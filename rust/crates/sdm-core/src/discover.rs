// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The source's facts of a table's columns, read once for `describe_source`
//! and `materialize` alike (T3.16, fix A): the catalog's columns and keys
//! (one step, then any index's columns), the preview, each big integer's
//! tier over the whole column ([`whole_column_tiers`]) and each exact
//! number's type by its role ([`exact_number::settle`]), limited to the
//! columns asked, with no inference, links or key counting.
//!
//! `describe_source` infers each table from what [`table`] reads;
//! `materialize` reads, by [`facts`], the facts of each column the host's
//! schema holds none of (a schema built by hand, or a column added to one
//! described), as `describe_source` would have read them, and nothing for a
//! schema whose columns all carry theirs. The cost, per table holding such
//! a column: the `columns` and `keys` catalog requests (with any
//! `index_columns` read), one preview, and one `column_ranges` read only
//! where the preview may have left rows unread and a big integer needs
//! widening; requests of one table are asked together in one step.

use std::collections::BTreeMap;
use std::sync::Arc;

use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogKey, KeyKind};
use sdm_contract::data::{DataRequest, DescribePart, Relation};
use sdm_contract::ops::{HeldSchema, NamedTable, SourceFacts};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_schema::held::ColumnAt;
use sdm_schema::infer::PREVIEW_ROWS;

use crate::describe::{another_part, ask, describe, describe_all, table_names};
use crate::run::{ArrowData, Asker, schema_mismatch};
use crate::{big_integer, exact_number};

/// What [`table`] read of one table.
pub(crate) struct Discovered {
    /// The catalog's columns read, in its order, as the preview holds them.
    pub declared: Vec<CatalogColumn>,
    /// The table's keys over the columns the schema keeps, each index's
    /// columns read.
    pub constraints: Vec<CatalogKey>,
    /// The first rows of the columns read, each big integer in its whole
    /// column's tier and each exact number in its role's type.
    pub preview: ArrowData,
    /// Each big integer's declared type ([`big_integer_types`]).
    pub big_types: Vec<(String, String)>,
}

/// The facts of each column of `held` that carries none, read from the
/// source as `describe_source` reads them ([`table`]), by table and column
/// (T3.16): none asked for a schema whose columns all carry theirs. A column
/// the source does not hold has none, and the schema read without it
/// ([`sdm_schema::held::read`]), as the reference's engine reads only the
/// columns its source holds; a table the source does not hold is refused as
/// describing it refuses one (`no_such_table`).
pub(crate) async fn facts(
    asker: &Asker,
    held: &HeldSchema,
) -> Result<BTreeMap<ColumnAt, SourceFacts>, CoreError> {
    let mut found = BTreeMap::new();
    for (table, lacking) in sdm_schema::held::lacking_facts(held) {
        let named = NamedTable {
            name: table.name.clone(),
            source: table.source.clone(),
            columns: Some(table.columns.iter().map(|c| c.name.clone()).collect()),
        };
        let asked: Vec<String> = lacking.iter().map(|c| (*c).to_owned()).collect();
        let read = self::table(asker, &named, true, Some(&asked)).await?;
        let types = read.big_types;
        for (name, mut facts) in sdm_schema::infer::source_facts(
            &table.source,
            &read.declared,
            &read.constraints,
            &read.preview.schema,
        )? {
            facts.declared_type = types
                .iter()
                .find(|(column, _)| *column == name)
                .map(|(_, declared)| declared.clone());
            found.insert((table.name.clone(), name), facts);
        }
    }
    Ok(found)
}

/// One table read as `describe_source` reads it, under the name and source
/// `table` gives: its catalog's columns and keys over the columns it keeps
/// (`table.columns`; every column when none), and the preview of those
/// `previewed` names (every one kept when none), each big integer in its whole
/// column's tier and each exact number in its role's type. A table the
/// source does not hold, or holds with no columns, is refused by name
/// (`no_such_table`); where the caller named it (`caller_named`), a
/// provider's failure reading its columns is asked of first.
pub(crate) async fn table(
    asker: &Asker,
    table: &NamedTable,
    caller_named: bool,
    previewed: Option<&[String]>,
) -> Result<Discovered, CoreError> {
    let one = || vec![table.source.clone()];
    let columns = DataRequest::Describe {
        tables: one(),
        part: DescribePart::Columns,
    };
    let keys = DataRequest::Describe {
        tables: one(),
        part: DescribePart::Keys,
    };
    // Whether a failed read of the table's keys is read as none, as the
    // reference reads Unity Catalog's informational constraints
    // (`DatabricksTable`, DB/table.py:271-303): the columns first, then the
    // keys, a failure of which is read as no keys.
    let asked = if asker.capabilities()?.keys_optional {
        match describe(asker, columns).await {
            Ok(columns) => {
                let keys = match describe(asker, keys).await {
                    Err(failed) if failed.code == ErrorCode::ProviderError => {
                        CatalogAnswer::Keys(Vec::new())
                    }
                    other => other?,
                };
                Ok([columns, keys])
            }
            Err(failed) => Err(failed),
        }
    } else {
        describe_all(asker, [columns, keys]).await
    };
    let [columns, keys] = match asked {
        Ok(answers) => answers,
        // A table the caller named that the source does not hold: a
        // dialect may fail reading its columns rather than answer none
        // (DuckDB's pragma does). The source, asked only then whether
        // it holds a relation of that name, says which, so the caller's
        // naming is refused as the reference refuses it, and any other
        // failure stays the provider's.
        Err(failed) if caller_named && failed.code == ErrorCode::ProviderError => {
            // Absence only where the source says so; the provider's
            // first error stands otherwise, and where the question
            // itself fails, whatever its error (a provider's, or an
            // answer that does not read), since that says nothing of
            // the table.
            return Err(match holds(asker, &table.source).await {
                Ok(false) => not_held(asker, &table.source).await,
                Ok(true) | Err(_) => failed,
            });
        }
        Err(failed) => return Err(failed),
    };
    let mut declared = declared_columns(columns)?;
    if declared.is_empty() {
        // A table the source answers with no columns, as the reference
        // refuses one it cannot read.
        return Err(not_held(asker, &table.source).await);
    }
    // Every later read of the table reads the columns its catalog found
    // (`Asker::guard`).
    asker.guard(
        &table.source,
        declared.iter().map(|column| column.name.clone()).collect(),
    );
    let mut keys = declared_keys(keys)?;
    if let Some(kept) = &table.columns {
        // Only the columns the schema keeps are read, and only the keys
        // over them, so a column it leaves out is never inferred, as the
        // reference reads only the columns a schema describes; one it
        // keeps that the source lacks is the description's to lack. A
        // key dropped for a column left out takes its marks off the
        // columns kept, so what the catalog says of a column and of its
        // keys agree (a primary key of `ts` and a column left out is no
        // primary key of `ts`).
        declared.retain(|c| kept.contains(&c.name));
        let (held, dropped): (Vec<_>, Vec<_>) = keys
            .into_iter()
            .partition(|k| k.columns.iter().all(|c| kept.contains(c)));
        keys = held;
        for key in dropped.iter().filter(|k| k.kind == KeyKind::Primary) {
            for column in declared
                .iter_mut()
                .filter(|c| key.columns.contains(&c.name))
            {
                column.primary_key = false;
            }
        }
    }
    let constraints = key_columns(asker, &table.source, keys).await?;
    if let Some(previewed) = previewed {
        // Only the columns asked are previewed, once their keys are read
        // over every column the schema keeps.
        declared.retain(|c| previewed.contains(&c.name));
        if declared.is_empty() {
            // None of them is the source's: nothing to preview.
            return Ok(Discovered {
                declared,
                constraints,
                preview: ArrowData::new(Arc::new(arrow_schema::Schema::empty()), Vec::new()),
                big_types: Vec::new(),
            });
        }
    }
    let column_names: Vec<String> = declared.iter().map(|c| c.name.clone()).collect();
    let preview = ask(
        asker,
        DataRequest::Preview {
            table: table.source.clone(),
            columns: column_names.clone(),
            limit: PREVIEW_ROWS,
        },
    )
    .await?;
    let returned: Vec<&String> = preview.schema.fields().iter().map(|f| f.name()).collect();
    if returned.len() != column_names.len()
        || returned.iter().zip(&column_names).any(|(a, b)| *a != b)
    {
        return Err(schema_mismatch(
            "expected",
            None,
            "a preview's columns are not the ones it asked for",
        ));
    }
    // A big integer's tier, the whole column's (P1B-43).
    let preview = whole_column_tiers(asker, &table.source, preview).await?;
    // A column of numbers handed over as their text, by its role: a
    // key's text, any other's doubles (crate::exact_number).
    let preview = exact_number::settle(preview, &constraints)?;
    let big_types = big_integer_types(&preview, &declared);
    Ok(Discovered {
        declared,
        constraints,
        preview,
        big_types,
    })
}

/// Whether the time column's choice falls back to the preview on
/// `failed`, the error asking `column_ranges` gave: the provider's own
/// failure (`PROVIDER_ERROR`), as the reference falls back on whatever its
/// fetch raises, or the request refused as one the provider does not
/// declare (`UNSUPPORTED_REQUEST` naming it), as a table that cannot ask
/// its source falls back there. Anything else is the core's error.
///
/// The second test reads only the code and the `request` field: any
/// `UNSUPPORTED_REQUEST` naming `column_ranges` falls back, whatever
/// raised it. Today only [`crate::compose::undeclared`] raises one; a
/// refusal of another kind that came to carry that field would fall back
/// too.
pub(crate) fn falls_back(failed: &CoreError) -> bool {
    match failed.code {
        ErrorCode::ProviderError => true,
        ErrorCode::UnsupportedRequest => {
            failed.fields.get("request") == Some(&Detail::text("column_ranges"))
        }
        _ => false,
    }
}

/// The catalog's declared type of each big-integer column the host layer
/// read from an opaque type and marked ([`crate::big_integer`]), by name:
/// what the described schema keeps (`DescribedColumn::declared_type`), so a
/// `decimal128(38, 0)` seed is joined to such a key in the key's own type
/// with no statement `materialize` adds (register P1B-43). Every other
/// column keeps none, and joins as before part 3.
fn big_integer_types(preview: &ArrowData, declared: &[CatalogColumn]) -> Vec<(String, String)> {
    preview
        .schema
        .fields()
        .iter()
        .filter(|f| big_integer::is_marked(f))
        .filter_map(|f| {
            let column = declared.iter().find(|c| &c.name == f.name())?;
            Some((column.name.clone(), column.declared_type.clone()))
        })
        .collect()
}

/// `preview` with each big-integer column the host layer read
/// ([`crate::big_integer`]) in the tier of its values over the whole
/// table, so the column is described in one tier every fetch is held to:
/// where the preview may have left rows unread, the column's smallest and
/// largest values are asked of the source (`column_ranges`, one request for
/// every such column), and the column takes the widest tier of the preview
/// and those two (a column already read as text, the widest tier, is not
/// asked). The request is the time column's own kind, but asked apart,
/// before inference: a big integer's tier decides its inference (an int or
/// a string, and so whether it may be the key), which the time column's
/// candidates follow. Where the source fails that request or does not take it,
/// as [`falls_back`] reads it, the preview's tier stands, and a later fetch
/// past it is refused by name.
async fn whole_column_tiers(
    asker: &Asker,
    source: &Relation,
    preview: ArrowData,
) -> Result<ArrowData, CoreError> {
    let big: Vec<(String, arrow_schema::DataType)> = preview
        .schema
        .fields()
        .iter()
        .filter_map(|f| big_integer::tier(f).map(|t| (f.name().clone(), t)))
        // Text is the widest tier: no range widens it.
        .filter(|(_, tier)| *tier != arrow_schema::DataType::Utf8)
        .collect();
    let rows: usize = preview.batches.iter().map(|b| b.num_rows()).sum();
    if big.is_empty() || (rows as u64) < PREVIEW_ROWS {
        return Ok(preview);
    }
    let asked: Vec<String> = big.iter().map(|(name, _)| name.clone()).collect();
    let read = match ask(
        asker,
        DataRequest::ColumnRanges {
            table: source.clone(),
            columns: asked.clone(),
        },
    )
    .await
    {
        Ok(read) => read,
        Err(failed) if falls_back(&failed) => return Ok(preview),
        Err(failed) => return Err(failed),
    };
    let fields = read.schema.fields();
    if fields.len() != 1 + 3 * asked.len() {
        return Err(schema_mismatch(
            "expected",
            None,
            "the column ranges hold another number of columns than asked",
        ));
    }
    let mut tiers = Vec::new();
    for (at, (name, previewed)) in big.iter().enumerate() {
        let mut tier = previewed.clone();
        for index in [1 + 3 * at, 2 + 3 * at] {
            tier = big_integer::tier(&fields[index])
                .and_then(|range| big_integer::wider(&tier, &range))
                .ok_or_else(|| {
                    schema_mismatch(
                        "expected",
                        Some(name),
                        "a big integer's range is not of a big integer's type",
                    )
                })?;
        }
        if &tier != previewed {
            tiers.push((name.clone(), tier));
        }
    }
    if tiers.is_empty() {
        return Ok(preview);
    }
    big_integer::widen_data(&preview, &tiers)
}

/// The refusal of a table to describe that the source does not hold, or
/// holds with no columns: `INVALID_REQUEST`, as the reference refuses a
/// table it cannot read (its `TableNotFoundError`), naming the table as
/// the source reads it (`source`): under the provider's namespace, or, for
/// a name of one part, under the namespace the source reads it in, which
/// the source says (`describe` of its namespace: a Postgres source's
/// current schema, as the reference names the table under it); and the
/// database the source is (`database`), which the host words.
pub(crate) async fn not_held(asker: &Asker, table: &Relation) -> CoreError {
    let mut read = asker.read_as(table);
    if read.source.len() == 1 {
        let asked = DataRequest::Describe {
            tables: vec![],
            part: DescribePart::Namespace,
        };
        // Only to name the table: a source that cannot say leaves it alone.
        if let Ok(CatalogAnswer::Namespace(parts)) = describe(asker, asked).await {
            read.source.splice(0..0, parts);
        }
    }
    let refused = CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "a table to describe is not one the source holds, or has no columns there",
    )
    .with_field("field", Detail::text("tables"))
    .with_field("rule", Detail::text("no_such_table"))
    .with_field(
        "source",
        Detail::List(read.source.into_iter().map(Scalar::Utf8).collect()),
    );
    match asker.source_database() {
        Some(database) => refused.with_field("database", Detail::text(database)),
        None => refused,
    }
}

/// Whether the source holds a relation, a table or a view, under `source`,
/// as the source itself resolves the name: `describe` of the tables with
/// `source` named, whose answer is empty when it holds none (sdm-data's
/// rendering decides how the dialect matches a name; the core matches
/// none).
async fn holds(asker: &Asker, source: &Relation) -> Result<bool, CoreError> {
    let held = table_names(
        describe(
            asker,
            DataRequest::Describe {
                tables: vec![source.clone()],
                part: DescribePart::Tables,
            },
        )
        .await?,
    )?;
    Ok(!held.is_empty())
}

fn declared_columns(answer: CatalogAnswer) -> Result<Vec<CatalogColumn>, CoreError> {
    match answer {
        CatalogAnswer::Columns(columns) => Ok(columns),
        _ => Err(another_part()),
    }
}

fn declared_keys(answer: CatalogAnswer) -> Result<Vec<CatalogKey>, CoreError> {
    match answer {
        CatalogAnswer::Keys(keys) => Ok(keys),
        _ => Err(another_part()),
    }
}

/// `keys`, each one the catalog lists as an index by its name (SQLite's
/// unique indexes) with its columns read by that name, one index after
/// another in the catalog's order, as the reference reads each
/// (`PRAGMA index_info`); the index is its table's schema's object, so its
/// relation is `source`'s with the index's name last.
async fn key_columns(
    asker: &Asker,
    source: &Relation,
    keys: Vec<CatalogKey>,
) -> Result<Vec<CatalogKey>, CoreError> {
    let mut read = Vec::with_capacity(keys.len());
    for mut key in keys {
        if let Some(index) = &key.index {
            let mut relation = source.clone();
            if let Some(last) = relation.source.last_mut() {
                last.clone_from(index);
            }
            let answer = describe(
                asker,
                DataRequest::Describe {
                    tables: vec![relation],
                    part: DescribePart::IndexColumns,
                },
            )
            .await?;
            let CatalogAnswer::IndexColumns(columns) = answer else {
                return Err(another_part());
            };
            // An index over an expression names no column, and keys none
            // the core can read.
            match columns.into_iter().collect::<Option<Vec<String>>>() {
                Some(columns) if !columns.is_empty() => key.columns = columns,
                _ => key.kind = KeyKind::Other,
            }
        }
        read.push(key);
    }
    Ok(read)
}
