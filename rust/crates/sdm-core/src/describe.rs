// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `describe_source`: the tables of a source, read through the data port
//! and inferred by `sdm-schema`, as `Schema.from_source` reads them.
//!
//! The steps are the reference's, in its order, one data request for each
//! statement it runs (design, T1.11a row): the table list; for each table,
//! its columns and constraints (one step), then its first 1,000 rows, and,
//! where those may not be every row and the table has timestamps to choose
//! its time column among, their ranges over every row (from T1.32e); the
//! declared foreign keys of every table (one step); and, when keys are
//! checked, one count over every key the source does not enforce. Each
//! catalog answer is read by the run's data layer, in no dialect's shape
//! (decision E54), before anything here reads it. It refuses by name
//! ([`sdm_schema::not_yet`]) an input whose description takes a rule it
//! does not port yet (register entry P1B-2). From T1.21d the description
//! is the schema as the reference's client assembles it
//! ([`sdm_schema::assemble`], decision E55): a keyless table keyed by the
//! column foreign keys refer to before the count, then the links, the edit
//! stamps, the columns joined on, and the warnings, which the run gives the
//! host as codes and fields. From T1.23 (decision E59) it takes tables
//! named, each read by its source name with no listing of the source's
//! tables, the keys and time columns named, applied between the reading and
//! the count, and the relationships given, in place of the declared and
//! inferred ones.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_schema::TimeUnit;
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey};
use sdm_contract::data::{DataRequest, DescribePart, TableKey};
use sdm_contract::ops::{ColumnRef, DescribedSource, NamedTable, OperationResult, SourceSpec};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_engine::time::Instant;
use sdm_engine::time::text::Reader;
use sdm_schema::assemble::{self, Assembled, Given};
use sdm_schema::catalog;
use sdm_schema::infer::{self, InferredTable, PREVIEW_ROWS};
use sdm_schema::keys;

use crate::run::{
    ArrowData, Ask, Asker, Attachment, FetchResult, Reply, internal, schema_mismatch,
};

/// The reader of text times for a run whose clock is `facts.now`: its
/// year, in UTC, places a two-digit year, as the date literals' reader
/// placed one before T1.6b; date literals now read the year in the local
/// zone (`pql::local_year`).
pub(crate) fn text_reader(facts: &HostFacts) -> Result<Reader, CoreError> {
    let Scalar::Timestamp { unit, value, .. } = &facts.now else {
        return Err(internal("the host's clock is not a ts scalar"));
    };
    let unit = match unit {
        Unit::Ns => TimeUnit::Nanosecond,
        Unit::Us => TimeUnit::Microsecond,
    };
    Ok(Reader::new(Instant::from_count(*value, unit).civil(0).year))
}

/// The body of a `describe_source` run.
pub(crate) async fn run(
    asker: Asker,
    spec: SourceSpec,
    check_keys: bool,
    facts: HostFacts,
) -> Result<OperationResult, CoreError> {
    let reader = text_reader(&facts)?;
    // The tables named, each under the schema's name and read by the
    // source's (`_require_table_names`); none named, every table the
    // source lists, under its own name.
    let named = named_tables(&spec)?;
    let caller_named = !named.is_empty();
    let names: Vec<NamedTable> = if named.is_empty() {
        let listed = table_names(
            describe(
                &asker,
                DataRequest::Describe {
                    tables: vec![],
                    part: DescribePart::Tables,
                },
            )
            .await?,
        )?;
        if listed.is_empty() {
            return Err(CoreError::new(
                ErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
                "the source holds no table to describe",
            )
            .with_field("field", Detail::text("tables")));
        }
        // Each listed under its own name, which must be one the schema
        // may give a table.
        let mut tables = Vec::with_capacity(listed.len());
        for name in listed {
            assemble::check_table_name(&name)?;
            tables.push(NamedTable {
                source: name.clone(),
                name,
            });
        }
        tables
    } else {
        named
    };

    let mut tables = Vec::with_capacity(names.len());
    for table in &names {
        let one = || vec![table.source.clone()];
        let asked = describe_all(
            &asker,
            [
                DataRequest::Describe {
                    tables: one(),
                    part: DescribePart::Columns,
                },
                DataRequest::Describe {
                    tables: one(),
                    part: DescribePart::Keys,
                },
            ],
        )
        .await;
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
                return Err(match holds(&asker, &table.source).await {
                    Ok(false) => not_held(),
                    Ok(true) | Err(_) => failed,
                });
            }
            Err(failed) => return Err(failed),
        };
        let declared = declared_columns(columns)?;
        if declared.is_empty() {
            // A table the source answers with no columns, as the reference
            // refuses one it cannot read.
            return Err(not_held());
        }
        let constraints = declared_keys(keys)?;
        let column_names: Vec<String> = declared.iter().map(|c| c.name.clone()).collect();
        let preview = ask(
            &asker,
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
        let previewed = infer::preview(
            &table.name,
            &table.source,
            &declared,
            &constraints,
            &preview.schema,
            &preview.batches,
            &reader,
        )?;
        // A preview that may have left rows unread: the time column's
        // choice reads its timestamps over every row, as the reference's
        // asks its source (`full_range`), in one statement.
        let asked = previewed.whole_table_columns();
        let inferred = if asked.is_empty() {
            infer::finish(previewed, None)?
        } else {
            let read = ask(
                &asker,
                DataRequest::ColumnRanges {
                    table: table.source.clone(),
                    columns: asked.clone(),
                },
            )
            .await?;
            check_ranges(&read, &asked, &preview)?;
            let whole = infer::whole_table(&asked, &read.batches, &reader)?;
            infer::finish(previewed, Some(&whole))?
        };
        // The engine says what it dropped as it reads each table, and the
        // day-first formats it guessed as it typed its columns, before
        // anything later can refuse.
        for warning in assemble::unread_times(&inferred)
            .into_iter()
            .chain(assemble::day_first(&inferred))
        {
            asker.warn(warning);
        }
        tables.push(inferred);
    }
    // The columns each table leaves out, said as the tables are described.
    for table in &tables {
        for warning in assemble::left_out(table) {
            asker.warn(warning);
        }
    }

    let foreign = describe_vec(
        &asker,
        names
            .iter()
            .map(|table| DataRequest::Describe {
                tables: vec![table.source.clone()],
                part: DescribePart::ForeignKeys,
            })
            .collect(),
    )
    .await?;
    let mut declared = Vec::new();
    for (table, answer) in tables.iter().zip(foreign) {
        declared.extend(declared_links(table, &foreign_keys(answer)?, &tables));
    }
    let given = spec.relationships.as_deref();
    if given.is_none() {
        // A keyless table keyed by the one column the declared foreign keys
        // refer to, before the keys are counted, as the reference keys it.
        assemble::key_by_reference(&mut tables, &declared)?;
    }
    // The keys and time columns named, between the reading and the count.
    assemble::apply_keys_and_times(&mut tables, &spec.primary_keys, &spec.time_columns)?;

    // The keys counted and held to the reference's rules, before the links
    // are inferred: a declined key is one no link can point at.
    let counted = if check_keys {
        let stated = spec.primary_keys.keys().cloned().collect();
        let held = count_keys(&asker, &mut tables, &stated, given.is_none()).await?;
        for warning in held.warnings {
            asker.warn(warning);
        }
        Some(held.duplicates)
    } else {
        None
    };
    let duplicates = counted.as_ref().map(|_| Scalar::UInt64(0));

    // The schema as the reference's client assembles it: the links, the
    // edit stamps they settle, the columns joined on marked, and the
    // warnings, given in the reference's order (decision E55).
    let Assembled {
        mut description,
        warnings,
    } = assemble::assemble(
        tables,
        declared,
        Given {
            time_columns: Some(&spec.time_columns),
            relationships: given,
        },
        duplicates,
    )?;
    for warning in warnings {
        asker.warn(warning);
    }
    // Each table's repeated keys, as the reference's count sets them.
    if let Some(counted) = counted {
        for (table, repeats) in description.tables.iter_mut().zip(counted) {
            table.duplicate_keys = Some(Scalar::UInt64(repeats));
        }
    }
    let bytes = serde_json::to_vec(&description)
        .map_err(|_| internal("a source description does not serialize"))?;
    let id = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::DescribeSource(DescribedSource {
        description: id,
    }))
}

/// Holds the answer to a `column_ranges` request over `asked` to its
/// shape, which the contract leaves the columns' own types to (`Discover`):
/// one row; the table's rows, then each column's smallest value, largest
/// value and rows holding one, each count a non-null 64-bit integer and
/// each value of the type the column's preview returned.
fn check_ranges(read: &ArrowData, asked: &[String], preview: &ArrowData) -> Result<(), CoreError> {
    let fields = read.schema.fields();
    if fields.len() != 1 + 3 * asked.len() {
        return Err(schema_mismatch(
            "expected",
            None,
            "the column ranges hold another number of columns than asked",
        ));
    }
    if read.batches.iter().map(|b| b.num_rows()).sum::<usize>() != 1 {
        return Err(schema_mismatch(
            "expected",
            None,
            "the column ranges are not one row",
        ));
    }
    let counts = std::iter::once(0).chain((0..asked.len()).map(|at| 3 + 3 * at));
    for index in counts {
        let nulls: usize = read
            .batches
            .iter()
            .map(|b| b.column(index).null_count())
            .sum();
        if *fields[index].data_type() != arrow_schema::DataType::Int64 || nulls > 0 {
            return Err(schema_mismatch(
                "expected",
                None,
                "a count of the column ranges is not a 64-bit integer",
            ));
        }
    }
    for (at, name) in asked.iter().enumerate() {
        let previewed = preview
            .schema
            .field_with_name(name)
            .map_err(|_| internal("a column asked over the whole table is not the preview's"))?
            .data_type();
        for index in [1 + 3 * at, 2 + 3 * at] {
            if fields[index].data_type() != previewed {
                return Err(schema_mismatch(
                    "expected",
                    Some(name),
                    "a column's range is not of the type its preview returned",
                ));
            }
        }
    }
    Ok(())
}

/// The refusal of a table to describe that the source does not hold, or
/// holds with no columns: `INVALID_REQUEST`, as the reference refuses a
/// table it cannot read (its `TableNotFoundError`).
fn not_held() -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "a table to describe is not one the source holds, or has no columns there",
    )
    .with_field("field", Detail::text("tables"))
}

/// Whether the source holds a relation, a table or a view, under `source`,
/// as the source itself resolves the name: `describe` of the tables with
/// `source` named, whose answer is empty when it holds none (sdm-data's
/// rendering decides how the dialect matches a name; the core matches
/// none).
async fn holds(asker: &Asker, source: &str) -> Result<bool, CoreError> {
    let held = table_names(
        describe(
            asker,
            DataRequest::Describe {
                tables: vec![source.to_owned()],
                part: DescribePart::Tables,
            },
        )
        .await?,
    )?;
    Ok(!held.is_empty())
}

/// The tables `spec` names, checked as the reference checks a mapping of
/// them (`_require_table_names`): each source name non-empty, no name
/// twice, and each name one the schema may give a table
/// ([`assemble::check_table_name`]). Refusals are `INVALID_REQUEST`
/// naming the field.
fn named_tables(spec: &SourceSpec) -> Result<Vec<NamedTable>, CoreError> {
    let refuse = |message: &'static str| {
        CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            message,
        )
        .with_field("field", Detail::text("tables"))
    };
    let mut seen = std::collections::BTreeSet::new();
    for table in &spec.tables {
        assemble::check_table_name(&table.name)?;
        if table.source.is_empty() {
            return Err(refuse("a table's name in the source is empty"));
        }
        if !seen.insert(table.name.as_str()) {
            return Err(refuse("a table is named twice"));
        }
    }
    Ok(spec.tables.clone())
}

/// The foreign keys `table` declares to tables of the source, as links
/// `[foreign key, referenced key]` by the schema's names: a constraint of
/// one column pair whose referenced table, a name in the source, is a
/// table described, and whose columns both tables hold. The referenced
/// table is found as the reference finds it (`_declared_links`,
/// `_table_named`): through a map from each source name to the schema's
/// name, the last table described under a source name holding it, then by
/// the exact source name or, failing that, the one source name equal to
/// it in any case. A column declared twice keeps its first place and its
/// last constraint, as the reference's mapping by column does.
fn declared_links(
    table: &InferredTable,
    parts: &[CatalogForeignKey],
    tables: &[InferredTable],
) -> Vec<Vec<ColumnRef>> {
    // Source name to the table described under it, the last one winning,
    // in the order each source name first appears.
    let mut by_source: Vec<(&str, &InferredTable)> = Vec::new();
    for described in tables {
        match by_source
            .iter_mut()
            .find(|(source, _)| *source == described.source)
        {
            Some(slot) => slot.1 = described,
            None => by_source.push((&described.source, described)),
        }
    }
    let mut keys: Vec<(&String, &CatalogForeignKey)> = Vec::new();
    for part in parts {
        let pairs = parts.iter().filter(|p| p.key == part.key).count();
        let ([column], [_, ..]) = (part.columns.as_slice(), part.referenced_columns.as_slice())
        else {
            continue;
        };
        if pairs != 1 {
            continue;
        }
        match keys.iter_mut().find(|(name, _)| *name == column) {
            Some(slot) => slot.1 = part,
            None => keys.push((column, part)),
        }
    }
    let mut links = Vec::new();
    for (column, part) in keys {
        let exact = by_source
            .iter()
            .find(|(source, _)| *source == part.referenced_table);
        let folded: Vec<&(&str, &InferredTable)> = by_source
            .iter()
            .filter(|(source, _)| source.to_lowercase() == part.referenced_table.to_lowercase())
            .collect();
        let target = match (exact, folded.as_slice()) {
            (Some((_, target)), _) => *target,
            (None, [(_, target)]) => *target,
            _ => continue,
        };
        let referenced = &part.referenced_columns[0];
        if table.column(column).is_none() || target.column(referenced).is_none() {
            continue;
        }
        links.push(vec![
            ColumnRef {
                table: table.name.clone(),
                column: column.clone(),
            },
            ColumnRef {
                table: target.name.clone(),
                column: referenced.clone(),
            },
        ]);
    }
    links
}

/// Counts every key the source does not enforce in one request, then, for
/// each that repeats a value, reads one such value (`repeated_key`), in
/// table order, as the reference's `count_keys` does, and holds the keys to
/// the reference's rules ([`keys::hold`]): a named key holding a null or a
/// named or declared key repeating in more than 1% of its rows refused, an
/// inferred key under 99% distinct declined, a key kept with nulls or
/// repeats said. `stated` names the tables whose key `primary_keys` named;
/// `infer_links` says whether the links are inferred.
async fn count_keys(
    asker: &Asker,
    tables: &mut [InferredTable],
    stated: &std::collections::BTreeSet<String>,
    infer_links: bool,
) -> Result<keys::Held, CoreError> {
    let positions: Vec<usize> = (0..tables.len())
        .filter(|&i| {
            tables[i]
                .primary_key
                .clone()
                .is_some_and(|key| !tables[i].enforces(&[key]))
        })
        .collect();
    if positions.is_empty() {
        return keys::hold(tables, &[], stated, infer_links);
    }
    let asked: Vec<TableKey> = positions
        .iter()
        .map(|&i| TableKey {
            table: tables[i].source.clone(),
            columns: tables[i].primary_key.clone().into_iter().collect(),
        })
        .collect();
    let answered = ask(
        asker,
        DataRequest::KeyCounts {
            keys: asked.clone(),
        },
    )
    .await?;
    let mut counts = catalog::key_counts(&answered.batches)?;
    // One row per key, each read by its position, as the reference reads
    // each row by its label.
    counts.sort_by_key(|c| c.position);
    if counts.len() != asked.len()
        || counts
            .iter()
            .enumerate()
            .any(|(at, c)| usize::try_from(c.position).ok() != Some(at))
    {
        return Err(schema_mismatch(
            "expected",
            None,
            "a key count does not hold one row per key",
        ));
    }
    let mut counted = Vec::with_capacity(counts.len());
    for ((&at, key), count) in positions.iter().zip(&asked).zip(counts) {
        let number = |n: i64| {
            u64::try_from(n)
                .map_err(|_| schema_mismatch("expected", None, "a key count is negative"))
        };
        let mut found = keys::Counted {
            rows: number(count.rows)?,
            distinct: number(count.distinct)?,
            missing: number(count.missing)?,
            example: None,
        };
        if found.distinct + found.missing > found.rows {
            return Err(schema_mismatch(
                "expected",
                None,
                "a key count holds more values than rows",
            ));
        }
        if found.repeats() > 0 {
            let read = ask(
                asker,
                DataRequest::RepeatedKey {
                    table: key.table.clone(),
                    key: key.columns.clone(),
                },
            )
            .await?;
            found.example = keys::example(&read.batches);
        }
        counted.push((at, found));
    }
    keys::hold(tables, &counted, stated, infer_links)
}

/// The data a request's answer holds, or the provider's error.
pub(crate) async fn ask(asker: &Asker, request: DataRequest) -> Result<ArrowData, CoreError> {
    data(asker.fetch(request)?.await)
}

/// The catalog's answer to `request`, a `describe`: fetched, admitted,
/// then read by the run's data layer (decision E54).
pub(crate) async fn describe(
    asker: &Asker,
    request: DataRequest,
) -> Result<CatalogAnswer, CoreError> {
    let data = ask(asker, request.clone()).await?;
    asker.read(&request, &data)
}

/// The catalog's answers to `requests`, asked together in one step, in
/// order.
async fn describe_all<const N: usize>(
    asker: &Asker,
    requests: [DataRequest; N],
) -> Result<[CatalogAnswer; N], CoreError> {
    let answers = describe_vec(asker, requests.into()).await?;
    answers
        .try_into()
        .map_err(|_| internal("a step answered another number of requests"))
}

async fn describe_vec(
    asker: &Asker,
    requests: Vec<DataRequest>,
) -> Result<Vec<CatalogAnswer>, CoreError> {
    let answers = ask_all_vec(asker, requests.clone()).await?;
    requests
        .iter()
        .zip(&answers)
        .map(|(request, data)| asker.read(request, data))
        .collect()
}

/// The refusal of a catalog answer the data layer read as another part
/// than the one asked for.
fn another_part() -> CoreError {
    internal("the data layer read another part of the catalog than was asked for")
}

fn table_names(answer: CatalogAnswer) -> Result<Vec<String>, CoreError> {
    match answer {
        CatalogAnswer::Tables(names) => Ok(names),
        _ => Err(another_part()),
    }
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

fn foreign_keys(answer: CatalogAnswer) -> Result<Vec<CatalogForeignKey>, CoreError> {
    match answer {
        CatalogAnswer::ForeignKeys(keys) => Ok(keys),
        _ => Err(another_part()),
    }
}

pub(crate) async fn ask_all_vec(
    asker: &Asker,
    requests: Vec<DataRequest>,
) -> Result<Vec<ArrowData>, CoreError> {
    let asks = requests
        .into_iter()
        .map(|request| asker.fetch(request))
        .collect::<Result<Vec<_>, _>>()?;
    Together::new(asks).await.into_iter().map(data).collect()
}

fn data(reply: Reply) -> Result<ArrowData, CoreError> {
    match reply {
        Reply::Fetch(FetchResult::Data(data)) => Ok(data),
        Reply::Fetch(FetchResult::Error(error)) => Err(CoreError::provider(&error)),
        Reply::Model(_) => Err(internal("a fetch was answered by the model")),
    }
}

/// Requests awaited together: each is asked at the first poll, so they make
/// one step, and the answers come back in the order asked.
struct Together {
    asks: Vec<Option<Ask>>,
    answers: Vec<Option<Reply>>,
}

impl Together {
    fn new(asks: Vec<Ask>) -> Self {
        let answers = asks.iter().map(|_| None).collect();
        Together {
            asks: asks.into_iter().map(Some).collect(),
            answers,
        }
    }
}

impl Future for Together {
    type Output = Vec<Reply>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Vec<Reply>> {
        let this = self.get_mut();
        for (slot, answer) in this.asks.iter_mut().zip(this.answers.iter_mut()) {
            if let Some(ask) = slot
                && let Poll::Ready(reply) = Pin::new(ask).poll(cx)
            {
                *answer = Some(reply);
                *slot = None;
            }
        }
        if this.asks.iter().any(Option::is_some) {
            return Poll::Pending;
        }
        Poll::Ready(this.answers.iter_mut().filter_map(Option::take).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use sdm_contract::ErrorCode;

    use super::{ArrowData, check_ranges};

    fn stamp() -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, None)
    }

    /// Data of one row (or `rows` rows), each column given as its type
    /// and a value of it.
    fn data(columns: &[(&str, DataType)], rows: usize, null_count: bool) -> ArrowData {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, t)| Field::new(*name, t.clone(), true))
                .collect::<Vec<_>>(),
        ));
        let arrays: Vec<ArrayRef> = columns
            .iter()
            .enumerate()
            .map(|(at, (_, t))| -> ArrayRef {
                match t {
                    DataType::Int64 if null_count && at == 0 => {
                        Arc::new(Int64Array::from(vec![None::<i64>; rows]))
                    }
                    DataType::Int64 => Arc::new(Int64Array::from(vec![3; rows])),
                    DataType::Utf8 => Arc::new(StringArray::from(vec!["x"; rows])),
                    _ => Arc::new(TimestampMicrosecondArray::from(vec![0; rows])),
                }
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        ArrowData::new(schema, vec![batch])
    }

    fn preview() -> ArrowData {
        data(&[("id", DataType::Int64), ("ts", stamp())], 2, false)
    }

    fn mismatch(read: &ArrowData) -> bool {
        check_ranges(read, &["ts".to_owned()], &preview())
            .is_err_and(|err| err.code == ErrorCode::ProviderSchemaMismatch)
    }

    #[test]
    fn the_column_ranges_are_held_to_their_shape() {
        // The control: the rows, then the column's low, high and count.
        let good = [
            ("n", DataType::Int64),
            ("lo", stamp()),
            ("hi", stamp()),
            ("c", DataType::Int64),
        ];
        assert!(check_ranges(&data(&good, 1, false), &["ts".to_owned()], &preview()).is_ok());
        // Seeded faults, each refused: no row or two; a column missing or
        // one more; a count null or of another type; a range of another
        // type than the preview's.
        assert!(mismatch(&data(&good, 0, false)));
        assert!(mismatch(&data(&good, 2, false)));
        assert!(mismatch(&data(&good[..3], 1, false)));
        let mut more = good.to_vec();
        more.push(("x", DataType::Int64));
        assert!(mismatch(&data(&more, 1, false)));
        assert!(mismatch(&data(&good, 1, true)));
        let mut retyped = good.to_vec();
        retyped[3] = ("c", DataType::Utf8);
        assert!(mismatch(&data(&retyped, 1, false)));
        let mut retyped = good.to_vec();
        retyped[0] = ("n", stamp());
        assert!(mismatch(&data(&retyped, 1, false)));
        let mut text = good.to_vec();
        text[1] = ("lo", DataType::Utf8);
        assert!(mismatch(&data(&text, 1, false)));
    }
}
