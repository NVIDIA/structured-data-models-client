// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `describe_source`: the tables of a source, read through the data port
//! and inferred by `sdm-schema`, as `Schema.from_source` reads them.
//!
//! The steps are the reference's, in its order, one data request for each
//! statement it runs (design, T1.11a row): the table list; for each table,
//! its columns and constraints (one step), then its first 1,000 rows; the
//! declared foreign keys of every table (one step); and, when keys are
//! checked, one count over every key the source does not enforce. Each
//! catalog answer is read by the run's data layer, in no dialect's shape
//! (decision E54), before anything here reads it. As built
//! at T1.11b the operation is thin: it describes every table of the source
//! with no overrides, and refuses by name ([`sdm_schema::not_yet`]) the
//! inputs whose description takes a rule it does not port yet (register
//! entry P1B-2). From T1.21d the description is the schema as the
//! reference's client assembles it ([`sdm_schema::assemble`], decision
//! E55): a keyless table keyed by the column foreign keys refer to before
//! the count, then the links, the edit stamps, the columns joined on, and
//! the warnings, which the run gives the host.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogKey};
use sdm_contract::data::{DataRequest, DescribePart, TableKey};
use sdm_contract::ops::{ColumnRef, DescribedSource, OperationResult, SourceSpec};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_schema::assemble::{self, Assembled};
use sdm_schema::catalog;
use sdm_schema::infer::{self, InferredTable, PREVIEW_ROWS};
use sdm_schema::not_yet;

use crate::run::{
    ArrowData, Ask, Asker, Attachment, FetchResult, Reply, internal, schema_mismatch,
};

/// The body of a `describe_source` run.
pub(crate) async fn run(
    asker: Asker,
    spec: SourceSpec,
    check_keys: bool,
) -> Result<OperationResult, CoreError> {
    // Tables named, and keys and time columns given, are the caller's
    // overrides, which the reference applies between inference and the key
    // count; not ported yet.
    if !spec.tables.is_empty() {
        return Err(not_yet("tables_named"));
    }
    if !spec.primary_keys.is_empty() {
        return Err(not_yet("primary_keys"));
    }
    if !spec.time_columns.is_empty() {
        return Err(not_yet("time_columns"));
    }

    let names = table_names(
        describe(
            &asker,
            DataRequest::Describe {
                tables: vec![],
                part: DescribePart::Tables,
            },
        )
        .await?,
    )?;
    if names.is_empty() {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "the source holds no table to describe",
        )
        .with_field("field", Detail::text("tables")));
    }

    let mut tables = Vec::with_capacity(names.len());
    for name in &names {
        let one = || vec![name.clone()];
        let [columns, keys] = describe_all(
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
        .await?;
        let declared = declared_columns(columns)?;
        if declared.is_empty() {
            return Err(CoreError::new(
                ErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
                "a table the source lists has no columns",
            )
            .with_field("field", Detail::text("tables")));
        }
        let constraints = declared_keys(keys)?;
        let column_names: Vec<String> = declared.iter().map(|c| c.name.clone()).collect();
        let preview = ask(
            &asker,
            DataRequest::Preview {
                table: name.clone(),
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
        tables.push(infer::table(
            name,
            &declared,
            &constraints,
            &preview.schema,
            &preview.batches,
        )?);
    }

    let foreign = describe_vec(
        &asker,
        names
            .iter()
            .map(|name| DataRequest::Describe {
                tables: vec![name.clone()],
                part: DescribePart::ForeignKeys,
            })
            .collect(),
    )
    .await?;
    let mut declared = Vec::new();
    for (table, answer) in tables.iter().zip(foreign) {
        declared.extend(declared_links(table, &foreign_keys(answer)?, &tables));
    }
    // A keyless table keyed by the one column the declared foreign keys
    // refer to, before the keys are counted, as the reference keys it.
    assemble::key_by_reference(&mut tables, &declared)?;

    let duplicates = if check_keys {
        count_keys(&asker, &tables).await?;
        Some(Scalar::UInt64(0))
    } else {
        None
    };

    // The schema as the reference's client assembles it: the links, the
    // edit stamps they settle, the columns joined on marked, and the
    // warnings, given in the reference's order (decision E55).
    let Assembled {
        description,
        warnings,
    } = assemble::assemble(tables, declared, duplicates);
    for warning in warnings {
        asker.warn(warning);
    }
    let bytes = serde_json::to_vec(&description)
        .map_err(|_| internal("a source description does not serialize"))?;
    let id = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::DescribeSource(DescribedSource {
        description: id,
    }))
}

/// The foreign keys `table` declares to tables of the source, as links
/// `[foreign key, referenced key]`: a constraint of one column pair whose
/// referenced table is a table described, by its exact name or, failing
/// that, by the one name equal to it in any case, and whose columns both
/// tables hold. A column declared twice keeps its first place and its last
/// constraint, as the reference's mapping by column does.
fn declared_links(
    table: &InferredTable,
    parts: &[CatalogForeignKey],
    tables: &[InferredTable],
) -> Vec<Vec<ColumnRef>> {
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
        let folded: Vec<&InferredTable> = tables
            .iter()
            .filter(|t| t.name.to_lowercase() == part.referenced_table.to_lowercase())
            .collect();
        let target =
            tables
                .iter()
                .find(|t| t.name == part.referenced_table)
                .or(match folded.as_slice() {
                    [one] => Some(*one),
                    _ => None,
                });
        let Some(target) = target else { continue };
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

/// Counts every key the source does not enforce in one request: a key
/// whose rows repeat a value or miss one is held to rules the reference
/// warns, declines or refuses by, which are not ported yet.
async fn count_keys(asker: &Asker, tables: &[InferredTable]) -> Result<(), CoreError> {
    let keys: Vec<TableKey> = tables
        .iter()
        .filter_map(|table| {
            let key = vec![table.primary_key.clone()?];
            (!table.enforces(&key)).then(|| TableKey {
                table: table.name.clone(),
                columns: key,
            })
        })
        .collect();
    if keys.is_empty() {
        return Ok(());
    }
    let counted = ask(asker, DataRequest::KeyCounts { keys: keys.clone() }).await?;
    let counts = catalog::key_counts(&counted.batches)?;
    if counts.len() != keys.len() {
        return Err(schema_mismatch(
            "expected",
            None,
            "a key count does not hold one row per key",
        ));
    }
    for count in counts {
        if count.missing > 0 {
            return Err(not_yet("key_missing"));
        }
        if count.rows - count.missing - count.distinct > 0 {
            return Err(not_yet("key_repeats"));
        }
    }
    Ok(())
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

async fn ask_all_vec(
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
