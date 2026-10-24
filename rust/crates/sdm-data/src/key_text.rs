// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A key set bound as JSON text (T3.28, design S4): the one binding of a
//! dialect whose statement reads its key set from a text parameter
//! (Databricks's `posexplode(from_json(?, ...))`, Postgres's
//! `jsonb_array_elements($1::jsonb)`), written here once over the
//! dialect's hooks ([`SqlDialect::key_rows`], [`SqlDialect::key_budget`],
//! [`SqlDialect::chunk_text`]), so no provider writes or splits one.
//!
//! [`bind`] answers `DataLayer::bind` for every built-in dialect: a plan
//! binding no key set, or one whose dialect binds the core's Arrow key set
//! as a relation (DuckDB, SQLite), is one fetch of the plan as it is; a key
//! set the dialect writes as text is split as the old client splits it
//! (`_chunk_rows`, `DB/sampler.py:32-44`, `PG/sampler.py:125-182`): each
//! seed's JSON counted with one separator and the array's brackets as two,
//! a chunk closed before the seed that would take it past the dialect's
//! budget, never an empty chunk unless the key set is empty, so one seed
//! past the budget alone is a chunk of its own. Each chunk is one fetch:
//! the plan with the key set's parameter a text parameter naming a local
//! reference, its text a one-row, one-column text batch under that
//! reference, and each statement reading it as the dialect reads a chunk
//! whose first seed is at that offset of the key set. The core installs
//! each batch as one of its run's attachments, never in the control
//! message, sends the fetches in one step and joins their answers in
//! order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{
    AttachmentId, BoundFetch, BoundRequest, DataRequest, FetchPlan, KeyData, SEED_BATCH, SEED_KEY,
    SqlParam, SqlStatement,
};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::sql::SqlDialect;

/// The reference's budget of a key set's JSON text, in bytes, below a
/// warehouse's native parameter limit of 1 MB (`_chunk_rows` on Databricks
/// and Postgres alike).
pub(crate) const BUDGET: usize = 900_000;

/// The field the old client names a seed's column by in its key set's
/// JSON (`_by_fkey`, `_by_time` on Databricks and Postgres alike): the key
/// `id`, the window's upper bound `e` and its lower `s`; another column by
/// its own name. In that order, the key first.
pub(crate) fn fields<'a>(
    columns: impl IntoIterator<Item = &'a str>,
    request: &DataRequest,
) -> Vec<(&'a str, String)> {
    let window = match request {
        DataRequest::Neighbors { window, .. } => window.as_ref(),
        _ => None,
    };
    let bound = |bound: &Option<sdm_contract::data::SeedBound>, column: &str| {
        bound.as_ref().is_some_and(|b| b.seed_column == column)
    };
    let mut named: Vec<(usize, &str, String)> = columns
        .into_iter()
        .map(|column| {
            if column == SEED_KEY {
                (0, column, "id".to_owned())
            } else if window.is_some_and(|w| bound(&w.upper, column)) {
                (1, column, "e".to_owned())
            } else if window.is_some_and(|w| bound(&w.lower, column)) {
                (2, column, "s".to_owned())
            } else {
                (3, column, column.to_owned())
            }
        })
        .collect();
    named.sort_by_key(|(rank, _, _)| *rank);
    named
        .into_iter()
        .map(|(_, column, field)| (column, field))
        .collect()
}

/// The key columns of a key set's columns `names`: the one key, or each
/// of a key of several, in order (`seed_key_columns`).
pub(crate) fn key_columns<'a>(names: &[&'a str]) -> Vec<&'a str> {
    if let Some(one) = names.iter().find(|n| **n == SEED_KEY) {
        return vec![*one];
    }
    (1..)
        .map_while(|at| {
            let name = format!("{SEED_KEY}_{at}");
            names.iter().find(|n| **n == name).copied()
        })
        .collect()
}

/// Refuses a key set whose batches are not its seeds' positions from 0,
/// which a statement reading its JSON numbers by position cannot read.
pub(crate) fn in_batch_order(
    schema: &arrow_schema::SchemaRef,
    batches: &[RecordBatch],
) -> Result<(), CoreError> {
    let Ok(at) = schema.index_of(SEED_BATCH) else {
        return Ok(());
    };
    let mut next = 0_i64;
    for batch in batches {
        let column = batch.column(at);
        let Some(positions) = column.as_primitive_opt::<arrow_array::types::Int64Type>() else {
            return Err(out_of_order());
        };
        for position in positions.iter() {
            if position != Some(next) {
                return Err(out_of_order());
            }
            next += 1;
        }
    }
    Ok(())
}

fn out_of_order() -> CoreError {
    CoreError::new(
        Sentence::Internal,
        "a key set's batches are not its seeds' positions",
    )
}

/// The local reference a fetch names its key set's text by; each fetch
/// holds one batch, so one reference serves them all.
const TEXT: AttachmentId = AttachmentId(1);

/// The name of a text batch's one column.
const COLUMN: &str = "keys";

/// `plan`, the dialect's answer to a request, bound with the key sets the
/// core handed over (the module's documentation).
pub(crate) fn bind(
    dialect: &dyn SqlDialect,
    request: &DataRequest,
    plan: FetchPlan,
    keys: &BTreeMap<AttachmentId, KeyData>,
    float_text: fn(f64) -> String,
) -> Result<BoundRequest, CoreError> {
    let as_it_is = |plan| BoundRequest {
        fetches: vec![BoundFetch {
            plan,
            batches: BTreeMap::new(),
        }],
    };
    let FetchPlan::Sql { statements, expect } = &plan else {
        return Ok(as_it_is(plan));
    };
    let named: BTreeSet<AttachmentId> = statements
        .iter()
        .flat_map(|statement| &statement.params)
        .filter_map(|param| match param {
            SqlParam::KeySet { keys, .. } => Some(keys.attachment),
            _ => None,
        })
        .collect();
    let Some(&attachment) = named.first() else {
        return Ok(as_it_is(plan));
    };
    let Some(made) = keys.get(&attachment) else {
        return Err(CoreError::new(
            Sentence::Internal,
            "a plan binds a key set the core did not hand over",
        )
        .with_field("attachment", Detail::Scalar(Scalar::UInt32(attachment.0))));
    };
    let Some(rows) = dialect.key_rows(request, &made.schema, &made.batches, float_text)? else {
        return Ok(as_it_is(plan));
    };
    if named.len() > 1 {
        return Err(CoreError::new(
            Sentence::Internal,
            "a plan binds several key sets its dialect writes as text",
        ));
    }
    let mut fetches = Vec::new();
    let mut offset = 0;
    for chunk in chunks(&rows, dialect.key_budget()) {
        let statements = statements
            .iter()
            .map(|statement| chunked(dialect, statement, attachment, offset))
            .collect::<Result<Vec<_>, CoreError>>()?;
        fetches.push(BoundFetch {
            plan: FetchPlan::Sql {
                statements,
                expect: expect.clone(),
            },
            batches: BTreeMap::from([(TEXT, cell(format!("[{}]", chunk.join(",")))?)]),
        });
        offset += chunk.len();
    }
    Ok(BoundRequest { fetches })
}

/// `statement`, binding `attachment` as a text parameter that names the
/// fetch's text, read as a chunk whose first seed is at `offset`; a
/// statement binding no key set as it is.
fn chunked(
    dialect: &dyn SqlDialect,
    statement: &SqlStatement,
    attachment: AttachmentId,
    offset: usize,
) -> Result<SqlStatement, CoreError> {
    let mut binds = false;
    let params = statement
        .params
        .iter()
        .map(|param| match param {
            SqlParam::KeySet { keys, .. } if keys.attachment == attachment => {
                binds = true;
                SqlParam::Text { attachment: TEXT }
            }
            other => other.clone(),
        })
        .collect();
    let text = if binds {
        dialect.chunk_text(&statement.text, offset)?
    } else {
        statement.text.clone()
    };
    Ok(SqlStatement {
        dialect: statement.dialect.clone(),
        text,
        params,
    })
}

/// `rows` split as the old client splits a key set (the module's
/// documentation): all of them in one chunk where there is no budget.
fn chunks(rows: &[String], budget: Option<usize>) -> Vec<&[String]> {
    let Some(budget) = budget else {
        return vec![rows];
    };
    let mut found = Vec::new();
    let (mut start, mut size) = (0, 2);
    for (at, row) in rows.iter().enumerate() {
        let more = row.len() + 1;
        if at > start && size + more > budget {
            found.push(&rows[start..at]);
            (start, size) = (at, 2);
        }
        size += more;
    }
    if start < rows.len() || found.is_empty() {
        found.push(&rows[start..]);
    }
    found
}

/// `text` as a one-row, one-column text batch, the attachment a text
/// parameter binds.
fn cell(text: String) -> Result<KeyData, CoreError> {
    let schema = Arc::new(Schema::new(vec![Field::new(COLUMN, DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![text]))],
    )
    .map_err(|_| CoreError::new(Sentence::Internal, "a key set's text does not build"))?;
    Ok(KeyData {
        schema,
        batches: vec![batch],
    })
}

#[cfg(test)]
mod tests {
    use super::chunks;

    fn rows(sizes: &[usize]) -> Vec<String> {
        sizes.iter().map(|n| "x".repeat(*n)).collect()
    }

    fn lengths(found: &[&[String]]) -> Vec<usize> {
        found.iter().map(|c| c.len()).collect()
    }

    #[test]
    fn a_chunk_closes_before_the_seed_that_would_take_it_past_the_budget() {
        // [aaaa,bbbb] is 2 + 5 + 5 = 12 bytes; a third row takes it to 17.
        let rows = rows(&[4, 4, 4, 4, 4]);
        assert_eq!(lengths(&chunks(&rows, Some(12))), [2, 2, 1]);
        assert_eq!(lengths(&chunks(&rows, Some(11))), [1, 1, 1, 1, 1]);
        assert_eq!(lengths(&chunks(&rows, None)), [5]);
    }

    #[test]
    fn one_seed_past_the_budget_is_a_chunk_of_its_own_and_none_is_one_empty() {
        let rows = rows(&[1, 50, 1]);
        assert_eq!(lengths(&chunks(&rows, Some(10))), [1, 1, 1]);
        assert_eq!(lengths(&chunks(&[], Some(10))), [0]);
    }
}
