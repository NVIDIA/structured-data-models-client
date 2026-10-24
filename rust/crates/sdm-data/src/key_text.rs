// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A key set bound as JSON text (T3.28, design S4): the one binding of a
//! dialect whose statement reads its key set from a text parameter
//! (Databricks's `posexplode(from_json(?, ...))`, Postgres's
//! `jsonb_array_elements($1::jsonb)`), written here once over the
//! dialect's hooks ([`SqlDialect::key_rows`], [`SqlDialect::key_budget`]),
//! so no provider writes or splits one.
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
//! reference, and its statements rendered for a chunk whose first seed is
//! at that offset of the key set (`sql::plan`), never found in rendered
//! text. The core installs
//! each batch as one of its run's attachments, never in the control
//! message, sends the fetches in one step and joins their answers in
//! order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampNanosecondType;
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Field as ArrowField, Schema, SchemaRef};
use sdm_contract::data::{
    AttachmentId, BoundFetch, BoundRequest, DataRequest, FetchPlan, KeyData, SEED_BATCH, SEED_KEY,
    SqlParam, SqlStatement,
};
use sdm_contract::types::ColumnType;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::json;
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
    render: &dyn Fn(usize) -> Result<FetchPlan, CoreError>,
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
    // The parameters the key set's statement binds after its text, read
    // from the whole key set (Snowflake's window bounds, `SNOW/sampler.py:
    // 350-360`).
    let after = dialect.key_bounds(request, &made.schema, &made.batches)?;
    let separator = dialect.key_separator();
    let mut fetches = Vec::new();
    let mut offset = 0;
    for chunk in chunks(&rows, dialect.key_budget()) {
        // A later chunk's statements rendered with its seeds' first batch.
        let rendered;
        let statements = if offset == 0 {
            statements
        } else {
            rendered = render(offset)?;
            match &rendered {
                FetchPlan::Sql { statements, .. } => statements,
                FetchPlan::Host { .. } => {
                    return Err(CoreError::new(
                        Sentence::Internal,
                        "a chunk of a SQL plan rendered as a host's",
                    ));
                }
            }
        };
        let statements = statements
            .iter()
            .map(|statement| chunked(statement, attachment, &after))
            .collect();
        fetches.push(BoundFetch {
            plan: FetchPlan::Sql {
                statements,
                expect: expect.clone(),
            },
            batches: BTreeMap::from([(TEXT, cell(format!("[{}]", chunk.join(separator)))?)]),
        });
        offset += chunk.len();
    }
    Ok(BoundRequest { fetches })
}

/// `statement` binding `attachment` as a text parameter that names the
/// fetch's text, `after` bound after it; a statement binding no key set as
/// it is.
fn chunked(statement: &SqlStatement, attachment: AttachmentId, after: &[SqlParam]) -> SqlStatement {
    let mut params = Vec::with_capacity(statement.params.len() + after.len());
    for param in &statement.params {
        match param {
            SqlParam::KeySet { keys, .. } if keys.attachment == attachment => {
                params.push(SqlParam::Text { attachment: TEXT });
                params.extend(after.iter().cloned());
            }
            other => params.push(other.clone()),
        }
    }
    SqlStatement {
        dialect: statement.dialect.clone(),
        text: statement.text.clone(),
        params,
    }
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

// --- a key set's cells, as the old client writes them -------------------------

/// One cell of a key set [`coerced`] for its JSON, as JSON: an integer's
/// digits, a double as `float_text` writes it, a text's string, ASCII
/// only, `null`.
pub(crate) fn json_cell(column: &ArrayRef, row: usize, float_text: fn(f64) -> String) -> String {
    if column.is_null(row) {
        return "null".to_owned();
    }
    if let Some(integers) = column.as_primitive_opt::<arrow_array::types::Int64Type>() {
        return integers.value(row).to_string();
    }
    if let Some(doubles) = column.as_primitive_opt::<arrow_array::types::Float64Type>() {
        return json::number(doubles.value(row), float_text);
    }
    json::string(column.as_string::<i32>().value(row), true)
}

/// A key set as this dialect writes it (the module's documentation): the
/// batch left out, the key coerced as the reference coerces it for its
/// JSON (`_coerce_id`), each seed time its wall clock's text to the
/// second.
pub(crate) fn coerced(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(SchemaRef, Vec<RecordBatch>), CoreError> {
    let unbound = || {
        CoreError::new(
            Sentence::Internal,
            "a key set holds a column this dialect does not bind",
        )
    };
    let mut fields = Vec::new();
    for field in schema.fields() {
        if field.name() == SEED_BATCH {
            continue;
        }
        let bound = if field.name() == SEED_KEY {
            match arrow_key_type(field.data_type()) {
                KeyType::Integer => DataType::Int64,
                KeyType::Double => DataType::Float64,
                KeyType::Text => DataType::Utf8,
            }
        } else {
            match field.data_type() {
                DataType::Timestamp(_, None) => DataType::Utf8,
                _ => return Err(unbound()),
            }
        };
        fields.push(ArrowField::new(field.name(), bound, true));
    }
    let out = Arc::new(Schema::new(fields));
    let mut rewritten = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut columns: Vec<ArrayRef> = Vec::new();
        for (at, field) in schema.fields().iter().enumerate() {
            if field.name() == SEED_BATCH {
                continue;
            }
            let column = batch.column(at);
            let bound: ArrayRef = if field.name() == SEED_KEY {
                match arrow_key_type(field.data_type()) {
                    KeyType::Integer => arrow_cast(column, &DataType::Int64)?,
                    KeyType::Double => arrow_cast(column, &DataType::Float64)?,
                    KeyType::Text => arrow_cast(column, &DataType::Utf8)?,
                }
            } else {
                let times = arrow_cast(
                    column,
                    &DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                )?;
                let times = times.as_primitive::<TimestampNanosecondType>();
                Arc::new(
                    times
                        .iter()
                        .map(|t| t.map(wall_clock_text))
                        .collect::<StringArray>(),
                )
            };
            columns.push(bound);
        }
        rewritten.push(RecordBatch::try_new(Arc::clone(&out), columns).map_err(|_| unbound())?);
    }
    Ok((out, rewritten))
}

/// How a key is written in a key set's JSON, as the old client coerces
/// one (`_coerce_id`) and casts it in SQL: an integer (or a decimal of
/// scale 0), a double (a float, or a decimal with a fraction), or text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyType {
    Integer,
    Double,
    Text,
}

/// The [`KeyType`] of a key of `column_type`.
pub(crate) fn key_type(column_type: &ColumnType) -> KeyType {
    use ColumnType as T;
    match column_type {
        T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::Uint8 | T::Uint16 | T::Uint32 | T::Uint64 => {
            KeyType::Integer
        }
        T::Decimal128 { scale: 0, .. } | T::Decimal256 { scale: 0, .. } => KeyType::Integer,
        T::Float32 | T::Float64 | T::Decimal128 { .. } | T::Decimal256 { .. } => KeyType::Double,
        _ => KeyType::Text,
    }
}

/// The [`KeyType`] of a key of Arrow type `data_type`.
fn arrow_key_type(data_type: &DataType) -> KeyType {
    match ColumnType::from_arrow("key", data_type) {
        Ok(column_type) => key_type(&column_type),
        Err(_) => KeyType::Text,
    }
}

fn arrow_cast(column: &ArrayRef, to: &DataType) -> Result<ArrayRef, CoreError> {
    if column.data_type() == to {
        return Ok(Arc::clone(column));
    }
    let converted: ArrayRef = match to {
        DataType::Int64 => Arc::new(
            (0..column.len())
                .map(|row| integer_at(column, row))
                .collect::<Option<Vec<_>>>()
                .map(Int64Array::from)
                .ok_or_else(|| {
                    CoreError::new(
                        Sentence::Internal,
                        "a key is not an integer the binding writes",
                    )
                })?,
        ),
        DataType::Float64 => Arc::new(
            (0..column.len())
                .map(|row| float_at(column, row))
                .collect::<Float64Array>(),
        ),
        _ => Arc::new(
            (0..column.len())
                .map(|row| text_cell(column, row))
                .collect::<Option<Vec<_>>>()
                .map(StringArray::from)
                .ok_or_else(|| {
                    CoreError::new(
                        Sentence::KeySetTypeUnbound,
                        "a key of this type is not one the binding writes as text",
                    )
                })?,
        ),
    };
    Ok(converted)
}

/// Row `row` of an integer or scale-0 decimal column as an `i64`; `Some`
/// of `None` for a null.
fn integer_at(column: &ArrayRef, row: usize) -> Option<Option<i64>> {
    use arrow_array::types as t;
    if column.is_null(row) {
        return Some(None);
    }
    let value: i128 = match column.data_type() {
        DataType::Int8 => column.as_primitive::<t::Int8Type>().value(row).into(),
        DataType::Int16 => column.as_primitive::<t::Int16Type>().value(row).into(),
        DataType::Int32 => column.as_primitive::<t::Int32Type>().value(row).into(),
        DataType::Int64 => column.as_primitive::<t::Int64Type>().value(row).into(),
        DataType::UInt8 => column.as_primitive::<t::UInt8Type>().value(row).into(),
        DataType::UInt16 => column.as_primitive::<t::UInt16Type>().value(row).into(),
        DataType::UInt32 => column.as_primitive::<t::UInt32Type>().value(row).into(),
        DataType::UInt64 => column.as_primitive::<t::UInt64Type>().value(row).into(),
        DataType::Decimal128(_, 0) => column.as_primitive::<t::Decimal128Type>().value(row),
        _ => return None,
    };
    i64::try_from(value).ok().map(Some)
}

/// Row `row` of a numeric column as a double, as the reference's `float`
/// of it.
#[allow(
    clippy::cast_precision_loss,
    reason = "a number to the nearest double, as the reference converts it"
)]
fn float_at(column: &ArrayRef, row: usize) -> Option<f64> {
    use arrow_array::types as t;
    if column.is_null(row) {
        return None;
    }
    Some(match column.data_type() {
        DataType::Float16 => column.as_primitive::<t::Float16Type>().value(row).to_f64(),
        DataType::Float32 => f64::from(column.as_primitive::<t::Float32Type>().value(row)),
        DataType::Float64 => column.as_primitive::<t::Float64Type>().value(row),
        DataType::Decimal128(_, scale) => {
            let value = column.as_primitive::<t::Decimal128Type>().value(row);
            format!("{value}e-{scale}").parse().ok()?
        }
        _ => integer_at(column, row).flatten()? as f64,
    })
}

/// Row `row` of a text column; `Some(None)` for a null, `None` for a
/// column of another type, which no key this dialect binds as text holds.
fn text_cell(column: &ArrayRef, row: usize) -> Option<Option<String>> {
    if column.is_null(row) {
        return Some(None);
    }
    if let Some(strings) = column.as_string_opt::<i32>() {
        return Some(Some(strings.value(row).to_owned()));
    }
    if let Some(strings) = column.as_string_opt::<i64>() {
        return Some(Some(strings.value(row).to_owned()));
    }
    if let Some(strings) = column.as_string_view_opt() {
        return Some(Some(strings.value(row).to_owned()));
    }
    None
}

/// A wall clock in nanoseconds since the epoch as its text to the second,
/// `%Y-%m-%d %H:%M:%S`, as the reference writes a bound.
pub(crate) fn wall_clock_text(nanos: i64) -> String {
    let seconds = nanos.div_euclid(1_000_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        of_day / 3600,
        (of_day % 3600) / 60,
        of_day % 60
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (
        year,
        u32::try_from(m).unwrap_or(1),
        u32::try_from(d).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::{chunks, wall_clock_text};

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

    #[test]
    fn a_wall_clock_is_written_to_the_second() {
        assert_eq!(wall_clock_text(0), "1970-01-01 00:00:00");
        assert_eq!(
            wall_clock_text(1_709_251_199_999_999_999),
            "2024-02-29 23:59:59"
        );
        assert_eq!(wall_clock_text(-1), "1969-12-31 23:59:59");
    }
}
