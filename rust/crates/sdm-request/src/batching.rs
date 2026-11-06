// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A call's predict rows taken apart, so a call past the most bytes one call
//! takes is asked as several smaller ones.
//!
//! A payload carries its context whole and the rows it scores beside it,
//! each row of its predict half's tables belonging to one of the rows
//! scored by its `instance_id`. [`predict_slice`] keeps the context as it
//! is and a run of the rows scored, with the rows of every related table
//! that belong to them, numbered as a payload of only those rows would be.
//! The context is sent with every call, so it is never split.

use std::collections::BTreeSet;
use std::ops::Range;

use sdm_contract::{CoreError, Sentence};
use sdm_wire::json::Value;

use crate::payload::INSTANCE_ID;

fn bad(reason: &str) -> CoreError {
    CoreError::new(
        Sentence::Internal,
        format!("a payload cannot be split: {reason}"),
    )
}

/// The rows a table of the payload holds: its column names and rows.
fn table_rows(table: &Value) -> Option<(Vec<&str>, &[Value])> {
    let Some(Value::Array(columns)) = table.get("columns") else {
        return None;
    };
    let Some(Value::Array(rows)) = table.get("rows") else {
        return None;
    };
    let names: Vec<&str> = columns.iter().filter_map(Value::as_text).collect();
    (names.len() == columns.len()).then_some((names, rows.as_slice()))
}

/// How many rows the payload scores: its predict half's instance rows.
pub fn predict_rows(payload: &Value) -> Option<usize> {
    let table = payload.get("predict")?.get("instance_table")?;
    table_rows(table).map(|(_, rows)| rows.len())
}

/// An instance id as a whole number, where it is one.
fn id_of(value: &Value) -> Option<i64> {
    match value {
        Value::Number(token) => token.parse::<i64>().ok(),
        _ => None,
    }
}

/// `table` with only the rows `keep` takes, each row's `instance_id` moved
/// `shift` places down where it is a whole number.
fn kept(
    table: &Value,
    keep: impl Fn(usize, &[Value]) -> bool,
    id_at: Option<usize>,
    shift: i64,
) -> Result<Value, CoreError> {
    let (_, rows) = table_rows(table).ok_or_else(|| bad("a table has no rows to take"))?;
    let rows = rows
        .iter()
        .enumerate()
        .filter_map(|(at, row)| {
            let Value::Array(cells) = row else {
                return Some(Err(bad("a row is not a list")));
            };
            if !keep(at, cells) {
                return None;
            }
            let mut cells = cells.clone();
            if shift != 0
                && let Some(at) = id_at
                && let Some(id) = cells.get(at).and_then(id_of)
            {
                cells[at] = Value::Number((id - shift).to_string());
            }
            Some(Ok(Value::Array(cells)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut table = table.clone();
    table.set("rows", Value::Array(rows));
    Ok(table)
}

/// `payload` scoring only its predict rows `range`, in order, with the rows
/// of each related table of its predict half that belong to them, and its
/// context whole. The rows kept are numbered from where the payload's own
/// were (`instance_id`), as a payload built of only those rows numbers
/// them, and its metadata counts them (`num_prediction_examples`).
pub fn predict_slice(payload: &Value, range: Range<usize>) -> Result<Value, CoreError> {
    let predict = payload
        .get("predict")
        .ok_or_else(|| bad("there is no predict half"))?;
    let instance = predict
        .get("instance_table")
        .ok_or_else(|| bad("there is no instance table"))?;
    let (names, rows) = table_rows(instance).ok_or_else(|| bad("the instance table is unread"))?;
    if range.end > rows.len() || range.start > range.end {
        return Err(bad("the rows asked for are not there"));
    }
    let id_at = names.iter().position(|name| *name == INSTANCE_ID);
    // The ids of the rows kept, and how far they move down: a run of rows
    // numbered on from the first row of the half is numbered from it again.
    let ids = |row: &Value| match (row, id_at) {
        (Value::Array(cells), Some(at)) => cells.get(at).and_then(id_of),
        _ => None,
    };
    let kept_ids: BTreeSet<i64> = rows[range.clone()].iter().filter_map(ids).collect();
    let shift = match (
        rows.first().and_then(ids),
        rows.get(range.start).and_then(ids),
    ) {
        (Some(first), Some(start)) if kept_ids.len() == range.len() => start - first,
        _ => 0,
    };
    let mut half = predict.clone();
    half.set(
        "instance_table",
        kept(instance, |at, _| range.contains(&at), id_at, shift)?,
    );
    if let Some(Value::Object(related)) = predict.get("related_tables") {
        let mut tables = Vec::with_capacity(related.len());
        for (name, table) in related {
            let (columns, _) = table_rows(table).ok_or_else(|| bad("a related table is unread"))?;
            let at = columns.iter().position(|column| *column == INSTANCE_ID);
            let Some(at) = at.filter(|_| id_at.is_some()) else {
                // A related table whose rows name no instance belongs to no
                // one row scored: kept whole.
                tables.push((name.clone(), table.clone()));
                continue;
            };
            let belongs = |_: usize, cells: &[Value]| {
                cells
                    .get(at)
                    .and_then(id_of)
                    .is_some_and(|id| kept_ids.contains(&id))
            };
            tables.push((name.clone(), kept(table, belongs, Some(at), shift)?));
        }
        half.set("related_tables", Value::Object(tables));
    }
    let mut sliced = payload.clone();
    sliced.set("predict", half);
    if let Some(metadata @ Value::Object(_)) = payload.get("metadata")
        && metadata.get("num_prediction_examples").is_some()
    {
        let mut metadata = metadata.clone();
        metadata.set(
            "num_prediction_examples",
            Value::Number(range.len().to_string()),
        );
        sliced.set("metadata", metadata);
    }
    Ok(sliced)
}

/// The calls `payload` is asked as so that each fits, in order, each its
/// rows' range: the whole where it fits (`fits`, given the payload a call
/// would carry), else halves, each split again until it fits. A call of one
/// row that does not fit cannot be split, and is refused by `refuse`, given
/// that one row's payload: the context is sent with every call.
pub fn fitted(
    payload: &Value,
    fits: &mut dyn FnMut(&Value) -> Result<bool, CoreError>,
    refuse: &mut dyn FnMut(&Value) -> CoreError,
) -> Result<Vec<(Range<usize>, Value)>, CoreError> {
    if fits(payload)? {
        let rows = predict_rows(payload).unwrap_or(0);
        return Ok(vec![(0..rows, payload.clone())]);
    }
    let rows = predict_rows(payload).ok_or_else(|| bad("the rows scored are unread"))?;
    let mut done = Vec::new();
    // Ranges still to fit, the first next.
    let mut pending: Vec<Range<usize>> = Vec::new();
    pending.push(0..rows);
    while let Some(range) = pending.pop() {
        let sliced = predict_slice(payload, range.clone())?;
        if range.len() != rows && fits(&sliced)? {
            done.push((range, sliced));
            continue;
        }
        if range.len() <= 1 {
            return Err(refuse(&sliced));
        }
        let middle = range.start + range.len() / 2;
        pending.push(middle..range.end);
        pending.push(range.start..middle);
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_wire::json;

    fn payload() -> Value {
        json::read(
            br#"{
  "model": "kumo-relational",
  "context": {"instance_table": {"format": "rows", "columns": ["instance_id", "TARGET"], "rows": [[0, 1], [1, 2]]},
              "related_tables": {"users": {"format": "rows", "columns": ["instance_id", "user_id"], "rows": [[0, 10], [1, 11]]}}},
  "predict": {"instance_table": {"format": "rows", "columns": ["instance_id", "__kumo_entity_ref_0"], "rows": [[2, 20], [3, 21], [4, 22], [5, 23]]},
              "related_tables": {"users": {"format": "rows", "columns": ["instance_id", "user_id"], "rows": [[2, 20], [3, 21], [4, 22], [5, 23], [3, 99]]},
                                 "plain": {"format": "rows", "columns": ["x"], "rows": [[7]]}}},
  "metadata": {"num_context_examples": 2, "num_prediction_examples": 4}
}"#,
        )
        .unwrap()
    }

    fn rows(value: &Value, path: &[&str]) -> String {
        let mut at = value;
        for key in path {
            at = at.get(key).unwrap();
        }
        String::from_utf8(at.get("rows").unwrap().to_bytes()).unwrap()
    }

    #[test]
    fn a_slice_keeps_the_context_and_the_rows_of_its_instances_renumbered() {
        let whole = payload();
        let sliced = predict_slice(&whole, 1..3).unwrap();
        assert_eq!(sliced.get("context"), whole.get("context"));
        assert_eq!(
            rows(&sliced, &["predict", "instance_table"]),
            "[[2,21],[3,22]]"
        );
        assert_eq!(
            rows(&sliced, &["predict", "related_tables", "users"]),
            "[[2,21],[3,22],[2,99]]"
        );
        assert_eq!(
            rows(&sliced, &["predict", "related_tables", "plain"]),
            "[[7]]"
        );
        assert_eq!(
            sliced
                .get("metadata")
                .unwrap()
                .get("num_prediction_examples"),
            Some(&Value::Number("2".into()))
        );
        // The first run is numbered as it was.
        let first = predict_slice(&whole, 0..1).unwrap();
        assert_eq!(rows(&first, &["predict", "instance_table"]), "[[2,20]]");
        assert_eq!(predict_slice(&whole, 0..4).unwrap(), whole);
        assert!(predict_slice(&whole, 2..5).is_err());
    }

    fn size(value: &Value) -> usize {
        value.to_bytes().len()
    }

    #[test]
    fn a_call_that_fits_is_one_call_and_one_past_is_halved_until_each_fits() {
        let whole = payload();
        let one = size(&predict_slice(&whole, 0..1).unwrap());
        let two = size(&predict_slice(&whole, 0..2).unwrap());
        let all = size(&whole);
        let mut refuse = |_: &Value| CoreError::new(Sentence::Internal, "refused");
        let calls = fitted(&whole, &mut |p| Ok(size(p) <= all), &mut refuse).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0..4);
        // One byte under the whole: two calls of two rows.
        let calls = fitted(&whole, &mut |p| Ok(size(p) < all), &mut refuse).unwrap();
        let ranges: Vec<_> = calls.iter().map(|(r, _)| r.clone()).collect();
        assert_eq!(ranges, [0..2, 2..4]);
        // Exactly two rows' worth: still two calls; one byte under it: four.
        let ranges: Vec<_> = fitted(&whole, &mut |p| Ok(size(p) <= two), &mut refuse)
            .unwrap()
            .into_iter()
            .map(|(r, _)| r)
            .collect();
        assert_eq!(ranges, [0..2, 2..4]);
        let ranges: Vec<_> = fitted(&whole, &mut |p| Ok(size(p) < two), &mut refuse)
            .unwrap()
            .into_iter()
            .map(|(r, _)| r)
            .collect();
        // The last two rows carry fewer related rows than the first two,
        // so together they fit where the first two do not.
        assert_eq!(ranges, [0..1, 1..2, 2..4]);
        // Under one row's worth, the context alone does not fit.
        let mut refused_rows = None;
        let error = fitted(&whole, &mut |p| Ok(size(p) < one), &mut |p| {
            refused_rows = predict_rows(p);
            CoreError::new(Sentence::Internal, "refused")
        })
        .unwrap_err();
        assert_eq!(error.message, "refused");
        assert_eq!(refused_rows, Some(1));
    }
}
