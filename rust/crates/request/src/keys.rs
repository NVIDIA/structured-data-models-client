// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/wire/keys.py`: which rows an encoded request is about, keyed
//! one way whichever side of the wire they were read from.

use std::cell::Cell;

use crate::pyops::TaggedValue;
use sdm_schema::pyval::PyVal;
use serde_json::Value;

use crate::frames::Table;
use crate::num::Dec;
use crate::pyops::{HostProtocols, InstanceFamily, is_instance, mapping_get, truth_with};
use crate::pyops::{decimal_value, is_real_type, py_float, py_str};
use crate::series::is_na;

const ANCHOR: &str = "__kumo_anchor_time";
const ENTITY_REF: &str = "__kumo_entity_ref";
const INSTANCE_ID: &str = "instance_id";

thread_local! {
    /// `_UNIDENTIFIED`: counts the rows that name no entity.
    static UNIDENTIFIED: Cell<u64> = const { Cell::new(0) };
}

/// A key as a name: `repr` of the tuple of its parts' spellings, or a fresh
/// name for a row that names no entity.
pub type Named = String;

/// `_anchor_column(payload)`.
pub fn anchor_column(payload: &Value) -> String {
    payload
        .py_get("task")
        .and_then(|t| t.py_get("anchor_time_column"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(ANCHOR)
        .to_string()
}

/// `_is_missing(value)` of keys.py: names no entity.
fn is_missing(value: &PyVal) -> bool {
    matches!(value, PyVal::None) || is_na(value)
}

/// `_one_spelling(value)`.
pub fn one_spelling(value: &PyVal) -> String {
    let is_number = !matches!(value, PyVal::Bool(_))
        && (is_real_type(value) || matches!(value, PyVal::Decimal(_)));
    if !is_number {
        return py_str(value);
    }
    let exact = match decimal_value(value) {
        Some(d) => Dec::parse(&d.to_sci()),
        None => match crate::pyops::item(value) {
            PyVal::Float(x) => {
                if !x.is_finite() {
                    return py_str(value);
                }
                Dec::parse(&crate::pyops::float_text(x))
            }
            other => Dec::parse(&py_str(&other)),
        },
    };
    let Some(exact) = exact else {
        return py_str(value);
    };
    if !exact.is_finite() {
        return py_str(value);
    }
    if exact.is_integral() {
        return exact.trunc().format_f();
    }
    exact.normalize().to_sci()
}

/// `_named_by(values)`.
pub fn named_by(values: &[PyVal]) -> Named {
    if values.iter().any(is_missing) {
        let n = UNIDENTIFIED.with(|c| {
            let n = c.get();
            c.set(n + 1);
            n
        });
        return format!("<unidentified {n}>");
    }
    let parts: Vec<String> = values
        .iter()
        .map(|v| sdm_pql::pyrepr::str_repr(&one_spelling(v)))
        .collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// A JSON cell as the Python value `json.loads` gives.
pub fn json_cell(value: &Value) -> PyVal {
    match value {
        Value::Null => PyVal::None,
        Value::Bool(b) => PyVal::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => PyVal::Int(i128::from(i)),
            None => match n.as_u64() {
                Some(u) => PyVal::Int(i128::from(u)),
                None => PyVal::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
        },
        Value::String(s) => PyVal::Str(s.clone()),
        Value::Array(items) => PyVal::List(items.iter().map(json_cell).collect()),
        Value::Object(map) => PyVal::Dict(
            map.iter()
                .map(|(k, v)| (PyVal::Str(k.clone()), json_cell(v)))
                .collect(),
        ),
    }
}

/// `rows_in(payload, half, with_anchor=...)`.
pub fn rows_in(
    payload: &Value,
    half: &str,
    with_anchor: bool,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<Vec<Named>> {
    if !is_instance(payload, InstanceFamily::Dict, protocols)? {
        return Ok(Vec::new());
    }
    let or_empty = |value: Value| -> crate::error::R<Value> {
        if truth_with(
            &PyVal::decode(&value).map_err(crate::error::value_error)?,
            protocols,
        )? {
            Ok(value)
        } else {
            Ok(serde_json::json!({}))
        }
    };
    let section = or_empty(mapping_get(payload, half, &Value::Null, protocols)?)?;
    let table = or_empty(mapping_get(
        &section,
        "instance_table",
        &Value::Null,
        protocols,
    )?)?;
    let columns = mapping_get(&table, "columns", &Value::Null, protocols)?;
    let rows = mapping_get(&table, "rows", &Value::Null, protocols)?;
    let (Some(columns), Some(rows)) = (columns.as_array(), rows.as_array()) else {
        return Ok(Vec::new());
    };
    let names: Vec<String> = columns
        .iter()
        .map(|c| c.as_str().map_or_else(|| c.to_string(), str::to_string))
        .collect();
    let wanted: Vec<String> = match names.iter().find(|n| n.starts_with(ENTITY_REF)) {
        Some(n) => vec![n.clone()],
        None => {
            let schema = or_empty(mapping_get(payload, "schema", &Value::Null, protocols)?)?;
            let instance = or_empty(mapping_get(
                &schema,
                "instance_table",
                &Value::Null,
                protocols,
            )?)?;
            match mapping_get(&instance, "primary_key", &Value::Null, protocols)? {
                Value::String(key) => vec![key],
                Value::Array(ks) => ks
                    .iter()
                    .filter_map(|k| k.as_str().map(str::to_string))
                    .collect(),
                _ => return Ok(Vec::new()),
            }
        }
    };
    if wanted.iter().any(|w| !names.contains(w)) {
        return Ok(Vec::new());
    }
    let mut at: Vec<usize> = wanted
        .iter()
        .map(|w| names.iter().position(|n| n == w).unwrap_or(0))
        .collect();
    if with_anchor {
        let task = mapping_get(&or_empty(payload.clone())?, "task", &Value::Null, protocols)?;
        let named = if is_instance(&task, InstanceFamily::Mapping, protocols)? {
            mapping_get(&task, "anchor_time_column", &Value::Null, protocols)?
        } else {
            Value::Null
        };
        let named = PyVal::decode(&named).map_err(crate::error::value_error)?;
        let anchor = if truth_with(&named, protocols)? {
            py_str(&named)
        } else {
            ANCHOR.to_string()
        };
        if let Some(i) = names.iter().position(|n| *n == anchor) {
            at.push(i);
        }
    }
    let top = at.iter().copied().max().unwrap_or(0);
    Ok(rows
        .iter()
        .filter_map(Value::as_array)
        .filter(|row| row.len() > top)
        .map(|row| named_by(&at.iter().map(|&i| json_cell(&row[i])).collect::<Vec<_>>()))
        .collect())
}

/// `keys_in(payload, half, with_anchor=...)`.
pub fn keys_in(
    payload: &Value,
    half: &str,
    with_anchor: bool,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<Vec<Named>> {
    let mut out: Vec<Named> = Vec::new();
    for key in rows_in(payload, half, with_anchor, protocols)? {
        if !out.contains(&key) {
            out.push(key);
        }
    }
    Ok(out)
}

/// `key_dtypes(table, key, widened_by)`.
pub fn key_dtypes(
    table: &sdm_schema::client::Table,
    key: &[String],
    widened_by: Option<&Table>,
) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();
    for name in key {
        let column = table.col(name)?;
        let mut dtype = if column.untyped {
            String::new()
        } else {
            column.dtype.clone()
        };
        if let Some(frame) = widened_by
            && let Some(series) = frame.series(name)
        {
            let raw = crate::tabular_adapter::raw_dtype_of(&series, Some(&dtype));
            dtype = crate::wire::widen_tfm_dtype(&[dtype.clone(), raw]);
        }
        out.push((
            name.clone(),
            if dtype.is_empty() {
                "string".to_string()
            } else {
                dtype
            },
        ));
    }
    Some(out)
}

/// `keys_from_a_frame(frame, key, dtypes)`.
pub fn keys_from_a_frame(
    frame: &Table,
    key: &[String],
    dtypes: Option<&[(String, String)]>,
) -> Vec<Named> {
    let Some(dtypes) = dtypes else {
        return Vec::new();
    };
    let mut columns = Vec::new();
    for name in key {
        let Some(series) = frame.series(name) else {
            return Vec::new();
        };
        let Some((_, dtype)) = dtypes.iter().find(|(n, _)| n == name) else {
            return Vec::new();
        };
        match crate::wire::serialize_column(&series, dtype) {
            Ok(cells) => columns.push(cells),
            Err(_) => return Vec::new(),
        }
    }
    (0..frame.len())
        .map(|i| named_by(&columns.iter().map(|c| json_cell(&c[i])).collect::<Vec<_>>()))
        .collect()
}

/// `without_the_keys(payload, keys)`: the context rows naming one of
/// `keys` taken out, with the related rows hanging off them; how many.
pub fn without_the_keys(payload: &mut Value, keys: &[Named]) -> usize {
    if keys.is_empty() {
        return 0;
    }
    let primary = payload
        .py_get("schema")
        .and_then(|s| s.py_get("instance_table"))
        .and_then(|t| t.py_get("primary_key"))
        .cloned();
    let Some(context) = crate::pyops::field_mut(payload, "context") else {
        return 0;
    };
    let Some(table) = crate::pyops::field_mut(context, "instance_table") else {
        return 0;
    };
    let names: Vec<String> = match table.py_get("columns").and_then(Value::as_array) {
        Some(c) => c
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect(),
        None => return 0,
    };
    let Some(rows) = table.py_get("rows").and_then(Value::as_array).cloned() else {
        return 0;
    };
    let wanted: Vec<String> = match names.iter().find(|n| n.starts_with(ENTITY_REF)) {
        Some(n) => vec![n.clone()],
        None => match primary {
            Some(Value::String(k)) => vec![k],
            Some(Value::Array(ks)) => ks
                .iter()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect(),
            _ => return 0,
        },
    };
    if wanted.iter().any(|w| !names.contains(w)) {
        return 0;
    }
    let at: Vec<usize> = wanted
        .iter()
        .map(|w| names.iter().position(|n| n == w).unwrap_or(0))
        .collect();
    let top = at.iter().copied().max().unwrap_or(0);
    let instance = names.iter().position(|n| n == INSTANCE_ID);
    let mut going = Vec::new();
    let mut leaving: Vec<Value> = Vec::new();
    for row in &rows {
        let Some(cells) = row.as_array().filter(|r| r.len() > top) else {
            going.push(row.clone());
            continue;
        };
        let named = named_by(&at.iter().map(|&i| json_cell(&cells[i])).collect::<Vec<_>>());
        if keys.contains(&named) {
            if let Some(i) = instance
                && cells.len() > i
            {
                leaving.push(cells[i].clone());
            }
            continue;
        }
        going.push(row.clone());
    }
    let removed = rows.len() - going.len();
    if removed == 0 {
        return 0;
    }
    crate::pyops::set_field(table, "rows", Some(Value::Array(going)));
    if let Some(related) = crate::pyops::field_mut(context, "related_tables") {
        for table in crate::pyops::mapping_values_mut(related) {
            drop_instances(table, &leaving);
        }
    }
    removed
}

/// `_drop_instances(table, instances)`.
pub fn drop_instances(table: &mut Value, instances: &[Value]) {
    if instances.is_empty() {
        return;
    }
    let Some(at) = table
        .py_get("columns")
        .and_then(Value::as_array)
        .and_then(|c| c.iter().position(|n| n.as_str() == Some(INSTANCE_ID)))
    else {
        return;
    };
    let Some(rows) = table.py_get("rows").and_then(Value::as_array) else {
        return;
    };
    let kept: Vec<Value> = rows
        .iter()
        .filter(|row| {
            !row.as_array()
                .is_some_and(|r| r.len() > at && instances.iter().any(|i| same_key(i, &r[at])))
        })
        .cloned()
        .collect();
    crate::pyops::set_field(table, "rows", Some(Value::Array(kept)));
}

/// Whether two JSON cells are one key, as Python's `in` on a set compares
/// them: `1 == 1.0`.
fn same_key(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.as_f64() == y.as_f64() && (x.as_i64() == y.as_i64() || x.is_f64() || y.is_f64())
        }
        _ => a == b,
    }
}

/// Whether a value is a number `_one_spelling` reads as one.
pub fn float_of(value: &PyVal) -> Option<f64> {
    py_float(value).ok()
}
