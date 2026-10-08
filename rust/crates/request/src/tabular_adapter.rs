// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/adapters/tabular.py`: the single-table envelope, and reading
//! the answer back onto the rows asked about.

use crate::pyops::TaggedValue;
use sdm_schema::client::Schema;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::{R, Warnings, invalid};
use crate::frames::Table;
use crate::models::{KUMO_RELATIONAL, KUMO_TABULAR};
use crate::request::{Request, column_of, is_classification, schema_table_or_none};
use crate::result::Item;
use crate::series::Series;
use crate::wire::{UNTYPED, infer_tfm_dtype, widen_tfm_dtype};

const SESSION_PINNED: [&str; 4] = ["model", "task", "schema", "context"];
const SESSION_SCORING: [&str; 4] = ["predict", "output", "inference", "metadata"];
const MAX_ID_LENGTH: usize = 1024;

/// `_DEFAULT_OUTPUTS[kind]`.
pub fn default_output(kind: &str) -> Vec<String> {
    match kind {
        "classification" => vec!["prediction".into(), "probabilities".into()],
        _ => vec!["prediction".into()],
    }
}

/// `_raw_dtype_of(series, declared)`.
pub fn raw_dtype_of(series: &Series, declared: Option<&str>) -> String {
    if let Some(declared) = declared
        && declared.starts_with("int")
        && series.dtype.is_float()
        && series.cells.iter().any(crate::series::is_na)
    {
        let present: Vec<f64> = series
            .cells
            .iter()
            .filter(|c| !crate::series::is_na(c))
            .filter_map(|c| match crate::pyops::item(c) {
                PyVal::Float(x) => Some(x),
                _ => None,
            })
            .collect();
        if present.iter().all(|x| x.trunc() == *x) {
            return declared.to_string();
        }
    }
    infer_tfm_dtype(series)
}

/// `_MEANING_OF`.
fn meaning_of(dtype: &str) -> &'static str {
    match dtype {
        "int64" | "int32" | "float64" | "float32" => "numerical",
        _ => "categorical",
    }
}

/// `_coherent_stype(stype, dtype, column)`.
fn coherent_stype(stype: &str, dtype: &str, column: &sdm_schema::client::Col) -> String {
    if sdm_schema::client::supports_dtype(stype, dtype) || column.stype_declared {
        return stype.to_string();
    }
    if dtype.starts_with("timestamp") {
        return "timestamp".to_string();
    }
    meaning_of(dtype).to_string()
}

/// `TabularAdapter._stype_for(request, inferred, column)`.
fn stype_for(request: &Request, inferred: &str, column: &str) -> String {
    if inferred == "text" {
        return "categorical".to_string();
    }
    if !matches!(&request.task.target.column_name, PyVal::Str(target) if target == column) {
        return inferred.to_string();
    }
    if request.task.kind.as_str().is_some_and(is_classification) {
        return "categorical".to_string();
    }
    if inferred == "categorical" {
        "numerical".to_string()
    } else {
        inferred.to_string()
    }
}

/// `encode_table(frame, dtypes)`.
pub fn encode_table(frame: &Table, dtypes: &[(String, String)]) -> R<Value> {
    let columns = frame.columns();
    if columns.is_empty() {
        return Ok(json!({
            "format": "arrays",
            "columns": [],
            "rows": (0..frame.len()).map(|_| json!([])).collect::<Vec<_>>(),
        }));
    }
    let mut serialized = Vec::new();
    for name in &columns {
        let series = frame.series(name).unwrap_or_else(|| {
            Series::new(Some(name.clone()), crate::series::Dtype::Object, Vec::new())
        });
        let dtype = dtypes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| crate::error::builtin("KeyError", sdm_pql::pyrepr::str_repr(name)))?;
        serialized.push(crate::wire::serialize_column(&series, &dtype)?);
    }
    let rows: Vec<Value> = (0..frame.len())
        .map(|i| Value::Array(serialized.iter().map(|c| c[i].clone()).collect()))
        .collect();
    Ok(json!({"format": "arrays", "columns": columns, "rows": rows}))
}

/// A request id: `sdm-` and 12 lowercase hexadecimal digits.
pub fn request_id() -> String {
    let mut bytes = [0u8; 6];
    if getrandom::fill(&mut bytes).is_err() {
        bytes = [0; 6];
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("sdm-{hex}")
}

/// `TabularAdapter._envelope(request, outputs, pinned_dtypes)`.
pub fn envelope(
    request: &Request,
    schema: &mut Schema,
    outputs: &[String],
    pinned: Option<&[(String, String)]>,
    said: &mut Warnings,
) -> R<Value> {
    if request.payloads().is_some() {
        return Err(crate::error::unsupported(
            format!(
                "this request was gathered for {KUMO_RELATIONAL} and carries the sampled \
                 payloads that model reads. Gather it with materialize(..., \
                 model='{KUMO_TABULAR}') to ask this one"
            ),
            Map::new(),
        ));
    }
    let mut checked = sdm_schema::client::Said::default();
    schema
        .check_consistent(&mut checked)
        .map_err(crate::request::from_schema)?;
    for warning in checked.warnings {
        said.push(sdm_protocol::Diagnostic::warning(
            &warning.category,
            warning.message,
        ));
    }
    let table = schema_table_or_none(
        schema,
        request.entity_table.as_ref().and_then(PyVal::as_str),
    )?;
    let halves = request
        .halves
        .as_ref()
        .ok_or_else(|| invalid("this request holds no rows"))?;
    let (context, predict) = (&halves.context, &halves.predict);
    let mut dtypes: Vec<(String, String)> = table
        .columns
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                if c.untyped {
                    UNTYPED.to_string()
                } else {
                    c.dtype.clone()
                },
            )
        })
        .collect();
    for name in predict.columns() {
        if let Some(entry) = dtypes.iter_mut().find(|(n, _)| *n == name)
            && let Some(series) = predict.series(&name)
        {
            let raw = raw_dtype_of(&series, Some(&entry.1));
            entry.1 = widen_tfm_dtype(&[entry.1.clone(), raw]);
        }
    }
    for entry in &mut dtypes {
        if entry.1.is_empty() {
            entry.1 = "string".to_string();
        }
    }
    if let Some(pinned) = pinned {
        for (name, dtype) in pinned {
            if dtype.is_empty() {
                continue;
            }
            match dtypes.iter_mut().find(|(n, _)| n == name) {
                Some(entry) => entry.1 = dtype.clone(),
                None => dtypes.push((name.clone(), dtype.clone())),
            }
        }
    }
    let target = crate::layout::encode_deep(&request.task.target.column_name);
    // The schema and the session name columns by text, so a label that is
    // not text is never among them.
    let missing: Vec<String> = context
        .columns()
        .into_iter()
        .filter(|name| {
            context.label_not_text(name).is_some() || !dtypes.iter().any(|(n, _)| n == name)
        })
        .collect();
    if !missing.is_empty() {
        return Err(invalid(format!(
            "schema does not describe context column(s) {}",
            context.labels_repr(&missing)
        )));
    }
    let dtype_of = |name: &str| -> String {
        dtypes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.clone())
            .unwrap_or_default()
    };
    let mut schema_columns = Map::new();
    let mut as_text = Vec::new();
    for name in context.columns() {
        let column = column_of(table, &name)?;
        if column.stype == "text" {
            as_text.push(name.clone());
        }
        let dtype = dtype_of(&name);
        schema_columns.insert(
            name.clone(),
            json!({
                "dtype": dtype,
                "stype": coherent_stype(&stype_for(request, &column.stype, &name), &dtype, column),
            }),
        );
    }
    if !as_text.is_empty() {
        said.sdm(format!(
            "{KUMO_TABULAR} cannot read a text column, so {} travel(s) as categorical instead. ",
            as_text.join(", ")
        ));
    }
    let mut instance = Map::new();
    instance.insert("columns".to_string(), Value::Object(schema_columns));
    let key = sdm_schema::client::key_names_of(&table.primary_key);
    if !key.is_empty() {
        instance.insert("primary_key".to_string(), json!(key));
    }
    let target_dtype = dtypes
        .iter()
        .find(|(name, _)| Some(name.as_str()) == target.as_str())
        .map(|(_, d)| d.clone())
        .ok_or_else(|| crate::error::builtin("KeyError", request.task.target.column_name.repr()))?;
    let mut envelope = Map::new();
    envelope.insert("model".into(), json!(KUMO_TABULAR));
    envelope.insert(
        "task".into(),
        json!({"kind": crate::layout::encode_deep(&request.task.kind), "target": {"column_name": target, "dtype": target_dtype}}),
    );
    envelope.insert(
        "schema".into(),
        json!({"instance_table": Value::Object(instance)}),
    );
    envelope.insert(
        "context".into(),
        json!({"instance_table": encode_table(context, &dtypes)?}),
    );
    envelope.insert(
        "predict".into(),
        json!({"instance_table": encode_table(predict, &dtypes)?}),
    );
    envelope.insert("output".into(), json!({"fields": outputs}));
    envelope.insert("metadata".into(), json!({"request_id": request_id()}));
    if !request.inference.is_empty() {
        let given = PyVal::Dict(request.inference.clone());
        let ready = crate::wire::json_ready(&decode_tags(&given))?;
        envelope.insert("inference".into(), inference_json(&ready));
    }
    Ok(Value::Object(envelope))
}

/// Tagged values in a mapping read from JSON, decoded.
fn decode_tags(value: &PyVal) -> PyVal {
    let json = inference_json(value);
    PyVal::decode(&json).unwrap_or_else(|_| value.clone())
}

fn inference_json(value: &PyVal) -> Value {
    match value {
        PyVal::Held { .. } => crate::layout::encode_deep(value),
        PyVal::Dict(items) => Value::Object(
            items
                .iter()
                .map(|(key, value)| (crate::pyops::py_str(key), inference_json(value)))
                .collect(),
        ),
        PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
            Value::Array(items.iter().map(inference_json).collect())
        }
        PyVal::Numpy(_, inner) => inference_json(inner),
        other => crate::cells::to_json(other),
    }
}

/// `session_plan`'s pinned half of an envelope.
pub fn pinned_half(envelope: &Value) -> Value {
    let mut pinned = Map::new();
    for key in SESSION_PINNED {
        if let Some(v) = envelope.py_get(key) {
            pinned.insert(key.to_string(), v.clone());
        }
    }
    pinned.insert(
        "metadata".to_string(),
        envelope.py_get("metadata").cloned().unwrap_or(Value::Null),
    );
    Value::Object(pinned)
}

/// `session_prediction_plan`'s scoring half of an envelope.
pub fn scoring_half(envelope: &Value) -> Value {
    let mut out = Map::new();
    for key in SESSION_SCORING {
        if let Some(v) = envelope.py_get(key) {
            out.insert(key.to_string(), v.clone());
        }
    }
    Value::Object(out)
}

/// `_declared_by(pinned)`: the dtypes a session pinned.
pub fn declared_by(pinned: &Value) -> Option<Vec<(String, String)>> {
    let columns = pinned
        .get("schema")?
        .get("instance_table")?
        .get("columns")?
        .as_object()?;
    Some(
        columns
            .iter()
            .filter_map(|(name, described)| {
                let dtype = described.get("dtype")?;
                let text = dtype
                    .as_str()
                    .map_or_else(|| dtype.to_string(), str::to_string);
                (!text.is_empty()).then(|| (name.clone(), text))
            })
            .collect(),
    )
}

/// `_key_named_by(value)`: the key an id names, or `None`.
fn key_named_by(value: &str) -> Option<String> {
    if value.chars().count() > MAX_ID_LENGTH {
        return None;
    }
    let named = match serde_json::from_str::<Value>(value) {
        Ok(v) => crate::keys::json_cell(&v),
        Err(_) => crate::pyliteral::literal_eval(value)?,
    };
    match named {
        PyVal::List(items) if items.len() == 1 => Some(crate::pyops::py_str(&items[0])),
        _ => None,
    }
}

/// `_named_entities(items, payload)` of the tabular adapter.
pub fn named_entities(items: Vec<Item>, payload: &Value) -> R<Vec<Item>> {
    let table = payload.get("predict").and_then(|p| p.get("instance_table"));
    let Some(rows) = table.and_then(|t| t.get("rows")).and_then(Value::as_array) else {
        return Ok(items);
    };
    let columns: Vec<String> = table
        .and_then(|t| t.get("columns"))
        .and_then(Value::as_array)
        .map(|c| {
            c.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    let instance = payload.get("schema").and_then(|s| s.get("instance_table"));
    let key: Vec<String> = instance
        .and_then(|i| i.get("primary_key"))
        .and_then(Value::as_array)
        .map(|k| {
            k.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let named_key = if key.len() == 1 {
        Some(key[0].clone())
    } else {
        None
    };
    let key_dtype = named_key.as_ref().and_then(|k| {
        instance
            .and_then(|i| i.get("columns"))
            .and_then(|c| c.get(k))
            .and_then(|c| c.get("dtype"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let keys: Option<Vec<PyVal>> = match &named_key {
        Some(k) if columns.contains(k) => {
            let at = columns.iter().position(|c| c == k).unwrap_or(0);
            Some(
                rows.iter()
                    .map(|row| {
                        let cell = row.get(at).map_or(PyVal::None, crate::keys::json_cell);
                        crate::wire::as_declared(&cell, key_dtype.as_deref())
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    let in_batch: Vec<String> = keys.iter().flatten().map(crate::pyops::py_str).collect();
    let mut named = Vec::new();
    let mut seen: Vec<i64> = Vec::new();
    for item in items {
        let Some(index) = item.row_index_i64() else {
            return Err(crate::error::structured(
                "INVALID_RESPONSE",
                "the response is missing a row_index, so its answer cannot be attributed to a row",
            ));
        };
        if index < 0 || index as usize >= rows.len() {
            return Err(crate::error::structured(
                "INVALID_RESPONSE",
                format!(
                    "the response names row_index {index}, and this batch asked about {} rows",
                    rows.len()
                ),
            ));
        }
        if seen.contains(&index) {
            return Err(crate::error::structured(
                "INVALID_RESPONSE",
                format!("the response answers row {index} more than once"),
            ));
        }
        if let (Some(keys), Some(id)) = (&keys, &item.id) {
            let answered = crate::pyops::py_str(id);
            let mut found = key_named_by(&answered);
            if found.is_none() && in_batch.contains(&answered) {
                found = Some(answered.clone());
            }
            if let Some(found) = found
                && found != crate::pyops::py_str(&keys[index as usize])
            {
                return Err(crate::error::structured(
                    "INVALID_RESPONSE",
                    format!(
                        "the response pairs row_index {index} with id {}, and this batch asked \
                         about {} there",
                        crate::pyops::py_repr(id),
                        crate::pyops::py_repr(&keys[index as usize])
                    ),
                ));
            }
        }
        seen.push(index);
        let value = keys.as_ref().map(|k| k[index as usize].clone());
        let mut item = item;
        if let Some(value) = value
            && !matches!(value, PyVal::None)
        {
            item.id = Some(value);
        }
        named.push(item);
    }
    let mut missing: Vec<i64> = (0..rows.len() as i64)
        .filter(|i| !seen.contains(i))
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        let shown: Vec<String> = missing.iter().map(i64::to_string).collect();
        return Err(crate::error::structured(
            "INVALID_RESPONSE",
            format!("the response left row(s) [{}] unanswered", shown.join(", ")),
        ));
    }
    Ok(named)
}

/// `_a_row_the_pinned_schema_cannot_hold(request, declared, error)`: a
/// failure to write the scored rows to the pinned schema, said of the
/// session.
pub fn a_row_the_pinned_schema_cannot_hold(
    request: &Request,
    tables: &crate::schemas::Tables,
    declared: Option<&[(String, String)]>,
    error: sdm_protocol::Diagnostic,
) -> sdm_protocol::Diagnostic {
    let _ = tables;
    let text = crate::relational_adapter::error_text(&error);
    // What is made here is raised from the error (`raise ... from error`).
    let beneath = error.as_links();
    let Some(declared) = declared else {
        if crate::parts::is_structured(&error) {
            return error;
        }
        return crate::error::invalid(error.message).caused_by(beneath);
    };
    let Some(halves) = &request.halves else {
        return crate::error::invalid(text).caused_by(beneath);
    };
    let predict = &halves.predict;
    let differing: Vec<String> = predict
        .columns()
        .iter()
        .filter_map(|name| {
            let (_, pinned) = declared.iter().find(|(n, _)| n == name)?;
            let held = predict.dtype(name).name();
            (held != *pinned)
                .then(|| format!("{name} was pinned as {pinned} and these rows hold {held}"))
        })
        .collect();
    if differing.is_empty() {
        return error;
    }
    crate::error::invalid(format!(
        "these rows cannot be written to the schema this session pinned: {}. A scoring call \
         does not send a schema. It was settled by fit(), so the rows have to be writable as it \
         describes them. Give the column the type it was pinned with, or fit() again on a \
         context that has the type you want. ({text})",
        differing.join("; ")
    ))
    .caused_by(beneath)
}

#[cfg(test)]
mod inference_tests {
    use super::*;

    #[test]
    fn inference_conversion_keeps_held_values_until_their_use_site() {
        let cyclic = PyVal::decode(&json!({"options": {"$held": {
            "reference": {"$host_ref": "0"},
            "value": {"$refers_back": "dict"}
        }}}))
        .unwrap();
        let refused = crate::wire::json_ready(&decode_tags(&cyclic)).unwrap_err();
        assert_eq!(
            refused.message,
            "this value refers back to itself, so it cannot be sent as JSON"
        );
        for view in [json!(4), json!({"$object": {"type": "Value", "repr": ""}})] {
            let held = PyVal::decode(&json!({"options": {"$held": {
                "reference": {"$host_ref": "1"},
                "owner": "operation",
                "type": "Value",
                "value": view
            }}}))
            .unwrap();
            let ready = crate::wire::json_ready(&decode_tags(&held)).unwrap();
            assert_eq!(ready, held);
            assert_eq!(PyVal::decode(&inference_json(&ready)).unwrap(), held);
        }
        let ordinary = PyVal::Numpy("int64".into(), Box::new(PyVal::Int(4)));
        assert_eq!(inference_json(&ordinary), json!(4));
        for ordinary in [
            json!(null),
            json!(true),
            json!(u64::MAX),
            json!({"$float": "inf"}),
            json!({"$decimal": "1.25"}),
            json!({"$tuple": [1, {"nested": [true, null]}]}),
            json!({"options": {"$np": {"type": "int64", "value": 4}}}),
            json!({"$dict": {"$np": {"type": "int64", "value": 4}}}),
        ] {
            let ordinary = PyVal::decode(&ordinary).unwrap();
            assert_eq!(inference_json(&ordinary), crate::cells::to_json(&ordinary));
        }
    }
}
