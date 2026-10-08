// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `sampling.*` operations of the parity fixtures, as
//! `tests/parity/_fixture_ops.py` runs them: the neighbor sampler with
//! `relationallib`'s checks of what it is handed, the graph store, `DateOffset`
//! arithmetic and the random stream of a seed.

use serde_json::{Map, Value, json};

use sdm_labels::date_offset::{Bound, DateOffset};
use sdm_schema::dt::Host;

use crate::graph_store::{GraphSpec, KeyValue, graph_store};
use crate::neighbor::{
    EdgeType, NeighborSampler, SampleError, Seed, TimeRanges, cast_error, rel_type,
};

/// What an operation raised: the Python exception's class and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError {
    pub code: String,
    pub message: String,
}

impl OpError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl From<SampleError> for OpError {
    fn from(error: SampleError) -> Self {
        match &error {
            SampleError::Runtime(m) => OpError::new("RuntimeError", m.clone()),
            SampleError::Index(m) => OpError::new("IndexError", m.clone()),
            SampleError::Key(_) => OpError::new("KeyError", error.to_string()),
        }
    }
}

fn key_error(name: &str) -> OpError {
    OpError::new("KeyError", sdm_schema::py::str_repr(name))
}

fn get<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value, OpError> {
    object.get(name).ok_or_else(|| key_error(name))
}

fn type_error(message: impl Into<String>) -> OpError {
    OpError::new("TypeError", message)
}

// --- numpy's view of what the sampler is handed --------------------------------

/// `np.asarray(value)` of a fixture value, as far as the sampler's checks
/// read it: its shape, its dtype's kind and item size, and its items.
#[derive(Debug)]
struct NdArray {
    shape: Vec<usize>,
    kind: char,
    itemsize: usize,
    items: Vec<Value>,
}

/// `value`'s shape, as numpy reads nested lists, and its items in order;
/// `None` when its lists are ragged, which numpy refuses.
fn flatten(value: &Value) -> Option<(Vec<usize>, Vec<Value>)> {
    let mut shape = Vec::new();
    let mut at = value;
    while let Value::Array(list) = at {
        shape.push(list.len());
        match list.first() {
            Some(first) => at = first,
            None => break,
        }
    }
    let mut items = Vec::new();
    gather(value, &shape, &mut items).then_some((shape, items))
}

fn gather(value: &Value, shape: &[usize], items: &mut Vec<Value>) -> bool {
    match (value, shape) {
        (Value::Array(list), [len, rest @ ..]) => {
            list.len() == *len && list.iter().all(|item| gather(item, rest, items))
        }
        (Value::Array(_), []) => false,
        (_, []) => {
            items.push(value.clone());
            true
        }
        (_, _) => false,
    }
}

/// pybind11's `cast<int64_t>()` of a fixture value, or its error's text. A
/// `bool` is an `int` to Python, so pybind11 casts it.
fn cast_int(value: &Value) -> Result<i64, String> {
    if let Value::Bool(b) = value {
        return Ok(i64::from(*b));
    }
    value.as_i64().ok_or_else(|| {
        cast_error(match value {
            Value::Null => "<class 'NoneType'>",
            Value::Bool(_) => "<class 'bool'>",
            Value::Number(n) if n.is_f64() => "<class 'float'>",
            Value::Number(_) => "<class 'int'>",
            Value::String(_) => "<class 'str'>",
            Value::Array(_) => "<class 'list'>",
            Value::Object(_) => "<class 'dict'>",
        })
    })
}

/// The dtypes [`NdArray::of`] reads by name; any other is `object`.
const KNOWN_DTYPES: [&str; 6] = ["int64", "int32", "uint64", "float64", "bool", "str"];

fn ragged() -> OpError {
    OpError::new(
        "ValueError",
        "setting an array element with a sequence. The requested array has an inhomogeneous \
         shape",
    )
}

/// The kind and item size numpy infers for `items`.
fn inferred_kind(items: &[Value]) -> (char, usize) {
    if items.iter().all(|v| v.is_i64() || v.is_u64()) {
        ('i', 8)
    } else if items.iter().all(Value::is_number) {
        ('f', 8)
    } else if items.iter().all(Value::is_boolean) {
        ('b', 1)
    } else if items.iter().all(Value::is_string) {
        ('U', 4)
    } else {
        ('O', 8)
    }
}

impl NdArray {
    /// `np.asarray(value)`, or for `{"$ndarray": {"dtype", "values"}}`,
    /// `np.array(values, dtype=dtype)`: ragged lists are numpy's
    /// `ValueError`, except as `object`, which holds the outer list's items.
    fn of(value: &Value) -> Result<NdArray, OpError> {
        let (values, dtype) = match value.get("$ndarray") {
            Some(Value::Object(body)) => (
                body.get("values").unwrap_or(&Value::Null),
                Some(
                    body.get("dtype")
                        .and_then(Value::as_str)
                        .unwrap_or("object"),
                ),
            ),
            _ => (value, None),
        };
        let (shape, items) = match (flatten(values), dtype, values) {
            (Some(flat), _, _) => flat,
            (None, Some(dtype), Value::Array(list)) if !KNOWN_DTYPES.contains(&dtype) => {
                (vec![list.len()], list.clone())
            }
            (None, _, _) => return Err(ragged()),
        };
        let (kind, itemsize) = match dtype {
            Some("int64") => ('i', 8),
            Some("int32") => ('i', 4),
            Some("uint64") => ('u', 8),
            Some("float64") => ('f', 8),
            Some("bool") => ('b', 1),
            Some("str") => ('U', 4),
            Some(_) => ('O', 8),
            None => inferred_kind(&items),
        };
        Ok(NdArray {
            shape,
            kind,
            itemsize,
            items,
        })
    }

    fn size(&self) -> usize {
        self.items.len()
    }

    /// `validate_arr`.
    fn check(&self, name: &str, dim: usize, int64: bool) -> Result<(), OpError> {
        if self.shape.len() != dim {
            return Err(OpError::new(
                "RuntimeError",
                format!("{name} is not {dim}D."),
            ));
        }
        if self.size() == 0 {
            return Ok(());
        }
        if int64 && (self.kind != 'i' || self.itemsize != 8) {
            return Err(OpError::new(
                "RuntimeError",
                format!("{name} must have dtype int64."),
            ));
        }
        Ok(())
    }

    fn ints(&self) -> Vec<i64> {
        self.items.iter().filter_map(Value::as_i64).collect()
    }
}

/// `validate_arr_dict` over a dict of arrays, giving each checked array.
fn checked(
    dict: &Map<String, Value>,
    name: &str,
    keys: &[String],
    dim: usize,
    int64: bool,
) -> Result<Vec<(String, NdArray)>, OpError> {
    let mut out = Vec::new();
    for (key, value) in dict {
        if !keys.contains(key) {
            return Err(OpError::new(
                "RuntimeError",
                format!("Invalid type '{key}' found in {name}."),
            ));
        }
        let array = NdArray::of(value)?;
        array.check(&format!("Type '{key}' in {name}"), dim, int64)?;
        out.push((key.clone(), array));
    }
    Ok(out)
}

fn dict<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>, OpError> {
    value
        .as_object()
        .ok_or_else(|| type_error(format!("{what} is a dict")))
}

// --- Integers of any size ---------------------------------------------------------

/// A Python integer of the fixture JSON: a number, or `{"$int": digits}`.
fn big_int(value: &Value) -> Result<(bool, Vec<u32>), OpError> {
    let digits = match value {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(_) => {
            return Err(type_error(
                "'float' object cannot be interpreted as an integer",
            ));
        }
        Value::Object(o) => match o.get("$int").and_then(Value::as_str) {
            Some(text) => text.to_string(),
            None => return Err(type_error("expected an integer")),
        },
        other => {
            return Err(type_error(format!(
                "'{}' object cannot be interpreted as an integer",
                match other {
                    Value::String(_) => "str",
                    Value::Bool(_) => "bool",
                    Value::Null => "NoneType",
                    _ => "list",
                }
            )));
        }
    };
    let (negative, digits) = match digits.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, digits),
    };
    let mut words: Vec<u32> = Vec::new();
    for c in digits.chars() {
        let d = c
            .to_digit(10)
            .ok_or_else(|| type_error("expected an integer"))?;
        let mut carry = u64::from(d);
        for word in &mut words {
            let v = u64::from(*word) * 10 + carry;
            *word = v as u32;
            carry = v >> 32;
        }
        if carry > 0 {
            words.push(carry as u32);
        }
    }
    while words.last() == Some(&0) {
        words.pop();
    }
    Ok((negative, words))
}

/// A seed as the sampler takes it.
fn sampler_seed(value: &Value) -> Result<Seed, OpError> {
    let (negative, words) = big_int(value)?;
    let text = || {
        if let Value::Object(o) = value {
            o.get("$int")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        } else {
            value.to_string()
        }
    };
    if negative {
        let magnitude = words.first().copied().map_or(0, i64::from);
        if words.len() > 1 || magnitude > 1 << 31 {
            return Err(type_error(format!(
                "a seed is an integer from -2**31 up, not {}",
                text()
            )));
        }
        return Ok(Seed::Word(-magnitude));
    }
    Ok(match words.as_slice() {
        [] => Seed::Word(0),
        [word] => Seed::Word(i64::from(*word)),
        _ => Seed::Words(words),
    })
}

// --- The operations ----------------------------------------------------------------

fn neighbor_sample(input: &Map<String, Value>) -> Result<Value, OpError> {
    let given = dict(get(input, "sampler")?, "sampler")?;
    let node_types: Vec<String> = get(given, "node_types")?
        .as_array()
        .ok_or_else(|| type_error("node_types is a list"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| type_error("a node type is text"))
        })
        .collect::<Result<_, _>>()?;
    let edge_types: Vec<EdgeType> = get(given, "edge_types")?
        .as_array()
        .ok_or_else(|| type_error("edge_types is a list"))?
        .iter()
        .map(|edge| {
            let parts: Vec<String> = edge
                .as_array()
                .map(|p| {
                    p.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            match <[String; 3]>::try_from(parts) {
                Ok([a, b, c]) => Ok((a, b, c)),
                Err(_) => Err(type_error("an edge type is three strings")),
            }
        })
        .collect::<Result<_, _>>()?;
    let rel_types: Vec<String> = edge_types.iter().map(rel_type).collect();
    let seed = match given.get("seed") {
        Some(seed) => sampler_seed(seed)?,
        None => Seed::Word(123),
    };
    let colptr = dict(get(given, "colptr")?, "colptr")?;
    let row = dict(get(given, "row")?, "row")?;
    // `node_time` is read as `dict(value)`: an empty list is an empty dict.
    let empty = Map::new();
    let node_time = match get(given, "node_time")? {
        Value::Object(map) => map,
        Value::Array(list) if list.is_empty() => &empty,
        _ => return Err(type_error("node_time is read as a dict")),
    };
    let ints = |entries: Vec<(String, NdArray)>| {
        entries
            .into_iter()
            .map(|(k, a)| (k, a.ints()))
            .collect::<Vec<_>>()
    };
    let colptr = ints(checked(colptr, "colptr", &rel_types, 1, true)?);
    let row = ints(checked(row, "row", &rel_types, 1, true)?);
    let node_time = ints(checked(node_time, "node_time", &node_types, 1, true)?);
    let mut sampler = NeighborSampler::new(node_types, edge_types, colptr, row, node_time, &seed)?;

    let mut out = Vec::new();
    for call in get(input, "calls")?
        .as_array()
        .ok_or_else(|| type_error("calls is a list"))?
    {
        let call = dict(call, "a call")?;
        if let Some(reseed) = call.get("reseed") {
            sampler.seed(&sampler_seed(reseed)?);
        }
        let counts = ints(checked(
            dict(get(call, "num_neighbors")?, "num_neighbors")?,
            "num_neighbors",
            &rel_types,
            1,
            true,
        )?);
        let ranges = checked(
            dict(get(call, "sampling_time_offsets")?, "sampling_time_offsets")?,
            "sampling_time_offsets",
            &rel_types,
            2,
            false,
        )?
        .into_iter()
        .map(|(key, array)| {
            let width = array.shape.get(1).copied().unwrap_or(0);
            let rows = array.shape.first().copied().unwrap_or(0);
            let ranges = if width == 2 {
                array
                    .items
                    .chunks(2)
                    .map(|pair| {
                        let start = match &pair[0] {
                            Value::Null => None,
                            value => Some(cast_int(value)?),
                        };
                        Ok((start, cast_int(&pair[1])?))
                    })
                    .collect()
            } else {
                vec![Ok((None, 0)); rows]
            };
            (key, TimeRanges { width, ranges })
        })
        .collect::<Vec<_>>();
        let entity = get(call, "entity_table")?
            .as_str()
            .ok_or_else(|| type_error("entity_table is text"))?;
        let seed_node = NdArray::of(get(call, "seed_node")?)?;
        let seed_time = NdArray::of(get(call, "seed_time")?)?;
        seed_node.check("seed_node", 1, true)?;
        seed_time.check("seed_time", 1, true)?;
        let sampled = sampler.sample(
            &counts,
            &ranges,
            entity,
            &seed_node.ints(),
            &seed_time.ints(),
        )?;
        let part = |entries: &[(String, Vec<i64>)]| {
            Value::Object(entries.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
        };
        out.push(json!({
            "row": part(&sampled.row),
            "col": part(&sampled.col),
            "node_id": part(&sampled.node_id),
            "batch": part(&sampled.batch),
            "num_sampled_nodes": part(&sampled.num_sampled_nodes),
            "num_sampled_edges": part(&sampled.num_sampled_edges),
        }));
    }
    Ok(Value::Array(out))
}

fn store(input: &Map<String, Value>) -> Result<Value, OpError> {
    let graph = GraphSpec::from_json(&get(input, "graph")?.to_string())
        .map_err(|m| OpError::new("ValueError", m))?;
    let mut tables = Vec::new();
    for (name, frame) in dict(get(input, "tables")?, "tables")? {
        let columns = frame
            .get("columns")
            .and_then(Value::as_object)
            .ok_or_else(|| key_error("columns"))?
            .iter()
            .map(|(column, spec)| {
                let series = sdm_labels::fixture::build_series(spec)
                    .and_then(|c| sdm_labels::arrow::to_arrow(&c))
                    .map_err(|e| OpError::new(&e.code, e.message))?;
                Ok((column.clone(), series))
            })
            .collect::<Result<Vec<_>, OpError>>()?;
        tables.push((name.clone(), columns));
    }
    let store = graph_store(&graph, &tables, &Host::assumed())
        .map_err(|e| OpError::new(&e.code, e.message))?;
    let by_name = |entries: Vec<(String, Value)>| Value::Object(entries.into_iter().collect());
    let key = |k: &KeyValue| match k {
        KeyValue::Int(v) => json!(v),
        KeyValue::Float(v) => json!(v),
        KeyValue::Text(v) => json!(v),
    };
    Ok(json!({
        "report": by_name(store.report.iter().map(|(n, r)| (n.clone(), json!({
            "input_rows": r.input_rows,
            "output_rows": r.output_rows,
            "null_primary_key_rows": r.null_primary_key_rows,
            "duplicate_primary_key_rows": r.duplicate_primary_key_rows,
            "null_time_rows": r.null_time_rows,
        }))).collect()),
        "masks": by_name(store.masks.iter().map(|(n, m)| (n.clone(), json!(m))).collect()),
        "keys": by_name(store.keys.iter().map(|(n, k)| (n.clone(), Value::Array(k.iter().map(key).collect()))).collect()),
        "node_ids": by_name(store.node_ids.iter().map(|(n, v)| (n.clone(), json!(v))).collect()),
        "time": by_name(store.time.iter().map(|(n, v)| (n.clone(), json!(v))).collect()),
        "time_range": by_name(store.time_range.iter().map(|(n, (lo, hi))| (n.clone(), json!([lo, hi]))).collect()),
        "colptr": by_name(store.colptr.iter().map(|(e, v)| (rel_type(e), json!(v))).collect()),
        "row": by_name(store.row.iter().map(|(e, v)| (rel_type(e), json!(v))).collect()),
    }))
}

/// `pd.DateOffset(**offset)`, with the error making it raises.
fn offset(input: &Map<String, Value>) -> Result<DateOffset, OpError> {
    let offset =
        DateOffset::from_json(get(input, "offset")?).map_err(|m| OpError::new("TypeError", m))?;
    offset
        .constructed()
        .map_err(|e| OpError::new(e.code, e.message))?;
    Ok(offset)
}

fn date_offset_apply(input: &Map<String, Value>) -> Result<Value, OpError> {
    let offset = offset(input)?;
    let sign = get(input, "sign")?.as_i64().unwrap_or(1);
    let sign = if sign > 0 { 1 } else { -1 };
    let out = get(input, "timestamps")?
        .as_array()
        .ok_or_else(|| type_error("timestamps is a list"))?
        .iter()
        .map(|t| {
            let t = t
                .as_i64()
                .ok_or_else(|| type_error("a timestamp is an integer"))?;
            Ok(match offset.apply(t, sign) {
                Ok(ns) => json!({"ns": ns}),
                Err(e) => json!({"error": {"code": e.code, "message": e.message}}),
            })
        })
        .collect::<Result<Vec<_>, OpError>>()?;
    Ok(Value::Array(out))
}

fn date_offset_to_seconds(input: &Map<String, Value>) -> Result<String, OpError> {
    let bound = get(input, "bound")?
        .as_str()
        .and_then(Bound::parse)
        .ok_or_else(|| OpError::new("ValueError", "bound is 'earliest' or 'latest'"))?;
    let seconds = offset(input)?
        .to_seconds(bound)
        .map_err(|e| OpError::new(e.code, e.message))?;
    // The exact integer, as JSON text: Python's `json` reads it back exactly,
    // where a `serde_json::Value` past 64 bits would have to be a float.
    Ok(seconds.to_decimal())
}

fn random_state(input: &Map<String, Value>) -> Result<Value, OpError> {
    let (negative, words) = big_int(get(input, "seed")?)?;
    if negative {
        return Err(OpError::new(
            "ValueError",
            "Seed must be between 0 and 2**32 - 1",
        ));
    }
    let int = |name: &str| {
        get(input, name)?
            .as_i64()
            .ok_or_else(|| type_error(format!("{name} is an integer")))
    };
    let mut state =
        sdm_random::random_state(&words).map_err(|e| OpError::new(e.code, e.message))?;
    let size = usize::try_from(int("size")?)
        .map_err(|_| OpError::new("ValueError", "negative dimensions are not allowed"))?;
    let drawn = state
        .randint(int("low")?, int("high")?, size)
        .map_err(|e| OpError::new(e.code, e.message))?;
    Ok(json!(drawn))
}

/// `sampling.to_list`: `Table._sanitize` of one text column with a list
/// dtype or the `sequence` stype; the column's dtype after, and its values,
/// a list as numpy's array of it.
fn to_list(input: &Map<String, Value>) -> Result<Value, OpError> {
    use std::sync::Arc;

    use crate::driver::sql::{dtype_of_name, list_kind};
    use crate::frame::{Dtype, Series};

    let values = get(input, "values")?
        .as_array()
        .ok_or_else(|| type_error("values is a list"))?
        .iter()
        .map(|v| match v {
            Value::Null => Ok(None),
            Value::String(s) => Ok(Some(s.clone())),
            _ => Err(type_error("a value is text or null")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, name) = match get(input, "series")?.as_str() {
        Some("arrow") => (Dtype::Arrow, "string[pyarrow]"),
        Some("str") => (Dtype::Extension("str".into()), "str"),
        Some("object") => (Dtype::Object, "object"),
        _ => return Err(type_error("series is arrow, str or object")),
    };
    let series = Series::new(dtype, Arc::new(arrow_array::StringArray::from(values)));
    let kind = list_kind(
        input
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(dtype_of_name),
        input.get("stype").and_then(Value::as_str),
    );
    let out = match kind {
        Some(kind) => crate::sequences::to_list(&series, kind)
            .map_err(|e| OpError::new(&e.class, e.message))?,
        None => series,
    };
    let array = out.array.as_ref();
    let values: Vec<Value> = match array.data_type() {
        arrow_schema::DataType::List(_) => {
            use arrow_array::Array;
            use arrow_array::cast::AsArray;
            use arrow_array::types::{Float32Type, Int64Type};
            let lists = array.as_list::<i32>();
            (0..lists.len())
                .map(|i| {
                    let row = lists.value(i);
                    match row.data_type() {
                        arrow_schema::DataType::Float32 => json!({"$ndarray": {
                            "dtype": "float32",
                            "values": row.as_primitive::<Float32Type>().values().iter()
                                .map(|&v| float_json(f64::from(v))).collect::<Vec<_>>(),
                        }}),
                        _ => json!({"$ndarray": {
                            "dtype": "int64",
                            "values": row.as_primitive::<Int64Type>().values().to_vec(),
                        }}),
                    }
                })
                .collect()
        }
        _ => crate::frame::texts(array)
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.map_or(Value::Null, Value::String))
            .collect(),
    };
    let name = if out.dtype == Dtype::Object {
        "object"
    } else {
        name
    };
    Ok(json!({"dtype": name, "values": values}))
}

/// A float as the fixtures write one: itself, or `{"$float": ...}` where
/// JSON has no number for it.
fn float_json(value: f64) -> Value {
    if value.is_nan() {
        json!({"$float": "nan"})
    } else if value.is_infinite() {
        json!({"$float": if value > 0.0 { "inf" } else { "-inf" }})
    } else {
        json!(value)
    }
}

/// Answers one `sampling.*` operation: its output as JSON text, or the error
/// it raises; `None` for an operation it doesn't know.
pub fn run(operation: &str, input: &Map<String, Value>) -> Option<Result<String, OpError>> {
    let text = |value: Result<Value, OpError>| value.map(|v| v.to_string());
    Some(match operation {
        "sampling.neighbor_sample" => text(neighbor_sample(input)),
        "sampling.graph_store" => text(store(input)),
        "sampling.date_offset_apply" => text(date_offset_apply(input)),
        "sampling.date_offset_to_seconds" => date_offset_to_seconds(input),
        "sampling.random_state" => text(random_state(input)),
        "sampling.to_list" => text(to_list(input)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_lists_are_refused_as_numpy_refuses_them() {
        let square = NdArray::of(&json!([[0, 1], [2, 3]])).unwrap();
        assert_eq!((square.shape, square.items.len()), (vec![2, 2], 4));
        assert_eq!(NdArray::of(&json!([[], []])).unwrap().shape, [2, 0]);
        for ragged in [json!([[0, 1], [2]]), json!([[0, 1], 2]), json!([0, [1]])] {
            assert_eq!(NdArray::of(&ragged).unwrap_err().code, "ValueError");
        }
        // As `object`, numpy holds the outer list's items instead.
        let held = json!({"$ndarray": {"dtype": "object", "values": [[0, 1], 2]}});
        assert_eq!(NdArray::of(&held).unwrap().shape, [2]);
        let typed = json!({"$ndarray": {"dtype": "int64", "values": [[0, 1], 2]}});
        assert!(NdArray::of(&typed).is_err());
    }
}
