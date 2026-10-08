// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `backend/local/graph_store.LocalGraphStore`: in-memory tables made into
//! the arrays the neighbor sampler walks.
//!
//! It sanitizes each table as `Table._sanitize` does (every column the
//! schema calls a time read as `pd.to_datetime(ser, unit='ns',
//! errors='coerce')` reads it), drops the rows with a null or repeated key
//! or a null time, keys each remaining row by its position, takes times in
//! floored seconds, and builds each link as CSC arrays, a neighborhood
//! sorted stably by time, and its reverse.

use std::collections::{HashMap, HashSet};
use std::fmt;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use sdm_schema::dt::{self, Host};
use serde_json::Value;

use crate::neighbor::EdgeType;

/// The prefix `SampledSubgraph.rev_edge_type` names a reverse link with.
pub const REVERSE_RELATION_PREFIX: &str = "__###REV###__";

/// `SampledSubgraph.rev_edge_type`.
pub fn rev_edge_type(edge: &EdgeType) -> EdgeType {
    let (src, rel, dst) = edge;
    match rel.strip_prefix(REVERSE_RELATION_PREFIX) {
        Some(rel) => (dst.clone(), rel.to_string(), src.clone()),
        None => (
            dst.clone(),
            format!("{REVERSE_RELATION_PREFIX}{rel}"),
            src.clone(),
        ),
    }
}

/// What the store raised: the exception's class and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreError {
    pub code: String,
    pub message: String,
    /// A `KeyError`'s argument, whose `repr` is its text.
    pub key: Option<MissingKey>,
}

/// What a `KeyError` was raised for: a name, or `None`, as `df[None]` raises
/// it for a table with no primary key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MissingKey {
    Name(String),
    None,
}

impl StoreError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            key: None,
        }
    }

    /// `KeyError(key)`, whose text is the key's `repr`.
    fn key_error(key: &str) -> Self {
        Self {
            code: "KeyError".to_string(),
            message: sdm_schema::py::str_repr(key),
            key: Some(MissingKey::Name(key.to_string())),
        }
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for StoreError {}

/// A column of a table, as the graph describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: String,
    /// The engine's `str(column.dtype)`, such as `int` or `time`.
    pub dtype: String,
    /// The engine's `str(column.stype)`, such as `ID` or `timestamp`.
    pub stype: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableSpec {
    pub name: String,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
    pub end_time_column: Option<String>,
    pub columns: Vec<ColumnSpec>,
}

/// A link: `src_table.fkey` references `dst_table`'s primary key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeSpec {
    pub src_table: String,
    pub fkey: String,
    pub dst_table: String,
}

/// The graph, as `graph_json` writes it
/// (`tests/rust_core/sampling/CONTRACT.md`, The graph store).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSpec {
    pub tables: Vec<TableSpec>,
    pub edges: Vec<EdgeSpec>,
}

fn text_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<String, String> {
    object
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("expected the text field {name:?}"))
}

fn optional_text(
    object: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Option<String>, String> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(other) => Err(format!("{name} is text or null, not {other}")),
    }
}

impl GraphSpec {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| format!("the graph is not JSON: {e}"))?;
        let Some(object) = value.as_object() else {
            return Err("the graph is a JSON object".to_string());
        };
        let tables = object
            .get("tables")
            .and_then(Value::as_array)
            .ok_or("the graph has no list of tables")?
            .iter()
            .map(|table| {
                let table = table.as_object().ok_or("a table is an object")?;
                let columns = table
                    .get("columns")
                    .and_then(Value::as_array)
                    .ok_or("a table has a list of columns")?
                    .iter()
                    .map(|column| {
                        let column = column.as_object().ok_or("a column is an object")?;
                        Ok(ColumnSpec {
                            name: text_field(column, "name")?,
                            dtype: optional_text(column, "dtype")?.unwrap_or_default(),
                            stype: optional_text(column, "stype")?.unwrap_or_default(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(TableSpec {
                    name: text_field(table, "name")?,
                    primary_key: optional_text(table, "primary_key")?,
                    time_column: optional_text(table, "time_column")?,
                    end_time_column: optional_text(table, "end_time_column")?,
                    columns,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let edges = object
            .get("edges")
            .and_then(Value::as_array)
            .ok_or("the graph has no list of edges")?
            .iter()
            .map(|edge| {
                let edge = edge.as_object().ok_or("an edge is an object")?;
                Ok(EdgeSpec {
                    src_table: text_field(edge, "src_table")?,
                    fkey: text_field(edge, "fkey")?,
                    dst_table: text_field(edge, "dst_table")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { tables, edges })
    }
}

/// A key as Python holds it: an `int`, a `float` or a `str`.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyValue {
    Int(i64),
    Float(f64),
    Text(String),
}

/// A key's identity for matching: pandas finds `1` and `1.0` the same key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum KeyId {
    Int(i128),
    Float(u64),
    Text(String),
}

impl KeyValue {
    fn id(&self) -> KeyId {
        match self {
            KeyValue::Int(v) => KeyId::Int(i128::from(*v)),
            KeyValue::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 1.7e38 {
                    KeyId::Int(*v as i128)
                } else {
                    KeyId::Float(v.to_bits())
                }
            }
            KeyValue::Text(v) => KeyId::Text(v.clone()),
        }
    }
}

/// A table's data, by column name: Arrow arrays, as
/// `pa.Table.from_pandas(local_table._data)` gives them.
pub type TableData = Vec<(String, ArrayRef)>;

/// `TableSanitizationReport`'s counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableReport {
    pub input_rows: usize,
    pub output_rows: usize,
    pub null_primary_key_rows: usize,
    pub duplicate_primary_key_rows: usize,
    pub null_time_rows: usize,
}

/// What `LocalGraphStore` holds, each in the order it holds it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphStore {
    pub report: Vec<(String, TableReport)>,
    pub masks: Vec<(String, Vec<bool>)>,
    /// `pkey_map_dict`: the valid keys and their row positions.
    pub keys: Vec<(String, Vec<KeyValue>)>,
    pub node_ids: Vec<(String, Vec<i64>)>,
    /// `time_dict`: every row's time in floored seconds.
    pub time: Vec<(String, Vec<i64>)>,
    /// `min_max_time_dict`, in nanoseconds.
    pub time_range: Vec<(String, (i64, i64))>,
    pub colptr: Vec<(EdgeType, Vec<i64>)>,
    pub row: Vec<(EdgeType, Vec<i64>)>,
}

/// `np.iinfo(np.int64)` and `pd.Timestamp`'s bounds.
const NAT: i64 = i64::MIN;

/// `df[name]`, or the `KeyError` it raises.
fn column<'a>(data: &'a TableData, name: &str) -> Result<&'a ArrayRef, StoreError> {
    data.iter()
        .find(|(n, _)| n == name)
        .map(|(_, a)| a)
        .ok_or_else(|| StoreError::key_error(name))
}

fn rows_of(data: &TableData) -> usize {
    data.first().map_or(0, |(_, array)| array.len())
}

/// A column as keys: `None` where pandas' `isna` holds.
fn keys_of(array: &ArrayRef) -> Result<Vec<Option<KeyValue>>, StoreError> {
    let n = array.len();
    let valid = |i: usize| array.is_valid(i);
    macro_rules! ints {
        ($t:ty) => {{
            let values = array.as_primitive::<$t>();
            (0..n)
                .map(|i| valid(i).then(|| KeyValue::Int(i64::from(values.value(i)))))
                .collect()
        }};
    }
    Ok(match array.data_type() {
        DataType::Int64 => ints!(Int64Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int8 => ints!(Int8Type),
        // A `uint64` key past `int64` wraps, one to one, as the sampler's
        // frames read it (`frame::ints`); the schema calls such a key text.
        DataType::UInt64 => {
            let values = array.as_primitive::<UInt64Type>();
            (0..n)
                .map(|i| valid(i).then(|| KeyValue::Int(values.value(i) as i64)))
                .collect()
        }
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            (0..n)
                .map(|i| {
                    let v = values.value(i);
                    (valid(i) && !v.is_nan()).then_some(KeyValue::Float(v))
                })
                .collect()
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            (0..n)
                .map(|i| {
                    let v = f64::from(values.value(i));
                    (valid(i) && !v.is_nan()).then_some(KeyValue::Float(v))
                })
                .collect()
        }
        DataType::Utf8 => {
            let values = array.as_string::<i32>();
            (0..n)
                .map(|i| valid(i).then(|| KeyValue::Text(values.value(i).to_string())))
                .collect()
        }
        DataType::LargeUtf8 => {
            let values = array.as_string::<i64>();
            (0..n)
                .map(|i| valid(i).then(|| KeyValue::Text(values.value(i).to_string())))
                .collect()
        }
        DataType::Null => vec![None; n],
        // A `Categorical` holds its categories' keys.
        DataType::Dictionary(..) => match crate::frame::decoded(array.as_ref()) {
            Some(values) => keys_of(&values)?,
            None => {
                return Err(StoreError::new(
                    "TypeError",
                    format!("keys of Arrow type {} are not supported", array.data_type()),
                ));
            }
        },
        other => {
            return Err(StoreError::new(
                "TypeError",
                format!("keys of Arrow type {other} are not supported"),
            ));
        }
    })
}

/// `to_datetime(ser, column_name)` of a column, in nanoseconds: `None` is
/// `NaT`.
pub(crate) fn to_datetime(array: &ArrayRef, host: &Host) -> Result<Vec<Option<i64>>, StoreError> {
    let n = array.len();
    let scaled = |values: Vec<Option<i128>>| -> Result<Vec<Option<i64>>, StoreError> {
        values
            .into_iter()
            .map(|v| match v {
                None => Ok(None),
                Some(ns) => i64::try_from(ns)
                    .ok()
                    .filter(|&ns| ns != NAT)
                    .map(Some)
                    .ok_or_else(|| StoreError::new("OutOfBoundsDatetime", out_of_bounds_text(ns))),
            })
            .collect()
    };
    macro_rules! stamps {
        ($t:ty, $per:expr) => {{
            let values = array.as_primitive::<$t>();
            scaled(
                (0..n)
                    .map(|i| {
                        array
                            .is_valid(i)
                            .then(|| i128::from(values.value(i)) * $per)
                    })
                    .collect(),
            )
        }};
    }
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, _) => stamps!(TimestampNanosecondType, 1),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            stamps!(TimestampMicrosecondType, 1_000)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            stamps!(TimestampMillisecondType, 1_000_000)
        }
        DataType::Timestamp(TimeUnit::Second, _) => stamps!(TimestampSecondType, 1_000_000_000),
        // A date is its midnight, as pandas reads a `datetime.date`.
        DataType::Date32 => stamps!(Date32Type, 86_400_000_000_000),
        DataType::Date64 => stamps!(Date64Type, 1_000_000),
        DataType::Int64 => stamps!(Int64Type, 1),
        DataType::Null => Ok(vec![None; n]),
        // A time of day is no point in time: pandas reads none of them.
        DataType::Time32(_) | DataType::Time64(_) => Ok(vec![None; n]),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let texts: Vec<Option<String>> = match array.data_type() {
                DataType::Utf8 => {
                    let values = array.as_string::<i32>();
                    (0..n)
                        .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                        .collect()
                }
                _ => {
                    let values = array.as_string::<i64>();
                    (0..n)
                        .map(|i| array.is_valid(i).then(|| values.value(i).to_string()))
                        .collect()
                }
            };
            let call = dt::Call {
                coerce: true,
                ..dt::Call::default()
            };
            let converted = dt::pandas::objects_to_datetime_public(&texts, call, host)
                .map_err(|e| StoreError::new(e.class, e.message))?;
            // Zoned text is converted to UTC and the zone dropped; the values
            // already are UTC.
            scaled(converted.values)
        }
        other => Err(StoreError::new(
            "TypeError",
            format!("times of Arrow type {other} are not supported"),
        )),
    }
}

/// `pd.Index(keys).get_indexer(values)`: each value's position among the
/// keys, or -1.
fn get_indexer(keys: &[KeyValue], values: &[Option<KeyValue>]) -> Vec<i64> {
    let index: HashMap<KeyId, i64> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.id(), i as i64))
        .collect();
    values
        .iter()
        .map(|v| {
            v.as_ref()
                .and_then(|v| index.get(&v.id()).copied())
                .unwrap_or(-1)
        })
        .collect()
}

/// `np.cumsum(np.bincount(col, minlength))`, after a leading 0.
fn compressed(col: &[i64], minlength: usize) -> Vec<i64> {
    let length = col
        .iter()
        .map(|&c| c as usize + 1)
        .max()
        .unwrap_or(0)
        .max(minlength);
    let mut counts = vec![0i64; length + 1];
    for &c in col {
        counts[c as usize + 1] += 1;
    }
    for i in 1..counts.len() {
        counts[i] += counts[i - 1];
    }
    counts
}

/// The graph's tables, by name.
struct Tables<'a> {
    graph: &'a GraphSpec,
    data: &'a [(String, TableData)],
}

impl Tables<'_> {
    fn data(&self, name: &str) -> Result<&TableData, StoreError> {
        self.data
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d)
            .ok_or_else(|| StoreError::key_error(name))
    }

    fn spec(&self, name: &str) -> Result<&TableSpec, StoreError> {
        self.graph
            .tables
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| StoreError::key_error(name))
    }
}

/// What sanitizing leaves for the later steps: each table's mask where it
/// drops rows, its keys and its time column.
#[derive(Default)]
struct Sanitized {
    masks: HashMap<String, Vec<bool>>,
    keys: HashMap<String, Vec<Option<KeyValue>>>,
    times: HashMap<String, Vec<Option<i64>>>,
}

/// `LocalGraphStore.sanitize`: every column the schema calls a time read
/// as one, then each table's rows with a null or repeated key, or a null
/// time, masked and counted.
fn sanitize(
    tables: &Tables<'_>,
    host: &Host,
    store: &mut GraphStore,
) -> Result<Sanitized, StoreError> {
    let mut read_times: HashMap<String, HashMap<String, Vec<Option<i64>>>> = HashMap::new();
    for table in &tables.graph.tables {
        let mut read = HashMap::new();
        for (name, array) in tables.data(&table.name)? {
            let Some(spec) = table.columns.iter().find(|c| &c.name == name) else {
                continue;
            };
            if spec.dtype == "time" || spec.stype == "timestamp" {
                read.insert(name.clone(), to_datetime(array, host)?);
            }
        }
        read_times.insert(table.name.clone(), read);
    }

    let mut out = Sanitized::default();
    for table in &tables.graph.tables {
        let data = tables.data(&table.name)?;
        let n = rows_of(data);
        let mut null_pkey = vec![false; n];
        let mut duplicate_pkey = vec![false; n];
        let mut null_time = vec![false; n];
        if let Some(pkey) = &table.primary_key {
            let keys = keys_of(column(data, pkey)?)?;
            let mut seen = HashSet::new();
            for (i, key) in keys.iter().enumerate() {
                match key {
                    // `duplicated()` marks a second null too, but the null
                    // mask clears it.
                    None => null_pkey[i] = true,
                    Some(key) => duplicate_pkey[i] = !seen.insert(key.id()),
                }
            }
            out.keys.insert(table.name.clone(), keys);
        }
        if let Some(time_column) = &table.time_column {
            let time = match read_times[&table.name].get(time_column) {
                Some(time) => time.clone(),
                None => to_datetime(column(data, time_column)?, host)?,
            };
            for i in 0..n {
                null_time[i] = time[i].is_none() && !null_pkey[i] && !duplicate_pkey[i];
            }
            out.times.insert(table.name.clone(), time);
        }
        let mask: Vec<bool> = (0..n)
            .map(|i| !(null_pkey[i] || duplicate_pkey[i] || null_time[i]))
            .collect();
        let count = |v: &[bool]| v.iter().filter(|&&b| b).count();
        store.report.push((
            table.name.clone(),
            TableReport {
                input_rows: n,
                output_rows: count(&mask),
                null_primary_key_rows: count(&null_pkey),
                duplicate_primary_key_rows: count(&duplicate_pkey),
                null_time_rows: count(&null_time),
            },
        ));
        if !mask.iter().all(|&b| b) {
            store.masks.push((table.name.clone(), mask.clone()));
            out.masks.insert(table.name.clone(), mask);
        }
    }
    Ok(out)
}

/// `LocalGraphStore.get_pkey_map_dict`: each keyed table's valid keys and
/// their rows, or the refusal of a table with none.
fn get_pkey_map_dict(
    tables: &Tables<'_>,
    sanitized: &Sanitized,
    store: &mut GraphStore,
) -> Result<(), StoreError> {
    for table in &tables.graph.tables {
        let Some(keys) = sanitized.keys.get(&table.name) else {
            continue;
        };
        let mask = sanitized.masks.get(&table.name);
        let mut valid_keys = Vec::new();
        let mut ids = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if mask.is_some_and(|m| !m[i]) {
                continue;
            }
            // A key the mask keeps is never null.
            if let Some(key) = key {
                valid_keys.push(key.clone());
                ids.push(i as i64);
            }
        }
        if valid_keys.is_empty() {
            let mut message = format!("Found no valid rows in table '{}'. ", table.name);
            if table.time_column.is_some() {
                message.push_str(
                    "Please make sure that there exists valid non-N/A primary key and time \
                     column pairs in this table.",
                );
            } else {
                message.push_str(
                    "Please make sure that there exists valid non-N/A primary keys in this table.",
                );
            }
            return Err(StoreError::new("ValueError", message));
        }
        store.keys.push((table.name.clone(), valid_keys));
        store.node_ids.push((table.name.clone(), ids));
    }
    Ok(())
}

/// `LocalGraphStore.get_time_data`: every row's time in floored seconds,
/// and each table's earliest and latest valid time.
fn get_time_data(
    tables: &Tables<'_>,
    sanitized: &Sanitized,
    store: &mut GraphStore,
) -> HashMap<String, Vec<i64>> {
    let mut seconds_of = HashMap::new();
    for table in &tables.graph.tables {
        let Some(time) = sanitized.times.get(&table.name) else {
            continue;
        };
        // `astype(int) // 10**9`: NaT is int64's minimum, floored too.
        let seconds: Vec<i64> = time
            .iter()
            .map(|t| t.unwrap_or(NAT).div_euclid(1_000_000_000))
            .collect();
        store.time.push((table.name.clone(), seconds.clone()));
        seconds_of.insert(table.name.clone(), seconds);
        let mask = sanitized.masks.get(&table.name);
        let kept: Vec<i64> = time
            .iter()
            .enumerate()
            .filter(|(i, _)| mask.is_none_or(|m| m[*i]))
            .filter_map(|(_, t)| *t)
            .collect();
        let range = match (kept.iter().min(), kept.iter().max()) {
            (Some(&lo), Some(&hi)) => (lo, hi),
            // (pd.Timestamp.max, pd.Timestamp.min)
            _ => (i64::MAX, i64::MIN + 1),
        };
        store.time_range.push((table.name.clone(), range));
    }
    seconds_of
}

/// `_argsort`, and `_lexsort` of the time within the destination: the
/// order of a link's edges by destination, then source time, ties kept in
/// source order.
fn argsort(dst: &[i64], src: &[i64], seconds: Option<&Vec<i64>>) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..dst.len()).collect();
    match seconds {
        Some(seconds) => perm.sort_by_key(|&i| (dst[i], seconds[src[i] as usize])),
        None => perm.sort_by_key(|&i| dst[i]),
    }
    perm
}

/// `LocalGraphStore.get_csc`: each link as CSC arrays over the rows of its
/// destination, sorted by [`argsort`], and its reverse.
fn get_csc(
    tables: &Tables<'_>,
    sanitized: &Sanitized,
    seconds_of: &HashMap<String, Vec<i64>>,
    store: &mut GraphStore,
) -> Result<(), StoreError> {
    let graph = tables.graph;
    // Each destination's valid keys, and the rows dropped before each.
    let mut index_of: HashMap<String, (Vec<KeyValue>, Option<Vec<i64>>)> = HashMap::new();
    for edge in &graph.edges {
        if index_of.contains_key(&edge.dst_table) {
            continue;
        }
        let dst = tables.spec(&edge.dst_table)?;
        // `df[table._primary_key]` of a table with none is `df[None]`, which
        // `Graph.validate` keeps a graph from reaching.
        let Some(pkey) = &dst.primary_key else {
            return Err(StoreError {
                code: "KeyError".to_string(),
                message: "None".to_string(),
                key: Some(MissingKey::None),
            });
        };
        let keys = match sanitized.keys.get(&edge.dst_table) {
            Some(keys) => keys.clone(),
            None => keys_of(column(tables.data(&edge.dst_table)?, pkey)?)?,
        };
        let (kept, offsets) = match sanitized.masks.get(&edge.dst_table) {
            Some(mask) => {
                let mut kept = Vec::new();
                let mut offsets = Vec::new();
                let mut dropped = 0i64;
                for (i, key) in keys.into_iter().enumerate() {
                    if !mask[i] {
                        dropped += 1;
                        continue;
                    }
                    if let Some(key) = key {
                        kept.push(key);
                        offsets.push(dropped);
                    }
                }
                (kept, Some(offsets))
            }
            None => (keys.into_iter().flatten().collect(), None),
        };
        index_of.insert(edge.dst_table.clone(), (kept, offsets));
    }
    for edge in &graph.edges {
        let src_data = tables.data(&edge.src_table)?;
        let dst_rows = rows_of(tables.data(&edge.dst_table)?);
        let src_rows = rows_of(src_data);
        let (keys, offsets) = &index_of[&edge.dst_table];
        let fkeys = keys_of(column(src_data, &edge.fkey)?)?;
        let found = get_indexer(keys, &fkeys);
        let src_mask = sanitized.masks.get(&edge.src_table);
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for (i, &position) in found.iter().enumerate() {
            if position < 0 || src_mask.is_some_and(|m| !m[i]) {
                continue;
            }
            let row = match offsets {
                Some(offsets) => position + offsets[position as usize],
                None => position,
            };
            src.push(i as i64);
            dst.push(row);
        }
        let seconds = seconds_of.get(&edge.src_table).filter(|_| !dst.is_empty());
        let perm = argsort(&dst, &src, seconds);
        let row: Vec<i64> = perm.iter().map(|&i| src[i]).collect();
        let col: Vec<i64> = perm.iter().map(|&i| dst[i]).collect();
        let edge_type = (
            edge.src_table.clone(),
            edge.fkey.clone(),
            edge.dst_table.clone(),
        );
        store
            .colptr
            .push((edge_type.clone(), compressed(&col, dst_rows)));
        store.row.push((edge_type.clone(), row));
        // The reverse link: each source row has at most one parent, so it
        // is in source order already.
        let reverse = rev_edge_type(&edge_type);
        store
            .colptr
            .push((reverse.clone(), compressed(&src, src_rows)));
        store.row.push((reverse, dst));
    }
    Ok(())
}

/// `LocalGraphStore(graph)`.
pub fn graph_store(
    graph: &GraphSpec,
    tables: &[(String, TableData)],
    host: &Host,
) -> Result<GraphStore, StoreError> {
    for (name, data) in tables {
        let rows = rows_of(data);
        if data.iter().any(|(_, array)| array.len() != rows) {
            // What `pd.DataFrame` says of such columns; a `pyarrow.Table`
            // can't have them, but a table built another way can.
            return Err(StoreError::new(
                "ValueError",
                format!("All arrays must be of the same length (table {name})"),
            ));
        }
    }
    let tables = Tables {
        graph,
        data: tables,
    };
    let mut store = GraphStore::default();
    let sanitized = sanitize(&tables, host, &mut store)?;
    get_pkey_map_dict(&tables, &sanitized, &mut store)?;
    let seconds_of = get_time_data(&tables, &sanitized, &mut store);
    get_csc(&tables, &sanitized, &seconds_of, &mut store)?;
    Ok(store)
}

/// The text of pandas' `OutOfBoundsDatetime` for a time `ns` nanoseconds
/// from the epoch that `datetime64[ns]` cannot hold: the time to the second,
/// its year unpadded (`check_dts_bounds`).
pub fn out_of_bounds_text(ns: i128) -> String {
    let seconds = ns.div_euclid(1_000_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    // Howard Hinnant's `civil_from_days`, in the proleptic Gregorian calendar.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i128::from(month <= 2);
    format!(
        "Out of bounds nanosecond timestamp: {year}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        of_day / 3600,
        of_day % 3600 / 60,
        of_day % 60
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int64Array, TimestampNanosecondArray};

    use super::*;

    #[test]
    fn a_time_past_datetime64_ns_is_named_as_pandas_names_it() {
        let ns = |days: i128, seconds: i128| (days * 86_400 + seconds) * 1_000_000_000;
        assert_eq!(
            out_of_bounds_text(ns(2_932_896, 3 * 3600 + 4 * 60 + 5)),
            "Out of bounds nanosecond timestamp: 9999-12-31 03:04:05"
        );
        assert_eq!(
            out_of_bounds_text(ns(-135_140, 0)),
            "Out of bounds nanosecond timestamp: 1600-01-01 00:00:00"
        );
    }

    fn spec(
        name: &str,
        key: &str,
        time: Option<&str>,
        columns: &[(&str, &str, &str)],
    ) -> TableSpec {
        TableSpec {
            name: name.to_string(),
            primary_key: Some(key.to_string()),
            time_column: time.map(str::to_string),
            end_time_column: None,
            columns: columns
                .iter()
                .map(|(name, dtype, stype)| ColumnSpec {
                    name: name.to_string(),
                    dtype: dtype.to_string(),
                    stype: stype.to_string(),
                })
                .collect(),
        }
    }

    /// Two tables with a repeated key, a null key, a null time, a time
    /// before 1970 and a link that misses, as `LocalGraphStore` stores them.
    #[test]
    fn stores_as_local_graph_store() {
        let graph = GraphSpec {
            tables: vec![
                spec(
                    "users",
                    "id",
                    None,
                    &[("id", "int", "ID"), ("x", "float", "numerical")],
                ),
                spec(
                    "orders",
                    "oid",
                    Some("ts"),
                    &[
                        ("oid", "int", "ID"),
                        ("uid", "int", "ID"),
                        ("ts", "time", "timestamp"),
                    ],
                ),
            ],
            edges: vec![EdgeSpec {
                src_table: "orders".into(),
                fkey: "uid".into(),
                dst_table: "users".into(),
            }],
        };
        let users: TableData = vec![
            (
                "id".into(),
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    Some(3),
                ])),
            ),
            ("x".into(), Arc::new(Float64Array::from(vec![1.0; 5]))),
        ];
        let orders: TableData = vec![
            (
                "oid".into(),
                Arc::new(Int64Array::from(vec![10, 11, 12, 13])),
            ),
            (
                "uid".into(),
                Arc::new(Int64Array::from(vec![Some(3), Some(1), Some(3), Some(9)])),
            ),
            (
                "ts".into(),
                Arc::new(TimestampNanosecondArray::from(vec![
                    Some(1_704_153_600_000_000_000),
                    Some(1_704_067_201_500_000_000),
                    None,
                    Some(-500_000_000),
                ])),
            ),
        ];
        let store = graph_store(
            &graph,
            &[("users".into(), users), ("orders".into(), orders)],
            &Host::assumed(),
        )
        .unwrap();
        assert_eq!(
            store.report[0].1,
            TableReport {
                input_rows: 5,
                output_rows: 3,
                null_primary_key_rows: 1,
                duplicate_primary_key_rows: 1,
                null_time_rows: 0,
            }
        );
        assert_eq!(store.masks[1].1, [true, true, false, true]);
        assert_eq!(store.node_ids[0].1, [0, 1, 4]);
        assert_eq!(
            store.time[0].1,
            [1_704_153_600, 1_704_067_201, -9_223_372_037, -1]
        );
        assert_eq!(
            store.time_range[0].1,
            (-500_000_000, 1_704_153_600_000_000_000)
        );
        assert_eq!(store.colptr[0].1, [0, 1, 1, 1, 1, 2]);
        assert_eq!(store.row[0].1, [1, 0]);
        assert_eq!(store.colptr[1].0.1, "__###REV###__uid");
        assert_eq!(store.colptr[1].1, [0, 1, 2, 2, 2]);
        assert_eq!(store.row[1].1, [4, 0]);
    }

    /// A `Categorical` key column holds its categories' values.
    #[test]
    fn categories_are_keys_of_their_values() {
        use arrow_array::types::Int8Type;
        use arrow_array::{DictionaryArray, Int8Array};
        let categories: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let array: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(2), None, Some(0)]),
                categories,
            )
            .unwrap(),
        );
        assert_eq!(
            keys_of(&array).unwrap(),
            [Some(KeyValue::Int(30)), None, Some(KeyValue::Int(10))]
        );
    }

    /// Columns of different lengths are refused, not read past their end.
    #[test]
    fn a_table_of_ragged_columns_is_a_value_error() {
        let graph = GraphSpec {
            tables: vec![spec(
                "users",
                "id",
                Some("ts"),
                &[("id", "int", "ID"), ("ts", "time", "timestamp")],
            )],
            edges: vec![],
        };
        let users: TableData = vec![
            ("id".into(), Arc::new(Int64Array::from(vec![1, 2, 3]))),
            (
                "ts".into(),
                Arc::new(TimestampNanosecondArray::from(vec![0, 1])),
            ),
        ];
        let error = graph_store(&graph, &[("users".into(), users)], &Host::assumed()).unwrap_err();
        assert_eq!(error.code, "ValueError");
    }
}
