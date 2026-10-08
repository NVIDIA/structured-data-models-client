// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain data: what crosses between a host and the core
//! (`tests/rust_core/request/CONTRACT.md`, "Names"). `None`, booleans,
//! integers, floats, text, bytes, Arrow tables, batches and arrays, and
//! lists, tuples and text-keyed mappings of them.
//!
//! Every argument of a run or function the core exports, and every value it
//! gives back, is plain data, so a call can be recorded and made again from
//! its record in any language.

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use serde_json::{Map, Number, Value};

/// A value of plain data.
#[derive(Clone, Debug)]
pub enum Plain {
    None,
    Bool(bool),
    Int(i64),
    /// An integer outside `i64`, as its decimal digits.
    BigInt(String),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Plain>),
    Tuple(Vec<Plain>),
    /// A mapping, in its order.
    Dict(Vec<(String, Plain)>),
    /// A `pyarrow.Table`: its schema and batches.
    Table(SchemaRef, Vec<RecordBatch>),
    /// A `pyarrow.RecordBatch`.
    Batch(RecordBatch),
    /// A `pyarrow.Array`.
    Array(ArrayRef),
    /// A `pyarrow.ChunkedArray`, as its type and chunks.
    Chunked(DataType, Vec<ArrayRef>),
}

impl Plain {
    /// Text, or `None`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Plain::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The value under `key` of a mapping.
    pub fn get(&self, key: &str) -> Option<&Plain> {
        match self {
            Plain::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// A mapping of `items`.
    pub fn dict<K: Into<String>>(items: impl IntoIterator<Item = (K, Plain)>) -> Plain {
        Plain::Dict(items.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// `Some(text)` as text, `None` as `None`.
    pub fn opt_str(text: Option<&str>) -> Plain {
        text.map_or(Plain::None, |t| Plain::Str(t.to_owned()))
    }

    /// The JSON value this holds, where it holds no Arrow data and no bytes:
    /// a tuple as a list, an integer outside `i64` as a float.
    pub fn to_json(&self) -> Option<Value> {
        Some(match self {
            Plain::None => Value::Null,
            Plain::Bool(b) => Value::Bool(*b),
            Plain::Int(i) => Value::from(*i),
            Plain::BigInt(digits) => {
                let n: f64 = digits.parse().ok()?;
                Value::Number(Number::from_f64(n)?)
            }
            Plain::Float(f) => Value::Number(Number::from_f64(*f)?),
            Plain::Str(s) => Value::String(s.clone()),
            Plain::List(items) | Plain::Tuple(items) => {
                Value::Array(items.iter().map(Plain::to_json).collect::<Option<_>>()?)
            }
            Plain::Dict(items) => {
                let mut map = Map::new();
                for (k, v) in items {
                    map.insert(k.clone(), v.to_json()?);
                }
                Value::Object(map)
            }
            Plain::Bytes(_)
            | Plain::Table(..)
            | Plain::Batch(_)
            | Plain::Array(_)
            | Plain::Chunked(..) => return None,
        })
    }

    /// Plain data of a JSON value: an object as a mapping, an array as a
    /// list, a number as an integer where it is one.
    pub fn from_json(value: &Value) -> Plain {
        match value {
            Value::Null => Plain::None,
            Value::Bool(b) => Plain::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Plain::Int(i),
                None => match n.as_u64() {
                    Some(u) => Plain::BigInt(u.to_string()),
                    None => Plain::Float(n.as_f64().unwrap_or(f64::NAN)),
                },
            },
            Value::String(s) => Plain::Str(s.clone()),
            Value::Array(items) => Plain::List(items.iter().map(Plain::from_json).collect()),
            Value::Object(map) => Plain::Dict(
                map.iter()
                    .map(|(k, v)| (k.clone(), Plain::from_json(v)))
                    .collect(),
            ),
        }
    }
}
