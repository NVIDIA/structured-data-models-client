// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An engine table as plain JSON, exactly: everything it read from its
//! source and keeps, so a later call can go on from where `Schema.from_source`
//! left it without reading the source again (`EngineTables`, which keeps
//! the tables `from_source` read for the materializer).

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use arrow_schema::DataType;
use sdm_pql::types::{Dtype, Stype};
use serde_json::{Value, json};

use super::infer::{SampleColumn, SampleFrame, SampleKind};
use super::{Backend, EColumn, EngineTable, Place, SourceColumn, SourceFk};
use crate::dt::civil::Wall;
use crate::pyval::PyVal;

fn wall_json(wall: &Wall) -> Value {
    json!(wall.epoch_nanos().to_string())
}

fn wall_of(value: &Value) -> Result<Wall, String> {
    let nanos: i128 = value
        .as_str()
        .ok_or("a time is its nanoseconds as text")?
        .parse()
        .map_err(|_| "a time's nanoseconds are a whole number")?;
    Ok(Wall::from_epoch_nanos(nanos))
}

/// A value as JSON that reads back as exactly that value.
pub fn value_json(value: &PyVal) -> Value {
    match value {
        PyVal::Held {
            reference,
            owner,
            type_name,
            value,
        } => json!(["Held", reference, value_json(value), owner, type_name]),
        PyVal::None => json!(["None"]),
        PyVal::Bool(b) => json!(["Bool", b]),
        PyVal::Int(n) => json!(["Int", n.to_string()]),
        PyVal::BigInt(s) => json!(["BigInt", s]),
        PyVal::Float(x) => json!(["Float", format!("{:?}", x.to_bits())]),
        PyVal::Str(s) => json!(["Str", s]),
        PyVal::Decimal(s) => json!(["Decimal", s]),
        PyVal::Date(y, m, d) => json!(["Date", y, m, d]),
        PyVal::DateTime(w, o) => json!(["DateTime", wall_json(w), o]),
        PyVal::Time {
            hour,
            minute,
            second,
            micro,
            offset,
        } => json!(["Time", hour, minute, second, micro, offset]),
        PyVal::Timestamp(w) => json!(["Timestamp", wall_json(w)]),
        PyVal::TimestampTz(w, o) => json!(["TimestampTz", wall_json(w), o]),
        PyVal::Timedelta(n) => json!(["Timedelta", n]),
        PyVal::NdArray(d, items) => json!([
            "NdArray",
            d,
            items.iter().map(value_json).collect::<Vec<_>>()
        ]),
        PyVal::Bytes(b) => json!(["Bytes", b]),
        PyVal::Uuid(s) => json!(["Uuid", s]),
        PyVal::Na => json!(["Na"]),
        PyVal::NaT => json!(["NaT"]),
        PyVal::List(items) => json!(["List", items.iter().map(value_json).collect::<Vec<_>>()]),
        PyVal::Tuple(items) => json!(["Tuple", items.iter().map(value_json).collect::<Vec<_>>()]),
        PyVal::Dict(pairs) => json!([
            "Dict",
            pairs
                .iter()
                .map(|(k, v)| json!([value_json(k), value_json(v)]))
                .collect::<Vec<_>>()
        ]),
        PyVal::Numpy(d, inner) => json!(["Numpy", d, value_json(inner)]),
        PyVal::Opaque {
            kind,
            type_name,
            repr,
        } => json!(["Opaque", kind, type_name, repr]),
    }
}

fn items_of(value: &Value) -> Result<Vec<PyVal>, String> {
    value
        .as_array()
        .ok_or("a list of values")?
        .iter()
        .map(value_of)
        .collect()
}

/// [`value_json`] read back.
pub fn value_of(value: &Value) -> Result<PyVal, String> {
    let parts = value.as_array().ok_or("a value is a tagged list")?;
    let tag = parts
        .first()
        .and_then(Value::as_str)
        .ok_or("a value names its kind")?;
    let at = |i: usize| parts.get(i).unwrap_or(&Value::Null);
    let text = |i: usize| -> Result<String, String> {
        at(i)
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{tag} holds text"))
    };
    let int = |i: usize| -> Result<i64, String> {
        at(i)
            .as_i64()
            .ok_or_else(|| format!("{tag} holds a number"))
    };
    let small = |i: usize| -> Result<u32, String> { int(i).map(|n| n as u32) };
    let offset = |i: usize| -> Option<i64> { at(i).as_i64() };
    Ok(match tag {
        "None" => PyVal::None,
        "Bool" => PyVal::Bool(at(1).as_bool().ok_or("Bool holds a flag")?),
        "Int" => PyVal::Int(text(1)?.parse().map_err(|_| "Int holds digits")?),
        "BigInt" => PyVal::BigInt(text(1)?),
        "Float" => PyVal::Float(f64::from_bits(
            text(1)?.parse().map_err(|_| "Float holds its bits")?,
        )),
        "Str" => PyVal::Str(text(1)?),
        "Decimal" => PyVal::Decimal(text(1)?),
        "Date" => PyVal::Date(int(1)?, small(2)?, small(3)?),
        "DateTime" => PyVal::DateTime(wall_of(at(1))?, offset(2)),
        "Time" => PyVal::Time {
            hour: small(1)?,
            minute: small(2)?,
            second: small(3)?,
            micro: small(4)?,
            offset: offset(5),
        },
        "Timestamp" => PyVal::Timestamp(wall_of(at(1))?),
        "TimestampTz" => PyVal::TimestampTz(wall_of(at(1))?, int(2)?),
        "Timedelta" => PyVal::Timedelta(int(1)?),
        "NdArray" => PyVal::NdArray(text(1)?, items_of(at(2))?),
        "Bytes" => PyVal::Bytes(
            at(1)
                .as_array()
                .ok_or("Bytes holds bytes")?
                .iter()
                .map(|b| b.as_u64().map(|b| b as u8).ok_or("a byte"))
                .collect::<Result<_, _>>()?,
        ),
        "Uuid" => PyVal::Uuid(text(1)?),
        "Na" => PyVal::Na,
        "NaT" => PyVal::NaT,
        "List" => PyVal::List(items_of(at(1))?),
        "Tuple" => PyVal::Tuple(items_of(at(1))?),
        "Dict" => PyVal::Dict(
            at(1)
                .as_array()
                .ok_or("Dict holds pairs")?
                .iter()
                .map(|pair| Ok((value_of(&pair[0])?, value_of(&pair[1])?)))
                .collect::<Result<_, String>>()?,
        ),
        "Numpy" => PyVal::Numpy(text(1)?, Box::new(value_of(at(2))?)),
        "Held" => PyVal::Held {
            reference: value[1].as_str().ok_or("a held reference is text")?.into(),
            value: Box::new(value_of(&value[2])?),
            owner: value[3].as_str().map(str::to_string),
            type_name: value[4].as_str().map(str::to_string),
        },
        "Opaque" => PyVal::Opaque {
            kind: text(1)?,
            type_name: text(2)?,
            repr: text(3)?,
        },
        other => return Err(format!("no kind of value {other:?}")),
    })
}

fn kind_json(kind: &SampleKind) -> Value {
    match kind {
        SampleKind::Arrow(dt) => json!(["Arrow", dt.to_string()]),
        SampleKind::Datetime64 => json!(["Datetime64"]),
        SampleKind::Nullable(name) => json!(["Nullable", name]),
        SampleKind::Object => json!(["Object"]),
    }
}

/// The names a nullable extension dtype is spelled with.
const NULLABLE: [&str; 15] = [
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64",
    "boolean", "string", "str", "float16", "category",
];

fn kind_of(value: &Value) -> Result<SampleKind, String> {
    let parts = value.as_array().ok_or("a sample kind is a tagged list")?;
    match parts.first().and_then(Value::as_str) {
        Some("Arrow") => {
            let text = parts
                .get(1)
                .and_then(Value::as_str)
                .ok_or("an Arrow type")?;
            DataType::from_str(text)
                .map(SampleKind::Arrow)
                .map_err(|e| e.to_string())
        }
        Some("Datetime64") => Ok(SampleKind::Datetime64),
        Some("Nullable") => {
            let name = parts.get(1).and_then(Value::as_str).ok_or("a dtype name")?;
            NULLABLE
                .iter()
                .find(|n| **n == name)
                .map(|n| SampleKind::Nullable(n))
                .ok_or_else(|| format!("no nullable dtype {name:?}"))
        }
        Some("Object") => Ok(SampleKind::Object),
        _ => Err("no such sample kind".to_string()),
    }
}

/// A sample frame as JSON.
pub fn frame_json(frame: &SampleFrame) -> Value {
    json!({
        "index": frame.index,
        "columns": frame.columns.iter().map(|c| json!({
            "name": c.name,
            "kind": kind_json(&c.kind),
            "cells": c.cells.iter().map(value_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

/// [`frame_json`] read back.
pub fn frame_of(value: &Value) -> Result<SampleFrame, String> {
    let columns = value
        .get("columns")
        .and_then(Value::as_array)
        .ok_or("a frame has columns")?
        .iter()
        .map(|c| {
            Ok(SampleColumn {
                name: c
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("a column has a name")?
                    .to_string(),
                kind: kind_of(c.get("kind").unwrap_or(&Value::Null))?,
                cells: items_of(c.get("cells").unwrap_or(&Value::Null))?,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(SampleFrame {
        columns,
        index: value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize,
    })
}

fn opt_text(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_string)
}

fn dtype_of(value: Option<&Value>) -> Result<Dtype, String> {
    let text = value.and_then(Value::as_str).ok_or("a dtype")?;
    Dtype::parse(text).ok_or_else(|| format!("no dtype {text:?}"))
}

fn source_column_json(c: &SourceColumn) -> Value {
    json!({
        "name": c.name,
        "dtype": c.dtype.map(Dtype::value),
        "is_primary_key": c.is_primary_key,
        "is_unique_key": c.is_unique_key,
        "is_nullable": c.is_nullable,
    })
}

fn source_column_of(value: &Value) -> Result<SourceColumn, String> {
    let flag = |k: &str| value.get(k).and_then(Value::as_bool).unwrap_or(false);
    Ok(SourceColumn {
        name: opt_text(value.get("name")).ok_or("a source column has a name")?,
        dtype: match value.get("dtype") {
            None | Some(Value::Null) => None,
            given => Some(dtype_of(given)?),
        },
        is_primary_key: flag("is_primary_key"),
        is_unique_key: flag("is_unique_key"),
        is_nullable: flag("is_nullable"),
    })
}

fn listing_json(rows: &[Vec<(String, PyVal)>]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| Value::Array(row.iter().map(|(k, v)| json!([k, value_json(v)])).collect()))
            .collect(),
    )
}

fn listing_of(value: &Value) -> Result<Vec<Vec<(String, PyVal)>>, String> {
    value
        .as_array()
        .ok_or("a listing is rows")?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or("a listed row is pairs")?
                .iter()
                .map(|pair| {
                    Ok((
                        opt_text(pair.get(0)).ok_or("a field name")?,
                        value_of(pair.get(1).unwrap_or(&Value::Null))?,
                    ))
                })
                .collect()
        })
        .collect()
}

impl EngineTable {
    /// The table as plain JSON: what it is and everything it read.
    pub fn to_json(&self) -> Value {
        self.state_json(true)
    }

    /// The table as plain JSON, but the rows its types were inferred from,
    /// which gathering does not read again.
    pub fn to_json_without_sample(&self) -> Value {
        self.state_json(false)
    }

    fn state_json(&self, with_sample: bool) -> Value {
        json!({
            "backend": self.backend.source_name(),
            "name": self.name,
            "raw_source_name": self.raw_source_name,
            "place": {
                "catalog": self.place.catalog,
                "schema": self.place.schema,
                "row_filter": self.place.row_filter,
            },
            "columns": self.columns.iter().map(|c| json!({
                "name": c.name,
                "expr": c.expr,
                "dtype": c.dtype.value(),
                "stype": c.stype.value(),
                "is_pk": c.is_pk,
                "is_time": c.is_time,
                "is_end": c.is_end,
            })).collect::<Vec<_>>(),
            "primary_key": self.primary_key,
            "time_column": self.time_column,
            "time_choice": self.time_choice,
            "time_inferred": self.time_inferred,
            "end_time_column": self.end_time_column,
            "expr_sample": frame_json(&self.expr_sample),
            "source_columns": self.source_columns.as_ref().map(|cs| cs.iter().map(source_column_json).collect::<Vec<_>>()),
            "source_fks": self.source_fks.as_ref().map(|fks| fks.iter().map(|f| json!([f.name, f.dst_table, f.primary_key])).collect::<Vec<_>>()),
            "source_sample": if with_sample { self.source_sample.as_ref().map(frame_json) } else { None },
            "unique_constraints": self.unique_constraints.iter().map(|c| c.iter().cloned().collect::<Vec<_>>()).collect::<Vec<_>>(),
            "spelled": self.spelled,
            "sample_source": self.sample_source,
            "listed": match &self.listed {
                None => Value::Null,
                Some(None) => json!({"failed": true}),
                Some(Some(rows)) => json!({"rows": listing_json(rows)}),
            },
            "is_hybrid": self.is_hybrid,
            "composite_key": self.composite_key,
        })
    }

    /// [`EngineTable::to_json`] read back.
    pub fn from_json(value: &Value) -> Result<EngineTable, String> {
        let backend_name = opt_text(value.get("backend")).ok_or("a table names its backend")?;
        let backend = match backend_name.as_str() {
            "snowflake" => Backend::Snow,
            other => Backend::parse(other).ok_or_else(|| format!("no backend {other:?}"))?,
        };
        let place = value.get("place").cloned().unwrap_or(Value::Null);
        let columns = value
            .get("columns")
            .and_then(Value::as_array)
            .ok_or("a table has columns")?
            .iter()
            .map(|c| {
                let flag = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(false);
                Ok(EColumn {
                    name: opt_text(c.get("name")).ok_or("a column has a name")?,
                    expr: opt_text(c.get("expr")),
                    dtype: dtype_of(c.get("dtype"))?,
                    stype: {
                        let text = opt_text(c.get("stype")).ok_or("a column has an stype")?;
                        Stype::parse(&text).ok_or_else(|| format!("no stype {text:?}"))?
                    },
                    is_pk: flag("is_pk"),
                    is_time: flag("is_time"),
                    is_end: flag("is_end"),
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(EngineTable {
            backend,
            name: opt_text(value.get("name")).ok_or("a table has a name")?,
            raw_source_name: opt_text(value.get("raw_source_name"))
                .ok_or("a table has a source name")?,
            place: Place {
                catalog: opt_text(place.get("catalog")),
                schema: opt_text(place.get("schema")),
                row_filter: opt_text(place.get("row_filter")),
            },
            columns,
            primary_key: opt_text(value.get("primary_key")),
            time_column: opt_text(value.get("time_column")),
            time_choice: value
                .get("time_choice")
                .filter(|choice| !choice.is_null())
                .map(|choice| {
                    serde_json::from_value(choice.clone()).map_err(|error| error.to_string())
                })
                .transpose()?,
            time_inferred: value
                .get("time_inferred")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            end_time_column: opt_text(value.get("end_time_column")),
            expr_sample: frame_of(value.get("expr_sample").unwrap_or(&Value::Null))?,
            source_columns: match value.get("source_columns") {
                Some(Value::Array(items)) => Some(
                    items
                        .iter()
                        .map(source_column_of)
                        .collect::<Result<_, _>>()?,
                ),
                _ => None,
            },
            source_fks: match value.get("source_fks") {
                Some(Value::Array(items)) => Some(
                    items
                        .iter()
                        .map(|f| {
                            Ok(SourceFk {
                                name: opt_text(f.get(0)).ok_or("a foreign key's column")?,
                                dst_table: opt_text(f.get(1)).ok_or("a foreign key's table")?,
                                primary_key: opt_text(f.get(2)).ok_or("a foreign key's key")?,
                            })
                        })
                        .collect::<Result<_, String>>()?,
                ),
                _ => None,
            },
            source_sample: match value.get("source_sample") {
                None | Some(Value::Null) => None,
                Some(frame) => Some(frame_of(frame)?),
            },
            unique_constraints: value
                .get("unique_constraints")
                .and_then(Value::as_array)
                .map(|cs| {
                    cs.iter()
                        .map(|c| {
                            c.as_array()
                                .map(|names| {
                                    names
                                        .iter()
                                        .filter_map(|n| n.as_str().map(str::to_string))
                                        .collect::<BTreeSet<_>>()
                                })
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            spelled: value
                .get("spelled")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect::<BTreeMap<_, _>>()
                })
                .unwrap_or_default(),
            sample_source: opt_text(value.get("sample_source")),
            listed: match value.get("listed") {
                None | Some(Value::Null) => None,
                Some(v) if v.get("failed").is_some() => Some(None),
                Some(v) => Some(Some(listing_of(v.get("rows").unwrap_or(&Value::Null))?)),
            },
            is_hybrid: value.get("is_hybrid").and_then(Value::as_bool),
            composite_key: value
                .get("composite_key")
                .and_then(Value::as_array)
                .map(|names| {
                    names
                        .iter()
                        .filter_map(|n| n.as_str().map(str::to_string))
                        .collect()
                }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_of_value_reads_back_as_itself() {
        let wall = Wall::from_epoch_nanos(1_700_000_000_123_456_789);
        let values = vec![
            PyVal::None,
            PyVal::Bool(true),
            PyVal::Int(-(1 << 100)),
            PyVal::BigInt("1".repeat(50)),
            PyVal::Float(0.1),
            PyVal::Float(f64::NAN),
            PyVal::Str("é\u{1F600}".into()),
            PyVal::Decimal("1.50".into()),
            PyVal::Date(2024, 2, 29),
            PyVal::DateTime(wall, Some(3600)),
            PyVal::Time {
                hour: 1,
                minute: 2,
                second: 3,
                micro: 4,
                offset: None,
            },
            PyVal::Timestamp(wall),
            PyVal::TimestampTz(wall, -18_000),
            PyVal::Timedelta(-5),
            PyVal::NdArray("int64".into(), vec![PyVal::Int(1)]),
            PyVal::Bytes(vec![0, 255]),
            PyVal::Uuid("u".into()),
            PyVal::Na,
            PyVal::NaT,
            PyVal::Tuple(vec![PyVal::List(vec![])]),
            PyVal::Dict(vec![(PyVal::Int(1), PyVal::Str("a".into()))]),
            PyVal::Numpy("float32".into(), Box::new(PyVal::Float(1.5))),
        ];
        for value in values {
            let back = value_of(&value_json(&value)).unwrap();
            assert_eq!(value_json(&back), value_json(&value));
        }
    }
}
