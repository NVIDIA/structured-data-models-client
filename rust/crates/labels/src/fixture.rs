// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `labels.*` operations of the parity fixtures, in the fixtures' JSON
//! (`tests/parity/FORMAT.md`): series built from `{"dtype", "values"}` as
//! `_fixture_ops.build_series` builds them, and written back as
//! `series_out` writes them.

use serde_json::{Map, Number, Value, json};

use crate::LabelError;
use crate::aggregate::Aggregation;
use crate::column::{Column, Obj, TimeUnit};
use crate::date_offset::civil_from_days;
use crate::executor::{
    Batch, Frame, execute_aggregation, execute_bool_op, execute_column, execute_member_op,
    execute_rel_op,
};
use crate::query::{Constant, ConstantValue, Node, Window};
use sdm_schema::dt::Host;

pub use crate::datetime::parse_timestamp;

/// `pd.Timestamp.isoformat()` of `ticks` in `unit`.
pub fn isoformat(unit: TimeUnit, ticks: i64) -> String {
    let ns = i128::from(ticks) * i128::from(unit.nanos());
    let day = ns.div_euclid(86_400_000_000_000);
    let rest = ns.rem_euclid(86_400_000_000_000);
    let (year, month, d) = civil_from_days(day);
    let seconds = rest / 1_000_000_000;
    let sub = rest % 1_000_000_000;
    let (micro, nano) = (sub / 1_000, sub % 1_000);
    let mut out = format!(
        "{year:04}-{month:02}-{d:02}T{:02}:{:02}:{:02}",
        seconds / 3_600,
        seconds / 60 % 60,
        seconds % 60
    );
    if micro != 0 {
        out.push_str(&format!(".{micro:06}"));
    }
    if nano != 0 {
        if micro != 0 {
            out.push_str(&format!("{nano:03}"));
        } else {
            out.push_str(&format!(".{sub:09}"));
        }
    }
    out
}

fn float_json(v: f64) -> Value {
    if v.is_nan() {
        json!({"$float": "nan"})
    } else if v.is_infinite() {
        json!({"$float": if v > 0.0 { "inf" } else { "-inf" }})
    } else {
        Number::from_f64(v).map_or(Value::Null, Value::Number)
    }
}

/// `_fixture_ops.encode` of a value.
pub fn encode(obj: &Obj) -> Value {
    let np = |kind: &str, value: Value| json!({"$np": {"type": kind, "value": value}});
    match obj {
        Obj::None => Value::Null,
        Obj::Na => json!({"$na": null}),
        Obj::NaT => json!({"$nat": null}),
        Obj::Bool(b) => Value::Bool(*b),
        Obj::Int(i) => Value::from(*i),
        Obj::Float(f) => float_json(*f),
        Obj::Str(s) => Value::String(s.clone()),
        Obj::Bytes(bytes) => json!({"$bytes": encode_bytes(bytes)}),
        Obj::Duration(unit, ticks) => {
            json!({"$timedelta_us": (i128::from(*ticks) * i128::from(unit.nanos())).div_euclid(1000)})
        }
        Obj::Time(unit, t) => json!({"$timestamp": isoformat(*unit, *t)}),
        Obj::Decimal(digits, scale) => {
            json!({"$decimal": sdm_schema::frame::decimal_text(*digits, *scale)})
        }
        Obj::List(items) => Value::Array(items.iter().map(encode).collect()),
        Obj::NpInt64(i) => np("int64", Value::from(*i)),
        Obj::NpFloat64(f) => np("float64", float_json(*f)),
        Obj::NpFloat32(f) => np("float32", float_json(f64::from(*f))),
        Obj::NpBool(b) => np("bool", Value::Bool(*b)),
    }
}

fn encode_bytes(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    for chunk in bytes.chunks(3) {
        let number = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for position in 0..4 {
            output.push(if position <= chunk.len() {
                ALPHABET[((number >> (18 - 6 * position)) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    output
}

/// `_fixture_ops.decode` of a value.
pub fn decode(value: &Value) -> Result<Obj, LabelError> {
    Ok(match value {
        Value::Null => Obj::None,
        Value::Bool(b) => Obj::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Obj::Int(i),
            None => Obj::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Obj::Str(s.clone()),
        Value::Array(items) => Obj::List(items.iter().map(decode).collect::<Result<_, _>>()?),
        Value::Object(object) => {
            let Some((tag, body)) = object.iter().next().filter(|_| object.len() == 1) else {
                return Err(unreadable("a value object other than a tag"));
            };
            match tag.as_str() {
                "$bytes" | "$timedelta_us" => {
                    match sdm_schema::pyval::PyVal::decode(value)
                        .map_err(|message| unreadable(&message))?
                    {
                        sdm_schema::pyval::PyVal::Bytes(bytes) => Obj::Bytes(bytes),
                        sdm_schema::pyval::PyVal::Timedelta(ticks) => {
                            Obj::Duration(TimeUnit::Ns, ticks)
                        }
                        _ => return Err(unreadable("a bytes or timedelta scalar")),
                    }
                }
                "$na" => Obj::Na,
                "$nat" => Obj::NaT,
                "$float" => Obj::Float(match body.as_str() {
                    Some("nan") => f64::NAN,
                    Some("inf") => f64::INFINITY,
                    Some("-inf") => f64::NEG_INFINITY,
                    Some(other) => {
                        crate::query::python_float(other).ok_or_else(|| unreadable("a $float"))?
                    }
                    None => return Err(unreadable("a $float")),
                }),
                "$int" => Obj::Int(
                    body.as_str()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| unreadable("an $int beyond int64"))?,
                ),
                "$timestamp" => Obj::Time(
                    TimeUnit::Ns,
                    body.as_str()
                        .and_then(parse_timestamp)
                        .ok_or_else(|| unreadable("a $timestamp"))?,
                ),
                "$np" => {
                    let kind = body.get("type").and_then(Value::as_str).unwrap_or("");
                    let inner = decode(body.get("value").unwrap_or(&Value::Null))?;
                    match (kind, inner) {
                        ("int64", Obj::Int(i)) => Obj::NpInt64(i),
                        ("float64", Obj::Float(f)) => Obj::NpFloat64(f),
                        ("float64", Obj::Int(i)) => Obj::NpFloat64(i as f64),
                        ("float32", Obj::Float(f)) => Obj::NpFloat32(f as f32),
                        ("bool", Obj::Bool(b)) => Obj::NpBool(b),
                        _ => return Err(unreadable("a $np value")),
                    }
                }
                other => return Err(unreadable(&format!("the tag {other}"))),
            }
        }
    })
}

fn unreadable(what: &str) -> LabelError {
    LabelError::new("ValueError", format!("the fixture input holds {what}"))
}

fn number(obj: &Obj) -> Option<f64> {
    match obj {
        Obj::Int(i) | Obj::NpInt64(i) => Some(*i as f64),
        Obj::Float(f) | Obj::NpFloat64(f) => Some(*f),
        Obj::NpFloat32(f) => Some(f64::from(*f)),
        Obj::Bool(b) | Obj::NpBool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// `build_series({"dtype", "values"})`.
pub fn build_series(spec: &Value) -> Result<Column, LabelError> {
    let dtype = spec
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| unreadable("a series without a dtype"))?;
    let values: Vec<Obj> = spec
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| unreadable("a series without values"))?
        .iter()
        .map(decode)
        .collect::<Result<_, _>>()?;
    let not_a = |what: &str, obj: &Obj| {
        LabelError::new(
            "TypeError",
            format!("a {what} series can't hold {}", obj.type_name()),
        )
    };
    if let Some(unit) = dtype
        .strip_prefix("datetime64[")
        .and_then(|rest| rest.strip_suffix(']'))
        .and_then(TimeUnit::parse)
    {
        return Ok(Column::Time(
            unit,
            values
                .iter()
                .map(|v| match v {
                    v if v.is_na() => Ok(None),
                    Obj::Time(u, t) => {
                        let ns = i128::from(*t) * i128::from(u.nanos());
                        Ok(Some(ns.div_euclid(i128::from(unit.nanos())) as i64))
                    }
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ));
    }
    Ok(match dtype {
        "float32" => Column::F32(
            values
                .iter()
                .map(|v| {
                    if v.is_na() {
                        Ok(f32::NAN)
                    } else {
                        number(v).map(|x| x as f32).ok_or_else(|| not_a(dtype, v))
                    }
                })
                .collect::<Result<_, _>>()?,
        ),
        "float64" => Column::F64(
            values
                .iter()
                .map(|v| {
                    if v.is_na() {
                        Ok(f64::NAN)
                    } else {
                        number(v).ok_or_else(|| not_a(dtype, v))
                    }
                })
                .collect::<Result<_, _>>()?,
        ),
        "Float64" => Column::MF64(
            values
                .iter()
                .map(|v| {
                    if v.is_na() {
                        Ok(None)
                    } else {
                        number(v).map(Some).ok_or_else(|| not_a(dtype, v))
                    }
                })
                .collect::<Result<_, _>>()?,
        ),
        "int64" => Column::I64(
            values
                .iter()
                .map(|v| match v {
                    Obj::Int(i) | Obj::NpInt64(i) => Ok(*i),
                    Obj::Bool(b) => Ok(i64::from(*b)),
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ),
        "Int64" => Column::MI64(
            values
                .iter()
                .map(|v| match v {
                    v if v.is_na() => Ok(None),
                    Obj::Int(i) | Obj::NpInt64(i) => Ok(Some(*i)),
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ),
        "bool" => Column::Bool(
            values
                .iter()
                .map(|v| match v {
                    Obj::Bool(b) | Obj::NpBool(b) => Ok(*b),
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ),
        "boolean" => Column::MBool(
            values
                .iter()
                .map(|v| match v {
                    v if v.is_na() => Ok(None),
                    Obj::Bool(b) | Obj::NpBool(b) => Ok(Some(*b)),
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ),
        "str" => Column::Str(
            values
                .iter()
                .map(|v| match v {
                    v if v.is_na() => Ok(None),
                    Obj::Str(s) => Ok(Some(s.clone())),
                    other => Err(not_a(dtype, other)),
                })
                .collect::<Result<_, _>>()?,
        ),
        "object" => Column::Obj(values),
        other => {
            return Err(LabelError::new(
                "TypeError",
                format!("the fixtures' dtype {other:?} is not supported"),
            ));
        }
    })
}

/// `series_out(series)`.
pub fn series_out(column: &Column) -> Value {
    json!({
        "dtype": column.dtype(),
        "values": column.objects().iter().map(encode).collect::<Vec<_>>(),
    })
}

fn labelled(out: &Column, mask: &[bool]) -> Value {
    json!({"out": series_out(out), "mask": mask})
}

fn get<'a>(input: &'a Map<String, Value>, name: &str) -> Result<&'a Value, LabelError> {
    input.get(name).ok_or_else(|| LabelError::key_error(name))
}

fn flag(input: &Map<String, Value>, name: &str) -> Result<bool, LabelError> {
    get(input, name)?
        .as_bool()
        .ok_or_else(|| unreadable(&format!("a {name} that is not a bool")))
}

fn text<'a>(input: &'a Map<String, Value>, name: &str) -> Result<&'a str, LabelError> {
    get(input, name)?
        .as_str()
        .ok_or_else(|| unreadable(&format!("a {name} that is not text")))
}

fn frames(tables: &Value) -> Result<Vec<(String, Frame)>, LabelError> {
    let tables = tables
        .as_object()
        .ok_or_else(|| unreadable("tables that are not an object"))?;
    tables
        .iter()
        .map(|(name, frame)| {
            let columns = frame
                .get("columns")
                .and_then(Value::as_object)
                .ok_or_else(|| unreadable("a frame without columns"))?
                .iter()
                .map(|(column, spec)| Ok((column.clone(), build_series(spec)?)))
                .collect::<Result<Vec<_>, LabelError>>()?;
            Ok((name.clone(), Frame { columns }))
        })
        .collect()
}

/// `_fixture_ops._constant(spec)`.
fn constant(spec: &Value) -> Result<Constant, LabelError> {
    let value = match spec.get("value") {
        Some(Value::Array(items)) => {
            ConstantValue::List(items.iter().map(constant).collect::<Result<_, _>>()?)
        }
        Some(Value::String(s)) => ConstantValue::Text(s.clone()),
        Some(other) => ConstantValue::Text(other.to_string()),
        None => return Err(LabelError::key_error("value")),
    };
    let dtype = match spec.get("dtype") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => return Err(unreadable(&format!("a constant dtype {other}"))),
    };
    Ok(Constant { value, dtype })
}

fn batches(given: &Value) -> Result<Vec<(String, Vec<i64>)>, LabelError> {
    given
        .as_object()
        .ok_or_else(|| unreadable("batches that are not an object"))?
        .iter()
        .map(|(name, rows)| {
            let rows = rows
                .as_array()
                .ok_or_else(|| unreadable("a batch that is not a list"))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| unreadable("a batch index that is not an integer"))
                })
                .collect::<Result<_, _>>()?;
            Ok((name.clone(), rows))
        })
        .collect()
}

/// Answers one `labels.*` operation on its input: its output, or the error
/// it raises, as today's Python runner answers it; `None` for an operation
/// it doesn't know.
pub fn run(
    operation: &str,
    input: &Map<String, Value>,
    host: &Host,
) -> Option<Result<Value, LabelError>> {
    run_with_warnings(operation, input, host, &mut Vec::new())
}

pub fn run_with_warnings(
    operation: &str,
    input: &Map<String, Value>,
    host: &Host,
    warnings: &mut Vec<crate::Warning>,
) -> Option<Result<Value, LabelError>> {
    Some(match operation {
        "labels.column" => (|| {
            let batch = Batch {
                feat: frames(get(input, "tables")?)?,
                ..Batch::default()
            };
            let result = execute_column(text(input, "fqn")?, &batch, flag(input, "filter_na")?);
            warnings.extend(batch.warnings.into_inner());
            let (out, mask) = result?;
            Ok(labelled(&out, &mask))
        })(),
        "labels.rel_op" => (|| {
            let left = build_series(get(input, "left")?)?;
            let out = execute_rel_op(
                &left,
                text(input, "op")?,
                &constant(get(input, "right")?)?,
                host,
            )?;
            Ok(series_out(&out))
        })(),
        "labels.member_op" => (|| {
            let left = build_series(get(input, "left")?)?;
            let right = constant(get(input, "right")?)?;
            let out = execute_member_op(&left, text(input, "op")?, &right, host)?;
            Ok(series_out(&out))
        })(),
        "labels.bool_op" => (|| {
            let left = build_series(get(input, "left")?)?;
            let right = match input.get("right") {
                None | Some(Value::Null) => None,
                Some(spec) => Some(build_series(spec)?),
            };
            let out = execute_bool_op(&left, text(input, "op")?, right.as_ref())?;
            Ok(series_out(&out))
        })(),
        "labels.aggregation_type" => (|| {
            let op = text(input, "op")?;
            let op = Aggregation::parse(op).ok_or_else(|| {
                LabelError::new(
                    "ValueError",
                    format!("'{op}' is not a valid AggregationType"),
                )
            })?;
            let feat = build_series(get(input, "feat")?)?;
            let batch = get(input, "batch")?
                .as_array()
                .ok_or_else(|| unreadable("a batch that is not a list"))?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .ok_or_else(|| unreadable("a batch index that is not an integer"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let size = get(input, "batch_size")?
                .as_u64()
                .ok_or_else(|| unreadable("a batch_size that is not a count"))?;
            let (out, mask) = crate::aggregate::aggregate_with_warnings(
                op,
                &feat,
                &batch,
                size as usize,
                flag(input, "filter_na")?,
                None,
                warnings,
            )?;
            Ok(labelled(&out, &mask))
        })(),
        "labels.aggregation" => (|| {
            let spec = get(input, "aggregation")?
                .as_object()
                .ok_or_else(|| unreadable("an aggregation that is not an object"))?;
            let aggr = text(spec, "aggr")?;
            let aggr = Aggregation::parse(aggr).ok_or_else(|| {
                LabelError::new(
                    "ValueError",
                    format!("'{aggr}' is not a valid AggregationType"),
                )
            })?;
            let unit = text(spec, "unit")?;
            let unit = crate::date_offset::Unit::parse(unit).ok_or_else(|| {
                LabelError::new("ValueError", format!("'{unit}' is not a valid TimeUnit"))
            })?;
            let start = match spec.get("start") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_i64()
                        .ok_or_else(|| unreadable("a start that is not an integer"))?,
                ),
            };
            let end = get(spec, "end")?
                .as_i64()
                .ok_or_else(|| unreadable("an end that is not an integer"))?;
            let window = Window::new(start, end, unit)?;
            let batch = Batch {
                feat: frames(get(input, "tables")?)?,
                time: get(input, "times")?
                    .as_object()
                    .ok_or_else(|| unreadable("times that are not an object"))?
                    .iter()
                    .map(|(name, spec)| Ok((name.clone(), build_series(spec)?)))
                    .collect::<Result<_, LabelError>>()?,
                batch: batches(get(input, "batches")?)?,
                anchor: build_series(get(input, "anchor_time")?)?,
                arrow_decimal_scales: Vec::new(),
                host: host.clone(),
                warnings: Default::default(),
            };
            let target = Node::Column {
                fqn: text(spec, "target")?.to_string(),
            };
            let result = execute_aggregation(
                &target,
                aggr,
                Some(&window),
                &batch,
                flag(input, "filter_na")?,
            );
            warnings.extend(batch.warnings.into_inner());
            let (out, mask) = result?;
            Ok(labelled(&out, &mask))
        })(),
        _ => return None,
    })
}
