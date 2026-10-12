// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The label vectors (T1.27, decision E19; register PK-2 and PK-3):
//! `sdm_engine::time::offset` and `sdm_engine::labels` against the
//! reference's own answers in `labels.json`, recorded by `labels_record.py`
//! under the reference's environment (a protected path, F4), compared
//! exactly: a window's offsets added to anchors at month ends, on leap days
//! and with nanoseconds, as the sampler's `_by_time` seeds are made; the
//! fixed step a window of months walks back by; the executor's
//! aggregations, in float32 bits; its comparisons; and a decimal read at
//! float32.
//!
//! [`problems`] fails closed: a perturbed answer, and evidence missing,
//! empty or retyped, each fail (seeded below).

use sdm_contract::pql::TimeUnit;
use sdm_engine::labels::{self, Aggregation, Bounds, Constant, Labels, RelOp, Values};
use sdm_engine::time::offset::{Offset, timeframe};
use serde_json::Value;

const VECTORS: &str = include_str!("labels.json");

/// How many vectors of each kind `problems` compared.
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    offsets: usize,
    timeframes: usize,
    aggregations: usize,
    conditions: usize,
    decimals: usize,
}

fn unit(name: &str) -> Result<TimeUnit, String> {
    Ok(match name {
        "seconds" => TimeUnit::Seconds,
        "minutes" => TimeUnit::Minutes,
        "hours" => TimeUnit::Hours,
        "days" => TimeUnit::Days,
        "weeks" => TimeUnit::Weeks,
        "months" => TimeUnit::Months,
        other => return Err(format!("a unit {other:?}")),
    })
}

fn list<'a>(document: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    document
        .get(key)
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| format!("{key}: no vectors"))
}

fn int(value: &Value, at: &str) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| format!("{at}: not an integer"))
}

fn digits(value: &Value, at: &str) -> Result<i128, String> {
    value
        .as_str()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| format!("{at}: not digits"))
}

fn bits(value: &Value, at: &str) -> Result<f32, String> {
    value
        .as_str()
        .and_then(|t| u32::from_str_radix(t, 16).ok())
        .map(f32::from_bits)
        .ok_or_else(|| format!("{at}: not float32 bits"))
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Each way the core departs from the vectors in `document`, and how
/// many of each it compared.
fn problems(document: &Value) -> (Vec<String>, Counts) {
    let mut found = Vec::new();
    let mut counts = Counts::default();
    if document.get("format").and_then(Value::as_str) != Some("sdm-labels-vectors/1") {
        return (vec!["not an sdm-labels-vectors/1 document".into()], counts);
    }
    let mut each =
        |key: &str, check: &mut dyn FnMut(&Value) -> Result<Option<String>, String>| match list(
            document, key,
        ) {
            Err(problem) => {
                found.push(problem);
                0
            }
            Ok(vectors) => {
                for (i, vector) in vectors.iter().enumerate() {
                    match check(vector) {
                        Ok(None) => {}
                        Ok(Some(problem)) => found.push(format!("{key}[{i}]: {problem}")),
                        Err(problem) => found.push(format!("{key}[{i}]: {problem}")),
                    }
                }
                vectors.len()
            }
        };
    counts.offsets = each("offsets", &mut |v| {
        let offset = Offset {
            count: int(&v["count"], "count")?,
            unit: unit(v["unit"].as_str().unwrap_or_default())?,
        };
        let anchor = int(&v["anchor"], "anchor")?;
        let wanted = match &v["at"] {
            Value::Null => None,
            other => Some(int(other, "at")?),
        };
        let mine = offset.add_to(anchor, "t").ok();
        Ok((mine != wanted).then(|| format!("{anchor} + {offset:?}: {mine:?}, not {wanted:?}")))
    });
    counts.timeframes = each("timeframes", &mut |v| {
        let unit = unit(v["unit"].as_str().unwrap_or_default())?;
        let start = int(&v["start"], "start")?;
        let end = int(&v["end"], "end")?;
        let wanted = i128::from(int(&v["nanos"], "nanos")?);
        let at = |n: i64| Offset { count: n, unit };
        let mine = timeframe(at(start), at(end), "t").map_err(|e| format!("{e:?}"))?;
        Ok((mine != wanted).then(|| format!("{start}..{end} {unit:?}: {mine}, not {wanted}")))
    });
    counts.aggregations = each("aggregations", &mut |v| {
        let aggregation = match v["aggregation"].as_str() {
            Some("sum") => Aggregation::Sum,
            Some("count") => Aggregation::Count,
            Some("avg") => Aggregation::Avg,
            Some("min") => Aggregation::Min,
            Some("max") => Aggregation::Max,
            other => return Err(format!("an aggregation {other:?}")),
        };
        let examples =
            usize::try_from(int(&v["examples"], "examples")?).map_err(|e| e.to_string())?;
        let batch: Vec<usize> = v["batch"]
            .as_array()
            .ok_or("batch: not a list")?
            .iter()
            .map(|b| int(b, "batch").map(|b| b as usize))
            .collect::<Result<_, _>>()?;
        let rows = v["values"].as_array().ok_or("values: not a list")?;
        let values = match v["kind"].as_str() {
            Some("float32") => Values::Float(
                rows.iter()
                    .map(|r| (!r.is_null()).then(|| bits(r, "value")).transpose())
                    .collect::<Result<_, _>>()?,
            ),
            Some("int64") => Values::Int(
                rows.iter()
                    .map(|r| {
                        (!r.is_null())
                            .then(|| digits(r, "value").map(|d| d as i64))
                            .transpose()
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some("decimal") => Values::Decimal {
                values: rows
                    .iter()
                    .map(|r| (!r.is_null()).then(|| digits(r, "value")).transpose())
                    .collect::<Result<_, _>>()?,
                scale: 18,
            },
            other => return Err(format!("a kind {other:?}")),
        };
        let wanted: Vec<f32> = v["labels"]
            .as_array()
            .ok_or("labels: not a list")?
            .iter()
            .map(|l| bits(l, "label"))
            .collect::<Result<_, _>>()?;
        let mask: Vec<bool> = v["mask"]
            .as_array()
            .ok_or("mask: not a list")?
            .iter()
            .map(|m| m.as_bool().ok_or("mask: not a boolean"))
            .collect::<Result<_, _>>()?;
        let ends = vec![0_i64; examples];
        let out = labels::aggregate(
            aggregation,
            examples,
            &batch,
            &values,
            &vec![Some(0); batch.len()],
            Bounds {
                starts: None,
                ends: &ends,
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        let mine: Vec<f32> = out
            .values
            .iter()
            .zip(&out.present)
            .filter(|(_, p)| **p)
            .map(|(v, _)| *v)
            .collect();
        let agree = out.present == mask
            && mine.len() == wanted.len()
            && mine.iter().zip(&wanted).all(|(a, b)| same_f32(*a, *b));
        Ok((!agree).then(|| format!("{mine:?} {:?}, not {wanted:?} {mask:?}", out.present)))
    });
    counts.conditions = each("conditions", &mut |v| {
        let op = match v["op"].as_str() {
            Some("eq") => RelOp::Eq,
            Some("ne") => RelOp::Ne,
            Some("lt") => RelOp::Lt,
            Some("le") => RelOp::Le,
            Some("gt") => RelOp::Gt,
            Some("ge") => RelOp::Ge,
            other => return Err(format!("an operator {other:?}")),
        };
        let rows = v["left"].as_array().ok_or("left: not a list")?;
        let left = match v["kind"].as_str() {
            Some("float32") => Labels::Float(
                rows.iter()
                    .map(|r| bits(r, "left"))
                    .collect::<Result<_, _>>()?,
            ),
            Some("int64") => Labels::Int(
                rows.iter()
                    .map(|r| digits(r, "left").map(|d| d as i64))
                    .collect::<Result<_, _>>()?,
            ),
            other => return Err(format!("a kind {other:?}")),
        };
        let text = v["constant"]["text"].as_str().ok_or("constant: no text")?;
        let constant = match v["constant"]["type"].as_str() {
            Some("int") => Constant::Int(text.parse().map_err(|_| "constant: not digits")?),
            Some("float") => Constant::Float(text.parse().map_err(|_| "constant: not a float")?),
            other => return Err(format!("a constant {other:?}")),
        };
        let wanted: Vec<bool> = v["answer"]
            .as_array()
            .ok_or("answer: not a list")?
            .iter()
            .map(|a| a.as_bool().ok_or("answer: not a boolean"))
            .collect::<Result<_, _>>()?;
        let mine = labels::compare(&left, op, constant).map_err(|e| format!("{e:?}"))?;
        Ok((mine != wanted).then(|| format!("{op:?} {text}: {mine:?}, not {wanted:?}")))
    });
    counts.decimals = each("decimals", &mut |v| {
        let unscaled = digits(&v["unscaled"], "unscaled")?;
        let scale = i8::try_from(int(&v["scale"], "scale")?).map_err(|e| e.to_string())?;
        let wanted = bits(&v["f32"], "f32")?;
        let mine = labels::decimal_to_f64(unscaled, scale) as f32;
        Ok((!same_f32(mine, wanted))
            .then(|| format!("{unscaled}e-{scale}: {mine:e}, not {wanted:e}")))
    });
    (found, counts)
}

fn vectors() -> Value {
    serde_json::from_str(VECTORS).unwrap()
}

#[test]
fn the_core_s_labels_and_offsets_are_the_reference_s() {
    let (found, counts) = problems(&vectors());
    assert_eq!(found, Vec::<String>::new());
    assert!(counts.offsets > 2_500, "{counts:?}");
    assert!(counts.timeframes > 1_000, "{counts:?}");
    assert!(counts.aggregations > 150, "{counts:?}");
    assert!(counts.conditions > 400, "{counts:?}");
    assert!(counts.decimals > 500, "{counts:?}");
}

#[test]
fn the_vector_check_fails_closed() {
    let document = vectors();
    // A perturbed answer of each kind.
    for (key, field) in [
        ("offsets", "at"),
        ("timeframes", "nanos"),
        ("conditions", "answer"),
        ("aggregations", "labels"),
        ("decimals", "f32"),
    ] {
        let mut changed = document.clone();
        let first = changed[key]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|v| match &v[field] {
                Value::Array(a) => !a.is_empty(),
                Value::Null => false,
                _ => true,
            })
            .unwrap();
        first[field] = match &first[field] {
            Value::Number(n) => Value::from(n.as_i64().unwrap() + 1),
            Value::String(_) => Value::from("7fc00001"),
            Value::Array(items) => {
                let mut items = items.clone();
                items[0] = match &items[0] {
                    Value::Bool(b) => Value::Bool(!b),
                    _ => Value::from("3f800001"),
                };
                Value::Array(items)
            }
            other => other.clone(),
        };
        assert!(!problems(&changed).0.is_empty(), "{key}");
    }
    // Evidence missing, empty or retyped.
    for key in [
        "offsets",
        "timeframes",
        "aggregations",
        "conditions",
        "decimals",
    ] {
        let mut missing = document.clone();
        missing.as_object_mut().unwrap().remove(key);
        assert!(!problems(&missing).0.is_empty(), "{key} missing");
        let mut empty = document.clone();
        empty[key] = Value::Array(vec![]);
        assert!(!problems(&empty).0.is_empty(), "{key} empty");
        let mut retyped = document.clone();
        retyped[key] = Value::from("vectors");
        assert!(!problems(&retyped).0.is_empty(), "{key} retyped");
    }
    let mut retyped = document.clone();
    retyped["offsets"][0]["anchor"] = Value::from("1706659200000000000");
    assert!(!problems(&retyped).0.is_empty());
    let mut format = document;
    format["format"] = Value::from("sdm-labels-vectors/2");
    assert!(!problems(&format).0.is_empty());
}
