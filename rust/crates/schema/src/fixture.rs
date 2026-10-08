// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The dates and composite-key operations of the parity fixtures
//! (`tests/parity/_fixture_ops.py`), on a case's JSON input, answered as
//! `{"output": X}` or `{"error": {"code", "message"}}`.

use std::fmt;

use serde_json::{Map, Value, json};

use crate::composite;
use crate::dates::{self, Conversion, TimeColumn};
use crate::dt::Host;
use crate::pyval::PyVal;

/// An operation that cannot be answered: unknown, or its input unreadable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputError(pub String);

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

fn bad(message: impl Into<String>) -> InputError {
    InputError(message.into())
}

fn output(value: Value) -> String {
    json!({ "output": value }).to_string()
}

fn error(code: &str, message: &str) -> String {
    json!({ "error": { "code": code, "message": message } }).to_string()
}

/// A column of text, as `pd.Series(values, dtype=object)` holds it.
fn strings(input: &Map<String, Value>) -> Result<Vec<Option<String>>, InputError> {
    let values = input
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("the input has no values"))?;
    values
        .iter()
        .map(|v| match v {
            Value::Null => Ok(None),
            Value::String(s) => Ok(Some(s.clone())),
            other => Err(bad(format!("a value that is not text: {other}"))),
        })
        .collect()
}

fn conversion_of(given: &Value) -> Result<Conversion, InputError> {
    let map = given
        .as_object()
        .ok_or_else(|| bad("a conversion is an object"))?;
    let mut conversion = Conversion::default();
    for (key, value) in map {
        match key.as_str() {
            "utc" => conversion.utc = value.as_bool().ok_or_else(|| bad("utc is a boolean"))?,
            "mixed" => {
                conversion.mixed = value.as_bool().ok_or_else(|| bad("mixed is a boolean"))?
            }
            "dayfirst" => {
                conversion.dayfirst = match value {
                    Value::Null => None,
                    Value::Bool(b) => Some(*b),
                    _ => return Err(bad("dayfirst is a boolean or null")),
                }
            }
            other => return Err(bad(format!("Conversion takes no argument {other:?}"))),
        }
    }
    Ok(conversion)
}

fn dates_op(
    operation: &str,
    input: &Map<String, Value>,
    host: &Host,
) -> Result<String, InputError> {
    Ok(match operation {
        "time_column.choose" => {
            let columns = frame(input.get("frame").ok_or_else(|| bad("a frame is given"))?)?;
            let sample = crate::engine::infer::SampleFrame {
                index: input.get("rows").and_then(Value::as_u64).unwrap_or(0) as usize,
                columns: columns
                    .into_iter()
                    .map(|(name, cells)| crate::engine::infer::SampleColumn {
                        name,
                        cells,
                        kind: crate::engine::infer::SampleKind::Object,
                    })
                    .collect(),
            };
            let candidates = string_list(input.get("candidates"), "candidates")?;
            let whole = input
                .get("whole")
                .filter(|whole| !whole.is_null())
                .map(|whole| {
                    let mut columns = std::collections::BTreeMap::new();
                    for (name, range) in whole
                        .get("columns")
                        .and_then(Value::as_object)
                        .ok_or_else(|| bad("ranges are an object"))?
                    {
                        columns.insert(
                            name.clone(),
                            crate::time_columns::Range {
                                low: PyVal::decode(&range[0]).map_err(bad)?,
                                high: PyVal::decode(&range[1]).map_err(bad)?,
                                held: range[2].as_u64().ok_or_else(|| bad("held is a count"))?
                                    as usize,
                            },
                        );
                    }
                    Ok::<_, InputError>(crate::time_columns::TableRange {
                        rows: whole
                            .get("rows")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| bad("rows is a count"))?
                            as usize,
                        columns,
                    })
                })
                .transpose()?;
            let choice = crate::time_columns::choose(
                &sample,
                &candidates,
                whole.as_ref(),
                input
                    .get("attempted")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                host,
            )
            .for_links(
                input
                    .get("linked")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            );
            let mut value =
                serde_json::to_value(&choice).map_err(|error| bad(error.to_string()))?;
            value["refused_why"] = json!(
                choice
                    .refused
                    .iter()
                    .map(crate::time_columns::Rejected::why)
                    .collect::<Vec<_>>()
            );
            value["unconfirmed_why"] = json!(
                choice
                    .unconfirmed
                    .iter()
                    .map(crate::time_columns::Rejected::why)
                    .collect::<Vec<_>>()
            );
            output(value)
        }
        "dates.day_comes_first" => output(json!(dates::day_comes_first(&strings(input)?))),
        "dates.looks_unambiguous" => output(json!(dates::looks_unambiguous(&strings(input)?))),
        "dates.shapes_written" => {
            let mut names: Vec<&str> = dates::shapes_written(&strings(input)?, host)
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            names.sort_unstable();
            output(json!(names))
        }
        "dates.reads_as_a_date" => output(json!(dates::reads_as_a_date(&strings(input)?, host))),
        "dates.conversion_that_works" => {
            let plausible = match input.get("only_plausible_years") {
                None => true,
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| bad("only_plausible_years is a boolean"))?,
            };
            let found = dates::conversion_that_works(&strings(input)?, plausible, false, host);
            output(found.map_or(Value::Null, |c| Value::Object(c.arguments())))
        }
        "dates.read" => {
            let values = strings(input)?;
            let conversion = match input.get("conversion") {
                Some(given) => conversion_of(given)?,
                None => match dates::conversion_that_works(&values, true, false, host) {
                    Some(c) => c,
                    None => return Ok(error("NO_CONVERSION", "no conversion reads the column")),
                },
            };
            match dates::read(&values, &conversion, host) {
                Ok(read) => output(json!(
                    (0..values.len())
                        .map(|i| read.isoformat(i))
                        .collect::<Vec<_>>()
                )),
                Err(e) => error(e.class, &e.message),
            }
        }
        "dates.pattern" => {
            let pattern = input
                .get("pattern")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("pattern names a pattern"))?;
            let value = input
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("value is text"))?;
            let whole = input.get("whole").and_then(Value::as_bool).unwrap_or(true);
            match dates::pattern_matches(pattern, value, whole) {
                Some(found) => output(json!(found)),
                None => error(
                    "AttributeError",
                    &format!("module 'sdm_client.core.dates' has no attribute '{pattern}'"),
                ),
            }
        }
        "dates.conversion_text" => {
            let conversion = conversion_of(&Value::Object(input.clone()))?;
            let arguments: Vec<String> = conversion.arguments().keys().cloned().collect();
            output(json!({"text": conversion.text(), "arguments": arguments}))
        }
        "time_column.read" => {
            let values = strings(input)?;
            let place = input
                .get("where")
                .and_then(Value::as_str)
                .unwrap_or("t.when");
            match dates::time_column(&values, place, host) {
                Ok(TimeColumn::Untouched(n)) => {
                    output(json!({"dtype": "object", "values": vec![Value::Null; n]}))
                }
                Ok(TimeColumn::Read(read)) => output(json!({
                    "dtype": read.dtype(),
                    "values": (0..values.len()).map(|i| read.isoformat(i)).collect::<Vec<_>>(),
                })),
                Err(refusal) => error(&refusal.code, &refusal.message),
            }
        }
        "infer.timestamp" => {
            let column = input
                .get("column")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("column names the column"))?;
            match dates::contains_timestamp(&strings(input)?, column, host).0 {
                Ok(found) => output(json!(found)),
                Err(e) => error(e.class, &e.message),
            }
        }
        other => return Err(bad(format!("no operation {other:?}"))),
    })
}

/// A fixture frame's columns, as pandas would build them.
fn frame(spec: &Value) -> Result<Vec<(String, Vec<PyVal>)>, InputError> {
    let columns = spec
        .get("columns")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("a frame has columns"))?;
    let mut out = Vec::new();
    for (name, column) in columns {
        let dtype = column
            .get("dtype")
            .and_then(Value::as_str)
            .unwrap_or("object");
        let values = column
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("a column has values"))?;
        let mut decoded = Vec::new();
        for v in values {
            let value = PyVal::decode(v).map_err(bad)?;
            decoded.push(match (dtype, value) {
                (d, PyVal::None) if d != "object" => PyVal::Float(f64::NAN),
                (d, PyVal::Int(n)) if d.starts_with("float") => PyVal::Float(n as f64),
                (_, v) => v,
            });
        }
        out.push((name.clone(), decoded));
    }
    Ok(out)
}

fn string_list(value: Option<&Value>, what: &str) -> Result<Vec<String>, InputError> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| bad(format!("{what} is a list")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| bad(format!("{what} holds text")))
        })
        .collect()
}

fn composite_op(operation: &str, input: &Map<String, Value>) -> Result<String, InputError> {
    Ok(match operation {
        "composite.encode_values" => {
            let parts = input.get("parts").ok_or_else(|| bad("parts are given"))?;
            let PyVal::List(parts) = PyVal::decode(parts).map_err(bad)? else {
                return Err(bad("parts is a list"));
            };
            match composite::encode_values(&parts) {
                Ok(text) => output(json!(text)),
                Err(e) => error("ValueError", &e),
            }
        }
        "composite.decode_value" => {
            let value = input
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("value is text"))?;
            let arity = input
                .get("arity")
                .and_then(Value::as_u64)
                .ok_or_else(|| bad("arity is a count"))?;
            match composite::decode_value(value, arity as usize) {
                Ok(parts) => output(json!(parts)),
                Err(e) => error("ValueError", &e),
            }
        }
        "composite.encode_identity" => {
            let columns = string_list(input.get("columns"), "columns")?;
            match composite::encode_identity(&columns) {
                Ok(text) => output(json!(text)),
                Err(e) => error("ValueError", &e),
            }
        }
        "composite.decode_identity" => {
            let derived = input
                .get("derived")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("derived is text"))?;
            output(composite::decode_identity(derived).map_or(Value::Null, |c| json!(c)))
        }
        "composite.encode_frame" => {
            let frame = frame(input.get("frame").ok_or_else(|| bad("a frame is given"))?)?;
            let columns = string_list(input.get("columns"), "columns")?;
            match composite::encode_frame(&frame, &columns) {
                Ok(rows) => output(json!(rows)),
                Err(e) => error("ValueError", &e),
            }
        }
        other => return Err(bad(format!("no operation {other:?}"))),
    })
}

/// One fixture operation on a case's whole input, with what it takes from
/// the host Python process.
pub fn fixture_op_in(operation: &str, input_json: &str, host: &Host) -> Result<String, InputError> {
    let input: Value =
        serde_json::from_str(input_json).map_err(|e| bad(format!("not JSON: {e}")))?;
    let input = input
        .as_object()
        .ok_or_else(|| bad("the input is a JSON object"))?;
    if operation.starts_with("composite.") {
        composite_op(operation, input)
    } else {
        dates_op(operation, input, host)
    }
}

/// [`fixture_op_in`] with [`Host::assumed`].
pub fn fixture_op(operation: &str, input_json: &str) -> Result<String, InputError> {
    fixture_op_in(operation, input_json, &Host::assumed())
}
