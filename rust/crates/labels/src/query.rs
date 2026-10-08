// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The resolved query the executor runs, read from the `tree` a validation
//! gives (`json_serde.to_json_dict` of a `ValidatedPredictiveQuery`, as the
//! PQL core writes it): the entity, the target, the `ASSUMING` condition,
//! and the nodes the executor walks. A node's kind is read from its fields,
//! as the tree carries no type tags.

use serde_json::{Map, Value};

use crate::LabelError;
use crate::aggregate::Aggregation;
use crate::column::{Obj, TimeUnit};
use crate::date_offset::{DateOffset, Unit};
use sdm_schema::dt::{self, Host};

/// `DateOffsetRange`: an aggregation's window, `(start, end]` in `unit`s
/// of the anchor, an open start reaching back without bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: Option<i64>,
    pub end: i64,
    pub unit: Unit,
}

impl Window {
    /// `DateOffsetRange(start, end, unit)`, with its checks: the start
    /// before the end, and each a 32-bit integer.
    pub fn new(start: Option<i64>, end: i64, unit: Unit) -> Result<Self, LabelError> {
        let refused = |message: String| LabelError::new("ValidationError", message);
        if let Some(start) = start
            && start >= end
        {
            return Err(refused(format!(
                "The start date offset '{start}' needs to be smaller than the end date offset \
                 '{end}' to specify a valid time range."
            )));
        }
        for value in start.into_iter().chain([end]) {
            if i32::try_from(value).is_err() {
                return Err(refused(format!(
                    "Unsupported constant. Constant {value} is outside of the range of \
                     supported integers (-2147483648, 2147483647)."
                )));
            }
        }
        Ok(Self { start, end, unit })
    }

    pub fn start_offset(&self) -> Option<DateOffset> {
        self.start.map(|start| DateOffset::single(self.unit, start))
    }

    pub fn end_offset(&self) -> DateOffset {
        DateOffset::single(self.unit, self.end)
    }
}

/// A `Constant`: its text and its PQL dtype.
#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    pub value: ConstantValue,
    pub dtype: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstantValue {
    Text(String),
    List(Vec<Constant>),
}

/// What `Constant.typed_value()` gives.
#[derive(Clone, Debug, PartialEq)]
pub enum Typed {
    None,
    Scalar(Obj),
    List(Vec<Typed>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Column {
        fqn: String,
    },
    Aggregation {
        target: Box<Node>,
        aggr: Aggregation,
        window: Option<Window>,
    },
    Join {
        rhs_target: Box<Node>,
    },
    Condition {
        target: Box<Node>,
        op: String,
        value: Constant,
    },
    Logical {
        left: Box<Node>,
        bool_op: String,
        right: Option<Box<Node>>,
    },
    Filter {
        target: Box<Node>,
        condition: Box<Node>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    pub entity: Node,
    pub target: Node,
    pub whatif: Option<Node>,
}

fn bad(message: impl Into<String>) -> LabelError {
    LabelError::new("ValueError", message)
}

fn field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value, LabelError> {
    object
        .get(name)
        .ok_or_else(|| bad(format!("a query node lacks {name:?}")))
}

fn text(value: &Value, what: &str) -> Result<String, LabelError> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| bad(format!("{what} is text, not {value}")))
}

fn window(value: &Value) -> Result<Option<Window>, LabelError> {
    let Value::Object(range) = value else {
        return Ok(None);
    };
    let unit = text(field(range, "unit")?, "a window's unit")?;
    let unit = Unit::parse(&unit).ok_or_else(|| bad(format!("no time unit {unit:?}")))?;
    let start = match range.get("start") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_i64()
                .ok_or_else(|| bad("a window's start is an integer"))?,
        ),
    };
    let end = field(range, "end")?
        .as_i64()
        .ok_or_else(|| bad("a window's end is an integer"))?;
    Window::new(start, end, unit).map(Some)
}

fn constant(value: &Value) -> Result<Constant, LabelError> {
    let Value::Object(object) = value else {
        return Err(bad(format!("a constant is an object, not {value}")));
    };
    let dtype = match object.get("dtype_maybe") {
        None | Some(Value::Null) => None,
        Some(Value::Object(array)) if array.len() == 1 && array.contains_key("nested_dtype") => {
            Some(format!(
                "array[{}]",
                text(&array["nested_dtype"], "an array's dtype")?
            ))
        }
        Some(v) => Some(text(v, "a constant's dtype")?),
    };
    let value = match field(object, "value")? {
        Value::String(s) => ConstantValue::Text(s.clone()),
        Value::Array(items) => {
            ConstantValue::List(items.iter().map(constant).collect::<Result<_, _>>()?)
        }
        Value::Null => ConstantValue::Text("None".to_string()),
        other => ConstantValue::Text(other.to_string()),
    };
    Ok(Constant { value, dtype })
}

impl Node {
    pub fn from_json(value: &Value) -> Result<Node, LabelError> {
        let Value::Object(object) = value else {
            return Err(bad(format!("a query node is an object, not {value}")));
        };
        let child = |name: &str| -> Result<Box<Node>, LabelError> {
            Ok(Box::new(Node::from_json(field(object, name)?)?))
        };
        if let Some(fqn) = object.get("fqn") {
            return Ok(Node::Column {
                fqn: text(fqn, "a column's name")?,
            });
        }
        if object.contains_key("aggr") {
            let aggr = text(field(object, "aggr")?, "an aggregation")?;
            return Ok(Node::Aggregation {
                target: child("target")?,
                aggr: Aggregation::parse(&aggr)
                    .ok_or_else(|| bad(format!("no aggregation {aggr:?}")))?,
                window: window(object.get("aggr_time_range").unwrap_or(&Value::Null))?,
            });
        }
        if object.contains_key("rhs_target") {
            return Ok(Node::Join {
                rhs_target: child("rhs_target")?,
            });
        }
        if object.contains_key("bool_op") {
            let right = match object.get("right") {
                None | Some(Value::Null) => None,
                Some(right) => Some(Box::new(Node::from_json(right)?)),
            };
            return Ok(Node::Logical {
                left: child("left")?,
                bool_op: text(field(object, "bool_op")?, "a boolean operator")?,
                right,
            });
        }
        if object.contains_key("condition") {
            return Ok(Node::Filter {
                target: child("target")?,
                condition: child("condition")?,
            });
        }
        if object.contains_key("op") {
            return Ok(Node::Condition {
                target: child("target")?,
                op: text(field(object, "op")?, "an operator")?,
                value: constant(field(object, "value")?)?,
            });
        }
        Err(bad("a query node of no kind the executor knows"))
    }

    /// `get_target_column_name()` of an aggregation's target.
    pub fn target_column(&self) -> Result<&str, LabelError> {
        match self {
            Node::Column { fqn } => Ok(fqn),
            Node::Filter { target, .. } => target.target_column(),
            Node::Aggregation { target, .. } => target.target_column(),
            _ => Err(LabelError::new(
                "AttributeError",
                "the node has no target column",
            )),
        }
    }
}

impl Query {
    /// The query of a resolved `tree`.
    pub fn from_json(text: &str) -> Result<Query, LabelError> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| bad(format!("the query is not JSON: {e}")))?;
        let Value::Object(object) = &value else {
            return Err(bad("the query is a JSON object"));
        };
        let whatif = match object.get("whatif_ast") {
            None | Some(Value::Null) => None,
            Some(node) => Some(Node::from_json(node)?),
        };
        Ok(Query {
            entity: Node::from_json(field(object, "entity_ast")?)?,
            target: Node::from_json(field(object, "target_ast")?)?,
            whatif,
        })
    }
}

/// Python's `int(text)`.
fn python_int(text: &str) -> Option<i64> {
    let t = text.trim();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    let digits: String = digits.chars().filter(|c| *c != '_').collect();
    digits.parse::<i64>().ok().map(|v| v * sign)
}

/// Python's `float(text)`.
pub fn python_float(text: &str) -> Option<f64> {
    let t = text.trim();
    let lower = t.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        let negative = lower.starts_with('-');
        let v = if unsigned == "nan" {
            f64::NAN
        } else {
            f64::INFINITY
        };
        return Some(if negative { -v } else { v });
    }
    if t.contains("__") || t.starts_with('_') || t.ends_with('_') {
        return None;
    }
    let cleaned: String = t.chars().filter(|c| *c != '_').collect();
    if cleaned
        .chars()
        .any(|c| !(c.is_ascii_digit() || "+-.eE".contains(c)))
    {
        return None;
    }
    cleaned.parse::<f64>().ok()
}

/// `str(value)[1:-1].encode('latin-1', 'backslashreplace').decode('unicode-escape')`.
fn unquote(text: &str) -> Result<String, LabelError> {
    let chars: Vec<char> = text.chars().collect();
    let inner: Vec<char> = if chars.len() >= 2 {
        chars[1..chars.len() - 1].to_vec()
    } else {
        Vec::new()
    };
    let mut out = String::new();
    let mut i = 0;
    let hex = |digits: &[char], width: usize, at: usize| -> Result<char, LabelError> {
        let text: String = digits.iter().take(width).collect();
        if text.chars().count() != width || !text.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(LabelError::new(
                "UnicodeDecodeError",
                format!(
                    "'unicodeescape' codec can't decode bytes in position {at}-{}: truncated \\xXX escape",
                    at + 1
                ),
            ));
        }
        u32::from_str_radix(&text, 16)
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| LabelError::new("UnicodeDecodeError", "illegal Unicode character"))
    };
    while i < inner.len() {
        let c = inner[i];
        if c != '\\' || i + 1 >= inner.len() {
            out.push(c);
            i += 1;
            continue;
        }
        let next = inner[i + 1];
        i += 2;
        match next {
            '\n' => {}
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'a' => out.push('\u{7}'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'v' => out.push('\u{b}'),
            'x' => {
                out.push(hex(&inner[i..], 2, i - 2)?);
                i += 2;
            }
            'u' => {
                out.push(hex(&inner[i..], 4, i - 2)?);
                i += 4;
            }
            'U' => {
                out.push(hex(&inner[i..], 8, i - 2)?);
                i += 8;
            }
            '0'..='7' => {
                let mut value = next.to_digit(8).unwrap_or(0);
                let mut taken = 0;
                while taken < 2 && i < inner.len() && inner[i].is_digit(8) {
                    value = value * 8 + inner[i].to_digit(8).unwrap_or(0);
                    i += 1;
                    taken += 1;
                }
                out.push(char::from_u32(value).unwrap_or('\u{fffd}'));
            }
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    Ok(out)
}

impl Constant {
    /// `typed_value()`; `host` is what reading a date takes from the
    /// Python process (`pd.to_datetime` of a time constant).
    pub fn typed(&self, host: &Host) -> Result<Typed, LabelError> {
        self.typed_with_warnings(host, &mut Vec::new())
    }

    pub fn typed_with_warnings(
        &self,
        host: &Host,
        warnings: &mut Vec<String>,
    ) -> Result<Typed, LabelError> {
        let Some(dtype) = &self.dtype else {
            return Ok(Typed::None);
        };
        let is_list = matches!(dtype.as_str(), "floatlist" | "intlist" | "stringlist")
            || dtype.starts_with("array[");
        match &self.value {
            ConstantValue::List(items) => {
                return Ok(Typed::List(
                    items
                        .iter()
                        .map(|item| item.typed_with_warnings(host, warnings))
                        .collect::<Result<_, _>>()?,
                ));
            }
            ConstantValue::Text(_) if is_list => {
                return Err(LabelError::new("AssertionError", ""));
            }
            ConstantValue::Text(_) => {}
        }
        let ConstantValue::Text(text) = &self.value else {
            return Ok(Typed::None);
        };
        Ok(Typed::Scalar(match dtype.as_str() {
            "bool" => Obj::Bool(text.to_lowercase() == "true"),
            "int" | "byte" | "int16" | "int32" | "int64" => {
                Obj::Int(python_int(text).ok_or_else(|| {
                    LabelError::new(
                        "ValueError",
                        format!("invalid literal for int() with base 10: '{text}'"),
                    )
                })?)
            }
            "float" | "float32" | "float64" => Obj::Float(python_float(text).ok_or_else(|| {
                LabelError::new(
                    "ValueError",
                    format!("could not convert string to float: '{text}'"),
                )
            })?),
            "string" | "binary" => Obj::Str(unquote(text)?),
            "date" | "time" => {
                // pd.to_datetime(value): a format guessed, else each value
                // read as ISO 8601 and then by dateutil.
                let (read, emitted) =
                    dt::to_datetime(&[Some(text.clone())], dt::Call::default(), host);
                warnings.extend(emitted);
                let read = read.map_err(|e| LabelError::new(e.class, e.message))?;
                match (read.tz, read.values.first().copied().flatten()) {
                    (dt::ColumnTz::Naive, Some(ns)) => Obj::Time(
                        TimeUnit::Ns,
                        i64::try_from(ns).map_err(|_| {
                            LabelError::new(
                                "OutOfBoundsDatetime",
                                format!("Out of bounds nanosecond timestamp: {text}"),
                            )
                        })?,
                    ),
                    (dt::ColumnTz::Naive, None) => Obj::NaT,
                    (_, _) => {
                        return Err(LabelError::new(
                            "TypeError",
                            "Invalid comparison between dtype=datetime64[ns] and Timestamp",
                        ));
                    }
                }
            }
            _ => Obj::Str(text.clone()),
        }))
    }
}
