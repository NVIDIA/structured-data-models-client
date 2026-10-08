// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python's `json.dumps`, byte for byte, as the dialects call it to write a
//! key list into one parameter: `ensure_ascii` escapes, `float.__repr__`,
//! integers of any size, and the `default` PostgreSQL's and Databricks'
//! `dumps` pass.

use sdm_pql::pyrepr::float_repr;

use crate::Error;
use crate::py::Py;

/// How `json.dumps` is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dumps {
    /// `separators=(',', ':')` rather than the default `(', ', ': ')`.
    pub compact: bool,
    /// `default=_json_default`: a value JSON has no form for is written as
    /// what its `item()` gives, or else as its `str`.
    pub default: bool,
}

impl Dumps {
    /// `json.dumps(obj, separators=(',', ':'))`.
    pub const COMPACT: Self = Self {
        compact: true,
        default: false,
    };
    /// `json.dumps(obj)`.
    pub const PLAIN: Self = Self {
        compact: false,
        default: false,
    };
    /// The dialects' own `dumps`: compact, with `_json_default`.
    pub const WITH_DEFAULT: Self = Self {
        compact: true,
        default: true,
    };

    /// `obj` as JSON text, or the `TypeError` Python raises.
    pub fn dumps(self, obj: &Py) -> Result<String, Error> {
        let mut out = String::new();
        self.write(obj, &mut out, 0)?;
        Ok(out)
    }

    fn write(self, obj: &Py, out: &mut String, depth: usize) -> Result<(), Error> {
        if depth > 900 {
            return Err(Error::new("maximum recursion depth exceeded"));
        }
        let (item_sep, key_sep) = if self.compact {
            (",", ":")
        } else {
            (", ", ": ")
        };
        match obj {
            Py::None => out.push_str("null"),
            Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Py::Str(s) => quote_into(s, out),
            // A `Dtype` is a `StrEnum`, which JSON writes as its text.
            Py::Dtype(dtype) => quote_into(dtype.value(), out),
            Py::Int(int) => out.push_str(&int.to_string()),
            Py::Float(x) => out.push_str(&float_text(*x)),
            // A numpy float64 is a Python float, which JSON writes as one.
            Py::Np(dtype, value) if dtype == "float64" => self.write(value, out, depth + 1)?,
            Py::List(items) | Py::Tuple(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item_sep);
                    }
                    self.write(item, out, depth + 1)?;
                }
                out.push(']');
            }
            Py::Dict(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item_sep);
                    }
                    quote_into(key, out);
                    out.push_str(key_sep);
                    self.write(value, out, depth + 1)?;
                }
                out.push('}');
            }
            other => {
                if !self.default {
                    return Err(not_serializable(other));
                }
                let replaced = json_default(other)?;
                self.write(&replaced, out, depth + 1)?;
            }
        }
        Ok(())
    }
}

/// `_json_default(obj)`: `obj.item()` where it has one, else `str(obj)`.
fn json_default(obj: &Py) -> Result<Py, Error> {
    match obj {
        Py::Np(_, value) => Ok((**value).clone()),
        Py::Ndarray(_, values) if values.len() == 1 => Ok(values[0].clone()),
        Py::Ndarray(..) => Err(Error::new(
            "can only convert an array of size 1 to a Python scalar",
        )),
        Py::Series(series) if series.values.len() == 1 => Ok(series.values[0].clone()),
        other => Ok(Py::Str(
            other.py_str().map_err(|_| not_serializable(other))?,
        )),
    }
}

fn not_serializable(obj: &Py) -> Error {
    Error::new(format!("Object of type {obj:?} is not JSON serializable"))
}

/// A float as `json.dumps` writes it.
pub fn float_text(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x == f64::INFINITY {
        "Infinity".to_string()
    } else if x == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        float_repr(x)
    }
}

/// `text` as a JSON string with `ensure_ascii`: printable ASCII as it is,
/// everything else escaped, outside the BMP as a surrogate pair.
pub fn quote_into(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// `text` as `json.dumps` writes a string.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    quote_into(text, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int::Int;

    #[test]
    fn writes_as_python() {
        let value = Py::List(vec![
            Py::Int(Int::parse("-100000000000000000000").unwrap()),
            Py::Float(1e16),
            Py::Float(f64::NAN),
            Py::Float(-0.0),
            Py::str("é\u{1f600}\u{7f}\"/\u{1}"),
            Py::Tuple(vec![Py::Bool(true), Py::None]),
            Py::Dict(vec![("k".into(), Py::List(vec![]))]),
        ]);
        assert_eq!(
            Dumps::PLAIN.dumps(&value).unwrap(),
            r#"[-100000000000000000000, 1e+16, NaN, -0.0, "\u00e9\ud83d\ude00\u007f\"/\u0001", [true, null], {"k": []}]"#
        );
        assert_eq!(
            Dumps::COMPACT.dumps(&value).unwrap(),
            r#"[-100000000000000000000,1e+16,NaN,-0.0,"\u00e9\ud83d\ude00\u007f\"/\u0001",[true,null],{"k":[]}]"#
        );
    }

    #[test]
    fn default_takes_item_then_str() {
        let value = Py::List(vec![
            Py::Np("int64".into(), Box::new(Py::int(3))),
            Py::Decimal("1.50".into()),
            Py::NaT,
        ]);
        assert_eq!(
            Dumps::WITH_DEFAULT.dumps(&value).unwrap(),
            r#"[3,"1.50","NaT"]"#
        );
        assert!(Dumps::COMPACT.dumps(&value).is_err());
    }
}
