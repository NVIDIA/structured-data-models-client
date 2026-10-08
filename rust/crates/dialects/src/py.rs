// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python values as the dialects take and give them, read from and written
//! to the JSON of the dialect-call capture
//! (`tests/rust_core/dialects/CONTRACT.md`, "Encoded values").

use serde_json::{Map, Number, Value, json};

use crate::Error;
use crate::arrow::{ArrowArray, ArrowType};
use crate::int::Int;
use crate::time::{Stamp, timedelta_text};

/// `predictive_query`'s `Dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Int,
    Byte,
    Int16,
    Int32,
    Int64,
    Float,
    Float32,
    Float64,
    String,
    Binary,
    Date,
    Time,
    Timedelta,
    FloatList,
    IntList,
    StringList,
    Unsupported,
}

impl Dtype {
    const ALL: [(Self, &'static str); 18] = [
        (Self::Bool, "bool"),
        (Self::Int, "int"),
        (Self::Byte, "byte"),
        (Self::Int16, "int16"),
        (Self::Int32, "int32"),
        (Self::Int64, "int64"),
        (Self::Float, "float"),
        (Self::Float32, "float32"),
        (Self::Float64, "float64"),
        (Self::String, "string"),
        (Self::Binary, "binary"),
        (Self::Date, "date"),
        (Self::Time, "time"),
        (Self::Timedelta, "timedelta"),
        (Self::FloatList, "floatlist"),
        (Self::IntList, "intlist"),
        (Self::StringList, "stringlist"),
        (Self::Unsupported, "unsupported"),
    ];

    pub fn parse(value: &str) -> Result<Self, Error> {
        Self::ALL
            .iter()
            .find(|(_, name)| *name == value)
            .map(|(dtype, _)| *dtype)
            .ok_or_else(|| Error::new(format!("no Dtype {value:?}")))
    }

    pub fn value(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(dtype, _)| *dtype == self)
            .map(|(_, name)| *name)
            .unwrap_or("unsupported")
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            Self::Int | Self::Byte | Self::Int16 | Self::Int32 | Self::Int64
        )
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::Float32 | Self::Float64)
    }

    pub fn is_string(self) -> bool {
        matches!(self, Self::String | Self::Binary)
    }

    pub fn is_list(self) -> bool {
        matches!(self, Self::FloatList | Self::IntList | Self::StringList)
    }
}

/// A `fractions.Fraction`, in lowest terms with a positive denominator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fraction {
    pub numerator: i128,
    pub denominator: i128,
}

impl Fraction {
    pub fn new(numerator: i128, denominator: i128) -> Result<Self, Error> {
        if denominator == 0 {
            return Err(Error::new("Fraction(n, 0)"));
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
        let sign = if denominator < 0 { -1 } else { 1 };
        let divisor = i128::try_from(divisor).map_err(|_| Error::new("a fraction too large"))?;
        Ok(Self {
            numerator: sign * numerator / divisor,
            denominator: sign * denominator / divisor,
        })
    }

    pub fn integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    /// `str(fraction)`.
    pub fn text(&self) -> String {
        if self.denominator == 1 {
            self.numerator.to_string()
        } else {
            format!("{}/{}", self.numerator, self.denominator)
        }
    }
}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        Some(left.cmp(&right))
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// A pandas `Series` with the default index.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub dtype: String,
    pub values: Vec<Py>,
    pub name: Box<Py>,
}

/// A Python value.
#[derive(Clone, Debug, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(Int),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    /// A `dict` with text keys, in insertion order.
    Dict(Vec<(String, Py)>),
    /// A `decimal.Decimal`, as `str` writes it.
    Decimal(String),
    /// A pandas `Timestamp`.
    Timestamp(Stamp),
    /// A `datetime.datetime`, as `isoformat` writes it.
    Datetime(String),
    /// A `datetime.date`, as `isoformat` writes it.
    Date(String),
    /// A `datetime.time`, as `isoformat` writes it.
    Time(String),
    /// A `datetime.timedelta`, in microseconds.
    Timedelta(i64),
    Bytes(Vec<u8>),
    Uuid(String),
    /// `pandas.NA`.
    Na,
    /// `pandas.NaT`.
    NaT,
    /// A numpy scalar of the dtype named, holding what `item()` gives.
    Np(String, Box<Py>),
    /// A numpy array of the dtype named.
    Ndarray(String, Vec<Py>),
    Dtype(Dtype),
    Fraction(Fraction),
    Arrow(ArrowArray),
    ArrowType(ArrowType),
    Series(Series),
    /// A `NamedTuple` or dataclass instance: its class, as
    /// `module.QualifiedName`, and its fields.
    Record(String, Vec<(String, Py)>),
    /// A function quoting one name with this character.
    Quote(char),
}

impl Py {
    pub fn int(value: i64) -> Self {
        Self::Int(Int::from(value))
    }

    pub fn str(text: impl Into<String>) -> Self {
        Self::Str(text.into())
    }

    /// The value `value`, the capture's JSON, stands for.
    pub fn decode(value: &Value) -> Result<Self, Error> {
        Ok(match value {
            Value::Null => Self::None,
            Value::Bool(b) => Self::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Self::Int(Int::from(i))
                } else if let Some(u) = n.as_u64() {
                    Self::Int(Int::from_i128(i128::from(u)))
                } else {
                    Self::Float(n.as_f64().ok_or_else(|| Error::new("a bad number"))?)
                }
            }
            Value::String(s) => Self::Str(s.clone()),
            Value::Array(items) => Self::List(decode_all(items)?),
            Value::Object(map) => {
                if map.len() == 1
                    && let Some((tag, body)) = map.iter().next()
                    && tag.starts_with('$')
                {
                    return decode_tagged(tag, body);
                }
                Self::Dict(decode_map(map)?)
            }
        })
    }

    /// The capture's JSON of this value, as `_codec.encode` writes it.
    pub fn encode(&self) -> Result<Value, Error> {
        Ok(match self {
            Self::None => Value::Null,
            Self::Bool(b) => Value::Bool(*b),
            Self::Int(int) => match int.to_i64() {
                Some(i) => json!(i),
                None => json!({"$int": int.to_string()}),
            },
            Self::Float(x) => float_json(*x),
            Self::Str(s) => Value::String(s.clone()),
            Self::List(items) => Value::Array(encode_all(items)?),
            Self::Tuple(items) => json!({"$tuple": encode_all(items)?}),
            Self::Dict(fields) => {
                let map = encode_fields(fields)?;
                if map.len() == 1 && map.keys().all(|k| k.starts_with('$')) {
                    json!({"$object": map})
                } else {
                    Value::Object(map)
                }
            }
            Self::Decimal(text) => json!({"$decimal": text}),
            Self::Timestamp(stamp) => json!({"$timestamp": stamp.isoformat('T')}),
            Self::Datetime(text) => json!({"$datetime": text}),
            Self::Date(text) => json!({"$date": text}),
            Self::Time(text) => json!({"$time": text}),
            Self::Timedelta(micros) => json!({"$timedelta_us": micros}),
            Self::Bytes(bytes) => json!({"$bytes": base64_encode(bytes)}),
            Self::Uuid(text) => json!({"$uuid": text}),
            Self::Na => json!({"$na": null}),
            Self::NaT => json!({"$nat": null}),
            Self::Np(dtype, value) => json!({"$np": {"type": dtype, "value": value.encode()?}}),
            Self::Ndarray(dtype, values) => {
                json!({"$ndarray": {"dtype": dtype, "values": encode_all(values)?}})
            }
            Self::Dtype(dtype) => json!({"$dtype": dtype.value()}),
            Self::Fraction(f) => json!({"$fraction": [
                Self::Int(Int::from_i128(f.numerator)).encode()?,
                Self::Int(Int::from_i128(f.denominator)).encode()?,
            ]}),
            Self::Arrow(array) => {
                let mut body = Map::new();
                body.insert("type".into(), json!(array.data_type.to_string()));
                body.insert("values".into(), Value::Array(encode_all(&array.values)?));
                if let Some(chunks) = &array.chunks {
                    body.insert("chunks".into(), json!(chunks));
                }
                json!({"$arrow": body})
            }
            Self::ArrowType(t) => json!({"$arrow_type": t.to_string()}),
            Self::Series(series) => json!({"$series": {
                "dtype": series.dtype,
                "values": encode_all(&series.values)?,
                "name": series.name.encode()?,
            }}),
            Self::Record(class, fields) => json!({"$record": {
                "class": class,
                "fields": encode_fields(fields)?,
            }}),
            Self::Quote(c) => json!({"$quote": c.to_string()}),
        })
    }

    /// `str(value)`, for the values that have a plain one.
    pub fn py_str(&self) -> Result<String, Error> {
        Ok(match self {
            Self::None => "None".into(),
            Self::Bool(b) => if *b { "True" } else { "False" }.into(),
            Self::Int(int) => int.to_string(),
            Self::Float(x) => sdm_pql::pyrepr::float_repr(*x),
            Self::Str(s) => s.clone(),
            Self::Decimal(text) | Self::Date(text) | Self::Time(text) | Self::Uuid(text) => {
                text.clone()
            }
            Self::Datetime(text) => replace_separator(text),
            Self::Timestamp(stamp) => stamp.isoformat(' '),
            Self::Timedelta(micros) => timedelta_text(*micros),
            Self::Bytes(bytes) => bytes_repr(bytes),
            Self::Na => "<NA>".into(),
            Self::NaT => "NaT".into(),
            Self::Fraction(f) => f.text(),
            Self::Np(_, value) => value.py_str()?,
            other => return Err(Error::new(format!("no plain str() of {other:?}"))),
        })
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

fn replace_separator(iso: &str) -> String {
    let mut out = iso.to_string();
    if out.len() > 10 && out.as_bytes()[10] == b'T' {
        out.replace_range(10..11, " ");
    }
    out
}

/// A float as JSON holds it in the capture: a number, or `$float` where
/// JSON has none.
pub fn float_json(x: f64) -> Value {
    if x.is_nan() {
        json!({"$float": "nan"})
    } else if x.is_infinite() {
        json!({"$float": if x > 0.0 { "inf" } else { "-inf" }})
    } else {
        Number::from_f64(x).map_or(Value::Null, Value::Number)
    }
}

fn decode_all(items: &[Value]) -> Result<Vec<Py>, Error> {
    items.iter().map(Py::decode).collect()
}

fn decode_map(map: &Map<String, Value>) -> Result<Vec<(String, Py)>, Error> {
    map.iter()
        .map(|(k, v)| Ok((k.clone(), Py::decode(v)?)))
        .collect()
}

fn encode_all(items: &[Py]) -> Result<Vec<Value>, Error> {
    items.iter().map(Py::encode).collect()
}

fn encode_fields(fields: &[(String, Py)]) -> Result<Map<String, Value>, Error> {
    fields
        .iter()
        .map(|(k, v)| Ok((k.clone(), v.encode()?)))
        .collect()
}

fn text_of<'a>(tag: &str, body: &'a Value) -> Result<&'a str, Error> {
    body.as_str()
        .ok_or_else(|| Error::new(format!("{tag} does not hold text")))
}

fn field<'a>(tag: &str, body: &'a Value, name: &str) -> Result<&'a Value, Error> {
    body.get(name)
        .ok_or_else(|| Error::new(format!("{tag} has no {name:?}")))
}

fn decode_tagged(tag: &str, body: &Value) -> Result<Py, Error> {
    Ok(match tag {
        "$int" => Py::Int(Int::parse(text_of(tag, body)?).ok_or_else(|| Error::new("a bad $int"))?),
        "$float" => Py::Float(match text_of(tag, body)? {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => other
                .parse()
                .map_err(|_| Error::new(format!("a bad $float {other:?}")))?,
        }),
        "$decimal" => Py::Decimal(text_of(tag, body)?.to_string()),
        "$timestamp" => Py::Timestamp(Stamp::parse(text_of(tag, body)?)?),
        "$datetime" => {
            let text = text_of(tag, body)?;
            Stamp::parse(text)?;
            Py::Datetime(text.to_string())
        }
        "$date" => Py::Date(text_of(tag, body)?.to_string()),
        "$time" => Py::Time(text_of(tag, body)?.to_string()),
        "$timedelta_us" => Py::Timedelta(
            body.as_i64()
                .ok_or_else(|| Error::new("a bad $timedelta_us"))?,
        ),
        "$bytes" => Py::Bytes(base64_decode(text_of(tag, body)?)?),
        "$uuid" => Py::Uuid(text_of(tag, body)?.to_string()),
        "$na" => Py::Na,
        "$nat" => Py::NaT,
        "$tuple" => Py::Tuple(decode_all(
            body.as_array()
                .ok_or_else(|| Error::new("$tuple does not hold a list"))?,
        )?),
        "$np" => {
            let dtype = text_of(tag, field(tag, body, "type")?)?;
            Py::Np(
                dtype.to_string(),
                Box::new(Py::decode(field(tag, body, "value")?)?),
            )
        }
        "$ndarray" => {
            let dtype = text_of(tag, field(tag, body, "dtype")?)?;
            let values = field(tag, body, "values")?
                .as_array()
                .ok_or_else(|| Error::new("$ndarray values are not a list"))?;
            Py::Ndarray(dtype.to_string(), decode_all(values)?)
        }
        "$dtype" => Py::Dtype(Dtype::parse(text_of(tag, body)?)?),
        "$fraction" => {
            let parts = body
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| Error::new("$fraction does not hold two integers"))?;
            let number = |v: &Value| match Py::decode(v)? {
                Py::Int(int) => int
                    .to_i128()
                    .ok_or_else(|| Error::new("a fraction too large")),
                _ => Err(Error::new("$fraction does not hold two integers")),
            };
            Py::Fraction(Fraction::new(number(&parts[0])?, number(&parts[1])?)?)
        }
        "$arrow" => {
            let data_type = ArrowType::parse(text_of(tag, field(tag, body, "type")?)?)?;
            let values = field(tag, body, "values")?
                .as_array()
                .ok_or_else(|| Error::new("$arrow values are not a list"))?;
            let chunks = match body.get("chunks") {
                None => None,
                Some(chunks) => Some(
                    chunks
                        .as_array()
                        .ok_or_else(|| Error::new("$arrow chunks are not a list"))?
                        .iter()
                        .map(|n| {
                            n.as_u64()
                                .map(|n| n as usize)
                                .ok_or_else(|| Error::new("a bad chunk length"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            };
            Py::Arrow(ArrowArray::new(data_type, decode_all(values)?, chunks)?)
        }
        "$arrow_type" => Py::ArrowType(ArrowType::parse(text_of(tag, body)?)?),
        "$series" => {
            let values = field(tag, body, "values")?
                .as_array()
                .ok_or_else(|| Error::new("$series values are not a list"))?;
            Py::Series(Series {
                dtype: text_of(tag, field(tag, body, "dtype")?)?.to_string(),
                values: decode_all(values)?,
                name: Box::new(Py::decode(field(tag, body, "name")?)?),
            })
        }
        "$record" => {
            let class = text_of(tag, field(tag, body, "class")?)?;
            let fields = field(tag, body, "fields")?
                .as_object()
                .ok_or_else(|| Error::new("$record fields are not an object"))?;
            Py::Record(class.to_string(), decode_map(fields)?)
        }
        "$quote" => {
            let text = text_of(tag, body)?;
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Py::Quote(c),
                _ => return Err(Error::new("$quote does not hold one character")),
            }
        }
        "$object" => Py::Dict(decode_map(
            body.as_object()
                .ok_or_else(|| Error::new("$object does not hold an object"))?,
        )?),
        _ => return Err(Error::new(format!("an unknown tag {tag}"))),
    })
}

/// `repr(b)` for Python `bytes`.
pub fn bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut out = format!("b{quote}");
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b if b == quote as u8 => {
                out.push('\\');
                out.push(quote);
            }
            0x20..=0x7e => out.push(b as char),
            b => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push(quote);
    out
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n =
            chunk.iter().fold(0u32, |acc, &b| (acc << 8) | u32::from(b)) << (8 * (3 - chunk.len()));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(BASE64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(text: &str) -> Result<Vec<u8>, Error> {
    let bad = || Error::new("bad base64");
    if !text.len().is_multiple_of(4) {
        return Err(bad());
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for chunk in text.as_bytes().chunks(4) {
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 2 {
            return Err(bad());
        }
        let mut n = 0u32;
        for &b in &chunk[..4 - pad] {
            let v = BASE64.iter().position(|&c| c == b).ok_or_else(bad)?;
            n = (n << 6) | v as u32;
        }
        n <<= 6 * pad;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let text = r#"[1, {"$int": "18446744073709551616"}, {"$float": "nan"}, 2.5,
            {"$tuple": ["a", null]}, {"$object": {"$x": 1}}, {"a": true},
            {"$timestamp": "2024-01-01T00:00:00.000000005+05:30"},
            {"$bytes": "YQBi"}, {"$np": {"type": "int64", "value": 3}},
            {"$arrow": {"type": "int64", "values": [1, null], "chunks": [1, 1]}},
            {"$fraction": [1, 3]}, {"$quote": "`"}, {"$dtype": "int"}]"#;
        let value: Value = serde_json::from_str(text).unwrap();
        let decoded = Py::decode(&value).unwrap();
        let again = decoded.encode().unwrap();
        assert_eq!(
            serde_json::to_string(&again).unwrap(),
            serde_json::to_string(&value).unwrap()
        );
    }

    #[test]
    fn bytes_as_python() {
        assert_eq!(
            bytes_repr(b"a\x00b\x7f\x80'\"\\"),
            r#"b'a\x00b\x7f\x80\'"\\'"#
        );
        assert_eq!(bytes_repr(b"it's"), "b\"it's\"");
        assert_eq!(base64_decode(&base64_encode(b"abcd")).unwrap(), b"abcd");
    }
}
