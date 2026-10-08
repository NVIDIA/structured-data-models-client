// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python values, as far as the schema core reads and writes them: the
//! values a YAML view holds, the cells a fixture names, and how Python
//! writes each with `str` and `repr`.

use serde_json::Value;

use crate::dt::civil::Wall;
use crate::py;

/// The [`PyVal::Opaque`] kind of a container met again inside itself: the
/// host writes it `{"$refers_back": TYPE}` where the value refers back to
/// one holding it.
pub const REFERS_BACK: &str = "refers_back";

/// A Python value.
#[derive(Clone, Debug, PartialEq)]
pub enum PyVal {
    Held {
        reference: String,
        owner: Option<String>,
        type_name: Option<String>,
        value: Box<PyVal>,
    },
    None,
    Bool(bool),
    /// An `int`; wider than 128 bits it is kept as its digits.
    Int(i128),
    BigInt(String),
    Float(f64),
    Str(String),
    /// `decimal.Decimal`, by its text.
    Decimal(String),
    /// `datetime.date`.
    Date(i64, u32, u32),
    /// `datetime.datetime`, with its UTC offset in seconds if aware.
    DateTime(Wall, Option<i64>),
    /// `datetime.time`, with its UTC offset in seconds if aware.
    Time {
        hour: u32,
        minute: u32,
        second: u32,
        micro: u32,
        offset: Option<i64>,
    },
    /// `pandas.Timestamp`, naive.
    Timestamp(Wall),
    /// `pandas.Timestamp` with a fixed UTC offset in seconds: its wall time
    /// at that offset.
    TimestampTz(Wall, i64),
    /// `pandas.Timedelta` (or `datetime.timedelta`), in nanoseconds.
    Timedelta(i64),
    /// A numpy array: its dtype name and its items as `tolist()` gives
    /// them.
    NdArray(String, Vec<PyVal>),
    Bytes(Vec<u8>),
    /// `uuid.UUID`, by its text.
    Uuid(String),
    /// `pandas.NA`.
    Na,
    /// `pandas.NaT`.
    NaT,
    List(Vec<PyVal>),
    Tuple(Vec<PyVal>),
    /// A `dict`, in insertion order.
    Dict(Vec<(PyVal, PyVal)>),
    /// A numpy scalar: its dtype name and value.
    Numpy(String, Box<PyVal>),
    /// Any other object, by what the host said of it: `kind` is
    /// `mapping`, `container` or `object`, or [`REFERS_BACK`] for a
    /// container met again inside itself, which plain data cannot hold.
    Opaque {
        kind: String,
        type_name: String,
        repr: String,
    },
}

fn date_text(y: i64, m: u32, d: u32) -> String {
    format!("{y:04}-{m:02}-{d:02}")
}

fn time_text(w: &Wall) -> String {
    let mut out = format!("{:02}:{:02}:{:02}", w.hour, w.minute, w.second);
    if w.nano / 1000 != 0 {
        out.push_str(&format!(".{:06}", w.nano / 1000));
    }
    out
}

fn offset_text(seconds: i64) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let total = seconds.unsigned_abs();
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    if s != 0 {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    } else {
        format!("{sign}{h:02}:{m:02}")
    }
}

impl PyVal {
    pub fn semantic(&self) -> &PyVal {
        match self {
            Self::Held { value, .. } => value.semantic(),
            other => other,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self.semantic() {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    /// Whether this is `None`, `pd.NA`, `pd.NaT` or a float NaN, which
    /// `pd.isna` calls missing.
    pub fn is_missing(&self) -> bool {
        match self.semantic() {
            PyVal::None | PyVal::Na | PyVal::NaT => true,
            PyVal::Float(x) => x.is_nan(),
            PyVal::Numpy(_, inner) => inner.is_missing(),
            _ => false,
        }
    }

    /// `str(value)`.
    pub fn str_(&self) -> String {
        match self.semantic() {
            PyVal::None => "None".to_string(),
            PyVal::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            PyVal::Int(n) => n.to_string(),
            PyVal::BigInt(digits) => digits.clone(),
            PyVal::Float(x) => py::float_repr(*x),
            PyVal::Str(s) => s.clone(),
            PyVal::Decimal(text) => text.clone(),
            PyVal::Date(y, m, d) => date_text(*y, *m, *d),
            PyVal::DateTime(w, tz) => {
                let mut out = format!("{} {}", date_text(w.year, w.month, w.day), time_text(w));
                if let Some(offset) = tz {
                    out.push_str(&offset_text(*offset));
                }
                out
            }
            PyVal::Timestamp(w) => {
                let mut out = format!("{} {}", date_text(w.year, w.month, w.day), time_text(w));
                if w.nano % 1000 != 0 {
                    out = format!(
                        "{} {:02}:{:02}:{:02}.{:09}",
                        date_text(w.year, w.month, w.day),
                        w.hour,
                        w.minute,
                        w.second,
                        w.nano
                    );
                }
                out
            }
            PyVal::Time {
                hour,
                minute,
                second,
                micro,
                offset,
            } => {
                let mut out = format!("{hour:02}:{minute:02}:{second:02}");
                if *micro != 0 {
                    out.push_str(&format!(".{micro:06}"));
                }
                if let Some(o) = offset {
                    out.push_str(&offset_text(*o));
                }
                out
            }
            PyVal::Uuid(text) => text.clone(),
            PyVal::Na => "<NA>".to_string(),
            PyVal::NaT => "NaT".to_string(),
            PyVal::TimestampTz(w, offset) => {
                let naive = PyVal::Timestamp(*w).str_();
                format!("{naive}{}", offset_text(*offset))
            }
            PyVal::Timedelta(nanos) => timedelta_str(*nanos),
            PyVal::NdArray(..) => self.repr(),
            PyVal::Numpy(_, inner) => match inner.as_ref() {
                PyVal::Bool(b) => if *b { "True" } else { "False" }.to_string(),
                other => other.str_(),
            },
            other => other.repr(),
        }
    }

    /// `repr(value)`.
    pub fn repr(&self) -> String {
        match self.semantic() {
            PyVal::Str(s) => py::str_repr(s),
            PyVal::Decimal(text) => format!("Decimal('{text}')"),
            PyVal::Date(y, m, d) => format!("datetime.date({y}, {m}, {d})"),
            PyVal::DateTime(w, tz) => {
                let mut parts = vec![
                    w.year.to_string(),
                    w.month.to_string(),
                    w.day.to_string(),
                    w.hour.to_string(),
                    w.minute.to_string(),
                ];
                let micro = w.nano / 1000;
                if w.second != 0 || micro != 0 {
                    parts.push(w.second.to_string());
                }
                if micro != 0 {
                    parts.push(micro.to_string());
                }
                let mut out = format!("datetime.datetime({}", parts.join(", "));
                if let Some(offset) = tz {
                    if *offset == 0 {
                        out.push_str(", tzinfo=datetime.timezone.utc");
                    } else {
                        out.push_str(&format!(
                            ", tzinfo=datetime.timezone({})",
                            timedelta_repr(*offset)
                        ));
                    }
                }
                out.push(')');
                out
            }
            PyVal::Time {
                hour,
                minute,
                second,
                micro,
                offset,
            } => {
                let mut parts = vec![hour.to_string(), minute.to_string()];
                if *second != 0 || *micro != 0 {
                    parts.push(second.to_string());
                }
                if *micro != 0 {
                    parts.push(micro.to_string());
                }
                let mut out = format!("datetime.time({}", parts.join(", "));
                match offset {
                    None => {}
                    Some(0) => out.push_str(", tzinfo=datetime.timezone.utc"),
                    Some(o) => out.push_str(&format!(
                        ", tzinfo=datetime.timezone({})",
                        timedelta_repr(*o)
                    )),
                }
                out.push(')');
                out
            }
            PyVal::Timestamp(_) => format!("Timestamp('{}')", self.str_()),
            PyVal::TimestampTz(w, offset) => {
                let naive = PyVal::Timestamp(*w).str_();
                let compact = offset_text(*offset).replace(':', "");
                let zone = if *offset == 0 {
                    "UTC".to_string()
                } else {
                    format!("UTC{}", offset_text(*offset))
                };
                format!("Timestamp('{naive}{compact}', tz='{zone}')")
            }
            PyVal::Timedelta(nanos) => timedelta_repr_micros(nanos.div_euclid(1000)),
            PyVal::NdArray(_, items) => {
                let parts: Vec<String> = items.iter().map(PyVal::repr).collect();
                format!("array([{}])", parts.join(", "))
            }
            PyVal::Bytes(bytes) => bytes_repr(bytes),
            PyVal::Uuid(text) => format!("UUID('{text}')"),
            PyVal::List(items) => {
                let parts: Vec<String> = items.iter().map(PyVal::repr).collect();
                format!("[{}]", parts.join(", "))
            }
            PyVal::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(PyVal::repr).collect();
                if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            PyVal::Dict(items) => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.repr(), v.repr()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            PyVal::Numpy(dtype, inner) => match inner.as_ref() {
                PyVal::Bool(b) => format!("np.{dtype}({})", if *b { "True" } else { "False" }),
                PyVal::Float(x) => format!("np.{dtype}({})", py::float_repr(*x)),
                other => format!("np.{dtype}({})", other.str_()),
            },
            PyVal::Na => "<NA>".to_string(),
            PyVal::NaT => "NaT".to_string(),
            PyVal::Opaque { repr, .. } => repr.clone(),
            other => other.str_(),
        }
    }

    /// A key that is equal for values Python compares equal, as `dict` and
    /// `set` do: `1`, `1.0` and `True` are one key.
    pub fn eq_key(&self) -> String {
        match self.semantic() {
            PyVal::Bool(b) => format!("n:{}", i32::from(*b)),
            PyVal::Int(n) => format!("n:{n}"),
            PyVal::BigInt(d) => format!("n:{d}"),
            PyVal::Float(x) => {
                if *x == 0.0 {
                    "n:0".to_string()
                } else if x.is_finite() && x.fract() == 0.0 {
                    format!("n:{x:.0}")
                } else if x.is_nan() {
                    // NaN is only equal to itself by identity.
                    format!("f:nan:{:p}", self)
                } else {
                    format!("x:{}", py::float_repr(*x))
                }
            }
            PyVal::Decimal(text) => decimal_key(text),
            PyVal::Str(s) => format!("s:{s}"),
            PyVal::None => "None".to_string(),
            PyVal::Numpy(_, inner) => inner.eq_key(),
            PyVal::Timestamp(w) | PyVal::DateTime(w, None) => format!("t:{}", w.epoch_nanos()),
            PyVal::Timedelta(nanos) => format!("td:{nanos}"),
            PyVal::Time {
                hour,
                minute,
                second,
                micro,
                offset,
            } => {
                let micros = ((i64::from(*hour) * 60 + i64::from(*minute)) * 60
                    + i64::from(*second))
                    * 1_000_000
                    + i64::from(*micro);
                match offset {
                    None => format!("time:{micros}"),
                    Some(o) => format!("timetz:{}", micros - o * 1_000_000),
                }
            }
            PyVal::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(PyVal::eq_key).collect();
                format!("t:({})", parts.join(","))
            }
            other => format!("o:{}", other.repr()),
        }
    }

    /// The fixture JSON a value is written as (`tests/parity/FORMAT.md`).
    pub fn decode(value: &Value) -> Result<PyVal, String> {
        Ok(match value {
            Value::Null => PyVal::None,
            Value::Bool(b) => PyVal::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    PyVal::Int(i128::from(i))
                } else if let Some(u) = n.as_u64() {
                    PyVal::Int(i128::from(u))
                } else {
                    PyVal::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::String(s) => PyVal::Str(s.clone()),
            Value::Array(items) => {
                PyVal::List(items.iter().map(PyVal::decode).collect::<Result<_, _>>()?)
            }
            Value::Object(map) => {
                if map.len() == 1 {
                    let (tag, body) = map.iter().next().expect("one entry");
                    if let Some(decoded) = decode_tag(tag, body)? {
                        return Ok(decoded);
                    }
                }
                let mut items = Vec::new();
                for (k, v) in map {
                    items.push((PyVal::Str(k.clone()), PyVal::decode(v)?));
                }
                PyVal::Dict(items)
            }
        })
    }
}

/// The key a `Decimal` compares by: a whole value as the integer it is.
fn decimal_key(text: &str) -> String {
    let t = text.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.trim_start_matches('+')),
    };
    let lower = body.to_ascii_lowercase();
    if lower.contains("inf") || lower.contains("nan") {
        return format!("d:{t}");
    }
    let (mantissa, exp) = match lower.split_once('e') {
        Some((m, e)) => (m.to_string(), e.parse::<i64>().unwrap_or(0)),
        None => (lower.clone(), 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa.as_str(), ""));
    let mut digits = format!("{int}{frac}");
    let mut point = int.len() as i64 + exp;
    let lead = digits.len() - digits.trim_start_matches('0').len();
    digits = digits.trim_start_matches('0').to_string();
    point -= lead as i64;
    let digits = digits.trim_end_matches('0').to_string();
    if digits.is_empty() {
        return "n:0".to_string();
    }
    let sign = if neg { "-" } else { "" };
    if point >= digits.len() as i64 {
        return format!(
            "n:{sign}{digits}{}",
            "0".repeat((point - digits.len() as i64) as usize)
        );
    }
    format!("x:{sign}{digits}e{point}")
}

/// `str(datetime.timedelta)` of `nanos`, to the microsecond.
fn timedelta_str(nanos: i64) -> String {
    let micros = nanos.div_euclid(1000);
    let days = micros.div_euclid(86_400_000_000);
    let rest = micros.rem_euclid(86_400_000_000);
    let (seconds, micro) = (rest / 1_000_000, rest % 1_000_000);
    let clock = format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        seconds % 3600 / 60,
        seconds % 60
    );
    let clock = if micro != 0 {
        format!("{clock}.{micro:06}")
    } else {
        clock
    };
    match days {
        0 => clock,
        1 | -1 => format!("{days} day, {clock}"),
        d => format!("{d} days, {clock}"),
    }
}

/// `repr(datetime.timedelta(microseconds=micros))`.
fn timedelta_repr_micros(micros: i64) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let rest = micros.rem_euclid(86_400_000_000);
    let (seconds, micro) = (rest / 1_000_000, rest % 1_000_000);
    let mut parts = Vec::new();
    if days != 0 {
        parts.push(format!("days={days}"));
    }
    if seconds != 0 {
        parts.push(format!("seconds={seconds}"));
    }
    if micro != 0 {
        parts.push(format!("microseconds={micro}"));
    }
    if parts.is_empty() {
        return "datetime.timedelta(0)".to_string();
    }
    format!("datetime.timedelta({})", parts.join(", "))
}

fn timedelta_repr(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    match (days, rest) {
        (0, r) => format!("datetime.timedelta(seconds={r})"),
        (d, 0) => format!("datetime.timedelta(days={d})"),
        (d, r) => format!("datetime.timedelta(days={d}, seconds={r})"),
    }
}

/// `repr(bytes)`.
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
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push(quote);
    out
}

fn parse_wall(text: &str) -> Result<(Wall, Option<i64>), String> {
    let bad = || format!("not an ISO date and time: {text:?}");
    let (date, time) = match text.find(['T', ' ']) {
        Some(i) => (&text[..i], &text[i + 1..]),
        None => (text, ""),
    };
    let mut d = date.split('-');
    let year: i64 = d.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
    let month: u32 = d.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
    let day: u32 = d.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
    let mut wall = Wall::date(year, month, day);
    let mut offset = None;
    if !time.is_empty() {
        let (clock, zone) = match time.find(['+', '-', 'Z']) {
            Some(i) => (&time[..i], Some(&time[i..])),
            None => (time, None),
        };
        let (hms, frac) = match clock.split_once('.') {
            Some((a, b)) => (a, b),
            None => (clock, ""),
        };
        let mut parts = hms.split(':');
        wall.hour = parts.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
        wall.minute = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        wall.second = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if !frac.is_empty() {
            let digits: String = format!("{frac:0<9}").chars().take(9).collect();
            wall.nano = digits.parse().map_err(|_| bad())?;
        }
        if let Some(zone) = zone {
            offset = Some(if zone == "Z" {
                0
            } else {
                let sign = if zone.starts_with('-') { -1 } else { 1 };
                let body = &zone[1..];
                let mut p = body.split(':');
                let h: i64 = p.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
                let m: i64 = p.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                sign * (h * 3600 + m * 60)
            });
        }
    }
    Ok((wall, offset))
}

fn decode_tag(tag: &str, body: &Value) -> Result<Option<PyVal>, String> {
    let text = || {
        body.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{tag} takes text"))
    };
    Ok(Some(match tag {
        "$held" => PyVal::Held {
            reference: body["reference"]["$host_ref"]
                .as_str()
                .ok_or("$held takes a local reference")?
                .to_string(),
            owner: body
                .get("owner")
                .and_then(Value::as_str)
                .map(str::to_string),
            value: Box::new(PyVal::decode(&body["value"])?),
            type_name: body.get("type").and_then(Value::as_str).map(str::to_string),
        },
        "$host_ref" => PyVal::Opaque {
            kind: "host_ref".to_string(),
            type_name: text()?,
            repr: String::new(),
        },
        "$int" => {
            let digits = text()?;
            match digits.parse::<i128>() {
                Ok(n) => PyVal::Int(n),
                Err(_) => PyVal::BigInt(digits),
            }
        }
        "$float" => {
            let t = text()?;
            PyVal::Float(match t.as_str() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => other.parse().map_err(|_| format!("not a float: {other}"))?,
            })
        }
        "$decimal" => PyVal::Decimal(text()?),
        "$datetime" => {
            let (wall, offset) = parse_wall(&text()?)?;
            PyVal::DateTime(wall, offset)
        }
        "$date" => {
            let (wall, _) = parse_wall(&text()?)?;
            PyVal::Date(wall.year, wall.month, wall.day)
        }
        "$timestamp" => {
            let (wall, offset) = parse_wall(&text()?)?;
            match offset {
                None => PyVal::Timestamp(wall),
                Some(offset) => PyVal::TimestampTz(wall, offset),
            }
        }
        "$timedelta_us" => {
            let micros = body.as_i64().ok_or("$timedelta_us takes an integer")?;
            PyVal::Timedelta(micros.saturating_mul(1000))
        }
        "$ndarray" => {
            let dtype = body
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or("$ndarray takes a dtype")?;
            let items = body
                .get("values")
                .and_then(Value::as_array)
                .ok_or("$ndarray takes values")?;
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                let value = PyVal::decode(item)?;
                values.push(match (dtype, value) {
                    (t, PyVal::Int(n)) if t.starts_with("float") => PyVal::Float(n as f64),
                    (_, v) => v,
                });
            }
            PyVal::NdArray(dtype.to_string(), values)
        }
        "$bytes" => PyVal::Bytes(base64_decode(&text()?)?),
        "$uuid" => PyVal::Uuid(text()?.to_lowercase()),
        "$mapping" => PyVal::Opaque {
            kind: "mapping".to_string(),
            type_name: text()?,
            repr: String::new(),
        },
        "$container" => PyVal::Opaque {
            kind: "container".to_string(),
            type_name: text()?,
            repr: String::new(),
        },
        "$object" => PyVal::Opaque {
            kind: "object".to_string(),
            type_name: body
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("object")
                .to_string(),
            repr: body
                .get("repr")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        },
        "$refers_back" => PyVal::Opaque {
            kind: REFERS_BACK.to_string(),
            type_name: body
                .as_str()
                .or_else(|| body.get("type").and_then(Value::as_str))
                .ok_or("$refers_back takes a container type")?
                .to_string(),
            repr: body
                .get("reference")
                .and_then(Value::as_u64)
                .map(|reference| reference.to_string())
                .unwrap_or_default(),
        },
        "$na" => PyVal::Na,
        "$nat" => PyVal::NaT,
        "$map" => {
            let pairs = body.as_array().ok_or("$map takes a list of pairs")?;
            let mut items = Vec::with_capacity(pairs.len());
            for pair in pairs {
                let pair = pair
                    .as_array()
                    .filter(|p| p.len() == 2)
                    .ok_or("$map takes pairs")?;
                items.push((PyVal::decode(&pair[0])?, PyVal::decode(&pair[1])?));
            }
            PyVal::Dict(items)
        }
        "$dict" => {
            let map = body.as_object().ok_or("$dict takes an object")?;
            let mut items = Vec::with_capacity(map.len());
            for (k, v) in map {
                items.push((PyVal::Str(k.clone()), PyVal::decode(v)?));
            }
            PyVal::Dict(items)
        }
        "$tuple" => {
            let items = body.as_array().ok_or("$tuple takes a list")?;
            PyVal::Tuple(items.iter().map(PyVal::decode).collect::<Result<_, _>>()?)
        }
        "$np" => {
            let dtype = body
                .get("type")
                .and_then(Value::as_str)
                .ok_or("$np takes a type")?;
            let inner = PyVal::decode(body.get("value").ok_or("$np takes a value")?)?;
            let inner = match (dtype, inner) {
                (t, PyVal::Int(n)) if t.starts_with("float") => PyVal::Float(n as f64),
                ("bool", PyVal::Int(n)) => PyVal::Bool(n != 0),
                (_, v) => v,
            };
            PyVal::Numpy(dtype.to_string(), Box::new(inner))
        }
        _ => return Ok(None),
    }))
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut acc: u32 = 0;
    let mut bits = 0;
    for c in text.bytes() {
        if c == b'=' {
            break;
        }
        let v = ALPHABET.iter().position(|&a| a == c).ok_or("not base64")? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod duration_identity_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn duration_keys_preserve_nanosecond_class_identity() {
        let values = [
            i64::MIN + 1,
            -1_001,
            -1_000,
            -999,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            999,
            1_000,
            1_001,
            i64::MAX,
        ];
        let keys: HashSet<_> = values
            .iter()
            .map(|value| PyVal::Timedelta(*value).eq_key())
            .collect();
        assert_eq!(keys.len(), values.len());
        for value in values {
            let duration = PyVal::Timedelta(value);
            assert_eq!(duration.eq_key(), duration.clone().eq_key());
            assert_eq!(
                duration.eq_key(),
                PyVal::Numpy("timedelta64[ns]".into(), Box::new(duration.clone())).eq_key()
            );
            assert_ne!(duration.eq_key(), PyVal::Int(i128::from(value)).eq_key());
            assert_ne!(duration.eq_key(), PyVal::Str(duration.repr()).eq_key());
        }
        assert_eq!(PyVal::Timedelta(1).repr(), PyVal::Timedelta(2).repr());
        assert_ne!(PyVal::Timedelta(1).eq_key(), PyVal::Timedelta(2).eq_key());
    }
}
