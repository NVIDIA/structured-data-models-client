// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

/// `encode(value)` of `_fixture_ops.py`: a Python value as fixture JSON.
pub fn encode(value: &PyVal) -> Value {
    match value {
        PyVal::None => Value::Null,
        PyVal::Bool(b) => Value::Bool(*b),
        PyVal::Na => json!({"$na": null}),
        PyVal::NaT => json!({"$nat": null}),
        PyVal::Int(n) => match i64::try_from(*n) {
            Ok(i) => Value::from(i),
            Err(_) => json!({"$int": n.to_string()}),
        },
        PyVal::BigInt(digits) => json!({"$int": digits}),
        PyVal::Float(x) => {
            if x.is_nan() {
                json!({"$float": "nan"})
            } else if x.is_infinite() {
                json!({"$float": if *x > 0.0 { "inf" } else { "-inf" }})
            } else {
                crate::wire::float_json(*x)
            }
        }
        PyVal::Str(s) => Value::String(s.clone()),
        PyVal::Decimal(_) => json!({"$decimal": crate::pyops::py_str(value)}),
        PyVal::Numpy(dtype, inner) => {
            if dtype.starts_with("datetime64") || dtype.starts_with("timedelta64") {
                return encode(inner);
            }
            json!({"$np": {"type": dtype, "value": encode(inner)}})
        }
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) => {
            json!({"$timestamp": timestamp_isoformat(value)})
        }
        PyVal::DateTime(..) => json!({"$datetime": datetime_isoformat(value)}),
        PyVal::Date(y, m, d) => json!({"$date": format!("{y:04}-{m:02}-{d:02}")}),
        PyVal::Timedelta(nanos) => json!({"$timedelta_us": nanos.div_euclid(1000)}),
        PyVal::Bytes(bytes) => json!({"$bytes": base64(bytes)}),
        PyVal::Uuid(text) => json!({"$uuid": text}),
        PyVal::Tuple(items) => json!({"$tuple": items.iter().map(encode).collect::<Vec<_>>()}),
        PyVal::NdArray(dtype, items) => json!({"$ndarray": {
            "dtype": if dtype.starts_with("<U") { "str" } else { dtype.as_str() },
            "values": items.iter().map(encode).collect::<Vec<_>>(),
        }}),
        PyVal::List(items) => Value::Array(items.iter().map(encode).collect()),
        PyVal::Dict(items) => Value::Object(
            items
                .iter()
                .map(|(k, v)| (crate::pyops::py_str(k), encode(v)))
                .collect::<Map<_, _>>(),
        ),
        other => Value::String(crate::pyops::py_repr(other)),
    }
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// `Timestamp.isoformat()`.
pub fn timestamp_isoformat(value: &PyVal) -> String {
    let (wall, offset) = match value {
        PyVal::Timestamp(w) => (*w, None),
        PyVal::TimestampTz(w, o) => (*w, Some(*o)),
        _ => return crate::pyops::py_str(value),
    };
    let mut out = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        wall.year, wall.month, wall.day, wall.hour, wall.minute, wall.second
    );
    if wall.nano % 1000 != 0 {
        out.push_str(&format!(".{:09}", wall.nano));
    } else if wall.nano != 0 {
        out.push_str(&format!(".{:06}", wall.nano / 1000));
    }
    if let Some(offset) = offset {
        out.push_str(&offset_text(offset));
    }
    out
}

fn datetime_isoformat(value: &PyVal) -> String {
    let PyVal::DateTime(wall, offset) = value else {
        return crate::pyops::py_str(value);
    };
    let mut out = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        wall.year, wall.month, wall.day, wall.hour, wall.minute, wall.second
    );
    if wall.nano / 1000 != 0 {
        out.push_str(&format!(".{:06}", wall.nano / 1000));
    }
    if let Some(offset) = offset {
        out.push_str(&offset_text(*offset));
    }
    out
}

fn offset_text(seconds: i64) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let total = seconds.unsigned_abs();
    format!("{sign}{:02}:{:02}", total / 3600, total % 3600 / 60)
}
