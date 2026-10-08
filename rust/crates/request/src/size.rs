// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `payload_size_bytes` (`rfm/inference/payload.py`): how many bytes a
//! payload is, as Python writes it with `json.dumps(payload,
//! allow_nan=False, separators=(',', ':'))`. The 30 MiB limit is counted in
//! those bytes, so the count follows Python's text exactly: `ensure_ascii`
//! escapes, a character outside the BMP as a surrogate pair, and every float
//! as its `repr`.

use serde_json::{Number, Value};

/// Why a text is not a payload whose size can be counted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotAPayload(pub String);

impl std::fmt::Display for NotAPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotAPayload {}

/// The bytes `json.dumps` writes for `text` as a JSON string.
fn string_bytes(text: &str) -> usize {
    text.chars()
        .map(|c| match c {
            '"' | '\\' | '\n' | '\r' | '\t' | '\u{8}' | '\u{c}' => 2,
            ' '..='~' => 1,
            c if (c as u32) > 0xffff => 12,
            _ => 6,
        })
        .sum::<usize>()
        + 2
}

fn number_bytes(number: &Number) -> Result<usize, NotAPayload> {
    if let Some(i) = number.as_i64() {
        return Ok(i.to_string().len());
    }
    if let Some(u) = number.as_u64() {
        return Ok(u.to_string().len());
    }
    let x = number
        .as_f64()
        .ok_or_else(|| NotAPayload("a number that is not one".to_string()))?;
    if !x.is_finite() {
        return Err(NotAPayload(
            "Out of range float values are not JSON compliant".to_string(),
        ));
    }
    Ok(sdm_pql::pyrepr::float_repr(x).len())
}

/// The bytes of `value` as Python's compact `json.dumps` writes it.
pub fn value_bytes(value: &Value) -> Result<usize, NotAPayload> {
    Ok(match value {
        Value::Null => 4,
        Value::Bool(true) => 4,
        Value::Bool(false) => 5,
        Value::Number(n) => number_bytes(n)?,
        Value::String(s) => string_bytes(s),
        Value::Array(items) => {
            let mut total = 2 + items.len().saturating_sub(1);
            for item in items {
                total += value_bytes(item)?;
            }
            total
        }
        Value::Object(map) => {
            let mut total = 2 + map.len().saturating_sub(1);
            for (key, item) in map {
                total += string_bytes(key) + 1 + value_bytes(item)?;
            }
            total
        }
    })
}

/// `payload_size_bytes(json.loads(text))`: a JSON object's size as Python
/// writes it, however the text spelled it.
pub fn payload_size_bytes(text: &str) -> Result<usize, NotAPayload> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| NotAPayload(format!("not JSON: {e}")))?;
    if !value.is_object() {
        return Err(NotAPayload("a payload is a JSON object".to_string()));
    }
    value_bytes(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_as_python_writes() {
        // Python writes this {"a":"<e-acute escaped><astral pair escaped>\n","b":[1e+16,-0.0,0.1]},
        // 49 bytes: the e-acute takes 6, the astral character 12.
        let text = "{\"a\": \"\u{e9}\u{1f600}\\n\", \"b\": [1e16, -0.0, 0.1]}";
        let python = 49;
        assert_eq!(payload_size_bytes(text), Ok(python));
    }

    #[test]
    fn refuses_what_is_not_a_payload() {
        for text in ["[1]", "3", "{\"a\": NaN}", "{\"a\": 1e400}", "", "{} x"] {
            assert!(payload_size_bytes(text).is_err(), "{text}");
        }
    }
}
