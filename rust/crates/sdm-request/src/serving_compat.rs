// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded Python compatibility for Databricks Model Serving (T3.13).
//!
//! Only JSON scalar formatting, text escaping, type names, presence and
//! the text comparisons used by the existing serving contract live here.
//! The parent module owns envelope shape, limits, endpoint response
//! validation and error-code policy. This module performs no I/O and
//! receives the existing float formatter from the core's parser adapter.
//!
//! The one-module exception to the parser-only emulation boundary needs
//! the owner's explicit approval; see T313-OWNER-2026-10-24-SERVING-EMULATION.

use sdm_wire::json::{Value, is_integer_token};

/// `text` case-folded as Python's `str.casefold` folds it, as far as the
/// fixed caller-fault wording in `serving` can tell: lower case, with the long
/// s and the Latin ligatures folded to their letters.
pub(super) fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ſ' => out.push('s'),
            'ß' | 'ẞ' => out.push_str("ss"),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' | 'ﬆ' => out.push_str("st"),
            _ => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// Whether Python's `str.isspace` (the `\s` of a text pattern) takes `c`:
/// Unicode's white space, and the information separators.
pub(super) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Whether Python's `bool` takes `value` as true.
pub(super) fn is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(token) => token.parse::<f64>().ok().is_none_or(|n| n != 0.0),
        Value::Text(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(members) => !members.is_empty(),
    }
}

/// The name of the Python type `json.loads` reads `value` as.
pub(super) fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(token) if is_integer_token(token) => "int",
        Value::Number(_) => "float",
        Value::Text(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

// --- Writing as Python's `json.dumps` writes ----------------------------------

/// `value` as `json.dumps(value, allow_nan=False, separators=(',', ':'))`
/// writes what `json.loads` read of it: members in order, an integer as
/// its digits (`-0` as `0`), a float as its `repr`, every text escaped to
/// ASCII; an error naming a float past a double's range, which Python
/// refuses to write.
pub(super) fn write_json(
    value: &Value,
    float_repr: fn(f64) -> String,
    out: &mut String,
) -> Result<(), &'static str> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(token) if is_integer_token(token) => {
            if token.trim_start_matches('-').bytes().all(|b| b == b'0') {
                out.push('0');
            } else {
                out.push_str(token);
            }
        }
        Value::Number(token) => {
            let number: f64 = token.parse().map_err(|_| "not_a_number")?;
            if !number.is_finite() {
                return Err("out_of_range_float");
            }
            out.push_str(&float_repr(number));
        }
        Value::Text(text) => write_text(text, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, float_repr, out)?;
            }
            out.push(']');
        }
        Value::Object(members) => {
            out.push('{');
            for (i, (name, member)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_text(name, out);
                out.push(':');
                write_json(member, float_repr, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// `text` as a JSON string Python writes with `ensure_ascii`: a quote and
/// a backslash escaped, the five control characters JSON names by letter
/// by it, any other character outside space to tilde as `\u` and four
/// lower-case hex digits, one past the BMP as its surrogate pair.
pub(super) fn write_text(text: &str, out: &mut String) {
    use std::fmt::Write;
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
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}
