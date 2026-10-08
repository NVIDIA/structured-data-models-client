// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A key spread across several columns, folded into one identity:
//! `rfm/base/composite_key.py`.

use crate::py;
use crate::pyval::PyVal;

pub const DERIVED_PREFIX: &str = "__kumo_key_";
pub const SEPARATOR: char = '\u{1f}';
pub const ESCAPE: char = '\\';
pub const MAX_DERIVED_NAME: usize = 200;

/// `escape_part`.
pub fn escape_part(text: &str) -> String {
    text.replace(ESCAPE, "\\\\").replace(SEPARATOR, "\\u")
}

/// `_render`: one key part's text; `Err` is its `ValueError`.
pub fn render(value: &PyVal) -> Result<String, String> {
    let null = || "a key column cannot hold a null value".to_string();
    match value {
        PyVal::None | PyVal::Na | PyVal::NaT => Err(null()),
        PyVal::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
        PyVal::Float(x) if x.is_nan() => Err(null()),
        PyVal::Numpy(_, inner) => match inner.as_ref() {
            PyVal::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            PyVal::Float(x) if x.is_nan() => Err(null()),
            other => Ok(other.str_()),
        },
        other => Ok(other.str_()),
    }
}

/// `sql_expression(columns, quote=, text_type=, chr_function=)`: the fold
/// as SQL, matching [`encode_frame`] value for value; the separator and the
/// escape written as character codes, which every dialect reads alike.
pub fn sql_expression(
    columns: &[String],
    quote: impl Fn(&str) -> String,
    text_type: &str,
    chr_function: &str,
) -> String {
    let backslash = format!("{chr_function}(92)");
    let separator = format!("{chr_function}(31)");
    columns
        .iter()
        .map(|name| {
            let rendered = format!("CAST({} AS {text_type})", quote(name));
            format!(
                "REPLACE(REPLACE({rendered}, {backslash}, {backslash} || {backslash}), \
                 {separator}, {backslash} || 'u')"
            )
        })
        .collect::<Vec<_>>()
        .join(&format!(" || {separator} || "))
}

/// `encode_values`.
pub fn encode_values(values: &[PyVal]) -> Result<String, String> {
    let parts: Result<Vec<String>, String> = values
        .iter()
        .map(|v| render(v).map(|t| escape_part(&t)))
        .collect();
    Ok(parts?.join(&SEPARATOR.to_string()))
}

/// `decode_value`.
pub fn decode_value(encoded: &str, arity: usize) -> Result<Vec<String>, String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for c in encoded.chars() {
        if escaped {
            current.push(if c == 'u' { SEPARATOR } else { c });
            escaped = false;
        } else if c == ESCAPE {
            escaped = true;
        } else if c == SEPARATOR {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    parts.push(current);
    if parts.len() != arity {
        return Err(format!(
            "expected a key of {arity} parts, got {}: {}",
            parts.len(),
            py::str_repr(encoded)
        ));
    }
    Ok(parts)
}

/// `encode_identity`.
pub fn encode_identity(columns: &[String]) -> Result<String, String> {
    let encoded: Vec<String> = columns
        .iter()
        .map(|name| name.bytes().map(|b| format!("{b:02x}")).collect())
        .collect();
    let encoded = encoded.join("_");
    if DERIVED_PREFIX.len() + encoded.len() > MAX_DERIVED_NAME {
        return Err(format!(
            "An identity over {} needs a column name longer than {MAX_DERIVED_NAME} characters, which a warehouse will not accept. Add a column holding that identity and declare it instead.",
            py::str_list_repr(columns)
        ));
    }
    Ok(encoded)
}

/// `bytes.fromhex(text)`.
fn from_hex(text: &str) -> Option<Vec<u8>> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let space = |c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c');
    while i < chars.len() {
        if space(chars[i]) {
            i += 1;
            continue;
        }
        let hi = chars[i].to_digit(16)?;
        let lo = chars.get(i + 1)?.to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

/// `decode_identity`.
pub fn decode_identity(derived: &str) -> Option<Vec<String>> {
    let rest = derived.strip_prefix(DERIVED_PREFIX)?;
    if rest.is_empty() {
        return None;
    }
    let mut columns = Vec::new();
    for part in rest.split('_') {
        let bytes = from_hex(part)?;
        columns.push(String::from_utf8(bytes).ok()?);
    }
    Some(columns)
}

/// `encode_frame` over named columns of values, in order.
pub fn encode_frame(
    frame: &[(String, Vec<PyVal>)],
    columns: &[String],
) -> Result<Vec<String>, String> {
    let missing: Vec<String> = columns
        .iter()
        .filter(|c| !frame.iter().any(|(name, _)| name == *c))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "composite key columns {} are not present; a table referencing this key must carry every one of {}",
            py::str_list_repr(&missing),
            py::str_list_repr(columns)
        ));
    }
    let rows = frame.first().map_or(0, |(_, v)| v.len());
    let mut joined: Vec<String> = vec![String::new(); rows];
    for (k, name) in columns.iter().enumerate() {
        let values = &frame
            .iter()
            .find(|(n, _)| n == name)
            .expect("checked above")
            .1;
        if values.iter().any(PyVal::is_missing) {
            return Err(format!(
                "composite key column '{name}' holds a null value; every part of a row's identity has to be present"
            ));
        }
        for (row, value) in values.iter().enumerate() {
            if k > 0 {
                joined[row].push(SEPARATOR);
            }
            joined[row].push_str(&escape_part(&render(value)?));
        }
    }
    Ok(joined)
}
