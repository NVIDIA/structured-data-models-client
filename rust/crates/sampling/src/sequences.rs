// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Table._sanitize`'s `_to_list` (`rfm/base/table.py`): a column of
//! sequences read back as text, as pandas, numpy and `json` turn it into
//! lists.
//!
//! A list dtype, or a `sequence` column, whose values are text is read row by
//! row with `np.fromstring(row.strip('[]'), sep=',', dtype=...)`: `int` for
//! `intlist`, `np.float32` otherwise. If any row fails, every row is read
//! with `json.loads` instead, and if any of those fails, the text is kept.
//!
//! Which rows numpy reads is ported exactly, from numpy's `array_from_text`
//! with CPython's `PyOS_string_to_double` and `PyOS_strtol` underneath (on
//! the 64-bit Linux and macOS builds, where numpy's `int` is a C `long`).
//! Which rows `json` accepts is ported too, as Python's `json.loads` reads
//! text. When every row is JSON but one is not a list numpy reads, the
//! column becomes the values Python's standard JSON decoder reads.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::builder::{Float32Builder, Int64Builder, ListBuilder};

use crate::driver::Raised;
use crate::driver::output::Warning;
use crate::frame::{Dtype, Series, texts};

mod json;

/// What a list column's rows are read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListKind {
    /// `intlist`: numpy's `int`, a C `long`.
    Int,
    /// `floatlist`, and a `sequence` column: `np.float32`.
    Float,
    /// `stringlist`: numpy is not asked, only `json`.
    Text,
}

/// The rows of a column numpy read.
enum Lists {
    Int(Vec<Vec<i64>>),
    Float(Vec<Vec<f32>>),
}

/// `_to_list(ser, dtype)`: the series as lists where today's Python makes
/// it lists, and unchanged where it keeps the text.
pub fn to_list(series: &Series, kind: ListKind) -> Result<Series, Raised> {
    to_list_with_policy(series, kind, false, &mut Vec::new())
}

pub fn to_list_with_policy(
    series: &Series,
    kind: ListKind,
    prefix: bool,
    warnings: &mut Vec<Warning>,
) -> Result<Series, Raised> {
    // `pd.api.types.is_string_dtype(ser)`: text of any dtype pandas holds
    // it in. Anything else, such as lists DuckDB returns as lists, is kept.
    let Some(rows) = texts(series.array.as_ref()) else {
        return Ok(series.clone());
    };
    // An empty column maps to itself. A missing row is `NA` or `NaN` in an
    // Arrow or `str` column, whose `.strip` and `json.loads` both raise, and
    // `None` in an `object` column, which `is_string_dtype` then refuses:
    // either way the text is kept.
    if rows.iter().any(Option::is_none) {
        if prefix && series.dtype != Dtype::Object && kind != ListKind::Text {
            for row in rows.iter().take_while(|row| row.is_some()).flatten() {
                let warned = match kind {
                    ListKind::Int => from_string_ints_with_policy(strip_brackets(row), true).1,
                    ListKind::Float => from_string_floats_with_policy(strip_brackets(row), true).1,
                    ListKind::Text => false,
                };
                if warned {
                    warnings.push(prefix_warning());
                }
            }
        }
        return Ok(series.clone());
    }
    let rows: Vec<String> = rows.into_iter().flatten().collect();
    if rows.is_empty() {
        return Ok(series.clone());
    }
    let read = match kind {
        ListKind::Int => rows
            .iter()
            .map(|row| {
                let (values, warned) = from_string_ints_with_policy(strip_brackets(row), prefix);
                if warned {
                    warnings.push(prefix_warning());
                }
                values
            })
            .collect::<Option<Vec<_>>>()
            .map(Lists::Int),
        ListKind::Float => rows
            .iter()
            .map(|row| {
                let (values, warned) = from_string_floats_with_policy(strip_brackets(row), prefix);
                if warned {
                    warnings.push(prefix_warning());
                }
                values
            })
            .collect::<Option<Vec<_>>>()
            .map(Lists::Float),
        ListKind::Text => None,
    };
    if let Some(lists) = read {
        return Ok(Series::new(Dtype::Object, list_array(lists)));
    }
    if let Some(values) = rows
        .iter()
        .map(|row| json::parse(row))
        .collect::<Option<Vec<_>>>()
    {
        return Ok(json::series(values));
    }
    if rows.iter().all(|row| json_accepts(row)) {
        return Err(Raised::new(
            "TypeError",
            "a sequence column whose text is JSON but not a list numpy reads is made of whatever \
             values json.loads or orjson.loads give, which the core does not reproduce",
        ));
    }
    Ok(series.clone())
}

fn list_array(lists: Lists) -> ArrayRef {
    match lists {
        Lists::Int(rows) => {
            let mut builder = ListBuilder::new(Int64Builder::new());
            for row in rows {
                builder.values().append_slice(&row);
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
        Lists::Float(rows) => {
            let mut builder = ListBuilder::new(Float32Builder::new());
            for row in rows {
                builder.values().append_slice(&row);
                builder.append(true);
            }
            Arc::new(builder.finish())
        }
    }
}

/// `row.strip('[]')`.
fn strip_brackets(row: &str) -> &str {
    row.trim_matches(|c| c == '[' || c == ']')
}

/// C's `isspace` in the C locale, CPython's `Py_ISSPACE` and numpy's
/// `NumPyOS_ascii_isspace`: the same six bytes.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// `np.fromstring(text, sep=',', dtype=np.float32)`, `None` where numpy
/// raises `ValueError`.
pub fn from_string_floats(text: &str) -> Option<Vec<f32>> {
    from_string_floats_with_policy(text, false).0
}

pub fn from_string_floats_with_policy(text: &str, prefix: bool) -> (Option<Vec<f32>>, bool) {
    // C's double-to-float conversion, which rounds to nearest.
    from_text(text.as_bytes(), prefix, |bytes, at| {
        let (value, end) = ascii_strtod(bytes, at);
        (value as f32, end)
    })
}

/// `np.fromstring(text, sep=',', dtype=int)`, `None` where numpy raises
/// `ValueError`.
pub fn from_string_ints(text: &str) -> Option<Vec<i64>> {
    from_string_ints_with_policy(text, false).0
}

pub fn from_string_ints_with_policy(text: &str, prefix: bool) -> (Option<Vec<i64>>, bool) {
    from_text(text.as_bytes(), prefix, strtol)
}

fn prefix_warning() -> Warning {
    Warning { category: "builtins.DeprecationWarning".to_owned(), message: "string or file could not be read to its end due to unmatched data; this will raise a ValueError in the future.".to_owned(), masked_float32_cast: false }
}

/// numpy's `array_from_text` with `fromstr_next_element` and
/// `fromstr_skip_separator`, over the whole text, its separator `","`
/// cleaned to `" ,"`: any run of whitespace, then a comma.
fn from_text<T>(
    bytes: &[u8],
    prefix: bool,
    read: impl Fn(&[u8], usize) -> (T, usize),
) -> (Option<Vec<T>>, bool) {
    let end = bytes.len();
    let mut at = 0;
    let mut out = Vec::new();
    loop {
        let (value, next) = read(bytes, at);
        if next == at {
            // Nothing read: the end of the text, or what numpy cannot read.
            return (((at >= end) || prefix).then_some(out), prefix && at < end);
        }
        out.push(value);
        at = next;
        // The separator: whitespace, then a comma.
        let mut comma = false;
        loop {
            if at >= end {
                return (Some(out), false);
            }
            if comma {
                break;
            }
            let c = bytes[at];
            if c == b',' {
                comma = true;
            } else if !is_space(c) {
                return (prefix.then_some(out), prefix);
            }
            at += 1;
        }
    }
}

/// numpy's `NumPyOS_ascii_strtod` at `at`: the value and where reading
/// stopped. `nan`, `nan(...)`, `inf` and `infinity` in any case and with a
/// sign are numpy's own; the rest is CPython's `PyOS_string_to_double`,
/// which on text it cannot read gives `-1.0` and stops where it started,
/// after the whitespace numpy skipped.
fn ascii_strtod(bytes: &[u8], at: usize) -> (f64, usize) {
    let mut s = at;
    while s < bytes.len() && is_space(bytes[s]) {
        s += 1;
    }
    let mut p = s;
    let mut sign = 1.0;
    match bytes.get(p) {
        Some(b'-') => {
            sign = -1.0;
            p += 1;
        }
        Some(b'+') => p += 1,
        _ => {}
    }
    let starts = |at: usize, word: &[u8]| {
        bytes.len() >= at + word.len() && bytes[at..at + word.len()].eq_ignore_ascii_case(word)
    };
    if starts(p, b"nan") {
        p += 3;
        if bytes.get(p) == Some(&b'(') {
            p += 1;
            while p < bytes.len() && (bytes[p].is_ascii_alphanumeric() || bytes[p] == b'_') {
                p += 1;
            }
            if bytes.get(p) == Some(&b')') {
                p += 1;
            }
        }
        return (f64::NAN, p);
    }
    if starts(p, b"inf") {
        p += 3;
        if starts(p, b"inity") {
            p += 5;
        }
        return (sign * f64::INFINITY, p);
    }
    match decimal_prefix(bytes, s) {
        Some(stop) => {
            let text = std::str::from_utf8(&bytes[s..stop]).unwrap_or_default();
            (text.parse::<f64>().unwrap_or(f64::NAN), stop)
        }
        None => (-1.0, s),
    }
}

/// Where `_Py_dg_strtod` stops reading a decimal at `at`: a sign, digits
/// with at most one point and at least one digit, then an exponent if one
/// is complete; `None` where there is no number.
fn decimal_prefix(bytes: &[u8], at: usize) -> Option<usize> {
    let digits = |mut p: usize| {
        while p < bytes.len() && bytes[p].is_ascii_digit() {
            p += 1;
        }
        p
    };
    let mut p = at;
    if matches!(bytes.get(p), Some(b'+' | b'-')) {
        p += 1;
    }
    let whole = digits(p);
    let mut stop = whole;
    let mut any = whole > p;
    if bytes.get(stop) == Some(&b'.') {
        let fraction = digits(stop + 1);
        any |= fraction > stop + 1;
        stop = fraction;
    }
    if !any {
        return None;
    }
    if matches!(bytes.get(stop), Some(b'e' | b'E')) {
        let mut e = stop + 1;
        if matches!(bytes.get(e), Some(b'+' | b'-')) {
            e += 1;
        }
        let exponent = digits(e);
        if exponent > e {
            stop = exponent;
        }
    }
    Some(stop)
}

/// CPython's `PyOS_strtol(text, &end, 10)` at `at`, for a 64-bit C `long`:
/// whitespace, a sign, whitespace again, digits; past the range, `LONG_MAX`,
/// unless the magnitude is exactly that of `LONG_MIN` and negative. Reading
/// stops after the last digit, or after the sign and whitespace where there
/// are none, so a lone sign reads as 0.
fn strtol(bytes: &[u8], at: usize) -> (i64, usize) {
    let mut p = at;
    while p < bytes.len() && is_space(bytes[p]) {
        p += 1;
    }
    let negative = bytes.get(p) == Some(&b'-');
    if matches!(bytes.get(p), Some(b'+' | b'-')) {
        p += 1;
    }
    while p < bytes.len() && is_space(bytes[p]) {
        p += 1;
    }
    let mut magnitude: u128 = 0;
    while p < bytes.len() && bytes[p].is_ascii_digit() {
        magnitude = (magnitude * 10 + u128::from(bytes[p] - b'0')).min(u128::from(u64::MAX) + 1);
        p += 1;
    }
    let value = if magnitude <= i64::MAX as u128 {
        let v = magnitude as i64;
        if negative { -v } else { v }
    } else if negative && magnitude == 1u128 << 63 {
        i64::MIN
    } else {
        i64::MAX
    };
    (value, p)
}

/// Whether Python's `json.loads` reads the text: one JSON value between
/// whitespace, with `NaN`, `Infinity` and `-Infinity` besides, and no
/// control character inside a string.
///
/// Nested deeper than this reads, Python's answer turns on how much of its
/// recursion limit the caller's stack has used, so a text that deep counts
/// as read, which makes the core refuse the column rather than guess.
pub fn json_accepts(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut p = skip_json_space(bytes, 0);
    let mut deep = false;
    if !json_value(bytes, &mut p, 0, &mut deep) {
        return deep;
    }
    skip_json_space(bytes, p) == bytes.len()
}

/// How deep the core reads JSON before it counts a text as read.
const JSON_DEPTH: usize = 200;

fn skip_json_space(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() && matches!(bytes[p], b' ' | b'\t' | b'\n' | b'\r') {
        p += 1;
    }
    p
}

/// One JSON value at `p`, `p` moved past it; `deep` set, and `false`,
/// past [`JSON_DEPTH`].
fn json_value(bytes: &[u8], p: &mut usize, depth: usize, deep: &mut bool) -> bool {
    if depth > JSON_DEPTH {
        *deep = true;
        return false;
    }
    let literal = |p: &mut usize, word: &[u8]| {
        if bytes[*p..].starts_with(word) {
            *p += word.len();
            true
        } else {
            false
        }
    };
    match bytes.get(*p) {
        Some(b'{') => {
            *p = skip_json_space(bytes, *p + 1);
            if bytes.get(*p) == Some(&b'}') {
                *p += 1;
                return true;
            }
            loop {
                if !json_string(bytes, p) {
                    return false;
                }
                *p = skip_json_space(bytes, *p);
                if bytes.get(*p) != Some(&b':') {
                    return false;
                }
                *p = skip_json_space(bytes, *p + 1);
                if !json_value(bytes, p, depth + 1, deep) {
                    return false;
                }
                *p = skip_json_space(bytes, *p);
                match bytes.get(*p) {
                    Some(b',') => *p = skip_json_space(bytes, *p + 1),
                    Some(b'}') => {
                        *p += 1;
                        return true;
                    }
                    _ => return false,
                }
            }
        }
        Some(b'[') => {
            *p = skip_json_space(bytes, *p + 1);
            if bytes.get(*p) == Some(&b']') {
                *p += 1;
                return true;
            }
            loop {
                if !json_value(bytes, p, depth + 1, deep) {
                    return false;
                }
                *p = skip_json_space(bytes, *p);
                match bytes.get(*p) {
                    Some(b',') => *p = skip_json_space(bytes, *p + 1),
                    Some(b']') => {
                        *p += 1;
                        return true;
                    }
                    _ => return false,
                }
            }
        }
        Some(b'"') => json_string(bytes, p),
        Some(b'n') => literal(p, b"null"),
        Some(b't') => literal(p, b"true"),
        Some(b'f') => literal(p, b"false"),
        Some(b'N') => literal(p, b"NaN"),
        Some(b'I') => literal(p, b"Infinity"),
        Some(b'-') if bytes[*p..].starts_with(b"-Infinity") => {
            *p += 9;
            true
        }
        Some(b'-' | b'0'..=b'9') => json_number(bytes, p),
        _ => false,
    }
}

/// `json`'s `NUMBER_RE`: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`,
/// where a fraction or exponent that is not complete is left unread.
fn json_number(bytes: &[u8], p: &mut usize) -> bool {
    let digits = |mut at: usize| {
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        at
    };
    let mut at = *p;
    if bytes.get(at) == Some(&b'-') {
        at += 1;
    }
    match bytes.get(at) {
        Some(b'0') => at += 1,
        Some(b'1'..=b'9') => at = digits(at),
        _ => return false,
    }
    if bytes.get(at) == Some(&b'.') && bytes.get(at + 1).is_some_and(u8::is_ascii_digit) {
        at = digits(at + 1);
    }
    if matches!(bytes.get(at), Some(b'e' | b'E')) {
        let mut e = at + 1;
        if matches!(bytes.get(e), Some(b'+' | b'-')) {
            e += 1;
        }
        if bytes.get(e).is_some_and(u8::is_ascii_digit) {
            at = digits(e);
        }
    }
    *p = at;
    true
}

/// A JSON string at `p`, as `json`'s strict scanner reads it: any escape of
/// `"\/bfnrt`, or `u` and four hex digits, and no character below `0x20`.
fn json_string(bytes: &[u8], p: &mut usize) -> bool {
    if bytes.get(*p) != Some(&b'"') {
        return false;
    }
    let mut at = *p + 1;
    loop {
        match bytes.get(at) {
            None => return false,
            Some(b'"') => {
                *p = at + 1;
                return true;
            }
            Some(b'\\') => match bytes.get(at + 1) {
                Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => at += 2,
                Some(b'u') => {
                    let hex = bytes.get(at + 2..at + 6);
                    if !hex.is_some_and(|h| h.iter().all(u8::is_ascii_hexdigit)) {
                        return false;
                    }
                    at += 6;
                }
                _ => return false,
            },
            Some(&c) if c < 0x20 => return false,
            Some(_) => at += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipping_sql_sanitizer_uses_explicit_numpy_capability() {
        for prefix in [false, true] {
            let series = Series::new(
                Dtype::Arrow,
                Arc::new(arrow_array::StringArray::from(vec!["x", "1,2"])),
            );
            let frame = crate::frame::Frame::new(vec!["items".into()], vec![series], 2);
            let mut host = sdm_schema::dt::Host::assumed();
            host.numpy_text_prefix = prefix;
            let mut warnings = Vec::new();
            let actual = crate::driver::sql::sanitize_with(
                &host,
                &[("items".into(), crate::driver::sql::dtype_of_name("intlist"))],
                &[],
                frame,
                &mut warnings,
            )
            .unwrap();
            assert_eq!(warnings.len(), usize::from(prefix));
            if prefix {
                assert_eq!(warnings[0].category, "builtins.DeprecationWarning");
                assert_eq!(actual.columns[0].dtype, Dtype::Object);
                assert!(matches!(
                    actual.columns[0].array.data_type(),
                    arrow_schema::DataType::List(_)
                ));
            } else {
                assert_eq!(
                    texts(actual.columns[0].array.as_ref()).unwrap(),
                    vec![Some("x".into()), Some("1,2".into())]
                );
            }
        }
    }

    #[test]
    fn missing_arrow_row_retains_only_prior_parser_warnings() {
        for dtype in [Dtype::Arrow, Dtype::Object] {
            let series = Series::new(
                dtype.clone(),
                Arc::new(arrow_array::StringArray::from(vec![
                    Some("x"),
                    None,
                    Some("y"),
                ])),
            );
            let mut warnings = Vec::new();
            let actual = to_list_with_policy(&series, ListKind::Int, true, &mut warnings).unwrap();
            assert_eq!(warnings.len(), usize::from(dtype == Dtype::Arrow));
            assert_eq!(actual.array.to_data(), series.array.to_data());
        }
    }

    #[test]
    fn numpy_reads_comma_separated_numbers() {
        assert_eq!(from_string_floats("1, 2.5,3"), Some(vec![1.0, 2.5, 3.0]));
        assert_eq!(from_string_floats(" 1 ,2 "), Some(vec![1.0, 2.0]));
        assert_eq!(from_string_floats(""), Some(vec![]));
        assert_eq!(from_string_floats("1,2,"), Some(vec![1.0, 2.0]));
        assert_eq!(from_string_floats("1 ,"), Some(vec![1.0]));
        assert_eq!(from_string_floats("1,\n2"), Some(vec![1.0, 2.0]));
        assert_eq!(from_string_floats("1e40"), Some(vec![f32::INFINITY]));
        assert_eq!(from_string_floats("1."), Some(vec![1.0]));
        assert_eq!(from_string_floats(".5"), Some(vec![0.5]));
        // Whitespace alone reads as CPython's failed conversion, -1.0.
        assert_eq!(from_string_floats("  "), Some(vec![-1.0]));
        assert_eq!(from_string_floats("1, "), Some(vec![1.0, -1.0]));
        for refused in [
            "1,,2", "1;2", "abc", "1.5,x", "0x10,1", ",1", "1 2", "1_000", "1e",
        ] {
            assert_eq!(from_string_floats(refused), None, "{refused}");
        }
        let special = from_string_floats("nan,inf,-Infinity,NaN(x_1)").unwrap();
        assert!(special[0].is_nan() && special[3].is_nan());
        assert_eq!(special[1..3], [f32::INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn numpy_reads_ints_as_a_c_long() {
        assert_eq!(from_string_ints(" 1 ,2 "), Some(vec![1, 2]));
        assert_eq!(from_string_ints("+3,-4"), Some(vec![3, -4]));
        assert_eq!(from_string_ints("  "), Some(vec![0]));
        assert_eq!(from_string_ints("-"), Some(vec![0]));
        assert_eq!(from_string_ints("- 5"), Some(vec![-5]));
        assert_eq!(
            from_string_ints("9223372036854775808"),
            Some(vec![i64::MAX])
        );
        assert_eq!(
            from_string_ints("-9223372036854775808"),
            Some(vec![i64::MIN])
        );
        assert_eq!(
            from_string_ints("-9223372036854775809"),
            Some(vec![i64::MAX])
        );
        assert_eq!(
            from_string_ints("-99999999999999999999999"),
            Some(vec![i64::MAX])
        );
        for refused in ["1.5", "1e5", "abc", "nan", "1,,2"] {
            assert_eq!(from_string_ints(refused), None, "{refused}");
        }
    }

    #[test]
    fn captured_numpy_text_policies_match_112_primitive_observations() {
        let observations = [
            ("int", false, "", Some(""), false),
            ("int", false, " ", Some("0000000000000000"), false),
            ("int", false, "\t", Some("0000000000000000"), false),
            ("int", false, "[]", Some(""), false),
            ("int", false, "[ ]", Some("0000000000000000"), false),
            ("int", false, "1", Some("0100000000000000"), false),
            (
                "int",
                false,
                "[1,2]",
                Some("01000000000000000200000000000000"),
                false,
            ),
            ("int", false, "x", None, false),
            ("int", false, "x,1", None, false),
            ("int", false, "1x", None, false),
            ("int", false, "1,x", None, false),
            ("int", false, "1,2x", None, false),
            ("int", false, "1,", Some("0100000000000000"), false),
            ("int", false, ",1", None, false),
            ("int", false, "1,,2", None, false),
            ("int", false, "[null7]", None, false),
            ("int", false, "null7", None, false),
            ("int", false, "nan", None, false),
            ("int", false, "inf", None, false),
            ("int", false, "-inf", None, false),
            ("int", false, "1e40", None, false),
            (
                "int",
                false,
                "9223372036854775807",
                Some("ffffffffffffff7f"),
                false,
            ),
            (
                "int",
                false,
                "9223372036854775808",
                Some("ffffffffffffff7f"),
                false,
            ),
            (
                "int",
                false,
                "-9223372036854775809",
                Some("ffffffffffffff7f"),
                false,
            ),
            ("int", false, "0x10", None, false),
            ("int", false, "1.5", None, false),
            ("int", false, "1e2", None, false),
            ("int", false, "[1][2]", None, false),
            ("float32", false, "", Some(""), false),
            ("float32", false, " ", Some("000080bf"), false),
            ("float32", false, "\t", Some("000080bf"), false),
            ("float32", false, "[]", Some(""), false),
            ("float32", false, "[ ]", Some("000080bf"), false),
            ("float32", false, "1", Some("0000803f"), false),
            ("float32", false, "[1,2]", Some("0000803f00000040"), false),
            ("float32", false, "x", None, false),
            ("float32", false, "x,1", None, false),
            ("float32", false, "1x", None, false),
            ("float32", false, "1,x", None, false),
            ("float32", false, "1,2x", None, false),
            ("float32", false, "1,", Some("0000803f"), false),
            ("float32", false, ",1", None, false),
            ("float32", false, "1,,2", None, false),
            ("float32", false, "[null7]", None, false),
            ("float32", false, "null7", None, false),
            ("float32", false, "nan", Some("0000c07f"), false),
            ("float32", false, "inf", Some("0000807f"), false),
            ("float32", false, "-inf", Some("000080ff"), false),
            ("float32", false, "1e40", Some("0000807f"), false),
            (
                "float32",
                false,
                "9223372036854775807",
                Some("0000005f"),
                false,
            ),
            (
                "float32",
                false,
                "9223372036854775808",
                Some("0000005f"),
                false,
            ),
            (
                "float32",
                false,
                "-9223372036854775809",
                Some("000000df"),
                false,
            ),
            ("float32", false, "0x10", None, false),
            ("float32", false, "1.5", Some("0000c03f"), false),
            ("float32", false, "1e2", Some("0000c842"), false),
            ("float32", false, "[1][2]", None, false),
            ("int", true, "", Some(""), false),
            ("int", true, " ", Some("0000000000000000"), false),
            ("int", true, "\t", Some("0000000000000000"), false),
            ("int", true, "[]", Some(""), false),
            ("int", true, "[ ]", Some("0000000000000000"), false),
            ("int", true, "1", Some("0100000000000000"), false),
            (
                "int",
                true,
                "[1,2]",
                Some("01000000000000000200000000000000"),
                false,
            ),
            ("int", true, "x", Some(""), true),
            ("int", true, "x,1", Some(""), true),
            ("int", true, "1x", Some("0100000000000000"), true),
            ("int", true, "1,x", Some("0100000000000000"), true),
            (
                "int",
                true,
                "1,2x",
                Some("01000000000000000200000000000000"),
                true,
            ),
            ("int", true, "1,", Some("0100000000000000"), false),
            ("int", true, ",1", Some(""), true),
            ("int", true, "1,,2", Some("0100000000000000"), true),
            ("int", true, "[null7]", Some(""), true),
            ("int", true, "null7", Some(""), true),
            ("int", true, "nan", Some(""), true),
            ("int", true, "inf", Some(""), true),
            ("int", true, "-inf", Some("0000000000000000"), true),
            ("int", true, "1e40", Some("0100000000000000"), true),
            (
                "int",
                true,
                "9223372036854775807",
                Some("ffffffffffffff7f"),
                false,
            ),
            (
                "int",
                true,
                "9223372036854775808",
                Some("ffffffffffffff7f"),
                false,
            ),
            (
                "int",
                true,
                "-9223372036854775809",
                Some("ffffffffffffff7f"),
                false,
            ),
            ("int", true, "0x10", Some("0000000000000000"), true),
            ("int", true, "1.5", Some("0100000000000000"), true),
            ("int", true, "1e2", Some("0100000000000000"), true),
            ("int", true, "[1][2]", Some("0100000000000000"), true),
            ("float32", true, "", Some(""), false),
            ("float32", true, " ", Some("000080bf"), false),
            ("float32", true, "\t", Some("000080bf"), false),
            ("float32", true, "[]", Some(""), false),
            ("float32", true, "[ ]", Some("000080bf"), false),
            ("float32", true, "1", Some("0000803f"), false),
            ("float32", true, "[1,2]", Some("0000803f00000040"), false),
            ("float32", true, "x", Some(""), true),
            ("float32", true, "x,1", Some(""), true),
            ("float32", true, "1x", Some("0000803f"), true),
            ("float32", true, "1,x", Some("0000803f"), true),
            ("float32", true, "1,2x", Some("0000803f00000040"), true),
            ("float32", true, "1,", Some("0000803f"), false),
            ("float32", true, ",1", Some(""), true),
            ("float32", true, "1,,2", Some("0000803f"), true),
            ("float32", true, "[null7]", Some(""), true),
            ("float32", true, "null7", Some(""), true),
            ("float32", true, "nan", Some("0000c07f"), false),
            ("float32", true, "inf", Some("0000807f"), false),
            ("float32", true, "-inf", Some("000080ff"), false),
            ("float32", true, "1e40", Some("0000807f"), false),
            (
                "float32",
                true,
                "9223372036854775807",
                Some("0000005f"),
                false,
            ),
            (
                "float32",
                true,
                "9223372036854775808",
                Some("0000005f"),
                false,
            ),
            (
                "float32",
                true,
                "-9223372036854775809",
                Some("000000df"),
                false,
            ),
            ("float32", true, "0x10", Some("00000000"), true),
            ("float32", true, "1.5", Some("0000c03f"), false),
            ("float32", true, "1e2", Some("0000c842"), false),
            ("float32", true, "[1][2]", Some("0000803f"), true),
        ];
        for (kind, prefix, text, expected, warned) in observations {
            let text = strip_brackets(text);
            let (actual, warnings) = if kind == "int" {
                let (values, warnings) = from_string_ints_with_policy(text, prefix);
                (
                    values.map(|values| {
                        values
                            .into_iter()
                            .flat_map(i64::to_le_bytes)
                            .map(|byte| format!("{byte:02x}"))
                            .collect::<String>()
                    }),
                    warnings,
                )
            } else {
                let (values, warnings) = from_string_floats_with_policy(text, prefix);
                (
                    values.map(|values| {
                        values
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .map(|byte| format!("{byte:02x}"))
                            .collect::<String>()
                    }),
                    warnings,
                )
            };
            assert_eq!(actual.as_deref(), expected, "{kind} {prefix} {text:?}");
            assert_eq!(warnings, warned, "{kind} {prefix} {text:?}");
        }
    }

    #[test]
    fn json_is_read_as_python_reads_it() {
        for accepted in [
            "[1, 2.5, -3e2]",
            " {\"a\": [true, false, null]} ",
            "NaN",
            "[-Infinity, Infinity]",
            "\"\\ud800\"",
            "0",
            "[]",
        ] {
            assert!(json_accepts(accepted), "{accepted}");
        }
        for refused in [
            "", " ", "1,,2", "[1,]", "01", "1.", "'a'", "[1] x", "\"a\nb\"", "-", "nan", "{1: 2}",
        ] {
            assert!(!json_accepts(refused), "{refused}");
        }
    }
}
