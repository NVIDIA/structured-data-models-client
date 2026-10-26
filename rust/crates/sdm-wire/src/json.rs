// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON as the wire carries it: objects keep the order their members were
//! written in, and a number is kept as the token its sender wrote, so a
//! value read and written again is the same text and never passes through
//! a float or a fixed-width integer.
//!
//! Reading is `serde_json`'s: the whole text is first read once, which
//! refuses what JSON does not allow (`NaN` and `Infinity` included) with the
//! position and category of the fault, and then each value's own text is
//! taken as `serde_json`'s `RawValue` gives it. A member named twice keeps
//! its first place and its last value, as a Python dictionary built from
//! the same text does.

use std::collections::BTreeMap;

use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

/// A JSON value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A number, as the token written: `1`, `-0`, `97.0`, `1e-05`.
    Number(String),
    Text(String),
    Array(Vec<Value>),
    /// Members in the order written.
    Object(Vec<(String, Value)>),
}

/// Why a text is not JSON: where, and what kind of fault, never the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadError {
    /// `syntax`, `eof` or `data`, as `serde_json` classifies the fault,
    /// `depth` for text nested past [`MAX_DEPTH`], `not_finite` for a
    /// `NaN` or `Infinity` token, or `encoding` for an answer's bytes that
    /// do not decode in the encoding they are detected in (UTF-16 or
    /// UTF-32).
    pub category: &'static str,
    pub line: usize,
    pub column: usize,
}

impl ReadError {
    fn of(error: &serde_json::Error) -> Self {
        use serde_json::error::Category;
        ReadError {
            category: match error.classify() {
                Category::Io => "io",
                Category::Syntax => "syntax",
                Category::Data => "data",
                Category::Eof => "eof",
            },
            line: error.line(),
            column: error.column(),
        }
    }
}

/// The deepest nesting of arrays and objects [`read`] takes: `serde_json`'s
/// own default limit, which every other JSON reading in the core keeps. A
/// prediction response nests a handful of levels, and a request state's
/// payload six; anything deeper is refused (category `depth`) before a
/// value is built, so neither building nor dropping one recurses past it.
pub const MAX_DEPTH: usize = 128;

/// Reads `bytes` as one JSON value, nested at most [`MAX_DEPTH`] deep, with
/// no number that is not finite: a `NaN` or `Infinity` token, which JSON
/// does not hold and Python's reader takes, is refused as category
/// `not_finite`, as a finite-looking token past a double's range is by
/// [`Value::all_finite`].
pub fn read(bytes: &[u8]) -> Result<Value, ReadError> {
    check_non_finite_tokens(bytes)?;
    // The whole text once, for its faults' positions within it. Ignoring a
    // value is iterative in `serde_json`, so this holds at any depth.
    serde_json::from_slice::<serde::de::IgnoredAny>(bytes).map_err(|e| ReadError::of(&e))?;
    // The text is JSON, so its nesting can be counted without recursion
    // before anything recursive reads it.
    check_depth(bytes)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|e| ReadError::of(&e))?;
    value(raw).map_err(|e| ReadError::of(&e))
}

/// Reads an endpoint's answer as [`read`] does, but takes a `NaN`,
/// `Infinity` or `-Infinity` token outside text as the number Python's
/// reader takes it for, as the reference's client reads an answer (owner
/// decision of 2026-10-12, relayed: a non-finite prediction is passed
/// through, register P1B-4): each such token is a [`Value::Number`] of the
/// token as written, which `str::parse::<f64>` reads as a NaN or an
/// infinity, and which [`Value::to_bytes`] writes back as written.
///
/// Each token is read as a JSON string literal standing for it, one whose
/// text the answer itself does not hold (`"\uE000NaN"`, the private-use
/// character's number raised until no such text is in the answer), and the
/// value whose raw text is that literal is read back as the token; so no
/// text the endpoint wrote is read as a number.
///
/// The bytes are first decoded to UTF-8 text as Python's `json.loads`
/// decodes bytes ([`answer_text`]), so an answer the reference reads in
/// UTF-8 with a byte order mark, UTF-16 or UTF-32 is read alike.
///
/// A lone surrogate, which Python's reader keeps in a string and UTF-8
/// cannot hold, is read as U+FFFD, the replacement character, and the
/// answer is read on (register P1B-24, the owner's KO2-93): one escaped in
/// text (`\ud800` with no low surrogate escaped after it, or a low one
/// alone), one in UTF-16 or UTF-32 ([`answer_text`]), and one written in
/// UTF-8 as its three bytes (`ED A0 80`), as Python's reader takes them
/// with `surrogatepass`. Nothing else in the answer changes.
pub fn read_answer(bytes: &[u8]) -> Result<Value, ReadError> {
    let text = answer_text(bytes)?;
    let text = lone_surrogates_replaced(&text);
    read_answer_utf8(&text)
}

/// `bytes` with each lone surrogate replaced by U+FFFD, each replacement
/// as long as what it replaces, so a fault's position stays where it was:
/// an escape `\uD800`..`\uDFFF` in a string that is not a high one
/// followed by an escaped low one becomes `\uFFFD`, and a surrogate's
/// three UTF-8 bytes (`ED` then `A0`..`BF`, then a continuation byte)
/// become `EF BF BD`. Escapes are read in order, so the `\u` after an
/// escaped backslash (`\\u`) is text, not an escape.
fn lone_surrogates_replaced(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    use std::borrow::Cow;
    fn hex4(b: &[u8]) -> Option<u16> {
        let s = std::str::from_utf8(b.get(..4)?).ok()?;
        if !s.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        u16::from_str_radix(s, 16).ok()
    }
    let high = |u: u16| (0xD800..=0xDBFF).contains(&u);
    let low = |u: u16| (0xDC00..=0xDFFF).contains(&u);
    let mut out: Option<Vec<u8>> = None;
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0xED
            && matches!(bytes.get(i + 1), Some(0xA0..=0xBF))
            && matches!(bytes.get(i + 2), Some(0x80..=0xBF))
        {
            out.get_or_insert_with(|| bytes.to_vec())[i..i + 3]
                .copy_from_slice("\u{FFFD}".as_bytes());
            i += 3;
            continue;
        }
        if !in_string {
            in_string = b == b'"';
            i += 1;
            continue;
        }
        match b {
            b'"' => {
                in_string = false;
                i += 1;
            }
            b'\\' if bytes.get(i + 1) == Some(&b'u') => {
                let Some(unit) = hex4(&bytes[i + 2..]) else {
                    i += 2;
                    continue;
                };
                let paired = high(unit)
                    && bytes.get(i + 6) == Some(&b'\\')
                    && bytes.get(i + 7) == Some(&b'u')
                    && hex4(&bytes[(i + 8).min(bytes.len())..]).is_some_and(low);
                if paired {
                    i += 12;
                } else if high(unit) || low(unit) {
                    out.get_or_insert_with(|| bytes.to_vec())[i + 2..i + 6]
                        .copy_from_slice(b"FFFD");
                    i += 6;
                } else {
                    i += 6;
                }
            }
            b'\\' => i += 2,
            _ => i += 1,
        }
    }
    out.map_or(Cow::Borrowed(bytes), Cow::Owned)
}

/// Reads the option attachment's nonfinite numeric tokens losslessly.
/// This does not decide which options admit them, and does not apply the
/// endpoint decoder's encoding detection or surrogate replacement.
pub fn read_options(bytes: &[u8]) -> Result<Value, ReadError> {
    read_answer_utf8(bytes)
}

/// Lossless nonfinite-token decoding shared by answer and option readers.
fn read_answer_utf8(bytes: &[u8]) -> Result<Value, ReadError> {
    let tokens = non_finite_tokens(bytes);
    if tokens.is_empty() {
        return read(bytes);
    }
    let marker = (0xE000u32..=0xF8FF)
        .map(|n| format!("\\u{n:04X}"))
        .find(|m| !contains(bytes, m.as_bytes()))
        .ok_or(ReadError {
            category: "syntax",
            line: 1,
            column: 0,
        })?;
    let mut text = Vec::with_capacity(bytes.len() + tokens.len() * 16);
    let mut from = 0;
    for &(at, token) in &tokens {
        text.extend_from_slice(&bytes[from..at]);
        text.push(b'"');
        text.extend_from_slice(marker.as_bytes());
        text.extend_from_slice(token.as_bytes());
        text.push(b'"');
        from = at + token.len();
    }
    text.extend_from_slice(&bytes[from..]);
    serde_json::from_slice::<serde::de::IgnoredAny>(&text).map_err(|e| ReadError::of(&e))?;
    check_depth(&text)?;
    let raw: &RawValue = serde_json::from_slice(&text).map_err(|e| ReadError::of(&e))?;
    value_with(raw, Some(&marker)).map_err(|e| ReadError::of(&e))
}

/// An answer's bytes as UTF-8 text, decoded as Python's `json.loads`
/// decodes bytes (`json.detect_encoding`, after RFC 8259 and RFC 4627): a
/// UTF-32 or UTF-16 byte order mark, then a UTF-8 one, names the encoding
/// and is dropped; without one, the zero bytes among the first four name
/// UTF-16 or UTF-32 and their byte order, as the first character of JSON
/// text is ASCII; anything else is UTF-8. UTF-8 without a mark is the bytes
/// as they are. Text that does not decode is refused, category `encoding`,
/// as Python's decoding raises; a lone surrogate, which Python's decoding
/// keeps (`surrogatepass`), is read as U+FFFD, as [`read_answer`] reads one
/// escaped (register P1B-24). The position is the first byte's.
pub fn answer_text(bytes: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>, ReadError> {
    use std::borrow::Cow;
    #[derive(Clone, Copy)]
    enum Width {
        Two,
        Four,
    }
    let (width, big_endian, skip) = match bytes {
        [0, 0, 0xFE, 0xFF, ..] => (Width::Four, true, 4),
        [0xFF, 0xFE, 0, 0, ..] => (Width::Four, false, 4),
        [0xFE, 0xFF, ..] => (Width::Two, true, 2),
        [0xFF, 0xFE, ..] => (Width::Two, false, 2),
        [0xEF, 0xBB, 0xBF, rest @ ..] => return Ok(Cow::Borrowed(rest)),
        [0, 0, _, _, ..] => (Width::Four, true, 0),
        [0, _, _, _, ..] | [0, _] => (Width::Two, true, 0),
        [_, 0, 0, 0, ..] => (Width::Four, false, 0),
        [_, 0, _, _, ..] | [_, 0] => (Width::Two, false, 0),
        _ => return Ok(Cow::Borrowed(bytes)),
    };
    let undecodable = ReadError {
        category: "encoding",
        line: 1,
        column: 0,
    };
    let body = &bytes[skip..];
    let mut text = String::with_capacity(body.len());
    match width {
        Width::Two => {
            if !body.len().is_multiple_of(2) {
                return Err(undecodable);
            }
            let units = body.chunks_exact(2).map(|u| {
                let u = [u[0], u[1]];
                if big_endian {
                    u16::from_be_bytes(u)
                } else {
                    u16::from_le_bytes(u)
                }
            });
            for c in char::decode_utf16(units) {
                text.push(c.unwrap_or(char::REPLACEMENT_CHARACTER));
            }
        }
        Width::Four => {
            if !body.len().is_multiple_of(4) {
                return Err(undecodable);
            }
            for u in body.chunks_exact(4) {
                let u = [u[0], u[1], u[2], u[3]];
                let n = if big_endian {
                    u32::from_be_bytes(u)
                } else {
                    u32::from_le_bytes(u)
                };
                text.push(match n {
                    0xD800..=0xDFFF => char::REPLACEMENT_CHARACTER,
                    _ => char::from_u32(n).ok_or(undecodable)?,
                });
            }
        }
    }
    Ok(Cow::Owned(text.into_bytes()))
}

/// Whether `haystack` holds `needle`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Each `NaN`, `Infinity` and `-Infinity` token in a value's place outside
/// text: where it starts, and the token. One where an object's member name
/// goes is left as it is, so the reading refuses it, as Python's does.
fn non_finite_tokens(bytes: &[u8]) -> Vec<(usize, &'static str)> {
    let mut found = Vec::new();
    let (mut in_text, mut escaped) = (false, false);
    // Whether each open bracket is an object's, and whether a member's
    // name is due.
    let mut objects: Vec<bool> = Vec::new();
    let mut name_due = false;
    let mut at = 0;
    while at < bytes.len() {
        let byte = bytes[at];
        if in_text {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_text = false,
                _ => {}
            }
            at += 1;
            continue;
        }
        let rest = &bytes[at..];
        let token = if name_due {
            None
        } else if rest.starts_with(b"NaN") {
            Some("NaN")
        } else if rest.starts_with(b"Infinity") {
            Some("Infinity")
        } else if rest.starts_with(b"-Infinity") {
            Some("-Infinity")
        } else {
            None
        };
        if let Some(token) = token {
            found.push((at, token));
            at += token.len();
            continue;
        }
        match byte {
            b'"' => in_text = true,
            b'{' => {
                objects.push(true);
                name_due = true;
            }
            b'[' => {
                objects.push(false);
                name_due = false;
            }
            b'}' | b']' => {
                objects.pop();
                name_due = false;
            }
            b':' => name_due = false,
            b',' => name_due = objects.last() == Some(&true),
            _ => {}
        }
        at += 1;
    }
    found
}

/// Refuses a `NaN`, `Infinity` or `-Infinity` token outside text, as the
/// non-finite number Python's reader takes it for. JSON's only bare words
/// are `true`, `false` and `null`, so a bare `N` or `I` is one of these or
/// text that is not JSON, which the reading after this refuses either way.
fn check_non_finite_tokens(bytes: &[u8]) -> Result<(), ReadError> {
    let (mut line, mut column) = (1usize, 0usize);
    let (mut in_text, mut escaped) = (false, false);
    for (at, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            line += 1;
            column = 0;
        } else {
            column += 1;
        }
        if in_text {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_text = false,
                _ => {}
            }
            continue;
        }
        let word = |w: &[u8]| bytes[at..].starts_with(w);
        match byte {
            b'"' => in_text = true,
            b'N' if word(b"NaN") => {
                return Err(ReadError {
                    category: "not_finite",
                    line,
                    column,
                });
            }
            b'I' if word(b"Infinity") => {
                return Err(ReadError {
                    category: "not_finite",
                    line,
                    column,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Refuses JSON text nested past [`MAX_DEPTH`], at the bracket that passes
/// it; the text has already been read as JSON.
fn check_depth(bytes: &[u8]) -> Result<(), ReadError> {
    let (mut depth, mut line, mut column) = (0usize, 1usize, 0usize);
    let (mut in_text, mut escaped) = (false, false);
    for &byte in bytes {
        if byte == b'\n' {
            line += 1;
            column = 0;
        } else {
            column += 1;
        }
        if in_text {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_text = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_text = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > MAX_DEPTH {
                    return Err(ReadError {
                        category: "depth",
                        line,
                        column,
                    });
                }
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

fn value(raw: &RawValue) -> Result<Value, serde_json::Error> {
    value_with(raw, None)
}

/// [`value`], reading a string literal of `marker` and a token
/// ([`read_answer`]) as that token.
fn value_with(raw: &RawValue, marker: Option<&str>) -> Result<Value, serde_json::Error> {
    let text = raw.get();
    if let Some(token) = marker.and_then(|marker| {
        text.strip_prefix('"')
            .and_then(|t| t.strip_suffix('"'))
            .and_then(|t| t.strip_prefix(marker))
            .filter(|t| matches!(*t, "NaN" | "Infinity" | "-Infinity"))
    }) {
        return Ok(Value::Number(token.to_owned()));
    }
    Ok(match text.as_bytes().first() {
        Some(b'{') => {
            let Members(members) = serde_json::from_str(text)?;
            let mut read: Vec<(String, Value)> = Vec::with_capacity(members.len());
            // Each name's place, so a repeated one is found in log time and
            // a wide object reads in n log n, not n squared.
            let mut placed: BTreeMap<String, usize> = BTreeMap::new();
            for (name, raw) in members {
                let member = value_with(raw, marker)?;
                match placed.get(&name) {
                    Some(&at) => read[at].1 = member,
                    None => {
                        placed.insert(name.clone(), read.len());
                        read.push((name, member));
                    }
                }
            }
            Value::Object(read)
        }
        Some(b'[') => {
            let items: Vec<&RawValue> = serde_json::from_str(text)?;
            Value::Array(
                items
                    .into_iter()
                    .map(|raw| value_with(raw, marker))
                    .collect::<Result<_, _>>()?,
            )
        }
        Some(b'"') => Value::Text(serde_json::from_str(text)?),
        Some(b't') => Value::Bool(true),
        Some(b'f') => Value::Bool(false),
        Some(b'n') => Value::Null,
        _ => Value::Number(text.to_owned()),
    })
}

/// An object's members, each its raw text, in order, names repeated.
struct Members<'a>(Vec<(String, &'a RawValue)>);

impl<'de: 'a, 'a> serde::Deserialize<'de> for Members<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Each;
        impl<'de> Visitor<'de> for Each {
            type Value = Members<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Members<'de>, A::Error> {
                let mut members = Vec::new();
                while let Some((name, raw)) = map.next_entry::<String, &'de RawValue>()? {
                    members.push((name, raw));
                }
                Ok(Members(members))
            }
        }
        deserializer.deserialize_map(Each)
    }
}

impl Value {
    /// An object's member, by name.
    pub fn get(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Sets an object's member: in its place if it is there, appended if
    /// not, as a dictionary's assignment does. Nothing for a non-object.
    pub fn set(&mut self, name: &str, value: Value) {
        if let Value::Object(members) = self {
            match members.iter_mut().find(|(n, _)| n == name) {
                Some(slot) => slot.1 = value,
                None => members.push((name.to_owned(), value)),
            }
        }
    }

    /// Removes an object's member, if it is there.
    pub fn remove(&mut self, name: &str) {
        if let Value::Object(members) = self {
            members.retain(|(n, _)| n != name);
        }
    }

    pub fn text(value: &str) -> Value {
        Value::Text(value.to_owned())
    }

    /// The value's text, if it is text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Whether a decoded value has only ordinary JSON numeric tokens.
    /// Unlike [`Self::all_finite`], this preserves finite numbers outside
    /// double range; only the three extra literal tokens admitted by
    /// [`read_options`] and [`read_answer`] are excluded.
    pub fn all_json_numbers(&self) -> bool {
        match self {
            Value::Number(token) => !matches!(token.as_str(), "NaN" | "Infinity" | "-Infinity"),
            Value::Array(items) => items.iter().all(Value::all_json_numbers),
            Value::Object(members) => members.iter().all(|(_, v)| v.all_json_numbers()),
            _ => true,
        }
    }

    /// Whether every number in the value reads as a finite double: an
    /// integer token always does, as an integer of any size; a float token
    /// past a double's range (`1e400`) reads as an infinity, and does not.
    pub fn all_finite(&self) -> bool {
        match self {
            Value::Number(token) => {
                is_integer_token(token) || token.parse::<f64>().is_ok_and(f64::is_finite)
            }
            Value::Array(items) => items.iter().all(Value::all_finite),
            Value::Object(members) => members.iter().all(|(_, v)| v.all_finite()),
            _ => true,
        }
    }

    /// The value's UTF-8 JSON text: compact, members in order, each number
    /// its token as read.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(token) => out.extend_from_slice(token.as_bytes()),
            Value::Text(text) => write_text(text, out),
            Value::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    item.write(out);
                }
                out.push(b']');
            }
            Value::Object(members) => {
                out.push(b'{');
                for (i, (name, member)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write_text(name, out);
                    out.push(b':');
                    member.write(out);
                }
                out.push(b'}');
            }
        }
    }
}

/// A value written as [`Value::to_bytes`] writes it: each number its token,
/// members in order. This is how a payload's cells are written
/// ([`crate::payload`]).
impl serde::Serialize for Value {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeMap, SerializeSeq};
        match self {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Number(token) => RawValue::from_string(token.clone())
                .map_err(|_| S::Error::custom("a number token that is not JSON"))?
                .serialize(serializer),
            Value::Text(text) => serializer.serialize_str(text),
            Value::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Value::Object(members) => {
                let mut map = serializer.serialize_map(Some(members.len()))?;
                for (name, member) in members {
                    map.serialize_entry(name, member)?;
                }
                map.end()
            }
        }
    }
}

/// Any JSON value: a payload's cell holds whatever its column's dtype
/// gives.
impl schemars::JsonSchema for Value {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Value".into()
    }

    fn inline_schema() -> bool {
        true
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::Schema::from(true)
    }
}

impl Value {
    /// An integer, as its digits.
    pub fn int(value: i64) -> Value {
        Value::Number(value.to_string())
    }

    /// A finite double, as the shortest text that reads back as the same
    /// double; `None` for a NaN or an infinity, which JSON does not hold.
    pub fn float(value: f64) -> Option<Value> {
        if !value.is_finite() {
            return None;
        }
        serde_json::to_string(&value).ok().map(Value::Number)
    }

    /// An array of texts.
    pub fn texts<'a>(values: impl IntoIterator<Item = &'a str>) -> Value {
        Value::Array(values.into_iter().map(Value::text).collect())
    }
}

fn write_text(text: &str, out: &mut Vec<u8>) {
    // serde_json escapes a string's quotes, backslashes and control
    // characters; writing a `str` to a `Vec` cannot fail.
    if serde_json::to_writer(&mut *out, text).is_err() {
        unreachable!("writing a string to memory cannot fail");
    }
}

/// A number token's value as a whole number, when it is one: an integer
/// token, or a finite float token with no fraction, as Python's
/// `int(float(...))` reads it. `None` for anything else, a whole number
/// past `i128` included.
pub fn whole(token: &str) -> Option<i128> {
    if let Ok(integer) = token.parse::<i128>() {
        return Some(integer);
    }
    if !token.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        // An integer token past i128.
        return None;
    }
    let float: f64 = token.parse().ok()?;
    if !float.is_finite() || float.fract() != 0.0 || float.abs() >= 2f64.powi(127) {
        return None;
    }
    // Whole and within i128, so the conversion is exact.
    Some(float as i128)
}

/// Whether a number token is an integer token, as JSON writes one: an
/// optional minus and digits, no fraction and no exponent.
pub fn is_integer_token(token: &str) -> bool {
    let digits = token.strip_prefix('-').unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod option_attachment_tests {
    use super::{Value, read, read_options};

    #[test]
    fn nonfinite_tokens_belong_only_to_the_option_decoder() {
        let bytes = br#"{"lag_timesteps":NaN,"text":"NaN","nested":[Infinity,-Infinity]}"#;
        assert!(read(bytes).is_err());
        let value = read_options(bytes).unwrap();
        assert_eq!(
            value.get("lag_timesteps"),
            Some(&Value::Number("NaN".into()))
        );
        assert_eq!(value.get("text"), Some(&Value::Text("NaN".into())));
        assert!(!value.get("nested").unwrap().all_finite());
        assert!(read_options(br#"{"lag_timesteps":NaNa}"#).is_err());
    }

    #[test]
    fn ordinary_json_numbers_keep_their_full_exponent_range() {
        let finite =
            read_options(br#"{"lag_timesteps":1e400,"nested":[-1e400,18446744073709551616]}"#)
                .unwrap();
        assert!(finite.all_json_numbers());
        assert!(!finite.all_finite(), "finite JSON need not fit a double");
        for token in ["NaN", "Infinity", "-Infinity"] {
            let nested = read_options(format!(r#"{{"nested":[{token}]}}"#).as_bytes()).unwrap();
            assert!(!nested.all_json_numbers(), "{token}");
            let text = read_options(format!(r#"{{"nested":["{token}"]}}"#).as_bytes()).unwrap();
            assert!(text.all_json_numbers(), "{token} text stays text");
        }
    }

    #[test]
    fn options_do_not_inherit_endpoint_encoding_or_surrogate_repair() {
        assert!(read_options(b"\xff\xfe{\0}\0").is_err());
        assert!(read_options(br#"{"lag_timesteps":NaN,"text":"\ud800"}"#).is_err());
        assert!(read_options(br#"{"lag_timesteps":NaN,"text":"\uE000NaN"}"#).is_ok());
    }
}
