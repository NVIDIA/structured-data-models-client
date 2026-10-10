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
    /// `syntax`, `eof` or `data`, as `serde_json` classifies the fault, or
    /// `depth` for text nested past [`MAX_DEPTH`].
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

/// Reads `bytes` as one JSON value, nested at most [`MAX_DEPTH`] deep.
pub fn read(bytes: &[u8]) -> Result<Value, ReadError> {
    // The whole text once, for its faults' positions within it. Ignoring a
    // value is iterative in `serde_json`, so this holds at any depth.
    serde_json::from_slice::<serde::de::IgnoredAny>(bytes).map_err(|e| ReadError::of(&e))?;
    // The text is JSON, so its nesting can be counted without recursion
    // before anything recursive reads it.
    check_depth(bytes)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|e| ReadError::of(&e))?;
    value(raw).map_err(|e| ReadError::of(&e))
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
    let text = raw.get();
    Ok(match text.as_bytes().first() {
        Some(b'{') => {
            let Members(members) = serde_json::from_str(text)?;
            let mut read: Vec<(String, Value)> = Vec::with_capacity(members.len());
            for (name, raw) in members {
                let member = value(raw)?;
                match read.iter_mut().find(|(n, _)| *n == name) {
                    Some(slot) => slot.1 = member,
                    None => read.push((name, member)),
                }
            }
            Value::Object(read)
        }
        Some(b'[') => {
            let items: Vec<&RawValue> = serde_json::from_str(text)?;
            Value::Array(items.into_iter().map(value).collect::<Result<_, _>>()?)
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
