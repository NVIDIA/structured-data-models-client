// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A view definition's YAML, read as the old client reads it: PyYAML
//! 6.0.3's `safe_load`, whose values are YAML 1.1's (design T3.5 part 5,
//! old-behaviour row 69). `yaml-rust2` reads the text into events; the
//! values are given here, by PyYAML's resolvers and constructors
//! (`yaml/resolver.py`, `yaml/constructor.py`): `yes`, `on` and `off` are
//! booleans and `y` and `n` text, `~` and an empty value are null, `017`
//! is octal and `1:30` base 60, `1e3` and `1.0e3` are text and `1.0e+3` a
//! float, `2024-01-01` a date; a quoted scalar is always text. A mapping
//! keeps its first key's place and its last value, keys equal as Python
//! compares them (`1`, `1.0` and `true` are one key), and `<<` merges.
//!
//! A value is data only: the texts the old client's messages quote it by,
//! Python's `repr` and `str` of it, are the core's to write
//! (`sdm-core::view_words`, through `sdm-pql`).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use yaml_rust2::parser::{Event, MarkedEventReceiver, Parser, Tag};
use yaml_rust2::scanner::{Marker, TScalarStyle};

/// A value of a view definition, as PyYAML's `safe_load` makes it.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer; PyYAML's are unbounded, and one past `i128` is refused.
    Int(i128),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Date(Date),
    Timestamp(Timestamp),
    Seq(Vec<Value>),
    /// The entries in the order their keys first appear.
    Map(Vec<(Value, Value)>),
}

/// A calendar date, as `!!timestamp` gives one with no time of day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Date {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// A date and time of day, with the offset from UTC it was written with,
/// in seconds, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timestamp {
    pub date: Date,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub microsecond: u32,
    pub offset: Option<i32>,
}

/// Why a text is no YAML value PyYAML would give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YamlError {
    pub reason: String,
}

impl fmt::Display for YamlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for YamlError {}

fn error(reason: impl Into<String>) -> YamlError {
    YamlError {
        reason: reason.into(),
    }
}

/// The one document `text` holds, as `safe_load` reads it: null for no
/// document, refused for more than one.
pub fn load(text: &str) -> Result<Value, YamlError> {
    // PyYAML's reader refuses a character outside its printable set before
    // anything is parsed (`yaml/reader.py`, `NON_PRINTABLE`).
    if let Some(c) = text.chars().find(|&c| !readable(c)) {
        return Err(error(format!(
            "the text holds the unacceptable character U+{:04X}",
            u32::from(c)
        )));
    }
    let mut composer = Composer::default();
    Parser::new_from_str(text)
        .load(&mut composer, true)
        .map_err(|scan| error(scan.to_string()))?;
    if let Some(reason) = composer.error {
        return Err(error(reason));
    }
    let mut documents = composer.documents;
    match documents.len() {
        0 => Ok(Value::Null),
        1 => construct(documents.remove(0)),
        _ => Err(error("the text holds more than one document")),
    }
}

/// `yaml/reader.py`'s printable set: tab, line feed, carriage return, the
/// printable ASCII, NEL, and the rest of Unicode but surrogates, U+FFFE and
/// U+FFFF.
fn readable(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | ' '..='~' | '\u{85}' | '\u{A0}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}')
        || u32::from(c) >= 0x10000
}

const CORE: &str = "tag:yaml.org,2002:";

/// A node of the document, its tag resolved, before any value is made.
#[derive(Clone, Debug)]
enum Node {
    Scalar {
        text: String,
        tag: String,
    },
    Seq {
        tag: String,
        items: Vec<Node>,
    },
    Map {
        tag: String,
        pairs: Vec<(Node, Node)>,
    },
}

enum Frame {
    Seq {
        anchor: usize,
        tag: String,
        items: Vec<Node>,
    },
    Map {
        anchor: usize,
        tag: String,
        pairs: Vec<(Node, Node)>,
        key: Option<Node>,
    },
}

/// Builds the nodes of each document from the parser's events, an alias
/// standing for a copy of its anchor's node.
#[derive(Default)]
struct Composer {
    stack: Vec<Frame>,
    anchors: BTreeMap<usize, Node>,
    documents: Vec<Node>,
    error: Option<String>,
}

impl Composer {
    fn add(&mut self, node: Node) {
        match self.stack.last_mut() {
            None => self.documents.push(node),
            Some(Frame::Seq { items, .. }) => items.push(node),
            Some(Frame::Map { pairs, key, .. }) => match key.take() {
                None => *key = Some(node),
                Some(k) => pairs.push((k, node)),
            },
        }
    }

    fn anchored(&mut self, anchor: usize, node: Node) {
        if anchor > 0 {
            self.anchors.insert(anchor, node.clone());
        }
        self.add(node);
    }
}

impl MarkedEventReceiver for Composer {
    fn on_event(&mut self, event: Event, _mark: Marker) {
        if self.error.is_some() {
            return;
        }
        match event {
            Event::Scalar(text, style, anchor, tag) => {
                let tag = match explicit(tag.as_ref()) {
                    Some(Some(tag)) => tag,
                    Some(None) => format!("{CORE}str"),
                    None if style == TScalarStyle::Plain => implicit(&text).to_owned(),
                    None => format!("{CORE}str"),
                };
                self.anchored(anchor, Node::Scalar { text, tag });
            }
            Event::SequenceStart(anchor, tag) => self.stack.push(Frame::Seq {
                anchor,
                tag: explicit(tag.as_ref())
                    .flatten()
                    .unwrap_or_else(|| format!("{CORE}seq")),
                items: Vec::new(),
            }),
            Event::MappingStart(anchor, tag) => self.stack.push(Frame::Map {
                anchor,
                tag: explicit(tag.as_ref())
                    .flatten()
                    .unwrap_or_else(|| format!("{CORE}map")),
                pairs: Vec::new(),
                key: None,
            }),
            Event::SequenceEnd | Event::MappingEnd => match self.stack.pop() {
                Some(Frame::Seq { anchor, tag, items }) => {
                    self.anchored(anchor, Node::Seq { tag, items });
                }
                Some(Frame::Map {
                    anchor, tag, pairs, ..
                }) => self.anchored(anchor, Node::Map { tag, pairs }),
                None => self.error = Some("a collection ends that never started".into()),
            },
            Event::Alias(anchor) => match self.anchors.get(&anchor).cloned() {
                Some(node) => self.add(node),
                None => self.error = Some("an alias names no anchor defined before it".into()),
            },
            Event::Nothing
            | Event::StreamStart
            | Event::StreamEnd
            | Event::DocumentStart
            | Event::DocumentEnd => {}
        }
    }
}

/// The tag written on a node: `None` if none, `Some(None)` for the
/// non-specific `!`, which PyYAML reads as the node's kind's own tag.
fn explicit(tag: Option<&Tag>) -> Option<Option<String>> {
    let tag = tag?;
    let full = format!("{}{}", tag.handle, tag.suffix);
    Some((full != "!").then_some(full))
}

/// PyYAML's implicit resolvers (`yaml/resolver.py`), each tried in the
/// order it was added, among those registered for the scalar's first
/// character.
fn implicit(text: &str) -> &'static str {
    static RESOLVERS: LazyLock<Vec<(&'static str, &'static str, Regex)>> = LazyLock::new(|| {
        [
            (
                "tag:yaml.org,2002:bool",
                "yYnNtTfFoO",
                r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$",
            ),
            (
                "tag:yaml.org,2002:float",
                "-+0123456789.",
                r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
            ),
            (
                "tag:yaml.org,2002:int",
                "-+0123456789",
                r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
            ),
            ("tag:yaml.org,2002:merge", "<", r"^(?:<<)$"),
            ("tag:yaml.org,2002:null", "~nN", r"^(?:~|null|Null|NULL|)$"),
            (
                "tag:yaml.org,2002:timestamp",
                "0123456789",
                r"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$",
            ),
            ("tag:yaml.org,2002:value", "=", r"^(?:=)$"),
        ]
        .into_iter()
        .map(|(tag, first, pattern)| (tag, first, Regex::new(pattern).unwrap()))
        .collect()
    });
    // The empty scalar is registered for the null resolver alone.
    let Some(first) = text.chars().next() else {
        return "tag:yaml.org,2002:null";
    };
    RESOLVERS
        .iter()
        .find(|(_, firsts, pattern)| firsts.contains(first) && pattern.is_match(text))
        .map_or("tag:yaml.org,2002:str", |(tag, _, _)| tag)
}

/// The value of `node`, by its tag, as PyYAML's `SafeConstructor` makes it.
fn construct(node: Node) -> Result<Value, YamlError> {
    let tag = match &node {
        Node::Scalar { tag, .. } | Node::Seq { tag, .. } | Node::Map { tag, .. } => tag.clone(),
    };
    let kind = tag.strip_prefix(CORE).unwrap_or("");
    match (kind, node) {
        ("seq", Node::Seq { items, .. }) => Ok(Value::Seq(
            items.into_iter().map(construct).collect::<Result<_, _>>()?,
        )),
        ("map", Node::Map { pairs, .. }) => mapping(pairs),
        ("null", Node::Scalar { .. }) => Ok(Value::Null),
        ("str", Node::Scalar { text, .. }) => Ok(Value::Str(text)),
        ("bool", Node::Scalar { text, .. }) => match text.to_lowercase().as_str() {
            "yes" | "true" | "on" => Ok(Value::Bool(true)),
            "no" | "false" | "off" => Ok(Value::Bool(false)),
            _ => Err(error(format!("'{text}' is no boolean"))),
        },
        ("int", Node::Scalar { text, .. }) => integer(&text).map(Value::Int),
        ("float", Node::Scalar { text, .. }) => float(&text).map(Value::Float),
        ("timestamp", Node::Scalar { text, .. }) => timestamp(&text),
        ("binary", Node::Scalar { text, .. }) => binary(&text).map(Value::Bytes),
        _ => Err(error(format!(
            "no value can be made for the tag '{tag}' here"
        ))),
    }
}

/// `construct_mapping` after `flatten_mapping`: the entries `<<` merges
/// come first, the later of two merged in first, then the node's own;
/// a key equal to an earlier one keeps that one's place and takes the
/// later value. A `=` key is text.
fn mapping(pairs: Vec<(Node, Node)>) -> Result<Value, YamlError> {
    let mut entries: Vec<(Value, Value)> = Vec::new();
    for (key, entry) in flatten(pairs)? {
        let key = match key {
            Node::Scalar { text, tag } if tag == format!("{CORE}value") => Value::Str(text),
            Node::Scalar { .. } => construct(key)?,
            Node::Seq { .. } | Node::Map { .. } => {
                return Err(error("a mapping has a key that is a collection"));
            }
        };
        let entry = construct(entry)?;
        match entries.iter_mut().find(|(k, _)| same_key(k, &key)) {
            Some((_, value)) => *value = entry,
            None => entries.push((key, entry)),
        }
    }
    Ok(Value::Map(entries))
}

fn flatten(pairs: Vec<(Node, Node)>) -> Result<Vec<(Node, Node)>, YamlError> {
    let merge_tag = format!("{CORE}merge");
    let (mut merged, mut own) = (Vec::new(), Vec::new());
    for (key, entry) in pairs {
        if matches!(&key, Node::Scalar { tag, .. } if *tag == merge_tag) {
            match entry {
                Node::Map { pairs, .. } => merged.extend(flatten(pairs)?),
                Node::Seq { items, .. } => {
                    let mut each = Vec::new();
                    for item in items {
                        let Node::Map { pairs, .. } = item else {
                            return Err(error("a merge lists something other than a mapping"));
                        };
                        each.push(flatten(pairs)?);
                    }
                    for pairs in each.into_iter().rev() {
                        merged.extend(pairs);
                    }
                }
                Node::Scalar { .. } => {
                    return Err(error("a merge names something other than a mapping"));
                }
            }
        } else {
            own.push((key, entry));
        }
    }
    merged.extend(own);
    Ok(merged)
}

/// Whether two keys are one, as Python compares them: numbers by value
/// (a boolean is 0 or 1), a date never equal to a date and time.
fn same_key(a: &Value, b: &Value) -> bool {
    fn number(value: &Value) -> Option<f64> {
        match value {
            Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
            #[allow(clippy::cast_precision_loss)]
            Value::Int(n) => Some(*n as f64),
            Value::Float(x) => Some(*x),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(_) | Value::Int(_), Value::Bool(_) | Value::Int(_)) => {
            let int = |v: &Value| match v {
                Value::Bool(flag) => i128::from(*flag),
                Value::Int(n) => *n,
                _ => unreachable!(),
            };
            int(a) == int(b)
        }
        _ => match (number(a), number(b)) {
            (Some(x), Some(y)) => x == y,
            (None, None) => a == b,
            _ => false,
        },
    }
}

/// `construct_yaml_int`: `_` dropped, a sign, then `0`, `0b` binary, `0x`
/// hexadecimal, a leading `0` octal, `:` base 60, else decimal.
fn integer(text: &str) -> Result<i128, YamlError> {
    let value = text.replace('_', "");
    let refused = || error(format!("'{text}' is no integer"));
    let (sign, digits) = match value.as_bytes().first() {
        Some(b'-') => (-1, &value[1..]),
        Some(b'+') => (1, &value[1..]),
        Some(_) => (1, value.as_str()),
        None => return Err(refused()),
    };
    let magnitude = if digits == "0" {
        Some(0)
    } else if let Some(rest) = digits.strip_prefix("0b") {
        radix(rest, 2)
    } else if let Some(rest) = digits.strip_prefix("0x") {
        radix(rest, 16)
    } else if digits.starts_with('0') {
        radix(digits, 8)
    } else if digits.contains(':') {
        digits
            .split(':')
            .rev()
            .try_fold((0i128, 1i128), |(sum, base), part| {
                let part = radix(part, 10)?;
                Some((
                    sum.checked_add(part.checked_mul(base)?)?,
                    base.checked_mul(60)?,
                ))
            })
            .map(|(sum, _)| sum)
    } else {
        radix(digits, 10)
    };
    magnitude.map(|n| sign * n).ok_or_else(refused)
}

/// Python's `int(text, base)` for what reaches it here: an optional sign,
/// then digits of the base, surrounding white space allowed.
fn radix(text: &str, base: u32) -> Option<i128> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: i128 = 0;
    for c in digits.chars() {
        value = value
            .checked_mul(i128::from(base))?
            .checked_add(i128::from(c.to_digit(base)?))?;
    }
    Some(if negative { -value } else { value })
}

/// `construct_yaml_float`: `_` dropped, lower case, a sign, then `.inf`,
/// `.nan`, `:` base 60, else Python's `float`.
fn float(text: &str) -> Result<f64, YamlError> {
    let value = text.replace('_', "").to_lowercase();
    let refused = || error(format!("'{text}' is no number"));
    let (sign, rest) = match value.as_bytes().first() {
        Some(b'-') => (-1.0, &value[1..]),
        Some(b'+') => (1.0, &value[1..]),
        Some(_) => (1.0, value.as_str()),
        None => return Err(refused()),
    };
    let parse = |part: &str| -> Result<f64, YamlError> {
        let part = part.trim();
        let plain = part.trim_start_matches(['+', '-']);
        // Rust also reads `infinity` and `nan`, as Python does; neither
        // reads anything else that is no decimal.
        if plain.is_empty() || plain.contains(['x', 'p', '_']) {
            return Err(refused());
        }
        part.parse::<f64>().map_err(|_| refused())
    };
    match rest {
        ".inf" => Ok(sign * f64::INFINITY),
        ".nan" => Ok(f64::NAN),
        _ if rest.contains(':') => {
            let (mut sum, mut base) = (0.0, 1.0);
            for part in rest.split(':').rev() {
                sum += parse(part)? * base;
                base *= 60.0;
            }
            Ok(sign * sum)
        }
        _ => Ok(sign * parse(rest)?),
    }
}

/// `construct_yaml_timestamp`: a date alone, or a date and time with a
/// fraction cut to microseconds and an offset or `Z`.
fn timestamp(text: &str) -> Result<Value, YamlError> {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"^(?P<year>[0-9][0-9][0-9][0-9])-(?P<month>[0-9][0-9]?)-(?P<day>[0-9][0-9]?)(?:(?:[Tt]|[ \t]+)(?P<hour>[0-9][0-9]?):(?P<minute>[0-9][0-9]):(?P<second>[0-9][0-9])(?:\.(?P<fraction>[0-9]*))?(?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)(?::(?P<tz_minute>[0-9][0-9]))?))?)?$",
        )
        .unwrap()
    });
    let refused = || error(format!("'{text}' is no date or time"));
    let groups = PATTERN.captures(text).ok_or_else(refused)?;
    let number =
        |name: &str| -> u32 { groups.name(name).map_or(0, |m| m.as_str().parse().unwrap()) };
    let date = Date {
        year: i32::try_from(number("year")).map_err(|_| refused())?,
        month: u8::try_from(number("month")).map_err(|_| refused())?,
        day: u8::try_from(number("day")).map_err(|_| refused())?,
    };
    if date.year < 1 || !(1..=12).contains(&date.month) || date.day < 1 || date.day > days_in(date)
    {
        return Err(refused());
    }
    if groups.name("hour").is_none() {
        return Ok(Value::Date(date));
    }
    let microsecond = match groups.name("fraction").map(|m| m.as_str()) {
        Some(fraction) if !fraction.is_empty() => {
            let mut digits: String = fraction.chars().take(6).collect();
            while digits.len() < 6 {
                digits.push('0');
            }
            digits.parse().unwrap()
        }
        _ => 0,
    };
    let offset = match groups.name("tz_sign").map(|m| m.as_str()) {
        Some(sign) => {
            let seconds = i32::try_from(number("tz_hour") * 3600 + number("tz_minute") * 60)
                .map_err(|_| refused())?;
            if seconds >= 86_400 {
                return Err(refused());
            }
            Some(if sign == "-" { -seconds } else { seconds })
        }
        None => groups.name("tz").map(|_| 0),
    };
    let (hour, minute, second) = (number("hour"), number("minute"), number("second"));
    if hour > 23 || minute > 59 || second > 59 {
        return Err(refused());
    }
    Ok(Value::Timestamp(Timestamp {
        date,
        hour: u8::try_from(hour).unwrap(),
        minute: u8::try_from(minute).unwrap(),
        second: u8::try_from(second).unwrap(),
        microsecond,
        offset,
    }))
}

fn days_in(date: Date) -> u8 {
    let leap = date.year % 4 == 0 && (date.year % 100 != 0 || date.year % 400 == 0);
    match date.month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// `construct_yaml_binary`: base64, as `base64.decodebytes` reads it,
/// characters outside the alphabet skipped.
fn binary(text: &str) -> Result<Vec<u8>, YamlError> {
    let refused = || error("the binary value is not base64");
    if !text.is_ascii() {
        return Err(refused());
    }
    let mut bits: u32 = 0;
    let (mut count, mut out, mut padding) = (0u32, Vec::new(), 0usize);
    for c in text.bytes() {
        let sextet = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                padding += 1;
                continue;
            }
            _ => continue,
        };
        if padding > 0 {
            return Err(refused());
        }
        bits = (bits << 6) | u32::from(sextet);
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push(u8::try_from((bits >> count) & 0xFF).unwrap());
        }
    }
    let leftover = (out.len() * 8 + count as usize) / 6 % 4;
    if leftover == 1 || (leftover != 0 && leftover + padding < 4) {
        return Err(refused());
    }
    Ok(out)
}

impl Date {
    /// The ISO 8601 text: `2024-01-01`.
    pub fn isoformat(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl Timestamp {
    /// The ISO 8601 text: `2024-01-01T10:00:00`, with `.ffffff` for a
    /// fraction and `+HH:MM` for an offset.
    pub fn isoformat(&self) -> String {
        let mut text = format!(
            "{}T{:02}:{:02}:{:02}",
            self.date.isoformat(),
            self.hour,
            self.minute,
            self.second
        );
        if self.microsecond > 0 {
            text.push_str(&format!(".{:06}", self.microsecond));
        }
        if let Some(offset) = self.offset {
            let sign = if offset < 0 { '-' } else { '+' };
            let seconds = offset.unsigned_abs();
            text.push_str(&format!(
                "{sign}{:02}:{:02}",
                seconds / 3600,
                seconds / 60 % 60
            ));
            if seconds % 60 > 0 {
                text.push_str(&format!(":{:02}", seconds % 60));
            }
        }
        text
    }
}

impl Value {
    /// The entry of a mapping whose key is the text `key`, as
    /// `dict.get(key)` finds it; `None` for no such entry or no mapping.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(pairs) => pairs
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(text) if text == key))
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// Whether a mapping has the entry `key` (`key in cfg`).
    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// The text, for a text value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// Whether Python reads the value as true in a condition: no null,
    /// `false`, zero, empty text or empty collection.
    pub fn is_set(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(flag) => *flag,
            Value::Int(n) => *n != 0,
            Value::Float(x) => *x != 0.0,
            Value::Str(text) => !text.is_empty(),
            Value::Bytes(bytes) => !bytes.is_empty(),
            Value::Seq(items) => !items.is_empty(),
            Value::Map(pairs) => !pairs.is_empty(),
            Value::Date(_) | Value::Timestamp(_) => true,
        }
    }
}
