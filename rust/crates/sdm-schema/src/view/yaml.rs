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
//! float, `2024-01-01` a date; a quoted scalar is always text. Integers are
//! unbounded. `!!set`, `!!omap`, `!!pairs` and `!!binary` are read as
//! PyYAML reads them. A mapping keeps its first key's place and its last
//! value, keys equal as Python compares them (`1`, `1.0` and `true` are one
//! key, as are two times of one instant), and `<<` merges.
//!
//! The values live in a [`Document`], each node once: an alias is the
//! node its anchor names, so a node aliased many times is held once, and a
//! collection may hold itself, as PyYAML's may.
//!
//! A value is data only: the texts the old client's messages quote it by,
//! Python's `repr` and `str` of it, are the core's to write
//! (`sdm-core::view_words`, through `sdm-pql`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use yaml_rust2::parser::{Event, MarkedEventReceiver, Parser, Tag};
use yaml_rust2::scanner::{Marker, Scanner, TScalarStyle, TokenType};

/// A node of a [`Document`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

/// A value of a view definition, as PyYAML's `safe_load` makes it; a
/// collection holds its members as nodes of its document.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(Integer),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Date(Date),
    Timestamp(Timestamp),
    Seq(Vec<NodeId>),
    /// The entries in the order their keys first appear.
    Map(Vec<(NodeId, NodeId)>),
    /// A `!!set`: its members in the order they are written.
    Set(Vec<NodeId>),
    /// A `!!omap` or `!!pairs`: a list of pairs.
    Pairs(Vec<(NodeId, NodeId)>),
}

/// An integer of any size, as its decimal digits, a `-` before them if it
/// is negative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Integer(String);

impl Integer {
    /// Its decimal text.
    pub fn text(&self) -> &str {
        &self.0
    }

    /// Its value, if it fits.
    pub fn to_i128(&self) -> Option<i128> {
        self.0.parse().ok()
    }

    fn from_i128(n: i128) -> Integer {
        Integer(n.to_string())
    }

    fn negative(&self) -> bool {
        self.0.starts_with('-')
    }
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

/// A loaded document: its nodes, and the one its root is.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    values: Vec<Value>,
    root: NodeId,
}

impl Document {
    /// The root's node.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The value of `node`.
    pub fn value(&self, node: NodeId) -> &Value {
        &self.values[node.0]
    }

    /// The entry of a mapping whose key is the text `key`, as
    /// `dict.get(key)` finds it; `None` for no such entry or no mapping.
    pub fn get(&self, node: NodeId, key: &str) -> Option<NodeId> {
        match self.value(node) {
            Value::Map(pairs) => pairs
                .iter()
                .find(|(k, _)| matches!(self.value(*k), Value::Str(text) if text == key))
                .map(|(_, value)| *value),
            _ => None,
        }
    }

    /// The entry of a mapping whose key Python finds equal to `True` (a
    /// bare `on:` key, or `1`).
    pub fn get_true(&self, node: NodeId) -> Option<NodeId> {
        match self.value(node) {
            Value::Map(pairs) => pairs
                .iter()
                .find(|(k, _)| match self.value(*k) {
                    Value::Bool(flag) => *flag,
                    Value::Int(n) => n.text() == "1",
                    Value::Float(x) => *x == 1.0,
                    _ => false,
                })
                .map(|(_, value)| *value),
            _ => None,
        }
    }

    /// Whether a mapping has the entry `key` (`key in cfg`).
    pub fn has(&self, node: NodeId, key: &str) -> bool {
        self.get(node, key).is_some()
    }

    /// The text, for a text value.
    pub fn text(&self, node: NodeId) -> Option<&str> {
        match self.value(node) {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// Whether the value is null.
    pub fn is_null(&self, node: NodeId) -> bool {
        matches!(self.value(node), Value::Null)
    }

    /// Whether Python reads the value as true in a condition: no null,
    /// `false`, zero, empty text or empty collection.
    pub fn is_set(&self, node: NodeId) -> bool {
        match self.value(node) {
            Value::Null => false,
            Value::Bool(flag) => *flag,
            Value::Int(n) => n.text() != "0",
            Value::Float(x) => *x != 0.0,
            Value::Str(text) => !text.is_empty(),
            Value::Bytes(bytes) => !bytes.is_empty(),
            Value::Seq(items) | Value::Set(items) => !items.is_empty(),
            Value::Map(pairs) | Value::Pairs(pairs) => !pairs.is_empty(),
            Value::Date(_) | Value::Timestamp(_) => true,
        }
    }

    /// The members `for x in value` visits: a list's items, a mapping's or
    /// a set's keys, a list of pairs' pairs, a text's characters, a byte
    /// string's numbers; `None` for a value that cannot be iterated. The
    /// characters and numbers are new values, which `iterate` adds.
    pub fn iterate(&mut self, node: NodeId) -> Option<Vec<NodeId>> {
        match self.value(node).clone() {
            Value::Seq(items) | Value::Set(items) => Some(items),
            Value::Map(pairs) => Some(pairs.iter().map(|(key, _)| *key).collect()),
            Value::Pairs(pairs) => Some(
                pairs
                    .into_iter()
                    .map(|(key, value)| self.push(Value::Pairs(vec![(key, value)])))
                    .collect(),
            ),
            Value::Str(text) => Some(
                text.chars()
                    .map(|c| self.push(Value::Str(c.to_string())))
                    .collect(),
            ),
            Value::Bytes(bytes) => Some(
                bytes
                    .into_iter()
                    .map(|b| self.push(Value::Int(Integer::from_i128(i128::from(b)))))
                    .collect(),
            ),
            _ => None,
        }
    }

    fn push(&mut self, value: Value) -> NodeId {
        self.values.push(value);
        NodeId(self.values.len() - 1)
    }
}

/// The one document `text` holds, as `safe_load` reads it: null for no
/// document, refused for more than one.
pub fn load(text: &str) -> Result<Document, YamlError> {
    // PyYAML skips a byte order mark at the very start.
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    // PyYAML's reader refuses a character outside its printable set before
    // anything is parsed (`yaml/reader.py`, `NON_PRINTABLE`).
    if let Some(c) = text.chars().find(|&c| !readable(c)) {
        return Err(error(format!(
            "the text holds the unacceptable character U+{:04X}",
            u32::from(c)
        )));
    }
    refuse_duplicate_anchors(text)?;
    // `yaml-rust2` reads a block scalar ending the text without a line
    // break as if it had one, more than one where a line of spaces ends it;
    // read with the break, then the clipped break is taken back.
    let broken;
    let read = if text.is_empty() || text.ends_with(['\n', '\r']) {
        text
    } else {
        broken = format!("{text}\n");
        &broken
    };
    let mut composer = Composer::default();
    Parser::new_from_str(read)
        .load(&mut composer, true)
        .map_err(|scan| error(scan.to_string()))?;
    if let Some(reason) = composer.error {
        return Err(error(reason));
    }
    composer.chomp_last_block(text);
    let root = match composer.documents.as_slice() {
        [] => None,
        [root] => Some(*root),
        _ => return Err(error("the text holds more than one document")),
    };
    Constructor::new(composer.nodes).document(root)
}

/// `yaml/reader.py`'s printable set: tab, line feed, carriage return, the
/// printable ASCII, NEL, and the rest of Unicode but surrogates, U+FFFE and
/// U+FFFF.
fn readable(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | ' '..='~' | '\u{85}' | '\u{A0}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}')
        || u32::from(c) >= 0x10000
}

/// PyYAML's composer refuses an anchor named twice in a document, where
/// `yaml-rust2` lets the second replace the first.
fn refuse_duplicate_anchors(text: &str) -> Result<(), YamlError> {
    let mut named: BTreeSet<String> = BTreeSet::new();
    for token in Scanner::new(text.chars()) {
        match token.1 {
            TokenType::DocumentStart | TokenType::DocumentEnd => named.clear(),
            TokenType::Anchor(name) if !named.insert(name.clone()) => {
                return Err(error(format!("the anchor '{name}' is named twice")));
            }
            _ => {}
        }
    }
    Ok(())
}

const CORE: &str = "tag:yaml.org,2002:";

/// A node as composed, its tag resolved, before any value is made.
#[derive(Clone, Debug)]
enum Raw {
    Scalar {
        text: String,
        tag: String,
    },
    Seq {
        tag: String,
        items: Vec<usize>,
    },
    Map {
        tag: String,
        pairs: Vec<(usize, usize)>,
    },
}

enum Frame {
    Seq(usize),
    Map { node: usize, key: Option<usize> },
}

/// Builds each document's nodes from the parser's events, once each: an
/// anchor names its node from its start, so an alias inside it is the node
/// itself.
#[derive(Default)]
struct Composer {
    nodes: Vec<Raw>,
    stack: Vec<Frame>,
    anchors: BTreeMap<usize, usize>,
    documents: Vec<usize>,
    error: Option<String>,
    /// The last scalar's node and whether it was a block scalar.
    last_scalar: Option<(usize, bool)>,
}

impl Composer {
    fn add(&mut self, node: usize) {
        match self.stack.last_mut() {
            None => self.documents.push(node),
            Some(Frame::Seq(seq)) => {
                if let Raw::Seq { items, .. } = &mut self.nodes[*seq] {
                    items.push(node);
                }
            }
            Some(Frame::Map { node: map, key }) => match key.take() {
                None => *key = Some(node),
                Some(k) => {
                    if let Raw::Map { pairs, .. } = &mut self.nodes[*map] {
                        pairs.push((k, node));
                    }
                }
            },
        }
    }

    fn open(&mut self, anchor: usize, raw: Raw) -> usize {
        self.nodes.push(raw);
        let node = self.nodes.len() - 1;
        if anchor > 0 {
            self.anchors.insert(anchor, node);
        }
        node
    }

    /// `yaml-rust2` ends a block scalar with clipped chomping that ends the
    /// text with a line break the text does not have; PyYAML keeps only a
    /// break the text holds.
    fn chomp_last_block(&mut self, text: &str) {
        let Some((node, true)) = self.last_scalar else {
            return;
        };
        if text.ends_with(['\n', '\r']) {
            return;
        }
        // Only a last line the block holds ends it without its break: a
        // comment after the block, or a line of spaces, does not.
        let last_line = text.rsplit(['\n', '\r']).next().unwrap_or("");
        let content = last_line.trim();
        if content.is_empty() || content.starts_with('#') {
            return;
        }
        if let Raw::Scalar { text: value, .. } = &mut self.nodes[node]
            && value.ends_with('\n')
        {
            value.pop();
        }
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
                let block = matches!(style, TScalarStyle::Literal | TScalarStyle::Folded);
                let node = self.open(anchor, Raw::Scalar { text, tag });
                self.last_scalar = Some((node, block));
                self.add(node);
            }
            Event::SequenceStart(anchor, tag) => {
                self.last_scalar = None;
                let tag = explicit(tag.as_ref())
                    .flatten()
                    .unwrap_or_else(|| format!("{CORE}seq"));
                let node = self.open(
                    anchor,
                    Raw::Seq {
                        tag,
                        items: Vec::new(),
                    },
                );
                self.stack.push(Frame::Seq(node));
            }
            Event::MappingStart(anchor, tag) => {
                self.last_scalar = None;
                let tag = explicit(tag.as_ref())
                    .flatten()
                    .unwrap_or_else(|| format!("{CORE}map"));
                let node = self.open(
                    anchor,
                    Raw::Map {
                        tag,
                        pairs: Vec::new(),
                    },
                );
                self.stack.push(Frame::Map { node, key: None });
            }
            Event::SequenceEnd | Event::MappingEnd => match self.stack.pop() {
                Some(Frame::Seq(node) | Frame::Map { node, .. }) => self.add(node),
                None => self.error = Some("a collection ends that never started".into()),
            },
            Event::Alias(anchor) => match self.anchors.get(&anchor).copied() {
                Some(node) => {
                    self.last_scalar = None;
                    self.add(node);
                }
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

/// Makes each node's value, once, as PyYAML's `SafeConstructor` does; the
/// values keep the nodes' places, so a collection names its members by
/// node.
struct Constructor {
    raw: Vec<Raw>,
    values: Vec<Option<Value>>,
    /// The nodes made from `.nan`, which PyYAML makes one float object, so
    /// each is the same key as any other.
    nans: BTreeSet<usize>,
    /// Each mapping's entries with its merges flattened in.
    flattened: Vec<Option<Vec<(usize, usize)>>>,
    flattening: BTreeSet<usize>,
}

impl Constructor {
    fn new(raw: Vec<Raw>) -> Constructor {
        let count = raw.len();
        Constructor {
            raw,
            values: vec![None; count],
            nans: BTreeSet::new(),
            flattened: vec![None; count],
            flattening: BTreeSet::new(),
        }
    }

    fn document(mut self, root: Option<usize>) -> Result<Document, YamlError> {
        let Some(root) = root else {
            return Ok(Document {
                values: vec![Value::Null],
                root: NodeId(0),
            });
        };
        // Each node reachable from the root is made, once: a key or a
        // merge PyYAML reads as such is made by its mapping.
        let mut stack = vec![root];
        let mut seen = vec![false; self.raw.len()];
        while let Some(node) = stack.pop() {
            if std::mem::replace(&mut seen[node], true) {
                continue;
            }
            self.make(node)?;
            match self.values[node].as_ref() {
                Some(Value::Seq(items) | Value::Set(items)) => {
                    stack.extend(items.iter().map(|n| n.0));
                }
                Some(Value::Map(pairs) | Value::Pairs(pairs)) => {
                    stack.extend(pairs.iter().flat_map(|(k, v)| [k.0, v.0]));
                }
                _ => {}
            }
        }
        Ok(Document {
            values: self
                .values
                .into_iter()
                .map(|v| v.unwrap_or(Value::Null))
                .collect(),
            root: NodeId(root),
        })
    }

    fn tag(&self, node: usize) -> &str {
        match &self.raw[node] {
            Raw::Scalar { tag, .. } | Raw::Seq { tag, .. } | Raw::Map { tag, .. } => tag,
        }
    }

    /// The value of `node`, made if it is not yet.
    fn make(&mut self, node: usize) -> Result<(), YamlError> {
        if self.values[node].is_some() {
            return Ok(());
        }
        let tag = self.tag(node).to_owned();
        let kind = tag.strip_prefix(CORE).unwrap_or("");
        let value = match (kind, self.raw[node].clone()) {
            ("seq", Raw::Seq { items, .. }) => Value::Seq(items.into_iter().map(NodeId).collect()),
            ("map", Raw::Map { .. }) => Value::Map(self.mapping(node)?),
            ("set", Raw::Map { .. }) => Value::Set(
                self.mapping(node)?
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect(),
            ),
            ("omap" | "pairs", Raw::Seq { items, .. }) => {
                let mut pairs = Vec::new();
                for item in items {
                    let Raw::Map { pairs: entry, .. } = &self.raw[item] else {
                        return Err(error(format!(
                            "an ordered map holds no mapping at node {item}"
                        )));
                    };
                    let [(key, value)] = entry.as_slice() else {
                        return Err(error("an ordered map's entry is not of one pair"));
                    };
                    pairs.push((NodeId(*key), NodeId(*value)));
                }
                Value::Pairs(pairs)
            }
            (_, Raw::Scalar { text, .. }) => self.scalar(node, kind, &tag, &text)?,
            _ => {
                return Err(error(format!(
                    "no value can be made for the tag '{tag}' here"
                )));
            }
        };
        self.values[node] = Some(value);
        Ok(())
    }

    fn scalar(
        &mut self,
        node: usize,
        kind: &str,
        tag: &str,
        text: &str,
    ) -> Result<Value, YamlError> {
        Ok(match kind {
            "null" => Value::Null,
            "str" => Value::Str(text.to_owned()),
            "bool" => match text.to_lowercase().as_str() {
                "yes" | "true" | "on" => Value::Bool(true),
                "no" | "false" | "off" => Value::Bool(false),
                _ => return Err(error(format!("'{text}' is no boolean"))),
            },
            "int" => Value::Int(integer(text)?),
            "float" => {
                let x = float(text)?;
                if text
                    .replace('_', "")
                    .to_lowercase()
                    .trim_start_matches(['+', '-'])
                    == ".nan"
                {
                    self.nans.insert(node);
                }
                Value::Float(x)
            }
            "timestamp" => timestamp(text)?,
            "binary" => Value::Bytes(binary(text)?),
            _ => {
                return Err(error(format!(
                    "no value can be made for the tag '{tag}' here"
                )));
            }
        })
    }

    /// `construct_mapping` after `flatten_mapping`: the entries `<<` merges
    /// come first, the later of two merged in first, then the node's own; a
    /// key equal to an earlier one keeps that one's place and takes the
    /// later value. A `=` key is text, and a key that is a collection is
    /// refused, as Python refuses an unhashable key.
    fn mapping(&mut self, node: usize) -> Result<Vec<(NodeId, NodeId)>, YamlError> {
        let pairs = self.flatten(node)?;
        let mut entries: Vec<(NodeId, NodeId)> = Vec::new();
        for (key, value) in pairs {
            if let Raw::Scalar { text, tag } = &self.raw[key]
                && *tag == format!("{CORE}value")
            {
                let text = text.clone();
                self.values[key] = Some(Value::Str(text));
            } else {
                self.make(key)?;
            }
            if matches!(self.raw[key], Raw::Seq { .. } | Raw::Map { .. }) {
                return Err(error("a mapping has a key that is a collection"));
            }
            match entries.iter_mut().find(|(k, _)| self.same_key(k.0, key)) {
                Some((_, slot)) => *slot = NodeId(value),
                None => entries.push((NodeId(key), NodeId(value))),
            }
        }
        Ok(entries)
    }

    fn flatten(&mut self, node: usize) -> Result<Vec<(usize, usize)>, YamlError> {
        if let Some(done) = &self.flattened[node] {
            return Ok(done.clone());
        }
        if self.flattening.contains(&node) {
            // A mapping merging itself: PyYAML has taken the merge out of
            // it before it reads it again, so it gives its other entries.
            let merge_tag = format!("{CORE}merge");
            let Raw::Map { pairs, .. } = &self.raw[node] else {
                unreachable!()
            };
            return Ok(pairs
                .iter()
                .filter(|(key, _)| !matches!(&self.raw[*key], Raw::Scalar { tag, .. } if *tag == merge_tag))
                .copied()
                .collect());
        }
        self.flattening.insert(node);
        let Raw::Map { pairs, .. } = self.raw[node].clone() else {
            return Err(error("a merge names something other than a mapping"));
        };
        let merge_tag = format!("{CORE}merge");
        let (mut merged, mut own) = (Vec::new(), Vec::new());
        for (key, value) in pairs {
            if !matches!(&self.raw[key], Raw::Scalar { tag, .. } if *tag == merge_tag) {
                own.push((key, value));
                continue;
            }
            match self.raw[value].clone() {
                Raw::Map { .. } => merged.extend(self.flatten(value)?),
                Raw::Seq { items, .. } => {
                    let mut each = Vec::new();
                    for item in items {
                        if !matches!(self.raw[item], Raw::Map { .. }) {
                            return Err(error("a merge lists something other than a mapping"));
                        }
                        each.push(self.flatten(item)?);
                    }
                    for pairs in each.into_iter().rev() {
                        merged.extend(pairs);
                    }
                }
                Raw::Scalar { .. } => {
                    return Err(error("a merge names something other than a mapping"));
                }
            }
        }
        merged.extend(own);
        self.flattening.remove(&node);
        self.flattened[node] = Some(merged.clone());
        Ok(merged)
    }

    /// Whether the keys at nodes `a` and `b` are one, as Python's `dict`
    /// finds them: the same node; numbers equal by value, exactly (a
    /// boolean is 0 or 1); a NaN from `.nan` the same as another; two times
    /// with offsets at one instant, two without equal in every field, never
    /// one of each; a date never a date and time.
    fn same_key(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        let (Some(x), Some(y)) = (&self.values[a], &self.values[b]) else {
            return false;
        };
        if let (Some(p), Some(q)) = (number(x), number(y)) {
            return match (p, q) {
                (Number::Float(p), Number::Float(q)) if p.is_nan() && q.is_nan() => {
                    self.nans.contains(&a) && self.nans.contains(&b)
                }
                (p, q) => numbers_equal(&p, &q),
            };
        }
        match (x, y) {
            (Value::Timestamp(p), Value::Timestamp(q)) => match (p.offset, q.offset) {
                (Some(_), Some(_)) => instant(p) == instant(q),
                (None, None) => p == q,
                _ => false,
            },
            _ => x == y,
        }
    }
}

/// A number as Python compares it.
enum Number {
    Int(String),
    Float(f64),
}

fn number(value: &Value) -> Option<Number> {
    match value {
        Value::Bool(flag) => Some(Number::Int(if *flag { "1" } else { "0" }.into())),
        Value::Int(n) => Some(Number::Int(n.text().to_owned())),
        Value::Float(x) => Some(Number::Float(*x)),
        _ => None,
    }
}

/// Python's exact comparison of two numbers: an integer equals a float
/// only if the float is that integer exactly.
fn numbers_equal(p: &Number, q: &Number) -> bool {
    let exact = |x: f64| (x.is_finite() && x.fract() == 0.0).then(|| format!("{:.0}", x));
    let integer = |text: &str| {
        if text == "-0" {
            "0".to_owned()
        } else {
            text.to_owned()
        }
    };
    match (p, q) {
        (Number::Int(a), Number::Int(b)) => a == b,
        (Number::Float(a), Number::Float(b)) => a == b,
        (Number::Int(a), Number::Float(b)) | (Number::Float(b), Number::Int(a)) => {
            exact(*b).is_some_and(|b| integer(&b) == *a)
        }
    }
}

/// A time with an offset as microseconds since 0001-01-01 00:00 UTC.
fn instant(moment: &Timestamp) -> i128 {
    let date = moment.date;
    let y = i128::from(date.year) - 1;
    let before: [i128; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = date.year % 4 == 0 && (date.year % 100 != 0 || date.year % 400 == 0);
    let mut days = y * 365 + y / 4 - y / 100
        + y / 400
        + before[usize::from(date.month - 1)]
        + i128::from(date.day);
    if date.month > 2 && leap {
        days += 1;
    }
    let seconds = days * 86_400
        + i128::from(moment.hour) * 3600
        + i128::from(moment.minute) * 60
        + i128::from(moment.second)
        - i128::from(moment.offset.unwrap_or(0));
    seconds * 1_000_000 + i128::from(moment.microsecond)
}

/// Python reads at most this many digits into an integer of base 10
/// (`sys.int_info.default_max_str_digits`).
const MOST_DIGITS: usize = 4300;

/// `construct_yaml_int`: `_` dropped, a sign, then `0`, `0b` binary, `0x`
/// hexadecimal, a leading `0` octal, `:` base 60, else decimal; of any size.
fn integer(text: &str) -> Result<Integer, YamlError> {
    let value = text.replace('_', "");
    let refused = || error(format!("'{text}' is no integer"));
    let (negative, digits) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        Some(_) => (false, value.as_str()),
        None => return Err(refused()),
    };
    let magnitude = if digits == "0" {
        Some(vec![0])
    } else if let Some(rest) = digits.strip_prefix("0b") {
        radix(rest, 2)
    } else if let Some(rest) = digits.strip_prefix("0x") {
        radix(rest, 16)
    } else if digits.starts_with('0') {
        radix(digits, 8)
    } else if digits.contains(':') {
        let mut sum = vec![0];
        for part in digits.split(':') {
            let part = radix(part, 10).ok_or_else(refused)?;
            sum = add(&times(&sum, 60), &part);
        }
        Some(sum)
    } else {
        if digits.trim().trim_start_matches(['+', '-']).len() > MOST_DIGITS {
            return Err(error(format!(
                "'{text}' has more digits than an integer of base 10 is read with"
            )));
        }
        radix(digits, 10)
    };
    let magnitude = magnitude.ok_or_else(refused)?;
    let mut decimal = decimal(&magnitude);
    if negative && decimal != "0" {
        decimal.insert(0, '-');
    }
    Ok(Integer(decimal))
}

/// Limbs of base one billion, least significant first.
const LIMB: u64 = 1_000_000_000;

fn times(limbs: &[u64], factor: u64) -> Vec<u64> {
    let mut carry = 0u64;
    let mut out: Vec<u64> = limbs
        .iter()
        .map(|limb| {
            let product = limb * factor + carry;
            carry = product / LIMB;
            product % LIMB
        })
        .collect();
    while carry > 0 {
        out.push(carry % LIMB);
        carry /= LIMB;
    }
    out
}

fn add(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut carry = 0;
    for i in 0..a.len().max(b.len()) {
        let sum = a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0) + carry;
        out.push(sum % LIMB);
        carry = sum / LIMB;
    }
    if carry > 0 {
        out.push(carry);
    }
    out
}

fn decimal(limbs: &[u64]) -> String {
    let mut limbs = limbs.to_vec();
    while limbs.len() > 1 && limbs.last() == Some(&0) {
        limbs.pop();
    }
    let mut out = limbs.last().unwrap().to_string();
    for limb in limbs.iter().rev().skip(1) {
        out.push_str(&format!("{limb:09}"));
    }
    out
}

/// Python's `int(text, base)` for what reaches it here, its sign taken
/// off: digits of the base, surrounding white space allowed.
fn radix(text: &str, base: u32) -> Option<Vec<u64>> {
    let digits = text.trim();
    if digits.is_empty() {
        return None;
    }
    let mut limbs = vec![0];
    for c in digits.chars() {
        limbs = add(
            &times(&limbs, u64::from(base)),
            &[u64::from(c.to_digit(base)?)],
        );
    }
    Some(limbs)
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
        part.trim().parse::<f64>().map_err(|_| refused())
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

/// `construct_yaml_binary`: base64 as Python's `base64.decodebytes` reads
/// it (`binascii.a2b_base64`, not strict): a character outside the
/// alphabet skipped, padding that completes a group ending the data, and a
/// group left open refused.
fn binary(text: &str) -> Result<Vec<u8>, YamlError> {
    if !text.is_ascii() {
        return Err(error("the binary value is not ASCII"));
    }
    let (mut out, mut quad, mut left, mut pads) = (Vec::new(), 0u8, 0u8, 0u8);
    for c in text.bytes() {
        if c == b'=' {
            pads += 1;
            if quad >= 2 && quad + pads >= 4 {
                return Ok(out);
            }
            continue;
        }
        let sextet = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        pads = 0;
        match quad {
            0 => {
                left = sextet;
                quad = 1;
            }
            1 => {
                out.push((left << 2) | (sextet >> 4));
                left = sextet & 0x0f;
                quad = 2;
            }
            2 => {
                out.push((left << 4) | (sextet >> 2));
                left = sextet & 0x03;
                quad = 3;
            }
            _ => {
                out.push((left << 6) | sextet);
                quad = 0;
            }
        }
    }
    match quad {
        0 => Ok(out),
        1 => Err(error("the binary value has one data character too many")),
        _ => Err(error("the binary value is padded wrongly")),
    }
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

impl Integer {
    /// Whether it is below zero.
    pub fn is_negative(&self) -> bool {
        self.negative()
    }
}
