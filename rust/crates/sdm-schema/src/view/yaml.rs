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
//! float, `2024-01-01` a date; a quoted scalar is always text. A block
//! scalar keeps the line breaks PyYAML keeps. A mapping keeps its first
//! key's place and its last value, keys equal as Python compares them
//! (`1`, `1.0` and `true` are one key, as are two times of one instant).
//!
//! Only the YAML view definitions use is read: scalars, block and flow
//! mappings and lists, block scalars, comments, and anchors and aliases as
//! plain references. Anything else is refused by name ([`Outside`],
//! register P3J-9): an explicit tag (`!!binary`, `!!set`, `!!omap`,
//! `!!pairs`, any other), a merge (`<<`), a key that is a collection (a
//! mapping keyed by itself among them), nesting deeper than [`MAX_DEPTH`],
//! aliases expanding past [`MAX_VALUES`] values, an integer of more digits
//! than Python writes, and the line breaks of YAML 1.1 alone.
//!
//! The values live in a [`Document`], each node once: an alias is the
//! node its anchor names, so a node aliased many times is held once, and a
//! collection may hold itself, as PyYAML's may. Reading is bounded: no
//! value is built past the text's own size, nesting, or Python's digits.
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

/// How deep collections may nest, counting the document's own and each
/// alias's: below the 255 flow levels `yaml-rust2` scans and the about
/// 490 levels at which PyYAML's composer meets Python's recursion limit.
pub const MAX_DEPTH: usize = 200;

/// How many values a document may hold with each alias read as a copy of
/// what it names, so that no walk of it is unbounded.
pub const MAX_VALUES: u64 = 1 << 22;

/// What reading takes from the host: the most decimal digits Python reads
/// into an integer (`sys.get_int_max_str_digits()`), which the core hands
/// in from `sdm-pql`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub most_digits: usize,
}

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
    Date(Date),
    Timestamp(Timestamp),
    Seq(Vec<NodeId>),
    /// The entries in the order their keys first appear.
    Map(Vec<(NodeId, NodeId)>),
}

/// An integer, as its decimal digits, a `-` before them if it is
/// negative; never more digits than [`Rules::most_digits`].
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

    /// Whether it is below zero.
    pub fn is_negative(&self) -> bool {
        self.0.starts_with('-')
    }
}

/// A calendar date, as `!!timestamp` gives one with no time of day.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// A date and time of day, with the offset from UTC it was written with,
/// in seconds, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub date: Date,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub microsecond: u32,
    pub offset: Option<i32>,
}

/// YAML outside what view definitions use, refused by name (register
/// P3J-9).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outside {
    /// An explicit tag, as written.
    Tag(String),
    /// A merge key, `<<`.
    Merge,
    /// A key that is a list or a mapping.
    CollectionKey,
    /// Collections nested deeper than [`MAX_DEPTH`].
    Nesting,
    /// Aliases that, each read as a copy, give more than [`MAX_VALUES`]
    /// values.
    Expansion,
    /// An integer of more decimal digits than Python writes.
    LongInteger,
    /// A line break of YAML 1.1 only (U+0085, U+2028, U+2029), which
    /// PyYAML reads as one and `yaml-rust2`, of YAML 1.2, does not.
    LineBreak,
}

impl fmt::Display for Outside {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outside::Tag(tag) => write!(f, "the explicit tag '{tag}'"),
            Outside::Merge => f.write_str("a merge key ('<<')"),
            Outside::CollectionKey => f.write_str("a key that is a list or a mapping"),
            Outside::Nesting => write!(f, "collections nested deeper than {MAX_DEPTH} levels"),
            Outside::Expansion => {
                write!(
                    f,
                    "aliases that, each read as a copy, give more than {MAX_VALUES} values"
                )
            }
            Outside::LongInteger => {
                f.write_str("an integer of more digits than an integer is written with")
            }
            Outside::LineBreak => f.write_str("a line break of U+0085, U+2028 or U+2029"),
        }
    }
}

/// Why a text is no YAML value PyYAML would give, or is YAML outside what
/// view definitions use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YamlError {
    pub reason: String,
    /// What the text holds outside the YAML view definitions use, if that
    /// is why.
    pub outside: Option<Outside>,
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
        outside: None,
    }
}

fn outside(what: Outside) -> YamlError {
    YamlError {
        reason: format!("the text holds {what}, which is outside the YAML view definitions use"),
        outside: Some(what),
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

    /// The entry of a mapping whose key Python finds equal to the number
    /// `n`, 0 or 1 (a key `0`, `0.0` or `false` is 0).
    fn get_number(&self, node: NodeId, n: u8) -> Option<NodeId> {
        let Value::Map(pairs) = self.value(node) else {
            return None;
        };
        pairs
            .iter()
            .find(|(k, _)| match self.value(*k) {
                Value::Bool(flag) => u8::from(*flag) == n,
                Value::Int(i) => i.text() == if n == 0 { "0" } else { "1" },
                Value::Float(x) => *x == f64::from(n),
                _ => false,
            })
            .map(|(_, value)| *value)
    }

    /// The entry of a mapping whose key Python finds equal to `True` (a
    /// bare `on:` key, or `1`).
    pub fn get_true(&self, node: NodeId) -> Option<NodeId> {
        self.get_number(node, 1)
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
            Value::Seq(items) => !items.is_empty(),
            Value::Map(pairs) => !pairs.is_empty(),
            Value::Date(_) | Value::Timestamp(_) => true,
        }
    }

    /// `len(value)`: a list's items, a mapping's entries, a text's
    /// characters; `None` for a value with no length.
    pub fn len(&self, node: NodeId) -> Option<usize> {
        match self.value(node) {
            Value::Seq(items) => Some(items.len()),
            Value::Map(pairs) => Some(pairs.len()),
            Value::Str(text) => Some(text.chars().count()),
            _ => None,
        }
    }

    /// `value[0]`: a list's first item, a mapping's entry keyed 0, a
    /// text's first character (a new value, which this adds); `None` where
    /// Python fails.
    pub fn first(&mut self, node: NodeId) -> Option<NodeId> {
        match self.value(node) {
            Value::Seq(items) => items.first().copied(),
            Value::Map(_) => self.get_number(node, 0),
            Value::Str(text) => {
                let c = text.chars().next()?;
                Some(self.push(Value::Str(c.to_string())))
            }
            _ => None,
        }
    }

    /// The members `for x in value` visits: a list's items, a mapping's
    /// keys, a text's characters; `None` for a value that cannot be
    /// iterated. The characters are new values, which `iterate` adds.
    pub fn iterate(&mut self, node: NodeId) -> Option<Vec<NodeId>> {
        match self.value(node) {
            Value::Seq(items) => Some(items.clone()),
            Value::Map(pairs) => Some(pairs.iter().map(|(key, _)| *key).collect()),
            Value::Str(text) => {
                let chars: Vec<char> = text.chars().collect();
                Some(
                    chars
                        .into_iter()
                        .map(|c| self.push(Value::Str(c.to_string())))
                        .collect(),
                )
            }
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
pub fn load(text: &str, rules: &Rules) -> Result<Document, YamlError> {
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
    if text.contains(['\u{85}', '\u{2028}', '\u{2029}']) {
        return Err(outside(Outside::LineBreak));
    }
    refuse_duplicate_anchors(text)?;
    // `yaml-rust2` reads a block scalar ending the text without a line
    // break as if it had one, more than one where a line of spaces ends it;
    // the text is read with the break, and each block scalar's line breaks
    // are then taken from the text itself, as PyYAML keeps them.
    let broken;
    let read = if text.is_empty() || text.ends_with(['\n', '\r']) {
        text
    } else {
        broken = format!("{text}\n");
        &broken
    };
    let mut composer = Composer::default();
    let parsed = Parser::new_from_str(read).load(&mut composer, true);
    // What the composer refused came before what the scanner did.
    if let Some(reason) = composer.error {
        return Err(reason);
    }
    parsed.map_err(|scan| {
        // `yaml-rust2` stops at 255 flow levels, past the bound.
        if scan.info() == "recursion limit exceeded" {
            outside(Outside::Nesting)
        } else {
            error(scan.to_string())
        }
    })?;
    composer.block_breaks(text);
    let root = match composer.documents.as_slice() {
        [] => None,
        [root] => Some(*root),
        _ => return Err(error("the text holds more than one document")),
    };
    construct(composer.nodes, root, rules)
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

/// A node as composed, its kind resolved, before any value is made.
#[derive(Clone, Debug)]
enum Raw {
    /// A scalar: its text, and its kind, `None` for quoted text.
    Scalar {
        text: String,
        kind: Option<&'static str>,
    },
    Seq(Vec<usize>),
    Map(Vec<(usize, usize)>),
}

enum Frame {
    Seq(usize),
    Map { node: usize, key: Option<usize> },
}

/// A block scalar composed: its node, and the character `yaml-rust2`
/// marks it at, its first content's or, with none, its indicator or the
/// line after.
#[derive(Clone, Copy)]
struct Block {
    node: usize,
    at: usize,
}

/// Builds each document's nodes from the parser's events, once each: an
/// anchor names its node from its start, so an alias inside it is the node
/// itself.
#[derive(Default)]
struct Composer {
    nodes: Vec<Raw>,
    /// Each node's depth: how many collections deep it reaches, an alias
    /// of a collection still open (the collection itself) counted as none;
    /// and its size, how many values it holds with each alias read as a
    /// copy, such an alias counted as one.
    depths: Vec<usize>,
    sizes: Vec<u64>,
    open: Vec<bool>,
    /// The open collections, each with the deepest member it has and the
    /// size of its members.
    stack: Vec<(Frame, usize, u64)>,
    anchors: BTreeMap<usize, usize>,
    documents: Vec<usize>,
    error: Option<YamlError>,
    blocks: Vec<Block>,
}

impl Composer {
    fn add(&mut self, node: usize) {
        let (depth, size) = if self.open[node] {
            (0, 1)
        } else {
            (self.depths[node], self.sizes[node])
        };
        let collection = matches!(self.nodes[node], Raw::Seq(_) | Raw::Map(_));
        let Some((frame, deepest, members)) = self.stack.last_mut() else {
            self.documents.push(node);
            return;
        };
        *deepest = (*deepest).max(depth);
        *members = members.saturating_add(size);
        match frame {
            Frame::Seq(seq) => {
                if let Raw::Seq(items) = &mut self.nodes[*seq] {
                    items.push(node);
                }
            }
            Frame::Map { node: map, key } => match key.take() {
                // PyYAML refuses a key that is a list or a mapping, as
                // Python refuses an unhashable key; it is refused here
                // before it is made, so a mapping keyed by itself is too.
                None if collection => self.error = Some(outside(Outside::CollectionKey)),
                None => *key = Some(node),
                Some(k) => {
                    if let Raw::Map(pairs) = &mut self.nodes[*map] {
                        pairs.push((k, node));
                    }
                }
            },
        }
    }

    fn open(&mut self, anchor: usize, raw: Raw) -> usize {
        let collection = matches!(raw, Raw::Seq(_) | Raw::Map(_));
        self.nodes.push(raw);
        self.depths.push(0);
        self.sizes.push(1);
        self.open.push(collection);
        let node = self.nodes.len() - 1;
        if anchor > 0 {
            self.anchors.insert(anchor, node);
        }
        node
    }

    fn start(&mut self, anchor: usize, raw: Raw) {
        if self.stack.len() >= MAX_DEPTH {
            self.error = Some(outside(Outside::Nesting));
            return;
        }
        let node = self.open(anchor, raw);
        let frame = match self.nodes[node] {
            Raw::Seq(_) => Frame::Seq(node),
            _ => Frame::Map { node, key: None },
        };
        self.stack.push((frame, 0, 0));
    }

    fn end(&mut self) {
        let Some((Frame::Seq(node) | Frame::Map { node, .. }, deepest, members)) = self.stack.pop()
        else {
            self.error = Some(error("a collection ends that never started"));
            return;
        };
        self.open[node] = false;
        self.depths[node] = deepest + 1;
        self.sizes[node] = members.saturating_add(1);
        if self.depths[node] > MAX_DEPTH {
            self.error = Some(outside(Outside::Nesting));
            return;
        }
        if self.sizes[node] > MAX_VALUES {
            self.error = Some(outside(Outside::Expansion));
            return;
        }
        self.add(node);
    }

    /// Each block scalar's line breaks as PyYAML keeps them, from the text:
    /// `yaml-rust2` gives a block with no content a line break, and the
    /// text was read with a break it may not end with.
    fn block_breaks(&mut self, text: &str) {
        for block in std::mem::take(&mut self.blocks) {
            let Raw::Scalar { text: value, .. } = &self.nodes[block.node] else {
                continue;
            };
            if let Some(kept) = block_value(text, block.at, value) {
                self.nodes[block.node] = Raw::Scalar {
                    text: kept,
                    kind: None,
                };
            }
        }
    }
}

/// A line of the text: its characters, without its break, and whether a
/// break ends it.
struct Line<'a> {
    text: &'a str,
    broken: bool,
}

/// The lines of `text` from byte `from`, each without its break.
fn lines_from(text: &str, from: usize) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    let mut rest = &text[from..];
    loop {
        match rest.find(['\n', '\r']) {
            None => {
                lines.push(Line {
                    text: rest,
                    broken: false,
                });
                return lines;
            }
            Some(at) => {
                lines.push(Line {
                    text: &rest[..at],
                    broken: true,
                });
                let width = if rest[at..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[at + width..];
                if rest.is_empty() {
                    return lines;
                }
            }
        }
    }
}

/// Where the line holding byte `at` starts.
fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind(['\n', '\r']).map_or(0, |b| b + 1)
}

/// A block scalar's header's chomping: `Some(true)` to keep (`+`),
/// `Some(false)` to strip (`-`), `None` to clip; read from the indicator
/// at the start of `header`, `None` for a line that is no header.
fn chomping(header: &str) -> Option<Option<bool>> {
    let header = header.split(['\n', '\r']).next().unwrap_or("");
    let mut chars = header.chars();
    if !matches!(chars.next(), Some('|' | '>')) {
        return None;
    }
    let rest = chars.as_str();
    let modifiers: String = rest
        .chars()
        .take_while(|c| matches!(c, '+' | '-' | '0'..='9'))
        .collect();
    let after = rest[modifiers.len()..].trim_start_matches([' ', '\t']);
    if !(after.is_empty() || after.starts_with('#')) {
        return None;
    }
    Some(if modifiers.contains('+') {
        Some(true)
    } else if modifiers.contains('-') {
        Some(false)
    } else {
        None
    })
}

/// The header of the block scalar whose content starts on the line at
/// byte `content`: the nearest line above it that is not blank, its
/// indicator the last `|` or `>` before its modifiers and comment. Its
/// chomping, and the byte its line ends at.
fn header_above(text: &str, content: usize) -> Option<(Option<bool>, usize)> {
    let mut end = line_start(text, content);
    loop {
        let before = end.checked_sub(1)?;
        let start = line_start(text, before);
        let line = text[start..before].trim_end_matches('\r');
        if !line.chars().all(|c| c == ' ') {
            // The indicator is the last `|` or `>` from which the rest of
            // the line reads as a header.
            let keep = line
                .char_indices()
                .rev()
                .filter(|(_, c)| matches!(c, '|' | '>'))
                .find_map(|(at, _)| chomping(&line[at..]))?;
            return Some((keep, start + line.len()));
        }
        end = start;
    }
}

/// The value of a block scalar as PyYAML reads it (`scan_block_scalar`),
/// `yaml-rust2` having read it as `value` and marked it at character
/// `at`; `None` where the text cannot be read so, the value left as is.
/// Its content is `yaml-rust2`'s without the breaks that end it; after the
/// content comes its last line's break unless it is stripped, and the
/// breaks of the blank lines after it if it is kept.
fn block_value(text: &str, at: usize, value: &str) -> Option<String> {
    let mark = text.char_indices().nth(at).map_or(text.len(), |(b, _)| b);
    let blank = |line: &str| line.chars().all(|c| c == ' ');
    let content = value.trim_end_matches('\n');
    if content.is_empty() && !value.is_empty() && !value.ends_with('\n') {
        return None;
    }
    if content.is_empty() {
        // No content: kept, the breaks of the blank lines after the
        // header; else nothing.
        let (keep, header_end) = match chomping(&text[mark..]) {
            Some(keep) => (
                keep,
                mark + text[mark..].find(['\n', '\r']).unwrap_or(text.len() - mark),
            ),
            None => header_above(text, mark)?,
        };
        if keep != Some(true) {
            return Some(String::new());
        }
        if header_end >= text.len() {
            return Some(String::new());
        }
        let lines = lines_from(text, header_end);
        // The header's own line's break opens the first.
        let breaks = lines
            .iter()
            .skip(1)
            .take_while(|line| blank(line.text) && line.broken)
            .count();
        return Some("\n".repeat(breaks));
    }
    let (keep, _) = header_above(text, mark)?;
    let start = line_start(text, mark);
    let indent = text[start..mark].chars().count();
    if !blank(&text[start..mark]) {
        return None;
    }
    let lines = lines_from(text, start);
    let is_content = |line: &Line| {
        let spaces = line.text.len() - line.text.trim_start_matches(' ').len();
        spaces >= indent && line.text.len() > indent
    };
    let mut last = 0;
    let mut stop = lines.len();
    for (i, line) in lines.iter().enumerate().skip(1) {
        if is_content(line) {
            last = i;
        } else if !(blank(line.text) && line.broken) {
            stop = i;
            break;
        }
    }
    let line_break = usize::from(lines[last].broken);
    let breaks = lines[last + 1..stop]
        .iter()
        .filter(|line| blank(line.text) && line.broken)
        .count();
    let tail = match keep {
        Some(false) => 0,
        None => line_break,
        Some(true) => line_break + breaks,
    };
    Some(format!("{content}{}", "\n".repeat(tail)))
}

impl MarkedEventReceiver for Composer {
    fn on_event(&mut self, event: Event, mark: Marker) {
        if self.error.is_some() {
            return;
        }
        match event {
            Event::Scalar(text, style, anchor, tag) => {
                if let Some(tag) = tag {
                    self.error = Some(outside(Outside::Tag(written(&tag))));
                    return;
                }
                let kind = (style == TScalarStyle::Plain).then(|| implicit(&text));
                if kind == Some("merge") {
                    self.error = Some(outside(Outside::Merge));
                    return;
                }
                let block = matches!(style, TScalarStyle::Literal | TScalarStyle::Folded);
                let node = self.open(anchor, Raw::Scalar { text, kind });
                if block {
                    self.blocks.push(Block {
                        node,
                        at: mark.index(),
                    });
                }
                self.add(node);
            }
            Event::SequenceStart(_, Some(tag)) | Event::MappingStart(_, Some(tag)) => {
                self.error = Some(outside(Outside::Tag(written(&tag))));
            }
            Event::SequenceStart(anchor, None) => self.start(anchor, Raw::Seq(Vec::new())),
            Event::MappingStart(anchor, None) => self.start(anchor, Raw::Map(Vec::new())),
            Event::SequenceEnd | Event::MappingEnd => self.end(),
            Event::Alias(anchor) => match self.anchors.get(&anchor).copied() {
                Some(node) => self.add(node),
                None => {
                    self.error = Some(error("an alias names no anchor defined before it"));
                }
            },
            Event::Nothing
            | Event::StreamStart
            | Event::StreamEnd
            | Event::DocumentStart
            | Event::DocumentEnd => {}
        }
    }
}

/// A tag as written: `!!binary` for YAML's own, which `yaml-rust2`
/// resolves.
fn written(tag: &Tag) -> String {
    match tag.handle.as_str() {
        "tag:yaml.org,2002:" => format!("!!{}", tag.suffix),
        handle => format!("{handle}{}", tag.suffix),
    }
}

/// PyYAML's implicit resolvers (`yaml/resolver.py`), each tried in the
/// order it was added, among those registered for the scalar's first
/// character: the kind of value a plain scalar is.
fn implicit(text: &str) -> &'static str {
    static RESOLVERS: LazyLock<Vec<(&'static str, &'static str, Regex)>> = LazyLock::new(|| {
        [
            (
                "bool",
                "yYnNtTfFoO",
                r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$",
            ),
            (
                "float",
                "-+0123456789.",
                r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
            ),
            (
                "int",
                "-+0123456789",
                r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
            ),
            ("merge", "<", r"^(?:<<)$"),
            ("null", "~nN", r"^(?:~|null|Null|NULL|)$"),
            (
                "timestamp",
                "0123456789",
                r"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$",
            ),
            ("value", "=", r"^(?:=)$"),
        ]
        .into_iter()
        .filter_map(|(kind, first, pattern)| Some((kind, first, Regex::new(pattern).ok()?)))
        .collect()
    });
    // The empty scalar is registered for the null resolver alone.
    let Some(first) = text.chars().next() else {
        return "null";
    };
    RESOLVERS
        .iter()
        .find(|(_, firsts, pattern)| firsts.contains(first) && pattern.is_match(text))
        .map_or("str", |(kind, _, _)| kind)
}

/// Makes each node's value, as PyYAML's `SafeConstructor` does, every node
/// composed (each is in the document, a value a later key replaces too);
/// the values keep the nodes' places, so a collection names its members by
/// node.
fn construct(raw: Vec<Raw>, root: Option<usize>, rules: &Rules) -> Result<Document, YamlError> {
    let Some(root) = root else {
        return Ok(Document {
            values: vec![Value::Null],
            root: NodeId(0),
        });
    };
    // A `=` is text as a key (`flatten_mapping`), and no value elsewhere.
    let mut keys = vec![false; raw.len()];
    for node in &raw {
        if let Raw::Map(pairs) = node {
            for &(key, _) in pairs {
                keys[key] = true;
            }
        }
    }
    let mut values = Vec::with_capacity(raw.len());
    for (at, node) in raw.iter().enumerate() {
        values.push(match node {
            Raw::Scalar {
                text,
                kind: Some("value"),
            } if keys[at] => Value::Str(text.clone()),
            Raw::Scalar { text, kind } => scalar(text, *kind, rules)?,
            Raw::Seq(items) => Value::Seq(items.iter().copied().map(NodeId).collect()),
            Raw::Map(_) => Value::Null,
        });
    }
    // Keys are scalars, made above.
    for (at, node) in raw.iter().enumerate() {
        if let Raw::Map(pairs) = node {
            values[at] = Value::Map(mapping(&values, pairs));
        }
    }
    Ok(Document {
        values,
        root: NodeId(root),
    })
}

/// A scalar's value: text for a quoted one, else by its resolved kind.
fn scalar(text: &str, kind: Option<&str>, rules: &Rules) -> Result<Value, YamlError> {
    Ok(match kind {
        None | Some("str") => Value::Str(text.to_owned()),
        Some("null") => Value::Null,
        Some("bool") => Value::Bool(matches!(
            text.to_lowercase().as_str(),
            "yes" | "true" | "on"
        )),
        Some("int") => Value::Int(integer(text, rules)?),
        Some("float") => Value::Float(float(text)?),
        Some("timestamp") => timestamp(text)?,
        Some(_) => return Err(error(format!("no value can be made of '{text}' here"))),
    })
}

/// `construct_mapping`: a key equal to an earlier one keeps that one's
/// place and takes the later value; a `=` key is text.
fn mapping(values: &[Value], pairs: &[(usize, usize)]) -> Vec<(NodeId, NodeId)> {
    let mut entries: Vec<(NodeId, NodeId)> = Vec::new();
    let mut places: BTreeMap<KeyOf, usize> = BTreeMap::new();
    for &(key, value) in pairs {
        let of = KeyOf::of(&values[key]);
        match places.get(&of) {
            Some(&at) => entries[at].1 = NodeId(value),
            None => {
                places.insert(of, entries.len());
                entries.push((NodeId(key), NodeId(value)));
            }
        }
    }
    entries
}

/// A key as Python's `dict` tells keys apart: numbers by value exactly (a
/// boolean is 0 or 1, a float that is an integer that integer), each NaN
/// PyYAML makes the one object it is, two times with offsets by instant,
/// two without by every field, never one of each, a date never a time.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum KeyOf {
    Null,
    /// An integer, or a float of an integer's value, as decimal digits.
    Integer(String),
    /// Another float, by its bits, `-0.0` being `0`.
    Float(u64),
    Nan,
    Text(String),
    Date(Date),
    Naive(Timestamp),
    Instant(i128),
    /// A value no key can be, which the composer refuses first.
    Other,
}

impl KeyOf {
    fn of(value: &Value) -> KeyOf {
        match value {
            Value::Null => KeyOf::Null,
            Value::Bool(flag) => KeyOf::Integer(if *flag { "1" } else { "0" }.into()),
            Value::Int(n) => KeyOf::Integer(n.text().to_owned()),
            Value::Float(x) if x.is_nan() => KeyOf::Nan,
            Value::Float(x) if x.is_finite() && x.fract() == 0.0 => {
                let digits = format!("{x:.0}");
                KeyOf::Integer(if digits == "-0" { "0".into() } else { digits })
            }
            Value::Float(x) => KeyOf::Float(x.to_bits()),
            Value::Str(text) => KeyOf::Text(text.clone()),
            Value::Date(date) => KeyOf::Date(*date),
            Value::Timestamp(moment) if moment.offset.is_some() => KeyOf::Instant(instant(moment)),
            Value::Timestamp(moment) => KeyOf::Naive(*moment),
            Value::Seq(_) | Value::Map(_) => KeyOf::Other,
        }
    }
}

/// A time with an offset as microseconds since 0001-01-01 00:00 UTC.
fn instant(moment: &Timestamp) -> i128 {
    let date = moment.date;
    let y = i128::from(date.year) - 1;
    let before: [i128; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = date.year % 4 == 0 && (date.year % 100 != 0 || date.year % 400 == 0);
    let month = usize::from(date.month.clamp(1, 12)) - 1;
    let mut days = y * 365 + y / 4 - y / 100 + y / 400 + before[month] + i128::from(date.day);
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

/// `construct_yaml_int`: `_` dropped, a sign, then `0`, `0b` binary, `0x`
/// hexadecimal, a leading `0` octal, `:` base 60, else decimal. Python
/// reads at most `rules.most_digits` decimal digits; an integer of more
/// digits than that from another base is outside the YAML view
/// definitions use, since Python would not write it.
fn integer(text: &str, rules: &Rules) -> Result<Integer, YamlError> {
    let value = text.replace('_', "");
    let refused = || error(format!("'{text}' is no integer"));
    let (negative, digits) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        Some(_) => (false, value.as_str()),
        None => return Err(refused()),
    };
    // How many decimal digits a number of `count` significant digits of
    // `base` has at least: too many is refused before any is computed.
    let too_long = |count: usize, base: f64| {
        count.saturating_sub(1) as f64 * base.log10() > (rules.most_digits + 1) as f64
    };
    let significant = |digits: &str| digits.trim_start_matches('0').len();
    let magnitude = if digits == "0" {
        vec![0]
    } else if let Some(rest) = digits.strip_prefix("0b") {
        if too_long(significant(rest), 2.0) {
            return Err(outside(Outside::LongInteger));
        }
        radix(rest, 2).ok_or_else(refused)?
    } else if let Some(rest) = digits.strip_prefix("0x") {
        if too_long(significant(rest), 16.0) {
            return Err(outside(Outside::LongInteger));
        }
        radix(rest, 16).ok_or_else(refused)?
    } else if digits.starts_with('0') {
        if too_long(significant(digits), 8.0) {
            return Err(outside(Outside::LongInteger));
        }
        radix(digits, 8).ok_or_else(refused)?
    } else if digits.contains(':') {
        let parts: Vec<&str> = digits.split(':').collect();
        let first = parts.first().copied().unwrap_or("");
        if first.len() > rules.most_digits {
            return Err(error(format!(
                "'{text}' has more digits than an integer of base 10 is read with"
            )));
        }
        // Each part after the first is below 60, so adds at most two
        // digits: the first part's digits and the parts' count bound it.
        let estimate =
            first.len().saturating_sub(1) as f64 + (parts.len() - 1) as f64 * 60f64.log10();
        if estimate > (rules.most_digits + 1) as f64 {
            return Err(outside(Outside::LongInteger));
        }
        let mut sum = vec![0];
        for part in parts {
            let part = radix(part, 10).ok_or_else(refused)?;
            sum = add(&times(&sum, 60), &part);
        }
        sum
    } else {
        if digits.len() > rules.most_digits {
            return Err(error(format!(
                "'{text}' has more digits than an integer of base 10 is read with"
            )));
        }
        radix(digits, 10).ok_or_else(refused)?
    };
    let mut decimal = decimal(&magnitude);
    if decimal.len() > rules.most_digits {
        return Err(outside(Outside::LongInteger));
    }
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
    let mut out = limbs.last().copied().unwrap_or(0).to_string();
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
    static PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^(?P<year>[0-9][0-9][0-9][0-9])-(?P<month>[0-9][0-9]?)-(?P<day>[0-9][0-9]?)(?:(?:[Tt]|[ \t]+)(?P<hour>[0-9][0-9]?):(?P<minute>[0-9][0-9]):(?P<second>[0-9][0-9])(?:\.(?P<fraction>[0-9]*))?(?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)(?::(?P<tz_minute>[0-9][0-9]))?))?)?$",
        )
        .ok()
    });
    let refused = || error(format!("'{text}' is no date or time"));
    let groups = PATTERN
        .as_ref()
        .and_then(|pattern| pattern.captures(text))
        .ok_or_else(refused)?;
    // Each group is at most four ASCII digits.
    let number = |name: &str| -> u32 {
        groups
            .name(name)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0)
    };
    let small = |n: u32| u8::try_from(n).map_err(|_| refused());
    let date = Date {
        year: i32::try_from(number("year")).map_err(|_| refused())?,
        month: small(number("month"))?,
        day: small(number("day"))?,
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
            digits.parse().map_err(|_| refused())?
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
        hour: small(hour)?,
        minute: small(minute)?,
        second: small(second)?,
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
