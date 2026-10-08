// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `yaml.safe_load` as PyYAML 6 runs it: its scanner, ported line for line,
//! its parser and composer, and the safe constructor with the YAML 1.1
//! implicit types (`on` and `yes` are booleans, `0x1A` and `1:30` integers,
//! `2021-01-01` a date). Errors carry PyYAML's class name and its text.

use std::collections::HashMap;

use crate::dt::civil::{self, Wall};
use crate::pyval::PyVal;

/// A position in the text, as PyYAML's `Mark`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mark {
    index: usize,
    line: usize,
    column: usize,
}

/// A `yaml.YAMLError` (or a `ValueError` a constructor raises).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YamlError {
    pub class: &'static str,
    pub message: String,
}

#[derive(Clone, Debug)]
enum Tok {
    StreamStart,
    StreamEnd,
    Directive,
    DocumentStart,
    DocumentEnd,
    BlockSequenceStart,
    BlockMappingStart,
    BlockEnd,
    FlowSequenceStart,
    FlowMappingStart,
    FlowSequenceEnd,
    FlowMappingEnd,
    Key,
    Value,
    BlockEntry,
    FlowEntry,
    Alias(String),
    Anchor(String),
    Tag(Option<String>, String),
    Scalar(String, bool),
}

impl Tok {
    fn id(&self) -> &'static str {
        match self {
            Tok::StreamStart => "<stream start>",
            Tok::StreamEnd => "<stream end>",
            Tok::Directive => "<directive>",
            Tok::DocumentStart => "<document start>",
            Tok::DocumentEnd => "<document end>",
            Tok::BlockSequenceStart => "<block sequence start>",
            Tok::BlockMappingStart => "<block mapping start>",
            Tok::BlockEnd => "<block end>",
            Tok::FlowSequenceStart => "[",
            Tok::FlowMappingStart => "{",
            Tok::FlowSequenceEnd => "]",
            Tok::FlowMappingEnd => "}",
            Tok::Key => "?",
            Tok::Value => ":",
            Tok::BlockEntry => "-",
            Tok::FlowEntry => ",",
            Tok::Alias(_) => "<alias>",
            Tok::Anchor(_) => "<anchor>",
            Tok::Tag(..) => "<tag>",
            Tok::Scalar(..) => "<scalar>",
        }
    }
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    start: Mark,
    end: Mark,
}

struct SimpleKey {
    token_number: usize,
    required: bool,
    index: usize,
    line: usize,
    column: usize,
    mark: Mark,
}

const BREAKZ: &str = "\0\r\n\u{85}\u{2028}\u{2029}";
const BLANKZ: &str = "\0 \t\r\n\u{85}\u{2028}\u{2029}";

fn is_in(c: char, set: &str) -> bool {
    set.contains(c)
}

struct Scanner<'a> {
    buffer: &'a [char],
    pointer: usize,
    index: usize,
    line: usize,
    column: usize,
    done: bool,
    flow_level: usize,
    tokens: Vec<Token>,
    indent: isize,
    indents: Vec<isize>,
    allow_simple_key: bool,
    possible_simple_keys: HashMap<usize, SimpleKey>,
}

type SResult<T> = Result<T, YamlError>;

fn snippet(buffer: &[char], pointer: usize) -> String {
    let indent = 4;
    let max_length = 75.0;
    let mut head = "";
    let mut start = pointer;
    while start > 0 && !is_in(buffer[start - 1], BREAKZ) {
        start -= 1;
        if (pointer - start) as f64 > max_length / 2.0 - 1.0 {
            head = " ... ";
            start += 5;
            break;
        }
    }
    let mut tail = "";
    let mut end = pointer;
    let len = buffer.len().saturating_sub(1);
    while end < len && !is_in(buffer[end], BREAKZ) {
        end += 1;
        if (end - pointer) as f64 > max_length / 2.0 - 1.0 {
            tail = " ... ";
            end -= 5;
            break;
        }
    }
    let text: String = buffer[start.min(end)..end].iter().collect();
    format!(
        "{}{head}{text}{tail}\n{}^",
        " ".repeat(indent),
        " ".repeat(indent + pointer - start + head.len())
    )
}

fn mark_text(buffer: &[char], mark: Mark) -> String {
    format!(
        "  in \"<unicode string>\", line {}, column {}:\n{}",
        mark.line + 1,
        mark.column + 1,
        snippet(buffer, mark.index)
    )
}

fn marked(
    buffer: &[char],
    class: &'static str,
    context: Option<&str>,
    context_mark: Option<Mark>,
    problem: &str,
    problem_mark: Option<Mark>,
) -> YamlError {
    let mut lines = Vec::new();
    if let Some(c) = context {
        lines.push(c.to_string());
    }
    if let Some(cm) = context_mark
        && problem_mark.is_none_or(|pm| pm.line != cm.line || pm.column != cm.column)
    {
        lines.push(mark_text(buffer, cm));
    }
    lines.push(problem.to_string());
    if let Some(pm) = problem_mark {
        lines.push(mark_text(buffer, pm));
    }
    YamlError {
        class,
        message: lines.join("\n"),
    }
}

/// `repr` of one character, as PyYAML's messages write it.
fn char_repr(c: char) -> String {
    crate::py::str_repr(&c.to_string())
}

impl<'a> Scanner<'a> {
    fn new(buffer: &'a [char]) -> Self {
        let mut s = Scanner {
            buffer,
            pointer: 0,
            index: 0,
            line: 0,
            column: 0,
            done: false,
            flow_level: 0,
            tokens: Vec::new(),
            indent: -1,
            indents: Vec::new(),
            allow_simple_key: true,
            possible_simple_keys: HashMap::new(),
        };
        let mark = s.mark();
        s.tokens.push(Token {
            tok: Tok::StreamStart,
            start: mark,
            end: mark,
        });
        s
    }

    fn mark(&self) -> Mark {
        Mark {
            index: self.pointer,
            line: self.line,
            column: self.column,
        }
    }

    fn peek(&self, i: usize) -> char {
        self.buffer.get(self.pointer + i).copied().unwrap_or('\0')
    }

    fn prefix(&self, n: usize) -> String {
        let end = (self.pointer + n).min(self.buffer.len());
        self.buffer[self.pointer..end].iter().collect()
    }

    fn forward(&mut self, n: usize) {
        for _ in 0..n {
            let ch = self.peek(0);
            self.pointer += 1;
            self.index += 1;
            if is_in(ch, "\n\u{85}\u{2028}\u{2029}") || (ch == '\r' && self.peek(0) != '\n') {
                self.line += 1;
                self.column = 0;
            } else if ch != '\u{feff}' {
                self.column += 1;
            }
        }
    }

    fn error(&self, context: Option<&str>, context_mark: Option<Mark>, problem: &str) -> YamlError {
        marked(
            self.buffer,
            "ScannerError",
            context,
            context_mark,
            problem,
            Some(self.mark()),
        )
    }

    fn scan_all(&mut self) -> SResult<Vec<Token>> {
        while !self.done {
            self.fetch_more_tokens()?;
        }
        Ok(std::mem::take(&mut self.tokens))
    }

    fn push(&mut self, tok: Tok, start: Mark, end: Mark) {
        self.tokens.push(Token { tok, start, end });
    }

    fn fetch_more_tokens(&mut self) -> SResult<()> {
        self.scan_to_next_token();
        self.stale_possible_simple_keys()?;
        self.unwind_indent(self.column as isize);
        let ch = self.peek(0);
        match ch {
            '\0' => return self.fetch_stream_end(),
            '%' if self.column == 0 => return self.fetch_directive(),
            '-' if self.check_document_indicator("---") => {
                return self.fetch_document_indicator(Tok::DocumentStart);
            }
            '.' if self.check_document_indicator("...") => {
                return self.fetch_document_indicator(Tok::DocumentEnd);
            }
            '[' => return self.fetch_flow_collection_start(Tok::FlowSequenceStart),
            '{' => return self.fetch_flow_collection_start(Tok::FlowMappingStart),
            ']' => return self.fetch_flow_collection_end(Tok::FlowSequenceEnd),
            '}' => return self.fetch_flow_collection_end(Tok::FlowMappingEnd),
            ',' => return self.fetch_flow_entry(),
            _ => {}
        }
        if ch == '-' && is_in(self.peek(1), BLANKZ) {
            return self.fetch_block_entry();
        }
        if ch == '?' && (self.flow_level > 0 || is_in(self.peek(1), BLANKZ)) {
            return self.fetch_key();
        }
        if ch == ':' && (self.flow_level > 0 || is_in(self.peek(1), BLANKZ)) {
            return self.fetch_value();
        }
        match ch {
            '*' => return self.fetch_anchor(true),
            '&' => return self.fetch_anchor(false),
            '!' => return self.fetch_tag(),
            '|' if self.flow_level == 0 => return self.fetch_block_scalar('|'),
            '>' if self.flow_level == 0 => return self.fetch_block_scalar('>'),
            '\'' => return self.fetch_flow_scalar('\''),
            '"' => return self.fetch_flow_scalar('"'),
            _ => {}
        }
        if self.check_plain() {
            return self.fetch_plain();
        }
        Err(self.error(
            Some("while scanning for the next token"),
            None,
            &format!(
                "found character {} that cannot start any token",
                char_repr(ch)
            ),
        ))
    }

    fn check_document_indicator(&self, what: &str) -> bool {
        self.column == 0 && self.prefix(3) == what && is_in(self.peek(3), BLANKZ)
    }

    fn check_plain(&self) -> bool {
        let ch = self.peek(0);
        !is_in(ch, "\0 \t\r\n\u{85}\u{2028}\u{2029}-?:,[]{}#&*!|>'\"%@`")
            || (!is_in(self.peek(1), BLANKZ)
                && (ch == '-' || (self.flow_level == 0 && (ch == '?' || ch == ':'))))
    }

    fn stale_possible_simple_keys(&mut self) -> SResult<()> {
        let levels: Vec<usize> = self.possible_simple_keys.keys().copied().collect();
        for level in levels {
            let key = &self.possible_simple_keys[&level];
            if key.line != self.line || self.index - key.index > 1024 {
                if key.required {
                    return Err(self.error(
                        Some("while scanning a simple key"),
                        Some(key.mark),
                        "could not find expected ':'",
                    ));
                }
                self.possible_simple_keys.remove(&level);
            }
        }
        Ok(())
    }

    fn save_possible_simple_key(&mut self) -> SResult<()> {
        let required = self.flow_level == 0 && self.indent == self.column as isize;
        if self.allow_simple_key {
            self.remove_possible_simple_key()?;
            let key = SimpleKey {
                token_number: self.tokens.len(),
                required,
                index: self.index,
                line: self.line,
                column: self.column,
                mark: self.mark(),
            };
            self.possible_simple_keys.insert(self.flow_level, key);
        }
        Ok(())
    }

    fn remove_possible_simple_key(&mut self) -> SResult<()> {
        if let Some(key) = self.possible_simple_keys.get(&self.flow_level) {
            if key.required {
                return Err(self.error(
                    Some("while scanning a simple key"),
                    Some(key.mark),
                    "could not find expected ':'",
                ));
            }
            self.possible_simple_keys.remove(&self.flow_level);
        }
        Ok(())
    }

    fn unwind_indent(&mut self, column: isize) {
        if self.flow_level > 0 {
            return;
        }
        while self.indent > column {
            let mark = self.mark();
            self.indent = self.indents.pop().unwrap_or(-1);
            self.push(Tok::BlockEnd, mark, mark);
        }
    }

    fn add_indent(&mut self, column: isize) -> bool {
        if self.indent < column {
            self.indents.push(self.indent);
            self.indent = column;
            return true;
        }
        false
    }

    fn fetch_stream_end(&mut self) -> SResult<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        self.possible_simple_keys.clear();
        let mark = self.mark();
        self.push(Tok::StreamEnd, mark, mark);
        self.done = true;
        Ok(())
    }

    fn fetch_directive(&mut self) -> SResult<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        let start = self.mark();
        self.forward(1);
        let mut length = 0;
        while self.peek(length).is_ascii_alphanumeric() || is_in(self.peek(length), "-_") {
            length += 1;
        }
        if length == 0 {
            return Err(self.error(
                Some("while scanning a directive"),
                Some(start),
                &format!(
                    "expected alphabetic or numeric character, but found {}",
                    char_repr(self.peek(0))
                ),
            ));
        }
        self.forward(length);
        let end = self.mark();
        while !is_in(self.peek(0), BREAKZ) {
            self.forward(1);
        }
        self.scan_line_break();
        self.push(Tok::Directive, start, end);
        Ok(())
    }

    fn fetch_document_indicator(&mut self, tok: Tok) -> SResult<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        let start = self.mark();
        self.forward(3);
        let end = self.mark();
        self.push(tok, start, end);
        Ok(())
    }

    fn fetch_flow_collection_start(&mut self, tok: Tok) -> SResult<()> {
        self.save_possible_simple_key()?;
        self.flow_level += 1;
        self.allow_simple_key = true;
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(tok, start, end);
        Ok(())
    }

    fn fetch_flow_collection_end(&mut self, tok: Tok) -> SResult<()> {
        self.remove_possible_simple_key()?;
        self.flow_level = self.flow_level.saturating_sub(1);
        self.allow_simple_key = false;
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(tok, start, end);
        Ok(())
    }

    fn fetch_flow_entry(&mut self) -> SResult<()> {
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(Tok::FlowEntry, start, end);
        Ok(())
    }

    fn fetch_block_entry(&mut self) -> SResult<()> {
        if self.flow_level == 0 {
            if !self.allow_simple_key {
                return Err(self.error(None, None, "sequence entries are not allowed here"));
            }
            if self.add_indent(self.column as isize) {
                let mark = self.mark();
                self.push(Tok::BlockSequenceStart, mark, mark);
            }
        }
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(Tok::BlockEntry, start, end);
        Ok(())
    }

    fn fetch_key(&mut self) -> SResult<()> {
        if self.flow_level == 0 {
            if !self.allow_simple_key {
                return Err(self.error(None, None, "mapping keys are not allowed here"));
            }
            if self.add_indent(self.column as isize) {
                let mark = self.mark();
                self.push(Tok::BlockMappingStart, mark, mark);
            }
        }
        self.allow_simple_key = self.flow_level == 0;
        self.remove_possible_simple_key()?;
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(Tok::Key, start, end);
        Ok(())
    }

    fn fetch_value(&mut self) -> SResult<()> {
        if let Some(key) = self.possible_simple_keys.remove(&self.flow_level) {
            self.tokens.insert(
                key.token_number,
                Token {
                    tok: Tok::Key,
                    start: key.mark,
                    end: key.mark,
                },
            );
            if self.flow_level == 0 && self.add_indent(key.column as isize) {
                self.tokens.insert(
                    key.token_number,
                    Token {
                        tok: Tok::BlockMappingStart,
                        start: key.mark,
                        end: key.mark,
                    },
                );
            }
            self.allow_simple_key = false;
        } else {
            if self.flow_level == 0 {
                if !self.allow_simple_key {
                    return Err(self.error(None, None, "mapping values are not allowed here"));
                }
                if self.add_indent(self.column as isize) {
                    let mark = self.mark();
                    self.push(Tok::BlockMappingStart, mark, mark);
                }
            }
            self.allow_simple_key = self.flow_level == 0;
            self.remove_possible_simple_key()?;
        }
        let start = self.mark();
        self.forward(1);
        let end = self.mark();
        self.push(Tok::Value, start, end);
        Ok(())
    }

    fn fetch_anchor(&mut self, alias: bool) -> SResult<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let start = self.mark();
        let name = if alias { "alias" } else { "anchor" };
        self.forward(1);
        let mut length = 0;
        while self.peek(length).is_ascii_alphanumeric() || is_in(self.peek(length), "-_") {
            length += 1;
        }
        if length == 0 {
            return Err(self.error(
                Some(&format!("while scanning an {name}")),
                Some(start),
                &format!(
                    "expected alphabetic or numeric character, but found {}",
                    char_repr(self.peek(0))
                ),
            ));
        }
        let value = self.prefix(length);
        self.forward(length);
        if !is_in(self.peek(0), "\0 \t\r\n\u{85}\u{2028}\u{2029}?:,]}%@`") {
            return Err(self.error(
                Some(&format!("while scanning an {name}")),
                Some(start),
                &format!(
                    "expected alphabetic or numeric character, but found {}",
                    char_repr(self.peek(0))
                ),
            ));
        }
        let end = self.mark();
        self.push(
            if alias {
                Tok::Alias(value)
            } else {
                Tok::Anchor(value)
            },
            start,
            end,
        );
        Ok(())
    }

    fn fetch_tag(&mut self) -> SResult<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let start = self.mark();
        let ch = self.peek(1);
        let (handle, suffix);
        if ch == '<' {
            self.forward(2);
            suffix = self.scan_tag_uri(start)?;
            if self.peek(0) != '>' {
                return Err(self.error(
                    Some("while parsing a tag"),
                    Some(start),
                    &format!("expected '>', but found {}", char_repr(self.peek(0))),
                ));
            }
            self.forward(1);
            handle = None;
        } else if is_in(ch, BLANKZ) {
            handle = None;
            suffix = "!".to_string();
            self.forward(1);
        } else {
            let mut length = 1;
            let mut use_handle = false;
            let mut c = ch;
            while !is_in(c, "\0 \r\n\u{85}\u{2028}\u{2029}") {
                if c == '!' {
                    use_handle = true;
                    break;
                }
                length += 1;
                c = self.peek(length);
            }
            if use_handle {
                let mut l = 1;
                let mut c = self.peek(l);
                if c != ' ' {
                    while c.is_ascii_alphanumeric() || is_in(c, "-_") {
                        l += 1;
                        c = self.peek(l);
                    }
                    if c != '!' {
                        self.forward(l);
                        return Err(self.error(
                            Some("while scanning a tag"),
                            Some(start),
                            &format!("expected '!', but found {}", char_repr(c)),
                        ));
                    }
                    l += 1;
                }
                handle = Some(self.prefix(l));
                self.forward(l);
            } else {
                handle = Some("!".to_string());
                self.forward(1);
            }
            suffix = self.scan_tag_uri(start)?;
        }
        if !is_in(self.peek(0), "\0 \r\n\u{85}\u{2028}\u{2029}") {
            return Err(self.error(
                Some("while scanning a tag"),
                Some(start),
                &format!("expected ' ', but found {}", char_repr(self.peek(0))),
            ));
        }
        let end = self.mark();
        self.push(Tok::Tag(handle, suffix), start, end);
        Ok(())
    }

    fn scan_tag_uri(&mut self, start: Mark) -> SResult<String> {
        let mut length = 0;
        let mut out = String::new();
        loop {
            let c = self.peek(length);
            if c.is_ascii_alphanumeric() || is_in(c, "-;/?:@&=+$,_.!~*'()[]%") {
                if c == '%' {
                    out.push_str(&self.prefix(length));
                    self.forward(length);
                    length = 0;
                    let mut codes = Vec::new();
                    while self.peek(0) == '%' {
                        self.forward(1);
                        let hex = self.prefix(2);
                        let code = u8::from_str_radix(&hex, 16).map_err(|_| {
                            self.error(
                                Some("while scanning a tag"),
                                Some(start),
                                "expected URI escape sequence of 2 hexadecimal numbers",
                            )
                        })?;
                        codes.push(code);
                        self.forward(2);
                    }
                    out.push_str(&String::from_utf8_lossy(&codes));
                } else {
                    length += 1;
                }
            } else {
                break;
            }
        }
        if length > 0 {
            out.push_str(&self.prefix(length));
            self.forward(length);
        }
        if out.is_empty() {
            return Err(self.error(
                Some("while parsing a tag"),
                Some(start),
                &format!("expected URI, but found {}", char_repr(self.peek(0))),
            ));
        }
        Ok(out)
    }

    fn scan_to_next_token(&mut self) {
        if self.index == 0 && self.peek(0) == '\u{feff}' {
            self.forward(1);
        }
        loop {
            while self.peek(0) == ' ' {
                self.forward(1);
            }
            if self.peek(0) == '#' {
                while !is_in(self.peek(0), BREAKZ) {
                    self.forward(1);
                }
            }
            if !self.scan_line_break().is_empty() {
                if self.flow_level == 0 {
                    self.allow_simple_key = true;
                }
            } else {
                break;
            }
        }
    }

    fn scan_line_break(&mut self) -> String {
        let ch = self.peek(0);
        if is_in(ch, "\r\n\u{85}") {
            if self.prefix(2) == "\r\n" {
                self.forward(2);
            } else {
                self.forward(1);
            }
            return "\n".to_string();
        }
        if is_in(ch, "\u{2028}\u{2029}") {
            self.forward(1);
            return ch.to_string();
        }
        String::new()
    }

    fn fetch_block_scalar(&mut self, style: char) -> SResult<()> {
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let folded = style == '>';
        let mut chunks = String::new();
        let start = self.mark();
        self.forward(1);
        let mut chomping: Option<bool> = None;
        let mut increment: Option<usize> = None;
        let ch = self.peek(0);
        if ch == '+' || ch == '-' {
            chomping = Some(ch == '+');
            self.forward(1);
            let c = self.peek(0);
            if c.is_ascii_digit() {
                if c == '0' {
                    return Err(self.error(
                        Some("while scanning a block scalar"),
                        Some(start),
                        "expected indentation indicator in the range 1-9, but found 0",
                    ));
                }
                increment = c.to_digit(10).map(|d| d as usize);
                self.forward(1);
            }
        } else if ch.is_ascii_digit() {
            if ch == '0' {
                return Err(self.error(
                    Some("while scanning a block scalar"),
                    Some(start),
                    "expected indentation indicator in the range 1-9, but found 0",
                ));
            }
            increment = ch.to_digit(10).map(|d| d as usize);
            self.forward(1);
            let c = self.peek(0);
            if c == '+' || c == '-' {
                chomping = Some(c == '+');
                self.forward(1);
            }
        }
        if !is_in(self.peek(0), "\0 \r\n\u{85}\u{2028}\u{2029}") {
            return Err(self.error(
                Some("while scanning a block scalar"),
                Some(start),
                &format!(
                    "expected chomping or indentation indicators, but found {}",
                    char_repr(self.peek(0))
                ),
            ));
        }
        while self.peek(0) == ' ' {
            self.forward(1);
        }
        if self.peek(0) == '#' {
            while !is_in(self.peek(0), BREAKZ) {
                self.forward(1);
            }
        }
        if !is_in(self.peek(0), BREAKZ) {
            return Err(self.error(
                Some("while scanning a block scalar"),
                Some(start),
                &format!(
                    "expected a comment or a line break, but found {}",
                    char_repr(self.peek(0))
                ),
            ));
        }
        self.scan_line_break();
        let min_indent = (self.indent + 1).max(1) as usize;
        let (mut breaks, indent, mut end) = match increment {
            None => {
                let mut breaks = Vec::new();
                let mut max_indent = 0;
                let mut end = self.mark();
                while is_in(self.peek(0), " \r\n\u{85}\u{2028}\u{2029}") {
                    if self.peek(0) != ' ' {
                        breaks.push(self.scan_line_break());
                        end = self.mark();
                    } else {
                        self.forward(1);
                        max_indent = max_indent.max(self.column);
                    }
                }
                (breaks, min_indent.max(max_indent), end)
            }
            Some(inc) => {
                let indent = min_indent + inc - 1;
                let (breaks, end) = self.block_breaks(indent);
                (breaks, indent, end)
            }
        };
        let mut line_break = String::new();
        while self.column == indent && self.peek(0) != '\0' {
            chunks.push_str(&breaks.concat());
            let leading_non_space = !is_in(self.peek(0), " \t");
            let mut length = 0;
            while !is_in(self.peek(length), BREAKZ) {
                length += 1;
            }
            chunks.push_str(&self.prefix(length));
            self.forward(length);
            line_break = self.scan_line_break();
            let (b, e) = self.block_breaks(indent);
            breaks = b;
            end = e;
            if self.column == indent && self.peek(0) != '\0' {
                if folded && line_break == "\n" && leading_non_space && !is_in(self.peek(0), " \t")
                {
                    if breaks.is_empty() {
                        chunks.push(' ');
                    }
                } else {
                    chunks.push_str(&line_break);
                }
            } else {
                break;
            }
        }
        if chomping != Some(false) {
            chunks.push_str(&line_break);
        }
        if chomping == Some(true) {
            chunks.push_str(&breaks.concat());
        }
        self.push(Tok::Scalar(chunks, false), start, end);
        Ok(())
    }

    fn block_breaks(&mut self, indent: usize) -> (Vec<String>, Mark) {
        let mut chunks = Vec::new();
        let mut end = self.mark();
        while self.column < indent && self.peek(0) == ' ' {
            self.forward(1);
        }
        while is_in(self.peek(0), "\r\n\u{85}\u{2028}\u{2029}") {
            chunks.push(self.scan_line_break());
            end = self.mark();
            while self.column < indent && self.peek(0) == ' ' {
                self.forward(1);
            }
        }
        (chunks, end)
    }

    fn fetch_flow_scalar(&mut self, style: char) -> SResult<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let double = style == '"';
        let start = self.mark();
        let quote = self.peek(0);
        self.forward(1);
        let mut chunks = String::new();
        self.flow_non_spaces(double, start, &mut chunks)?;
        while self.peek(0) != quote {
            self.flow_spaces(double, start, &mut chunks)?;
            self.flow_non_spaces(double, start, &mut chunks)?;
        }
        self.forward(1);
        let end = self.mark();
        self.push(Tok::Scalar(chunks, false), start, end);
        Ok(())
    }

    fn flow_non_spaces(&mut self, double: bool, start: Mark, chunks: &mut String) -> SResult<()> {
        loop {
            let mut length = 0;
            while !is_in(self.peek(length), "'\"\\\0 \t\r\n\u{85}\u{2028}\u{2029}") {
                length += 1;
            }
            if length > 0 {
                chunks.push_str(&self.prefix(length));
                self.forward(length);
            }
            let ch = self.peek(0);
            if !double && ch == '\'' && self.peek(1) == '\'' {
                chunks.push('\'');
                self.forward(2);
            } else if (double && ch == '\'') || (!double && (ch == '"' || ch == '\\')) {
                chunks.push(ch);
                self.forward(1);
            } else if double && ch == '\\' {
                self.forward(1);
                let c = self.peek(0);
                let replacement = match c {
                    '0' => Some('\0'),
                    'a' => Some('\u{7}'),
                    'b' => Some('\u{8}'),
                    't' | '\t' => Some('\t'),
                    'n' => Some('\n'),
                    'v' => Some('\u{b}'),
                    'f' => Some('\u{c}'),
                    'r' => Some('\r'),
                    'e' => Some('\u{1b}'),
                    ' ' => Some(' '),
                    '"' => Some('"'),
                    '\\' => Some('\\'),
                    '/' => Some('/'),
                    'N' => Some('\u{85}'),
                    '_' => Some('\u{a0}'),
                    'L' => Some('\u{2028}'),
                    'P' => Some('\u{2029}'),
                    _ => None,
                };
                if let Some(r) = replacement {
                    chunks.push(r);
                    self.forward(1);
                } else if let Some(n) = match c {
                    'x' => Some(2),
                    'u' => Some(4),
                    'U' => Some(8),
                    _ => None,
                } {
                    self.forward(1);
                    for k in 0..n {
                        if !self.peek(k).is_ascii_hexdigit() {
                            return Err(self.error(
                                Some("while scanning a double-quoted scalar"),
                                Some(start),
                                &format!(
                                    "expected escape sequence of {n} hexadecimal numbers, but found {}",
                                    char_repr(self.peek(k))
                                ),
                            ));
                        }
                    }
                    let code = u32::from_str_radix(&self.prefix(n), 16).unwrap_or(0);
                    let Some(c) = char::from_u32(code) else {
                        return Err(YamlError {
                            class: "ValueError",
                            message: "chr() arg not in range(0x110000)".to_string(),
                        });
                    };
                    chunks.push(c);
                    self.forward(n);
                } else if is_in(c, "\r\n\u{85}\u{2028}\u{2029}") {
                    self.scan_line_break();
                    let breaks = self.flow_breaks(start)?;
                    chunks.push_str(&breaks.concat());
                } else {
                    return Err(self.error(
                        Some("while scanning a double-quoted scalar"),
                        Some(start),
                        &format!("found unknown escape character {}", char_repr(c)),
                    ));
                }
            } else {
                return Ok(());
            }
        }
    }

    fn flow_spaces(&mut self, _double: bool, start: Mark, chunks: &mut String) -> SResult<()> {
        let mut length = 0;
        while is_in(self.peek(length), " \t") {
            length += 1;
        }
        let whitespaces = self.prefix(length);
        self.forward(length);
        let ch = self.peek(0);
        if ch == '\0' {
            return Err(self.error(
                Some("while scanning a quoted scalar"),
                Some(start),
                "found unexpected end of stream",
            ));
        } else if is_in(ch, "\r\n\u{85}\u{2028}\u{2029}") {
            let line_break = self.scan_line_break();
            let breaks = self.flow_breaks(start)?;
            if line_break != "\n" {
                chunks.push_str(&line_break);
            } else if breaks.is_empty() {
                chunks.push(' ');
            }
            chunks.push_str(&breaks.concat());
        } else {
            chunks.push_str(&whitespaces);
        }
        Ok(())
    }

    fn flow_breaks(&mut self, start: Mark) -> SResult<Vec<String>> {
        let mut chunks = Vec::new();
        loop {
            let prefix = self.prefix(3);
            if (prefix == "---" || prefix == "...") && is_in(self.peek(3), BLANKZ) {
                return Err(self.error(
                    Some("while scanning a quoted scalar"),
                    Some(start),
                    "found unexpected document separator",
                ));
            }
            while is_in(self.peek(0), " \t") {
                self.forward(1);
            }
            if is_in(self.peek(0), "\r\n\u{85}\u{2028}\u{2029}") {
                chunks.push(self.scan_line_break());
            } else {
                return Ok(chunks);
            }
        }
    }

    fn fetch_plain(&mut self) -> SResult<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let mut chunks = String::new();
        let start = self.mark();
        let mut end = start;
        let indent = (self.indent + 1) as usize;
        let mut spaces: Option<String> = Some(String::new());
        loop {
            let mut length = 0;
            if self.peek(0) == '#' {
                break;
            }
            loop {
                let ch = self.peek(length);
                let stop_colon = ch == ':'
                    && (is_in(self.peek(length + 1), BLANKZ)
                        || (self.flow_level > 0 && is_in(self.peek(length + 1), ",[]{}")));
                if is_in(ch, BLANKZ) || stop_colon || (self.flow_level > 0 && is_in(ch, ",?[]{}")) {
                    break;
                }
                length += 1;
            }
            if length == 0 {
                break;
            }
            self.allow_simple_key = false;
            if let Some(s) = &spaces {
                chunks.push_str(s);
            }
            chunks.push_str(&self.prefix(length));
            self.forward(length);
            end = self.mark();
            spaces = self.plain_spaces();
            match &spaces {
                None => break,
                Some(s) if s.is_empty() => break,
                _ => {}
            }
            if self.peek(0) == '#' || (self.flow_level == 0 && self.column < indent) {
                break;
            }
        }
        self.push(Tok::Scalar(chunks, true), start, end);
        Ok(())
    }

    /// `scan_plain_spaces`: `None` where it returns nothing at a document
    /// separator.
    fn plain_spaces(&mut self) -> Option<String> {
        let mut length = 0;
        while self.peek(length) == ' ' {
            length += 1;
        }
        let whitespaces = self.prefix(length);
        self.forward(length);
        let ch = self.peek(0);
        let mut chunks = String::new();
        if is_in(ch, "\r\n\u{85}\u{2028}\u{2029}") {
            let line_break = self.scan_line_break();
            self.allow_simple_key = true;
            let prefix = self.prefix(3);
            if (prefix == "---" || prefix == "...") && is_in(self.peek(3), BLANKZ) {
                return None;
            }
            let mut breaks = String::new();
            let mut any_break = false;
            while is_in(self.peek(0), " \r\n\u{85}\u{2028}\u{2029}") {
                if self.peek(0) == ' ' {
                    self.forward(1);
                } else {
                    breaks.push_str(&self.scan_line_break());
                    any_break = true;
                    let prefix = self.prefix(3);
                    if (prefix == "---" || prefix == "...") && is_in(self.peek(3), BLANKZ) {
                        return None;
                    }
                }
            }
            if line_break != "\n" {
                chunks.push_str(&line_break);
            } else if !any_break {
                chunks.push(' ');
            }
            chunks.push_str(&breaks);
        } else if !whitespaces.is_empty() {
            chunks.push_str(&whitespaces);
        }
        Some(chunks)
    }
}

// --- The parser and composer ------------------------------------------------

#[derive(Clone, Debug)]
enum Node {
    Scalar {
        tag: String,
        value: String,
        start: Mark,
    },
    Seq {
        tag: String,
        items: Vec<Node>,
        start: Mark,
    },
    Map {
        tag: String,
        pairs: Vec<(Node, Node)>,
        start: Mark,
    },
}

impl Node {
    fn tag(&self) -> &str {
        match self {
            Node::Scalar { tag, .. } | Node::Seq { tag, .. } | Node::Map { tag, .. } => tag,
        }
    }

    fn start(&self) -> Mark {
        match self {
            Node::Scalar { start, .. } | Node::Seq { start, .. } | Node::Map { start, .. } => {
                *start
            }
        }
    }

    fn id(&self) -> &'static str {
        match self {
            Node::Scalar { .. } => "scalar",
            Node::Seq { .. } => "sequence",
            Node::Map { .. } => "mapping",
        }
    }
}

const STR: &str = "tag:yaml.org,2002:str";
const SEQ: &str = "tag:yaml.org,2002:seq";
const MAP: &str = "tag:yaml.org,2002:map";

struct Parser<'a> {
    buffer: &'a [char],
    tokens: Vec<Token>,
    pos: usize,
    anchors: HashMap<String, Node>,
    tag_handles: HashMap<String, String>,
}

type PResult<T> = Result<T, YamlError>;

fn resolve_scalar(value: &str) -> &'static str {
    use regex::Regex;
    use std::sync::OnceLock;
    static RES: OnceLock<Vec<(&'static str, Regex, &'static str)>> = OnceLock::new();
    let res = RES.get_or_init(|| {
        let list: Vec<(&'static str, &'static str, &'static str)> = vec![
            ("tag:yaml.org,2002:bool", r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)\n?\z", "yYnNtTfFoO"),
            ("tag:yaml.org,2002:float", r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))\n?\z", "-+0123456789."),
            ("tag:yaml.org,2002:int", r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)\n?\z", "-+0123456789"),
            ("tag:yaml.org,2002:merge", r"^(?:<<)\z", "<"),
            ("tag:yaml.org,2002:null", r"^(?:~|null|Null|NULL|)\n?\z", "~nN"),
            ("tag:yaml.org,2002:timestamp", r"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)\n?\z", "0123456789"),
            ("tag:yaml.org,2002:value", r"^(?:=)\z", "="),
        ];
        list.into_iter()
            .map(|(tag, pattern, first)| (tag, Regex::new(pattern).expect("resolver pattern compiles"), first))
            .collect()
    });
    if value.is_empty() {
        return "tag:yaml.org,2002:null";
    }
    let first = value.chars().next().unwrap_or('\0');
    for (tag, re, firsts) in res {
        if firsts.contains(first) && re.is_match(value) {
            return tag;
        }
    }
    STR
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn check(&self, f: impl Fn(&Tok) -> bool) -> bool {
        f(&self.peek().tok)
    }

    fn next(&mut self) -> Token {
        let t = self.peek().clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn error(
        &self,
        context: Option<&str>,
        context_mark: Option<Mark>,
        problem: &str,
        mark: Mark,
    ) -> YamlError {
        marked(
            self.buffer,
            "ParserError",
            context,
            context_mark,
            problem,
            Some(mark),
        )
    }

    fn token_repr(t: &Tok) -> String {
        crate::py::str_repr(t.id())
    }

    /// `get_single_data`: the stream's one document, or nothing.
    fn single(&mut self) -> PResult<Option<Node>> {
        self.next(); // stream start
        if matches!(self.peek().tok, Tok::StreamEnd) {
            return Ok(None);
        }
        let doc = self.document()?;
        if !matches!(self.peek().tok, Tok::StreamEnd) {
            // A second document.
            let t = self.peek().clone();
            if matches!(t.tok, Tok::DocumentStart | Tok::Directive) {
                return Err(marked(
                    self.buffer,
                    "ComposerError",
                    Some("expected a single document in the stream"),
                    None,
                    "but found another document",
                    Some(t.start),
                ));
            }
            return Err(self.error(
                None,
                None,
                &format!(
                    "expected '<document start>', but found {}",
                    Self::token_repr(&t.tok)
                ),
                t.start,
            ));
        }
        Ok(Some(doc))
    }

    fn document(&mut self) -> PResult<Node> {
        self.tag_handles = HashMap::from([
            ("!".to_string(), "!".to_string()),
            ("!!".to_string(), "tag:yaml.org,2002:".to_string()),
        ]);
        let explicit = matches!(self.peek().tok, Tok::Directive | Tok::DocumentStart);
        let node;
        if explicit {
            while matches!(self.peek().tok, Tok::Directive) {
                self.next();
            }
            if !matches!(self.peek().tok, Tok::DocumentStart) {
                let t = self.peek().clone();
                return Err(self.error(
                    None,
                    None,
                    &format!(
                        "expected '<document start>', but found {}",
                        Self::token_repr(&t.tok)
                    ),
                    t.start,
                ));
            }
            self.next();
            if self.check(|t| {
                matches!(
                    t,
                    Tok::Directive | Tok::DocumentStart | Tok::DocumentEnd | Tok::StreamEnd
                )
            }) {
                let mark = self.peek().start;
                node = Node::Scalar {
                    tag: "tag:yaml.org,2002:null".to_string(),
                    value: String::new(),
                    start: mark,
                };
            } else {
                node = self.node(true, false)?;
            }
        } else {
            node = self.node(true, false)?;
        }
        while matches!(self.peek().tok, Tok::DocumentEnd) {
            self.next();
        }
        self.anchors.clear();
        Ok(node)
    }

    fn empty(&self, mark: Mark) -> Node {
        Node::Scalar {
            tag: "tag:yaml.org,2002:null".to_string(),
            value: String::new(),
            start: mark,
        }
    }

    fn node(&mut self, block: bool, indentless: bool) -> PResult<Node> {
        if let Tok::Alias(name) = &self.peek().tok {
            let name = name.clone();
            let t = self.next();
            return match self.anchors.get(&name) {
                Some(node) => Ok(node.clone()),
                None => Err(marked(
                    self.buffer,
                    "ComposerError",
                    None,
                    None,
                    &format!("found undefined alias {}", crate::py::str_repr(&name)),
                    Some(t.start),
                )),
            };
        }
        let mut anchor: Option<String> = None;
        let mut tag: Option<(Option<String>, String)> = None;
        let mut start: Option<Mark> = None;
        if let Tok::Anchor(a) = &self.peek().tok {
            anchor = Some(a.clone());
            start = Some(self.next().start);
            if let Tok::Tag(h, s) = &self.peek().tok {
                tag = Some((h.clone(), s.clone()));
                self.next();
            }
        } else if let Tok::Tag(h, s) = &self.peek().tok {
            tag = Some((h.clone(), s.clone()));
            start = Some(self.next().start);
            if let Tok::Anchor(a) = &self.peek().tok {
                anchor = Some(a.clone());
                self.next();
            }
        }
        let tag: Option<String> = match tag {
            None => None,
            Some((None, suffix)) => Some(suffix),
            Some((Some(handle), suffix)) => match self.tag_handles.get(&handle) {
                Some(prefix) => Some(format!("{prefix}{suffix}")),
                None => {
                    return Err(self.error(
                        Some("while parsing a node"),
                        start,
                        &format!(
                            "found undefined tag handle {}",
                            crate::py::str_repr(&handle)
                        ),
                        start.unwrap_or(self.peek().start),
                    ));
                }
            },
        };
        let start = start.unwrap_or(self.peek().start);
        let implicit_tag = tag.is_none() || tag.as_deref() == Some("!");
        let node = if indentless && matches!(self.peek().tok, Tok::BlockEntry) {
            let mut items = Vec::new();
            while matches!(self.peek().tok, Tok::BlockEntry) {
                let t = self.next();
                if self
                    .check(|t| matches!(t, Tok::BlockEntry | Tok::Key | Tok::Value | Tok::BlockEnd))
                {
                    items.push(self.empty(t.end));
                } else {
                    items.push(self.node(true, false)?);
                }
            }
            Node::Seq {
                tag: if implicit_tag {
                    SEQ.to_string()
                } else {
                    tag.clone().unwrap_or_default()
                },
                items,
                start,
            }
        } else if let Tok::Scalar(value, plain) = &self.peek().tok {
            let (value, plain) = (value.clone(), *plain);
            self.next();
            let resolved = match tag.as_deref() {
                None if plain => resolve_scalar(&value).to_string(),
                Some("!") => resolve_scalar(&value).to_string(),
                None => STR.to_string(),
                Some(t) => t.to_string(),
            };
            Node::Scalar {
                tag: resolved,
                value,
                start,
            }
        } else if matches!(self.peek().tok, Tok::FlowSequenceStart) {
            self.flow_sequence(tag.clone().filter(|_| !implicit_tag), start)?
        } else if matches!(self.peek().tok, Tok::FlowMappingStart) {
            self.flow_mapping(tag.clone().filter(|_| !implicit_tag), start)?
        } else if block && matches!(self.peek().tok, Tok::BlockSequenceStart) {
            self.block_sequence(tag.clone().filter(|_| !implicit_tag), start)?
        } else if block && matches!(self.peek().tok, Tok::BlockMappingStart) {
            self.block_mapping(tag.clone().filter(|_| !implicit_tag), start)?
        } else if anchor.is_some() || tag.is_some() {
            let resolved = match tag.as_deref() {
                None | Some("!") => resolve_scalar("").to_string(),
                Some(t) => t.to_string(),
            };
            Node::Scalar {
                tag: resolved,
                value: String::new(),
                start,
            }
        } else {
            let t = self.peek().clone();
            return Err(self.error(
                Some(&format!(
                    "while parsing a {} node",
                    if block { "block" } else { "flow" }
                )),
                Some(start),
                &format!(
                    "expected the node content, but found {}",
                    Self::token_repr(&t.tok)
                ),
                t.start,
            ));
        };
        if let Some(name) = anchor {
            if self.anchors.contains_key(&name) {
                return Err(marked(
                    self.buffer,
                    "ComposerError",
                    Some(&format!(
                        "found duplicate anchor {}; first occurrence",
                        crate::py::str_repr(&name)
                    )),
                    None,
                    "second occurrence",
                    Some(start),
                ));
            }
            self.anchors.insert(name, node.clone());
        }
        Ok(node)
    }

    fn block_sequence(&mut self, tag: Option<String>, start: Mark) -> PResult<Node> {
        let first = self.next();
        let mut items = Vec::new();
        loop {
            if matches!(self.peek().tok, Tok::BlockEntry) {
                let t = self.next();
                if self.check(|t| matches!(t, Tok::BlockEntry | Tok::BlockEnd)) {
                    items.push(self.empty(t.end));
                } else {
                    items.push(self.node(true, false)?);
                }
                continue;
            }
            if !matches!(self.peek().tok, Tok::BlockEnd) {
                let t = self.peek().clone();
                return Err(self.error(
                    Some("while parsing a block collection"),
                    Some(first.start),
                    &format!(
                        "expected <block end>, but found {}",
                        Self::token_repr(&t.tok)
                    ),
                    t.start,
                ));
            }
            self.next();
            break;
        }
        Ok(Node::Seq {
            tag: tag.unwrap_or_else(|| SEQ.to_string()),
            items,
            start,
        })
    }

    fn block_mapping(&mut self, tag: Option<String>, start: Mark) -> PResult<Node> {
        let first = self.next();
        let mut pairs = Vec::new();
        loop {
            let key = if matches!(self.peek().tok, Tok::Key) {
                let t = self.next();
                if self.check(|t| matches!(t, Tok::Key | Tok::Value | Tok::BlockEnd)) {
                    self.empty(t.end)
                } else {
                    self.node(true, true)?
                }
            } else {
                if !matches!(self.peek().tok, Tok::BlockEnd) {
                    let t = self.peek().clone();
                    return Err(self.error(
                        Some("while parsing a block mapping"),
                        Some(first.start),
                        &format!(
                            "expected <block end>, but found {}",
                            Self::token_repr(&t.tok)
                        ),
                        t.start,
                    ));
                }
                self.next();
                break;
            };
            let value = if matches!(self.peek().tok, Tok::Value) {
                let t = self.next();
                if self.check(|t| matches!(t, Tok::Key | Tok::Value | Tok::BlockEnd)) {
                    self.empty(t.end)
                } else {
                    self.node(true, true)?
                }
            } else {
                let mark = self.peek().start;
                self.empty(mark)
            };
            pairs.push((key, value));
        }
        Ok(Node::Map {
            tag: tag.unwrap_or_else(|| MAP.to_string()),
            pairs,
            start,
        })
    }

    fn flow_sequence(&mut self, tag: Option<String>, start: Mark) -> PResult<Node> {
        let first = self.next();
        let mut items = Vec::new();
        let mut is_first = true;
        loop {
            if matches!(self.peek().tok, Tok::FlowSequenceEnd) {
                self.next();
                break;
            }
            if !is_first {
                if matches!(self.peek().tok, Tok::FlowEntry) {
                    self.next();
                } else {
                    let t = self.peek().clone();
                    return Err(self.error(
                        Some("while parsing a flow sequence"),
                        Some(first.start),
                        &format!("expected ',' or ']', but got {}", Self::token_repr(&t.tok)),
                        t.start,
                    ));
                }
            }
            is_first = false;
            if matches!(self.peek().tok, Tok::Key) {
                let t = self.next();
                let key = if self
                    .check(|t| matches!(t, Tok::Value | Tok::FlowEntry | Tok::FlowSequenceEnd))
                {
                    self.empty(t.end)
                } else {
                    self.node(false, false)?
                };
                let value = if matches!(self.peek().tok, Tok::Value) {
                    let v = self.next();
                    if self.check(|t| matches!(t, Tok::FlowEntry | Tok::FlowSequenceEnd)) {
                        self.empty(v.end)
                    } else {
                        self.node(false, false)?
                    }
                } else {
                    let mark = self.peek().start;
                    self.empty(mark)
                };
                items.push(Node::Map {
                    tag: MAP.to_string(),
                    pairs: vec![(key, value)],
                    start: t.start,
                });
            } else if !matches!(self.peek().tok, Tok::FlowSequenceEnd) {
                items.push(self.node(false, false)?);
            }
        }
        Ok(Node::Seq {
            tag: tag.unwrap_or_else(|| SEQ.to_string()),
            items,
            start,
        })
    }

    fn flow_mapping(&mut self, tag: Option<String>, start: Mark) -> PResult<Node> {
        let first = self.next();
        let mut pairs = Vec::new();
        let mut is_first = true;
        loop {
            if matches!(self.peek().tok, Tok::FlowMappingEnd) {
                self.next();
                break;
            }
            if !is_first {
                if matches!(self.peek().tok, Tok::FlowEntry) {
                    self.next();
                } else {
                    let t = self.peek().clone();
                    return Err(self.error(
                        Some("while parsing a flow mapping"),
                        Some(first.start),
                        &format!("expected ',' or '}}', but got {}", Self::token_repr(&t.tok)),
                        t.start,
                    ));
                }
            }
            is_first = false;
            if matches!(self.peek().tok, Tok::Key) {
                let t = self.next();
                let key = if self
                    .check(|t| matches!(t, Tok::Value | Tok::FlowEntry | Tok::FlowMappingEnd))
                {
                    self.empty(t.end)
                } else {
                    self.node(false, false)?
                };
                let value = if matches!(self.peek().tok, Tok::Value) {
                    let v = self.next();
                    if self.check(|t| matches!(t, Tok::FlowEntry | Tok::FlowMappingEnd)) {
                        self.empty(v.end)
                    } else {
                        self.node(false, false)?
                    }
                } else {
                    let mark = self.peek().start;
                    self.empty(mark)
                };
                pairs.push((key, value));
            } else if !matches!(self.peek().tok, Tok::FlowMappingEnd) {
                let key = self.node(false, false)?;
                let mark = self.peek().start;
                pairs.push((key, self.empty(mark)));
            }
        }
        Ok(Node::Map {
            tag: tag.unwrap_or_else(|| MAP.to_string()),
            pairs,
            start,
        })
    }
}

// --- The constructor ---------------------------------------------------------

struct Constructor<'a> {
    buffer: &'a [char],
}

impl Constructor<'_> {
    fn error(
        &self,
        context: Option<&str>,
        context_mark: Option<Mark>,
        problem: &str,
        mark: Mark,
    ) -> YamlError {
        marked(
            self.buffer,
            "ConstructorError",
            context,
            context_mark,
            problem,
            Some(mark),
        )
    }

    fn scalar<'n>(&self, node: &'n Node) -> Result<&'n str, YamlError> {
        match node {
            Node::Scalar { value, .. } => Ok(value),
            Node::Map { pairs, .. } => {
                for (k, v) in pairs {
                    if k.tag() == "tag:yaml.org,2002:value" {
                        return self.scalar(v);
                    }
                }
                Err(self.error(
                    None,
                    None,
                    &format!("expected a scalar node, but found {}", node.id()),
                    node.start(),
                ))
            }
            other => Err(self.error(
                None,
                None,
                &format!("expected a scalar node, but found {}", other.id()),
                other.start(),
            )),
        }
    }

    fn construct(&self, node: &Node) -> Result<PyVal, YamlError> {
        let tag = node.tag();
        match tag {
            "tag:yaml.org,2002:null" => {
                self.scalar(node)?;
                Ok(PyVal::None)
            }
            "tag:yaml.org,2002:bool" => {
                let v = self.scalar(node)?.to_lowercase();
                Ok(PyVal::Bool(matches!(v.as_str(), "yes" | "true" | "on")))
            }
            "tag:yaml.org,2002:int" => construct_int(self.scalar(node)?),
            "tag:yaml.org,2002:float" => Ok(PyVal::Float(construct_float(self.scalar(node)?))),
            "tag:yaml.org,2002:timestamp" => construct_timestamp(self.scalar(node)?),
            "tag:yaml.org,2002:str" => Ok(PyVal::Str(self.scalar(node)?.to_string())),
            "tag:yaml.org,2002:seq" => match node {
                Node::Seq { items, .. } => Ok(PyVal::List(
                    items
                        .iter()
                        .map(|n| self.construct(n))
                        .collect::<Result<_, _>>()?,
                )),
                other => Err(self.error(
                    None,
                    None,
                    &format!("expected a sequence node, but found {}", other.id()),
                    other.start(),
                )),
            },
            "tag:yaml.org,2002:map" => self.mapping(node),
            other => Err(self.error(
                None,
                None,
                &format!(
                    "could not determine a constructor for the tag {}",
                    crate::py::str_repr(other)
                ),
                node.start(),
            )),
        }
    }

    fn flatten(&self, pairs: &[(Node, Node)], start: Mark) -> Result<Vec<(Node, Node)>, YamlError> {
        let mut merge: Vec<(Node, Node)> = Vec::new();
        let mut rest: Vec<(Node, Node)> = Vec::new();
        for (k, v) in pairs {
            if k.tag() == "tag:yaml.org,2002:merge" {
                match v {
                    Node::Map {
                        pairs, start: s, ..
                    } => merge.extend(self.flatten(pairs, *s)?),
                    Node::Seq { items, .. } => {
                        let mut submerge = Vec::new();
                        for sub in items {
                            match sub {
                                Node::Map {
                                    pairs, start: s, ..
                                } => submerge.push(self.flatten(pairs, *s)?),
                                other => {
                                    return Err(self.error(
                                        Some("while constructing a mapping"),
                                        Some(start),
                                        &format!(
                                            "expected a mapping for merging, but found {}",
                                            other.id()
                                        ),
                                        other.start(),
                                    ));
                                }
                            }
                        }
                        submerge.reverse();
                        for value in submerge {
                            merge.extend(value);
                        }
                    }
                    other => {
                        return Err(self.error(
                            Some("while constructing a mapping"),
                            Some(start),
                            &format!(
                                "expected a mapping or list of mappings for merging, but found {}",
                                other.id()
                            ),
                            other.start(),
                        ));
                    }
                }
            } else if k.tag() == "tag:yaml.org,2002:value" {
                let mut key = k.clone();
                if let Node::Scalar { tag, .. } = &mut key {
                    *tag = STR.to_string();
                }
                rest.push((key, v.clone()));
            } else {
                rest.push((k.clone(), v.clone()));
            }
        }
        merge.extend(rest);
        Ok(merge)
    }

    fn mapping(&self, node: &Node) -> Result<PyVal, YamlError> {
        let Node::Map { pairs, start, .. } = node else {
            return Err(self.error(
                None,
                None,
                &format!("expected a mapping node, but found {}", node.id()),
                node.start(),
            ));
        };
        let pairs = self.flatten(pairs, *start)?;
        let mut out: Vec<(PyVal, PyVal)> = Vec::new();
        for (k, v) in &pairs {
            let key = self.construct(k)?;
            if matches!(key, PyVal::List(_) | PyVal::Dict(_)) {
                return Err(self.error(
                    Some("while constructing a mapping"),
                    Some(*start),
                    "found unhashable key",
                    k.start(),
                ));
            }
            let value = self.construct(v)?;
            let id = key.eq_key();
            if let Some(slot) = out.iter_mut().find(|(existing, _)| existing.eq_key() == id) {
                slot.1 = value;
            } else {
                out.push((key, value));
            }
        }
        Ok(PyVal::Dict(out))
    }
}

fn construct_int(text: &str) -> Result<PyVal, YamlError> {
    let value: String = text.chars().filter(|&c| c != '_').collect();
    let mut sign: i128 = 1;
    let mut body = value.as_str();
    if body.starts_with('-') {
        sign = -1;
    }
    if body.starts_with(['+', '-']) {
        body = &body[1..];
    }
    let parse = |digits: &str, radix: u32| -> Result<PyVal, YamlError> {
        match i128::from_str_radix(digits, radix) {
            Ok(n) => Ok(PyVal::Int(sign * n)),
            Err(_) => Err(YamlError {
                class: "ValueError",
                message: format!(
                    "invalid literal for int() with base {radix}: {}",
                    crate::py::str_repr(digits)
                ),
            }),
        }
    };
    if body == "0" {
        return Ok(PyVal::Int(0));
    }
    if let Some(rest) = body.strip_prefix("0b") {
        return parse(rest, 2);
    }
    if let Some(rest) = body.strip_prefix("0x") {
        return parse(rest, 16);
    }
    if body.starts_with('0') {
        return parse(body, 8);
    }
    if body.contains(':') {
        let mut total: i128 = 0;
        for part in body.split(':') {
            let digit: i128 = part.parse().unwrap_or(0);
            total = total * 60 + digit;
        }
        return Ok(PyVal::Int(sign * total));
    }
    match body.parse::<i128>() {
        Ok(n) => Ok(PyVal::Int(sign * n)),
        Err(_) => Ok(PyVal::BigInt(format!(
            "{}{}",
            if sign < 0 { "-" } else { "" },
            body.trim_start_matches('0')
        ))),
    }
}

fn construct_float(text: &str) -> f64 {
    let value: String = text
        .chars()
        .filter(|&c| c != '_')
        .collect::<String>()
        .to_lowercase();
    let mut sign = 1.0;
    let mut body = value.as_str();
    if body.starts_with('-') {
        sign = -1.0;
    }
    if body.starts_with(['+', '-']) {
        body = &body[1..];
    }
    if body == ".inf" {
        return sign * f64::INFINITY;
    }
    if body == ".nan" {
        return f64::NAN;
    }
    if body.contains(':') {
        let mut total = 0.0;
        for part in body.split(':') {
            let digit = crate::py::float(part).unwrap_or(0.0);
            total = total * 60.0 + digit;
        }
        return sign * total;
    }
    sign * crate::py::float(body).unwrap_or(f64::NAN)
}

fn construct_timestamp(text: &str) -> Result<PyVal, YamlError> {
    use regex::Regex;
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"^(?P<year>[0-9][0-9][0-9][0-9])-(?P<month>[0-9][0-9]?)-(?P<day>[0-9][0-9]?)(?:(?:[Tt]|[ \t]+)(?P<hour>[0-9][0-9]?):(?P<minute>[0-9][0-9]):(?P<second>[0-9][0-9])(?:\.(?P<fraction>[0-9]*))?(?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)(?::(?P<tz_minute>[0-9][0-9]))?))?)?\n?\z",
        )
        .expect("timestamp pattern compiles")
    });
    let Some(m) = re.captures(text) else {
        return Err(YamlError {
            class: "AttributeError",
            message: "'NoneType' object has no attribute 'groupdict'".to_string(),
        });
    };
    let num = |name: &str| -> i64 { m.name(name).map_or(0, |v| v.as_str().parse().unwrap_or(0)) };
    let (year, month, day) = (num("year"), num("month"), num("day"));
    let date_error = |message: String| YamlError {
        class: "ValueError",
        message,
    };
    if !(1..=12).contains(&month) {
        return Err(date_error("month must be in 1..12".to_string()));
    }
    if !(1..=9999).contains(&year) {
        return Err(date_error(format!("year {year} is out of range")));
    }
    if day < 1 || day > i64::from(civil::days_in_month(year, month as u32)) {
        return Err(date_error("day is out of range for month".to_string()));
    }
    let Some(hour) = m.name("hour").filter(|h| !h.as_str().is_empty()) else {
        return Ok(PyVal::Date(year, month as u32, day as u32));
    };
    let hour: i64 = hour.as_str().parse().unwrap_or(0);
    let (minute, second) = (num("minute"), num("second"));
    if hour > 23 {
        return Err(date_error("hour must be in 0..23".to_string()));
    }
    if minute > 59 {
        return Err(date_error("minute must be in 0..59".to_string()));
    }
    if second > 59 {
        return Err(date_error("second must be in 0..59".to_string()));
    }
    let mut micro = 0u32;
    if let Some(fraction) = m
        .name("fraction")
        .map(|f| f.as_str())
        .filter(|f| !f.is_empty())
    {
        let six: String = format!("{fraction:0<6}").chars().take(6).collect();
        micro = six.parse().unwrap_or(0);
    }
    let tz = if let Some(sign) = m.name("tz_sign") {
        let offset = num("tz_hour") * 3600 + num("tz_minute") * 60;
        let offset = if sign.as_str() == "-" {
            -offset
        } else {
            offset
        };
        if offset.abs() >= 86_400 {
            return Err(date_error("offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).".to_string()));
        }
        Some(offset)
    } else if m.name("tz").is_some() {
        Some(0)
    } else {
        None
    };
    Ok(PyVal::DateTime(
        Wall {
            year,
            month: month as u32,
            day: day as u32,
            hour: hour as u32,
            minute: minute as u32,
            second: second as u32,
            nano: micro * 1000,
        },
        tz,
    ))
}

/// PyYAML's reader refuses characters it calls special.
fn check_printable(buffer: &[char]) -> Result<(), YamlError> {
    for (i, &c) in buffer.iter().enumerate() {
        let ok = matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{7e}' | '\u{85}' | '\u{a0}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}');
        if !ok {
            return Err(YamlError {
                class: "ReaderError",
                message: format!(
                    "unacceptable character #x{:04x}: special characters are not allowed\n  in \"<unicode string>\", position {i}",
                    c as u32
                ),
            });
        }
    }
    Ok(())
}

/// `yaml.safe_load(text)`.
pub fn safe_load(text: &str) -> Result<PyVal, YamlError> {
    let mut buffer: Vec<char> = text.chars().collect();
    check_printable(&buffer)?;
    buffer.push('\0');
    let mut scanner = Scanner::new(&buffer);
    let tokens = scanner.scan_all();
    // PyYAML scans lazily and reports the parser's error where one comes
    // before the scanner's; scanning ahead keeps what was read before it.
    let (tokens, scan_error) = match tokens {
        Ok(tokens) => (tokens, None),
        Err(error) => {
            let mut partial = std::mem::take(&mut scanner.tokens);
            let mark = scanner.mark();
            partial.push(Token {
                tok: Tok::StreamEnd,
                start: mark,
                end: mark,
            });
            (partial, Some(error))
        }
    };
    let mut parser = Parser {
        buffer: &buffer,
        tokens,
        pos: 0,
        anchors: HashMap::new(),
        tag_handles: HashMap::new(),
    };
    let node = match parser.single() {
        Ok(node) => node,
        Err(error) => return Err(scan_error.unwrap_or(error)),
    };
    if let Some(error) = scan_error {
        return Err(error);
    }
    let Some(node) = node else {
        return Ok(PyVal::None);
    };
    Constructor { buffer: &buffer }.construct(&node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_yaml_one_one() {
        let v =
            safe_load("on: yes\nx: 0x1A\ny: 1:30\nz: 2021-01-01\nw: [a, 'b', \"c\"]\n").unwrap();
        assert_eq!(
            v.repr(),
            "{True: True, 'x': 26, 'y': 90, 'z': datetime.date(2021, 1, 1), 'w': ['a', 'b', 'c']}"
        );
        let v = safe_load("- name: a\n  joins:\n  - name: b\n").unwrap();
        assert_eq!(v.repr(), "[{'name': 'a', 'joins': [{'name': 'b'}]}]");
        assert_eq!(safe_load("").unwrap(), PyVal::None);
    }
}
