// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ast.literal_eval` of the Python literals an endpoint writes a key list
//! as: lists, tuples, text in either quote, numbers, `None`, `True` and
//! `False`. Anything else is not read (`None`), as `literal_eval` raises.

use sdm_schema::pyval::PyVal;

struct Reader<'a> {
    chars: Vec<char>,
    at: usize,
    _text: &'a str,
}

impl Reader<'_> {
    fn skip_space(&mut self) {
        while self.at < self.chars.len() && self.chars[self.at].is_whitespace() {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn value(&mut self, depth: usize) -> Option<PyVal> {
        if depth > 100 {
            return None;
        }
        self.skip_space();
        match self.peek()? {
            '[' | '(' => {
                let close = if self.peek()? == '[' { ']' } else { ')' };
                let tuple = close == ')';
                self.at += 1;
                let mut items = Vec::new();
                let mut trailing_comma = false;
                loop {
                    self.skip_space();
                    if self.peek()? == close {
                        self.at += 1;
                        break;
                    }
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    match self.peek()? {
                        ',' => {
                            self.at += 1;
                            trailing_comma = true;
                        }
                        c if c == close => {
                            self.at += 1;
                            trailing_comma = false;
                            break;
                        }
                        _ => return None,
                    }
                }
                if tuple {
                    if items.len() == 1 && !trailing_comma {
                        return items.pop();
                    }
                    return Some(PyVal::Tuple(items));
                }
                Some(PyVal::List(items))
            }
            '\'' | '"' => self.string(),
            _ => self.atom(),
        }
    }

    fn string(&mut self) -> Option<PyVal> {
        let quote = self.peek()?;
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.at += 1;
            if c == quote {
                break;
            }
            if c == '\n' {
                return None;
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let e = self.peek()?;
            self.at += 1;
            match e {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                '0' => out.push('\0'),
                'a' => out.push('\u{7}'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'v' => out.push('\u{b}'),
                'x' | 'u' | 'U' => {
                    let width = match e {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let digits: String = self.chars.get(self.at..self.at + width)?.iter().collect();
                    self.at += width;
                    out.push(char::from_u32(u32::from_str_radix(&digits, 16).ok()?)?);
                }
                '\n' => {}
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        }
        Some(PyVal::Str(out))
    }

    fn atom(&mut self) -> Option<PyVal> {
        let start = self.at;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || matches!(c, '.' | '_' | '+' | '-') {
                self.at += 1;
            } else {
                break;
            }
        }
        let word: String = self.chars[start..self.at].iter().collect();
        match word.as_str() {
            "None" => Some(PyVal::None),
            "True" => Some(PyVal::Bool(true)),
            "False" => Some(PyVal::Bool(false)),
            text => {
                if let Some(n) = crate::pyops::parse_int(text) {
                    return Some(crate::wire::dec_to_pyval(&n));
                }
                let lowered = text.to_ascii_lowercase();
                if lowered.contains("inf") || lowered.contains("nan") {
                    return None;
                }
                crate::pyops::parse_float(text).map(PyVal::Float)
            }
        }
    }
}

/// `ast.literal_eval(text)`, for the literals above.
pub fn literal_eval(text: &str) -> Option<PyVal> {
    let mut reader = Reader {
        chars: text.chars().collect(),
        at: 0,
        _text: text,
    };
    let value = reader.value(0)?;
    reader.skip_space();
    (reader.at == reader.chars.len()).then_some(value)
}
