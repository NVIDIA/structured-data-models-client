// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How each dialect splits an expression into tokens
//! (`RE/view_expr.py:20-189`). Every check reads quoted text through these,
//! so no two can disagree about where a literal ends. A `Str` is data, and
//! a `Quoted` token is a name.

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::{ExprDialect, stripped};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Comment,
    Str,
    Quoted,
    Number,
    Word,
    Punct,
    Other,
    /// Stands past the end of the tokens, and before the first.
    End,
}

/// A token, with where it starts and ends in the expression, in bytes.
#[derive(Clone, Debug)]
pub(crate) struct Token {
    pub kind: Kind,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl Token {
    /// A word's text in lower case; nothing for any other token.
    pub fn folded(&self) -> String {
        if self.kind == Kind::Word {
            self.text.to_lowercase()
        } else {
            String::new()
        }
    }

    /// The name a word or quoted token gives, unquoted.
    pub fn unquoted(&self) -> String {
        if self.kind == Kind::Quoted {
            let quote = &self.text[..1];
            let inner = &self.text[1..self.text.len() - 1];
            inner.replace(&quote.repeat(2), quote)
        } else {
            self.text.clone()
        }
    }

    /// The name, folded, a word or quoted token gives; nothing else does.
    pub fn folded_name(&self) -> String {
        match self.kind {
            Kind::Word | Kind::Quoted => self.unquoted().to_lowercase(),
            _ => String::new(),
        }
    }
}

/// What a lexeme of the pattern is read as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lexeme {
    Is(Kind),
    /// `/*`, whose end is found by hand: Databricks nests these.
    Open,
    Space,
}

const ESCAPED_STRING: &str = r"'(?:''|\\[\s\S]|[^'\\])*'";
const ESCAPED_DOUBLE: &str = r#""(?:""|\\[\s\S]|[^"\\])*""#;
const BACKTICKED: &str = r"`(?:``|[^`])*`";
const DOLLARS: &str = r"\$\$[\s\S]*?\$\$";
const WORD: &str = r"[A-Za-z_][\w$]*";
/// Databricks reads a run of letters, digits and `_` as a name unless the
/// whole run is a number, as `1e5`, `2d` and `10L` are.
const WHOLE_NUMBER: &str = r"(?i:\d+(?:e\d+)?(?:bd|d|f)?|\d+[lsy])(?![A-Za-z0-9_])";

/// A run Databricks reads as a name (`BARE_NAME`).
pub(crate) fn bare_name() -> String {
    format!("(?!{WHOLE_NUMBER})[A-Za-z0-9_]+")
}

/// A signed exponent outruns any name, and a decimal point ends a number
/// only where no name character follows, as in Spark's lexer.
fn databricks_number() -> String {
    format!(
        r"(?i:\d+e[+-]\d+(?:bd|d|f)?|(?:\d+\.\d*|\.\d+)(?:e[+-]?\d+)?(?:bd|d|f)?(?![A-Za-z0-9_]))|{WHOLE_NUMBER}"
    )
}

/// Words and numbers elsewhere. A word may hold `$`, as a Snowflake name
/// does, so `a$$` opens no literal.
fn loose_words() -> Vec<(Lexeme, String)> {
    vec![
        (Lexeme::Open, r"/\*".into()),
        (
            Lexeme::Is(Kind::Number),
            r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*".into(),
        ),
        (Lexeme::Is(Kind::Word), WORD.into()),
    ]
}

struct Lexer {
    pattern: Regex,
    lexemes: Vec<Lexeme>,
}

impl Lexer {
    fn new(own: Vec<(Lexeme, String)>) -> Lexer {
        // `?::` is Databricks' try-cast, read wherever `::` is.
        let everywhere = [
            (
                Lexeme::Is(Kind::Punct),
                r"<=>|<=|>=|<>|\?::|::|->|=>|[.:(),\[\]<>]".to_owned(),
            ),
            (Lexeme::Space, r"\s+".to_owned()),
            (Lexeme::Is(Kind::Other), r"[\s\S]".to_owned()),
        ];
        let all: Vec<(Lexeme, String)> = own.into_iter().chain(everywhere).collect();
        let pattern = all
            .iter()
            .map(|(_, lexeme)| format!("({lexeme})"))
            .collect::<Vec<_>>()
            .join("|");
        Lexer {
            pattern: Regex::new(&format!("^(?:{pattern})")).unwrap(),
            lexemes: all.into_iter().map(|(lexeme, _)| lexeme).collect(),
        }
    }
}

fn lexer(dialect: Option<ExprDialect>) -> &'static Lexer {
    static DATABRICKS: LazyLock<Lexer> = LazyLock::new(|| {
        Lexer::new(vec![
            // A line comment goes on past a backslash ending its line.
            (Lexeme::Is(Kind::Comment), r"--(?:\\\n|[^\r\n])*".into()),
            // A raw string takes no escapes; double quotes delimit a
            // string, not a name.
            (
                Lexeme::Is(Kind::Str),
                format!(r#"[rR]'[^']*'|[rR]"[^"]*"|{ESCAPED_STRING}|{ESCAPED_DOUBLE}"#),
            ),
            (Lexeme::Is(Kind::Quoted), BACKTICKED.into()),
            (Lexeme::Open, r"/\*".into()),
            (Lexeme::Is(Kind::Number), databricks_number()),
            (Lexeme::Is(Kind::Word), bare_name()),
        ])
    });
    // Double quotes delimit a name, which takes no escapes. Snowflake
    // refuses a backtick, so reading one as a quote hides nothing from it.
    static SNOWFLAKE: LazyLock<Lexer> = LazyLock::new(|| {
        let mut own = vec![
            (Lexeme::Is(Kind::Comment), r"(?:--|//)[^\r\n]*".to_owned()),
            (Lexeme::Is(Kind::Str), format!("{DOLLARS}|{ESCAPED_STRING}")),
            (
                Lexeme::Is(Kind::Quoted),
                format!(r#""(?:""|[^"])*"|{BACKTICKED}"#),
            ),
        ];
        own.extend(loose_words());
        Lexer::new(own)
    });
    // What a check that knows no dialect reads as quoted.
    static NEITHER: LazyLock<Lexer> = LazyLock::new(|| {
        let mut own = vec![
            (Lexeme::Is(Kind::Comment), r"--[^\r\n]*".to_owned()),
            (
                Lexeme::Is(Kind::Str),
                format!("{DOLLARS}|{ESCAPED_STRING}|{ESCAPED_DOUBLE}"),
            ),
            (Lexeme::Is(Kind::Quoted), BACKTICKED.to_owned()),
        ];
        own.extend(loose_words());
        Lexer::new(own)
    });
    match dialect {
        Some(ExprDialect::Databricks) => &DATABRICKS,
        Some(ExprDialect::Snowflake) => &SNOWFLAKE,
        None => &NEITHER,
    }
}

/// Where the `/*` comment at `start` ends, or `None` if it never does.
/// Databricks nests one such comment in another.
fn comment_end(expr: &str, start: usize, nested: bool) -> Option<usize> {
    let (mut depth, mut at) = (1, start + 2);
    while depth > 0 {
        let close = at + expr[at..].find("*/")?;
        let open = if nested {
            expr[at..close].find("/*").map(|found| at + found)
        } else {
            None
        };
        match open {
            Some(open) => {
                depth += 1;
                at = open + 2;
            }
            None => {
                depth -= 1;
                at = close + 2;
            }
        }
    }
    Some(at)
}

/// The tokens of `expr` as `dialect` reads them, spaces left out. A quote
/// or `/*` that is never closed is a token of kind `Other`.
pub(crate) fn lex(expr: &str, dialect: Option<ExprDialect>) -> Vec<Token> {
    let lexer = lexer(dialect);
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < expr.len() {
        // Some lexeme matches any character, so a match always exists.
        let captures = lexer
            .pattern
            .captures(&expr[at..])
            .ok()
            .flatten()
            .expect("a lexeme matches any character");
        let group = (1..=lexer.lexemes.len())
            .find(|&group| captures.get(group).is_some())
            .unwrap();
        let mut end = at + captures.get(group).unwrap().end();
        let kind = match lexer.lexemes[group - 1] {
            Lexeme::Space => None,
            Lexeme::Is(kind) => Some(kind),
            Lexeme::Open => match comment_end(expr, at, dialect == Some(ExprDialect::Databricks)) {
                Some(close) => {
                    end = close;
                    Some(Kind::Comment)
                }
                None => {
                    end = at + 1;
                    Some(Kind::Other)
                }
            },
        };
        if let Some(kind) = kind {
            tokens.push(Token {
                kind,
                text: expr[at..end].to_owned(),
                start: at,
                end,
            });
        }
        at = end;
    }
    tokens
}

/// A Databricks expression with each comment replaced by a space, trimmed
/// (`strip_comments`). A comment hides nothing from a check of what is
/// left, and what is left is what runs.
pub fn strip_comments(expr: &str) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for token in lex(expr, Some(ExprDialect::Databricks)) {
        if token.kind == Kind::Comment {
            out.push_str(&expr[last..token.start]);
            out.push(' ');
            last = token.end;
        }
    }
    out.push_str(&expr[last..]);
    stripped(&out).to_owned()
}

/// `expr` with each string literal and quoted name replaced by `mask`.
pub(crate) fn masked(expr: &str, dialect: Option<ExprDialect>, mask: &str) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for token in lex(expr, dialect) {
        if matches!(token.kind, Kind::Str | Kind::Quoted) {
            out.push_str(&expr[last..token.start]);
            out.push_str(mask);
            last = token.end;
        }
    }
    out.push_str(&expr[last..]);
    out
}

/// Where each string literal of a Databricks expression starts and ends,
/// in bytes.
pub fn string_literals(expr: &str) -> Vec<std::ops::Range<usize>> {
    lex(expr, Some(ExprDialect::Databricks))
        .into_iter()
        .filter(|token| token.kind == Kind::Str)
        .map(|token| token.start..token.end)
        .collect()
}
