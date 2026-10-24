// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How each dialect splits an expression into tokens
//! (`RE/view_expr.py:20-189`). Every check reads quoted text through these,
//! so no two can disagree about where a literal ends. A `Str` is data, and
//! a `Quoted` token is a name.

use std::sync::LazyLock;

use super::names::Unreadable;
use super::pattern::{Pattern, Read};
use super::{ExprDialect, Spaces};

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

/// How a lexeme is matched: by its pattern, or, for Databricks' numbers
/// and names, by hand. Those two end with a lookahead that a megabyte-long
/// run of digits or letters overflows the pattern engine's stack on (T3.5
/// fix 1); since the lookahead only asks that the run of name characters
/// end there, it is decided on that run exactly.
enum Matcher {
    Pattern(Pattern),
    DatabricksNumber,
    DatabricksName,
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

/// The markers of the two lexemes matched by hand.
const DATABRICKS_NUMBER: &str = "<databricks number>";
const DATABRICKS_NAME: &str = "<databricks name>";

/// Whether `c` is a character of a Databricks name: `[A-Za-z0-9_]`.
fn name_character(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The run of name characters `text` starts with, in bytes; `None` where
/// a digit outside ASCII stands at its start or right after it, where the
/// old client's `\d` reads further than the run, so the pattern decides.
fn run(text: &str) -> Option<usize> {
    let end = text
        .find(|c: char| !name_character(c))
        .unwrap_or(text.len());
    let other_digit = |c: char| !c.is_ascii() && c.is_numeric();
    if text[end..].chars().next().is_some_and(other_digit)
        || text.chars().next().is_some_and(other_digit)
    {
        return None;
    }
    Some(end)
}

/// Whether a whole run is `WHOLE_NUMBER`'s: digits, an optional `e` and
/// digits, an optional `bd`, `d` or `f`; or digits and one of `l`, `s`,
/// `y`; in any case.
fn whole_number(run: &str) -> bool {
    let lower = run.to_ascii_lowercase();
    let digits = lower.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let rest = &lower[digits..];
    if matches!(rest, "l" | "s" | "y") {
        return true;
    }
    let rest = match rest.strip_prefix('e') {
        Some(after) => {
            let exponent = after.bytes().take_while(u8::is_ascii_digit).count();
            if exponent == 0 {
                return false;
            }
            &after[exponent..]
        }
        None => rest,
    };
    matches!(rest, "" | "bd" | "d" | "f")
}

/// `BARE_NAME` at the start of `text`, decided on its run: the run, unless
/// the whole run is a number.
fn name_length(text: &str) -> Read<Option<usize>> {
    let Some(end) = run(text) else {
        return fallback(&bare_name(), text);
    };
    Ok((end > 0 && !whole_number(&text[..end])).then_some(end))
}

/// `_DATABRICKS_NUMBER` at the start of `text`, its three forms tried in
/// order: digits, a signed exponent and a suffix, with nothing asked of what
/// follows; a decimal with an optional exponent and suffix, which must end
/// the run; a whole number, which must be the whole run.
fn number_length(text: &str) -> Read<Option<usize>> {
    let Some(end) = run(text) else {
        return fallback(&databricks_number(), text);
    };
    let bytes = text.as_bytes();
    let digits_at = |at: usize| {
        bytes[at.min(bytes.len())..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let lower = |at: usize| bytes.get(at).map(u8::to_ascii_lowercase);
    // The suffixes a number may end with at `at`, longest first.
    let suffixes = |at: usize| -> Vec<usize> {
        let mut found = Vec::new();
        if lower(at) == Some(b'b') && lower(at + 1) == Some(b'd') {
            found.push(2);
        }
        if matches!(lower(at), Some(b'd' | b'f')) {
            found.push(1);
        }
        found.push(0);
        found
    };
    // `\d+e[+-]\d+(?:bd|d|f)?`
    let whole = digits_at(0);
    if whole > 0 && lower(whole) == Some(b'e') && matches!(bytes.get(whole + 1), Some(b'+' | b'-'))
    {
        let exponent = digits_at(whole + 2);
        if exponent > 0 {
            let at = whole + 2 + exponent;
            return Ok(Some(at + suffixes(at)[0]));
        }
    }
    // `(?:\d+\.\d*|\.\d+)(?:e[+-]?\d+)?(?:bd|d|f)?(?![A-Za-z0-9_])`
    let mantissa = if whole > 0 && bytes.get(whole) == Some(&b'.') {
        Some(whole + 1 + digits_at(whole + 1))
    } else if whole == 0 && bytes.first() == Some(&b'.') && digits_at(1) > 0 {
        Some(1 + digits_at(1))
    } else {
        None
    };
    if let Some(mantissa) = mantissa {
        let ends = |at: usize| !text[at..].chars().next().is_some_and(name_character);
        let mut exponents = Vec::new();
        if lower(mantissa) == Some(b'e') {
            let signed = usize::from(matches!(bytes.get(mantissa + 1), Some(b'+' | b'-')));
            let digits = digits_at(mantissa + 1 + signed);
            if digits > 0 {
                exponents.push(mantissa + 1 + signed + digits);
            }
        }
        exponents.push(mantissa);
        for at in exponents {
            if let Some(length) = suffixes(at)
                .into_iter()
                .map(|s| at + s)
                .find(|&end| ends(end))
            {
                return Ok(Some(length));
            }
        }
    }
    // `WHOLE_NUMBER`.
    Ok((end > 0 && whole_number(&text[..end])).then_some(end))
}

/// The pattern's own answer, for the runs it alone reads right.
fn fallback(pattern: &str, text: &str) -> Read<Option<usize>> {
    Ok(Pattern::new(&format!("^(?:{pattern})"))
        .find(text)?
        .map(|span| span.end))
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

/// A dialect's lexemes, each its own pattern anchored at the token's
/// start, tried in the order the old client's alternation tries them, so the
/// first that matches wins as it does there.
struct Lexer {
    lexemes: Vec<(Lexeme, Matcher)>,
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
        Lexer {
            lexemes: own
                .into_iter()
                .chain(everywhere)
                .map(|(lexeme, text)| {
                    let matcher = match text.as_str() {
                        DATABRICKS_NUMBER => Matcher::DatabricksNumber,
                        DATABRICKS_NAME => Matcher::DatabricksName,
                        _ => Matcher::Pattern(Pattern::new(&format!("^(?:{text})"))),
                    };
                    (lexeme, matcher)
                })
                .collect(),
        }
    }

    /// The first lexeme matching at the start of `text`, and its length.
    fn next(&self, text: &str) -> Read<(Lexeme, usize)> {
        for (lexeme, matcher) in &self.lexemes {
            let length = match matcher {
                Matcher::Pattern(pattern) => pattern.find(text)?.map(|span| span.end),
                Matcher::DatabricksNumber => number_length(text)?,
                Matcher::DatabricksName => name_length(text)?,
            };
            if let Some(length) = length {
                return Ok((*lexeme, length));
            }
        }
        // The last lexeme matches any character.
        Err(Unreadable::TooDeep)
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
            (Lexeme::Is(Kind::Number), DATABRICKS_NUMBER.to_owned()),
            (Lexeme::Is(Kind::Word), DATABRICKS_NAME.to_owned()),
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
/// or `/*` that is never closed is a token of kind `Other`. A pattern that
/// fails on the text makes it unreadable.
pub(crate) fn lex(expr: &str, dialect: Option<ExprDialect>) -> Read<Vec<Token>> {
    let lexer = lexer(dialect);
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < expr.len() {
        let (lexeme, length) = lexer.next(&expr[at..])?;
        let mut end = at + length;
        let kind = match lexeme {
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
    Ok(tokens)
}

/// A Databricks expression with each comment replaced by a space, trimmed
/// by Python's white space, `spaces` (`strip_comments`). A comment hides
/// nothing from a check of what is left, and what is left is what runs.
pub fn strip_comments(expr: &str, spaces: Spaces) -> Result<String, Unreadable> {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for token in lex(expr, Some(ExprDialect::Databricks))? {
        if token.kind == Kind::Comment {
            out.push_str(&expr[last..token.start]);
            out.push(' ');
            last = token.end;
        }
    }
    out.push_str(&expr[last..]);
    Ok(spaces.strip(&out).to_owned())
}

/// `expr` with each string literal and quoted name replaced by `mask`.
pub(crate) fn masked(expr: &str, dialect: Option<ExprDialect>, mask: &str) -> Read<String> {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for token in lex(expr, dialect)? {
        if matches!(token.kind, Kind::Str | Kind::Quoted) {
            out.push_str(&expr[last..token.start]);
            out.push_str(mask);
            last = token.end;
        }
    }
    out.push_str(&expr[last..]);
    Ok(out)
}

/// Where each string literal of a Databricks expression starts and ends,
/// in bytes.
pub fn string_literals(expr: &str) -> Result<Vec<std::ops::Range<usize>>, Unreadable> {
    Ok(lex(expr, Some(ExprDialect::Databricks))?
        .into_iter()
        .filter(|token| token.kind == Kind::Str)
        .map(|token| token.start..token.end)
        .collect())
}
