// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Why an expression is unsafe to splice into a statement, and why it
//! calls a function that is not trusted (`RE/view_expr.py:192-1044`).

use std::sync::LazyLock;

use super::lex::{Kind, lex, masked};
use super::names::Unreadable;
use super::pattern::{Pattern, Read};
use super::words::{self, has};
use super::{ExprDialect, rstrip};

/// Why an expression is refused (old-behaviour rows 64 to 66), typed so
/// the core words it as the old client did (`sdm-core::view_words`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprRefusal {
    /// It holds `;`, `--`, `/*`, `*/`, `|>` or `->>`, anywhere.
    Contains(&'static str),
    /// `$$` on Databricks, which quotes nothing with it.
    DollarQuotes,
    /// A Snowflake comment, `//` or another.
    SlashComment,
    /// A backslash inside a Snowflake double-quoted name.
    BackslashInQuotedName,
    /// A character outside quotes that is not ASCII.
    NotAscii(char),
    /// A quote left unpaired: `$$`, `'`, `"` or a backtick.
    Unpaired(&'static str),
    ClosesUnopened,
    LeavesUnclosed,
    /// A statement keyword outside quotes, as written.
    Keyword(String),
    SubQuery,
    /// A call of what a bracket or an index gives.
    CallsResult,
    /// A call of something that is no name, with the ten characters before
    /// its `(`, quoted text masked.
    CallsAfter(String),
    /// A call of a quoted name.
    QuotedFunction,
    /// A call of a qualified name: its last part.
    QualifiedFunction(String),
    /// A function named through `IDENTIFIER()`.
    Identifier,
    /// A call of a function off the dialect's allow-list, as written.
    NotPure(String),
    /// A session word read bare, as written.
    SessionValue(String),
    /// A session word read quoted, its quotes included.
    QuotedSessionValue(String),
    /// A pattern the check reads it by failed on it, so it cannot be read
    /// safely (T3.5 fix 1; the old client has no such limit).
    Unreadable(Unreadable),
}

impl ExprRefusal {
    /// The old client's words for the refusal, `quote` writing a piece of
    /// the expression as it quoted it (Python's `repr`, which the core
    /// hands in: no code here writes Python's text).
    pub fn words(&self, quote: &dyn Fn(&str) -> String) -> String {
        match self {
            ExprRefusal::Contains(token) => format!("it contains {}", quote(token)),
            ExprRefusal::DollarQuotes => {
                "it contains '$$', which Databricks SQL does not quote with".into()
            }
            ExprRefusal::SlashComment => "it contains '//'".into(),
            ExprRefusal::BackslashInQuotedName => {
                "it has a backslash inside a double-quoted name".into()
            }
            ExprRefusal::NotAscii(c) => format!(
                "it has {} outside quotes, which is not ASCII",
                quote(&c.to_string())
            ),
            ExprRefusal::Unpaired(delimiter) => format!(
                "it leaves {} unpaired, so which part of it is quoted text cannot be established",
                quote(delimiter)
            ),
            ExprRefusal::ClosesUnopened => "it closes a '(' it never opened".into(),
            ExprRefusal::LeavesUnclosed => "it leaves a '(' unclosed".into(),
            ExprRefusal::Keyword(keyword) => format!("it contains the SQL keyword '{keyword}'"),
            ExprRefusal::SubQuery => "it contains a sub-query".into(),
            ExprRefusal::CallsResult => "it calls the result of an expression".into(),
            ExprRefusal::CallsAfter(before) => {
                format!("it calls what precedes '(' after {}", quote(before))
            }
            ExprRefusal::QuotedFunction => "it calls a function by a quoted name".into(),
            ExprRefusal::QualifiedFunction(name) => {
                format!("it calls the qualified function '{name}'")
            }
            ExprRefusal::Identifier => "it names a function through IDENTIFIER()".into(),
            ExprRefusal::NotPure(name) => {
                format!("it calls '{name}', which is not a pure built-in SQL function")
            }
            ExprRefusal::SessionValue(word) => format!("it reads the session value '{word}'"),
            ExprRefusal::QuotedSessionValue(quoted) => {
                format!("it reads the session value {quoted}")
            }
            ExprRefusal::Unreadable(unreadable) => unreadable.to_string(),
        }
    }
}

const UNSAFE_TOKENS: [&str; 6] = [";", "--", "/*", "*/", "|>", "->>"];
const QUOTE_DELIMITERS: [&str; 4] = ["$$", "'", "\"", "`"];

/// A private-use character stands in for quoted text, so a call to a
/// quoted name stays visible as a call.
pub(crate) const QUOTED: char = '\u{E000}';

static UNSAFE_KEYWORDS: LazyLock<Pattern> = LazyLock::new(|| {
    Pattern::new(
        r"(?i)(?<![\w.])(?:(?:select|table|update|delete|merge|create|alter|drop|grant|revoke|execute)\b|(?:insert|truncate)\b(?!\s*\())",
    )
});
// Databricks runs each of these as a sub-query where it opens the
// expression or a `(`; the script transforms need a USING before their
// bracket closes.
static QUERY_START: LazyLock<Pattern> = LazyLock::new(|| {
    Pattern::new(
        r"(?i)(?:^|\()\s*(?:(?:from|values)\b|with\s+(?:recursive\s+)?(?:[\w$]+\s*)?(?:\([^()]*\)\s*)?as\s*\(|(?P<script>(?:map|reduce)\b(?!\s*\()))",
    )
});
static USING: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"(?i)\busing\b"));
static NOT_ASCII: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"[^\x00-\x7f]"));
static SESSION_VALUES: LazyLock<Pattern> = LazyLock::new(|| {
    Pattern::new(&format!(
        r"(?i)(?<![\w$.\x{{e000}}])({})\b(?!\s*\()",
        words::SESSION_WORDS.join("|")
    ))
});
pub(crate) static LAST_TOKEN: LazyLock<Pattern> =
    LazyLock::new(|| Pattern::new(r"([\w$\x{e000}]+)\s*$"));
pub(crate) static QUALIFIED: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"\.\s*$"));
static AFTER_DOUBLE_COLON: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"::\s*$"));
static AFTER_AS: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"(?i)(?<![\w$])as\s*$"));
// The field of `EXTRACT(YEAR FROM d)` is a value, whatever its name.
static DATETIME_FIELD: LazyLock<Pattern> =
    LazyLock::new(|| Pattern::new(r"(?i)(?<![\w$.])(?:extract|date_part)\s*\(\s*$"));
static LAMBDA_PARAMETERS: LazyLock<Pattern> =
    LazyLock::new(|| Pattern::new(r"(?:\(\s*([\w$]+(?:\s*,\s*[\w$]+)*)\s*\)|([\w$]+))\s*->"));
static WORD_OR_BRACKET: LazyLock<Pattern> =
    LazyLock::new(|| Pattern::new(r"[\w$\x{e000}]+|[()\[\]]"));
static WHEN: LazyLock<Pattern> = LazyLock::new(|| Pattern::new(r"(?i)^\s*when\b"));

/// The first match of `pattern` in `text`: where its first group starts,
/// and that group's text.
pub(crate) fn first(pattern: &Pattern, text: &str) -> Read<Option<(usize, String)>> {
    Ok(pattern
        .groups(text)?
        .and_then(|groups| groups.get(1))
        .map(|span| (span.start, text[span].to_owned())))
}

/// Keywords that an expression may follow with `(` without calling
/// anything (`_KEYWORDS`), with what each may stand after: the keyword
/// pairs SQL writes (`_FOLLOWS_A_KEYWORD`, `:760-781`).
fn follows_a_keyword(before: &str) -> &'static [&'static str] {
    match before {
        "not" => &[
            "between", "distinct", "exists", "ilike", "in", "like", "regexp", "rlike",
        ],
        // Only LIKE and ILIKE take a quantifier.
        "like" | "ilike" => &["all", "any", "some"],
        "is" => &["distinct", "not"],
        "distinct" => &["from"],
        // WHEN, THEN and ELSE each take an operand.
        "case" => &["when"],
        _ => &[],
    }
}

const CASTS: [&str; 2] = ["cast", "try_cast"];
/// What may stand right before a `(` that opens a group rather than a call.
const OPENS_A_GROUP: &str = "+-*/%=<>!|&^~,([?:";

/// Why an expression from a view is unsafe, or `None`
/// (`_unsafe_expr_reason`, `:231-315`).
///
/// A scalar expression never needs a statement separator, a comment or a
/// sub-query, so those are refused, checked on the raw text so that no
/// reading of the quoting can hide one. Then, read as `dialect` lexes it
/// with its quoted text masked: anything not ASCII outside quotes, a quote
/// left unpaired, brackets that do not balance, a statement keyword, or a
/// sub-query's start. `None` for the dialect checks only what both share.
/// An expression the check cannot read is refused.
pub fn unsafe_reason(expr: &str, dialect: Option<ExprDialect>) -> Option<ExprRefusal> {
    unsafe_read(expr, dialect)
        .unwrap_or_else(|unreadable| Some(ExprRefusal::Unreadable(unreadable)))
}

fn unsafe_read(expr: &str, dialect: Option<ExprDialect>) -> Read<Option<ExprRefusal>> {
    for token in UNSAFE_TOKENS {
        if expr.contains(token) {
            return Ok(Some(ExprRefusal::Contains(token)));
        }
    }
    // Databricks has no `$$` literal.
    if dialect == Some(ExprDialect::Databricks) && expr.contains("$$") {
        return Ok(Some(ExprRefusal::DollarQuotes));
    }
    if dialect == Some(ExprDialect::Snowflake) {
        let tokens = lex(expr, dialect)?;
        // Snowflake also comments out the rest of a line with `//`.
        if tokens.iter().any(|token| token.kind == Kind::Comment) {
            return Ok(Some(ExprRefusal::SlashComment));
        }
        if tokens
            .iter()
            .any(|token| token.text.starts_with('"') && token.text.contains('\\'))
        {
            return Ok(Some(ExprRefusal::BackslashInQuotedName));
        }
    }
    let unquoted = masked(expr, dialect, " ")?;
    if let Some(other) = NOT_ASCII.find(&unquoted)? {
        return Ok(Some(ExprRefusal::NotAscii(
            unquoted[other].chars().next().unwrap(),
        )));
    }
    for delimiter in QUOTE_DELIMITERS {
        if unquoted.contains(delimiter) {
            return Ok(Some(ExprRefusal::Unpaired(delimiter)));
        }
    }
    let mut depth: i64 = 0;
    for c in unquoted.chars() {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth < 0 {
            return Ok(Some(ExprRefusal::ClosesUnopened));
        }
    }
    if depth != 0 {
        return Ok(Some(ExprRefusal::LeavesUnclosed));
    }
    if let Some(keyword) = UNSAFE_KEYWORDS.find(&unquoted)? {
        return Ok(Some(ExprRefusal::Keyword(unquoted[keyword].to_owned())));
    }
    if starts_a_query(&unquoted)? {
        return Ok(Some(ExprRefusal::SubQuery));
    }
    Ok(None)
}

/// Whether `unquoted`, its quoted text masked, opens a sub-query: `FROM`,
/// `VALUES` or `WITH .. AS (` where the expression or a `(` opens, or
/// `MAP` or `REDUCE` with a `USING` before its bracket closes.
fn starts_a_query(unquoted: &str) -> Read<bool> {
    for groups in QUERY_START.groups_all(unquoted)? {
        if groups.name("script").is_none() {
            return Ok(true);
        }
        let after = groups.get(0).unwrap().end;
        let mut depth: i64 = 0;
        let mut end = unquoted.len();
        for (at, c) in unquoted[after..].char_indices() {
            depth += match c {
                '(' => 1,
                ')' => -1,
                _ => 0,
            };
            if depth < 0 {
                end = after + at;
                break;
            }
        }
        // Searched as Python searches `[after, end)`: the text ends at
        // `end`, and what precedes `after` is still seen.
        if USING.find_from(&unquoted[..end], after)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether a `CASE` is open at the end of `lead`, in the same group
/// (`_closes_a_case`).
fn closes_a_case(lead: &str) -> Read<bool> {
    let mut open_cases = vec![0usize];
    for span in WORD_OR_BRACKET.find_all(lead)? {
        let text = &lead[span.clone()];
        if text == "(" || text == "[" {
            open_cases.push(0);
        } else if text == ")" || text == "]" {
            if open_cases.len() > 1 {
                open_cases.pop();
            }
        } else if QUALIFIED.is_match(&lead[..span.start])? {
            continue;
        } else if text.to_lowercase() == "case" {
            *open_cases.last_mut().unwrap() += 1;
        } else if text.to_lowercase() == "end" && *open_cases.last().unwrap() > 0 {
            *open_cases.last_mut().unwrap() -= 1;
        }
    }
    Ok(*open_cases.last().unwrap() > 0)
}

/// Whether `lead` ends with something a comparison can be about
/// (`_follows_a_value`): a closing bracket, quoted text, a literal, a
/// qualified name, a type after `::`, the field of `EXTRACT`, the `END` of
/// an open `CASE`, or a bare word naming one of `columns` the parser cannot
/// read as grammar instead.
fn follows_a_value(lead: &str, columns: &[String]) -> Read<bool> {
    let lead = rstrip(lead);
    let Some(last_char) = lead.chars().last() else {
        return Ok(false);
    };
    if matches!(last_char, ')' | ']' | QUOTED) {
        return Ok(true);
    }
    let Some((start, word)) = first(&LAST_TOKEN, lead)? else {
        return Ok(false);
    };
    let before = &lead[..start];
    if QUALIFIED.is_match(before)?
        || AFTER_DOUBLE_COLON.is_match(before)?
        || DATETIME_FIELD.is_match(before)?
    {
        return Ok(true);
    }
    let word = word.to_lowercase();
    if word.chars().next().is_some_and(char::is_numeric)
        || matches!(word.as_str(), "true" | "false" | "null")
    {
        return Ok(true);
    }
    if word == "end" {
        return Ok(closes_a_case(before)? && follows_a_value(before, columns)?);
    }
    // A table may name a column `where`, which the parser still reads as
    // WHERE.
    Ok(columns.contains(&word) && !has(words::GRAMMAR_WORDS, &word))
}

/// Whether `CASE` before `group` starts `CASE (x) WHEN ...`: where a CASE
/// can start, and a WHEN follows the bracket (`_opens_a_simple_case`).
fn opens_a_simple_case(lead: &str, group: &str) -> Read<bool> {
    let lead = rstrip(lead);
    if let Some(c) = lead.chars().last()
        && c != '('
        && c != ','
    {
        return Ok(false);
    }
    let mut depth: i64 = 0;
    for (at, c) in group.char_indices() {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth == 0 {
            return WHEN.is_match(&group[at + c.len_utf8()..]);
        }
    }
    Ok(false)
}

/// Whether `word`, followed by `(`, is a keyword rather than a call
/// (`_acts_as_keyword`): `NOT` always; `CASE` where a CASE may start; any
/// other where a keyword may stand.
fn acts_as_keyword(word: &str, lead: &str, group: &str, columns: &[String]) -> Read<bool> {
    match word {
        "not" => Ok(true),
        "case" => opens_a_simple_case(lead, group),
        _ => in_keyword_position(word, lead, columns),
    }
}

/// Whether `word` sits where a keyword may: after a value, or after a
/// keyword SQL writes it after, that one itself in keyword position
/// (`_in_keyword_position`).
fn in_keyword_position(word: &str, lead: &str, columns: &[String]) -> Read<bool> {
    if follows_a_value(lead, columns)? {
        return Ok(true);
    }
    let lead = rstrip(lead);
    let Some((start, previous)) = first(&LAST_TOKEN, lead)? else {
        return Ok(false);
    };
    let before = previous.to_lowercase();
    if !follows_a_keyword(&before).contains(&word) {
        return Ok(false);
    }
    if before == "case" {
        return Ok(true);
    }
    in_keyword_position(&before, &lead[..start], columns)
}

/// Whether the word after `lead` stands where only a type can: after
/// `::`, or after an `AS` directly inside `CAST(` or `TRY_CAST(`
/// (`_names_a_type`).
fn names_a_type(lead: &str) -> Read<bool> {
    if AFTER_DOUBLE_COLON.is_match(lead)? {
        return Ok(true);
    }
    let Some(after_as) = AFTER_AS.find(lead)? else {
        return Ok(false);
    };
    let mut depth: i64 = 0;
    for (index, c) in lead[..after_as.start].char_indices().rev() {
        depth += match c {
            ')' => 1,
            '(' => -1,
            _ => 0,
        };
        if depth < 0 {
            let opener = rstrip(&lead[..index]);
            return Ok(match first(&LAST_TOKEN, opener)? {
                Some((start, name)) => {
                    CASTS.contains(&name.to_lowercase().as_str())
                        && !QUALIFIED.is_match(&opener[..start])?
                }
                None => false,
            });
        }
    }
    Ok(false)
}

/// The folded names `text` binds in lambdas, as in `x -> x + 1`.
fn lambda_parameters(text: &str) -> Read<Vec<String>> {
    let mut names = Vec::new();
    for groups in LAMBDA_PARAMETERS.groups_all(text)? {
        let Some(bound) = groups.get(1).or_else(|| groups.get(2)) else {
            continue;
        };
        names.extend(
            text[bound]
                .split(',')
                .map(|name| super::stripped(name).to_lowercase()),
        );
    }
    Ok(names)
}

/// Why an expression calls a function that is not trusted, or reads a
/// session value, or `None` (`_untrusted_call_reason`, `:964-1044`).
///
/// Every `(` must open a group, follow a keyword acting as grammar, give a
/// type its parameters, or call a function on the dialect's allow-list by
/// its plain name; a quoted or qualified name, or `IDENTIFIER()`, can name
/// anything and is refused. A session word (`current_user`) is refused,
/// bare or quoted, unless one of `columns` names it where the dialect
/// lets the column win (Databricks only). Apply [`unsafe_reason`] first,
/// which establishes that every quote is paired. An expression the check
/// cannot read is refused.
pub fn untrusted_call_reason<S: AsRef<str>>(
    expr: &str,
    dialect: ExprDialect,
    columns: &[S],
) -> Option<ExprRefusal> {
    calls_read(expr, dialect, columns)
        .unwrap_or_else(|unreadable| Some(ExprRefusal::Unreadable(unreadable)))
}

fn calls_read<S: AsRef<str>>(
    expr: &str,
    dialect: ExprDialect,
    columns: &[S],
) -> Read<Option<ExprRefusal>> {
    let trusted = match dialect {
        ExprDialect::Databricks => words::DATABRICKS_FUNCTIONS,
        ExprDialect::Snowflake => words::SNOWFLAKE_FUNCTIONS,
    };
    let text = masked(expr, Some(dialect), &QUOTED.to_string())?;
    let named: Vec<String> = columns.iter().map(|c| c.as_ref().to_lowercase()).collect();
    let mut visible = named.clone();
    visible.extend(lambda_parameters(&text)?);
    for (index, c) in text.char_indices() {
        if c != '(' {
            continue;
        }
        let before = rstrip(&text[..index]);
        let Some(last) = before.chars().last() else {
            continue;
        };
        if OPENS_A_GROUP.contains(last) {
            continue;
        }
        if last == ')' || last == ']' {
            return Ok(Some(ExprRefusal::CallsResult));
        }
        let Some((start, name)) = first(&LAST_TOKEN, before)? else {
            let tail: String = {
                let chars: Vec<char> = before.chars().collect();
                chars[chars.len().saturating_sub(10)..].iter().collect()
            };
            return Ok(Some(ExprRefusal::CallsAfter(tail)));
        };
        if name.contains(QUOTED) {
            return Ok(Some(ExprRefusal::QuotedFunction));
        }
        if QUALIFIED.is_match(&before[..start])? {
            return Ok(Some(ExprRefusal::QualifiedFunction(name)));
        }
        let folded = name.to_lowercase();
        if folded == "identifier" {
            return Ok(Some(ExprRefusal::Identifier));
        }
        if has(words::KEYWORDS, &folded)
            && acts_as_keyword(&folded, &before[..start], &text[index..], &visible)?
        {
            continue;
        }
        if has(words::TYPE_NAMES, &folded) && names_a_type(&before[..start])? {
            continue;
        }
        if !has(trusted, &folded) {
            return Ok(Some(ExprRefusal::NotPure(name)));
        }
    }
    // A lambda parameter is in scope only inside its lambda, so only the
    // table's columns can stand in for a session word.
    let overridden = |word: &str| {
        dialect == ExprDialect::Databricks
            && named.iter().any(|n| n == word)
            && has(words::DATABRICKS_COLUMN_OVER_SESSION, word)
    };
    for groups in SESSION_VALUES.groups_all(&text)? {
        let session = &text[groups.get(1).unwrap()];
        if !overridden(&session.to_lowercase()) {
            return Ok(Some(ExprRefusal::SessionValue(session.to_owned())));
        }
    }
    // Quoting the word does not keep it from the session; a column of the
    // name wins for every session word quoted, on Databricks alone.
    let quoted_overridden = |word: &str| {
        dialect == ExprDialect::Databricks
            && named.iter().any(|n| n == word)
            && has(words::SESSION_WORDS, word)
    };
    for token in lex(expr, Some(dialect))? {
        if token.kind != Kind::Quoted || QUALIFIED.is_match(&expr[..token.start])? {
            continue;
        }
        let word = token.unquoted().to_lowercase();
        if has(words::SESSION_WORDS, &word) && !quoted_overridden(&word) {
            return Ok(Some(ExprRefusal::QuotedSessionValue(token.text)));
        }
    }
    Ok(None)
}
