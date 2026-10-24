// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Why an expression is unsafe to splice into a statement, and why it
//! calls a function that is not trusted (`RE/view_expr.py:192-1044`).

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::lex::{Kind, lex, masked};
use super::words::{self, has};
use super::{ExprDialect, quoted, rstrip};

const UNSAFE_TOKENS: [&str; 6] = [";", "--", "/*", "*/", "|>", "->>"];
const QUOTE_DELIMITERS: [&str; 4] = ["$$", "'", "\"", "`"];

/// A private-use character stands in for quoted text, so a call to a
/// quoted name stays visible as a call.
pub(crate) const QUOTED: char = '\u{E000}';

fn pattern(text: &str) -> Regex {
    Regex::new(text).unwrap()
}

static UNSAFE_KEYWORDS: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)(?<![\w.])(?:(?:select|table|update|delete|merge|create|alter|drop|grant|revoke|execute)\b|(?:insert|truncate)\b(?!\s*\())",
    )
});
// Databricks runs each of these as a sub-query where it opens the
// expression or a `(`; the script transforms need a USING before their
// bracket closes.
static QUERY_START: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)(?:^|\()\s*(?:(?:from|values)\b|with\s+(?:recursive\s+)?(?:[\w$]+\s*)?(?:\([^()]*\)\s*)?as\s*\(|(?P<script>(?:map|reduce)\b(?!\s*\()))",
    )
});
static USING: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)\busing\b"));
static NOT_ASCII: LazyLock<Regex> = LazyLock::new(|| pattern(r"[^\x00-\x7f]"));
static SESSION_VALUES: LazyLock<Regex> = LazyLock::new(|| {
    pattern(&format!(
        r"(?i)(?<![\w$.\x{{e000}}])({})\b(?!\s*\()",
        words::SESSION_WORDS.join("|")
    ))
});
pub(crate) static LAST_TOKEN: LazyLock<Regex> = LazyLock::new(|| pattern(r"([\w$\x{e000}]+)\s*$"));
pub(crate) static QUALIFIED: LazyLock<Regex> = LazyLock::new(|| pattern(r"\.\s*$"));
static AFTER_DOUBLE_COLON: LazyLock<Regex> = LazyLock::new(|| pattern(r"::\s*$"));
static AFTER_AS: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)(?<![\w$])as\s*$"));
// The field of `EXTRACT(YEAR FROM d)` is a value, whatever its name.
static DATETIME_FIELD: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(?i)(?<![\w$.])(?:extract|date_part)\s*\(\s*$"));
static LAMBDA_PARAMETERS: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(?:\(\s*([\w$]+(?:\s*,\s*[\w$]+)*)\s*\)|([\w$]+))\s*->"));
static WORD_OR_BRACKET: LazyLock<Regex> = LazyLock::new(|| pattern(r"[\w$\x{e000}]+|[()\[\]]"));
static WHEN: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)^\s*when\b"));

/// Whether `regex` finds a match in `text`.
pub(crate) fn found(regex: &Regex, text: &str) -> bool {
    regex.is_match(text).unwrap_or(false)
}

/// The first match of `regex` in `text`, its range and first group's.
pub(crate) fn first(regex: &Regex, text: &str) -> Option<(usize, String)> {
    let captures = regex.captures(text).ok()??;
    let group = captures.get(1).unwrap();
    Some((group.start(), group.as_str().to_owned()))
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
pub fn unsafe_reason(expr: &str, dialect: Option<ExprDialect>) -> Option<String> {
    for token in UNSAFE_TOKENS {
        if expr.contains(token) {
            return Some(format!("it contains '{token}'"));
        }
    }
    // Databricks has no `$$` literal.
    if dialect == Some(ExprDialect::Databricks) && expr.contains("$$") {
        return Some("it contains '$$', which Databricks SQL does not quote with".into());
    }
    if dialect == Some(ExprDialect::Snowflake) {
        let tokens = lex(expr, dialect);
        // Snowflake also comments out the rest of a line with `//`.
        if tokens.iter().any(|token| token.kind == Kind::Comment) {
            return Some("it contains '//'".into());
        }
        if tokens
            .iter()
            .any(|token| token.text.starts_with('"') && token.text.contains('\\'))
        {
            return Some("it has a backslash inside a double-quoted name".into());
        }
    }
    let unquoted = masked(expr, dialect, " ");
    if let Ok(Some(other)) = NOT_ASCII.find(&unquoted) {
        return Some(format!(
            "it has {} outside quotes, which is not ASCII",
            quoted(other.as_str())
        ));
    }
    for delimiter in QUOTE_DELIMITERS {
        if unquoted.contains(delimiter) {
            return Some(format!(
                "it leaves {} unpaired, so which part of it is quoted text cannot be established",
                quoted(delimiter)
            ));
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
            return Some("it closes a '(' it never opened".into());
        }
    }
    if depth != 0 {
        return Some("it leaves a '(' unclosed".into());
    }
    if let Ok(Some(keyword)) = UNSAFE_KEYWORDS.find(&unquoted) {
        return Some(format!(
            "it contains the SQL keyword '{}'",
            keyword.as_str()
        ));
    }
    if starts_a_query(&unquoted) {
        return Some("it contains a sub-query".into());
    }
    None
}

/// Whether `unquoted`, its quoted text masked, opens a sub-query: `FROM`,
/// `VALUES` or `WITH .. AS (` where the expression or a `(` opens, or
/// `MAP` or `REDUCE` with a `USING` before its bracket closes.
fn starts_a_query(unquoted: &str) -> bool {
    for captures in QUERY_START.captures_iter(unquoted) {
        let Ok(captures) = captures else {
            return true;
        };
        if captures.name("script").is_none() {
            return true;
        }
        let after = captures.get(0).unwrap().end();
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
        if matches!(USING.find_from_pos(&unquoted[..end], after), Ok(Some(_))) {
            return true;
        }
    }
    false
}

/// Whether a `CASE` is open at the end of `lead`, in the same group
/// (`_closes_a_case`).
fn closes_a_case(lead: &str) -> bool {
    let mut open_cases = vec![0usize];
    for token in WORD_OR_BRACKET.find_iter(lead) {
        let Ok(token) = token else { break };
        let text = token.as_str();
        if text == "(" || text == "[" {
            open_cases.push(0);
        } else if text == ")" || text == "]" {
            if open_cases.len() > 1 {
                open_cases.pop();
            }
        } else if found(&QUALIFIED, &lead[..token.start()]) {
            continue;
        } else if text.to_lowercase() == "case" {
            *open_cases.last_mut().unwrap() += 1;
        } else if text.to_lowercase() == "end" && *open_cases.last().unwrap() > 0 {
            *open_cases.last_mut().unwrap() -= 1;
        }
    }
    *open_cases.last().unwrap() > 0
}

/// Whether `lead` ends with something a comparison can be about
/// (`_follows_a_value`): a closing bracket, quoted text, a literal, a
/// qualified name, a type after `::`, the field of `EXTRACT`, the `END` of
/// an open `CASE`, or a bare word naming one of `columns` the parser cannot
/// read as grammar instead.
fn follows_a_value(lead: &str, columns: &[String]) -> bool {
    let lead = rstrip(lead);
    let Some(last_char) = lead.chars().last() else {
        return false;
    };
    if matches!(last_char, ')' | ']' | QUOTED) {
        return true;
    }
    let Some((start, word)) = first(&LAST_TOKEN, lead) else {
        return false;
    };
    let before = &lead[..start];
    if found(&QUALIFIED, before)
        || found(&AFTER_DOUBLE_COLON, before)
        || found(&DATETIME_FIELD, before)
    {
        return true;
    }
    let word = word.to_lowercase();
    if word.chars().next().is_some_and(char::is_numeric)
        || matches!(word.as_str(), "true" | "false" | "null")
    {
        return true;
    }
    if word == "end" {
        return closes_a_case(before) && follows_a_value(before, columns);
    }
    // A table may name a column `where`, which the parser still reads as
    // WHERE.
    columns.contains(&word) && !has(words::GRAMMAR_WORDS, &word)
}

/// Whether `CASE` before `group` starts `CASE (x) WHEN ...`: where a CASE
/// can start, and a WHEN follows the bracket (`_opens_a_simple_case`).
fn opens_a_simple_case(lead: &str, group: &str) -> bool {
    let lead = rstrip(lead);
    if let Some(c) = lead.chars().last()
        && c != '('
        && c != ','
    {
        return false;
    }
    let mut depth: i64 = 0;
    for (at, c) in group.char_indices() {
        depth += match c {
            '(' => 1,
            ')' => -1,
            _ => 0,
        };
        if depth == 0 {
            return found(&WHEN, &group[at + c.len_utf8()..]);
        }
    }
    false
}

/// Whether `word`, followed by `(`, is a keyword rather than a call
/// (`_acts_as_keyword`): `NOT` always; `CASE` where a CASE may start; any
/// other where a keyword may stand.
fn acts_as_keyword(word: &str, lead: &str, group: &str, columns: &[String]) -> bool {
    match word {
        "not" => true,
        "case" => opens_a_simple_case(lead, group),
        _ => in_keyword_position(word, lead, columns),
    }
}

/// Whether `word` sits where a keyword may: after a value, or after a
/// keyword SQL writes it after, that one itself in keyword position
/// (`_in_keyword_position`).
fn in_keyword_position(word: &str, lead: &str, columns: &[String]) -> bool {
    if follows_a_value(lead, columns) {
        return true;
    }
    let lead = rstrip(lead);
    let Some((start, previous)) = first(&LAST_TOKEN, lead) else {
        return false;
    };
    let before = previous.to_lowercase();
    if !follows_a_keyword(&before).contains(&word) {
        return false;
    }
    if before == "case" {
        return true;
    }
    in_keyword_position(&before, &lead[..start], columns)
}

/// Whether the word after `lead` stands where only a type can: after
/// `::`, or after an `AS` directly inside `CAST(` or `TRY_CAST(`
/// (`_names_a_type`).
fn names_a_type(lead: &str) -> bool {
    if found(&AFTER_DOUBLE_COLON, lead) {
        return true;
    }
    let Ok(Some(after_as)) = AFTER_AS.find(lead) else {
        return false;
    };
    let mut depth: i64 = 0;
    for (index, c) in lead[..after_as.start()].char_indices().rev() {
        depth += match c {
            ')' => 1,
            '(' => -1,
            _ => 0,
        };
        if depth < 0 {
            let opener = rstrip(&lead[..index]);
            return match first(&LAST_TOKEN, opener) {
                Some((start, name)) => {
                    CASTS.contains(&name.to_lowercase().as_str())
                        && !found(&QUALIFIED, &opener[..start])
                }
                None => false,
            };
        }
    }
    false
}

/// The folded names `text` binds in lambdas, as in `x -> x + 1`.
fn lambda_parameters(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for captures in LAMBDA_PARAMETERS.captures_iter(text) {
        let Ok(captures) = captures else { break };
        let bound = captures
            .get(1)
            .or_else(|| captures.get(2))
            .unwrap()
            .as_str();
        names.extend(bound.split(',').map(|name| name.trim().to_lowercase()));
    }
    names
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
/// which establishes that every quote is paired.
pub fn untrusted_call_reason<S: AsRef<str>>(
    expr: &str,
    dialect: ExprDialect,
    columns: &[S],
) -> Option<String> {
    let trusted = match dialect {
        ExprDialect::Databricks => words::DATABRICKS_FUNCTIONS,
        ExprDialect::Snowflake => words::SNOWFLAKE_FUNCTIONS,
    };
    let text = masked(expr, Some(dialect), &QUOTED.to_string());
    let named: Vec<String> = columns.iter().map(|c| c.as_ref().to_lowercase()).collect();
    let mut visible = named.clone();
    visible.extend(lambda_parameters(&text));
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
            return Some("it calls the result of an expression".into());
        }
        let Some((start, name)) = first(&LAST_TOKEN, before) else {
            let tail: String = {
                let chars: Vec<char> = before.chars().collect();
                chars[chars.len().saturating_sub(10)..].iter().collect()
            };
            return Some(format!(
                "it calls what precedes '(' after {}",
                quoted(&tail)
            ));
        };
        if name.contains(QUOTED) {
            return Some("it calls a function by a quoted name".into());
        }
        if found(&QUALIFIED, &before[..start]) {
            return Some(format!("it calls the qualified function '{name}'"));
        }
        let folded = name.to_lowercase();
        if folded == "identifier" {
            return Some("it names a function through IDENTIFIER()".into());
        }
        if has(words::KEYWORDS, &folded)
            && acts_as_keyword(&folded, &before[..start], &text[index..], &visible)
        {
            continue;
        }
        if has(words::TYPE_NAMES, &folded) && names_a_type(&before[..start]) {
            continue;
        }
        if !has(trusted, &folded) {
            return Some(format!(
                "it calls '{name}', which is not a pure built-in SQL function"
            ));
        }
    }
    // A lambda parameter is in scope only inside its lambda, so only the
    // table's columns can stand in for a session word.
    let overridden = |word: &str| {
        dialect == ExprDialect::Databricks
            && named.iter().any(|n| n == word)
            && has(words::DATABRICKS_COLUMN_OVER_SESSION, word)
    };
    for captures in SESSION_VALUES.captures_iter(&text) {
        let Ok(captures) = captures else { break };
        let session = captures.get(1).unwrap().as_str();
        if !overridden(&session.to_lowercase()) {
            return Some(format!("it reads the session value '{session}'"));
        }
    }
    // Quoting the word does not keep it from the session; a column of the
    // name wins for every session word quoted, on Databricks alone.
    let quoted_overridden = |word: &str| {
        dialect == ExprDialect::Databricks
            && named.iter().any(|n| n == word)
            && has(words::SESSION_WORDS, word)
    };
    for token in lex(expr, Some(dialect)) {
        if token.kind != Kind::Quoted || found(&QUALIFIED, &expr[..token.start]) {
            continue;
        }
        let word = token.unquoted().to_lowercase();
        if has(words::SESSION_WORDS, &word) && !quoted_overridden(&word) {
            return Some(format!("it reads the session value {}", token.text));
        }
    }
    None
}
