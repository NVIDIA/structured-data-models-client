// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's expression helpers, ported from the old client's
//! `DB/metric_view.py` (and `_renamed_column`, `RE/graph.py:424-430`): its
//! table and view names, a join's condition and keys, a table's qualifier
//! in an expression, the column a time dimension truncates or casts, a
//! parameter's default as a typed literal, parameters and earlier fields
//! substituted, the star, the filter's check, and what each measure reads.
//!
//! Each is a pure function of text; the core calls each in the old
//! client's order and words each refusal (`sdm-core::view_words`).

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::check::{ExprRefusal, unsafe_reason};
use super::lex::{bare_name, string_literals};
use super::names::{Unreadable, name_references};
use super::{ExprDialect, stripped};

fn pattern(text: &str) -> Regex {
    Regex::new(text).unwrap()
}

/// A part of a name: backticked, or a run Databricks reads as a name.
fn ident() -> String {
    format!("`(?:[^`]|``)+`|{}", bare_name())
}

fn chain() -> String {
    let ident = ident();
    format!(r"(?:{ident})(?:\s*\.\s*(?:{ident}))*")
}

static TABLE_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    let ident = ident();
    pattern(&format!(
        r"^\s*({ident})(?:\.({ident}))?(?:\.({ident}))?\s*$"
    ))
});
// An unquoted part of a view's name is any run of characters but a dot, a
// backtick and white space, as a caller writes `my-catalog` unquoted.
static VIEW_NAME: LazyLock<Regex> = LazyLock::new(|| {
    let part = r"`(?:[^`]|``)+`|[^\s.`]+";
    pattern(&format!(r"^\s*({part})(?:\.({part}))?(?:\.({part}))?\s*$"))
});
static JOIN_CONDITION: LazyLock<Regex> = LazyLock::new(|| {
    let chain = chain();
    pattern(&format!(r"^\s*({chain})\s*=\s*({chain})\s*$"))
});
static IDENT: LazyLock<Regex> = LazyLock::new(|| pattern(&ident()));
static WHOLE_IDENT: LazyLock<Regex> = LazyLock::new(|| pattern(&format!("^(?:{})$", ident())));
static CHAIN: LazyLock<Regex> = LazyLock::new(|| pattern(&chain()));
// `- expr: source.*` or `- expr: customers.* EXCEPT (a, b)`.
static WILDCARD: LazyLock<Regex> = LazyLock::new(|| {
    let chain = chain();
    pattern(&format!(
        r"(?i)^\s*({chain})\s*\.\s*\*(?:\s*EXCEPT\s*\(\s*({chain}(?:\s*,\s*{chain})*)\s*\))?\s*$"
    ))
});
static RENAMED: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^(?:`((?:[^`]|``)+)`|([A-Za-z_]\w*))$"));
static WHOLE_BARE_NAME: LazyLock<Regex> =
    LazyLock::new(|| pattern(&format!("^(?:{})$", bare_name())));
// What a timestamp dimension may wrap a column in and still say when the
// row happened: a truncation, or a cast to a date or time.
static DERIVATIONS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let time = "(?:date|timestamp(?:_ntz|_ltz)?)";
    [
        r#"\(\s*(?P<inner>.+)\s*\)"#.to_owned(),
        r#"date_trunc\s*\(\s*(?:'\w+'|"\w+")\s*,\s*(?P<inner>.+)\s*\)"#.to_owned(),
        r#"trunc\s*\(\s*(?P<inner>.+?)\s*,\s*(?:'\w+'|"\w+")\s*\)"#.to_owned(),
        format!(r"cast\s*\(\s*(?P<inner>.+)\s+as\s+{time}\s*\)"),
        r"(?:to_date|date|to_timestamp)\s*\(\s*(?P<inner>.+)\s*\)".to_owned(),
        format!(r"(?P<inner>.+?)\s*::\s*{time}"),
    ]
    .iter()
    .map(|p| pattern(&format!("(?is)^(?:{p})$")))
    .collect()
});
// The first part of a name, backticked or not.
static FIRST_PART: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^(?:`(?:``|[^`])*`|[A-Za-z0-9_]+)"));
static SIMPLE_MEASURE: LazyLock<Regex> = LazyLock::new(|| {
    pattern(&format!(
        r"(?i)^\s*(sum|avg|min|max|count)\s*\(\s*(distinct\s+)?(\*|1|{})\s*\)\s*$",
        chain()
    ))
});
static INTEGER_TEXT: LazyLock<Regex> = LazyLock::new(|| pattern(r"^[+-]?\d+$"));
static DECIMAL_TYPE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^(?:decimal|dec|numeric)(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?$"));
static STRING_TYPE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^(?:string|(?:var)?char\(\s*\d+\s*\))$"));

const SOURCE_ALIAS: &str = "source";
const INTEGER_TYPES: &[&str] = &[
    "bigint", "byte", "int", "integer", "long", "short", "smallint", "tinyint",
];
const FLOAT_TYPES: &[&str] = &["double", "float", "real"];
const TIMESTAMP_TYPES: &[&str] = &["timestamp", "timestamp_ltz", "timestamp_ntz"];

fn matches(regex: &Regex, text: &str) -> bool {
    regex.is_match(text).unwrap_or(false)
}

/// One side of a join condition: the (dot-joined) table alias or join path
/// qualifying the column, if any, and the column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRef {
    pub qualifier: Option<String>,
    pub column: String,
}

/// A name without its backticks, if it is backticked (`unquote_ident`).
pub fn unquote(name: &str) -> String {
    let name = stripped(name);
    match name
        .strip_prefix('`')
        .and_then(|rest| rest.strip_suffix('`'))
    {
        Some(inner) if name.len() >= 2 => inner.replace("``", "`"),
        _ => name.to_owned(),
    }
}

fn name_parts(regex: &Regex, name: &str) -> Option<Vec<String>> {
    let captures = regex.captures(name).ok()??;
    Some(
        (1..=3)
            .filter_map(|group| captures.get(group))
            .map(|part| unquote(part.as_str()))
            .collect(),
    )
}

/// A plain, one-to-three-part table name's parts, unquoted; `None` for
/// anything else, a query say (`parse_table_reference`).
pub fn table_reference(name: &str) -> Option<Vec<String>> {
    name_parts(&TABLE_REFERENCE, name)
}

/// A metric view's name as a caller passes it, its parts unquoted; an
/// unquoted part may hold what SQL reads only quoted (`parse_view_name`).
pub fn view_name(name: &str) -> Option<Vec<String>> {
    name_parts(&VIEW_NAME, name)
}

fn key_ref(chain: &str) -> KeyRef {
    let mut parts: Vec<String> = IDENT
        .find_iter(chain)
        .filter_map(Result::ok)
        .map(|part| unquote(part.as_str()))
        .collect();
    let column = parts.pop().unwrap_or_default();
    KeyRef {
        qualifier: (!parts.is_empty()).then(|| parts.join(".")),
        column,
    }
}

/// A single-column equality join condition's two sides; `None` for any
/// other (`parse_join_condition`).
pub fn join_condition(condition: &str) -> Option<[KeyRef; 2]> {
    let captures = JOIN_CONDITION.captures(condition).ok()??;
    Some([
        key_ref(captures.get(1)?.as_str()),
        key_ref(captures.get(2)?.as_str()),
    ])
}

/// A wildcard dimension, `alias.*` or `alias.* EXCEPT (a, b)`: the folded
/// path it names, and the columns it leaves out, unquoted.
pub fn wildcard(expr: &str) -> Option<(Vec<String>, Vec<String>)> {
    let captures = WILDCARD.captures(expr).ok()??;
    let path = IDENT
        .find_iter(captures.get(1)?.as_str())
        .filter_map(Result::ok)
        .map(|part| unquote(part.as_str()).to_lowercase())
        .collect();
    let excluded = captures.get(2).map_or_else(Vec::new, |list| {
        CHAIN
            .find_iter(list.as_str())
            .filter_map(Result::ok)
            .filter_map(|chain| {
                IDENT
                    .find_iter(chain.as_str())
                    .filter_map(Result::ok)
                    .last()
                    .map(|last| unquote(last.as_str()))
            })
            .collect()
    });
    Some((path, excluded))
}

/// The pattern of `part`, backticked or, where it is a bare name, bare.
fn quoted_variants(part: &str) -> String {
    let quoted = fancy_regex::escape(&format!("`{}`", part.replace('`', "``"))).into_owned();
    if matches(&WHOLE_BARE_NAME, part) {
        format!("{quoted}|{}", fancy_regex::escape(part))
    } else {
        quoted
    }
}

/// The pattern of a qualifier `path.` in any case, not after a name's
/// character, a backtick, `$` or a dot (`_path_pattern`).
fn path_pattern<S: AsRef<str>>(path: &[S]) -> Regex {
    let chain = path
        .iter()
        .map(|part| format!("(?:{})", quoted_variants(part.as_ref())))
        .collect::<Vec<_>>()
        .join(r"\s*\.\s*");
    pattern(&format!(r"(?i)(?<![A-Za-z0-9_`$.]){chain}\s*\."))
}

/// `expr` with each string literal emptied (`_masked_strings`).
fn masked_strings(expr: &str) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for span in string_literals(expr) {
        out.push_str(&expr[last..span.start]);
        out.push_str("''");
        last = span.end;
    }
    out.push_str(&expr[last..]);
    out
}

/// Whether `expr` reads a column of the table `path` names, outside its
/// literals (`references_alias`).
pub fn references_alias<S: AsRef<str>>(expr: &str, path: &[S]) -> bool {
    matches(&path_pattern(path), &masked_strings(expr))
}

/// `expr` with every qualifier `path.` taken out, its literals as they are,
/// trimmed (`strip_alias_qualifier`).
pub fn strip_alias<S: AsRef<str>>(expr: &str, path: &[S]) -> String {
    let pattern = path_pattern(path);
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for span in string_literals(expr) {
        out.push_str(&pattern.replace_all(&expr[last..span.start], ""));
        out.push_str(&expr[span.clone()]);
        last = span.end;
    }
    out.push_str(&pattern.replace_all(&expr[last..], ""));
    stripped(&out).to_owned()
}

/// The column `expr` truncates or casts to a date or time, as
/// `order_date` in `DATE_TRUNC('MONTH', order_date)`; `None` for anything
/// else, a bare column included (`derived_column`).
pub fn derived_column(expr: &str) -> Option<String> {
    let column = column_under(expr)?;
    let bare = stripped(expr).trim_matches(['(', ')', ' ']);
    (!matches(&WHOLE_IDENT, bare)).then_some(column)
}

fn column_under(expr: &str) -> Option<String> {
    let expr = stripped(expr);
    if matches(&WHOLE_IDENT, expr) {
        return Some(unquote(expr));
    }
    for derivation in DERIVATIONS.iter() {
        let Ok(Some(captures)) = derivation.captures(expr) else {
            continue;
        };
        let inner = captures.name("inner").unwrap().as_str();
        // `.+` reaches past a bracket that closes early, as in
        // `date(a) - date(b)`, so what it holds must balance.
        let mut depth: i64 = 0;
        for c in masked_strings(inner).chars() {
            depth += match c {
                '(' => 1,
                ')' => -1,
                _ => 0,
            };
            if depth < 0 {
                break;
            }
        }
        if depth == 0
            && let Some(column) = column_under(inner)
        {
            return Some(column);
        }
    }
    None
}

/// The column an expression is, if it is one column and nothing else
/// (`_renamed_column`).
pub fn renamed_column(expr: &str) -> Option<String> {
    let captures = RENAMED.captures(stripped(expr)).ok()??;
    match (captures.get(1), captures.get(2)) {
        (Some(quoted), _) => Some(quoted.as_str().replace("``", "`")),
        (None, Some(bare)) => Some(bare.as_str().to_owned()),
        _ => None,
    }
}

/// A parameter's default as the data layer reads it: the kind of value it
/// is, and the text Python's `str` gives it, which the core writes.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterDefault {
    pub kind: DefaultKind,
    pub text: String,
}

/// The kinds of default `_default_text` tells apart.
#[derive(Clone, Debug, PartialEq)]
pub enum DefaultKind {
    Bool(bool),
    Int(i128),
    /// A date with no time of day: its text is its ISO form.
    Date,
    /// A date and time: its text is its ISO form with a space, and its
    /// `isoformat()` with a `T` is `iso`.
    DateTime {
        iso: String,
    },
    Other,
}

/// Why a default gives no literal of its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultRefusal {
    /// The type has no literal here.
    NoLiteral,
    /// The default is no value of the type.
    NotOfType,
}

/// The text a parameter's default casts from, checked against its type so
/// that nothing but the value reaches the SQL (`_default_text`).
pub fn default_text(data_type: &str, default: &ParameterDefault) -> Result<String, DefaultRefusal> {
    let folded = data_type
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let string_type = matches(&STRING_TYPE, &folded);
    let text = stripped(&default.text);
    let answer = if let DefaultKind::Bool(flag) = default.kind {
        (folded == "boolean" || string_type).then(|| flag.to_string())
    } else if folded == "boolean" {
        let lower = text.to_lowercase();
        (lower == "true" || lower == "false").then_some(lower)
    } else if INTEGER_TYPES.contains(&folded.as_str()) {
        match default.kind {
            DefaultKind::Int(n) => Some(n.to_string()),
            _ if matches(&INTEGER_TEXT, text) => integer_text(text),
            _ => None,
        }
    } else if FLOAT_TYPES.contains(&folded.as_str()) || matches(&DECIMAL_TYPE, &folded) {
        decimal_text(text)
    } else if string_type {
        Some(match &default.kind {
            DefaultKind::DateTime { iso } => iso.clone(),
            _ => default.text.clone(),
        })
    } else if folded == "date" {
        match default.kind {
            DefaultKind::Date => Some(default.text.clone()),
            DefaultKind::DateTime { .. } => None,
            _ => iso_date(text)
                .filter(|(_, used)| *used == text.len())
                .map(|(day, _)| day),
        }
    } else if TIMESTAMP_TYPES.contains(&folded.as_str()) {
        match default.kind {
            DefaultKind::Date | DefaultKind::DateTime { .. } => Some(default.text.clone()),
            _ => iso_datetime(text),
        }
    } else {
        return Err(DefaultRefusal::NoLiteral);
    };
    answer.ok_or(DefaultRefusal::NotOfType)
}

/// `str(int(text))` for a text of digits, an optional sign first.
fn integer_text(text: &str) -> Option<String> {
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    let digits: String = digits
        .chars()
        .map(|c| c.to_digit(10).map_or(c, |d| char::from(b'0' + d as u8)))
        .collect();
    let digits = digits.trim_start_matches('0');
    Some(match (negative, digits) {
        (_, "") => "0".into(),
        (true, d) => format!("-{d}"),
        (false, d) => d.to_owned(),
    })
}

/// `format(Decimal(text), 'f')` for a finite decimal: the digits of its
/// coefficient placed by its exponent, no exponent written.
fn decimal_text(text: &str) -> Option<String> {
    let text = text.replace('_', "");
    let (negative, rest) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text.as_str()),
    };
    let (mantissa, exponent) = match rest.find(['e', 'E']) {
        Some(at) => (&rest[..at], rest[at + 1..].parse::<i64>().ok()?),
        None => (rest, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if (whole.is_empty() && fraction.is_empty())
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{whole}{fraction}");
    let digits = match digits.trim_start_matches('0') {
        "" => "0",
        kept => kept,
    };
    let exponent = exponent - i64::try_from(fraction.len()).ok()?;
    let sign = if negative { "-" } else { "" };
    if exponent >= 0 {
        if digits == "0" {
            return Some(format!("{sign}0"));
        }
        return Some(format!(
            "{sign}{digits}{}",
            "0".repeat(usize::try_from(exponent).ok()?)
        ));
    }
    let places = usize::try_from(-exponent).ok()?;
    let padded = if digits.len() <= places {
        format!("{}{digits}", "0".repeat(places - digits.len() + 1))
    } else {
        digits.to_owned()
    };
    let (whole, fraction) = padded.split_at(padded.len() - places);
    Some(format!("{sign}{whole}.{fraction}"))
}

/// A date in one of the ISO forms Python 3.12's `date.fromisoformat`
/// reads (`2024-01-01`, `20240101`, `2024-W01-2`, `2024W012`), as
/// `YYYY-MM-DD`, with how many characters it took.
fn iso_date(text: &str) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let digits = |from: usize, count: usize| -> Option<u32> {
        let slice = text.get(from..from + count)?;
        slice
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| slice.parse().ok())?
    };
    let year = digits(0, 4)?;
    let extended = bytes.get(4) == Some(&b'-');
    let at = if extended { 5 } else { 4 };
    let (date, used) = if bytes.get(at) == Some(&b'W') {
        let week = digits(at + 1, 2)?;
        let day_at = at + 3 + usize::from(extended);
        let (day, used) = match (extended, bytes.get(at + 3)) {
            (true, Some(b'-')) => (digits(day_at, 1)?, day_at + 1),
            (false, Some(b)) if b.is_ascii_digit() => (digits(day_at, 1)?, day_at + 1),
            _ => (1, at + 3),
        };
        (iso_week_date(year, week, day)?, used)
    } else {
        let month = digits(at, 2)?;
        let day_at = at + 2 + usize::from(extended);
        if extended && bytes.get(at + 2) != Some(&b'-') {
            return None;
        }
        let day = digits(day_at, 2)?;
        ((year, month, day), day_at + 2)
    };
    let (year, month, day) = date;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1..=12 => 31,
        _ => return None,
    };
    (year >= 1 && (1..=days).contains(&day))
        .then(|| (format!("{year:04}-{month:02}-{day:02}"), used))
}

/// The calendar date of ISO year `year`'s week `week`, day `day`.
fn iso_week_date(year: u32, week: u32, day: u32) -> Option<(u32, u32, u32)> {
    if !(1..=53).contains(&week) || !(1..=7).contains(&day) {
        return None;
    }
    // Days from 0001-01-01 (a Monday) to `year`-01-04, which week 1 holds.
    let ordinal = |y: u32, m: u32, d: u32| -> i64 {
        let y = i64::from(y) - 1;
        let leap = |y: i64| y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let before: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let mut days =
            y * 365 + y / 4 - y / 100 + y / 400 + before[(m - 1) as usize] + i64::from(d);
        if m > 2 && leap(y + 1) {
            days += 1;
        }
        days
    };
    let jan4 = ordinal(year, 1, 4);
    let week1 = jan4 - (jan4 - 1).rem_euclid(7);
    let target = week1 + i64::from(week - 1) * 7 + i64::from(day - 1);
    // Back to a calendar date.
    let mut y = u32::try_from((target * 400 / 146_097).max(1)).ok()?;
    while ordinal(y + 1, 1, 1) <= target {
        y += 1;
    }
    while ordinal(y, 1, 1) > target {
        y -= 1;
    }
    let mut m = 12;
    while ordinal(y, m, 1) > target {
        m -= 1;
    }
    let d = u32::try_from(target - ordinal(y, m, 1) + 1).ok()?;
    // Week 53 exists only in a year that has it.
    let next_week1 = {
        let jan4 = ordinal(year + 1, 1, 4);
        jan4 - (jan4 - 1).rem_euclid(7)
    };
    (target < next_week1).then_some((y, m, d))
}

/// A date and time in the ISO forms Python 3.12's `datetime.fromisoformat`
/// reads, written as `isoformat(' ')` writes it.
fn iso_datetime(text: &str) -> Option<String> {
    let (day, used) = iso_date(text)?;
    let rest = &text[used..];
    if rest.is_empty() {
        return Some(format!("{day} 00:00:00"));
    }
    // Any one character stands between the date and the time.
    let mut chars = rest.chars();
    chars.next()?;
    let time = chars.as_str();
    let zone_at = time.find(['Z', '+', '-']).unwrap_or(time.len());
    let (clock, zone) = time.split_at(zone_at);
    let (hour, minute, second, micro) = iso_clock(clock)?;
    let mut out = format!("{day} {hour:02}:{minute:02}:{second:02}");
    if micro > 0 {
        out.push_str(&format!(".{micro:06}"));
    }
    if !zone.is_empty() {
        out.push_str(&iso_zone(zone)?);
    }
    Some(out)
}

/// `HH[:MM[:SS[.f]]]` or `HH[MM[SS[.f]]]`, a fraction cut to
/// microseconds.
fn iso_clock(clock: &str) -> Option<(u32, u32, u32, u32)> {
    let (main, fraction) = match clock.find(['.', ',']) {
        Some(at) => (&clock[..at], Some(&clock[at + 1..])),
        None => (clock, None),
    };
    let parts: Vec<&str> = if main.contains(':') {
        main.split(':').collect()
    } else {
        (0..main.len())
            .step_by(2)
            .map(|at| main.get(at..at + 2).unwrap_or(""))
            .collect()
    };
    if parts.is_empty()
        || parts.len() > 3
        || parts
            .iter()
            .any(|p| p.len() != 2 || !p.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let field = |i: usize| parts.get(i).map_or(0, |p| p.parse().unwrap());
    let (hour, minute, second) = (field(0), field(1), field(2));
    if fraction.is_some() && parts.len() != 3 {
        return None;
    }
    let micro = match fraction {
        None => 0,
        Some(f) if !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()) => {
            let mut digits: String = f.chars().take(6).collect();
            while digits.len() < 6 {
                digits.push('0');
            }
            digits.parse().unwrap()
        }
        Some(_) => return None,
    };
    (hour < 24 && minute < 60 && second < 60).then_some((hour, minute, second, micro))
}

/// `Z`, or `±HH[:MM[:SS[.f]]]` or `±HHMM[SS]`, written as `+HH:MM[:SS]`.
fn iso_zone(zone: &str) -> Option<String> {
    if zone == "Z" {
        return Some("+00:00".into());
    }
    let (sign, rest) = zone.split_at(1);
    let (hour, minute, second, micro) = iso_clock(rest)?;
    let mut out = format!("{sign}{hour:02}:{minute:02}");
    if second > 0 || micro > 0 {
        out.push_str(&format!(":{second:02}"));
    }
    if micro > 0 {
        out.push_str(&format!(".{micro:06}"));
    }
    Some(out)
}

/// What a parameter of the view is given, as the core hands it over: its
/// name, its type as the definition writes it if text, and its default
/// unless null.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterInput {
    pub name: String,
    pub data_type: Option<String>,
    pub default: Option<ParameterDefault>,
}

/// A parameter as a query of the view without arguments reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub read: Result<ParameterRead, Unread>,
}

/// A parameter's default as a typed SQL literal, and as text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterRead {
    pub literal: String,
    pub text: String,
}

/// Why a parameter has no literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unread {
    NoDefault,
    NoType,
    /// Its default is refused for its type, which is given.
    Refused {
        refusal: DefaultRefusal,
        data_type: String,
    },
}

/// A Databricks string literal holding `text`, a quote and a backslash
/// escaped with a backslash (`_sql_string`).
fn sql_string(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// The `parameters` section read, by folded name, a later one replacing an
/// earlier of the same name in its place (`_read_parameters`). The type
/// Databricks describes, `described`, wins over the definition's.
pub fn read_parameters(
    inputs: &[ParameterInput],
    described: &[(String, String)],
) -> Vec<(String, Parameter)> {
    let mut parameters: Vec<(String, Parameter)> = Vec::new();
    for input in inputs {
        let folded = input.name.to_lowercase();
        let data_type = described
            .iter()
            .rev()
            .find(|(name, _)| name.to_lowercase() == folded)
            .map(|(_, data_type)| data_type.clone())
            .or_else(|| input.data_type.clone());
        let read = match (&input.default, data_type) {
            (None, _) => Err(Unread::NoDefault),
            (Some(_), None) => Err(Unread::NoType),
            (Some(default), Some(data_type)) => match default_text(&data_type, default) {
                Ok(text) => {
                    let sql_type = data_type
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_uppercase();
                    Ok(ParameterRead {
                        literal: format!("CAST({} AS {sql_type})", sql_string(&text)),
                        text,
                    })
                }
                Err(refusal) => Err(Unread::Refused { refusal, data_type }),
            },
        };
        let parameter = Parameter {
            name: input.name.clone(),
            read,
        };
        match parameters.iter_mut().find(|(name, _)| *name == folded) {
            Some((_, slot)) => *slot = parameter,
            None => parameters.push((folded, parameter)),
        }
    }
    parameters
}

/// `expr` with each bare reference to a parameter that has a literal
/// replaced by it, and the names of every parameter it references, once
/// each, in order (`_substitute_parameters`).
pub fn substitute_parameters(
    expr: &str,
    parameters: &[(String, Parameter)],
) -> Result<(String, Vec<String>), Unreadable> {
    let found: Vec<(std::ops::Range<usize>, &Parameter)> = name_references(expr)?
        .into_iter()
        .filter(|reference| reference.bare)
        .filter_map(|reference| {
            let folded = reference.parts[0].to_lowercase();
            let parameter = parameters.iter().find(|(name, _)| *name == folded)?;
            Some((reference.span, &parameter.1))
        })
        .collect();
    let mut out = expr.to_owned();
    for (span, parameter) in found.iter().rev() {
        if let Ok(read) = &parameter.read {
            out.replace_range(span.clone(), &read.literal);
        }
    }
    let mut used: Vec<String> = Vec::new();
    for (_, parameter) in &found {
        if !used.contains(&parameter.name) {
            used.push(parameter.name.clone());
        }
    }
    Ok((out, used))
}

/// `expr` with each reference to an earlier field replaced by its
/// expression in brackets, and the name, as written, of a refused field it
/// references (`_substitute_fields`). `fields` are each earlier field's
/// expression by folded name, `None` for one refused; `qualifiers` every
/// join path or alias, folded.
pub fn substitute_fields(
    expr: &str,
    fields: &[(String, Option<String>)],
    qualifiers: &[Vec<String>],
) -> Result<(String, Option<String>), Unreadable> {
    let mut out = expr.to_owned();
    let mut refused = None;
    for reference in name_references(expr)?.into_iter().rev() {
        let parts: Vec<String> = reference.parts.iter().map(|p| p.to_lowercase()).collect();
        let qualified = qualifiers
            .iter()
            .any(|path| parts.len() > path.len() && parts[..path.len()] == path[..]);
        let Some((_, field)) = fields.iter().find(|(name, _)| *name == parts[0]) else {
            continue;
        };
        if qualified {
            continue;
        }
        let Some(field) = field else {
            refused = Some(reference.parts[0].clone());
            continue;
        };
        let start = reference.span.start;
        let first = FIRST_PART
            .find(&out[start..])
            .ok()
            .flatten()
            .map_or(start, |m| start + m.end());
        out.replace_range(start..first, &format!("({field})"));
    }
    Ok((out, refused))
}

/// Whether `expr` reads every column of a table through `*`, as
/// `struct(*)` does; `COUNT(*)` counts rows instead (`_star`).
pub fn reads_star(expr: &str) -> Result<bool, Unreadable> {
    Ok(name_references(expr)?
        .iter()
        .any(|reference| reference.parts.last().is_some_and(|p| p == "*")))
}

/// Why the filter cannot be applied to the source table alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterRefusal {
    /// It is no text, or blank.
    NotACondition,
    Unsafe(ExprRefusal),
    /// It reads a joined table, by alias.
    JoinedTable(String),
}

/// The filter, trimmed, if it can be applied to the source table alone
/// (`_check_filter`): text, safe, and reading no joined table. `filter` is
/// `None` for a filter that is no text; `candidates` each join path or
/// alias with the alias it resolves to, longest first.
pub fn check_filter(
    filter: Option<&str>,
    candidates: &[(Vec<String>, String)],
) -> Result<String, FilterRefusal> {
    let Some(filter) = filter.filter(|f| !stripped(f).is_empty()) else {
        return Err(FilterRefusal::NotACondition);
    };
    if let Some(reason) = unsafe_reason(filter, Some(ExprDialect::Databricks)) {
        return Err(FilterRefusal::Unsafe(reason));
    }
    let masked = masked_strings(filter);
    for (path, alias) in candidates {
        if alias != SOURCE_ALIAS && references_alias(&masked, path) {
            return Err(FilterRefusal::JoinedTable(alias.clone()));
        }
    }
    Ok(stripped(filter).to_owned())
}

/// A measure as the core hands it over: an entry of `measures` that is a
/// mapping with a text name, its `expr` if text, and whether it has a
/// `window`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureInput {
    pub name: String,
    pub expr: Option<String>,
    pub window: bool,
}

/// A dimension resolved onto one table: its name, the table's alias, and
/// its expression with that table's qualifier taken out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionRef {
    pub name: String,
    pub alias: String,
    pub expr: String,
}

/// What a measure reads: each column with its table's alias, and the
/// predictive query aggregation it is, if one plain aggregate of the
/// source table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureRead {
    pub name: String,
    pub inputs: Vec<(String, String)>,
    pub aggregation: Option<(String, Option<String>)>,
}

/// A measure left out, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureRefusal {
    pub name: String,
    pub reason: MeasureReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasureReason {
    Unreadable(Unreadable),
    /// It names a refused dimension, as written.
    NamesRefused(String),
}

/// Which columns each measure reads, never running it (`_read_measures`).
/// A name that is a dimension, a measure or a parameter is no column to
/// add, and a measure over more than one table reads none.
pub fn read_measures(
    measures: &[MeasureInput],
    candidates: &[(Vec<String>, String)],
    kept: &[String],
    dimensions: &[(String, DimensionRef)],
    refused: &[String],
    parameters: &[String],
) -> (Vec<MeasureRead>, Vec<MeasureRefusal>) {
    let paths: Vec<(Vec<String>, &str)> = {
        let mut paths: Vec<(Vec<String>, &str)> = Vec::new();
        for (path, alias) in candidates {
            let folded: Vec<String> = path.iter().map(|p| p.to_lowercase()).collect();
            match paths.iter_mut().find(|(p, _)| *p == folded) {
                Some(slot) => slot.1 = alias,
                None => paths.push((folded, alias)),
            }
        }
        paths
    };
    let path_alias = |folded: &[String]| {
        paths
            .iter()
            .find(|(p, _)| p.as_slice() == folded)
            .map(|(_, alias)| *alias)
    };
    let dimension = |folded: &str| {
        dimensions
            .iter()
            .find(|(name, _)| name == folded)
            .map(|(_, d)| d)
    };
    let measure_names: Vec<String> = measures.iter().map(|m| m.name.to_lowercase()).collect();
    let (mut read, mut left_out) = (Vec::new(), Vec::new());
    for measure in measures {
        let Some(expr) = &measure.expr else {
            continue;
        };
        let expr = super::strip_comments(expr);
        let names = match name_references(&expr) {
            Ok(names) => names,
            Err(error) => {
                left_out.push(MeasureRefusal {
                    name: measure.name.clone(),
                    reason: MeasureReason::Unreadable(error),
                });
                continue;
            }
        };
        let named_refused = names.iter().find(|named| {
            let folded = named.parts[0].to_lowercase();
            named.parts.len() == 1
                && refused.contains(&folded)
                && !measure_names.contains(&folded)
                && !parameters.contains(&folded)
        });
        if let Some(named) = named_refused {
            left_out.push(MeasureRefusal {
                name: measure.name.clone(),
                reason: MeasureReason::NamesRefused(named.parts[0].clone()),
            });
            continue;
        }
        let mut inputs: Vec<(String, String)> = Vec::new();
        let mut tables: Vec<String> = Vec::new();
        let reads = |alias: &str, tables: &mut Vec<String>| {
            if !tables.iter().any(|t| t == alias) {
                tables.push(alias.to_owned());
            }
        };
        for named in &names {
            let parts = &named.parts;
            if parts.last().is_some_and(|p| p == "*") {
                let table: Vec<String> = parts[..parts.len() - 1]
                    .iter()
                    .map(|p| p.to_lowercase())
                    .collect();
                if !table.is_empty()
                    && let Some(alias) = path_alias(&table)
                {
                    reads(alias, &mut tables);
                }
                continue;
            }
            let reference = if parts.len() == 1 {
                let folded = parts[0].to_lowercase();
                if measure_names.contains(&folded)
                    || parameters.contains(&folded)
                    || path_alias(std::slice::from_ref(&folded)).is_some()
                {
                    continue;
                }
                if let Some(dimension) = dimension(&folded) {
                    reads(&dimension.alias, &mut tables);
                    continue;
                }
                (SOURCE_ALIAS.to_owned(), parts[0].clone())
            } else {
                let qualifier: Vec<String> = parts[..parts.len() - 1]
                    .iter()
                    .map(|p| p.to_lowercase())
                    .collect();
                let Some(alias) = path_alias(&qualifier) else {
                    continue; // A field of a struct, or nothing known.
                };
                (alias.to_owned(), parts[parts.len() - 1].clone())
            };
            reads(&reference.0, &mut tables);
            if !inputs.contains(&reference) {
                inputs.push(reference);
            }
        }
        let readable = |alias: &String| alias == SOURCE_ALIAS || kept.contains(alias);
        if tables.len() > 1 || !tables.iter().all(readable) {
            inputs.clear();
        }
        let aggregation = aggregation(measure.window, &expr, &inputs, &dimension);
        read.push(MeasureRead {
            name: measure.name.clone(),
            inputs,
            aggregation,
        });
    }
    (read, left_out)
}

/// The predictive query aggregation a measure is, if one plain aggregate of
/// a column or dimension of the source table, or a count of its rows
/// (`_aggregation`).
fn aggregation<'a>(
    window: bool,
    expr: &str,
    inputs: &[(String, String)],
    dimension: &dyn Fn(&str) -> Option<&'a DimensionRef>,
) -> Option<(String, Option<String>)> {
    if window {
        return None;
    }
    let masked = masked_strings(expr);
    let captures = SIMPLE_MEASURE.captures(&masked).ok()??;
    let function = captures.get(1)?.as_str().to_uppercase();
    let distinct = captures.get(2).is_some();
    let argument = captures.get(3)?.as_str();
    if argument == "*" || argument == "1" {
        return (function == "COUNT" && !distinct).then(|| ("COUNT".to_owned(), None));
    }
    let column = match dimension(&unquote(argument).to_lowercase()) {
        Some(d) if d.alias == SOURCE_ALIAS => d.name.clone(),
        _ if inputs.len() == 1 && inputs[0].0 == SOURCE_ALIAS => inputs[0].1.clone(),
        _ => return None,
    };
    if distinct {
        return (function == "COUNT").then(|| ("COUNT_DISTINCT".to_owned(), Some(column)));
    }
    Some((function, Some(column)))
}

/// A join as its keys are resolved: its alias, the alias it is declared
/// on, its path, and its condition's two sides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinRef {
    pub alias: String,
    pub parent_alias: String,
    pub path: Vec<String>,
    pub condition: [KeyRef; 2],
}

/// A join condition resolved into `(other_alias, other_key, child_key)`,
/// unqualified names the child's first, as Databricks reads them; `None`
/// when it cannot be (`resolve_join_keys`).
pub fn resolve_join_keys<S: AsRef<str>>(
    join: &JoinRef,
    child_columns: &[S],
    parent_columns: &[S],
) -> Option<(String, String, String)> {
    let child_names = [
        join.alias.to_lowercase(),
        join.path.join(".").to_lowercase(),
    ];
    let mut parent_names = vec![join.parent_alias.to_lowercase()];
    if join.path.len() > 1 {
        parent_names.push(join.path[..join.path.len() - 1].join(".").to_lowercase());
    }
    let holds = |columns: &[S], name: &str| {
        let folded = name.to_lowercase();
        columns.iter().any(|c| c.as_ref().to_lowercase() == folded)
    };
    let side = |key: &KeyRef| -> Option<String> {
        match &key.qualifier {
            Some(qualifier) => {
                let qualifier = qualifier.to_lowercase();
                if child_names.contains(&qualifier) {
                    Some(join.alias.clone())
                } else if parent_names.contains(&qualifier) {
                    Some(join.parent_alias.clone())
                } else if qualifier == SOURCE_ALIAS {
                    Some(SOURCE_ALIAS.to_owned())
                } else {
                    None
                }
            }
            None if holds(child_columns, &key.column) => Some(join.alias.clone()),
            None if holds(parent_columns, &key.column) => Some(join.parent_alias.clone()),
            None => None,
        }
    };
    let [first, second] = &join.condition;
    let (first_side, second_side) = (side(first), side(second));
    let child = Some(join.alias.clone());
    if first_side == child && second_side.is_some() && second_side != child {
        return Some((second_side?, second.column.clone(), first.column.clone()));
    }
    if second_side == child && first_side.is_some() && first_side != child {
        return Some((first_side?, first.column.clone(), second.column.clone()));
    }
    None
}
