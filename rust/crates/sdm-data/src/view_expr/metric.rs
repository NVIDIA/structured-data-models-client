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

use super::check::{ExprRefusal, unsafe_reason};
use super::lex::{bare_name, string_literals};
use super::names::{Unreadable, name_references};
use super::pattern::{Lazy, Pattern, Read};
use super::{ExprDialect, Spaces};

fn pattern(text: &str, spaces: &Spaces) -> Pattern {
    Pattern::new(text, spaces)
}

/// A part of a name: backticked, or a run Databricks reads as a name.
fn ident() -> String {
    format!("`(?:[^`]|``)+`|{}", bare_name())
}

fn chain() -> String {
    let ident = ident();
    format!(r"(?:{ident})(?:\s*\.\s*(?:{ident}))*")
}

static TABLE_REFERENCE: Lazy<Pattern> = Lazy::new(|spaces| {
    let ident = ident();
    pattern(
        &format!(r"^\s*({ident})(?:\.({ident}))?(?:\.({ident}))?\s*$"),
        spaces,
    )
});
// An unquoted part of a view's name is any run of characters but a dot, a
// backtick and white space, as a caller writes `my-catalog` unquoted.
static VIEW_NAME: Lazy<Pattern> = Lazy::new(|spaces| {
    let part = r"`(?:[^`]|``)+`|[^\s.`]+";
    pattern(
        &format!(r"^\s*({part})(?:\.({part}))?(?:\.({part}))?\s*$"),
        spaces,
    )
});
static JOIN_CONDITION: Lazy<Pattern> = Lazy::new(|spaces| {
    let chain = chain();
    pattern(&format!(r"^\s*({chain})\s*=\s*({chain})\s*$"), spaces)
});
static IDENT: Lazy<Pattern> = Lazy::new(|spaces| pattern(&ident(), spaces));
static WHOLE_IDENT: Lazy<Pattern> =
    Lazy::new(|spaces| pattern(&format!("^(?:{})$", ident()), spaces));
static CHAIN: Lazy<Pattern> = Lazy::new(|spaces| pattern(&chain(), spaces));
// `- expr: source.*` or `- expr: customers.* EXCEPT (a, b)`.
static WILDCARD: Lazy<Pattern> = Lazy::new(|spaces| {
    let chain = chain();
    pattern(
        &format!(
            r"(?i)^\s*({chain})\s*\.\s*\*(?:\s*EXCEPT\s*\(\s*({chain}(?:\s*,\s*{chain})*)\s*\))?\s*$"
        ),
        spaces,
    )
});
static RENAMED: Lazy<Pattern> =
    Lazy::new(|spaces| pattern(r"^(?:`((?:[^`]|``)+)`|([A-Za-z_]\w*))$", spaces));
static WHOLE_BARE_NAME: Lazy<Pattern> =
    Lazy::new(|spaces| pattern(&format!("^(?:{})$", bare_name()), spaces));
// What a timestamp dimension may wrap a column in and still say when the
// row happened: a truncation, or a cast to a date or time.
static DERIVATIONS: Lazy<Vec<Pattern>> = Lazy::new(|spaces| {
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
    .map(|p| pattern(&format!("(?is)^(?:{p})$"), spaces))
    .collect()
});
// The first part of a name, backticked or not.
static FIRST_PART: Lazy<Pattern> =
    Lazy::new(|spaces| pattern(r"^(?:`(?:``|[^`])*`|[A-Za-z0-9_]+)", spaces));
static SIMPLE_MEASURE: Lazy<Pattern> = Lazy::new(|spaces| {
    pattern(
        &format!(
            r"(?i)^\s*(sum|avg|min|max|count)\s*\(\s*(distinct\s+)?(\*|1|{})\s*\)\s*$",
            chain()
        ),
        spaces,
    )
});
static DECIMAL_TYPE: Lazy<Pattern> = Lazy::new(|spaces| {
    pattern(
        r"^(?:decimal|dec|numeric)(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?$",
        spaces,
    )
});
static STRING_TYPE: Lazy<Pattern> =
    Lazy::new(|spaces| pattern(r"^(?:string|(?:var)?char\(\s*\d+\s*\))$", spaces));

const SOURCE_ALIAS: &str = "source";

/// The old client's words for the `fields` section passed over for the
/// `dimensions` (`DB/metric_view.py:953-957`), which name the warehouse.
pub const FIELDS_IGNORED: &str = "Ignored the 'fields' section since the view also has 'dimensions', and Databricks reads \
     only those";
const INTEGER_TYPES: &[&str] = &[
    "bigint", "byte", "int", "integer", "long", "short", "smallint", "tinyint",
];
const FLOAT_TYPES: &[&str] = &["double", "float", "real"];
const TIMESTAMP_TYPES: &[&str] = &["timestamp", "timestamp_ltz", "timestamp_ntz"];

/// Whether `pattern` matches, where a failed match may only refuse: a name
/// not read as one, a type with no literal, no derived column.
fn matches(pattern: &Pattern, text: &str) -> bool {
    pattern.is_match(text).unwrap_or(false)
}

/// One side of a join condition: the (dot-joined) table alias or join path
/// qualifying the column, if any, and the column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRef {
    pub qualifier: Option<String>,
    pub column: String,
}

/// A name without its backticks, if it is backticked (`unquote_ident`).
pub fn unquote(name: &str, spaces: &Spaces) -> String {
    let name = spaces.strip(name);
    match name
        .strip_prefix('`')
        .and_then(|rest| rest.strip_suffix('`'))
    {
        Some(inner) if name.len() >= 2 => inner.replace("``", "`"),
        _ => name.to_owned(),
    }
}

fn name_parts(pattern: &Pattern, name: &str, spaces: &Spaces) -> Option<Vec<String>> {
    let groups = pattern.groups(name).ok()??;
    Some(
        (1..=3)
            .filter_map(|group| groups.get(group))
            .map(|part| unquote(&name[part], spaces))
            .collect(),
    )
}

/// A plain, one-to-three-part table name's parts, unquoted; `None` for
/// anything else, a query say (`parse_table_reference`).
pub fn table_reference(name: &str, spaces: &Spaces) -> Option<Vec<String>> {
    name_parts(&TABLE_REFERENCE.get(spaces), name, spaces)
}

/// A metric view's name as a caller passes it, its parts unquoted; an
/// unquoted part may hold what SQL reads only quoted (`parse_view_name`).
pub fn view_name(name: &str, spaces: &Spaces) -> Option<Vec<String>> {
    name_parts(&VIEW_NAME.get(spaces), name, spaces)
}

fn key_ref(chain: &str, spaces: &Spaces) -> KeyRef {
    let mut parts: Vec<String> = IDENT
        .get(spaces)
        .find_all(chain)
        .unwrap_or_default()
        .into_iter()
        .map(|part| unquote(&chain[part], spaces))
        .collect();
    let column = parts.pop().unwrap_or_default();
    KeyRef {
        qualifier: (!parts.is_empty()).then(|| parts.join(".")),
        column,
    }
}

/// A single-column equality join condition's two sides; `None` for any
/// other (`parse_join_condition`).
pub fn join_condition(condition: &str, spaces: &Spaces) -> Option<[KeyRef; 2]> {
    let groups = JOIN_CONDITION.get(spaces).groups(condition).ok()??;
    Some([
        key_ref(&condition[groups.get(1)?], spaces),
        key_ref(&condition[groups.get(2)?], spaces),
    ])
}

/// A wildcard dimension, `alias.*` or `alias.* EXCEPT (a, b)`: the folded
/// path it names, and the columns it leaves out, unquoted.
pub fn wildcard(expr: &str, spaces: &Spaces) -> Option<(Vec<String>, Vec<String>)> {
    let groups = WILDCARD.get(spaces).groups(expr).ok()??;
    let named = &expr[groups.get(1)?];
    let path = IDENT
        .get(spaces)
        .find_all(named)
        .ok()?
        .into_iter()
        .map(|part| unquote(&named[part], spaces).to_lowercase())
        .collect();
    let excluded = match groups.get(2) {
        None => Vec::new(),
        Some(list) => {
            let list = &expr[list];
            let mut excluded = Vec::new();
            for chain in CHAIN.get(spaces).find_all(list).ok()? {
                let chain = &list[chain];
                if let Some(last) = IDENT.get(spaces).find_all(chain).ok()?.pop() {
                    excluded.push(unquote(&chain[last], spaces));
                }
            }
            excluded
        }
    };
    Some((path, excluded))
}

/// The pattern of `part`, backticked or, where it is a bare name, bare.
fn quoted_variants(part: &str, spaces: &Spaces) -> String {
    let quoted = fancy_regex::escape(&format!("`{}`", part.replace('`', "``"))).into_owned();
    if matches(&WHOLE_BARE_NAME.get(spaces), part) {
        format!("{quoted}|{}", fancy_regex::escape(part))
    } else {
        quoted
    }
}

/// The pattern of a qualifier `path.` in any case, not after a name's
/// character, a backtick, `$` or a dot (`_path_pattern`).
fn path_pattern<S: AsRef<str>>(path: &[S], spaces: &Spaces) -> Read<Pattern> {
    let chain = path
        .iter()
        .map(|part| format!("(?:{})", quoted_variants(part.as_ref(), spaces)))
        .collect::<Vec<_>>()
        .join(r"\s*\.\s*");
    Pattern::build(&format!(r"(?i)(?<![A-Za-z0-9_`$.]){chain}\s*\."), spaces)
}

/// `expr` with each string literal emptied (`_masked_strings`).
fn masked_strings(expr: &str, spaces: &Spaces) -> Read<String> {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for span in string_literals(expr, spaces)? {
        out.push_str(&expr[last..span.start]);
        out.push_str("''");
        last = span.end;
    }
    out.push_str(&expr[last..]);
    Ok(out)
}

/// Whether `expr` reads a column of the table `path` names, outside its
/// literals (`references_alias`).
pub fn references_alias<S: AsRef<str>>(
    expr: &str,
    path: &[S],
    spaces: &Spaces,
) -> Result<bool, Unreadable> {
    path_pattern(path, spaces)?.is_match(&masked_strings(expr, spaces)?)
}

/// `expr` with every qualifier `path.` taken out, its literals as they are,
/// trimmed (`strip_alias_qualifier`).
pub fn strip_alias<S: AsRef<str>>(
    expr: &str,
    path: &[S],
    spaces: &Spaces,
) -> Result<String, Unreadable> {
    let pattern = path_pattern(path, spaces)?;
    let mut out = String::with_capacity(expr.len());
    let mut last = 0;
    for span in string_literals(expr, spaces)? {
        out.push_str(&pattern.replace_all(&expr[last..span.start], "")?);
        out.push_str(&expr[span.clone()]);
        last = span.end;
    }
    out.push_str(&pattern.replace_all(&expr[last..], "")?);
    Ok(spaces.strip(&out).to_owned())
}

/// The column `expr` truncates or casts to a date or time, as
/// `order_date` in `DATE_TRUNC('MONTH', order_date)`; `None` for anything
/// else, a bare column included (`derived_column`).
pub fn derived_column(expr: &str, spaces: &Spaces) -> Option<String> {
    let column = column_under(expr, spaces)?;
    let bare = spaces.strip(expr).trim_matches(['(', ')', ' ']);
    (!matches(&WHOLE_IDENT.get(spaces), bare)).then_some(column)
}

fn column_under(expr: &str, spaces: &Spaces) -> Option<String> {
    let expr = spaces.strip(expr);
    if matches(&WHOLE_IDENT.get(spaces), expr) {
        return Some(unquote(expr, spaces));
    }
    for derivation in DERIVATIONS.get(spaces).iter() {
        let Ok(Some(groups)) = derivation.groups(expr) else {
            continue;
        };
        let inner = &expr[groups.name("inner").unwrap()];
        // `.+` reaches past a bracket that closes early, as in
        // `date(a) - date(b)`, so what it holds must balance.
        let mut depth: i64 = 0;
        for c in masked_strings(inner, spaces).ok()?.chars() {
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
            && let Some(column) = column_under(inner, spaces)
        {
            return Some(column);
        }
    }
    None
}

/// The column an expression is, if it is one column and nothing else
/// (`_renamed_column`).
pub fn renamed_column(expr: &str, spaces: &Spaces) -> Option<String> {
    let expr = spaces.strip(expr);
    let groups = RENAMED.get(spaces).groups(expr).ok()??;
    match (groups.get(1), groups.get(2)) {
        (Some(quoted), _) => Some(expr[quoted].replace("``", "`")),
        (None, Some(bare)) => Some(expr[bare].to_owned()),
        _ => None,
    }
}

/// The literal a parameter's type takes, as `_default_text` tells types
/// apart (`DB/metric_view.py:513-566`): by the type in lower case, its
/// white space single.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeClass {
    Boolean,
    /// `int`, `bigint` and the like.
    Integer,
    /// `double`, `float`, `real`, and `decimal` with or without its
    /// precision.
    Decimal,
    /// `string`, `char(n)` and `varchar(n)`.
    String,
    Date,
    /// `timestamp`, `timestamp_ntz` and `timestamp_ltz`.
    Timestamp,
    /// Any other type, which has no literal here.
    NoLiteral,
}

/// The class of `data_type`.
pub fn type_class(data_type: &str, spaces: &Spaces) -> TypeClass {
    let folded = spaces
        .split(&data_type.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    if folded == "boolean" {
        TypeClass::Boolean
    } else if INTEGER_TYPES.contains(&folded.as_str()) {
        TypeClass::Integer
    } else if FLOAT_TYPES.contains(&folded.as_str()) || matches(&DECIMAL_TYPE.get(spaces), &folded)
    {
        TypeClass::Decimal
    } else if matches(&STRING_TYPE.get(spaces), &folded) {
        TypeClass::String
    } else if folded == "date" {
        TypeClass::Date
    } else if TIMESTAMP_TYPES.contains(&folded.as_str()) {
        TypeClass::Timestamp
    } else {
        TypeClass::NoLiteral
    }
}

/// Why a default gives no literal of its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultRefusal {
    /// The type has no literal here.
    NoLiteral,
    /// The default is no value of the type.
    NotOfType,
    /// An integer of more digits than Python reads, so many.
    TooManyDigits(usize),
    /// A text `int` cannot read though the old client's pattern took it.
    NotAnInteger,
    /// A value whose text would be longer than the core writes.
    TooLong,
}

/// What a parameter of the view is given, as the core hands it over: its
/// name, its type as the definition writes it if text, and its default
/// unless null, as the core holds it (`D`).
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterInput<D> {
    pub name: String,
    pub data_type: Option<String>,
    pub default: Option<D>,
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
/// Databricks describes, `described`, wins over the definition's. The
/// core converts each default to the text its type casts from
/// (`convert`, `_default_text`).
pub fn read_parameters<D>(
    inputs: &[ParameterInput<D>],
    described: &[(String, String)],
    convert: impl Fn(&str, &D) -> Result<String, DefaultRefusal>,
    spaces: &Spaces,
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
            (Some(default), Some(data_type)) => match convert(&data_type, default) {
                Ok(text) => {
                    let sql_type = spaces
                        .split(&data_type)
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
    spaces: &Spaces,
) -> Result<(String, Vec<String>), Unreadable> {
    let found: Vec<(std::ops::Range<usize>, &Parameter)> = name_references(expr, spaces)?
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
    spaces: &Spaces,
) -> Result<(String, Option<String>), Unreadable> {
    let mut out = expr.to_owned();
    let mut refused = None;
    for reference in name_references(expr, spaces)?.into_iter().rev() {
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
            .get(spaces)
            .find(&out[start..])?
            .map_or(start, |m| start + m.end);
        out.replace_range(start..first, &format!("({field})"));
    }
    Ok((out, refused))
}

/// Whether `expr` reads every column of a table through `*`, as
/// `struct(*)` does; `COUNT(*)` counts rows instead (`_star`).
pub fn reads_star(expr: &str, spaces: &Spaces) -> Result<bool, Unreadable> {
    Ok(name_references(expr, spaces)?
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
    /// A pattern the check reads it by failed on it.
    Unreadable(Unreadable),
}

/// The filter, trimmed, if it can be applied to the source table alone
/// (`_check_filter`): text, safe, and reading no joined table. `filter` is
/// `None` for a filter that is no text; `candidates` each join path or
/// alias with the alias it resolves to, longest first.
pub fn check_filter(
    filter: Option<&str>,
    candidates: &[(Vec<String>, String)],
    spaces: &Spaces,
) -> Result<String, FilterRefusal> {
    let Some(filter) = filter.filter(|f| !spaces.strip(f).is_empty()) else {
        return Err(FilterRefusal::NotACondition);
    };
    if let Some(reason) = unsafe_reason(filter, Some(ExprDialect::Databricks), spaces) {
        return Err(FilterRefusal::Unsafe(reason));
    }
    let masked = masked_strings(filter, spaces).map_err(FilterRefusal::Unreadable)?;
    for (path, alias) in candidates {
        if alias != SOURCE_ALIAS
            && references_alias(&masked, path, spaces).map_err(FilterRefusal::Unreadable)?
        {
            return Err(FilterRefusal::JoinedTable(alias.clone()));
        }
    }
    Ok(spaces.strip(filter).to_owned())
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
    spaces: &Spaces,
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
        let (expr, names) = match super::strip_comments(expr, spaces).and_then(|expr| {
            let names = name_references(&expr, spaces)?;
            Ok((expr, names))
        }) {
            Ok(read) => read,
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
        let aggregation = aggregation(measure.window, &expr, &inputs, &dimension, spaces);
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
    spaces: &Spaces,
) -> Option<(String, Option<String>)> {
    if window {
        return None;
    }
    let masked = masked_strings(expr, spaces).ok()?;
    let groups = SIMPLE_MEASURE.get(spaces).groups(&masked).ok()??;
    let function = masked[groups.get(1)?].to_uppercase();
    let distinct = groups.get(2).is_some();
    let argument = &masked[groups.get(3)?];
    if argument == "*" || argument == "1" {
        return (function == "COUNT" && !distinct).then(|| ("COUNT".to_owned(), None));
    }
    let column = match dimension(&unquote(argument, spaces).to_lowercase()) {
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
