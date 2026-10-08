// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks metric view's YAML, read into the parts a graph can hold:
//! `rfm/backend/databricks/metric_view.py::parse_metric_view`, with Python's
//! typing kept where the code leans on it (iterating whatever a key holds,
//! `get(True)` for a bare `on`, `repr` in messages).

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::py;
use crate::pyval::PyVal;
use crate::yaml;

pub const SOURCE_ALIAS: &str = "source";

const BARE_IDENT: &str = "[A-Za-z_][A-Za-z0-9_]*";
const IDENT: &str = "`(?:[^`]|``)+`|[A-Za-z_][A-Za-z0-9_]*";

/// An exception the parser raises, by class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raised {
    pub class: String,
    pub message: String,
}

fn raised(class: &str, message: impl Into<String>) -> Raised {
    Raised {
        class: class.to_string(),
        message: message.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinKeyRef {
    pub qualifier: Option<String>,
    pub column: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricViewJoin {
    pub alias: String,
    pub table: Vec<String>,
    pub parent_alias: String,
    pub path: Vec<String>,
    pub condition: (JoinKeyRef, JoinKeyRef),
    pub cardinality: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricViewColumn {
    pub name: String,
    pub alias: String,
    pub expr: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricViewSpec {
    pub source: Vec<String>,
    pub joins: Vec<MetricViewJoin>,
    pub columns: Vec<MetricViewColumn>,
    pub messages: Vec<String>,
    pub filter: Option<MetricViewFilter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricViewFilter {
    sql: String,
}

impl MetricViewFilter {
    pub fn checked(&self, columns: &[String]) -> Result<String, Raised> {
        if let Some(reason) =
            sdm_dialects::view_expr::untrusted_call_reason(&self.sql, "databricks", columns)
                .map_err(|error| raised("KeyError", error.to_string()))?
        {
            return Err(refuse_filter(&PyVal::Str(self.sql.clone()), &reason));
        }
        Ok(self.sql.clone())
    }

    pub fn trusted(&self) -> String {
        self.sql.clone()
    }
}

impl MetricViewSpec {
    /// `dataclasses.asdict(spec)`, tuples as lists.
    pub fn to_json(&self) -> Value {
        let key = |k: &JoinKeyRef| json!({"qualifier": k.qualifier, "column": k.column});
        json!({
            "source": self.source,
            "joins": self.joins.iter().map(|j| json!({
                "alias": j.alias,
                "table": j.table,
                "parent_alias": j.parent_alias,
                "path": j.path,
                "condition": [key(&j.condition.0), key(&j.condition.1)],
                "cardinality": j.cardinality,
            })).collect::<Vec<_>>(),
            "columns": self.columns.iter().map(|c| json!({
                "name": c.name, "alias": c.alias, "expr": c.expr,
            })).collect::<Vec<_>>(),
            "messages": self.messages,
            "filter": self.filter.as_ref().map(|filter| json!({"_sql": filter.sql})),
        })
    }
}

/// `unquote_ident`.
pub fn unquote_ident(name: &str) -> String {
    let name = py::strip(name);
    if name.chars().count() >= 2 && name.starts_with('`') && name.ends_with('`') {
        let inner: String = name
            .chars()
            .skip(1)
            .take(name.chars().count() - 2)
            .collect();
        return inner.replace("``", "`");
    }
    name.to_string()
}

fn table_reference_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"\A\s*({IDENT})(?:\.({IDENT}))?(?:\.({IDENT}))?\s*\n?\z"
        ))
        .expect("pattern compiles")
    })
}

/// `parse_table_reference`.
pub fn parse_table_reference(name: Option<&PyVal>) -> Option<Vec<String>> {
    let Some(PyVal::Str(name)) = name else {
        return None;
    };
    let found = table_reference_re().captures(name)?;
    Some(
        (1..=3)
            .filter_map(|i| found.get(i))
            .map(|m| unquote_ident(m.as_str()))
            .collect(),
    )
}

fn ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(IDENT).expect("pattern compiles"))
}

/// `parse_join_condition`.
pub fn parse_join_condition(condition: Option<&PyVal>) -> Option<(JoinKeyRef, JoinKeyRef)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        let chain = format!(r"(?:{IDENT})(?:\s*\.\s*(?:{IDENT}))*");
        Regex::new(&format!(r"\A\s*({chain})\s*=\s*({chain})\s*\n?\z")).expect("pattern compiles")
    });
    let Some(PyVal::Str(text)) = condition else {
        return None;
    };
    let found = re.captures(text)?;
    let make = |chain: &str| -> JoinKeyRef {
        let parts: Vec<String> = ident_re()
            .find_iter(chain)
            .map(|m| unquote_ident(m.as_str()))
            .collect();
        let qualifier = parts[..parts.len() - 1].join(".");
        JoinKeyRef {
            qualifier: if qualifier.is_empty() {
                None
            } else {
                Some(qualifier)
            },
            column: parts.last().cloned().unwrap_or_default(),
        }
    };
    Some((make(found.get(1)?.as_str()), make(found.get(2)?.as_str())))
}

fn is_bare(name: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"\A(?:{BARE_IDENT})\z")).expect("pattern compiles"))
        .is_match(name)
}

/// `_quoted_variants`: each spelling of a name, in the order the pattern
/// tries them.
fn variants(name: &str) -> Vec<String> {
    let quoted = regex::escape(&format!("`{}`", name.replace('`', "``")));
    if is_bare(name) {
        vec![quoted, regex::escape(name)]
    } else {
        vec![quoted]
    }
}

fn lookbehind_ok(chars: &[char], at: usize) -> bool {
    at == 0
        || !(chars[at - 1].is_ascii_alphanumeric()
            || matches!(chars[at - 1], '_' | '`' | '$' | '.'))
}

fn byte_offsets(text: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    offsets.push(text.len());
    offsets
}

/// A pattern with the lookbehind `(?<![A-Za-z0-9_`$.])` Rust's regular
/// expressions lack: matched at each start where the character before
/// allows it.
struct Behind {
    body: Regex,
    /// For `_field_pattern`: each alternative, with the lookahead
    /// `(?![A-Za-z0-9_`])` checked after it.
    alternatives: Option<Vec<Regex>>,
}

impl Behind {
    fn path(path: &[String]) -> Behind {
        let chain: Vec<String> = path
            .iter()
            .map(|part| format!("(?:{})", variants(part).join("|")))
            .collect();
        let body = format!(r"(?i)\A{}\s*\.", chain.join(r"\s*\.\s*"));
        Behind {
            body: Regex::new(&body).expect("pattern compiles"),
            alternatives: None,
        }
    }

    fn field(name: &str) -> Behind {
        let alternatives: Vec<Regex> = variants(name)
            .iter()
            .map(|v| Regex::new(&format!(r"(?i)\A(?:{v})")).expect("pattern compiles"))
            .collect();
        Behind {
            body: Regex::new("^$").expect("pattern compiles"),
            alternatives: Some(alternatives),
        }
    }

    /// The match starting at char `at`, as a byte range of `text`.
    fn match_at(
        &self,
        text: &str,
        chars: &[char],
        offsets: &[usize],
        at: usize,
    ) -> Option<(usize, usize)> {
        if !lookbehind_ok(chars, at) {
            return None;
        }
        let start = offsets[at];
        let rest = &text[start..];
        match &self.alternatives {
            None => self.body.find(rest).map(|m| (start, start + m.end())),
            Some(alts) => {
                for alt in alts {
                    if let Some(m) = alt.find(rest) {
                        let end = start + m.end();
                        let next = text[end..].chars().next();
                        if next.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '`'))
                        {
                            return Some((start, end));
                        }
                    }
                }
                None
            }
        }
    }

    fn search(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let offsets = byte_offsets(text);
        (0..=chars.len()).any(|at| self.match_at(text, &chars, &offsets, at).is_some())
    }

    fn sub(&self, text: &str, replacement: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let offsets = byte_offsets(text);
        let mut out = String::new();
        let mut at = 0;
        while at <= chars.len() {
            if let Some((start, end)) = self.match_at(text, &chars, &offsets, at) {
                out.push_str(replacement);
                // Continue after the match, by characters.
                let next = offsets
                    .iter()
                    .position(|&o| o == end)
                    .unwrap_or(chars.len());
                if end == start {
                    if at < chars.len() {
                        out.push(chars[at]);
                    }
                    at += 1;
                } else {
                    at = next;
                }
                continue;
            }
            if at < chars.len() {
                out.push(chars[at]);
            }
            at += 1;
        }
        out
    }
}

/// `references_alias`.
pub fn references_alias(expr: &str, path: &[String]) -> bool {
    Behind::path(path).search(&string_literal_re().replace_all(expr, "''"))
}

/// `strip_alias_qualifier`.
pub fn strip_alias_qualifier(expr: &str, path: &[String]) -> String {
    py::strip(&sub_outside_literals(&Behind::path(path), expr, "")).to_string()
}

fn sub_outside_literals(pattern: &Behind, expr: &str, replacement: &str) -> String {
    let mut out = String::new();
    let mut last = 0;
    for literal in string_literal_re().find_iter(expr) {
        out.push_str(&pattern.sub(&expr[last..literal.start()], replacement));
        out.push_str(literal.as_str());
        last = literal.end();
    }
    out.push_str(&pattern.sub(&expr[last..], replacement));
    out
}

fn string_literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"'(?:''|\\[\s\S]|[^'\\])*'|"(?:""|\\[\s\S]|[^"\\])*""#)
            .expect("pattern compiles")
    })
}

/// `dict.get(key)` with Python's key equality.
pub fn get<'a>(dict: &'a PyVal, key: &PyVal) -> Option<&'a PyVal> {
    let PyVal::Dict(items) = dict else {
        return None;
    };
    let id = key.eq_key();
    items.iter().find(|(k, _)| k.eq_key() == id).map(|(_, v)| v)
}

fn get_str<'a>(dict: &'a PyVal, key: &str) -> Option<&'a PyVal> {
    get(dict, &PyVal::Str(key.to_string()))
}

/// Python truthiness.
pub fn truthy(value: &PyVal) -> bool {
    match value {
        PyVal::None | PyVal::Na | PyVal::NaT => false,
        PyVal::Bool(b) => *b,
        PyVal::Int(n) => *n != 0,
        PyVal::BigInt(_) => true,
        PyVal::Float(x) => *x != 0.0,
        PyVal::Str(s) => !s.is_empty(),
        PyVal::List(v) | PyVal::Tuple(v) => !v.is_empty(),
        PyVal::Dict(v) => !v.is_empty(),
        PyVal::Bytes(b) => !b.is_empty(),
        _ => true,
    }
}

/// `type(value).__name__`, as an error message names it.
pub fn type_name(value: &PyVal) -> &'static str {
    match value {
        PyVal::None => "NoneType",
        PyVal::Bool(_) => "bool",
        PyVal::Int(_) | PyVal::BigInt(_) => "int",
        PyVal::Float(_) => "float",
        PyVal::Str(_) => "str",
        PyVal::Date(..) => "datetime.date",
        PyVal::DateTime(..) => "datetime.datetime",
        PyVal::Time { .. } => "datetime.time",
        PyVal::List(_) => "list",
        PyVal::Tuple(_) => "tuple",
        PyVal::Dict(_) => "dict",
        PyVal::Bytes(_) => "bytes",
        _ => "object",
    }
}

/// `for item in (value or [])`.
pub fn iterate(value: Option<&PyVal>) -> Result<Vec<PyVal>, Raised> {
    let Some(value) = value.filter(|v| truthy(v)) else {
        return Ok(Vec::new());
    };
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => Ok(items.clone()),
        PyVal::Dict(items) => Ok(items.iter().map(|(k, _)| k.clone()).collect()),
        PyVal::Str(s) => Ok(s.chars().map(|c| PyVal::Str(c.to_string())).collect()),
        PyVal::Bytes(b) => Ok(b.iter().map(|&x| PyVal::Int(i128::from(x))).collect()),
        other => Err(raised(
            "TypeError",
            format!("'{}' object is not iterable", type_name(other)),
        )),
    }
}

fn is_hashable(value: &PyVal) -> bool {
    !matches!(value, PyVal::List(_) | PyVal::Dict(_))
}

/// An ordered `dict` of strings, as `defined` is: assigning an existing
/// name keeps its place.
#[derive(Default)]
struct Defined(Vec<(String, String)>);

impl Defined {
    fn set(&mut self, name: &str, value: String) {
        if let Some(slot) = self.0.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value;
        } else {
            self.0.push((name.to_string(), value));
        }
    }
}

struct Joins<'a> {
    joins: Vec<MetricViewJoin>,
    aliases: Vec<String>,
    kept: Vec<String>,
    messages: &'a mut Vec<String>,
}

impl Joins<'_> {
    fn add_alias(&mut self, alias: &str) {
        if !self.aliases.iter().any(|a| a == alias) {
            self.aliases.push(alias.to_string());
        }
    }

    fn collect_aliases(&mut self, cfgs: Option<&PyVal>) -> Result<(), Raised> {
        for cfg in iterate(cfgs)? {
            if let PyVal::Dict(_) = &cfg {
                if let Some(PyVal::Str(name)) = get_str(&cfg, "name") {
                    let name = name.clone();
                    self.add_alias(&name);
                }
                self.collect_aliases(get_str(&cfg, "joins"))?;
            }
        }
        Ok(())
    }

    fn add(&mut self, cfgs: Option<&PyVal>, parent_path: &[String]) -> Result<(), Raised> {
        let parent_alias = parent_path
            .last()
            .cloned()
            .unwrap_or_else(|| SOURCE_ALIAS.to_string());
        for cfg in iterate(cfgs)? {
            if !matches!(cfg, PyVal::Dict(_)) {
                self.messages.push(format!(
                    "Failed to add join {} since it is not a mapping",
                    cfg.repr()
                ));
                continue;
            }
            let Some(PyVal::Str(alias)) = get_str(&cfg, "name") else {
                self.messages.push(format!(
                    "Failed to add join {} since it misses a name",
                    cfg.repr()
                ));
                continue;
            };
            let alias = alias.clone();
            let skip = |me: &mut Self, msg: String| -> Result<(), Raised> {
                me.messages
                    .push(format!("Failed to add join '{alias}' since {msg}"));
                me.add_alias(&alias);
                me.collect_aliases(get_str(&cfg, "joins"))
            };
            if alias == SOURCE_ALIAS || self.kept.contains(&alias) {
                skip(self, format!("the name '{alias}' is already in use"))?;
                continue;
            }
            let Some(table) = parse_table_reference(get_str(&cfg, "source")) else {
                skip(
                    self,
                    "only plain table names are supported as join sources".to_string(),
                )?;
                continue;
            };
            let default = PyVal::Str("many_to_one".to_string());
            let cardinality = match get(&cfg, &PyVal::Str("cardinality".into())) {
                Some(value) => value.clone(),
                None => default,
            };
            if !is_hashable(&cardinality) {
                return Err(raised(
                    "TypeError",
                    format!("unhashable type: '{}'", type_name(&cardinality)),
                ));
            }
            let cardinality_text = match &cardinality {
                PyVal::Str(s) if s == "many_to_one" || s == "one_to_many" => s.clone(),
                other => {
                    skip(
                        self,
                        format!("of its unsupported cardinality '{}'", other.str_()),
                    )?;
                    continue;
                }
            };
            let mut using = get_str(&cfg, "using").cloned();
            if let Some(PyVal::Str(s)) = &using {
                using = Some(PyVal::List(vec![PyVal::Str(s.clone())]));
            }
            let condition = match using {
                Some(PyVal::None) | None => {
                    let on = match get_str(&cfg, "on") {
                        Some(v) => Some(v.clone()),
                        None => get(&cfg, &PyVal::Bool(true)).cloned(),
                    };
                    match parse_join_condition(on.as_ref()) {
                        Some(parsed) => parsed,
                        None => {
                            skip(
                                self,
                                "only single-column equality join conditions are supported"
                                    .to_string(),
                            )?;
                            continue;
                        }
                    }
                }
                Some(PyVal::List(items)) if items.len() == 1 => {
                    let column = unquote_ident(&items[0].str_());
                    (
                        JoinKeyRef {
                            qualifier: Some(parent_alias.clone()),
                            column: column.clone(),
                        },
                        JoinKeyRef {
                            qualifier: Some(alias.clone()),
                            column,
                        },
                    )
                }
                Some(_) => {
                    skip(
                        self,
                        "composite key references are not yet supported".to_string(),
                    )?;
                    continue;
                }
            };
            let mut path = parent_path.to_vec();
            path.push(alias.clone());
            self.joins.push(MetricViewJoin {
                alias: alias.clone(),
                table,
                parent_alias: parent_alias.clone(),
                path: path.clone(),
                condition,
                cardinality: cardinality_text,
            });
            self.add_alias(&alias);
            if !self.kept.contains(&alias) {
                self.kept.push(alias.clone());
            }
            self.add(get_str(&cfg, "joins"), &path)?;
        }
        Ok(())
    }
}

fn check_filter(
    row_filter: Option<&PyVal>,
    candidates: &[(Vec<String>, String)],
) -> Result<Option<MetricViewFilter>, Raised> {
    let Some(row_filter) = row_filter.filter(|v| !matches!(v, PyVal::None)) else {
        return Ok(None);
    };
    let mut reason: Option<String> = match row_filter {
        PyVal::Str(text) if !py::strip(text).is_empty() => {
            sdm_dialects::view_expr::unsafe_expr_reason(text, Some("databricks"))
        }
        _ => Some("it is not a SQL condition".to_string()),
    };
    if reason.is_none()
        && let PyVal::Str(text) = row_filter
    {
        let masked = string_literal_re().replace_all(text, "''");
        for (path, alias) in candidates {
            if alias != SOURCE_ALIAS && references_alias(&masked, path) {
                reason = Some(format!("it references the joined table '{alias}'"));
                break;
            }
        }
    }
    if let Some(reason) = reason {
        return Err(refuse_filter(row_filter, &reason));
    }
    match row_filter {
        PyVal::Str(text) => Ok(Some(MetricViewFilter {
            sql: py::strip(text).to_string(),
        })),
        _ => Ok(None),
    }
}

fn refuse_filter(row_filter: &PyVal, reason: &str) -> Raised {
    raised(
        "UnsupportedFilterError",
        format!(
            "Cannot apply the metric view filter {} since {reason}. The graph would otherwise read rows the view leaves out. Create a regular view with this WHERE clause and point the metric view's source at it",
            row_filter.repr()
        ),
    )
}

/// `parse_metric_view(definition, trust_expressions=trust)`.
pub fn parse_metric_view(definition: &str, _trust: bool) -> Result<MetricViewSpec, Raised> {
    let cfg = yaml::safe_load(definition).map_err(|e| raised(e.class, e.message))?;
    if !matches!(cfg, PyVal::Dict(_)) {
        return Err(raised(
            "ValueError",
            "The metric view definition is not a YAML mapping",
        ));
    }
    let source = get_str(&cfg, "source").filter(|v| !matches!(v, PyVal::None));
    let Some(source) = source else {
        return Err(raised(
            "ValueError",
            "The metric view definition misses a 'source'",
        ));
    };
    let Some(source_ref) = parse_table_reference(Some(source)) else {
        return Err(raised(
            "ValueError",
            format!(
                "Unsupported metric view source {}. Only plain table names are supported as the source of a metric view when converting it into a graph",
                source.repr()
            ),
        ));
    };
    let mut messages: Vec<String> = Vec::new();
    for key in ["parameters", "materialization"] {
        if get_str(&cfg, key).is_some() {
            messages.push(format!(
                "Ignored the '{key}' section since it cannot be represented in a graph"
            ));
        }
    }
    let mut builder = Joins {
        joins: Vec::new(),
        aliases: Vec::new(),
        kept: Vec::new(),
        messages: &mut messages,
    };
    builder.add(get_str(&cfg, "joins"), &[])?;
    let Joins {
        joins,
        aliases,
        kept,
        ..
    } = builder;
    let mut candidates: Vec<(Vec<String>, String)> = Vec::new();
    for join in &joins {
        candidates.push((join.path.clone(), join.alias.clone()));
        if join.path.len() > 1 {
            candidates.push((vec![join.alias.clone()], join.alias.clone()));
        }
    }
    for alias in aliases.iter().filter(|a| !kept.contains(a)) {
        candidates.push((vec![alias.clone()], alias.clone()));
    }
    candidates.push((vec![SOURCE_ALIAS.to_string()], SOURCE_ALIAS.to_string()));
    candidates.sort_by_key(|(path, _)| std::cmp::Reverse(path.len()));

    let row_filter = check_filter(get_str(&cfg, "filter"), &candidates)?;

    let mut columns = Vec::new();
    let mut defined = Defined::default();
    let mut dims = iterate(get_str(&cfg, "dimensions"))?;
    dims.extend(iterate(get_str(&cfg, "fields"))?);
    for dim in dims {
        if !matches!(dim, PyVal::Dict(_)) {
            messages.push(format!(
                "Failed to add dimension {} since it is not a mapping",
                dim.repr()
            ));
            continue;
        }
        let (Some(PyVal::Str(name)), Some(PyVal::Str(expr))) =
            (get_str(&dim, "name"), get_str(&dim, "expr"))
        else {
            messages.push(format!(
                "Failed to add dimension {} since it misses a name or expression",
                dim.repr()
            ));
            continue;
        };
        let name = name.clone();
        let mut expr = expr.clone();
        for (field_name, field_expr) in &defined.0 {
            expr = sub_outside_literals(
                &Behind::field(field_name),
                &expr,
                &format!("({field_expr})"),
            );
        }
        defined.set(&name, expr.clone());
        let mut refs: Vec<String> = Vec::new();
        for (path, path_alias) in &candidates {
            if references_alias(&expr, path) {
                if !refs.contains(path_alias) {
                    refs.push(path_alias.clone());
                }
                expr = strip_alias_qualifier(&expr, path);
            }
        }
        if refs.len() > 1 {
            messages.push(format!(
                "Failed to add dimension '{name}' since its expression references multiple tables"
            ));
            continue;
        }
        let alias = refs.pop().unwrap_or_else(|| SOURCE_ALIAS.to_string());
        if alias != SOURCE_ALIAS && !kept.contains(&alias) {
            messages.push(format!(
                "Failed to add dimension '{name}' since it references the skipped join '{alias}'"
            ));
            continue;
        }
        columns.push(MetricViewColumn { name, alias, expr });
    }
    Ok(MetricViewSpec {
        source: source_ref,
        joins,
        columns,
        messages,
        filter: row_filter,
    })
}

/// `metric_view_parse(yaml, trust)`: `{"spec"}` or `{"error"}`, as JSON.
pub fn parse_json(definition: &str, trust: bool) -> String {
    match parse_metric_view(definition, trust) {
        Ok(spec) => json!({ "spec": spec.to_json() }).to_string(),
        Err(e) => json!({ "error": { "code": e.class, "message": e.message } }).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_checks_source_scope_before_handing_out_sql() {
        let spec = parse_metric_view("source: orders\nfilter: status IN ('VALID')", false).unwrap();
        let filter = spec.filter.unwrap();
        assert!(filter.checked(&[]).is_err());
        assert_eq!(
            filter.checked(&["status".into()]).unwrap(),
            "status IN ('VALID')"
        );
        let spec =
            parse_metric_view("source: orders\nfilter: my_udf(status) = 'VALID'", false).unwrap();
        let filter = spec.filter.unwrap();
        assert!(filter.checked(&["status".into()]).is_err());
        assert_eq!(filter.trusted(), "my_udf(status) = 'VALID'");
        assert!(parse_metric_view("source: orders\nfilter: (SELECT 1)", true).is_err());
    }

    #[test]
    fn literals_survive_alias_and_dimension_substitution() {
        let definition = r#"
source: cat.sch.orders
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: gold
    expr: customers.segment = 'gold'
  - name: tag
    expr: CONCAT('source.', source.channel, "customers.x", 'gold')
  - name: label
    expr: CASE WHEN gold THEN 'gold' END
"#;
        let spec = parse_metric_view(definition, false).unwrap();
        assert!(spec.messages.is_empty());
        assert_eq!(spec.columns[0].expr, "segment = 'gold'");
        assert_eq!(spec.columns[1].alias, "source");
        assert_eq!(
            spec.columns[1].expr,
            "CONCAT('source.', channel, \"customers.x\", 'gold')"
        );
        assert_eq!(spec.columns[2].alias, "customers");
        assert_eq!(
            spec.columns[2].expr,
            "CASE WHEN (segment = 'gold') THEN 'gold' END"
        );
        for literal in [
            "'source.a'",
            "'source.\\'a'",
            "'source.''a'",
            "\"source.a\"",
        ] {
            assert!(!references_alias(literal, &["source".into()]));
            assert_eq!(strip_alias_qualifier(literal, &["source".into()]), literal);
        }
    }
}
