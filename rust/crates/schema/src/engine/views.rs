// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Graph.from_snowflake_semantic_view` and
//! `Graph.from_databricks_metric_view`, as `sdm_client`'s
//! `semantic_model_graph` calls them: a semantic model's tables, keys and
//! links, and the conversion warning naming what it dropped.

use std::sync::OnceLock;

use regex::Regex;
use sdm_pql::types::Stype;

use super::{
    Backend, ColumnSpec, Ctx, EColumn, ER, EngineTable, Graph, PkArg, Place, PyExc, TableSpec,
    databricks_dtype, must, quote, rows, snow_dtype, text_of,
};
use crate::metric_view::{self, SOURCE_ALIAS, get, type_name};
use crate::pyval::PyVal;
use crate::source::SourceInfo;

/// The warning class a view conversion warns with.
pub const VIEW_CONVERSION_WARNING: &str = crate::diagnostics::VIEW_CONVERSION_WARNING;

fn raise<T>(class: &str, message: impl Into<String>) -> ER<T> {
    Err(PyExc::new(class, message))
}

// --- Python's reading of a YAML document -------------------------------------------

/// `value[key]`.
fn subscript(value: &PyVal, key: &str) -> ER<PyVal> {
    match value {
        PyVal::Dict(_) => get(value, &PyVal::Str(key.to_string()))
            .cloned()
            .ok_or_else(|| PyExc::key_error(key)),
        PyVal::List(_) => raise(
            "TypeError",
            "list indices must be integers or slices, not str",
        ),
        PyVal::Tuple(_) => raise(
            "TypeError",
            "tuple indices must be integers or slices, not str",
        ),
        PyVal::Str(_) => raise("TypeError", "string indices must be integers, not 'str'"),
        other => raise(
            "TypeError",
            format!("'{}' object is not subscriptable", type_name(other)),
        ),
    }
}

/// `value[index]` of a list.
fn item(value: &PyVal, index: usize) -> ER<PyVal> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => items
            .get(index)
            .cloned()
            .ok_or_else(|| PyExc::new("IndexError", "list index out of range")),
        PyVal::Dict(_) => get(value, &PyVal::Int(index as i128))
            .cloned()
            .ok_or_else(|| PyExc::new("KeyError", index.to_string())),
        PyVal::Str(s) => s
            .chars()
            .nth(index)
            .map(|c| PyVal::Str(c.to_string()))
            .ok_or_else(|| PyExc::new("IndexError", "string index out of range")),
        other => raise(
            "TypeError",
            format!("'{}' object is not subscriptable", type_name(other)),
        ),
    }
}

/// `value.get(key, default)`.
fn dict_get(value: &PyVal, key: &str) -> ER<Option<PyVal>> {
    match value {
        PyVal::Dict(_) => Ok(get(value, &PyVal::Str(key.to_string())).cloned()),
        other => raise(
            "AttributeError",
            format!("'{}' object has no attribute 'get'", type_name(other)),
        ),
    }
}

/// `key in value`.
fn contains(value: &PyVal, key: &str) -> ER<bool> {
    match value {
        PyVal::Dict(_) => Ok(get(value, &PyVal::Str(key.to_string())).is_some()),
        PyVal::List(items) | PyVal::Tuple(items) => {
            Ok(items.iter().any(|i| matches!(i, PyVal::Str(s) if s == key)))
        }
        PyVal::Str(s) => Ok(s.contains(key)),
        other => raise(
            "TypeError",
            format!("argument of type '{}' is not iterable", type_name(other)),
        ),
    }
}

/// `len(value)`.
fn len(value: &PyVal) -> ER<usize> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => Ok(items.len()),
        PyVal::Dict(items) => Ok(items.len()),
        PyVal::Str(s) => Ok(s.chars().count()),
        PyVal::Bytes(b) => Ok(b.len()),
        other => raise(
            "TypeError",
            format!("object of type '{}' has no len()", type_name(other)),
        ),
    }
}

/// `for x in value`.
fn iterate(value: &PyVal) -> ER<Vec<PyVal>> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => Ok(items.clone()),
        PyVal::Dict(items) => Ok(items.iter().map(|(k, _)| k.clone()).collect()),
        PyVal::Str(s) => Ok(s.chars().map(|c| PyVal::Str(c.to_string())).collect()),
        other => raise(
            "TypeError",
            format!("'{}' object is not iterable", type_name(other)),
        ),
    }
}

/// `_view_name(name)`: a quoted name spells what is inside its quotes.
fn view_name(name: &PyVal) -> PyVal {
    if let PyVal::Str(s) = name {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() >= 2 && chars[0] == '"' && chars[chars.len() - 1] == '"' {
            let inner: String = chars[1..chars.len() - 1].iter().collect();
            return PyVal::Str(inner.replace("\"\"", "\""));
        }
    }
    name.clone()
}

/// A name as the engine holds it, once it is text.
fn name_text(name: &PyVal) -> String {
    text_of(name)
}

// --- Expressions -----------------------------------------------------------------------

fn string_literal() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("'(?:''|[^'])*'").expect("pattern compiles"))
}

/// `_mask_literals(expr)`.
fn mask_literals(expr: &str) -> String {
    string_literal()
        .replace_all(expr, |m: &regex::Captures<'_>| {
            " ".repeat(m[0].chars().count())
        })
        .into_owned()
}

/// `_qualifier_pattern(table_name)`: `<name>.`, the lookbehind
/// `(?<![A-Za-z0-9_$."])` checked at each start.
struct Qualifier {
    body: Regex,
}

impl Qualifier {
    fn new(table: &str) -> Qualifier {
        let body = format!(
            r"(?i)\A(?:{}|{})\s*\.",
            regex::escape(&quote(table)),
            regex::escape(table)
        );
        Qualifier {
            body: Regex::new(&body).expect("pattern compiles"),
        }
    }

    fn allowed_after(c: Option<char>) -> bool {
        !c.is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.' | '"'))
    }

    /// Each match, as byte ranges, scanning as `re.sub` does.
    fn matches(&self, text: &str) -> Vec<(usize, usize)> {
        let mut found = Vec::new();
        let mut previous: Option<char> = None;
        let mut at = 0;
        while at <= text.len() {
            if Self::allowed_after(previous)
                && let Some(m) = self.body.find(&text[at..])
            {
                let end = at + m.end();
                found.push((at, end));
                previous = text[..end].chars().next_back();
                at = end;
                continue;
            }
            match text[at..].chars().next() {
                Some(c) => {
                    previous = Some(c);
                    at += c.len_utf8();
                }
                None => break,
            }
        }
        found
    }

    fn search(&self, text: &str) -> bool {
        !self.matches(text).is_empty()
    }

    fn strip(&self, text: &str) -> String {
        let mut out = String::new();
        let mut last = 0;
        for (start, end) in self.matches(text) {
            out.push_str(&text[last..start]);
            last = end;
        }
        out.push_str(&text[last..]);
        out
    }
}

/// `_sub_outside_literals(pattern, expr)`.
fn strip_outside_literals(pattern: &Qualifier, expr: &str) -> String {
    let mut out = String::new();
    let mut last = 0;
    for m in string_literal().find_iter(expr) {
        out.push_str(&pattern.strip(&expr[last..m.start()]));
        out.push_str(m.as_str());
        last = m.end();
    }
    out.push_str(&pattern.strip(&expr[last..]));
    out
}

/// `_view_expr_message(column, expr, view, dialect, trust)`.
fn view_expr_message(
    column: &str,
    expr: &str,
    view: &str,
    dialect: &str,
    trust: bool,
    source_columns: &[String],
) -> Option<String> {
    let mut hatch = "";
    let mut reason = sdm_dialects::view_expr::unsafe_expr_reason(expr, Some(dialect));
    if reason.is_none() && !trust {
        reason = sdm_dialects::view_expr::untrusted_call_reason(expr, dialect, source_columns)
            .ok()
            .flatten();
        hatch = ". Pass trust_expressions=True to keep it if you trust whoever authored the view";
    }
    let reason = reason?;
    Some(format!(
        "Failed to add {column} since {reason}, and expressions taken from the {view} are executed as SQL under your warehouse credentials{hatch}"
    ))
}

fn py_strip(s: &str) -> String {
    crate::py::strip(s).to_string()
}

fn warn_conversion(ctx: &mut Ctx<'_>, title: String, msgs: &[String]) {
    if msgs.is_empty() {
        return;
    }
    let lines: Vec<String> = msgs.iter().map(|m| format!("- {m}")).collect();
    ctx.warnings.0.push((
        VIEW_CONVERSION_WARNING.to_string(),
        format!("{title}{}", lines.join("\n")),
    ));
}

/// What the engine's errors are, as `except (ValueError, KeyError)` sees
/// them.
fn is_value_or_key_error(e: &PyExc) -> bool {
    e.is_value_error() || e.class == "KeyError"
}

async fn time_candidates_and_infer(
    ctx: &mut Ctx<'_>,
    table: &mut EngineTable,
    preferred: &[String],
) -> ER<()> {
    let is_time = |c: &EColumn| c.stype == Stype::Timestamp;
    let mut candidates: Vec<String> = preferred
        .iter()
        .filter(|n| table.column(n).is_some_and(is_time))
        .cloned()
        .collect();
    if candidates.is_empty() {
        candidates = table
            .columns
            .iter()
            .filter(|c| is_time(c))
            .map(|c| c.name.clone())
            .collect();
    }
    if candidates.is_empty() {
        return Ok(());
    }
    table.infer_time_column_among(ctx, &candidates).await?;
    Ok(())
}

/// `_reraise_as_graph_error(error, source)` for what is read before a
/// view's graph exists.
fn as_graph_error(e: PyExc, source: &str) -> PyExc {
    match e {
        e if !is_builtin_or_own(&e.class) => PyExc::new(
            "GraphConstructionError",
            format!(
                "Reading the {source} source failed while building the graph: {}: {}",
                e.class, e.message
            ),
        ),
        other => other,
    }
}

fn is_builtin_or_own(class: &str) -> bool {
    matches!(
        class,
        "ValueError"
            | "KeyError"
            | "RuntimeError"
            | "TypeError"
            | "IndexError"
            | "AttributeError"
            | "AssertionError"
            | "NotImplementedError"
            | "TableNotFoundError"
            | "MetricViewAsTableError"
            | "NoTablesFoundError"
            | "UnsupportedJoinError"
            | "UnsupportedFilterError"
            | "GraphConstructionError"
    )
}

// --- Snowflake semantic views ---------------------------------------------------------

/// `Graph.from_snowflake_semantic_view(name, source, trust_expressions=trust)`.
pub async fn from_snowflake_semantic_view(ctx: &mut Ctx<'_>, name: &str, trust: bool) -> ER<Graph> {
    let cfg: ER<PyVal> = async {
        let read = rows(
            ctx.db,
            Backend::Snow,
            &sdm_dialects::snow::semantic_view(name),
        )
        .await?;
        let Some(first) = read.first().and_then(|r| r.first()) else {
            return raise("IndexError", "list index out of range");
        };
        let PyVal::Str(text) = first else {
            return raise(
                "AttributeError",
                format!("'{}' object has no attribute 'read'", type_name(first)),
            );
        };
        crate::yaml::safe_load(text).map_err(|e| PyExc::new(e.class, e.message))
    }
    .await;
    let cfg = cfg.map_err(|e| as_graph_error(e, "Snowflake"))?;
    let mut graph = Graph::default();
    let mut msgs: Vec<String> = Vec::new();
    let tables = subscript(&cfg, "tables")?;
    let mut table_names: Vec<String> = Vec::new();
    for t in iterate(&tables)? {
        let n = name_text(&view_name(&subscript(&t, "name")?));
        if !table_names.contains(&n) {
            table_names.push(n);
        }
    }
    for table_cfg in iterate(&tables)? {
        let table_name = name_text(&view_name(&subscript(&table_cfg, "name")?));
        let base = subscript(&table_cfg, "base_table")?;
        let source_table_name = name_text(&view_name(&subscript(&base, "table")?));
        let database = view_name(&subscript(&base, "database")?);
        let schema = view_name(&subscript(&base, "schema")?);
        let mut primary_key: Option<String> = None;
        if contains(&table_cfg, "primary_key")? {
            let key_cfg = subscript(&table_cfg, "primary_key")?;
            let columns = subscript(&key_cfg, "columns")?;
            let n = len(&columns)?;
            if n == 1 {
                primary_key = Some(name_text(&view_name(&item(&columns, 0)?)));
            } else if n > 1 {
                msgs.push(format!(
                    "Failed to add primary key for table '{table_name}' since composite primary keys are not yet supported"
                ));
            }
        }
        let given = |value: &PyVal| match value {
            PyVal::None => None,
            other => Some(name_text(other)),
        };
        let mut table = EngineTable::new(
            ctx,
            Backend::Snow,
            TableSpec {
                name: table_name.clone(),
                source_name: Some(source_table_name),
                place: Place {
                    catalog: given(&database),
                    schema: given(&schema),
                    row_filter: None,
                },
                columns: Some(Vec::new()),
                primary_key: PkArg::None,
                time_column: None,
            },
        )
        .await?;
        let self_pattern = Qualifier::new(&table_name);
        let other_patterns: Vec<Qualifier> = table_names
            .iter()
            .filter(|n| **n != table_name)
            .map(|n| Qualifier::new(n))
            .collect();
        let mut columns: Vec<ColumnSpec> = Vec::new();
        let mut unsupported: Vec<String> = Vec::new();
        let mut sections: Vec<PyVal> = Vec::new();
        for section in ["dimensions", "time_dimensions", "facts"] {
            if let Some(found) = dict_get(&table_cfg, section)? {
                sections.extend(iterate(&found)?);
            }
        }
        for column_cfg in sections {
            let written = subscript(&column_cfg, "name")?;
            let column_name = name_text(&view_name(&written));
            let expr = dict_get(&column_cfg, "expr")?.unwrap_or(PyVal::None);
            let data_type = dict_get(&column_cfg, "data_type")?.unwrap_or(PyVal::None);
            let expr = match expr {
                PyVal::None => {
                    columns.push(ColumnSpec::plain(&column_name));
                    continue;
                }
                PyVal::Str(s) => s,
                other => {
                    return raise(
                        "TypeError",
                        format!(
                            "expected string or bytes-like object, got '{}'",
                            type_name(&other)
                        ),
                    );
                }
            };
            let expr = py_strip(&strip_outside_literals(&self_pattern, &expr));
            if matches!(&written, PyVal::Str(w) if *w == expr) {
                columns.push(ColumnSpec::plain(&column_name));
                continue;
            }
            let masked = mask_literals(&expr);
            if other_patterns.iter().any(|p| p.search(&masked)) {
                unsupported.push(column_name);
                continue;
            }
            if let Some(msg) = view_expr_message(
                &format!("column '{column_name}' of table '{table_name}'"),
                &expr,
                "semantic view",
                "snowflake",
                trust,
                &table
                    .source_column_dict(ctx)
                    .await?
                    .iter()
                    .map(|column| column.name.clone())
                    .collect::<Vec<_>>(),
            ) {
                msgs.push(msg);
                continue;
            }
            let dtype = match &data_type {
                PyVal::None => None,
                PyVal::Str(s) => snow_dtype(Some(s)),
                other => {
                    return raise(
                        "AttributeError",
                        format!("'{}' object has no attribute 'strip'", type_name(other)),
                    );
                }
            };
            columns.push(ColumnSpec {
                name: column_name,
                expr: Some(expr),
                dtype,
                stype: None,
            });
        }
        if unsupported.len() == 1 {
            msgs.push(format!(
                "Failed to add column '{}' of table '{table_name}' since its expression references other tables",
                unsupported[0]
            ));
        } else if unsupported.len() > 1 {
            msgs.push(format!(
                "Failed to add columns '{}' of table '{table_name}' since their expressions reference other tables",
                crate::py::str_list_repr(&unsupported)
            ));
        }
        table.add_columns(ctx, columns).await?;
        if let Some(key) = primary_key {
            if table.column(&key).is_none() {
                table
                    .add_columns(ctx, vec![ColumnSpec::plain(&key)])
                    .await?;
            }
            table.set_primary_key(Some(&key))?;
        }
        let mut preferred: Vec<String> = Vec::new();
        if let Some(found) = dict_get(&table_cfg, "time_dimensions")? {
            for c in iterate(&found)? {
                preferred.push(name_text(&view_name(&subscript(&c, "name")?)));
            }
        }
        time_candidates_and_infer(ctx, &mut table, &preferred).await?;
        graph.add_table(table)?;
    }
    let relationships = dict_get(&cfg, "relationships")?.unwrap_or(PyVal::List(Vec::new()));
    for relation_cfg in iterate(&relationships)? {
        let name = name_text(&view_name(&subscript(&relation_cfg, "name")?));
        let pairs = subscript(&relation_cfg, "relationship_columns")?;
        if len(&pairs)? != 1 {
            msgs.push(format!(
                "Failed to add relationship '{name}' since composite key references are not yet supported"
            ));
            continue;
        }
        let pair = item(&pairs, 0)?;
        let left_table = name_text(&view_name(&subscript(&relation_cfg, "left_table")?));
        let left_key = name_text(&view_name(&subscript(&pair, "left_column")?));
        let right_table = name_text(&view_name(&subscript(&relation_cfg, "right_table")?));
        let right_key = name_text(&view_name(&subscript(&pair, "right_column")?));
        let mut added_column = false;
        let attempt: ER<Option<String>> = async {
            let right = graph.get(&right_table)?;
            if graph.tables[right].primary_key.as_deref() != Some(right_key.as_str()) {
                return Ok(Some(format!(
                    "Failed to add relationship '{name}' since the referenced key '{right_key}' of table '{right_table}' is not a primary key"
                )));
            }
            let left = graph.get(&left_table)?;
            if graph.tables[left].primary_key.as_deref() == Some(left_key.as_str()) {
                return Ok(Some(format!(
                    "Failed to add relationship '{name}' since the referencing key '{left_key}' of table '{left_table}' is a primary key"
                )));
            }
            if !graph.tables[left].has_column(&left_key) {
                graph.tables[left].add_columns(ctx, vec![ColumnSpec::plain(&left_key)]).await?;
                added_column = true;
            }
            graph.link(&left_table, &left_key, &right_table)?;
            Ok(None)
        }
        .await;
        match attempt {
            Ok(Some(msg)) => msgs.push(msg),
            Ok(None) => {}
            Err(e) if is_value_or_key_error(&e) => {
                if added_column {
                    let left = graph.get(&left_table)?;
                    graph.tables[left].remove_column(&left_key)?;
                }
                msgs.push(format!(
                    "Failed to add relationship '{name}' since its keys could not be linked: {}",
                    e.message
                ));
            }
            Err(other) => return Err(other),
        }
    }
    msgs.extend(graph.drop_invalid_edges()?);
    graph.settle_time_columns()?;
    graph.validate()?;
    warn_conversion(
        ctx,
        format!("Could not fully convert the semantic view definition '{name}' into a graph:\n"),
        &msgs,
    );
    Ok(graph)
}

// --- Databricks metric views ------------------------------------------------------------

const MEASURE_SUFFIX: &str = " measure";

/// A metric view's definition, the data types of its dimensions by name,
/// and its catalog and schema.
type Definition = (
    String,
    Vec<(String, String)>,
    Option<String>,
    Option<String>,
);

/// `read_metric_view_definition(source, quoted_name)`.
async fn read_metric_view_definition(ctx: &mut Ctx<'_>, quoted_name: &str) -> ER<Definition> {
    let found = must(
        ctx.db,
        Backend::Databricks,
        &sdm_dialects::databricks::describe_extended(quoted_name),
    )
    .await?
    .rows();
    let mut dtypes: Vec<(String, String)> = Vec::new();
    let mut details: Vec<(String, String)> = Vec::new();
    let mut in_columns = true;
    for row in &found {
        let col_name = match row.first() {
            None | Some(PyVal::None) => String::new(),
            Some(v) => py_strip(&v.str_()),
        };
        let data_type = match row.get(1) {
            None | Some(PyVal::None) => String::new(),
            Some(v) => v.str_(),
        };
        if in_columns {
            if col_name.is_empty() || col_name.starts_with('#') {
                in_columns = false;
                continue;
            }
            if !data_type.ends_with(MEASURE_SUFFIX) {
                match dtypes.iter_mut().find(|(n, _)| *n == col_name) {
                    Some(slot) => slot.1 = data_type,
                    None => dtypes.push((col_name, data_type)),
                }
            }
        } else if !col_name.is_empty() && !details.iter().any(|(n, _)| *n == col_name) {
            details.push((col_name, data_type));
        }
    }
    let detail = |k: &str| details.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    if detail("Type").as_deref() != Some("METRIC_VIEW") {
        return raise(
            "ValueError",
            format!(
                "'{quoted_name}' is not a metric view (got type '{}')",
                detail("Type").unwrap_or_else(|| "UNKNOWN".to_string())
            ),
        );
    }
    let definition = detail("View Text").unwrap_or_default();
    if definition.is_empty() {
        return raise(
            "ValueError",
            format!("Could not read the definition of metric view '{quoted_name}'"),
        );
    }
    Ok((definition, dtypes, detail("Catalog"), detail("Database")))
}

fn table_parts(
    parts: &[String],
    view: &str,
    catalog: &Option<String>,
    schema: &Option<String>,
) -> ER<(String, String, String)> {
    let mut parts = parts.to_vec();
    if parts.len() == 2 {
        if let Some(c) = catalog {
            parts.insert(0, c.clone());
        }
    } else if parts.len() == 1
        && let (Some(c), Some(s)) = (catalog, schema)
        && !c.is_empty()
        && !s.is_empty()
    {
        parts.insert(0, s.clone());
        parts.insert(0, c.clone());
    }
    if parts.len() != 3 {
        return raise(
            "ValueError",
            format!(
                "Could not fully qualify table '{}' referenced by metric view '{view}'",
                parts.join(".")
            ),
        );
    }
    Ok((parts[0].clone(), parts[1].clone(), parts[2].clone()))
}

/// `resolve_join_keys(join, child_columns, parent_columns)`.
fn resolve_join_keys(
    join: &metric_view::MetricViewJoin,
    child: &[String],
    parent: &[String],
) -> ER<(String, String, String)> {
    let mut child_names = vec![join.alias.clone(), join.path.join(".")];
    child_names.dedup();
    let mut parent_names = vec![join.parent_alias.clone()];
    if join.path.len() > 1 {
        parent_names.push(join.path[..join.path.len() - 1].join("."));
    }
    let side = |r: &metric_view::JoinKeyRef| -> Option<String> {
        if let Some(q) = &r.qualifier {
            if child_names.contains(q) {
                return Some(join.alias.clone());
            }
            if parent_names.contains(q) {
                return Some(join.parent_alias.clone());
            }
            if q == SOURCE_ALIAS {
                return Some(SOURCE_ALIAS.to_string());
            }
            return None;
        }
        let in_child = child.contains(&r.column);
        let in_parent = parent.contains(&r.column);
        if in_child && !in_parent {
            return Some(join.alias.clone());
        }
        if in_parent && !in_child {
            return Some(join.parent_alias.clone());
        }
        None
    };
    let (first, second) = &join.condition;
    let (a, b) = (side(first), side(second));
    let own = |side: &Option<String>| side.as_deref() == Some(join.alias.as_str());
    if own(&a)
        && let Some(other) = b.clone().filter(|o| *o != join.alias)
    {
        return Ok((other, second.column.clone(), first.column.clone()));
    }
    if own(&b)
        && let Some(other) = a.filter(|o| *o != join.alias)
    {
        return Ok((other, first.column.clone(), second.column.clone()));
    }
    raise(
        "UnsupportedJoinError",
        format!(
            "Failed to add join '{}' since its join condition could not be resolved into a key reference between '{}' and '{}'",
            join.alias, join.parent_alias, join.alias
        ),
    )
}

/// Where [`at`] finds the join's own table, which is not in the graph yet.
const OWN: usize = usize::MAX;

async fn metric_source_table(
    ctx: &mut Ctx<'_>,
    alias: &str,
    parts: &(String, String, String),
) -> ER<EngineTable> {
    EngineTable::new(
        ctx,
        Backend::Databricks,
        TableSpec {
            name: alias.to_string(),
            source_name: Some(parts.2.clone()),
            place: Place {
                catalog: Some(parts.0.clone()),
                schema: Some(parts.1.clone()),
                row_filter: None,
            },
            columns: Some(Vec::new()),
            primary_key: PkArg::None,
            time_column: None,
        },
    )
    .await
}

async fn metric_columns_of(
    ctx: &mut Ctx<'_>,
    alias: &str,
    tables: &mut Vec<(String, EngineTable)>,
    spec: &metric_view::MetricViewSpec,
    view: &str,
    catalog: &Option<String>,
    schema: &Option<String>,
) -> ER<Vec<String>> {
    if !tables.iter().any(|(name, _)| name == alias) {
        let join = spec
            .joins
            .iter()
            .find(|join| join.alias == alias)
            .ok_or_else(|| PyExc::key_error(alias))?;
        let parts = table_parts(&join.table, view, catalog, schema)?;
        tables.push((
            alias.to_string(),
            metric_source_table(ctx, alias, &parts).await?,
        ));
    }
    let table = &mut tables
        .iter_mut()
        .find(|(name, _)| name == alias)
        .expect("table inserted")
        .1;
    match table.source_column_dict(ctx).await {
        Ok(columns) => Ok(columns.iter().map(|column| column.name.clone()).collect()),
        Err(error) if alias != SOURCE_ALIAS && error.is_value_error() => Ok(Vec::new()),
        Err(error) => Err(error),
    }
}

/// The table at `i` of the graph, or the join's own table.
fn at<'a>(graph: &'a mut Graph, own: &'a mut EngineTable, i: usize) -> &'a mut EngineTable {
    if i == OWN { own } else { &mut graph.tables[i] }
}

/// What adding a join changed, to undo if it is refused.
enum Undo {
    /// A column added where there was none.
    Added(usize, String),
    /// A derived column replaced by its source column.
    Replaced(usize, String, EColumn),
}

/// `Graph.from_databricks_metric_view(name, source, catalog=..., schema=...,
/// trust_expressions=trust)`.
pub async fn from_databricks_metric_view(
    ctx: &mut Ctx<'_>,
    view: &str,
    catalog: Option<&str>,
    schema: Option<&str>,
    trust: bool,
) -> ER<Graph> {
    let prepared: ER<_> = async {
        let Some(mut parts) = metric_view::parse_table_reference(Some(&PyVal::Str(view.to_string()))) else {
            return raise("ValueError", format!("Invalid metric view name '{view}'"));
        };
        if parts.len() == 1
            && let Some(s) = schema {
                parts.insert(0, s.to_string());
            }
        if parts.len() == 2
            && let Some(c) = catalog {
                parts.insert(0, c.to_string());
            }
        let quoted_name: Vec<String> = parts.iter().map(|p| Backend::Databricks.quote(p)).collect();
        let (definition, dtype_names, view_catalog, view_schema) = read_metric_view_definition(ctx, &quoted_name.join(".")).await?;
        let spec = metric_view::parse_metric_view(&definition, trust)
            .map_err(|e| PyExc::new(&e.class, e.message))?;
        let mut msgs = spec.messages.clone();
        let dtypes: Vec<(String, Option<sdm_pql::types::Dtype>)> =
            dtype_names.iter().map(|(n, d)| (n.clone(), databricks_dtype(d))).collect();
        let fact = table_parts(&spec.source, view, &view_catalog, &view_schema)?;
        if spec.joins.iter().any(|join| join.alias == fact.2) {
            return raise("ValueError", format!("Cannot convert metric view '{view}' into a graph since the join name '{}' collides with the name of the source table", fact.2));
        }
        let fact_table = metric_source_table(ctx, &fact.2, &fact).await?;
        let mut tables = vec![(SOURCE_ALIAS.to_string(), fact_table)];
        if let Some(filter) = &spec.filter {
            let checked = if trust {
                filter.trusted()
            } else {
                let columns = metric_columns_of(ctx, SOURCE_ALIAS, &mut tables, &spec, view, &view_catalog, &view_schema).await?;
                filter.checked(&columns).map_err(|error| PyExc::new(&error.class, error.message))?
            };
            tables[0].1.set_row_filter(Some(checked))?;
        }
        let mut by_alias: Vec<(String, Vec<ColumnSpec>)> = Vec::new();
        for column in &spec.columns {
            let spec_of = if metric_view::unquote_ident(&column.expr) == column.name {
                ColumnSpec::plain(&column.name)
            } else if let Some(msg) = view_expr_message(
                &format!("column '{}'", column.name),
                &column.expr,
                "metric view",
                "databricks",
                trust,
                &metric_columns_of(ctx, &column.alias, &mut tables, &spec, view, &view_catalog, &view_schema).await?,
            ) {
                msgs.push(msg);
                continue;
            } else {
                ColumnSpec {
                    name: column.name.clone(),
                    expr: Some(column.expr.clone()),
                    dtype: dtypes.iter().find(|(n, _)| *n == column.name).and_then(|(_, d)| *d),
                    stype: None,
                }
            };
            match by_alias.iter_mut().find(|(a, _)| *a == column.alias) {
                Some((_, v)) => v.push(spec_of),
                None => by_alias.push((column.alias.clone(), vec![spec_of])),
            }
        }
        Ok((spec, msgs, by_alias, fact, view_catalog, view_schema, tables))
    }
    .await;
    let prepared = prepared.map_err(|e| as_graph_error(e, "Databricks"))?;
    let (spec, mut msgs, by_alias, fact, view_catalog, view_schema, mut tables) = prepared;
    let columns_of = |alias: &str| {
        by_alias
            .iter()
            .find(|(a, _)| a == alias)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let mut graph = Graph::default();
    let mut fact_table = tables.remove(0).1;
    fact_table
        .add_columns(ctx, columns_of(SOURCE_ALIAS))
        .await?;
    graph.add_table(fact_table)?;
    // Aliases to the graph's table names.
    let mut table_dict: Vec<(String, String)> = vec![(SOURCE_ALIAS.to_string(), fact.2.clone())];
    for join in &spec.joins {
        let Some(parent_name) = table_dict
            .iter()
            .find(|(a, _)| *a == join.parent_alias)
            .map(|(_, n)| n.clone())
        else {
            msgs.push(format!(
                "Failed to add join '{}' since its parent join '{}' was not added",
                join.alias, join.parent_alias
            ));
            continue;
        };
        let parts = table_parts(&join.table, view, &view_catalog, &view_schema)?;
        let mut table = match tables.iter().position(|(alias, _)| *alias == join.alias) {
            Some(index) => tables.remove(index).1,
            None => metric_source_table(ctx, &join.alias, &parts).await?,
        };
        table.add_columns(ctx, columns_of(&join.alias)).await?;
        let mut undo: Vec<Undo> = Vec::new();
        let mut undo_key: Option<(usize, String, Stype)> = None;
        let mut join_msgs: Vec<String> = Vec::new();
        let mut added = false;
        let attempt: ER<()> = async {
            let child_columns: Vec<String> = table
                .source_column_dict(ctx).await?
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let parent = graph.get(&parent_name)?;
            let parent_columns: Vec<String> = graph.tables[parent]
                .source_column_dict(ctx).await?
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let (other_alias, other_key, child_key) =
                resolve_join_keys(join, &child_columns, &parent_columns)?;
            let Some(other_name) = table_dict
                .iter()
                .find(|(a, _)| *a == other_alias)
                .map(|(_, n)| n.clone())
            else {
                return Err(PyExc::key_error(&other_alias));
            };
            let other = graph.get(&other_name)?;
            let ((one, one_key), (many, many_key)) = if join.cardinality == "many_to_one" {
                ((OWN, child_key), (other, other_key))
            } else {
                ((other, other_key), (OWN, child_key))
            };
            let one_ref = at(&mut graph, &mut table, one);
            if let Some(pk) = &one_ref.primary_key
                && *pk != one_key
            {
                return raise(
                    "UnsupportedJoinError",
                    format!(
                        "Failed to add join '{}' since table '{}' already uses '{pk}' as its primary key",
                        join.alias, one_ref.name
                    ),
                );
            }
            for (i, key) in [(one, one_key.clone()), (many, many_key.clone())] {
                let t = at(&mut graph, &mut table, i);
                match t.column(&key) {
                    None => {
                        t.add_columns(ctx, vec![ColumnSpec::plain(&key)]).await?;
                        undo.push(Undo::Added(i, key.clone()));
                    }
                    Some(c) if !c.is_source() => {
                        let old = c.clone();
                        undo.push(Undo::Replaced(i, key.clone(), old));
                        t.remove_column(&key)?;
                        t.add_columns(ctx, vec![ColumnSpec::plain(&key)]).await?;
                        join_msgs.push(format!(
                            "Replaced the derived column '{key}' of table '{}' with its physical source column since join '{}' references it as a key",
                            t.name, join.alias
                        ));
                    }
                    Some(_) => {}
                }
            }
            let one_ref = at(&mut graph, &mut table, one);
            if one_ref.primary_key.is_none() {
                let stype = one_ref.column(&one_key).map(|c| c.stype).ok_or_else(|| {
                    PyExc::key_error(&format!(
                        "Column '{one_key}' not found in table '{}'",
                        one_ref.name
                    ))
                })?;
                undo_key = Some((one, one_key.clone(), stype));
                one_ref.set_primary_key(Some(&one_key))?;
            }
            let many_name = if many == OWN {
                join.alias.clone()
            } else {
                graph.tables[many].name.clone()
            };
            let one_name = if one == OWN {
                join.alias.clone()
            } else {
                graph.tables[one].name.clone()
            };
            graph.add_table(table.clone())?;
            added = true;
            table_dict.push((join.alias.clone(), join.alias.clone()));
            graph.link(&many_name, &many_key, &one_name)?;
            Ok(())
        }
        .await;
        match attempt {
            Ok(()) => msgs.extend(join_msgs),
            Err(e) if e.is_value_error() => {
                // The join's own table: in the graph if it got that far.
                let own_index = if added {
                    graph.index(&join.alias)
                } else {
                    None
                };
                let mut own = match own_index {
                    Some(i) => {
                        let t = graph.tables[i].clone();
                        graph.remove_table(&join.alias)?;
                        table_dict.retain(|(a, _)| *a != join.alias);
                        t
                    }
                    None => table.clone(),
                };
                if let Some((i, key, stype)) = &undo_key {
                    let t = at(&mut graph, &mut own, *i);
                    t.set_primary_key(None)?;
                    match t.columns.iter_mut().find(|c| c.name == *key) {
                        Some(c) => c.set_stype(*stype)?,
                        None => {
                            return Err(PyExc::key_error(&format!(
                                "Column '{key}' not found in table '{}'",
                                t.name
                            )));
                        }
                    }
                }
                for step in undo.into_iter().rev() {
                    let (i, key, old) = match step {
                        Undo::Added(i, key) => (i, key, None),
                        Undo::Replaced(i, key, old) => (i, key, Some(old)),
                    };
                    let t = at(&mut graph, &mut own, i);
                    if t.has_column(&key) {
                        t.remove_column(&key)?;
                    }
                    if let Some(old) = old {
                        t.columns.push(old);
                    }
                }
                if e.class == "UnsupportedJoinError" {
                    msgs.push(e.message);
                } else {
                    msgs.push(format!(
                        "Failed to add join '{}' since its keys could not be linked: {}",
                        join.alias, e.message
                    ));
                }
            }
            Err(other) => return Err(other),
        }
    }
    for table in &mut graph.tables {
        table.infer_time_column(ctx).await?;
    }
    msgs.extend(graph.drop_invalid_edges()?);
    graph.settle_time_columns()?;
    graph.validate()?;
    warn_conversion(
        ctx,
        format!("Could not fully convert the metric view definition '{view}' into a graph:\n"),
        &msgs,
    );
    Ok(graph)
}

/// `semantic_model_graph(model)`.
pub async fn semantic_model_graph(ctx: &mut Ctx<'_>, info: &SourceInfo) -> ER<Graph> {
    let name = info.semantic_model.clone().unwrap_or_default();
    let get = |k: &str| {
        info.namespace
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.as_str().map(str::to_string))
    };
    match info.backend {
        Backend::Databricks => {
            from_databricks_metric_view(
                ctx,
                &name,
                get("catalog").as_deref(),
                get("schema").as_deref(),
                info.trust_expressions,
            )
            .await
        }
        Backend::Snow => {
            let qualified = if name.contains('.') {
                name
            } else {
                let mut parts: Vec<String> = ["database", "schema"]
                    .iter()
                    .filter(|p| info.namespace.iter().any(|(n, _)| n == *p))
                    .map(|p| {
                        info.namespace
                            .iter()
                            .find(|(n, _)| n == p)
                            .map(|(_, v)| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                            .unwrap_or_default()
                    })
                    .collect();
                parts.push(name);
                parts.join(".")
            };
            from_snowflake_semantic_view(ctx, &qualified, info.trust_expressions).await
        }
        other => raise(
            "StructuredDataError",
            format!(
                "no semantic model is kept in a '{}' source",
                other.source_name()
            ),
        ),
    }
}
