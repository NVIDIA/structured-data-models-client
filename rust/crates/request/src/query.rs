// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/query.py`'s `PredictiveQuery`: a question resolved against a
//! schema for a profile, the checks `materialize` makes of its arguments,
//! and the route a request is gathered on.

use sdm_pql::ast;
use sdm_schema::client::Schema;
use sdm_schema::dt::Host;
use serde_json::Value;

use crate::error::{R, Warnings, invalid};
use crate::models::Profile;

/// An aggregation a target ends in, as decomposing it needs it.
#[derive(Clone, Debug)]
pub struct Aggregation {
    pub aggr: String,
    /// `str(aggregation.target)`.
    pub target_text: String,
    pub start: Option<i64>,
    pub end: i64,
    pub unit: String,
}

/// A query the language resolved, as much of it as gathering reads.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The validated query as the sampler reads it.
    pub tree: Value,
    /// `query.to_string()`.
    pub text: String,
    pub entity_column: String,
    pub target_column: String,
    pub target_stype: Option<String>,
    /// `str(query.problem_type)`'s value, such as `RANK`.
    pub problem_type: Option<String>,
    pub top_k: Option<i64>,
    pub num_forecasts: i64,
    pub for_each: String,
    /// `str(query.entity_ast)`.
    pub entity_text: String,
    /// `str(query.entity_ast.condition)` where the entity is filtered.
    pub entity_filter: Option<String>,
    /// Whether the target is a window (`target_ast.date_offset_range`).
    pub windowed: bool,
    pub exclude_cols: Vec<(String, Vec<String>)>,
    pub final_aggregation: Option<Aggregation>,
}

/// `table_name(full_name)`.
pub fn table_name(full: &str) -> String {
    let parts: Vec<&str> = full.split('.').collect();
    if parts.len() < 2 {
        return full.to_string();
    }
    parts[..parts.len() - 1].join(".")
}

/// `col_name(full_name)`.
pub fn col_name(full: &str) -> String {
    full.rsplit('.').next().unwrap_or(full).to_string()
}

fn summary(query: &sdm_pql::ValidatedQuery) -> Resolved {
    let parsed = &query.query;
    let final_aggregation = query.get_final_target_aggregation().and_then(|node| {
        let window = ast::aggr_time_range(&node)?;
        let window = window.borrow();
        Some(Aggregation {
            aggr: ast::aggr(&node).map_or_else(String::new, |a| a.value().to_string()),
            target_text: ast::target(&node).map_or_else(String::new, |t| ast::to_string(&t)),
            start: window.start,
            end: window.end,
            unit: window.unit.value().to_string(),
        })
    });
    let entity_filter = match &parsed.entity_ast.borrow().kind {
        ast::Kind::Filter { condition, .. } => Some(ast::to_string(condition)),
        _ => None,
    };
    Resolved {
        tree: serde_json::from_str(&parsed.to_json(&query.validation_response).render())
            .unwrap_or(Value::Null),
        text: parsed.to_string_with(false, false),
        entity_column: parsed.entity_column(),
        target_column: query.target_column(),
        target_stype: query.target_stype().map(|s| s.value().to_string()),
        problem_type: parsed.problem_type.map(|p| p.value().to_string()),
        top_k: parsed.top_k.as_ref().and_then(|k| k.to_i64()),
        num_forecasts: parsed.num_forecasts.to_i64().unwrap_or(1),
        for_each: parsed.for_each.clone(),
        entity_text: ast::to_string(&parsed.entity_ast),
        entity_filter,
        windowed: ast::date_offset_range(&parsed.target_ast).is_some(),
        exclude_cols: query.get_exclude_cols_dict().into_iter().collect(),
        final_aggregation,
    }
}

/// `pql::Host` of a schema core host.
fn pql_host(host: &Host) -> sdm_pql::dates::Host {
    sdm_pql::dates::Host {
        year: host.year,
        pads_year: host.pads_year,
    }
}

/// `to_schema_definition(schema)` as the language reads it: each column an
/// object, each group of joined columns an object of its columns.
fn pql_schema(definition: &Value) -> Value {
    let mut tables = serde_json::Map::new();
    if let Some(Value::Object(given)) = definition.get("tables") {
        for (name, table) in given {
            let cols: Vec<Value> = table
                .get("cols")
                .and_then(Value::as_array)
                .map(|cols| {
                    cols.iter()
                        .map(|c| {
                            serde_json::json!({
                                "name": c.get(0).cloned().unwrap_or(Value::Null),
                                "stype": c.get(1).cloned().unwrap_or(Value::Null),
                                "dtype": c.get(2).cloned().unwrap_or(Value::Null),
                                "timestamp_format": c.get(3).cloned().unwrap_or(Value::Null),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut out = table.as_object().cloned().unwrap_or_default();
            out.insert("cols".to_string(), Value::Array(cols));
            tables.insert(name.clone(), Value::Object(out));
        }
    }
    let groups: Vec<Value> = definition
        .get("col_groups")
        .and_then(Value::as_array)
        .map(|groups| {
            groups
                .iter()
                .map(|g| {
                    let columns: Vec<Value> = g
                        .as_array()
                        .map(|pairs| {
                            pairs
                                .iter()
                                .map(|p| {
                                    serde_json::json!({
                                        "table_name": p.get(0).cloned().unwrap_or(Value::Null),
                                        "col_name": p.get(1).cloned().unwrap_or(Value::Null),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    serde_json::json!({ "columns": columns })
                })
                .collect()
        })
        .unwrap_or_default();
    serde_json::json!({ "tables": tables, "col_groups": groups })
}

/// What validating gave, sent back from the thread it ran on.
struct Validated {
    resolved: Option<Resolved>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

/// `PredictiveQuery._resolve(profile)`.
pub fn resolve(
    text: &str,
    schema: &Schema,
    profile: Profile,
    host: &Host,
    said: &mut Warnings,
) -> R<Resolved> {
    let definition = sdm_schema::client::definition(schema);
    if let Some(Value::Array(given)) = definition.get("warnings") {
        for warning in given {
            if let (Some(category), Some(message)) = (
                warning.get("category").and_then(Value::as_str),
                warning.get("message").and_then(Value::as_str),
            ) {
                said.push(sdm_protocol::Diagnostic::warning(category, message));
            }
        }
    }
    let schema_def =
        sdm_pql::SchemaDefinition::from_value(&pql_schema(&definition)).map_err(|e| {
            crate::error::value_error(format!("the schema is not one the language reads: {e}"))
        })?;
    let pql_profile = sdm_pql::Profile::by_name(profile.name())
        .ok_or_else(|| crate::error::value_error("no such profile"))?;
    let model = pql_profile.capabilities.model.clone();
    // The validation runs on a stack of its own, however deep the query.
    let validation = sdm_pql::validate_in(text, &schema_def, &pql_profile, pql_host(host));
    let validated = Validated {
        resolved: validation
            .query
            .as_ref()
            .filter(|_| validation.response.ok())
            .map(summary),
        errors: validation
            .response
            .errors
            .iter()
            .map(|e| e.message.clone())
            .collect(),
        warnings: validation.warnings.clone(),
    };
    for warning in validated.warnings {
        said.user(warning);
    }
    if let Some(resolved) = validated.resolved {
        return Ok(resolved);
    }
    let joined = validated
        .errors
        .iter()
        .map(|m| without_the_validator_trace(m))
        .collect::<Vec<_>>()
        .join("; ");
    let reasons = if joined.is_empty() {
        "the query did not resolve".to_string()
    } else {
        joined
    };
    Err(invalid(format!(
        "{} is not a query {model} can answer: {}{}{}",
        sdm_pql::pyrepr::str_repr(text),
        without_a_remedy_this_caller_cannot_run(&reasons),
        why_the_target_is_text(text, &reasons, profile, schema)?,
        how_to_resolve_an_ambiguous_link(&reasons, schema),
    )))
}

/// `_without_the_validator_trace(message)`.
pub fn without_the_validator_trace(message: &str) -> String {
    let pattern = regex::Regex::new(r"(?s)Value error, (.+?) \[type=").expect("a pattern");
    let found: Vec<String> = pattern
        .captures_iter(message)
        .filter_map(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .collect();
    if found.is_empty() {
        message.to_string()
    } else {
        found.join("; ")
    }
}

/// `_without_a_remedy_this_caller_cannot_run(reasons)`.
pub fn without_a_remedy_this_caller_cannot_run(reasons: &str) -> String {
    let pattern = regex::Regex::new(
        r"\s*(?:Keep the one this query means and drop the rest, for example:|You can drop the extra foreign keys? with:)[^.]*?(?:schema|graph)\.unlink\([^)]*\)(?:\s*;\s*(?:schema|graph)\.unlink\([^)]*\))*\.?",
    )
    .expect("a pattern");
    pattern.replace_all(reasons, "").into_owned()
}

/// `_target_named_by_the_query()`: the table and column `PREDICT t.c`
/// names, where the schema has the table.
fn target_named_by_the_query(text: &str, schema: &Schema) -> Option<(String, String)> {
    let pattern =
        regex::Regex::new(r"(?i)PREDICT\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s").expect("a pattern");
    let found = pattern.captures(text)?;
    let (table, name) = (found.get(1)?.as_str(), found.get(2)?.as_str());
    schema
        .tables
        .iter()
        .any(|t| t.name == table)
        .then(|| (table.to_string(), name.to_string()))
}

/// `_why_the_target_is_text(reasons, profile)`.
fn why_the_target_is_text(
    text: &str,
    reasons: &str,
    profile: Profile,
    schema: &Schema,
) -> R<String> {
    if !reasons.contains("semantic type text") {
        return Ok(String::new());
    }
    let Some((table, name)) = target_named_by_the_query(text, schema) else {
        return Ok(String::new());
    };
    let Some(described) = schema.tables.iter().find(|t| t.name == table) else {
        return Ok(String::new());
    };
    let Some(column) = described.col(&name) else {
        return Ok(String::new());
    };
    if column.stype != "text" || column.stype_declared {
        return Ok(String::new());
    }
    let distinct = described
        .data
        .as_ref()
        .and_then(|frame| frame.column(&name))
        .map(sdm_schema::frame::Column::nunique)
        .transpose()
        .map_err(|refusal| {
            sdm_protocol::Diagnostic::refusal(&refusal.raised_as(), None, refusal.message)
        })?;
    let holds = match distinct {
        Some(n) => format!(
            "holds {} distinct values",
            sdm_schema::py::thousands(n as i64)
        ),
        None => "was read as prose".to_string(),
    };
    let said = format!(
        ". {table}.{name} {holds}, and a string column with more than 64 is typed text rather \
         than categorical. That is fair for a feature and a poor one for a label"
    );
    let ceiling =
        (profile == Profile::Relational).then_some(crate::models::relational_limits::CLASSES_MAX);
    if let (Some(ceiling), Some(distinct)) = (ceiling, distinct)
        && distinct as i64 > ceiling
    {
        let model = sdm_pql::Profile::by_name(profile.name())
            .map(|p| p.capabilities.model)
            .unwrap_or_default();
        return Ok(format!(
            "{said}. Typing it categorical would not help here, though: {model} classifies at \
             most {ceiling} classes and this column holds {}, so the send would be refused. \
             Predict a column with fewer classes, or group these into that many first",
            sdm_schema::py::thousands(distinct as i64)
        ));
    }
    Ok(format!(
        "{said}. If it is a label, say so with schema[{}][{}].stype = 'categorical'",
        sdm_pql::pyrepr::str_repr(&table),
        sdm_pql::pyrepr::str_repr(&name)
    ))
}

/// `_how_to_resolve_an_ambiguous_link(reasons)`.
fn how_to_resolve_an_ambiguous_link(reasons: &str, schema: &Schema) -> String {
    if !reasons.contains("is not unique") {
        return String::new();
    }
    let mut ways: Vec<((String, String), Vec<String>)> = Vec::new();
    for link in &schema.relationships {
        let pair = (link.source_table.clone(), link.target_table.clone());
        match ways.iter_mut().find(|(p, _)| *p == pair) {
            Some(entry) => entry.1.push(link.source_column.clone()),
            None => ways.push((pair, vec![link.source_column.clone()])),
        }
    }
    let ambiguous: Vec<((String, String), Vec<String>)> = ways
        .into_iter()
        .filter(|(_, c)| c.len() > 1)
        .map(|(p, mut c)| {
            c.sort();
            (p, c)
        })
        .collect();
    if ambiguous.is_empty() {
        return String::new();
    }
    let crossed: Vec<&((String, String), Vec<String>)> = ambiguous
        .iter()
        .filter(|((s, t), _)| {
            reasons.contains(&format!("table {s} ")) && reasons.contains(&format!("table {t} "))
        })
        .collect();
    if ambiguous.len() > 1 && crossed.is_empty() {
        return String::new();
    }
    let mut chosen: Vec<&((String, String), Vec<String>)> = if crossed.is_empty() {
        ambiguous.iter().collect()
    } else {
        crossed
    };
    chosen.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let ((source, target), columns) = chosen[0];
    let kept = &columns[0];
    let rest: Vec<String> = columns[1..].to_vec();
    let named: Vec<String> = columns
        .iter()
        .map(|c| sdm_pql::pyrepr::str_repr(c))
        .collect();
    let s = sdm_pql::pyrepr::str_repr(source);
    let t = sdm_pql::pyrepr::str_repr(target);
    format!(
        " In this schema {s} reaches {t} through {}; to keep {}, schema.relationships = [link \
         for link in schema.relationships if not (link.source_table == {s} and \
         link.target_table == {t} and link.source_column in {})].",
        named.join(", "),
        sdm_pql::pyrepr::str_repr(kept),
        sdm_schema::py::str_list_repr(&rest)
    )
}
