// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data-request vocabulary carries no expression and no SQL (decision
//! E28): the structural guard that replaced the L0 scan's weak SQL tier.
//!
//! A `DataRequest` names what it wants with typed fields; a provider, or
//! `sdm-data` for a SQL provider, decides how to fetch it. So no field of
//! the vocabulary, at any depth, may hold a free-text expression or a SQL
//! string. The test reads the generated JSON Schema of `DataRequest` and
//! every definition it reaches, and holds:
//!
//! - no property is named as SQL or an expression (`DENIED`), anywhere;
//! - every text property (a string, or a list or map of strings) outside
//!   the typed values (`TYPED_VALUES`: a scalar, an Arrow field and its
//!   type) is named as an identifier the request refers to (`IDENTIFIERS`).
//!
//! A new text field fails until it is added to `IDENTIFIERS` in review, so
//! it is a name the request refers to, never text a provider interprets.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use sdm_contract::data::DataRequest;
use serde_json::Value;

/// Property names that hold SQL or an expression; never allowed.
const DENIED: &[&str] = &[
    "sql",
    "text",
    "query",
    "statement",
    "expression",
    "expr",
    "predicate",
    "condition",
    "clause",
    "where",
    "having",
    "order_by",
    "raw",
    "dialect",
];

/// Text properties that name what a request refers to.
const IDENTIFIERS: &[&str] = &["table", "tables", "column", "columns", "key"];

/// Definitions that are typed values, not request structure: their text is
/// a scalar's encoded value, a field's name, type or Arrow metadata.
const TYPED_VALUES: &[&str] = &["Scalar", "Field", "ColumnType"];

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("a schema serializes")
}

/// Whether a property's schema is text: a string, or a list or map of them,
/// not held to closed values (a serde tag is a `const`, an enum an `enum`).
fn is_text(property: &Value) -> bool {
    let string = |v: &Value| match v.get("type") {
        _ if v.get("const").is_some() || v.get("enum").is_some() => false,
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        _ => false,
    };
    string(property)
        || property.get("items").is_some_and(string)
        || property.get("additionalProperties").is_some_and(string)
}

/// Every property of `value` at any depth, outside the typed values, with
/// the path that reaches it; `denied` gathers denied names everywhere.
fn walk(value: &Value, path: &str, text: &mut Vec<(String, String)>, denied: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if key == "$defs" {
                    for (name, def) in child.as_object().into_iter().flatten() {
                        let mut inner = Vec::new();
                        walk(def, &format!("{path}/$defs/{name}"), &mut inner, denied);
                        if !TYPED_VALUES.contains(&name.as_str()) {
                            text.extend(inner);
                        }
                    }
                    continue;
                }
                if key == "properties" {
                    for (name, property) in child.as_object().into_iter().flatten() {
                        if DENIED.contains(&name.to_ascii_lowercase().as_str()) {
                            denied.push(format!("{path}/{name}"));
                        }
                        if is_text(property) {
                            text.push((format!("{path}/{name}"), name.clone()));
                        }
                    }
                }
                walk(child, &format!("{path}/{key}"), text, denied);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                walk(item, &format!("{path}/{i}"), text, denied);
            }
        }
        _ => {}
    }
}

/// What in a vocabulary's schema could carry SQL or an expression.
fn problems(schema: &Value) -> Vec<String> {
    let (mut text, mut denied) = (Vec::new(), Vec::new());
    walk(schema, "#", &mut text, &mut denied);
    let mut found: Vec<String> = denied
        .into_iter()
        .map(|p| format!("{p}: named as SQL or an expression"))
        .collect();
    found.extend(
        text.into_iter()
            .filter(|(_, name)| !IDENTIFIERS.contains(&name.as_str()))
            .map(|(p, _)| format!("{p}: a text field that names no identifier")),
    );
    found
}

#[test]
fn data_requests_hold_no_expression_or_sql() {
    let found = problems(&schema::<DataRequest>());
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn the_lists_do_not_overlap() {
    let denied: BTreeSet<_> = DENIED.iter().collect();
    assert!(IDENTIFIERS.iter().all(|name| !denied.contains(name)));
}

#[test]
fn the_vocabulary_has_identifier_text() {
    // The walk sees the request's own text fields; were it blind, the first
    // test would pass on anything.
    let (mut text, mut denied) = (Vec::new(), Vec::new());
    walk(&schema::<DataRequest>(), "#", &mut text, &mut denied);
    let names: BTreeSet<_> = text.iter().map(|(_, name)| name.as_str()).collect();
    assert!(
        names.contains("table") && names.contains("columns"),
        "{names:?}"
    );
}

// Seeded faults: shapes the guard must refuse.

#[derive(JsonSchema)]
#[serde(tag = "request")]
#[allow(dead_code, reason = "only its schema is read")]
enum WithSql {
    Scan { table: String, sql: String },
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Inner {
    order: Vec<String>,
}

#[derive(JsonSchema)]
#[serde(tag = "request")]
#[allow(dead_code, reason = "only its schema is read")]
enum WithExpression {
    Scan {
        table: String,
        filter: Option<String>,
    },
    Sorted {
        table: String,
        by: Inner,
    },
    Where {
        expression: String,
    },
}

#[test]
fn a_sql_field_is_refused() {
    let found = problems(&schema::<WithSql>());
    assert!(
        found
            .iter()
            .any(|p| p.ends_with("/sql: named as SQL or an expression")),
        "{found:#?}"
    );
}

#[test]
fn free_text_fields_are_refused() {
    let found = problems(&schema::<WithExpression>());
    for field in ["/filter:", "/order:", "/expression: named"] {
        assert!(
            found.iter().any(|p| p.contains(field)),
            "{field} in {found:#?}"
        );
    }
}
