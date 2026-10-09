// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL corpus (`tests/pql/corpus.json`): 417 queries, 193 the reference
//! accepts and 224 it refuses, each run through the adapter and compared
//! with the reference's answer.
//!
//! An accepted query gives the reference's validated query: its tree, read
//! into the validated-query format, equals the adapter's, field for field,
//! and each date literal reads as the reference reads it (`reading`,
//! recorded by `record_readings.py`). A refused query gives the same errors
//! in count, code and span, unknown-name errors as a multiset, since the
//! reference orders them from a Python set; messages are not promised. The
//! date warnings the reference gave are compared by count.
//!
//! Until T1.6b completes the parser, not every case agrees: the cases that
//! differ, and the parts of each that do, are pinned in
//! `tests/pql/known-differing.json`. Any other difference fails, and so does
//! a pinned part that agrees, so the list can only shrink.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the corpus file"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ColumnRef, ProfileSpec, QueryColumn, QuerySchema, QueryTable, Severity};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::resolve;
use sdm_core::profiles::Profile;
use serde_json::{Map, Value, json};

fn read(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/pql")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn corpus() -> Value {
    read("corpus.json")
}

/// The pinned differences: each case's id and the parts of it that differ.
fn known_differing() -> BTreeMap<String, BTreeSet<String>> {
    read("known-differing.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let parts = case["parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| text(p).unwrap())
                .collect();
            (text(&case["id"]).unwrap(), parts)
        })
        .collect()
}

/// The host facts of the recording: the clock in 2026.
fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
    }
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// A schema as the reference writes it, in the contract's form.
fn query_schema(schema: &Value) -> QuerySchema {
    let tables = schema["tables"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, table)| QueryTable {
            name: name.clone(),
            columns: table["cols"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| QueryColumn {
                    name: text(&c["name"]).unwrap(),
                    dtype: serde_json::from_value(c["dtype"].clone()).unwrap(),
                    stype: serde_json::from_value(c["stype"].clone()).unwrap(),
                })
                .collect(),
            primary_key: text(&table["pkey"]),
            time_column: text(&table["time_col"]),
            end_time_column: text(&table["end_time_col"]),
        })
        .collect();
    let links = schema["col_groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| {
            group["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| ColumnRef {
                    table: text(&c["table_name"]).unwrap(),
                    column: text(&c["col_name"]).unwrap(),
                })
                .collect()
        })
        .collect();
    QuerySchema { tables, links }
}

fn profile(name: &str) -> Profile {
    Profile::resolve(&ProfileSpec {
        name: serde_json::from_value(json!(name)).unwrap(),
        model: None,
        capabilities: BTreeMap::new(),
    })
}

// --- The reference's tree, in the validated-query format ---------------------

fn column_ref(fqn: &Value) -> Value {
    match fqn.as_str() {
        None => Value::Null,
        Some(fqn) => {
            let (table, column) = fqn.rsplit_once('.').unwrap();
            json!({"table": table, "column": column})
        }
    }
}

fn snake(text: &Value) -> Value {
    json!(text.as_str().unwrap().to_lowercase().replace(' ', "_"))
}

fn digits(text: &str) -> String {
    let (sign, body) = match text.strip_prefix('-') {
        Some(body) => ("-", body),
        None => ("", text),
    };
    let body = body.trim_start_matches('0');
    match body {
        "" => "0".into(),
        _ => format!("{sign}{body}"),
    }
}

/// A string literal's text without its quotes, its escapes read; the
/// corpus's strings use only `\\`, `\'` and `\"`.
fn unquote(text: &str) -> String {
    let inner: String = text
        .chars()
        .skip(1)
        .take(text.chars().count() - 2)
        .collect();
    let mut out = String::new();
    let mut escaped = false;
    for c in inner.chars() {
        if escaped || c != '\\' {
            out.push(c);
            escaped = false;
        } else {
            escaped = true;
        }
    }
    out
}

fn literal(value: &Value, dtype: &Value, reading: &Value) -> Value {
    if let Some(items) = value.as_array() {
        return json!({"type": "array", "items": items.iter().map(node).collect::<Vec<_>>()});
    }
    let text = value.as_str().unwrap();
    match dtype.as_str() {
        None => json!({"type": "null"}),
        Some("bool") => json!({"type": "bool", "value": text.to_lowercase() == "true"}),
        Some("int" | "byte" | "int16" | "int32" | "int64") => {
            json!({"type": "int", "value": digits(text)})
        }
        Some("float" | "float32" | "float64") => json!({"type": "float", "text": text}),
        Some("string" | "binary") => json!({"type": "string", "value": unquote(text)}),
        Some("date" | "time") => {
            // The validated-query format holds whole seconds.
            assert_eq!(reading["nanos"], 0, "the reading of {text}");
            json!({
                "type": "time",
                "text": text,
                "value": reading["value"],
                "offset": reading["offset"],
            })
        }
        Some(other) => panic!("a constant of type {other}"),
    }
}

fn node_type(dtype: &Value) -> Value {
    match dtype {
        Value::Null => Value::Null,
        Value::String(d) => json!({"shape": "value", "dtype": d}),
        Value::Object(array) => json!({"shape": "array", "dtype": array["nested_dtype"]}),
        other => panic!("a dtype {other}"),
    }
}

fn span(location: &Value) -> Value {
    match location {
        Value::Object(l) if l["data_available"] == true => json!({
            "start_row": l["start_row"], "start_col": l["start_col"],
            "end_row": l["end_row"], "end_col": l["end_col"],
        }),
        _ => Value::Null,
    }
}

fn optional_node(value: &Value) -> Value {
    if value.is_null() {
        Value::Null
    } else {
        node(value)
    }
}

/// One node of the reference's tree, whose kind its fields say.
fn node(tree: &Value) -> Value {
    let n = tree.as_object().unwrap();
    let kind = if n.contains_key("fqn") {
        json!({"node": "column", "column": column_ref(&n["fqn"])})
    } else if n.contains_key("aggr") {
        json!({
            "node": "aggregation",
            "aggregation": snake(&n["aggr"]),
            "target": node(&n["target"]),
            "window": n["aggr_time_range"],
            "group_by": column_ref(&n["group_by"]),
            "time_column": column_ref(&n["time_col"]),
        })
    } else if n.contains_key("rhs_target") {
        json!({
            "node": "join",
            "target": node(&n["rhs_target"]),
            "lhs_key": column_ref(&n["lhs_key"]),
            "rhs_key": column_ref(&n["rhs_key"]),
        })
    } else if n.contains_key("bool_op") {
        json!({
            "node": "logical",
            "op": snake(&n["bool_op"]),
            "left": node(&n["left"]),
            "right": optional_node(&n["right"]),
        })
    } else if n.contains_key("condition") {
        json!({"node": "filter", "target": node(&n["target"]), "condition": node(&n["condition"])})
    } else if n.contains_key("op") {
        let op = match n["op"].as_str().unwrap() {
            "=" => "eq",
            "!=" => "ne",
            "<" => "lt",
            "<=" => "le",
            ">" => "gt",
            ">=" => "ge",
            "IN" => "in",
            "IS IN" => "is_in",
            "STARTS WITH" => "starts_with",
            "ENDS WITH" => "ends_with",
            "CONTAINS" => "contains",
            "NOT CONTAINS" => "not_contains",
            other => panic!("an operator {other}"),
        };
        json!({
            "node": "condition",
            "target": node(&n["target"]),
            "op": op,
            "value": node(&n["value"]),
            "from_like": !n["input_op"].is_null(),
        })
    } else {
        let reading = n.get("reading").cloned().unwrap_or(Value::Null);
        json!({"node": "constant", "literal": literal(&n["value"], &n["dtype_maybe"], &reading)})
    };
    json!({
        "dtype": node_type(&n["dtype_maybe"]),
        "stype": n["stype_maybe"],
        "range": n["date_offset_range"],
        "span": span(&n["location"]),
        "kind": kind,
    })
}

/// The reference's validated query in the validated-query format.
fn validated(tree: &Value) -> Value {
    let response = &tree["validation_response"];
    for list in ["warnings", "errors", "info_items"] {
        assert_eq!(
            response[list],
            json!([]),
            "the corpus's trees carry no {list}"
        );
    }
    json!({
        "entity": node(&tree["entity_ast"]),
        "target": node(&tree["target_ast"]),
        "assuming": optional_node(&tree["whatif_ast"]),
        "entity_ids": optional_node(&tree["rfm_entity_ids"]),
        "problem_type": if tree["problem_type"].is_null() { Value::Null } else { snake(&tree["problem_type"]) },
        "top_k": tree["top_k"].as_i64().map(|k| k.to_string()),
        "num_forecasts": tree["num_forecasts"],
        "entity_clause": snake(&tree["for_each"]),
        "mode": if tree["rfm_query"] == true { "inference" } else { "training" },
        "evaluate": tree["evaluate"],
        "explain": tree["explain"],
    })
}

/// The tree without its date literals' readings, to tell a reading that
/// differs from a tree that does.
fn without_readings(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type") == Some(&json!("time")) {
                map.remove("value");
                map.remove("offset");
            }
            map.values_mut().for_each(without_readings);
        }
        Value::Array(items) => items.iter_mut().for_each(without_readings),
        _ => {}
    }
}

/// Where two JSON values first differ.
fn first_difference(a: &Value, b: &Value, path: &str) -> String {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys = |m: &Map<String, Value>| m.keys().cloned().collect::<Vec<_>>();
            if keys(x) != keys(y) {
                return format!("{path}: keys {:?} != {:?}", keys(x), keys(y));
            }
            x.iter()
                .find(|(k, v)| y[*k] != **v)
                .map_or(format!("{path}: equal"), |(k, v)| {
                    first_difference(v, &y[k], &format!("{path}.{k}"))
                })
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .find(|(_, (p, q))| p != q)
            .map_or(format!("{path}: equal"), |(i, (p, q))| {
                first_difference(p, q, &format!("{path}[{i}]"))
            }),
        _ => format!("{path}: {a} != {b}"),
    }
}

/// Error codes the reference gives in the order of a Python set.
fn names_unknown(code: &str) -> bool {
    matches!(
        code,
        "PQL_SCHEMA_UNKNOWN_COLUMN" | "PQL_SCHEMA_UNKNOWN_TABLE"
    )
}

/// Errors in order, each run of unknown-name errors sorted, so that run is
/// compared as a multiset and everything else in order.
fn in_comparable_order(mut errors: Vec<(String, Value)>) -> Vec<(String, Value)> {
    let mut start = 0;
    while start < errors.len() {
        let mut end = start;
        while end < errors.len() && names_unknown(&errors[end].0) {
            end += 1;
        }
        if end > start {
            errors[start..end].sort_by_key(|(code, span)| (code.clone(), span.to_string()));
            start = end;
        } else {
            start += 1;
        }
    }
    errors
}

/// How the adapter's answer differs from the reference's: each part that
/// differs, and how.
fn differences(case: &Value, schema: &QuerySchema) -> BTreeMap<&'static str, String> {
    let mut out = BTreeMap::new();
    let query = case["query"].as_str().unwrap();
    let profile = profile(case["profile"].as_str().unwrap());
    let got = match resolve(query, schema, &profile, &facts()) {
        Ok(got) => got,
        Err(error) => {
            out.insert(
                "answer",
                format!("no answer: {} {}", error.code, error.message),
            );
            return out;
        }
    };
    // Warnings, by count, where the corpus records them all: an accepted
    // case's tree holds the validator's warnings (none, which `validated`
    // checks) and `warnings` the date reader's; a refused case's record
    // holds no warnings of the validator's, so its are not compared.
    let want_warnings = case
        .get("warnings")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let found_warnings = got
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .count();
    if case.get("tree").is_some() && want_warnings != found_warnings {
        out.insert(
            "warnings",
            format!("{found_warnings} warnings, where the reference gives {want_warnings}"),
        );
    }
    let found_errors: Vec<(String, Value)> = got
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| (d.code.clone(), serde_json::to_value(d.span).unwrap()))
        .collect();
    if let Some(tree) = case.get("tree") {
        let Some(found) = &got.query else {
            out.insert("validity", format!("refused: {found_errors:?}"));
            return out;
        };
        let want = validated(tree);
        let found = serde_json::to_value(found).unwrap();
        if want != found {
            let (mut bare_want, mut bare_found) = (want.clone(), found.clone());
            without_readings(&mut bare_want);
            without_readings(&mut bare_found);
            if bare_want == bare_found {
                out.insert("readings", first_difference(&want, &found, "$"));
            } else {
                out.insert("tree", first_difference(&bare_want, &bare_found, "$"));
            }
        }
        if !found_errors.is_empty() {
            out.insert(
                "errors",
                format!("errors on an accepted query: {found_errors:?}"),
            );
        }
        return out;
    }
    if got.query.is_some() {
        out.insert("validity", "accepted, where the reference refuses".into());
        return out;
    }
    let want: Vec<(String, Value)> = case["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (text(&e["code"]).unwrap(), e["span"].clone()))
        .collect();
    if in_comparable_order(want.clone()) != in_comparable_order(found_errors.clone()) {
        out.insert(
            "errors",
            format!("errors {found_errors:?}, where the reference gives {want:?}"),
        );
    }
    out
}

#[test]
fn unknown_name_errors_compare_as_a_multiset_and_the_rest_in_order() {
    let error = |code: &str, col: u32| (code.to_owned(), json!({"start_col": col}));
    let unknown = "PQL_SCHEMA_UNKNOWN_COLUMN";
    let a = vec![error(unknown, 1), error(unknown, 9), error("PQL_X", 3)];
    let b = vec![error(unknown, 9), error(unknown, 1), error("PQL_X", 3)];
    assert_eq!(in_comparable_order(a.clone()), in_comparable_order(b));
    let c = vec![error("PQL_X", 3), error(unknown, 1), error(unknown, 9)];
    assert_ne!(in_comparable_order(a), in_comparable_order(c));
    let d = vec![error("PQL_X", 3), error("PQL_Y", 1)];
    let e = vec![error("PQL_Y", 1), error("PQL_X", 3)];
    assert_ne!(in_comparable_order(d), in_comparable_order(e));
}

#[test]
fn the_corpus_agrees_with_the_reference() {
    let corpus = corpus();
    let schemas: BTreeMap<&str, QuerySchema> = corpus["schemas"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, schema)| (name.as_str(), query_schema(schema)))
        .collect();
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 417, "the corpus holds 417 queries");
    let accepted = cases.iter().filter(|c| c.get("tree").is_some()).count();
    assert_eq!((accepted, cases.len() - accepted), (193, 224));
    let known = known_differing();
    let ids: BTreeSet<&str> = cases.iter().map(|c| c["id"].as_str().unwrap()).collect();
    let mut failures = Vec::new();
    for id in known.keys() {
        if !ids.contains(id.as_str()) {
            failures.push(format!("{id}: pinned as differing, and not in the corpus"));
        }
    }
    let mut differing = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let schema = &schemas[case["schema"].as_str().unwrap()];
        let found = differences(case, schema);
        if !found.is_empty() {
            differing += 1;
        }
        let pinned = known.get(id).cloned().unwrap_or_default();
        for (part, how) in &found {
            let status = if pinned.contains(*part) {
                "pinned"
            } else {
                "NOT PINNED"
            };
            println!("  {id}: {part} ({status}): {how}");
            if !pinned.contains(*part) {
                failures.push(format!("{id}: {part} differs: {how}"));
            }
        }
        for part in &pinned {
            if !found.contains_key(part.as_str()) {
                failures.push(format!(
                    "{id}: {part} is pinned as differing and now agrees; remove it from \
                     known-differing.json"
                ));
            }
        }
    }
    println!(
        "PQL corpus: {} of {} queries agree with the reference; {differing} differ, {} pinned",
        cases.len() - differing,
        cases.len(),
        known.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
