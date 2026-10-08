// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL corpus (`tests/pql/corpus.json`): 417 queries, 193 the reference
//! accepts and 224 it refuses, each run through the adapter and compared
//! with the reference's answer.
//!
//! An accepted query gives the reference's validated query: its tree, read
//! into the validated-query format, equals the adapter's, field for field,
//! except a date literal's reading, which the date tests check. A refused
//! query gives the same errors in count, code and span; messages are not
//! promised. The date warnings the reference gave are compared too.
//!
//! Until T1.6b completes the parser, not every case agrees: the test reports
//! how many do and which differ, and fails if fewer agree than
//! [`AGREEING_AT_LEAST`], so the count can only rise.

use std::collections::BTreeMap;
use std::path::Path;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ColumnRef, ProfileSpec, QueryColumn, QuerySchema, QueryTable, Severity};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::resolve;
use sdm_core::profiles::Profile;
use serde_json::{Map, Value, json};

/// How many cases agreed when the parser was ported (T1.6a).
const AGREEING_AT_LEAST: usize = 416;

fn corpus() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pql/corpus.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
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

fn literal(value: &Value, dtype: &Value) -> Value {
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
        Some("date" | "time") => json!({"type": "time", "text": text}),
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
        json!({"node": "constant", "literal": literal(&n["value"], &n["dtype_maybe"])})
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

/// A date literal's reading is checked by the date tests, not here.
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

/// How the adapter's answer differs from the reference's, if it does.
fn difference(case: &Value, schema: &QuerySchema) -> Option<String> {
    let query = case["query"].as_str().unwrap();
    let profile = profile(case["profile"].as_str().unwrap());
    let got = match resolve(query, schema, &profile, &facts()) {
        Ok(got) => got,
        Err(error) => return Some(format!("no answer: {} {}", error.code, error.message)),
    };
    if case.get("warnings").is_some() {
        return Some("the reference's date warnings are not given yet".into());
    }
    if let Some(tree) = case.get("tree") {
        let Some(found) = &got.query else {
            return Some(format!("refused: {:?}", got.diagnostics));
        };
        let mut want = validated(tree);
        let mut found = serde_json::to_value(found).unwrap();
        without_readings(&mut want);
        without_readings(&mut found);
        if want != found {
            return Some(format!("tree {}", first_difference(&want, &found, "$")));
        }
        return (!got.diagnostics.is_empty())
            .then(|| format!("diagnostics on an accepted query: {:?}", got.diagnostics));
    }
    if got.query.is_some() {
        return Some("accepted, where the reference refuses".into());
    }
    let want: Vec<(String, Value)> = case["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (text(&e["code"]).unwrap(), e["span"].clone()))
        .collect();
    let found: Vec<(String, Value)> = got
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| (d.code.clone(), serde_json::to_value(d.span).unwrap()))
        .collect();
    (want != found).then(|| format!("errors {found:?}, where the reference gives {want:?}"))
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
    let mut differences = Vec::new();
    for case in cases {
        let schema = &schemas[case["schema"].as_str().unwrap()];
        if let Some(problem) = difference(case, schema) {
            differences.push(format!("{}: {problem}", case["id"].as_str().unwrap()));
        }
    }
    let agreeing = cases.len() - differences.len();
    println!(
        "PQL corpus: {agreeing} of {} queries agree with the reference; {} differ",
        cases.len(),
        differences.len()
    );
    for difference in &differences {
        println!("  {difference}");
    }
    assert!(
        agreeing >= AGREEING_AT_LEAST,
        "{agreeing} corpus queries agree, fewer than the {AGREEING_AT_LEAST} of T1.6a:\n{}",
        differences.join("\n")
    );
}
