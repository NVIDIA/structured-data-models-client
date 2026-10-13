// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The comparison the PQL corpus, quirk and custom-profile tests share: a
//! case as the reference recorded it (a tree as `json_serde` writes it, or
//! errors with their spans, and the date warnings it gave) against the
//! adapter's answer, part by part ([`differences`]).

#![allow(dead_code, reason = "each test file uses part of the comparison")]

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ColumnRef, ProfileSpec, QueryColumn, QuerySchema, QueryTable, Severity};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_core::pql::{DATE_LITERAL_DAY_FIRST, resolve};
use sdm_core::profiles::Profile;
use serde_json::{Map, Value, json};

/// Zones that hold nothing: these facts name no local zone, so none is
/// resolved.
pub struct NoZones;

impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}

/// The host facts of the recording: the clock in 2026.
pub fn facts() -> HostFacts {
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

pub fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// A schema as the reference writes it, in the contract's form.
pub fn query_schema(schema: &Value) -> QuerySchema {
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

pub fn profile(name: &str) -> Profile {
    Profile::resolve(&ProfileSpec {
        name: serde_json::from_value(json!(name)).unwrap(),
        model: None,
        capabilities: BTreeMap::new(),
    })
}

// --- The reference's tree, in the validated-query format ---------------------

pub fn column_ref(fqn: &Value) -> Value {
    match fqn.as_str() {
        None => Value::Null,
        Some(fqn) => {
            let (table, column) = fqn.rsplit_once('.').unwrap();
            json!({"table": table, "column": column})
        }
    }
}

pub fn snake(text: &Value) -> Value {
    json!(text.as_str().unwrap().to_lowercase().replace(' ', "_"))
}

pub fn digits(text: &str) -> String {
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
pub fn unquote(text: &str) -> String {
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

pub fn literal(value: &Value, dtype: &Value, reading: &Value) -> Value {
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

pub fn node_type(dtype: &Value) -> Value {
    match dtype {
        Value::Null => Value::Null,
        Value::String(d) => json!({"shape": "value", "dtype": d}),
        Value::Object(array) => json!({"shape": "array", "dtype": array["nested_dtype"]}),
        other => panic!("a dtype {other}"),
    }
}

pub fn span(location: &Value) -> Value {
    match location {
        Value::Object(l) if l["data_available"] == true => json!({
            "start_row": l["start_row"], "start_col": l["start_col"],
            "end_row": l["end_row"], "end_col": l["end_col"],
        }),
        _ => Value::Null,
    }
}

pub fn optional_node(value: &Value) -> Value {
    if value.is_null() {
        Value::Null
    } else {
        node(value)
    }
}

/// One node of the reference's tree, whose kind its fields say.
pub fn node(tree: &Value) -> Value {
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
pub fn validated(tree: &Value) -> Value {
    let response = &tree["validation_response"];
    for list in ["errors", "info_items"] {
        assert_eq!(response[list], json!([]), "a tree carries no {list}");
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
pub fn without_readings(value: &mut Value) {
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
pub fn first_difference(a: &Value, b: &Value, path: &str) -> String {
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

/// The format of the reference's day-first warning, pandas' `UserWarning`
/// "Parsing dates in <format> format when dayfirst=False ...".
pub fn reference_format(message: &str) -> String {
    let rest = message
        .strip_prefix("Parsing dates in ")
        .unwrap_or_else(|| panic!("not a day-first warning: {message}"));
    let (format, _) = rest
        .split_once(" format when dayfirst=False")
        .unwrap_or_else(|| panic!("not a day-first warning: {message}"));
    format.to_owned()
}

/// Whether `query` holds a date literal with a run of five digits or more
/// (D8), which the core refuses by name from T1.6c: a `DATE` token, three
/// runs of digits joined by one of `/`, `-` or `.`, with such a run, or
/// followed by a `TIME` token, three runs joined by `:`, with one.
pub fn holds_packed_date(query: &str) -> bool {
    let words: Vec<&str> = query
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ','))
        .collect();
    let runs = |word: &str, separators: &[char]| -> Option<Vec<usize>> {
        let sep = word.chars().find(|c| separators.contains(c))?;
        let parts: Vec<&str> = word.split(sep).collect();
        let digits = |p: &&str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
        (parts.len() == 3 && parts.iter().all(digits))
            .then(|| parts.iter().map(|p| p.len()).collect())
    };
    words.iter().enumerate().any(|(i, word)| {
        let Some(date) = runs(word, &['/', '-', '.']) else {
            return false;
        };
        let time = words.get(i + 1).and_then(|next| runs(next, &[':']));
        date.iter().chain(time.iter().flatten()).any(|n| *n >= 5)
    })
}

/// Whether `error` is the core's refusal by name of a D8 literal (T1.6c):
/// `INVALID_REQUEST` for `validate_query`'s rule `date_literal_packed`.
pub fn is_packed_refusal(error: &CoreError) -> bool {
    let text = |name: &str| match error.fields.get(name) {
        Some(Detail::Scalar(Scalar::Utf8(text))) => Some(text.as_str()),
        _ => None,
    };
    error.kind == ErrorKind::InvalidRequest
        && error.code == ErrorCode::InvalidRequest
        && text("operation") == Some("validate_query")
        && text("rule") == Some("date_literal_packed")
}

/// Error codes the reference gives in the order of a Python set.
pub fn names_unknown(code: &str) -> bool {
    matches!(
        code,
        "PQL_SCHEMA_UNKNOWN_COLUMN" | "PQL_SCHEMA_UNKNOWN_TABLE"
    )
}

/// Errors in order, each run of unknown-name errors sorted, so that run is
/// compared as a multiset and everything else in order.
pub fn in_comparable_order(mut errors: Vec<(String, Value)>) -> Vec<(String, Value)> {
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
pub fn differences(
    case: &Value,
    schema: &QuerySchema,
    profile: &Profile,
) -> BTreeMap<&'static str, String> {
    let mut out = BTreeMap::new();
    let query = case["query"].as_str().unwrap();
    let got = match resolve(query, schema, profile, &facts(), &NoZones) {
        Ok(got) => got,
        Err(error) => {
            out.insert(
                "answer",
                format!("no answer: {} {}", error.code, error.message),
            );
            return out;
        }
    };
    // Warnings, where the record holds them all: an accepted case's. The
    // validator's, on the tree, by code and span in order. The date
    // reader's (D4): each format the reference warned of, the adapter warns
    // of, and no other; the reference repeats a warning each time it reads
    // the literal (twice in a validation), the adapter gives it once per
    // literal (register PQL-3), so the formats are compared as a set. A
    // refused case's record holds no warnings, so its are not compared.
    let validator_warnings: Vec<(String, Value)> = got
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .map(|d| (d.code.clone(), serde_json::to_value(d.span).unwrap()))
        .collect();
    let want_validator: Vec<(String, Value)> = case
        .get("tree")
        .and_then(|t| t["validation_response"]["warnings"].as_array())
        .into_iter()
        .flatten()
        .map(|w| (text(&w["code"]).unwrap(), w["span"].clone()))
        .collect();
    let want_formats: BTreeSet<String> = case
        .get("warnings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|w| reference_format(w["message"].as_str().unwrap()))
        .collect();
    let found_formats: BTreeSet<String> = got
        .warnings
        .iter()
        .map(|w| {
            assert_eq!(w.code, DATE_LITERAL_DAY_FIRST);
            match &w.fields["format"] {
                Detail::Scalar(Scalar::Utf8(text)) => text.clone(),
                other => panic!("a format that is not text: {other:?}"),
            }
        })
        .collect();
    // The core gives one warning per day-first literal (PQL-3): two at one
    // literal's place are a doubled warning, whatever the reference's count.
    let places: Vec<String> = got
        .warnings
        .iter()
        .map(|w| {
            format!(
                "{:?} {:?}",
                w.fields.get("start_row"),
                w.fields.get("start_col")
            )
        })
        .collect();
    let distinct: BTreeSet<&String> = places.iter().collect();
    if distinct.len() != places.len() {
        out.insert(
            "warnings",
            format!("a day-first literal warned more than once: {places:?}"),
        );
    }
    if case.get("tree").is_some()
        && (validator_warnings != want_validator || want_formats != found_formats)
    {
        out.insert(
            "warnings",
            format!(
                "validator warnings {validator_warnings:?} and date formats {found_formats:?}, \
                 where the reference gives {want_validator:?} and {want_formats:?}"
            ),
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
