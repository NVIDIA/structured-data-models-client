// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's edits on the pure-Rust host (T1.24b, decision E67): each
//! `edit.*` case's (`contract/fixtures/edit-cases`) edits are the core's `edit_schema`, as on python-core,
//! and what this host starts the core with, the core's error code and the
//! request state are the ones `contract/pinned/edits.core.json` pins from
//! python-core's run (`contract/tests/test_edit_cases.py`), so CI gates the
//! two hosts' equality, refused edits included. The check fails closed on
//! a pin missing, empty or retyped, seeded below.

use std::collections::BTreeMap;
use std::path::PathBuf;

use sdm_testkit::edits::{Link, SchemaEdit};
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case};
use serde_json::{Value, json};

const FIELDS: [&str; 3] = ["inputs", "error", "materialized"];

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

/// `value` as canonical JSON, keys sorted at every depth, as
/// `test_edit_cases.digest` writes it.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(members) => {
            let mut sorted: Vec<(&String, &Value)> = members.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k.clone(), canonical(v)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn digest(value: &Value) -> String {
    case::sha256(serde_json::to_string(&canonical(value)).unwrap().as_bytes())
}

/// A case's edit as this host makes it, every edit of the case format
/// named (`SchemaEdit::from_case` names the two a gate's checkpoint loader
/// holds, until it holds these).
fn schema_edit(edit: &case::Edit) -> SchemaEdit {
    match edit {
        case::Edit::ClearTimeColumn { table } => SchemaEdit::ClearTimeColumn {
            table: table.0.clone(),
        },
        case::Edit::DeleteColumn { table, column } => SchemaEdit::DeleteColumn {
            table: table.0.clone(),
            column: column.0.clone(),
        },
        case::Edit::SetPrimaryKey { table, column } => SchemaEdit::SetPrimaryKey {
            table: table.0.clone(),
            column: column.0.clone(),
        },
        case::Edit::SetTimeColumn { table, column } => SchemaEdit::SetTimeColumn {
            table: table.0.clone(),
            column: column.0.clone(),
        },
        case::Edit::SetStype {
            table,
            column,
            stype,
        } => SchemaEdit::SetStype {
            table: table.0.clone(),
            column: column.0.clone(),
            stype: stype.0.clone(),
        },
        case::Edit::SetRelationships { relationships } => SchemaEdit::SetRelationships {
            relationships: relationships
                .iter()
                .map(|link| Link {
                    source_table: link.source_table.0.clone(),
                    source_column: link.source_column.0.clone(),
                    target_table: link.target_table.0.clone(),
                    target_column: link.target_column.0.clone(),
                })
                .collect(),
        },
    }
}

/// Each edit case, run on this host: what the pin holds of it, by case.
fn observed() -> BTreeMap<String, Value> {
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(contract().join("fixtures/edit-cases"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("edit.") && name.ends_with(".json"))
        })
        .collect();
    paths.sort();
    let mut seen = BTreeMap::new();
    for loaded in case::load(&paths).unwrap() {
        let edits: Vec<SchemaEdit> = loaded
            .case
            .input
            .edits
            .iter()
            .flatten()
            .map(schema_edit)
            .collect();
        let ran = host::run_edited(&loaded.case, &bases, &edits).unwrap();
        let error = match &ran.answer {
            Answer::Error { code, .. } => Value::from(code.clone()),
            _ => Value::Null,
        };
        seen.insert(
            loaded.case.id.clone(),
            json!({
                "inputs": digest(&Value::from(ran.inputs.clone())),
                "error": error,
                "materialized": ran.materialized.as_ref().map_or(Value::Null, |m| digest(m).into()),
            }),
        );
    }
    seen
}

fn pin() -> Value {
    serde_json::from_slice(&std::fs::read(contract().join("pinned/edits.core.json")).unwrap())
        .unwrap()
}

/// Each way `seen` departs from `pin`, as `test_edit_cases.problems`
/// reads them; missing, empty or retyped evidence is a problem.
fn problems(pin: &Value, seen: &BTreeMap<String, Value>) -> Vec<String> {
    if pin.get("format").and_then(Value::as_str) != Some("sdm-edits-pin/1") {
        return vec!["the pin is not an sdm-edits-pin/1 document".into()];
    }
    let Some(cases) = pin
        .get("cases")
        .and_then(Value::as_object)
        .filter(|c| !c.is_empty())
    else {
        return vec!["the pin holds no case".into()];
    };
    let mut names: Vec<&String> = cases.keys().chain(seen.keys()).collect();
    names.sort();
    names.dedup();
    let mut found = Vec::new();
    for name in names {
        let Some(wanted) = cases
            .get(name)
            .and_then(Value::as_object)
            .filter(|w| w.len() == FIELDS.len() && FIELDS.iter().all(|f| w.contains_key(*f)))
        else {
            found.push(format!("{name}: not pinned as {}", FIELDS.join(", ")));
            continue;
        };
        if !wanted["inputs"].as_str().is_some_and(|d| d.len() == 64) {
            found.push(format!("{name}: inputs is not a digest"));
            continue;
        }
        let Some(mine) = seen.get(name) else {
            found.push(format!("{name}: pinned, and not run"));
            continue;
        };
        for field in FIELDS {
            if mine[field] != wanted[field] {
                found.push(format!("{name}: {field} is not the pin's"));
            }
        }
    }
    found
}

#[test]
fn the_edit_cases_on_this_host_are_the_pin_s() {
    let seen = observed();
    assert_eq!(seen.len(), 8);
    assert_eq!(problems(&pin(), &seen), Vec::<String>::new());
    // Seven refused at an edit, as the reference refuses them.
    let refused = seen
        .values()
        .filter(|v| v["error"] == "INVALID_REQUEST")
        .count();
    assert_eq!(refused, 7);
}

#[test]
fn the_pin_check_fails_closed() {
    let pin = pin();
    let seen: BTreeMap<String, Value> = pin["cases"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert!(problems(&pin, &seen).is_empty());
    let name = seen.keys().next().unwrap().clone();
    let mut differing = seen.clone();
    differing.get_mut(&name).unwrap()["error"] = "INTERNAL_ERROR".into();
    assert_eq!(
        problems(&pin, &differing),
        [format!("{name}: error is not the pin's")]
    );
    let mut missing = seen.clone();
    missing.remove(&name);
    assert_eq!(
        problems(&pin, &missing),
        [format!("{name}: pinned, and not run")]
    );
    let mut empty = pin.clone();
    empty["cases"] = json!({});
    assert_eq!(problems(&empty, &seen), ["the pin holds no case"]);
    let mut other = pin.clone();
    other["format"] = "other".into();
    assert!(!problems(&other, &seen).is_empty());
    let mut retyped = pin.clone();
    retyped["cases"][&name] = json!(["inputs"]);
    assert!(problems(&retyped, &seen)[0].starts_with(&format!("{name}: not pinned")));
    let mut short = pin.clone();
    short["cases"][&name]["inputs"] = "x".into();
    assert_eq!(
        problems(&short, &seen),
        [format!("{name}: inputs is not a digest")]
    );
}
