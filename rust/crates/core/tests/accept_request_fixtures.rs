// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P85.fixtures and P85.diagnostics: every request fixture case gives,
//! through `sdm_core::fixture_op`, what it records
//! (`tests/rust_core/request/CONTRACT.md`, "Fixtures").
//!
//! A request fixture case is a case of any `tests/parity/fixtures/*.json`
//! whose operation starts with `client.`, `decompose.`, `payload.`,
//! `tabular.` or `wire.`. The cases listed in
//! `scripts/parity/request_fixture_exclusions.json` are skipped; the
//! Phase 8.5 checks hold that list to its rules, and a reviewer reads it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The operations of the request fixture cases.
const PREFIXES: [&str; 5] = ["client.", "decompose.", "payload.", "tabular.", "wire."];

/// The fewest request fixture cases the corpus holds (Phase 8 left 549).
const AT_LEAST: usize = 500;

/// The keys an `expected_error` may record, each compared where it does.
const ERROR_KEYS: [&str; 4] = ["class", "code", "message", "details_keys"];

const CORRECTED_ID: &str = "request-gaps/client.run/a-tabular-target-dropped-after-its-schema";

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Correction {
    fixture_path: String,
    fix: String,
    original_case: Value,
    core_outcome: CorrectedOutcome,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CorrectedOutcome {
    error: CorrectedError,
    warnings: Vec<Value>,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CorrectedError {
    class: String,
    code: String,
    message: String,
    details_keys: Vec<String>,
}

fn correction(root: &Path) -> Correction {
    let text = fs::read_to_string(root.join("tests/rust_core/request/fixture_corrections.json"))
        .expect("the governed correction is present");
    let entry: Correction =
        serde_json::from_str(&text).expect("the governed correction has its exact schema");
    assert_eq!(
        entry.fixture_path,
        "tests/parity/fixtures/request-gaps.json"
    );
    assert_eq!(entry.fix, "dropped-target-or-entity-column");
    assert_eq!(entry.original_case["id"], CORRECTED_ID);
    assert_eq!(entry.original_case["operation"], "client.run");
    assert!(entry.original_case.get("expected").is_none());
    assert_eq!(
        entry.original_case["expected_error"],
        serde_json::json!({
            "class": "builtins.KeyError",
            "code": "KeyError",
            "message": "'churn'",
            "details_keys": []
        })
    );
    assert_eq!(
        entry.original_case["expected_warnings"],
        serde_json::json!([])
    );
    assert_eq!(
        entry.core_outcome.error.class,
        "sdm_client.errors.StructuredDataError"
    );
    assert_eq!(entry.core_outcome.error.code, "INVALID_REQUEST");
    assert!(!entry.core_outcome.error.message.is_empty());
    assert!(entry.core_outcome.error.details_keys.is_empty());
    assert!(entry.core_outcome.warnings.is_empty());
    entry
}

fn expected_case(case: &Value, entry: &Correction) -> Value {
    if case.get("id").and_then(Value::as_str) != Some(CORRECTED_ID) {
        return case.clone();
    }
    assert_eq!(
        case, &entry.original_case,
        "the original correction case changed"
    );
    let mut expected = case.clone();
    expected["expected_error"] = serde_json::to_value(&entry.core_outcome.error).unwrap();
    expected["expected_warnings"] = serde_json::to_value(&entry.core_outcome.warnings).unwrap();
    expected
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read(path: &Path) -> Value {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not JSON: {error}", path.display()))
}

fn short(value: &Value) -> String {
    let text = value.to_string();
    if text.chars().count() <= 300 {
        return text;
    }
    let cut: String = text.chars().take(300).collect();
    format!("{cut}...")
}

/// The excluded cases, by id.
fn exclusions(root: &Path) -> BTreeMap<String, String> {
    let path = root.join("scripts/parity/request_fixture_exclusions.json");
    if !path.is_file() {
        return BTreeMap::new();
    }
    match read(&path) {
        Value::Object(map) => map
            .into_iter()
            .map(|(id, reason)| match reason {
                Value::String(reason) => (id, reason),
                other => panic!("{}: the reason for {id} is {other}", path.display()),
            })
            .collect(),
        other => panic!("{} is not a JSON object: {}", path.display(), short(&other)),
    }
}

/// Why `got`, what `fixture_op` gave for `case`, is not what the case
/// records; `None` when it is.
fn difference(case: &Value, got: &Value) -> Option<String> {
    let Some(map) = got.as_object() else {
        return Some(format!("gave {}, not an object", short(got)));
    };
    let warnings = map.get("warnings");
    if !matches!(warnings, Some(Value::Array(_))) {
        return Some(format!("gave {}, without a warnings list", short(got)));
    }
    if let Some(expected) = case.get("expected") {
        match (map.get("output"), map.get("error")) {
            (Some(output), None) if output == expected => {}
            (Some(output), None) => {
                return Some(format!(
                    "output {} instead of {}",
                    short(output),
                    short(expected)
                ));
            }
            _ => {
                return Some(format!(
                    "gave {}, where it records the output {}",
                    short(got),
                    short(expected)
                ));
            }
        }
    }
    if let Some(want) = case.get("expected_error") {
        let (None, Some(error)) = (map.get("output"), map.get("error")) else {
            return Some(format!(
                "gave {}, where it records the error {}",
                short(got),
                short(want)
            ));
        };
        for key in ERROR_KEYS {
            if let Some(expected) = want.get(key)
                && error.get(key) != Some(expected)
            {
                return Some(format!(
                    "error {key} {} instead of {}",
                    error.get(key).map_or_else(|| "missing".to_owned(), short),
                    short(expected)
                ));
            }
        }
    }
    if let Some(expected) = case.get("expected_warnings")
        && warnings != Some(expected)
    {
        return Some(format!(
            "warnings {} instead of {}",
            warnings.map_or_else(|| "missing".to_owned(), short),
            short(expected)
        ));
    }
    None
}

#[test]
fn every_request_fixture_case_gives_what_it_records() {
    let root = repo();
    let fixtures = std::env::var_os("SDM_PARITY_FIXTURES_DIR")
        .map_or_else(|| root.join("tests/parity/fixtures"), PathBuf::from);
    let excluded = exclusions(&root);
    let correction = correction(&root);
    assert!(!excluded.contains_key(CORRECTED_ID));
    let mut corrected = 0_usize;
    let mut files: Vec<PathBuf> = fs::read_dir(&fixtures)
        .unwrap_or_else(|error| panic!("cannot list {}: {error}", fixtures.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    let mut counted = 0_usize;
    let mut skipped = 0_usize;
    let mut failures = Vec::new();
    for file in &files {
        let document = read(file);
        let Some(cases) = document.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            let operation = case
                .get("operation")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !PREFIXES.iter().any(|prefix| operation.starts_with(prefix)) {
                continue;
            }
            counted += 1;
            let id = case.get("id").and_then(Value::as_str).unwrap_or_default();
            if id == CORRECTED_ID {
                corrected += 1;
                assert_eq!(
                    file.file_name().and_then(|name| name.to_str()),
                    Some("request-gaps.json")
                );
                assert_eq!(case, &correction.original_case);
            }
            if excluded.contains_key(id) {
                skipped += 1;
                continue;
            }
            let input = case
                .get("input")
                .cloned()
                .unwrap_or(Value::Null)
                .to_string();
            let found = match sdm_core::fixture_op(operation, &input) {
                Err(error) => Some(format!("refused: {error}")),
                Ok(text) => match serde_json::from_str::<Value>(&text) {
                    Err(error) => Some(format!("gave text that is not JSON: {error}")),
                    Ok(got) => difference(&expected_case(case, &correction), &got),
                },
            };
            if let Some(why) = found {
                failures.push(format!("{id}: {why}"));
            }
        }
    }

    assert_eq!(corrected, 1, "the corrected case must run exactly once");
    assert!(
        counted >= AT_LEAST,
        "only {counted} request fixture cases under {}",
        fixtures.display()
    );
    let mut report = String::new();
    for failure in failures.iter().take(60) {
        let _ = writeln!(report, "{failure}");
    }
    assert!(
        failures.is_empty(),
        "{} of {counted} request fixture cases ({skipped} excluded) differ:\n{report}",
        failures.len()
    );
}

#[test]
fn correction_is_exact_and_never_accepts_the_old_core_result() {
    let entry = correction(&repo());
    let expected = expected_case(&entry.original_case, &entry);
    let got = serde_json::to_value(&entry.core_outcome).unwrap();
    assert!(difference(&expected, &got).is_none());
    let old = serde_json::json!({
        "error": entry.original_case["expected_error"],
        "warnings": entry.original_case["expected_warnings"]
    });
    assert!(difference(&entry.original_case, &old).is_none());
    assert!(difference(&expected, &old).is_some());
    assert!(difference(&entry.original_case, &got).is_some());
    assert!(
        difference(
            &expected,
            &serde_json::json!({
                "output": null, "warnings": []
            })
        )
        .is_some()
    );
    for key in ERROR_KEYS {
        let mut altered = got.clone();
        altered["error"][key] = serde_json::json!("wrong");
        assert!(difference(&expected, &altered).is_some());
    }
    let mut warned = got.clone();
    warned["warnings"] = serde_json::json!([["UserWarning", "unexpected"]]);
    assert!(difference(&expected, &warned).is_some());
    let mut changed_input = entry.original_case.clone();
    changed_input["input"] = Value::Null;
    assert!(std::panic::catch_unwind(|| expected_case(&changed_input, &entry)).is_err());
    let mut changed_old = entry.original_case.clone();
    changed_old["expected_error"]["message"] = serde_json::json!("wrong");
    assert!(std::panic::catch_unwind(|| expected_case(&changed_old, &entry)).is_err());
    let unrelated = serde_json::json!({
        "id": "unrelated", "expected": 1, "expected_warnings": []
    });
    assert_eq!(expected_case(&unrelated, &entry), unrelated);
}

#[test]
fn actual_corrected_refusal_rejects_a_seeded_wrong_message() {
    let entry = correction(&repo());
    let case = &entry.original_case;
    let text = sdm_core::fixture_op("client.run", &case["input"].to_string()).unwrap();
    let mut actual: Value = serde_json::from_str(&text).unwrap();
    let expected = expected_case(case, &entry);
    assert!(difference(&expected, &actual).is_none());
    actual["error"]["message"] = serde_json::json!("seeded wrong refusal");
    assert!(difference(&expected, &actual).is_some());
}
