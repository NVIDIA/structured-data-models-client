// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The S0 cases of `contract/cases` through the pure-Rust host: each loads,
//! runs through `HostRun` and meets its expected outcome and code. CI's
//! `rust-test` job also runs `sdm-contract-run` over them and compares its
//! observations with the reference's (`scripts/rust_core/compare.py`).

use std::path::PathBuf;

use sdm_testkit::case::{self, Status, Until};
use sdm_testkit::host::{self, Answer};

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/cases")
}

#[test]
fn the_s0_cases_meet_their_expectation_on_the_pure_rust_host() {
    let loaded = case::load(&[cases_dir()]).unwrap();
    let s0: Vec<_> = loaded
        .iter()
        .filter(|each| each.case.slice == "S0")
        .collect();
    assert!(s0.len() >= 2, "the accepted and the refused S0 case");
    let mut answers = Vec::new();
    for each in s0 {
        let case = &each.case;
        assert_eq!(case.status, Status::Acceptance);
        assert_eq!(case.until, Until::Materialize);
        let observed = host::run(case).unwrap();
        assert_eq!(observed.operations, ["validate_query"]);
        let (result, code) = observed.answer.outcome();
        assert_eq!(result, case.expect.outcome, "{}", case.id);
        assert_eq!(code, case.expect.code, "{}", case.id);
        answers.push(observed.answer);
    }
    assert!(answers.contains(&Answer::Validated(true)));
    assert!(answers.contains(&Answer::Validated(false)));
}

/// The accepted S0 case as JSON, for a test to change.
fn accepted() -> serde_json::Value {
    let path = cases_dir().join("s0/s0.accept.sum-window.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// A scratch directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("sdm-testkit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// Loads `case` written to a file: the loader's error, if any.
fn load_error(name: &str, case: &serde_json::Value) -> Option<String> {
    let directory = scratch(name);
    let path = directory.join("case.json");
    std::fs::write(&path, serde_json::to_vec(case).unwrap()).unwrap();
    let error = case::load(&[path]).err().map(|error| error.to_string());
    std::fs::remove_dir_all(&directory).unwrap();
    error
}

#[test]
fn the_accepted_case_loads_unchanged() {
    assert_eq!(load_error("unchanged", &accepted()), None);
}

#[test]
fn an_unknown_key_names_the_file_and_position_and_never_the_key() {
    let mut case = accepted();
    case["secret-canary"] = serde_json::json!(1);
    let error = load_error("top", &case).expect("refused");
    assert!(
        error.contains("case.json") && error.contains("line 1"),
        "{error}"
    );
    assert!(!error.contains("secret-canary"), "{error}");
}

/// The same key `contract/tests` shows the Python loader refusing.
#[test]
fn an_unknown_key_inside_the_call_arguments_is_refused() {
    let mut case = accepted();
    case["input"]["materialize"]["ignored_typo"] = serde_json::json!(true);
    assert!(load_error("nested-materialize", &case).is_some());
    let mut case = accepted();
    case["input"]["from_source"] = serde_json::json!({"ignored_typo": {}});
    assert!(load_error("nested-from-source", &case).is_some());
}

#[test]
fn a_failed_run_leaves_no_output_an_earlier_run_wrote() {
    let directory = scratch("stale");
    let mut case = accepted();
    case.as_object_mut().unwrap().remove("core");
    let cases = directory.join("case.json");
    std::fs::write(&cases, serde_json::to_vec(&case).unwrap()).unwrap();
    let out = directory.join("rust-host.jsonl");
    std::fs::write(&out, "an earlier run's observations\n").unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .arg("--out")
        .arg(&out)
        .arg(&cases)
        .output()
        .unwrap();
    let left = out.exists();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("runner failure"));
    assert!(!left, "the earlier run's output survived");
}

#[test]
fn the_host_records_the_inputs_it_encoded() {
    let mut loaded = case::load(&[cases_dir()]).unwrap().remove(0).case;
    // A profile spelled with an explicit empty capabilities map: the host
    // encodes it without one, and records what it encoded.
    let spec = loaded
        .core
        .as_mut()
        .unwrap()
        .validate_query
        .as_mut()
        .unwrap();
    spec.profile["capabilities"] = serde_json::json!({});
    let observed = host::run(&loaded).unwrap();
    assert!(observed.inputs[0]["profile"].get("capabilities").is_none());
}

#[test]
fn a_case_without_a_core_operation_is_this_hosts_failure() {
    let mut case = case::load(&[cases_dir()]).unwrap().remove(0).case;
    case.core = None;
    assert!(host::run(&case).is_err());
}

/// `contract/tests/index_literals.json`, which the Python loader's tests
/// read too.
fn index_literals() -> serde_json::Value {
    let path = cases_dir().join("../tests/index_literals.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The accepted case with `literal` as its one index, written as is.
fn with_index(name: &str, literal: &str) -> Result<case::Index, String> {
    let mut case = accepted();
    case["input"]["materialize"]["indices"] = serde_json::json!(["@index@"]);
    let text = serde_json::to_string(&case)
        .unwrap()
        .replace("\"@index@\"", literal);
    let directory = scratch(name);
    let path = directory.join("case.json");
    std::fs::write(&path, text).unwrap();
    let loaded = case::load(&[path]);
    std::fs::remove_dir_all(&directory).unwrap();
    loaded
        .map(|mut loaded| loaded.remove(0).case.input.materialize.indices.remove(0))
        .map_err(|error| error.to_string())
}

#[test]
fn an_index_is_an_integer_literal_of_any_size_or_text() {
    let literals = index_literals();
    for literal in literals["accepted"].as_array().unwrap() {
        let literal = literal.as_str().unwrap();
        let index = with_index("index-accepted", literal).unwrap();
        let expected = match serde_json::from_str::<serde_json::Value>(literal) {
            Ok(serde_json::Value::String(text)) => case::Index::Text(text),
            _ => case::Index::Integer(literal.to_string()),
        };
        assert_eq!(index, expected, "{literal}");
    }
    for literal in literals["refused"].as_array().unwrap() {
        let literal = literal.as_str().unwrap();
        assert!(with_index("index-refused", literal).is_err(), "{literal}");
    }
}
