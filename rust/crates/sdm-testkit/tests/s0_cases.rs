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

#[test]
fn a_case_names_its_file_and_position_when_it_does_not_load() {
    let directory = std::env::temp_dir().join(format!("sdm-testkit-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("broken.json");
    std::fs::write(
        &path,
        br#"{"format": "sdm-case/1", "id": "x", "secret-canary": 1}"#,
    )
    .unwrap();
    let error = case::load(&[path]).unwrap_err().to_string();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(
        error.contains("broken.json") && error.contains("line 1"),
        "{error}"
    );
    assert!(!error.contains("secret-canary"), "{error}");
}

#[test]
fn a_case_without_a_core_operation_is_this_hosts_failure() {
    let mut case = case::load(&[cases_dir()]).unwrap().remove(0).case;
    case.core = None;
    assert!(host::run(&case).is_err());
}
