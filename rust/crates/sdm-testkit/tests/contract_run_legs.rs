// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm-contract-run` runs a case only where the case runs (T3.3): a
//! warehouse-only S2 case (its `warehouses`) is selected out of a DuckDB
//! run, never run on DuckDB, while an S0 case beside it still runs.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn a_warehouse_only_case_is_not_run_on_duckdb() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let scratch = std::env::temp_dir().join(format!("sdm-legs-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let s0 = root.join("contract/cases/s0/s0.accept.binary-classification.json");
    let s2 = scratch.join("s2.json");
    std::fs::write(&s2, include_str!("contract-run-legs-s2.json")).unwrap();
    let out = scratch.join("out.jsonl");
    let done = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .current_dir(&root)
        .args([
            "--out".as_ref(),
            out.as_os_str(),
            s0.as_os_str(),
            s2.as_os_str(),
        ])
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let lines = std::fs::read_to_string(&out).unwrap();
    let cases: Vec<String> = lines
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["case"].to_string())
        .collect();
    assert_eq!(cases, ["\"s0.accept.binary-classification\""], "{cases:?}");
    // Alone, it is no case of a DuckDB run: named, never run.
    let alone = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .current_dir(&root)
        .args(["--out".as_ref(), out.as_os_str(), s2.as_os_str()])
        .output()
        .unwrap();
    assert!(!alone.status.success());
    assert!(
        String::from_utf8_lossy(&alone.stderr).contains("no case runs on the database asked for")
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
