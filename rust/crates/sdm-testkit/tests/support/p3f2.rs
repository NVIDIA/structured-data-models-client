// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only delegation to the existing owner reader and shared FK assertions.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

pub fn problems(
    root: &Path,
    observed: &BTreeMap<String, Value>,
    reference: &BTreeMap<String, Value>,
) -> BTreeMap<String, Option<String>> {
    let python = std::env::var("SDM_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let mut child = Command::new(python)
        .arg("-I")
        .arg(root.join("contract/tests/_p3f2_payload.py"))
        .arg("--rust-pairs")
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("P3F-2 comparison starts with the existing test Python");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            &serde_json::to_vec(&json!({"observed": observed, "reference": reference})).unwrap(),
        )
        .expect("complete observed/reference bodies reach the comparison");
    let done = child.wait_with_output().unwrap();
    assert!(
        done.status.success(),
        "P3F-2 comparison failed: {}",
        String::from_utf8_lossy(&done.stderr)
    );
    serde_json::from_slice(&done.stdout).expect("P3F-2 comparison returns complete keyed problems")
}
