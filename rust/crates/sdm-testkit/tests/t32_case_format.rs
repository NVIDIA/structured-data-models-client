// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.2's case-format exits on the Rust loader (the package's test
//! author's), beside `tests/test_t32_case_format.py`, which holds the
//! Python loader to the same vectors:
//!
//! - `the_rust_loader_takes_and_refuses_the_warehouse_vectors`: every
//!   change of `contract/tests/t32-case-format-vectors.json` (`warehouses`,
//!   `input.view`, `input.connect`, `num_context_samples`; design choice 1
//!   and the dry run's gaps) is taken or refused by `case::load` as the
//!   file lists it;
//! - `the_rust_loader_refuses_the_texts_json_cannot_hold`:
//!   `contract/tests/malformed-cases.json`'s `refused_texts` (P2-T2.1i-1: a
//!   `fail_call` of `-0`, a `rows_per_call` of NaN, Infinity or -Infinity)
//!   are refused; the Rust loader refused them before T3.2, and this holds
//!   it there while the Python loader is aligned.
//!
//! Gating: `contract/tests/t32-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each pending test
//! must still fail (strict); once one passes it fails, naming itself,
//! until it is taken out.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::case;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(relative: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join(relative)).unwrap()).unwrap()
}

/// The `rust:` names of the pending file's `list`.
fn named(list: &str) -> BTreeSet<String> {
    read_json("contract/tests/t32-pending.json")[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .map(str::to_owned)
        .collect()
}

/// Runs `check` and holds it to the pending file: a pending test must
/// fail, any other must pass.
fn gated(name: &str, check: impl FnOnce() -> Vec<String>) {
    assert!(named("authored").contains(name), "{name} is not authored");
    let problems = check();
    if named("pending").contains(name) {
        assert!(
            !problems.is_empty(),
            "{name} passes now: take rust:{name} out of contract/tests/t32-pending.json pending"
        );
    } else {
        assert!(problems.is_empty(), "{name}:\n{}", problems.join("\n"));
    }
}

/// A scratch directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("sdm-testkit-t32-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// `case::load`'s error for one case file holding `text`, if any.
fn load_text_error(name: &str, text: &str) -> Option<String> {
    let directory = scratch(name);
    let path = directory.join("case.json");
    std::fs::write(&path, text).unwrap();
    let error = case::load(&[path]).err().map(|error| format!("{error:#}"));
    std::fs::remove_dir_all(&directory).unwrap();
    error
}

/// `case` with one vector's `set` items applied, then its `remove` paths.
fn changed(case: &Value, change: &Value) -> Value {
    let mut out = case.clone();
    for item in change["set"].as_array().unwrap() {
        let path = item["path"].as_array().unwrap();
        let mut at = &mut out;
        for key in &path[..path.len() - 1] {
            at = match key {
                Value::String(name) => &mut at[name.as_str()],
                _ => &mut at[key.as_u64().unwrap() as usize],
            };
        }
        let last = path[path.len() - 1].as_str().unwrap();
        at.as_object_mut()
            .unwrap()
            .insert(last.to_owned(), item["value"].clone());
    }
    for path in change
        .get("remove")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let path = path.as_array().unwrap();
        let mut at = &mut out;
        for key in &path[..path.len() - 1] {
            at = &mut at[key.as_str().unwrap()];
        }
        at.as_object_mut()
            .unwrap()
            .remove(path[path.len() - 1].as_str().unwrap());
    }
    out
}

#[test]
fn the_rust_loader_takes_and_refuses_the_warehouse_vectors() {
    gated(
        "the_rust_loader_takes_and_refuses_the_warehouse_vectors",
        || {
            let vectors = read_json("contract/tests/t32-case-format-vectors.json");
            let case = &vectors["case"];
            let mut problems = Vec::new();
            let accepted = vectors["accepted"].as_array().unwrap();
            let refused = vectors["refused"].as_array().unwrap();
            assert!(accepted.len() >= 20 && refused.len() >= 20);
            for change in accepted {
                let text = serde_json::to_string(&changed(case, change)).unwrap();
                if let Some(error) = load_text_error("accepted", &text) {
                    problems.push(format!("accepted {}: refused: {error}", change["why"]));
                }
            }
            for change in refused {
                let text = serde_json::to_string(&changed(case, change)).unwrap();
                if load_text_error("refused", &text).is_none() {
                    problems.push(format!("refused {}: taken", change["why"]));
                }
            }
            problems
        },
    );
}

#[test]
fn the_rust_loader_refuses_the_texts_json_cannot_hold() {
    gated("the_rust_loader_refuses_the_texts_json_cannot_hold", || {
        const PLACEHOLDER: &str = "\u{1}t32-the-text\u{1}";
        let listed = read_json("contract/tests/malformed-cases.json");
        let case: Value = read_json(listed["case"].as_str().unwrap());
        assert_eq!(
            load_text_error("adventureworks", &serde_json::to_string(&case).unwrap()),
            None
        );
        let texts = listed["refused_texts"].as_array().unwrap();
        assert_eq!(texts.len(), 4);
        let mut problems = Vec::new();
        for change in texts {
            let mut out = case.clone();
            let path = change["path"].as_array().unwrap();
            let mut at = &mut out;
            for key in path {
                at = &mut at[key.as_str().unwrap()];
            }
            *at = Value::String(PLACEHOLDER.to_owned());
            let placeholder = serde_json::to_string(PLACEHOLDER).unwrap();
            let text = serde_json::to_string(&out)
                .unwrap()
                .replace(&placeholder, change["text"].as_str().unwrap());
            if load_text_error("text", &text).is_none() {
                problems.push(format!("{}: taken", change["why"]));
            }
        }
        problems
    });
}
