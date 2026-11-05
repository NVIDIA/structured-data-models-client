// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Case-format acceptance and rejection using shared malformed-input vectors.

use std::path::PathBuf;

use sdm_testkit::case;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(relative: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join(relative)).unwrap()).unwrap()
}

/// Assert all case-format checks, reporting every mismatch.
fn check_cases(name: &str, check: impl FnOnce() -> Vec<String>) {
    let problems = check();
    assert!(problems.is_empty(), "{name}:\n{}", problems.join("\n"));
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
    check_cases(
        "the_rust_loader_takes_and_refuses_the_warehouse_vectors",
        || {
            let vectors = read_json("contract/tests/case-format-vectors.json");
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
    check_cases("the_rust_loader_refuses_the_texts_json_cannot_hold", || {
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
