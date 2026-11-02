// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executed SQL for this type family must remain covered by native goldens.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Observed};

const STAGES: [&str; 2] = ["from_source", "materialize"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn nested_cases() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| c.id.starts_with("nested."))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(cases.len(), 5, "the nested family's five cases");
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host, through `predict` against the scripted
/// endpoint, in process.
fn observed() -> Vec<(Case, Observed)> {
    let (cases, bases) = nested_cases();
    cases
        .into_iter()
        .map(|case| {
            let seen = host::run(&case, &bases).unwrap_or_else(|e| panic!("{}: {e}", case.id));
            (case, seen)
        })
        .collect()
}

/// The statements the provider ran, by public call, in order.
fn statements(seen: &Observed) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let driver = seen.driver.as_ref().expect("a driver record");
    // The reads plan B adds taken out (register P3A-1).
    for statement in &sdm_testkit::plan_b::as_the_old_client_ran(
        driver["capture"]["statements"].as_array().unwrap(),
    ) {
        out.entry(statement["stage"].as_str().unwrap().to_owned())
            .or_default()
            .push(statement["sql"].as_str().unwrap().to_owned());
    }
    out
}

/// Read Rust literals used as exact SQL expectations.
fn literals(source: &str) -> Vec<String> {
    let s: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            '/' if s.get(i + 1) == Some(&'/') => {
                while i < s.len() && s[i] != '\n' {
                    i += 1;
                }
            }
            '\'' => {
                if s.get(i + 2) == Some(&'\'') {
                    i += 3;
                } else if s.get(i + 1) == Some(&'\\') {
                    let close = (i + 2..(i + 12).min(s.len())).find(|j| s[*j] == '\'');
                    i = close.map_or(i + 1, |j| j + 1);
                } else {
                    i += 1;
                }
            }
            'r' if (s.get(i + 1) == Some(&'"') || s.get(i + 1) == Some(&'#'))
                && (i == 0 || !(s[i - 1].is_alphanumeric() || s[i - 1] == '_')) =>
            {
                let mut j = i + 1;
                let mut hashes = 0;
                while s.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if s.get(j) != Some(&'"') {
                    i += 1;
                    continue;
                }
                let close: String = std::iter::once('"')
                    .chain(std::iter::repeat_n('#', hashes))
                    .collect();
                let body_start = j + 1;
                let rest: String = s[body_start..].iter().collect();
                let end = rest.find(&close).expect("a closed raw string");
                out.push(rest[..end].to_owned());
                i = body_start + rest[..end].chars().count() + close.chars().count();
            }
            '"' => {
                let mut text = String::new();
                i += 1;
                while i < s.len() && s[i] != '"' {
                    if s[i] == '\\' {
                        i += 1;
                        match s[i] {
                            'n' => text.push('\n'),
                            't' => text.push('\t'),
                            'r' => text.push('\r'),
                            '0' => text.push('\0'),
                            '\n' => {
                                while i + 1 < s.len() && s[i + 1].is_whitespace() {
                                    i += 1;
                                }
                            }
                            other => text.push(other),
                        }
                    } else {
                        text.push(s[i]);
                    }
                    i += 1;
                }
                out.push(text);
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}

/// The text pins: every string literal of `sdm-data`'s tests.
fn pins(dir: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.extend(literals(&std::fs::read_to_string(&path).unwrap()));
        }
    }
    out
}

/// The statements not pinned; every one is when the evidence is empty.
fn unpinned(ran: &[(String, String)], pinned: &BTreeSet<String>) -> Vec<String> {
    if ran.is_empty() {
        return vec!["no statement ran: no evidence".to_owned()];
    }
    ran.iter()
        .filter(|(_, sql)| !pinned.contains(sql))
        .map(|(id, sql)| format!("{id}: {sql}"))
        .collect()
}

fn family_statements() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (case, seen) in observed() {
        for (stage, texts) in statements(&seen) {
            if STAGES.contains(&stage.as_str()) {
                out.extend(
                    texts
                        .into_iter()
                        .map(|t| (format!("{} {stage}", case.id), t)),
                );
            }
        }
    }
    out
}

#[test]
fn the_literal_reader_reads_rust_s_strings() {
    let source = "let a = \"x\\n\\\"y\\\"\"; // \"not a pin\"\n\
                  let b = r#\"raw \"q\"\"#; let c = '\"'; let d = \"long \\\n    line\";";
    assert_eq!(
        literals(source),
        vec!["x\n\"y\"", "raw \"q\"", "long line"],
        "{source}"
    );
    // duckdb_catalog.rs's own pins read back as the statements they pin.
    let pinned = pins(&root().join("rust/crates/sdm-data/tests"));
    assert!(pinned.contains(
        "SELECT \"order_id\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000"
    ));
    assert!(pinned.iter().any(|p| p.starts_with(
        "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()"
    )));
}

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails() {
    // The check itself, on seeded evidence, whatever the goldens hold.
    let ran = family_statements();
    assert!(ran.len() > 5 * 6, "{} statements", ran.len());
    let pinned: BTreeSet<String> = ran.iter().map(|(_, s)| s.clone()).collect();
    assert!(unpinned(&ran, &pinned).is_empty());
    // One statement's text changed, its pin not: it fails.
    let mut seeded = ran.clone();
    let at = ran
        .iter()
        .position(|(_, sql)| sql.contains("__sdm_seeds__"))
        .expect("a statement over the seeds");
    seeded[at].1 = seeded[at].1.replacen("SELECT ", "SELECT DISTINCT ", 1);
    assert_ne!(seeded[at].1, ran[at].1);
    assert_eq!(unpinned(&seeded, &pinned).len(), 1);
    // No pins, or no statements, fail closed.
    assert_eq!(unpinned(&ran, &BTreeSet::new()).len(), ran.len());
    assert_eq!(unpinned(&[], &pinned).len(), 1);
}

#[test]
fn every_statement_the_nested_family_runs_is_pinned_in_sdm_data_s_tests() {
    let pinned = pins(&root().join("rust/crates/sdm-data/tests"));
    let ran = family_statements();
    let missing: BTreeSet<String> = unpinned(&ran, &pinned)
        .into_iter()
        .map(|m| m.split_once(": ").map_or(m.clone(), |(_, s)| s.to_owned()))
        .collect();
    assert!(
        missing.is_empty(),
        "{} statements of the nested family have no golden pin in \
                 rust/crates/sdm-data/tests:\n{}",
        missing.len(),
        missing.into_iter().collect::<Vec<_>>().join("\n---\n")
    );
}
