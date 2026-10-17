// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32b's SQL exits for the decimal family on `rust-host`, held from
//! outside the implementation (the package's independent test author).
//! Each `decimal.*` case of `contract/cases/s1` runs through this host,
//! its statements executed on DuckDB, through `predict` against the
//! scripted endpoint:
//!
//! - Exit "the DuckDB SQL goldens for the family's own statements, executed
//!   on DuckDB in CI, as extensions of `sdm-data`'s `tests/duckdb_catalog.rs`
//!   text pins" (E53(c), T1.30 folded in), and "a seeded SQL change without
//!   a golden update and an entry fails": every statement the family runs
//!   (`from_source` and `materialize`) must be pinned, verbatim, as a
//!   string literal in `rust/crates/sdm-data/tests/*.rs`; a statement whose
//!   text changed without its pin changing is unpinned and fails. The check
//!   is shown to fail on a seeded change (in the statement over the seeds of
//!   the wide decimal feature's case, and in the decimal target's first
//!   describe statement), on no pins and on no statements.
//! - Exit "statement counts equal the reference's or are registered,
//!   cross-checked against DuckDB's query log": per public call, the
//!   number of statements this host's provider ran equals the `python-ref`
//!   observation's (`contract/reference/<commit>.jsonl`), unless an
//!   approved register entry other than P1B-1 lists the case in its
//!   `cases` (a proposed one excuses nothing); and the provider's record is
//!   complete, its count equal to DuckDB's query log with no logged
//!   statement differing.
//! - Exit "`python-core` and `rust-host` agree on `core` fields for the
//!   family", this host's half: each case's outcome is the `python-ref`
//!   observation's (`success`, or the refusal's code), or an approved entry
//!   other than P1B-1 lists the case. The family is T1.32b's three cases
//!   and T2.0c's twelve exact-decimal ones; where the core answers past the
//!   reference (`decimal.sum-past-38`), the approved entry listing the case
//!   (P2B-1) is what lets it differ. `contract/tests/test_t132b_decimal_family.py`
//!   holds `python-core` the same way, and `tests/t132b_decimal_vectors.rs`
//!   this host over the family's vectors.
//!
//! Gating: the tests named in `contract/tests/t132b-pending.json`'s
//! `pending["exits"]` (as `rust:<test name>`) failed when written, and
//! must still fail (strict); the implementer removes each as it passes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::Value;

const STAGES: [&str; 2] = ["from_source", "materialize"];
const FAMILY: [&str; 1] = ["decimal."];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn pending(test: &str) -> bool {
    let text = std::fs::read_to_string(root().join("contract/tests/t132b-pending.json")).unwrap();
    let gating: Value = serde_json::from_str(&text).unwrap();
    gating["pending"]["exits"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t132b-pending.json's pending[\"exits\"]"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

fn family_cases() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| FAMILY.iter().any(|p| c.id.starts_with(p)))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    // T1.32b's three cases and T2.0c's twelve (exact decimals, KO2-50).
    assert_eq!(
        cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        [
            "decimal.class-target-38-18",
            "decimal.class-target-collide",
            "decimal.compare-finer-eq",
            "decimal.compare-finer-gt",
            "decimal.compare-finer-negative",
            "decimal.compare-past-precision",
            "decimal.compare-past-precision-gt",
            "decimal.compare-same-scale",
            "decimal.feature-38-0-wide",
            "decimal.feature-inexact-38-18",
            "decimal.key-wide",
            "decimal.sum-10-2",
            "decimal.sum-past-38",
            "decimal.target-38-18",
            "decimal.target-regression-collide",
        ],
        "the family's fifteen cases"
    );
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host, through `predict` against the scripted
/// endpoint, in process.
fn observed() -> Vec<(Case, Observed)> {
    let (cases, bases) = family_cases();
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
    for statement in driver["capture"]["statements"].as_array().unwrap() {
        out.entry(statement["stage"].as_str().unwrap().to_owned())
            .or_default()
            .push(statement["sql"].as_str().unwrap().to_owned());
    }
    out
}

/// The python-ref observations, by case.
fn reference() -> BTreeMap<String, Value> {
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    text.lines()
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
        })
        .collect()
}

/// A list field of each approved register entry (a proposed one excuses
/// nothing), read from the YAML's text entry by entry, in its flow or
/// block form.
fn register_lists(field: &str) -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let head = format!("  {field}:");
    let mut out = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim();
        if status != "approved" {
            continue;
        }
        let lines: Vec<&str> = block.lines().collect();
        let Some(at) = lines.iter().position(|l| l.starts_with(&head)) else {
            continue;
        };
        let mut items = Vec::new();
        let rest = lines[at][head.len()..].trim();
        if let Some(flow) = rest.strip_prefix('[') {
            let mut flow = flow.to_owned();
            let mut next = at + 1;
            while !flow.contains(']') && next < lines.len() {
                flow.push(' ');
                flow.push_str(lines[next].trim());
                next += 1;
            }
            let inner = flow.split(']').next().unwrap_or_default();
            items.extend(inner.split(',').map(|r| r.trim().to_owned()));
        } else {
            for line in &lines[at + 1..] {
                match line.strip_prefix("  - ") {
                    Some(item) => items.push(item.trim().to_owned()),
                    None => break,
                }
            }
        }
        let items = items
            .into_iter()
            .map(|i| i.trim_matches(|c| c == '\'' || c == '"').to_owned())
            .filter(|i| !i.is_empty())
            .collect();
        out.push((id, items));
    }
    out
}

/// The umbrellas, the placeholder P1B-1 among them: approved by the owner as
/// E92's Phase 2 class, they excuse no case, even one they list (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry that lists the case in its `cases`.
/// P1B-1 does not count: it is the inventory's placeholder, whose own notes
/// say each case whose outcome changes gets an entry of its own.
fn registered_case(id: &str) -> Option<String> {
    register_lists("cases")
        .into_iter()
        .find(|(entry, cases)| {
            !UMBRELLAS.contains(&entry.as_str()) && cases.iter().any(|c| c == id)
        })
        .map(|(entry, _)| entry)
}

// --- the goldens --------------------------------------------------------------

/// Every string literal of a Rust source, its escapes read: `"..."` (with
/// `\n`, `\t`, `\\`, `\"`, `\'`, `\0` and a line continuation) and raw
/// `r"..."`, `r#"..."#`; comments and character literals skipped.
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

/// A seeded change: the statement of a case and stage holding a text, and
/// how its text is changed.
type Seed = (&'static str, &'static str, fn(&str) -> String);

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails() {
    // The check itself, on seeded evidence, whatever the goldens hold.
    let ran = family_statements();
    assert!(ran.len() > 3 * 10, "{} statements", ran.len());
    let pinned: BTreeSet<String> = ran.iter().map(|(_, s)| s.clone()).collect();
    assert!(unpinned(&ran, &pinned).is_empty());
    // One statement's text changed, its pin not: it fails; the statement
    // over the seeds of the wide decimal feature's case (a word added), and
    // the first the decimal target's description runs (a trailing space:
    // pins are verbatim).
    let changes: [Seed; 2] = [
        (
            "decimal.feature-38-0-wide materialize",
            "__sdm_seeds__",
            |sql| sql.replacen("SELECT ", "SELECT DISTINCT ", 1),
        ),
        ("decimal.target-38-18 from_source", "", |sql| {
            format!("{sql} ")
        }),
    ];
    for (case, needle, change) in changes {
        let at = ran
            .iter()
            .position(|(id, sql)| id == case && sql.contains(needle))
            .unwrap_or_else(|| panic!("{case}: a statement holding {needle:?}"));
        let mut seeded = ran.clone();
        seeded[at].1 = change(&seeded[at].1);
        assert_ne!(seeded[at].1, ran[at].1, "{case}");
        assert_eq!(unpinned(&seeded, &pinned).len(), 1, "{case}");
    }
    // No pins, or no statements, fail closed.
    assert_eq!(unpinned(&ran, &BTreeSet::new()).len(), ran.len());
    assert_eq!(unpinned(&[], &pinned).len(), 1);
}

#[test]
fn every_statement_the_family_runs_is_pinned_in_sdm_data_s_tests() {
    gated(
        "every_statement_the_family_runs_is_pinned_in_sdm_data_s_tests",
        || {
            let pinned = pins(&root().join("rust/crates/sdm-data/tests"));
            let ran = family_statements();
            let missing: BTreeSet<String> = unpinned(&ran, &pinned)
                .into_iter()
                .map(|m| m.split_once(": ").map_or(m.clone(), |(_, s)| s.to_owned()))
                .collect();
            assert!(
                missing.is_empty(),
                "{} statements of the decimal family have no golden pin in \
                 rust/crates/sdm-data/tests:\n{}",
                missing.len(),
                missing.into_iter().collect::<Vec<_>>().join("\n---\n")
            );
        },
    );
}

#[test]
fn statement_counts_equal_the_reference_s_or_are_registered() {
    gated(
        "statement_counts_equal_the_reference_s_or_are_registered",
        || {
            let reference = reference();
            let mut problems = Vec::new();
            for (case, seen) in observed() {
                // Complete: the provider's record against DuckDB's log.
                let driver = seen.driver.as_ref().unwrap();
                let recorded = driver["capture"]["statements"].as_array().unwrap().len();
                let logged = &driver["query_log"];
                if logged["statements"].as_u64() != Some(recorded as u64)
                    || logged["differing"].as_array().is_none_or(|d| !d.is_empty())
                {
                    problems.push(format!("{}: the record is not DuckDB's log", case.id));
                }
                let ours = statements(&seen);
                let theirs = &reference[&case.id]["driver"]["capture"]["statements"];
                for stage in STAGES {
                    let mine = ours.get(stage).map_or(0, Vec::len);
                    let want = theirs
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|s| s["stage"] == stage)
                        .count();
                    if mine != want && registered_case(&case.id).is_none() {
                        problems.push(format!(
                            "{} {stage}: {mine} statements, the reference {want}",
                            case.id
                        ));
                    }
                }
            }
            assert!(problems.is_empty(), "{}", problems.join("\n"));
        },
    );
}

#[test]
fn rust_host_answers_each_case_as_python_ref() {
    gated("rust_host_answers_each_case_as_python_ref", || {
        let reference = reference();
        let mut problems = Vec::new();
        for (case, seen) in observed() {
            let theirs = &reference[&case.id]["outcome"];
            let (result, code) = match &seen.answer {
                Answer::Predicted(_) => ("success", None),
                Answer::Error { code, .. } => ("refusal", Some(code.clone())),
                // A query refused first (a case carrying validate_query in
                // place of materialize): the refusal INVALID_REQUEST, as
                // the route maps it.
                Answer::Validated(false) => ("refusal", Some("INVALID_REQUEST".to_owned())),
                other => {
                    problems.push(format!("{}: no answer of predict: {other:?}", case.id));
                    continue;
                }
            };
            let want_result = theirs["result"].as_str().unwrap();
            let want_code = theirs["code"].as_str().map(str::to_owned);
            if (result != want_result || code != want_code) && registered_case(&case.id).is_none() {
                problems.push(format!(
                    "{}: {result} {code:?}, python-ref {want_result} {want_code:?}",
                    case.id
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}
