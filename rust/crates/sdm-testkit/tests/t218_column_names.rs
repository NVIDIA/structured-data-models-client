// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.18's exits on `rust-host` (the package's test author's): any column
//! name (KO2-36). `contract/tests/test_t218_column_names.py` holds
//! `python-core` to the same exits and to the reference's vectors.
//!
//! The family is twelve of the fifteen `name.*` S1 cases: a column named
//! `a.b`, one named with a backtick and one with a line feed, each as a
//! feature the query does not name, the target, the entity (users'
//! declared primary key) and an aggregation's column. The filter role's
//! three (`name.*-filter`) need the core's `target_filter` rule lifted,
//! which is T2.10's: `tests/t210_targets.rs` holds them. Each test's line
//! says what it proves:
//!
//! - step 1, "the core keeps every column as data": on DuckDB and SQLite,
//!   the description `describe_source` answers for each case holds the
//!   column under its own name;
//! - "cases written for all three databases" and "the differences from
//!   `main` registered": on DuckDB and SQLite, this host answers each case
//!   through `predict`, as the reference answers the same data and query
//!   with the column named `a_b` (`contract/column-name-vectors.json`'s
//!   twins), and an approved register entry citing no temporary approval
//!   names each key whose outcome is not python-ref's committed line's.
//!   Postgres is held on this host by CI's `cross-host (postgres)` job,
//!   which replays python-core's fetches.
//!
//! The SQLite runs load each case through the contract loader, a Python
//! with DuckDB and pyarrow (`SDM_PYTHON`, else `python3`), as T2.2's do.
//!
//! Gating: the tests named in `contract/tests/t218-pending.json`'s
//! `pending` (as `rust:<test name>`) failed when written, and must still
//! fail (strict); the implementer removes each as it passes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn pending(test: &str) -> bool {
    let text = std::fs::read_to_string(root().join("contract/tests/t218-pending.json")).unwrap();
    let gating: Value = serde_json::from_str(&text).unwrap();
    gating["pending"]
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
             contract/tests/t218-pending.json's pending"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

/// The column each case names, by the case ID's word for it.
fn column_of(case_id: &str) -> (&'static str, &'static str) {
    let name = if case_id.starts_with("name.dot-") {
        "a.b"
    } else if case_id.starts_with("name.backtick-") {
        "a`b"
    } else if case_id.starts_with("name.line-break-") {
        "a\nb"
    } else {
        panic!("{case_id}: not a name case")
    };
    let table = if case_id.ends_with("-aggregation") {
        "orders"
    } else {
        "users"
    };
    (table, name)
}

fn family() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| c.id.starts_with("name.") && !c.id.ends_with("-filter"))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        cases.len(),
        12,
        "three names in four roles, T2.10's filter left out"
    );
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host on `database`, through `predict` against
/// the scripted endpoint: its key and what it observed, or why it could
/// not run.
fn observed(database: &str) -> Vec<(Case, String, Result<Observed, String>)> {
    let (cases, bases) = family();
    cases
        .into_iter()
        .map(|case| {
            let (key, seen) = match database {
                "duckdb" => (case.id.clone(), host::run(&case, &bases)),
                "sqlite" => (
                    format!("{}@sqlite", case.id),
                    host::run_sqlite(&case, &bases, &root()),
                ),
                other => panic!("no {other} run here"),
            };
            (case, key, seen.map_err(|e| e.to_string()))
        })
        .collect()
}

/// The python-ref observations, by key.
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

/// The decision rows that approve only until a package lands, as
/// `scripts/rust_core/register.py`'s `TEMPORARY` names them.
fn temporary() -> Vec<String> {
    let text = std::fs::read_to_string(root().join("scripts/rust_core/register.py")).unwrap();
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix("TEMPORARY = ("))
        .expect("register.py names TEMPORARY");
    line.trim_end_matches(')')
        .split(',')
        .map(|r| r.trim().trim_matches('\'').to_owned())
        .filter(|r| !r.is_empty())
        .collect()
}

/// The register's umbrellas: approved as E92's Phase 2 class, they excuse
/// no case (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry, no umbrella and citing no temporary approval, whose
/// text names `wanted` (a line key in its `cases` or `case_outcomes`):
/// each entry read from the register's text, block by block.
fn registered(wanted: &str) -> Option<String> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let temporary = temporary();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let approved = block.lines().any(|l| l.trim_end() == "  status: approved");
        let row = block
            .lines()
            .find(|l| l.starts_with("  decision_ref:"))
            .and_then(|l| l.split("row: ").nth(1))
            .map(|r| r.trim_end_matches('}').trim().to_owned());
        let names = block.split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | '[' | ']' | '{' | '}' | '\'' | '"')
        });
        if approved
            && !UMBRELLAS.contains(&id.as_str())
            && !row.is_some_and(|r| temporary.contains(&r))
            && names.into_iter().any(|n| n == wanted)
        {
            return Some(id);
        }
    }
    None
}

/// Whether `described` holds `column` in `table`.
fn describes(described: &Value, table: &str, column: &str) -> bool {
    described["tables"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["name"] == table)
        .flat_map(|t| t["columns"].as_array().into_iter().flatten())
        .any(|c| c["name"] == column)
}

fn keeps_every_column(database: &str) {
    let mut problems = Vec::new();
    for (case, key, seen) in observed(database) {
        let (table, column) = column_of(&case.id);
        match seen {
            Ok(seen) => match &seen.described {
                Some(described) if describes(described, table, column) => {}
                Some(_) => problems.push(format!("{key}: {table} leaves out {column:?}")),
                None => problems.push(format!("{key}: nothing described ({:?})", seen.answer)),
            },
            Err(error) => problems.push(format!("{key}: {error}")),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

fn answers_through_predict(database: &str) {
    let reference = reference();
    let mut problems = Vec::new();
    for (_, key, seen) in observed(database) {
        let seen = match seen {
            Ok(seen) => seen,
            Err(error) => {
                problems.push(format!("{key}: {error}"));
                continue;
            }
        };
        if !matches!(seen.answer, Answer::Predicted(_)) {
            problems.push(format!("{key}: {:?}", seen.answer));
            continue;
        }
        let theirs = &reference[&key]["outcome"];
        if theirs["result"] != "success" && registered(&key).is_none() {
            problems.push(format!(
                "{key}: answered where python-ref is {theirs}, and no approved, permanent \
                 entry names it"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn rust_host_keeps_every_column_whatever_its_name_on_duckdb() {
    gated(
        "rust_host_keeps_every_column_whatever_its_name_on_duckdb",
        || keeps_every_column("duckdb"),
    );
}

#[test]
fn rust_host_keeps_every_column_whatever_its_name_on_sqlite() {
    gated(
        "rust_host_keeps_every_column_whatever_its_name_on_sqlite",
        || keeps_every_column("sqlite"),
    );
}

#[test]
fn rust_host_answers_each_name_case_through_predict_on_duckdb() {
    gated(
        "rust_host_answers_each_name_case_through_predict_on_duckdb",
        || answers_through_predict("duckdb"),
    );
}

#[test]
fn rust_host_answers_each_name_case_through_predict_on_sqlite() {
    gated(
        "rust_host_answers_each_name_case_through_predict_on_sqlite",
        || answers_through_predict("sqlite"),
    );
}
