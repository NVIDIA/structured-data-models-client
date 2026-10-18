// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.5's and T2.6's exits on `rust-host`'s providers (the package's test
//! author's): the provider conformance kit and the in-memory fallback.
//! `contract/tests/test_t25_provider_kit.py` holds the Python host's
//! providers to the same exits, over the same data.
//!
//! The kit's data is `contract/fixtures/provider-kit/`: `dataset.sql`, the
//! one fixture dataset, and `expected.json`, the cases and their expected
//! answers (some hand-checked, marked `checked: hand`), whose `about` says
//! how a case runs (its `page_rows`, `seeds`, `capabilities` and `inject`)
//! and is judged, and whose `faults` say what each seeded fault does.
//! `record.py` beside them recomputes every answer from the dataset.
//!
//! The interface assumed, the kit's runner on this host:
//!
//! ```text
//! sdm-contract-run --kit DATABASE [--fault truncate|silent_capability] --out REPORT
//! ```
//!
//! It builds the dataset on DATABASE (DuckDB from `dataset.sql`; SQLite
//! loaded from it by the one contract loader, as `--database sqlite` loads
//! a case), runs each case through the core on this host's provider for
//! that database (`sdm_testkit::duck`, `sdm_testkit::sqlite`), with the
//! fault named seeded in the provider path, and writes REPORT in the format
//! the Python file's docstring gives, `host` `rust-host`. It exits 0 when
//! every case passes its own judgment and 1 when any case fails. This file
//! judges the report itself too, by `expected.json`'s rules.
//!
//! Postgres is not run here: this host's Postgres is a replay of the
//! Python provider's recorded answers (`sdm_testkit::replay`, KO2-20), no
//! provider path of its own, and the Python file runs the kit on the Python
//! host's Postgres provider.
//!
//! Each test's line says what it proves:
//!
//! - T2.5, "every provider landed by then passes": the kit passes on
//!   DuckDB and on SQLite (paging, limits, the expected answers, capability
//!   failure and retryability);
//! - T2.6, "kit cases with the capability undeclared give the pushdown's
//!   answer": each `same_as` case answers as its pushdown case does, the
//!   provider never asked the request withheld;
//! - T2.6, "a limit exceeded is a named error": `UNSUPPORTED_REQUEST`,
//!   rule `fallback_limit`, naming the request and the limit, where the
//!   same request within its limits answers;
//! - T2.5, the seeded faults "in each host's provider path": a truncation
//!   (each case the fault names answered short, no error) and a silent
//!   capability failure (each neighbors case answered empty, no error,
//!   every other case passing) each fail the kit, which exits 1.
//!
//! A control passing before the implementation holds every case's request
//! to the contract's own types: each parses as a `DataRequest` and passes
//! `DataRequest::check`, each withheld name is an `OptionalRequest` and
//! each fallback a `FallbackLimits`.
//!
//! Gating: `contract/tests/t25-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each must still fail
//! (strict); once one passes it fails, naming itself, until it is taken
//! out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use sdm_contract::ColumnType;
use sdm_contract::data::{
    AttachmentId, DataRequest, FallbackLimits, KeySet, OptionalRequest, field,
};
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t25-pending.json";
const FORMAT: &str = "sdm-provider-kit/1";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn pending(test: &str) -> bool {
    read_json(&root().join(PENDING))["pending"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!("{test} passes now: take rust:{test} out of {PENDING}'s pending"),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let gating = read_json(&root().join(PENDING));
    let names = |key: &str| -> BTreeSet<String> {
        gating[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap().to_owned())
            .collect()
    };
    assert!(names("pending").is_subset(&names("authored")));
}

// --- the cases, resolved ----------------------------------------------------

fn expected() -> &'static Value {
    static EXPECTED: OnceLock<Value> = OnceLock::new();
    EXPECTED.get_or_init(|| read_json(&root().join("contract/fixtures/provider-kit/expected.json")))
}

fn cases() -> &'static [Value] {
    expected()["cases"].as_array().unwrap()
}

fn case(id: &str) -> &'static Value {
    cases()
        .iter()
        .find(|c| c["id"] == id)
        .unwrap_or_else(|| panic!("no case {id}"))
}

fn page_rows() -> usize {
    expected()["page_rows"].as_u64().unwrap() as usize
}

/// The case whose request and seeds `case` runs.
fn request_case(case_: &'static Value) -> &'static Value {
    match case_
        .get("same_as")
        .or_else(|| case_.get("request_of"))
        .and_then(Value::as_str)
    {
        Some(named) => case(named),
        None => case_,
    }
}

/// The answer `case` expects, its own or its `same_as` case's.
fn wanted_answer(case_: &'static Value) -> Option<&'static Value> {
    match case_.get("same_as").and_then(Value::as_str) {
        Some(named) => case(named).get("answer"),
        None => case_.get("answer"),
    }
}

fn wanted_rows(answer: &Value) -> usize {
    match answer.get("any_of") {
        Some(_) => answer["count"].as_u64().unwrap() as usize,
        None => answer["rows"].as_array().unwrap().len(),
    }
}

fn withheld(case_: &Value) -> Vec<String> {
    case_
        .pointer("/capabilities/withhold")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .map(|n| n.as_str().unwrap().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn fault_fails(name: &str) -> Vec<String> {
    expected()["faults"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap_or_else(|| panic!("no fault {name}"))["fails"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap().to_owned())
        .collect()
}

// --- the judgment -------------------------------------------------------------

/// A value as compared: a number by its value, anything else as it is.
fn value(v: &Value) -> Value {
    match v.as_f64() {
        Some(number) if v.is_number() => json!(number),
        _ => v.clone(),
    }
}

fn row_key(row: &[Value]) -> String {
    serde_json::to_string(&row.iter().map(value).collect::<Vec<_>>()).unwrap()
}

/// `got`'s rows, the listed columns in order, or why not.
fn projected(got: &Value, columns: &[Value]) -> Result<Vec<Vec<Value>>, String> {
    let names: Vec<&str> = got["columns"]
        .as_array()
        .map(|c| c.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut at = Vec::new();
    for column in columns {
        let column = column.as_str().unwrap();
        match names.iter().position(|n| *n == column) {
            Some(i) => at.push(i),
            None => return Err(format!("no column {column} among {names:?}")),
        }
    }
    let rows = got["rows"].as_array().cloned().unwrap_or_default();
    rows.iter()
        .map(|row| {
            let row = row.as_array().ok_or("a row that is not a list")?;
            at.iter()
                .map(|i| {
                    row.get(*i)
                        .map(value)
                        .ok_or_else(|| "a short row".to_owned())
                })
                .collect()
        })
        .collect()
}

/// How `got` differs from the answer `wanted`; `None` when it does not.
fn answer_problem(got: &Value, wanted: &Value) -> Option<String> {
    let rows = match projected(got, wanted["columns"].as_array().unwrap()) {
        Ok(rows) => rows,
        Err(why) => return Some(why),
    };
    let lists = |key: &str| -> Vec<Vec<Value>> {
        wanted[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(value).collect())
            .collect()
    };
    if wanted.get("any_of").is_some() {
        let count = wanted["count"].as_u64().unwrap() as usize;
        if rows.len() != count {
            return Some(format!("{} rows, not {count}", rows.len()));
        }
        let mut pool: Vec<String> = lists("any_of").iter().map(|r| row_key(r)).collect();
        for row in &rows {
            match pool.iter().position(|k| *k == row_key(row)) {
                Some(i) => {
                    pool.remove(i);
                }
                None => return Some(format!("{row:?} is not one of the rows it may be")),
            }
        }
        return None;
    }
    let want = lists("rows");
    if wanted["ordered"].as_bool().unwrap() {
        if rows != want {
            return Some(format!("rows {rows:?}, not {want:?} in order"));
        }
    } else {
        let sorted = |r: &[Vec<Value>]| {
            let mut keys: Vec<String> = r.iter().map(|row| row_key(row)).collect();
            keys.sort();
            keys
        };
        if sorted(&rows) != sorted(&want) {
            return Some(format!("rows {rows:?}, not the multiset {want:?}"));
        }
    }
    None
}

/// An error field's value as text: the value itself, or the value of the
/// contract's scalar as it serializes (`{"t": ..., "v": ...}`).
fn text(detail: Option<&Value>) -> String {
    match detail {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(scalar)) if scalar.contains_key("v") => text(scalar.get("v")),
        Some(other) => other.to_string(),
        None => "None".to_owned(),
    }
}

fn error_problem(got: &Value, wanted: &Value) -> Option<String> {
    for name in ["code", "kind", "retryable"] {
        if got.get(name) != wanted.get(name) {
            return Some(format!("{name} {:?}, not {}", got.get(name), wanted[name]));
        }
    }
    for (name, value) in wanted["fields"].as_object().unwrap() {
        let seen = text(got.pointer(&format!("/fields/{name}")));
        if seen != value.as_str().unwrap() {
            return Some(format!("field {name} {seen}, not {value}"));
        }
    }
    None
}

/// Why the report's `seen` does not pass `case`; empty when it does.
fn judge(case_: &'static Value, seen: Option<&Value>) -> Vec<String> {
    let Some(seen) = seen else {
        return vec!["not reported".to_owned()];
    };
    let fetched: Vec<&str> = seen["fetched"]
        .as_array()
        .map(|f| f.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if let Some(wanted) = wanted_answer(case_) {
        let Some(got) = seen.get("answer").filter(|_| seen.get("error").is_none()) else {
            return vec![format!("an error, {}, not an answer", seen["error"])];
        };
        if let Some(problem) = answer_problem(got, wanted) {
            return vec![problem];
        }
        let name = request_case(case_)["request"]["request"].as_str().unwrap();
        let held_back = withheld(case_);
        if !held_back.is_empty() {
            // The fallback answered: the provider was asked only what it
            // declares, and something.
            if fetched.is_empty() || fetched.iter().any(|f| held_back.iter().any(|h| h == f)) {
                return vec![format!(
                    "fetched {fetched:?}: the fallback asks none of {held_back:?}, and asks something"
                )];
            }
        } else {
            let pages = wanted_rows(wanted).div_ceil(page_rows());
            if fetched.iter().filter(|f| **f == name).count() < pages {
                return vec![format!(
                    "fetched {fetched:?}: {} rows come in {pages} pages of {} or more",
                    wanted_rows(wanted),
                    page_rows()
                )];
            }
        }
        return vec![];
    }
    let Some(got) = seen.get("error").filter(|_| seen.get("answer").is_none()) else {
        return vec![format!("an answer, not the error {}", case_["error"])];
    };
    if let Some(problem) = error_problem(got, &case_["error"]) {
        return vec![problem];
    }
    if let Some(page) = case_.pointer("/inject/page").and_then(Value::as_u64)
        && (fetched.len() as u64) < page
    {
        return vec![format!("fetched {fetched:?}: page {page} was never asked")];
    }
    vec![]
}

// --- the kit's runs -----------------------------------------------------------

type Ran = Result<(i32, Value), String>;

/// The kit's runner on `database`, with `fault` seeded: its exit status
/// and report, or why there is none. Each run is made once per process.
fn kit(database: &str, fault: Option<&str>) -> Ran {
    static RUNS: OnceLock<Mutex<BTreeMap<String, Ran>>> = OnceLock::new();
    let name = format!("{database}-{}", fault.unwrap_or("none"));
    let mut runs = RUNS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    runs.entry(name.clone())
        .or_insert_with(|| run_kit(database, fault, &name))
        .clone()
}

fn run_kit(database: &str, fault: Option<&str>, name: &str) -> Ran {
    let dir = std::env::temp_dir().join(format!("t25-provider-kit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let out = dir.join(format!("{name}.json"));
    let mut arguments = vec!["--kit", database];
    if let Some(fault) = fault {
        arguments.extend(["--fault", fault]);
    }
    let out_text = out.to_str().unwrap().to_owned();
    arguments.extend(["--out", &out_text]);
    let done = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .args(&arguments)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| e.to_string())?;
    let report: Value = match std::fs::read_to_string(&out) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| e.to_string())?,
        Err(_) => {
            return Err(format!(
                "sdm-contract-run --kit {database} wrote no report ({}): {}{}",
                done.status,
                String::from_utf8_lossy(&done.stdout),
                String::from_utf8_lossy(&done.stderr)
            ));
        }
    };
    std::fs::remove_file(&out).ok();
    for (field, want) in [
        ("format", json!(FORMAT)),
        ("host", json!("rust-host")),
        ("database", json!(database)),
        ("fault", json!(fault)),
    ] {
        if report.get(field) != Some(&want) {
            return Err(format!(
                "the report's {field} is {:?}, not {want}",
                report.get(field)
            ));
        }
    }
    Ok((done.status.code().unwrap_or(-1), report))
}

fn reported(report: &Value) -> BTreeMap<String, Value> {
    report["cases"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|c| (c["id"].as_str().unwrap_or_default().to_owned(), c))
        .collect()
}

/// Each case's problems in `report`, by case.
fn judged(report: &Value) -> BTreeMap<String, Vec<String>> {
    let seen = reported(report);
    cases()
        .iter()
        .map(|c| {
            let id = c["id"].as_str().unwrap().to_owned();
            let problems = judge(c, seen.get(&id));
            (id, problems)
        })
        .collect()
}

fn failing(
    problems: &BTreeMap<String, Vec<String>>,
    ids: &[&str],
) -> BTreeMap<String, Vec<String>> {
    ids.iter()
        .filter(|id| !problems[**id].is_empty())
        .map(|id| ((*id).to_owned(), problems[*id].clone()))
        .collect()
}

// --- control ------------------------------------------------------------------

/// Control: every case's request is the contract's own, `seeds` filled as
/// the runner fills them, and passes the request's own check; each
/// withheld name is an optional request, and each fallback the contract's
/// limits.
#[test]
fn every_kit_request_is_a_contract_request() {
    let seeds = KeySet {
        attachment: AttachmentId(1),
        columns: vec![
            field("key", ColumnType::Int64, false),
            field("batch", ColumnType::Int64, false),
        ],
    };
    let mut seen = 0;
    for case_ in cases() {
        if let Some(request) = case_.get("request") {
            let mut request = request.clone();
            if case_.get("seeds").is_some() {
                request["seeds"] = serde_json::to_value(&seeds).unwrap();
            }
            let parsed: DataRequest = serde_json::from_value(request)
                .unwrap_or_else(|e| panic!("{}: not a DataRequest: {e}", case_["id"]));
            parsed
                .check()
                .unwrap_or_else(|e| panic!("{}: refused: {e:?}", case_["id"]));
            seen += 1;
        }
        for name in withheld(case_) {
            serde_json::from_value::<OptionalRequest>(json!(name))
                .unwrap_or_else(|e| panic!("{}: {name}: {e}", case_["id"]));
        }
        if let Some(limits) = case_.pointer("/capabilities/fallback") {
            serde_json::from_value::<FallbackLimits>(limits.clone())
                .unwrap_or_else(|e| panic!("{}: {e}", case_["id"]));
        }
    }
    assert!(seen >= 10, "the kit runs {seen} requests of its own");
}

// --- T2.5: the kit on each provider ---------------------------------------------

fn passes(database: &str) {
    let (status, report) = kit(database, None).unwrap();
    let problems: BTreeMap<_, _> = judged(&report)
        .into_iter()
        .filter(|(_, p)| !p.is_empty())
        .collect();
    assert!(problems.is_empty(), "{problems:#?}");
    let ids: BTreeSet<String> = cases()
        .iter()
        .map(|c| c["id"].as_str().unwrap().to_owned())
        .collect();
    let seen: BTreeSet<String> = reported(&report).into_keys().collect();
    assert_eq!(seen, ids);
    assert_eq!(status, 0, "the kit exited {status}");
}

#[test]
fn the_kit_passes_on_duckdb() {
    gated("the_kit_passes_on_duckdb", || passes("duckdb"));
}

#[test]
fn the_kit_passes_on_sqlite() {
    gated("the_kit_passes_on_sqlite", || passes("sqlite"));
}

// --- T2.6: the in-memory fallback -----------------------------------------------

fn fallback_answers(database: &str) {
    let (_, report) = kit(database, None).unwrap();
    let problems = judged(&report);
    let fallback: Vec<&str> = cases()
        .iter()
        .filter(|c| c.get("same_as").is_some())
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert!(!fallback.is_empty());
    assert!(
        failing(&problems, &fallback).is_empty(),
        "{:#?}",
        failing(&problems, &fallback)
    );
    let pushdown: Vec<&str> = fallback
        .iter()
        .map(|id| case(id)["same_as"].as_str().unwrap())
        .collect();
    assert!(
        failing(&problems, &pushdown).is_empty(),
        "{:#?}",
        failing(&problems, &pushdown)
    );
}

#[test]
fn the_fallback_gives_the_pushdown_s_answer_on_duckdb() {
    gated("the_fallback_gives_the_pushdown_s_answer_on_duckdb", || {
        fallback_answers("duckdb")
    });
}

#[test]
fn the_fallback_gives_the_pushdown_s_answer_on_sqlite() {
    gated("the_fallback_gives_the_pushdown_s_answer_on_sqlite", || {
        fallback_answers("sqlite")
    });
}

fn limits_named(database: &str) {
    let (_, report) = kit(database, None).unwrap();
    let problems = judged(&report);
    let named = [
        "fallback.rows-exceeded",
        "fallback.bytes-exceeded",
        "fallback.neighbors-ordered",
    ];
    assert!(
        failing(&problems, &named).is_empty(),
        "{:#?}",
        failing(&problems, &named)
    );
}

#[test]
fn a_fallback_limit_exceeded_is_a_named_error_on_duckdb() {
    gated(
        "a_fallback_limit_exceeded_is_a_named_error_on_duckdb",
        || limits_named("duckdb"),
    );
}

#[test]
fn a_fallback_limit_exceeded_is_a_named_error_on_sqlite() {
    gated(
        "a_fallback_limit_exceeded_is_a_named_error_on_sqlite",
        || limits_named("sqlite"),
    );
}

// --- T2.5: the seeded faults ----------------------------------------------------

fn truncation_fails(database: &str) {
    let (status, report) = kit(database, Some("truncate")).unwrap();
    let seen = reported(&report);
    let problems = judged(&report);
    for name in fault_fails("truncate") {
        assert!(!problems[&name].is_empty(), "{name} passes truncated");
        let got = &seen[&name];
        assert!(got.get("error").is_none(), "{name}: {got}");
        let rows = got["answer"]["rows"].as_array().unwrap().len();
        let wanted = wanted_rows(&case(&name)["answer"]);
        assert!(
            rows < wanted,
            "{name}: {rows} rows, truncated from {wanted}"
        );
    }
    assert_eq!(status, 1, "the kit exited {status}, not 1");
}

#[test]
fn a_seeded_truncation_fails_the_kit_on_duckdb() {
    gated("a_seeded_truncation_fails_the_kit_on_duckdb", || {
        truncation_fails("duckdb")
    });
}

#[test]
fn a_seeded_truncation_fails_the_kit_on_sqlite() {
    gated("a_seeded_truncation_fails_the_kit_on_sqlite", || {
        truncation_fails("sqlite")
    });
}

fn silent_capability_fails(database: &str) {
    let (status, report) = kit(database, Some("silent_capability")).unwrap();
    let seen = reported(&report);
    let problems = judged(&report);
    let fails = fault_fails("silent_capability");
    for name in &fails {
        assert!(!problems[name].is_empty(), "{name} passes with no rows");
        let got = &seen[name];
        assert!(got.get("error").is_none(), "{name}: {got}");
        assert!(
            got["answer"]["rows"].as_array().unwrap().is_empty(),
            "{name}: {got}"
        );
        assert!(
            got["fetched"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "neighbors"),
            "{name}: {got}"
        );
    }
    let others: BTreeMap<_, _> = problems
        .into_iter()
        .filter(|(id, p)| !p.is_empty() && !fails.contains(id))
        .collect();
    assert!(others.is_empty(), "{others:#?}");
    assert_eq!(status, 1, "the kit exited {status}, not 1");
}

#[test]
fn a_seeded_silent_capability_failure_fails_the_kit_on_duckdb() {
    gated(
        "a_seeded_silent_capability_failure_fails_the_kit_on_duckdb",
        || silent_capability_fails("duckdb"),
    );
}

#[test]
fn a_seeded_silent_capability_failure_fails_the_kit_on_sqlite() {
    gated(
        "a_seeded_silent_capability_failure_fails_the_kit_on_sqlite",
        || silent_capability_fails("sqlite"),
    );
}
