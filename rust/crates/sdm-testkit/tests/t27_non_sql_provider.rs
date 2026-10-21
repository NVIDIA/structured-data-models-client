// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.7's exits (the package's test author's): a minimal non-SQL test
//! provider and a second `DataLayer` composed at the composition root.
//!
//! The interface assumed, on `sdm-contract-run`:
//!
//! ```text
//! sdm-contract-run --kit memory [--fault truncate|silent_capability] --out REPORT
//! sdm-contract-run --database memory [--layer default] --out FILE CASE...
//! ```
//!
//! `memory` names the non-SQL provider: each table read whole into Arrow
//! in memory (built by DuckDB from the kit's `dataset.sql`, or from a
//! case's base and setup), and every typed data request answered from
//! those tables, never through SQL. `--kit memory` runs the provider
//! conformance kit (T2.5) on it, as `--kit duckdb` does on DuckDB's
//! provider, the report's `database` `memory`. `--database memory` runs
//! each case on the swapped data layer: a second `DataLayer`, not
//! `sdm_data::Data`, composed into its own `CompositionRoot` and handed to
//! `HostRun::start`, with the non-SQL provider answering every fetch; each
//! observation keyed `<case>@memory`, as `--database sqlite` keys
//! `<case>@sqlite`. With `--layer default`, the same provider answers the
//! same case through the default composition root's layer
//! (`sdm_host::default_root`, `sdm-data`'s), which serves host providers
//! too: the one change between the two runs is the data layer.
//!
//! Each test's line says what it proves:
//!
//! - "the provider passes the kit, including two schema tables over one
//!   source": every kit case, judged here by `expected.json`'s rules,
//!   passes on `memory`, `key-counts.one-source-twice` among them (two
//!   keys over the one relation `orders`, each counted alone);
//! - the slice gate's seeded faults (design, check 5) fail the kit on the
//!   non-SQL provider too: a truncated page and a capability that fails
//!   silently each fail the cases `expected.json`'s `faults` name;
//! - "the same prediction runs on the swapped layer": S1's `base.binary`
//!   (`from_source`, `materialize`, `predict`) with the non-SQL provider
//!   gives, on the swapped layer, the whole result it gives on the default
//!   layer (outcome, description, the data it asked for, materialized
//!   request and every model call body); and against DuckDB's SQL
//!   provider, everything but the in-context examples' rows, which each
//!   provider draws by its own seeded sample (DuckDB's reservoir shuffles
//!   even a table it keeps whole; design, "Data access": a provider's own
//!   sampling is repeatable only for that provider), every fetch on the
//!   swapped layer planned as the typed request, no statement;
//! - "a guard shows no domain crate changed for it": no commit of T2.7's
//!   (subject `T2.7: `, `T2.7 fix <n>: `, `T2.7 tests: `, or its
//!   landing) changes a file of a crate `scripts/rust_core/l0_deps.py`
//!   classes as domain; a control shows the guard catches such an edit.
//!
//! The kit case `key-counts.one-source-twice` also runs on every SQL
//! provider through T2.5's tests on both hosts, which it passes as written.
//!
//! Gating: `contract/tests/t27-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each must still fail
//! (strict); once one passes it fails, naming itself, until it is taken
//! out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t27-pending.json";
const MEMORY: &str = "memory";
/// The S1 case the swapped layer runs: `from_source`, `materialize` and a
/// binary `predict`, on the shop base.
const PREDICTION: &str = "contract/cases/s1/base.binary.json";
/// The kit case of two schema tables over one source.
const ONE_SOURCE_TWICE: &str = "key-counts.one-source-twice";

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

/// `sdm-contract-run` with `arguments`, from the repository's root.
fn contract_run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .args(arguments)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// A directory of this test process's own, `name` apart.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("t27-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn said(done: &Output) -> String {
    format!(
        "{}: {}{}",
        done.status,
        String::from_utf8_lossy(&done.stdout),
        String::from_utf8_lossy(&done.stderr)
    )
}

// --- the kit's cases, judged by expected.json's rules ---------------------------

fn expected() -> Value {
    read_json(&root().join("contract/fixtures/provider-kit/expected.json"))
}

fn by_id(kit: &Value) -> BTreeMap<String, Value> {
    kit["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["id"].as_str().unwrap().to_owned(), c.clone()))
        .collect()
}

/// A value as compared: a number by its value, a zero of either sign one
/// value, anything else as it is.
fn value(v: &Value) -> Value {
    match v.as_f64() {
        Some(number) if v.is_number() => json!(number + 0.0),
        _ => v.clone(),
    }
}

fn row_key(row: &[Value]) -> String {
    serde_json::to_string(&row.iter().map(value).collect::<Vec<_>>()).unwrap()
}

fn rows(of: &Value) -> Vec<Vec<Value>> {
    of.as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(value).collect())
        .collect()
}

/// How `got` differs from the answer `wanted`; `None` when it does not.
fn answer_problem(got: &Value, wanted: &Value) -> Option<String> {
    let names: Vec<&str> = got["columns"]
        .as_array()
        .map(|c| c.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut at = Vec::new();
    for column in wanted["columns"].as_array().unwrap() {
        match names.iter().position(|n| Some(*n) == column.as_str()) {
            Some(i) => at.push(i),
            None => return Some(format!("no column {column} among {names:?}")),
        }
    }
    let Some(got_rows) = got["rows"].as_array() else {
        return Some("no rows".to_owned());
    };
    let mut have = Vec::new();
    for row in got_rows {
        let row = row.as_array().map(Vec::as_slice).unwrap_or_default();
        if at.iter().any(|&i| i >= row.len()) {
            return Some(format!("a short row {row:?}"));
        }
        have.push(row_key(
            &at.iter().map(|&i| row[i].clone()).collect::<Vec<_>>(),
        ));
    }
    if let Some(any_of) = wanted.get("any_of") {
        let count = wanted["count"].as_u64().unwrap() as usize;
        if have.len() != count {
            return Some(format!("{} rows, not {count}", have.len()));
        }
        let mut pool: Vec<String> = rows(any_of).iter().map(|r| row_key(r)).collect();
        for row in &have {
            match pool.iter().position(|k| k == row) {
                Some(i) => {
                    pool.remove(i);
                }
                None => return Some(format!("{row} is not one of the rows it may be")),
            }
        }
        return None;
    }
    let mut want: Vec<String> = rows(&wanted["rows"]).iter().map(|r| row_key(r)).collect();
    if wanted["ordered"] != json!(true) {
        have.sort();
        want.sort();
    }
    (have != want).then(|| format!("rows {have:?}, not {want:?}"))
}

/// An error field's value as text: the value, or a scalar's value.
fn text(detail: Option<&Value>) -> String {
    match detail {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(scalar)) if scalar.contains_key("v") => text(scalar.get("v")),
        Some(other) => other.to_string(),
        None => "None".to_owned(),
    }
}

/// Why the report's `seen` does not pass `case`; `None` when it does.
fn judge(case: &Value, seen: Option<&Value>, cases: &BTreeMap<String, Value>) -> Option<String> {
    let Some(seen) = seen else {
        return Some("not reported".to_owned());
    };
    let wanted = match case.get("same_as").and_then(Value::as_str) {
        Some(named) => cases[named].get("answer"),
        None => case.get("answer"),
    };
    if let Some(wanted) = wanted {
        let Some(got) = seen.get("answer").filter(|_| seen.get("error").is_none()) else {
            return Some(format!("an error, {}, not an answer", seen["error"]));
        };
        if let Some(problem) = answer_problem(got, wanted) {
            return Some(problem);
        }
    } else {
        let Some(got) = seen.get("error").filter(|_| seen.get("answer").is_none()) else {
            return Some(format!("an answer, not the error {}", case["error"]));
        };
        let error = &case["error"];
        for name in ["code", "kind", "retryable"] {
            if got.get(name) != error.get(name) {
                return Some(format!("{name} {:?}, not {}", got.get(name), error[name]));
            }
        }
        for (name, want) in error["fields"].as_object().unwrap() {
            let have = text(got.pointer(&format!("/fields/{name}")));
            if have != text(Some(want)) {
                return Some(format!("field {name} {have}, not {want}"));
            }
        }
    }
    if let Some(exact) = case.get("fetched")
        && seen.get("fetched") != Some(exact)
    {
        return Some(format!("fetched {}, not exactly {exact}", seen["fetched"]));
    }
    None
}

/// `--kit memory`, with `fault` seeded: its exit status and report, the
/// report's own header held to the run asked for.
fn kit_on_memory(fault: Option<&str>) -> (i32, Value) {
    let dir = scratch("kit");
    let out = dir.join(format!("{}.json", fault.unwrap_or("none")));
    let out_text = out.to_str().unwrap().to_owned();
    let mut arguments = vec!["--kit", MEMORY];
    if let Some(fault) = fault {
        arguments.extend(["--fault", fault]);
    }
    arguments.extend(["--out", &out_text]);
    let done = contract_run(&arguments);
    let report: Value = match std::fs::read_to_string(&out) {
        Ok(text) => serde_json::from_str(&text).unwrap(),
        Err(_) => panic!("--kit {MEMORY} wrote no report ({})", said(&done)),
    };
    std::fs::remove_file(&out).ok();
    for (field, want) in [
        ("format", json!("sdm-provider-kit/1")),
        ("host", json!("rust-host")),
        ("database", json!(MEMORY)),
        ("fault", json!(fault)),
    ] {
        assert_eq!(report.get(field), Some(&want), "the report's {field}");
    }
    (done.status.code().unwrap_or(-1), report)
}

/// Each case's problem in `report`, by case; a passing case maps to `None`.
fn judged(report: &Value) -> BTreeMap<String, Option<String>> {
    let kit = expected();
    let cases = by_id(&kit);
    let seen: BTreeMap<String, Value> = report["cases"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|c| (c["id"].as_str().unwrap_or_default().to_owned(), c))
        .collect();
    assert_eq!(
        seen.keys().collect::<BTreeSet<_>>(),
        cases.keys().collect::<BTreeSet<_>>(),
        "the report holds every kit case, once"
    );
    cases
        .iter()
        .map(|(id, case)| (id.clone(), judge(case, seen.get(id), &cases)))
        .collect()
}

/// Control: the kit holds the case of two schema tables over one source,
/// a key_counts over two equal relations whose keys count differently, so
/// a provider that merges them, or keys what it holds by relation, fails.
#[test]
fn the_kit_holds_two_schema_tables_over_one_source() {
    let kit = expected();
    let case = &by_id(&kit)[ONE_SOURCE_TWICE];
    let keys = case["request"]["keys"].as_array().unwrap();
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0]["table"], keys[1]["table"], "one source");
    assert_ne!(keys[0]["columns"], keys[1]["columns"]);
    let counts = rows(&case["answer"]["rows"]);
    assert_eq!(counts.len(), 2);
    assert_ne!(counts[0][1..], counts[1][1..], "the keys count differently");
}

/// T2.7: the non-SQL provider passes the kit, including two schema tables
/// over one source.
#[test]
fn the_kit_passes_on_the_non_sql_provider() {
    gated("the_kit_passes_on_the_non_sql_provider", || {
        let (status, report) = kit_on_memory(None);
        let problems: BTreeMap<_, _> = judged(&report)
            .into_iter()
            .filter_map(|(id, p)| p.map(|p| (id, p)))
            .collect();
        assert!(problems.is_empty(), "{problems:#?}");
        assert_eq!(status, 0, "the kit exited {status}");
    });
}

/// T2.7, the slice gate's seeded faults on the non-SQL provider: each
/// fails every case `expected.json`'s `faults` name for it, each answered
/// with no error (a silenced capability fails no other case), and the kit
/// exits 1.
#[test]
fn a_seeded_fault_fails_the_kit_on_the_non_sql_provider() {
    gated(
        "a_seeded_fault_fails_the_kit_on_the_non_sql_provider",
        || {
            let kit = expected();
            for fault in kit["faults"].as_array().unwrap() {
                let name = fault["name"].as_str().unwrap();
                let fails: BTreeSet<String> = fault["fails"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_str().unwrap().to_owned())
                    .collect();
                let (status, report) = kit_on_memory(Some(name));
                let failed: BTreeSet<String> = judged(&report)
                    .into_iter()
                    .filter_map(|(id, p)| p.map(|_| id))
                    .collect();
                assert!(
                    fails.is_subset(&failed),
                    "{name}: fails {failed:?}, not every one of {fails:?}"
                );
                if name == "silent_capability" {
                    // Only neighbors is silenced: every other case passes.
                    assert_eq!(failed, fails, "{name}: the cases it fails");
                }
                for case in report["cases"].as_array().unwrap() {
                    if fails.contains(case["id"].as_str().unwrap()) {
                        assert!(case.get("error").is_none(), "{name}: {case}");
                    }
                }
                assert_eq!(status, 1, "{name}: the kit exited {status}, not 1");
            }
        },
    );
}

// --- the same prediction on the swapped layer -----------------------------------

/// The one observation `sdm-contract-run` writes for `arguments`.
fn observed(arguments: &[&str], name: &str) -> Value {
    let dir = scratch("predict");
    let out = dir.join(format!("{name}.jsonl"));
    let out_text = out.to_str().unwrap().to_owned();
    let mut all: Vec<&str> = arguments.to_vec();
    all.extend(["--out", &out_text, PREDICTION]);
    let done = contract_run(&all);
    let Ok(text) = std::fs::read_to_string(&out) else {
        panic!(
            "sdm-contract-run {arguments:?} wrote no observation ({})",
            said(&done)
        );
    };
    std::fs::remove_file(&out).ok();
    let lines: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 1, "{text}");
    lines.into_iter().next().unwrap()
}

/// What two runs of one case must hold equal, whatever their provider:
/// the calls made, the outcome, the description, and the core's error.
const SAME: &[&str] = &[
    "/identity/operations",
    "/outcome",
    "/core/accepted",
    "/core/error",
    "/core/described",
];

/// `value` with each model request body's in-context examples (a body:
/// an object holding `model`, `task` and `context`) cut to their shape:
/// each table's place, format, columns and count of rows, without the
/// rows, which a provider's own seeded sample draws and orders.
fn examples_shaped(value: &Value) -> Value {
    match value {
        Value::Object(fields) => {
            let body = fields.contains_key("model") && fields.contains_key("task");
            Value::Object(
                fields
                    .iter()
                    .map(|(key, field)| {
                        let field = if body && key == "context" {
                            shaped(field)
                        } else {
                            examples_shaped(field)
                        };
                        (key.clone(), field)
                    })
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(examples_shaped).collect()),
        other => other.clone(),
    }
}

/// `context` with each table's `rows` replaced by their count.
fn shaped(context: &Value) -> Value {
    match context {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, field)| match (key.as_str(), field) {
                    ("rows", Value::Array(rows)) => (key.clone(), json!(rows.len())),
                    _ => (key.clone(), shaped(field)),
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The swapped run's fetches: each planned as the typed request, no
/// statement; and their requests.
fn host_fetches(run: &Value, which: &str) -> Vec<Value> {
    let fetches = run["core"]["fetches"].as_array().unwrap();
    assert!(!fetches.is_empty(), "{which}: the run reads its data");
    for fetch in fetches {
        assert_eq!(
            fetch["plan"]["plan"], "host",
            "{which}: a fetch planned as SQL: {fetch}"
        );
    }
    fetches.iter().map(|f| f["request"].clone()).collect()
}

/// T2.7: the same prediction runs on the swapped layer. With the non-SQL
/// provider, the swapped layer's run is the default layer's, whole; and
/// it is DuckDB's but for the in-context examples' rows, every fetch
/// planned as the typed request.
#[test]
fn the_same_prediction_runs_on_the_swapped_layer() {
    gated("the_same_prediction_runs_on_the_swapped_layer", || {
        let duckdb = observed(&[], "duckdb");
        assert_eq!(duckdb["case"], "base.binary");
        assert_eq!(
            duckdb["outcome"]["result"], "success",
            "{}",
            duckdb["outcome"]
        );
        assert!(duckdb["core"]["predicted"].is_object(), "DuckDB predicts");
        let swapped = observed(&["--database", MEMORY], MEMORY);
        assert_eq!(swapped["case"], "base.binary@memory");
        let default = observed(&["--database", MEMORY, "--layer", "default"], "default");
        assert_eq!(default["case"], "base.binary@memory");

        // The whole result: only the data layer changed.
        for field in SAME {
            assert_eq!(
                swapped.pointer(field),
                default.pointer(field),
                "layers: {field}"
            );
        }
        assert_eq!(
            host_fetches(&swapped, "swapped"),
            host_fetches(&default, "default"),
            "layers: the data the core asked for"
        );
        assert_eq!(
            swapped["core"]["materialized"], default["core"]["materialized"],
            "layers: the materialized request"
        );
        assert_eq!(
            swapped["core"]["predicted"], default["core"]["predicted"],
            "layers: every model call body"
        );

        // Against DuckDB: all but the examples' rows.
        for field in SAME {
            assert_eq!(
                swapped.pointer(field),
                duckdb.pointer(field),
                "DuckDB: {field}"
            );
        }
        assert!(
            swapped["core"]["predicted"]["body"]["context"].is_object(),
            "the call holds in-context examples"
        );
        assert_eq!(
            examples_shaped(&swapped["core"]["materialized"]),
            examples_shaped(&duckdb["core"]["materialized"]),
            "DuckDB: the materialized request"
        );
        assert_eq!(
            examples_shaped(&swapped["core"]["predicted"]),
            examples_shaped(&duckdb["core"]["predicted"]),
            "DuckDB: every model call body"
        );
    });
}

/// Control: cutting the examples to their shape keeps everything else of
/// a body, and a body's other fields, a count of rows, or the examples'
/// columns still differ.
#[test]
fn only_the_examples_rows_are_left_out() {
    let body = |rows: Value, columns: Value, target: &str| {
        json!({"payload": {"model": "m", "task": {"target": target},
            "context": {"instance_table": {"format": "arrays", "columns": columns, "rows": rows}},
            "predict": {"rows": [[1]]}}})
    };
    let one = body(json!([[1, 2], [3, 4]]), json!(["a", "b"]), "t");
    let shuffled = body(json!([[3, 4], [1, 2]]), json!(["a", "b"]), "t");
    assert_eq!(examples_shaped(&one), examples_shaped(&shuffled));
    for other in [
        body(json!([[1, 2]]), json!(["a", "b"]), "t"),
        body(json!([[1, 2], [3, 4]]), json!(["a", "c"]), "t"),
        body(json!([[1, 2], [3, 4]]), json!(["a", "b"]), "u"),
    ] {
        assert_ne!(examples_shaped(&one), examples_shaped(&other), "{other}");
    }
    assert_eq!(
        examples_shaped(&one)["payload"]["predict"],
        json!({"rows": [[1]]}),
        "rows outside the examples are kept"
    );
}

// --- the guard: no domain crate changed for T2.7 ---------------------------------

/// The crates `scripts/rust_core/l0_deps.py` classes as domain, read from
/// its `CLASSES` table (`'<crate>': DOMAIN,`).
fn domain_crates() -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join("scripts/rust_core/l0_deps.py")).unwrap();
    let start = text
        .find("CLASSES: dict[str, str] = {")
        .expect("l0_deps.py's CLASSES");
    let table = &text[start..start + text[start..].find('}').unwrap()];
    table
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_suffix(": DOMAIN,")
                .map(|name| name.trim_matches('\'').to_owned())
        })
        .collect()
}

/// The files of `changed` inside a domain crate's directory.
fn in_domain_crates<'a>(changed: &'a [String], domain: &BTreeSet<String>) -> Vec<&'a String> {
    changed
        .iter()
        .filter(|path| {
            domain
                .iter()
                .any(|name| path.starts_with(&format!("rust/crates/{name}/")))
        })
        .collect()
}

fn git(arguments: &[&str]) -> String {
    let done = Command::new("git")
        .args(arguments)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(done.status.success(), "git {arguments:?}: {}", said(&done));
    String::from_utf8(done.stdout).unwrap()
}

/// Whether `subject` is one of T2.7's commits: its implementation, a fix
/// round, its tests, its tracker, or its landing.
fn is_t27(subject: &str) -> bool {
    let subject = subject.strip_prefix("Land ").unwrap_or(subject);
    subject
        .strip_prefix("T2.7")
        .is_some_and(|rest| rest.starts_with(':') || rest.starts_with(' '))
}

/// Control: the guard reads the domain crates from l0_deps.py, and flags a
/// file of one (a seeded edit of the composition root's crate) while
/// passing the test kit's and the host's.
#[test]
fn the_guard_catches_a_domain_crate_edit() {
    let domain = domain_crates();
    for name in [
        "sdm-contract",
        "sdm-core",
        "sdm-data",
        "sdm-request",
        "sdm-schema",
    ] {
        assert!(domain.contains(name), "{name} in {domain:?}");
    }
    assert!(!domain.contains("sdm-testkit") && !domain.contains("sdm-host"));
    let changed: Vec<String> = [
        "rust/crates/sdm-testkit/src/memory.rs",
        "rust/crates/sdm-host/src/lib.rs",
        "rust/crates/sdm-core/src/compose.rs",
        "rust/crates/sdm-data-extra/src/lib.rs",
        "contract/fixtures/provider-kit/expected.json",
    ]
    .map(str::to_owned)
    .to_vec();
    assert_eq!(
        in_domain_crates(&changed, &domain),
        vec!["rust/crates/sdm-core/src/compose.rs"]
    );
    assert!(is_t27("T2.7: the non-SQL provider"));
    assert!(is_t27("T2.7 fix 2: paging"));
    assert!(is_t27("T2.7 tests: the kit on memory"));
    assert!(is_t27("Land T2.7: the swapped layer"));
    assert!(!is_t27("T2.70: another package"));
    assert!(!is_t27("T2.5: the kit"));
}

/// T2.7: a guard shows no domain crate changed for it. Every commit of
/// T2.7's in this checkout's history is read, the implementation's among
/// them (so the guard holds something), and none touches a domain crate.
#[test]
fn no_domain_crate_changed_for_t2_7() {
    gated("no_domain_crate_changed_for_t2_7", || {
        assert_eq!(
            git(&["rev-parse", "--is-shallow-repository"]).trim(),
            "false",
            "a shallow checkout cannot show T2.7's commits"
        );
        let domain = domain_crates();
        let mut implemented = false;
        let mut touched = BTreeMap::new();
        for line in git(&["log", "--no-merges", "--format=%H %s", "HEAD"]).lines() {
            let (sha, subject) = line.split_once(' ').unwrap();
            if !is_t27(subject) {
                continue;
            }
            implemented |= subject.starts_with("T2.7: ");
            let changed: Vec<String> =
                git(&["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
                    .lines()
                    .map(str::to_owned)
                    .collect();
            let hit: Vec<String> = in_domain_crates(&changed, &domain)
                .into_iter()
                .cloned()
                .collect();
            if !hit.is_empty() {
                touched.insert(format!("{sha} {subject}"), hit);
            }
        }
        assert!(implemented, "no `T2.7: ` commit in this history");
        assert!(
            touched.is_empty(),
            "T2.7 changed domain crates: {touched:#?}"
        );
    });
}
