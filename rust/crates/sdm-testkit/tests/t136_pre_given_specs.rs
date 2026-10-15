// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.36: `sdm-contract-run --pre-given-specs` runs a case written before
//! contract 23, held from outside the implementation (the independent test
//! author), black-box through the binary.
//!
//! A case frozen before T1.32h writes `core.describe_source.source` in
//! contract 22's shape: each declared key a bare list of names, each time
//! column a bare name or null. Without the flag the runner's preflight
//! refuses it (`is not a SourceSpec`, the design's seed 12). Under the
//! flag the runner reads the spec as contract 22's `SourceSpec` and writes
//! each key as `{"value": [...]}` and each time column as
//! `{"value": name-or-null}` before the host runs it, and records the spec
//! it sent: so the case observes exactly as its twin written in the
//! current shape and run without the flag, but for what is read off the
//! case file's bytes. A spec not in contract 22's shape is a runner
//! failure under the flag (`is not a contract-22 SourceSpec`); a case
//! without `describe_source` records as it does without the flag.
//!
//! Gating: `contract/tests/t136ck-pending.json`'s `pending` lists the
//! checks that failed when this file was written; each must still fail
//! until taken out, and the lists only shrink. The implementer changes
//! nothing else of this file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use sdm_testkit::bases;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t136ck-pending.json";
const GROUP: &str = "rust-host";
const FLAG: &str = "--pre-given-specs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{PENDING}: a group that is not a list"))
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn pending(group: &str) -> BTreeSet<String> {
    let gating = read_json(&root().join(PENDING));
    let Some(listed) = gating["pending"].get(group) else {
        panic!("{PENDING}: no pending group {group}");
    };
    ids(listed)
}

/// Holds one check to the pending list: a check not pending must pass, a
/// pending one must still fail.
fn hold(name: &str, problems: Vec<String>) {
    let listed = pending(GROUP).contains(name);
    match (problems.is_empty(), listed) {
        (true, true) => panic!("{name} passes now: take it out of {PENDING}'s pending list"),
        (false, false) => panic!("{name}:\n{}", problems.join("\n")),
        _ => {}
    }
}

#[test]
fn the_pending_lists_only_shrink() {
    let gating = read_json(&root().join(PENDING));
    assert!(gating["about"].is_string());
    let groups: BTreeSet<&String> = gating["authored"].as_object().unwrap().keys().collect();
    let pending_groups: BTreeSet<&String> = gating["pending"].as_object().unwrap().keys().collect();
    assert_eq!(groups, pending_groups);
    for group in groups {
        let authored = ids(&gating["authored"][group]);
        let now = ids(&gating["pending"][group]);
        assert!(
            now.is_subset(&authored),
            "{group}: pending ids added: {:?}",
            now.difference(&authored).collect::<Vec<_>>()
        );
    }
}

/// A scratch directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("sdm-testkit-t136-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// One run of the binary: whether it succeeded, its stderr, and its
/// observations when it left an output file.
struct Run {
    success: bool,
    stderr: String,
    observations: Option<Vec<Value>>,
}

impl Run {
    fn problems(&self) -> Vec<String> {
        if self.success && self.observations.is_some() {
            return Vec::new();
        }
        vec![format!(
            "the run did not record (success {}): {}",
            self.success,
            self.stderr.trim()
        )]
    }

    /// The observations by case id.
    fn by_case(&self) -> BTreeMap<String, Value> {
        self.observations
            .iter()
            .flatten()
            .map(|o| (o["case"].as_str().unwrap().to_owned(), o.clone()))
            .collect()
    }
}

/// `sdm-contract-run [--pre-given-specs] --out OUT CASES...`, from the
/// checkout's root; an earlier output written first, which a failed run
/// must not leave.
fn run(flag: bool, out: &Path, cases: &[PathBuf]) -> Run {
    std::fs::write(out, "an earlier run's observations\n").unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"));
    command.current_dir(root()).stdin(Stdio::null());
    if flag {
        command.arg(FLAG);
    }
    let output = command.arg("--out").arg(out).args(cases).output().unwrap();
    let observations = std::fs::read_to_string(out).ok().map(|text| {
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap_or(Value::String(line.to_owned())))
            .collect()
    });
    Run {
        success: output.status.success(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        observations,
    }
}

/// A failed run's problems against `wanted` in its stderr: none when it
/// failed, said so and left no output.
fn refused(run: &Run, wanted: &str) -> Vec<String> {
    let mut problems = Vec::new();
    if run.success {
        problems.push("the run succeeded".to_owned());
    }
    if !run.stderr.contains("runner failure") || !run.stderr.contains(wanted) {
        problems.push(format!(
            "stderr does not say `{wanted}`: {}",
            run.stderr.trim()
        ));
    }
    if run.observations.is_some() {
        problems.push("an output file was left".to_owned());
    }
    problems
}

/// A case over the `shop` base that ends at `from_source`, its
/// `describe_source` reading `source` as written.
fn shop_case(id: &str, from_source: Value, source: Value) -> Value {
    json!({
        "format": "sdm-case/1",
        "id": id,
        "status": "diagnostic",
        "slice": "S1",
        "paths": [],
        "routes": [],
        "cells": [],
        "input": {
            "base": "shop",
            "from_source": from_source,
            "pql": "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id",
            "materialize": {"indices": [0, 1], "anchor_time": "2024-03-01"},
        },
        "until": "from_source",
        "core": {"describe_source": {
            "provider": {"provider": "sql", "dialect": "duckdb"},
            "source": source,
            "check_keys": true,
        }},
        "expect": {"outcome": "success", "stage": "from_source"},
        "why": format!("T1.36 test case {id}"),
        "source": "rust/crates/sdm-testkit/tests/t136_pre_given_specs.rs",
    })
}

fn write_case(directory: &Path, name: &str, case: &Value) -> PathBuf {
    let path = directory.join(name);
    std::fs::write(&path, serde_json::to_vec_pretty(case).unwrap()).unwrap();
    path
}

/// An observation without what legitimately differs between a case and
/// its twin: the digest of the case file's bytes, the checkout's state
/// (read anew by each run), and the row order of a statement's result,
/// which DuckDB does not fix between two runs of the same case (its
/// multiset, `unordered`, is kept).
fn comparable(observation: &Value) -> Value {
    let mut observation = observation.clone();
    if let Some(identity) = observation["identity"].as_object_mut() {
        identity.remove("commit");
        if let Some(inputs) = identity.get_mut("inputs").and_then(Value::as_object_mut) {
            inputs.remove("case");
        }
    }
    if let Some(statements) = observation
        .pointer_mut("/driver/capture/statements")
        .and_then(Value::as_array_mut)
    {
        for statement in statements {
            for result in statement["results"].as_array_mut().into_iter().flatten() {
                if let Some(values) = result["values"].as_object_mut() {
                    values.remove("ordered");
                }
            }
        }
    }
    observation
}

/// Where two values differ, by JSON pointer, a few at most.
fn differences(a: &Value, b: &Value) -> Vec<String> {
    fn walk(a: &Value, b: &Value, at: String, found: &mut Vec<String>) {
        if a == b || found.len() >= 8 {
            return;
        }
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
                for key in keys {
                    let none = Value::Null;
                    let (u, v) = (x.get(key).unwrap_or(&none), y.get(key).unwrap_or(&none));
                    walk(u, v, format!("{at}/{key}"), found);
                }
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
                for (i, (u, v)) in x.iter().zip(y).enumerate() {
                    walk(u, v, format!("{at}/{i}"), found);
                }
            }
            _ => {
                let short = |v: &Value| v.to_string().chars().take(200).collect::<String>();
                found.push(format!("{at}: {} against {}", short(a), short(b)));
            }
        }
    }
    let mut found = Vec::new();
    walk(a, b, String::new(), &mut found);
    found
}

/// The flagged run's observation of each case against the unflagged run's
/// of its twin, case by case.
fn as_twins(flagged: &Run, twins: &Run) -> Vec<String> {
    let mut problems = flagged.problems();
    if !problems.is_empty() {
        return problems;
    }
    let (mine, theirs) = (flagged.by_case(), twins.by_case());
    let names = |m: &BTreeMap<String, Value>| m.keys().cloned().collect::<Vec<_>>();
    if names(&mine) != names(&theirs) {
        return vec![format!(
            "cases recorded {:?}, against {:?}",
            names(&mine),
            names(&theirs)
        )];
    }
    for (id, observation) in &mine {
        for difference in differences(&comparable(observation), &comparable(&theirs[id])) {
            problems.push(format!("{id}: {difference}"));
        }
    }
    problems
}

fn twin_answer(case: &Value) -> Answer {
    let case: Case = serde_json::from_value(case.clone()).expect("the twin is a case");
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    host::run(&case, &bases).expect("the twin runs").answer
}

/// The twin runs without the flag, as it does today; its observation
/// records `given` as the source sent.
fn twin_run(directory: &Path, twin: &Path, given: &Value) -> Run {
    let ran = run(false, &directory.join("twin.jsonl"), &[twin.to_path_buf()]);
    assert!(ran.success, "the twin runs today: {}", ran.stderr);
    let observations = ran.observations.as_ref().unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(
        observations[0]["core"]["inputs"][0]["source"], *given,
        "the twin records the Given-shaped source it sent"
    );
    ran
}

/// The old-shape case under the flag against its Given-shaped twin without
/// it.
fn old_against_twin(name: &str, from_source: Value, old: Value, given: Value) -> Vec<String> {
    let directory = scratch(name);
    let id = format!("t136.{name}");
    let old_path = write_case(
        &directory,
        "old.json",
        &shop_case(&id, from_source.clone(), old),
    );
    let twin_path = write_case(
        &directory,
        "twin.json",
        &shop_case(&id, from_source, given.clone()),
    );
    let twins = twin_run(&directory, &twin_path, &given);
    let flagged = run(true, &directory.join("old.jsonl"), &[old_path]);
    let mut problems = as_twins(&flagged, &twins);
    if problems.is_empty() {
        let observation = &flagged.observations.as_ref().unwrap()[0];
        if observation["core"]["inputs"][0]["source"] != given {
            problems.push(format!(
                "core.inputs holds {}, not the Given-shaped source",
                observation["core"]["inputs"][0]["source"]
            ));
        }
    }
    std::fs::remove_dir_all(&directory).unwrap();
    problems
}

fn composite_old() -> Value {
    json!({"tables": [], "primary_keys": {"orders": ["order_id", "user_id"]}, "time_columns": {}})
}

fn composite_given() -> Value {
    json!({
        "tables": [],
        "primary_keys": {"orders": {"value": ["order_id", "user_id"]}},
        "time_columns": {},
    })
}

fn composite_from_source() -> Value {
    json!({"primary_keys": {"orders": ["order_id", "user_id"]}})
}

/// 1. Without the flag a case in contract 22's shape stops at the
///    runner's preflight and leaves no output (passes before the fix).
#[test]
fn without_the_flag_an_old_shape_case_stops_at_the_preflight() {
    let directory = scratch("preflight");
    let path = write_case(
        &directory,
        "old.json",
        &shop_case("t136.preflight", composite_from_source(), composite_old()),
    );
    let ran = run(false, &directory.join("out.jsonl"), &[path]);
    std::fs::remove_dir_all(&directory).unwrap();
    hold(
        "preflight-without-flag",
        refused(&ran, "is not a SourceSpec"),
    );
}

/// 2. A composite key in contract 22's shape, under the flag, observes as
///    its Given-shaped twin: the core's refusal of a composite key.
#[test]
fn an_old_composite_key_observes_as_its_given_twin() {
    let twin = shop_case("t136.composite", composite_from_source(), composite_given());
    match twin_answer(&twin) {
        Answer::Error { code, fields, .. } => {
            assert_eq!(code, "UNSUPPORTED_REQUEST");
            assert_eq!(
                fields.get("rule").map(String::as_str),
                Some("composite_key")
            );
        }
        other => panic!("the twin answered {other:?}"),
    }
    let problems = old_against_twin(
        "composite",
        composite_from_source(),
        composite_old(),
        composite_given(),
    );
    hold("composite-key-as-given-twin", problems);
}

/// 3. A single-column key in contract 22's shape, under the flag, observes
///    as its Given-shaped twin: described, the key as declared.
#[test]
fn an_old_single_column_key_observes_as_its_given_twin() {
    let from_source = json!({"primary_keys": {"orders": ["order_id"]}});
    let old = json!({"tables": [], "primary_keys": {"orders": ["order_id"]}, "time_columns": {}});
    let given = json!({
        "tables": [],
        "primary_keys": {"orders": {"value": ["order_id"]}},
        "time_columns": {},
    });
    match twin_answer(&shop_case(
        "t136.single-key",
        from_source.clone(),
        given.clone(),
    )) {
        Answer::Described(described) => {
            let orders = described["tables"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == "orders")
                .unwrap();
            assert_eq!(orders["primary_key"], json!(["order_id"]));
        }
        other => panic!("the twin answered {other:?}"),
    }
    let problems = old_against_twin("single-key", from_source, old, given);
    hold("single-key-as-given-twin", problems);
}

/// 4. Time columns in contract 22's shape, a bare name and a bare null,
///    under the flag, observe as their Given-shaped twins.
#[test]
fn old_time_columns_observe_as_their_given_twins() {
    let mut problems = Vec::new();
    for (name, time) in [("time-name", json!("ts")), ("time-null", Value::Null)] {
        let from_source = json!({"time_columns": {"orders": time}});
        let old = json!({"tables": [], "primary_keys": {}, "time_columns": {"orders": time}});
        let given = json!({
            "tables": [],
            "primary_keys": {},
            "time_columns": {"orders": {"value": time}},
        });
        let id = format!("t136.{name}");
        match twin_answer(&shop_case(&id, from_source.clone(), given.clone())) {
            Answer::Described(described) => {
                let orders = described["tables"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|t| t["name"] == "orders")
                    .unwrap();
                assert_eq!(orders["time_column"], time, "{name}");
            }
            other => panic!("{name}: the twin answered {other:?}"),
        }
        for problem in old_against_twin(name, from_source, old, given) {
            problems.push(format!("{name}: {problem}"));
        }
    }
    hold("time-columns-as-given-twin", problems);
}

/// Contract 22's shape of a Given-shaped spec, as t132h's `old_shape`
/// writes it; `None` when a value has no contract-22 spelling (a null or
/// unreadable key, an unreadable time column).
fn old_shape(spec: &Value) -> Option<Value> {
    let mut old = spec.clone();
    for field in ["primary_keys", "time_columns"] {
        for value in old[field].as_object_mut()?.values_mut() {
            let inner = value.get("value")?.clone();
            if field == "primary_keys" && !inner.is_array() {
                return None;
            }
            *value = inner;
        }
    }
    Some(old)
}

fn declares_any(spec: &Value) -> (bool, bool) {
    let any = |field: &str| spec[field].as_object().is_some_and(|m| !m.is_empty());
    (any("primary_keys"), any("time_columns"))
}

/// 5. Every public case declaring a key or a time column, rewritten into
///    contract 22's shape and run under the flag, observes as the public
///    case run without it, through every stage the case goes on to.
#[test]
fn every_public_case_declaring_keys_or_times_round_trips_under_the_flag() {
    let directory = scratch("corpus");
    let old_directory = directory.join("old");
    std::fs::create_dir_all(&old_directory).unwrap();
    let mut public = Vec::new();
    let mut old = Vec::new();
    let (mut keys, mut times) = (0, 0);
    let files = sdm_testkit::case::case_files(&[root().join("contract/cases")]).unwrap();
    for path in files {
        let case = read_json(&path);
        let spec = &case["core"]["describe_source"]["source"];
        let (key, time) = declares_any(spec);
        if !key && !time {
            continue;
        }
        // A value with no contract-22 spelling could never have been
        // written before contract 23.
        let Some(spec_old) = old_shape(spec) else {
            continue;
        };
        assert_ne!(&spec_old, spec, "{}", path.display());
        keys += usize::from(key);
        times += usize::from(time);
        let mut rewritten = case.clone();
        rewritten["core"]["describe_source"]["source"] = spec_old;
        let name = path.file_name().unwrap().to_str().unwrap();
        old.push(write_case(&old_directory, name, &rewritten));
        public.push(path);
    }
    assert!(keys > 0, "no public case declares a key");
    assert!(times > 0, "no public case declares a time column");
    let twins = run(false, &directory.join("public.jsonl"), &public);
    assert!(
        twins.success,
        "the public cases run today: {}",
        twins.stderr
    );
    let flagged = run(true, &directory.join("old.jsonl"), &old);
    let problems = as_twins(&flagged, &twins);
    std::fs::remove_dir_all(&directory).unwrap();
    hold("corpus-round-trip", problems);
}

/// 6. Under the flag a spec not in contract 22's shape is a runner
///    failure, leaving no output: a key or time column already written as
///    Given, an `unreadable`, a null key, a field contract 22 lacks.
#[test]
fn under_the_flag_a_spec_not_in_contract_22_s_shape_is_refused() {
    let directory = scratch("not-22");
    let variants = [
        ("given-key", composite_from_source(), composite_given()),
        (
            "given-time",
            json!({"time_columns": {"orders": "ts"}}),
            json!({"tables": [], "primary_keys": {}, "time_columns": {"orders": {"value": "ts"}}}),
        ),
        (
            "unreadable-key",
            json!({"primary_keys": {"orders": 3}}),
            json!({"tables": [], "primary_keys": {"orders": "unreadable"}, "time_columns": {}}),
        ),
        (
            "null-key",
            json!({"primary_keys": {"orders": null}}),
            json!({"tables": [], "primary_keys": {"orders": null}, "time_columns": {}}),
        ),
        (
            "unknown-field",
            json!({"primary_keys": {"orders": ["order_id"]}}),
            json!({
                "tables": [],
                "primary_keys": {"orders": ["order_id"]},
                "time_columns": {},
                "check_keys": true,
            }),
        ),
    ];
    let mut problems = Vec::new();
    for (name, from_source, source) in variants {
        let path = write_case(
            &directory,
            &format!("{name}.json"),
            &shop_case(&format!("t136.{name}"), from_source, source),
        );
        let ran = run(true, &directory.join(format!("{name}.jsonl")), &[path]);
        for problem in refused(&ran, "is not a contract-22 SourceSpec") {
            problems.push(format!("{name}: {problem}"));
        }
    }
    std::fs::remove_dir_all(&directory).unwrap();
    hold("given-spec-refused-under-flag", problems);
}

/// 7. Under the flag a case without `describe_source` (the S0 corpus,
///    `validate_query` alone) records exactly as without it.
#[test]
fn under_the_flag_a_case_without_describe_source_records_as_before() {
    let directory = scratch("s0");
    let s0 = [root().join("contract/cases/s0")];
    let plain = run(false, &directory.join("plain.jsonl"), &s0);
    assert!(plain.success, "the S0 cases run today: {}", plain.stderr);
    assert!(
        plain
            .observations
            .iter()
            .flatten()
            .all(|o| o["identity"]["operations"] == json!(["validate_query"])),
        "an S0 case started more than validate_query"
    );
    let flagged = run(true, &directory.join("flagged.jsonl"), &s0);
    let mut problems = as_twins(&flagged, &plain);
    if problems.is_empty() {
        // The same files: their digests too.
        for (id, observation) in flagged.by_case() {
            let digest = &observation["identity"]["inputs"]["case"];
            if *digest != plain.by_case()[&id]["identity"]["inputs"]["case"] {
                problems.push(format!("{id}: the case digest differs"));
            }
        }
    }
    std::fs::remove_dir_all(&directory).unwrap();
    hold("case-without-describe-source-unchanged", problems);
}
