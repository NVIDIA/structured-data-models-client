// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The AdventureWorks case's refusal map on the pure-Rust host (T1.21b).
//!
//! `contract/refusal-maps/base.adventureworks.json` lists each refusal the
//! core route hits on E27's AdventureWorks journey and the probes that reach
//! them, each probe the case with the features refused before it removed.
//! This host runs every probe the map gives it, starting the core
//! operations the probe names over the case's base and the probe's setup,
//! and holds the run to the map: a refusal the map does not list fails, and
//! so does an entry for this host no probe reaches. The same check on
//! `python-core` is `contract/tests/test_refusal_map.py`'s, through
//! `contract/runners/refusal_map.py`, whose rules this file's [`check`]
//! follows. The seeded faults below show the check failing when the map or
//! the evidence is missing, empty or retyped, not only when it differs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::{self, Case, CoreInputs};
use sdm_testkit::host::{self, Answer};
use serde_json::{Value, json};

const RUNNER: &str = "rust-host";
const FIELDS: [&str; 3] = ["code", "operation", "rule"];

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn map() -> Value {
    let bytes = std::fs::read(contract().join("refusal-maps/base.adventureworks.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn the_case(map: &Value) -> Case {
    let loaded = case::load(&[contract().join("cases/s1/base.adventureworks.json")]).unwrap();
    let case = loaded.into_iter().next().unwrap().case;
    assert_eq!(map["case"], json!(case.id));
    case
}

fn success(until: &Value) -> Value {
    json!({"result": "success", "stage": until})
}

/// What this host observes of one run: its first refusal's code and the
/// fields `operation` and `rule`, or the stage whose result it reached.
fn observation(answer: &Answer) -> Value {
    match answer {
        Answer::Error { code, fields, .. } => json!({
            "code": code,
            "operation": fields.get("operation"),
            "rule": fields.get("rule"),
        }),
        Answer::Described(_) => success(&json!("from_source")),
        Answer::Materialized(_) => success(&json!("materialize")),
        Answer::Predicted(_) => success(&json!("predict")),
        Answer::Validated(_) => json!({"result": "validated"}),
    }
}

/// Each probe the map gives this host, run, by its id.
fn run(map: &Value) -> BTreeMap<String, Value> {
    let case = the_case(map);
    let bases: Bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut observed = BTreeMap::new();
    for probe in map["probes"].as_array().unwrap() {
        let Some(spec) = probe.get(RUNNER) else {
            continue;
        };
        let mut given = case.clone();
        given.input.setup = serde_json::from_value(spec["setup"].clone()).unwrap();
        // materialize's query is the case's text, here the query the
        // probe's python-core run sent the core.
        if let Some(pql) = spec.get("pql") {
            given.input.pql = pql.as_str().unwrap().to_owned();
        }
        let core: CoreInputs = serde_json::from_value(spec["core"].clone()).unwrap();
        given.core = Some(core);
        let ran = host::run(&given, &bases).unwrap();
        observed.insert(
            probe["id"].as_str().unwrap().to_owned(),
            observation(&ran.answer),
        );
    }
    observed
}

fn observed() -> &'static BTreeMap<String, Value> {
    static OBSERVED: OnceLock<BTreeMap<String, Value>> = OnceLock::new();
    OBSERVED.get_or_init(|| run(&map()))
}

/// Whether `seen` is a refusal as this host observes it: exactly its
/// fields, each text or null.
fn a_refusal(seen: &Value) -> bool {
    seen.as_object().is_some_and(|fields| {
        fields.len() == FIELDS.len()
            && FIELDS.iter().all(|name| {
                fields
                    .get(*name)
                    .is_some_and(|v| v.is_null() || v.as_str().is_some_and(|s| !s.is_empty()))
            })
    })
}

/// Each way `observed` departs from `map` for this host; empty when the run
/// is the map's. Missing, empty or retyped evidence fails.
fn check(map: &Value, observed: &BTreeMap<String, Value>) -> Vec<String> {
    let mut failures = Vec::new();
    let empty = vec![];
    let probes: Vec<&Value> = map["probes"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter(|probe| probe.get(RUNNER).is_some())
        .collect();
    if probes.is_empty() {
        return vec!["the map gives this host no probe".into()];
    }
    if observed.is_empty() {
        return vec!["nothing was observed".into()];
    }
    let entries: Vec<&Value> = map["entries"].as_array().unwrap_or(&empty).iter().collect();
    let given: BTreeSet<&str> = probes.iter().filter_map(|p| p["id"].as_str()).collect();
    for name in observed.keys() {
        if !given.contains(name.as_str()) {
            failures.push(format!("{name}: observed, and not a probe of {RUNNER}"));
        }
    }
    let mut hit = BTreeSet::new();
    for probe in &probes {
        let name = probe["id"].as_str().unwrap_or("?");
        let spec = &probe[RUNNER];
        let Some(seen) = observed.get(name) else {
            failures.push(format!("{name}: not observed"));
            continue;
        };
        let reached = success(&spec["until"]);
        if *seen != reached && !a_refusal(seen) {
            failures.push(format!("{name}: not the fields the runner observes"));
            continue;
        }
        let expect = spec["expect"].as_str().unwrap_or("");
        let wanted = if expect == "success" {
            Some(reached.clone())
        } else {
            entries
                .iter()
                .find(|entry| entry["id"] == expect)
                .and_then(|entry| entry["observed"].get(RUNNER))
                .cloned()
        };
        if wanted.as_ref() == Some(seen) {
            if expect != "success" {
                hit.insert(expect.to_owned());
            }
            continue;
        }
        let listed = entries
            .iter()
            .find(|entry| entry["observed"].get(RUNNER) == Some(seen));
        if *seen == reached {
            failures.push(format!("{name}: succeeded, where the map has {expect}"));
        } else if let Some(entry) = listed {
            failures.push(format!(
                "{name}: hit {}, where the map has {expect}",
                entry["id"].as_str().unwrap_or("?")
            ));
        } else {
            failures.push(format!("{name}: a refusal the map does not list, {seen}"));
        }
    }
    for entry in &entries {
        let id = entry["id"].as_str().unwrap_or("?");
        if entry["observed"].get(RUNNER).is_some() && !hit.contains(id) {
            failures.push(format!("{id}: listed for {RUNNER} and not hit"));
        }
    }
    failures
}

#[test]
fn the_rust_host_run_is_held_to_the_map() {
    let map = map();
    let observed = observed();
    assert!(observed.len() >= 10, "{observed:?}");
    assert_eq!(check(&map, observed), Vec::<String>::new());
    // The ladder ends at predict's result.
    assert!(
        observed
            .values()
            .any(|seen| *seen == success(&json!("predict")))
    );
}

#[test]
fn a_refusal_the_map_does_not_list_fails() {
    let mut map = map();
    map["entries"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["id"] != "describe_source/time_name");
    let failures = check(&map, observed());
    assert!(
        failures
            .iter()
            .any(|f| f.starts_with("no-text-dates: a refusal the map does not list")),
        "{failures:?}"
    );
}

#[test]
fn an_entry_no_probe_hits_fails() {
    let mut map = map();
    map["entries"].as_array_mut().unwrap().push(json!({
        "id": "materialize/seeded",
        "observed": {"rust-host": {"code": "UNSUPPORTED_REQUEST", "operation": "materialize", "rule": "seeded"}},
    }));
    let failures = check(&map, observed());
    assert_eq!(
        failures,
        ["materialize/seeded: listed for rust-host and not hit"]
    );
}

#[test]
fn missing_empty_or_retyped_evidence_fails() {
    let map = map();
    assert_eq!(check(&map, &BTreeMap::new()), ["nothing was observed"]);
    let mut missing = observed().clone();
    missing.remove("no-keys");
    let failures = check(&map, &missing);
    assert!(
        failures.contains(&"no-keys: not observed".to_owned()),
        "{failures:?}"
    );
    assert!(
        failures
            .contains(&"describe_source/time_columns: listed for rust-host and not hit".to_owned()),
        "{failures:?}"
    );
    for retyped in [json!(7), Value::Null, json!(""), json!(["time_columns"])] {
        let mut changed = observed().clone();
        changed.get_mut("no-keys").unwrap()["rule"] = retyped.clone();
        assert!(!check(&map, &changed).is_empty(), "{retyped}");
    }
    let mut extra = observed().clone();
    extra.get_mut("no-keys").unwrap()["stage"] = json!("from_source");
    assert!(!check(&map, &extra).is_empty());
    let mut unknown = observed().clone();
    unknown.insert("not-a-probe".into(), success(&json!("predict")));
    assert_eq!(
        check(&map, &unknown),
        ["not-a-probe: observed, and not a probe of rust-host"]
    );
    let mut no_probes = map.clone();
    no_probes["probes"] = json!([]);
    assert_eq!(
        check(&no_probes, observed()),
        ["the map gives this host no probe"]
    );
}

#[test]
fn a_probe_that_succeeds_or_hits_another_entry_fails() {
    let mut map = map();
    let probes = map["probes"].as_array_mut().unwrap();
    let probe = probes
        .iter_mut()
        .find(|probe| probe["id"] == "no-text")
        .unwrap();
    probe[RUNNER]["expect"] = json!("materialize/text_column");
    let failures = check(&map, observed());
    assert!(
        failures
            .contains(&"no-text: succeeded, where the map has materialize/text_column".to_owned()),
        "{failures:?}"
    );
    let mut swapped = observed().clone();
    let deeper = swapped["no-inference-config"].clone();
    swapped.insert("one-hop".into(), deeper);
    let failures = check(&self::map(), &swapped);
    assert!(
        failures.contains(
            &"one-hop: hit materialize/deeper_hop, where the map has materialize/text_column"
                .to_owned()
        ),
        "{failures:?}"
    );
}
