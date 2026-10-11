// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The refusal maps on the pure-Rust host: the AdventureWorks case's
//! (T1.21b) and the Phase 1b inventory's families' (T1.21e).
//!
//! `contract/refusal-maps/base.adventureworks.json` lists each refusal the
//! core route hits on E27's AdventureWorks journey and the probes that reach
//! them, each probe the case with the features refused before it removed;
//! each family map (`keys.json` to `baseline.json`) does the same for every
//! case of one of the inventory's families, each probe naming its case.
//! This host runs every probe a map gives it, starting the core
//! operations the probe names over its case's base and the probe's setup,
//! and holds the run to the map: a refusal the map does not list fails, and
//! so does an entry for this host no probe reaches. The same check on
//! `python-core` is `contract/tests/test_refusal_map.py`'s, through
//! `contract/runners/refusal_map.py`, whose rules this file's [`check`]
//! follows. The seeded faults below show the check failing when the map or
//! the evidence is missing, empty or retyped, not only when it differs.
//!
//! With `SDM_REFUSAL_MAP_OBSERVED` naming a file, the family maps' test
//! writes there what this host observed of every map's probes, by map and
//! probe, before it checks them: what `refusal_map.py write
//! --rust-observed` reads to regenerate a map.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::{self, Case, CoreInputs};
use sdm_testkit::edits::SchemaEdit;
use sdm_testkit::host::{self, Answer};
use serde_json::{Value, json};

const RUNNER: &str = "rust-host";
const FIELDS: [&str; 3] = ["code", "operation", "rule"];

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn map() -> Value {
    read_map("base.adventureworks.json")
}

fn read_map(name: &str) -> Value {
    let bytes = std::fs::read(contract().join("refusal-maps").join(name)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The family maps, by file name: every map but the AdventureWorks case's.
fn family_maps() -> Vec<(String, Value)> {
    let mut names: Vec<String> = std::fs::read_dir(contract().join("refusal-maps"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".json") && name != "base.adventureworks.json")
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let map = read_map(&name);
            (name, map)
        })
        .collect()
}

/// Every contract case by id, as the case loader reads it, and as JSON.
fn cases() -> &'static BTreeMap<String, (Case, Value)> {
    static CASES: OnceLock<BTreeMap<String, (Case, Value)>> = OnceLock::new();
    CASES.get_or_init(|| {
        let loaded = case::load(&[contract().join("cases")]).unwrap();
        loaded
            .into_iter()
            .map(|each| {
                let path = contract()
                    .join("cases")
                    .join(each.case.slice.to_lowercase())
                    .join(format!("{}.json", each.case.id));
                let raw: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                (each.case.id.clone(), (each.case, raw))
            })
            .collect()
    })
}

/// The case `probe` of `map` runs: its own `case`, or the map's.
fn case_of<'a>(map: &'a Value, probe: &'a Value) -> &'static (Case, Value) {
    let id = probe
        .get("case")
        .or_else(|| map.get("case"))
        .and_then(Value::as_str)
        .unwrap();
    &cases()[id]
}

fn the_case(map: &Value) -> Case {
    let case = cases()["base.adventureworks"].0.clone();
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

/// What one run gave of its core operations where it reached
/// `describe_source`: the digests of the description and of the schema
/// materialize sampled (the pin's `AGREE`, `contract/runners/refusal_map.py`).
type Digests = BTreeMap<String, BTreeMap<String, String>>;

/// `value` as canonical JSON, as `refusal_map.digest` writes it: keys
/// sorted at every depth, no whitespace, UTF-8.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(members) => {
            let mut sorted: Vec<(&String, &Value)> = members.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k.clone(), canonical(v)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn digest(value: &Value) -> String {
    case::sha256(serde_json::to_string(&canonical(value)).unwrap().as_bytes())
}

/// Each probe the map gives this host, run: what it observed, by the
/// probe's id, and the digests of its core operations where it reached
/// `describe_source`.
fn run_with_digests(map: &Value) -> (BTreeMap<String, Value>, Digests) {
    let (observed, digests, _) = run_probes(map);
    (observed, digests)
}

/// [`run_with_digests`], with each description reached, by probe.
fn run_probes(map: &Value) -> (BTreeMap<String, Value>, Digests, BTreeMap<String, Value>) {
    let bases: Bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut observed = BTreeMap::new();
    let mut digests = Digests::new();
    let mut descriptions = BTreeMap::new();
    for probe in map["probes"].as_array().unwrap() {
        let Some(spec) = probe.get(RUNNER) else {
            continue;
        };
        let mut given = case_of(map, probe).0.clone();
        given.input.setup = serde_json::from_value(spec["setup"].clone()).unwrap();
        // materialize's query is the case's text, here the query the
        // probe's python-core run sent the core.
        if let Some(pql) = spec.get("pql") {
            given.input.pql = pql.as_str().unwrap().to_owned();
        }
        let core: CoreInputs = serde_json::from_value(spec["core"].clone()).unwrap();
        given.core = Some(core);
        // The probe's edits, made to this host's own description: the
        // probe names no schema for materialize (T1.21d).
        let edits: Vec<SchemaEdit> = match spec.get("edits") {
            Some(edits) => serde_json::from_value(edits.clone()).unwrap(),
            None => vec![],
        };
        let ran = host::run_edited(&given, &bases, &edits).unwrap();
        let name = probe["id"].as_str().unwrap().to_owned();
        if let Some(described) = &ran.described {
            let mut agreed = BTreeMap::from([
                ("described".to_owned(), digest(described)),
                (
                    "warnings".to_owned(),
                    digest(&Value::from(ran.warnings.clone())),
                ),
            ]);
            for input in &ran.inputs {
                if input["op"] == "materialize" {
                    agreed.insert("materialize_schema".into(), digest(&input["schema"]));
                }
            }
            digests.insert(name.clone(), agreed);
            descriptions.insert(name.clone(), described.clone());
        }
        observed.insert(name, observation(&ran.answer));
    }
    (observed, digests, descriptions)
}

/// `described` with each zoned column's received zone written as the
/// family pin writes it (`refusal_map.zone_free`): DuckDB hands a
/// `TIMESTAMPTZ` over in the database's zone, the machine's.
fn zone_free(described: &Value) -> Value {
    let mut described = described.clone();
    if let Some(tables) = described["tables"].as_array_mut() {
        for table in tables {
            if let Some(columns) = table["columns"].as_array_mut() {
                for column in columns {
                    // Read, never indexed into, so nothing is added.
                    if let Some(zone) = column
                        .get_mut("received")
                        .and_then(|received| received.get_mut("zone"))
                        .filter(|zone| !zone.is_null())
                    {
                        *zone = json!("<the database zone>");
                    }
                }
            }
        }
    }
    described
}

fn observed() -> &'static BTreeMap<String, Value> {
    &ran().0
}

fn ran() -> &'static (BTreeMap<String, Value>, Digests) {
    static RAN: OnceLock<(BTreeMap<String, Value>, Digests)> = OnceLock::new();
    RAN.get_or_init(|| {
        let map = map();
        the_case(&map);
        run_with_digests(&map)
    })
}

/// What this host observed of each family map's probes, by file name.
/// Each family map's run on this host, by map: what each probe observed,
/// and the digests of its core operations where it reached
/// `describe_source`.
#[allow(clippy::type_complexity)]
fn families_ran() -> &'static BTreeMap<String, (BTreeMap<String, Value>, Digests)> {
    static RAN: OnceLock<BTreeMap<String, (BTreeMap<String, Value>, Digests)>> = OnceLock::new();
    RAN.get_or_init(|| {
        family_maps()
            .into_iter()
            .map(|(name, map)| {
                let (observed, mut digests, descriptions) = run_probes(&map);
                // Not the schema materialize samples, which the family pin
                // leaves out: the hosts refuse some questions at different
                // points (python-core's bridge before materialize starts,
                // this host in the core).
                for agreed in digests.values_mut() {
                    agreed.remove("materialize_schema");
                }
                // A zoned column's received zone is the database's, the
                // machine's, so the pin writes it as one marker
                // (refusal_map.zone_free).
                for (name, described) in &descriptions {
                    if let Some(agreed) = digests.get_mut(name) {
                        agreed.insert("described".into(), digest(&zone_free(described)));
                    }
                }
                (name, (observed, digests))
            })
            .collect()
    })
}

fn families_observed() -> &'static BTreeMap<String, BTreeMap<String, Value>> {
    static OBSERVED: OnceLock<BTreeMap<String, BTreeMap<String, Value>>> = OnceLock::new();
    OBSERVED.get_or_init(|| {
        families_ran()
            .iter()
            .map(|(name, (observed, _))| (name.clone(), observed.clone()))
            .collect()
    })
}

fn pin() -> Value {
    let bytes = std::fs::read(contract().join("pinned/base.adventureworks.probes.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Each way `observed`'s digests depart from `pin`'s, as
/// `refusal_map.agree` reads them; empty when they are the pin's. Missing,
/// empty or retyped evidence fails.
fn agree(pin: &Value, observed: &Digests) -> Vec<String> {
    const AGREE: [&str; 3] = ["described", "materialize_schema", "warnings"];
    let Some(probes) = pin
        .get("probes")
        .and_then(Value::as_object)
        .filter(|p| !p.is_empty())
    else {
        return vec!["the pin lists no probe".into()];
    };
    let hex = |v: &Value| {
        v.as_str()
            .is_some_and(|t| t.len() == 64 && t.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')))
    };
    let names: std::collections::BTreeSet<&String> = probes.keys().chain(observed.keys()).collect();
    let mut failures = Vec::new();
    for name in names {
        let Some(wanted) = probes
            .get(name)
            .and_then(Value::as_object)
            .filter(|w| !w.is_empty())
        else {
            failures.push(format!("{name}: reached describe_source, not pinned"));
            continue;
        };
        if !wanted.contains_key("described")
            || wanted
                .iter()
                .any(|(k, v)| !AGREE.contains(&k.as_str()) || !hex(v))
        {
            failures.push(format!("{name}: the pin is not digests"));
            continue;
        }
        let Some(seen) = observed.get(name) else {
            failures.push(format!("{name}: pinned, and describe_source not reached"));
            continue;
        };
        let differing: Vec<&str> = AGREE
            .into_iter()
            .filter(|k| seen.get(*k).map(String::as_str) != wanted.get(*k).and_then(Value::as_str))
            .collect();
        if !differing.is_empty() {
            failures.push(format!("{name}: {} not the pin's", differing.join(", ")));
        }
    }
    failures
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
    // The first probe is the case's own core operations; the Python test
    // binds every later one to the case (refusal_map.bound).
    let case: Value = serde_json::from_slice(
        &std::fs::read(contract().join("cases/s1/base.adventureworks.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(map["probes"][0]["rust-host"]["core"], case["core"]);
    let observed = observed();
    // From T1.24 the probes keep the notebook's inputs whose rules have
    // landed, so the ladder is five rust-host probes: the notebook's call,
    // the text dates dropped, and the three past materialize's refusals.
    assert!(observed.len() >= 5, "{observed:?}");
    assert_eq!(check(&map, observed), Vec::<String>::new());
    // The ladder ends at predict's result.
    assert!(
        observed
            .values()
            .any(|seen| *seen == success(&json!("predict")))
    );
    // Where both hosts reach describe_source, this host's description and
    // the schema its materialize samples, its own description with the
    // probe's edits made, are the pin python-core is held to (T1.21d).
    let digests = &ran().1;
    assert_eq!(agree(&pin(), digests), Vec::<String>::new());
    assert!(digests.len() >= 4);
    assert!(
        digests
            .values()
            .any(|d| d.contains_key("materialize_schema"))
    );
}

#[test]
fn the_descriptions_are_held_to_the_pin_fail_closed() {
    let pin = pin();
    let seen: Digests = serde_json::from_value(pin["probes"].clone()).unwrap();
    assert_eq!(agree(&pin, &seen), Vec::<String>::new());
    assert!(!agree(&pin, &Digests::new()).is_empty());
    assert_eq!(agree(&json!({}), &seen), ["the pin lists no probe"]);
    assert_eq!(
        agree(&json!({"probes": {}}), &seen),
        ["the pin lists no probe"]
    );
    let name = seen.keys().next().unwrap().clone();
    let mut other = seen.clone();
    other
        .get_mut(&name)
        .unwrap()
        .insert("described".into(), "0".repeat(64));
    assert_eq!(
        agree(&pin, &other),
        [format!("{name}: described not the pin's")]
    );
    let mut missing = seen.clone();
    missing.remove(&name);
    assert_eq!(
        agree(&pin, &missing),
        [format!("{name}: pinned, and describe_source not reached")]
    );
    let mut extra = seen.clone();
    extra.insert("not-a-probe".into(), seen[&name].clone());
    assert_eq!(
        agree(&pin, &extra),
        ["not-a-probe: reached describe_source, not pinned"]
    );
    for retyped in [json!(7), Value::Null, json!(""), json!("abc"), json!(["x"])] {
        let mut broken = pin.clone();
        broken["probes"][&name]["described"] = retyped.clone();
        assert_eq!(
            agree(&broken, &seen),
            [format!("{name}: the pin is not digests")],
            "{retyped}"
        );
    }
    // The digest is of canonical JSON, the one refusal_map.digest writes.
    assert_eq!(
        digest(&json!({"b": 1, "a": [1, "é"]})),
        digest(&serde_json::from_str::<Value>(r#"{ "a" : [1, "\u00e9"], "b": 1 }"#).unwrap())
    );
}

#[test]
fn a_refusal_the_map_does_not_list_fails() {
    let mut map = map();
    map["entries"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["id"] != "describe_source/text_time");
    let failures = check(&map, observed());
    assert!(
        failures
            .iter()
            .any(|f| f.starts_with("notebook: a refusal the map does not list")),
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
    missing.remove("no-inference-config");
    let failures = check(&map, &missing);
    assert!(
        failures.contains(&"no-inference-config: not observed".to_owned()),
        "{failures:?}"
    );
    assert!(
        failures.contains(&"materialize/deeper_hop: listed for rust-host and not hit".to_owned()),
        "{failures:?}"
    );
    for retyped in [json!(7), Value::Null, json!(""), json!(["deeper_hop"])] {
        let mut changed = observed().clone();
        changed.get_mut("no-inference-config").unwrap()["rule"] = retyped.clone();
        assert!(!check(&map, &changed).is_empty(), "{retyped}");
    }
    let mut extra = observed().clone();
    extra.get_mut("no-inference-config").unwrap()["stage"] = json!("materialize");
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

#[test]
fn the_family_maps_are_held_on_the_rust_host() {
    let maps = family_maps();
    let families: Vec<&str> = maps
        .iter()
        .map(|(_, map)| map["family"].as_str().unwrap())
        .collect();
    assert_eq!(
        families.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "baseline",
            "decimal",
            "empty-null",
            "keys",
            "nested",
            "refusal",
            "time"
        ])
    );
    let observed = families_observed();
    // What refusal_map.py write --rust-observed reads, every map's probes
    // by map and probe, written before the check so a map can be made from
    // a run it does not yet hold.
    if let Ok(path) = std::env::var("SDM_REFUSAL_MAP_OBSERVED") {
        let mut every: BTreeMap<String, BTreeMap<String, Value>> = observed.clone();
        every.insert("base.adventureworks.json".into(), self::observed().clone());
        std::fs::write(path, serde_json::to_vec_pretty(&every).unwrap()).unwrap();
    }
    // Where both hosts reach describe_source, this host's description, the
    // schema its materialize samples and the core's warnings are the pin
    // python-core is held to (T1.23).
    let pinned: Value = serde_json::from_slice(
        &std::fs::read(contract().join("pinned/families.probes.json")).unwrap(),
    )
    .unwrap();
    for (name, map) in &maps {
        let seen = &observed[name];
        assert_eq!(check(map, seen), Vec::<String>::new(), "{name}");
        let family = map["family"].as_str().unwrap();
        assert_eq!(
            agree(&pinned["families"][family], &families_ran()[name].1),
            Vec::<String>::new(),
            "{name}"
        );
        // Each case's first probe is the case's own core operations; the
        // Python test binds every later one to its case (refusal_map.bound).
        let mut started = BTreeSet::new();
        for probe in map["probes"].as_array().unwrap() {
            let Some(spec) = probe.get(RUNNER) else {
                continue;
            };
            if spec["derive"]["from"] == "case" {
                let (case, raw) = case_of(map, probe);
                assert_eq!(spec["core"], raw["core"], "{name}: {}", probe["id"]);
                started.insert(case.id.clone());
            }
        }
        let wanted: BTreeSet<String> = map["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(started, wanted, "{name}");
    }
}

/// A family map with one of its refused probes, and what this host saw.
fn a_family_refusal() -> (String, Value, String, String) {
    for (name, map) in family_maps() {
        for probe in map["probes"].as_array().unwrap() {
            if let Some(expect) = probe
                .get(RUNNER)
                .and_then(|spec| spec["expect"].as_str())
                .filter(|expect| *expect != "success")
            {
                let id = probe["id"].as_str().unwrap().to_owned();
                return (name, map.clone(), id, expect.to_owned());
            }
        }
    }
    panic!("no family map has a refusal on this host")
}

#[test]
fn a_family_map_fails_on_an_unlisted_refusal_and_a_stale_entry() {
    // Seeded: the entry a probe hits removed, so its refusal is one the
    // map does not list; and an entry no probe hits added.
    let (name, map, probe, expect) = a_family_refusal();
    let seen = &families_observed()[&name];
    assert_eq!(check(&map, seen), Vec::<String>::new());
    let mut unlisted = map.clone();
    unlisted["entries"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["id"] != expect.as_str());
    let failures = check(&unlisted, seen);
    assert!(
        failures
            .iter()
            .any(|f| f.starts_with(&format!("{probe}: a refusal the map does not list"))),
        "{failures:?}"
    );
    let mut stale = map.clone();
    stale["entries"].as_array_mut().unwrap().push(json!({
        "id": "materialize/stale",
        "observed": {"rust-host": {"code": "UNSUPPORTED_REQUEST", "operation": "materialize", "rule": "stale"}},
    }));
    assert_eq!(
        check(&stale, seen),
        ["materialize/stale: listed for rust-host and not hit"]
    );
    assert_eq!(check(&map, &BTreeMap::new()), ["nothing was observed"]);
}
