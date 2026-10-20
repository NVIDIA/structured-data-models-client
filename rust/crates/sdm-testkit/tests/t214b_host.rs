// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.14b's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t214b_host.py` and
//! `contract/tests/test_t214b_decimal_stype.py`, which hold `python-core` to
//! the same cases. On this host no Python decides anything, so what passes
//! here is the core's own (learnings lesson 17).
//!
//! - `rank.repeated-candidates`: a ranking given the candidates 2, 1, 2, 1
//!   and 0, which the reference's `dict.fromkeys` keeps as 2, 1 and 0. This
//!   host hands the core each candidate as given; the questions the core
//!   asks are python-ref's, in its order, and its payloads python-ref's
//!   (each but its request id).
//! - `decimal.stype-past-float-precision`: `orders.d`, a `DECIMAL(38, 18)`
//!   whose 80 values differ only past float64's precision, is described
//!   with python-ref's semantic type (categorical on DuckDB; text on SQLite,
//!   where the contract loader stores a decimal as text). The rides-along
//!   decimal stype fix.
//!
//! Each runs on DuckDB (`host::run`) and SQLite (`host::run_sqlite`, the
//! contract loader); this host's Postgres pass replays python-core's
//! fetches (KO2-20), which `cross-host (postgres)` holds to python-core.
//!
//! Gating: `contract/tests/t214b-pending.json`'s `pending` names, as
//! `host:rust:<case key>` and `stype:rust:<case key>`, the keys that failed
//! when written. Each pending key must still fail (strict); once one
//! passes it fails, naming itself, until it is taken out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Whether `name` is pending; it must be authored.
fn pending(name: &str) -> bool {
    let gating = read_json(&root().join("contract/tests/t214b-pending.json"));
    let listed = |list: &str| {
        gating[list]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key.as_str() == Some(name))
    };
    assert!(listed("authored"), "{name} is not among the authored keys");
    listed("pending")
}

/// The case key of `id` on `database`, as the runners key it.
fn key(id: &str, database: &str) -> String {
    if database == "duckdb" {
        id.to_owned()
    } else {
        format!("{id}@{database}")
    }
}

/// python-ref's committed lines, by key.
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

fn observe(id: &str, database: &str) -> host::Observed {
    let contract = root().join("contract");
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let loaded = case::load(&[contract.join(format!("cases/s1/{id}.json"))]).unwrap();
    let case: &Case = &loaded[0].case;
    match database {
        "duckdb" => host::run(case, &bases),
        _ => host::run_sqlite(case, &bases, &root()),
    }
    .unwrap_or_else(|failure| panic!("{id}@{database}: host failure: {failure:#}"))
}

/// Runs `check` for `name`: it must give no problem, or, while `name` is
/// pending, give one.
fn held(name: &str, check: impl FnOnce() -> Option<String>) {
    match (check(), pending(name)) {
        (Some(why), false) => panic!("{name}: {why}"),
        (None, true) => {
            panic!("{name} passes now: take it out of contract/tests/t214b-pending.json's pending")
        }
        _ => {}
    }
}

/// A request as python-ref's endpoint received it, its request id left out.
fn without_request_id(request: &Value) -> Value {
    let mut request = request.clone();
    if let Some(metadata) = request.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    request
}

/// What each request asks about: its query's condition on the candidate.
fn asked(requests: &[Value]) -> Vec<String> {
    requests
        .iter()
        .map(|r| {
            let query = r["metadata"]["source_query"].as_str().unwrap_or_default();
            query
                .split_once(" WHERE ")
                .map_or(query, |(_, rest)| rest.split(',').next().unwrap_or(rest))
                .to_owned()
        })
        .collect()
}

fn repeats(database: &str) {
    let id = "rank.repeated-candidates";
    let key = key(id, database);
    let line = reference()
        .remove(&key)
        .unwrap_or_else(|| panic!("{key}: no python-ref line"));
    let sent: Vec<Value> = line["public"]["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| without_request_id(&p["parsed"]))
        .collect();
    // Control: the line asks each candidate once, as dict.fromkeys keeps them.
    assert_eq!(
        asked(&sent),
        ["buys.item_id = 2", "buys.item_id = 1", "buys.item_id = 0"]
    );
    held(&format!("host:rust:{key}"), || {
        let observed = observe(id, database);
        let (result, code) = observed.answer.outcome();
        if result != "success" {
            return Some(format!("{result} ({code:?}) where python-ref answers"));
        }
        let Answer::Predicted(predicted) = &observed.answer else {
            return Some(format!("no answer of predict: {:?}", observed.answer));
        };
        let written: Vec<Value> = match &predicted["calls"] {
            Value::Array(bodies) => bodies.iter().map(without_request_id).collect(),
            _ => vec![without_request_id(&predicted["body"])],
        };
        if asked(&written) != asked(&sent) {
            return Some(format!(
                "asks {:?} where python-ref asks {:?}",
                asked(&written),
                asked(&sent)
            ));
        }
        None
    });
}

#[test]
fn rust_host_asks_repeated_candidates_as_the_reference_keeps_them_on_duckdb() {
    repeats("duckdb");
}

#[test]
fn rust_host_asks_repeated_candidates_as_the_reference_keeps_them_on_sqlite() {
    repeats("sqlite");
}

fn stype(database: &str) {
    let id = "decimal.stype-past-float-precision";
    let key = key(id, database);
    let line = reference()
        .remove(&key)
        .unwrap_or_else(|| panic!("{key}: no python-ref line"));
    let of_d = |tables: &Value| -> Option<String> {
        tables.as_array()?.iter().find(|t| t["name"] == "orders")?["columns"]
            .as_array()?
            .iter()
            .find(|c| c["name"] == "d")?["stype"]
            .as_str()
            .map(str::to_owned)
    };
    let want = of_d(&line["public"]["schema"]["tables"]).expect("python-ref describes orders.d");
    held(&format!("stype:rust:{key}"), || {
        let observed = observe(id, database);
        let Some(described) = observed.described.as_ref() else {
            return Some("no description".to_owned());
        };
        match of_d(&described["tables"]) {
            Some(got) if got == want => None,
            got => Some(format!("orders.d is {got:?} where python-ref has {want:?}")),
        }
    });
}

#[test]
fn rust_host_describes_a_scaled_decimal_as_the_reference_on_duckdb() {
    stype("duckdb");
}

#[test]
fn rust_host_describes_a_scaled_decimal_as_the_reference_on_sqlite() {
    stype("sqlite");
}
