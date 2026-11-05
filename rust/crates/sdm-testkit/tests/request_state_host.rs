// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ranking questions deduplicate candidates while preserving their input order.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
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
    let observed = observe("rank.repeated-candidates", database);
    let Answer::Predicted(predicted) = &observed.answer else {
        panic!("expected prediction, got {:?}", observed.answer);
    };
    let written = match &predicted["calls"] {
        Value::Array(bodies) => bodies.clone(),
        _ => vec![predicted["body"].clone()],
    };
    assert_eq!(
        asked(&written),
        ["buys.item_id = 2", "buys.item_id = 1", "buys.item_id = 0"]
    );
}

#[test]
fn ranking_preserves_unique_candidate_order_on_duckdb() {
    repeats("duckdb");
}

#[test]
fn ranking_preserves_unique_candidate_order_on_sqlite() {
    repeats("sqlite");
}
