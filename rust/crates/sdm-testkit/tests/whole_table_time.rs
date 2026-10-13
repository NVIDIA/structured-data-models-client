// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32e part 1, the implementer's own checks on `rust-host` (the test
//! author's acceptance tests are `t132e_time_inference.rs`): the time
//! column chosen over the whole table, asked of DuckDB in the reference's
//! one statement, and register P1B-38's diagnostic, a whole-table read the
//! source fails refused with the provider's error where the reference
//! falls back to its preview.

use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// `describe_source` over `setup` on the `shop` base, with `events` named
/// as its table.
fn described(setup: &[&str]) -> Observed {
    let case = json!({
        "format": "sdm-case/1",
        "id": "t132e.whole-table",
        "status": "diagnostic",
        "slice": "S1",
        "paths": [],
        "routes": [],
        "cells": [],
        "input": {
            "base": "shop",
            "setup": setup,
            "pql": "PREDICT COUNT(events.*, 0, 30, days) FOR EACH events.event_id",
            "materialize": {"indices": [0, 1], "anchor_time": "2024-03-01"},
        },
        "until": "from_source",
        "core": {
            "describe_source": {
                "provider": {"provider": "sql", "dialect": "duckdb"},
                "source": {
                    "tables": [{"name": "events", "source": "events"}],
                    "primary_keys": {},
                    "time_columns": {},
                },
                "check_keys": true,
            }
        },
        "expect": {"outcome": "success", "stage": "from_source"},
        "why": "T1.32e: the time column over the whole table",
        "source": "rust/crates/sdm-testkit/tests/whole_table_time.rs",
    });
    let case: Case = serde_json::from_value(case).unwrap();
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    host::run(&case, &bases).unwrap()
}

fn time_column(observed: &Observed) -> Value {
    let Answer::Described(described) = &observed.answer else {
        panic!("{:?}", observed.answer);
    };
    described["tables"][0]["time_column"].clone()
}

#[test]
fn a_full_preview_asks_the_whole_table_in_the_reference_s_statement() {
    // A view of 3,000 rows: one value in the first 1,000, others past
    // them, and a load stamp, never asked.
    let observed = described(&[
        "CREATE VIEW events AS SELECT i::BIGINT AS event_id, CASE WHEN i < 1000 \
         THEN TIMESTAMP '2023-01-01' ELSE TIMESTAMP '2023-01-01' + to_minutes(i::INTEGER) \
         END AS happened_at, TIMESTAMP '2020-01-01' AS loaded_at FROM range(3000) t(i)",
    ]);
    assert_eq!(time_column(&observed), json!("happened_at"));
    let asked: Vec<&Value> = observed
        .fetches
        .iter()
        .filter(|f| f["request"]["request"] == "column_ranges")
        .collect();
    assert_eq!(asked.len(), 1, "{:?}", observed.fetches);
    assert_eq!(asked[0]["request"]["columns"], json!(["happened_at"]));
    // Under a full preview nothing is asked.
    let small = described(&[
        "CREATE VIEW events AS SELECT i::BIGINT AS event_id, TIMESTAMP '2023-01-01' \
         + to_minutes(i::INTEGER) AS happened_at FROM range(999) t(i)",
    ]);
    assert_eq!(time_column(&small), json!("happened_at"));
    assert!(
        small
            .fetches
            .iter()
            .all(|f| f["request"]["request"] != "column_ranges")
    );
}

#[test]
fn a_whole_table_read_the_source_fails_is_its_error() {
    // Register P1B-38 (proposed), diagnostic: a view whose rows past the
    // preview fail. The reference catches the failure and chooses from its
    // preview (`happened_at`, measured at b413e9ad in ref-pandas3); the
    // core refuses with the provider's error.
    let observed = described(&[
        "CREATE VIEW events AS SELECT i::BIGINT AS event_id, (i % 7)::DOUBLE AS amount, \
         CASE WHEN i >= 9000 THEN error('past the preview') ELSE TIMESTAMP '2023-01-01' \
         + to_minutes(i::INTEGER) END AS happened_at FROM range(10000) t(i)",
    ]);
    let Answer::Error { code, .. } = &observed.answer else {
        panic!("{:?}", observed.answer);
    };
    assert_eq!(code, "PROVIDER_ERROR");
}
