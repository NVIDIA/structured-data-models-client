// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit time-column selection remains honored beyond the preview size.

use sdm_testkit::bases;
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use serde_json::json;
use std::path::PathBuf;

#[test]
fn named_time_column_is_honored_on_large_tables() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let bases = bases::load(&root.join("contract/bases.yaml")).unwrap();
    for rows in [1_000, 1_001, 5_000] {
        for named in ["created_at", "billed_at"] {
            let case: Case = serde_json::from_value(json!({
            "format": "sdm-case/1", "id": format!("named-time.{rows}"),
            "status": "diagnostic", "slice": "S1", "paths": [], "routes": [], "cells": [],
            "input": {
                "base": "shop",
                "setup": [format!("CREATE TABLE events AS SELECT i AS event_id, TIMESTAMP '2020-01-01' AS created_at, TIMESTAMP '2024-01-01' AS billed_at FROM range({rows}) t(i)")],
                "from_source": {"tables": {"events": "events"}, "time_columns": {"events": named}},
                "pql": "PREDICT COUNT(events.*, 0, 30, days) FOR EACH events.event_id",
                "materialize": {"indices": [0, 1], "anchor_time": "2024-03-01"}
            },
            "until": "from_source",
            "core": {"describe_source": {
                "provider": {"provider": "sql", "dialect": "duckdb"},
                "source": {"tables": [{"name": "events", "source": {"source": ["events"]}}], "primary_keys": {}, "time_columns": {"events": {"value": named}}},
                "check_keys": true
            }},
            "expect": {"outcome": "success", "stage": "from_source"},
            "why": "Explicit time column overrides inference beyond preview size",
            "source": "authored current contract"
        })).unwrap();
            let observed = host::run(&case, &bases).unwrap();
            let Answer::Described(schema) = observed.answer else {
                panic!("description failed at {rows} rows");
            };
            let table = schema["tables"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == "events")
                .unwrap();
            assert_eq!(table["time_column"], named, "{rows} rows");
        }
    }
}
