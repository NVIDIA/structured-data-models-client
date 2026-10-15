// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The sampler's two `neighbors` reads (`sdm_engine::sample::by_fkey`
//! and `by_time`) are the requests `sdm-testkit`'s `neighbors.rs` runs on
//! DuckDB against the reference's rows (`neighbor-requests.json`, register
//! ORD-2): built here for the vectors' tables, each must be the one listed
//! there, so the L1 vectors hold the core's own requests.

use sdm_contract::data::{AttachmentId, DataRequest, KeySet, NullOrder};
use sdm_contract::ops::{DescribedColumn, Dtype, Stype};
use sdm_contract::{ColumnType, Unit};
use sdm_engine::graph::Table;
use sdm_engine::sample::{by_fkey, by_time};
use serde_json::Value;

const REQUESTS: &str = include_str!("../../sdm-testkit/tests/neighbor-requests.json");

fn column(name: &str, received: ColumnType, dtype: Dtype, stype: Stype) -> DescribedColumn {
    DescribedColumn {
        name: name.into(),
        received,
        dtype,
        stype,
        stype_before_key: None,
        nullable: true,
        unique: false,
        declared_type: None,
    }
}

/// A table of the vectors: `eid` its key, `uid` referring to the users,
/// `ts` its time column (zoned for `evtz`), `val` a number.
fn table(name: &str) -> Table {
    let zone = (name == "evtz").then(|| "UTC".to_owned());
    Table {
        name: name.into(),
        source: name.into(),
        columns: vec![
            column("eid", ColumnType::Int64, Dtype::Int, Stype::Id),
            column("uid", ColumnType::Int64, Dtype::Int, Stype::Id),
            column(
                "ts",
                ColumnType::Timestamp {
                    unit: Unit::Us,
                    zone,
                },
                Dtype::Date,
                Stype::Timestamp,
            ),
            column("val", ColumnType::Float64, Dtype::Float, Stype::Numerical),
        ],
        primary_key: Some("eid".into()),
        time_column: Some("ts".into()),
    }
}

fn seeds() -> KeySet {
    KeySet {
        attachment: AttachmentId(0),
        columns: vec![],
    }
}

fn columns() -> Vec<String> {
    ["eid", "uid", "ts", "val"].map(String::from).to_vec()
}

#[test]
fn the_vectors_run_the_core_s_own_requests() {
    let listed: Value = serde_json::from_str(REQUESTS).unwrap();
    let built: Vec<(&str, DataRequest)> = vec![
        (
            "fkey/ev",
            by_fkey(&table("ev"), "uid", columns(), seeds(), 32, NullOrder::Last).unwrap(),
        ),
        (
            "fkey/evtz",
            by_fkey(
                &table("evtz"),
                "uid",
                columns(),
                seeds(),
                32,
                NullOrder::Last,
            )
            .unwrap(),
        ),
        (
            "time/ev",
            by_time(&table("ev"), "uid", columns(), seeds(), true).unwrap(),
        ),
        (
            "time/ev/no-start",
            by_time(&table("ev"), "uid", columns(), seeds(), false).unwrap(),
        ),
        (
            "time/evtz",
            by_time(&table("evtz"), "uid", columns(), seeds(), true).unwrap(),
        ),
    ];
    let names = listed["requests"].as_object().unwrap();
    assert_eq!(
        names.len(),
        built.len(),
        "a listed request is not built here"
    );
    for (name, request) in built {
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            listed["requests"][name],
            "{name}"
        );
    }
}
