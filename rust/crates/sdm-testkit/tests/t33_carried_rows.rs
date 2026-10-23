// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.3's acceptance tests for the two `sdm-testkit` rows it closes
//! (migration-plan.md section 8, carried into Phase 3), written by the
//! package's test author before the implementation:
//!
//! - P2-T2.7-1: "The second data layer drops a whole key list when one name
//!   in it is null, where `sdm-data` drops only the null name
//!   (`sdm-testkit/src/layer.rs`)": a key's and a foreign key's column list
//!   holding a null name is read with the null name dropped and every other
//!   name kept, as `sdm-data`'s reader (`read.rs`, `names`) reads it;
//! - P2-T2.7-2: "The memory test provider ... does not answer
//!   `column_ranges`, so on a table bigger than the preview it picks the
//!   time column from the preview (`sdm-testkit/src/memory.rs`)": the
//!   provider declares `column_ranges` and answers it over every row of a
//!   table whose times all lie past its first rows.
//!
//! Gating: `contract/tests/t33-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each must still fail
//! (strict); once one passes it fails, naming itself, until it is taken
//! out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, TimestampMicrosecondType};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, OptionalRequest, ProviderChoice, catalog_columns as c,
};
use sdm_testkit::layer::HostLayer;
use sdm_testkit::memory::Memory;
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t33-pending.json";
const AUTHORED: [&str; 2] = [
    "a_null_name_drops_only_that_name_from_a_key_list",
    "the_memory_provider_answers_column_ranges_over_every_row",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The pending file's `list`, its `rust:` keys of this file's tests.
fn named(list: &str) -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join(PENDING)).unwrap();
    let file: Value = serde_json::from_str(&text).unwrap();
    file[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .filter(|n| AUTHORED.contains(n))
        .map(str::to_owned)
        .collect()
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(key: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, named("pending").contains(key)) {
        (Ok(()), true) => panic!("{key} passes now: take rust:{key} out of {PENDING}'s pending"),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    assert!(named("pending").is_subset(&named("authored")));
    let authored: BTreeSet<String> = AUTHORED.iter().map(|&n| n.to_owned()).collect();
    assert_eq!(named("authored"), authored);
}

/// A list-of-names column, each row's names, `None` a null name.
fn lists(rows: &[&[Option<&str>]]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for name in *row {
            builder.values().append_option(*name);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn describe(part: &str) -> DataRequest {
    serde_json::from_value(json!({
        "request": "describe",
        "tables": [{"source": ["orders"]}],
        "part": part,
    }))
    .unwrap()
}

fn host() -> ProviderChoice {
    ProviderChoice::Host {
        capabilities: Memory::capabilities(),
    }
}

#[test]
fn a_null_name_drops_only_that_name_from_a_key_list() {
    gated("a_null_name_drops_only_that_name_from_a_key_list", || {
        let keys = RecordBatch::try_from_iter([
            (
                c::KIND,
                Arc::new(StringArray::from(vec![c::KIND_PRIMARY])) as ArrayRef,
            ),
            (c::COLUMNS, lists(&[&[Some("a"), None, Some("b")]])),
        ])
        .unwrap();
        let read = HostLayer.read(&describe("keys"), &host(), &[keys]).unwrap();
        let CatalogAnswer::Keys(found) = read else {
            panic!("not a keys answer: {read:?}");
        };
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].columns, ["a", "b"]);

        let foreign = RecordBatch::try_from_iter([
            (c::KEY, Arc::new(Int64Array::from(vec![0])) as ArrayRef),
            (
                c::REFERENCED_TABLE,
                Arc::new(StringArray::from(vec!["users"])) as ArrayRef,
            ),
            (c::COLUMNS, lists(&[&[Some("user_id"), None]])),
            (c::REFERENCED_COLUMNS, lists(&[&[None, Some("id")]])),
        ])
        .unwrap();
        let read = HostLayer
            .read(&describe("foreign_keys"), &host(), &[foreign])
            .unwrap();
        let CatalogAnswer::ForeignKeys(found) = read else {
            panic!("not a foreign-keys answer: {read:?}");
        };
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].columns, ["user_id"]);
        assert_eq!(found[0].referenced_columns, ["id"]);
    });
}

#[test]
fn the_memory_provider_answers_column_ranges_over_every_row() {
    gated(
        "the_memory_provider_answers_column_ranges_over_every_row",
        || {
            assert!(
                Memory::capabilities()
                    .optional
                    .contains(&OptionalRequest::ColumnRanges)
            );
            // 3,000 rows, the first 2,000 with no time: past any preview.
            let duck = duckdb::Connection::open_in_memory().unwrap();
            duck.execute_batch(
                "CREATE TABLE events AS SELECT i::BIGINT AS event_id, \
                 CASE WHEN i >= 2000 THEN TIMESTAMP '2024-01-01' + to_days(i::INTEGER) END AS ts \
                 FROM range(3000) t(i)",
            )
            .unwrap();
            let memory = Memory::from_duckdb(&duck).unwrap();
            let request: DataRequest = serde_json::from_value(json!({
                "request": "column_ranges",
                "table": {"source": ["events"]},
                "columns": ["ts"],
            }))
            .unwrap();
            let answer = memory.answer(&request, &BTreeMap::new()).unwrap();
            assert_eq!(answer.num_rows(), 1);
            let day = 86_400_000_000_i64;
            let first = 1_704_067_200_000_000_i64 + 2000 * day;
            let last = 1_704_067_200_000_000_i64 + 2999 * day;
            let mut times = Vec::new();
            let mut counts = Vec::new();
            for column in answer.columns() {
                match column.data_type() {
                    DataType::Timestamp(TimeUnit::Microsecond, _) => {
                        times.push(column.as_primitive::<TimestampMicrosecondType>().value(0));
                    }
                    DataType::Int64 => counts.push(column.as_primitive::<Int64Type>().value(0)),
                    _ => {}
                }
            }
            assert_eq!(times, [first, last], "{answer:?}");
            assert!(counts.contains(&3000), "{answer:?}");
        },
    );
}
