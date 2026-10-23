// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core's sampling for a provider sized as the old client's warehouse
//! samplers are (T3.4 fix 1, round 1's P1-4): base.binary materialized
//! over the non-SQL test provider declaring `sample_rows` and
//! `neighbor_lookback` (contract v28), each request it was asked logged.
//!
//! - every table's rows counted once, in the graph's order, before the
//!   entity sample, which names its table's rows (`num_rows_dict`, the
//!   `TABLESAMPLE` percentage's denominator);
//! - each neighbor read by a foreign key over a table with a time column
//!   bounded below by a `start` seed column, exclusive, beside the anchor
//!   (`_by_fkey`'s look-back window);
//! - a provider declaring neither is asked no row count and no look-back.

use std::path::PathBuf;

use sdm_contract::data::{DataRequest, SeedBound};
use sdm_host::default_root;
use sdm_testkit::bases;
use sdm_testkit::case;
use sdm_testkit::duck;
use sdm_testkit::host;
use sdm_testkit::memory::Memory;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// base.binary's tables, copied into the test provider, declaring the
/// warehouse samplers' sizing where `warehouse`.
fn memory(warehouse: bool) -> (case::Case, Memory) {
    let path = repository().join("contract/cases/s1/base.binary.json");
    let case = case::load(&[path]).unwrap().remove(0).case;
    let bases = bases::load(&repository().join("contract/bases.yaml")).unwrap();
    let connection = duck::open().unwrap();
    for statement in &bases[&case.input.base] {
        connection.execute_batch(statement).unwrap();
    }
    let mut memory = Memory::from_duckdb(&connection).unwrap();
    if warehouse {
        let mut capabilities = Memory::capabilities();
        capabilities.sample_rows = true;
        capabilities.neighbor_lookback = true;
        memory.declared = Some(capabilities);
    }
    (case, memory)
}

fn asked(memory: &Memory) -> Vec<DataRequest> {
    memory.asked.lock().unwrap().clone()
}

#[test]
fn a_warehouse_sized_provider_is_asked_each_table_s_rows_then_a_sample_naming_them() {
    let (case, memory) = memory(true);
    host::run_memory_over(&case, &memory, &default_root()).unwrap();
    let asked = asked(&memory);
    let first_sample = asked
        .iter()
        .position(|r| matches!(r, DataRequest::SampleEntities { .. }))
        .expect("a sample was asked");
    let counted: Vec<&str> = asked[..first_sample]
        .iter()
        .filter_map(|r| match r {
            DataRequest::CountRows { table, .. } => table.single(),
            _ => None,
        })
        .collect();
    let tables: Vec<&str> = memory.tables.iter().map(|t| t.name.as_str()).collect();
    let mut sorted_counted = counted.clone();
    sorted_counted.sort_unstable();
    let mut sorted_tables = tables.clone();
    sorted_tables.sort_unstable();
    assert_eq!(sorted_counted, sorted_tables, "each table counted once");
    let DataRequest::SampleEntities { table, rows, .. } = &asked[first_sample] else {
        unreachable!()
    };
    let held = memory
        .tables
        .iter()
        .find(|t| Some(t.name.as_str()) == table.single())
        .unwrap();
    assert_eq!(*rows, Some(held.rows.num_rows() as u64));
    // Counted once for the materializer: no count after the sample.
    assert!(
        !asked[first_sample..]
            .iter()
            .any(|r| matches!(r, DataRequest::CountRows { .. }))
    );
}

#[test]
fn a_warehouse_sized_provider_s_neighbor_reads_by_a_foreign_key_look_back() {
    let (case, memory) = memory(true);
    host::run_memory_over(&case, &memory, &default_root()).unwrap();
    let windowed: Vec<DataRequest> = asked(&memory)
        .into_iter()
        .filter(|r| {
            matches!(
                r,
                DataRequest::Neighbors {
                    per_seed: Some(_),
                    window: Some(_),
                    ..
                }
            )
        })
        .collect();
    assert!(
        !windowed.is_empty(),
        "no neighbor read by a foreign key over a timed table"
    );
    for read in windowed {
        let DataRequest::Neighbors {
            window: Some(window),
            seeds,
            ..
        } = read
        else {
            unreachable!()
        };
        assert_eq!(
            window.lower,
            Some(SeedBound {
                seed_column: "start".into(),
                inclusive: false,
            })
        );
        assert!(seeds.columns.iter().any(|c| c.name == "start"));
    }
}

#[test]
fn a_provider_sized_otherwise_is_asked_no_row_count_and_no_look_back() {
    let (case, memory) = memory(false);
    host::run_memory_over(&case, &memory, &default_root()).unwrap();
    let asked = asked(&memory);
    assert!(
        !asked
            .iter()
            .any(|r| matches!(r, DataRequest::CountRows { .. }))
    );
    assert!(!asked.iter().any(|r| matches!(
        r,
        DataRequest::Neighbors { window: Some(w), per_seed: Some(_), .. } if w.lower.is_some()
    )));
    assert!(
        asked
            .iter()
            .any(|r| matches!(r, DataRequest::SampleEntities { rows: None, .. }))
    );
}
