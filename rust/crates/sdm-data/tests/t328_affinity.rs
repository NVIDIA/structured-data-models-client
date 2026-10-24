// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQLite holds no column types, only affinities (T3.28 fix 3): a column's
//! declared type steers how a value is stored, never which values it
//! holds, so the old client takes a caller's type over any SQLite column
//! (`SQLiteTable._to_dtype` types a column by its affinity alone, and its
//! `_sanitize` reads the values as the caller's type: a time as
//! `to_datetime` reads text). Plan B's check of a caller's type against
//! the type the source holds (T3.4) asks the dialect whether its source
//! holds one; every other dialect does.

use sdm_contract::data::{DataLayer, ProviderChoice};
use sdm_data::Data;

#[test]
fn only_sqlite_holds_no_column_type_a_caller_s_type_must_match() {
    let data = Data::new();
    for (dialect, holds) in [
        ("sqlite", false),
        ("duckdb", true),
        ("postgres", true),
        ("databricks", true),
    ] {
        assert_eq!(
            data.holds_column_types(&ProviderChoice::sql(dialect)),
            holds,
            "{dialect}"
        );
    }
}

#[test]
fn a_host_provider_s_columns_hold_their_types() {
    let data = Data::new();
    let host = ProviderChoice::Host {
        capabilities: data.capabilities(&ProviderChoice::sql("duckdb")).unwrap(),
    };
    assert!(data.holds_column_types(&host));
}
