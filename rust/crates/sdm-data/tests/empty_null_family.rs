// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The empty, null and refuse family's DuckDB statements (T1.32d, T1.30
//! folded in), pinned as text beside `duckdb_catalog.rs`'s: each data
//! request the `empty.*`, `null.*` and `refuse.*` cases make through
//! `from_source` and `materialize` on the pure-Rust host that no other file
//! here pins already, as `sdm-testkit` recorded it, with the one statement
//! this layer renders for it. A statement that changes without its pin
//! changing here fails this test. The family's nulls and empty tables
//! change no statement's text, only what it reads: the one statement of its
//! own is the lookup of `null.feature-all-null`'s users, which reads the
//! all-null column `f` with the rest.

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;

/// Each request, as the contract writes it, and its statement.
const PINS: &[(&str, &str)] = &[(
    r#"{"columns": ["user_id", "age", "country", "f"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"#,
    r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."user_id", t."age", t."country", t."f"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."user_id" = s."key"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
)];

#[test]
fn each_request_of_the_empty_null_family_renders_its_pinned_statement() {
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
        namespace: Vec::new(),
        view: None,
    };
    assert!(!PINS.is_empty());
    for (written, pinned) in PINS {
        let request: DataRequest = serde_json::from_str(written).expect("a data request");
        let FetchPlan::Sql { statements, .. } = Data::new().plan(&request, &duckdb).unwrap() else {
            panic!("not SQL: {written}");
        };
        assert_eq!(statements.len(), 1, "{written}");
        assert_eq!(statements[0].text, *pinned, "{written}");
    }
}
