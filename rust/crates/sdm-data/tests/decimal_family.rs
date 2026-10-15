// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The decimal family's DuckDB statements (T1.32b, T1.30 folded in), pinned
//! as text beside `duckdb_catalog.rs`'s: each data request the `decimal.*`
//! cases make through `from_source` and `materialize` on the pure-Rust host
//! that no other file here pins already, as `sdm-testkit` recorded it, with
//! the one statement this layer renders for it. A statement that changes
//! without its pin changing here fails this test, and `sdm-testkit`'s
//! `tests/t132b_decimal_sql.rs` runs the cases on DuckDB and fails on any
//! statement they run that is pinned nowhere in these files. A decimal
//! column changes no statement's text, only what it reads: the two of its
//! own are `decimal.class-target-38-18`'s preview of `users`, which reads
//! its static target `tier` with the rest, and its entity sample, which
//! reads `tier` beside the key for the labels.

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;

/// Each request, as the contract writes it, and its statement.
const PINS: &[(&str, &str)] = &[
    (
        r#"{"columns": ["user_id", "age", "country", "tier"], "limit": 1000, "request": "preview", "table": "users"}"#,
        r#"SELECT "user_id", "age", "country", "tier" FROM "users" LIMIT 1000"#,
    ),
    (
        r#"{"columns": ["user_id", "tier"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": "users"}"#,
        r#"SELECT "user_id", "tier"
FROM (SELECT * FROM "users"
WHERE "user_id" IS NOT NULL)
USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)"#,
    ),
];

#[test]
fn each_request_of_the_decimal_family_renders_its_pinned_statement() {
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
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
