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
//!
//! T2.0c's exact-decimal cases add seven more, as `sdm-testkit` recorded
//! them: the previews of the tables their decimal columns (`d`, `f` and
//! `g`, `amount`) widen, the entity samples of their static targets,
//! and `decimal.sum-10-2`'s and `decimal.feature-inexact-38-18`'s
//! neighbour and lookup reads over `amount`, `f` and `g`.

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
    // decimal.sum-10-2 (T2.0c)
    (
        r#"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "limit": 1000, "request": "preview", "table": "orders"}"#,
        r#"SELECT "order_id", "user_id", "price", "ts", "amount" FROM "orders" LIMIT 1000"#,
    ),
    // decimal.compare-finer-gt (T2.0c)
    (
        r#"{"columns": ["user_id", "age", "country", "d"], "limit": 1000, "request": "preview", "table": "users"}"#,
        r#"SELECT "user_id", "age", "country", "d" FROM "users" LIMIT 1000"#,
    ),
    // decimal.feature-inexact-38-18 (T2.0c)
    (
        r#"{"columns": ["user_id", "age", "country", "f", "g"], "limit": 1000, "request": "preview", "table": "users"}"#,
        r#"SELECT "user_id", "age", "country", "f", "g" FROM "users" LIMIT 1000"#,
    ),
    // decimal.compare-finer-gt (T2.0c)
    (
        r#"{"columns": ["user_id", "d"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": "users"}"#,
        r#"SELECT "user_id", "d"
FROM (SELECT * FROM "users"
WHERE "user_id" IS NOT NULL)
USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)"#,
    ),
    // decimal.feature-inexact-38-18 (T2.0c)
    (
        r#"{"columns": ["user_id", "age", "country", "f", "g"], "key": "user_id", "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": "users"}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."user_id", t."age", t."country", t."f", t."g"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."user_id" = s."key"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
    ),
    // decimal.sum-10-2 (T2.0c)
    (
        r#"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "key": "user_id", "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": "orders", "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."order_id", t."user_id", t."price", t."ts", t."amount"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = s."key"
 AND CAST(t."ts" AS TIMESTAMP) <= s."time"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
  ORDER BY CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST
) <= 32
ORDER BY s."batch", CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST"#,
    ),
    // decimal.sum-10-2 (T2.0c)
    (
        r#"{"columns": ["ts", "amount"], "key": "user_id", "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": "orders", "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."ts", t."amount"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = s."key"
 AND CAST(t."ts" AS TIMESTAMP) <= s."end"
 AND CAST(t."ts" AS TIMESTAMP) > s."start""#,
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
