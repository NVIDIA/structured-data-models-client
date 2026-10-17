// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The time and feature family's DuckDB statements (T1.32e, T1.30 folded
//! in), pinned as text beside `duckdb_catalog.rs`'s: each data request the
//! `time.*` and `feature.*` cases make through `from_source` and
//! `materialize` on the pure-Rust host that no other file here pins
//! already, as `sdm-testkit` recorded it, with the one statement this layer
//! renders for it. A statement that changes without its pin changing here
//! fails this test, and `sdm-testkit`'s `tests/t132e_time_feature_sql.rs`
//! runs the cases on DuckDB and fails on any statement they run that is
//! pinned nowhere in these files. The family's own statements are those
//! naming its columns: `time.text-declared`'s time column `happened` (its
//! bounds, preview and both neighbor reads) and
//! `time.feature-date-and-timestamp-with-null`'s `f_date` and `f_ts` (the
//! preview, the entities sampled with `f_date` as the time column, and the
//! lookup).

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;

/// Each request, as the contract writes it, and its statement.
const PINS: &[(&str, &str)] = &[
    (
        r#"{"columns": [{"column": "happened", "table": {"source": ["orders"]}}], "request": "time_bounds"}"#,
        r#"SELECT
  0 as table_index,
  MIN(CAST("happened" AS TIMESTAMP)) as min_date,
  MAX(CAST("happened" AS TIMESTAMP)) as max_date
FROM "orders""#,
    ),
    (
        r#"{"columns": ["order_id", "user_id", "price", "happened"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"#,
        r#"SELECT "order_id", "user_id", "price", "happened" FROM "orders" LIMIT 1000"#,
    ),
    (
        r#"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"#,
        r#"SELECT "user_id", "age", "country", "f_date", "f_ts" FROM "users" LIMIT 1000"#,
    ),
    (
        r#"{"columns": ["user_id", "f_date"], "filters": [{"column": "user_id", "filter": "not_null"}, {"column": "f_date", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"#,
        r#"SELECT "user_id", "f_date"
FROM (SELECT * FROM "users"
WHERE "user_id" IS NOT NULL AND "f_date" IS NOT NULL)
USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)"#,
    ),
    (
        r#"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."user_id", t."age", t."country", t."f_date", t."f_ts"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."user_id" = s."key"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
    ),
    (
        r#"{"columns": ["order_id", "user_id", "price", "happened"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "happened", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."order_id", t."user_id", t."price", t."happened"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = s."key"
 AND CAST(t."happened" AS TIMESTAMP) <= s."time"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
  ORDER BY CAST(t."happened" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST
) <= 32
ORDER BY s."batch", CAST(t."happened" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST"#,
    ),
    (
        r#"{"columns": ["price", "happened"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."price", t."happened"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = s."key"
 AND CAST(t."happened" AS TIMESTAMP) <= s."end"
 AND CAST(t."happened" AS TIMESTAMP) > s."start""#,
    ),
];

#[test]
fn each_request_of_the_time_and_feature_family_renders_its_pinned_statement() {
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
