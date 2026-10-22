// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The keys family's DuckDB statements (T1.32a, T1.30 folded in), pinned
//! as text beside `duckdb_catalog.rs`'s: each data request the `key.*`
//! cases make through `from_source` and `materialize` on the pure-Rust
//! host, as `sdm-testkit` recorded it, with the one statement this layer
//! renders for it. A statement that changes without its pin changing here
//! fails this test, and `sdm-testkit`'s `tests/t132a_keys_sql.rs` runs
//! the cases on DuckDB and fails on any statement they run that is pinned
//! nowhere in these files. The key's type changes no statement but one:
//! its values travel in the key set the statements join, which DuckDB reads
//! in the key's own type; a `decimal128(38, 0)` seed key naming an integer
//! key type the core read from the catalog is cast to that type (T1.33
//! part 3, register P1B-43: no one SQL type compares it exactly with a
//! VARINT and a UHUGEINT key alike), so its statements are pinned beside
//! the narrower decimal's.

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;

/// Each request, as the contract writes it, and its statement.
const PINS: &[(&str, &str)] = &[
    (
        r#"{"part": "tables", "request": "describe", "tables": []}"#,
        r#"SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND database_name = current_database() AND schema_name = current_schema()"#,
    ),
    (
        r#"{"part": "columns", "request": "describe", "tables": [{"source": ["orders"]}]}"#,
        r#"PRAGMA table_info("orders")"#,
    ),
    (
        r#"{"part": "keys", "request": "describe", "tables": [{"source": ["orders"]}]}"#,
        r#"SELECT constraint_type, constraint_column_names
FROM duckdb_constraints()
WHERE table_name = 'orders'
  AND database_name = current_database() AND schema_name = current_schema()"#,
    ),
    (
        r#"{"columns": ["order_id", "user_id", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"#,
        r#"SELECT "order_id", "user_id", "price", "ts" FROM "orders" LIMIT 1000"#,
    ),
    (
        r#"{"part": "columns", "request": "describe", "tables": [{"source": ["users"]}]}"#,
        r#"PRAGMA table_info("users")"#,
    ),
    (
        r#"{"part": "keys", "request": "describe", "tables": [{"source": ["users"]}]}"#,
        r#"SELECT constraint_type, constraint_column_names
FROM duckdb_constraints()
WHERE table_name = 'users'
  AND database_name = current_database() AND schema_name = current_schema()"#,
    ),
    (
        r#"{"columns": ["user_id", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"#,
        r#"SELECT "user_id", "age", "country" FROM "users" LIMIT 1000"#,
    ),
    (
        r#"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["orders"]}]}"#,
        r#"SELECT constraint_index, referenced_table,
       constraint_column_names, referenced_column_names
FROM duckdb_constraints()
WHERE table_name = 'orders'
  AND database_name = current_database() AND schema_name = current_schema()
  AND constraint_type = 'FOREIGN KEY'"#,
    ),
    (
        r#"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["users"]}]}"#,
        r#"SELECT constraint_index, referenced_table,
       constraint_column_names, referenced_column_names
FROM duckdb_constraints()
WHERE table_name = 'users'
  AND database_name = current_database() AND schema_name = current_schema()
  AND constraint_type = 'FOREIGN KEY'"#,
    ),
    (
        r#"{"keys": [{"columns": ["order_id"], "table": {"source": ["orders"]}}, {"columns": ["user_id"], "table": {"source": ["users"]}}], "request": "key_counts"}"#,
        r#"SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT "order_id") AS d, COUNT(CASE WHEN "order_id" IS NULL THEN 1 END) AS z
FROM "orders"
UNION ALL
SELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT "user_id") AS d, COUNT(CASE WHEN "user_id" IS NULL THEN 1 END) AS z
FROM "users""#,
    ),
    (
        r#"{"part": "indexes", "request": "describe", "tables": []}"#,
        r#"SELECT table_name, expressions FROM duckdb_indexes()"#,
    ),
    (
        r#"{"columns": [{"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"#,
        r#"SELECT
  0 as table_index,
  MIN(CAST("ts" AS TIMESTAMP)) as min_date,
  MAX(CAST("ts" AS TIMESTAMP)) as max_date
FROM "orders""#,
    ),
    (
        r#"{"columns": ["user_id"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"#,
        r#"SELECT "user_id"
FROM (SELECT * FROM "users"
WHERE "user_id" IS NOT NULL)
USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)"#,
    ),
    (
        r#"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."price", t."ts"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = s."key"
 AND CAST(t."ts" AS TIMESTAMP) <= s."end"
 AND CAST(t."ts" AS TIMESTAMP) > s."start""#,
    ),
    (
        r#"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."user_id", t."age", t."country"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."user_id" = s."key"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
    ),
    (
        r#"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."order_id", t."user_id", t."price", t."ts"
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
    (
        r#"{"columns": ["oid", "buyer", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"#,
        r#"SELECT "oid", "buyer", "price", "ts" FROM "orders" LIMIT 1000"#,
    ),
    (
        r#"{"columns": ["uid", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"#,
        r#"SELECT "uid", "age", "country" FROM "users" LIMIT 1000"#,
    ),
    (
        r#"{"columns": ["uid"], "filters": [], "key": ["uid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"#,
        r#"SELECT "uid"
FROM (SELECT * FROM "users")
USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)"#,
    ),
    (
        r#"{"columns": ["price", "ts"], "key": ["buyer"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."price", t."ts"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."buyer" = s."key"
 AND CAST(t."ts" AS TIMESTAMP) <= s."end"
 AND CAST(t."ts" AS TIMESTAMP) > s."start""#,
    ),
    (
        r#"{"columns": ["uid", "age", "country"], "key": ["uid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."uid", t."age", t."country"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."uid" = s."key"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
    ),
    (
        r#"{"columns": ["oid", "buyer", "price", "ts"], "key": ["buyer"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."oid", t."buyer", t."price", t."ts"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."buyer" = s."key"
 AND CAST(t."ts" AS TIMESTAMP) <= s."time"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
  ORDER BY CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."oid" ASC NULLS LAST
) <= 32
ORDER BY s."batch", CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."oid" ASC NULLS LAST"#,
    ),
    (
        r#"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"metadata": {"sdm:key_declared_type": "HUGEINT"}, "name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."price", t."ts"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = TRY_CAST(s."key" AS HUGEINT)
 AND CAST(t."ts" AS TIMESTAMP) <= s."end"
 AND CAST(t."ts" AS TIMESTAMP) > s."start""#,
    ),
    (
        r#"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_declared_type": "HUGEINT"}, "name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", COUNT(*) OVER (PARTITION BY s."batch") AS "__kumo_matches__", t."user_id", t."age", t."country"
FROM "__sdm_seeds__" s
JOIN "users" t
  ON t."user_id" = TRY_CAST(s."key" AS HUGEINT)
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
) = 1"#,
    ),
    (
        r#"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"metadata": {"sdm:key_declared_type": "HUGEINT"}, "name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"#,
        r#"SELECT s."batch" AS "__kumo_batch__", t."order_id", t."user_id", t."price", t."ts"
FROM "__sdm_seeds__" s
JOIN "orders" t
  ON t."user_id" = TRY_CAST(s."key" AS HUGEINT)
 AND CAST(t."ts" AS TIMESTAMP) <= s."time"
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s."batch"
  ORDER BY CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST
) <= 32
ORDER BY s."batch", CAST(t."ts" AS TIMESTAMP) DESC NULLS LAST, t."order_id" ASC NULLS LAST"#,
    ),
];

#[test]
fn each_request_of_the_keys_family_renders_its_pinned_statement() {
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
        namespace: Vec::new(),
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
