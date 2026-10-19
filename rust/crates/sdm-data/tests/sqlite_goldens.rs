// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4 on SQLite (T2.2; checked against the run since T2.4): every statement
//! the SQLite dialect rendered for the contract cases and the key-pair
//! vectors on SQLite, on both core hosts, pinned verbatim beside the
//! request it answers. Each request is planned here and must render exactly
//! its statements, so a change of the SQL without a golden update fails;
//! and every fetch python-core's run of the cases on SQLite made in CI
//! (`contract-python (ref-pandas3, sqlite)`, which executed each statement
//! on SQLite, its result held to the reference's by the comparator) is a
//! golden, checked in `l4-goldens (sqlite)` ([`goldens`]). Collected from
//! those runs' recorded fetches (`core.fetches`) when the dialect changes,
//! and reviewed as any golden.

mod goldens;

/// Each request, as the contract writes it, and the statements its plan
/// carries on SQLite, in order.
const GOLDENS: goldens::Goldens = &[
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &["PRAGMA foreign_key_list(\"addresses\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &["PRAGMA foreign_key_list(\"adventureworks_customers\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &["PRAGMA foreign_key_list(\"adventureworks_products\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &["PRAGMA foreign_key_list(\"adventureworks_sales_order_details\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &["PRAGMA foreign_key_list(\"adventureworks_sales_order_headers\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &["PRAGMA foreign_key_list(\"buyers\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &["PRAGMA foreign_key_list(\"customers\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &["PRAGMA foreign_key_list(\"details\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &["PRAGMA foreign_key_list(\"headers\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &["PRAGMA foreign_key_list(\"lines\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA foreign_key_list(\"orders\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &["PRAGMA foreign_key_list(\"products\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &["PRAGMA foreign_key_list(\"stores\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA foreign_key_list(\"users\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &["PRAGMA index_list(\"addresses\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_customers\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_products\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_sales_order_details\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_sales_order_headers\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &["PRAGMA index_list(\"buyers\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &["PRAGMA index_list(\"customers\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &["PRAGMA index_list(\"details\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &["PRAGMA index_list(\"headers\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &["PRAGMA index_list(\"lines\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA index_list(\"orders\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &["PRAGMA index_list(\"products\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &["PRAGMA index_list(\"stores\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA index_list(\"users\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &["PRAGMA table_info(\"addresses\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &["PRAGMA table_info(\"adventureworks_customers\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &["PRAGMA table_info(\"adventureworks_products\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &["PRAGMA table_info(\"adventureworks_sales_order_details\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &["PRAGMA table_info(\"adventureworks_sales_order_headers\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &["PRAGMA table_info(\"buyers\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &["PRAGMA table_info(\"customers\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &["PRAGMA table_info(\"details\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &["PRAGMA table_info(\"headers\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &["PRAGMA table_info(\"lines\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA table_info(\"orders\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &["PRAGMA table_info(\"products\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &["PRAGMA table_info(\"stores\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA table_info(\"users\")"],
    ),
    (
        r##"{"part": "tables", "request": "describe", "tables": []}"##,
        &["SELECT name FROM sqlite_master WHERE type='table'"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_customers\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_products\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_sales_order_details\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &["PRAGMA index_list(\"adventureworks_sales_order_headers\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA index_list(\"orders\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA index_list(\"users\")"],
    ),
    (
        r##"{"keys": [{"columns": ["CustomerID"], "table": {"source": ["adventureworks_customers"]}}, {"columns": ["ProductID"], "table": {"source": ["adventureworks_products"]}}, {"columns": ["SalesOrderDetailID"], "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"CustomerID\") AS d, COUNT(CASE WHEN \"CustomerID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_customers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"ProductID\") AS d, COUNT(CASE WHEN \"ProductID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_products\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"SalesOrderDetailID\") AS d, COUNT(CASE WHEN \"SalesOrderDetailID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["aid"], "table": {"source": ["addresses"]}}, {"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["lid"], "table": {"source": ["lines"]}}, {"columns": ["oid"], "table": {"source": ["orders"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"aid\") AS d, COUNT(CASE WHEN \"aid\" IS NULL THEN 1 END) AS z\nFROM \"addresses\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"lid\") AS d, COUNT(CASE WHEN \"lid\" IS NULL THEN 1 END) AS z\nFROM \"lines\"\nUNION ALL\nSELECT 3 AS k, COUNT(*) AS n, COUNT(DISTINCT \"oid\") AS d, COUNT(CASE WHEN \"oid\" IS NULL THEN 1 END) AS z\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["aid"], "table": {"source": ["addresses"]}}, {"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["oid"], "table": {"source": ["orders"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"aid\") AS d, COUNT(CASE WHEN \"aid\" IS NULL THEN 1 END) AS z\nFROM \"addresses\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"oid\") AS d, COUNT(CASE WHEN \"oid\" IS NULL THEN 1 END) AS z\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["cid"], "table": {"source": ["buyers"]}}, {"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["oid"], "table": {"source": ["orders"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"buyers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"oid\") AS d, COUNT(CASE WHEN \"oid\" IS NULL THEN 1 END) AS z\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["cid"], "table": {"source": ["buyers"]}}, {"columns": ["cid"], "table": {"source": ["customers"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"buyers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["did"], "table": {"source": ["details"]}}, {"columns": ["hid"], "table": {"source": ["headers"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"did\") AS d, COUNT(CASE WHEN \"did\" IS NULL THEN 1 END) AS z\nFROM \"details\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"hid\") AS d, COUNT(CASE WHEN \"hid\" IS NULL THEN 1 END) AS z\nFROM \"headers\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["hid"], "table": {"source": ["headers"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"hid\") AS d, COUNT(CASE WHEN \"hid\" IS NULL THEN 1 END) AS z\nFROM \"headers\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["cid"], "table": {"source": ["customers"]}}, {"columns": ["oid"], "table": {"source": ["orders"]}}, {"columns": ["pid"], "table": {"source": ["products"]}}, {"columns": ["sid"], "table": {"source": ["stores"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"cid\") AS d, COUNT(CASE WHEN \"cid\" IS NULL THEN 1 END) AS z\nFROM \"customers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"oid\") AS d, COUNT(CASE WHEN \"oid\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"pid\") AS d, COUNT(CASE WHEN \"pid\" IS NULL THEN 1 END) AS z\nFROM \"products\"\nUNION ALL\nSELECT 3 AS k, COUNT(*) AS n, COUNT(DISTINCT \"sid\") AS d, COUNT(CASE WHEN \"sid\" IS NULL THEN 1 END) AS z\nFROM \"stores\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["order_id"], "table": {"source": ["orders"]}}, {"columns": ["user_id"], "table": {"source": ["users"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_customers\" WHERE \"CustomerID\" == s.\"key\") AS \"__kumo_matches__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_f613941cc9692445__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_products\" WHERE \"ProductID\" == s.\"key\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_34ab0cadb26f9a15__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_products\" WHERE \"ProductID\" == s.\"key\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_531ddf2689261dd9__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\"\nFROM temp.\"__sdm_seeds_80f465c089dc6cc7__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_0f8c101f6aa2042c__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_2ee1b62316397eae__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "float64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_6bb6a57457e04c89__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_dccb91ba8f587019__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\"\nFROM temp.\"__sdm_seeds_44e5f787c236f710__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\", t.\"g\"\nFROM temp.\"__sdm_seeds_521de18bd7093c5f__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f_date\", t.\"f_ts\"\nFROM temp.\"__sdm_seeds_e46ce5d2c8b6773f__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_b5005b990adfc5b8__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_d3486eae0a069df1__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_ca2778a1aa37b882__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"aid\", t.\"city\", t.\"tag\"\nFROM temp.\"__sdm_seeds_8c86de7a897c4696__\" s\nJOIN \"addresses\" t\n  ON t.\"aid\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"addresses\" WHERE \"aid\" == s.\"key\") AS \"__kumo_matches__\", t.\"aid\", t.\"city\", t.\"tag\"\nFROM temp.\"__sdm_seeds_52e9c370f77751e2__\" s\nJOIN \"addresses\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"addresses\"\n    WHERE \"aid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["aid", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"addresses\" WHERE \"aid\" == s.\"key\") AS \"__kumo_matches__\", t.\"aid\", t.\"tag\"\nFROM temp.\"__sdm_seeds_0c19054dd75f7b56__\" s\nJOIN \"addresses\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"addresses\"\n    WHERE \"aid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"buyers\" WHERE \"cid\" == s.\"key\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_1b714f319225d9d8__\" s\nJOIN \"buyers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"buyers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"customers\" WHERE \"cid\" == s.\"key\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_ec46213a60d8279d__\" s\nJOIN \"customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"customers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_1986c936fe713ad2__\" s\nJOIN \"buyers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"buyers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_4515f57e455ec979__\" s\nJOIN \"customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"customers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_3294f14ccac0db3d__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_9542387fdea30ebb__\" s\nJOIN \"customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"customers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["cid", "seg"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"customers\" WHERE \"cid\" == s.\"key\") AS \"__kumo_matches__\", t.\"cid\", t.\"seg\"\nFROM temp.\"__sdm_seeds_bbad3ddf0df06d3b__\" s\nJOIN \"customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"customers\"\n    WHERE \"cid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["hid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["headers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_a5f0d3ecaeb32cba__\" s\nJOIN \"headers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"headers\"\n    WHERE \"hid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"products\" WHERE \"pid\" == s.\"key\") AS \"__kumo_matches__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM temp.\"__sdm_seeds_7b749117acaeeeda__\" s\nJOIN \"products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"products\"\n    WHERE \"pid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM temp.\"__sdm_seeds_6871999bf17091cc__\" s\nJOIN \"products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"products\"\n    WHERE \"pid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM temp.\"__sdm_seeds_41ce1d0d69f89623__\" s\nJOIN \"products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"products\"\n    WHERE \"pid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"stores\" WHERE \"sid\" == s.\"key\") AS \"__kumo_matches__\", t.\"sid\", t.\"size\", t.\"tag\"\nFROM temp.\"__sdm_seeds_c5bde6793459d8ce__\" s\nJOIN \"stores\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"stores\"\n    WHERE \"sid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"sid\", t.\"size\", t.\"tag\"\nFROM temp.\"__sdm_seeds_a969a6fa56cbd2af__\" s\nJOIN \"stores\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"stores\"\n    WHERE \"sid\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "key": ["uid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"uid\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_4147eb4816a8d16e__\" s\nJOIN \"users\" t\n  ON t.\"uid\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "key": ["CustomerID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_7ab73cffa59df4a2__\" s\nJOIN \"adventureworks_sales_order_headers\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND t.\"OrderDate\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"OrderDate\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["OrderQty", "OrderDate"], "key": ["ProductID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderQty\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_13fd732bf8ff7595__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND t.\"OrderDate\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"OrderDate\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "ModifiedDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "AccountNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "BillToAddressID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Comment", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CreditCardApprovalCode", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CreditCardID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CurrencyRateID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CustomerID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "DueDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Freight", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ModifiedDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "OnlineOrderFlag", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "PurchaseOrderNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "RevisionNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesPersonID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipMethodID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipToAddressID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Status", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SubTotal", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TaxAmt", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TerritoryID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TotalDue", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"RevisionNumber\", t.\"OrderDate\", t.\"DueDate\", t.\"ShipDate\", t.\"Status\", t.\"OnlineOrderFlag\", t.\"SalesOrderNumber\", t.\"PurchaseOrderNumber\", t.\"AccountNumber\", t.\"CustomerID\", t.\"SalesPersonID\", t.\"TerritoryID\", t.\"BillToAddressID\", t.\"ShipToAddressID\", t.\"ShipMethodID\", t.\"CreditCardID\", t.\"CreditCardApprovalCode\", t.\"CurrencyRateID\", t.\"SubTotal\", t.\"TaxAmt\", t.\"Freight\", t.\"TotalDue\", t.\"Comment\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_6872696b92d75d28__\" s\nJOIN \"adventureworks_sales_order_headers\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_headers\"\n  WHERE \"CustomerID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"AccountNumber\", \"BillToAddressID\", \"Comment\", \"CreditCardApprovalCode\", \"CreditCardID\", \"CurrencyRateID\", \"CustomerID\", \"DueDate\", \"Freight\", \"ModifiedDate\", \"OnlineOrderFlag\", \"PurchaseOrderNumber\", \"RevisionNumber\", \"SalesOrderID\", \"SalesOrderNumber\", \"SalesPersonID\", \"ShipDate\", \"ShipMethodID\", \"ShipToAddressID\", \"Status\", \"SubTotal\", \"TaxAmt\", \"TerritoryID\", \"TotalDue\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"AccountNumber\", t.\"BillToAddressID\", t.\"Comment\", t.\"CreditCardApprovalCode\", t.\"CreditCardID\", t.\"CurrencyRateID\", t.\"CustomerID\", t.\"DueDate\", t.\"Freight\", t.\"ModifiedDate\", t.\"OnlineOrderFlag\", t.\"PurchaseOrderNumber\", t.\"RevisionNumber\", t.\"SalesOrderID\", t.\"SalesOrderNumber\", t.\"SalesPersonID\", t.\"ShipDate\", t.\"ShipMethodID\", t.\"ShipToAddressID\", t.\"Status\", t.\"SubTotal\", t.\"TaxAmt\", t.\"TerritoryID\", t.\"TotalDue\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_06cd748ba7d62b2e__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"CustomerID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_452420de76cc9d39__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"ProductID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_736c1f4771f9b220__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"ProductID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_cbaac5239dff16c0__\" s\nJOIN \"customers\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"customers\"\n  WHERE \"aid\" = s.\"key\"\n  ORDER BY \"cid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"cid\"",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM temp.\"__sdm_seeds_d8edbc08965d49f1__\" s\nJOIN \"customers\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"customers\"\n  WHERE \"aid\" = s.\"key\"\n  ORDER BY \"cid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"cid\"",
        ],
    ),
    (
        r##"{"columns": ["did", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_97610f73eab6b892__\" s\nJOIN \"details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"details\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"cid\", \"did\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"cid\", t.\"did\"",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_144effdd4926c632__\" s\nJOIN \"details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"details\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"cid\", \"did\", \"hid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"cid\", t.\"did\", t.\"hid\"",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_32733c8906a7a82d__\" s\nJOIN \"details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"details\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"did\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"did\"",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_6985cf8c5add6346__\" s\nJOIN \"details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"details\"\n  WHERE \"hid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"cid\", \"did\", \"hid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"cid\", t.\"did\", t.\"hid\"",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_c84a3c1c63dd4921__\" s\nJOIN \"details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"details\"\n  WHERE \"hid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"did\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"did\"",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["headers"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM temp.\"__sdm_seeds_cbac9e10a143e307__\" s\nJOIN \"headers\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"headers\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"hid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"hid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "key": ["buyer"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"buyer\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_f27296d1591af8b8__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"buyer\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["aid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM temp.\"__sdm_seeds_4a20a4b4d4c84164__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"aid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM temp.\"__sdm_seeds_169810670be1e1e7__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM temp.\"__sdm_seeds_a65b8c82bd1fbc71__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"cid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["pid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM temp.\"__sdm_seeds_b8d98ed4f58a85d8__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"pid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["sid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM temp.\"__sdm_seeds_b13b7d349be62697__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"sid\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "day", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"day\"\nFROM temp.\"__sdm_seeds_5229d63177ed19bf__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"day\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"day\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"day\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "happened", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"happened\"\nFROM temp.\"__sdm_seeds_f8ceb4c0d9726260__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"happened\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"happened\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"happened\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "01/01/2023"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_0c4b24cbdbf3d60c__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_307cfada9ea8ae8e__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8) || '+00:00'\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_49b596eef87b9888__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_591339924e9e364d__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "float64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_6e8351a1e15ef5eb__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_7a40c2e6d02b2ef6__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-08 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_8453feb1e8e62ae9__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_fb05d3994c2f8a20__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["price", "day"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"day\"\nFROM temp.\"__sdm_seeds_e07a8d3b0f9773e3__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"day\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"day\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "happened"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"happened\"\nFROM temp.\"__sdm_seeds_aa45322e9634fa25__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"happened\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"happened\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-08 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-08 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_133484094e2876a7__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "float64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_38635bbd8aa99f80__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_66c99f241c1ac4a5__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "01/01/2023"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "01/01/2023"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_91f4796acc1a55b5__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["buyer"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_bba59a9ea5f66f03__\" s\nJOIN \"orders\" t\n  ON t.\"buyer\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_bd21edb6d02f06e1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_edb8d46170709e81__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8) || '+00:00'\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8) || '+00:00'",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_f016993c1efaca15__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["aid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_11ed4e001309341d__\" s\nJOIN \"orders\" t\n  ON t.\"aid\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1dd4fdea90e69436__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_f2c6870d1c89b9d3__\" s\nJOIN \"orders\" t\n  ON t.\"cid\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["pid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_89edf4f61e4ffb50__\" s\nJOIN \"orders\" t\n  ON t.\"pid\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["sid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2024-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_0696e019c64a58e3__\" s\nJOIN \"orders\" t\n  ON t.\"sid\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_c3aff1e1327eaf9e__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT \"CustomerID\", \"PersonID\", \"StoreID\", \"TerritoryID\", \"AccountNumber\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_customers\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\", \"Name\", \"ProductNumber\", \"MakeFlag\", \"FinishedGoodsFlag\", \"Color\", \"SafetyStockLevel\", \"ReorderPoint\", \"StandardCost\", \"ListPrice\", \"Size\", \"SizeUnitMeasureCode\", \"WeightUnitMeasureCode\", \"Weight\", \"DaysToManufacture\", \"ProductLine\", \"Class\", \"Style\", \"ProductSubcategoryID\", \"ProductModelID\", \"SellStartDate\", \"SellEndDate\", \"DiscontinuedDate\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_products\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_sales_order_headers"]}}"##,
        &[
            "SELECT \"SalesOrderID\", \"RevisionNumber\", \"OrderDate\", \"DueDate\", \"ShipDate\", \"Status\", \"OnlineOrderFlag\", \"SalesOrderNumber\", \"PurchaseOrderNumber\", \"AccountNumber\", \"CustomerID\", \"SalesPersonID\", \"TerritoryID\", \"BillToAddressID\", \"ShipToAddressID\", \"ShipMethodID\", \"CreditCardID\", \"CreditCardApprovalCode\", \"CurrencyRateID\", \"SubTotal\", \"TaxAmt\", \"Freight\", \"TotalDue\", \"Comment\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_sales_order_headers\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "rowguid", "ModifiedDate", "CustomerID", "OrderDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_sales_order_details"]}}"##,
        &[
            "SELECT \"SalesOrderID\", \"SalesOrderDetailID\", \"CarrierTrackingNumber\", \"OrderQty\", \"ProductID\", \"SpecialOfferID\", \"UnitPrice\", \"UnitPriceDiscount\", \"LineTotal\", \"rowguid\", \"ModifiedDate\", \"CustomerID\", \"OrderDate\" FROM \"adventureworks_sales_order_details\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["addresses"]}}"##,
        &["SELECT \"aid\", \"city\", \"tag\" FROM \"addresses\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["buyers"]}}"##,
        &["SELECT \"cid\", \"aid\", \"seg\", \"tag\" FROM \"buyers\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["customers"]}}"##,
        &["SELECT \"cid\", \"aid\", \"seg\", \"tag\" FROM \"customers\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "seg"], "limit": 1000, "request": "preview", "table": {"source": ["customers"]}}"##,
        &["SELECT \"cid\", \"seg\" FROM \"customers\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["did", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["details"]}}"##,
        &["SELECT \"did\", \"cid\", \"ts\" FROM \"details\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["details"]}}"##,
        &["SELECT \"did\", \"hid\", \"cid\", \"ts\" FROM \"details\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["headers"]}}"##,
        &["SELECT \"hid\", \"cid\", \"ts\" FROM \"headers\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["lid", "oid", "qty", "ts", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["lines"]}}"##,
        &[
            "SELECT \"lid\", \"oid\", \"qty\", \"ts\", \"tag\" FROM \"lines\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"oid\", \"buyer\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"oid\", \"cid\", \"aid\", \"pid\", \"sid\", \"ts\", \"tag\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"day\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"happened\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"amount\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["products"]}}"##,
        &["SELECT \"pid\", \"price\", \"tag\" FROM \"products\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["stores"]}}"##,
        &["SELECT \"sid\", \"size\", \"tag\" FROM \"stores\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"uid\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "d"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"d\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"f\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"f\", \"g\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"f_date\", \"f_ts\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "t"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"t\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "tier"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"tier\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["CustomerID"], "filters": [{"column": "CustomerID", "filter": "not_null"}], "key": ["CustomerID"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT \"CustomerID\"\nFROM \"adventureworks_customers\"\nWHERE \"CustomerID\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_customers') WHERE name = 'CustomerID')), 1) = 1 THEN ((((787846414 * ((\"CustomerID\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"CustomerID\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"CustomerID\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_customers') WHERE name = 'CustomerID')), 1) = 1 THEN \"CustomerID\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["ProductID"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\"\nFROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_products') WHERE name = 'ProductID')), 1) = 1 THEN ((((787846414 * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_products') WHERE name = 'ProductID')), 1) = 1 THEN \"ProductID\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Color"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\", \"Color\"\nFROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_products') WHERE name = 'ProductID')), 1) = 1 THEN ((((787846414 * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"ProductID\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('adventureworks_products') WHERE name = 'ProductID')), 1) = 1 THEN \"ProductID\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid", "city"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\", \"city\"\nFROM \"addresses\"\nWHERE \"aid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN ((((787846414 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN \"aid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [], "key": ["aid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\"\nFROM \"addresses\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN ((((787846414 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN \"aid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\"\nFROM \"addresses\"\nWHERE \"aid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN ((((787846414 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"aid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('addresses') WHERE name = 'aid')), 1) = 1 THEN \"aid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT \"cid\"\nFROM \"buyers\"\nWHERE \"cid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('buyers') WHERE name = 'cid')), 1) = 1 THEN ((((787846414 * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('buyers') WHERE name = 'cid')), 1) = 1 THEN \"cid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT \"cid\"\nFROM \"customers\"\nWHERE \"cid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('customers') WHERE name = 'cid')), 1) = 1 THEN ((((787846414 * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"cid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('customers') WHERE name = 'cid')), 1) = 1 THEN \"cid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["pid"], "filters": [{"column": "pid", "filter": "not_null"}], "key": ["pid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT \"pid\"\nFROM \"products\"\nWHERE \"pid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('products') WHERE name = 'pid')), 1) = 1 THEN ((((787846414 * ((\"pid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"pid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"pid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('products') WHERE name = 'pid')), 1) = 1 THEN \"pid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["sid"], "filters": [{"column": "sid", "filter": "not_null"}], "key": ["sid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT \"sid\"\nFROM \"stores\"\nWHERE \"sid\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('stores') WHERE name = 'sid')), 1) = 1 THEN ((((787846414 * ((\"sid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"sid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"sid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('stores') WHERE name = 'sid')), 1) = 1 THEN \"sid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["uid"], "filters": [], "key": ["uid"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"uid\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'uid')), 1) = 1 THEN ((((787846414 * ((\"uid\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"uid\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"uid\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'uid')), 1) = 1 THEN \"uid\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "key_dtype": "string", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 0) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 0) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "country"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"country\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "f_date"], "filters": [{"column": "user_id", "filter": "not_null"}, {"column": "f_date", "filter": "not_null"}], "key": ["user_id"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"f_date\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL AND \"f_date\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "tier"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "key_dtype": "int", "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"tier\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "filters": [{"column": "OrderDate", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["adventureworks_sales_order_details"]}}"##,
        &[
            "SELECT \"OrderDate\" FROM \"adventureworks_sales_order_details\" WHERE \"OrderDate\" IS NOT NULL LIMIT 1",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "filters": [{"column": "OrderDate", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["adventureworks_sales_order_headers"]}}"##,
        &[
            "SELECT \"OrderDate\" FROM \"adventureworks_sales_order_headers\" WHERE \"OrderDate\" IS NOT NULL LIMIT 1",
        ],
    ),
    (
        r##"{"columns": ["day"], "filters": [{"column": "day", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["orders"]}}"##,
        &["SELECT \"day\" FROM \"orders\" WHERE \"day\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns": ["happened"], "filters": [{"column": "happened", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["orders"]}}"##,
        &["SELECT \"happened\" FROM \"orders\" WHERE \"happened\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns": ["ts"], "filters": [{"column": "ts", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["details"]}}"##,
        &["SELECT \"ts\" FROM \"details\" WHERE \"ts\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns": ["ts"], "filters": [{"column": "ts", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["headers"]}}"##,
        &["SELECT \"ts\" FROM \"headers\" WHERE \"ts\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns": ["ts"], "filters": [{"column": "ts", "filter": "not_null"}], "limit": 1, "request": "scan", "table": {"source": ["orders"]}}"##,
        &["SELECT \"ts\" FROM \"orders\" WHERE \"ts\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"OrderDate\") as min_date,\n  MAX(\"OrderDate\") as max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"OrderDate\") as min_date,\n  MAX(\"OrderDate\") as max_date\nFROM \"adventureworks_sales_order_headers\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}, {"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"OrderDate\") as min_date,\n  MAX(\"OrderDate\") as max_date\nFROM \"adventureworks_sales_order_headers\"\nUNION ALL\nSELECT\n  1 as table_index,\n  MIN(\"OrderDate\") as min_date,\n  MAX(\"OrderDate\") as max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "day", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"day\") as min_date,\n  MAX(\"day\") as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "happened", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"happened\") as min_date,\n  MAX(\"happened\") as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["lines"]}}, {"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"lines\"\nUNION ALL\nSELECT\n  1 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &["PRAGMA index_list(\"addresses\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &["PRAGMA index_list(\"buyers\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &["PRAGMA index_list(\"customers\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &["PRAGMA index_list(\"details\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &["PRAGMA index_list(\"headers\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &["PRAGMA index_list(\"lines\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA index_list(\"orders\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &["PRAGMA index_list(\"products\")"],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &["PRAGMA index_list(\"stores\")"],
    ),
    // T2.18 (KO2-36): the name.* cases, a column named with a dot, a
    // backtick or a line feed, each quoted alone as any name is.
    (
        r##"{"columns":["a.b","age","country"],"key":["a.b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"a.b\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_49ee09cb323805e6__\" s\nJOIN \"users\" t\n  ON t.\"a.b\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["a.b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a.b\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns":["a.b"],"filters":[],"key":["a.b"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a.b\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a.b')), 1) = 1 THEN ((((787846414 * ((\"a.b\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"a.b\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"a.b\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a.b')), 1) = 1 THEN \"a.b\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"key":["a\nb"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"a\nb\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_c8dc9e8151385582__\" s\nJOIN \"users\" t\n  ON t.\"a\nb\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a\nb\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns":["a\nb"],"filters":[],"key":["a\nb"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a\nb\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a\nb')), 1) = 1 THEN ((((787846414 * ((\"a\nb\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"a\nb\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"a\nb\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a\nb')), 1) = 1 THEN \"a\nb\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"key":["a`b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"a`b\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_a1c410adc6da787a__\" s\nJOIN \"users\" t\n  ON t.\"a`b\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a`b\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns":["a`b"],"filters":[],"key":["a`b"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a`b\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a`b')), 1) = 1 THEN ((((787846414 * ((\"a`b\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"a`b\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"a`b\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'a`b')), 1) = 1 THEN \"a`b\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a.b"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a.b\"\nFROM temp.\"__sdm_seeds_3fe5166b2eb04ad5__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a.b"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a.b\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a\nb"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a\nb\"\nFROM temp.\"__sdm_seeds_88fbebce19848f7d__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a\nb"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a\nb\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a`b"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a`b\"\nFROM temp.\"__sdm_seeds_b9447ee50a66958f__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a`b"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a`b\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["ts","a.b"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a.b\"\nFROM temp.\"__sdm_seeds_569d4d13872f9851__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["ts","a\nb"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a\nb\"\nFROM temp.\"__sdm_seeds_4aadab5a062f9549__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["ts","a`b"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a`b\"\nFROM temp.\"__sdm_seeds_1dc284b9209c5d7b__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","a.b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a.b\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","a\nb"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a\nb\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","a`b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a`b\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a.b\"\nFROM temp.\"__sdm_seeds_cc384e8f9ed541df__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"a.b\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a\nb\"\nFROM temp.\"__sdm_seeds_34e6ba58676ec485__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"a\nb\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a`b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a`b\"\nFROM temp.\"__sdm_seeds_70edda45ebea2495__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a`b"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"a`b\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["user_id","score","cat"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"score\", \"cat\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"keys":[{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"score\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","cat"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"cat\"\nFROM temp.\"__sdm_seeds_74b7a8d1a9cd2256__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"score\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_c395fe92b47d5b2a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_64510407f8b59345__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_f1fa6ba0c5a32ad8__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a1245a5f308e2e83__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_70a36c826a64c9ae__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_1e8d4819e383bb19__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_12d1bc10526ab91c__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_0263da31da641ca7__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_d59b6c5dcce97da2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_ca7adb4bb462b410__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_5c4d0f6c562d011d__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_3af59e7085f59342__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9e40d9cd1c4f99c7__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_73cf2d5ddaee7bf4__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_86806d93a2d73631__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":16,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_5f1294c08aafdc46__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":17,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_6042ccf46375a63b__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":18,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9de598fdb1b55378__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":19,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_291855a3ee30dbe5__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":20,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9e243fb67d501a95__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":21,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_dc4e33d5c8aa10e8__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":22,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_aa4642871ed054df__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":23,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_48a1da7d56f9e43a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":24,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_89ea5d8c3de60a29__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":25,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_20633335e13dfdec__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":26,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_4f3d9f5d3322d453__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":27,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_903967b39c99573e__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":28,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_cf8dae3d8f1f8c8d__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":29,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_0d6941b2e5e7f0c0__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":30,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_0f3a74e23175246a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":31,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_8a47b76f116f78cf__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":32,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_4eed198033415518__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":33,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_c20b15cf70273685__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":34,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9df3cd1613c519ee__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":35,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_5b1d91f2bc5a63c3__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":36,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_4317d2cdae05dc5c__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":37,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_fd5ae86f55846259__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":38,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_dfa3a3894317f8e2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":39,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_1e0a7dea6931abe7__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":40,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_1d788d3ed2c0814f__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":41,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a26b4ab1f2c62cea__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":42,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_553beb9f31783f05__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":43,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_e21def4ff4925d98__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":44,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_ee4e67c27dab6c43__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":45,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_3124a2e5d516226e__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":46,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_908bbe3f16d56ad9__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":47,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_d648a89d6f56e4dc__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":48,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_b13b53ba2a82b467__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":49,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_72d4795904690162__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":50,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_73a5440f1c81e74c__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":51,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_76b57e6b98739a09__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":52,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9e0c138c229fce1e__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":53,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a821b70522f2c733__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":54,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_0a1845a32a032dc8__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":55,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_d11107be31df8475__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":56,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_0b7253fc9160bc9a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":57,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_ef7e234f91b45cbf__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":58,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_2441e3ab9f3ab884__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":59,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_4009f0a25dbc5301__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":60,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_f40a8a56c9226b61__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":61,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_86235adf57421964__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":62,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_1e759938118a04eb__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":63,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_2cd70acf22233736__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":64,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_35e256cb7ee2f24d__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":65,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_af16ffa412808f80__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":66,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_6d4d3d480d543a77__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":67,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_c72d62c2e628e8b2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":68,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_55a846bf5bba8de9__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":69,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a7f4e21b58b2daac__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":70,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_81d6386d74e69426__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":71,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_616db919d86ae99b__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":72,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_161c80560df80e54__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":73,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_8fc0022248ec2f11__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":74,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_eaf6301a462e7ba2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":75,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_17be9dee53a91aa7__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":76,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_5bb605e83c301170__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":77,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_584a40fbfc83fe7d__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":78,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_85fe303ee3a9c7ae__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":79,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_b67f1e1ba9d32c83__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":80,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_f49e8ee999bbf21b__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":81,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_15070e3d36379ca6__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":82,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_22f0d7f20a3d3791__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":83,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a94d5625cf4916d4__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":84,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_aaef73be14fa2327__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":85,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_7e2705ea077f8422__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":86,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_eb7b16cbbdd506fd__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":87,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_eee6dbb7fd8119f0__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":88,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_49aff3eb6b243503__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":89,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_192f060ea4fad02e__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":90,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_754ec51fb5e36618__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":91,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_e86cc16ef2c94785__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":92,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_359c2081b417356a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":93,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_b0a9630e941189cf__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":94,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_69797e6d30a7ed5c__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":95,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_23bc940ed8267359__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":96,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_c45578b596672aee__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":97,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_817f3d923efc74c3__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":98,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_d67142a056ce52b0__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":99,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_4cd2db6841d671bd__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":100,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_5ba10528cd5f6a52__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_b038b5417177dfd8__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_2bd81e5ce387c672__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_81d448336052102a__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_344db008255c0474__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_0780d6111d30a00e__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_228f4da8a48a4845__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_5950b5ca914d2b79__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"filters":[],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"score\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a307dd1ead4b4efe__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_811f9786b52e9b1b__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_351e9b7ce7d33864__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_a946e984639054a1__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_67490fd2b12732aa__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_9faad80325cd2667__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_296bfe05b70d74a0__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_7d818d7bef1a5d5d__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_17d87d3f7bbef856__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\"\nFROM temp.\"__sdm_seeds_84652635a91e753c__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"part":"index_columns","request":"describe","tables":[{"source":["sqlite_autoindex_users_1"]}]}"##,
        &["PRAGMA index_info(\"sqlite_autoindex_users_1\")"],
    ),
    (
        r##"{"columns":["user_id","region","age"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"region\", \"age\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"columns":["order_id","user_id","region","price","ts"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"region\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["user_id"],"filters":[],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\"\nFROM \"users\"\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","region","age"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"user_id\", t.\"region\", t.\"age\"\nFROM temp.\"__sdm_seeds_5518bcefd93373da__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","region","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"region\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_d53a54b966039a56__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"part":"index_columns","request":"describe","tables":[{"source":["sqlite_autoindex_orders_1"]}]}"##,
        &["PRAGMA index_info(\"sqlite_autoindex_orders_1\")"],
    ),
    (
        r##"{"columns":["order_no","line","user_id","price","ts"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_no\", \"line\", \"user_id\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_no","line","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"line","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"order_no","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"price","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"user_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_no\", t.\"line\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_e37ae112010b6bb3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"line\", \"order_no\", \"price\", \"user_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"line\", t.\"order_no\", t.\"price\", t.\"user_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","line"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"line\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"keys":[{"columns":["order_id","line"],"table":{"source":["orders"]}},{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, (SELECT COUNT(*) FROM (SELECT DISTINCT \"order_id\", \"line\" FROM \"orders\" WHERE \"order_id\" IS NOT NULL AND \"line\" IS NOT NULL)) AS d, COUNT(CASE WHEN \"order_id\" IS NULL OR \"line\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"keys":[{"columns":["order_id"],"table":{"source":["orders"]}},{"columns":["user"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user\") AS d, COUNT(CASE WHEN \"user\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user"],"filters":[{"column":"user","filter":"not_null"}],"key":["user"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user\"\nFROM \"users\"\nWHERE \"user\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user')), 1) = 1 THEN ((((787846414 * ((\"user\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user')), 1) = 1 THEN \"user\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user","age","country"],"key":["user"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user\" == s.\"key\") AS \"__kumo_matches__\", t.\"user\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_c4dfd543f903c1c6__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["user_id","userid","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"userid\", \"age\", \"country\" FROM \"users\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"keys":[{"columns":["order_id"],"table":{"source":["orders"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns":["order_id","orderid","user_id","price","ts"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"orderid\", \"user_id\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","orderid","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"orderid","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"price","descending":false,"nulls_last":false},{"as_timestamp":false,"column":"user_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"orderid\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_49c938c29232ea1f__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\", \"orderid\", \"price\", \"user_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\", t.\"orderid\", t.\"price\", t.\"user_id\"",
        ],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["buys"]}]}"##,
        &["PRAGMA table_info(\"buys\")"],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &["PRAGMA index_list(\"buys\")"],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"limit":1000,"request":"preview","table":{"source":["buys"]}}"##,
        &[
            "SELECT \"buy_id\", \"user_id\", \"item_id\", \"ts\" FROM \"buys\" ORDER BY rowid LIMIT 1000",
        ],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["items"]}]}"##,
        &["PRAGMA table_info(\"items\")"],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &["PRAGMA index_list(\"items\")"],
    ),
    (
        r##"{"columns":["item_id","kind"],"limit":1000,"request":"preview","table":{"source":["items"]}}"##,
        &["SELECT \"item_id\", \"kind\" FROM \"items\" ORDER BY rowid LIMIT 1000"],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &["PRAGMA foreign_key_list(\"buys\")"],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &["PRAGMA foreign_key_list(\"items\")"],
    ),
    (
        r##"{"keys":[{"columns":["buy_id"],"table":{"source":["buys"]}},{"columns":["item_id"],"table":{"source":["items"]}},{"columns":["order_id"],"table":{"source":["orders"]}},{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"buy_id\") AS d, COUNT(CASE WHEN \"buy_id\" IS NULL THEN 1 END) AS z\nFROM \"buys\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"item_id\") AS d, COUNT(CASE WHEN \"item_id\" IS NULL THEN 1 END) AS z\nFROM \"items\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 3 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"part":"indexes","request":"describe","tables":[{"source":["buys"]}]}"##,
        &["PRAGMA index_list(\"buys\")"],
    ),
    (
        r##"{"part":"indexes","request":"describe","tables":[{"source":["items"]}]}"##,
        &["PRAGMA index_list(\"items\")"],
    ),
    (
        r##"{"columns":[{"column":"ts","table":{"source":["buys"]}}],"request":"time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"buys\"",
        ],
    ),
    (
        r##"{"columns":["ts"],"filters":[{"column":"ts","filter":"not_null"}],"limit":1,"request":"scan","table":{"source":["buys"]}}"##,
        &["SELECT \"ts\" FROM \"buys\" WHERE \"ts\" IS NOT NULL LIMIT 1"],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_8e2659944b530c31__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_45066ce332a9ed7c__\" s\nJOIN \"buys\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"buys\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"buy_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"buy_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_9abd82e5a7ddd817__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"kind\"\nFROM temp.\"__sdm_seeds_223b081ff123b3b6__\" s\nJOIN \"items\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"items\"\n    WHERE \"item_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_c021acae602bb1b8__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_bbe392cfb1103f8f__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_d7b1960b837021bb__\" s\nJOIN \"buys\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"buys\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"buy_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"buy_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_cc86df34deaafb3e__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"kind\"\nFROM temp.\"__sdm_seeds_9f21c8125139f814__\" s\nJOIN \"items\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"items\"\n    WHERE \"item_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_4690e709a0d8f8ae__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_3701b53c0bf993eb__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM temp.\"__sdm_seeds_690cf0a7f51297f3__\" s\nJOIN \"buys\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"buys\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"buy_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"buy_id\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_3bd0dc0ac1e4eeac__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"kind\"\nFROM temp.\"__sdm_seeds_e2dc20944e8eecc7__\" s\nJOIN \"items\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"items\"\n    WHERE \"item_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_2a5b1f045c296375__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_72fa0c024b4c0610__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_5f62a3ffdc62e383__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_5442d82155d25209__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1fcce85fbf04b7d4__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":false},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":false}],"per_seed":32,"request":"neighbors","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_826bd5cfe904ea33__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_ae1939a5de27d3d7__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_bf00063a4087c554__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"metadata":{"sdm:stored_time":"2023-01-01 00:00:00"},"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_d0d824a02ac1a6a7__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns":["user_id","age"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"key_dtype":"int","n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN ((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 ELSE ((((787846414 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((rowid % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647 END, CASE WHEN COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info('users') WHERE name = 'user_id')), 1) = 1 THEN \"user_id\" END, rowid LIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"country\"\nFROM temp.\"__sdm_seeds_e40cd6630c91afb4__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
];

#[test]
fn every_golden_request_renders_its_statements_on_sqlite() {
    goldens::render("sqlite", GOLDENS);
}

#[test]
fn every_fetch_the_sqlite_run_made_is_a_golden() {
    goldens::run_is_pinned("sqlite", GOLDENS);
}

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails_on_sqlite() {
    goldens::seeded_faults("sqlite", GOLDENS);
}
