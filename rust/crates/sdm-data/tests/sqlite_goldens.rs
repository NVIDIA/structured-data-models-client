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
