// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4 on DuckDB for the whole corpus (T2.4): every statement the DuckDB
//! dialect rendered for the contract cases, pinned verbatim beside the
//! request it answers, beside the families' pins this directory already
//! holds (`duckdb_catalog.rs` and the `*_family.rs` files, which
//! sdm-testkit's `t132*_sql.rs` execute on DuckDB). Each request is planned
//! here and must render exactly its statements, so a change of the SQL
//! without a golden update fails; and every fetch python-core's run of the
//! cases on DuckDB made in CI (`contract-python (ref-pandas3, duckdb)`,
//! which executed each statement on DuckDB, its result held to the
//! reference's by the gate's comparator) is a golden, checked in
//! `l4-goldens (duckdb)` ([`goldens`]). Collected from that run's
//! recorded fetches (`core.fetches`) when the dialect changes, and
//! reviewed as any golden.

mod goldens;

/// Each request, as the contract writes it, and the statements its plan
/// carries on DuckDB, in order.
const GOLDENS: goldens::Goldens = &[
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
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA table_info(\"orders\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA table_info(\"users\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_customers'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_products'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_sales_order_details'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_sales_order_headers'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'orders'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'users'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "indexes", "request": "describe", "tables": []}"##,
        &["SELECT table_name, expressions FROM duckdb_indexes()"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_customers'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_products'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_sales_order_details'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'adventureworks_sales_order_headers'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'orders'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'users'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "tables", "request": "describe", "tables": []}"##,
        &[
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &["PRAGMA table_info(\"addresses\")"],
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
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &["PRAGMA table_info(\"products\")"],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &["PRAGMA table_info(\"stores\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'addresses'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'buyers'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'customers'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'details'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'headers'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'lines'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'products'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'stores'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'addresses'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'buyers'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'customers'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'details'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'headers'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'lines'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'products'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'stores'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["CustomerID"], "table": {"source": ["adventureworks_customers"]}}, {"columns": ["ProductID"], "table": {"source": ["adventureworks_products"]}}, {"columns": ["SalesOrderDetailID"], "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"CustomerID\") AS d, COUNT(CASE WHEN \"CustomerID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_customers\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"ProductID\") AS d, COUNT(CASE WHEN \"ProductID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_products\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"SalesOrderDetailID\") AS d, COUNT(CASE WHEN \"SalesOrderDetailID\" IS NULL THEN 1 END) AS z\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"keys": [{"columns": ["order_id"], "table": {"source": ["orders"]}}, {"columns": ["user_id"], "table": {"source": ["users"]}}], "request": "key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
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
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.\"CustomerID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.\"CustomerID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.\"CustomerID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_products\" t\n  ON t.\"ProductID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_products\" t\n  ON t.\"ProductID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_products\" t\n  ON t.\"ProductID\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "key": ["uid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"uid\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"uid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\", t.\"g\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f_date\", t.\"f_ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"aid\", t.\"city\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"addresses\" t\n  ON t.\"aid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"aid\", t.\"city\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"addresses\" t\n  ON t.\"aid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["aid", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"aid\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"addresses\" t\n  ON t.\"aid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buyers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buyers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["cid", "seg"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"cid\", t.\"seg\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"cid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["hid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["headers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"headers\" t\n  ON t.\"hid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"products\" t\n  ON t.\"pid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"products\" t\n  ON t.\"pid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"pid\", t.\"price\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"products\" t\n  ON t.\"pid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"sid\", t.\"size\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"stores\" t\n  ON t.\"sid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"sid\", t.\"size\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"stores\" t\n  ON t.\"sid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "key": ["CustomerID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_headers\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["OrderQty", "OrderDate"], "key": ["ProductID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderQty\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "ModifiedDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "AccountNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "BillToAddressID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Comment", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CreditCardApprovalCode", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CreditCardID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CurrencyRateID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CustomerID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "DueDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Freight", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ModifiedDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "OnlineOrderFlag", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "PurchaseOrderNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "RevisionNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesPersonID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipMethodID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipToAddressID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Status", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SubTotal", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TaxAmt", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TerritoryID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TotalDue", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"RevisionNumber\", t.\"OrderDate\", t.\"DueDate\", t.\"ShipDate\", t.\"Status\", t.\"OnlineOrderFlag\", t.\"SalesOrderNumber\", t.\"PurchaseOrderNumber\", t.\"AccountNumber\", t.\"CustomerID\", t.\"SalesPersonID\", t.\"TerritoryID\", t.\"BillToAddressID\", t.\"ShipToAddressID\", t.\"ShipMethodID\", t.\"CreditCardID\", t.\"CreditCardApprovalCode\", t.\"CurrencyRateID\", t.\"SubTotal\", t.\"TaxAmt\", t.\"Freight\", t.\"TotalDue\", t.\"Comment\", t.\"ModifiedDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_headers\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"AccountNumber\" ASC NULLS LAST, t.\"BillToAddressID\" ASC NULLS LAST, t.\"Comment\" ASC NULLS LAST, t.\"CreditCardApprovalCode\" ASC NULLS LAST, t.\"CreditCardID\" ASC NULLS LAST, t.\"CurrencyRateID\" ASC NULLS LAST, t.\"CustomerID\" ASC NULLS LAST, t.\"DueDate\" ASC NULLS LAST, t.\"Freight\" ASC NULLS LAST, t.\"ModifiedDate\" ASC NULLS LAST, t.\"OnlineOrderFlag\" ASC NULLS LAST, t.\"PurchaseOrderNumber\" ASC NULLS LAST, t.\"RevisionNumber\" ASC NULLS LAST, t.\"SalesOrderID\" ASC NULLS LAST, t.\"SalesOrderNumber\" ASC NULLS LAST, t.\"SalesPersonID\" ASC NULLS LAST, t.\"ShipDate\" ASC NULLS LAST, t.\"ShipMethodID\" ASC NULLS LAST, t.\"ShipToAddressID\" ASC NULLS LAST, t.\"Status\" ASC NULLS LAST, t.\"SubTotal\" ASC NULLS LAST, t.\"TaxAmt\" ASC NULLS LAST, t.\"TerritoryID\" ASC NULLS LAST, t.\"TotalDue\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"AccountNumber\" ASC NULLS LAST, t.\"BillToAddressID\" ASC NULLS LAST, t.\"Comment\" ASC NULLS LAST, t.\"CreditCardApprovalCode\" ASC NULLS LAST, t.\"CreditCardID\" ASC NULLS LAST, t.\"CurrencyRateID\" ASC NULLS LAST, t.\"CustomerID\" ASC NULLS LAST, t.\"DueDate\" ASC NULLS LAST, t.\"Freight\" ASC NULLS LAST, t.\"ModifiedDate\" ASC NULLS LAST, t.\"OnlineOrderFlag\" ASC NULLS LAST, t.\"PurchaseOrderNumber\" ASC NULLS LAST, t.\"RevisionNumber\" ASC NULLS LAST, t.\"SalesOrderID\" ASC NULLS LAST, t.\"SalesOrderNumber\" ASC NULLS LAST, t.\"SalesPersonID\" ASC NULLS LAST, t.\"ShipDate\" ASC NULLS LAST, t.\"ShipMethodID\" ASC NULLS LAST, t.\"ShipToAddressID\" ASC NULLS LAST, t.\"Status\" ASC NULLS LAST, t.\"SubTotal\" ASC NULLS LAST, t.\"TaxAmt\" ASC NULLS LAST, t.\"TerritoryID\" ASC NULLS LAST, t.\"TotalDue\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND CAST(t.\"OrderDate\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"OrderDate\" AS TIMESTAMP) DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "key": ["buyer"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"buyer\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"buyer\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "day", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"day\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"day\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"day\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"day\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "happened", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"happened\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"happened\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"happened\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"happened\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"amount\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["price", "day"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"day\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"day\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"day\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "happened"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"happened\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"happened\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"happened\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["buyer"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"buyer\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "uint8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts", "amount"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"amount\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"aid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY t.\"cid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", t.\"cid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"cid\", t.\"aid\", t.\"seg\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"customers\" t\n  ON t.\"aid\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY t.\"cid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", t.\"cid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["did", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"did\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"did\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"hid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"did\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"hid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"did\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"did\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["headers"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"hid\", t.\"cid\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"headers\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"hid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"hid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["aid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"aid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["pid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"pid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["sid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"cid\", t.\"aid\", t.\"pid\", t.\"sid\", t.\"ts\", t.\"tag\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"sid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"oid\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["aid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"aid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"cid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["pid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"pid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["sid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"sid\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT \"CustomerID\", \"PersonID\", \"StoreID\", \"TerritoryID\", \"AccountNumber\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_customers\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\", \"Name\", \"ProductNumber\", \"MakeFlag\", \"FinishedGoodsFlag\", \"Color\", \"SafetyStockLevel\", \"ReorderPoint\", \"StandardCost\", \"ListPrice\", \"Size\", \"SizeUnitMeasureCode\", \"WeightUnitMeasureCode\", \"Weight\", \"DaysToManufacture\", \"ProductLine\", \"Class\", \"Style\", \"ProductSubcategoryID\", \"ProductModelID\", \"SellStartDate\", \"SellEndDate\", \"DiscontinuedDate\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_products\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "rowguid", "ModifiedDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_sales_order_headers"]}}"##,
        &[
            "SELECT \"SalesOrderID\", \"RevisionNumber\", \"OrderDate\", \"DueDate\", \"ShipDate\", \"Status\", \"OnlineOrderFlag\", \"SalesOrderNumber\", \"PurchaseOrderNumber\", \"AccountNumber\", \"CustomerID\", \"SalesPersonID\", \"TerritoryID\", \"BillToAddressID\", \"ShipToAddressID\", \"ShipMethodID\", \"CreditCardID\", \"CreditCardApprovalCode\", \"CurrencyRateID\", \"SubTotal\", \"TaxAmt\", \"Freight\", \"TotalDue\", \"Comment\", \"rowguid\", \"ModifiedDate\" FROM \"adventureworks_sales_order_headers\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "rowguid", "ModifiedDate", "CustomerID", "OrderDate"], "limit": 1000, "request": "preview", "table": {"source": ["adventureworks_sales_order_details"]}}"##,
        &[
            "SELECT \"SalesOrderID\", \"SalesOrderDetailID\", \"CarrierTrackingNumber\", \"OrderQty\", \"ProductID\", \"SpecialOfferID\", \"UnitPrice\", \"UnitPriceDiscount\", \"LineTotal\", \"rowguid\", \"ModifiedDate\", \"CustomerID\", \"OrderDate\" FROM \"adventureworks_sales_order_details\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"oid\", \"buyer\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"order_id\", \"user_id\", \"price\", \"day\" FROM \"orders\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"order_id\", \"user_id\", \"price\", \"happened\" FROM \"orders\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"amount\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"order_id\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"uid\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "d"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"d\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"f\", \"g\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"f\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\", \"country\", \"f_date\", \"f_ts\" FROM \"users\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "t"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"t\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "tier"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"tier\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "limit": 1000, "request": "preview", "table": {"source": ["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["addresses"]}}"##,
        &["SELECT \"aid\", \"city\", \"tag\" FROM \"addresses\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["buyers"]}}"##,
        &["SELECT \"cid\", \"aid\", \"seg\", \"tag\" FROM \"buyers\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["customers"]}}"##,
        &["SELECT \"cid\", \"aid\", \"seg\", \"tag\" FROM \"customers\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["cid", "seg"], "limit": 1000, "request": "preview", "table": {"source": ["customers"]}}"##,
        &["SELECT \"cid\", \"seg\" FROM \"customers\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["did", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["details"]}}"##,
        &["SELECT \"did\", \"cid\", \"ts\" FROM \"details\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["details"]}}"##,
        &["SELECT \"did\", \"hid\", \"cid\", \"ts\" FROM \"details\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["headers"]}}"##,
        &["SELECT \"hid\", \"cid\", \"ts\" FROM \"headers\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["lid", "oid", "qty", "ts", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["lines"]}}"##,
        &["SELECT \"lid\", \"oid\", \"qty\", \"ts\", \"tag\" FROM \"lines\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &[
            "SELECT \"oid\", \"cid\", \"aid\", \"pid\", \"sid\", \"ts\", \"tag\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["products"]}}"##,
        &["SELECT \"pid\", \"price\", \"tag\" FROM \"products\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "limit": 1000, "request": "preview", "table": {"source": ["stores"]}}"##,
        &["SELECT \"sid\", \"size\", \"tag\" FROM \"stores\" LIMIT 1000"],
    ),
    (
        r##"{"columns": ["CustomerID"], "filters": [{"column": "CustomerID", "filter": "not_null"}], "key": ["CustomerID"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT \"CustomerID\"\nFROM (SELECT * FROM \"adventureworks_customers\"\nWHERE \"CustomerID\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Color"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\", \"Color\"\nFROM (SELECT * FROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["ProductID"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT \"ProductID\"\nFROM (SELECT * FROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["uid"], "filters": [], "key": ["uid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"uid\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "country"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"country\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "d"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"d\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "f_date"], "filters": [{"column": "user_id", "filter": "not_null"}, {"column": "f_date", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"f_date\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL AND \"f_date\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "t"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"t\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "tier"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\", \"tier\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["user_id"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT \"user_id\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["aid", "city"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\", \"city\"\nFROM (SELECT * FROM \"addresses\"\nWHERE \"aid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\"\nFROM (SELECT * FROM \"addresses\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT \"aid\"\nFROM (SELECT * FROM \"addresses\"\nWHERE \"aid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT \"cid\"\nFROM (SELECT * FROM \"buyers\"\nWHERE \"cid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT \"cid\"\nFROM (SELECT * FROM \"customers\"\nWHERE \"cid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["pid"], "filters": [{"column": "pid", "filter": "not_null"}], "key": ["pid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT \"pid\"\nFROM (SELECT * FROM \"products\"\nWHERE \"pid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": ["sid"], "filters": [{"column": "sid", "filter": "not_null"}], "key": ["sid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT \"sid\"\nFROM (SELECT * FROM \"stores\"\nWHERE \"sid\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"OrderDate\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"OrderDate\" AS TIMESTAMP)) as max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}, {"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"OrderDate\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"OrderDate\" AS TIMESTAMP)) as max_date\nFROM \"adventureworks_sales_order_headers\"\nUNION ALL\nSELECT\n  1 as table_index,\n  MIN(CAST(\"OrderDate\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"OrderDate\" AS TIMESTAMP)) as max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"OrderDate\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"OrderDate\" AS TIMESTAMP)) as max_date\nFROM \"adventureworks_sales_order_headers\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "day", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"day\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"day\" AS TIMESTAMP)) as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "happened", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"happened\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"happened\" AS TIMESTAMP)) as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["lines"]}}, {"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"lines\"\nUNION ALL\nSELECT\n  1 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"orders\"",
        ],
    ),
    // T2.18 (KO2-36): the name.* cases, a column named with a dot, a
    // backtick or a line feed, each quoted alone as any name is.
    (
        r##"{"columns":["a.b","age","country"],"key":["a.b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"a.b\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"a.b\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["a.b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a.b\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a.b"],"filters":[],"key":["a.b"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a.b\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"key":["a\nb"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"a\nb\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"a\nb\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a\nb\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a\nb"],"filters":[],"key":["a\nb"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a\nb\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"key":["a`b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"a`b\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"a`b\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a`b\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a`b"],"filters":[],"key":["a`b"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"a`b\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a.b"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a.b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a.b"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a.b\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a\nb"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a\nb\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a\nb"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a\nb\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a`b"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"a`b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a`b"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"a`b\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["ts","a.b"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a.b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["ts","a\nb"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a\nb\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["ts","a`b"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a`b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","a.b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a.b\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","a\nb"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a\nb\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","a`b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"a`b\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a.b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"a.b\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a\nb\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"a\nb\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","a`b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"a`b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a`b"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"a`b\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","score","cat"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"score\", \"cat\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"keys":[{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"score\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","cat"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"cat\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"score\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":16,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":17,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":18,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":19,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":20,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":21,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":22,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":23,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":24,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":25,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":26,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":27,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":28,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":29,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":30,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":31,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":32,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":33,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":34,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":35,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":36,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":37,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":38,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":39,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":40,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":41,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":42,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":43,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":44,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":45,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":46,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":47,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":48,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":49,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":50,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":51,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":52,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":53,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":54,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":55,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":56,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":57,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":58,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":59,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":60,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":61,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":62,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":63,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":64,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":65,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":66,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":67,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":68,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":69,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":70,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":71,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":72,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":73,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":74,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":75,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":76,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":77,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":78,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":79,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":80,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":81,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":82,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":83,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":84,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":85,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":86,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":87,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":88,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":89,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":90,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":91,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":92,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":93,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":94,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":95,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":96,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":97,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":98,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":99,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":100,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"filters":[],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"score\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","region","price","ts"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"region\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["user_id","region","age"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"region\", \"age\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"keys":[{"columns":["user_id"],"label":1,"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user_id"],"filters":[],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\"\nFROM (SELECT * FROM \"users\")\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","region","age"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"region\", t.\"age\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","region","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"region\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_no","line","user_id","price","ts"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_no\", \"line\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_no","line","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"line","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"order_no","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"price","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"user_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_no\", t.\"line\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"line\" ASC NULLS LAST, t.\"order_no\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"line\" ASC NULLS LAST, t.\"order_no\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","line"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"line\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"keys":[{"columns":["order_id","line"],"table":{"source":["orders"]}},{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT (\"order_id\", \"line\")) FILTER (WHERE \"order_id\" IS NOT NULL AND \"line\" IS NOT NULL) AS d, COUNT(CASE WHEN \"order_id\" IS NULL OR \"line\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","score"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"score\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","score"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"score\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"keys":[{"columns":["order_id"],"table":{"source":["orders"]}},{"columns":["user"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user\") AS d, COUNT(CASE WHEN \"user\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["user"],"filters":[{"column":"user","filter":"not_null"}],"key":["user"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user","age","country"],"key":["user"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","userid","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"userid\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
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
            "SELECT \"order_id\", \"orderid\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["order_id","orderid","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"orderid","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"price","descending":false,"nulls_last":true},{"as_timestamp":false,"column":"user_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"orderid\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST, t.\"orderid\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST, t.\"orderid\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["user_id","age","feature"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"feature\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","feature"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"feature\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"float32"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"float32"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"float32"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["user_id","age"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT \"user_id\", \"age\"\nFROM (SELECT * FROM \"users\"\nWHERE \"user_id\" IS NOT NULL)\nUSING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)",
        ],
    ),
    (
        r##"{"columns":["user_id","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["user_id","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"float32"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["buys"]}]}"##,
        &["PRAGMA table_info(\"buys\")"],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'buys'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"limit":1000,"request":"preview","table":{"source":["buys"]}}"##,
        &["SELECT \"buy_id\", \"user_id\", \"item_id\", \"ts\" FROM \"buys\" LIMIT 1000"],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["items"]}]}"##,
        &["PRAGMA table_info(\"items\")"],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &[
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'items'\n  AND database_name = current_database() AND schema_name = current_schema()",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"limit":1000,"request":"preview","table":{"source":["items"]}}"##,
        &["SELECT \"item_id\", \"kind\" FROM \"items\" LIMIT 1000"],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'buys'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &[
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE table_name = 'items'\n  AND database_name = current_database() AND schema_name = current_schema()\n  AND constraint_type = 'FOREIGN KEY'",
        ],
    ),
    (
        r##"{"keys":[{"columns":["buy_id"],"table":{"source":["buys"]}},{"columns":["item_id"],"table":{"source":["items"]}},{"columns":["order_id"],"table":{"source":["orders"]}},{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"buy_id\") AS d, COUNT(CASE WHEN \"buy_id\" IS NULL THEN 1 END) AS z\nFROM \"buys\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"item_id\") AS d, COUNT(CASE WHEN \"item_id\" IS NULL THEN 1 END) AS z\nFROM \"items\"\nUNION ALL\nSELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT \"order_id\") AS d, COUNT(CASE WHEN \"order_id\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 3 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":[{"column":"ts","table":{"source":["buys"]}}],"request":"time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(CAST(\"ts\" AS TIMESTAMP)) as min_date,\n  MAX(CAST(\"ts\" AS TIMESTAMP)) as max_date\nFROM \"buys\"",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"item_id\", t.\"kind\"\nFROM \"__sdm_seeds__\" s\nJOIN \"items\" t\n  ON t.\"item_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"item_id\", t.\"kind\"\nFROM \"__sdm_seeds__\" s\nJOIN \"items\" t\n  ON t.\"item_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"buy_id\", t.\"user_id\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"item_id\", t.\"kind\"\nFROM \"__sdm_seeds__\" s\nJOIN \"items\" t\n  ON t.\"item_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"precision":18,"scale":0,"type":"decimal128"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"item_id\", t.\"kind\"\nFROM \"__sdm_seeds__\" s\nJOIN \"items\" t\n  ON t.\"item_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","i.d","ts"],"limit":1000,"request":"preview","table":{"source":["buys"]}}"##,
        &["SELECT \"buy_id\", \"user_id\", \"i.d\", \"ts\" FROM \"buys\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["i.d","kind"],"limit":1000,"request":"preview","table":{"source":["items"]}}"##,
        &["SELECT \"i.d\", \"kind\" FROM \"items\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", COUNT(*) OVER (PARTITION BY s.\"batch\") AS \"__kumo_matches__\", t.\"user_id\", t.\"country\"\nFROM \"__sdm_seeds__\" s\nJOIN \"users\" t\n  ON t.\"user_id\" = s.\"key\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n) = 1",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","channel"],"limit":1000,"request":"preview","table":{"source":["orders"]}}"##,
        &[
            "SELECT \"order_id\", \"user_id\", \"price\", \"ts\", \"channel\" FROM \"orders\" LIMIT 1000",
        ],
    ),
    (
        r##"{"columns":["price","ts","channel"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\", t.\"channel\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"end\"\n AND CAST(t.\"ts\" AS TIMESTAMP) > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","channel"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\", t.\"channel\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND CAST(t.\"ts\" AS TIMESTAMP) <= s.\"time\"\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.\"batch\"\n  ORDER BY CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n) <= 32\nORDER BY s.\"batch\", CAST(t.\"ts\" AS TIMESTAMP) DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST",
        ],
    ),
];

#[test]
fn every_golden_request_renders_its_statements_on_duckdb() {
    goldens::render("duckdb", GOLDENS);
}

#[test]
fn every_fetch_the_duckdb_run_made_is_a_golden() {
    goldens::run_is_pinned("duckdb", GOLDENS);
}

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails_on_duckdb() {
    goldens::seeded_faults("duckdb", GOLDENS);
}
