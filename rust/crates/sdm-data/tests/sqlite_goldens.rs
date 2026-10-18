// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4 on SQLite (T2.2): every statement the SQLite dialect rendered for the
//! contract cases and the key-pair vectors on SQLite, on both core hosts,
//! pinned verbatim beside the request it answers. Each request is planned
//! here and must render exactly its statements, so a change of the SQL
//! without a golden update fails; the case runs (python-core and rust-host
//! on SQLite) execute them, each result held to the reference's by the
//! comparator (`sql.result`). Collected from those runs' recorded fetches
//! (`core.fetches`) when the dialect changes, and reviewed as any golden.

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;

/// Each request, as the contract writes it, and the statements its plan
/// carries on SQLite, in order.
const GOLDENS: &[(&str, &[&str])] = &[
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
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA foreign_key_list(\"orders\")"],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA foreign_key_list(\"users\")"],
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
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA index_list(\"orders\")"],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &["PRAGMA index_list(\"users\")"],
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
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &["PRAGMA table_info(\"orders\")"],
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
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_customers\" WHERE \"CustomerID\" == s.\"key\") AS \"__kumo_matches__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_products\" WHERE \"ProductID\" == s.\"key\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"adventureworks_products\" WHERE \"ProductID\" == s.\"key\") AS \"__kumo_matches__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f\", t.\"g\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", (SELECT COUNT(*) FROM \"users\" WHERE \"user_id\" == s.\"key\") AS \"__kumo_matches__\", t.\"user_id\", t.\"age\", t.\"country\", t.\"f_date\", t.\"f_ts\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"users\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"users\"\n    WHERE \"user_id\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"CustomerID\", t.\"PersonID\", t.\"StoreID\", t.\"TerritoryID\", t.\"AccountNumber\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_4__\" s\nJOIN \"adventureworks_customers\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_customers\"\n    WHERE \"CustomerID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ProductID\", t.\"Name\", t.\"ProductNumber\", t.\"MakeFlag\", t.\"FinishedGoodsFlag\", t.\"Color\", t.\"SafetyStockLevel\", t.\"ReorderPoint\", t.\"StandardCost\", t.\"ListPrice\", t.\"Size\", t.\"SizeUnitMeasureCode\", t.\"WeightUnitMeasureCode\", t.\"Weight\", t.\"DaysToManufacture\", t.\"ProductLine\", t.\"Class\", t.\"Style\", t.\"ProductSubcategoryID\", t.\"ProductModelID\", t.\"SellStartDate\", t.\"SellEndDate\", t.\"DiscontinuedDate\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_5__\" s\nJOIN \"adventureworks_products\" t\n  ON t.rowid = (\n    SELECT rowid\n    FROM \"adventureworks_products\"\n    WHERE \"ProductID\" == s.\"key\"\n    LIMIT 1\n)",
        ],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "key": ["uid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"uid\", t.\"age\", t.\"country\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"users\" t\n  ON t.\"uid\" = s.\"key\"",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "key": ["CustomerID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"adventureworks_sales_order_headers\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND t.\"OrderDate\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"OrderDate\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["OrderQty", "OrderDate"], "key": ["ProductID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderQty\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND t.\"OrderDate\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"OrderDate\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "ModifiedDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "AccountNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "BillToAddressID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Comment", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CreditCardApprovalCode", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CreditCardID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CurrencyRateID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "CustomerID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "DueDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Freight", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ModifiedDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "OnlineOrderFlag", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "PurchaseOrderNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "RevisionNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderNumber", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SalesPersonID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipDate", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipMethodID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "ShipToAddressID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "Status", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "SubTotal", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TaxAmt", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TerritoryID", "descending": false, "nulls_last": false}, {"as_timestamp": false, "column": "TotalDue", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"RevisionNumber\", t.\"OrderDate\", t.\"DueDate\", t.\"ShipDate\", t.\"Status\", t.\"OnlineOrderFlag\", t.\"SalesOrderNumber\", t.\"PurchaseOrderNumber\", t.\"AccountNumber\", t.\"CustomerID\", t.\"SalesPersonID\", t.\"TerritoryID\", t.\"BillToAddressID\", t.\"ShipToAddressID\", t.\"ShipMethodID\", t.\"CreditCardID\", t.\"CreditCardApprovalCode\", t.\"CurrencyRateID\", t.\"SubTotal\", t.\"TaxAmt\", t.\"Freight\", t.\"TotalDue\", t.\"Comment\", t.\"ModifiedDate\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"adventureworks_sales_order_headers\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_headers\"\n  WHERE \"CustomerID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"AccountNumber\", \"BillToAddressID\", \"Comment\", \"CreditCardApprovalCode\", \"CreditCardID\", \"CurrencyRateID\", \"CustomerID\", \"DueDate\", \"Freight\", \"ModifiedDate\", \"OnlineOrderFlag\", \"PurchaseOrderNumber\", \"RevisionNumber\", \"SalesOrderID\", \"SalesOrderNumber\", \"SalesPersonID\", \"ShipDate\", \"ShipMethodID\", \"ShipToAddressID\", \"Status\", \"SubTotal\", \"TaxAmt\", \"TerritoryID\", \"TotalDue\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"AccountNumber\", t.\"BillToAddressID\", t.\"Comment\", t.\"CreditCardApprovalCode\", t.\"CreditCardID\", t.\"CurrencyRateID\", t.\"CustomerID\", t.\"DueDate\", t.\"Freight\", t.\"ModifiedDate\", t.\"OnlineOrderFlag\", t.\"PurchaseOrderNumber\", t.\"RevisionNumber\", t.\"SalesOrderID\", t.\"SalesOrderNumber\", t.\"SalesPersonID\", t.\"ShipDate\", t.\"ShipMethodID\", t.\"ShipToAddressID\", t.\"Status\", t.\"SubTotal\", t.\"TaxAmt\", t.\"TerritoryID\", t.\"TotalDue\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"ProductID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"ProductID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2022-06-30 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"SalesOrderID\", t.\"SalesOrderDetailID\", t.\"CarrierTrackingNumber\", t.\"OrderQty\", t.\"ProductID\", t.\"SpecialOfferID\", t.\"UnitPrice\", t.\"UnitPriceDiscount\", t.\"LineTotal\", t.\"ModifiedDate\", t.\"CustomerID\", t.\"OrderDate\"\nFROM temp.\"__sdm_seeds_4__\" s\nJOIN \"adventureworks_sales_order_details\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"adventureworks_sales_order_details\"\n  WHERE \"CustomerID\" = s.\"key\"\n  AND \"OrderDate\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"OrderDate\" DESC, \"SalesOrderDetailID\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"OrderDate\" DESC, t.\"SalesOrderDetailID\"",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "key": ["buyer"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"oid\", t.\"buyer\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"buyer\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"oid\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"oid\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "day", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"day\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"day\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"day\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"day\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "happened", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"happened\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"happened\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"happened\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"happened\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_2__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8)\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": false}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": false}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"order_id\", t.\"user_id\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_3__\" s\nJOIN \"orders\" t\nON t.rowid IN (\n  SELECT rowid\n  FROM \"orders\"\n  WHERE \"user_id\" = s.\"key\"\n  AND \"ts\" <= substr(s.\"time\", 1, 10) || ' ' || substr(s.\"time\", 12, 8) || '+00:00'\n  ORDER BY \"ts\" DESC, \"order_id\"\n  LIMIT 32\n)\nORDER BY \"__kumo_batch__\", t.\"ts\" DESC, t.\"order_id\"",
        ],
    ),
    (
        r##"{"columns": ["price", "day"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"day\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"day\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"day\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "happened"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"happened\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"happened\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"happened\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["buyer"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"buyer\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 05:00:00+00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8) || '+00:00'\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8) || '+00:00'",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"metadata": {"sdm:stored_time": "2023-01-01 00:00:00"}, "name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "SELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM temp.\"__sdm_seeds_1__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= substr(s.\"end\", 1, 10) || ' ' || substr(s.\"end\", 12, 8)\n AND t.\"ts\" > substr(s.\"start\", 1, 10) || ' ' || substr(s.\"start\", 12, 8)",
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
        r##"{"columns": ["oid", "buyer", "price", "ts"], "limit": 1000, "request": "preview", "table": {"source": ["orders"]}}"##,
        &["SELECT \"oid\", \"buyer\", \"price\", \"ts\" FROM \"orders\" ORDER BY rowid LIMIT 1000"],
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
        r##"{"columns": [{"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(\"ts\") as min_date,\n  MAX(\"ts\") as max_date\nFROM \"orders\"",
        ],
    ),
];

#[test]
fn every_golden_request_renders_its_statements_on_sqlite() {
    let sqlite = ProviderChoice::Sql {
        dialect: Dialect::new("sqlite"),
    };
    assert!(!GOLDENS.is_empty());
    for (request, statements) in GOLDENS {
        let request: DataRequest = serde_json::from_str(request).unwrap();
        let FetchPlan::Sql {
            statements: planned,
            ..
        } = Data::new().plan(&request, &sqlite).unwrap()
        else {
            panic!("{request:?}: not SQL");
        };
        let texts: Vec<&str> = planned.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(&texts, statements, "{request:?}");
    }
}
