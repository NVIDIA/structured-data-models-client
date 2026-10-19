// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4 on Postgres (T2.4): every statement the Postgres dialect rendered for
//! the contract cases, pinned verbatim beside the request it answers. Each
//! request is planned here and must render exactly its statements, so a
//! change of the SQL without a golden update fails; and every fetch
//! python-core's run of the cases on Postgres made in CI
//! (`contract-python (ref-pandas3, postgres)`, which executed each
//! statement on the pinned Postgres service, its result held to the
//! reference's by the comparator) is a golden, checked in
//! `l4-goldens (postgres)` ([`goldens`]). Collected from that run's
//! recorded fetches (`core.fetches`) when the dialect changes, and
//! reviewed as any golden. The provider's own test executes every request
//! kind on Postgres too (`tests_test_ops/test_the_postgres_provider.py`).

mod goldens;

/// Each request, as the contract writes it, and the statements its plan
/// carries on Postgres, in order.
const GOLDENS: goldens::Goldens = &[
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'adventureworks_customers'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'adventureworks_customers'\n           AND c.oid = to_regclass('\"adventureworks_customers\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'adventureworks_products'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'adventureworks_products'\n           AND c.oid = to_regclass('\"adventureworks_products\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'adventureworks_sales_order_details'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'adventureworks_sales_order_details'\n           AND c.oid = to_regclass('\"adventureworks_sales_order_details\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'adventureworks_sales_order_headers'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'adventureworks_sales_order_headers'\n           AND c.oid = to_regclass('\"adventureworks_sales_order_headers\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'orders'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'orders'\n           AND c.oid = to_regclass('\"orders\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'users'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'users'\n           AND c.oid = to_regclass('\"users\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_customers'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_products'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_sales_order_details'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_sales_order_headers'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'orders'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'users'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_customers"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_customers'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_products"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_products'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_details"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_sales_order_details'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["adventureworks_sales_order_headers"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'adventureworks_sales_order_headers'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["orders"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'orders'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["users"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'users'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "tables", "request": "describe", "tables": []}"##,
        &[
            "SELECT table_name\nFROM information_schema.tables\nWHERE table_schema = current_schema()\n  AND table_type = 'BASE TABLE'\nORDER BY table_name",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'addresses'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'addresses'\n           AND c.oid = to_regclass('\"addresses\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'buyers'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'buyers'\n           AND c.oid = to_regclass('\"buyers\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'customers'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'customers'\n           AND c.oid = to_regclass('\"customers\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'details'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'details'\n           AND c.oid = to_regclass('\"details\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'headers'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'headers'\n           AND c.oid = to_regclass('\"headers\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'lines'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'lines'\n           AND c.oid = to_regclass('\"lines\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'products'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'products'\n           AND c.oid = to_regclass('\"products\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "columns", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'stores'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'stores'\n           AND c.oid = to_regclass('\"stores\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'addresses'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'buyers'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'customers'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'details'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'headers'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'lines'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'products'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "foreign_keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'stores'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["addresses"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'addresses'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["buyers"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'buyers'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["customers"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'customers'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["details"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'details'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["headers"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'headers'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["lines"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'lines'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["products"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'products'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"part": "keys", "request": "describe", "tables": [{"source": ["stores"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'stores'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"CustomerID\", l.c1 AS \"PersonID\", l.c2 AS \"StoreID\", l.c3 AS \"TerritoryID\", l.c4 AS \"AccountNumber\", l.c5 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"CustomerID\" AS c0, t.\"PersonID\" AS c1, t.\"StoreID\" AS c2, t.\"TerritoryID\" AS c3, t.\"AccountNumber\" AS c4, t.\"ModifiedDate\" AS c5,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"CustomerID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_customers\" t\n    ON t.\"CustomerID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"CustomerID\", l.c1 AS \"PersonID\", l.c2 AS \"StoreID\", l.c3 AS \"TerritoryID\", l.c4 AS \"AccountNumber\", l.c5 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"CustomerID\" AS c0, t.\"PersonID\" AS c1, t.\"StoreID\" AS c2, t.\"TerritoryID\" AS c3, t.\"AccountNumber\" AS c4, t.\"ModifiedDate\" AS c5,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"CustomerID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_customers\" t\n    ON t.\"CustomerID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber", "ModifiedDate"], "key": ["CustomerID"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"CustomerID\", l.c1 AS \"PersonID\", l.c2 AS \"StoreID\", l.c3 AS \"TerritoryID\", l.c4 AS \"AccountNumber\", l.c5 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"CustomerID\" AS c0, t.\"PersonID\" AS c1, t.\"StoreID\" AS c2, t.\"TerritoryID\" AS c3, t.\"AccountNumber\" AS c4, t.\"ModifiedDate\" AS c5,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"CustomerID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_customers\" t\n    ON t.\"CustomerID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"ProductID\", l.c1 AS \"Name\", l.c2 AS \"ProductNumber\", l.c3 AS \"MakeFlag\", l.c4 AS \"FinishedGoodsFlag\", l.c5 AS \"Color\", l.c6 AS \"SafetyStockLevel\", l.c7 AS \"ReorderPoint\", l.c8 AS \"StandardCost\", l.c9 AS \"ListPrice\", l.c10 AS \"Size\", l.c11 AS \"SizeUnitMeasureCode\", l.c12 AS \"WeightUnitMeasureCode\", l.c13 AS \"Weight\", l.c14 AS \"DaysToManufacture\", l.c15 AS \"ProductLine\", l.c16 AS \"Class\", l.c17 AS \"Style\", l.c18 AS \"ProductSubcategoryID\", l.c19 AS \"ProductModelID\", l.c20 AS \"SellStartDate\", l.c21 AS \"SellEndDate\", l.c22 AS \"DiscontinuedDate\", l.c23 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"ProductID\" AS c0, t.\"Name\" AS c1, t.\"ProductNumber\" AS c2, t.\"MakeFlag\" AS c3, t.\"FinishedGoodsFlag\" AS c4, t.\"Color\" AS c5, t.\"SafetyStockLevel\" AS c6, t.\"ReorderPoint\" AS c7, t.\"StandardCost\" AS c8, t.\"ListPrice\" AS c9, t.\"Size\" AS c10, t.\"SizeUnitMeasureCode\" AS c11, t.\"WeightUnitMeasureCode\" AS c12, t.\"Weight\" AS c13, t.\"DaysToManufacture\" AS c14, t.\"ProductLine\" AS c15, t.\"Class\" AS c16, t.\"Style\" AS c17, t.\"ProductSubcategoryID\" AS c18, t.\"ProductModelID\" AS c19, t.\"SellStartDate\" AS c20, t.\"SellEndDate\" AS c21, t.\"DiscontinuedDate\" AS c22, t.\"ModifiedDate\" AS c23,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"ProductID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_products\" t\n    ON t.\"ProductID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"ProductID\", l.c1 AS \"Name\", l.c2 AS \"ProductNumber\", l.c3 AS \"MakeFlag\", l.c4 AS \"FinishedGoodsFlag\", l.c5 AS \"Color\", l.c6 AS \"SafetyStockLevel\", l.c7 AS \"ReorderPoint\", l.c8 AS \"StandardCost\", l.c9 AS \"ListPrice\", l.c10 AS \"Size\", l.c11 AS \"SizeUnitMeasureCode\", l.c12 AS \"WeightUnitMeasureCode\", l.c13 AS \"Weight\", l.c14 AS \"DaysToManufacture\", l.c15 AS \"ProductLine\", l.c16 AS \"Class\", l.c17 AS \"Style\", l.c18 AS \"ProductSubcategoryID\", l.c19 AS \"ProductModelID\", l.c20 AS \"SellStartDate\", l.c21 AS \"SellEndDate\", l.c22 AS \"DiscontinuedDate\", l.c23 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"ProductID\" AS c0, t.\"Name\" AS c1, t.\"ProductNumber\" AS c2, t.\"MakeFlag\" AS c3, t.\"FinishedGoodsFlag\" AS c4, t.\"Color\" AS c5, t.\"SafetyStockLevel\" AS c6, t.\"ReorderPoint\" AS c7, t.\"StandardCost\" AS c8, t.\"ListPrice\" AS c9, t.\"Size\" AS c10, t.\"SizeUnitMeasureCode\" AS c11, t.\"WeightUnitMeasureCode\" AS c12, t.\"Weight\" AS c13, t.\"DaysToManufacture\" AS c14, t.\"ProductLine\" AS c15, t.\"Class\" AS c16, t.\"Style\" AS c17, t.\"ProductSubcategoryID\" AS c18, t.\"ProductModelID\" AS c19, t.\"SellStartDate\" AS c20, t.\"SellEndDate\" AS c21, t.\"DiscontinuedDate\" AS c22, t.\"ModifiedDate\" AS c23,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"ProductID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_products\" t\n    ON t.\"ProductID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size", "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture", "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ModifiedDate"], "key": ["ProductID"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"ProductID\", l.c1 AS \"Name\", l.c2 AS \"ProductNumber\", l.c3 AS \"MakeFlag\", l.c4 AS \"FinishedGoodsFlag\", l.c5 AS \"SafetyStockLevel\", l.c6 AS \"ReorderPoint\", l.c7 AS \"StandardCost\", l.c8 AS \"ListPrice\", l.c9 AS \"Size\", l.c10 AS \"SizeUnitMeasureCode\", l.c11 AS \"WeightUnitMeasureCode\", l.c12 AS \"Weight\", l.c13 AS \"DaysToManufacture\", l.c14 AS \"ProductLine\", l.c15 AS \"Class\", l.c16 AS \"Style\", l.c17 AS \"ProductSubcategoryID\", l.c18 AS \"ProductModelID\", l.c19 AS \"SellStartDate\", l.c20 AS \"SellEndDate\", l.c21 AS \"DiscontinuedDate\", l.c22 AS \"ModifiedDate\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"ProductID\" AS c0, t.\"Name\" AS c1, t.\"ProductNumber\" AS c2, t.\"MakeFlag\" AS c3, t.\"FinishedGoodsFlag\" AS c4, t.\"SafetyStockLevel\" AS c5, t.\"ReorderPoint\" AS c6, t.\"StandardCost\" AS c7, t.\"ListPrice\" AS c8, t.\"Size\" AS c9, t.\"SizeUnitMeasureCode\" AS c10, t.\"WeightUnitMeasureCode\" AS c11, t.\"Weight\" AS c12, t.\"DaysToManufacture\" AS c13, t.\"ProductLine\" AS c14, t.\"Class\" AS c15, t.\"Style\" AS c16, t.\"ProductSubcategoryID\" AS c17, t.\"ProductModelID\" AS c18, t.\"SellStartDate\" AS c19, t.\"SellEndDate\" AS c20, t.\"DiscontinuedDate\" AS c21, t.\"ModifiedDate\" AS c22,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"ProductID\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"adventureworks_products\" t\n    ON t.\"ProductID\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["uid", "age", "country"], "key": ["uid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"uid\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"uid\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"uid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"uid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f", "g"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"f\", l.c4 AS \"g\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"f\" AS c3, t.\"g\" AS c4,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"f\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"f\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country", "f_date", "f_ts"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"f_date\", l.c4 AS \"f_ts\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"f_date\" AS c3, t.\"f_ts\" AS c4,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 20, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int16"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age", "country"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT k.\"user_id\" AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"users\" AS base FROM \"users\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('user_id', r.value->0)) AS k\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["user_id", "age"], "key": ["user_id"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"aid\", l.c1 AS \"city\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"aid\" AS c0, t.\"city\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"aid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"addresses\" t\n    ON t.\"aid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["aid", "city", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"aid\", l.c1 AS \"city\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"aid\" AS c0, t.\"city\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"aid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"addresses\" t\n    ON t.\"aid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["aid", "tag"], "key": ["aid"], "request": "lookup", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["addresses"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"aid\", l.c1 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"aid\" AS c0, t.\"tag\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"aid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"addresses\" t\n    ON t.\"aid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"buyers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"customers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["buyers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"buyers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"customers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"metadata": {"sdm:key_unique": "true"}, "name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"customers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"aid\", l.c2 AS \"seg\", l.c3 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"customers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["cid", "seg"], "key": ["cid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"cid\", l.c1 AS \"seg\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"cid\" AS c0, t.\"seg\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"cid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"customers\" t\n    ON t.\"cid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["hid"], "request": "lookup", "seeds": {"attachment": 6, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["headers"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"hid\", l.c1 AS \"cid\", l.c2 AS \"ts\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"hid\" AS c0, t.\"cid\" AS c1, t.\"ts\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"hid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"headers\" t\n    ON t.\"hid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"pid\", l.c1 AS \"price\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"pid\" AS c0, t.\"price\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"pid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"products\" t\n    ON t.\"pid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"pid\", l.c1 AS \"price\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"pid\" AS c0, t.\"price\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"pid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"products\" t\n    ON t.\"pid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["pid", "price", "tag"], "key": ["pid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["products"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"pid\", l.c1 AS \"price\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"pid\" AS c0, t.\"price\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"pid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"products\" t\n    ON t.\"pid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"sid\", l.c1 AS \"size\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"sid\" AS c0, t.\"size\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"sid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"stores\" t\n    ON t.\"sid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["sid", "size", "tag"], "key": ["sid"], "request": "lookup", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"metadata": {"sdm:count_unread": "true"}, "name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["stores"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"sid\", l.c1 AS \"size\", l.c2 AS \"tag\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"sid\" AS c0, t.\"size\" AS c1, t.\"tag\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"sid\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"stores\" t\n    ON t.\"sid\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns": ["OrderDate"], "key": ["CustomerID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"end\", w3.\"OrderDate\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_headers\" AS base FROM \"adventureworks_sales_order_headers\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_headers\" t\n  ON t.\"CustomerID\" = s.\"key\"\n AND t.\"OrderDate\" <= s.\"end\"\n AND t.\"OrderDate\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["OrderQty", "OrderDate"], "key": ["ProductID"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"end\", w3.\"OrderDate\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_details\" AS base FROM \"adventureworks_sales_order_details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"OrderQty\", t.\"OrderDate\"\nFROM \"__sdm_seeds__\" s\nJOIN \"adventureworks_sales_order_details\" t\n  ON t.\"ProductID\" = s.\"key\"\n AND t.\"OrderDate\" <= s.\"end\"\n AND t.\"OrderDate\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID", "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "ModifiedDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "AccountNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "BillToAddressID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Comment", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CreditCardApprovalCode", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CreditCardID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CurrencyRateID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "CustomerID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "DueDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Freight", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ModifiedDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "OnlineOrderFlag", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "PurchaseOrderNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "RevisionNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderNumber", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SalesPersonID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipDate", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipMethodID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "ShipToAddressID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "Status", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "SubTotal", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TaxAmt", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TerritoryID", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "TotalDue", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_headers"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_headers\" AS base FROM \"adventureworks_sales_order_headers\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"SalesOrderID\", n.c1 AS \"RevisionNumber\", n.c2 AS \"OrderDate\", n.c3 AS \"DueDate\", n.c4 AS \"ShipDate\", n.c5 AS \"Status\", n.c6 AS \"OnlineOrderFlag\", n.c7 AS \"SalesOrderNumber\", n.c8 AS \"PurchaseOrderNumber\", n.c9 AS \"AccountNumber\", n.c10 AS \"CustomerID\", n.c11 AS \"SalesPersonID\", n.c12 AS \"TerritoryID\", n.c13 AS \"BillToAddressID\", n.c14 AS \"ShipToAddressID\", n.c15 AS \"ShipMethodID\", n.c16 AS \"CreditCardID\", n.c17 AS \"CreditCardApprovalCode\", n.c18 AS \"CurrencyRateID\", n.c19 AS \"SubTotal\", n.c20 AS \"TaxAmt\", n.c21 AS \"Freight\", n.c22 AS \"TotalDue\", n.c23 AS \"Comment\", n.c24 AS \"ModifiedDate\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"SalesOrderID\" AS c0, t.\"RevisionNumber\" AS c1, t.\"OrderDate\" AS c2, t.\"DueDate\" AS c3, t.\"ShipDate\" AS c4, t.\"Status\" AS c5, t.\"OnlineOrderFlag\" AS c6, t.\"SalesOrderNumber\" AS c7, t.\"PurchaseOrderNumber\" AS c8, t.\"AccountNumber\" AS c9, t.\"CustomerID\" AS c10, t.\"SalesPersonID\" AS c11, t.\"TerritoryID\" AS c12, t.\"BillToAddressID\" AS c13, t.\"ShipToAddressID\" AS c14, t.\"ShipMethodID\" AS c15, t.\"CreditCardID\" AS c16, t.\"CreditCardApprovalCode\" AS c17, t.\"CurrencyRateID\" AS c18, t.\"SubTotal\" AS c19, t.\"TaxAmt\" AS c20, t.\"Freight\" AS c21, t.\"TotalDue\" AS c22, t.\"Comment\" AS c23, t.\"ModifiedDate\" AS c24, ROW_NUMBER() OVER (ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"AccountNumber\" ASC NULLS LAST, t.\"BillToAddressID\" ASC NULLS LAST, t.\"Comment\" ASC NULLS LAST, t.\"CreditCardApprovalCode\" ASC NULLS LAST, t.\"CreditCardID\" ASC NULLS LAST, t.\"CurrencyRateID\" ASC NULLS LAST, t.\"CustomerID\" ASC NULLS LAST, t.\"DueDate\" ASC NULLS LAST, t.\"Freight\" ASC NULLS LAST, t.\"ModifiedDate\" ASC NULLS LAST, t.\"OnlineOrderFlag\" ASC NULLS LAST, t.\"PurchaseOrderNumber\" ASC NULLS LAST, t.\"RevisionNumber\" ASC NULLS LAST, t.\"SalesOrderID\" ASC NULLS LAST, t.\"SalesOrderNumber\" ASC NULLS LAST, t.\"SalesPersonID\" ASC NULLS LAST, t.\"ShipDate\" ASC NULLS LAST, t.\"ShipMethodID\" ASC NULLS LAST, t.\"ShipToAddressID\" ASC NULLS LAST, t.\"Status\" ASC NULLS LAST, t.\"SubTotal\" ASC NULLS LAST, t.\"TaxAmt\" ASC NULLS LAST, t.\"TerritoryID\" ASC NULLS LAST, t.\"TotalDue\" ASC NULLS LAST) AS r\n  FROM \"adventureworks_sales_order_headers\" t\n  WHERE t.\"CustomerID\" = s.\"key\"\n    AND t.\"OrderDate\" <= s.\"time\"\n  ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"AccountNumber\" ASC NULLS LAST, t.\"BillToAddressID\" ASC NULLS LAST, t.\"Comment\" ASC NULLS LAST, t.\"CreditCardApprovalCode\" ASC NULLS LAST, t.\"CreditCardID\" ASC NULLS LAST, t.\"CurrencyRateID\" ASC NULLS LAST, t.\"CustomerID\" ASC NULLS LAST, t.\"DueDate\" ASC NULLS LAST, t.\"Freight\" ASC NULLS LAST, t.\"ModifiedDate\" ASC NULLS LAST, t.\"OnlineOrderFlag\" ASC NULLS LAST, t.\"PurchaseOrderNumber\" ASC NULLS LAST, t.\"RevisionNumber\" ASC NULLS LAST, t.\"SalesOrderID\" ASC NULLS LAST, t.\"SalesOrderNumber\" ASC NULLS LAST, t.\"SalesPersonID\" ASC NULLS LAST, t.\"ShipDate\" ASC NULLS LAST, t.\"ShipMethodID\" ASC NULLS LAST, t.\"ShipToAddressID\" ASC NULLS LAST, t.\"Status\" ASC NULLS LAST, t.\"SubTotal\" ASC NULLS LAST, t.\"TaxAmt\" ASC NULLS LAST, t.\"TerritoryID\" ASC NULLS LAST, t.\"TotalDue\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["CustomerID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_details\" AS base FROM \"adventureworks_sales_order_details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"SalesOrderID\", n.c1 AS \"SalesOrderDetailID\", n.c2 AS \"CarrierTrackingNumber\", n.c3 AS \"OrderQty\", n.c4 AS \"ProductID\", n.c5 AS \"SpecialOfferID\", n.c6 AS \"UnitPrice\", n.c7 AS \"UnitPriceDiscount\", n.c8 AS \"LineTotal\", n.c9 AS \"ModifiedDate\", n.c10 AS \"CustomerID\", n.c11 AS \"OrderDate\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"SalesOrderID\" AS c0, t.\"SalesOrderDetailID\" AS c1, t.\"CarrierTrackingNumber\" AS c2, t.\"OrderQty\" AS c3, t.\"ProductID\" AS c4, t.\"SpecialOfferID\" AS c5, t.\"UnitPrice\" AS c6, t.\"UnitPriceDiscount\" AS c7, t.\"LineTotal\" AS c8, t.\"ModifiedDate\" AS c9, t.\"CustomerID\" AS c10, t.\"OrderDate\" AS c11, ROW_NUMBER() OVER (ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST) AS r\n  FROM \"adventureworks_sales_order_details\" t\n  WHERE t.\"CustomerID\" = s.\"key\"\n    AND t.\"OrderDate\" <= s.\"time\"\n  ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_details\" AS base FROM \"adventureworks_sales_order_details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"SalesOrderID\", n.c1 AS \"SalesOrderDetailID\", n.c2 AS \"CarrierTrackingNumber\", n.c3 AS \"OrderQty\", n.c4 AS \"ProductID\", n.c5 AS \"SpecialOfferID\", n.c6 AS \"UnitPrice\", n.c7 AS \"UnitPriceDiscount\", n.c8 AS \"LineTotal\", n.c9 AS \"ModifiedDate\", n.c10 AS \"CustomerID\", n.c11 AS \"OrderDate\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"SalesOrderID\" AS c0, t.\"SalesOrderDetailID\" AS c1, t.\"CarrierTrackingNumber\" AS c2, t.\"OrderQty\" AS c3, t.\"ProductID\" AS c4, t.\"SpecialOfferID\" AS c5, t.\"UnitPrice\" AS c6, t.\"UnitPriceDiscount\" AS c7, t.\"LineTotal\" AS c8, t.\"ModifiedDate\" AS c9, t.\"CustomerID\" AS c10, t.\"OrderDate\" AS c11, ROW_NUMBER() OVER (ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST) AS r\n  FROM \"adventureworks_sales_order_details\" t\n  WHERE t.\"ProductID\" = s.\"key\"\n    AND t.\"OrderDate\" <= s.\"time\"\n  ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty", "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "ModifiedDate", "CustomerID", "OrderDate"], "key": ["ProductID"], "order": [{"as_timestamp": true, "column": "OrderDate", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "SalesOrderDetailID", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["adventureworks_sales_order_details"]}, "window": {"column": "OrderDate", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"OrderDate\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"adventureworks_sales_order_details\" AS base FROM \"adventureworks_sales_order_details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('OrderDate', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"SalesOrderID\", n.c1 AS \"SalesOrderDetailID\", n.c2 AS \"CarrierTrackingNumber\", n.c3 AS \"OrderQty\", n.c4 AS \"ProductID\", n.c5 AS \"SpecialOfferID\", n.c6 AS \"UnitPrice\", n.c7 AS \"UnitPriceDiscount\", n.c8 AS \"LineTotal\", n.c9 AS \"ModifiedDate\", n.c10 AS \"CustomerID\", n.c11 AS \"OrderDate\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"SalesOrderID\" AS c0, t.\"SalesOrderDetailID\" AS c1, t.\"CarrierTrackingNumber\" AS c2, t.\"OrderQty\" AS c3, t.\"ProductID\" AS c4, t.\"SpecialOfferID\" AS c5, t.\"UnitPrice\" AS c6, t.\"UnitPriceDiscount\" AS c7, t.\"LineTotal\" AS c8, t.\"ModifiedDate\" AS c9, t.\"CustomerID\" AS c10, t.\"OrderDate\" AS c11, ROW_NUMBER() OVER (ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST) AS r\n  FROM \"adventureworks_sales_order_details\" t\n  WHERE t.\"ProductID\" = s.\"key\"\n    AND t.\"OrderDate\" <= s.\"time\"\n  ORDER BY t.\"OrderDate\" DESC NULLS LAST, t.\"SalesOrderDetailID\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "buyer", "price", "ts"], "key": ["buyer"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"buyer\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"buyer\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"buyer\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "day"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "day", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"day\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('day', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"day\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"day\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"day\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"day\" <= s.\"time\"\n  ORDER BY t.\"day\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "happened"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "happened", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"happened\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('happened', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"happened\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"happened\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"happened\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"happened\" <= s.\"time\"\n  ORDER BY t.\"happened\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts", "amount"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.c4 AS \"amount\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, t.\"amount\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 20, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int16"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["order_id", "user_id", "price", "ts"], "key": ["user_id"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "order_id", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT k.\"user_id\" AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('user_id', r.value->0)) AS k\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["price", "day"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "day", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"day\" AS \"end\", w3.\"day\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('day', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('day', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"day\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"day\" <= s.\"end\"\n AND t.\"day\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "happened"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "happened", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"happened\" AS \"end\", w3.\"happened\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('happened', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('happened', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"happened\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"happened\" <= s.\"end\"\n AND t.\"happened\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["buyer"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int32"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"buyer\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 18, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 20, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 0, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"precision": 38, "scale": 10, "type": "decimal128"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::numeric AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int16"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["price", "ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "utf8"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT k.\"user_id\" AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('user_id', r.value->0)) AS k\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts", "amount"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"amount\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["user_id"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 2, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"cid\", n.c1 AS \"aid\", n.c2 AS \"seg\", n.c3 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"cid\" ASC NULLS LAST) AS r\n  FROM \"customers\" t\n  WHERE t.\"aid\" = s.\"key\"\n  ORDER BY t.\"cid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["cid", "aid", "seg", "tag"], "key": ["aid"], "order": [{"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}, "table": {"source": ["customers"]}, "window": null}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"cid\", n.c1 AS \"aid\", n.c2 AS \"seg\", n.c3 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"cid\" AS c0, t.\"aid\" AS c1, t.\"seg\" AS c2, t.\"tag\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"cid\" ASC NULLS LAST) AS r\n  FROM \"customers\" t\n  WHERE t.\"aid\" = s.\"key\"\n  ORDER BY t.\"cid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["did", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"did\", n.c1 AS \"cid\", n.c2 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"did\" AS c0, t.\"cid\" AS c1, t.\"ts\" AS c2, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST) AS r\n  FROM \"details\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"did\", n.c1 AS \"hid\", n.c2 AS \"cid\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"did\" AS c0, t.\"hid\" AS c1, t.\"cid\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST) AS r\n  FROM \"details\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"did\", n.c1 AS \"hid\", n.c2 AS \"cid\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"did\" AS c0, t.\"hid\" AS c1, t.\"cid\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"did\" ASC NULLS LAST) AS r\n  FROM \"details\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"did\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "cid", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"did\", n.c1 AS \"hid\", n.c2 AS \"cid\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"did\" AS c0, t.\"hid\" AS c1, t.\"cid\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST) AS r\n  FROM \"details\" t\n  WHERE t.\"hid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"cid\" ASC NULLS LAST, t.\"did\" ASC NULLS LAST, t.\"hid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["did", "hid", "cid", "ts"], "key": ["hid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "did", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"did\", n.c1 AS \"hid\", n.c2 AS \"cid\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"did\" AS c0, t.\"hid\" AS c1, t.\"cid\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"did\" ASC NULLS LAST) AS r\n  FROM \"details\" t\n  WHERE t.\"hid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"did\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["hid", "cid", "ts"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "hid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["headers"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"headers\" AS base FROM \"headers\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"hid\", n.c1 AS \"cid\", n.c2 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"hid\" AS c0, t.\"cid\" AS c1, t.\"ts\" AS c2, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"hid\" ASC NULLS LAST) AS r\n  FROM \"headers\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"hid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["aid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 4, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"cid\", n.c2 AS \"aid\", n.c3 AS \"pid\", n.c4 AS \"sid\", n.c5 AS \"ts\", n.c6 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"cid\" AS c1, t.\"aid\" AS c2, t.\"pid\" AS c3, t.\"sid\" AS c4, t.\"ts\" AS c5, t.\"tag\" AS c6, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"aid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"cid\", n.c2 AS \"aid\", n.c3 AS \"pid\", n.c4 AS \"sid\", n.c5 AS \"ts\", n.c6 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"cid\" AS c1, t.\"aid\" AS c2, t.\"pid\" AS c3, t.\"sid\" AS c4, t.\"ts\" AS c5, t.\"tag\" AS c6, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["cid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 5, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"cid\", n.c2 AS \"aid\", n.c3 AS \"pid\", n.c4 AS \"sid\", n.c5 AS \"ts\", n.c6 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"cid\" AS c1, t.\"aid\" AS c2, t.\"pid\" AS c3, t.\"sid\" AS c4, t.\"ts\" AS c5, t.\"tag\" AS c6, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"cid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["pid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"cid\", n.c2 AS \"aid\", n.c3 AS \"pid\", n.c4 AS \"sid\", n.c5 AS \"ts\", n.c6 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"cid\" AS c1, t.\"aid\" AS c2, t.\"pid\" AS c3, t.\"sid\" AS c4, t.\"ts\" AS c5, t.\"tag\" AS c6, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"pid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["oid", "cid", "aid", "pid", "sid", "ts", "tag"], "key": ["sid"], "order": [{"as_timestamp": true, "column": "ts", "descending": true, "nulls_last": true}, {"as_timestamp": false, "column": "oid", "descending": false, "nulls_last": true}], "per_seed": 32, "request": "neighbors", "seeds": {"attachment": 3, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "time", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": null, "upper": {"inclusive": true, "seed_column": "time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"oid\", n.c1 AS \"cid\", n.c2 AS \"aid\", n.c3 AS \"pid\", n.c4 AS \"sid\", n.c5 AS \"ts\", n.c6 AS \"tag\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"oid\" AS c0, t.\"cid\" AS c1, t.\"aid\" AS c2, t.\"pid\" AS c3, t.\"sid\" AS c4, t.\"ts\" AS c5, t.\"tag\" AS c6, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"sid\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"oid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["aid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"aid\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["details"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"details\" AS base FROM \"details\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"details\" t\n  ON t.\"cid\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["cid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"cid\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["pid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"pid\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns": ["ts"], "key": ["sid"], "order": null, "per_seed": null, "request": "neighbors", "seeds": {"attachment": 1, "columns": [{"name": "key", "nullable": true, "type": {"type": "int64"}}, {"name": "batch", "nullable": false, "type": {"type": "int64"}}, {"name": "end", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}, {"name": "start", "nullable": false, "type": {"type": "timestamp", "unit": "ns", "zone": null}}]}, "table": {"source": ["orders"]}, "window": {"column": "ts", "lower": {"inclusive": false, "seed_column": "start"}, "upper": {"inclusive": true, "seed_column": "end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"sid\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
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
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"CustomerID\"\nFROM \"adventureworks_customers\"\nWHERE \"CustomerID\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["ProductID", "Color"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"ProductID\", \"Color\"\nFROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["ProductID"], "filters": [{"column": "ProductID", "filter": "not_null"}], "key": ["ProductID"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["adventureworks_products"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"ProductID\"\nFROM \"adventureworks_products\"\nWHERE \"ProductID\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["uid"], "filters": [], "key": ["uid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"uid\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "country"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"country\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "d"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"d\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "f_date"], "filters": [{"column": "user_id", "filter": "not_null"}, {"column": "f_date", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"f_date\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL AND \"f_date\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "t"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"t\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id", "tier"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"tier\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["user_id"], "filters": [{"column": "user_id", "filter": "not_null"}], "key": ["user_id"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid", "city"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"aid\", \"city\"\nFROM \"addresses\"\nWHERE \"aid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"aid\"\nFROM \"addresses\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["aid"], "filters": [{"column": "aid", "filter": "not_null"}], "key": ["aid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["addresses"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"aid\"\nFROM \"addresses\"\nWHERE \"aid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["buyers"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"cid\"\nFROM \"buyers\"\nWHERE \"cid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["cid"], "filters": [{"column": "cid", "filter": "not_null"}], "key": ["cid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["customers"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"cid\"\nFROM \"customers\"\nWHERE \"cid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["pid"], "filters": [{"column": "pid", "filter": "not_null"}], "key": ["pid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["products"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"pid\"\nFROM \"products\"\nWHERE \"pid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": ["sid"], "filters": [{"column": "sid", "filter": "not_null"}], "key": ["sid"], "n": 10000, "request": "sample_entities", "seed": {"t": "u64", "v": "42"}, "table": {"source": ["stores"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"sid\"\nFROM \"stores\"\nWHERE \"sid\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"OrderDate\") AS min_date, MAX(\"OrderDate\") AS max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}, {"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"OrderDate\") AS min_date, MAX(\"OrderDate\") AS max_date\nFROM \"adventureworks_sales_order_headers\"\nUNION ALL\nSELECT 1 AS table_index, MIN(\"OrderDate\") AS min_date, MAX(\"OrderDate\") AS max_date\nFROM \"adventureworks_sales_order_details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "OrderDate", "table": {"source": ["adventureworks_sales_order_headers"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"OrderDate\") AS min_date, MAX(\"OrderDate\") AS max_date\nFROM \"adventureworks_sales_order_headers\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "day", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"day\") AS min_date, MAX(\"day\") AS max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "happened", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"happened\") AS min_date, MAX(\"happened\") AS max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"ts\") AS min_date, MAX(\"ts\") AS max_date\nFROM \"orders\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["details"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"ts\") AS min_date, MAX(\"ts\") AS max_date\nFROM \"details\"",
        ],
    ),
    (
        r##"{"columns": [{"column": "ts", "table": {"source": ["lines"]}}, {"column": "ts", "table": {"source": ["orders"]}}], "request": "time_bounds"}"##,
        &[
            "SELECT 0 AS table_index, MIN(\"ts\") AS min_date, MAX(\"ts\") AS max_date\nFROM \"lines\"\nUNION ALL\nSELECT 1 AS table_index, MIN(\"ts\") AS min_date, MAX(\"ts\") AS max_date\nFROM \"orders\"",
        ],
    ),
    // T2.18 (KO2-36): the name.* cases, a column named with a dot, a
    // backtick or a line feed, each quoted alone as any name is.
    (
        r##"{"columns":["a.b","age","country"],"key":["a.b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"a.b\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"a.b\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"a.b\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"a.b\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["a.b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a.b\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a.b"],"filters":[],"key":["a.b"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"a.b\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"key":["a\nb"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"a\nb\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"a\nb\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"a\nb\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"a\nb\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["a\nb","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a\nb\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a\nb"],"filters":[],"key":["a\nb"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"a\nb\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"key":["a`b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"a`b\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"a`b\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"a`b\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"a`b\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["a`b","age","country"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"a`b\", \"age\", \"country\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["a`b"],"filters":[],"key":["a`b"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"a`b\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts","a.b"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.c4 AS \"a.b\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, t.\"a.b\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.c4 AS \"a\nb\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, t.\"a\nb\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.c4 AS \"a`b\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, t.\"a`b\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a.b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["ts","a\nb"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a\nb\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["ts","a`b"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"ts\", t.\"a`b\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","a.b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"a.b\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","a\nb"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"a\nb\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","a`b"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"a`b\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"a.b\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"a.b\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a.b"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"a.b\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"a\nb\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"a\nb\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country","a\nb"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"country\", \"a\nb\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","a`b"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"a`b\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"a`b\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
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
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"score\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","cat"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"cat\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"cat\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"score\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":16,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":17,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":18,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":19,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":20,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":21,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":22,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":23,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":24,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":25,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":26,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":27,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":28,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":29,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":30,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":31,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":32,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":33,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":34,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":35,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":36,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":37,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":38,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":39,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":40,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":41,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":42,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":43,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":44,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":45,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":46,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":47,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":48,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":49,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":50,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":51,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":52,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":53,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":54,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":55,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":56,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":57,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":58,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":59,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":60,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":61,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":62,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":63,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":64,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":65,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":66,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":67,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":68,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":69,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":70,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":71,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":72,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":73,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":74,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":75,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":76,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":77,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":78,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":79,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":80,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":81,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":82,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":83,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":84,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":85,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":86,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":87,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":88,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":89,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":90,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":91,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":92,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":93,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":94,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":95,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":96,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":97,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":98,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":99,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":100,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["price","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"price\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"orders\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["user_id","score"],"filters":[],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"score\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":3,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":4,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":6,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":8,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":9,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"metadata":{"sdm:key_unique":"true"},"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
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
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\"\nFROM \"users\"\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","region","age"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"region\", l.c2 AS \"age\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"region\" AS c1, t.\"age\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","region","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"region\", n.c3 AS \"price\", n.c4 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"region\" AS c2, t.\"price\" AS c3, t.\"ts\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_no\", n.c1 AS \"line\", n.c2 AS \"user_id\", n.c3 AS \"price\", n.c4 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_no\" AS c0, t.\"line\" AS c1, t.\"user_id\" AS c2, t.\"price\" AS c3, t.\"ts\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"line\" ASC NULLS LAST, t.\"order_no\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"line\" ASC NULLS LAST, t.\"order_no\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.c3 AS \"score\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2, t.\"score\" AS c3,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
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
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user\"\nFROM \"users\"\nWHERE \"user\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user","age","country"],"key":["user"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
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
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"orderid\", n.c2 AS \"user_id\", n.c3 AS \"price\", n.c4 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"orderid\" AS c1, t.\"user_id\" AS c2, t.\"price\" AS c3, t.\"ts\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST, t.\"orderid\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST, t.\"orderid\" ASC NULLS LAST, t.\"price\" ASC NULLS LAST, t.\"user_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["user_id","age","feature"],"limit":1000,"request":"preview","table":{"source":["users"]}}"##,
        &["SELECT \"user_id\", \"age\", \"feature\" FROM \"users\" LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","feature"],"key":["user_id"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"feature\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"feature\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"keys":[{"columns":["orderid"],"table":{"source":["orders"]}},{"columns":["user_id"],"table":{"source":["users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT \"orderid\") AS d, COUNT(CASE WHEN \"orderid\" IS NULL THEN 1 END) AS z\nFROM \"orders\"\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT \"user_id\") AS d, COUNT(CASE WHEN \"user_id\" IS NULL THEN 1 END) AS z\nFROM \"users\"",
        ],
    ),
    (
        r##"{"columns":["order_id","orderid","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"orderid","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"orderid\", n.c2 AS \"user_id\", n.c3 AS \"price\", n.c4 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"orderid\" AS c1, t.\"user_id\" AS c2, t.\"price\" AS c3, t.\"ts\" AS c4, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"orderid\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"orderid\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["user_id","age"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["users"]}}"##,
        &[
            "SELECT setseed(-0.31481207155156876)",
            "SELECT \"user_id\", \"age\"\nFROM \"users\"\nWHERE \"user_id\" IS NOT NULL\nORDER BY random()\nLIMIT 10000",
        ],
    ),
    (
        r##"{"columns":["user_id","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT k.\"user_id\" AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"users\" AS base FROM \"users\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('user_id', r.value->0)) AS k\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"country\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":2,"columns":[{"name":"key","nullable":true,"type":{"type":"float64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT k.\"user_id\" AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('user_id', r.value->0)) AS k\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["buys"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'buys'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'buys'\n           AND c.oid = to_regclass('\"buys\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'buys'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"limit":1000,"request":"preview","table":{"source":["buys"]}}"##,
        &["SELECT \"buy_id\", \"user_id\", \"item_id\", \"ts\" FROM \"buys\" LIMIT 1000"],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["items"]}]}"##,
        &[
            "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = current_schema() AND c.table_name = 'items'\n  AND rel.oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = current_schema() AND c.relname = 'items'\n           AND c.oid = to_regclass('\"items\"'))\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = current_schema() AND tc.table_name = 'items'\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"limit":1000,"request":"preview","table":{"source":["items"]}}"##,
        &["SELECT \"item_id\", \"kind\" FROM \"items\" LIMIT 1000"],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["buys"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'buys'\nORDER BY tc.constraint_name, kcu.ordinal_position",
        ],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["items"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       target_kcu.table_schema, target_kcu.table_name,\n       target_kcu.column_name, kcu.ordinal_position\nFROM information_schema.table_constraints tc\nJOIN information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN information_schema.key_column_usage target_kcu\n  ON rc.unique_constraint_catalog = target_kcu.constraint_catalog\n AND rc.unique_constraint_schema = target_kcu.constraint_schema\n AND rc.unique_constraint_name = target_kcu.constraint_name\n AND target_kcu.ordinal_position = kcu.position_in_unique_constraint\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = current_schema() AND tc.table_name = 'items'\nORDER BY tc.constraint_name, kcu.ordinal_position",
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
            "SELECT 0 AS table_index, MIN(\"ts\") AS min_date, MAX(\"ts\") AS max_date\nFROM \"buys\"",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":1,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":3,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"buy_id\", n.c1 AS \"user_id\", n.c2 AS \"item_id\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"buy_id\" AS c0, t.\"user_id\" AS c1, t.\"item_id\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST) AS r\n  FROM \"buys\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":4,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":5,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"item_id\", l.c1 AS \"kind\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"item_id\" AS c0, t.\"kind\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"item_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"items\" t\n    ON t.\"item_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":6,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":7,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":8,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"buy_id\", n.c1 AS \"user_id\", n.c2 AS \"item_id\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"buy_id\" AS c0, t.\"user_id\" AS c1, t.\"item_id\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST) AS r\n  FROM \"buys\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":9,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":10,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"item_id\", l.c1 AS \"kind\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"item_id\" AS c0, t.\"kind\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"item_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"items\" t\n    ON t.\"item_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["item_id","ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{"attachment":11,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}},{"name":"start","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":{"inclusive":false,"seed_column":"start"},"upper":{"inclusive":true,"seed_column":"end"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"end\", w3.\"ts\" AS \"start\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->3)) AS w3\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", t.\"item_id\", t.\"ts\"\nFROM \"__sdm_seeds__\" s\nJOIN \"buys\" t\n  ON t.\"user_id\" = s.\"key\"\n AND t.\"ts\" <= s.\"end\"\n AND t.\"ts\" > s.\"start\"",
        ],
    ),
    (
        r##"{"columns":["user_id","age","country"],"key":["user_id"],"request":"lookup","seeds":{"attachment":12,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["users"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"user_id\", l.c1 AS \"age\", l.c2 AS \"country\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"user_id\" AS c0, t.\"age\" AS c1, t.\"country\" AS c2,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"user_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"users\" t\n    ON t.\"user_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
    (
        r##"{"columns":["buy_id","user_id","item_id","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"buy_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":13,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["buys"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"buys\" AS base FROM \"buys\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"buy_id\", n.c1 AS \"user_id\", n.c2 AS \"item_id\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"buy_id\" AS c0, t.\"user_id\" AS c1, t.\"item_id\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST) AS r\n  FROM \"buys\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"buy_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"key":["user_id"],"order":[{"as_timestamp":true,"column":"ts","descending":true,"nulls_last":true},{"as_timestamp":false,"column":"order_id","descending":false,"nulls_last":true}],"per_seed":32,"request":"neighbors","seeds":{"attachment":14,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"name":"batch","nullable":false,"type":{"type":"int64"}},{"name":"time","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}}]},"table":{"source":["orders"]},"window":{"column":"ts","lower":null,"upper":{"inclusive":true,"seed_column":"time"}}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\", w2.\"ts\" AS \"time\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n  CROSS JOIN (SELECT ROW(t.*)::\"orders\" AS base FROM \"orders\" AS t LIMIT 1) AS b\n  CROSS JOIN LATERAL jsonb_populate_record(b.base, jsonb_build_object('ts', r.value->2)) AS w2\n)\nSELECT s.\"batch\" AS \"__kumo_batch__\", n.c0 AS \"order_id\", n.c1 AS \"user_id\", n.c2 AS \"price\", n.c3 AS \"ts\", n.r AS \"__kumo_row__\"\nFROM \"__sdm_seeds__\" s\nCROSS JOIN LATERAL (\n  SELECT t.\"order_id\" AS c0, t.\"user_id\" AS c1, t.\"price\" AS c2, t.\"ts\" AS c3, ROW_NUMBER() OVER (ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST) AS r\n  FROM \"orders\" t\n  WHERE t.\"user_id\" = s.\"key\"\n    AND t.\"ts\" <= s.\"time\"\n  ORDER BY t.\"ts\" DESC NULLS LAST, t.\"order_id\" ASC NULLS LAST\n  LIMIT 32\n) AS n\nORDER BY s.\"batch\", n.r",
        ],
    ),
    (
        r##"{"columns":["item_id","kind"],"key":["item_id"],"request":"lookup","seeds":{"attachment":15,"columns":[{"name":"key","nullable":true,"type":{"type":"int64"}},{"metadata":{"sdm:count_unread":"true"},"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["items"]}}"##,
        &[
            "WITH \"__sdm_seeds__\" AS (\n  SELECT (r.value->>0)::bigint AS \"key\", (r.value->>1)::bigint AS \"batch\"\n  FROM jsonb_array_elements($1::jsonb) AS r(value)\n)\nSELECT l.b AS \"__kumo_batch__\", l.c0 AS \"item_id\", l.c1 AS \"kind\", l.r AS \"__kumo_row__\", l.m AS \"__kumo_matches__\"\nFROM (\n  SELECT s.\"batch\" AS b, t.\"item_id\" AS c0, t.\"kind\" AS c1,\n         ROW_NUMBER() OVER (PARTITION BY s.\"batch\"\n           ORDER BY t.\"item_id\") AS r,\n         COUNT(*) OVER (PARTITION BY s.\"batch\") AS m\n  FROM \"__sdm_seeds__\" s\n  JOIN \"items\" t\n    ON t.\"item_id\" = s.\"key\"\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        ],
    ),
];

#[test]
fn every_golden_request_renders_its_statements_on_postgres() {
    goldens::render("postgres", GOLDENS);
}

#[test]
fn every_fetch_the_postgres_run_made_is_a_golden() {
    goldens::run_is_pinned("postgres", GOLDENS);
}

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails_on_postgres() {
    goldens::seeded_faults("postgres", GOLDENS);
}
