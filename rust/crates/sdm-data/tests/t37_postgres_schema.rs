// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lakebase is Postgres (T3.7): every Postgres source reads its session's
//! schema once per run, where it names none, and names it in every
//! statement, as the old client reads `current_schema()` once per graph
//! (`Graph.from_postgres`) and writes `"schema"."table"` into every read
//! (`old-behaviour/T3.7-lakebase.md`, rows 16, 21 and 38).
//!
//! - A source given no schema is settled first: the one statement that
//!   asks the session is the namespace read, the old client's own
//!   `SELECT current_schema()`.
//! - Every request kind, under a schema, names it in every statement, and
//!   none asks the session for its schema or finds a relation by the
//!   search path (`to_regclass`): a temporary or `pg_catalog` relation of
//!   a table's name is never read in its place, nor refused.
//! - The catalog's column read is the old client's statement, but for the
//!   schema and table it binds and the core writes in place, as recorded
//!   on Lakebase (C-1, `contract/captures/lakebase/base.binary`).
//! - A dotted name is one table under the schema, as the old client never
//!   splits a name (row 20).

use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, FetchPlan, ProviderChoice, Relation,
};
use sdm_data::Data;
use serde_json::{Value, json};

/// Every data request kind on every base (`contract/request-kinds.json`).
const KINDS: &str = include_str!("../../../../contract/request-kinds.json");

/// The old client's column read on Lakebase, as C-1 recorded it for
/// `base.binary` (`PostgresTable._get_source_columns`), its two `%s` the
/// schema and the table it binds.
const REFERENCE_COLUMNS: &str = "SELECT c.column_name, c.data_type, c.udt_name,\n       c.numeric_scale, c.is_nullable, c.table_schema,\n       c.table_name,\n       pg_catalog.format_type(a.atttypid, a.atttypmod)\nFROM information_schema.columns c\nJOIN pg_catalog.pg_namespace n\n  ON n.nspname = c.table_schema\nJOIN pg_catalog.pg_class rel\n  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\nJOIN pg_catalog.pg_attribute a\n  ON a.attrelid = rel.oid AND a.attname = c.column_name\n AND a.attnum > 0 AND NOT a.attisdropped\nWHERE c.table_schema = %s AND c.table_name = %s\nORDER BY c.ordinal_position";

fn postgres(namespace: Value) -> ProviderChoice {
    serde_json::from_value(json!({
        "provider": "sql", "dialect": "postgres", "namespace": namespace
    }))
    .unwrap()
}

fn statements(request: &DataRequest, provider: &ProviderChoice) -> Vec<String> {
    match Data::new().plan(request, provider) {
        Ok(FetchPlan::Sql { statements, .. }) => statements.into_iter().map(|s| s.text).collect(),
        other => panic!("{request:?}: not planned as SQL: {other:?}"),
    }
}

fn describe(part: DescribePart, tables: Vec<Relation>) -> DataRequest {
    DataRequest::Describe { tables, part }
}

#[test]
fn a_source_given_no_schema_reads_the_session_s_once_and_reads_every_table_under_it() {
    let data = Data::new();
    let given_none = postgres(json!([]));
    assert!(data.unsettled(&given_none));
    assert!(data.unsettled(&postgres(json!([null]))));
    assert!(!data.unsettled(&postgres(json!(["shop"]))));

    // The one read of the session: the old client's own statement
    // (`Graph.from_postgres`), its schema null where the search path holds
    // none, which reads as no schema.
    let asked = describe(DescribePart::Namespace, vec![]);
    assert_eq!(statements(&asked, &given_none), ["SELECT current_schema()"]);
    let answer = |schema: Option<&str>| {
        data.read(
            &asked,
            &given_none,
            &[arrow_array::RecordBatch::try_from_iter([(
                "current_schema",
                std::sync::Arc::new(arrow_array::StringArray::from(vec![schema]))
                    as arrow_array::ArrayRef,
            )])
            .unwrap()],
        )
        .unwrap()
    };
    assert_eq!(
        answer(Some("sess")),
        CatalogAnswer::Namespace(vec!["sess".into()])
    );
    assert_eq!(answer(None), CatalogAnswer::Namespace(vec![]));

    // Every later read names the schema the session answered.
    let settled = data.settled(&given_none, vec!["sess".into()]);
    let preview = DataRequest::Preview {
        table: Relation::named("orders"),
        columns: vec!["id".into()],
        limit: 1000,
    };
    assert_eq!(
        statements(&preview, &settled),
        [r#"SELECT "id" FROM "sess"."orders" LIMIT 1000"#]
    );
    assert_eq!(
        statements(&describe(DescribePart::Tables, vec![]), &settled),
        [
            "SELECT table_name\nFROM information_schema.tables\nWHERE table_schema = 'sess'\n  AND table_type = 'BASE TABLE'\nORDER BY table_name"
        ]
    );
}

/// Each request kind of `contract/request-kinds.json`, its seeds given as
/// the keyed fetches give them.
fn kinds() -> Vec<(String, DataRequest)> {
    let kinds: Value = serde_json::from_str(KINDS).unwrap();
    kinds["fetches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fetch| {
            let mut request = fetch["request"].clone();
            if fetch.get("keys").is_some() {
                request["seeds"] = json!({
                    "attachment": 0,
                    "columns": [
                        {"name": "key", "type": {"type": "int64"}, "nullable": true},
                        {"name": "batch", "type": {"type": "int64"}, "nullable": true},
                    ],
                });
            }
            (
                format!("{}: {}", fetch["base"], fetch["name"]),
                serde_json::from_value(request).unwrap(),
            )
        })
        .collect()
}

#[test]
fn every_request_kind_names_the_schema_in_every_statement_and_never_asks_the_session() {
    let shop = postgres(json!(["shop"]));
    let mut catalog = vec![
        describe(DescribePart::Tables, vec![]),
        describe(DescribePart::Indexes, vec![]),
    ];
    for part in [
        DescribePart::Tables,
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
        DescribePart::IndexColumns,
    ] {
        catalog.push(describe(part, vec![Relation::named("orders")]));
    }
    let mut planned = 0;
    for (place, request) in kinds()
        .into_iter()
        .chain(catalog.into_iter().map(|r| (r.name().to_owned(), r)))
    {
        let names: Vec<String> = request
            .relations()
            .iter()
            .map(|(_, r)| r.source.join("."))
            .collect();
        for text in statements(&request, &shop) {
            assert!(!text.contains("current_schema"), "{place}: {text}");
            assert!(!text.contains("regclass"), "{place}: {text}");
            assert!(!text.contains("search_path"), "{place}: {text}");
            for name in &names {
                let quoted = format!(r#""{name}""#);
                let literal = format!("'{name}'");
                assert!(
                    text.contains(&format!(r#""shop".{quoted}"#))
                        || (text.contains("'shop'") && text.contains(&literal))
                        || !(text.contains(&quoted) || text.contains(&literal)),
                    "{place}: {name} is read under no schema: {text}"
                );
            }
            planned += 1;
        }
    }
    assert!(planned >= 50, "{planned}");
}

#[test]
fn the_column_read_is_the_old_client_s_statement_with_the_schema_and_table_in_place() {
    let read = statements(
        &describe(DescribePart::Columns, vec![Relation::named("orders")]),
        &postgres(json!(["sdm_shop"])),
    );
    assert_eq!(
        read,
        [REFERENCE_COLUMNS
            .replacen("%s", "'sdm_shop'", 1)
            .replacen("%s", "'orders'", 1)]
    );
}

#[test]
fn a_named_table_is_found_in_its_schema_never_where_the_search_path_finds_it() {
    let read = statements(
        &describe(DescribePart::Tables, vec![Relation::named("pg_roles")]),
        &postgres(json!(["shop"])),
    );
    assert_eq!(
        read,
        [
            "SELECT relname AS table_name\nFROM pg_catalog.pg_class\nWHERE oid = (SELECT c.oid FROM pg_catalog.pg_class c\n         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace\n         WHERE n.nspname = 'shop' AND c.relname = 'pg_roles')"
        ]
    );
}

#[test]
fn a_dotted_name_is_one_table_under_the_schema() {
    let scan = DataRequest::Scan {
        table: Relation::named("shop.users"),
        columns: vec!["id".into()],
        filters: vec![],
        limit: None,
    };
    let text = statements(&scan, &postgres(json!(["sess"]))).join("\n");
    assert!(text.contains(r#"FROM "sess"."shop.users""#), "{text}");
    let columns = statements(
        &describe(DescribePart::Columns, vec![Relation::named("shop.users")]),
        &postgres(json!(["sess"])),
    )
    .join("\n");
    assert!(
        columns.contains("WHERE c.table_schema = 'sess' AND c.table_name = 'shop.users'"),
        "{columns}"
    );
}
