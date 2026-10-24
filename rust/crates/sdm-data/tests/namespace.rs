// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A SQL provider's namespace (contract v28, T3.8): a Postgres source given
//! its schema is read under it, every table qualified by it and the source's
//! tables listed in it, as the reference's `PostgresTable` and
//! `Graph.from_postgres` read the schema they were given; a part left
//! `null` is the connection's current one, as before v28; DuckDB's and
//! SQLite's sources name none, so a namespace given them is refused by
//! name. A Postgres source given no schema has the session's current one
//! read once per run and every relation named under it (T3.7).

use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation,
};
use sdm_contract::{Detail, ErrorCode, Sentence};
use sdm_data::Data;

fn sql(name: &str, namespace: &[Option<&str>]) -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new(name),
        namespace: namespace.iter().map(|p| p.map(str::to_owned)).collect(),
        view: None,
    }
}

fn text(request: &DataRequest, provider: &ProviderChoice) -> String {
    match Data::new().plan(request, provider).unwrap() {
        FetchPlan::Sql { statements, .. } => statements
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        FetchPlan::Host { .. } => panic!("a SQL dialect planned a host fetch"),
    }
}

fn listing() -> DataRequest {
    DataRequest::Describe {
        tables: vec![],
        part: DescribePart::Tables,
    }
}

fn preview(table: Relation) -> DataRequest {
    DataRequest::Preview {
        table,
        columns: vec!["id".into()],
        limit: 10,
    }
}

#[test]
fn a_postgres_namespace_lists_and_reads_under_its_schema() {
    let under = sql("postgres", &[Some("sales")]);
    assert!(text(&listing(), &under).contains("WHERE table_schema = 'sales'"));
    assert_eq!(
        text(&preview(Relation::named("orders")), &under),
        r#"SELECT "id" FROM "sales"."orders" LIMIT 10"#
    );
    // A relation naming its schema reads where it says.
    let named = Relation {
        columns: Vec::new(),
        filter: None,
        found: false,
        source: vec!["other".into(), "orders".into()],
    };
    assert!(text(&preview(named), &under).contains(r#"FROM "other"."orders""#));
}

/// A Postgres source given no schema is read under the session's current
/// one, which the core reads once per run before any table (T3.7): until
/// then a relation of one part is under no schema, refused as the
/// reference refuses a session with none, never read where the search path
/// finds it.
#[test]
fn no_namespace_or_a_null_part_reads_no_relation_before_the_schema_is_settled() {
    for current in [sql("postgres", &[]), sql("postgres", &[None])] {
        for request in [listing(), preview(Relation::named("orders"))] {
            let refused = Data::new().plan(&request, &current).unwrap_err();
            assert_eq!(refused.sentence, Sentence::NoCurrentSchema, "{request:?}");
            assert_eq!(refused.code, ErrorCode::InvalidRequest);
            assert_eq!(
                refused.fields.get("database"),
                Some(&Detail::text("postgres"))
            );
        }
    }
}

#[test]
fn a_namespace_the_dialect_cannot_write_is_refused_by_name() {
    for (provider, request) in [
        (sql("postgres", &[Some("db"), Some("sales")]), listing()),
        (
            sql("postgres", &[Some("db"), None]),
            preview(Relation::named("t")),
        ),
        (sql("duckdb", &[Some("main")]), listing()),
        (
            sql("sqlite", &[Some("main")]),
            preview(Relation::named("t")),
        ),
    ] {
        let refused = Data::new().plan(&request, &provider).unwrap_err();
        assert_eq!(refused.code, ErrorCode::UnsupportedRequest, "{provider:?}");
        assert_eq!(
            refused.fields.get("field"),
            Some(&Detail::text("namespace")),
            "{provider:?}: {refused:?}"
        );
    }
    // DuckDB and SQLite read as before with a null part, their current one.
    assert!(
        Data::new()
            .plan(&listing(), &sql("duckdb", &[None]))
            .is_ok()
    );
}

/// The namespace a relation of one part is read in, which the core asks for
/// only to name a table the source lacks (T3.8 fix 2): a Postgres source's
/// given schema, or its current one; none for DuckDB's and SQLite's, whose
/// missing tables the reference names alone.
#[test]
fn the_namespace_asked_for_is_the_one_a_table_is_named_under() {
    let asked = DataRequest::Describe {
        tables: vec![],
        part: DescribePart::Namespace,
    };
    // The old client's own read of the session's schema (T3.7).
    assert_eq!(
        text(&asked, &sql("postgres", &[])),
        "SELECT current_schema()"
    );
    assert_eq!(
        text(&asked, &sql("postgres", &[None])),
        "SELECT current_schema()"
    );
    assert_eq!(
        text(&asked, &sql("postgres", &[Some("sales")])),
        "SELECT 'sales' AS current_schema"
    );
    for alone in ["duckdb", "sqlite"] {
        let statement = text(&asked, &sql(alone, &[]));
        assert!(
            statement.contains("WHERE FALSE") || statement.contains("WHERE 0"),
            "{statement}"
        );
    }
}
