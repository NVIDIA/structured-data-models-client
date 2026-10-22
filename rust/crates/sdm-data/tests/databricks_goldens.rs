// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4 on Databricks (T3.4, the brief's exit 2): the goldens of every
//! request kind the `databricks` dialect renders, each beside the statement
//! the old client sent for the same request, so the diff between them is
//! reviewed (non-gating, in the package's evidence) and the statements the
//! core must copy verbatim are held to the old client's.
//!
//! The reference side is `contract/fixtures/databricks/reference-statements.json`,
//! recorded by `contract/fixtures/databricks/record.py` from the reference
//! (43e55f4f) running the cases on a stand-in Databricks connection, until
//! C-1's recorded statements replace it; `record.py --diff` prints each
//! golden beside the reference's statement of its kind, the evidence's
//! diff. The relations are `main`.`shop`.<table>, the catalog and schema
//! the fixture records under (its `{catalog}` and `{schema}`).
//!
//! What gates, by the design (choice 2, exact replay; choice 7 and KO3-41,
//! the old client's sampling SQL; Scope item 1, key sets bound as the old
//! client binds them):
//! - each golden request renders exactly its statements ([`goldens::render`]),
//!   and every request kind has one;
//! - the catalog statements are the old client's, text and parameters, after
//!   whitespace is normalised, so the stand-in replays C-1's catalog answers
//!   and the catalog reader reads the warehouse's real rows;
//! - every entity draw the old client writes (the percentage as Python's
//!   `str(float)`, `REPEATABLE (seed)`, `LIMIT n`) is a golden's statement;
//! - every key set is bound through `posexplode(from_json(?, 'array<...>'))`;
//! - each part of a relation is quoted alone with backticks, a backtick in a
//!   name doubled, and a relation of more than three parts refused.
//!
//! The other statements' text is the dialect's, pinned here and reviewed
//! against the old client's: a golden seeded with the old client's text may
//! be changed with the request it answers, the diff then shown by
//! `record.py --diff`.

mod goldens;

use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderChoice, SqlParam};
use sdm_contract::scalar::Scalar;
use sdm_data::Data;
use serde_json::Value;

/// The old client's statements, recorded (see the module's documentation).
const REFERENCE: &str =
    include_str!("../../../../contract/fixtures/databricks/reference-statements.json");

/// Each request, as the contract writes it, and the statements its plan
/// carries on Databricks, in order. Seeded by T3.4's test author with the
/// old client's statement where the request is the one the old client
/// answered with it; the dialect's own goldens join them.
const GOLDENS: goldens::Goldens = &[
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["main","shop","orders"]}]}"##,
        &[
            "SELECT c.column_name, c.full_data_type, c.is_nullable,\n       c.table_catalog, c.table_schema, c.table_name,\n       t.table_type\nFROM `main`.information_schema.columns c\nJOIN `main`.information_schema.tables t\n  ON t.table_catalog = c.table_catalog\n AND t.table_schema = c.table_schema\n AND t.table_name = c.table_name\nWHERE c.table_schema = lower(?)\n  AND c.table_name = lower(?)\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part":"columns","request":"describe","tables":[{"source":["main","shop","users"]}]}"##,
        &[
            "SELECT c.column_name, c.full_data_type, c.is_nullable,\n       c.table_catalog, c.table_schema, c.table_name,\n       t.table_type\nFROM `main`.information_schema.columns c\nJOIN `main`.information_schema.tables t\n  ON t.table_catalog = c.table_catalog\n AND t.table_schema = c.table_schema\n AND t.table_name = c.table_name\nWHERE c.table_schema = lower(?)\n  AND c.table_name = lower(?)\nORDER BY c.ordinal_position",
        ],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["main","shop","orders"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name\nFROM `main`.information_schema.table_constraints tc\nJOIN `main`.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')",
        ],
    ),
    (
        r##"{"part":"keys","request":"describe","tables":[{"source":["main","shop","users"]}]}"##,
        &[
            "SELECT tc.constraint_type, tc.constraint_name,\n       kcu.column_name\nFROM `main`.information_schema.table_constraints tc\nJOIN `main`.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nWHERE tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)\n  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')",
        ],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["main","shop","orders"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       ccu.table_catalog, ccu.table_schema,\n       ccu.table_name, ccu.column_name\nFROM `main`.information_schema.table_constraints tc\nJOIN `main`.information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN `main`.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN `main`.information_schema.constraint_column_usage ccu\n  ON rc.unique_constraint_catalog = ccu.constraint_catalog\n AND rc.unique_constraint_schema = ccu.constraint_schema\n AND rc.unique_constraint_name = ccu.constraint_name\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)",
        ],
    ),
    (
        r##"{"part":"foreign_keys","request":"describe","tables":[{"source":["main","shop","users"]}]}"##,
        &[
            "SELECT tc.constraint_name, kcu.column_name,\n       ccu.table_catalog, ccu.table_schema,\n       ccu.table_name, ccu.column_name\nFROM `main`.information_schema.table_constraints tc\nJOIN `main`.information_schema.referential_constraints rc\n  ON tc.constraint_catalog = rc.constraint_catalog\n AND tc.constraint_schema = rc.constraint_schema\n AND tc.constraint_name = rc.constraint_name\nJOIN `main`.information_schema.key_column_usage kcu\n  ON tc.constraint_catalog = kcu.constraint_catalog\n AND tc.constraint_schema = kcu.constraint_schema\n AND tc.constraint_name = kcu.constraint_name\nJOIN `main`.information_schema.constraint_column_usage ccu\n  ON rc.unique_constraint_catalog = ccu.constraint_catalog\n AND rc.unique_constraint_schema = ccu.constraint_schema\n AND rc.unique_constraint_name = ccu.constraint_name\nWHERE tc.constraint_type = 'FOREIGN KEY'\n  AND tc.table_schema = lower(?)\n  AND tc.table_name = lower(?)",
        ],
    ),
    (
        r##"{"columns":["order_id","user_id","price","ts"],"limit":1000,"request":"preview","table":{"source":["main","shop","orders"]}}"##,
        &["SELECT `order_id`, `user_id`, `price`, `ts` FROM `main`.`shop`.`orders` LIMIT 1000"],
    ),
    (
        r##"{"columns":["user_id","age","country","home.city","o'clock","back`tick","gr\u00f6\u00dfe"],"limit":1000,"request":"preview","table":{"source":["main","shop","users"]}}"##,
        &[
            "SELECT `user_id`, `age`, `country`, `home.city`, `o'clock`, `back``tick`, `gr\u{f6}\u{df}e` FROM `main`.`shop`.`users` LIMIT 1000",
        ],
    ),
    (
        r##"{"filters":[],"request":"count_rows","table":{"source":["main","shop","orders"]}}"##,
        &["SELECT COUNT(*) FROM `main`.`shop`.`orders`"],
    ),
    (
        r##"{"keys":[{"columns":["order_id"],"table":{"source":["main","shop","orders"]}},{"columns":["user_id"],"table":{"source":["main","shop","users"]}}],"request":"key_counts"}"##,
        &[
            "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT `order_id`) AS d, COUNT(CASE WHEN `order_id` IS NULL THEN 1 END) AS z\nFROM `main`.`shop`.`orders`\nUNION ALL\nSELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT `user_id`) AS d, COUNT(CASE WHEN `user_id` IS NULL THEN 1 END) AS z\nFROM `main`.`shop`.`users`",
        ],
    ),
    (
        r##"{"key":["user_id"],"request":"repeated_key","table":{"source":["main","shop","users"]}}"##,
        &[
            "SELECT `user_id`\nFROM `main`.`shop`.`users`\nWHERE `user_id` IS NOT NULL\nGROUP BY `user_id`\nHAVING COUNT(*) > 1\nLIMIT 1",
        ],
    ),
    (
        r##"{"columns":["ts"],"request":"column_ranges","table":{"source":["main","shop","visits"]}}"##,
        &["SELECT COUNT(*), MIN(`ts`), MAX(`ts`), COUNT(`ts`)\nFROM `main`.`shop`.`visits`"],
    ),
    (
        r##"{"columns":[{"column":"ts","table":{"source":["main","shop","orders"]}}],"request":"time_bounds"}"##,
        &[
            "SELECT\n  0 as table_index,\n  MIN(`ts`) as min_date,\n  MAX(`ts`) as max_date\nFROM `main`.`shop`.`orders`",
        ],
    ),
    (
        r##"{"columns":["user_id"],"filters":[{"column":"user_id","filter":"not_null"}],"key":["user_id"],"n":10000,"request":"sample_entities","seed":{"t":"u64","v":"42"},"table":{"source":["main","shop","users"]}}"##,
        &[
            "SELECT `user_id`\nFROM `main`.`shop`.`users`\nTABLESAMPLE (100.0 PERCENT) REPEATABLE (42)\nWHERE `user_id` IS NOT NULL\nLIMIT 10000",
        ],
    ),
];

/// The `databricks` dialect's choice, built from the contract's JSON so the
/// test reads whatever optional fields the choice gains.
fn databricks() -> ProviderChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"databricks"}"#).unwrap()
}

/// The choice naming the source's catalog and schema (T3.8's `namespace`),
/// which a listing of the schema's tables reads, no table naming them
/// (`{catalog}.information_schema.tables WHERE table_schema = ?`).
fn databricks_in_main_shop() -> ProviderChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"databricks","namespace":["main","shop"]}"#)
        .unwrap()
}

/// The choice a golden request is planned on: the namespaced one for a
/// listing of the schema's tables, the plain one otherwise.
fn choice_for(request: &DataRequest) -> ProviderChoice {
    match request {
        DataRequest::Describe { tables, .. } if tables.is_empty() => databricks_in_main_shop(),
        _ => databricks(),
    }
}

fn reference() -> Value {
    serde_json::from_str(REFERENCE).unwrap()
}

/// `text` with every run of whitespace one space, as exact replay compares
/// statements (design choice 2).
fn normalised(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn request(text: &str) -> DataRequest {
    serde_json::from_str(text).unwrap_or_else(|error| panic!("{text}: {error}"))
}

/// The statements `request` plans on Databricks, with their parameters.
fn planned(request: &DataRequest) -> Vec<(String, Vec<SqlParam>)> {
    match Data::new().plan(request, &choice_for(request)) {
        Ok(FetchPlan::Sql { statements, .. }) => statements
            .into_iter()
            .map(|statement| {
                assert_eq!(statement.dialect.name(), "databricks");
                (statement.text, statement.params)
            })
            .collect(),
        other => panic!("{request:?}: not planned as Databricks SQL: {other:?}"),
    }
}

/// A request's kind as the reference file names it: the request's name,
/// and a describe's part.
fn kind(request: &DataRequest) -> String {
    match request {
        DataRequest::Describe { part, tables } => {
            let part = serde_json::to_value(part).unwrap();
            let named = if tables.is_empty() { "" } else { "/named" };
            format!("describe/{}{named}", part.as_str().unwrap())
        }
        other => other.name().to_owned(),
    }
}

fn the_reference_s(kind: &str) -> String {
    let reference = reference();
    let statements = reference["kinds"][kind].as_array().unwrap();
    assert!(!statements.is_empty(), "the reference holds no {kind}");
    statements[0]["text"].as_str().unwrap().to_owned()
}

/// A text parameter's value.
fn text_of(param: &SqlParam) -> Option<&str> {
    match param {
        SqlParam::Value {
            value: Scalar::Utf8(text),
        } => Some(text),
        _ => None,
    }
}

#[test]
fn every_golden_request_renders_its_statements_on_databricks() {
    goldens::render("databricks", GOLDENS);
}

/// L4's run check, as on every dialect: the run `SDM_L4_OBSERVATIONS` names
/// (none until a Databricks leg's run is named in CI) holds no fetch that is
/// no golden.
#[test]
fn every_fetch_the_databricks_run_made_is_a_golden() {
    goldens::run_is_pinned("databricks", GOLDENS);
}

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails_on_databricks() {
    goldens::seeded_faults("databricks", GOLDENS);
}

/// Exit 2: "the goldens of every request kind". Every kind of
/// `DataRequest` a SQL dialect answers, each describe part a Databricks
/// catalog has (it declares no index), and the listing of a schema's tables
/// with none named (the old client's `Graph.from_databricks`) beside one
/// with tables named.
#[test]
fn every_request_kind_has_a_golden_on_databricks() {
    let kinds: Vec<String> = GOLDENS
        .iter()
        .map(|(text, _)| kind(&request(text)))
        .collect();
    let missing: Vec<&str> = [
        "describe/tables",
        "describe/tables/named",
        "describe/columns/named",
        "describe/keys/named",
        "describe/foreign_keys/named",
        "scan",
        "preview",
        "count_rows",
        "key_counts",
        "time_bounds",
        "column_ranges",
        "repeated_key",
        "sample_entities",
        "neighbors",
        "lookup",
    ]
    .into_iter()
    .filter(|wanted| !kinds.iter().any(|k| k == wanted))
    .collect();
    assert!(missing.is_empty(), "no Databricks golden of: {missing:?}");
}

/// Exits 2 and 4: the catalog statements are the old client's (DB/table.py
/// 212-246, 271-300, 318-370; RE/graph.py:1199), text and parameters, so
/// exact replay serves C-1's catalog answers to the core's catalog reader
/// (design choice 2). The schema and the table are bound as the old client
/// binds them, `lower(?)` each, never written into the text.
#[test]
fn the_catalog_statements_are_the_old_client_s_with_its_parameters() {
    let mut seen = Vec::new();
    for (text, _) in GOLDENS {
        let request = request(text);
        let DataRequest::Describe { tables, .. } = &request else {
            continue;
        };
        let kind = kind(&request);
        let reference_kind = kind.trim_end_matches("/named");
        if reference_kind == "describe/tables" && !tables.is_empty() {
            // The old client has no statement for a listing of named tables.
            continue;
        }
        let statements = planned(&request);
        let (last, params) = statements.last().unwrap();
        assert_eq!(
            normalised(last),
            normalised(&the_reference_s(reference_kind)),
            "{text}: not the old client's catalog statement"
        );
        let bound: Vec<Option<&str>> = params.iter().map(text_of).collect();
        let wanted: Vec<Option<&str>> = match tables.as_slice() {
            [] => vec![Some("shop")],
            [table] => {
                let parts = &table.source;
                assert_eq!(
                    parts.len(),
                    3,
                    "{text}: a golden names catalog, schema and table"
                );
                vec![Some(parts[1].as_str()), Some(parts[2].as_str())]
            }
            _ => panic!("{text}: the old client describes one table a statement"),
        };
        assert_eq!(bound, wanted, "{text}: not the old client's parameters");
        seen.push(reference_kind.to_owned());
    }
    for wanted in [
        "describe/tables",
        "describe/columns",
        "describe/keys",
        "describe/foreign_keys",
    ] {
        assert!(seen.iter().any(|k| k == wanted), "no golden of {wanted}");
    }
}

/// Exit 2, RS-6 under KO3-41 (design choice 7): every entity draw the old
/// client writes (DB/sampler.py:201-277), its percentage Python's `str` of
/// `min(100.0, 100.0 * n / rows * 3.0)` (scientific below 1e-4, the inexact
/// `0.30000000000000004`, capped at `100.0`, whole as `6.0`), its seed 0,
/// negative and past 2**31, is a golden's statement, so the same rows are
/// drawn live; the stand-in refuses a draw with another fraction text or
/// seed by name.
#[test]
fn every_draw_the_old_client_writes_is_a_golden_s_statement() {
    let rendered: Vec<String> = GOLDENS
        .iter()
        .filter(|(text, _)| matches!(request(text), DataRequest::SampleEntities { .. }))
        .flat_map(|(text, _)| planned(&request(text)))
        .map(|(statement, _)| normalised(&statement))
        .collect();
    let reference = reference();
    let draws = reference["sampling"].as_array().unwrap();
    assert!(draws.len() >= 8);
    let missing: Vec<String> = draws
        .iter()
        .filter(|draw| !rendered.contains(&normalised(draw["text"].as_str().unwrap())))
        .map(|draw| {
            format!(
                "n {} of {} rows, seed {}",
                draw["n"], draw["rows"], draw["seed"]
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "no golden draws as the old client: {missing:?}"
    );
}

/// Exit 2, Scope item 1: "key sets bound as the old client binds them
/// (`posexplode(from_json(?, ...))`)" (DB/sampler.py:233-241, 339-365,
/// 396-482): every statement binding a key set reads it so, as one JSON
/// text parameter.
#[test]
fn key_sets_are_bound_as_the_old_client_binds_them() {
    let mut bound = 0;
    for (text, _) in GOLDENS {
        let request = request(text);
        if !matches!(
            request,
            DataRequest::Lookup { .. } | DataRequest::Neighbors { .. }
        ) {
            continue;
        }
        let statements = planned(&request);
        let binding: Vec<&(String, Vec<SqlParam>)> = statements
            .iter()
            .filter(|(_, params)| params.iter().any(|p| matches!(p, SqlParam::KeySet { .. })))
            .collect();
        assert!(!binding.is_empty(), "{text}: binds no key set");
        for (statement, _) in binding {
            assert!(
                statement.contains("posexplode(from_json(?, 'array<"),
                "{text}: a key set not bound as the old client binds it:\n{statement}"
            );
            bound += 1;
        }
    }
    assert!(bound > 0, "no golden binds a key set");
}

/// Design choice 4: each part of a relation quoted alone with backticks,
/// never split on a dot, a backtick in a name doubled (DB/table.py:37, :62,
/// 170-195; CN/sql.py:247).
#[test]
fn each_name_part_is_quoted_alone_with_backticks() {
    let preview = request(
        r#"{"columns":["home.city","back`tick"],"limit":1000,"request":"preview","table":{"source":["my-catalog","sch.ema","o`dd"]}}"#,
    );
    let statements = planned(&preview);
    let (text, _) = statements.last().unwrap();
    for quoted in [
        "`my-catalog`.`sch.ema`.`o``dd`",
        "`home.city`",
        "`back``tick`",
    ] {
        assert!(text.contains(quoted), "{quoted} not in:\n{text}");
    }
}

/// Design choice 4: Databricks takes one to three name parts (catalog,
/// schema, table); none or more is refused by name, as `relation` refuses.
#[test]
fn a_relation_of_no_part_or_more_than_three_is_refused() {
    for source in [r#"[]"#, r#"["a","b","c","d"]"#] {
        let preview = request(&format!(
            r#"{{"columns":["x"],"limit":1,"request":"preview","table":{{"source":{source}}}}}"#
        ));
        let refused = Data::new().plan(&preview, &databricks());
        assert!(refused.is_err(), "{source}: planned: {refused:?}");
    }
}
