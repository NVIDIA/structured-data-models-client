// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snowflake query planning, catalog interpretation and bound-key semantics.
//! Small authored inputs check behavior independently of recorded client outputs.
//! Four connector-produced Arrow samples retain external type interoperability coverage.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::catalog::{CatalogAnswer, CatalogForeignKey, KeyKind};
use sdm_contract::data::{
    DataLayer, DataRequest, Determinism, Dialect, ExpectedSchema, FetchPlan, NullOrder,
    ProviderChoice, ProviderError, Relation, SqlParam,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Detail, ErrorKind, Scalar, Sentence};
use serde_json::{Value, json};

/// The database and schema the recordings name, as their placeholders.
const DB: &str = "{catalog}";
const SCHEMA: &str = "{schema}";

fn captures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/fixtures/maintained")
}

fn arrow(_case: &str, file: &str) -> Vec<RecordBatch> {
    let path = captures().join("snowflake").join(file);
    let reader =
        arrow_ipc::reader::StreamReader::try_new(BufReader::new(File::open(path).unwrap()), None)
            .unwrap();
    reader.map(Result::unwrap).collect()
}

fn layer() -> Arc<dyn DataLayer> {
    Arc::clone(sdm_host::default_root().data_layer())
}

/// The Snowflake provider's choice, its database and schema each given or
/// left to the session.
fn snowflake(namespace: &[Option<&str>]) -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new("snowflake"),
        namespace: namespace.iter().map(|p| p.map(str::to_owned)).collect(),
        view: None,
    }
}

fn named() -> ProviderChoice {
    snowflake(&[Some(DB), Some(SCHEMA)])
}

fn relation(parts: &[&str]) -> Relation {
    Relation::of(parts.iter().map(|p| (*p).to_owned()).collect())
}

/// `request`'s plan on `provider`: each statement's text and parameters.
fn planned(
    provider: &ProviderChoice,
    request: Value,
) -> Result<Vec<(String, Vec<String>)>, CoreError> {
    let request: DataRequest = serde_json::from_value(request).unwrap();
    let FetchPlan::Sql { statements, .. } = layer().plan(&request, provider)? else {
        panic!("not a SQL plan");
    };
    Ok(statements
        .into_iter()
        .map(|s| {
            assert_eq!(s.dialect.name(), "snowflake");
            let params = s
                .params
                .into_iter()
                .map(|p| match p {
                    SqlParam::Value {
                        value: Scalar::Utf8(text),
                    } => text,
                    other => panic!("a parameter that is not text: {other:?}"),
                })
                .collect();
            (s.text, params)
        })
        .collect())
}

/// Check one SQL request without a historical statement-membership oracle.
fn one_statement(_case: &str, provider: &ProviderChoice, request: Value) -> (String, Vec<String>) {
    let mut statements = planned(provider, request.clone()).unwrap();
    assert_eq!(statements.len(), 1, "{request}: {statements:?}");
    let statement = statements.remove(0);
    let (text, params) = &statement;
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let quote = |name: &str| format!("\"{}\"", name.replace('"', "\"\""));
    let source = |name: &str| format!("\"{DB}\".\"{SCHEMA}\".{}", quote(name));
    match request["request"].as_str().unwrap() {
        "describe" => match request["part"].as_str().unwrap() {
            "namespace" => assert_eq!(text, "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()"),
            "view_definition" => assert_eq!(text, "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)"),
            "key_enforcement" => {
                assert_eq!(text, &format!(r#"SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}""#))
            }
            part => panic!("unchecked description {part}"),
        },
        "preview" => {
            let name = request["table"]["source"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .as_str()
                .unwrap();
            let columns = request["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| quote(c.as_str().unwrap()))
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(
                flat,
                format!(
                    "SELECT {columns} FROM {} LIMIT {}",
                    source(name),
                    request["limit"]
                )
            );
        }
        "key_counts" => {
            let keys = request["keys"].as_array().unwrap();
            let parts = flat.split(" UNION ALL ").collect::<Vec<_>>();
            assert_eq!(parts.len(), keys.len());
            for (i, (key, part)) in keys.iter().zip(parts).enumerate() {
                let columns = key["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| quote(c.as_str().unwrap()))
                    .collect::<Vec<_>>();
                let missing = columns
                    .iter()
                    .map(|c| format!("{c} IS NULL"))
                    .collect::<Vec<_>>()
                    .join(" OR ");
                let name = key["table"]["source"][0].as_str().unwrap();
                assert_eq!(
                    part,
                    format!(
                        "SELECT {i} AS k, COUNT(*) AS n, COUNT(DISTINCT {}) AS d, COUNT(CASE WHEN {missing} THEN 1 END) AS z FROM {}",
                        columns.join(", "),
                        source(name)
                    )
                );
            }
        }
        "time_bounds" => assert_eq!(
            flat,
            format!(
                "SELECT 0 as table_index, MIN(\"ts\") as min_date, MAX(\"ts\") as max_date FROM {}",
                source("orders")
            )
        ),
        "column_ranges" => assert_eq!(
            flat,
            format!(
                "SELECT COUNT(*), MIN(\"ts\"), MAX(\"ts\"), COUNT(\"ts\") FROM {}",
                source("orders")
            )
        ),
        "sample_entities" => {
            let n = request["n"].as_u64().unwrap().min(1_000_000);
            let filter = if request["filters"].as_array().unwrap().is_empty() {
                ""
            } else {
                " WHERE \"user_id\" IS NOT NULL"
            };
            assert_eq!(
                flat,
                format!(
                    "SELECT \"user_id\" FROM {} SAMPLE ROW ({n} ROWS){filter}",
                    source("users")
                )
            );
        }
        kind => panic!("unchecked request {kind}"),
    }
    if request["part"] != "view_definition" {
        assert!(params.is_empty());
    }
    statement
}

fn table(name: &str) -> Value {
    json!({"source": [name]})
}

#[test]
fn catalog_statements_preserve_namespace_and_request_kind() {
    let p = &named();
    let listing = planned(p, json!({"request":"describe","part":"tables","tables":[]})).unwrap();
    assert_eq!(listing.len(), 1);
    assert_eq!(
        listing[0]
            .0
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        format!("SELECT TABLE_NAME FROM \"{DB}\".INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?")
    );
    assert_eq!(listing[0].1, [SCHEMA]);
    for name in ["users", "orders", "USERS"] {
        for (part, prefix) in [
            ("columns", "DESCRIBE TABLE"),
            ("keys", "DESCRIBE TABLE"),
            ("foreign_keys", "SHOW IMPORTED KEYS IN TABLE"),
        ] {
            let got = planned(
                p,
                json!({"request":"describe","part":part,"tables":[table(name)]}),
            )
            .unwrap();
            assert_eq!(
                got,
                vec![(format!("{prefix} \"{DB}\".\"{SCHEMA}\".\"{name}\""), vec![])]
            );
        }
    }
}

#[test]
fn session_namespace_is_read_once() {
    // Each part left to the session, or the schema only, or the database
    // only: one statement, the old client's, whatever is named.
    for namespace in [
        vec![Some(DB), None],
        vec![None, None],
        vec![None, Some(SCHEMA)],
    ] {
        let (text, params) = one_statement(
            "snowflake.name.database-only",
            &snowflake(&namespace),
            json!({"request": "describe", "part": "namespace", "tables": []}),
        );
        assert_eq!(text, "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()");
        assert!(params.is_empty());
    }
}

#[test]
fn previews_key_counts_and_time_ranges_preserve_sql_semantics() {
    let p = &named();
    one_statement(
        "base.binary",
        p,
        json!({"request": "preview", "table": table("users"),
               "columns": ["user_id", "age", "country"], "limit": 1000}),
    );
    one_statement(
        "base.binary",
        p,
        json!({"request": "preview", "table": table("orders"),
               "columns": ["order_id", "user_id", "price", "ts"], "limit": 1000}),
    );
    one_statement(
        "snowflake.name.upper-case-folded",
        p,
        json!({"request": "preview", "table": {"source": [DB, SCHEMA, "USERS"]},
               "columns": ["user_id", "age", "country"], "limit": 1000}),
    );
    one_statement(
        "base.binary",
        p,
        json!({"request": "key_counts", "keys": [
            {"table": table("users"), "columns": ["user_id"]},
            {"table": table("orders"), "columns": ["order_id"]}]}),
    );
    // A key of two columns: `COUNT(DISTINCT a, b)`, Snowflake's, which
    // skips a row missing either part itself.
    one_statement(
        "rule.composite-key-given",
        p,
        json!({"request": "key_counts", "keys": [
            {"table": table("orders"), "columns": ["order_id", "line"]},
            {"table": table("users"), "columns": ["user_id"]}]}),
    );
    one_statement(
        "base.binary",
        p,
        json!({"request": "time_bounds", "columns": [{"table": table("orders"), "column": "ts"}]}),
    );
    one_statement(
        "snowflake.sampled.entity-table",
        p,
        json!({"request": "column_ranges", "table": table("orders"), "columns": ["ts"]}),
    );
}

#[test]
fn entity_sampling_preserves_row_cap_and_ignores_seed() {
    let p = &named();
    let sample = |n: u64, seed: Value, filters: Value| {
        json!({"request": "sample_entities", "table": table("users"), "key": ["user_id"],
               "n": n, "seed": seed, "columns": ["user_id"], "filters": filters})
    };
    let not_null = json!([{"filter": "not_null", "column": "user_id"}]);
    let seed = |v: &str| json!({"t": "u64", "v": v});
    let (text, _) = one_statement(
        "base.binary",
        p,
        sample(10_000, seed("42"), not_null.clone()),
    );
    // `SAMPLE ROW` takes no seed: every seed draws with the same text.
    assert_eq!(
        planned(p, sample(10_000, seed("7"), not_null.clone())).unwrap(),
        vec![(text, vec![])]
    );
    one_statement(
        "sample.max-pq-iterations-1",
        p,
        sample(1_000, seed("42"), not_null.clone()),
    );
    // A key the source declares not null takes no filter.
    one_statement(
        "key.past-2-53.catalog",
        p,
        sample(10_000, seed("42"), json!([])),
    );
    // Snowflake's limit on `SAMPLE ROW`, as the old client caps it.
    let capped = planned(p, sample(2_000_000, seed("42"), not_null)).unwrap();
    assert_eq!(
        capped[0].0,
        format!(
            "SELECT \"user_id\"\nFROM \"{DB}\".\"{SCHEMA}\".\"users\"\nSAMPLE ROW (1000000 ROWS)\nWHERE \"user_id\" IS NOT NULL"
        )
    );
}

#[test]
fn row_counts_name_the_exact_object_and_support_views() {
    for name in ["users", "user_s", "users%", "USERS", "users_v"] {
        let got = planned(
            &named(),
            json!({"request": "count_rows", "table": table(name), "filters": []}),
        )
        .unwrap();
        assert_eq!(
            got,
            vec![(
                format!("SELECT COUNT(*) AS \"rows\" FROM \"{DB}\".\"{SCHEMA}\".\"{name}\""),
                vec![]
            )]
        );
    }
}

#[test]
fn a_semantic_views_definition_is_read_by_its_name_as_the_old_client_binds_it() {
    let p = &named();
    let (text, params) = one_statement(
        "view.semantic.forecast",
        p,
        json!({"request": "describe", "part": "view_definition",
               "tables": [table("view_semantic_forecast")]}),
    );
    assert_eq!(text, "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)");
    assert_eq!(
        params,
        vec![format!("{DB}.{SCHEMA}.view_semantic_forecast")]
    );
    // A name with a dot is taken as qualified already, and bound whole
    // (`CL/materializers/sources.py:318-320`).
    let dotted = planned(
        p,
        json!({"request": "describe", "part": "view_definition",
               "tables": [table("other_db.other_schema.v")]}),
    )
    .unwrap();
    assert_eq!(
        dotted,
        vec![(
            "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)".to_owned(),
            vec!["other_db.other_schema.v".to_owned()]
        )]
    );
}

#[test]
fn each_name_part_is_quoted_alone_in_double_quotes() {
    let p = &named();
    let statements = planned(
        p,
        json!({"request": "preview", "table": table("a.b"),
               "columns": ["say \"hi\""], "limit": 1000}),
    )
    .unwrap();
    assert_eq!(
        statements[0].0,
        format!(r#"SELECT "say ""hi""" FROM "{DB}"."{SCHEMA}"."a.b" LIMIT 1000"#)
    );
}

#[test]
fn what_the_dialect_cannot_read_is_refused_by_name() {
    // A relation past a database, a schema and a table.
    let refused = planned(
        &named(),
        json!({"request": "preview", "table": {"source": ["a", "b", "c", "d"]},
               "columns": ["x"], "limit": 1}),
    )
    .unwrap_err();
    assert_eq!(refused.sentence, Sentence::RelationTooManyParts);
    // A session with no current schema: the source's tables cannot be
    // listed, as the old client refuses (`RE/graph.py:1035-1040`)...
    let schemaless = snowflake(&[Some(DB)]);
    let refused = planned(
        &schemaless,
        json!({"request": "describe", "part": "tables", "tables": []}),
    )
    .unwrap_err();
    assert_eq!(refused.sentence, Sentence::NoCurrentSchema);
    assert_eq!(
        refused.fields.get("database"),
        Some(&Detail::text("snowflake"))
    );
    //... and a table named has no schema to be read in
    // (`SNOW/table.py:88-93`).
    let refused = planned(
        &schemaless,
        json!({"request": "describe", "part": "columns", "tables": [table("users")]}),
    )
    .unwrap_err();
    assert_eq!(refused.sentence, Sentence::UnspecifiedSchema);
    assert_eq!(refused.fields.get("table"), Some(&Detail::text("users")));
    assert_eq!(refused.fields.get("in_database"), Some(&Detail::text(DB)));
    // Snowflake keeps no index the old client reads.
    let refused = planned(
        &named(),
        json!({"request": "describe", "part": "indexes", "tables": []}),
    )
    .unwrap_err();
    assert_eq!(refused.sentence, Sentence::RenderingRefused);
}

// --- the catalog reader -------------------------------------------------------

/// A Python `str` of a recorded cell, as the row adapter writes it.
fn read(request: Value, batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    let request: DataRequest = serde_json::from_value(request).unwrap();
    layer().read(&request, &named(), batches)
}

/// Read authored DESCRIBE rows as both columns and key declarations.
fn described(rows: &[Described<'_>], name: &str) -> (CatalogAnswer, CatalogAnswer) {
    let rows = describe_rows(rows);
    let ask = |part: &str| json!({"request": "describe", "part": part, "tables": [table(name)]});
    (
        read(ask("columns"), &rows).unwrap(),
        read(ask("keys"), &rows).unwrap(),
    )
}

#[test]
fn a_tables_columns_are_read_from_its_describe_as_the_old_client_reads_them() {
    let (columns, keys) = described(
        &[
            ("user_id", "NUMBER(38,0)", "Y", "N", "N"),
            ("age", "NUMBER(38,0)", "Y", "N", "N"),
            ("country", "VARCHAR(16777216)", "Y", "N", "N"),
        ],
        "users",
    );
    let CatalogAnswer::Columns(columns) = columns else {
        panic!("not columns");
    };
    let read: Vec<(&str, &str, bool, Option<Dtype>)> = columns
        .iter()
        .map(|c| {
            (
                c.name.as_str(),
                c.declared_type.as_str(),
                c.not_null,
                c.dtype,
            )
        })
        .collect();
    assert_eq!(
        read,
        vec![
            ("user_id", "NUMBER(38,0)", false, Some(Dtype::Int)),
            ("age", "NUMBER(38,0)", false, Some(Dtype::Int)),
            ("country", "VARCHAR(16777216)", false, Some(Dtype::String)),
        ]
    );
    assert_eq!(keys, CatalogAnswer::Keys(vec![]));
}

#[test]
fn a_declared_primary_key_is_read_from_the_describe_and_not_enforced() {
    let (columns, keys) = described(
        &[
            ("user_id", "NUMBER(38,0)", "N", "Y", "N"),
            ("age", "NUMBER(38,0)", "Y", "N", "N"),
        ],
        "users",
    );
    let CatalogAnswer::Columns(columns) = columns else {
        panic!("not columns");
    };
    // `null?` is `N` for the key.
    assert!(columns[0].not_null && !columns[1].not_null);
    let CatalogAnswer::Keys(keys) = keys else {
        panic!("not keys");
    };
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].kind, KeyKind::Primary);
    assert_eq!(keys[0].columns, vec!["user_id".to_owned()]);
    // Only a hybrid table enforces its key: the key is counted.
    assert!(!keys[0].enforced);
    // Two columns flagged: a key of both, which makes no table's key
    // (`RE/base/table.py:1016-1024`).
    let (_, keys) = described(
        &[
            ("order_no", "NUMBER(38,0)", "N", "Y", "N"),
            ("line", "NUMBER(38,0)", "N", "Y", "N"),
        ],
        "orders",
    );
    let CatalogAnswer::Keys(keys) = keys else {
        panic!("not keys");
    };
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].kind, KeyKind::Primary);
    assert_eq!(
        keys[0].columns,
        vec!["order_no".to_owned(), "line".to_owned()]
    );
}

#[test]
fn each_declared_type_is_read_by_the_old_clients_type_map() {
    for (declared, dtype) in [
        ("NUMBER(10,2)", Some(Dtype::Float)),
        ("NUMBER(38,18)", Some(Dtype::Float)),
        ("NUMBER(38,0)", Some(Dtype::Int)),
        ("NUMBER", Some(Dtype::Float)),
        ("NUMBER(38,x)", Some(Dtype::Float)),
        ("FLOAT", Some(Dtype::Float)),
        ("VARCHAR(10)", Some(Dtype::String)),
        ("BINARY(8388608)", Some(Dtype::Binary)),
        ("BOOLEAN", Some(Dtype::Bool)),
        ("DATE", Some(Dtype::Date)),
        ("TIMESTAMP_NTZ(9)", Some(Dtype::Date)),
        ("TIMESTAMP_TZ(9)", Some(Dtype::Date)),
        ("TIME(9)", Some(Dtype::Time)),
        ("VECTOR(INT, 3)", Some(Dtype::Intlist)),
        ("VECTOR(FLOAT, 256)", Some(Dtype::Floatlist)),
        ("ARRAY(NUMBER(38,0))", Some(Dtype::Intlist)),
        ("ARRAY(FLOAT)", Some(Dtype::Floatlist)),
        ("ARRAY(VARCHAR(16777216))", Some(Dtype::Stringlist)),
        ("ARRAY(BOOLEAN)", Some(Dtype::Unsupported)),
        ("ARRAY", Some(Dtype::Unsupported)),
        ("DECFLOAT", Some(Dtype::Unsupported)),
        ("VARIANT", Some(Dtype::Unsupported)),
        ("OBJECT", Some(Dtype::Unsupported)),
        ("MAP(VARCHAR, NUMBER)", Some(Dtype::Unsupported)),
        ("FILE", Some(Dtype::Unsupported)),
        ("GEOGRAPHY", Some(Dtype::Unsupported)),
        ("GEOMETRY", Some(Dtype::Unsupported)),
        ("UUID", None),
    ] {
        let rows = describe_rows(&[("c", declared, "Y", "N", "N")]);
        let CatalogAnswer::Columns(columns) = read(
            json!({"request": "describe", "part": "columns", "tables": [table("t")]}),
            &rows,
        )
        .unwrap() else {
            panic!("not columns");
        };
        assert_eq!(columns[0].dtype, dtype, "{declared}");
    }
}

/// One `DESCRIBE TABLE` row: name, type, `null?`, primary key, unique key.
type Described<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str);

/// A `DESCRIBE TABLE` answer in the row adapter's text form: name, type,
/// kind, `null?`, default, primary key, unique key.
fn describe_rows(rows: &[Described<'_>]) -> Vec<RecordBatch> {
    let names = [
        "name",
        "type",
        "kind",
        "null?",
        "default",
        "primary key",
        "unique key",
    ];
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|n| Field::new(*n, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    ));
    let column = |pick: &dyn Fn(&Described<'_>) -> Option<String>| -> ArrayRef {
        Arc::new(rows.iter().map(pick).collect::<StringArray>())
    };
    vec![
        RecordBatch::try_new(
            schema,
            vec![
                column(&|r| Some(r.0.to_owned())),
                column(&|r| Some(r.1.to_owned())),
                column(&|_| Some("COLUMN".to_owned())),
                column(&|r| Some(r.2.to_owned())),
                column(&|_| None),
                column(&|r| Some(r.3.to_owned())),
                column(&|r| Some(r.4.to_owned())),
            ],
        )
        .unwrap(),
    ]
}

#[test]
fn a_unique_key_is_one_column_flagged_unique() {
    let ask = json!({"request": "describe", "part": "keys", "tables": [table("t")]});
    let CatalogAnswer::Keys(keys) = read(
        ask.clone(),
        &describe_rows(&[
            ("a", "NUMBER(38,0)", "N", "N", "Y"),
            ("b", "FLOAT", "Y", "N", "N"),
        ]),
    )
    .unwrap() else {
        panic!("not keys");
    };
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].kind, KeyKind::Unique);
    assert_eq!(keys[0].columns, vec!["a".to_owned()]);
    assert!(!keys[0].enforced);
    // The flags are read after strip and upper-case, as the old client reads
    // them.
    let CatalogAnswer::Keys(keys) = read(
        ask,
        &describe_rows(&[("a", "NUMBER(38,0)", "n", " y ", "n")]),
    )
    .unwrap() else {
        panic!("not keys");
    };
    assert_eq!(keys[0].kind, KeyKind::Primary);
}

/// Minimal row-adapter representation of SHOW IMPORTED KEYS.
fn imported_rows(keys: &[(&str, &str, &str)]) -> Vec<RecordBatch> {
    let names = [
        "created_on",
        "pk_database_name",
        "pk_schema_name",
        "pk_table_name",
        "pk_column_name",
        "fk_database_name",
        "fk_schema_name",
        "fk_table_name",
        "fk_column_name",
        "key_sequence",
        "update_rule",
        "delete_rule",
        "fk_name",
        "pk_name",
        "deferrability",
        "rely",
        "comment",
    ];
    let columns = names.iter().map(|name| {
        let values = keys
            .iter()
            .map(|(constraint, from, to)| match *name {
                "pk_database_name" | "fk_database_name" => Some(DB),
                "pk_schema_name" | "fk_schema_name" => Some(SCHEMA),
                "pk_table_name" => Some("users"),
                "pk_column_name" => Some(*to),
                "fk_table_name" => Some("orders"),
                "fk_column_name" => Some(*from),
                "fk_name" => Some(*constraint),
                "pk_name" => Some("users_pk"),
                _ => None,
            })
            .collect::<StringArray>();
        (*name, Arc::new(values) as ArrayRef)
    });
    vec![RecordBatch::try_from_iter(columns).unwrap()]
}

fn foreign_keys(keys: &[(&str, &str, &str)]) -> Vec<CatalogForeignKey> {
    let CatalogAnswer::ForeignKeys(keys) = read(
        json!({"request":"describe","part":"foreign_keys","tables":[table("orders")]}),
        &imported_rows(keys),
    )
    .unwrap() else {
        panic!("not foreign keys")
    };
    keys
}

#[test]
fn a_foreign_key_is_read_from_the_imported_keys_by_its_referenced_keys_name() {
    let keys = foreign_keys(&[("buyer_fk", "buyer", "uid")]);
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].columns, vec!["buyer".to_owned()]);
    assert_eq!(keys[0].referenced_columns, vec!["uid".to_owned()]);
    // The target table joined with dots, unquoted.
    assert_eq!(keys[0].referenced_table, format!("{DB}.{SCHEMA}.users"));
    assert!(foreign_keys(&[]).is_empty());
    // A key of two columns: its two rows share the referenced key's name,
    // so they are one key, which links no table (`SNOW/table.py:174-189`).
    let keys = foreign_keys(&[
        ("buyer_region_fk", "user_id", "user_id"),
        ("buyer_region_fk", "region", "region"),
    ]);
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].key, keys[1].key);
}

#[test]
fn the_tables_and_the_session_are_read_from_their_answers() {
    let listing = vec![
        RecordBatch::try_from_iter([(
            "TABLE_NAME",
            Arc::new(StringArray::from(vec!["users", "orders"])) as ArrayRef,
        )])
        .unwrap(),
    ];
    assert_eq!(
        read(
            json!({"request": "describe", "part": "tables", "tables": []}),
            &listing
        )
        .unwrap(),
        CatalogAnswer::Tables(vec!["users".to_owned(), "orders".to_owned()])
    );
    let session = vec![
        RecordBatch::try_from_iter([
            (
                "CURRENT_DATABASE()",
                Arc::new(StringArray::from(vec![DB])) as ArrayRef,
            ),
            (
                "CURRENT_SCHEMA()",
                Arc::new(StringArray::from(vec![SCHEMA])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    let ask = json!({"request": "describe", "part": "namespace", "tables": []});
    assert_eq!(
        read(ask.clone(), &session).unwrap(),
        CatalogAnswer::Namespace(vec![DB.to_owned(), SCHEMA.to_owned()])
    );
    // No current schema: the database alone, which no table is read under.
    let current = |database: Option<&str>, schema: Option<&str>| {
        let schema_ = Arc::new(Schema::new(vec![
            Field::new("CURRENT_DATABASE()", DataType::Utf8, true),
            Field::new("CURRENT_SCHEMA()", DataType::Utf8, true),
        ]));
        vec![
            RecordBatch::try_new(
                schema_,
                vec![
                    Arc::new(StringArray::from(vec![database])) as ArrayRef,
                    Arc::new(StringArray::from(vec![schema])) as ArrayRef,
                ],
            )
            .unwrap(),
        ]
    };
    assert_eq!(
        read(ask.clone(), &current(Some("D"), None)).unwrap(),
        CatalogAnswer::Namespace(vec!["D".to_owned()])
    );
    // No current database: no part, so a database the source names is kept
    // and the schema is refused as missing (`RE/graph.py:1030-1040`), and
    // one it does not name is an internal error, as the old client's
    // assert is (P3D-4).
    assert_eq!(
        read(ask.clone(), &current(None, Some("S"))).unwrap(),
        CatalogAnswer::Namespace(vec![])
    );
    assert_eq!(
        read(ask, &current(None, None)).unwrap(),
        CatalogAnswer::Namespace(vec![])
    );
}

#[test]
fn a_session_with_no_current_database_keeps_the_database_the_source_names() {
    let layer = layer();
    let settled = layer.settled(&snowflake(&[Some(DB), None]), vec![]);
    assert_eq!(settled, snowflake(&[Some(DB)]));
    let refused = planned(
        &settled,
        json!({"request": "describe", "part": "tables", "tables": []}),
    )
    .unwrap_err();
    assert_eq!(refused.sentence, Sentence::NoCurrentSchema);
    // A source naming no database on such a session: the old client's bare
    // assert, an internal error here (P3D-4).
    let nowhere = layer.settled(&snowflake(&[None, None]), vec![]);
    for request in [
        json!({"request": "describe", "part": "tables", "tables": []}),
        json!({"request": "describe", "part": "columns", "tables": [table("users")]}),
    ] {
        let refused = planned(&nowhere, request).unwrap_err();
        assert_eq!(refused.sentence, Sentence::NoCurrentDatabase);
    }
}

#[test]
fn a_semantic_views_definition_is_its_yaml() {
    let yaml = "name: v\ntables: []\n";
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)",
            DataType::Utf8,
            true,
        )])),
        vec![Arc::new(StringArray::from(vec![Some(yaml)])) as ArrayRef],
    )
    .unwrap();
    let CatalogAnswer::ViewDefinition(definition) = read(
        json!({"request": "describe", "part": "view_definition", "tables": [table("v")]}),
        &[batch],
    )
    .unwrap() else {
        panic!("not a view's definition");
    };
    assert_eq!(definition.kind, sdm_contract::view::ViewKind::Semantic);
    assert_eq!(definition.text, yaml);
    assert!(definition.located.is_empty() && definition.parameters.is_empty());
}

// --- conform ------------------------------------------------------------------

/// Original connector bytes retained only for distinct physical Arrow representations.
fn connector_sample(kind: &str) -> Vec<RecordBatch> {
    let file = match kind {
        "narrow-integers" => {
            "a4d1c73e1c1c48bdaeb829617953b6d38bcf89f3e23ed330e1b8e3abcf42fbc2.arrow"
        }
        "fractional-decimal" => {
            "49087a631e81db9c4c1a573e575141cfe09ba86de71c56de9a648acd5aa50fdb.arrow"
        }
        "wide-decimal" => "b3673dc382515d8461ed952ef45ea16395215d79203c28f6ec27e2a303a00ec8.arrow",
        "zoned-timestamp" => {
            "d7e37050ac434076070cf7566f04218d29ca546ee3c163fd967bf90675fe37c7.arrow"
        }
        _ => panic!("unknown connector sample {kind}"),
    };
    arrow("", file)
}

fn conformed(batches: &[RecordBatch]) -> Vec<RecordBatch> {
    let schema = batches[0].schema();
    match layer()
        .conform(&named(), &schema, batches, |f| f.to_string())
        .unwrap()
    {
        Some((_, batches)) => batches,
        None => batches.to_vec(),
    }
}

#[test]
fn the_connectors_arrow_is_read_as_the_old_client_reads_it() {
    // Integers the connector narrows by their values: `int64`.
    let read = conformed(&connector_sample("narrow-integers"));
    assert_eq!(read[0].column(0).data_type(), &DataType::Int64);
    assert_eq!(
        read[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .values()
            .as_ref(),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(read[0].column(2).data_type(), &DataType::Utf8);
    // A `NUMBER(38,18)`: a double.
    let read = conformed(&connector_sample("fractional-decimal"));
    assert_eq!(read[0].column(2).data_type(), &DataType::Float64);
    assert_eq!(
        read[0]
            .column(2)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(1),
        11.5
    );
    // A `NUMBER(38,0)` past `int64`: left a decimal.
    let read = conformed(&connector_sample("wide-decimal"));
    assert_eq!(read[0].column(3).data_type(), &DataType::Decimal128(38, 0));
    assert_eq!(
        read[0]
            .column(3)
            .as_primitive::<arrow_array::types::Decimal128Type>()
            .value(0),
        12_345_678_901_234_567_890_123_456_789i128
    );
    // A `TIMESTAMP_LTZ` in a Los Angeles session: its UTC instant, no zone.
    let given = connector_sample("zoned-timestamp");
    let read = conformed(&given);
    assert_eq!(
        read[0].column(3).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(
        read[0]
            .column(3)
            .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        given[0]
            .column(3)
            .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0)
    );
    // A `NUMBER(38,0)` that fits: `int64`, exactly.
    let fits = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "k",
            DataType::Decimal128(38, 0),
            true,
        )])),
        vec![Arc::new(
            arrow_array::Decimal128Array::from(vec![Some(9_007_199_254_740_995_i128), None])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        ) as ArrayRef],
    )
    .unwrap();
    let read = conformed(&[fits]);
    let keys = read[0].column(0).as_primitive::<Int64Type>();
    assert_eq!(keys.value(0), 9_007_199_254_740_995);
    assert!(keys.is_null(1));
}

// --- the data layer's answers ------------------------------------------------

fn failed(vendor_code: Option<&str>, sqlstate: Option<&str>) -> ProviderError {
    said(vendor_code, sqlstate, None)
}

fn said(vendor_code: Option<&str>, sqlstate: Option<&str>, message: Option<&str>) -> ProviderError {
    ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: sqlstate.map(str::to_owned),
        vendor_code: vendor_code.map(str::to_owned),
        error_class: None,
        driver_message: message.map(str::to_owned),
    }
}

#[test]
fn errno_2003_or_a_message_saying_does_not_exist_says_an_object_is_not_found() {
    let layer = layer();
    let p = &named();
    // The old client's `_is_missing_object_error` (`SNOW/table.py:34-45`):
    // errno 2003, or the driver's message saying "does not exist", in any
    // case.
    assert!(
        layer
            .failure(p, &failed(Some("2003"), Some("02000")))
            .not_found
    );
    assert!(layer.failure(p, &failed(Some("2003"), None)).not_found);
    assert!(
        layer
            .failure(
                p,
                &said(
                    Some("2043"),
                    Some("02000"),
                    Some("Object does not exist, or operation cannot be performed.")
                )
            )
            .not_found
    );
    assert!(
        layer
            .failure(p, &said(None, None, Some("Schema 'X' DOES NOT EXIST")))
            .not_found
    );
    // A login, a permission or a connection failure says nothing of the
    // table, so no other spelling is tried.
    for (code, message) in [
        ("390100", "Incorrect username or password was specified."),
        ("390114", "Authentication token has expired."),
        ("3001", "Insufficient privileges to operate on table."),
        ("250001", "Could not connect to Snowflake backend."),
        ("2043", "Operation cannot be performed."),
    ] {
        assert!(
            !layer
                .failure(p, &said(Some(code), None, Some(message)))
                .not_found,
            "{code}"
        );
    }
    assert!(!layer.failure(p, &failed(None, Some("02000"))).not_found);
}

#[test]
fn a_name_is_tried_as_given_then_upper_cased_once() {
    let layer = layer();
    let p = &named();
    assert_eq!(
        layer.spellings(p, &relation(&["db", "Sales", "users"])),
        vec![
            relation(&["db", "Sales", "users"]),
            relation(&["DB", "SALES", "USERS"]),
        ]
    );
    assert_eq!(
        layer.spellings(p, &relation(&["DB", "S", "USERS"])),
        vec![relation(&["DB", "S", "USERS"])]
    );
}

#[test]
fn the_session_fills_only_the_parts_the_source_leaves_out() {
    let layer = layer();
    let session = || vec!["SESSION_DB".to_owned(), "SESSION_SCHEMA".to_owned()];
    for (given, settled) in [
        (vec![Some(DB), None], vec![Some(DB), Some("SESSION_SCHEMA")]),
        (
            vec![None, Some(SCHEMA)],
            vec![Some("SESSION_DB"), Some(SCHEMA)],
        ),
        (
            vec![None, None],
            vec![Some("SESSION_DB"), Some("SESSION_SCHEMA")],
        ),
    ] {
        let source = snowflake(&given);
        assert!(layer.unsettled(&source));
        assert_eq!(layer.settled(&source, session()), snowflake(&settled));
    }
    // A session with no current schema: the database named, alone.
    assert_eq!(
        layer.settled(&snowflake(&[Some(DB), None]), vec!["SESSION_DB".to_owned()]),
        snowflake(&[Some(DB)])
    );
    assert!(!layer.unsettled(&named()));
}

#[test]
fn snowflake_declares_what_the_old_clients_sampler_does() {
    let layer = layer();
    let capabilities = layer.capabilities(&named()).unwrap();
    // `SAMPLE ROW (n ROWS)` takes no seed (owner option C).
    assert_eq!(capabilities.sampling, Determinism::Unseeded);
    assert_eq!(capabilities.null_order, Some(NullOrder::Last));
    assert!(capabilities.neighbor_lookback);
    assert!(!capabilities.sample_rows);
    assert!(!capabilities.keys_optional);
    // The schema's hybrid tables are asked before a key is counted.
    assert!(capabilities.key_enforcement);
    // Semantic-view tables use derived relations.
    assert!(capabilities.derived_relations);
    assert!(capabilities.fallback.is_none());
    assert_eq!(layer.source_database(&named()), Some("snowflake"));
    assert!(layer.reads_tables_as_they_stand(&named()));
    // The key counts in the case Snowflake folds an unquoted alias to.
    let FetchPlan::Sql { expect, .. } = layer
        .plan(
            &serde_json::from_value(json!({"request": "key_counts", "keys": [
                {"table": table("users"), "columns": ["user_id"]}]}))
            .unwrap(),
            &named(),
        )
        .unwrap()
    else {
        panic!("not SQL");
    };
    let ExpectedSchema::Schema { fields } = expect else {
        panic!("no schema expected");
    };
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["K", "N", "D", "Z"]);
}

// --- the key reads -------------------------------------------------------------

/// `text`, `%Y-%m-%d %H:%M:%S`, as nanoseconds since the epoch.
fn wall_clock(text: &str) -> i64 {
    let number = |range: std::ops::Range<usize>| text[range].parse::<i64>().unwrap();
    let (y, m, d) = (number(0..4), number(5..7), number(8..10));
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let seconds = days * 86_400 + number(11..13) * 3600 + number(14..16) * 60 + number(17..19);
    seconds * 1_000_000_000
}

/// Authored seed JSON: each id, and each of
/// the window's bounds where the rows carry them, in batch order.
fn seed_data(payload: &str) -> sdm_contract::data::KeyData {
    use arrow_array::{Int64Array, TimestampNanosecondArray};
    let rows: Vec<Value> = serde_json::from_str(payload).unwrap();
    let id = |row: &Value| match row {
        Value::Array(cells) => cells[0].as_i64().unwrap(),
        cell => cell.as_i64().unwrap(),
    };
    let time = |row: &Value, at: usize| wall_clock(row[at].as_str().unwrap());
    let mut fields = vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(rows.iter().map(id).collect::<Int64Array>()),
        Arc::new(Int64Array::from_iter_values(0..rows.len() as i64)),
    ];
    let width = rows[0].as_array().map_or(0, Vec::len);
    for (at, name) in [(1, "end"), (2, "start")] {
        if at < width {
            fields.push(Field::new(
                name,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ));
            columns.push(Arc::new(TimestampNanosecondArray::from_iter_values(
                rows.iter().map(|row| time(row, at)),
            )));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    sdm_contract::data::KeyData {
        schema,
        batches: vec![batch],
    }
}

/// `request` planned and bound on Snowflake with `keys` its seeds: each
/// fetch's statements, the text each binds and its other parameters.
fn bound(request: Value, keys: sdm_contract::data::KeyData) -> Vec<(String, String, Vec<String>)> {
    use sdm_contract::data::{AttachmentId, KeyData};
    let request: DataRequest = serde_json::from_value(request).unwrap();
    let plan = layer().plan(&request, &named()).unwrap();
    let handed: std::collections::BTreeMap<AttachmentId, KeyData> =
        [(AttachmentId(2), keys)].into();
    let bound = layer()
        .bind(&named(), &request, plan, &handed, |x| format!("{x:?}"))
        .unwrap();
    let mut found = Vec::new();
    for fetch in bound.fetches {
        let FetchPlan::Sql { statements, .. } = fetch.plan else {
            panic!("not SQL");
        };
        for statement in statements {
            let mut text = String::new();
            let mut others = Vec::new();
            for param in statement.params {
                match param {
                    SqlParam::Text { attachment } => {
                        let batch = &fetch.batches[&attachment].batches[0];
                        text = batch.column(0).as_string::<i32>().value(0).to_owned();
                    }
                    SqlParam::Value {
                        value: Scalar::Utf8(value),
                    } => others.push(value),
                    other => panic!("an unexpected parameter {other:?}"),
                }
            }
            found.push((statement.text, text, others));
        }
    }
    found
}

fn assert_bound_payload(sent: &[(String, String, Vec<String>)], payload: &str) -> String {
    assert_eq!(sent.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&sent[0].1).unwrap(),
        serde_json::from_str::<Value>(payload).unwrap()
    );
    let sql = sent[0].0.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(sql.contains("f.index as __KUMO_BATCH__"));
    assert!(sql.contains("FLATTEN(INPUT => PARSE_JSON(?))"));
    assert!(sql.contains("ON \"user_id\" = TMP.__KUMO_ID__"));
    sql
}

fn assert_window(sent: &[(String, String, Vec<String>)], payload: &str) -> String {
    let sql = assert_bound_payload(sent, payload);
    for clause in [
        "f.value[0]::NUMBER as __KUMO_ID__",
        "f.value[1]::TIMESTAMP_NTZ as __KUMO_END_TIME__",
        "f.value[2]::TIMESTAMP_NTZ as __KUMO_START_TIME__",
        "JOIN \"{catalog}\".\"{schema}\".\"orders\"",
        "\"ts\" <= TMP.__KUMO_END_TIME__",
        "\"ts\" > TMP.__KUMO_START_TIME__",
        "WHERE \"ts\" <= ? AND \"ts\" > ?",
    ] {
        assert!(sql.contains(clause), "missing {clause}: {sql}");
    }
    assert_eq!(sql.matches('?').count(), 3);
    assert_eq!(sent[0].2, ["2024-03-01 00:00:00", "2024-01-01 00:00:00"]);
    sql
}

const WINDOW_SEEDS: &str = r#"[[7,"2024-03-01 00:00:00","2024-02-01 00:00:00"],[7,"2024-02-01 00:00:00","2024-01-01 00:00:00"]]"#;

fn seeds_json(columns: &[(&str, Value)], flags: &[(&str, &str)]) -> Value {
    Value::Array(
        columns
            .iter()
            .map(|(name, column_type)| {
                let metadata: serde_json::Map<String, Value> = flags
                    .iter()
                    .filter(|(column, _)| column == name)
                    .map(|(_, flag)| ((*flag).to_owned(), json!("true")))
                    .collect();
                let mut column =
                    json!({"name": name, "nullable": *name != "batch", "type": column_type});
                if !metadata.is_empty() {
                    column["metadata"] = Value::Object(metadata);
                }
                column
            })
            .collect(),
    )
}

fn time_type() -> Value {
    json!({"type": "timestamp", "unit": "ns", "zone": null})
}

#[test]
fn window_reads_bind_each_seed_and_global_bounds() {
    let payload = WINDOW_SEEDS;
    let request = json!({
        "request": "neighbors", "table": table("orders"), "key": ["user_id"],
        "columns": ["ts"],
        "seeds": {"attachment": 2, "columns": seeds_json(&[
            ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"})),
            ("end", time_type()), ("start", time_type())], &[])},
        "window": {"column": "ts",
                   "lower": {"seed_column": "start", "inclusive": false},
                   "upper": {"seed_column": "end", "inclusive": true}},
        "per_seed": null, "order": null});
    let sent = bound(request, seed_data(payload));
    let sql = assert_window(&sent, payload);
    assert!(sql.contains("SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, \"ts\" FROM TMP"));
    assert!(!sql.contains("QUALIFY"));
    assert!(!sql.contains("ORDER BY"));
}

#[test]
fn neighbor_reads_preserve_order_and_per_seed_cap() {
    let payload = WINDOW_SEEDS;
    let term = |column: &str, time: bool| json!({"column": column, "as_timestamp": time, "descending": time, "nulls_last": true});
    let request = json!({
        "request": "neighbors", "table": table("orders"), "key": ["user_id"],
        "columns": ["order_id", "user_id", "price", "ts"],
        "seeds": {"attachment": 2, "columns": seeds_json(&[
            ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"})),
            ("end", time_type()), ("start", time_type())], &[])},
        "window": {"column": "ts",
                   "lower": {"seed_column": "start", "inclusive": false},
                   "upper": {"seed_column": "end", "inclusive": true}},
        "per_seed": 32, "order": [term("ts", true), term("order_id", false)]});
    let sent = bound(request, seed_data(payload));
    let sql = assert_window(&sent, payload);
    assert!(sql.contains("SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, \"order_id\", \"user_id\", \"price\", \"ts\" FROM TMP"));
    assert!(sql.contains("QUALIFY ROW_NUMBER() OVER ( PARTITION BY TMP.__KUMO_BATCH__ ORDER BY \"ts\" DESC, \"order_id\" ) <= 32"));
    assert!(sql.ends_with("ORDER BY TMP.__KUMO_BATCH__, \"ts\" DESC, \"order_id\""));
}

#[test]
fn lookup_preserves_seed_identity_and_duplicate_policy() {
    let payload = "[7,7,3]";
    let lookup = |flags: &[(&str, &str)]| {
        json!({
            "request": "lookup", "table": table("users"), "key": ["user_id"],
            "columns": ["user_id", "age", "country"],
            "seeds": {"attachment": 2, "columns": seeds_json(&[
                ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"}))], flags)}})
    };
    for (flags, qualify, count) in [
        (vec![("key", "sdm:key_unique")], false, false),
        (vec![], true, true),
        (vec![("batch", "sdm:count_unread")], true, false),
    ] {
        let sent = bound(lookup(&flags), seed_data(payload));
        let sql = assert_bound_payload(&sent, payload);
        assert!(sent[0].2.is_empty());
        assert_eq!(sql.matches('?').count(), 1);
        assert!(sql.contains("f.value::NUMBER as __KUMO_ID__"));
        assert!(sql.contains("JOIN \"{catalog}\".\"{schema}\".\"users\""));
        assert!(sql.contains(
            "SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, \"user_id\", \"age\", \"country\""
        ));
        assert_eq!(sql.contains("QUALIFY ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__ ORDER BY \"user_id\") = 1"), qualify);
        assert_eq!(
            sql.contains("COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) AS \"__kumo_matches__\""),
            count
        );
    }
}

#[test]
fn a_float_or_text_key_is_cast_and_written_as_the_old_client_writes_it() {
    use arrow_array::{Float64Array, Int64Array};
    let lookup = |key_type: Value| {
        json!({
            "request": "lookup", "table": table("users"), "key": ["user_id"],
            "columns": ["user_id"],
            "seeds": {"attachment": 2, "columns": seeds_json(&[
                ("key", key_type), ("batch", json!({"type": "int64"}))],
                &[("key", "sdm:key_unique")])}})
    };
    let keys = |column: ArrayRef| {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", column.data_type().clone(), true),
            Field::new("batch", DataType::Int64, false),
        ]));
        let rows = column.len() as i64;
        sdm_contract::data::KeyData {
            batches: vec![
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![column, Arc::new(Int64Array::from_iter_values(0..rows))],
                )
                .unwrap(),
            ],
            schema,
        }
    };
    let found = bound(
        lookup(json!({"type": "float64"})),
        keys(Arc::new(Float64Array::from(vec![1.5, 2.0]))),
    );
    assert!(found[0].0.contains("f.value::FLOAT as __KUMO_ID__"));
    assert_eq!(found[0].1, "[1.5, 2.0]");
    let found = bound(
        lookup(json!({"type": "utf8"})),
        keys(Arc::new(StringArray::from(vec!["a\"b", "é"]))),
    );
    assert!(found[0].0.contains("f.value::VARCHAR as __KUMO_ID__"));
    assert_eq!(found[0].1, "[\"a\\\"b\", \"\\u00e9\"]");
}

#[test]
fn a_missing_anchor_is_json_null_and_other_seeds_keep_their_window() {
    use arrow_array::{Int64Array, TimestampNanosecondArray};
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
        Field::new("end", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]));
    let keys = sdm_contract::data::KeyData {
        batches: vec![
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![0, 1])),
                    Arc::new(Int64Array::from(vec![0, 1])),
                    Arc::new(TimestampNanosecondArray::from(vec![
                        None,
                        Some(wall_clock("2024-03-01 00:00:00")),
                    ])),
                ],
            )
            .unwrap(),
        ],
        schema,
    };
    let request = json!({
        "request": "neighbors", "table": table("orders"), "key": ["user_id"],
        "columns": ["ts"],
        "seeds": {"attachment": 2, "columns": seeds_json(&[
            ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"})),
            ("end", time_type())], &[])},
        "window": {"column": "ts", "lower": null,
                   "upper": {"seed_column": "end", "inclusive": true}},
        "per_seed": null, "order": null});
    let found = bound(request, keys);
    assert_eq!(found[0].1, r#"[[0, null], [1, "2024-03-01 00:00:00"]]"#);
    assert!(found[0].0.ends_with("WHERE \"ts\" <= ?"));
    assert_eq!(found[0].2, vec!["2024-03-01 00:00:00".to_owned()]);
}

// --- key enforcement and the row count ----------------------------------------

#[test]
fn the_schemas_hybrid_tables_are_listed_and_read_as_the_old_client_reads_them() {
    let p = &named();
    // Asked of a table, answered for its schema (`SNOW/table.py:264-285`).
    let (text, params) = one_statement(
        "key.declared-int32",
        p,
        json!({"request": "describe", "part": "key_enforcement", "tables": [table("users")]}),
    );
    assert_eq!(text, format!(r#"SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}""#));
    assert!(params.is_empty());
    let ask = json!({"request": "describe", "part": "key_enforcement", "tables": [table("users")]});
    // Ordinary tables do not provide enforcement evidence.
    let listing = vec![
        RecordBatch::try_from_iter([
            (
                "name",
                Arc::new(StringArray::from(vec!["users", "orders"])) as ArrayRef,
            ),
            (
                "is_hybrid",
                Arc::new(StringArray::from(vec!["N", "N"])) as ArrayRef,
            ),
        ])
        .unwrap(),
    ];
    assert_eq!(
        read(ask.clone(), &listing).unwrap(),
        CatalogAnswer::KeyEnforcement(vec![])
    );
    // A hybrid table by its name, `is_hybrid` `Y` after strip and
    // upper-case, read by column name.
    let rows = |names: Vec<&str>, hybrid: Vec<&str>| {
        let schema = Arc::new(Schema::new(vec![
            Field::new("created_on", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("is_hybrid", DataType::Utf8, true),
        ]));
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec![None::<&str>; names.len()])) as ArrayRef,
                    Arc::new(StringArray::from(names)) as ArrayRef,
                    Arc::new(StringArray::from(hybrid)) as ArrayRef,
                ],
            )
            .unwrap(),
        ]
    };
    assert_eq!(
        read(
            ask.clone(),
            &rows(vec!["users", "orders"], vec![" y ", "N"])
        )
        .unwrap(),
        CatalogAnswer::KeyEnforcement(vec!["users".to_owned()])
    );
    // A listing without those columns says no table enforces its key.
    let other = vec![
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["users"])) as ArrayRef],
        )
        .unwrap(),
    ];
    assert_eq!(
        read(ask, &other).unwrap(),
        CatalogAnswer::KeyEnforcement(vec![])
    );
}

#[test]
fn independent_foreign_keys_to_one_primary_key_stay_independent() {
    let rows = imported_rows(&[("buyer_fk", "buyer", "uid"), ("seller_fk", "seller", "uid")]);
    let CatalogAnswer::ForeignKeys(keys) = read(
        json!({"request": "describe", "part": "foreign_keys", "tables": [table("orders")]}),
        &rows,
    )
    .unwrap() else {
        panic!("keys")
    };
    assert_eq!(keys.len(), 2);
    assert_ne!(keys[0].key, keys[1].key);
}

#[test]
fn zoned_windows_keep_utc_bounds_at_non_utc_dst_edges() {
    for end in [
        "2026-03-08 09:59:59",
        "2026-03-08 10:00:00",
        "2026-11-01 08:59:59",
        "2026-11-01 09:00:00",
    ] {
        let payload = format!("[[1, \"{end}\"]]");
        let request = json!({
            "request": "neighbors", "table": table("orders"), "key": ["user_id"], "columns": ["ts"],
            "seeds": {"attachment": 2, "columns": seeds_json(&[
                ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"})), ("end", time_type())], &[])},
            "window": {"column": "ts", "source_type": "TIMESTAMP_LTZ(6)", "lower": null,
                "upper": {"seed_column": "end", "inclusive": true}}, "per_seed": null, "order": null});
        let sent = bound(request, seed_data(&payload));
        assert_eq!(sent[0].1, payload);
        assert_eq!(sent[0].2, vec![end.to_owned()]);
        assert!(
            sent[0]
                .0
                .contains("CONVERT_TIMEZONE('UTC', \"ts\")::TIMESTAMP_NTZ")
        );
    }
}

#[test]
fn an_unreadable_semantic_definition_is_refused_by_name() {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![None::<&str>])),
        Arc::new(StringArray::from(vec![""])),
        Arc::new(arrow_array::Int64Array::from(vec![1])),
    ];
    for values in columns {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "yaml",
                values.data_type().clone(),
                true,
            )])),
            vec![values],
        )
        .unwrap();
        let error = read(
            json!({"request": "describe", "part": "view_definition", "tables": [table("v")]}),
            &[batch],
        )
        .unwrap_err();
        assert_eq!(error.sentence, Sentence::ViewSemanticDefinitionUnread);
        assert_eq!(error.fields["view"], Detail::text("v"));
    }
}
