// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snowflake on the core (T3.6), held to what the old client sent and read
//! on Snowflake: T3.20's recordings (`contract/captures/snowflake/` and
//! `contract/captures/s2/snowflake/`, the old client at 43e55f4f, connector
//! 4.8.0, database and schema recorded as `{catalog}` and `{schema}`).
//!
//! - Goldens: each request kind the `snowflake` dialect renders without a
//!   key set is planned under the source's database and schema, and each
//!   statement is the one the old client sent for the same step, text and
//!   parameters exactly (the catalog reads, the preview, the key counts, the
//!   time bounds, the entity sample, the column ranges, the row count's
//!   listing, a semantic view's YAML read). The row count's second
//!   statement, which reads the listing's `rows` column, is the dialect's
//!   own and pinned here.
//! - The catalog reader: each recorded `DESCRIBE TABLE`, `SHOW IMPORTED
//!   KEYS`, table listing and session read, rebuilt in the Python row
//!   adapter's text form (every column nullable text, in the result's
//!   order, each non-null cell its `str`), read into the contract's catalog
//!   answer as the old client reads the rows (`SNOW/table.py:160-189`,
//!   `RE/graph.py:1024-1051`), the type map included (`:330-383`).
//! - `conform` over the connector's own Arrow as recorded: narrow integers
//!   and a `NUMBER(38,0)` that fits read as `int64`, one past it left a
//!   decimal, a `NUMBER(p,s)` with a fraction read as a double, a zoned time
//!   at its UTC instant with no zone (`RE/base/table.py:1093-1186`).
//! - The data layer's answers for Snowflake: a failure with errno 2003 is
//!   "not found", nothing else is; a name's spellings are as given, then
//!   all three parts upper-cased, once; the session's parts fill only those
//!   the source leaves out; the capabilities.

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

/// The row count's second statement: the listing's `rows`, first row.
const ROWS_OF_THE_LISTING: &str =
    r#"SELECT "rows" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) LIMIT 1"#;

fn captures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/captures")
}

/// The directory of `case`'s capture: C-1's, else the packages' own (S2).
fn capture_dir(case: &str) -> PathBuf {
    let c1 = captures().join("snowflake").join(case);
    if c1.join("capture.json").exists() {
        return c1;
    }
    let s2 = captures().join("s2").join("snowflake").join(case);
    assert!(s2.join("capture.json").exists(), "no capture of {case}");
    s2
}

fn capture(case: &str) -> Value {
    let text = std::fs::read_to_string(capture_dir(case).join("capture.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A parameter as the recording writes it, as text.
fn param_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Each statement `case` recorded: its text and parameters.
fn recorded(case: &str) -> Vec<(String, Vec<String>)> {
    capture(case)["statements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["text"].as_str().unwrap().to_owned(),
                s["parameters"]
                    .as_array()
                    .map(|p| p.iter().map(param_text).collect())
                    .unwrap_or_default(),
            )
        })
        .collect()
}

/// The recorded result of the first statement of `case` whose text is
/// `text` and whose parameters are `params`.
fn recorded_result(case: &str, text: &str, params: &[&str]) -> Vec<RecordBatch> {
    let capture = capture(case);
    let statement = capture["statements"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| {
            s["text"].as_str() == Some(text)
                && s["parameters"]
                    .as_array()
                    .map(|p| p.iter().map(param_text).collect::<Vec<_>>())
                    .unwrap_or_default()
                    == params
        })
        .unwrap_or_else(|| panic!("{case} recorded no {text:?} {params:?}"));
    arrow(case, statement["result"].as_str().unwrap())
}

/// A recorded table, as the Arrow IPC stream holds it.
fn arrow(case: &str, file: &str) -> Vec<RecordBatch> {
    let path = capture_dir(case).join(file);
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

/// `request`'s one statement, which `case` recorded exactly.
fn one_recorded(case: &str, provider: &ProviderChoice, request: Value) -> (String, Vec<String>) {
    let mut statements = planned(provider, request.clone()).unwrap();
    assert_eq!(statements.len(), 1, "{request}: {statements:?}");
    let statement = statements.remove(0);
    assert!(
        recorded(case).contains(&statement),
        "{case} recorded no statement {statement:?} for {request}"
    );
    statement
}

fn table(name: &str) -> Value {
    json!({"source": [name]})
}

#[test]
fn the_catalog_statements_are_the_old_clients() {
    let p = &named();
    one_recorded(
        "base.binary",
        p,
        json!({"request": "describe", "part": "tables", "tables": []}),
    );
    for name in ["users", "orders"] {
        let (text, _) = one_recorded(
            "base.binary",
            p,
            json!({"request": "describe", "part": "columns", "tables": [table(name)]}),
        );
        assert_eq!(
            text,
            format!(r#"DESCRIBE TABLE "{DB}"."{SCHEMA}"."{name}""#)
        );
        // A table's keys are read from the same DESCRIBE as its columns.
        one_recorded(
            "base.binary",
            p,
            json!({"request": "describe", "part": "keys", "tables": [table(name)]}),
        );
        one_recorded(
            "base.binary",
            p,
            json!({"request": "describe", "part": "foreign_keys", "tables": [table(name)]}),
        );
    }
    // The upper-cased spelling, as the old client wrote it after a miss.
    one_recorded(
        "snowflake.name.upper-case-folded",
        p,
        json!({"request": "describe", "part": "columns",
               "tables": [{"source": [DB, SCHEMA, "USERS"]}]}),
    );
}

#[test]
fn the_session_is_read_with_the_old_clients_one_statement() {
    // Each part left to the session, or the schema only, or the database
    // only: one statement, the old client's, whatever is named.
    for namespace in [
        vec![Some(DB), None],
        vec![None, None],
        vec![None, Some(SCHEMA)],
    ] {
        let (text, params) = one_recorded(
            "snowflake.name.database-only",
            &snowflake(&namespace),
            json!({"request": "describe", "part": "namespace", "tables": []}),
        );
        assert_eq!(text, "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()");
        assert!(params.is_empty());
    }
}

#[test]
fn the_reads_are_the_old_clients() {
    let p = &named();
    one_recorded(
        "base.binary",
        p,
        json!({"request": "preview", "table": table("users"),
               "columns": ["user_id", "age", "country"], "limit": 1000}),
    );
    one_recorded(
        "base.binary",
        p,
        json!({"request": "preview", "table": table("orders"),
               "columns": ["order_id", "user_id", "price", "ts"], "limit": 1000}),
    );
    one_recorded(
        "snowflake.name.upper-case-folded",
        p,
        json!({"request": "preview", "table": {"source": [DB, SCHEMA, "USERS"]},
               "columns": ["user_id", "age", "country"], "limit": 1000}),
    );
    one_recorded(
        "base.binary",
        p,
        json!({"request": "key_counts", "keys": [
            {"table": table("users"), "columns": ["user_id"]},
            {"table": table("orders"), "columns": ["order_id"]}]}),
    );
    // A key of two columns: `COUNT(DISTINCT a, b)`, Snowflake's, which
    // skips a row missing either part itself.
    one_recorded(
        "rule.composite-key-given",
        p,
        json!({"request": "key_counts", "keys": [
            {"table": table("orders"), "columns": ["order_id", "line"]},
            {"table": table("users"), "columns": ["user_id"]}]}),
    );
    one_recorded(
        "base.binary",
        p,
        json!({"request": "time_bounds", "columns": [{"table": table("orders"), "column": "ts"}]}),
    );
    one_recorded(
        "snowflake.sampled.entity-table",
        p,
        json!({"request": "column_ranges", "table": table("orders"), "columns": ["ts"]}),
    );
}

#[test]
fn an_entity_sample_is_the_old_clients_unseeded_sample_of_rows() {
    let p = &named();
    let sample = |n: u64, seed: Value, filters: Value| {
        json!({"request": "sample_entities", "table": table("users"), "key": ["user_id"],
               "n": n, "seed": seed, "columns": ["user_id"], "filters": filters})
    };
    let not_null = json!([{"filter": "not_null", "column": "user_id"}]);
    let seed = |v: &str| json!({"t": "u64", "v": v});
    let (text, _) = one_recorded(
        "base.binary",
        p,
        sample(10_000, seed("42"), not_null.clone()),
    );
    // `SAMPLE ROW` takes no seed: every seed draws with the same text.
    assert_eq!(
        planned(p, sample(10_000, seed("7"), not_null.clone())).unwrap(),
        vec![(text, vec![])]
    );
    one_recorded(
        "sample.max-pq-iterations-1",
        p,
        sample(1_000, seed("42"), not_null.clone()),
    );
    // A key the source declares not null takes no filter.
    one_recorded(
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
fn a_row_count_is_the_old_clients_listing_then_its_rows() {
    let p = &named();
    for (case, name) in [
        ("base.binary", "users"),
        ("base.binary", "orders"),
        ("snowflake.name.upper-case-folded", "USERS"),
    ] {
        let source = if name == "USERS" {
            json!({"source": [DB, SCHEMA, name]})
        } else {
            table(name)
        };
        let statements = planned(
            p,
            json!({"request": "count_rows", "table": source, "filters": []}),
        )
        .unwrap();
        assert_eq!(statements.len(), 2, "{statements:?}");
        assert!(
            recorded(case).contains(&statements[0]),
            "{case} recorded no {:?}",
            statements[0]
        );
        assert_eq!(
            statements[0],
            (
                format!(r#"SHOW TABLES LIKE ? IN SCHEMA "{DB}"."{SCHEMA}""#),
                vec![name.to_owned()]
            )
        );
        assert_eq!(statements[1], (ROWS_OF_THE_LISTING.to_owned(), vec![]));
    }
}

#[test]
fn a_semantic_views_definition_is_read_by_its_name_as_the_old_client_binds_it() {
    let p = &named();
    let (text, params) = one_recorded(
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
    // listed, as the old client refuses (`RE/graph.py:1035-1040`) ...
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
    // ... and a table named has no schema to be read in
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
fn python_text(column: &ArrayRef, row: usize) -> Option<String> {
    if column.is_null(row) {
        return None;
    }
    Some(match column.data_type() {
        DataType::Utf8 => column.as_string::<i32>().value(row).to_owned(),
        DataType::LargeUtf8 => column.as_string::<i64>().value(row).to_owned(),
        DataType::Int64 => column.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Boolean => if column.as_boolean().value(row) {
            "True"
        } else {
            "False"
        }
        .to_owned(),
        // A date or time cell (`created_on`) is never read.
        _ => "<unread>".to_owned(),
    })
}

/// `batches` in the row adapter's text form: each column nullable text, in
/// the result's order, each non-null cell its `str`.
fn as_rows(batches: &[RecordBatch]) -> Vec<RecordBatch> {
    batches
        .iter()
        .map(|batch| {
            let schema = Arc::new(Schema::new(
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| Field::new(f.name(), DataType::Utf8, true))
                    .collect::<Vec<_>>(),
            ));
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|column| {
                    Arc::new(
                        (0..column.len())
                            .map(|row| python_text(column, row))
                            .collect::<StringArray>(),
                    ) as ArrayRef
                })
                .collect();
            RecordBatch::try_new(schema, columns).unwrap()
        })
        .collect()
}

fn read(request: Value, batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    let request: DataRequest = serde_json::from_value(request).unwrap();
    layer().read(&request, &named(), batches)
}

/// The recorded `DESCRIBE TABLE` of `name` in `case`, read as its columns
/// and its keys.
fn described(case: &str, name: &str) -> (CatalogAnswer, CatalogAnswer) {
    let rows = as_rows(&recorded_result(
        case,
        &format!(r#"DESCRIBE TABLE "{DB}"."{SCHEMA}"."{name}""#),
        &[],
    ));
    let ask = |part: &str| json!({"request": "describe", "part": part, "tables": [table(name)]});
    (
        read(ask("columns"), &rows).unwrap(),
        read(ask("keys"), &rows).unwrap(),
    )
}

#[test]
fn a_tables_columns_are_read_from_its_describe_as_the_old_client_reads_them() {
    let (columns, keys) = described("base.binary", "users");
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
    let (columns, keys) = described("key.past-2-53.catalog", "users");
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
    let (_, keys) = described("rule.composite-key-declared", "orders");
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
    for (case, name, column, dtype) in [
        (
            "decimal.target-38-18",
            "orders",
            "price",
            Some(Dtype::Float),
        ),
        ("decimal.feature-38-0-wide", "users", "f", Some(Dtype::Int)),
        ("time.timestamptz", "orders", "ts", Some(Dtype::Date)),
        ("time.timestamp-ns", "orders", "ts", Some(Dtype::Date)),
        ("time.feature-time-of-day", "users", "f", Some(Dtype::Time)),
        ("feature.bool", "users", "f", Some(Dtype::Bool)),
        ("base.binary", "orders", "price", Some(Dtype::Float)),
        ("nested.list-int", "users", "f", Some(Dtype::Unsupported)),
        ("nested.map", "users", "f", Some(Dtype::Unsupported)),
        ("nested.union", "users", "f", Some(Dtype::Unsupported)),
    ] {
        let (CatalogAnswer::Columns(columns), _) = described(case, name) else {
            panic!("not columns");
        };
        let found = columns.iter().find(|c| c.name == column).unwrap();
        assert_eq!(
            found.dtype, dtype,
            "{case} {name}.{column} {}",
            found.declared_type
        );
    }
    // The declarations no recording holds, each as `_to_dtype` reads it.
    for (declared, dtype) in [
        ("NUMBER(10,2)", Some(Dtype::Float)),
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

/// A `DESCRIBE TABLE` answer in the row adapter's text form: name, type,
/// kind, `null?`, default, primary key, unique key.
fn describe_rows(rows: &[(&str, &str, &str, &str, &str)]) -> Vec<RecordBatch> {
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
    let column = |pick: &dyn Fn(&(&str, &str, &str, &str, &str)) -> Option<String>| -> ArrayRef {
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

fn foreign_keys(case: &str, name: &str) -> Vec<CatalogForeignKey> {
    let rows = as_rows(&recorded_result(
        case,
        &format!(r#"SHOW IMPORTED KEYS IN TABLE "{DB}"."{SCHEMA}"."{name}""#),
        &[],
    ));
    let CatalogAnswer::ForeignKeys(keys) = read(
        json!({"request": "describe", "part": "foreign_keys", "tables": [table(name)]}),
        &rows,
    )
    .unwrap() else {
        panic!("not foreign keys");
    };
    keys
}

#[test]
fn a_foreign_key_is_read_from_the_imported_keys_by_its_referenced_keys_name() {
    let keys = foreign_keys("snowflake.forecast.catalog-keys", "orders");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].columns, vec!["buyer".to_owned()]);
    assert_eq!(keys[0].referenced_columns, vec!["uid".to_owned()]);
    // The target table joined with dots, unquoted.
    assert_eq!(keys[0].referenced_table, format!("{DB}.{SCHEMA}.users"));
    assert!(foreign_keys("base.binary", "orders").is_empty());
    // A key of two columns: its two rows share the referenced key's name,
    // so they are one key, which links no table (`SNOW/table.py:174-189`).
    let keys = foreign_keys("rule.composite-foreign-key-declared", "orders");
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].key, keys[1].key);
}

#[test]
fn the_tables_and_the_session_are_read_from_their_answers() {
    let listing = recorded_result(
        "base.binary",
        "\n                    SELECT TABLE_NAME\n                    FROM \"{catalog}\".INFORMATION_SCHEMA.TABLES\n                    WHERE TABLE_SCHEMA = ?\n                    ",
        &[SCHEMA],
    );
    assert_eq!(
        read(
            json!({"request": "describe", "part": "tables", "tables": []}),
            &listing
        )
        .unwrap(),
        CatalogAnswer::Tables(vec!["users".to_owned(), "orders".to_owned()])
    );
    let session = recorded_result(
        "snowflake.name.database-only",
        "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()",
        &[],
    );
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
    // No current database: refused, never read as a name of no parts.
    assert!(read(ask, &current(None, Some("S"))).is_err());
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

/// The recorded preview of `columns` from `name` in `case`, as the
/// connector handed it to the old client.
fn previewed(case: &str, name: &str, columns: &[&str]) -> Vec<RecordBatch> {
    let quoted: Vec<String> = columns.iter().map(|c| format!("\"{c}\"")).collect();
    recorded_result(
        case,
        &format!(
            r#"SELECT {} FROM "{DB}"."{SCHEMA}"."{name}" LIMIT 1000"#,
            quoted.join(", ")
        ),
        &[],
    )
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
    let read = conformed(&previewed(
        "base.binary",
        "users",
        &["user_id", "age", "country"],
    ));
    assert_eq!(read[0].column(0).data_type(), &DataType::Int64);
    assert_eq!(read[0].column(2).data_type(), &DataType::Utf8);
    // A `NUMBER(38,18)`: a double.
    let read = conformed(&previewed(
        "decimal.target-38-18",
        "orders",
        &["order_id", "user_id", "price", "ts"],
    ));
    assert_eq!(read[0].column(2).data_type(), &DataType::Float64);
    assert_eq!(
        read[0]
            .column(2)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(1),
        11.5
    );
    // A `NUMBER(38,0)` past `int64`: left a decimal.
    let read = conformed(&previewed(
        "decimal.feature-38-0-wide",
        "users",
        &["user_id", "age", "country", "f"],
    ));
    assert_eq!(read[0].column(3).data_type(), &DataType::Decimal128(38, 0));
    // A `TIMESTAMP_LTZ` in a Los Angeles session: its UTC instant, no zone.
    let given = previewed(
        "time.timestamptz",
        "orders",
        &["order_id", "user_id", "price", "ts"],
    );
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
    ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: sqlstate.map(str::to_owned),
        vendor_code: vendor_code.map(str::to_owned),
        error_class: None,
    }
}

#[test]
fn only_snowflakes_errno_2003_says_an_object_is_not_found() {
    let layer = layer();
    let p = &named();
    assert!(
        layer
            .failure(p, &failed(Some("2003"), Some("02000")))
            .not_found
    );
    assert!(layer.failure(p, &failed(Some("2003"), None)).not_found);
    // A login, a permission or a connection failure says nothing of the
    // table, so no other spelling is tried.
    for code in ["390100", "390114", "3001", "250001", "2043"] {
        assert!(
            !layer.failure(p, &failed(Some(code), None)).not_found,
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
    assert!(!capabilities.derived_relations);
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
