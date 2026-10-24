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
//!   listing, a semantic view's YAML read).
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

/// The seeds the old client's JSON `payload` holds: each id, and each of
/// the window's bounds where the rows carry them, in batch order.
fn recorded_seeds(payload: &str) -> sdm_contract::data::KeyData {
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

/// The recorded key read of `case` whose text starts so and contains
/// `marker`: its text, its key set's JSON and its other parameters.
fn recorded_key_read(case: &str, marker: &str) -> (String, String, Vec<String>) {
    let (text, params) = recorded(case)
        .into_iter()
        .find(|(text, _)| text.contains("FLATTEN") && text.contains(marker))
        .unwrap_or_else(|| panic!("{case} recorded no key read with {marker:?}"));
    (text, params[0].clone(), params[1..].to_vec())
}

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
fn a_window_read_by_time_is_the_old_clients_statement_and_json() {
    let (text, payload, others) = recorded_key_read("base.binary", "\"ts\"\nFROM TMP");
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
    assert_eq!(
        bound(request, recorded_seeds(&payload)),
        vec![(text, payload, others)]
    );
}

#[test]
fn a_neighbor_read_is_the_old_clients_statement_and_json() {
    let (text, payload, others) = recorded_key_read("base.binary", "QUALIFY");
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
    assert_eq!(
        bound(request, recorded_seeds(&payload)),
        vec![(text, payload, others)]
    );
}

#[test]
fn a_key_lookup_is_the_old_clients_statement_and_json_one_row_per_seed() {
    let (text, payload, others) = recorded_key_read("base.binary", "f.value::NUMBER");
    let lookup = |flags: &[(&str, &str)]| {
        json!({
            "request": "lookup", "table": table("users"), "key": ["user_id"],
            "columns": ["user_id", "age", "country"],
            "seeds": {"attachment": 2, "columns": seeds_json(&[
                ("key", json!({"type": "int64"})), ("batch", json!({"type": "int64"}))], flags)}})
    };
    // A key the source keeps unique: the old client's plain join.
    assert_eq!(
        bound(
            lookup(&[("key", "sdm:key_unique")]),
            recorded_seeds(&payload)
        ),
        vec![(text.clone(), payload.clone(), others.clone())]
    );
    // Any other: the old client's first row of each seed, and how many rows
    // hold its key, which the old client counts after its fetch
    // (`SNOW/sampler.py:246-260`), here in the statement.
    let (select, rest) = text.split_once("\nFROM TMP").unwrap();
    let counted = format!(
        "{select}, COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) AS \"__kumo_matches__\"\nFROM TMP{rest}\nQUALIFY ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__ ORDER BY \"user_id\") = 1"
    );
    assert_eq!(
        bound(lookup(&[]), recorded_seeds(&payload)),
        vec![(counted, payload.clone(), others.clone())]
    );
    // A count not read: the first row of each seed alone.
    let first = format!(
        "{text}\nQUALIFY ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__ ORDER BY \"user_id\") = 1"
    );
    assert_eq!(
        bound(
            lookup(&[("batch", "sdm:count_unread")]),
            recorded_seeds(&payload)
        ),
        vec![(first, payload, others)]
    );
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
    let (text, params) = one_recorded(
        "key.declared-int32",
        p,
        json!({"request": "describe", "part": "key_enforcement", "tables": [table("users")]}),
    );
    assert_eq!(text, format!(r#"SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}""#));
    assert!(params.is_empty());
    let ask = json!({"request": "describe", "part": "key_enforcement", "tables": [table("users")]});
    // The recorded listing, every table plain: none enforces its key.
    let listing = as_rows(&recorded_result(
        "key.declared-int32",
        &format!(r#"SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}""#),
        &[],
    ));
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
    let statement = format!("SHOW IMPORTED KEYS IN TABLE \"{DB}\".\"{SCHEMA}\".\"orders\"");
    let mut rows = as_rows(&recorded_result(
        "rule.composite-foreign-key-declared",
        &statement,
        &[],
    ));
    let batch = &rows[0];
    let mut columns = batch.columns().to_vec();
    columns[12] = Arc::new(StringArray::from(vec!["buyer_fk", "seller_fk"]));
    columns[13] = Arc::new(StringArray::from(vec!["users_pk", "users_pk"]));
    rows[0] = RecordBatch::try_new(batch.schema(), columns).unwrap();
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
        let sent = bound(request, recorded_seeds(&payload));
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
