// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL kit answers a typed request by running exactly the fetches the
//! core's run asks a SQL provider's host to run today (`sdm-core`'s
//! `Asker::fetch`): the plan the data layer renders, bound as it binds it
//! (one fetch, or one per chunk of a key set written as text), preceded by
//! a schema-only read where the source's timestamps need one, its answers
//! joined in order.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::ErrorKind;
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, Dialect, FetchPlan, KeyData, ProviderChoice,
    ProviderError, SqlTimestampEncoding,
};
use sdm_sql::Data;
use sdm_sql::kit::{Answered, KitError, KitFetch, SqlKit};

fn choice(dialect: &str) -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new(dialect),
        namespace: if dialect == "postgres" {
            vec![Some("public".to_owned())]
        } else {
            Vec::new()
        },
        view: None,
        timestamp_encoding: None,
    }
}

/// A lookup of `users` against seeds of an `int64` key and the batch, on a
/// relation of `parts`.
fn lookup(parts: &str) -> DataRequest {
    serde_json::from_str(&format!(
        r#"{{"columns":["user_id","age"],"key":["user_id"],"request":"lookup","seeds":{{"attachment":2,"columns":[{{"name":"key","nullable":true,"type":{{"type":"int64"}}}},{{"name":"batch","nullable":false,"type":{{"type":"int64"}}}}]}},"table":{{"source":[{parts}]}}}}"#
    ))
    .unwrap()
}

/// Seeds `0..count` and their batch positions, as attachment 2.
fn seeds(count: i64) -> BTreeMap<AttachmentId, KeyData> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..count)),
            Arc::new(Int64Array::from_iter_values(0..count)),
        ],
    )
    .unwrap();
    BTreeMap::from([(
        AttachmentId(2),
        KeyData {
            schema,
            batches: vec![batch],
        },
    )])
}

fn float_text(x: f64) -> String {
    format!("{x:?}")
}

/// An answer of the lookup's columns, one row per fetch.
fn rows(fetch: usize) -> Answered {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int64, true),
        Field::new("age", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![fetch as i64])),
            Arc::new(Int64Array::from(vec![30])),
            Arc::new(Int64Array::from(vec![fetch as i64])),
        ],
    )
    .unwrap();
    Answered {
        schema,
        batches: vec![batch],
    }
}

/// What a kit's `run` was asked: each fetch's plan and the key sets it was
/// handed, by id, as row counts.
#[derive(Debug, PartialEq)]
struct Asked {
    plan: FetchPlan,
    keys: BTreeMap<AttachmentId, usize>,
}

fn counted(keys: &BTreeMap<AttachmentId, KeyData>) -> BTreeMap<AttachmentId, usize> {
    keys.iter()
        .map(|(id, made)| (*id, made.batches.iter().map(RecordBatch::num_rows).sum()))
        .collect()
}

/// `kit`'s answer to `request`, and every fetch it asked, each answered
/// by `answer`.
fn ask(
    kit: &SqlKit,
    request: &DataRequest,
    keys: &BTreeMap<AttachmentId, KeyData>,
    mut answer: impl FnMut(usize, &KitFetch<'_>) -> Result<Answered, ProviderError>,
) -> (Result<Answered, KitError>, Vec<Asked>) {
    let mut asked = Vec::new();
    let result = kit.answer(request, keys, float_text, &mut |fetch| {
        asked.push(Asked {
            plan: fetch.plan.clone(),
            keys: counted(fetch.keys),
        });
        answer(asked.len() - 1, &fetch)
    });
    (result, asked)
}

/// The fetches the core's run asks today for `request` on `provider`: the
/// data layer's plan, bound with `keys`.
fn core_fetches(
    provider: &ProviderChoice,
    request: &DataRequest,
    keys: &BTreeMap<AttachmentId, KeyData>,
) -> Vec<Asked> {
    let data = Data::new();
    let plan = data.plan(request, provider).unwrap();
    assert!(
        data.result_metadata(provider, request, &plan)
            .unwrap()
            .is_none()
    );
    data.bind(provider, request, plan, keys, float_text)
        .unwrap()
        .fetches
        .into_iter()
        .map(|fetch| {
            let mut handed = keys.clone();
            handed.extend(fetch.batches);
            let named = fetch.plan.key_sets();
            handed.retain(|id, _| named.contains(id));
            Asked {
                plan: fetch.plan,
                keys: counted(&handed),
            }
        })
        .collect()
}

#[test]
fn a_duckdb_lookup_runs_the_one_fetch_the_core_asks() {
    let provider = choice("duckdb");
    let request = lookup(r#""users""#);
    let keys = seeds(3);
    let kit = SqlKit::new(provider.clone(), None).unwrap();
    let (result, asked) = ask(&kit, &request, &keys, |at, _| Ok(rows(at)));
    assert_eq!(asked, core_fetches(&provider, &request, &keys));
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].keys, BTreeMap::from([(AttachmentId(2), 3)]));
    assert_eq!(result.unwrap().batches.len(), 1);
}

#[test]
fn a_postgres_key_set_past_the_budget_runs_one_fetch_per_chunk_joined_in_order() {
    let provider = choice("postgres");
    let request = lookup(r#""users""#);
    let keys = seeds(300_000);
    let kit = SqlKit::new(provider.clone(), None).unwrap();
    let (result, asked) = ask(&kit, &request, &keys, |at, _| Ok(rows(at)));
    let expected = core_fetches(&provider, &request, &keys);
    assert!(expected.len() >= 2, "{}", expected.len());
    assert_eq!(asked, expected);
    let answered = result.unwrap();
    let firsts: Vec<i64> = answered
        .batches
        .iter()
        .map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        })
        .collect();
    assert_eq!(firsts, (0..expected.len() as i64).collect::<Vec<_>>());
}

#[test]
fn a_databricks_source_needing_microsecond_timestamps_reads_the_result_schema_first() {
    let provider = ProviderChoice::Sql {
        dialect: Dialect::new("databricks"),
        namespace: Vec::new(),
        view: None,
        timestamp_encoding: Some(SqlTimestampEncoding::Microseconds),
    };
    let request = lookup(r#""main","shop","users""#);
    let keys = seeds(3);
    let data = Data::new();
    let plan = data.plan(&request, &provider).unwrap();
    let metadata = data
        .result_metadata(&provider, &request, &plan)
        .unwrap()
        .expect("a schema-only read first");
    let described_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("col_name", DataType::Utf8, false),
        Field::new("data_type", DataType::Utf8, false),
    ]));
    let described = RecordBatch::try_new(
        Arc::clone(&described_schema),
        vec![
            Arc::new(StringArray::from(vec!["user_id", "joined", "batch"])),
            Arc::new(StringArray::from(vec!["bigint", "timestamp_ntz", "bigint"])),
        ],
    )
    .unwrap();
    let projected = data
        .project_result(&provider, &request, &plan, std::slice::from_ref(&described))
        .unwrap();
    assert_ne!(
        projected, plan,
        "the timestamp column is read as microseconds"
    );

    let kit = SqlKit::new(provider.clone(), None).unwrap();
    let (result, asked) = ask(&kit, &request, &keys, |at, _| {
        Ok(if at == 0 {
            Answered {
                schema: Arc::clone(&described_schema),
                batches: vec![described.clone()],
            }
        } else {
            rows(at)
        })
    });
    result.unwrap();
    // The schema-only read binds the key set empty, then the projected
    // plan binds it whole, as the core's run binds them.
    let empty = BTreeMap::from([(
        AttachmentId(2),
        KeyData {
            schema: keys[&AttachmentId(2)].schema.clone(),
            batches: Vec::new(),
        },
    )]);
    let mut expected: Vec<Asked> = data
        .bind(&provider, &request, metadata, &empty, float_text)
        .unwrap()
        .fetches
        .into_iter()
        .map(|fetch| Asked {
            plan: fetch.plan,
            keys: fetch
                .batches
                .iter()
                .map(|(id, made)| (*id, made.batches.iter().map(RecordBatch::num_rows).sum()))
                .collect(),
        })
        .collect();
    expected.extend(
        data.bind(&provider, &request, projected, &keys, float_text)
            .unwrap()
            .fetches
            .into_iter()
            .map(|fetch| Asked {
                plan: fetch.plan,
                keys: fetch
                    .batches
                    .iter()
                    .map(|(id, made)| (*id, made.batches.iter().map(RecordBatch::num_rows).sum()))
                    .collect(),
            }),
    );
    assert_eq!(
        asked.iter().map(|a| &a.plan).collect::<Vec<_>>(),
        expected.iter().map(|a| &a.plan).collect::<Vec<_>>()
    );
}

#[test]
fn a_driver_failure_is_the_provider_s_error_with_what_it_means() {
    let kit = SqlKit::new(choice("duckdb"), None).unwrap();
    let failed = ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: Some("42P01".to_owned()),
        vendor_code: None,
        error_class: None,
        driver_message: None,
        facts: None,
    };
    let (result, _) = ask(&kit, &lookup(r#""users""#), &seeds(1), |_, _| {
        Err(failed.clone())
    });
    match result {
        Err(KitError::Provider(error)) => {
            let facts = error.facts.clone().expect("the engine's reading");
            assert_eq!(
                facts.reason,
                Some(sdm_contract::data::FailureReason::TableUnresolved)
            );
            assert!(!facts.denied);
            assert_eq!(
                ProviderError {
                    facts: None,
                    ..error
                },
                failed
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_host_provider_or_an_unknown_dialect_has_no_kit() {
    let host: ProviderChoice =
        serde_json::from_str(r#"{"provider":"host","capabilities":{"optional":[],"sampling":"seeded_stable","max_seed":0}}"#)
            .unwrap();
    assert!(SqlKit::new(host, None).is_err());
    assert!(SqlKit::new(choice("oracle"), None).is_err());
}

#[test]
fn the_kit_declares_what_the_core_declares_for_its_source() {
    for dialect in ["duckdb", "sqlite", "postgres", "databricks", "snowflake"] {
        let provider = choice(dialect);
        let mut declared = SqlKit::new(provider.clone(), None)
            .unwrap()
            .capabilities()
            .unwrap();
        assert_eq!(declared.database.as_deref(), Some(dialect));
        assert!(declared.reads_tables_as_they_stand, "{dialect}");
        assert_eq!(declared.column_affinities, dialect == "sqlite", "{dialect}");
        assert_eq!(
            declared.tries_upper_case_names,
            dialect == "snowflake",
            "{dialect}"
        );
        let namespace: Vec<String> = if dialect == "postgres" {
            vec!["public".to_owned()]
        } else {
            Vec::new()
        };
        assert_eq!(declared.namespace, namespace, "{dialect}");
        declared.database = None;
        declared.reads_tables_as_they_stand = false;
        declared.column_affinities = false;
        declared.tries_upper_case_names = false;
        declared.namespace = Vec::new();
        assert_eq!(
            declared,
            Data::new().capabilities(&provider).unwrap(),
            "{dialect}"
        );
    }
}

/// A source answering through the kit, as the core sees it (its declared
/// capabilities), reads as the SQL source it is: the same database named,
/// column types, tables read as they stand, names read under its
/// namespace and spelled as it folds them.
#[test]
fn a_source_answering_through_the_kit_reads_as_its_sql_source() {
    use sdm_contract::data::Relation;
    for dialect in ["duckdb", "sqlite", "postgres", "databricks", "snowflake"] {
        let sql = choice(dialect);
        let host = ProviderChoice::Host {
            capabilities: SqlKit::new(sql.clone(), None)
                .unwrap()
                .capabilities()
                .unwrap(),
        };
        let data = Data::new();
        assert_eq!(data.source_database(&host), data.source_database(&sql));
        assert_eq!(
            data.holds_column_types(&host),
            data.holds_column_types(&sql)
        );
        assert_eq!(
            data.reads_tables_as_they_stand(&host),
            data.reads_tables_as_they_stand(&sql)
        );
        for relation in [
            Relation::of(vec!["users".to_owned()]),
            Relation::of(vec!["Main".to_owned(), "Users".to_owned()]),
        ] {
            assert_eq!(
                data.read_as(&host, &relation),
                data.read_as(&sql, &relation)
            );
            assert_eq!(
                data.spellings(&host, &relation),
                data.spellings(&sql, &relation)
            );
        }
    }
}
