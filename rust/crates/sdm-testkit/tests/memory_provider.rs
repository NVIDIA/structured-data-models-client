// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The implementer's own checks of T2.7, beside the test author's
//! acceptance tests (`t27_non_sql_provider.rs`), on the minimal non-SQL
//! test provider (`sdm_testkit::memory`) through the second data layer
//! (`sdm_testkit::layer`):
//!
//! - a prediction whose two schema tables read one source is the one the
//!   same provider gives over two equal sources, whole;
//! - against DuckDB's SQL, a prediction's description, its result and its
//!   request but the in-context examples are DuckDB's; the examples are
//!   drawn by each provider's own seeded sample (DuckDB's reservoir
//!   shuffles even a table it keeps whole), so they are not compared;
//! - the swapped layer refuses a SQL dialect by name.

use std::path::PathBuf;

use sdm_contract::ErrorCode;
use sdm_contract::data::{DataLayer, Dialect, ProviderChoice};
use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::layer::{self, HostLayer};
use serde_json::{Value, json};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn case(id: &str) -> Case {
    let path = repository().join(format!("contract/cases/s1/{id}.json"));
    case::load(&[path]).unwrap().remove(0).case
}

fn bases() -> bases::Bases {
    bases::load(&repository().join("contract/bases.yaml")).unwrap()
}

/// A run's request state and prediction, which must be a prediction.
fn prediction(observed: &Observed) -> (Value, Value) {
    let Answer::Predicted(answer) = &observed.answer else {
        panic!("{:?}", observed.answer);
    };
    (observed.materialized.clone().unwrap(), answer.clone())
}

/// `value` with the request's in-context examples, the provider's own
/// draw, taken out of each request body in it.
fn without_examples(mut value: Value) -> Value {
    for key in ["payload", "body"] {
        if let Some(body) = value.get_mut(key).and_then(Value::as_object_mut) {
            body.remove("context");
        }
    }
    value
}

#[test]
fn a_prediction_is_duckdbs_but_for_its_drawn_examples() {
    let bases = bases();
    for id in ["base.binary", "shape.same-table-twice.buyers"] {
        let given = case(id);
        let on_duckdb = host::run(&given, &bases).unwrap();
        let swapped = host::run_memory(&given, &bases, &layer::root()).unwrap();
        assert!(swapped.driver.is_none(), "{id}: the swapped run ran SQL");
        assert!(
            swapped
                .fetches
                .iter()
                .all(|f| f["plan"]["plan"] == json!("host")),
            "{id}: a fetch was planned as SQL"
        );
        assert_eq!(swapped.described, on_duckdb.described, "{id}");
        let (materialized, predicted) = prediction(&swapped);
        let (duck_materialized, duck_predicted) = prediction(&on_duckdb);
        assert_eq!(predicted["result"], duck_predicted["result"], "{id}");
        assert_eq!(
            without_examples(materialized),
            without_examples(duck_materialized),
            "{id}"
        );
        assert_eq!(
            without_examples(predicted),
            without_examples(duck_predicted),
            "{id}"
        );
    }
}

#[test]
fn two_schema_tables_over_one_source_predict_as_two_sources() {
    // buyers and customers hold the same rows: read both from customers
    // alone, buyers' own table dropped, and the prediction is the one
    // the same provider gives over the two tables.
    let bases = bases();
    let given = case("shape.same-table-twice.buyers");
    let two_sources = host::run_memory(&given, &bases, &layer::root()).unwrap();
    let mut one_source = given.clone();
    one_source
        .input
        .setup
        .get_or_insert_with(Vec::new)
        .push("DROP TABLE buyers".into());
    let spec = one_source
        .core
        .as_mut()
        .and_then(|core| core.describe_source.as_mut())
        .unwrap();
    for table in spec.source["tables"].as_array_mut().unwrap() {
        if table["name"] == "buyers" {
            table["source"] = json!({"source": ["customers"]});
        }
    }
    let swapped = host::run_memory(&one_source, &bases, &layer::root()).unwrap();
    let read: Vec<&Value> = swapped
        .fetches
        .iter()
        .filter(|f| f["request"]["request"] == "preview")
        .map(|f| &f["request"]["table"])
        .collect();
    assert_eq!(
        read.iter()
            .filter(|t| **t == &json!({"source": ["customers"]}))
            .count(),
        2,
        "{read:?}"
    );
    assert_eq!(prediction(&swapped), prediction(&two_sources));
}

#[test]
fn the_swapped_layer_renders_no_sql_dialect() {
    let refused = HostLayer
        .capabilities(&ProviderChoice::Sql {
            dialect: Dialect::new("duckdb"),
        })
        .unwrap_err();
    assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
    assert_eq!(
        refused.fields.get("dialect"),
        Some(&sdm_contract::Detail::text("duckdb"))
    );
}

/// The key counts the core asks are labelled as the dialects label them
/// (T2.12): a source whose enforced key comes before a counted one, so a
/// counted key's label is not its position, is described as on DuckDB.
#[test]
fn a_key_count_is_labelled_as_duckdbs() {
    let given = case("rule.composite-foreign-key-declared");
    let bases = bases();
    let on_duckdb = host::run(&given, &bases).unwrap();
    let swapped = host::run_memory(&given, &bases, &layer::root()).unwrap();
    assert!(on_duckdb.described.is_some());
    assert_eq!(swapped.described, on_duckdb.described);
}
