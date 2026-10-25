// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQL foreign-key LIST_DISTINCT through the public host's real
//! describe/materialize/predict operations, with frozen43e's existing
//! label vectors and a separate scripted successful ranking response.
//! No model is called and no original reference observation is changed.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use sdm_testkit::case::{AsWritten, Case, Until};
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::{bases, case, endpoint};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn link_case(reverse: bool, predict: bool) -> Case {
    let loaded =
        case::load(&[root().join("contract/cases/s1/target.aggregation.list-distinct.json")])
            .unwrap();
    let mut case = loaded[0].case.clone();
    if reverse {
        case.input.pql = "PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) \
                          FOR EACH items.item_id"
            .into();
    }
    if !predict {
        case.until = Until::Materialize;
        case.core.as_mut().unwrap().predict = None;
    }
    case
}

fn run(case: &Case) -> Observed {
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    host::run(case, &bases).unwrap()
}

fn payload(observed: &Observed) -> &Value {
    &observed
        .materialized
        .as_ref()
        .unwrap_or_else(|| panic!("materialize refused: {:?}", observed.answer))["payload"]
}

/// Existing reference vectors include the ordered targets of both FK
/// directions, and missing RHS references in users/items context rows.
fn reference(reverse: bool) -> Value {
    let vectors: Value = serde_json::from_slice(
        &std::fs::read(root().join("contract/t210-target-vectors.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        vectors["reference"],
        "43e55f4f58924384c3cb52f67c8ed2e73ba45fa1"
    );
    let scenario = if reverse {
        "aggregation.list-distinct.users@duckdb"
    } else {
        "aggregation.list-distinct@duckdb"
    };
    vectors["vectors"][scenario]["payloads"][0].clone()
}

#[test]
fn both_fk_directions_keep_the_reference_labels_entity_roles_and_occurrences() {
    for reverse in [false, true] {
        let observed = run(&link_case(reverse, false));
        assert_eq!(observed.operations, ["describe_source", "materialize"]);
        assert!(matches!(&observed.answer, Answer::Materialized(_)));
        let actual = payload(&observed);
        let expected = host::numbers_as_text(reference(reverse));
        assert_eq!(actual["task"], expected["task"]);
        for half in ["context", "predict"] {
            assert_eq!(actual[half]["instance_table"], expected[half], "{half}");
        }
        assert_eq!(actual["output"], json!({"fields": ["rankings"]}));
        assert_eq!(
            actual["inference"]["inference_config"],
            host::numbers_as_text(json!({
                "kind": "base", "num_estimators": 1,
                "column_shuffle": false, "category_shuffle": false,
                "hop_shuffle": false,
            }))
        );
        assert_eq!(
            actual["schema"]["instance_table"]["columns"]["TARGET"],
            json!({"dtype": "stringlist", "stype": "multicategorical"})
        );
    }
}

#[test]
fn repeated_entity_roles_keep_both_names_but_only_the_last_alias_is_an_id() {
    let mut case = link_case(false, false);
    case.input.pql = "PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) \
                      FOR EACH users.user_id"
        .into();
    let observed = run(&case);
    let actual = payload(&observed);
    assert_eq!(
        actual["task"]["entity_table_names"],
        json!(["users", "users"])
    );
    let columns = &actual["schema"]["instance_table"]["columns"];
    assert!(columns["__kumo_entity_ref_0"].get("stype").is_none());
    assert_eq!(columns["__kumo_entity_ref_1"]["stype"], "id");
    let entity_links: Vec<_> = actual["schema"]["relationships"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|link| link.get("source_table").is_none())
        .cloned()
        .collect();
    assert_eq!(
        entity_links,
        vec![
            json!({
                "source_columns": ["__kumo_entity_ref_1"],
                "target_table": "users", "target_columns": ["user_id"],
            });
            2
        ]
    );
}

#[test]
fn a_non_fk_list_target_names_the_unqualified_column_before_any_model_call() {
    let mut case = link_case(false, true);
    case.input.pql = "PREDICT LIST_DISTINCT(orders.price, 0, 30, days) \
                      FOR EACH users.user_id"
        .into();
    let observed = endpoint::answered_by(
        |_, _| panic!("a non-FK list target must not call the model"),
        || run(&case),
    );
    assert_eq!(observed.operations, ["describe_source", "materialize"]);
    assert!(observed.materialized.is_none());
    let Answer::Error { code, fields, .. } = observed.answer else {
        panic!("the target must refuse as unsupported multilabel classification");
    };
    assert_eq!(code, "UNSUPPORTED_REQUEST");
    assert_eq!(fields.get("column").map(String::as_str), Some("price"));
}

#[test]
fn rhs_references_keep_float_and_missing_sampled_values_instead_of_target_labels() {
    let observed = run(&link_case(false, false));
    let actual = payload(&observed);
    let schema = &actual["schema"]["instance_table"]["columns"];
    assert_eq!(schema["__kumo_entity_ref_0"]["dtype"], "int64");
    assert_eq!(schema["__kumo_entity_ref_1"]["dtype"], "float64");
    let rows = actual["context"]["instance_table"]["rows"]
        .as_array()
        .unwrap();
    assert_eq!(rows.len(), 6);
    // These users have future-positive labels but no historical sampled
    // RHS row in their own batch. The reference leaves that reference null.
    for (at, target) in [(1, "1"), (5, "2")] {
        assert_eq!(rows[at][2], Value::Null);
        assert_eq!(rows[at][4], json!([target]));
    }
    assert_eq!(rows[0][2], json!({"number": "2.0"}));
    assert_eq!(rows[2][2], json!({"number": "3.0"}));
}

#[test]
fn no_historical_rhs_table_refuses_before_a_model_call() {
    let mut case = link_case(false, true);
    // One hop reaches buys from users, but cannot reach items through
    // buys.item_id. Zero hops would only test invalid option validation.
    case.input.materialize.num_hops = Some(AsWritten("1".into()));
    let observed = endpoint::answered_by(
        |_, _| panic!("a subgraph with no historical RHS must not call the model"),
        || run(&case),
    );
    assert_eq!(observed.operations, ["describe_source", "materialize"]);
    assert!(observed.materialized.is_none());
    let Answer::Error { kind, code, .. } = observed.answer else {
        panic!("expected the historical-target refusal");
    };
    assert_eq!(kind, "invalid_request");
    assert_eq!(code, "INVALID_REQUEST");
}

#[test]
fn an_unwindowed_link_query_keeps_the_reference_validation_refusal() {
    let mut case = link_case(false, true);
    case.input.pql = "PREDICT LIST_DISTINCT(buys.item_id) FOR EACH users.user_id".into();
    let observed = endpoint::answered_by(
        |_, _| panic!("the invalid static link query must not call the model"),
        || run(&case),
    );
    assert_eq!(observed.operations, ["describe_source", "materialize"]);
    assert!(observed.materialized.is_none());
    let Answer::Error { code, .. } = observed.answer else {
        panic!("the query's existing temporal validation must refuse");
    };
    assert_eq!(code, "INVALID_REQUEST");
}

#[test]
fn the_original_case_reaches_its_unchanged_no_classes_endpoint_refusal() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let recorded = Rc::clone(&calls);
    let observed = endpoint::answered_by(
        move |_, body| {
            let sent: Value = serde_json::from_slice(body).unwrap();
            recorded.borrow_mut().push(sent);
            let (status, response) = endpoint::answer("POST", endpoint::PREDICTIONS, body);
            assert_eq!(status, 400);
            assert_eq!(response, json!({"error": "no_classes"}));
            (status, serde_json::to_vec(&response).unwrap())
        },
        || run(&link_case(false, true)),
    );
    assert_eq!(calls.borrow().len(), 1);
    assert_eq!(
        observed.operations,
        ["describe_source", "materialize", "predict"]
    );
    let Answer::Error { kind, code, fields } = observed.answer else {
        panic!("the original endpoint refuses this payload");
    };
    assert_eq!(kind, "model_endpoint");
    assert_eq!(code, "INVALID_RESPONSE");
    assert_eq!(fields.get("field").map(String::as_str), Some("status"));
    assert_eq!(
        fields.get("reason").map(String::as_str),
        Some("not_a_success")
    );
}

#[test]
fn a_separate_successful_script_preserves_rankings_and_restores_public_entity_ids() {
    let bodies = Rc::new(RefCell::new(Vec::new()));
    let recorded = Rc::clone(&bodies);
    let observed = endpoint::answered_by(
        move |call, body| {
            assert_eq!(call, 0);
            let sent: Value = serde_json::from_slice(body).unwrap();
            let rows = sent["predict"]["instance_table"]["rows"]
                .as_array()
                .unwrap();
            // Transport ids are 6/7, whereas public users are 0/1.
            assert_eq!(rows[0][0], 6);
            assert_eq!(rows[1][0], 7);
            let predictions: Vec<Value> = rows
                .iter()
                .enumerate()
                .map(|(at, row)| {
                    json!({
                        "row_index": at, "id": row[0].to_string(),
                        "prediction": null,
                        "rankings": [{"item": 3, "score": 0.9},
                                     {"item": 1, "score": 0.4}],
                    })
                })
                .collect();
            recorded.borrow_mut().push(sent);
            (
                200,
                serde_json::to_vec(&json!({
                    "id": "offline-link-proof", "model": "kumo-relational",
                    "metadata": {}, "predictions": predictions,
                }))
                .unwrap(),
            )
        },
        || run(&link_case(false, true)),
    );
    assert_eq!(
        observed.operations,
        ["describe_source", "materialize", "predict"]
    );
    let Answer::Predicted(predicted) = observed.answer else {
        panic!("a genuine temporal-link request must decode its valid response");
    };
    let bodies = bodies.borrow();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["task"]["kind"], "temporal_link_prediction");
    assert_eq!(bodies[0]["output"], json!({"fields": ["rankings"]}));
    // Only caller-asked inference settings travel to Predict.
    assert!(bodies[0]["inference"].get("inference_config").is_none());
    let columns = predicted["result"]["columns"].as_array().unwrap();
    let at = |name: &str| columns.iter().position(|c| c["name"] == name).unwrap();
    let rows = predicted["result"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for (row, id) in rows.iter().zip(["0", "1"]) {
        assert_eq!(row[at("id")], json!({"i64": id}));
        let rankings = row[at("rankings")]["list"].as_array().unwrap();
        let values: Vec<Value> = rankings
            .iter()
            .map(|value| serde_json::from_str(value["utf8"].as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(
            values,
            [
                json!({"item": 3, "score": 0.9}),
                json!({"item": 1, "score": 0.4})
            ]
        );
    }
}
