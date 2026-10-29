// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual source/edit/materialize journeys retain this host's schema metadata.

use std::collections::BTreeMap;
use std::path::PathBuf;

use sdm_contract::ops::Operation;
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case};
use serde_json::{Value, json};

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn materializing(edits: Vec<Value>) -> case::Case {
    let mut case = case::load(&[contract().join("cases/s1/base.regression.json")])
        .unwrap()
        .remove(0)
        .case;
    case.until = case::Until::Materialize;
    case.core.as_mut().unwrap().predict = None;
    case.input.setup = Some(vec![
        "ALTER TABLE orders ADD COLUMN other_ts TIMESTAMP DEFAULT TIMESTAMP '2024-01-01'".into(),
        "ALTER TABLE orders ADD COLUMN age INTEGER DEFAULT 1".into(),
    ]);
    case.input.edits = (!edits.is_empty()).then(|| {
        edits
            .into_iter()
            .map(|edit| serde_json::from_value(edit).unwrap())
            .collect()
    });
    case
}

fn flags(schema: &Value) -> BTreeMap<(String, String), (bool, bool)> {
    let mut flags = BTreeMap::new();
    for table in schema["tables"].as_array().unwrap() {
        for column in table["columns"].as_array().unwrap() {
            let key = (
                table["name"].as_str().unwrap().to_owned(),
                column["name"].as_str().unwrap().to_owned(),
            );
            assert!(
                flags
                    .insert(
                        key,
                        (
                            column["untyped"]
                                .as_bool()
                                .expect("explicit untyped Boolean"),
                            column["stype_declared"]
                                .as_bool()
                                .expect("explicit declaration Boolean"),
                        ),
                    )
                    .is_none()
            );
        }
    }
    flags
}

fn only_declared(schema: &Value, declared: &[(&str, &str)]) {
    let actual = flags(schema);
    assert!(!actual.is_empty());
    for ((table, column), (untyped, assigned)) in &actual {
        assert!(!untyped);
        assert_eq!(
            *assigned,
            declared.contains(&(table.as_str(), column.as_str())),
            "{table}.{column}"
        );
    }
    for (table, column) in declared {
        assert!(actual.contains_key(&(table.to_string(), column.to_string())));
    }
}

fn materialize_input(observed: &host::Observed) -> &Value {
    let mut inputs = observed
        .inputs
        .iter()
        .filter(|input| input["op"] == "materialize");
    let input = inputs.next().expect("actual materialize Start");
    assert!(inputs.next().is_none());
    input
}

#[test]
fn actual_initial_source_input_writes_every_flag_and_only_declares_the_chosen_time() {
    let case = materializing(vec![]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&case, &bases).unwrap();
    assert!(matches!(observed.answer, Answer::Materialized(_)));
    assert_eq!(observed.operations, ["describe_source", "materialize"]);
    only_declared(&materialize_input(&observed)["schema"], &[("orders", "ts")]);
    assert_eq!(
        observed.described.as_ref().unwrap()["tables"][0]["time_column"],
        "ts"
    );
}

#[test]
fn actual_declared_reference_survives_an_unrelated_edit_and_same_named_column() {
    let case = materializing(vec![
        json!({"op":"set_stype","table":"users","column":"age","stype":"numerical"}),
        json!({"op":"set_primary_key","table":"users","column":"user_id"}),
    ]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&case, &bases).unwrap();
    assert!(matches!(observed.answer, Answer::Materialized(_)));
    assert_eq!(observed.edits.len(), 2);
    only_declared(&observed.edits[0]["schema"], &[("orders", "ts")]);
    only_declared(
        &observed.edits[1]["schema"],
        &[("orders", "ts"), ("users", "age")],
    );
    only_declared(
        &materialize_input(&observed)["schema"],
        &[("orders", "ts"), ("users", "age")],
    );
}

#[test]
fn actual_time_assignment_and_clear_keep_both_columns_declaration_history() {
    let case = materializing(vec![
        json!({"op":"set_time_column","table":"orders","column":"other_ts"}),
        json!({"op":"clear_time_column","table":"orders"}),
        json!({"op":"set_primary_key","table":"users","column":"user_id"}),
    ]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&case, &bases).unwrap();
    assert_eq!(observed.edits.len(), 3);
    for input in &observed.edits[1..] {
        only_declared(
            &input["schema"],
            &[("orders", "ts"), ("orders", "other_ts")],
        );
    }
    let input = materialize_input(&observed);
    only_declared(
        &input["schema"],
        &[("orders", "ts"), ("orders", "other_ts")],
    );
    assert!(input["schema"]["tables"][0]["time_column"].is_null());
    assert_eq!(observed.operations.last().unwrap(), "materialize");
}

#[test]
fn actual_delete_does_not_transfer_a_declared_flag_to_another_table() {
    let case = materializing(vec![
        json!({"op":"set_stype","table":"users","column":"age","stype":"numerical"}),
        json!({"op":"delete_column","table":"users","column":"age"}),
    ]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&case, &bases).unwrap();
    assert!(matches!(observed.answer, Answer::Materialized(_)));
    let schema = &materialize_input(&observed)["schema"];
    only_declared(schema, &[("orders", "ts")]);
    assert!(!flags(schema).contains_key(&("users".into(), "age".into())));
    assert_eq!(
        flags(schema)[&("orders".into(), "age".into())],
        (false, false)
    );
}

#[test]
fn actual_refused_edit_has_no_later_materialize_or_reconstructed_schema() {
    let case = materializing(vec![
        json!({"op":"set_stype","table":"users","column":"age","stype":"not_a_type"}),
    ]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&case, &bases).unwrap();
    assert_eq!(observed.operations, ["describe_source", "edit_schema"]);
    assert!(observed.materialized.is_none());
    let Answer::Error { kind, code, .. } = observed.answer else {
        panic!("edit was not refused")
    };
    assert_eq!(kind, "invalid_request");
    assert_eq!(code, "INVALID_REQUEST");
    only_declared(&observed.edits[0]["schema"], &[("orders", "ts")]);
}

#[test]
fn actual_repeated_and_selected_operation_paths_use_the_same_complete_encoder() {
    let case = materializing(vec![]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    for observed in host::materialize_repeatedly(&case, &bases, 2).unwrap() {
        assert!(matches!(observed.answer, Answer::Materialized(_)));
        only_declared(&materialize_input(&observed)["schema"], &[("orders", "ts")]);
    }
    let (observed, driven) = host::materialize_as(&case, &bases, &Operation::Materialize).unwrap();
    assert!(driven.is_some());
    only_declared(&materialize_input(&observed)["schema"], &[("orders", "ts")]);
}
