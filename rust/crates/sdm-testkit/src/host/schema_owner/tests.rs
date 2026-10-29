// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn description() -> SourceDescription {
    let pin: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../contract/pinned/base.regression.core.json"),
        )
        .unwrap(),
    )
    .unwrap();
    serde_json::from_value(pin["described"].clone()).unwrap()
}

fn flag(schema: &HeldSchema, table: &str, column: &str) -> (bool, bool) {
    let column = schema
        .tables
        .iter()
        .find(|held| held.name == table)
        .unwrap()
        .columns
        .iter()
        .find(|held| held.name == column)
        .unwrap();
    (column.untyped, column.stype_declared)
}

#[test]
fn initial_flags_follow_the_actual_initial_time_assignment_not_physical_dtype() {
    let mut source = description();
    let mut extra = source.tables[0].columns.last().unwrap().clone();
    assert_eq!(extra.name, "ts");
    extra.name = "other_time".into();
    source.tables[0].columns.push(extra.clone());
    extra.name = "ts".into();
    source.tables[1].columns.push(extra);
    source.tables[0].time_column = Some("other_time".into());
    let owner = SchemaOwner::new(source);
    assert_eq!(flag(owner.held(), "orders", "ts"), (false, true));
    assert_eq!(flag(owner.held(), "orders", "other_time"), (false, true));
    assert_eq!(flag(owner.held(), "users", "ts"), (false, false));
}

#[test]
fn adoption_keeps_host_flags_and_the_actual_core_fields_and_declared_reference() {
    let mut owner = SchemaOwner::new(description());
    let mut answer = owner.held().clone();
    for table in &mut answer.tables {
        for column in &mut table.columns {
            column.untyped = false;
            column.stype_declared = false;
        }
    }
    answer.tables[0].duplicate_keys = None;
    owner.adopt(answer, None).unwrap();
    assert_eq!(flag(owner.held(), "orders", "ts"), (false, true));
    assert_eq!(owner.held().tables[0].duplicate_keys, None);
    let answer = owner.held().clone();
    owner
        .adopt(
            answer,
            Some(&ColumnRef {
                table: "users".into(),
                column: "age".into(),
            }),
        )
        .unwrap();
    assert_eq!(flag(owner.held(), "users", "age"), (false, true));
    assert_eq!(flag(owner.held(), "orders", "price"), (false, false));
    assert_eq!(flag(owner.held(), "orders", "ts"), (false, true));
}

#[test]
fn invalid_adoption_never_replaces_half_the_owned_schema() {
    let mut owner = SchemaOwner::new(description());
    let before = owner.held().clone();
    let answer = before.clone();
    assert!(
        owner
            .adopt(
                answer,
                Some(&ColumnRef {
                    table: "users".into(),
                    column: "absent".into(),
                }),
            )
            .is_err()
    );
    assert_eq!(owner.held(), &before);
    let mut answer = before.clone();
    answer.tables[0].columns[0].source = None;
    assert!(owner.adopt(answer, None).is_err());
    assert_eq!(owner.held(), &before);
    let mut answer = before.clone();
    answer.tables[0].columns[0].name = "unexpected_new_column".into();
    assert!(owner.adopt(answer, None).is_err());
    assert_eq!(owner.held(), &before);
}

#[test]
fn encoding_writes_both_actual_booleans_without_changing_other_typed_data() {
    let owner = SchemaOwner::new(description());
    let ordinary = serde_json::to_value(owner.held()).unwrap();
    let mut actual: Value = serde_json::from_slice(&encode(owner.held()).unwrap()).unwrap();
    for (at, table) in owner.held().tables.iter().enumerate() {
        for (column_at, column) in table.columns.iter().enumerate() {
            let written = &mut actual["tables"][at]["columns"][column_at];
            assert_eq!(written["untyped"], Value::Bool(column.untyped));
            assert_eq!(
                written["stype_declared"],
                Value::Bool(column.stype_declared)
            );
            if !column.untyped {
                assert!(
                    ordinary["tables"][at]["columns"][column_at]
                        .get("untyped")
                        .is_none()
                );
                written.as_object_mut().unwrap().remove("untyped");
            }
            if !column.stype_declared {
                assert!(
                    ordinary["tables"][at]["columns"][column_at]
                        .get("stype_declared")
                        .is_none()
                );
                written.as_object_mut().unwrap().remove("stype_declared");
            }
        }
    }
    assert_eq!(actual, ordinary);
}
