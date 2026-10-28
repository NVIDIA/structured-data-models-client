// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native constructor meaning and later key metadata remain separate values.

use super::*;
use sdm_contract::ops::{HeldColumn, HeldTable};

fn table() -> FrameTableMeaning {
    let column = |name: &str, dtype: &str| HeldColumn {
        name: name.into(),
        untyped: false,
        stype_declared: false,
        dtype: dtype.into(),
        stype: Stype::Id,
        stype_before_key: None,
        source: None,
    };
    FrameTableMeaning {
        table: HeldTable {
            name: "history".into(),
            source: "history".into(),
            columns: vec![column("a", "string"), column("b", "int64")],
            primary_key: vec!["a".into(), "b".into()],
            frame_key_form: Some(sdm_contract::frames::FrameKeyForm::Sequence),
            time_column: None,
            duplicate_keys: None,
        },
        constructed: vec![
            FrameConstructedColumn {
                name: "a".into(),
                dtype: Dtype::String,
                stype: Stype::Text,
            },
            FrameConstructedColumn {
                name: "b".into(),
                dtype: Dtype::Int,
                stype: Stype::Numerical,
            },
        ],
    }
}

fn reached_metadata(key: FrameKeyMeaning, table: HeldTable) -> FrameTableMetadata {
    let derived = key
        .composite
        .as_ref()
        .filter(|definition| {
            !key.table
                .constructed
                .iter()
                .any(|column| column.name == definition.derived)
        })
        .map(|definition| FrameConstructedColumn {
            name: definition.derived.clone(),
            dtype: Dtype::String,
            stype: Stype::Id,
        });
    FrameTableMetadata {
        key,
        table,
        derived,
    }
}

#[test]
fn native_physical_float_and_enumerated_float_have_different_constructed_meaning() {
    assert_eq!(
        source_keys::constructed(&FrameConstructorValues::Typed {
            physical: ColumnType::Float64
        })
        .unwrap(),
        Dtype::Float
    );
    assert_eq!(
        source_keys::constructed(&FrameConstructorValues::Enumerated).unwrap(),
        Dtype::String
    );
    for (content, dtype) in [
        (FrameElementContent::Integer, Dtype::Int),
        (FrameElementContent::Real, Dtype::Float),
        (FrameElementContent::Boolean, Dtype::Bool),
        (FrameElementContent::Duration, Dtype::Timedelta),
        (FrameElementContent::Other, Dtype::String),
    ] {
        assert_eq!(
            source_keys::constructed(&FrameConstructorValues::Elements { content }).unwrap(),
            dtype
        );
    }
    let mut fixed = table();
    // Caller metadata says string, but the actual fixed constructor was float.
    fixed.constructed[0].dtype = Dtype::Float;
    let error = source_keys::key(fixed, vec!["a".into(), "b".into()]).unwrap_err();
    assert_eq!(
        error.fields["native_key_reason"],
        Detail::text("approximate_component")
    );
    assert_eq!(error.fields["column"], Detail::text("a"));
}

#[test]
fn later_metadata_preserves_captured_key_cast_dtype_and_current_time_stype() {
    let fixed = table();
    let key = source_keys::key(fixed.clone(), vec!["a".into(), "b".into()]).unwrap();
    let mut live = fixed.table;
    live.primary_key = vec!["b".into()];
    live.columns[1].dtype = "string".into();
    live.columns[1].stype = Stype::Categorical;
    let resolved = source_keys::metadata(reached_metadata(key, live)).unwrap();
    assert_eq!(resolved.table.primary_key, ["a", "b"]);
    assert_eq!(resolved.table.columns[1].dtype, "int64");
    assert_eq!(resolved.table.columns[1].stype, Stype::Categorical);
}

#[test]
fn preexisting_derived_column_is_admitted_then_current_primary_stype_is_checked() {
    let mut fixed = table();
    let derived = sdm_engine::graph::folded_name(&fixed.table.primary_key);
    let mut column = fixed.table.columns[1].clone();
    column.name.clone_from(&derived);
    fixed.table.columns.push(column);
    fixed.constructed.push(FrameConstructedColumn {
        name: derived,
        dtype: Dtype::Int,
        stype: Stype::Numerical,
    });
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    assert!(source_keys::metadata(reached_metadata(key.clone(), fixed.table.clone())).is_ok());
    let mut live = fixed.table;
    live.columns[2].stype = Stype::Numerical;
    let error = source_keys::metadata(reached_metadata(key, live)).unwrap_err();
    assert_eq!(
        error.fields["native_key_reason"],
        Detail::text("primary_stype")
    );
}

#[test]
fn malformed_definition_or_table_key_companions_never_select_another_meaning() {
    let fixed = table();
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    for change in 0..4 {
        let mut forged = key.clone();
        match change {
            0 => forged.composite.as_mut().unwrap().columns.reverse(),
            1 => forged.composite.as_mut().unwrap().source = "another".into(),
            2 => forged.composite.as_mut().unwrap().derived.push('x'),
            _ => forged.composite.as_mut().unwrap().columns[0].dtype = Dtype::Int64,
        }
        assert!(source_keys::metadata(reached_metadata(forged, fixed.table.clone())).is_err());
    }
}

#[test]
fn native_meaning_checks_keep_full_identity_and_current_component_assignments() {
    let mut fixed = table();
    fixed.table.columns[0].stype = Stype::Categorical;
    fixed.table.columns[1].stype_before_key = Some(Stype::Categorical);
    let read = sdm_schema::held::read(
        sdm_contract::ops::HeldSchema {
            tables: vec![fixed.table],
            links: vec![],
        },
        &BTreeMap::new(),
        true,
    )
    .unwrap();
    let mut description = read.description;
    super::columns::consistent_native(&mut description, &read.types).unwrap();
    super::columns::kept_native(&description).unwrap();
    assert_eq!(description.tables[0].primary_key, ["a", "b"]);
    assert_eq!(description.tables[0].columns[0].stype, Stype::Categorical);
    // Unlike caller Schema edits, native graph metadata does not release a
    // current assignment merely because it has an earlier caller key mark.
    assert_eq!(description.tables[0].columns[1].stype, Stype::Id);
}

#[test]
fn native_metadata_uses_fixed_types_and_requires_the_actual_new_derived_meaning() {
    let mut fixed = table();
    fixed.table.columns[1].dtype = "string".into();
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    let derived = FrameConstructedColumn {
        name: key.composite.as_ref().unwrap().derived.clone(),
        dtype: Dtype::String,
        stype: Stype::Id,
    };
    assert!(
        source_keys::metadata(FrameTableMetadata {
            key: key.clone(),
            table: fixed.table.clone(),
            derived: None
        })
        .is_err()
    );
    let mut wrong = derived.clone();
    wrong.name.push('x');
    assert!(
        source_keys::metadata(FrameTableMetadata {
            key: key.clone(),
            table: fixed.table.clone(),
            derived: Some(wrong)
        })
        .is_err()
    );
    let value = source_keys::metadata(FrameTableMetadata {
        key,
        table: fixed.table,
        derived: Some(derived),
    })
    .unwrap();
    assert_eq!(value.table.columns[1].dtype, "int64");
    assert_eq!(value.table.columns.last().unwrap().stype, Stype::Id);
}

#[test]
fn native_stype_assignment_refuses_before_later_table_effects_with_a_native_cause_recipe() {
    let mut fixed = table();
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    fixed.table.columns[0].stype = Stype::Numerical;
    let derived = FrameConstructedColumn {
        name: key.composite.as_ref().unwrap().derived.clone(),
        dtype: Dtype::String,
        stype: Stype::Id,
    };
    let error = source_keys::metadata(FrameTableMetadata {
        key,
        table: fixed.table,
        derived: Some(derived),
    })
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::IncompatibleSemanticType);
    assert_eq!(error.fields["column"], Detail::text("a"));
    assert_eq!(
        error.fields["native_key_reason"],
        Detail::text("metadata_validation")
    );
}

#[test]
fn native_role_assignment_precedes_caller_column_stype_order() {
    let mut fixed = table();
    let mut ts = fixed.table.columns[1].clone();
    ts.name = "ts".into(); ts.dtype = "timestamp[us]".into(); ts.stype = Stype::Timestamp;
    fixed.table.time_column = Some("ts".into());
    fixed.table.columns.push(ts);
    fixed.constructed.push(FrameConstructedColumn { name: "ts".into(), dtype: Dtype::Int, stype: Stype::Categorical });
    fixed.table.columns[0].stype = Stype::Numerical;
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    let error = source_keys::metadata(reached_metadata(key, fixed.table)).unwrap_err();
    assert_eq!(error.sentence, Sentence::IncompatibleSemanticType);
    assert_eq!(error.fields["column"], Detail::text("ts"));
    assert_eq!(error.fields["stype"], Detail::text("timestamp"));
    let mut fixed = table();
    fixed.table.primary_key = vec!["b".into()];
    fixed.table.time_column = Some("a".into());
    fixed.constructed[1].dtype = Dtype::Date;
    fixed.constructed[1].stype = Stype::Timestamp;
    let key = source_keys::key(fixed.clone(), fixed.table.primary_key.clone()).unwrap();
    let error = source_keys::metadata(reached_metadata(key, fixed.table)).unwrap_err();
    assert_eq!(error.fields["column"], Detail::text("b"));
    assert_eq!(error.fields["stype"], Detail::text("ID"));
}
