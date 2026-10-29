// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use sdm_contract::frames::materialize::{
    FrameLogicalText, FrameNativeColumn, FramePhysicalDeclaration, FrameTextBacking,
    FrameTextMissing,
};

fn source(column: FrameNativeColumn) -> Vec<FrameNativeTable> {
    vec![FrameNativeTable {
        source: "input_users".into(),
        columns: vec![column],
    }]
}

fn logical() -> FrameNativeColumn {
    FrameNativeColumn {
        name: "id".into(),
        constructed: None,
        declared: None,
        logical_text: Some(FrameLogicalText {
            backing: FrameTextBacking::ElementValues,
            missing: FrameTextMissing::NullScalar,
        }),
        strings: Some(StringEncoding::Utf8),
    }
}

#[test]
fn byte_identical_exports_keep_original_backing_missing_and_declaration_identity() {
    let original = logical();
    let mut offset = original.clone();
    offset.logical_text.as_mut().unwrap().backing = FrameTextBacking::OffsetUtf8;
    let mut floating = original.clone();
    floating.logical_text.as_mut().unwrap().missing = FrameTextMissing::IeeeNan;
    let mut fixed = original.clone();
    fixed.logical_text = None;
    fixed.declared = Some(FramePhysicalDeclaration::Physical {
        physical: sdm_contract::ColumnType::Utf8,
    });
    let mut ordinary = original.clone();
    ordinary.logical_text = None;
    let identities = [original.clone(), offset, floating, fixed, ordinary]
        .map(|column| native_identity("identical_exported_rows", &source(column)).unwrap());
    assert_eq!(
        identities.iter().collect::<BTreeSet<_>>().len(),
        identities.len()
    );
    assert_eq!(
        identities[0],
        native_identity("identical_exported_rows", &source(original)).unwrap()
    );
}

#[test]
fn malformed_native_declarations_cannot_bypass_canonical_identity() {
    let owner = super::super::retained_tests::owner();
    let frames = owner.frames();
    let frames = lock(&frames);
    let prepared = owner.prepared();
    let description = &prepared.description;
    let mut physical = FrameNativeColumn {
        name: "id".into(),
        constructed: None,
        declared: Some(FramePhysicalDeclaration::Physical {
            physical: sdm_contract::ColumnType::Int64,
        }),
        logical_text: None,
        strings: None,
    };
    assert!(validate_native(&frames, description, &source(physical.clone())).is_ok());
    physical.logical_text = logical().logical_text;
    assert!(validate_native(&frames, description, &source(physical.clone())).is_err());
    physical.logical_text = None;
    physical.declared = Some(FramePhysicalDeclaration::Physical {
        physical: sdm_contract::ColumnType::Uint64,
    });
    assert!(validate_native(&frames, description, &source(physical.clone())).is_err());
    physical.declared = Some(FramePhysicalDeclaration::Physical {
        physical: sdm_contract::ColumnType::Decimal128 {
            precision: 9,
            scale: 0,
        },
    });
    assert!(validate_native(&frames, description, &source(physical)).is_err());
}

#[test]
fn fixed_float_declarations_bind_exact_width_without_changing_native_error_rules() {
    use arrow_array::{Float32Array, Float64Array};
    use sdm_contract::ColumnType;
    let owner = super::super::retained_tests::owner();
    let prepared = owner.prepared();
    let frames = owner.frames();
    let mut frames = lock(&frames);
    for (physical, array) in [
        (
            ColumnType::Float32,
            Arc::new(Float32Array::from(vec![1.0, -0.0])) as ArrayRef,
        ),
        (
            ColumnType::Float64,
            Arc::new(Float64Array::from(vec![1.0, -0.0])) as ArrayRef,
        ),
    ] {
        let declaration = FramePhysicalDeclaration::Physical {
            physical: physical.clone(),
        };
        assert!(check_declaration(&declaration).is_ok());
        assert_eq!(declared_export(&declaration), *array.data_type());
        let batch = RecordBatch::try_from_iter([("id", array)]).unwrap();
        frames.tables[0].1 = ArrowData::new(batch.schema(), vec![batch]);
        let native = FrameNativeColumn {
            name: "id".into(),
            constructed: None,
            declared: Some(declaration),
            logical_text: None,
            strings: None,
        };
        assert!(validate_native(&frames, &prepared.description, &source(native.clone())).is_ok());
        let mut mismatch = native.clone();
        mismatch.declared = Some(FramePhysicalDeclaration::Physical {
            physical: if physical == ColumnType::Float32 {
                ColumnType::Float64
            } else {
                ColumnType::Float32
            },
        });
        assert_eq!(
            validate_native(&frames, &prepared.description, &source(mismatch))
                .unwrap_err()
                .sentence,
            Sentence::InvalidMessage
        );
        let mut logical = native.clone();
        logical.logical_text = Some(FrameLogicalText {
            backing: FrameTextBacking::ElementValues,
            missing: FrameTextMissing::NullScalar,
        });
        assert!(validate_native(&frames, &prepared.description, &source(logical)).is_err());
        let mut encoded = native.clone();
        encoded.strings = Some(StringEncoding::Utf8);
        assert!(validate_native(&frames, &prepared.description, &source(encoded)).is_err());
        let fixed = native_identity("same_export", &source(native.clone())).unwrap();
        let mut ordinary = native;
        ordinary.declared = None;
        assert_ne!(
            fixed,
            native_identity("same_export", &source(ordinary)).unwrap()
        );
    }
}

#[test]
fn large_offsets_are_accepted_for_logical_text_and_refused_only_for_fixed_declaration() {
    use arrow_array::LargeStringArray;
    let owner = super::super::retained_tests::owner();
    let frames = owner.frames();
    let mut frames = lock(&frames);
    let batch = RecordBatch::try_from_iter([(
        "id",
        Arc::new(LargeStringArray::from(vec![Some("a"), None])) as ArrayRef,
    )])
    .unwrap();
    frames.tables[0].1 = ArrowData::new(batch.schema(), vec![batch]);
    let prepared = owner.prepared();
    let description = &prepared.description;
    assert!(validate_native(&frames, description, &source(logical())).is_ok());
    let mut fixed = logical();
    fixed.logical_text = None;
    fixed.declared = Some(FramePhysicalDeclaration::LargeUtf8);
    let error = validate_native(&frames, description, &source(fixed)).unwrap_err();
    assert_eq!(error.sentence, Sentence::FrameNativeInternal);
    assert_eq!(
        error.fields["native_declaration_reason"],
        Detail::text("fixed_large_utf8")
    );
    assert!(!error.fields.contains_key("native_cause_message"));
}

#[test]
fn original_native_facts_and_owner_survive_rebinding_under_different_attachment_ids() {
    let owner = super::super::retained_tests::owner();
    let first = BoundGraph::take_retained(
        &mut BTreeMap::from([(AttachmentId(5), Attachment::FrameGraph(owner.clone()))]),
        AttachmentId(5),
    )
    .unwrap();
    let source = first.source.clone();
    let facts = first.native.clone();
    assert!(owner.same_owner(&first.retained()));
    drop(first);
    let second = BoundGraph::take_retained(
        &mut BTreeMap::from([(AttachmentId(91), Attachment::FrameGraph(owner.clone()))]),
        AttachmentId(91),
    )
    .unwrap();
    assert!(owner.same_owner(&second.retained()));
    assert_eq!(second.source, source);
    assert_eq!(second.native, facts);
}

#[test]
fn original_decimal_declaration_retains_full_scale_through_zero_scale_export_checks() {
    use sdm_contract::frames::NativeDecimal;
    for scale in [i32::MIN, -129, 0, 129, i32::MAX] {
        for native in [
            NativeDecimal::Decimal32 {
                precision: 9,
                scale,
            },
            NativeDecimal::Decimal64 {
                precision: 9,
                scale,
            },
            NativeDecimal::Decimal128 {
                precision: 9,
                scale,
            },
            NativeDecimal::Decimal256 {
                precision: 9,
                scale,
            },
        ] {
            let declaration = FramePhysicalDeclaration::Decimal {
                native: native.clone(),
            };
            assert!(check_declaration(&declaration).is_ok());
            let expected = if matches!(native, NativeDecimal::Decimal256 { .. }) {
                DataType::Decimal256(9, 0)
            } else {
                DataType::Decimal128(9, 0)
            };
            assert_eq!(declared_export(&declaration), expected);
        }
    }
    for native in [
        NativeDecimal::Decimal32 {
            precision: 10,
            scale: 0,
        },
        NativeDecimal::Decimal64 {
            precision: 19,
            scale: 0,
        },
        NativeDecimal::Decimal128 {
            precision: 0,
            scale: 0,
        },
        NativeDecimal::Decimal256 {
            precision: 77,
            scale: 0,
        },
    ] {
        assert!(check_declaration(&FramePhysicalDeclaration::Decimal { native }).is_err());
    }
}

#[test]
fn fixed_constructed_text_survives_numeric_export_without_bypassing_raw_type_guards() {
    use arrow_array::Float64Array;
    let owner = super::super::retained_tests::owner();
    let frames = owner.frames();
    let mut frames = lock(&frames);
    let batch = RecordBatch::try_from_iter([(
        "id",
        Arc::new(Float64Array::from(vec![1.25, 2.5])) as ArrayRef,
    )])
    .unwrap();
    frames.tables[0].1 = ArrowData::new(batch.schema(), vec![batch]);
    let mut description = owner.prepared().description.clone();
    description.tables[0].primary_key.clear();
    description.tables[0].columns[0].dtype = Dtype::String;
    description.tables[0].columns[0].stype = Stype::Text;
    assert!(frames.validate_text_types(&description).is_err());
    let mut column = FrameNativeColumn {
        name: "id".into(),
        declared: None,
        logical_text: None,
        strings: None,
        constructed: Some(Dtype::String),
    };
    assert!(validate_native(&frames, &description, &source(column.clone())).is_ok());
    assert!(
        frames
            .validate_native_text_types(&description, &source(column.clone()))
            .is_ok()
    );
    let fixed = native_identity("same_export", &source(column.clone())).unwrap();
    column.constructed = None;
    assert_ne!(
        fixed,
        native_identity("same_export", &source(column.clone())).unwrap()
    );
    assert!(
        frames
            .validate_native_text_types(&description, &source(column.clone()))
            .is_err()
    );
    column.constructed = Some(Dtype::Int);
    assert!(validate_native(&frames, &description, &source(column)).is_err());
}

#[test]
fn native_timestamp_constructor_date_fact_matches_both_timestamp_wire_aliases() {
    use arrow_array::TimestampMicrosecondArray;
    let owner = super::super::retained_tests::owner();
    let frames = owner.frames();
    let mut frames = lock(&frames);
    let batch = RecordBatch::try_from_iter([(
        "id",
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2])) as ArrayRef,
    )])
    .unwrap();
    frames.tables[0].1 = ArrowData::new(batch.schema(), vec![batch]);
    let mut description = owner.prepared().description.clone();
    description.tables[0].primary_key.clear();
    description.tables[0].columns[0].stype = Stype::Timestamp;
    let native = source(FrameNativeColumn {
        name: "id".into(),
        constructed: Some(Dtype::Date),
        declared: None,
        logical_text: None,
        strings: None,
    });
    for dtype in [Dtype::Date, Dtype::Time] {
        description.tables[0].columns[0].dtype = dtype;
        assert!(validate_native(&frames, &description, &native).is_ok());
    }
    description.tables[0].columns[0].dtype = Dtype::String;
    assert!(validate_native(&frames, &description, &native).is_err());
}
