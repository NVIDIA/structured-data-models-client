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
