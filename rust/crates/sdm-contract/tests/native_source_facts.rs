// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Original declarations and observed content are portable value facts.

use sdm_contract::ColumnType;
use sdm_contract::frames::NativeDecimal;
use sdm_contract::frames::materialize::{FrameObservedContent, FramePhysicalDeclaration};
use serde_json::json;

#[test]
fn original_declarations_keep_decimal_width_precision_and_full_scale() {
    for scale in [i32::MIN, -129, 0, 129, i32::MAX] {
        for native in [
            NativeDecimal::Decimal32 {
                precision: 9,
                scale,
            },
            NativeDecimal::Decimal64 {
                precision: 18,
                scale,
            },
            NativeDecimal::Decimal128 {
                precision: 38,
                scale,
            },
            NativeDecimal::Decimal256 {
                precision: 76,
                scale,
            },
        ] {
            let declaration = FramePhysicalDeclaration::Decimal { native };
            let wire = serde_json::to_value(&declaration).unwrap();
            assert_eq!(wire["native"]["scale"], json!(scale));
            assert_eq!(
                serde_json::from_value::<FramePhysicalDeclaration>(wire).unwrap(),
                declaration
            );
        }
    }
    for declaration in [
        FramePhysicalDeclaration::Physical {
            physical: ColumnType::Int64,
        },
        FramePhysicalDeclaration::Physical {
            physical: ColumnType::Utf8,
        },
        FramePhysicalDeclaration::LargeUtf8,
    ] {
        assert_eq!(
            serde_json::from_value::<FramePhysicalDeclaration>(
                serde_json::to_value(&declaration).unwrap()
            )
            .unwrap(),
            declaration
        );
    }
}

#[test]
fn observed_content_is_closed_and_does_not_accept_native_inference_labels() {
    for (fact, wire) in [
        (FrameObservedContent::Text, "text"),
        (FrameObservedContent::Decimal, "decimal"),
        (FrameObservedContent::Temporal, "temporal"),
        (FrameObservedContent::Other, "other"),
    ] {
        assert_eq!(serde_json::to_value(fact).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<FrameObservedContent>(json!(wire)).unwrap(),
            fact
        );
    }
    for name in [
        "string",
        "datetime",
        "datetime64",
        "date",
        "timedelta",
        "empty",
        "mixed",
    ] {
        assert!(serde_json::from_value::<FrameObservedContent>(json!(name)).is_err());
    }
}

#[test]
fn logical_text_alternatives_reject_unknown_or_mixed_storage_vocabulary() {
    use sdm_contract::frames::materialize::FrameLogicalText;
    for malformed in [
        json!({"backing":"python","missing":"null_scalar"}),
        json!({"backing":"offset_utf8","missing":"pandas_na"}),
        json!({"backing":"element_values","missing":"null_scalar","storage":"extra"}),
        json!({"backing":"element_values"}),
    ] {
        assert!(serde_json::from_value::<FrameLogicalText>(malformed).is_err());
    }
}

#[test]
fn completed_scalar_content_and_time_outcomes_are_closed_facts() {
    use sdm_contract::materialize::MaterializeSourceFacts;
    for input in [
        json!({"storage":"column_values","inferred":"temporal","inferred_empty":false}),
        json!({"storage":"column_values","inferred":"other","inferred_empty":true}),
        json!({"storage":"time_conversion","rows":2,"unread":1,"examples":[91,55296,56320,93]}),
    ] {
        let facts = serde_json::from_value::<MaterializeSourceFacts>(input.clone()).unwrap();
        assert_eq!(serde_json::to_value(facts).unwrap(), input);
    }
    for input in [
        json!({"storage":"column_values","inferred":"datetime","inferred_empty":false}),
        json!({"storage":"column_values","inferred":"temporal"}),
        json!({"storage":"column_values","inferred":"other","inferred_empty":null}),
        json!({"storage":"time_conversion","rows":2,"unread":1}),
        json!({"storage":"time_conversion","rows":2,"unread":1,"examples":[1114112]}),
        json!({"storage":"time_conversion","rows":2,"unread":1,"examples":"UTF8 is not lossless native text"}),
        json!({"storage":"time_conversion","rows":2,"unread":1,"examples":"[]","policy":true}),
    ] {
        assert!(serde_json::from_value::<MaterializeSourceFacts>(input).is_err());
    }
}
