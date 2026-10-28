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
