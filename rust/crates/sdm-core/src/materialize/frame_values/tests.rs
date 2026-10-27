// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authored before native profile/order and context range consumers.

use super::*;
use sdm_engine::words::{CasingProfile, UnicodeDataVersion};

#[test]
fn every_builtin_database_and_actual_result_storage_selects_core_casing() {
    for (database, version) in [
        ("13.0.0", UnicodeDataVersion::V13_0),
        ("14.0.0", UnicodeDataVersion::V14_0),
        ("15.0.0", UnicodeDataVersion::V15_0),
        ("15.1.0", UnicodeDataVersion::V15_1),
        ("16.0.0", UnicodeDataVersion::V16_0),
        ("17.0.0", UnicodeDataVersion::V17_0),
    ] {
        for target in [
            NativeStringTarget::NumpyUnicode,
            NativeStringTarget::PandasStringPython,
        ] {
            let facts = NativeTextFacts {
                target,
                python_unicode_database: database.to_owned(),
                arrow_simple_identity: None,
            };
            assert_eq!(
                casing_profile(&facts).unwrap(),
                CasingProfile::FullRoot(version)
            );
        }
    }
    let facts = NativeTextFacts {
        target: NativeStringTarget::PandasStringPyarrow,
        python_unicode_database: "17.0.0".to_owned(),
        arrow_simple_identity: Some("iσⰯᲉ꟎".to_owned()),
    };
    assert_eq!(
        casing_profile(&facts).unwrap(),
        CasingProfile::Simple(UnicodeDataVersion::V13_0)
    );
    assert!(
        casing_profile(&NativeTextFacts {
            arrow_simple_identity: None,
            ..facts
        })
        .is_err()
    );
}

#[test]
fn context_ranges_are_exact_core_selected_batches() {
    assert!(check_range(FrameBatchRange { start: 0, stop: 2 }, 3, 2).is_ok());
    assert!(check_range(FrameBatchRange { start: 2, stop: 3 }, 3, 2).is_ok());
    for range in [
        FrameBatchRange { start: 1, stop: 3 },
        FrameBatchRange { start: 0, stop: 1 },
        FrameBatchRange { start: 2, stop: 4 },
        FrameBatchRange { start: 3, stop: 3 },
    ] {
        assert!(check_range(range, 3, 2).is_err());
    }
}

#[test]
fn native_conversion_failure_prefix_identifies_the_actual_next_column() {
    let text = [
        ColumnRef {
            table: "users".into(),
            column: "note".into(),
        },
        ColumnRef {
            table: "orders".into(),
            column: "note".into(),
        },
    ];
    assert!(check_text_order(&text, &[text[0].clone()], Some(&text[1])).is_ok());
    assert!(check_text_order(&text, &[], Some(&text[1])).is_err());
    assert!(check_text_order(&text, &[text[1].clone()], Some(&text[0])).is_err());
    assert!(check_text_order(&text, &[text[0].clone()], None).is_err());
    assert!(check_text_order(&text, &text, None).is_ok());
}
