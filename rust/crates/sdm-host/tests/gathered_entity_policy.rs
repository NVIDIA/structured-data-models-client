// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::scalar::native_value::NativeValue;
use sdm_contract::scalar::unicode::UnicodeText;

#[test]
fn unicode13_fallback_is_selected_after_an_observed_native_number() {
    let integer = NativeValue::Integer { digits: "5".into() };
    for shown in ["\u{16ac5}", "\u{11f55}", "1\u{11f55}"] {
        let original = NativeValue::Unicode {
            points: shown.chars().map(u32::from).collect(),
        };
        let selected =
            sdm_host::gathered_entity_text(&original, &integer, &UnicodeText::from("int64"))
                .unwrap();
        assert_eq!(selected, Some(UnicodeText::from(shown)));
    }
    for shown in ["5", "\u{665}", " +0_05 "] {
        let original = NativeValue::Unicode {
            points: shown.chars().map(u32::from).collect(),
        };
        assert!(
            sdm_host::gathered_entity_text(&original, &integer, &UnicodeText::from("int64"),)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn nonnumeric_native_owners_and_unrelated_declared_families_remain_authoritative() {
    let original = NativeValue::Unicode {
        points: vec![0x11f55],
    };
    assert!(
        sdm_host::gathered_entity_text(&original, &original, &UnicodeText::from("int64"),)
            .unwrap()
            .is_none()
    );
    assert!(
        sdm_host::gathered_entity_text(
            &original,
            &NativeValue::Integer { digits: "5".into() },
            &UnicodeText::from("string"),
        )
        .unwrap()
        .is_none()
    );
    let malformed = NativeValue::Unicode {
        points: vec![0x110000],
    };
    assert!(
        sdm_host::gathered_entity_text(&malformed, &original, &UnicodeText::from("int64"),)
            .is_err()
    );
}
