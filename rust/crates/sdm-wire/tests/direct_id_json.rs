// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The inner string-JSON boundary established by frozen public direct IDs.

use sdm_wire::id_json::{self, Member};

#[test]
fn string_json_keeps_native_root_shapes_and_fallback_distinctions() {
    let nested = format!("{}0{}", "[".repeat(300), "]".repeat(300));
    assert!(id_json::read(&nested, 4300).is_some());
    assert!(id_json::read("\u{FEFF}[5]", 4300).is_none());
    assert_eq!(
        id_json::read("{\"a\":4}", 4300).unwrap().value,
        Member::Object
    );
    assert!(id_json::read("[4,]", 4300).is_none());
    assert!(id_json::read("[04]", 4300).is_none());
    assert_eq!(
        id_json::read("[-0]", 4300).unwrap().members,
        Some(vec![Member::Integer("0".into())])
    );
    for (text, expected) in [
        ("[Infinity]", f64::INFINITY),
        ("[-Infinity]", f64::NEG_INFINITY),
    ] {
        assert_eq!(
            id_json::read(text, 4300).unwrap().members,
            Some(vec![Member::Float(expected)])
        );
    }
    let value = id_json::read("[NaN]", 4300).unwrap();
    assert!(matches!(value.members.unwrap()[0], Member::Float(value) if value.is_nan()));
}

#[test]
fn inner_json_preserves_unicode_points_and_current_decimal_limit() {
    assert_eq!(
        id_json::read(r#"["\ud83d\ude00\ud800"]"#, 4300)
            .unwrap()
            .members,
        Some(vec![Member::Unicode(vec![0x1F600, 0xD800])])
    );
    let decimal = format!("[{}]", "1".repeat(650));
    assert!(id_json::read(&decimal, 640).is_none());
    assert!(id_json::read(&decimal, 4300).is_some());
    // Ignored nested members are still converted by the native JSON reader.
    assert!(id_json::read(&format!("{{\"ignored\":{}}}", "1".repeat(650)), 640).is_none());
}
