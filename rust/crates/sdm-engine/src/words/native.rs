// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Unicode words, pending coordinated contract35 linking.
//! Builtin conversion has already completed over the whole selected value.
//! Clipping, casing and word policy remain owned by the engine.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sdm_contract::scalar::native_value::{NativeObjectKey, NativeValue};
    use sdm_contract::scalar::unicode::UnicodeText;
    use serde::Deserialize;

    use super::normalize;
    use crate::words::{CasingProfile, UnicodeDataVersion};

    #[derive(Deserialize)]
    struct Capture {
        records: BTreeMap<String, Record>,
    }

    #[derive(Deserialize)]
    struct Record {
        inspection_native_values: NativeValue,
    }

    fn member<'a>(value: &'a NativeValue, name: &str) -> &'a NativeValue {
        let NativeValue::Object { members } = value else { panic!("object") };
        let expected: Vec<_> = name.chars().map(u32::from).collect();
        &members.iter().find(|member| match &member.key {
            NativeObjectKey::Unicode { points } => points == &expected,
        }).unwrap().value
    }

    fn items(value: &NativeValue) -> &[NativeValue] {
        let NativeValue::Array { items } = value else { panic!("array") };
        items
    }

    fn points(value: &NativeValue) -> &[u32] {
        let NativeValue::Unicode { points } = value else { panic!("text") };
        points
    }

    #[test]
    fn full_native_words_keep_frozen_surrogates_pair_lengths_and_caps() {
        let capture: Capture = serde_json::from_str(include_str!(
            "../../../../../packages/structured-data-models-client/tests/t44_native_unicode_journeys_reference.json"
        )).unwrap();
        for (case, input) in [
            ("text", vec![vec![0xd800, 32, 0x39f, 0x3a3], vec![0xdfff, 32, 0x130]]),
            ("pair-text", vec![
                [vec![65; 298], vec![0xd83d, 0xde3a, 120]].concat(),
                [vec![65; 298], vec![0x1f63a, 120]].concat(),
            ]),
        ] {
            let payload = &items(&capture.records[case].inspection_native_values)[0];
            let users = member(member(member(payload, "predict"), "related_tables"), "users");
            let at = items(member(users, "columns")).iter()
                .position(|value| points(value) == [110, 111, 116, 101]).unwrap();
            let rows = items(member(users, "rows"));
            for (row, input) in rows.iter().zip(input) {
                let expected: Vec<_> = items(&items(row)[at]).iter()
                    .map(|value| points(value).to_vec()).collect();
                for version in [
                    UnicodeDataVersion::V13_0, UnicodeDataVersion::V14_0,
                    UnicodeDataVersion::V15_0, UnicodeDataVersion::V15_1,
                    UnicodeDataVersion::V16_0,
                ] {
                    let input = UnicodeText::from_points(input.iter().copied()).unwrap();
                    let result = normalize(Some(input.as_slice()), CasingProfile::FullRoot(version));
                    assert_eq!(
                        result.iter().map(|value| value.points().collect::<Vec<_>>()).collect::<Vec<_>>(),
                        expected,
                        "{case} {version:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_native_unit_is_an_inert_context_boundary_without_replacement() {
        let input = UnicodeText::from_points([
            0x391, 0xd800, 0x3a3, 32, 0x391, 0x3a3, 0xdfff, 0x391,
        ]).unwrap();
        let result = normalize(
            Some(input.as_slice()), CasingProfile::FullRoot(UnicodeDataVersion::V15_0),
        );
        assert_eq!(result[0].points().collect::<Vec<_>>(), [0x3b1, 0xd800, 0x3c3]);
        assert_eq!(result[1].points().collect::<Vec<_>>(), [0x3b1, 0x3c2, 0xdfff, 0x3b1]);
        assert!(normalize(None, CasingProfile::FullRoot(UnicodeDataVersion::V15_0)).is_empty());
    }
}
