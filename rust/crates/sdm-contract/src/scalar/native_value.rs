// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete native presentation data, pending coordinated contract35 linking.
//! This is a value encoding, not a model schema, validation policy or workflow.
//! Input-sized trees travel as owned attachments, never inline control data.

#[cfg(test)]
mod tests {
    use super::{NativeObjectKey, NativeValue};
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct Capture {
        records: BTreeMap<String, Record>,
    }

    #[derive(Deserialize)]
    struct Record {
        inspection_native_values: NativeValue,
        result_native_values: NativeValue,
    }

    fn member<'a>(value: &'a NativeValue, name: &str) -> &'a NativeValue {
        let NativeValue::Object { members } = value else {
            panic!("not an object")
        };
        let expected: Vec<_> = name.chars().map(u32::from).collect();
        &members
            .iter()
            .find(|member| match &member.key {
                NativeObjectKey::Unicode { points } => points == &expected,
            })
            .unwrap()
            .value
    }

    fn items(value: &NativeValue) -> &[NativeValue] {
        let NativeValue::Array { items } = value else {
            panic!("not an array")
        };
        items
    }

    fn points(value: &NativeValue) -> &[u32] {
        let NativeValue::Unicode { points } = value else {
            panic!("not Unicode")
        };
        points
    }

    #[test]
    fn complete_frozen_native_data_keeps_pair_scalar_ids_and_character_caps() {
        // Deserialize only these explicitly typed native data fields. The
        // same immutable receipt separately retains its ordinary JSON body,
        // whose standard decoding intentionally combines escaped pairs.
        let capture: Capture = serde_json::from_str(include_str!(
            "../../../../../packages/structured-data-models-client/tests/t44_native_unicode_journeys_reference.json"
        ))
        .unwrap();
        assert_eq!(capture.records.len(), 6);
        for record in capture.records.values() {
            let encoded = serde_json::to_vec(&record.inspection_native_values).unwrap();
            let rebound: NativeValue = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(rebound, record.inspection_native_values);
        }
        let record = &capture.records["pair-key"];
        let rows = items(member(&record.result_native_values, "data"));
        assert_eq!(points(&items(&rows[0])[1]), [0xd83d, 0xde3a]);
        assert_eq!(points(&items(&rows[1])[1]), [0x1f63a]);

        let payload = &items(&capture.records["pair-text"].inspection_native_values)[0];
        let table = member(
            member(member(payload, "predict"), "related_tables"),
            "users",
        );
        let at = items(member(table, "columns"))
            .iter()
            .position(|value| points(value) == ['n' as u32, 'o' as u32, 't' as u32, 'e' as u32])
            .unwrap();
        let rows = items(member(table, "rows"));
        let first = points(&items(&items(&rows[0])[at])[0]);
        let second = points(&items(&items(&rows[1])[at])[0]);
        assert_eq!(first.len(), 300);
        assert_eq!(second.len(), 300);
        assert_eq!(&first[298..], [0xd83d, 0xde3a]);
        assert_eq!(&second[298..], [0x1f63a, u32::from('x')]);
    }

    #[test]
    fn native_value_data_rejects_unknown_fields_and_nonstring_object_keys() {
        for value in [
            r#"{"kind":"unicode","points":[65],"extra":true}"#,
            r#"{"kind":"float","bits":[0]}"#,
            r#"{"kind":"object","members":[{"key":{"kind":"integer","digits":"1"},"value":{"kind":"null"}}]}"#,
        ] {
            assert!(serde_json::from_str::<NativeValue>(value).is_err());
        }
    }
}
