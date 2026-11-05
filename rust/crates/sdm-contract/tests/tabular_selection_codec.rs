// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{SelectTabularRows, TabularKey};
use serde_json::json;

#[test]
fn integer_keys_round_trip_inside_the_real_selection_attachment() {
    for number in [i128::MIN, -1, 0, i128::from(u64::MAX), i128::MAX] {
        let key = json!({"kind":"integer", "value":number.to_string()});
        for mode in [
            json!({"kind":"indices", "source":[key.clone()], "requested":[key.clone()]}),
            json!({"kind":"supplied", "source":[key.clone()], "scoring":[key.clone()]}),
        ] {
            let encoded = json!({"mode":mode,"known":[true],"requested_context":null});
            let parsed: SelectTabularRows = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), encoded);
        }
        let parsed: TabularKey = serde_json::from_value(key).unwrap();
        assert_eq!(parsed, TabularKey::Integer { value: number });
    }
}

#[test]
fn noncanonical_or_rounded_integer_keys_are_not_admitted() {
    for value in [json!(1), json!(1.5), json!("01"), json!("-0"), json!("+1")] {
        assert!(
            serde_json::from_value::<TabularKey>(json!({"kind":"integer","value":value})).is_err()
        );
    }
}
