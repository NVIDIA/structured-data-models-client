// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen direct integer decoding and unchanged gathered-width admission.

use sdm_wire::{json, response::PredictionResponse};

#[test]
fn direct_indices_retain_exact_digits_and_native_whole_float_conversion() {
    for (token, expected) in [
        (
            "1606938044258990275541962092341162602522202993782792835301376",
            "1606938044258990275541962092341162602522202993782792835301376",
        ),
        (
            "-1606938044258990275541962092341162602522202993782792835301376",
            "-1606938044258990275541962092341162602522202993782792835301376",
        ),
        ("-0", "0"),
        ("-0.0", "0"),
        ("0.0", "0"),
        ("1e20", "100000000000000000000"),
    ] {
        let body=json::read(format!(r#"{{"id":"server-nonce","model":"kumo-tabular","metadata":{{}},"predictions":[{{"row_index":{token}}}]}}"#).as_bytes()).unwrap();
        let response = PredictionResponse::read_direct(&body).unwrap();
        assert_eq!(response.predictions[0].row_index.digits(), expected);
    }
}

#[test]
fn shared_decoder_preserves_gathered_admission_and_fault_ordinal() {
    let body=json::read(br#"{"id":"server-nonce","model":"kumo-tabular","metadata":{},"predictions":[{"row_index":1606938044258990275541962092341162602522202993782792835301376},{"row_index":1,"probabilities":[]}]}"#).unwrap();
    let gathered = PredictionResponse::read(&body).unwrap_err();
    assert_eq!(
        (gathered.field, gathered.reason, gathered.item),
        ("prediction.row_index", "out_of_range", Some(0))
    );
    // Direct bounds are later than complete generated decoding, so the later
    // malformed object is selected before attribution can inspect the index.
    let direct = PredictionResponse::read_direct(&body).unwrap_err();
    assert_eq!(
        (direct.field, direct.reason, direct.item),
        ("prediction.probabilities", "not_an_object", Some(1))
    );
}

#[test]
fn gathered_whole_float_boundary_precedes_later_fields_without_limiting_direct() {
    let bound = 1u128 << 127;
    // Exact adjacent binary64 integers, independently represented as u128.
    let inside = bound - (1u128 << 74);
    let outside = bound + (1u128 << 75);
    for (token, digits, gathered_admits) in [
        ("0".to_owned(), "0".to_owned(), true),
        (format!("-{bound}"), format!("-{bound}"), true),
        ((bound - 1).to_string(), (bound - 1).to_string(), true),
        (format!("-{}", bound + 1), format!("-{}", bound + 1), false),
        (bound.to_string(), bound.to_string(), false),
        (format!("-{bound}.0"), format!("-{bound}"), false),
        (format!("{bound}.0"), bound.to_string(), false),
        (format!("-{inside}.0"), format!("-{inside}"), true),
        (format!("{inside}.0"), inside.to_string(), true),
        (format!("-{outside}.0"), format!("-{outside}"), false),
        (format!("{outside}.0"), outside.to_string(), false),
    ] {
        for mode in 0..3 {
            let first = if mode == 1 {
                r#", "probabilities":[]"#
            } else {
                ""
            };
            let second = if mode == 2 {
                r#", "probabilities":[]"#
            } else {
                ""
            };
            let body = json::read(format!(r#"{{"id":"r","model":"kumo-relational","metadata":{{}},"predictions":[{{"row_index":{token}{first}}},{{"row_index":1{second}}}]}}"#).as_bytes()).unwrap();
            let gathered = PredictionResponse::read(&body);
            if !gathered_admits {
                let fault = gathered.unwrap_err();
                assert_eq!(
                    (fault.field, fault.reason, fault.item),
                    ("prediction.row_index", "out_of_range", Some(0)),
                    "{token}/{mode}"
                );
            } else if mode == 0 {
                assert_eq!(gathered.unwrap().predictions[0].row_index.digits(), digits);
            } else {
                let fault = gathered.unwrap_err();
                assert_eq!(
                    (fault.field, fault.reason, fault.item),
                    (
                        "prediction.probabilities",
                        "not_an_object",
                        Some(usize::from(mode == 2))
                    ),
                    "{token}/{mode}"
                );
            }
            let direct = PredictionResponse::read_direct(&body);
            if mode == 0 {
                assert_eq!(direct.unwrap().predictions[0].row_index.digits(), digits);
            } else {
                let fault = direct.unwrap_err();
                assert_eq!(
                    (fault.field, fault.reason, fault.item),
                    (
                        "prediction.probabilities",
                        "not_an_object",
                        Some(usize::from(mode == 2))
                    ),
                    "{token}/{mode}"
                );
            }
        }
    }
}
