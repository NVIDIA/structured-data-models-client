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
