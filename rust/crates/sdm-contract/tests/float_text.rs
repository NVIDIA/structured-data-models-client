// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Float scalars read at their declared width, from the vector file the
//! Python host's codec test also reads (`float_text.json`): each text gives
//! the bits it names, or is refused, finite text that overflows included.

use sdm_contract::Scalar;
use serde_json::Value;

#[test]
fn float_text_reads_at_its_width_as_the_vectors_say() {
    let document: Value =
        serde_json::from_str(include_str!("float_text.json")).expect("the vector file reads");
    assert_eq!(document["format"], "sdm-float-text/1");
    let vectors = document["vectors"].as_array().unwrap();
    assert!(vectors.len() >= 20);
    for vector in vectors {
        let scalar = serde_json::json!({"t": vector["t"], "v": vector["v"]});
        let read = serde_json::from_value::<Scalar>(scalar.clone());
        if vector["refused"] == true {
            assert!(read.is_err(), "{scalar} was read");
            continue;
        }
        let read = read.unwrap_or_else(|e| panic!("{scalar} was refused: {e}"));
        let (bits, nan) = match read {
            Scalar::Float32(value) => (format!("{:08x}", value.to_bits()), value.is_nan()),
            Scalar::Float64(value) => (format!("{:016x}", value.to_bits()), value.is_nan()),
            other => panic!("{scalar} read as {other:?}"),
        };
        if vector["nan"] == true {
            assert!(nan, "{scalar}");
        } else {
            assert_eq!(bits, vector["bits"].as_str().unwrap(), "{scalar}");
        }
    }
}
