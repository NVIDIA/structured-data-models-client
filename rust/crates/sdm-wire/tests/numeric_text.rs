// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A probability written as text is read as Python's `float` reads it, as
//! the bindings' `_finite` does (T1.28b fix round 1): each text of
//! `numeric-text.json`, recorded from the reference's interpreter, read to
//! the same double or refused alike.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the recorded texts"
)]

use sdm_wire::response::numeric_text;

#[test]
fn text_is_read_as_the_reference_s_float_reads_it() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/numeric-text.json");
    let recorded: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = recorded["cases"].as_array().unwrap();
    assert!(cases.len() >= 40);
    let (mut read, mut refused) = (0, 0);
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let got = numeric_text(text).map(|v| format!("{:016x}", v.to_bits()));
        let want = case["float"].as_str().map(str::to_owned);
        // Python's NaN and this reading's are both NaN; their payloads may
        // differ.
        if want.as_deref() == Some("7ff8000000000000") {
            assert!(numeric_text(text).is_some_and(f64::is_nan), "{text:?}");
        } else {
            assert_eq!(got, want, "{text:?}");
        }
        if want.is_some() {
            read += 1;
        } else {
            refused += 1;
        }
    }
    // Not vacuous: both outcomes are held.
    assert!(
        read >= 15 && refused >= 15,
        "{read} read, {refused} refused"
    );
}

/// The digits every supported Python takes, and no more (T1.28b fix round
/// 2): Kawi's, which Unicode 15.0 added and the reference's Python 3.12
/// takes, are refused, as Python 3.10, the floor, refuses them (register
/// P1B-25); Arabic-Indic and fullwidth digits, in Unicode 13.0, are read.
#[test]
fn only_the_digits_every_supported_python_reads_are_read() {
    assert_eq!(numeric_text("\u{11f50}.\u{11f55}"), None);
    assert_eq!(numeric_text("\u{16ac0}"), None);
    assert_eq!(numeric_text("\u{0660}.\u{0665}"), Some(0.5));
    assert_eq!(numeric_text("\u{ff10}.\u{ff15}"), Some(0.5));
}
