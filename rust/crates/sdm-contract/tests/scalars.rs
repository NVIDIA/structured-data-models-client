// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The exact scalar encodings, at their edges.
//!
//! `tests/scalar_edges.json` is what these edge values encode to; the test
//! regenerates it (`SDM_CONTRACT_WRITE=1` writes it) and the root Python
//! test `tests/test_rust_contract.py` decodes it with only `json` and
//! `decimal`.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads and, with SDM_CONTRACT_WRITE, rewrites its vector file"
)]

use std::path::{Path, PathBuf};
use std::str::FromStr;

use arrow_buffer::i256;
use sdm_contract::{Scalar, Unit};
use serde_json::{Map, Value, json};

fn edges() -> Vec<(&'static str, Scalar)> {
    let nines = |n: usize| "9".repeat(n);
    vec![
        ("i8_min", Scalar::Int8(i8::MIN)),
        ("i64_min", Scalar::Int64(i64::MIN)),
        ("i64_max", Scalar::Int64(i64::MAX)),
        ("u64_max", Scalar::UInt64(u64::MAX)),
        ("u64_above_i64_max", Scalar::UInt64(i64::MAX as u64 + 2)),
        (
            "d128_38_0_max",
            Scalar::Decimal128 {
                precision: 38,
                scale: 0,
                value: i128::from_str(&nines(38)).unwrap(),
            },
        ),
        (
            "d128_38_0_min",
            Scalar::Decimal128 {
                precision: 38,
                scale: 0,
                value: -i128::from_str(&nines(38)).unwrap(),
            },
        ),
        (
            "d128_38_10",
            Scalar::Decimal128 {
                precision: 38,
                scale: 10,
                value: -12_345_678_901_234_567_890_123_456,
            },
        ),
        (
            "d256_76_0_max",
            Scalar::Decimal256 {
                precision: 76,
                scale: 0,
                value: i256::from_string(&nines(76)).unwrap(),
            },
        ),
        (
            "d256_76_0_min",
            Scalar::Decimal256 {
                precision: 76,
                scale: 0,
                value: i256::from_string(&format!("-{}", nines(76))).unwrap(),
            },
        ),
        (
            "d256_76_38",
            Scalar::Decimal256 {
                precision: 76,
                scale: 38,
                value: i256::from_string(&format!("1{}", "0".repeat(75))).unwrap(),
            },
        ),
        ("f64_neg_zero", Scalar::Float64(-0.0)),
        ("f64_zero", Scalar::Float64(0.0)),
        ("f64_nan", Scalar::Float64(f64::NAN)),
        ("f64_inf", Scalar::Float64(f64::INFINITY)),
        ("f64_neg_inf", Scalar::Float64(f64::NEG_INFINITY)),
        ("f64_tenth", Scalar::Float64(0.1)),
        ("f64_max", Scalar::Float64(f64::MAX)),
        ("f64_min_subnormal", Scalar::Float64(5e-324)),
        ("f32_tenth", Scalar::Float32(0.1)),
        ("f32_neg_zero", Scalar::Float32(-0.0)),
        (
            "ts_i64_min",
            Scalar::Timestamp {
                unit: Unit::Ns,
                zone: None,
                value: i64::MIN,
            },
        ),
        (
            "ts_i64_max",
            Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("America/New_York".into()),
                value: i64::MAX,
            },
        ),
        ("date32_min", Scalar::Date32(i32::MIN)),
        (
            "duration_min",
            Scalar::Duration {
                unit: Unit::Us,
                value: i64::MIN,
            },
        ),
        (
            "time64_last_ns",
            Scalar::Time64 {
                unit: Unit::Ns,
                value: 86_399_999_999_999,
            },
        ),
        ("utf8", Scalar::Utf8("a\u{0}\u{e9}\u{1f600}\"".into())),
        ("binary", Scalar::Binary(vec![0, 255, 1, 254])),
        ("bool", Scalar::Bool(true)),
        ("null", Scalar::Null),
    ]
}

fn vector_file() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/scalar_edges.json")
}

fn encoded() -> String {
    let mut map = Map::new();
    for (name, scalar) in edges() {
        map.insert(name.to_owned(), serde_json::to_value(&scalar).unwrap());
    }
    let mut text =
        serde_json::to_string_pretty(&json!({"format": "sdm-scalar-edges/1", "edges": map}))
            .unwrap();
    text.push('\n');
    text
}

#[test]
fn the_edge_vectors_are_what_the_edges_encode_to() {
    let text = encoded();
    if std::env::var_os("SDM_CONTRACT_WRITE").is_some() {
        std::fs::write(vector_file(), &text).unwrap();
    }
    assert_eq!(std::fs::read_to_string(vector_file()).unwrap(), text);
}

#[test]
fn every_edge_round_trips_exactly() {
    for (name, scalar) in edges() {
        let text = serde_json::to_string(&scalar).unwrap();
        let back: Scalar = serde_json::from_str(&text).unwrap();
        assert_eq!(back, scalar, "{name}: {text}");
        assert_eq!(serde_json::to_string(&back).unwrap(), text, "{name}");
    }
}

#[test]
fn the_encoding_carries_no_json_number() {
    let file: Value = serde_json::from_str(&encoded()).unwrap();
    for (name, value) in file["edges"].as_object().unwrap() {
        let v = &value["v"];
        assert!(
            v.is_string() || v.is_boolean() || v.is_null(),
            "{name}: {v}"
        );
    }
}

#[test]
fn signs_and_nan_survive() {
    let decode = |text: &str| serde_json::from_str::<Scalar>(text).unwrap();
    let Scalar::Float64(z) = decode(r#"{"t":"f64","v":"-0.0"}"#) else {
        panic!()
    };
    assert!(z == 0.0 && z.is_sign_negative());
    assert_ne!(decode(r#"{"t":"f64","v":"-0.0"}"#), Scalar::Float64(0.0));
    let Scalar::Float64(n) = decode(r#"{"t":"f64","v":"NaN"}"#) else {
        panic!()
    };
    assert!(n.is_nan());
    // Another encoder's shortest form reads back to the same value.
    assert_eq!(
        decode(r#"{"t":"f64","v":"1e+300"}"#),
        Scalar::Float64(1e300)
    );
}

#[test]
fn malformed_and_overflowing_values_are_refused() {
    let refused = [
        r#"{"t":"i64","v":"9223372036854775808"}"#,
        r#"{"t":"i64","v":"-9223372036854775809"}"#,
        r#"{"t":"u64","v":"-1"}"#,
        r#"{"t":"u64","v":"18446744073709551616"}"#,
        r#"{"t":"i8","v":"+1"}"#,
        r#"{"t":"i8","v":"01"}"#,
        r#"{"t":"i8","v":"-0"}"#,
        r#"{"t":"i8","v":" 1"}"#,
        r#"{"t":"i8","v":1}"#,
        r#"{"t":"i64","v":"1.0"}"#,
        r#"{"t":"f64","v":"0x1p3"}"#,
        r#"{"t":"f64","v":"Infinity"}"#,
        r#"{"t":"f64","v":"nan"}"#,
        r#"{"t":"f64","v":1.5}"#,
        // Finite text that rounds to infinity is no float's text.
        r#"{"t":"f64","v":"1e400"}"#,
        r#"{"t":"f64","v":"-1e309"}"#,
        r#"{"t":"f32","v":"1e39"}"#,
        r#"{"t":"f32","v":"-3.5e38"}"#,
        r#"{"t":"d128","p":39,"s":0,"v":"1"}"#,
        r#"{"t":"d128","p":0,"s":0,"v":"0"}"#,
        r#"{"t":"d128","p":5,"s":6,"v":"1"}"#,
        r#"{"t":"d128","p":2,"s":0,"v":"100"}"#,
        r#"{"t":"d256","p":77,"s":0,"v":"1"}"#,
        r#"{"t":"d256","p":3,"s":0,"v":"+12"}"#,
        r#"{"t":"binary","v":"not base64!"}"#,
        r#"{"t":"ts","unit":"ms","zone":null,"v":"0"}"#,
        r#"{"t":"ts","unit":"us","zone":"","v":"0"}"#,
        r#"{"t":"i64","v":"1","extra":true}"#,
        r#"{"t":"i128","v":"1"}"#,
        r#"{"v":"1"}"#,
        r#"1"#,
    ];
    for text in refused {
        assert!(
            serde_json::from_str::<Scalar>(text).is_err(),
            "{text} was accepted"
        );
    }
}

#[test]
fn the_widest_finite_floats_are_kept() {
    for (text, want) in [
        (
            r#"{"t":"f64","v":"1.7976931348623157e308"}"#,
            Scalar::Float64(f64::MAX),
        ),
        (
            r#"{"t":"f32","v":"3.4028235e38"}"#,
            Scalar::Float32(f32::MAX),
        ),
        (
            r#"{"t":"f32","v":"-3.4028235e38"}"#,
            Scalar::Float32(f32::MIN),
        ),
        (
            r#"{"t":"f64","v":"-inf"}"#,
            Scalar::Float64(f64::NEG_INFINITY),
        ),
        (r#"{"t":"f32","v":"inf"}"#, Scalar::Float32(f32::INFINITY)),
    ] {
        assert_eq!(
            serde_json::from_str::<Scalar>(text).unwrap(),
            want,
            "{text}"
        );
    }
}

/// Values a `Scalar` can hold that the decoder refuses.
fn unencodable() -> Vec<Scalar> {
    vec![
        Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some(String::new()),
            value: 0,
        },
        Scalar::Decimal128 {
            precision: 0,
            scale: 0,
            value: 0,
        },
        Scalar::Decimal128 {
            precision: 39,
            scale: 0,
            value: 1,
        },
        Scalar::Decimal128 {
            precision: 5,
            scale: 6,
            value: 1,
        },
        Scalar::Decimal128 {
            precision: 2,
            scale: 0,
            value: 100,
        },
        Scalar::Decimal128 {
            precision: 2,
            scale: 0,
            value: -100,
        },
        Scalar::Decimal256 {
            precision: 0,
            scale: 0,
            value: i256::ZERO,
        },
        Scalar::Decimal256 {
            precision: 77,
            scale: 0,
            value: i256::ONE,
        },
        Scalar::Decimal256 {
            precision: 3,
            scale: 0,
            value: i256::from_i128(1000),
        },
    ]
}

#[test]
fn the_encoder_refuses_what_the_decoder_refuses() {
    for scalar in unencodable() {
        assert!(scalar.validate().is_err(), "{scalar:?} validated");
        let err = serde_json::to_string(&scalar).unwrap_err();
        assert!(!err.to_string().is_empty());
    }
    for (_, scalar) in edges() {
        scalar.validate().unwrap();
    }
}
