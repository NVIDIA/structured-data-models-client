// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bounded, property-based fuzz test of the contract's decoders: any
//! input is decoded or refused, never a panic, every value the encoder
//! writes reads back exactly, and a value the decoder would refuse is not
//! written at all.

mod corpus;

use arrow_buffer::i256;
use proptest::prelude::*;
use sdm_contract::{ControlMessage, Scalar, TimestampUnit, Unit};
use serde_json::Value;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 2_000,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn unit() -> impl Strategy<Value = Unit> {
    prop_oneof![Just(Unit::Ns), Just(Unit::Us)]
}

fn timestamp_unit() -> impl Strategy<Value = TimestampUnit> {
    prop_oneof![
        Just(TimestampUnit::S),
        Just(TimestampUnit::Ms),
        Just(TimestampUnit::Us),
        Just(TimestampUnit::Ns)
    ]
}

fn scalar() -> impl Strategy<Value = Scalar> {
    prop_oneof![
        Just(Scalar::Null),
        any::<bool>().prop_map(Scalar::Bool),
        any::<i8>().prop_map(Scalar::Int8),
        any::<i16>().prop_map(Scalar::Int16),
        any::<i32>().prop_map(Scalar::Int32),
        any::<i64>().prop_map(Scalar::Int64),
        any::<u8>().prop_map(Scalar::UInt8),
        any::<u16>().prop_map(Scalar::UInt16),
        any::<u32>().prop_map(Scalar::UInt32),
        any::<u64>().prop_map(Scalar::UInt64),
        any::<u32>().prop_map(|b| Scalar::Float32(f32::from_bits(b))),
        any::<u64>().prop_map(|b| Scalar::Float64(f64::from_bits(b))),
        (1u8..=38, any::<i128>(), -5i8..=38).prop_map(|(p, v, s)| {
            let limit = 10i128.pow(u32::from(p));
            Scalar::Decimal128 {
                precision: p,
                scale: s.min(p as i8),
                value: v % limit,
            }
        }),
        (1u8..=76, any::<[u8; 32]>(), -5i8..=76).prop_map(|(p, bytes, s)| {
            let limit = i256::from_i128(10).wrapping_pow(u32::from(p));
            Scalar::Decimal256 {
                precision: p,
                scale: s.min(p as i8),
                value: i256::from_le_bytes(bytes).wrapping_rem(limit),
            }
        }),
        ".*".prop_map(Scalar::Utf8),
        prop::collection::vec(any::<u8>(), 0..64).prop_map(Scalar::Binary),
        any::<i32>().prop_map(Scalar::Date32),
        (
            timestamp_unit(),
            prop::option::of("[A-Za-z_/]{1,20}"),
            any::<i64>()
        )
            .prop_map(|(unit, zone, value)| Scalar::Timestamp { unit, zone, value }),
        (unit(), any::<i64>()).prop_map(|(unit, value)| Scalar::Duration { unit, value }),
        (unit(), any::<i64>()).prop_map(|(unit, value)| Scalar::Time64 { unit, value }),
    ]
}

/// Scalars the decoder refuses: an empty zone, a decimal precision of 0 or
/// past its type's limit, a scale past the precision, or a value with more
/// digits than the precision.
fn unencodable() -> impl Strategy<Value = Scalar> {
    prop_oneof![
        (timestamp_unit(), any::<i64>()).prop_map(|(unit, value)| Scalar::Timestamp {
            unit,
            zone: Some(String::new()),
            value,
        }),
        (
            prop_oneof![Just(0u8), 39u8..=u8::MAX],
            any::<i8>(),
            any::<i128>()
        )
            .prop_map(|(precision, scale, value)| Scalar::Decimal128 {
                precision,
                scale,
                value,
            }),
        (1u8..=37, any::<i128>(), any::<bool>()).prop_map(|(p, v, negative)| {
            // At least 10^p, so p + 1 digits; below 2 * 10^37, so it fits.
            let floor = 10i128.pow(u32::from(p));
            let value = floor + v.rem_euclid(floor);
            Scalar::Decimal128 {
                precision: p,
                scale: 0,
                value: if negative { -value } else { value },
            }
        }),
        (1u8..=38, 1i8..=10).prop_map(|(p, over)| Scalar::Decimal128 {
            precision: p,
            scale: (p as i8).saturating_add(over),
            value: 0,
        }),
        (prop_oneof![Just(0u8), 77u8..=u8::MAX], any::<i8>()).prop_map(|(precision, scale)| {
            Scalar::Decimal256 {
                precision,
                scale,
                value: i256::ZERO,
            }
        }),
        (1u8..=75).prop_map(|p| Scalar::Decimal256 {
            precision: p,
            scale: 0,
            value: i256::from_i128(10).wrapping_pow(u32::from(p)),
        }),
    ]
}

/// Arbitrary JSON, a few levels deep, with the contract's own keys and tags
/// over-represented so decoding gets past the first field.
fn json() -> impl Strategy<Value = Value> {
    let key = prop_oneof![
        Just("message".to_owned()),
        Just("step".to_owned()),
        Just("t".to_owned()),
        Just("v".to_owned()),
        Just("p".to_owned()),
        Just("s".to_owned()),
        Just("type".to_owned()),
        Just("op".to_owned()),
        "[a-z_]{1,8}",
    ];
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        any::<f64>().prop_map(Value::from),
        prop_oneof![
            Just("start".to_owned()),
            Just("fetch".to_owned()),
            Just("d256".to_owned()),
            Just("ts".to_owned()),
            Just("-0".to_owned()),
            Just("NaN".to_owned()),
            "-?[0-9]{1,80}",
            ".*",
        ]
        .prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 64, 8, move |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..6).prop_map(Value::from),
            prop::collection::btree_map(key.clone(), inner, 0..6)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn every_scalar_round_trips(scalar in scalar()) {
        let text = serde_json::to_string(&scalar).unwrap();
        let back: Scalar = serde_json::from_str(&text).unwrap();
        prop_assert_eq!(&back, &scalar);
        prop_assert_eq!(serde_json::to_string(&back).unwrap(), text);
    }

    #[test]
    fn no_unencodable_scalar_is_written(scalar in unencodable()) {
        prop_assert!(scalar.validate().is_err());
        prop_assert!(serde_json::to_string(&scalar).is_err());
    }

    #[test]
    fn any_text_decodes_or_is_refused(text in ".*") {
        let _ = serde_json::from_str::<ControlMessage>(&text);
        let _ = serde_json::from_str::<Scalar>(&text);
    }

    #[test]
    fn any_json_decodes_or_is_refused(value in json()) {
        let text = value.to_string();
        if let Ok(message) = serde_json::from_str::<ControlMessage>(&text) {
            let again = serde_json::to_string(&message).unwrap();
            prop_assert_eq!(serde_json::from_str::<ControlMessage>(&again).unwrap(), message);
        }
        if let Ok(scalar) = serde_json::from_str::<Scalar>(&text) {
            let again = serde_json::to_string(&scalar).unwrap();
            prop_assert_eq!(serde_json::from_str::<Scalar>(&again).unwrap(), scalar);
        }
    }

    #[test]
    fn a_damaged_scalar_decodes_or_is_refused(
        scalar in scalar(),
        at in any::<prop::sample::Index>(),
        byte in any::<u8>(),
    ) {
        let mut bytes = serde_json::to_vec(&scalar).unwrap();
        let i = at.index(bytes.len());
        bytes[i] = byte;
        if let Ok(back) = serde_json::from_slice::<Scalar>(&bytes) {
            let again = serde_json::to_string(&back).unwrap();
            prop_assert_eq!(serde_json::from_str::<Scalar>(&again).unwrap(), back);
        }
    }

    #[test]
    fn a_damaged_message_decodes_or_is_refused(
        which in any::<prop::sample::Index>(),
        damage in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4),
    ) {
        let corpus = corpus::messages();
        let message = &corpus[which.index(corpus.len())];
        let mut bytes = serde_json::to_vec(message).unwrap();
        for (at, byte) in damage {
            let i = at.index(bytes.len());
            bytes[i] = byte;
        }
        if let Ok(back) = serde_json::from_slice::<ControlMessage>(&bytes) {
            let again = serde_json::to_string(&back).unwrap();
            prop_assert_eq!(serde_json::from_str::<ControlMessage>(&again).unwrap(), back);
        }
    }
}
