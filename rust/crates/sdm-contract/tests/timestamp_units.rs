// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Timestamp units retain their raw count; duration/time-of-day keep their narrower domain.

use arrow_schema::{DataType, TimeUnit};
use sdm_contract::{ColumnType, Scalar};
use serde_json::json;

#[test]
fn every_timestamp_unit_roundtrips_exact_extreme_counts_and_arrow_types() {
    for (unit, arrow) in [
        ("s", TimeUnit::Second),
        ("ms", TimeUnit::Millisecond),
        ("us", TimeUnit::Microsecond),
        ("ns", TimeUnit::Nanosecond),
    ] {
        for zone in [None, Some("America/Los_Angeles")] {
            let value = json!({"type":"timestamp","unit":unit,"zone":zone});
            let typed: ColumnType = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(&typed).unwrap(), value);
            let expected = DataType::Timestamp(arrow, zone.map(Into::into));
            assert_eq!(typed.to_arrow(), expected);
            assert_eq!(
                ColumnType::from_arrow("observed_at", &expected).unwrap(),
                typed
            );
            for count in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX] {
                let wire = json!({"t":"ts","unit":unit,"zone":zone,"v":count.to_string()});
                let scalar: Scalar = serde_json::from_value(wire.clone()).unwrap();
                assert!(matches!(&scalar, Scalar::Timestamp { value, .. } if *value == count));
                assert_eq!(serde_json::to_value(scalar).unwrap(), wire);
            }
        }
    }
}

#[test]
fn coarse_timestamp_units_do_not_expand_duration_or_time_of_day() {
    for unit in ["s", "ms"] {
        for kind in ["duration", "time64"] {
            assert!(
                serde_json::from_value::<Scalar>(json!({"t":kind,"unit":unit,"v":"1"})).is_err()
            );
            assert!(
                serde_json::from_value::<ColumnType>(json!({"type":kind,"unit":unit})).is_err()
            );
        }
    }
    for unit in ["", "h", "MS", "US", "nanosecond", "microsecond"] {
        assert!(
            serde_json::from_value::<Scalar>(json!({"t":"ts","unit":unit,"zone":null,"v":"1"}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<ColumnType>(
                json!({"type":"timestamp","unit":unit,"zone":null})
            )
            .is_err()
        );
    }
    for unit in ["us", "ns"] {
        for kind in ["duration", "time64"] {
            let scalar = json!({"t":kind,"unit":unit,"v":"-1"});
            let read: Scalar = serde_json::from_value(scalar.clone()).unwrap();
            assert_eq!(serde_json::to_value(read).unwrap(), scalar);
        }
    }
}
