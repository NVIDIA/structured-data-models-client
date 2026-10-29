// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use sdm_contract::frames::NativeDecimal;
use sdm_contract::frames::materialize::FramePhysicalDeclaration;
use sdm_contract::materialize::{
    MaterializeScalarMeaning as Meaning, MaterializeSourceFacts as Facts,
};

#[test]
fn raw_cell_nullable_meaning_preserves_original_wide_scalar_constructors() {
    for dtype in [
        Dtype::Byte,
        Dtype::Int,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
    ] {
        assert_eq!(
            scalar_meaning(Some(dtype), &Facts::CellRows {}).unwrap(),
            Meaning::Nullable {
                physical: ColumnType::Int64
            }
        );
    }
    for dtype in [Dtype::Float, Dtype::Float32, Dtype::Float64] {
        assert_eq!(
            scalar_meaning(Some(dtype), &Facts::CellRows {}).unwrap(),
            Meaning::Nullable {
                physical: ColumnType::Float64
            }
        );
    }
    assert_eq!(
        scalar_meaning(Some(Dtype::Bool), &Facts::CellRows {}).unwrap(),
        Meaning::Nullable {
            physical: ColumnType::Bool
        }
    );
    assert_eq!(
        scalar_meaning(Some(Dtype::String), &Facts::CellRows {}).unwrap(),
        Meaning::Nullable {
            physical: ColumnType::Utf8
        }
    );
    for dtype in [
        None,
        Some(Dtype::Time),
        Some(Dtype::Intlist),
        Some(Dtype::Stringlist),
    ] {
        assert_eq!(
            scalar_meaning(dtype, &Facts::CellRows {}).unwrap(),
            Meaning::Original {}
        );
    }
}

#[test]
fn actual_decimal_scale_and_declaration_select_one_native_cast_without_narrowing_scale() {
    for scale in [i32::MIN, -129, -1, 0, 1, 129, i32::MAX] {
        for native in [
            NativeDecimal::Decimal32 {
                precision: 7,
                scale,
            },
            NativeDecimal::Decimal64 {
                precision: 16,
                scale,
            },
            NativeDecimal::Decimal128 {
                precision: 38,
                scale,
            },
            NativeDecimal::Decimal256 {
                precision: 76,
                scale,
            },
        ] {
            let input = Facts::TypedColumns {
                declaration: Some(FramePhysicalDeclaration::Decimal { native }),
            };
            for dtype in [Dtype::Byte, Dtype::Int16, Dtype::Int32, Dtype::Int64] {
                assert_eq!(
                    scalar_meaning(Some(dtype), &input).unwrap(),
                    Meaning::ExactDecimal {
                        physical: ColumnType::Int64
                    }
                );
            }
            for dtype in [Dtype::Float32, Dtype::Float64] {
                assert_eq!(
                    scalar_meaning(Some(dtype), &input).unwrap(),
                    Meaning::ExactDecimal {
                        physical: ColumnType::Float64
                    }
                );
            }
            assert_eq!(
                scalar_meaning(Some(Dtype::String), &input).unwrap(),
                Meaning::Decimal {
                    physical: ColumnType::Utf8
                }
            );
            assert_eq!(
                scalar_meaning(None, &input).unwrap(),
                Meaning::ExactDecimal {
                    physical: if scale == 0 {
                        ColumnType::Int64
                    } else {
                        ColumnType::Float64
                    }
                }
            );
            assert_eq!(
                scalar_meaning(Some(Dtype::Bool), &input).unwrap(),
                Meaning::Original {}
            );
        }
    }
}

#[test]
fn ordinary_typed_scalar_storage_keeps_its_actual_source_array() {
    for physical in [
        ColumnType::Int8,
        ColumnType::Int64,
        ColumnType::Uint64,
        ColumnType::Float32,
        ColumnType::Float64,
        ColumnType::Utf8,
    ] {
        let input = Facts::TypedColumns {
            declaration: Some(FramePhysicalDeclaration::Physical { physical }),
        };
        for dtype in [Dtype::Int64, Dtype::Float64, Dtype::String] {
            assert_eq!(
                scalar_meaning(Some(dtype), &input).unwrap(),
                Meaning::Original {}
            );
        }
    }
}

#[test]
fn native_decimal_descriptor_cannot_alias_an_invalid_or_redundant_declaration() {
    for declaration in [
        FramePhysicalDeclaration::Decimal {
            native: NativeDecimal::Decimal32 {
                precision: 10,
                scale: 0,
            },
        },
        FramePhysicalDeclaration::Decimal {
            native: NativeDecimal::Decimal128 {
                precision: 0,
                scale: 0,
            },
        },
        FramePhysicalDeclaration::Physical {
            physical: ColumnType::Decimal128 {
                precision: 10,
                scale: 0,
            },
        },
    ] {
        assert_eq!(
            scalar_meaning(
                Some(Dtype::Int64),
                &Facts::TypedColumns {
                    declaration: Some(declaration)
                }
            )
            .unwrap_err()
            .code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn typed_timestamp_requires_both_reached_semantics_and_actual_storage() {
    let timestamp = Facts::TypedColumns {
        declaration: Some(FramePhysicalDeclaration::Physical {
            physical: ColumnType::Timestamp {
                unit: sdm_contract::types::TimestampUnit::Us,
                zone: Some("UTC".into()),
            },
        }),
    };
    let expected = Some(Meaning::Nullable {
        physical: ColumnType::Timestamp {
            unit: sdm_contract::types::TimestampUnit::Ns,
            zone: None,
        },
    });
    for dtype in [
        None,
        Some(Dtype::Time),
        Some(Dtype::Date),
        Some(Dtype::String),
    ] {
        assert_eq!(
            timestamp_meaning(dtype, Stype::Timestamp, &timestamp),
            expected
        );
    }
    assert_eq!(
        timestamp_meaning(Some(Dtype::Time), Stype::Categorical, &timestamp),
        expected
    );
    for dtype in [None, Some(Dtype::Date), Some(Dtype::String)] {
        assert_eq!(
            timestamp_meaning(dtype, Stype::Categorical, &timestamp),
            None
        );
    }
    for input in [
        Facts::CellRows {},
        Facts::TypedColumns { declaration: None },
        Facts::TypedColumns {
            declaration: Some(FramePhysicalDeclaration::Physical {
                physical: ColumnType::Utf8,
            }),
        },
    ] {
        assert_eq!(
            timestamp_meaning(Some(Dtype::Time), Stype::Timestamp, &input),
            None
        );
    }
}

#[test]
fn boxed_temporal_content_and_missingness_keep_present_text_outside_native_time() {
    use sdm_contract::frames::materialize::FrameObservedContent as Content;
    for (inferred, inferred_empty, temporal) in [
        (Content::Temporal, false, true),
        (Content::Other, true, true),
        (Content::Text, false, false),
        (Content::Other, false, false),
        (Content::Decimal, false, false),
    ] {
        let facts = Facts::ColumnValues {
            inferred,
            inferred_empty,
        };
        assert_eq!(
            scalar_meaning(Some(Dtype::Time), &facts).unwrap(),
            Meaning::Original {}
        );
        assert_eq!(
            timestamp_meaning(Some(Dtype::Time), Stype::Categorical, &facts).is_some(),
            temporal
        );
        assert_eq!(
            timestamp_meaning(Some(Dtype::String), Stype::Timestamp, &facts).is_some(),
            temporal
        );
        assert_eq!(
            timestamp_meaning(Some(Dtype::String), Stype::Categorical, &facts),
            None
        );
    }
}

#[test]
fn native_time_conversion_notice_is_completed_data_with_existing_warning_policy() {
    let input = Facts::TimeConversion {
        rows: 4,
        unread: 2,
        examples: "[datetime.datetime(3000, 1, 1, 0, 0)]".into(),
    };
    let notice = time_notice(Some(Dtype::Time), Stype::Timestamp, "ts", &input)
        .unwrap()
        .unwrap();
    assert_eq!(notice.code, "TIME_VALUES_UNREAD");
    assert_eq!(notice.sentence, Sentence::NativeTimeValuesUnread);
    assert!(notice.holds_to_table());
    assert_eq!(
        notice.fields,
        std::collections::BTreeMap::from([
            ("column".to_owned(), Detail::text("ts")),
            ("rows".to_owned(), Detail::Scalar(Scalar::UInt64(4))),
            ("unread".to_owned(), Detail::Scalar(Scalar::UInt64(2))),
            (
                "examples_points".to_owned(),
                Detail::List(
                    "[datetime.datetime(3000, 1, 1, 0, 0)]"
                        .chars()
                        .map(|point| Scalar::UInt32(u32::from(point)))
                        .collect()
                )
            ),
        ])
    );
    assert!(time_notice(Some(Dtype::String), Stype::Categorical, "ts", &input).is_err());
    for (rows, unread, examples) in [(0, 1, "[x]"), (1, 2, "[x]"), (2, 0, "[x]"), (2, 1, "")] {
        assert!(
            time_notice(
                Some(Dtype::Time),
                Stype::Timestamp,
                "ts",
                &Facts::TimeConversion {
                    rows,
                    unread,
                    examples: examples.into()
                }
            )
            .is_err()
        );
    }
    assert_eq!(
        time_notice(
            Some(Dtype::Time),
            Stype::Timestamp,
            "ts",
            &Facts::CellRows {}
        )
        .unwrap(),
        None
    );
}

#[test]
fn native_time_notice_preserves_surrogate_units_separately_from_unicode_scalars() {
    use sdm_contract::scalar::unicode::UnicodeText;
    for points in [vec![91, 0xd800, 0xdc00, 93], vec![91, 0x10000, 93]] {
        let input = Facts::TimeConversion {
            rows: 2,
            unread: 1,
            examples: UnicodeText::from_points(points.clone()).unwrap(),
        };
        let wire = serde_json::to_value(&input).unwrap();
        assert_eq!(wire["examples"], serde_json::json!(points));
        let decoded = serde_json::from_value::<Facts>(wire).unwrap();
        let notice = time_notice(Some(Dtype::Time), Stype::Timestamp, "ts", &decoded)
            .unwrap()
            .unwrap();
        assert_eq!(
            notice.fields["examples_points"],
            Detail::List(points.into_iter().map(Scalar::UInt32).collect())
        );
    }
}
