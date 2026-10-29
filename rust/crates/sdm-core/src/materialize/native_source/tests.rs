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
