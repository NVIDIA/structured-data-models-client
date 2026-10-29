// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed native constructor scalar storage, without native object handles.

use super::object_decimal::{Key, key};
use sdm_contract::frames::{FrameDecimalExponent, FrameScalar, NativeObjectMeaning};
use sdm_contract::ops::Dtype;
use sdm_contract::scalar::unicode::UnicodeText;
use sdm_contract::{CoreError, Sentence};

pub(crate) struct Column {
    pub dtype: Dtype,
    pub values: Vec<Option<Key>>,
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The fixed native tuple layout is composed from the existing FrameScalar
/// Decimal fields. The separate tag retains full exponent/special variants.
pub(crate) fn storage_type() -> arrow_schema::DataType {
    use arrow_schema::{DataType, Field};
    DataType::Struct(
        vec![
            Field::new("kind", DataType::UInt8, false),
            Field::new("negative", DataType::Boolean, true),
            Field::new("digits", DataType::Utf8, true),
            Field::new("exponent", DataType::Int64, true),
            Field::new("special", DataType::UInt8, true),
            Field::new("boolean", DataType::Boolean, true),
            Field::new("integer", DataType::Utf8, true),
            Field::new("float_bits", DataType::UInt64, true),
            Field::new("text", DataType::Binary, true),
        ]
        .into(),
    )
}

pub(crate) fn scalar(
    array: &dyn arrow_array::Array,
    row: usize,
) -> Result<Option<FrameScalar>, CoreError> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int64Type, UInt8Type, UInt64Type};
    use sdm_contract::Scalar;
    if array.data_type() != &storage_type() || row >= array.len() || array.is_null(row) {
        return Err(invalid(
            "a native object has a different typed scalar layout",
        ));
    }
    let fields = array.as_struct();
    let required = |indices: &[usize]| -> Result<(), CoreError> {
        if indices.iter().any(|&at| fields.column(at).is_null(row)) {
            Err(invalid(
                "a native scalar is missing a selected typed component",
            ))
        } else {
            Ok(())
        }
    };
    required(&[0])?;
    Ok(Some(
        match fields.column(0).as_primitive::<UInt8Type>().value(row) {
            0 => FrameScalar::Native,
            1 => return Ok(None),
            2 => {
                required(&[1, 2, 3, 4])?;
                let exponent = match fields.column(4).as_primitive::<UInt8Type>().value(row) {
                    0 => FrameDecimalExponent::Finite {
                        value: fields.column(3).as_primitive::<Int64Type>().value(row),
                    },
                    1 => FrameDecimalExponent::Infinity,
                    2 => FrameDecimalExponent::Nan,
                    3 => FrameDecimalExponent::SignalingNan,
                    _ => return Err(invalid("a native Decimal has an unknown exponent tag")),
                };
                FrameScalar::Decimal {
                    negative: fields.column(1).as_boolean().value(row),
                    digits: fields.column(2).as_string::<i32>().value(row).to_owned(),
                    exponent,
                }
            }
            3 => {
                required(&[5])?;
                FrameScalar::Value {
                    value: Scalar::Bool(fields.column(5).as_boolean().value(row)),
                }
            }
            4 => {
                required(&[6])?;
                let digits = fields.column(6).as_string::<i32>().value(row);
                if let Ok(value) = digits.parse::<i64>() {
                    FrameScalar::Value {
                        value: Scalar::Int64(value),
                    }
                } else if let Ok(value) = digits.parse::<u64>() {
                    FrameScalar::Value {
                        value: Scalar::UInt64(value),
                    }
                } else {
                    FrameScalar::Decimal {
                        negative: digits.starts_with('-'),
                        digits: digits.trim_start_matches('-').to_owned(),
                        exponent: FrameDecimalExponent::Finite { value: 0 },
                    }
                }
            }
            5 => {
                required(&[7])?;
                FrameScalar::Value {
                    value: Scalar::Float64(f64::from_bits(
                        fields.column(7).as_primitive::<UInt64Type>().value(row),
                    )),
                }
            }
            6 => {
                required(&[8])?;
                let value = UnicodeText::from_utf8_surrogatepass(
                    fields.column(8).as_binary::<i32>().value(row).to_vec(),
                )
                .map_err(|_| invalid("native text is not valid Unicode scalar storage"))?;
                FrameScalar::Text { value }
            }
            _ => return Err(invalid("a native scalar has an unknown kind tag")),
        },
    ))
}

fn scalar_key(value: FrameScalar, row: usize) -> Result<Option<Key>, CoreError> {
    use sdm_contract::Scalar;
    match value {
        FrameScalar::Native => Ok(Some(Key::Native { row })),
        FrameScalar::Text { value } => Ok(Some(Key::Text(value))),
        FrameScalar::Decimal { .. } => key(value),
        FrameScalar::Value { value } => match value {
            Scalar::Bool(value) => key(FrameScalar::Decimal {
                negative: false,
                digits: if value { "1" } else { "0" }.into(),
                exponent: FrameDecimalExponent::Finite { value: 0 },
            }),
            Scalar::Int64(value) => key(FrameScalar::Decimal {
                negative: value < 0,
                digits: value.unsigned_abs().to_string(),
                exponent: FrameDecimalExponent::Finite { value: 0 },
            }),
            Scalar::UInt64(value) => key(FrameScalar::Decimal {
                negative: false,
                digits: value.to_string(),
                exponent: FrameDecimalExponent::Finite { value: 0 },
            }),
            Scalar::Float64(value) if value.is_nan() => Ok(None),
            Scalar::Float64(value) if value.is_infinite() => Ok(Some(Key::Infinity {
                negative: value.is_sign_negative(),
            })),
            Scalar::Float64(value) => Ok(Some(Key::Float(if value == 0.0 {
                0
            } else {
                value.to_bits()
            }))),
            _ => Err(invalid(
                "native constructor scalar storage contains a different primitive kind",
            )),
        },
    }
}

pub(crate) fn read(
    batches: &[arrow_array::RecordBatch],
    at: usize,
    meaning: Option<NativeObjectMeaning>,
) -> Result<Column, CoreError> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .columns()
            .get(at)
            .ok_or_else(|| invalid("a native object Decimal field is missing"))?;
        for row in 0..batch.num_rows() {
            values.push(
                scalar(array.as_ref(), row)?
                    .map(|value| scalar_key(value, values.len()))
                    .transpose()?
                    .flatten(),
            );
        }
    }
    let dtype = if let Some(meaning) = meaning {
        if !matches!(
            meaning.dtype,
            Dtype::Int64 | Dtype::Float64 | Dtype::String | Dtype::Bool | Dtype::Date
        ) || (meaning.untyped && meaning.dtype != Dtype::String)
        {
            return Err(invalid(
                "a native object Decimal has an inconsistent inferred meaning",
            ));
        }
        meaning.dtype
    } else if values.iter().flatten().any(Key::fractional) {
        Dtype::Float64
    } else if values.iter().flatten().all(Key::fits_int64) {
        Dtype::Int64
    } else {
        Dtype::String
    };
    Ok(Column { dtype, values })
}
