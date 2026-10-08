// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::{ArrayRef, ListArray, new_null_array};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field};
use sdm_schema::pyval::PyVal;

use crate::driver::Raised;

use super::{Dtype, Series};

pub(crate) fn compatible(cells: &[PyVal]) -> bool {
    cells.iter().all(|cell| match cell {
        PyVal::None | PyVal::Na => true,
        PyVal::List(items) => items
            .iter()
            .all(|item| matches!(item, PyVal::Decimal(_) | PyVal::None | PyVal::Na)),
        _ => false,
    })
}

pub(crate) fn series(cells: &[PyVal]) -> Result<Series, Raised> {
    if !cells.iter().any(|cell| matches!(cell, PyVal::List(_))) {
        return Ok(
            Series::new(Dtype::Object, new_null_array(&DataType::Null, cells.len()))
                .with_original_values(cells.to_vec()),
        );
    }
    let items = cells
        .iter()
        .flat_map(|cell| match cell {
            PyVal::List(items) => items.as_slice(),
            _ => &[],
        })
        .cloned()
        .collect::<Vec<_>>();
    let values = if items.iter().any(|item| matches!(item, PyVal::Decimal(_))) {
        decimal_list_items(&items)
    } else {
        new_null_array(&DataType::Null, items.len())
    };
    Ok(list_series(cells, values)?.with_original_values(cells.to_vec()))
}

pub(crate) fn list_series(cells: &[PyVal], values: ArrayRef) -> Result<Series, Raised> {
    let mut offsets = Vec::with_capacity(cells.len() + 1);
    let mut valid = Vec::with_capacity(cells.len());
    let mut count = 0usize;
    offsets.push(0i32);
    for cell in cells {
        if let PyVal::List(items) = cell {
            count += items.len();
            valid.push(true);
        } else {
            valid.push(false);
        }
        offsets.push(
            i32::try_from(count)
                .map_err(|_| Raised::internal("a list column too long for one array"))?,
        );
    }
    let field = Arc::new(Field::new("item", values.data_type().clone(), true));
    let array = ListArray::try_new(
        field,
        OffsetBuffer::new(offsets.into()),
        values,
        Some(NullBuffer::from(valid)),
    )
    .map(|array| Arc::new(array) as ArrayRef)
    .map_err(|error| Raised::internal(format!("cannot build a list column: {error}")))?;
    Ok(Series::new(Dtype::Object, array))
}

pub(crate) fn decimal_list_items(cells: &[PyVal]) -> ArrayRef {
    arrow_decimal_list_items(cells).unwrap_or_else(|_| {
        crate::frame::encoded_objects(
            cells
                .iter()
                .map(|cell| match cell {
                    PyVal::Decimal(text) => serde_json::json!({"$decimal": text}).to_string(),
                    _ => "null".to_string(),
                })
                .collect(),
        )
    })
}

fn arrow_decimal_list_items(cells: &[PyVal]) -> Result<ArrayRef, Raised> {
    let invalid = || Raised::new("ArrowInvalid", "Decimal precision out of range [1, 76]");
    let parsed: Vec<Option<(String, usize)>> = cells
        .iter()
        .map(|cell| {
            let PyVal::Decimal(text) = cell else {
                return Ok(None);
            };
            let (mantissa, exponent) = match text.find(['e', 'E']) {
                Some(position) => (
                    &text[..position],
                    text[position + 1..].parse::<i64>().map_err(|_| invalid())?,
                ),
                None => (text.as_str(), 0),
            };
            let negative = mantissa.starts_with('-');
            let unsigned = mantissa.trim_start_matches(['-', '+']);
            let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
            let digits = format!("{whole}{fraction}");
            if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
                return Err(Raised::new(
                    "ArrowInvalid",
                    format!("The string '{text}' is not a valid decimal128 number"),
                ));
            }
            let mut digits = digits.trim_start_matches('0').to_string();
            if digits.is_empty() {
                digits.push('0');
            }
            let scale = fraction.len() as i64 - exponent;
            if scale < 0 {
                if -scale > 76 {
                    return Err(invalid());
                }
                digits.push_str(&"0".repeat((-scale) as usize));
            }
            if negative {
                digits.insert(0, '-');
            }
            Ok(Some((digits, scale.max(0) as usize)))
        })
        .collect::<Result<_, Raised>>()?;
    let scale = parsed
        .iter()
        .flatten()
        .map(|(_, scale)| *scale)
        .max()
        .unwrap_or(0);
    let precision = parsed
        .iter()
        .flatten()
        .map(|(digits, own_scale)| {
            digits
                .trim_start_matches('-')
                .len()
                .saturating_sub(*own_scale)
                + scale
        })
        .max()
        .unwrap_or(1)
        .max(1);
    if precision > 76 {
        return Err(invalid());
    }
    let values = parsed
        .iter()
        .map(|value| {
            value
                .as_ref()
                .map(|(digits, own_scale)| format!("{digits}{}", "0".repeat(scale - own_scale)))
        })
        .collect::<Vec<_>>();
    if precision <= 38 {
        let values = values
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map(|digits| digits.parse::<i128>().map_err(|_| invalid()))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        arrow_array::Decimal128Array::from(values)
            .with_precision_and_scale(precision as u8, scale as i8)
            .map(|array| Arc::new(array) as ArrayRef)
            .map_err(|_| invalid())
    } else {
        let values = values
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map(|digits| arrow_buffer::i256::from_string(digits).ok_or_else(invalid))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        arrow_array::Decimal256Array::from(values)
            .with_precision_and_scale(precision as u8, scale as i8)
            .map(|array| Arc::new(array) as ArrayRef)
            .map_err(|_| invalid())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int64Array, StringArray};

    use super::*;
    use crate::frame::concat_series;

    fn decimal(text: &str) -> PyVal {
        PyVal::Decimal(text.into())
    }

    #[test]
    fn decimal_batches_reinfer_arrow_and_keep_original_rows_in_both_orders() {
        for (left, right, expected_type) in [
            ("1.0", "123.0", DataType::Decimal128(5, 2)),
            ("1.0", "0.00001", DataType::Decimal128(6, 5)),
            ("1.0", "1E-39", DataType::Decimal256(40, 39)),
            ("1E-39", "1E-40", DataType::Decimal256(41, 40)),
        ] {
            for reverse in [false, true] {
                let left_cells = vec![PyVal::List(vec![decimal(left), decimal("2")])];
                let right_cells = vec![
                    PyVal::List(vec![decimal(right), decimal("2.00")]),
                    PyVal::None,
                    PyVal::List(vec![]),
                    PyVal::List(vec![PyVal::None]),
                ];
                let left = series(&left_cells).unwrap();
                let right = series(&right_cells).unwrap();
                let null = Series::new(Dtype::Object, new_null_array(&DataType::Null, 1));
                let empty = Series::new(Dtype::Object, new_null_array(&DataType::Null, 0));
                let empty_list =
                    list_series(&[PyVal::List(vec![])], new_null_array(&DataType::Null, 0))
                        .unwrap();
                let mut parts = vec![&empty, &left, &null, &empty_list, &right, &left, &empty];
                if reverse {
                    parts.reverse();
                }
                let expected = parts
                    .iter()
                    .flat_map(|part| {
                        part.object_values().unwrap_or_else(|| {
                            (0..part.len())
                                .map(|position| {
                                    sdm_schema::engine::result::value_at(&part.array, position)
                                })
                                .collect()
                        })
                    })
                    .collect::<Vec<_>>();
                let joined = concat_series(&parts).unwrap();
                assert_eq!(joined.object_values(), Some(expected.clone()));
                assert_eq!(joined.len(), expected.len());
                let DataType::List(field) = joined.array.data_type() else {
                    panic!("expected an Arrow list");
                };
                assert_eq!(field.data_type(), &expected_type);
                assert_eq!(
                    joined.array.to_data(),
                    series(&expected).unwrap().array.to_data()
                );
                let selected = joined.take(&[4, 0, 4, 2]).unwrap();
                assert_eq!(
                    selected.object_values(),
                    Some(
                        [4, 0, 4, 2]
                            .map(|position| expected[position].clone())
                            .to_vec()
                    )
                );
            }
        }
    }

    #[test]
    fn decimal_sidecars_join_arrow_only_parts_without_widening_unrelated_objects() {
        let originals = vec![PyVal::List(vec![decimal("1.0"), decimal("2.00")])];
        let with_originals = series(&originals).unwrap();
        let arrow_only = Series::new(Dtype::Object, with_originals.array.clone());
        for parts in [
            [&with_originals, &arrow_only],
            [&arrow_only, &with_originals],
        ] {
            let joined = concat_series(&parts).unwrap();
            let expected = parts
                .iter()
                .map(|part| {
                    part.object_values().map_or_else(
                        || sdm_schema::engine::result::value_at(&part.array, 0),
                        |cells| cells[0].clone(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(joined.object_values(), Some(expected));
        }
        let text = Series::new(Dtype::Object, Arc::new(StringArray::from(vec!["1.0"])));
        let integers = list_series(
            &[PyVal::List(vec![PyVal::Int(1)])],
            Arc::new(Int64Array::from(vec![1])),
        )
        .unwrap();
        let arrow_dtype = Series::new(Dtype::Arrow, with_originals.array.clone());
        for other in [&text, &integers, &arrow_dtype] {
            assert!(concat_series(&[&with_originals, other]).is_err());
            assert!(concat_series(&[other, &with_originals]).is_err());
        }
        let nullable = series(&[originals[0].clone(), PyVal::None]).unwrap();
        let null_sidecar = nullable.take(&[1]).unwrap();
        let null_arrow = Series::new(Dtype::Object, new_null_array(&DataType::Null, 1));
        let joined = concat_series(&[&null_sidecar, &null_arrow]).unwrap();
        assert_eq!(joined.array.data_type(), &DataType::Null);
        assert_eq!(joined.object_values(), Some(vec![PyVal::None, PyVal::None]));
    }

    #[test]
    fn decimal_union_preserves_cells_beyond_arrow_precision() {
        for text in [
            "9".repeat(76),
            "9".repeat(77),
            "NaN".into(),
            "Infinity".into(),
        ] {
            let first = series(&[PyVal::List(vec![decimal("1.0")])]).unwrap();
            let second = series(&[PyVal::List(vec![decimal(&text)])]).unwrap();
            for parts in [[&first, &second], [&second, &first]] {
                let joined = concat_series(&parts).unwrap();
                let expected = parts
                    .iter()
                    .flat_map(|part| part.object_values().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(joined.object_values(), Some(expected));
                let DataType::List(field) = joined.array.data_type() else {
                    panic!("expected an Arrow list");
                };
                assert!(matches!(field.data_type(), DataType::Struct(_)));
            }
        }
    }
}
