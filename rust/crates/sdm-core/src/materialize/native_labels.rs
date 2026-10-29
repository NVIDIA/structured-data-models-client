// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared typed label data and lossless scalar readers for every source host.

use super::*;
use arrow_array::UInt64Array;
use sdm_contract::data::strings::StringEncoding;

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

pub(super) fn group_rows(
    labels: &task::LabelledContext,
    scalar: Option<&DataType>,
) -> Result<RecordBatch, CoreError> {
    let rows = labelled_rows(&labels.keys, &labels.anchors, &labels.labels)?;
    if let Some(scalar) = scalar
        && let Some(list) = rows.column(2).as_list_opt::<i32>()
        && list.values().is_empty()
    {
        let target: ArrayRef = Arc::new(
            arrow_array::ListArray::try_new(
                Arc::new(arrow_schema::Field::new("item", scalar.clone(), false)),
                list.offsets().clone(),
                arrow_array::new_empty_array(scalar),
                list.nulls().cloned(),
            )
            .map_err(|_| protocol("empty native groups do not retain their scalar meaning"))?,
        );
        return RecordBatch::try_from_iter([
            ("entity", Arc::clone(rows.column(0))),
            ("anchor", Arc::clone(rows.column(1))),
            ("target", target),
        ])
        .map_err(|_| protocol("empty native groups do not preserve their completed scalar type"));
    }
    Ok(rows)
}

/// Validate complete group identities while leaving native values authoritative.
pub(super) fn restored(
    before: &task::LabelledContext,
    expected: &RecordBatch,
    rows: &RecordBatch,
    expected_encoding: Option<sdm_contract::data::strings::StringEncoding>,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
) -> Result<labels::Labels, CoreError> {
    let scalar = expected
        .column(2)
        .as_list_opt::<i32>()
        .map(|list| list.value_type());
    restored_with_meaning(
        before,
        expected,
        rows,
        expected_encoding,
        encoding,
        scalar.as_ref(),
    )
}

/// Native group identities and mathematical group masks are separate products.
pub(super) fn restored_with_meaning(
    before: &task::LabelledContext,
    expected: &RecordBatch,
    rows: &RecordBatch,
    expected_encoding: Option<StringEncoding>,
    encoding: Option<StringEncoding>,
    mathematical_scalar: Option<&DataType>,
) -> Result<labels::Labels, CoreError> {
    if rows.num_columns() != 3
        || rows
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>()
            != ["entity", "anchor", "target"]
        || rows.num_rows() != expected.num_rows()
        || rows.column(2).null_count() != 0
        || rows.column(0).to_data() != expected.column(0).to_data()
        || rows.column(1).data_type()
            != &DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None)
        || times_ns(rows.column(1), "anchor")? != times_ns(expected.column(1), "anchor")?
    {
        return Err(protocol(
            "restored native groups do not retain their complete ordered entity and anchor identities",
        ));
    }
    let exact_decimal = expected_encoding.is_none()
        && encoding.is_none()
        && matches!(mathematical_scalar, Some(DataType::Decimal128(_, _)));
    let numeric_completion = rows.column(2).as_list_opt::<i32>().is_some_and(|list| {
        matches!(
            list.value_type(),
            DataType::Int64 | DataType::UInt64 | DataType::Float32 | DataType::Decimal128(_, _)
        )
    });
    let logical_text = encoding.is_some() && expected_encoding.is_some();
    if !(logical_text || exact_decimal && numeric_completion)
        && (encoding != expected_encoding
            || rows.column(2).data_type() != expected.column(2).data_type())
    {
        return Err(protocol(
            "restored native groups do not retain their actual source scalar family and encoding",
        ));
    }
    let actual = read_link_labels(rows.column(2), encoding)?;
    // P2B-1 keeps exact class/key digits through the actual native restoration.
    // Its completed rows remain retained by the group owner for later reads.
    Ok(if exact_decimal {
        before.labels.clone()
    } else {
        actual
    })
}

pub(super) fn labelled_rows(
    keys: &ArrayRef,
    anchors: &[i64],
    labels: &labels::Labels,
) -> Result<RecordBatch, CoreError> {
    use arrow_array::{BooleanArray, Decimal128Array, Float32Array, Int64Array, ListArray};
    use sdm_engine::labels::LinkValue;
    let labels::Labels::Lists(lists) = labels else {
        return task::labelled_rows(keys, anchors, labels);
    };
    let mut offsets = vec![0_i32];
    let mut flat = Vec::new();
    for list in lists {
        flat.extend(list.iter());
        offsets.push(i32::try_from(flat.len()).map_err(|_| {
            internal("link label offsets exceed the declared Arrow representation")
        })?);
    }
    let mixed = || protocol("typed link labels mix incompatible native value families");
    let values: ArrayRef = match flat.first() {
        None | Some(LinkValue::Int(_)) => Arc::new(Int64Array::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Int(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::UInt(_)) => Arc::new(UInt64Array::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::UInt(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Float(_)) => Arc::new(Float32Array::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Float(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Bool(_)) => Arc::new(BooleanArray::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Bool(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Text(_)) => {
            sdm_engine::strings::encode(
                &flat
                    .iter()
                    .map(|value| match value {
                        LinkValue::Text(value) => Ok(value.clone()),
                        _ => Err(mixed()),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .0
        }
        Some(LinkValue::Decimal(_, scale)) => {
            let scale = *scale;
            let values = flat
                .iter()
                .map(|value| match value {
                    LinkValue::Decimal(value, observed) if *observed == scale => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(38, scale)
                    .map_err(|_| mixed())?,
            )
        }
    };
    let target: ArrayRef = Arc::new(
        ListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                values.data_type().clone(),
                false,
            )),
            arrow_buffer::OffsetBuffer::new(offsets.into()),
            values,
            None,
        )
        .map_err(|_| mixed())?,
    );
    RecordBatch::try_from_iter([
        ("entity", Arc::clone(keys)),
        (
            "anchor",
            Arc::new(TimestampNanosecondArray::from(anchors.to_vec())) as ArrayRef,
        ),
        ("target", target),
    ])
    .map_err(|_| protocol("typed link labels do not align with their entity and anchor rows"))
}

/// Text representation belongs to the same typed labels, not to physical
/// Binary inference or a separately replayed host conversion.
pub(super) fn label_encoding(
    labels: &labels::Labels,
) -> Option<sdm_contract::data::strings::StringEncoding> {
    use sdm_contract::data::strings::StringEncoding;
    let exceptional = match labels {
        labels::Labels::Text(values) => values.iter().any(|value| value.utf8().is_none()),
        labels::Labels::Lists(values)
            if values
                .iter()
                .flatten()
                .any(|value| matches!(value, labels::LinkValue::Text(_))) =>
        {
            values.iter().flatten().any(
                |value| matches!(value, labels::LinkValue::Text(text) if text.utf8().is_none()),
            )
        }
        _ => return None,
    };
    Some(if exceptional {
        StringEncoding::Utf8Surrogatepass
    } else {
        StringEncoding::Utf8
    })
}

pub(super) fn read_link_labels(
    target: &ArrayRef,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
) -> Result<labels::Labels, CoreError> {
    use sdm_engine::labels::LinkValue;
    let target = target
        .as_list_opt::<i32>()
        .ok_or_else(|| protocol("typed frame link labels are not a declared Arrow list"))?;
    let lists = (0..target.len())
        .map(|row| {
            let values = target.value(row);
            if values.null_count() != 0 {
                return Err(protocol(
                    "typed frame link labels contain missing child values",
                ));
            }
            Ok(match values.data_type() {
                _ if encoding.is_some() => sdm_engine::strings::owned(&values, encoding.unwrap())
                    .map_err(|_| protocol("frame link text does not match its explicit encoding"))?
                    .into_iter()
                    .map(|value| {
                        value
                            .map(LinkValue::Text)
                            .ok_or_else(|| protocol("frame link text contains missing values"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                DataType::Int64 => values
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied()
                    .map(LinkValue::Int)
                    .collect(),
                DataType::UInt64 => values
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .map(LinkValue::UInt)
                    .collect(),
                DataType::Float32 => values
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .copied()
                    .map(LinkValue::Float)
                    .collect(),
                DataType::Boolean => values
                    .as_boolean()
                    .values()
                    .iter()
                    .map(LinkValue::Bool)
                    .collect(),
                DataType::Utf8 => values
                    .as_string::<i32>()
                    .iter()
                    .map(|value| LinkValue::Text(value.unwrap().into()))
                    .collect(),
                DataType::LargeUtf8 => values
                    .as_string::<i64>()
                    .iter()
                    .map(|value| LinkValue::Text(value.unwrap().into()))
                    .collect(),
                DataType::Decimal128(_, scale) => values
                    .as_primitive::<Decimal128Type>()
                    .values()
                    .iter()
                    .map(|&value| LinkValue::Decimal(value, *scale))
                    .collect(),
                _ => {
                    return Err(protocol(
                        "typed frame link labels contain an incompatible value family",
                    ));
                }
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(labels::Labels::Lists(lists))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_decimal_restoration_preserves_exact_mathematical_lists_and_full_identity() {
        let labels = labels::Labels::Lists(vec![
            vec![labels::LinkValue::Decimal(9_007_199_254_740_993, 3)],
            vec![],
        ]);
        let before = task::LabelledContext {
            keys: Arc::new(arrow_array::Int64Array::from(vec![1, 2])),
            anchors: vec![0, 0],
            labels: labels.clone(),
        };
        let expected = group_rows(&before, Some(&DataType::Decimal128(38, 3))).unwrap();
        let native = task::LabelledContext {
            keys: Arc::clone(&before.keys),
            anchors: before.anchors.clone(),
            labels: labels::Labels::Lists(vec![vec![labels::LinkValue::Float(1.0)], vec![]]),
        };
        let actual = labelled_rows(&native.keys, &native.anchors, &native.labels).unwrap();
        assert_eq!(
            restored(&before, &expected, &actual, None, None).unwrap(),
            labels
        );
        let short = actual.slice(0, 1);
        assert!(restored(&before, &expected, &short, None, None).is_err());
        let text = task::LabelledContext {
            labels: labels::Labels::Lists(vec![
                vec![labels::LinkValue::Text("wrong".into())],
                vec![],
            ]),
            ..native
        };
        let text_rows = labelled_rows(&text.keys, &text.anchors, &text.labels).unwrap();
        assert!(
            restored(
                &before,
                &expected,
                &text_rows,
                None,
                Some(StringEncoding::Utf8)
            )
            .is_err()
        );
    }

    #[test]
    fn no_groups_and_all_empty_groups_keep_the_bound_decimal_scalar() {
        for count in [0, 2] {
            let before = task::LabelledContext {
                keys: Arc::new(arrow_array::Int64Array::from(vec![1; count])),
                anchors: vec![0; count],
                labels: labels::Labels::Lists(vec![vec![]; count]),
            };
            let expected = group_rows(&before, Some(&DataType::Decimal128(38, 3))).unwrap();
            let list = expected.column(2).as_list::<i32>();
            assert_eq!(list.value_type(), DataType::Decimal128(38, 3));
            assert_eq!(list.len(), count);
            assert_eq!(
                restored(&before, &expected, &expected, None, None).unwrap(),
                before.labels
            );
        }
    }

    #[test]
    fn native_and_mathematical_group_masks_have_independent_ordered_identities() {
        let mathematical = task::LabelledContext {
            keys: Arc::new(arrow_array::Int64Array::from(vec![1])),
            anchors: vec![3],
            labels: labels::Labels::Lists(vec![vec![labels::LinkValue::Decimal(1001, 3)]]),
        };
        let native = task::LabelledContext {
            keys: Arc::new(arrow_array::Int64Array::from(vec![2])),
            anchors: vec![4],
            labels: labels::Labels::Lists(vec![vec![labels::LinkValue::Float(1.0)]]),
        };
        let expected = group_rows(&native, Some(&DataType::Float32)).unwrap();
        assert_eq!(
            restored_with_meaning(
                &mathematical,
                &expected,
                &expected,
                None,
                None,
                Some(&DataType::Decimal128(38, 3))
            )
            .unwrap(),
            mathematical.labels
        );
        let wrong = task::LabelledContext {
            keys: Arc::new(arrow_array::Int64Array::from(vec![3])),
            ..native
        };
        let wrong = group_rows(&wrong, Some(&DataType::Float32)).unwrap();
        assert!(
            restored_with_meaning(
                &mathematical,
                &expected,
                &wrong,
                None,
                None,
                Some(&DataType::Decimal128(38, 3))
            )
            .is_err()
        );
    }
}
