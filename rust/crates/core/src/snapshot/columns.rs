// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_labels::column::{Column, Obj, TimeUnit};
use sdm_labels::executor::{Batch, Frame};
use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use super::{Error, Result, array, count, field, fields, require, text};

fn mapping(value: &Value) -> Result<Vec<(&str, &Value)>> {
    fields(value, &["mapping"])?;
    let mut result = Vec::new();
    for pair in array(field(value, "mapping")?)? {
        let pair = array(pair)?;
        require(pair.len() == 2, "invalid typed column mapping")?;
        fields(&pair[0], &["scalar"])?;
        let key = text(field(&pair[0], "scalar")?)?;
        require(
            !result.iter().any(|(previous, _)| *previous == key),
            "duplicate typed column mapping",
        )?;
        result.push((key, &pair[1]));
    }
    Ok(result)
}

fn list(value: &Value) -> Result<&[Value]> {
    fields(value, &["list"])?;
    array(field(value, "list")?)
}

fn range_index(value: &Value, length: usize) -> Result<()> {
    fields(value, &["index"])?;
    let index = field(value, "index")?;
    fields(
        index,
        &["class", "dtype", "names", "values", "levels", "codes"],
    )?;
    require(
        field(index, "class")? == "RangeIndex"
            && field(index, "dtype")? == "int64"
            && field(index, "names")? == &json!({"list": [{"scalar": null}]})
            && field(index, "levels")?.is_null()
            && field(index, "codes")?.is_null(),
        "unsupported typed column index",
    )?;
    let values = list(field(index, "values")?)?;
    require(
        values.len() == length
            && values
                .iter()
                .enumerate()
                .all(|(position, value)| value == &json!({"scalar": position})),
        "typed column index differs from rows",
    )
}

fn cell(value: &Value) -> Result<Obj> {
    let decoded = super::host::input_value(value)?;
    Ok(match decoded.semantic() {
        PyVal::None => Obj::None,
        PyVal::Na => Obj::Na,
        PyVal::NaT => Obj::NaT,
        PyVal::Bool(value) => Obj::Bool(*value),
        PyVal::Int(value) => {
            Obj::Int(i64::try_from(*value).map_err(|_| Error("wide typed label integer"))?)
        }
        PyVal::Float(value) => Obj::Float(*value),
        PyVal::Str(value) => Obj::Str(value.clone()),
        PyVal::List(_) => Obj::List(list(value)?.iter().map(cell).collect::<Result<_>>()?),
        PyVal::Timestamp(wall) => {
            let pandas = field(value, "pandas_scalar")?;
            let numpy = field(field(pandas, "value")?, "numpy_scalar")?;
            let dtype = text(field(numpy, "dtype")?)?;
            let unit = dtype
                .strip_prefix("<M8[")
                .and_then(|value| value.strip_suffix(']'))
                .and_then(TimeUnit::parse)
                .ok_or(Error("invalid typed label timestamp unit"))?;
            let nanos = wall.epoch_nanos();
            require(
                nanos % i128::from(unit.nanos()) == 0,
                "typed label timestamp loses precision",
            )?;
            Obj::Time(
                unit,
                i64::try_from(nanos / i128::from(unit.nanos()))
                    .map_err(|_| Error("typed label timestamp overflow"))?,
            )
        }
        _ => return Err(Error("unsupported typed label scalar")),
    })
}

fn integer(value: &Obj) -> Result<i64> {
    match value {
        Obj::Int(value) => Ok(*value),
        _ => Err(Error("invalid typed integer column")),
    }
}

fn boolean(value: &Obj) -> Result<bool> {
    match value {
        Obj::Bool(value) => Ok(*value),
        _ => Err(Error("invalid typed boolean column")),
    }
}

fn float(value: &Obj) -> Result<f64> {
    match value {
        Obj::Float(value) => Ok(*value),
        _ => Err(Error("invalid typed float column")),
    }
}

fn series(input: &Value) -> Result<Column> {
    fields(input, &["series"])?;
    let input = field(input, "series")?;
    fields(
        input,
        &[
            "dtype",
            "name",
            "index",
            "attrs",
            "values",
            "categories",
            "ordered",
        ],
    )?;
    require(
        field(input, "attrs")? == &json!({"mapping": []})
            && field(input, "categories")?.is_null()
            && field(input, "ordered")?.is_null(),
        "unsupported typed series metadata",
    )?;
    let values = list(field(input, "values")?)?
        .iter()
        .map(cell)
        .collect::<Result<Vec<_>>>()?;
    range_index(field(input, "index")?, values.len())?;
    let dtype = text(field(input, "dtype")?)?;
    Ok(match dtype {
        "object" => Column::Obj(values),
        "int64" => Column::I64(values.iter().map(integer).collect::<Result<_>>()?),
        "bool" => Column::Bool(values.iter().map(boolean).collect::<Result<_>>()?),
        "float64" => Column::F64(values.iter().map(float).collect::<Result<_>>()?),
        "float32" => Column::F32(
            values
                .iter()
                .map(|value| {
                    let value = float(value)?;
                    require(
                        value.is_nan() || f64::from(value as f32) == value,
                        "typed float32 input loses precision",
                    )?;
                    Ok(value as f32)
                })
                .collect::<Result<_>>()?,
        ),
        "Int64" => Column::MI64(
            values
                .iter()
                .map(|value| {
                    if value.is_na() {
                        Ok(None)
                    } else {
                        integer(value).map(Some)
                    }
                })
                .collect::<Result<_>>()?,
        ),
        "Float64" => Column::MF64(
            values
                .iter()
                .map(|value| {
                    if value.is_na() {
                        Ok(None)
                    } else {
                        float(value).map(Some)
                    }
                })
                .collect::<Result<_>>()?,
        ),
        "boolean" => Column::MBool(
            values
                .iter()
                .map(|value| {
                    if value.is_na() {
                        Ok(None)
                    } else {
                        boolean(value).map(Some)
                    }
                })
                .collect::<Result<_>>()?,
        ),
        "str" => Column::Str(
            values
                .iter()
                .map(|value| match value {
                    value if value.is_na() => Ok(None),
                    Obj::Str(value) => Ok(Some(value.clone())),
                    _ => Err(Error("invalid typed string column")),
                })
                .collect::<Result<_>>()?,
        ),
        dtype => {
            let unit = dtype
                .strip_prefix("datetime64[")
                .and_then(|value| value.strip_suffix(']'))
                .and_then(TimeUnit::parse)
                .ok_or(Error("unsupported typed column dtype"))?;
            Column::Time(
                unit,
                values
                    .iter()
                    .map(|value| match value {
                        value if value.is_na() => Ok(None),
                        Obj::Time(actual, ticks) if *actual == unit => Ok(Some(*ticks)),
                        _ => Err(Error("typed timestamp unit mismatch")),
                    })
                    .collect::<Result<_>>()?,
            )
        }
    })
}

fn frame(input: &Value) -> Result<Frame> {
    fields(input, &["frame"])?;
    let frame = field(input, "frame")?;
    fields(frame, &["columns", "index", "attrs", "series"])?;
    require(
        field(frame, "attrs")? == &json!({"mapping": []}),
        "unsupported typed frame attributes",
    )?;
    let index = field(field(frame, "columns")?, "index")?;
    fields(field(frame, "columns")?, &["index"])?;
    fields(
        index,
        &["class", "dtype", "names", "values", "levels", "codes"],
    )?;
    require(
        field(index, "class")? == "Index"
            && matches!(text(field(index, "dtype")?)?, "str" | "object")
            && field(index, "names")? == &json!({"list": [{"scalar": null}]})
            && field(index, "levels")?.is_null()
            && field(index, "codes")?.is_null(),
        "unsupported typed frame columns",
    )?;
    let names = list(field(index, "values")?)?;
    require(!names.is_empty(), "unsupported empty typed frame")?;
    let values = array(field(frame, "series")?)?;
    require(
        names.len() == values.len(),
        "typed frame column count mismatch",
    )?;
    let mut columns = Vec::new();
    for (name, value) in names.iter().zip(values) {
        fields(name, &["scalar"])?;
        let name = text(field(name, "scalar")?)?;
        require(
            !columns.iter().any(|(previous, _)| previous == name)
                && field(field(value, "series")?, "name")? == &json!({"scalar": name}),
            "typed frame column name mismatch",
        )?;
        let column = series(value)?;
        range_index(field(frame, "index")?, column.len())?;
        columns.push((name.to_owned(), column));
    }
    Ok(Frame { columns })
}

pub(super) fn batch(input: &Value) -> Result<Batch> {
    let mut batch = Batch::default();
    let inputs = mapping(input)?;
    require(inputs.len() == 4, "unexpected typed batch fields")?;
    for (name, value) in inputs {
        match name {
            "feat" => {
                batch.feat = mapping(value)?
                    .into_iter()
                    .map(|(name, value)| Ok((name.to_owned(), frame(value)?)))
                    .collect::<Result<_>>()?;
            }
            "times" => {
                batch.time = mapping(value)?
                    .into_iter()
                    .map(|(name, value)| Ok((name.to_owned(), series(value)?)))
                    .collect::<Result<_>>()?;
            }
            "anchor" => batch.anchor = series(value)?,
            "batches" => {
                for (name, value) in mapping(value)? {
                    fields(value, &["ndarray"])?;
                    let value = field(value, "ndarray")?;
                    fields(value, &["dtype", "descriptor", "shape", "values"])?;
                    let values = list(field(value, "values")?)?;
                    require(
                        field(value, "dtype")? == "<i8"
                            && field(value, "descriptor")?
                                == &json!({"list": [{"tuple": [{"scalar": ""}, {"scalar": "<i8"}]}]})
                            && array(field(value, "shape")?)?.len() == 1
                            && count(&field(value, "shape")?[0])? == values.len(),
                        "invalid typed batch ndarray",
                    )?;
                    batch.batch.push((
                        name.to_owned(),
                        values
                            .iter()
                            .map(|value| integer(&cell(value)?))
                            .collect::<Result<_>>()?,
                    ));
                }
            }
            _ => return Err(Error("unknown typed batch field")),
        }
    }
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Value {
        json!({"series": {
            "dtype": "int64", "name": {"scalar": "value"},
            "index": {"index": {"class": "RangeIndex", "dtype": "int64", "names": {"list": [{"scalar": null}]}, "values": {"list": [{"scalar": 0}]}, "levels": null, "codes": null}},
            "attrs": {"mapping": []}, "values": {"list": [{"scalar": 7}]}, "categories": null, "ordered": null
        }})
    }

    #[test]
    fn typed_column_preserves_computed_cells() {
        let mut value = input();
        assert_eq!(series(&value).unwrap(), Column::I64(vec![7]));
        value["series"]["values"]["list"][0]["scalar"] = json!(9);
        assert_eq!(series(&value).unwrap(), Column::I64(vec![9]));
    }

    #[test]
    fn typed_column_refuses_unknown_tags_and_mismatched_metadata() {
        let original = input();
        let mutations = [
            ("/series/values/list/0/unknown", json!(true)),
            ("/series/index/index/values/list/0/scalar", json!(1)),
            ("/series/index/index/class", json!("Index")),
            ("/series/index/index/values/list", json!([])),
            ("/series/dtype", json!("complex128")),
            (
                "/series/attrs/mapping",
                json!([[{"scalar": "x"}, {"scalar": 1}]]),
            ),
            ("/series/categories", json!({"list": []})),
        ];
        for (path, replacement) in mutations {
            let mut value = original.clone();
            if path.ends_with("/unknown") {
                value["series"]["values"]["list"][0]["unknown"] = replacement;
            } else {
                *value.pointer_mut(path).unwrap() = replacement;
            }
            assert!(series(&value).is_err(), "{path}");
        }
        assert!(batch(&json!({"mapping": []})).is_err());
    }

    #[test]
    fn empty_frame_cannot_bypass_index_validation() {
        let value = json!({"frame": {
            "columns": {"index": {"class": "Index", "dtype": "object", "names": {"list": [{"scalar": null}]}, "values": {"list": []}, "levels": null, "codes": null}},
            "index": {"invalid": true}, "attrs": {"mapping": []}, "series": []
        }});
        assert_eq!(
            frame(&value).unwrap_err().to_string(),
            "unsupported empty typed frame"
        );
    }
}
