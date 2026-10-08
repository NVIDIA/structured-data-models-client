// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;

use sdm_request::series::{Dtype, Series};
use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use super::{
    Error, Result, array, count, encode_typed, field, fields, float_hex, object, require, text,
    typed_observation, unhex,
};

fn unit_scale(unit: &str) -> Result<i128> {
    match unit {
        "s" => Ok(1_000_000_000),
        "ms" => Ok(1_000_000),
        "us" => Ok(1_000),
        "ns" => Ok(1),
        _ => Err(Error("unsupported timestamp unit")),
    }
}

fn numpy(input: &Value) -> Result<PyVal> {
    fields(input, &["dtype", "bytes_hex"])?;
    let dtype = text(field(input, "dtype")?)?;
    let bytes = unhex(text(field(input, "bytes_hex")?)?)?;
    Ok(match dtype {
        "<i8" => PyVal::Int(i128::from(i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| Error("invalid numpy integer"))?,
        ))),
        "<f8" => PyVal::Float(f64::from_le_bytes(
            bytes.try_into().map_err(|_| Error("invalid numpy float"))?,
        )),
        "<f4" => PyVal::Float(f64::from(f32::from_le_bytes(
            bytes.try_into().map_err(|_| Error("invalid numpy float"))?,
        ))),
        dtype if dtype.starts_with("<M8[") && dtype.ends_with(']') => {
            let ticks = i64::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| Error("invalid numpy timestamp"))?,
            );
            if ticks == i64::MIN {
                PyVal::NaT
            } else {
                PyVal::Timestamp(sdm_schema::dt::civil::Wall::from_epoch_nanos(
                    i128::from(ticks) * unit_scale(&dtype[4..dtype.len() - 1])?,
                ))
            }
        }
        _ => return Err(Error("unsupported numpy input dtype")),
    })
}

fn scalar_input(input: &Value) -> Result<PyVal> {
    fields(
        input,
        if input.get("numpy_scalar").is_some() {
            &["numpy_scalar"]
        } else {
            &["pandas_scalar"]
        },
    )?;
    if let Some(numpy_input) = input.get("numpy_scalar") {
        return numpy(numpy_input);
    }
    let pandas = field(input, "pandas_scalar")?;
    fields(pandas, &["class", "value", "timezone", "fold"])?;
    require(
        field(pandas, "class")? == "Timestamp"
            && field(pandas, "timezone")?.is_null()
            && field(pandas, "fold")? == 0,
        "unsupported pandas scalar",
    )?;
    let value = field(pandas, "value")?;
    fields(value, &["numpy_scalar"])?;
    let value = numpy(field(value, "numpy_scalar")?)?;
    require(
        matches!(value, PyVal::Timestamp(_) | PyVal::NaT),
        "invalid pandas timestamp storage",
    )?;
    Ok(value)
}

pub(super) fn input_value(input: &Value) -> Result<PyVal> {
    let mapping = object(input)?;
    require(mapping.len() == 1, "invalid host typed input")?;
    let (kind, contents) = mapping
        .iter()
        .next()
        .ok_or(Error("missing host typed input"))?;
    Ok(match kind.as_str() {
        "numpy_scalar" | "pandas_scalar" => PyVal::Held {
            reference: serde_json::to_string(input)
                .map_err(|_| Error("invalid input scalar metadata"))?,
            owner: Some("snapshot-scalar".to_owned()),
            type_name: None,
            value: Box::new(scalar_input(input)?),
        },
        "list" => PyVal::List(
            array(contents)?
                .iter()
                .map(input_value)
                .collect::<Result<_>>()?,
        ),
        "tuple" => PyVal::Tuple(
            array(contents)?
                .iter()
                .map(input_value)
                .collect::<Result<_>>()?,
        ),
        "mapping" => {
            let mut pairs = Vec::new();
            for entry in array(contents)? {
                let entry = array(entry)?;
                require(entry.len() == 2, "invalid host mapping pair")?;
                let key = input_value(&entry[0])?;
                require(
                    !pairs.iter().any(|(previous, _)| previous == &key),
                    "duplicate host mapping key",
                )?;
                pairs.push((key, input_value(&entry[1])?));
            }
            PyVal::Dict(pairs)
        }
        _ => super::typed::value(input)?,
    })
}

struct Tape<'a> {
    entries: &'a [Value],
    position: usize,
}

pub(super) fn descriptor(input: &Value) -> Result<sdm_schema::dt::Host> {
    fields(
        input,
        &[
            "year",
            "today",
            "pads_year",
            "tznames",
            "timestamp_series_units",
        ],
    )?;
    let mut host = sdm_schema::dt::Host::assumed();
    host.year = field(input, "year")?
        .as_i64()
        .ok_or(Error("invalid host year"))?;
    host.today = serde_json::from_value(field(input, "today")?.clone())
        .map_err(|_| Error("invalid host date"))?;
    require(
        (1..=9999).contains(&host.year)
            && sdm_schema::dt::civil::valid_date(host.today.0, host.today.1, host.today.2),
        "invalid host calendar facts",
    )?;
    host.pads_year = field(input, "pads_year")?
        .as_bool()
        .ok_or(Error("invalid host year formatting"))?;
    host.tznames = serde_json::from_value(field(input, "tznames")?.clone())
        .map_err(|_| Error("invalid host zones"))?;
    host.timestamp_series_units = sdm_schema::dt::pandas::TimestampSeriesUnits::from_json(field(
        input,
        "timestamp_series_units",
    )?)
    .map_err(|_| Error("invalid timestamp constructor facts"))?;
    Ok(host)
}

impl<'a> Tape<'a> {
    fn new(input: &'a Value) -> Result<Self> {
        fields(input, &["host", "primitives"])?;
        let host = field(input, "host")?;
        fields(
            host,
            &[
                "year",
                "today",
                "pads_year",
                "tznames",
                "timestamp_series_units",
            ],
        )?;
        require(
            field(host, "year")?.as_i64().is_some() && field(host, "pads_year")?.is_boolean(),
            "invalid host descriptor",
        )?;
        let _: (i64, u32, u32) = serde_json::from_value(field(host, "today")?.clone())
            .map_err(|_| Error("invalid host date"))?;
        let _: Vec<String> = serde_json::from_value(field(host, "tznames")?.clone())
            .map_err(|_| Error("invalid host zones"))?;
        sdm_schema::dt::pandas::TimestampSeriesUnits::from_json(field(
            host,
            "timestamp_series_units",
        )?)
        .map_err(|_| Error("invalid timestamp constructor facts"))?;
        Ok(Self {
            entries: array(field(input, "primitives")?)?,
            position: 0,
        })
    }

    fn take(&mut self, operation: &str, arguments: Value) -> Result<&'a Value> {
        let entry = self
            .entries
            .get(self.position)
            .ok_or(Error("missing host primitive"))?;
        fields(entry, &["operation", "arguments", "outcome"])?;
        require(
            field(entry, "operation")? == operation,
            "host primitive order disagrees",
        )?;
        require(
            field(entry, "arguments")? == &arguments,
            match operation {
                "infer_column" => "computed inference arguments disagree",
                "column_order" => "computed sorting arguments disagree",
                "index_constructor" => "computed index arguments disagree",
                _ => "unsupported computed host primitive",
            },
        )?;
        self.position += 1;
        field(entry, "outcome")
    }

    fn finish(&self) -> Result<()> {
        require(
            self.position == self.entries.len(),
            "unused host primitives",
        )
    }
}

fn scalar(value: &PyVal) -> Result<Value> {
    Ok(match value {
        PyVal::Held {
            reference,
            owner,
            value,
            ..
        } if owner.as_deref() == Some("snapshot-scalar") => {
            let input: Value =
                serde_json::from_str(reference).map_err(|_| Error("invalid scalar metadata"))?;
            require(
                sdm_request::layout::encode_deep(&scalar_input(&input)?)
                    == sdm_request::layout::encode_deep(value),
                "computed scalar disagrees with input metadata",
            )?;
            input
        }
        PyVal::None => json!({"scalar": null}),
        PyVal::Bool(value) => json!({"scalar": value}),
        PyVal::Str(value) => json!({"scalar": value}),
        PyVal::Int(value) => {
            if let Ok(value) = i64::try_from(*value) {
                json!({"scalar": value})
            } else if let Ok(value) = u64::try_from(*value) {
                json!({"scalar": value})
            } else {
                return Err(Error("wide host integer requires exact typed output"));
            }
        }
        PyVal::Float(value) => {
            json!({"float": if value.is_nan() { "nan".to_owned() } else if *value == f64::INFINITY { "inf".to_owned() } else if *value == f64::NEG_INFINITY { "-inf".to_owned() } else { float_hex(*value) }})
        }
        PyVal::List(values) => {
            json!({"list": values.iter().map(scalar).collect::<Result<Vec<_>>>()?})
        }
        PyVal::Tuple(values) => {
            json!({"tuple": values.iter().map(scalar).collect::<Result<Vec<_>>>()?})
        }
        PyVal::Dict(values) => {
            json!({"mapping": values.iter().map(|(key, value)| Ok(json!([scalar(key)?, scalar(value)?]))).collect::<Result<Vec<_>>>()?})
        }
        PyVal::Na | PyVal::NaT | PyVal::Decimal(_) => {
            json!({"fixture_scalar": sdm_request::layout::encode_deep(value)})
        }
        _ => return Err(Error("unsupported host scalar representation")),
    })
}

fn index(class: &str, dtype: &str, values: Value) -> Value {
    json!({"index": {"class": class, "dtype": dtype, "names": {"list": [{"scalar": null}]}, "values": values, "levels": null, "codes": null}})
}

fn range_index(length: usize) -> Value {
    index(
        "RangeIndex",
        "int64",
        json!({"list": (0..length).map(|position| json!({"scalar": position})).collect::<Vec<_>>()}),
    )
}

fn typed_series(name: Value, dtype: &str, cells: &[PyVal]) -> Result<Value> {
    let values = cells.iter().map(|cell| {
        if let PyVal::Timestamp(wall) = cell {
            let unit = dtype.strip_prefix("datetime64[").and_then(|unit| unit.strip_suffix(']')).ok_or(Error("timestamp output lacks representation fact"))?;
            let scale = unit_scale(unit)?;
            require(wall.epoch_nanos() % scale == 0, "timestamp cast loses precision")?;
            let ticks = i64::try_from(wall.epoch_nanos() / scale).map_err(|_| Error("timestamp representation overflow"))?;
            let bytes = ticks.to_le_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
            Ok(json!({"pandas_scalar": {"class": "Timestamp", "value": {"numpy_scalar": {"dtype": format!("<M8[{unit}]"), "bytes_hex": bytes}}, "timezone": null, "fold": 0}}))
        } else { scalar(cell) }
    }).collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"series": {"dtype": dtype, "name": name, "index": range_index(cells.len()), "attrs": {"mapping": []}, "values": {"list": values}, "categories": null, "ordered": null}}),
    )
}

fn frame(result: &sdm_request::result::Result_, input: &Value) -> Result<Value> {
    let tape = RefCell::new(Tape::new(input)?);
    let columns: RefCell<Vec<(String, Vec<PyVal>)>> = RefCell::new(Vec::new());
    let layout = sdm_request::layout::host_layout_with(
        result,
        |cells| {
            let arguments = scalar(&PyVal::List(cells.to_vec()))?;
            let outcome = tape.borrow_mut().take("infer_column", arguments)?;
            fields(outcome, &["status", "dtype", "index_class", "length"])?;
            let dtype = text(field(outcome, "dtype")?)?;
            require(
                field(outcome, "status")? == "success"
                    && field(outcome, "index_class")? == "RangeIndex"
                    && matches!(
                        dtype,
                        "object"
                            | "int64"
                            | "float64"
                            | "bool"
                            | "string"
                            | "str"
                            | "datetime64[s]"
                            | "datetime64[ms]"
                            | "datetime64[us]"
                            | "datetime64[ns]"
                    )
                    && count(field(outcome, "length")?)? == cells.len(),
                "invalid column inference fact",
            )?;
            let values = cells
                .iter()
                .map(|cell| {
                    if dtype == "object" {
                        cell.clone()
                    } else {
                        cell.semantic().clone()
                    }
                })
                .collect();
            let converted = Series::build(None, Dtype::parse(dtype), values);
            let mut held = columns.borrow_mut();
            let reference = held.len().to_string();
            held.push((dtype.to_owned(), converted.cells));
            Ok((reference, cells.len()))
        },
        |reference, stable, ascending, missing_last| {
            let position: usize = reference
                .parse()
                .map_err(|_| Error("invalid computed column reference"))?;
            let held = columns.borrow();
            let (dtype, cells) = held.get(position).ok_or(Error("missing computed column"))?;
            let name = if cells.is_empty() {
                json!({"scalar": null})
            } else {
                json!({"numpy_scalar": {"dtype": "<i8", "bytes_hex": "0000000000000000"}})
            };
            let arguments = json!({"column": typed_series(name, dtype, cells)?, "stable": stable, "ascending": ascending, "missing_last": missing_last});
            let outcome = tape.borrow_mut().take("column_order", arguments)?;
            fields(outcome, &["status", "permutation"])?;
            require(
                field(outcome, "status")? == "success",
                "unsupported host sorting refusal",
            )?;
            array(field(outcome, "permutation")?)?
                .iter()
                .map(count)
                .collect()
        },
        |_| Error("invalid host column layout"),
    )?;
    let entries = array(field(&layout, "columns")?)?;
    let names = entries
        .iter()
        .map(|entry| text(field(entry, "name")?).map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let outcome = tape.borrow_mut().take("index_constructor", json!({"values": encode_typed(json!(names))?, "name": {"scalar": null}, "row_count": result.items.len()}))?;
    fields(outcome, &["status", "dtype", "class", "length"])?;
    let dtype = text(field(outcome, "dtype")?)?;
    let class = text(field(outcome, "class")?)?;
    require(
        field(outcome, "status")? == "success"
            && count(field(outcome, "length")?)? == names.len()
            && if !names.is_empty() {
                class == "Index" && matches!(dtype, "str" | "object")
            } else if result.items.is_empty() {
                class == "RangeIndex" && dtype == "int64"
            } else {
                class == "Index" && dtype == "object"
            },
        "invalid index representation",
    )?;
    let column_index = index(class, dtype, encode_typed(json!(names))?);
    let held = columns.borrow();
    let series = entries
        .iter()
        .map(|entry| {
            let positions: Vec<usize> = array(field(entry, "positions")?)?
                .iter()
                .map(count)
                .collect::<Result<_>>()?;
            let reference: usize = text(field(entry, "reference")?)?
                .parse()
                .map_err(|_| Error("invalid column reference"))?;
            let (dtype, cells) = held
                .get(reference)
                .ok_or(Error("missing computed column"))?;
            let selected = positions
                .iter()
                .map(|position| {
                    cells
                        .get(*position)
                        .cloned()
                        .ok_or(Error("invalid row permutation"))
                })
                .collect::<Result<Vec<_>>>()?;
            typed_series(
                encode_typed(field(entry, "name")?.clone())?,
                dtype,
                &selected,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    tape.borrow().finish()?;
    Ok(
        json!({"frame": {"columns": column_index, "index": range_index(result.items.len()), "attrs": {"mapping": []}, "series": series}}),
    )
}

pub(super) fn replay(name: &str, arguments: &Value, facts: Option<&Value>) -> Result<Value> {
    let input = input_value(arguments)?;
    let result = if name == "request.results" {
        super::typed::result(&input)?
    } else {
        let PyVal::Dict(mapping) = &input else {
            return Err(Error("invalid merge input"));
        };
        let parts = mapping
            .iter()
            .find_map(|(key, value)| {
                matches!(key, PyVal::Str(name) if name == "parts").then_some(value)
            })
            .ok_or(Error("missing merge parts"))?;
        let PyVal::List(parts) = parts else {
            return Err(Error("invalid merge parts"));
        };
        let parts = parts
            .iter()
            .map(super::typed::result)
            .collect::<Result<Vec<_>>>()?;
        match sdm_request::result::Result_::merge(parts) {
            Ok(result) => result,
            Err(error) => {
                require(facts.is_none(), "unused host facts on refused merge")?;
                return typed_observation(
                    json!({"scalar": null}),
                    json!({"class": error.class, "code": error.code, "message": error.message, "details": error.details}),
                );
            }
        }
    };
    let rendered = json!({"tuple": [frame(&result, facts.ok_or(Error("missing request host facts"))?)?, {"scalar": null}]});
    let output = if name == "request.results" {
        rendered
    } else {
        let attributes = json!({"model": result.model, "request_id": result.request_id,
            "metadata": sdm_request::values::encode(&result.metadata), "outputs": result.outputs,
            "unscored_candidates": sdm_request::values::encode(&PyVal::List(result.unscored_candidates)),
            "explanation": result.explanation.as_ref().map(sdm_request::values::encode),
        });
        json!({"mapping": [[{"scalar": "attributes"}, encode_typed(attributes)?], [{"scalar": "frame"}, rendered]]})
    };
    typed_observation(output, Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_metadata_retains_input_units_and_rejects_changed_cells() {
        for unit in ["s", "ms", "us", "ns"] {
            let input = json!({"pandas_scalar": {"class": "Timestamp", "value": {"numpy_scalar": {"dtype": format!("<M8[{unit}]"), "bytes_hex": "0100000000000000"}}, "timezone": null, "fold": 0}});
            let mut decoded = input_value(&input).unwrap();
            assert_eq!(scalar(&decoded).unwrap(), input);
            let PyVal::Held { value, .. } = &mut decoded else {
                panic!("missing input metadata");
            };
            **value = PyVal::None;
            assert!(scalar(&decoded).is_err());
        }
        assert!(
            input_value(
                &json!({"numpy_scalar": {"dtype": ">i8", "bytes_hex": "0000000000000001"}})
            )
            .is_err()
        );
    }

    #[test]
    fn timestamp_output_uses_explicit_units_without_precision_loss() {
        let cells = vec![PyVal::Timestamp(
            sdm_schema::dt::civil::Wall::from_epoch_nanos(1),
        )];
        assert!(typed_series(Value::Null, "datetime64[us]", &cells).is_err());
        let output = typed_series(Value::Null, "datetime64[ns]", &cells).unwrap();
        assert_eq!(
            output["series"]["values"]["list"][0]["pandas_scalar"]["value"]["numpy_scalar"]["dtype"],
            "<M8[ns]"
        );
    }
}
