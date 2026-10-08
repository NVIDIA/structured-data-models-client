// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type};
use arrow_schema::DataType;
use sdm_sampling::frame::{Dtype, Series};
use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use super::{
    Error, Result, array, decode_typed, encode_typed, field, fields, float_hex, require, text,
    typed_observation,
};

use super::exact::Value as Exact;

pub(super) fn replay(input: &Value, facts: Option<&Value>) -> Result<Value> {
    replay_exact(input, facts)?.narrow()
}

pub(super) fn replay_exact(input: &Value, facts: Option<&Value>) -> Result<Exact> {
    let prefix = if let Some(facts) = facts {
        fields(facts, &["host", "primitives"])?;
        require(
            array(field(facts, "primitives")?)?.is_empty(),
            "unused sequence host primitives",
        )?;
        let mut host = field(facts, "host")?.clone();
        let prefix = host
            .as_object_mut()
            .ok_or(Error("invalid sequence host"))?
            .remove("numpy_text_prefix")
            .ok_or(Error("missing NumPy parser capability"))?;
        super::host::descriptor(&host)?;
        prefix
            .as_bool()
            .ok_or(Error("invalid NumPy parser capability"))?
    } else {
        false
    };
    let input = decode_typed(input)?;
    fields(&input, &["calls"])?;
    let mut warnings = Vec::new();
    let mut outputs = Vec::new();
    for call in array(field(&input, "calls")?)? {
        fields(call, &["series", "values", "dtype", "stype"])?;
        require(
            field(call, "series")? == "arrow",
            "sequence replay requires explicit Arrow string constructor",
        )?;
        let values = array(field(call, "values")?)?
            .iter()
            .map(|value| {
                if value.is_null() {
                    Ok(None)
                } else {
                    text(value).map(|value| Some(value.to_owned()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = text(field(call, "dtype")?)?;
        let stype = field(call, "stype")?;
        require(
            matches!(
                (dtype, stype.as_str()),
                ("string", Some("sequence")) | ("intlist" | "stringlist", None)
            ) && (stype.is_null() || stype.is_string()),
            "unsupported original sequence operation",
        )?;
        let kind = sdm_sampling::driver::sql::list_kind(
            sdm_sampling::driver::sql::dtype_of_name(dtype),
            stype.as_str(),
        )
        .ok_or(Error("invalid sequence kind"))?;
        let series = Series::new(
            Dtype::Arrow,
            Arc::new(arrow_array::StringArray::from(values)),
        );
        match sdm_sampling::sequences::to_list_with_policy(&series, kind, prefix, &mut warnings) {
            Ok(output) => {
                let array = output.array.as_ref();
                let values = if let Some(values) = output.object_values() {
                    values
                        .iter()
                        .map(typed_output)
                        .collect::<Result<Vec<_>>>()?
                } else if matches!(array.data_type(), DataType::List(_)) {
                    let lists = array.as_list::<i32>();
                    (0..lists.len()).map(|position| {
                        let row = lists.value(position);
                        match row.data_type() {
                            DataType::Int64 => Ok(json!({"$ndarray": {"dtype": "int64", "values": row.as_primitive::<Int64Type>().values().to_vec()}})),
                            DataType::Float32 => Ok(json!({"$ndarray": {"dtype": "float32", "values": row.as_primitive::<Float32Type>().values().iter().map(|value| floating(f64::from(*value))).collect::<Vec<_>>()}})),
                            _ => Err(Error("unsupported computed sequence array")),
                        }
                    }).collect::<Result<Vec<_>>>()?.into_iter().map(|value| encode_typed(value).map(Exact::from_json)).collect::<Result<Vec<_>>>()?
                } else {
                    (0..array.len())
                        .map(|position| {
                            let value =
                                sdm_schema::engine::result::value_at(&output.array, position);
                            typed_output(&value)
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                let dtype = if output.dtype == Dtype::Arrow {
                    "string[pyarrow]".to_owned()
                } else {
                    output.dtype.tag()
                };
                outputs.push(typed_mapping(vec![(
                    Exact::from_json(json!({"scalar": "output"})),
                    typed_mapping(vec![
                        (
                            Exact::from_json(json!({"scalar": "dtype"})),
                            Exact::from_json(json!({"scalar": dtype})),
                        ),
                        (
                            Exact::from_json(json!({"scalar": "values"})),
                            Exact::Object(vec![("list".into(), Exact::Array(values))]),
                        ),
                    ]),
                )]));
            }
            Err(error) => outputs.push(Exact::from_json(encode_typed(
                json!({"error": {"code": error.class, "message": error.message}}),
            )?)),
        }
    }
    let result = Exact::Object(vec![("list".into(), Exact::Array(outputs))]);
    let mut observation = typed_observation(Value::Null, Value::Null)?;
    observation["observation"]["warnings"] = json!(warnings.into_iter().map(|warning| json!({"class": warning.category, "message": warning.message, "code": null, "details": null})).collect::<Vec<_>>());
    let Exact::Object(mut envelope) = Exact::from_json(observation) else {
        unreachable!()
    };
    let (_, Exact::Object(fields)) = envelope
        .iter_mut()
        .find(|(name, _)| name == "observation")
        .unwrap()
    else {
        unreachable!()
    };
    fields
        .iter_mut()
        .find(|(name, _)| name == "result")
        .unwrap()
        .1 = result;
    Ok(Exact::Object(envelope))
}

fn typed_mapping(pairs: Vec<(Exact, Exact)>) -> Exact {
    Exact::Object(vec![(
        "mapping".into(),
        Exact::Array(
            pairs
                .into_iter()
                .map(|(key, value)| Exact::Array(vec![key, value]))
                .collect(),
        ),
    )])
}

fn typed_output(value: &PyVal) -> Result<Exact> {
    if matches!(value, PyVal::Float(value) if value.is_nan()) {
        return typed_json(&PyVal::None);
    }
    typed_json(value)
}

fn typed_json(value: &PyVal) -> Result<Exact> {
    Ok(match value {
        PyVal::None | PyVal::Na => Exact::from_json(json!({"scalar": null})),
        PyVal::Bool(value) => Exact::from_json(json!({"scalar": value})),
        PyVal::Int(value) => typed_integer(&value.to_string()),
        PyVal::BigInt(value) => typed_integer(value),
        PyVal::Float(value) => Exact::from_json(
            json!({"float": if value.is_nan() { "nan".to_owned() } else if value.is_infinite() { if value.is_sign_negative() { "-inf".to_owned() } else { "inf".to_owned() } } else { float_hex(*value) }}),
        ),
        PyVal::Str(value) => Exact::from_json(json!({"scalar": value})),
        PyVal::List(values) => Exact::Object(vec![(
            "list".into(),
            Exact::Array(values.iter().map(typed_json).collect::<Result<Vec<_>>>()?),
        )]),
        PyVal::Dict(pairs) => typed_mapping(
            pairs
                .iter()
                .map(|(key, value)| Ok((typed_json(key)?, typed_json(value)?)))
                .collect::<Result<Vec<_>>>()?,
        ),
        _ => return Err(Error("non-JSON sequence value")),
    })
}

fn typed_integer(digits: &str) -> Exact {
    Exact::Object(vec![("scalar".into(), Exact::Number(digits.to_owned()))])
}

fn floating(value: f64) -> Value {
    if value.is_nan() {
        json!({"$float": "nan"})
    } else if value.is_infinite() {
        json!({"$float": if value.is_sign_negative() { "-inf" } else { "inf" }})
    } else {
        json!(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_missing_projection_is_not_recursive() {
        assert_eq!(
            typed_output(&PyVal::Float(f64::NAN))
                .unwrap()
                .narrow()
                .unwrap(),
            json!({"scalar": null})
        );
        assert_eq!(
            typed_output(&PyVal::None).unwrap().narrow().unwrap(),
            json!({"scalar": null})
        );
        assert_eq!(
            typed_output(&PyVal::List(vec![PyVal::Float(f64::NAN)]))
                .unwrap()
                .narrow()
                .unwrap(),
            json!({"list": [{"float": "nan"}]})
        );
        for (value, spelling) in [(f64::INFINITY, "inf"), (f64::NEG_INFINITY, "-inf")] {
            assert_eq!(
                typed_output(&PyVal::Float(value))
                    .unwrap()
                    .narrow()
                    .unwrap(),
                json!({"float": spelling})
            );
        }
    }

    #[test]
    fn mapped_json_projects_only_top_level_missing_scalars() {
        for (values, dtype, expected) in [
            (
                json!(["NaN", "[NaN]"]),
                "object",
                json!([{ "scalar": null }, {"list": [{"float": "nan"}]}]),
            ),
            (
                json!(["NaN", "1"]),
                "float64",
                json!([{ "scalar": null }, {"float": "0x1.0000000000000p+0"}]),
            ),
        ] {
            let input = encode_typed(json!({"calls": [{"series": "arrow", "values": values, "dtype": "stringlist", "stype": null}]})).unwrap();
            let observed = replay(&input, None).unwrap();
            assert_eq!(
                observed["observation"]["result"],
                json!({"list": [{"mapping": [[{"scalar": "output"}, {"mapping": [[{"scalar": "dtype"}, {"scalar": dtype}], [{"scalar": "values"}, {"list": expected}]]}]]}]})
            );
        }
    }

    #[test]
    fn mixed_json_replay_emits_computed_typed_values() {
        let input = encode_typed(json!({"calls": [{
            "series": "arrow", "values": ["Infinity", "[1]"],
            "dtype": "stringlist", "stype": null,
        }]}))
        .unwrap();
        let observed = replay(&input, None).unwrap();
        assert_eq!(
            observed["observation"]["result"],
            json!({"list": [{"mapping": [
                [{"scalar": "output"}, {"mapping": [
                    [{"scalar": "dtype"}, {"scalar": "object"}],
                    [{"scalar": "values"}, {"list": [{"float": "inf"}, {"list": [{"scalar": 1}]}]}],
                ]}],
            ]}]})
        );
        assert_eq!(observed["observation"]["warnings"], json!([]));
        assert!(observed["observation"]["error"].is_null());
    }

    #[test]
    fn integer_transport_never_rounds_to_a_float() {
        for digits in ["9223372036854775808", "123456789012345678901234567890"] {
            assert_eq!(
                typed_integer(digits)
                    .field("scalar")
                    .unwrap()
                    .integer()
                    .unwrap(),
                digits
            );
        }
    }
}
