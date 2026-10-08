// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::{BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array};
use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use crate::frame::{Dtype, Series, encoded_objects};

pub(super) fn parse(text: &str) -> Option<PyVal> {
    let mut position = super::skip_json_space(text.as_bytes(), 0);
    let value = read(text, &mut position, 0)?;
    (super::skip_json_space(text.as_bytes(), position) == text.len()).then_some(value)
}

fn read(text: &str, position: &mut usize, depth: usize) -> Option<PyVal> {
    if depth > super::JSON_DEPTH {
        return None;
    }
    let bytes = text.as_bytes();
    *position = super::skip_json_space(bytes, *position);
    let start = *position;
    match bytes.get(start)? {
        b'"' => {
            super::json_string(bytes, position).then_some(())?;
            serde_json::from_str(&text[start..*position])
                .ok()
                .map(PyVal::Str)
        }
        b'[' | b'{' => {
            let mapping = bytes[start] == b'{';
            let close = if mapping { b'}' } else { b']' };
            *position = super::skip_json_space(bytes, start + 1);
            let mut items = Vec::new();
            let mut pairs = Vec::new();
            if bytes.get(*position) != Some(&close) {
                loop {
                    let key = if mapping {
                        let PyVal::Str(key) = read(text, position, depth + 1)? else {
                            return None;
                        };
                        *position = super::skip_json_space(bytes, *position);
                        if bytes.get(*position) != Some(&b':') {
                            return None;
                        }
                        *position += 1;
                        Some(PyVal::Str(key))
                    } else {
                        None
                    };
                    let value = read(text, position, depth + 1)?;
                    if let Some(key) = key {
                        if let Some((_, previous)) = pairs.iter_mut().find(|(seen, _)| *seen == key)
                        {
                            *previous = value;
                        } else {
                            pairs.push((key, value));
                        }
                    } else {
                        items.push(value);
                    }
                    *position = super::skip_json_space(bytes, *position);
                    if bytes.get(*position) == Some(&close) {
                        break;
                    }
                    if bytes.get(*position) != Some(&b',') {
                        return None;
                    }
                    *position += 1;
                }
            }
            *position += 1;
            Some(if mapping {
                PyVal::Dict(pairs)
            } else {
                PyVal::List(items)
            })
        }
        _ => {
            for (literal, value) in [
                ("null", PyVal::None),
                ("true", PyVal::Bool(true)),
                ("false", PyVal::Bool(false)),
                ("NaN", PyVal::Float(f64::NAN)),
                ("Infinity", PyVal::Float(f64::INFINITY)),
                ("-Infinity", PyVal::Float(f64::NEG_INFINITY)),
            ] {
                if text[start..].starts_with(literal) {
                    *position += literal.len();
                    return Some(value);
                }
            }
            super::json_number(bytes, position).then_some(())?;
            let number = &text[start..*position];
            if number.contains(['.', 'e', 'E']) {
                number.parse().ok().map(PyVal::Float)
            } else {
                Some(
                    number
                        .parse()
                        .map_or_else(|_| PyVal::BigInt(number.into()), PyVal::Int),
                )
            }
        }
    }
}

pub(super) fn series(mut values: Vec<PyVal>) -> Series {
    if values.iter().all(|value| matches!(value, PyVal::Bool(_))) {
        let array = BooleanArray::from(
            values
                .iter()
                .map(|value| matches!(value, PyVal::Bool(true)))
                .collect::<Vec<_>>(),
        );
        return Series::new(Dtype::Numpy("bool".into()), Arc::new(array));
    }
    if values.iter().all(|value| matches!(value, PyVal::Int(_))) {
        let signed: Option<Vec<_>> = values
            .iter()
            .map(|value| match value {
                PyVal::Int(value) => i64::try_from(*value).ok(),
                _ => None,
            })
            .collect();
        if let Some(values) = signed {
            return Series::new(
                Dtype::Numpy("int64".into()),
                Arc::new(Int64Array::from(values)),
            );
        }
        let unsigned: Option<Vec<_>> = values
            .iter()
            .map(|value| match value {
                PyVal::Int(value) => u64::try_from(*value).ok(),
                _ => None,
            })
            .collect();
        if let Some(values) = unsigned {
            return Series::new(
                Dtype::Numpy("uint64".into()),
                Arc::new(UInt64Array::from(values)),
            );
        }
    }
    if values
        .iter()
        .any(|value| matches!(value, PyVal::Int(_) | PyVal::Float(_)))
        && values
            .iter()
            .all(|value| matches!(value, PyVal::None | PyVal::Int(_) | PyVal::Float(_)))
        && values.iter().all(|value| {
            !matches!(value, PyVal::Int(value)
            if *value < i128::from(i64::MIN) || *value > i128::from(u64::MAX))
        })
        && !(values
            .iter()
            .any(|value| matches!(value, PyVal::Int(value) if *value < 0))
            && values
                .iter()
                .any(|value| matches!(value, PyVal::Int(value) if *value > i128::from(i64::MAX))))
    {
        let numbers = values
            .iter()
            .map(|value| match value {
                PyVal::Int(value) => *value as f64,
                PyVal::Float(value) => *value,
                _ => f64::NAN,
            })
            .collect::<Vec<_>>();
        return Series::new(
            Dtype::Numpy("float64".into()),
            Arc::new(Float64Array::from(numbers)),
        );
    }
    if values.iter().any(|value| matches!(value, PyVal::Str(_)))
        && values
            .iter()
            .all(|value| matches!(value, PyVal::None | PyVal::Str(_)))
    {
        let texts = values
            .iter()
            .map(|value| match value {
                PyVal::Str(value) => Some(value.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        for value in &mut values {
            if matches!(value, PyVal::None) {
                *value = PyVal::Float(f64::NAN);
            }
        }
        return Series::new(
            Dtype::Extension("str".into()),
            Arc::new(StringArray::from(texts)),
        )
        .with_original_values(values);
    }
    let encoded = values
        .iter()
        .map(|value| fixture(value).to_string())
        .collect();
    Series::new(Dtype::Object, encoded_objects(encoded)).with_original_values(values)
}

fn fixture(value: &PyVal) -> Value {
    match value {
        PyVal::None => Value::Null,
        PyVal::Bool(value) => json!(value),
        PyVal::Int(value) => json!({"$int": value.to_string()}),
        PyVal::BigInt(value) => json!({"$int": value}),
        PyVal::Float(value) if !value.is_finite() => {
            json!({"$float": if value.is_nan() { "nan" } else if value.is_sign_negative() { "-inf" } else { "inf" }})
        }
        PyVal::Float(value) => json!(value),
        PyVal::Str(value) => json!(value),
        PyVal::List(values) => Value::Array(values.iter().map(fixture).collect()),
        PyVal::Dict(pairs) => {
            json!({"$map": pairs.iter().map(|(key, value)| json!([fixture(key), fixture(value)])).collect::<Vec<_>>()})
        }
        _ => unreachable!("JSON parser returned a non-JSON value"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    #[test]
    fn numeric_inference_keeps_out_of_range_and_mixed_signed_integers_exact() {
        for values in [
            vec![PyVal::Int(i128::from(u64::MAX) + 1), PyVal::Float(1.5)],
            vec![PyVal::Int(i128::MAX), PyVal::Float(1.5)],
            vec![
                PyVal::Int(-1),
                PyVal::Int(i128::from(u64::MAX)),
                PyVal::Float(1.5),
            ],
            vec![PyVal::Int(i128::from(i64::MIN) - 1), PyVal::Float(1.5)],
        ] {
            let output = series(values.clone());
            assert_eq!(output.dtype, Dtype::Object);
            assert_eq!(output.object_values(), Some(values));
        }
        for tail in [PyVal::None, PyVal::Float(1.5)] {
            let output = series(vec![PyVal::Int(i128::from(u64::MAX)), tail]);
            assert_eq!(output.dtype, Dtype::Numpy("float64".into()));
        }
    }

    #[test]
    fn decoder_keeps_json_types_order_and_numeric_precision() {
        let value =
            parse(r#"{"a":1,"b":[null,true,"NaN",NaN,Infinity,-Infinity,-0.0],"a":2}"#).unwrap();
        let PyVal::Dict(pairs) = value else {
            panic!("mapping expected")
        };
        assert_eq!(pairs[0], (PyVal::Str("a".into()), PyVal::Int(2)));
        let PyVal::List(values) = &pairs[1].1 else {
            panic!("list expected")
        };
        assert_eq!(
            values[..3],
            [PyVal::None, PyVal::Bool(true), PyVal::Str("NaN".into())]
        );
        assert!(matches!(values[3], PyVal::Float(value) if value.is_nan()));
        assert_eq!(
            values[4..6],
            [PyVal::Float(f64::INFINITY), PyVal::Float(f64::NEG_INFINITY)]
        );
        assert!(
            matches!(values[6], PyVal::Float(value) if value == 0.0 && value.is_sign_negative())
        );
        assert_eq!(
            parse("9223372036854775808"),
            Some(PyVal::Int(9223372036854775808))
        );
        assert_eq!(
            parse("1234567890123456789012345678901234567890"),
            Some(PyVal::BigInt(
                "1234567890123456789012345678901234567890".into()
            ))
        );
        assert_eq!(parse("1e400"), Some(PyVal::Float(f64::INFINITY)));
    }

    #[test]
    fn malformed_json_does_not_partially_decode() {
        for value in [
            "",
            "01",
            "[1,]",
            "{\"a\":1,}",
            "[1] trailing",
            "nan",
            "true false",
            "1.",
            "--1",
            "\"bad\ntext\"",
        ] {
            assert!(parse(value).is_none(), "{value}");
        }
    }

    #[test]
    fn shipping_sanitizer_preserves_mixed_json_values() {
        let source = Series::new(
            Dtype::Arrow,
            Arc::new(StringArray::from(vec![
                "{\"a\":[true,null]}",
                "[1]",
                "9223372036854775808",
            ])),
        );
        let frame = crate::frame::Frame::new(vec!["items".into()], vec![source], 3);
        let mut warnings = Vec::new();
        let output = crate::driver::sql::sanitize_with(
            &sdm_schema::dt::Host::assumed(),
            &[(
                "items".into(),
                crate::driver::sql::dtype_of_name("stringlist"),
            )],
            &[],
            frame,
            &mut warnings,
        )
        .unwrap();
        let column = &output.columns[0];
        assert_eq!(column.dtype, Dtype::Object);
        assert_eq!(
            column.object_values().unwrap(),
            vec![
                PyVal::Dict(vec![(
                    PyVal::Str("a".into()),
                    PyVal::List(vec![PyVal::Bool(true), PyVal::None])
                )]),
                PyVal::List(vec![PyVal::Int(1)]),
                PyVal::Int(9223372036854775808),
            ]
        );
        assert_eq!(
            crate::frame::object_values(&column.array),
            column.object_values()
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn malformed_row_keeps_the_entire_original_column() {
        let source = Series::new(
            Dtype::Arrow,
            Arc::new(StringArray::from(vec!["[1]", "invalid"])),
        );
        let output = super::super::to_list(&source, super::super::ListKind::Text).unwrap();
        assert_eq!(output.array.to_data(), source.array.to_data());
        assert_eq!(output.dtype, source.dtype);
    }
}
