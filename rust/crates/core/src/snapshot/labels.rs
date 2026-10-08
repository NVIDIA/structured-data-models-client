// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::Array;
use arrow_schema::{DataType, TimeUnit};
use sdm_labels::column::{Column, Obj};
use sdm_labels::executor::{Batch, Frame};
use serde_json::{Value, json};

use super::{
    Error, Result, array, decode_typed, encode_typed, field, fields, object, require, text,
    typed_observation,
};

fn host(facts: Option<&Value>) -> Result<Option<sdm_schema::dt::Host>> {
    facts
        .map(|facts| {
            fields(facts, &["host", "primitives"])?;
            require(
                array(field(facts, "primitives")?)?.is_empty(),
                "unused label host primitives",
            )?;
            super::host::descriptor(field(facts, "host")?)
        })
        .transpose()
}

pub(super) fn questions(input: &Value, facts: Option<&Value>) -> Result<Value> {
    let host = host(facts)?;
    fields(input, &["mapping"])?;
    let mut arguments = serde_json::Map::new();
    for entry in array(field(input, "mapping")?)? {
        let pair = array(entry)?;
        require(pair.len() == 2, "invalid typed question mapping")?;
        fields(&pair[0], &["scalar"])?;
        let name = text(field(&pair[0], "scalar")?)?;
        require(
            arguments.insert(name.to_owned(), pair[1].clone()).is_none(),
            "duplicate typed question field",
        )?;
    }
    let arguments = Value::Object(arguments);
    fields(&arguments, &["query", "schema", "resolved", "batches"])?;
    let query = decode_typed(field(&arguments, "query")?)?;
    let schema = decode_typed(field(&arguments, "schema")?)?;
    let resolved = sdm_pql::validate_json(text(&query)?, &schema.to_string(), "RELATIONAL", None)
        .map_err(|_| Error("invalid question query"))?;
    let resolved: Value =
        serde_json::from_str(&resolved).map_err(|_| Error("invalid question resolution"))?;
    let tree = field(&resolved, "tree")?;
    require(
        host.is_some() || !needs_host(tree),
        "label date constants require explicit host facts",
    )?;
    let batches = field(&arguments, "batches")?;
    fields(batches, &["list"])?;
    let mut results = Vec::new();
    let mut warnings = Vec::new();
    for input in array(field(batches, "list")?)? {
        let mut batch = super::columns::batch(input)?;
        if let Some(host) = &host {
            batch.host = host.clone();
        }
        let result = match sdm_labels::execute(&tree.to_string(), &batch) {
            Ok((output, mask)) => json!({"mapping": [
                [{"scalar": "dtype"}, {"scalar": output.dtype()}],
                [{"scalar": "values"}, {"list": output.objects().iter().map(plain).collect::<Result<Vec<_>>>()?}],
                [{"scalar": "mask"}, encode_typed(json!(mask))?],
            ]}),
            Err(error) => {
                encode_typed(json!({"error": {"code": error.code, "message": error.message}}))?
            }
        };
        results.push(result);
        warnings.extend(batch.warnings.into_inner().into_iter().filter(|warning| !warning.masked_float32_cast()).map(|warning| json!({"class": warning.category(), "code": null, "message": warning.message(), "details": null})));
    }
    let mut observation = typed_observation(json!({"list": results}), Value::Null)?;
    observation["observation"]["warnings"] = json!(warnings);
    Ok(observation)
}

fn needs_host(value: &Value) -> bool {
    match value {
        Value::Object(fields) => {
            fields.get("dtype_maybe").is_some_and(|dtype| {
                ["date", "time", "timestamp"]
                    .iter()
                    .any(|name| dtype == name || dtype.get("nested_dtype") == Some(&json!(name)))
            }) || fields.values().any(needs_host)
        }
        Value::Array(values) => values.iter().any(needs_host),
        _ => false,
    }
}

fn column(value: &Value, dtype: Option<&str>) -> Result<Column> {
    let table = super::sampling::arrow_table(value)?;
    if table.num_columns() != 1 || table.schema().field(0).name() != "values" {
        return Err(Error("invalid label array input"));
    }
    sdm_labels::arrow::column(table.column(0).as_ref(), dtype)
        .map_err(|_| Error("invalid label Arrow values"))
}

fn type_name(dtype: &DataType) -> Result<String> {
    Ok(match dtype {
        DataType::Null => "null".into(),
        DataType::Boolean => "bool".into(),
        DataType::Int64 => "int64".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 => "string".into(),
        DataType::LargeUtf8 => "large_string".into(),
        DataType::Binary => "binary".into(),
        DataType::Duration(unit) => format!(
            "duration[{}]",
            match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            }
        ),
        DataType::Timestamp(unit, None) => format!(
            "timestamp[{}]",
            match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            }
        ),
        DataType::Decimal128(precision, scale) => format!("decimal128({precision}, {scale})"),
        DataType::List(element) => format!(
            "list<{}: {}>",
            element.name(),
            type_name(element.data_type())?
        ),
        _ => return Err(Error("unsupported label Arrow output type")),
    })
}

fn plain(value: &Obj) -> Result<Value> {
    let value = match value {
        Obj::None | Obj::Na | Obj::NaT => Value::Null,
        Obj::Bool(value) | Obj::NpBool(value) => json!(value),
        Obj::Int(value) | Obj::NpInt64(value) => json!(value),
        Obj::Float(value) | Obj::NpFloat64(value) if value.is_nan() => json!("nan"),
        Obj::Float(value) | Obj::NpFloat64(value) if value.is_finite() => json!(value),
        Obj::NpFloat32(value) if value.is_nan() => json!("nan"),
        Obj::NpFloat32(value) if value.is_finite() => json!(f64::from(*value)),
        Obj::Str(value) => json!(value),
        Obj::Bytes(_) => return Ok(json!({"fixture_scalar": sdm_labels::fixture::encode(value)})),
        Obj::Duration(unit, ticks) => {
            let bytes: String = ticks
                .to_le_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            return Ok(json!({"pandas_scalar": {
                "class": "Timedelta",
                "value": {"numpy_scalar": {"dtype": format!("<m8[{}]", unit.name()), "bytes_hex": bytes}},
                "timezone": null, "fold": null
            }}));
        }
        Obj::Time(unit, ticks) => json!([
            "ts",
            i64::try_from(i128::from(*ticks) * i128::from(unit.nanos()))
                .map_err(|_| Error("wide label timestamp requires lossless typed encoding"))?
        ]),
        Obj::Decimal(digits, scale) => json!(sdm_schema::frame::decimal_text(*digits, *scale)),
        Obj::List(values) => {
            return Ok(json!({"list": values.iter().map(plain).collect::<Result<Vec<_>>>()?}));
        }
        Obj::Float(value) | Obj::NpFloat64(value) => {
            return Ok(json!({"float": if value.is_sign_negative() { "-inf" } else { "inf" }}));
        }
        Obj::NpFloat32(value) => {
            return Ok(json!({"float": if value.is_sign_negative() { "-inf" } else { "inf" }}));
        }
    };
    encode_typed(value)
}

pub(super) fn execute(query: &Value, arguments: &Value, facts: Option<&Value>) -> Result<Value> {
    let host = host(facts)?;
    require(
        host.is_some() || !needs_host(query),
        "label date constants require explicit host facts",
    )?;
    require(
        host.is_none() || needs_host(query),
        "unused array label host facts",
    )?;
    let mut batch = Batch::default();
    if let Some(host) = host {
        batch.host = host;
    }
    for (name, input) in object(field(arguments, "feat")?)? {
        let table = super::sampling::arrow_table(input)?;
        let metadata = table
            .schema()
            .metadata()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let dtypes = sdm_labels::arrow::pandas_dtypes(&metadata);
        let columns = table
            .schema()
            .fields()
            .iter()
            .zip(table.columns())
            .map(|(field, array)| {
                Ok((
                    field.name().clone(),
                    sdm_labels::arrow::column(
                        array.as_ref(),
                        dtypes.get(field.name()).map(String::as_str),
                    )
                    .map_err(|_| Error("invalid label feature column"))?,
                ))
            })
            .collect::<Result<_>>()?;
        batch.feat.push((name.clone(), Frame { columns }));
    }
    for (name, input) in object(field(arguments, "times")?)? {
        batch.time.push((name.clone(), column(input, None)?));
    }
    for (name, input) in object(field(arguments, "batches")?)? {
        let Column::I64(indices) = column(input, Some("int64"))? else {
            return Err(Error("label batch indices must be int64"));
        };
        batch.batch.push((name.clone(), indices));
    }
    batch.anchor = column(field(arguments, "anchor")?, None)?;
    match sdm_labels::execute(&query.to_string(), &batch)
        .and_then(|(output, mask)| Ok((sdm_labels::arrow::to_arrow(&output)?, mask)))
    {
        Ok((arrow, mask)) => {
            let output = sdm_labels::arrow::column(arrow.as_ref(), None)
                .map_err(|_| Error("invalid label output array"))?;
            let values = output
                .objects()
                .iter()
                .enumerate()
                .map(|(position, value)| {
                    if arrow.is_null(position) {
                        Ok(json!({"scalar": null}))
                    } else if matches!(value, Obj::Duration(unit, _) if *unit != sdm_labels::column::TimeUnit::Ns) {
                        Ok(json!({"fixture_scalar": sdm_labels::fixture::encode(value)}))
                    } else {
                        plain(value)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({"mapping": [
                [{"scalar": "type"}, {"scalar": type_name(arrow.data_type())?}],
                [{"scalar": "values"}, {"list": values}],
                [{"scalar": "mask"}, encode_typed(json!(mask))?],
            ]}))
        }
        Err(error) => {
            encode_typed(json!({"error": {"code": error.code, "message": error.message}}))
        }
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BinaryArray, Int64Array, RecordBatch};
    use arrow_schema::{Field, Schema};

    fn encoded_table(columns: Vec<(&str, ArrayRef)>) -> Value {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, values)| Field::new(*name, values.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            columns.into_iter().map(|(_, values)| values).collect(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let encoded = sdm_labels::fixture::encode(&Obj::Bytes(bytes));
        json!({"arrow_ipc_base64": encoded["$bytes"]})
    }

    #[test]
    fn binary_and_duration_arrow_values_survive_full_snapshot_replay() {
        let mut inputs: Vec<ArrayRef> = vec![Arc::new(BinaryArray::from(vec![
            Some(&b"\x00\xff"[..]),
            Some(&b""[..]),
            None,
        ]))];
        for unit in [
            sdm_labels::column::TimeUnit::S,
            sdm_labels::column::TimeUnit::Ms,
            sdm_labels::column::TimeUnit::Us,
            sdm_labels::column::TimeUnit::Ns,
        ] {
            inputs.push(
                sdm_labels::arrow::to_arrow(&Column::Obj(vec![
                    Obj::Duration(unit, -1),
                    Obj::Duration(unit, 2),
                    Obj::NaT,
                ]))
                .unwrap(),
            );
            let scalar = plain(&Obj::Duration(unit, i64::MAX - 1)).unwrap();
            assert_eq!(
                scalar["pandas_scalar"]["value"]["numpy_scalar"]["bytes_hex"],
                "feffffffffffff7f"
            );
            assert_eq!(
                scalar["pandas_scalar"]["value"]["numpy_scalar"]["dtype"],
                format!("<m8[{}]", unit.name())
            );
        }
        for values in inputs {
            let dtype = values.data_type().clone();
            let expected_values = match dtype {
                DataType::Binary => {
                    json!([{"fixture_scalar": {"$bytes": "AP8="}}, {"fixture_scalar": {"$bytes": ""}}])
                }
                DataType::Duration(TimeUnit::Nanosecond) => json!([
                    plain(&Obj::Duration(sdm_labels::column::TimeUnit::Ns, -1)).unwrap(),
                    plain(&Obj::Duration(sdm_labels::column::TimeUnit::Ns, 2)).unwrap()
                ]),
                DataType::Duration(ref unit) => {
                    let micros = match unit {
                        TimeUnit::Second => 1_000_000,
                        TimeUnit::Millisecond => 1_000,
                        TimeUnit::Microsecond => 1,
                        TimeUnit::Nanosecond => unreachable!(),
                    };
                    json!([{"fixture_scalar": {"$timedelta_us": -micros}}, {"fixture_scalar": {"$timedelta_us": 2 * micros}}])
                }
                _ => unreachable!(),
            };
            let arguments = json!({"kind": "execute", "query_json": json!({"entity_ast": {"fqn": "items.id"}, "target_ast": {"fqn": "items.target"}}).to_string(),
                "feat": {"items": encoded_table(vec![("id", Arc::new(Int64Array::from(vec![1, 2, 3]))), ("target", values)])},
                "times": {}, "batches": {}, "anchor": encoded_table(vec![("values", Arc::new(Int64Array::from(vec![0, 0, 0])))])});
            let mut case = json!({"id": "labels", "depends_on": [], "expected": "never used", "replay_input": {
                "generator": {"name": "sampling.hashseed_array_jobs", "stream_seed": 1, "ordinal": 0}, "generator_input": encode_typed(arguments).unwrap()}});
            let expected = json!({"mapping": [
                [{"scalar": "type"}, {"scalar": type_name(&dtype).unwrap()}],
                [{"scalar": "values"}, {"list": expected_values}],
                [{"scalar": "mask"}, encode_typed(json!([true, true, false])).unwrap()]
            ]});
            let replay = |case: &Value| {
                let bytes = json!({"format": "p97-generators-v1", "cases": [case]}).to_string();
                let encoded: String = bytes.bytes().map(|byte| format!("{byte:02x}")).collect();
                super::super::replay_snapshot(&json!({"files": {"tests/parity/frozen-generators/labels.json": encoded}, "cases": ["labels"], "routes": {"labels": {"kind": "native_semantic", "native_functions": ["replay_snapshot"]}}}).to_string()).unwrap()
            };
            let output = replay(&case);
            assert_eq!(
                output["observations"]["labels"]["observation"]["result"]["mapping"][0][1]["list"]
                    [0],
                expected
            );
            case["expected"] = json!({"corrupt": true});
            assert_eq!(replay(&case), output);
        }
    }

    #[test]
    fn array_date_constants_use_explicit_host_facts_through_full_replay() {
        let times = sdm_labels::arrow::to_arrow(&Column::Time(
            sdm_labels::column::TimeUnit::Ns,
            vec![Some(0), Some(1_735_689_600_000_000_000)],
        ))
        .unwrap();
        let facts = json!({"host": {
            "year": 2026, "today": [2026, 10, 7], "pads_year": true,
            "tznames": ["UTC"], "timestamp_series_units": ["s", "ms", "us", "ns"]
        }, "primitives": []});
        let replay = |case: &Value| {
            let bytes = json!({"format": "p97-generators-v1", "cases": [case]}).to_string();
            let encoded: String = bytes.bytes().map(|byte| format!("{byte:02x}")).collect();
            super::super::replay_snapshot(
                &json!({"files": {
                    "tests/parity/frozen-generators/array-date.json": encoded
                }, "cases": ["array-date"], "routes": {"array-date": {
                    "kind": "native_semantic", "native_functions": ["replay_snapshot"]
                }}})
                .to_string(),
            )
        };
        for literal in ["2024-01-01", "2024-06-30 12:00:00"] {
            let query = json!({"entity_ast": {"fqn": "items.id"}, "target_ast": {
                "target": {"fqn": "items.at"}, "op": ">",
                "value": {"dtype_maybe": "time", "value": literal}
            }});
            let arguments = json!({"kind": "execute", "query_json": query.to_string(),
                "feat": {"items": encoded_table(vec![
                    ("id", Arc::new(Int64Array::from(vec![1, 2]))), ("at", times.clone())
                ])}, "times": {}, "batches": {},
                "anchor": encoded_table(vec![("values", Arc::new(Int64Array::from(vec![0, 0])))])});
            let mut case = json!({"id": "array-date", "depends_on": [], "expected": "unused",
                "replay_input": {"generator": {"name": "sampling.hashseed_array_jobs",
                "stream_seed": 1, "ordinal": 0}, "generator_input": encode_typed(arguments.clone()).unwrap()}});
            assert_eq!(
                replay(&case).unwrap_err().0,
                "label date constants require explicit host facts"
            );
            case["replay_input"]["host_inputs"] = facts.clone();
            let actual = replay(&case).unwrap();
            assert_eq!(
                actual["observations"]["array-date"]["observation"]["result"],
                json!({"mapping": [
                    [{"scalar": "answers"}, {"list": [{"mapping": [
                        [{"scalar": "type"}, {"scalar": "bool"}],
                        [{"scalar": "values"}, {"list": [{"scalar": false}, {"scalar": true}]}],
                        [{"scalar": "mask"}, {"list": [{"scalar": true}, {"scalar": true}]}]
                    ]}]}],
                    [{"scalar": "safe_python"}, {"list": []}],
                    [{"scalar": "purpose"}, {"scalar": "native-hashseed-consistency"}]
                ]})
            );
            case["expected"] = json!({"corrupted": true});
            assert_eq!(replay(&case).unwrap(), actual);
            for invalid in [
                json!({}),
                json!({"host": facts["host"], "primitives": ["unused"]}),
                json!({"host": facts["host"], "primitives": [], "extra": true}),
            ] {
                case["replay_input"]["host_inputs"] = invalid;
                assert!(replay(&case).is_err());
            }
            case["replay_input"]["host_inputs"] = facts.clone();
            case["replay_input"]["generator"]["name"] = json!("sampling.random_state");
            assert!(replay(&case).is_err());
            let mut unsupported = arguments.clone();
            unsupported["kind"] = json!("store");
            assert!(
                super::super::sampling::replay(
                    "sampling.hashseed_array_jobs",
                    &unsupported,
                    Some(&facts)
                )
                .is_err()
            );
            let non_date =
                json!({"entity_ast": {"fqn": "items.id"}, "target_ast": {"fqn": "items.id"}});
            assert_eq!(
                execute(&non_date, &arguments, Some(&facts)).unwrap_err().0,
                "unused array label host facts"
            );
        }
    }

    #[test]
    fn actual_date_constant_names_require_host_facts() {
        for name in ["date", "time", "timestamp"] {
            assert!(needs_host(
                &json!({"dtype_maybe": name, "value": "29.02.2024"})
            ));
            assert!(needs_host(
                &json!({"dtype_maybe": {"nested_dtype": name}, "value": []})
            ));
        }
        assert!(!needs_host(
            &json!({"dtype_maybe": "string", "value": "29.02.2024"})
        ));
    }
}
