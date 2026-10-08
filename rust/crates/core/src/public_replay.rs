// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::Cursor;

use sdm_protocol::{Outcome, Param, QueryResult};
use sdm_schema::dt::Host;
use serde_json::{Value, json};

use crate::plain::Plain;

mod carry_result;

type Result<T> = std::result::Result<T, String>;

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| format!("missing public input {name}"))
}

fn text(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| "public input must be text".into())
}

pub(crate) fn decode(
    value: &Value,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<Plain> {
    Ok(match value {
        Value::Null => Plain::None,
        Value::Bool(value) => Plain::Bool(*value),
        Value::String(value) => Plain::Str(value.clone()),
        Value::Number(value) => match value.as_i64() {
            Some(value) => Plain::Int(value),
            None if value.is_u64() => Plain::BigInt(value.to_string()),
            None => Plain::Float(value.as_f64().ok_or("invalid number")?),
        },
        Value::Array(values) => Plain::List(
            values
                .iter()
                .map(|value| decode(value, blobs, directory))
                .collect::<Result<_>>()?,
        ),
        Value::Object(values) => {
            if values.len() == 1 {
                let (tag, body) = values.iter().next().ok_or("missing plain value")?;
                match tag.as_str() {
                    "$table" | "$batch" | "$array" | "$chunked" => {
                        let digest = text(body)?;
                        if digest.len() != 64
                            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                        {
                            return Err("invalid Arrow input identity".into());
                        }
                        let name = format!("{directory}/tables/{digest}.arrow");
                        let raw = blobs.get(&name).ok_or("missing Arrow input dependency")?;
                        let reader = arrow_ipc::reader::FileReader::try_new(Cursor::new(raw), None)
                            .map_err(|_| "invalid Arrow input")?;
                        let schema = reader.schema();
                        let batches = reader
                            .collect::<std::result::Result<Vec<_>, _>>()
                            .map_err(|_| "invalid Arrow input batch")?;
                        return match tag.as_str() {
                            "$table" => Ok(Plain::Table(schema, batches)),
                            "$batch" => arrow_select::concat::concat_batches(&schema, &batches)
                                .map(Plain::Batch)
                                .map_err(|_| "invalid Arrow batch".into()),
                            _ => {
                                let position = schema
                                    .index_of("values")
                                    .map_err(|_| "missing Arrow values")?;
                                let chunks: Vec<_> = batches
                                    .iter()
                                    .map(|batch| batch.column(position).clone())
                                    .collect();
                                if tag == "$chunked" {
                                    Ok(Plain::Chunked(
                                        schema.field(position).data_type().clone(),
                                        chunks,
                                    ))
                                } else {
                                    let arrays: Vec<_> =
                                        chunks.iter().map(|array| array.as_ref()).collect();
                                    arrow_select::concat::concat(&arrays)
                                        .map(Plain::Array)
                                        .map_err(|_| "invalid Arrow array".into())
                                }
                            }
                        };
                    }
                    "$tuple" => {
                        return Ok(Plain::Tuple(
                            body.as_array()
                                .ok_or("invalid plain tuple")?
                                .iter()
                                .map(|value| decode(value, blobs, directory))
                                .collect::<Result<_>>()?,
                        ));
                    }
                    "$int" => {
                        let digits = text(body)?;
                        sdm_dialects::int::Int::parse(digits).ok_or("invalid plain integer")?;
                        return Ok(Plain::BigInt(digits.into()));
                    }
                    "$float" => {
                        return Ok(Plain::Float(match text(body)? {
                            "nan" => f64::NAN,
                            "inf" => f64::INFINITY,
                            "-inf" => f64::NEG_INFINITY,
                            _ => return Err("invalid nonfinite float".into()),
                        }));
                    }
                    "$dict" => {
                        let values = body.as_object().ok_or("invalid plain mapping")?;
                        return Ok(Plain::Dict(
                            values
                                .iter()
                                .map(|(key, value)| {
                                    Ok((key.clone(), decode(value, blobs, directory)?))
                                })
                                .collect::<Result<_>>()?,
                        ));
                    }
                    "$bytes" => {
                        if let sdm_schema::pyval::PyVal::Bytes(bytes) =
                            sdm_schema::pyval::PyVal::decode(value)
                                .map_err(|_| "invalid plain bytes")?
                        {
                            return Ok(Plain::Bytes(bytes));
                        }
                        return Err("invalid plain bytes".into());
                    }
                    _ => {}
                }
            }
            Plain::Dict(
                values
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), decode(value, blobs, directory)?)))
                    .collect::<Result<_>>()?,
            )
        }
    })
}

pub(crate) fn outcome(
    value: &Value,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<Outcome> {
    let values = value.as_object().ok_or("invalid host outcome")?;
    if values.len() != 1 {
        return Err("ambiguous host outcome".into());
    }
    let (kind, body) = values.iter().next().ok_or("missing host outcome")?;
    match kind.as_str() {
        "data" => match decode(body, blobs, directory)? {
            Plain::Table(schema, batches) => QueryResult::try_new(schema, batches)
                .map(Outcome::Data)
                .map_err(|_| "invalid source data".into()),
            Plain::Batch(batch) => Ok(Outcome::Data(QueryResult::from_batch(batch))),
            _ => Err("source data must be Arrow".into()),
        },
        "response" => Ok(Outcome::Response(text(body)?.into())),
        "model_error" => Ok(Outcome::ModelError(text(body)?.into())),
        "error" => Ok(Outcome::Error {
            sqlstate: body
                .get("sqlstate")
                .and_then(Value::as_str)
                .map(str::to_owned),
            vendor_code: body.get("vendor_code").and_then(Value::as_i64),
            message: text(field(body, "message")?)?.into(),
        }),
        _ => Err("unsupported host outcome".into()),
    }
}

pub fn replay(input: &Value, blobs: &BTreeMap<String, &[u8]>, directory: &str) -> Result<Value> {
    if input.get("carry").is_some() {
        return carry(input, blobs, directory);
    }
    let recipe = field(input, "recipe")?;
    let declared = recipe.pointer("/input/constructor");
    if declared.is_some() && declared.and_then(Value::as_str) != Some("declared_table_data") {
        return Err("unsupported public constructor".into());
    }
    exact_fields(
        input,
        if declared.is_some() {
            &[
                "recipe",
                "initial_schema",
                "host",
                "outcomes",
                "declared_table_data",
            ]
        } else {
            &["recipe", "initial_schema", "host", "outcomes"]
        },
    )?;
    let host = host(field(input, "host")?)?;
    materialize(input, recipe, declared.is_some(), host, blobs, directory)
}

fn host(host: &Value) -> Result<Host> {
    exact_fields(
        host,
        if host.get("numpy_text_prefix").is_some() {
            &[
                "year",
                "today",
                "pads_year",
                "tznames",
                "timestamp_series_units",
                "numpy_text_prefix",
            ]
        } else {
            &[
                "year",
                "today",
                "pads_year",
                "tznames",
                "timestamp_series_units",
            ]
        },
    )?;
    let today: (i64, u32, u32) =
        serde_json::from_value(field(host, "today")?.clone()).map_err(|_| "invalid host date")?;
    Ok(Host {
        numpy_text_prefix: host
            .get("numpy_text_prefix")
            .map(|value| value.as_bool().ok_or("invalid NumPy text policy"))
            .transpose()?
            .unwrap_or(false),
        year: field(host, "year")?.as_i64().ok_or("invalid host year")?,
        today,
        pads_year: field(host, "pads_year")?
            .as_bool()
            .ok_or("invalid host year formatting")?,
        tznames: serde_json::from_value(field(host, "tznames")?.clone())
            .map_err(|_| "invalid host timezones")?,
        timestamp_series_units: sdm_schema::dt::pandas::TimestampSeriesUnits::from_json(field(
            host,
            "timestamp_series_units",
        )?)?,
    })
}

fn materialize(
    input: &Value,
    recipe: &Value,
    declared: bool,
    host: Host,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<Value> {
    let initial = field(input, "initial_schema")?;
    exact_fields(initial, &["function", "args"])?;
    let function = text(field(initial, "function")?)?;
    let args = field(initial, "args")?
        .as_array()
        .ok_or("invalid initial schema arguments")?
        .iter()
        .map(|value| decode(value, blobs, directory))
        .collect::<Result<Vec<_>>>()?;
    let replacement = if declared {
        if function != "data_schema"
            || recipe.pointer("/input/backend").and_then(Value::as_str) != Some("memory")
            || recipe.get("matrix").is_none()
        {
            return Err("declared data requires an in-memory matrix constructor".into());
        }
        let replacement = field(input, "declared_table_data")?;
        exact_fields(replacement, &["frames", "tags"])?;
        let frames = decode(field(replacement, "frames")?, blobs, directory)?;
        let tags: Value = serde_json::from_str(text(field(replacement, "tags")?)?)
            .map_err(|_| "invalid declared frame tags")?;
        let Plain::Dict(tables) = &frames else {
            return Err("declared frames must be a mapping".into());
        };
        let names = recipe
            .pointer("/input/table_names")
            .and_then(Value::as_array)
            .ok_or("missing declared table names")?;
        let tags_map = tags.as_object().ok_or("invalid declared frame tags")?;
        if names.is_empty()
            || tables.len() != names.len()
            || tags_map.len() != tables.len()
            || tables.iter().zip(names).any(|((name, table), expected)| {
                expected
                    .as_str()
                    .is_none_or(|expected| name != &format!("schema/{expected}"))
                    || !tags_map.contains_key(name)
                    || !matches!(table, Plain::Table(_, _) | Plain::Batch(_))
            })
        {
            return Err("declared frame domain does not match the recipe".into());
        }
        Some((frames, tags))
    } else {
        None
    };
    let outcomes = outcomes(input, blobs, directory)?;
    crate::flow::public_materialize(recipe, function, args, host, outcomes, replacement)
        .map_err(|error| error.0)
}

fn outcomes(
    input: &Value,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<Vec<(bool, Outcome)>> {
    field(input, "outcomes")?
        .as_array()
        .ok_or("invalid external outcomes")?
        .iter()
        .map(|value| {
            exact_fields(value, &["kind", "outcome"])?;
            let model = match text(field(value, "kind")?)? {
                "query" => false,
                "model" => true,
                _ => return Err("invalid external outcome kind".into()),
            };
            Ok((model, outcome(field(value, "outcome")?, blobs, directory)?))
        })
        .collect::<Result<_>>()
}

fn carry(input: &Value, blobs: &BTreeMap<String, &[u8]>, directory: &str) -> Result<Value> {
    let recipe = field(input, "recipe")?;
    let shape = text(field(input, "carry")?)?;
    let host = host(field(input, "host")?)?;
    let outcomes = outcomes(input, blobs, directory)?;
    match shape {
        "client.run" => {
            exact_fields(
                input,
                &[
                    "carry",
                    "recipe",
                    "frames",
                    "table_order",
                    "cleanup",
                    "tags",
                    "sources",
                    "replacements",
                    "host",
                    "outcomes",
                ],
            )?;
            if recipe.get("operation").and_then(Value::as_str) != Some("client.run")
                || recipe
                    .pointer("/input/carry_case_id")
                    .and_then(Value::as_str)
                    .is_none()
            {
                return Err("carry flow recipe differs".into());
            }
            let Plain::Dict(mut frames) = decode(field(input, "frames")?, blobs, directory)? else {
                return Err("carry frames must be a mapping".into());
            };
            let mut ordered = Vec::new();
            for name in field(input, "table_order")?
                .as_array()
                .ok_or("missing carry table order")?
            {
                let name = text(name)?;
                let index = frames
                    .iter()
                    .position(|(key, _)| key == name)
                    .ok_or("carry table order differs")?;
                ordered.push(frames.remove(index));
            }
            if !frames.is_empty() {
                return Err("carry table order is incomplete".into());
            }
            let frames = Plain::Dict(ordered);
            let cleanup = field(input, "cleanup")?
                .as_array()
                .ok_or("invalid carry cleanup")?
                .iter()
                .map(|row| {
                    exact_fields(row, &["path", "status", "body"])?;
                    if row["status"] != 204 || row["body"] != "" {
                        return Err("carry cleanup was not successful".into());
                    }
                    Ok(text(field(row, "path")?)?.to_owned())
                })
                .collect::<Result<Vec<_>>>()?;
            let tags = serde_json::from_str(text(field(input, "tags")?)?)
                .map_err(|_| "invalid carry tags")?;
            let sources = field(input, "sources")?
                .as_array()
                .ok_or("invalid carry sources")?
                .iter()
                .map(|arguments| {
                    arguments
                        .as_array()
                        .ok_or("invalid carry source arguments")?
                        .iter()
                        .map(|value| decode(value, blobs, directory))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let replacements = field(input, "replacements")?
                .as_array()
                .ok_or("invalid carry caller frames")?
                .iter()
                .map(|value| {
                    exact_fields(value, &["operation", "frames", "tags"])?;
                    Ok((
                        text(field(value, "operation")?)?.to_owned(),
                        decode(field(value, "frames")?, blobs, directory)?,
                        serde_json::from_str(text(field(value, "tags")?)?)
                            .map_err(|_| "invalid carry caller tags")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            crate::flow::public_client_flow(
                recipe,
                frames,
                tags,
                host,
                outcomes,
                crate::flow::CarryInputs::new(sources, replacements, cleanup),
            )
            .map_err(|error| error.0)
        }
        "postgres" => {
            exact_fields(
                input,
                &["carry", "recipe", "initial_schema", "host", "outcomes"],
            )?;
            let initial = field(input, "initial_schema")?;
            exact_fields(initial, &["function", "args"])?;
            if text(field(initial, "function")?)? != "schema_source_run" {
                return Err("carry source requires source constructor".into());
            }
            let args = field(initial, "args")?
                .as_array()
                .ok_or("invalid carry source arguments")?
                .iter()
                .map(|value| decode(value, blobs, directory))
                .collect::<Result<Vec<_>>>()?;
            if recipe.get("operation").and_then(Value::as_str) != Some("postgres.carry")
                || recipe.pointer("/input/backend").and_then(Value::as_str) != Some("postgres")
            {
                return Err("carry PostgreSQL recipe differs".into());
            }
            match recipe
                .pointer("/input/restore_tags")
                .and_then(Value::as_bool)
            {
                Some(true) => {
                    return crate::flow::public_materialize(
                        recipe,
                        "schema_source_run",
                        args,
                        host,
                        outcomes,
                        None,
                    )
                    .map_err(|error| error.0);
                }
                Some(false) => {}
                None => return Err("missing carry PostgreSQL restoration mode".into()),
            }
            crate::flow::public_source_schema(args, host, outcomes).map_err(|error| error.0)
        }
        "result" => {
            exact_fields(
                input,
                &[
                    "carry",
                    "recipe",
                    "result",
                    "primitives",
                    "host",
                    "outcomes",
                ],
            )?;
            if recipe.get("operation").and_then(Value::as_str) != Some("result.to_pandas")
                || !outcomes.is_empty()
            {
                return Err("carry result operation or outcomes differ".into());
            }
            carry_result::replay(field(input, "result")?, field(input, "primitives")?)
        }
        _ => Err("unsupported public carry shape".into()),
    }
}

fn exact_fields(value: &Value, names: &[&str]) -> Result<()> {
    let fields = value.as_object().ok_or("invalid public input mapping")?;
    if fields.len() != names.len() || names.iter().any(|name| !fields.contains_key(*name)) {
        return Err("unexpected public input fields".into());
    }
    Ok(())
}

pub(crate) fn params(params: &[Param]) -> Value {
    if params.is_empty() {
        return Value::Null;
    }
    Value::Array(
        params
            .iter()
            .map(|param| match param {
                Param::Null => Value::Null,
                Param::Bool(value) => json!(value),
                Param::Int(value) => json!(value),
                Param::Float(value) if value.is_nan() => json!({"$float": "nan"}),
                Param::Float(value) if value.is_infinite() => {
                    json!({"$float": if *value < 0.0 { "-inf" } else { "inf" }})
                }
                Param::Float(value) => json!(value),
                Param::Text(value) => json!(value),
            })
            .collect(),
    )
}

#[cfg(test)]
mod declared_tests {
    use super::*;
    use arrow_array::{
        BinaryArray, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    };
    use std::sync::Arc;

    fn arrow_file(batch: &RecordBatch) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut writer =
            arrow_ipc::writer::FileWriter::try_new(&mut bytes, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    fn input() -> (Value, BTreeMap<String, Vec<u8>>) {
        let initial = RecordBatch::try_from_iter([
            (
                "product_id",
                Arc::new(Int64Array::from(vec![1, 2])) as arrow_array::ArrayRef,
            ),
            ("category", Arc::new(StringArray::from(vec!["one", "two"]))),
            (
                "ordered_at",
                Arc::new(TimestampMicrosecondArray::from(vec![1_000_000, 2_000_000])),
            ),
        ])
        .unwrap();
        let replaced = RecordBatch::try_from_iter([
            ("product_id", initial.column(0).clone()),
            ("category", initial.column(1).clone()),
            (
                "ordered_at",
                Arc::new(BinaryArray::from(vec![
                    b"one".as_slice(),
                    b"two".as_slice(),
                ])) as arrow_array::ArrayRef,
            ),
        ])
        .unwrap();
        let initial_sha = "1".repeat(64);
        let replaced_sha = "2".repeat(64);
        let files = BTreeMap::from([
            (
                format!("public/tables/{initial_sha}.arrow"),
                arrow_file(&initial),
            ),
            (
                format!("public/tables/{replaced_sha}.arrow"),
                arrow_file(&replaced),
            ),
        ]);
        let value = json!({
            "recipe": {
                "operation": "materialize",
                "entrypoint": "PredictiveQuery.materialize",
                "matrix": {"dtype": "arrow/binary", "template": "declared-memory-time_column-classification", "values": []},
                "input": {"constructor": "declared_table_data", "backend": "memory", "table_names": ["products"], "query": "PREDICT products.category FOR EACH products.product_id", "options": {"indices": [1, 2], "random_seed": 7, "model": {"$model": "KumoRelational"}}}
            },
            "initial_schema": {"function": "data_schema", "args": [
                {"products": {"$table": initial_sha}},
                json!({"products": {"product_id": {"dtype": "int64"}, "category": {"dtype": "str"}, "ordered_at": {"dtype": "datetime64[us]"}}}).to_string(),
                json!({"primary_keys": {"products": "product_id"}, "time_columns": {"products": "ordered_at"}, "relationships": null, "edits": []}).to_string()
            ]},
            "declared_table_data": {
                "frames": {"schema/products": {"$table": replaced_sha}},
                "tags": json!({"schema/products": {"product_id": {"dtype": "int64"}, "category": {"dtype": "str"}, "ordered_at": {"dtype": "binary[pyarrow]"}}}).to_string()
            },
            "host": {"year": 2026, "today": [2026, 10, 6], "pads_year": true, "tznames": ["UTC"], "timestamp_series_units": ["s", "ms", "us", "ns"], "numpy_text_prefix": false},
            "outcomes": []
        });
        (value, files)
    }

    fn execute(value: &Value, files: &BTreeMap<String, Vec<u8>>) -> Result<Value> {
        replay(
            value,
            &files
                .iter()
                .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
                .collect(),
            "public",
        )
    }

    #[test]
    fn declared_raw_frames_replace_data_without_replacing_schema() {
        let (value, files) = input();
        let observed = execute(&value, &files).unwrap();
        assert_eq!(observed["status"], "refused");
        assert_eq!(
            observed["error"]["message"],
            "Column 'ordered_at' received an incompatible semantic type (got dtype='binary' and stype='timestamp')"
        );
        let mut ordinary = value;
        ordinary
            .as_object_mut()
            .unwrap()
            .remove("declared_table_data");
        ordinary["recipe"]["input"]
            .as_object_mut()
            .unwrap()
            .remove("constructor");
        assert_ne!(execute(&ordinary, &files).unwrap(), observed);
    }

    #[test]
    fn declared_replacement_requires_exact_mode_and_input_domain() {
        let (value, files) = input();
        for mutation in [
            "missing", "ordinary", "unknown", "backend", "function", "matrix", "name", "tags",
            "frames", "extra",
        ] {
            let mut changed = value.clone();
            match mutation {
                "missing" => {
                    changed
                        .as_object_mut()
                        .unwrap()
                        .remove("declared_table_data");
                }
                "ordinary" => {
                    changed["recipe"]["input"]
                        .as_object_mut()
                        .unwrap()
                        .remove("constructor");
                }
                "unknown" => changed["recipe"]["input"]["constructor"] = json!("unknown"),
                "backend" => changed["recipe"]["input"]["backend"] = json!("sqlite"),
                "function" => changed["initial_schema"]["function"] = json!("schema_source_run"),
                "matrix" => {
                    changed["recipe"].as_object_mut().unwrap().remove("matrix");
                }
                "name" => changed["recipe"]["input"]["table_names"] = json!(["other"]),
                "tags" => changed["declared_table_data"]["tags"] = json!("{}"),
                "frames" => {
                    changed["declared_table_data"]["frames"] = json!({"schema/products": 7})
                }
                "extra" => changed["declared_table_data"]["schema"] = json!({"saved": true}),
                _ => unreachable!(),
            }
            assert!(execute(&changed, &files).is_err(), "{mutation}");
        }
        let mut missing_asset = files;
        missing_asset.remove(&format!("public/tables/{}.arrow", "2".repeat(64)));
        assert!(execute(&value, &missing_asset).is_err());
    }

    #[test]
    fn declared_frames_cannot_replace_initial_schema_inputs() {
        let (mut value, files) = input();
        value["initial_schema"]["args"][0]["products"] = json!({"$table": "2".repeat(64)});
        value["initial_schema"]["args"][1] = json!(json!({"products": {"product_id": {"dtype": "int64"}, "category": {"dtype": "str"}, "ordered_at": {"dtype": "binary[pyarrow]"}}}).to_string());
        assert_eq!(
            execute(&value, &files).unwrap_err(),
            "declared data requires successful initial schema construction"
        );
    }

    #[test]
    fn declared_frames_preserve_rows_and_column_domain() {
        let (value, files) = input();
        let path = format!("public/tables/{}.arrow", "2".repeat(64));
        let reader =
            arrow_ipc::reader::FileReader::try_new(Cursor::new(&files[&path]), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let renamed = RecordBatch::try_from_iter([
            ("product_id", batch.column(0).clone()),
            ("other", batch.column(1).clone()),
            ("ordered_at", batch.column(2).clone()),
        ])
        .unwrap();
        for changed in [
            batch.slice(0, 1),
            batch.project(&[0, 1]).unwrap(),
            batch.project(&[1, 0, 2]).unwrap(),
            renamed,
        ] {
            let mut altered = files.clone();
            altered.insert(path.clone(), arrow_file(&changed));
            assert_eq!(
                execute(&value, &altered).unwrap_err(),
                "declared frame names, columns, or rows changed"
            );
        }
    }
}
