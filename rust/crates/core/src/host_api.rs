// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The functions and runs a host calls to gather, send and read a request
//! (`tests/rust_core/request/CONTRACT.md`, "The binding"), by name, on plain
//! data. The bindings wrap [`call`], and a recording of their calls is
//! made again through it, in any language.
//!
//! Most take `state_json`, the JSON text of what the call is about, and
//! `tables`, a mapping of names to the Arrow tables the state names, each a
//! frame as `sdm_core.describe.describe_frames` described it (its tags are
//! in the state under `tags`).

use arrow_array::RecordBatch;
use sdm_protocol::{AsyncRun, Diagnostic, Run};
use sdm_request::error::Warnings;
use sdm_request::frames::Table;
use sdm_request::request::{FrameArg, FrameArgs};
use sdm_request::schemas::{DataSchema, Tables};
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::plain::Plain;

/// What reading dates takes from the host process.
pub type HostInfo = Host;

#[cfg(test)]
mod sequence_capability_tests {
    use super::*;

    #[test]
    fn exported_materialize_rejects_nonboolean_parser_capabilities() {
        for capability in [
            Value::Null,
            json!({}),
            json!([]),
            json!(0),
            json!(1),
            json!("prefix"),
        ] {
            let arguments = [
                Plain::Str(json!({"numpy_text_prefix": capability}).to_string()),
                Plain::Dict(Vec::new()),
            ];
            match call("materialize_run", &arguments, &Host::assumed()) {
                Err(refusal) => {
                    assert_eq!(refusal.error.class.as_deref(), Some("builtins.ValueError"));
                    assert_eq!(refusal.error.message, "invalid NumPy text policy");
                }
                Ok(_) => panic!("invalid capability was accepted"),
            }
        }
    }
}

/// What a call gives: a value, or a run to drive.
pub enum Called {
    Value(Plain),
    Run(Box<dyn Run<Plain> + Send>),
}

/// What a call refused with, and the warnings it gave before it did.
#[derive(Debug)]
pub struct Refused {
    pub error: Box<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
}

impl From<Diagnostic> for Refused {
    fn from(error: Diagnostic) -> Self {
        Self {
            error: Box::new(error),
            warnings: Vec::new(),
        }
    }
}

fn bad(message: impl Into<String>) -> Diagnostic {
    sdm_request::error::value_error(message)
}

fn text(value: Option<&Plain>, what: &str) -> Result<String, Diagnostic> {
    match value {
        Some(Plain::Str(s)) => Ok(s.clone()),
        _ => Err(bad(format!("{what} is text"))),
    }
}

fn json_arg(value: Option<&Plain>, what: &str) -> Result<Value, Diagnostic> {
    let text = text(value, what)?;
    let mut depth = 0usize;
    let mut deepest = 0;
    let mut quoted = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'[' | b'{' => {
                    depth += 1;
                    deepest = deepest.max(depth);
                    if depth > 512 {
                        return Err(bad(format!("{what} nests more than 512 levels deep")));
                    }
                }
                b']' | b'}' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    let parsed = if deepest < 120 {
        serde_json::from_str(&text)
    } else {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let mut decoder = serde_json::Deserializer::from_str(&text);
                decoder.disable_recursion_limit();
                let value = Value::deserialize(&mut decoder)?;
                decoder.end()?;
                Ok(value)
            })
            .map_err(|error| bad(error.to_string()))?
            .join()
            .map_err(|_| bad("the JSON reader failed"))?
    };
    parsed.map_err(|error| bad(format!("{what} is not JSON: {error}")))
}

/// One batch of a table's batches.
fn one_batch(
    schema: &arrow_schema::SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch, Diagnostic> {
    arrow_select::concat::concat_batches(schema, batches).map_err(|e| bad(e.to_string()))
}

/// The tables a call is given, with their tags.
fn tables_arg(value: Option<&Plain>, tags: &Value) -> Result<Tables, Diagnostic> {
    let mut batches = Vec::new();
    match value {
        None | Some(Plain::None) => {}
        Some(Plain::Dict(items)) => {
            for (name, table) in items {
                let batch = match table {
                    Plain::Table(schema, parts) => one_batch(schema, parts)?,
                    Plain::Batch(batch) => batch.clone(),
                    _ => return Err(bad(format!("table {name} is not a pyarrow table"))),
                };
                batches.push((name.clone(), batch));
            }
        }
        _ => return Err(bad("the tables are a dict of pyarrow tables")),
    }
    Ok(Tables::new(batches, tags))
}

/// Tables as plain data.
fn tables_value(tables: &Tables, only: &[&str]) -> Plain {
    Plain::Dict(
        tables
            .tables
            .iter()
            .filter(|(name, _)| only.is_empty() || only.contains(&name.as_str()))
            .map(|(name, table)| {
                (
                    name.clone(),
                    Plain::Table(table.batch.schema(), vec![table.batch.clone()]),
                )
            })
            .collect(),
    )
}

/// A state's warnings and value as a run's done value.
fn done(value: Plain) -> Plain {
    value
}

/// A frame argument of a state: `{"$frame": NAME}`, or what it was instead.
fn frame_arg(value: &Value, tables: &Tables) -> Result<FrameArg, Diagnostic> {
    if let Some(name) = value.get("$frame").and_then(Value::as_str) {
        return tables
            .get(name)
            .cloned()
            .map(FrameArg::Frame)
            .ok_or_else(|| bad(format!("no table {name} was given")));
    }
    Ok(FrameArg::Other(PyVal::decode(value).map_err(bad)?))
}

fn decode(value: Option<&Value>) -> Result<Option<PyVal>, Diagnostic> {
    match value {
        None => Ok(None),
        Some(v) => PyVal::decode(v).map(Some).map_err(bad),
    }
}

/// `request_build(state_json, tables)`: `Request(...)` built from frames.
fn request_build(args: &[Plain], host: &Host) -> Result<Called, Refused> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let mut tables = tables_arg(args.get(1), &tags)?;
    let given = state.get("args").cloned().unwrap_or_else(|| json!({}));
    let get = |k: &str| given.get(k);
    let related = match get("related_tables") {
        Some(Value::Object(map)) => map
            .iter()
            .map(|(name, v)| Ok((name.clone(), frame_arg(v, &tables)?)))
            .collect::<Result<Vec<_>, Diagnostic>>()?,
        _ => Vec::new(),
    };
    let schema = match get("schema") {
        Some(Value::Object(_)) => {
            Some(DataSchema::from_json(get("schema").unwrap_or(&Value::Null)).map_err(bad)?)
        }
        _ => None,
    };
    let object = |k: &str| {
        get(k)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    };
    let args = FrameArgs {
        context: frame_arg(get("context").unwrap_or(&Value::Null), &tables)?,
        predict: frame_arg(get("predict").unwrap_or(&Value::Null), &tables)?,
        target: decode(get("target"))?.unwrap_or(PyVal::None),
        task: decode(get("task"))?.unwrap_or(PyVal::None),
        related,
        schema,
        entity_table: get("entity_table")
            .and_then(Value::as_str)
            .unwrap_or("instance")
            .to_string(),
        inference: object("inference"),
        task_options: object("task_options"),
    };
    let mut said = Warnings::default();
    let built = sdm_request::request::from_frames(args, &mut tables, host, &mut said);
    let warnings = said.take();
    match built {
        Ok(request) => {
            let mut out = Tables::default();
            let state = request.to_json(&mut out);
            for (name, table) in &tables.tables {
                if name.starts_with("schema/") && out.get(name).is_none() {
                    out.insert(name, table.clone());
                }
            }
            let mut answer = Map::new();
            answer.insert("request".into(), state);
            answer.insert("tags".into(), out.tags());
            answer.insert(
                "warnings".into(),
                Value::Array(
                    warnings
                        .iter()
                        .map(sdm_request::error::warning_json)
                        .collect(),
                ),
            );
            Ok(Called::Value(Plain::Tuple(vec![
                Plain::Str(Value::Object(answer).to_string()),
                tables_value(&out, &[]),
            ])))
        }
        Err(error) => Err(Refused {
            error: Box::new(error),
            warnings,
        }),
    }
}

/// `request_materialized(state_json, tables)`: `Request.materialized(...)`,
/// a request of the envelopes a caller holds. The state names `payloads`,
/// `schema` (a schema handle's JSON, its frames among `tables`), `query`,
/// `indices`, `entity_table` (`null` where the caller named none),
/// `windowed` and `requested_inference` (`null` where not given), each value
/// as tagged JSON; done with `(answer_json, tables)` as `request_build`.
/// Entire envelopes use the shared `PyVal` codec, including escaped user
/// dictionaries. Semantic reads decode these values; the original tagged
/// envelope stays distinct from constructor-coerced options and inference.
/// Model-call bodies retain the codec until the host decodes the complete
/// body immediately before its transport performs ordinary JSON conversion.
/// Cyclic objects remain owned by the host; the shared cycle marker can
/// carry a call-local reference number which the host resolves to that
/// original object, preserving its transport's refusal and its timing.
fn request_materialized(
    args: &[Plain],
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let payloads = match state.get("payloads") {
        Some(Value::Array(items)) => items.clone(),
        _ => return Err(bad("the payloads are a list of envelopes")),
    };
    let schema = DataSchema::from_json(state.get("schema").unwrap_or(&Value::Null)).map_err(bad)?;
    let indices = PyVal::decode(state.get("indices").unwrap_or(&json!([]))).map_err(bad)?;
    let optional = |name: &str| decode(state.get(name).filter(|value| !value.is_null()));
    let request = sdm_request::relational::from_envelope_values(
        payloads,
        schema,
        optional("query")?,
        indices,
        optional("entity_table")?,
        PyVal::decode(state.get("windowed").unwrap_or(&json!(false))).map_err(bad)?,
        (
            state
                .get("requested_inference")
                .cloned()
                .unwrap_or_default(),
            protocols,
        ),
    )?;
    let mut out = Tables::default();
    let built = request.to_json(&mut out);
    for (name, table) in &tables.tables {
        if name.starts_with("schema/") && out.get(name).is_none() {
            out.insert(name, table.clone());
        }
    }
    let answer = json!({"request": built, "tags": out.tags(), "warnings": []});
    Ok(Called::Value(Plain::Tuple(vec![
        Plain::Str(answer.to_string()),
        tables_value(&out, &[]),
    ])))
}

/// `query_text(query)`: the query a `PredictiveQuery` is asked, refused as
/// today where it is not text.
fn query_text(args: &[Plain]) -> Result<Called, Diagnostic> {
    match args.first() {
        Some(Plain::Str(s)) if !s.trim().is_empty() => Ok(Called::Value(Plain::Str(s.clone()))),
        _ => Err(sdm_request::error::invalid(
            "query must be a non-empty string",
        )),
    }
}

/// `materialize_run(state_json, tables)`: `materialize(**args)` of a query
/// over a schema, the schema's frames as they stand and, after the query's
/// first relational gather, its model (`model`, its graph's tables named
/// `model/NAME`); done with `(answer_json, tables)`, the request and the
/// model as this gather left it. A gather refused after the model was built
/// is done too, with the refusal in place of the request (`"refused"`, as
/// `refusal_json` writes it), so the host keeps the model before raising it.
fn materialize_run(args: &[Plain], host: &Host) -> Result<Called, Diagnostic> {
    materialize_run_with_indices(args, host, None)
}

pub fn materialize_run_with_indices(
    args: &[Plain],
    host: &Host,
    indices: Option<sdm_request::materialize::SequenceProtocol>,
) -> Result<Called, Diagnostic> {
    materialize_run_with_sequences(args, host, indices, None)
}

pub fn materialize_run_with_sequences(
    args: &[Plain],
    host: &Host,
    indices: Option<sdm_request::materialize::SequenceProtocol>,
    candidates: Option<sdm_request::materialize::SequenceProtocol>,
) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let mut host = host.clone();
    if let Some(policy) = state.get("numpy_text_prefix") {
        host.numpy_text_prefix = policy
            .as_bool()
            .ok_or_else(|| bad("invalid NumPy text policy"))?;
    }
    if let Some(facts) = state.get("timestamp_series_units") {
        host.timestamp_series_units =
            sdm_schema::dt::pandas::TimestampSeriesUnits::from_json(facts).map_err(bad)?;
    }
    let run = AsyncRun::new(move |asker| async move {
        let mut said = Warnings::live(asker.clone());
        let mut tables = tables;
        let schema_handle =
            DataSchema::from_json(state.get("schema").unwrap_or(&Value::Null)).map_err(bad)?;
        let schema = schema_handle.build(&tables, &host)?;
        let query = state
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let given = state.get("args").cloned().unwrap_or_else(|| json!({}));
        let mut margs = sdm_request::materialize::Args {
            indices_protocol: indices,
            candidates_protocol: candidates,
            ..sdm_request::materialize::Args::default()
        };
        if let Some(Value::Object(map)) = given.as_object().map(|m| Value::Object(m.clone())) {
            for (key, value) in map {
                match key.as_str() {
                    "indices" => margs.indices = decode(Some(&value))?,
                    "predict" => margs.predict = Some(frame_arg(&value, &tables)?),
                    "num_context_samples" => margs.num_context_samples = decode(Some(&value))?,
                    "model" => margs.model = decode(Some(&value))?,
                    "candidates" => margs.candidates = decode(Some(&value))?,
                    "task" => margs.task = decode(Some(&value))?,
                    other => margs.options.push((
                        other.to_string(),
                        decode(Some(&value))?.unwrap_or(PyVal::None),
                    )),
                }
            }
        }
        let mut model = state.get("model").filter(|m| !m.is_null()).map(|json| {
            sdm_request::relational::Model {
                json: json.clone(),
                tables: tables
                    .tables
                    .iter()
                    .filter_map(|(name, table)| {
                        let graph = name.strip_prefix("model/")?;
                        Some((graph.to_string(), table.batch.clone()))
                    })
                    .collect(),
            }
        });
        let mut coverage_warned = state
            .get("coverage_warned")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let gathered = sdm_request::materialize::materialize(
            &asker,
            &query,
            sdm_request::materialize::Gathering {
                handle: &schema_handle,
                schema: &schema,
                tables: &mut tables,
                host: &host,
                model: &mut model,
                coverage_warned: &mut coverage_warned,
            },
            margs,
            &mut said,
        )
        .await;
        let mut out = Tables::default();
        let mut answer = Map::new();
        match gathered {
            Ok(request) => {
                answer.insert("request".into(), request.to_json(&mut out));
            }
            // A gather refused after the query's model was built leaves that
            // model with the query, as today's does; the host keeps it and
            // then raises the refusal.
            Err(refusal) if model.is_some() && refusal.class.is_some() => {
                answer.insert("refused".into(), sdm_request::error::refusal_json(&refusal));
            }
            Err(refusal) => return Err(refusal),
        }
        answer.insert("tags".into(), out.tags());
        answer.insert(
            "model".into(),
            model.as_ref().map_or(Value::Null, |m| m.json.clone()),
        );
        if state.get("coverage_warned").is_some() {
            answer.insert("coverage_warned".into(), json!(coverage_warned));
        }
        let answer = Value::Object(answer);
        for (name, batch) in model.map(|m| m.tables).unwrap_or_default() {
            out.insert(
                &sdm_request::relational::Model::table_key(&name),
                Table::new(batch, None),
            );
        }
        Ok(done(Plain::Tuple(vec![
            Plain::Str(answer.to_string()),
            tables_value(&out, &[]),
        ])))
    });
    Ok(Called::Run(Box::new(run)))
}

/// `predict_run(state_json, tables)`: `SDMClient.predict(request, model=,
/// outputs=)`; done with the result's plain JSON.
fn predict_run(
    args: &[Plain],
    host: &Host,
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let request = sdm_request::request::Request::from_json(
        state.get("request").unwrap_or(&Value::Null),
        &tables,
    )
    .map_err(bad)?
    .with_protocols(protocols);
    let asked = sdm_request::client::Asked {
        model: decode(state.get("model"))?.unwrap_or(PyVal::None),
        outputs: outputs_of(state.get("outputs"))?,
    };
    let host = host.clone();
    let run = AsyncRun::new(move |asker| async move {
        let mut said = Warnings::live(asker.clone());
        let result =
            sdm_request::client::predict(&asker, &request, &tables, &asked, &host, &mut said)
                .await?;
        Ok(Plain::Str(
            sdm_request::layout::encode_result(&result).to_string(),
        ))
    });
    Ok(Called::Run(Box::new(run)))
}

/// `fit_run(state_json, tables)`: `SDMClient.fit(request, model=,
/// outputs=, holdout=)`; done with the fitted model's plain JSON.
fn fit_run(
    args: &[Plain],
    host: &Host,
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Diagnostic> {
    fit_run_with_protocols(args, host, protocols, None)
}

pub fn fit_run_with_holdout(
    args: &[Plain],
    host: &Host,
    protocol: Option<sdm_request::materialize::SequenceProtocol>,
) -> Result<Called, Diagnostic> {
    fit_run_with_protocols(args, host, None, protocol)
}

pub fn fit_run_with_protocols(
    args: &[Plain],
    host: &Host,
    protocols: Option<sdm_request::pyops::HostProtocols>,
    protocol: Option<sdm_request::materialize::SequenceProtocol>,
) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let request = sdm_request::request::Request::from_json(
        state.get("request").unwrap_or(&Value::Null),
        &tables,
    )
    .map_err(bad)?
    .with_protocols(protocols);
    let asked = sdm_request::client::Asked {
        model: decode(state.get("model"))?.unwrap_or(PyVal::None),
        outputs: outputs_of(state.get("outputs"))?,
    };
    let holdout = decode(state.get("holdout"))?;
    let host = host.clone();
    let run = AsyncRun::new(move |asker| async move {
        let mut said = Warnings::live(asker.clone());
        let fitted = sdm_request::sessions::fit(
            &asker,
            &request,
            &tables,
            sdm_request::sessions::Fit {
                asked: &asked,
                holdout: holdout.as_ref(),
                holdout_protocol: protocol.as_ref(),
            },
            &host,
            &mut said,
        )
        .await?;
        Ok(Plain::Str(fitted.to_json().to_string()))
    });
    Ok(Called::Run(Box::new(run)))
}

/// `session_predict_run(state_json, tables)`: `FittedModel.predict(request)`,
/// the fitted model as `fit_run` gave it; done with the result's plain JSON.
fn session_predict_run(
    args: &[Plain],
    host: &Host,
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let request = sdm_request::request::Request::from_json(
        state.get("request").unwrap_or(&Value::Null),
        &tables,
    )
    .map_err(bad)?
    .with_protocols(protocols);
    let fitted =
        sdm_request::sessions::Fitted::from_json(state.get("fitted").unwrap_or(&Value::Null))
            .map_err(bad)?;
    let host = host.clone();
    let run = AsyncRun::new(move |asker| async move {
        let mut said = Warnings::live(asker.clone());
        let result = sdm_request::sessions::session_predict(
            &asker, &fitted, &request, &tables, &host, &mut said,
        )
        .await?;
        Ok(Plain::Str(
            sdm_request::layout::encode_result(&result).to_string(),
        ))
    });
    Ok(Called::Run(Box::new(run)))
}

/// The output fields a call names, or `None`, refused as the client
/// refuses anything but a sequence of names.
fn outputs_of(value: Option<&Value>) -> Result<Option<Vec<String>>, Diagnostic> {
    let given = decode(value)?.unwrap_or(PyVal::None);
    sdm_request::client::require_output_names(&given)
}

/// `data_schema(tables, column_tags_json, options_json)`:
/// `Schema.from_data(tables, **options)`, as the request core holds it;
/// `(handle_json, warnings_json)`, the handle `{"error": {code, message}}`
/// where it was refused. The handle says what the schema says; the frames
/// are given again, as they stand, at each call that reads them.
fn data_schema(args: &[Plain], host: &Host) -> Result<Called, Diagnostic> {
    let tags: Value = json_arg(args.get(1), "the tags")?;
    let options: Value = json_arg(args.get(2), "the options")?;
    sdm_schema::binding::data_args(&options).map_err(bad)?;
    let given = tables_arg(args.first(), &tags)?;
    let names: Vec<String> = given.tables.iter().map(|(n, _)| n.clone()).collect();
    let mut tables = Tables::default();
    for (name, table) in given.tables {
        tables.insert(&DataSchema::frame_key(&name), table);
    }
    let handle = DataSchema::over_frames(names, options);
    let (found, said) = match handle.describe(&tables, host).map_err(bad)? {
        Ok((schema, said)) => (handle.saying(schema).to_json(), said),
        Err((refusal, said)) => (json!({"error": refusal.to_json()}), said),
    };
    let warnings = sdm_schema::client::warnings_json(&said.warnings);
    Ok(Called::Value(Plain::Tuple(vec![
        Plain::Str(found.to_string()),
        Plain::Str(warnings.to_string()),
    ])))
}

/// `schema_source_run(backend, source_json, args_json)`:
/// `Schema.from_source(source, **args)`, as a run; done with the schema
/// handle's JSON (or `{"error": {code, message}}`) and the warnings' JSON.
fn schema_source_run(args: &[Plain], host: &Host) -> Result<Called, Diagnostic> {
    let backend = text(args.first(), "the backend")?;
    let source = json_arg(args.get(1), "the source")?;
    let given = json_arg(args.get(2), "the arguments")?;
    let (info, source_args) =
        sdm_schema::binding::source_args(&backend, &source, &given).map_err(bad)?;
    let mut described = source.clone();
    if let Some(object) = described.as_object_mut() {
        object.insert("backend".into(), json!(backend));
    }
    let host = host.clone();
    let run = AsyncRun::new(move |asker| async move {
        let db = sdm_schema::engine::Db(asker);
        let mut ctx = sdm_schema::engine::Ctx {
            db: &db,
            host: &host,
            warnings: sdm_schema::engine::Warnings::default(),
        };
        let mut said = sdm_schema::client::Said::default();
        let mut kept = Vec::new();
        let result = sdm_schema::source::from_source_keeping(
            &mut ctx,
            &info,
            &source_args,
            &mut said,
            &mut kept,
        )
        .await;
        let found = match result {
            Ok(schema) => DataSchema {
                names: schema.tables.iter().map(|t| t.name.clone()).collect(),
                options: json!({}),
                described: None,
                source: Some(Box::new(sdm_request::schemas::SourceSchema {
                    source: described,
                    schema,
                    engine: kept,
                })),
            }
            .to_json(),
            Err(refusal) => {
                json!({"error": refusal.to_json()})
            }
        };
        let warnings = sdm_schema::client::warnings_json(&said.warnings);
        Ok(Plain::Tuple(vec![
            Plain::Str(found.to_string()),
            Plain::Str(warnings.to_string()),
        ]))
    });
    Ok(Called::Run(Box::new(run)))
}

/// `schema_edit(state_json, tables)`: an edit a host makes on its own
/// objects of a schema (`sdm_schema::edits`). The state is `{"scope":
/// "column" | "table" | "schema", "schema": SCHEMA, "op": EDIT}`, `SCHEMA`
/// as `schema_json` writes it; a table whose rows an edit reads is given
/// as `schema/NAME`. `(schema_json, warnings_json)`: what the scope says
/// afterwards, and what the edit said; refused with the warnings given
/// before the refusal.
fn schema_edit(args: &[Plain], host: &Host) -> Result<Called, Refused> {
    let _ = host;
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let scope = state
        .get("scope")
        .and_then(Value::as_str)
        .and_then(sdm_schema::edits::Scope::parse)
        .ok_or_else(|| bad("the scope is column, table or schema"))?;
    let mut schema = sdm_request::schemas::schema_of(state.get("schema").unwrap_or(&Value::Null))
        .map_err(bad)?;
    for table in &mut schema.tables {
        let key = sdm_request::schemas::DataSchema::frame_key(&table.name);
        if let Some(frame) = tables.get(&key) {
            table.data = Some(std::sync::Arc::new(frame.schema_frame().map_err(bad)?));
        }
    }
    let op = state.get("op").cloned().unwrap_or(Value::Null);
    let mut said = sdm_schema::client::Said::default();
    let done = sdm_schema::edits::apply(&mut schema, scope, &op, &mut said);
    let warnings: Vec<Diagnostic> = said
        .warnings
        .iter()
        .map(|w| Diagnostic::warning(&w.category, w.message.clone()))
        .collect();
    match done {
        Ok(()) => Ok(Called::Value(Plain::Tuple(vec![
            Plain::Str(sdm_request::schemas::schema_json(&schema).to_string()),
            Plain::Str(sdm_schema::client::warnings_json(&said.warnings).to_string()),
        ]))),
        Err(refusal) => Err(Refused {
            error: Box::new(sdm_request::schemas::schema_refusal(&refusal)),
            warnings,
        }),
    }
}

/// `model_spec(model_json)`: `spec_for(model)`, a model id or a handle
/// (`{"$model": CLASS}`), as the registry names it: `{"model_id",
/// "profile", "shape": {CAPABILITY: bool}}`, or today's refusal.
fn model_spec(args: &[Plain]) -> Result<Called, Diagnostic> {
    let given = json_arg(args.first(), "the model")?;
    let model = PyVal::decode(&given).map_err(bad)?;
    let spec = sdm_request::models::spec_for(&model)?;
    let shape: Map<String, Value> = [
        "relationships",
        "multiple_tables",
        "link_prediction",
        "forecasting",
        "static",
        "temporal",
    ]
    .iter()
    .map(|c| ((*c).to_string(), json!(spec.profile.supports(c))))
    .collect();
    let found = json!({
        "model_id": spec.model_id,
        "profile": spec.profile.name(),
        "model": spec.profile.model(),
        "shape": shape,
    });
    Ok(Called::Value(Plain::Str(found.to_string())))
}

/// `gather_graph_run(state_json, tables)`: the graph a gather of a schema
/// over a source samples, as a run (reading the source where the schema
/// holds nothing it read, as a schema built by hand does); done with
/// `{"backend", "tables": [[NAME, TABLE]], "edges": [[SRC, FKEY, DST]]}`,
/// each table as the core keeps one between calls.
fn gather_graph_run(args: &[Plain], host: &Host) -> Result<Called, Diagnostic> {
    let state = json_arg(args.first(), "the state")?;
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(1), &tags)?;
    let host = host.clone();
    let run = AsyncRun::new(move |asker| async move {
        let handle =
            DataSchema::from_json(state.get("schema").unwrap_or(&Value::Null)).map_err(bad)?;
        let Some(source) = &handle.source else {
            return Err(bad("the schema is over frames, not a source"));
        };
        let mut schema = handle.build(&tables, &host)?;
        let mut checked = sdm_schema::client::Said::default();
        schema
            .check_consistent(&mut checked)
            .map_err(|r| sdm_request::schemas::schema_refusal(&r))?;
        let db = sdm_schema::engine::Db(asker);
        let mut ctx = sdm_schema::engine::Ctx {
            db: &db,
            host: &host,
            warnings: sdm_schema::engine::Warnings::default(),
        };
        let graph = sdm_request::source_gather::build_graph(&mut ctx, source, &schema)
            .await
            .map_err(|refused| match refused {
                sdm_request::source_gather::Refused::Client(diagnostic) => diagnostic,
                sdm_request::source_gather::Refused::Engine(error) => {
                    sdm_request::error::value_error(format!("{}: {}", error.class, error.message))
                }
            })?;
        let found = json!({
            "backend": source.backend().map_err(bad)?.source_name(),
            "tables": graph
                .tables
                .iter()
                .map(|t| json!([t.name, t.to_json_without_sample()]))
                .collect::<Vec<_>>(),
            "edges": graph
                .edges
                .iter()
                .map(|e| json!([e.src_table, e.fkey, e.dst_table]))
                .collect::<Vec<_>>(),
        });
        Ok(Plain::Str(found.to_string()))
    });
    Ok(Called::Run(Box::new(run)))
}

/// `request_op(op, state_json, tables)`: what a request's wrapper asks of
/// the core about a request it holds (`request_state`'s JSON and frames):
/// `question` (`request._question`, as JSON), `refuse_if_gathered` (with
/// `half` in the state; `None`, or today's refusal) and `entity_table_in`
/// (of the state's `payloads`).
fn request_op(
    args: &[Plain],
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Diagnostic> {
    let op = text(args.first(), "the operation")?;
    let state = json_arg(args.get(1), "the state")?;
    if op == "entity_table_in" {
        let payloads = state
            .get("payloads")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        return Ok(Called::Value(Plain::Str(
            sdm_request::relational::entity_table_in(&payloads),
        )));
    }
    let tags = state.get("tags").cloned().unwrap_or_else(|| json!({}));
    let tables = tables_arg(args.get(2), &tags)?;
    let request = sdm_request::request::Request::from_json(
        state.get("request").unwrap_or(&Value::Null),
        &tables,
    )
    .map_err(bad)?
    .with_protocols(protocols);
    match op.as_str() {
        "question" => Ok(Called::Value(Plain::Str(
            sdm_request::sessions::question_of(&request)?
                .to_json()
                .to_string(),
        ))),
        "refuse_if_gathered" => {
            let half = state
                .get("half")
                .and_then(Value::as_str)
                .unwrap_or("context");
            request.refuse_if_gathered(half)?;
            Ok(Called::Value(Plain::None))
        }
        other => Err(bad(format!("no request operation {other}"))),
    }
}

/// A result's plain JSON, read.
fn result_arg(value: Option<&Plain>) -> Result<sdm_request::result::Result_, Diagnostic> {
    let text = text(value, "a result")?;
    let parsed: Value =
        serde_json::from_str(&text).map_err(|e| bad(format!("the result is not JSON: {e}")))?;
    sdm_request::layout::decode_result(&parsed).map_err(|e| bad(format!("not a result: {e}")))
}

/// `flatten(result_json)`: the layout `to_pandas()` is built from.
fn flatten(args: &[Plain]) -> Result<Called, Diagnostic> {
    let result = result_arg(args.first())?;
    if sdm_request::layout::requires_host_columns(&result) {
        return Err(bad("result presentation requires host column facts"));
    }
    let batch = sdm_request::layout::layout(&result);
    Ok(Called::Value(Plain::Table(batch.schema(), vec![batch])))
}

/// `merge(parts_json)`: `PredictionResult.merge(parts)`'s plain JSON.
fn merge(args: &[Plain]) -> Result<Called, Diagnostic> {
    let parts = match args.first() {
        Some(Plain::List(items) | Plain::Tuple(items)) => items
            .iter()
            .map(|p| result_arg(Some(p)))
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(bad("the parts are a list of results")),
    };
    let merged = sdm_request::result::Result_::merge(parts)?;
    Ok(Called::Value(Plain::Str(
        sdm_request::layout::encode_result(&merged).to_string(),
    )))
}

/// `absent(result_json, requested)`: which of `requested` the result does
/// not carry.
fn absent(args: &[Plain]) -> Result<Called, Diagnostic> {
    let result = result_arg(args.first())?;
    let requested: Vec<String> = match args.get(1) {
        Some(Plain::List(items) | Plain::Tuple(items)) => items
            .iter()
            .map(|p| {
                p.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad("a name is text"))
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(bad("the names are a list")),
    };
    Ok(Called::Value(Plain::List(
        result
            .absent(&requested)
            .into_iter()
            .map(Plain::Str)
            .collect(),
    )))
}

/// Call the host function `function` with `args`.
pub fn call(function: &str, args: &[Plain], host: &Host) -> Result<Called, Refused> {
    call_with_protocols(function, args, host, None)
}

/// Call with language protocol facts owned by this operation, not serialized state.
pub fn call_with_protocols(
    function: &str,
    args: &[Plain],
    host: &Host,
    protocols: Option<sdm_request::pyops::HostProtocols>,
) -> Result<Called, Refused> {
    match function {
        "query_text" => Ok(query_text(args)?),
        "request_build" => request_build(args, host),
        "request_materialized" => Ok(request_materialized(args, protocols)?),
        "materialize_run" => Ok(materialize_run(args, host)?),
        "predict_run" => Ok(predict_run(args, host, protocols)?),
        "fit_run" => Ok(fit_run(args, host, protocols)?),
        "session_predict_run" => Ok(session_predict_run(args, host, protocols)?),
        "schema_source_run" => Ok(schema_source_run(args, host)?),
        "data_schema" => Ok(data_schema(args, host)?),
        "schema_edit" => schema_edit(args, host),
        "model_spec" => Ok(model_spec(args)?),
        "request_op" => Ok(request_op(args, protocols)?),
        "gather_graph_run" => Ok(gather_graph_run(args, host)?),
        "flatten" => Ok(flatten(args)?),
        "merge" => Ok(merge(args)?),
        "absent" => Ok(absent(args)?),
        other => Err(bad(format!("no host function {other}")).into()),
    }
}

/// A table of a plain value, for a caller holding one.
pub fn table_of(value: &Plain) -> Option<Table> {
    match value {
        Plain::Table(schema, batches) => {
            one_batch(schema, batches).ok().map(|b| Table::new(b, None))
        }
        _ => None,
    }
}
