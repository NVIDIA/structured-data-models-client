// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request fixture operations that take a whole flow
//! (`tests/parity/_fixture_ops.py`): `client.run`, public calls over the
//! input's frames with the model answered from the input's responses, and
//! `tabular.task`, `decompose.parts` and `payload.feature_column`, which
//! each gather one request.
//!
//! The calls are made as a host makes them (`scripts/parity/request_host.py`):
//! through [`crate::host_api::call`], with plain data, each run driven to its
//! end, each model call answered by a stand-in of the client's transport
//! (`Answering`), and the stand-in's answers turned into the errors the
//! transport raises.

use std::collections::VecDeque;

use arrow_array::RecordBatch;
use sdm_protocol::{Diagnostic, Endpoint, ModelCall, Outcome, Run, Step};
use sdm_request::error::{STRUCTURED, invalid};
use sdm_request::fixture::{InputError, answer, refusal};
use sdm_request::frames::Table;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::host_api::{Called, call};
use crate::plain::Plain;

mod carry;
pub(crate) use carry::Inputs as CarryInputs;

fn bad(message: impl Into<String>) -> InputError {
    InputError(message.into())
}

/// Frames by name, as a host describes them, with their tags.
#[derive(Clone, Debug, Default)]
struct Frames {
    tables: Vec<(String, RecordBatch)>,
    tags: Map<String, Value>,
}

impl Frames {
    /// Those frames whose name starts with `prefix`, and the others.
    fn split(&self, prefix: &str) -> (Frames, Frames) {
        let (mut named, mut others) = (Frames::default(), Frames::default());
        for (name, batch) in &self.tables {
            let tag = self.tags.get(name).cloned().unwrap_or(json!({}));
            let into = if name.starts_with(prefix) {
                &mut named
            } else {
                &mut others
            };
            into.add_in_place(name, batch.clone(), tag);
        }
        (named, others)
    }

    fn add(&self, other: &Frames) -> Frames {
        let mut out = self.clone();
        for (name, batch) in &other.tables {
            match out.tables.iter_mut().find(|(n, _)| n == name) {
                Some(entry) => entry.1 = batch.clone(),
                None => out.tables.push((name.clone(), batch.clone())),
            }
        }
        for (k, v) in &other.tags {
            out.tags.insert(k.clone(), v.clone());
        }
        out
    }

    fn insert(&mut self, name: &str, table: &Table) {
        self.add_in_place(name, table.batch.clone(), table.tags_json());
    }

    fn add_in_place(&mut self, name: &str, batch: RecordBatch, tag: Value) {
        match self.tables.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = batch,
            None => self.tables.push((name.to_string(), batch)),
        }
        self.tags.insert(name.to_string(), tag);
    }

    fn remove(&mut self, name: &str) {
        self.tables.retain(|(n, _)| n != name);
        self.tags.remove(name);
    }

    fn holds(&self, name: &str) -> bool {
        self.tables.iter().any(|(n, _)| n == name)
    }

    fn plain(&self) -> Plain {
        Plain::Dict(
            self.tables
                .iter()
                .map(|(n, b)| (n.clone(), Plain::Batch(b.clone())))
                .collect(),
        )
    }

    /// The tables a run gave back, with their tags.
    fn of(value: &Plain, tags: &Value) -> Frames {
        let mut out = Frames::default();
        if let Plain::Dict(items) = value {
            for (name, table) in items {
                let batch = match table {
                    Plain::Batch(b) => Some(b.clone()),
                    Plain::Table(schema, batches) => {
                        arrow_select::concat::concat_batches(schema, batches).ok()
                    }
                    _ => None,
                };
                if let Some(batch) = batch {
                    out.tables.push((name.clone(), batch));
                }
            }
        }
        if let Value::Object(map) = tags {
            out.tags = map.clone();
        }
        out
    }
}

/// A schema the core described.
#[derive(Clone, Debug)]
struct SchemaH {
    state: Value,
    frames: Frames,
}

/// A request the core built or gathered.
#[derive(Clone, Debug)]
struct RequestH {
    state: Value,
    frames: Frames,
}

/// A `FittedModel`, as `fit_run` gave it.
#[derive(Clone, Debug)]
struct FittedH {
    state: Value,
    closed: bool,
}

/// The stand-in of the client's transport: each request answered by the
/// next stored response, after checking it is the request stored.
#[derive(Debug, Default)]
struct Answering {
    responses: Vec<Value>,
    requests: Vec<Value>,
    bodies: Vec<Value>,
}

/// `NimRequestError`'s module-qualified class.
const NIM_REQUEST: &str = "sdm_client.errors.NimRequestError";

/// What the transport raises, as a host hands it to a run.
fn model_error(
    class: &str,
    status: Option<i64>,
    code: Option<&str>,
    message: &str,
    details: Map<String, Value>,
) -> String {
    json!({
        "class": class,
        "status": status,
        "code": code,
        "message": message,
        "details": details,
    })
    .to_string()
}

/// `_snippet(text)`: one line, at most 500 characters.
fn snippet(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let count = text.chars().count();
    if count <= 500 {
        return text;
    }
    let cut: String = text.chars().take(500).collect();
    format!("{cut}... [truncated, {count} chars total]")
}

/// `format_invalid_params(params)`.
fn format_invalid_params(params: Option<&Value>) -> String {
    let Some(Value::Array(params)) = params else {
        return String::new();
    };
    let mut entries = Vec::new();
    for param in params.iter().take(10) {
        let Value::Object(param) = param else {
            continue;
        };
        let text = |k: &str| match param.get(k) {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s.is_empty() => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Bool(false)) => None,
            Some(other) => Some(sdm_request::pyops::py_str(&sdm_request::keys::json_cell(
                other,
            ))),
        };
        match (text("name"), text("reason")) {
            (Some(name), Some(reason)) => entries.push(format!("{name}: {reason}")),
            (Some(one), None) | (None, Some(one)) => entries.push(one),
            (None, None) => {}
        }
    }
    if entries.is_empty() {
        return String::new();
    }
    let omitted = params.len() as i64 - entries.len() as i64;
    let more = if omitted > 0 {
        format!(" (and {omitted} more)")
    } else {
        String::new()
    };
    format!(" {}{more}", entries.join("; "))
}

/// `json.dumps(value)` with Python's defaults: `", "` and `": "` between
/// items, every character outside ASCII escaped, floats by `repr`. The
/// stand-in's responses are written so (`_fixture_ops._Response`).
fn py_dumps(value: &Value) -> String {
    let mut out = String::new();
    py_dump(value, &mut out);
    out
}

fn py_dump(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => {
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&sdm_schema::py::float_repr(f));
                }
            }
            _ => out.push_str(&n.to_string()),
        },
        Value::String(text) => py_dump_str(text, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dump(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dump_str(key, out);
                out.push_str(": ");
                py_dump(item, out);
            }
            out.push('}');
        }
    }
}

fn py_dump_str(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// `_to_nim_error(status_code, content)`, `content` the stand-in's body as
/// it writes it.
fn nim_error(status: i64, body: &Value) -> String {
    let content = match body {
        Value::Null => String::new(),
        other => py_dumps(other),
    };
    let object = body.as_object().cloned().unwrap_or_default();
    let code = object.get("code").cloned().unwrap_or(Value::Null);
    let truthy = |v: Option<&Value>| match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(v.clone()),
    };
    let reported = truthy(object.get("detail"))
        .or_else(|| truthy(object.get("title")))
        .unwrap_or_else(|| Value::String(content.clone()));
    let reported_text = match &reported {
        Value::String(s) => s.clone(),
        other => sdm_request::pyops::py_str(&sdm_request::keys::json_cell(other)),
    };
    let mut message = if reported_text.is_empty() {
        format!("NIM request failed with status {status}")
    } else {
        snippet(&reported_text)
    };
    message.push_str(&format_invalid_params(object.get("invalid_params")));
    let mut details: Map<String, Value> = object
        .iter()
        .filter(|(k, _)| {
            !matches!(
                k.as_str(),
                "code" | "detail" | "title" | "type" | "status" | "instance"
            )
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut code_text = match &code {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    if status >= 500
        && let Value::String(said) = &reported
    {
        let wording = "failed to collect any context examples";
        if said.to_lowercase().contains(wording) {
            details.insert("endpoint_code".into(), code.clone());
            details.insert("diagnosis".into(), json!("EMPTY_CONTEXT"));
            code_text = Some("INVALID_REQUEST".to_string());
            let found = message.to_lowercase().find(wording);
            let kept = match found {
                Some(at) => message[at..].trim().to_string(),
                None => message.clone(),
            };
            let leaked = regex::Regex::new(
                r"\b(?:failed unexpectedly with\s+)?[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\s*:\s*",
            )
            .expect("pattern compiles");
            let alone = leaked.replace_all(&kept, "").trim().to_string();
            message = if alone.is_empty() { message } else { alone };
        }
    }
    model_error(
        NIM_REQUEST,
        Some(status),
        code_text.as_deref(),
        &message,
        details,
    )
}

impl Answering {
    fn record_external(&mut self, call: &ModelCall) -> Result<(), Diagnostic> {
        let session = call.session_id.as_deref().unwrap_or_default();
        let (method, path) = match call.endpoint {
            Endpoint::Predict => ("POST", "/v1/predictions".to_string()),
            Endpoint::SessionCreate => ("POST", "/v1/sessions".to_string()),
            Endpoint::SessionPredict => ("POST", format!("/v1/sessions/{session}/predictions")),
            Endpoint::SessionDelete => ("DELETE", format!("/v1/sessions/{session}")),
        };
        self.requests.push(json!([method, path]));
        if method == "POST" {
            let body: Value = serde_json::from_str(call.body.as_deref().unwrap_or_default())
                .map_err(|_| invalid("invalid computed model body"))?;
            let body = PyVal::decode(&body).map_err(invalid)?;
            let body = sdm_request::wire::json_ready(&body)?;
            self.bodies.push(sdm_request::values::encode(&body));
        }
        Ok(())
    }

    /// The request `method path`, answered.
    fn serve(&mut self, method: &str, path: &str) -> Result<(i64, Value), String> {
        let at = self.requests.len();
        self.requests.push(json!([method, path]));
        let Some(given) = self.responses.get(at) else {
            return Err(model_error(
                "builtins.AssertionError",
                None,
                None,
                &format!("request {at} ({method} {path}) has no response in the fixture"),
                Map::new(),
            ));
        };
        let (want_method, want_path) = (
            given.get("method").and_then(Value::as_str).unwrap_or(""),
            given.get("path").and_then(Value::as_str).unwrap_or(""),
        );
        if (want_method, want_path) != (method, path) {
            return Err(model_error(
                "builtins.AssertionError",
                None,
                None,
                &format!(
                    "request {at} is {method} {path}, and the fixture answers {want_method} {want_path}"
                ),
                Map::new(),
            ));
        }
        let status = given.get("status").and_then(Value::as_i64).unwrap_or(200);
        Ok((status, given.get("body").cloned().unwrap_or(Value::Null)))
    }

    /// `transport._post(path, payload)`.
    fn post(&mut self, path: &str, body: &str) -> Outcome {
        let payload: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        self.bodies.push(payload);
        match self.serve("POST", path) {
            Err(error) => Outcome::ModelError(error),
            Ok((status, body)) if status >= 400 => Outcome::ModelError(nim_error(status, &body)),
            Ok((_, body)) => match body {
                Value::Object(_) => Outcome::Response(body.to_string()),
                other => Outcome::ModelError(model_error(
                    "sdm_client.errors.StructuredDataError",
                    None,
                    Some("TRANSPORT_ERROR"),
                    &format!(
                        "Expected a JSON object from http://localhost:8000{path}, got {}",
                        json_type_name(&other)
                    ),
                    json!({"url": format!("http://localhost:8000{path}")})
                        .as_object()
                        .cloned()
                        .unwrap_or_default(),
                )),
            },
        }
    }

    /// `transport.delete_session(session_id)`.
    fn delete(&mut self, session_id: &str) -> Result<(), String> {
        let path = format!("/v1/sessions/{session_id}");
        let (status, body) = self.serve("DELETE", &path)?;
        if status >= 400 {
            return Err(nim_error(status, &body));
        }
        Ok(())
    }

    /// One call a run asked for, answered.
    fn answer(&mut self, call: &ModelCall) -> Outcome {
        let body = call.body.clone().unwrap_or_default();
        let session = call.session_id.clone().unwrap_or_default();
        match call.endpoint {
            Endpoint::Predict => self.post("/v1/predictions", &body),
            Endpoint::SessionCreate => self.post("/v1/sessions", &body),
            Endpoint::SessionPredict => {
                self.post(&format!("/v1/sessions/{session}/predictions"), &body)
            }
            Endpoint::SessionDelete => match self.delete(&session) {
                Ok(()) => Outcome::Response("null".to_string()),
                Err(error) => Outcome::ModelError(error),
            },
        }
    }
}

/// `type(json.loads(...)).__name__` of a JSON value.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// A transport error, as the run that made the call would have raised it.
fn error_of_model(text: &str) -> Diagnostic {
    sdm_request::client::model_error(text)
}

/// Column labels, each with its `repr` where it is not text.
type Labels = Vec<(String, Option<String>)>;

/// The flow: what the calls made so far made.
struct Flow {
    tables: Vec<(String, Table)>,
    session: Answering,
    host: Host,
    schema: Option<SchemaH>,
    /// The query asked last: its text, its schema, and its model with the
    /// graph's tables, once it has gathered relationally.
    query: Option<(String, SchemaH, Value, Frames)>,
    request: Option<RequestH>,
    result: Option<Value>,
    results: Vec<Value>,
    fitted: Option<FittedH>,
    pinned: Vec<FittedH>,
    client: bool,
    frame: Option<Vec<sdm_request::layout::Column>>,
    /// What `PredictionResult.absent` gave, when asked.
    absent: Option<Vec<String>>,
    /// The envelopes `Request.payloads` read, when asked.
    payloads: Option<Vec<Value>>,
    warnings: Vec<Diagnostic>,
    external: Option<VecDeque<(bool, Outcome)>>,
    expected_queries: Option<VecDeque<(String, Value)>>,
    external_failed: bool,
    queries: Vec<sdm_protocol::Query>,
    carry: carry::Inputs,
}

/// An in-memory DuckDB database's catalog, holding `tables`: what
/// `duckdb_tables()` lists, and the error `PRAGMA table_info` gives of a
/// table it does not hold. Any other statement reads rows the fixture flow
/// has no database for, and is refused, so a case needing one fails rather
/// than being answered wrongly.
struct DuckdbCatalog {
    tables: Vec<String>,
}

impl DuckdbCatalog {
    fn answer(&self, query: &sdm_protocol::Query) -> Result<Outcome, InputError> {
        if query.sql == sdm_dialects::duckdb::list_tables().0 {
            let mut names = self.tables.clone();
            names.sort();
            let column = arrow_array::StringArray::from(names);
            let batch = RecordBatch::try_from_iter([(
                "table_name",
                std::sync::Arc::new(column) as arrow_array::ArrayRef,
            )])
            .map_err(|e| bad(e.to_string()))?;
            return Ok(Outcome::Data(sdm_protocol::QueryResult::from_batch(batch)));
        }
        let named = query
            .sql
            .strip_prefix("PRAGMA table_info(\"")
            .and_then(|rest| rest.strip_suffix("\")"))
            .map(|quoted| quoted.replace("\"\"", "\""));
        if let Some(name) = named.filter(|name| !self.tables.contains(name)) {
            return Ok(Outcome::Error {
                sqlstate: None,
                vendor_code: None,
                message: format!("Catalog Error: Table with name {name} does not exist!"),
            });
        }
        Err(bad(format!(
            "a fixture flow's DuckDB database answers only its catalog, not {}",
            query.sql
        )))
    }
}

/// A fixture argument as the host describes it (`sdm_core.answers.encode`):
/// `{"$cyclic": KEY}`, a mapping holding itself under `KEY`, is that
/// mapping with the reference back written `{"$refers_back": "dict"}`.
fn host_form(raw: &Value) -> Value {
    match raw {
        Value::Object(map) if map.len() == 1 && map.contains_key("$cyclic") => {
            let key = match &map["$cyclic"] {
                Value::String(key) => key.clone(),
                other => other.to_string(),
            };
            json!({ key: {"$refers_back": "dict"} })
        }
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), host_form(v))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(host_form).collect()),
        other => other.clone(),
    }
}

/// A fixture value of `call`'s arguments that is not one the flow reads.
fn unsupported(call: &str) -> InputError {
    bad(format!("client.run cannot make the call {call}"))
}

impl Flow {
    /// A flow over `tables`, the model answered with `responses`.
    fn new(tables: Vec<(String, Table)>, responses: Vec<Value>) -> Self {
        Flow {
            tables,
            session: Answering {
                responses,
                ..Answering::default()
            },
            host: Host::assumed(),
            schema: None,
            query: None,
            request: None,
            result: None,
            results: Vec::new(),
            fitted: None,
            pinned: Vec::new(),
            client: false,
            frame: None,
            absent: None,
            payloads: None,
            warnings: Vec::new(),
            external: None,
            expected_queries: None,
            external_failed: false,
            queries: Vec::new(),
            carry: carry::Inputs::default(),
        }
    }

    /// `change` made in place to the input's frame `name`: every frame the
    /// flow holds as that frame, the schema's and the query's, changes too,
    /// as a pandas `DataFrame` a schema keeps by reference does.
    fn change_frame(
        &mut self,
        name: &str,
        change: impl Fn(&Table) -> Result<Table, InputError>,
    ) -> Result<(), InputError> {
        let changed = change(&self.table(name)?)?;
        if let Some(entry) = self.tables.iter_mut().find(|(n, _)| n == name) {
            entry.1 = changed.clone();
        }
        let key = sdm_request::schemas::DataSchema::frame_key(name);
        for schema in self
            .schema
            .iter_mut()
            .chain(self.query.iter_mut().map(|query| &mut query.1))
        {
            if schema.frames.holds(&key) {
                schema.frames.insert(&key, &changed);
            }
        }
        Ok(())
    }

    /// The labels `Frame.set_columns` gives, each with its `repr` where it
    /// is not text, and whether they make a `MultiIndex`: a label given as
    /// a list is a tuple, named by its `str()` as the describer names it.
    fn labels(given: &[Value]) -> Result<(Labels, bool), InputError> {
        let multi_index = given.iter().any(Value::is_array);
        let labels = given
            .iter()
            .map(|label| {
                let value = PyVal::decode(label).map_err(bad)?;
                let value = match value {
                    PyVal::List(items) => PyVal::Tuple(items),
                    other => other,
                };
                Ok(match value {
                    PyVal::Str(text) => (text, None),
                    other => (
                        sdm_request::pyops::py_str(&other),
                        Some(sdm_request::pyops::py_repr(&other)),
                    ),
                })
            })
            .collect::<Result<Vec<_>, InputError>>()?;
        Ok((labels, multi_index))
    }

    fn table(&self, name: &str) -> Result<Table, InputError> {
        self.tables
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .ok_or_else(|| bad(format!("no table {name}")))
    }

    /// `{"$frame": {"table", "rows", "drop"}}`'s frame.
    fn frame_of(&self, body: &Value) -> Result<Table, InputError> {
        let name = body
            .get("table")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("a $frame names its table"))?;
        let mut frame = self.table(name)?;
        if let Some(Value::Array(rows)) = body.get("rows") {
            let start = rows.first().and_then(Value::as_u64).unwrap_or(0) as usize;
            let stop = rows.get(1).and_then(Value::as_u64).unwrap_or(0) as usize;
            let stop = stop.min(frame.len());
            let positions: Vec<usize> = (start.min(stop)..stop).collect();
            frame = frame.take(&positions);
        }
        if let Some(Value::Array(drop)) = body.get("drop") {
            let names: Vec<String> = drop
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            frame = frame.drop(&names);
        }
        Ok(frame)
    }

    fn frame_tag(raw: &Value) -> Option<&Value> {
        match raw {
            Value::Object(map) if map.len() == 1 => map.get("$frame"),
            _ => None,
        }
    }

    /// Drive `run` to its end, each model call answered by the stand-in.
    fn drive(&mut self, run: Box<dyn Run<Plain> + Send>) -> Result<Plain, Diagnostic> {
        self.drive_over(run, None)
    }

    /// Drive `run` to its end, each model call answered by the stand-in and
    /// each query by `catalog`, where a flow has one.
    fn drive_over(
        &mut self,
        mut run: Box<dyn Run<Plain> + Send>,
        catalog: Option<&DuckdbCatalog>,
    ) -> Result<Plain, Diagnostic> {
        loop {
            let step = run.next();
            self.warnings.extend(run.take_warnings());
            match step {
                Err(error) => return Err(error),
                Ok(Step::Done(value)) => return Ok(value),
                Ok(Step::RunQueries(queries)) => {
                    if self.external.is_some() {
                        for query in queries {
                            if let Some(expected) = &mut self.expected_queries {
                                let found = (
                                    query.sql.clone(),
                                    crate::public_replay::params(&query.params),
                                );
                                if expected.pop_front().as_ref() != Some(&found) {
                                    self.external_failed = true;
                                    return Err(invalid(
                                        "external query statement or parameters differ",
                                    ));
                                }
                            }
                            let outcome = self.external_outcome(false)?;
                            self.queries.push(query.clone());
                            run.submit(query.id, outcome)?;
                        }
                        continue;
                    }
                    let Some(catalog) = catalog else {
                        return Err(invalid("a fixture flow has no database to run a query in"));
                    };
                    for query in queries {
                        let outcome = catalog.answer(&query).map_err(|e| invalid(e.0))?;
                        run.submit(query.id, outcome)?;
                    }
                }
                Ok(Step::CallModel(call)) => {
                    let outcome = if self.external.is_some() {
                        self.session.record_external(&call)?;
                        self.external_outcome(true)?
                    } else {
                        self.session.answer(&call)
                    };
                    run.submit(call.id, outcome)?;
                }
            }
        }
    }

    fn external_outcome(&mut self, model: bool) -> Result<Outcome, Diagnostic> {
        match self.external.as_mut().and_then(VecDeque::pop_front) {
            Some((recorded_model, outcome)) if recorded_model == model => Ok(outcome),
            _ => {
                self.external_failed = true;
                Err(invalid("missing or mismatched external host outcome"))
            }
        }
    }

    /// A host function on `args`: a value, or a run driven to its end.
    fn call(&mut self, function: &str, args: Vec<Plain>) -> Result<Plain, Diagnostic> {
        match call(function, &args, &self.host) {
            Ok(Called::Value(value)) => Ok(value),
            Ok(Called::Run(run)) => self.drive(run),
            Err(refused) => {
                self.warnings.extend(refused.warnings);
                Err(*refused.error)
            }
        }
    }

    fn give(&mut self, given: Option<&Value>) {
        for warning in given.and_then(Value::as_array).into_iter().flatten() {
            if warning.get("masked_float32_cast") == Some(&Value::Bool(true)) {
                continue;
            }
            self.warnings.push(Diagnostic::warning(
                warning
                    .get("category")
                    .and_then(Value::as_str)
                    .unwrap_or(sdm_request::error::USER_WARNING),
                warning
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ));
        }
    }

    fn public_diagnostics(&self, failure: Option<&Diagnostic>) -> (Value, Value) {
        let error = failure.map(sdm_schema::diagnostics::public_observation);
        let warnings = self
            .warnings
            .iter()
            .filter(|warning| {
                warning.details.get("masked_float32_cast") != Some(&Value::Bool(true))
            })
            .map(sdm_schema::diagnostics::public_observation)
            .collect::<Vec<_>>();
        (json!(error), json!(warnings))
    }

    fn public_observation(
        &self,
        failure: Option<&Diagnostic>,
        payloads: Vec<Value>,
        result: Value,
    ) -> Result<Value, InputError> {
        if self.external_failed {
            return Err(bad("missing or mismatched external host outcome"));
        }
        if self
            .external
            .as_ref()
            .is_some_and(|outcomes| !outcomes.is_empty())
        {
            return Err(bad("unused external host outcomes"));
        }
        let mut columns = Vec::new();
        if let Some(schema) = &self.schema {
            for table in schema.state["schema"]["tables"]
                .as_array()
                .into_iter()
                .flatten()
            {
                for column in table["columns"].as_array().into_iter().flatten() {
                    columns.push(json!({"table": table["name"], "column": column["name"],
                        "dtype": column["dtype"], "stype": column["stype"],
                        "is_time": column["name"] == table["time_column"]}));
                }
            }
        }
        let (error, warnings) = self.public_diagnostics(failure);
        Ok(json!({
            "status": if failure.is_some() { "refused" } else { "success" },
            "schema": columns, "payloads": payloads, "result": result,
            "sql": self.queries.iter().map(|query| json!({"statement": query.sql,
                "parameters": crate::public_replay::params(&query.params)})).collect::<Vec<_>>(),
            "error": error, "warnings": warnings,
        }))
    }

    /// What the host raises of a schema's refusal.
    fn schema_refused(found: &Value) -> Option<Diagnostic> {
        let refused = found.get("error")?;
        let code = refused.get("code").and_then(Value::as_str).unwrap_or("");
        let message = refused
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Some(
            if code
                .replace('_', "")
                .chars()
                .all(|c| c.is_ascii_uppercase())
                && !code.is_empty()
            {
                Diagnostic::refusal(STRUCTURED, Some(code), message)
            } else {
                sdm_request::error::builtin(
                    if code.is_empty() { "ValueError" } else { code },
                    message,
                )
            },
        )
    }

    fn schema_from_data(
        &mut self,
        names: &[String],
        args: &Map<String, Value>,
    ) -> Result<SchemaH, Diagnostic> {
        let mut described = Frames::default();
        for name in names {
            let table = self
                .table(name)
                .map_err(|e| sdm_request::error::value_error(e.0))?;
            described.insert(name, &table);
        }
        let options = json!({
            "primary_keys": args.get("primary_keys").cloned().unwrap_or(Value::Null),
            "time_columns": args.get("time_columns").cloned().unwrap_or(Value::Null),
            "relationships": args.get("relationships").cloned().unwrap_or(Value::Null),
            "edits": [],
        });
        let found = self.call(
            "data_schema",
            vec![
                described.plain(),
                Plain::Str(Value::Object(described.tags.clone()).to_string()),
                Plain::Str(options.to_string()),
            ],
        )?;
        let (found, said) = match found {
            Plain::Tuple(items) if items.len() == 2 => (items[0].clone(), items[1].clone()),
            _ => return Err(invalid("data_schema gave no pair")),
        };
        let said: Value = serde_json::from_str(said.as_str().unwrap_or("[]")).unwrap_or(json!([]));
        self.give(Some(&said));
        let found: Value =
            serde_json::from_str(found.as_str().unwrap_or("{}")).unwrap_or(json!({}));
        if let Some(refused) = Self::schema_refused(&found) {
            return Err(refused);
        }
        let mut frames = Frames::default();
        for (name, batch) in &described.tables {
            frames.add_in_place(
                &format!("schema/{name}"),
                batch.clone(),
                described.tags.get(name).cloned().unwrap_or(json!({})),
            );
        }
        Ok(SchemaH {
            state: found,
            frames,
        })
    }

    /// `Schema.from_source` over an in-memory DuckDB database holding the
    /// input's tables named in `load`, all of them by default, as the host
    /// reads one (`request_host.schema_from_source`). The database answers
    /// only what a catalog says ([`DuckdbCatalog`]), so a flow reaches the
    /// refusals of a source with no table or without the table named, and
    /// no further.
    fn schema_from_duckdb(
        &mut self,
        args: &Map<String, Value>,
    ) -> Result<SchemaH, Result<Diagnostic, InputError>> {
        let loaded: Vec<String> = match args.get("load") {
            Some(Value::Array(names)) => names
                .iter()
                .map(|n| {
                    n.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| Err(bad("load names tables")))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(Err(bad("load names tables"))),
            None => self.tables.iter().map(|(n, _)| n.clone()).collect(),
        };
        for name in &loaded {
            self.table(name).map_err(Err)?;
        }
        let given = json!({
            "tables": args.get("tables").cloned().unwrap_or(Value::Null),
            "primary_keys": args.get("primary_keys").cloned().unwrap_or(Value::Null),
            "time_columns": args.get("time_columns").cloned().unwrap_or(Value::Null),
            "relationships": args.get("relationships").cloned().unwrap_or(Value::Null),
            "check_keys": args.get("check_keys").cloned().unwrap_or(Value::Bool(true)),
        });
        let source = json!({"namespace": {}, "semantic_model": null, "trust_expressions": false});
        let catalog = DuckdbCatalog { tables: loaded };
        let args = vec![
            Plain::Str("duckdb".into()),
            Plain::Str(source.to_string()),
            Plain::Str(given.to_string()),
        ];
        let found = match call("schema_source_run", &args, &self.host) {
            Ok(Called::Run(run)) => self.drive_over(run, Some(&catalog)).map_err(Ok)?,
            Ok(Called::Value(_)) => {
                return Err(Ok(invalid(
                    "schema_source_run gave a value where a run was expected",
                )));
            }
            Err(refused) => {
                self.warnings.extend(refused.warnings);
                return Err(Ok(*refused.error));
            }
        };
        let (found, said) = match found {
            Plain::Tuple(items) if items.len() == 2 => (items[0].clone(), items[1].clone()),
            _ => return Err(Ok(invalid("schema_source_run gave no pair"))),
        };
        let said: Value = serde_json::from_str(said.as_str().unwrap_or("[]")).unwrap_or(json!([]));
        self.give(Some(&said));
        let found: Value =
            serde_json::from_str(found.as_str().unwrap_or("{}")).unwrap_or(json!({}));
        if let Some(refused) = Self::schema_refused(&found) {
            return Err(Ok(refused));
        }
        if self.external.is_some() {
            return Ok(SchemaH {
                state: found,
                frames: Frames::default(),
            });
        }
        Err(Err(bad(
            "a fixture flow reads a DuckDB source only as far as its catalog, so a schema \
             described from one cannot be used further",
        )))
    }

    /// `setattr`, `del` or `append` on the schema made last, made on what
    /// it says; refused here as the assignment is.
    fn edit_schema(&mut self, edit: Value) -> Result<(), Diagnostic> {
        let Some(schema) = self.schema.clone() else {
            return Err(invalid("no schema has been made to edit"));
        };
        let tables = sdm_request::schemas::Tables::new(
            schema.frames.tables.clone(),
            &Value::Object(schema.frames.tags.clone()),
        );
        let before = sdm_request::schemas::DataSchema::from_json(&schema.state)
            .map_err(sdm_request::error::value_error)?;
        let (after, said) = before.edited(edit, &tables, &self.host)?;
        for warning in said.warnings {
            self.warnings
                .push(Diagnostic::warning(&warning.category, warning.message));
        }
        let state = after.to_json();
        self.schema = Some(SchemaH {
            state,
            frames: schema.frames,
        });
        Ok(())
    }

    /// An argument as the host passes it: a frame named, a model as its
    /// tag, a mapping as tagged values, anything else as tagged JSON.
    fn encode_args(
        &self,
        args: &Map<String, Value>,
        prefix: &str,
    ) -> Result<(Map<String, Value>, Frames), InputError> {
        let mut described = Frames::default();
        let mut encoded = Map::new();
        for (name, raw) in args {
            if let Some(body) = Self::frame_tag(raw) {
                let frame = self.frame_of(body)?;
                let key = format!("{prefix}{name}");
                described.insert(&key, &frame);
                encoded.insert(name.clone(), json!({"$frame": key}));
            } else {
                encoded.insert(name.clone(), host_form(raw));
            }
        }
        Ok((encoded, described))
    }

    fn run_call(&mut self, call: &Value) -> Result<(), Result<Diagnostic, InputError>> {
        let name = call
            .get("call")
            .and_then(Value::as_str)
            .ok_or_else(|| Err(bad("a call is named")))?;
        let args = call
            .get("args")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let fail = |e: Diagnostic| Ok(e);
        if name.starts_with("Carry.") {
            return self.carry_call(name, &args);
        }
        match name {
            "Schema.from_duckdb" => {
                let schema = self.schema_from_duckdb(&args).map_err(|e| match e {
                    Ok(refused) => Ok(refused),
                    Err(input) => Err(input),
                })?;
                self.schema = Some(schema);
            }
            "Schema.from_data" => {
                let names: Vec<String> = match args.get("tables") {
                    Some(Value::Array(items)) => items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect(),
                    _ => self.tables.iter().map(|(n, _)| n.clone()).collect(),
                };
                let schema = self.schema_from_data(&names, &args).map_err(fail)?;
                self.schema = Some(schema);
            }
            "Schema.set" => {
                let edit = json!({
                    "table": args.get("table").cloned().unwrap_or(Value::Null),
                    "column": args.get("column").cloned().unwrap_or(Value::Null),
                    "set": args.get("attribute").cloned().unwrap_or(Value::Null),
                    "value": args.get("value").cloned().unwrap_or(Value::Null),
                });
                self.edit_schema(edit).map_err(fail)?;
            }
            "Schema.set_table" => {
                let edit = json!({
                    "table": args.get("table").cloned().unwrap_or(Value::Null),
                    "set_table": args.get("attribute").cloned().unwrap_or(Value::Null),
                    "value": args.get("value").cloned().unwrap_or(Value::Null),
                });
                self.edit_schema(edit).map_err(fail)?;
            }
            "Schema.delete" => {
                let edit = json!({
                    "table": args.get("table").cloned().unwrap_or(Value::Null),
                    "delete": args.get("column").cloned().unwrap_or(Value::Null),
                });
                self.edit_schema(edit).map_err(fail)?;
            }
            "Schema.link" => {
                let edit = json!({"link": args.get("link").cloned().unwrap_or(Value::Null)});
                self.edit_schema(edit).map_err(fail)?;
            }
            "Schema.put_source_table" => {
                // A `SourceTable` holds no rows: the table's frame leaves the
                // schema, and what reads its rows refuses it.
                let table = args
                    .get("table")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Err(bad("Schema.put_source_table names a table")))?;
                let key = sdm_request::schemas::DataSchema::frame_key(table);
                if self.schema.is_none() {
                    return Err(Err(bad("no schema has been made to edit")));
                }
                for schema in self
                    .schema
                    .iter_mut()
                    .chain(self.query.iter_mut().map(|query| &mut query.1))
                {
                    schema.frames.remove(&key);
                }
            }
            "Frame.set_columns" => {
                let table = args
                    .get("table")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Err(bad("Frame.set_columns names a table")))?
                    .to_string();
                let given = args
                    .get("columns")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Err(bad("Frame.set_columns gives columns")))?;
                let (labels, multi_index) = Self::labels(given).map_err(Err)?;
                self.change_frame(&table, |frame| {
                    if labels.len() != frame.columns().len() {
                        return Err(bad("Frame.set_columns gives a label per column"));
                    }
                    Ok(frame.relabelled(&labels, multi_index))
                })
                .map_err(Err)?;
            }
            "Frame.drop" => {
                let table = args
                    .get("table")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Err(bad("Frame.drop names a table")))?
                    .to_string();
                let columns: Vec<String> = args
                    .get("columns")
                    .and_then(Value::as_array)
                    .map(|names| {
                        names
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                self.change_frame(&table, |frame| {
                    let missing: Vec<&String> = columns.iter().filter(|c| !frame.has(c)).collect();
                    if !missing.is_empty() {
                        return Err(bad(format!("Frame.drop drops {missing:?}, which it lacks")));
                    }
                    Ok(frame.drop(&columns))
                })
                .map_err(Err)?;
            }
            "PredictiveQuery" => {
                let query = args.get("query").cloned().unwrap_or(Value::Null);
                let text = self
                    .call("query_text", vec![Plain::from_json(&query)])
                    .map_err(fail)?;
                let schema = match args.get("schema") {
                    None | Some(Value::Null) => self.schema.clone(),
                    Some(Value::Object(map)) if map.contains_key("$schema") => self.schema.clone(),
                    Some(Value::Object(map)) if map.contains_key("$frames") => {
                        let names: Vec<String> = map["$frames"]
                            .as_array()
                            .map(|n| {
                                n.iter()
                                    .filter_map(|v| v.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(self.schema_from_data(&names, &Map::new()).map_err(fail)?)
                    }
                    Some(_) => return Err(Err(unsupported("PredictiveQuery"))),
                };
                let schema = schema.ok_or_else(|| Err(bad("no schema to ask")))?;
                self.query = Some((
                    text.as_str().unwrap_or_default().to_string(),
                    schema,
                    Value::Null,
                    Frames::default(),
                ));
            }
            "PredictiveQuery.materialize" => {
                let (text, schema, model, graph) = self
                    .query
                    .clone()
                    .ok_or_else(|| Err(bad("no query to materialize")))?;
                let (encoded, described) = self.encode_args(&args, "arg/").map_err(Err)?;
                let frames = schema.frames.add(&described).add(&graph);
                let state = json!({
                    "schema": schema.state,
                    "query": text,
                    "args": encoded,
                    "tags": frames.tags,
                    "model": model,
                });
                let done = self
                    .call(
                        "materialize_run",
                        vec![Plain::Str(state.to_string()), frames.plain()],
                    )
                    .map_err(fail)?;
                let (answer, tables) = match done {
                    Plain::Tuple(items) if items.len() == 2 => (items[0].clone(), items[1].clone()),
                    _ => return Err(Err(bad("materialize_run gave no pair"))),
                };
                let found: Value =
                    serde_json::from_str(answer.as_str().unwrap_or("{}")).unwrap_or(json!({}));
                let (graph, gathered) =
                    Frames::of(&tables, found.get("tags").unwrap_or(&Value::Null)).split("model/");
                if let Some(query) = self.query.as_mut() {
                    query.2 = found.get("model").cloned().unwrap_or(Value::Null);
                    query.3 = graph;
                }
                if let Some(refused) = found.get("refused") {
                    return Err(fail(sdm_request::error::refusal_of(refused)));
                }
                self.request = Some(RequestH {
                    state: found.get("request").cloned().unwrap_or(Value::Null),
                    frames: schema.frames.add(&gathered),
                });
            }
            "Request" => {
                let mut given = args.clone();
                let schema = given.remove("schema");
                let related = given.remove("related_tables").unwrap_or(json!({}));
                let mut described = Frames::default();
                let mut encoded = Map::new();
                for (name, raw) in &given {
                    if (name == "context" || name == "predict")
                        && let Some(body) = Self::frame_tag(raw)
                    {
                        let frame = self.frame_of(body).map_err(Err)?;
                        let key = format!("arg/{name}");
                        described.insert(&key, &frame);
                        encoded.insert(name.clone(), json!({"$frame": key}));
                    } else {
                        encoded.insert(name.clone(), host_form(raw));
                    }
                }
                let mut related_out = Map::new();
                if let Value::Object(map) = &related {
                    for (name, raw) in map {
                        if let Some(body) = Self::frame_tag(raw) {
                            let frame = self.frame_of(body).map_err(Err)?;
                            let key = format!("arg/related/{name}");
                            described.insert(&key, &frame);
                            related_out.insert(name.clone(), json!({"$frame": key}));
                        } else {
                            related_out.insert(name.clone(), raw.clone());
                        }
                    }
                }
                encoded.insert("related_tables".into(), Value::Object(related_out));
                if let Some(Value::Object(map)) = &schema
                    && map.contains_key("$schema")
                    && let Some(current) = &self.schema
                {
                    encoded.insert("schema".into(), current.state.clone());
                    described = described.add(&current.frames);
                }
                let state = json!({"args": encoded, "tags": described.tags});
                let done = self
                    .call(
                        "request_build",
                        vec![Plain::Str(state.to_string()), described.plain()],
                    )
                    .map_err(fail)?;
                let (answer, tables) = match done {
                    Plain::Tuple(items) if items.len() == 2 => (items[0].clone(), items[1].clone()),
                    _ => return Err(Err(bad("request_build gave no pair"))),
                };
                let found: Value =
                    serde_json::from_str(answer.as_str().unwrap_or("{}")).unwrap_or(json!({}));
                self.give(found.get("warnings"));
                self.request = Some(RequestH {
                    state: found.get("request").cloned().unwrap_or(Value::Null),
                    frames: Frames::of(&tables, found.get("tags").unwrap_or(&Value::Null)),
                });
            }
            "Request.assign" => {
                let half = args
                    .get("half")
                    .and_then(Value::as_str)
                    .unwrap_or("context")
                    .to_string();
                let body = args
                    .get("value")
                    .and_then(Self::frame_tag)
                    .ok_or_else(|| Err(unsupported("Request.assign")))?;
                let frame = self.frame_of(body).map_err(Err)?;
                let request = self
                    .request
                    .as_mut()
                    .ok_or_else(|| Err(bad("no request to assign to")))?;
                let tables = sdm_request::schemas::Tables::new(
                    request.frames.tables.clone(),
                    &Value::Object(request.frames.tags.clone()),
                );
                let built = sdm_request::request::Request::from_json(&request.state, &tables)
                    .map_err(|e| Err(bad(e)))?;
                built.refuse_if_gathered(&half).map_err(Ok)?;
                request.frames.insert(&half, &frame);
            }
            "SDMClient.predict" => {
                self.client = true;
                let request = self.request_arg(&args)?;
                let state = json!({
                    "request": request.state,
                    "model": args.get("model").cloned().unwrap_or(Value::Null),
                    "outputs": args.get("outputs").cloned().unwrap_or(Value::Null),
                    "tags": request.frames.tags,
                });
                let done = self
                    .call(
                        "predict_run",
                        vec![Plain::Str(state.to_string()), request.frames.plain()],
                    )
                    .map_err(fail)?;
                let result: Value =
                    serde_json::from_str(done.as_str().unwrap_or("null")).unwrap_or(Value::Null);
                self.results.push(result.clone());
                self.result = Some(result);
            }
            "SDMClient.fit" => {
                self.client = true;
                let request = self.request_arg(&args)?;
                let state = json!({
                    "request": request.state,
                    "model": args.get("model").cloned().unwrap_or(Value::Null),
                    "outputs": args.get("outputs").cloned().unwrap_or(Value::Null),
                    "holdout": args.get("holdout").cloned().unwrap_or(Value::Null),
                    "tags": request.frames.tags,
                });
                let done = self
                    .call(
                        "fit_run",
                        vec![Plain::Str(state.to_string()), request.frames.plain()],
                    )
                    .map_err(fail)?;
                let fitted: Value =
                    serde_json::from_str(done.as_str().unwrap_or("null")).unwrap_or(Value::Null);
                let handle = FittedH {
                    state: fitted,
                    closed: false,
                };
                self.pinned.push(handle.clone());
                self.fitted = Some(handle);
            }
            "FittedModel.predict" => {
                let request = self.request_arg(&args)?;
                let fitted = self
                    .fitted
                    .clone()
                    .ok_or_else(|| Err(bad("no fitted model")))?;
                let mut state = fitted.state.clone();
                if let Some(object) = state.as_object_mut() {
                    object.insert("closed".into(), json!(fitted.closed));
                }
                let state = json!({
                    "request": request.state,
                    "fitted": state,
                    "tags": request.frames.tags,
                });
                let done = self
                    .call(
                        "session_predict_run",
                        vec![Plain::Str(state.to_string()), request.frames.plain()],
                    )
                    .map_err(fail)?;
                let result: Value =
                    serde_json::from_str(done.as_str().unwrap_or("null")).unwrap_or(Value::Null);
                self.results.push(result.clone());
                self.result = Some(result);
            }
            "FittedModel.close" => {
                let Some(fitted) = self.fitted.clone() else {
                    return Err(Err(bad("no fitted model")));
                };
                self.close_fitted(&fitted).map_err(fail)?;
            }
            "PredictionResult.merge" => {
                let chosen: Vec<Value> = match args.get("parts") {
                    Some(Value::Array(at)) => at
                        .iter()
                        .filter_map(Value::as_u64)
                        .filter_map(|i| self.results.get(i as usize).cloned())
                        .collect(),
                    _ => self.results.clone(),
                };
                let parts = chosen
                    .iter()
                    .map(sdm_request::layout::decode_result)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| Err(bad(e)))?;
                let merged = sdm_request::result::Result_::merge(parts).map_err(fail)?;
                self.result = Some(sdm_request::layout::encode_result(&merged));
            }
            "PredictionResult.absent" => {
                let requested: Vec<String> = args
                    .get("requested")
                    .and_then(Value::as_array)
                    .map(|names| {
                        names
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let result = self
                    .result
                    .as_ref()
                    .ok_or_else(|| Err(bad("no result to ask")))?;
                let decoded =
                    sdm_request::layout::decode_result(result).map_err(|e| Err(bad(e)))?;
                self.absent = Some(decoded.absent(&requested));
            }
            "Request.payloads" => {
                let request = self
                    .request
                    .as_ref()
                    .ok_or_else(|| Err(bad("no request has been made")))?;
                let payloads = request
                    .state
                    .get("payloads")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                self.payloads = Some(payloads.iter().map(without_request_id).collect());
            }
            "PredictionResult.to_pandas" => {
                let result = self
                    .result
                    .as_ref()
                    .ok_or_else(|| Err(bad("no result to read")))?;
                let decoded =
                    sdm_request::layout::decode_result(result).map_err(|e| Err(bad(e)))?;
                self.frame = Some(sdm_request::layout::flatten(&decoded));
            }
            other => return Err(Err(unsupported(other))),
        }
        Ok(())
    }

    /// The request a call names, or the request made last; a value that is
    /// not a request is refused as the client refuses it.
    fn request_arg(
        &self,
        args: &Map<String, Value>,
    ) -> Result<RequestH, Result<Diagnostic, InputError>> {
        if let Some(given) = args.get("request") {
            let value = PyVal::decode(given).map_err(|e| Err(bad(e)))?;
            return Err(Ok(invalid(format!(
                "request must be a Request, got {}; build one with Request(context=..., \
                 predict=...) or PredictiveQuery(...).materialize(...)",
                sdm_request::models::type_name(&value)
            ))));
        }
        self.request
            .clone()
            .ok_or_else(|| Err(bad("no request has been made")))
    }

    /// `FittedModel.close()`: the session released, through the transport.
    fn close_fitted(&mut self, fitted: &FittedH) -> Result<(), Diagnostic> {
        if fitted.closed {
            return Ok(());
        }
        let session = fitted
            .state
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if self.external.is_some() {
            if !self.carry.close(&session) {
                self.external_failed = true;
                return Err(invalid("missing or mismatched carry cleanup outcome"));
            }
            self.session
                .requests
                .push(json!(["DELETE", format!("/v1/sessions/{session}")]));
        } else {
            self.session
                .delete(&session)
                .map_err(|e| error_of_model(&e))?;
        }
        for handle in self.pinned.iter_mut().chain(self.fitted.iter_mut()) {
            if handle.state.get("session_id") == fitted.state.get("session_id") {
                handle.closed = true;
            }
        }
        self.pinned
            .retain(|h| h.state.get("session_id") != fitted.state.get("session_id"));
        Ok(())
    }

    /// `client.close()`: every context still pinned released, the first
    /// failure raised.
    fn close(&mut self) -> Result<(), Diagnostic> {
        let mut first = None;
        for fitted in self.pinned.clone() {
            if let Err(error) = self.close_fitted(&fitted) {
                first.get_or_insert(error);
            }
        }
        self.pinned.clear();
        first.map_or(Ok(()), Err)
    }

    fn output(&self, show_bodies: bool) -> Value {
        let mut out = Map::new();
        out.insert(
            "requests".into(),
            Value::Array(self.session.requests.clone()),
        );
        if let Some(absent) = &self.absent {
            out.insert("absent".into(), json!(absent));
        }
        if let Some(payloads) = &self.payloads {
            out.insert("payloads".into(), Value::Array(payloads.clone()));
        }
        if let Some(columns) = &self.frame {
            let mut frame = Map::new();
            for column in columns {
                frame.insert(
                    column.name.clone(),
                    json!({
                        "dtype": column.dtype,
                        "values": column.cells.iter().map(sdm_request::fixture::encode).collect::<Vec<_>>(),
                    }),
                );
            }
            out.insert("frame".into(), json!({"columns": frame}));
        }
        if let Some(result) = &self.result
            && let Ok(decoded) = sdm_request::layout::decode_result(result)
        {
            if !decoded.unscored_candidates.is_empty() {
                out.insert(
                    "unscored_candidates".into(),
                    sdm_request::fixture::encode(&PyVal::List(decoded.unscored_candidates.clone())),
                );
            }
            if let Some(explanation) = &decoded.explanation {
                out.insert(
                    "explanation".into(),
                    sdm_request::fixture::encode(explanation),
                );
            }
        }
        if show_bodies {
            out.insert(
                "bodies".into(),
                Value::Array(self.session.bodies.iter().map(without_request_id).collect()),
            );
        }
        Value::Object(out)
    }
}

/// A posted body with its `metadata.request_id` set to `<request-id>`.
fn without_request_id(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut)
        && metadata.contains_key("request_id")
    {
        metadata.insert("request_id".into(), json!("<request-id>"));
    }
    body
}

/// `client.run`: the input's calls, made in order.
pub fn client_run(input: &Map<String, Value>) -> Result<String, InputError> {
    let mut tables = Vec::new();
    if let Some(Value::Object(specs)) = input.get("tables") {
        for (name, spec) in specs {
            tables.push((name.clone(), Table::from_spec(spec).map_err(bad)?));
        }
    }
    let responses = input
        .get("responses")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut flow = Flow::new(tables, responses);
    let calls = input
        .get("calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut failed: Option<Diagnostic> = None;
    for call in &calls {
        match flow.run_call(call) {
            Ok(()) => {}
            Err(Ok(error)) => {
                failed = Some(error);
                break;
            }
            Err(Err(input_error)) => return Err(input_error),
        }
    }
    if flow.client
        && let Err(error) = flow.close()
    {
        // An error leaving `finally` replaces the one being raised.
        failed = Some(error);
    }
    let show_bodies = input
        .get("show_bodies")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(match failed {
        Some(error) => refusal(&error, &flow.warnings),
        None => answer(flow.output(show_bodies), &flow.warnings),
    })
}

pub(crate) fn semantic_source(
    arguments: Vec<Plain>,
    host: Host,
    queries: Vec<(String, Value)>,
    outcomes: Vec<(bool, Outcome)>,
) -> Result<(Value, Value), InputError> {
    let mut flow = Flow::new(Vec::new(), Vec::new());
    flow.host = host;
    flow.external = Some(outcomes.into());
    flow.expected_queries = Some(queries.into());
    let result = flow.call("schema_source_run", arguments);
    if flow.external_failed {
        return Err(bad("semantic external query or outcome differs"));
    }
    if flow
        .external
        .as_ref()
        .is_some_and(|values| !values.is_empty())
        || flow
            .expected_queries
            .as_ref()
            .is_some_and(|values| !values.is_empty())
    {
        return Err(bad("unused semantic external queries or outcomes"));
    }
    let observation = match result {
        Err(error) => {
            return Err(bad(format!(
                "semantic source call failed: {}",
                error.message
            )));
        }
        Ok(Plain::Tuple(values)) if values.len() == 2 => {
            let state: Value = serde_json::from_str(
                values[0]
                    .as_str()
                    .ok_or_else(|| bad("invalid computed schema state"))?,
            )
            .map_err(|_| bad("invalid computed schema state JSON"))?;
            let warnings: Value = serde_json::from_str(
                values[1]
                    .as_str()
                    .ok_or_else(|| bad("invalid computed schema warnings"))?,
            )
            .map_err(|_| bad("invalid computed schema warnings JSON"))?;
            if !warnings.is_array() {
                return Err(bad("computed schema warnings must be a list"));
            }
            match state.get("error") {
                Some(error) => {
                    json!({"error":{"code":error.get("code"),"message":error.get("message")},"warnings":warnings})
                }
                None => {
                    let schema = sdm_request::schemas::schema_of(
                        state
                            .get("schema")
                            .ok_or_else(|| bad("missing computed schema"))?,
                    )
                    .map_err(bad)?;
                    json!({"schema":sdm_schema::client::describe(&schema),"warnings":warnings})
                }
            }
        }
        _ => return Err(bad("invalid computed semantic schema result")),
    };
    let log = Value::Array(
        flow.queries
            .iter()
            .map(|query| json!([query.sql, crate::public_replay::params(&query.params)]))
            .collect(),
    );
    Ok((observation, log))
}

fn public_request_payloads(request: &RequestH) -> Result<Vec<Value>, InputError> {
    let tables = sdm_request::schemas::Tables::new(
        request.frames.tables.clone(),
        &Value::Object(request.frames.tags.clone()),
    );
    let built = sdm_request::request::Request::from_json(&request.state, &tables).map_err(bad)?;
    Ok(built
        .payloads()
        .unwrap_or_default()
        .iter()
        .map(without_request_id)
        .collect())
}

pub(crate) fn public_client_flow(
    recipe: &Value,
    raw_frames: Plain,
    tags: Value,
    host: Host,
    outcomes: Vec<(bool, Outcome)>,
    caller_inputs: CarryInputs,
) -> Result<Value, InputError> {
    let given = recipe
        .get("input")
        .ok_or_else(|| bad("missing carry input"))?;
    let described = Frames::of(&raw_frames, &tags);
    let names = given
        .get("tables")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("missing carry tables"))?;
    if names.len() != described.tables.len()
        || tags
            .as_object()
            .is_none_or(|tags| tags.len() != names.len())
        || described
            .tables
            .iter()
            .any(|(name, _)| !names.contains_key(name) || tags.get(name).is_none())
    {
        return Err(bad("carry frame domain differs from its recipe"));
    }
    let tables = described
        .tables
        .into_iter()
        .map(|(name, batch)| {
            let table = Table::new(batch, tags.get(&name));
            (name, table)
        })
        .collect();
    let mut flow = Flow::new(tables, Vec::new());
    flow.host = host;
    flow.external = Some(outcomes.into());
    flow.carry = caller_inputs;
    let calls = given
        .get("calls")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing carry calls"))?;
    let mut failure = None;
    for call in calls {
        match flow.run_call(call) {
            Ok(()) => {}
            Err(Ok(error)) => {
                failure = Some(error);
                break;
            }
            Err(Err(error)) => return Err(error),
        }
    }
    if flow.client
        && let Err(error) = flow.close()
    {
        failure = Some(error);
    }
    flow.carry.finish()?;
    let result = flow.output(false);
    let payloads = flow.session.bodies.iter().map(without_request_id).collect();
    flow.public_observation(failure.as_ref(), payloads, result)
}

pub(crate) fn public_source_schema(
    arguments: Vec<Plain>,
    host: Host,
    outcomes: Vec<(bool, Outcome)>,
) -> Result<Value, InputError> {
    let mut flow = Flow::new(Vec::new(), Vec::new());
    flow.host = host;
    flow.external = Some(outcomes.into());
    let failure;
    match flow.call("schema_source_run", arguments) {
        Err(error) => failure = Some(error),
        Ok(Plain::Tuple(parts)) if parts.len() == 2 => {
            let state: Value = serde_json::from_str(
                parts[0]
                    .as_str()
                    .ok_or_else(|| bad("source schema state is not text"))?,
            )
            .map_err(|_| bad("invalid source schema state"))?;
            let warnings: Value = serde_json::from_str(
                parts[1]
                    .as_str()
                    .ok_or_else(|| bad("source schema warnings are not text"))?,
            )
            .map_err(|_| bad("invalid source schema warnings"))?;
            flow.give(Some(&warnings));
            failure = Flow::schema_refused(&state);
            if failure.is_none() {
                flow.schema = Some(SchemaH {
                    state,
                    frames: Frames::default(),
                });
            }
        }
        Ok(_) => return Err(bad("source schema returned no state")),
    }
    let result = if failure.is_none() {
        json!({"schema_constructed": true})
    } else {
        Value::Null
    };
    flow.public_observation(failure.as_ref(), Vec::new(), result)
}

pub fn public_materialize(
    recipe: &Value,
    function: &str,
    arguments: Vec<Plain>,
    host: Host,
    outcomes: Vec<(bool, Outcome)>,
    declared_table_data: Option<(Plain, Value)>,
) -> Result<Value, InputError> {
    let postgres = recipe.get("operation").and_then(Value::as_str) == Some("postgres.carry");
    if recipe.get("operation").and_then(Value::as_str) != Some("materialize") && !postgres {
        return Err(bad("unsupported public operation"));
    }
    if !matches!(function, "data_schema" | "schema_source_run") {
        return Err(bad(
            "initial public schema must be constructed from original inputs",
        ));
    }
    let given = recipe
        .get("input")
        .ok_or_else(|| bad("missing public input"))?;
    let mut flow = Flow::new(Vec::new(), Vec::new());
    flow.host = host;
    flow.external = Some(outcomes.into());
    let mut frames = Frames::default();
    if function == "data_schema" {
        let raw = arguments
            .first()
            .ok_or_else(|| bad("missing original frames"))?;
        let tags: Value = serde_json::from_str(
            arguments
                .get(1)
                .and_then(Plain::as_str)
                .ok_or_else(|| bad("missing original frame tags"))?,
        )
        .map_err(|_| bad("invalid original frame tags"))?;
        let described = Frames::of(raw, &tags);
        for (name, batch) in &described.tables {
            frames.add_in_place(
                &format!("schema/{name}"),
                batch.clone(),
                described.tags.get(name).cloned().unwrap_or(json!({})),
            );
        }
    }
    let mut failure = None;
    let constructed = flow.call(function, arguments);
    match constructed {
        Err(error) => failure = Some(error),
        Ok(Plain::Tuple(parts)) if parts.len() == 2 => {
            let state: Value = serde_json::from_str(
                parts[0]
                    .as_str()
                    .ok_or_else(|| bad("schema state is not text"))?,
            )
            .map_err(|_| bad("invalid schema state"))?;
            let warnings: Value = serde_json::from_str(
                parts[1]
                    .as_str()
                    .ok_or_else(|| bad("schema warnings are not text"))?,
            )
            .map_err(|_| bad("invalid schema warnings"))?;
            flow.give(Some(&warnings));
            if let Some(error) = Flow::schema_refused(&state) {
                failure = Some(error);
            } else {
                flow.schema = Some(SchemaH { state, frames });
            }
        }
        Ok(_) => return Err(bad("schema construction returned no state")),
    }
    if let Some((raw, tags)) = declared_table_data {
        let replacement = Frames::of(&raw, &tags);
        let schema = flow
            .schema
            .as_mut()
            .ok_or_else(|| bad("declared data requires successful initial schema construction"))?;
        if replacement.tables.len() != schema.frames.tables.len()
            || replacement.tables.iter().zip(&schema.frames.tables).any(
                |((name, batch), (original_name, original))| {
                    name != original_name
                        || batch.num_rows() != original.num_rows()
                        || batch
                            .schema()
                            .fields()
                            .iter()
                            .map(|field| field.name())
                            .ne(original.schema().fields().iter().map(|field| field.name()))
                },
            )
        {
            return Err(bad("declared frame names, columns, or rows changed"));
        }
        schema.frames = replacement;
    }
    if failure.is_none() {
        for edit in given
            .get("edits")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Err(error) = flow.edit_schema(edit.clone()) {
                failure = Some(error);
                break;
            }
        }
    }
    if failure.is_none() && postgres {
        match flow.carry_column("users", json!("tags"), "tags", "string", "multicategorical") {
            Ok(()) => {}
            Err(Ok(error)) => failure = Some(error),
            Err(Err(error)) => return Err(error),
        }
    }
    if failure.is_none() {
        let mut options = given.get("options").cloned().unwrap_or(json!({}));
        if postgres
            && given
                .get("explicit_relational_model")
                .and_then(Value::as_bool)
                == Some(true)
        {
            options["model"] = json!({"$model": "KumoRelational"});
        }
        let calls = [
            json!({"call": "PredictiveQuery", "args": {"query": given.get("query")}}),
            json!({"call": "PredictiveQuery.materialize", "args": options}),
        ];
        for call in calls {
            match flow.run_call(&call) {
                Ok(()) => {}
                Err(Ok(error)) => {
                    failure = Some(error);
                    break;
                }
                Err(Err(error)) => return Err(error),
            }
        }
    }
    if flow.external_failed {
        return Err(bad("missing or mismatched external host outcome"));
    }
    if flow
        .external
        .as_ref()
        .is_some_and(|outcomes| !outcomes.is_empty())
    {
        return Err(bad("unused external host outcomes"));
    }
    let mut payloads = Vec::new();
    let mut result = Value::Null;
    if failure.is_none() {
        let request = flow
            .request
            .as_ref()
            .ok_or_else(|| bad("materialization returned no request"))?;
        payloads = public_request_payloads(request)?;
        if payloads.is_empty() {
            return Err(bad("public frame-only projection is not yet supported"));
        }
        result = json!({"payload_count": payloads.len()});
    }
    flow.public_observation(failure.as_ref(), payloads, result)
}

/// `tabular.task`: the task a single-table question is gathered as, and
/// the payload's target column.
pub fn tabular_task(input: &Map<String, Value>) -> Result<String, InputError> {
    let frame = Table::from_spec(input.get("frame").ok_or_else(|| bad("a frame"))?).map_err(bad)?;
    let table = input
        .get("table")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("a table"))?
        .to_string();
    let target = input
        .get("target")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("a target"))?
        .to_string();
    let scored_rows = input.get("scored").and_then(Value::as_u64).unwrap_or(2) as usize;
    let scored = frame
        .take(&(0..scored_rows.min(frame.len())).collect::<Vec<_>>())
        .drop(std::slice::from_ref(&target));
    let options = input.get("options").cloned().unwrap_or(json!({}));
    let mut flow_input = Map::new();
    let mut specs = Map::new();
    specs.insert(
        table.clone(),
        input.get("frame").cloned().unwrap_or(Value::Null),
    );
    flow_input.insert("tables".into(), Value::Object(specs));
    let mut flow = Flow::new(
        vec![
            (table.clone(), frame.clone()),
            ("#scored".to_string(), scored),
        ],
        Vec::new(),
    );
    let query = input.get("query").cloned().unwrap_or(Value::Null);
    let mut materialize = options.as_object().cloned().unwrap_or_default();
    materialize.insert("predict".into(), json!({"$frame": {"table": "#scored"}}));
    let calls = vec![
        json!({"call": "Schema.from_data", "args": {"tables": [table]}}),
        json!({"call": "PredictiveQuery", "args": {"query": query}}),
        json!({"call": "PredictiveQuery.materialize", "args": materialize}),
    ];
    for call in &calls {
        match flow.run_call(call) {
            Ok(()) => {}
            Err(Ok(error)) => return Ok(refusal(&error, &flow.warnings)),
            Err(Err(input_error)) => return Err(input_error),
        }
    }
    let request = flow.request.clone().ok_or_else(|| bad("no request"))?;
    let tables = sdm_request::schemas::Tables::new(
        request.frames.tables.clone(),
        &Value::Object(request.frames.tags.clone()),
    );
    let built = sdm_request::request::Request::from_json(&request.state, &tables).map_err(bad)?;
    let mut said = sdm_request::error::Warnings::default();
    let envelope = (|| {
        let mut schema = built.schema.build(&tables, &flow.host)?;
        sdm_request::tabular_adapter::envelope(
            &built,
            &mut schema,
            &["prediction".to_string()],
            None,
            &mut said,
        )
    })();
    flow.warnings.extend(said.take());
    let envelope = match envelope {
        Ok(envelope) => envelope,
        Err(error) => return Ok(refusal(&error, &flow.warnings)),
    };
    let output = json!({
        "kind": sdm_request::layout::encode_deep(&built.task.kind),
        "target_dtype": sdm_request::layout::encode_deep(&built.task.target.dtype),
        "declared": envelope
            .pointer(&format!("/schema/instance_table/columns/{}", target.replace('~', "~0").replace('/', "~1")))
            .cloned()
            .unwrap_or(Value::Null),
        "warnings": flow.warnings.iter().map(|w| json!(w.message)).collect::<Vec<_>>(),
    });
    Ok(answer(output, &[]))
}

/// `decompose.parts`: the questions a ranking or forecast is asked as, over
/// given frames.
pub fn decompose_parts(input: &Map<String, Value>) -> Result<String, InputError> {
    let mut tables = Vec::new();
    let mut names = Vec::new();
    if let Some(Value::Object(specs)) = input.get("tables") {
        for (name, spec) in specs {
            tables.push((name.clone(), Table::from_spec(spec).map_err(bad)?));
            names.push(json!(name));
        }
    }
    let mut flow = Flow::new(tables, Vec::new());
    let mut schema_args = Map::new();
    schema_args.insert("tables".into(), Value::Array(names));
    for key in ["primary_keys", "time_columns"] {
        if let Some(v) = input.get(key) {
            schema_args.insert(key.into(), v.clone());
        }
    }
    let calls = vec![
        json!({"call": "Schema.from_data", "args": schema_args}),
        json!({"call": "PredictiveQuery", "args": {"query": input.get("query").cloned().unwrap_or(Value::Null)}}),
        json!({"call": "PredictiveQuery.materialize", "args": input.get("options").cloned().unwrap_or(json!({}))}),
    ];
    for call in &calls {
        match flow.run_call(call) {
            Ok(()) => {}
            Err(Ok(error)) => return Ok(refusal(&error, &[])),
            Err(Err(input_error)) => return Err(input_error),
        }
    }
    let request = flow.request.clone().ok_or_else(|| bad("no request"))?;
    let rewritten = request
        .state
        .get("decomposition")
        .cloned()
        .unwrap_or(Value::Null);
    if rewritten.is_null() {
        return Ok(answer(Value::Null, &[]));
    }
    Ok(answer(
        json!({
            "kind": rewritten.get("kind").cloned().unwrap_or(Value::Null),
            "top": rewritten.get("top").cloned().unwrap_or(Value::Null),
            "parts": rewritten.get("parts").cloned().unwrap_or(json!([])),
        }),
        &[],
    ))
}

/// `payload.feature_column`: a `FEATURE` column on a users table, as a
/// relational payload sends it, over the graph `test_payload_dtype_fidelity.py`
/// builds: users with the feature, orders over a year, and a regression task
/// of fixed rows.
pub fn payload_feature_column(input: &Map<String, Value>) -> Result<String, InputError> {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, TimestampNanosecondArray};
    use sdm_request::series::{Dtype, Series};
    use sdm_sampling::frame::{Dtype as SDtype, Frame, Series as SSeries};

    let mut feature =
        Series::from_spec(input.get("feature").ok_or_else(|| bad("a feature"))?).map_err(bad)?;
    feature.name = Some("FEATURE".to_string());
    let n = feature.len();
    if n < 2 {
        return Err(bad("a feature of at least two users"));
    }
    let int64 = |name: &str, values: Vec<i64>| {
        Series::new(
            Some(name.to_string()),
            Dtype::Int("int64".to_string()),
            values
                .into_iter()
                .map(|v| PyVal::Int(i128::from(v)))
                .collect(),
        )
    };
    let users = vec![int64("USER_ID", (0..n as i64).collect()), feature];
    let orders_n = 6 * n;
    let day = 86_400_000_000_000_i128;
    let start = sdm_schema::dt::civil::Wall::date(2024, 1, 1).epoch_nanos();
    let orders = vec![
        int64("ORDER_ID", (0..orders_n as i64).collect()),
        int64(
            "USER_ID",
            (0..orders_n as i64).map(|i| i % n as i64).collect(),
        ),
        Series::new(
            Some("AMOUNT".to_string()),
            Dtype::Float(64),
            (0..orders_n)
                .map(|i| PyVal::Float(5.0 + (i % 95) as f64))
                .collect(),
        ),
        Series::new(
            Some("TIME".to_string()),
            Dtype::Datetime("ns".to_string(), None),
            (0..orders_n)
                .map(|i| {
                    PyVal::Timestamp(sdm_schema::dt::civil::Wall::from_epoch_nanos(
                        start + day * ((i * 37) % 420) as i128,
                    ))
                })
                .collect(),
        ),
    ];
    let host = Host::assumed();
    let mut quiet = sdm_request::error::Warnings::default();
    let built = (|| -> Result<sdm_request::gather::EngineGraph, sdm_schema::engine::PyExc> {
        let mut graph = sdm_request::gather::EngineGraph::default();
        let mut users = sdm_request::gather::LocalTable::new("USERS", users, n, &host, &mut quiet)?;
        users.set_primary_key(
            Some(&sdm_schema::client::Key::One("USER_ID".to_string())),
            &host,
            &mut quiet,
        )?;
        let mut orders =
            sdm_request::gather::LocalTable::new("ORDERS", orders, orders_n, &host, &mut quiet)?;
        orders.set_primary_key(
            Some(&sdm_schema::client::Key::One("ORDER_ID".to_string())),
            &host,
            &mut quiet,
        )?;
        orders.set_time_column(Some("TIME"))?;
        graph.tables.push(users);
        graph.tables.push(orders);
        graph.link("ORDERS", "USER_ID", "USERS")?;
        Ok(graph)
    })();
    let raised = |e: sdm_schema::engine::PyExc| {
        let class = if e.class.contains('.') {
            e.class.clone()
        } else {
            format!("builtins.{}", e.class)
        };
        refusal(&Diagnostic::refusal(&class, None, e.message.clone()), &[])
    };
    let graph = match built {
        Ok(graph) => graph,
        Err(e) => return Ok(raised(e)),
    };
    let stype = graph
        .table("USERS")
        .ok()
        .and_then(|t| t.column("FEATURE"))
        .map(|c| c.stype.value().to_string())
        .unwrap_or_default();
    let ns = |rows: usize| -> ArrayRef {
        let at = sdm_schema::dt::civil::Wall::date(2025, 2, 20).epoch_nanos() as i64;
        Arc::new(TimestampNanosecondArray::from(vec![at; rows]))
    };
    let s = |dtype: &str, array: ArrayRef| SSeries::new(SDtype::Numpy(dtype.to_string()), array);
    let context_rows = n - 2;
    let context = Frame::new(
        vec!["ENTITY".into(), "TARGET".into(), "ANCHOR_TIMESTAMP".into()],
        vec![
            s(
                "int64",
                Arc::new(Int64Array::from(
                    (0..context_rows as i64).collect::<Vec<_>>(),
                )),
            ),
            s(
                "float64",
                Arc::new(Float64Array::from(
                    (0..context_rows).map(|i| i as f64).collect::<Vec<_>>(),
                )),
            ),
            s("datetime64[ns]", ns(context_rows)),
        ],
        context_rows,
    );
    let pred = Frame::new(
        vec!["ENTITY".into(), "ANCHOR_TIMESTAMP".into()],
        vec![
            s(
                "int64",
                Arc::new(Int64Array::from(vec![n as i64 - 2, n as i64 - 1])),
            ),
            s("datetime64[ns]", ns(2)),
        ],
        2,
    );
    let task = match sdm_request::relational::TaskTable::of_frames(
        "regression",
        context,
        pred,
        "USERS",
        Some("ANCHOR_TIMESTAMP"),
    ) {
        Ok(task) => task,
        Err(e) => return Ok(raised(e)),
    };
    let mut run = sdm_protocol::AsyncRun::new(move |asker| async move {
        Ok(sdm_request::relational::materialize_task_over(&asker, &graph, task, &host).await)
    });
    let payloads = match run.next() {
        Ok(Step::Done(Ok(payloads))) => payloads,
        Ok(Step::Done(Err(e))) => return Ok(raised(e)),
        Ok(_) => return Err(bad("gathering asked the host for something")),
        Err(e) => return Err(bad(e.message)),
    };
    let payload = payloads.first().ok_or_else(|| bad("no payload"))?;
    let table = payload
        .pointer("/context/related_tables/USERS")
        .ok_or_else(|| bad("no USERS table"))?;
    let index = table
        .get("columns")
        .and_then(Value::as_array)
        .and_then(|c| c.iter().position(|n| n == "FEATURE"))
        .ok_or_else(|| bad("no FEATURE column"))?;
    let cells: Vec<Value> = table
        .get("rows")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().map(|r| r[index].clone()).collect())
        .unwrap_or_default();
    Ok(answer(
        json!({
            "stype": stype,
            "declared": payload.pointer("/schema/related_tables/USERS/columns/FEATURE").cloned().unwrap_or(Value::Null),
            "cells": cells,
        }),
        &[],
    ))
}

#[cfg(test)]
mod public_diagnostic_tests {
    use super::*;
    use sdm_schema::diagnostics;

    fn gathered(payloads: Value, parts: Value) -> Value {
        json!({
            "task": {"target": {}, "kind": "regression", "options": {}},
            "schema": {"kind": "data", "tables": [], "options": {}},
            "payloads": payloads,
            "parts": parts
        })
    }

    #[test]
    fn public_payloads_flatten_parts_and_preserve_fallback_order() {
        let parent = json!({"candidate": "parent"});
        let expected: Vec<Value> = (1..=5)
            .map(|candidate| json!({"candidate": candidate, "metadata": {"request_id": "<request-id>"}}))
            .collect();
        let parts: Vec<Value> = (1..=5)
            .map(|candidate| {
                gathered(
                    json!([{"candidate": candidate, "metadata": {"request_id": "actual-id"}}]),
                    Value::Null,
                )
            })
            .collect();
        let nested = gathered(Value::Null, json!([parts[1], parts[2]]));
        for children in [
            Value::Array(parts.clone()),
            json!([parts[0], nested, parts[3], parts[4]]),
        ] {
            let request = RequestH {
                state: gathered(json!([parent]), children),
                frames: Frames::default(),
            };
            assert_eq!(public_request_payloads(&request).unwrap(), expected);
        }
        for children in [
            Value::Null,
            json!([]),
            json!([gathered(json!([]), Value::Null)]),
        ] {
            let request = RequestH {
                state: gathered(json!([parent]), children),
                frames: Frames::default(),
            };
            assert_eq!(
                public_request_payloads(&request).unwrap(),
                vec![parent.clone()]
            );
        }
        let request = RequestH {
            state: gathered(json!([]), json!([])),
            frames: Frames::default(),
        };
        assert!(public_request_payloads(&request).unwrap().is_empty());
    }

    #[test]
    fn flow_serializes_canonical_warning_without_internal_metadata() {
        let mut flow = Flow::new(Vec::new(), Vec::new());
        flow.warnings.push(diagnostics::warning(
            diagnostics::VIEW_CONVERSION_WARNING,
            "converted view".into(),
        ));
        flow.warnings.push(
            Diagnostic::warning("builtins.UserWarning", "ordinary")
                .with_details(serde_json::from_value(json!({"retained": 7})).unwrap()),
        );
        flow.warnings.push(
            Diagnostic::warning("builtins.RuntimeWarning", "overflow encountered in cast")
                .with_details(
                    serde_json::from_value(json!({"masked_float32_cast": true})).unwrap(),
                ),
        );
        let refusal = Diagnostic::refusal("builtins.ValueError", None, "refused").with_details(
            serde_json::from_value(
                json!({"retained": 3, "_sdm_internal_warning": "ordinary error detail"}),
            )
            .unwrap(),
        );
        let (error, warnings) = flow.public_diagnostics(Some(&refusal));
        let raw = serde_json::to_string(&json!({"error": error, "warnings": warnings})).unwrap();
        let decoded: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            decoded["warnings"][0],
            json!({"class":"sdm_connectors._warnings.ViewConversionWarning","code":null,"message":"converted view","details":null})
        );
        assert_eq!(decoded["warnings"][1]["details"], json!({"retained":7}));
        assert_eq!(decoded["warnings"].as_array().unwrap().len(), 2);
        assert_eq!(
            decoded["error"],
            json!({"class":"builtins.ValueError","code":null,"message":"refused","details":{"retained":3,"_sdm_internal_warning":"ordinary error detail"}})
        );
        assert!(flow.public_diagnostics(None).0.is_null());
        assert_eq!(
            flow.warnings[0].details[diagnostics::INTERNAL_WARNING],
            diagnostics::VIEW_CONVERSION_WARNING
        );
    }
}
