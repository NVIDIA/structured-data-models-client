// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/request.py`'s `Request`: what to ask, built from frames or
//! gathered as wire envelopes, and the checks building one makes.
//!
//! Between calls a request is plain data: its JSON form (below) and the
//! frames it names, among the tables a call is given. A request built from
//! frames names `context` and `predict`, and its schema's frames
//! `schema/NAME`; a gathered request carries its envelopes in the JSON.

use crate::pyops::TaggedValue;
use sdm_protocol::Diagnostic;
use sdm_schema::client::Schema;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::{R, Warnings, invalid, unsupported};
use crate::frames::Table;
use crate::layout::encode_deep;
use crate::pyops::py_repr;
use crate::schemas::{DataSchema, Tables, schema_refusal};
use crate::series::is_na;
use crate::values::encode;
use crate::wire::{UNTYPED, infer_tfm_dtype, is_container};

/// `TASK_KINDS`.
pub const TASK_KINDS: [&str; 7] = [
    "classification",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "forecasting",
    "temporal_link_prediction",
    "ranking",
];

const CLASSIFICATION_KINDS: [&str; 3] = [
    "classification",
    "binary_classification",
    "multiclass_classification",
];

/// Whether `kind` is a classification.
pub fn is_classification(kind: &str) -> bool {
    CLASSIFICATION_KINDS.contains(&kind)
}

/// `Target`.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub column_name: PyVal,
    pub dtype: PyVal,
    pub table_name: Option<String>,
}

/// `Task`.
#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub kind: PyVal,
    pub target: Target,
    pub entity_table: Option<PyVal>,
    pub options: Vec<(PyVal, PyVal)>,
}

/// `Decomposition`: the questions a ranking or forecast is asked as.
#[derive(Clone, Debug, PartialEq)]
pub struct Decomposition {
    pub kind: String,
    /// Each part's query and what it is about: a candidate or a step.
    pub parts: Vec<(String, PyVal)>,
    pub top: Option<i64>,
}

/// The halves of a request built from frames.
#[derive(Clone, Debug)]
pub struct Halves {
    pub context: Table,
    pub predict: Table,
}

/// A request.
#[derive(Clone, Debug)]
pub struct Request {
    pub protocols: Option<crate::pyops::HostProtocols>,
    pub coerced_mappings: Map<String, Value>,
    pub task: Task,
    pub schema: DataSchema,
    pub entity_table: Option<PyVal>,
    pub query: Option<PyVal>,
    pub inference: Vec<(PyVal, PyVal)>,
    pub requested_inference: Value,
    pub halves: Option<Halves>,
    pub payloads: Option<Vec<Value>>,
    pub indices: PyVal,
    pub windowed: PyVal,
    pub parts: Option<Vec<Request>>,
    pub decomposition: Option<Decomposition>,
    pub asked: Option<String>,
    /// What stands for a context built from frames, compared by sessions.
    pub context_fingerprint: Option<String>,
}

/// The table names a request built from frames gives its halves.
pub const CONTEXT: &str = "context";
pub const PREDICT: &str = "predict";

impl Request {
    /// Attach operation-local host protocols without changing serialized request state.
    pub fn with_protocols(mut self, protocols: Option<crate::pyops::HostProtocols>) -> Self {
        if let Some(parts) = &mut self.parts {
            for part in parts {
                *part = part.clone().with_protocols(protocols.clone());
            }
        }
        self.protocols = protocols;
        self
    }

    /// `request.payloads`: a gathered request's envelopes, its parts' where
    /// it is asked as several.
    pub fn payloads(&self) -> Option<Vec<Value>> {
        let mine = self.payloads.clone().filter(|p| !p.is_empty());
        match &self.parts {
            None => mine,
            Some(parts) => {
                let gathered: Vec<Value> = parts
                    .iter()
                    .flat_map(|p| p.payloads().unwrap_or_default())
                    .collect();
                if gathered.is_empty() {
                    mine
                } else {
                    Some(gathered)
                }
            }
        }
    }

    /// `_refuse_if_gathered(half)`: a half of a gathered request is not
    /// replaced, since its rows are already encoded.
    pub fn refuse_if_gathered(&self, half: &str) -> crate::error::R<()> {
        if self.payloads().is_none() {
            return Ok(());
        }
        Err(crate::error::invalid(format!(
            "{half} cannot be replaced on a request gathered by a predictive query: its rows are \
             already encoded, and the encoding carries what a frame does not. Materialize the \
             query again for the rows you want instead."
        )))
    }

    /// `request.nothing_to_score`.
    pub fn nothing_to_score(&self) -> R<bool> {
        if let Some(payloads) = self.payloads() {
            let counts = payloads
                .iter()
                .map(|payload| envelope_rows_with(payload, PREDICT, self.protocols.as_ref()))
                .collect::<R<Vec<_>>>()?;
            return Ok(
                counts.iter().all(Option::is_some) && !counts.iter().any(|c| c.unwrap_or(0) > 0)
            );
        }
        Ok(self.halves.as_ref().is_none_or(|h| h.predict.is_empty()))
    }

    /// `request.context_rows`.
    pub fn context_rows(&self) -> R<usize> {
        if let Some(payloads) = self.payloads() {
            return payloads
                .iter()
                .map(|payload| {
                    envelope_rows_with(payload, CONTEXT, self.protocols.as_ref())
                        .map(|count| count.unwrap_or(0))
                })
                .sum();
        }
        Ok(self.rows(CONTEXT))
    }

    /// `request.predict_rows`.
    pub fn predict_rows(&self) -> usize {
        self.rows(PREDICT)
    }

    fn rows(&self, half: &str) -> usize {
        if let Some(payloads) = self.payloads() {
            return payloads
                .iter()
                .map(|p| envelope_rows(p, half).unwrap_or(0))
                .sum();
        }
        match &self.halves {
            None => 0,
            Some(h) => {
                if half == CONTEXT {
                    h.context.len()
                } else {
                    h.predict.len()
                }
            }
        }
    }

    /// The JSON form, and the frames it names.
    pub fn to_json(&self, tables: &mut Tables) -> Value {
        if let Some(halves) = &self.halves {
            tables.insert(CONTEXT, halves.context.clone());
            tables.insert(PREDICT, halves.predict.clone());
        }
        let mut parts = Vec::new();
        for part in self.parts.iter().flatten() {
            parts.push(part.to_json(tables));
        }
        let mut said = json!({
            "task": {
                "kind": encode_deep(&self.task.kind),
                "target": {
                    "column_name": encode_deep(&self.task.target.column_name),
                    "dtype": encode_deep(&self.task.target.dtype),
                    "table_name": self.task.target.table_name,
                },
                "entity_table": self.task.entity_table.as_ref().map(encode_deep),
                "options": self.coerced_mappings.get("options").cloned().unwrap_or_else(|| crate::layout::encode_mapping(&self.task.options)),
            },
            "schema": self.schema.to_request_json(),
            "entity_table": self.entity_table.as_ref().map(encode_deep),
            "query": self.query.as_ref().map(encode_deep),
            "inference": self.coerced_mappings.get("inference").cloned().unwrap_or_else(|| crate::layout::encode_mapping(&self.inference)),
            "requested_inference": self.requested_inference,
            "frames": self.halves.is_some(),
            "payloads": self.payloads,
            "indices": encode_deep(&self.indices),
            "windowed": encode_deep(&self.windowed),
            "parts": if self.parts.is_some() { Value::Array(parts) } else { Value::Null },
            "decomposition": self.decomposition.as_ref().map(|d| json!({
                "kind": d.kind,
                "parts": d.parts.iter().map(|(t, a)| json!([t, encode_deep(a)])).collect::<Vec<_>>(),
                "top": d.top,
            })),
            "asked": self.asked,
            "context_fingerprint": self.context_fingerprint,
        });
        // Where each half of frames came from in the frames a host gave, which
        // it reads them back from.
        if let Some(halves) = &self.halves
            && (halves.context.rows.is_some() || halves.predict.rows.is_some())
        {
            said["rows"] = json!({
                "context": halves.context.rows,
                "predict": halves.predict.rows,
            });
        }
        said
    }

    /// A request read back from its JSON form and the tables it names.
    pub fn from_json(value: &Value, tables: &Tables) -> Result<Request, String> {
        let task = value.py_get("task").ok_or("a request has a task")?;
        let target = task.py_get("target").ok_or("a task has a target")?;
        let text = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_string);
        let optional = |value: Option<&Value>| {
            value
                .filter(|value| !value.is_null())
                .map(PyVal::decode)
                .transpose()
        };
        let mapping =
            |value: Option<&Value>| match PyVal::decode(value.unwrap_or(&json!({})))?.semantic() {
                PyVal::Dict(pairs) => Ok(pairs.clone()),
                _ => Err("a request's options and inference must be mappings".to_string()),
            };
        let halves = if value
            .py_get("frames")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            Some(Halves {
                context: tables
                    .get(CONTEXT)
                    .cloned()
                    .ok_or("no context frame was given")?,
                predict: tables
                    .get(PREDICT)
                    .cloned()
                    .ok_or("no predict frame was given")?,
            })
        } else {
            None
        };
        let parts = match value.py_get("parts") {
            Some(Value::Array(items)) => Some(
                items
                    .iter()
                    .map(|p| Request::from_json(p, tables))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            _ => None,
        };
        let decomposition = match value.py_get("decomposition") {
            Some(Value::Object(d)) => Some(Decomposition {
                kind: text(d.py_get("kind")).unwrap_or_default(),
                parts: d
                    .py_get("parts")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .map(|pair| {
                                Ok((
                                    text(pair.get(0)).unwrap_or_default(),
                                    PyVal::decode(pair.get(1).unwrap_or(&Value::Null))?,
                                ))
                            })
                            .collect::<Result<Vec<_>, String>>()
                    })
                    .transpose()?
                    .unwrap_or_default(),
                top: d.py_get("top").and_then(Value::as_i64),
            }),
            _ => None,
        };
        // What stands for a context of frames, where the request read back
        // carries none: the context it holds now, as `_question` reads it.
        let context_fingerprint = text(value.py_get("context_fingerprint"))
            .or_else(|| halves.as_ref().map(|h| fingerprint(&h.context)));
        Ok(Request {
            protocols: None,
            coerced_mappings: [
                ("options", task.py_get("options")),
                ("inference", value.py_get("inference")),
            ]
            .into_iter()
            .filter_map(|(name, value)| {
                value
                    .filter(|value| value.get("$held").is_some())
                    .map(|value| (name.into(), value.clone()))
            })
            .collect(),
            task: Task {
                kind: PyVal::decode(task.py_get("kind").unwrap_or(&json!("")))?,
                target: Target {
                    column_name: PyVal::decode(target.py_get("column_name").unwrap_or(&json!("")))?,
                    dtype: PyVal::decode(target.py_get("dtype").unwrap_or(&json!("")))?,
                    table_name: text(target.py_get("table_name")),
                },
                entity_table: optional(task.py_get("entity_table"))?,
                options: mapping(task.py_get("options"))?,
            },
            schema: DataSchema::from_json(value.py_get("schema").unwrap_or(&Value::Null))?,
            entity_table: optional(value.py_get("entity_table"))?,
            query: optional(value.py_get("query"))?,
            inference: mapping(value.py_get("inference"))?,
            requested_inference: value
                .py_get("requested_inference")
                .cloned()
                .unwrap_or(json!({})),
            halves,
            payloads: value.py_get("payloads").and_then(Value::as_array).cloned(),
            indices: PyVal::decode(value.py_get("indices").unwrap_or(&json!([])))?,
            windowed: PyVal::decode(value.py_get("windowed").unwrap_or(&json!(false)))?,
            parts,
            decomposition,
            asked: text(value.py_get("asked")),
            context_fingerprint,
        })
    }
}

/// `_envelope_rows(payload, half)`.
pub fn envelope_rows(payload: &Value, half: &str) -> Option<usize> {
    payload
        .py_get(half)?
        .py_get("instance_table")?
        .py_get("rows")?
        .as_array()
        .map(Vec::len)
}

fn envelope_rows_with(
    payload: &Value,
    half: &str,
    protocols: Option<&crate::pyops::HostProtocols>,
) -> R<Option<usize>> {
    let mut value = payload.clone();
    for key in [half, "instance_table", "rows"] {
        if !crate::pyops::is_instance(&value, crate::pyops::InstanceFamily::Mapping, protocols)? {
            return Ok(None);
        }
        value = crate::pyops::mapping_get(&value, key, &Value::Null, protocols)?;
    }
    Ok(crate::pyops::sequence(&value).map(Vec::len))
}

/// A frame argument as the host passes it: a frame it described, or what
/// it was instead.
#[derive(Clone, Debug)]
pub enum FrameArg {
    Frame(Table),
    Other(PyVal),
}

/// `_require_frame(value, name)`.
pub fn require_frame(value: &FrameArg, name: &str) -> R<Table> {
    match value {
        FrameArg::Frame(table) => Ok(table.clone()),
        FrameArg::Other(other) => {
            let type_name = crate::models::type_name(other);
            let hint = if matches!(other, PyVal::Dict(_)) || type_name == "Series" {
                "; wrap a single row with pd.DataFrame([row])"
            } else {
                "; build one with pd.DataFrame(...)"
            };
            Err(invalid(format!(
                "{name} must be a pandas DataFrame, got {type_name}{hint}"
            )))
        }
    }
}

/// `_refuse_tables_no_model_can_be_given(related)`.
pub fn refuse_tables_no_model_can_be_given(related: &[String]) -> R<()> {
    if related.is_empty() {
        return Ok(());
    }
    let mut names = related.to_vec();
    names.sort();
    let shown = sdm_schema::py::str_list_repr(&names);
    Err(unsupported(
        format!(
            "related_tables {shown} makes this a relational request, and no model can be sent one \
             built from frames: the relational model is sent tables gathered around each entity, \
             and the tabular model reads a single table. Gather it with \
             PredictiveQuery(...).materialize(indices=[...], model='kumo-relational'), whose \
             schema holds the related tables and the relationships that reach them."
        ),
        crate::error::details([("related_tables", json!(names))]),
    ))
}

/// What `Request(...)` is given.
pub struct FrameArgs {
    pub context: FrameArg,
    pub predict: FrameArg,
    pub target: PyVal,
    pub task: PyVal,
    pub related: Vec<(String, FrameArg)>,
    /// A schema of the caller's, with the frames it names.
    pub schema: Option<DataSchema>,
    pub entity_table: String,
    pub inference: Map<String, Value>,
    pub task_options: Map<String, Value>,
}

fn sorted_texts(names: &[String]) -> String {
    let mut sorted = names.to_vec();
    sorted.sort();
    sdm_schema::py::str_list_repr(&sorted)
}

/// `Request(context, predict, target=..., task=..., ...)`: the checks and
/// the request, with its schema's frames put among `tables`.
pub fn from_frames(
    args: FrameArgs,
    tables: &mut Tables,
    host: &Host,
    said: &mut Warnings,
) -> R<Request> {
    let context = require_frame(&args.context, "context")?;
    let predict = require_frame(&args.predict, "predict")?;
    let mut related = Vec::new();
    for (name, frame) in &args.related {
        require_frame(
            frame,
            &format!("related_tables[{}]", sdm_pql::pyrepr::str_repr(name)),
        )?;
        related.push(name.clone());
    }
    refuse_tables_no_model_can_be_given(&related)?;
    let task = match &args.task {
        PyVal::Str(kind) if TASK_KINDS.contains(&kind.as_str()) => kind.clone(),
        other => {
            let kinds: Vec<String> = TASK_KINDS.iter().map(|k| k.to_string()).collect();
            return Err(invalid(format!(
                "task must be one of {}, got {}",
                sdm_schema::py::str_list_repr(&kinds),
                py_repr(other)
            )));
        }
    };
    let target = match &args.target {
        PyVal::Str(t) => Some(t.clone()),
        _ => None,
    };
    let in_context = target.as_ref().is_some_and(|t| context.has(t));
    if let Some(t) = &target
        && in_context
        && context
            .series(t)
            .is_some_and(|s| s.cells.iter().any(is_container))
    {
        return Err(crate::error::structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "{} holds a list per row, which is a multi-label target. These models predict \
                 one value per row, and the contract has no form for a set of labels, so this \
                 cannot be asked or answered",
                sdm_pql::pyrepr::str_repr(t)
            ),
        ));
    }
    if context.is_empty() {
        return Err(invalid(
            "context has no rows, so there is nothing for the model to learn the answer from",
        ));
    }
    if !in_context {
        return Err(invalid(format!(
            "context has no target column {}; it has {}",
            py_repr(&args.target),
            sorted_texts(&context.columns())
        )));
    }
    let target = target.unwrap_or_default();
    if predict.has(&target) {
        let name = sdm_pql::pyrepr::str_repr(&target);
        return Err(invalid(format!(
            "predict carries the target column {name}. Those are the rows whose answer is being \
             asked for, so the column has to be absent; drop it with \
             predict.drop(columns=[{name}])"
        )));
    }
    let answers = context.series(&target).map(|s| s.cells).unwrap_or_default();
    if answers.iter().all(is_na) {
        return Err(invalid(
            "context must contain at least one row with a known target",
        ));
    }
    let context_columns = context.columns();
    let missing: Vec<String> = predict
        .columns()
        .into_iter()
        .filter(|c| !context.has_label_of(&predict, c))
        .collect();
    if !missing.is_empty() {
        return Err(invalid(format!(
            "predict has column(s) {} that context does not; the model is shown context first, \
             so it cannot score a feature it has never been given",
            predict.labels_repr(&missing)
        )));
    }
    let schema_handle = match args.schema {
        Some(schema) => schema,
        None => {
            let handle = DataSchema::over_frames(
                vec![args.entity_table.clone()],
                json!({"relationships": []}),
            );
            tables.insert(&DataSchema::frame_key(&args.entity_table), context.clone());
            let (handle, built_said) = handle.described(tables, host)?;
            for warning in built_said.warnings {
                said.push(Diagnostic::warning(&warning.category, warning.message));
            }
            handle
        }
    };
    let schema = schema_handle.build(tables, host)?;
    check_scored_values_mean_what_the_schema_says(&predict, &schema, &args.entity_table)?;
    let table = schema_table(&schema, &args.entity_table)?;
    let identity = sdm_schema::client::key_names_of(&table.primary_key);
    let absent: Vec<String> = context_columns
        .iter()
        .filter(|c| {
            let text = context.label_not_text(c).is_none();
            !(text && (**c == target || identity.contains(c))) && !predict.has_label_of(&context, c)
        })
        .cloned()
        .collect();
    if !absent.is_empty() {
        return Err(invalid(format!(
            "predict is missing column(s) {} that context carries. Every feature the model is \
             shown has to be supplied for the rows being scored; drop them from context, or \
             supply them",
            context.labels_repr(&absent)
        )));
    }
    judge_a_context_that_teaches_nothing(&target, &answers, &task, said);
    let dtype = column_of(table, &target)?.dtype.clone();
    let fingerprint = Some(fingerprint(&context));
    Ok(Request {
        protocols: None,
        coerced_mappings: Map::new(),
        task: Task {
            kind: PyVal::Str(task),
            target: Target {
                column_name: PyVal::Str(target),
                dtype: PyVal::Str(dtype),
                table_name: Some(args.entity_table.clone()),
            },
            entity_table: None,
            options: args
                .task_options
                .into_iter()
                .map(|(key, value)| (PyVal::Str(key), crate::keys::json_cell(&value)))
                .collect(),
        },
        schema: schema_handle,
        entity_table: Some(PyVal::Str(args.entity_table)),
        query: None,
        inference: args
            .inference
            .iter()
            .map(|(key, value)| (PyVal::Str(key.clone()), crate::keys::json_cell(value)))
            .collect(),
        requested_inference: Value::Object(args.inference),
        halves: Some(Halves { context, predict }),
        payloads: None,
        indices: PyVal::List(Vec::new()),
        windowed: PyVal::Bool(false),
        parts: None,
        decomposition: None,
        asked: None,
        context_fingerprint: fingerprint,
    })
}

/// `schema[name]`.
pub fn schema_table<'a>(schema: &'a Schema, name: &str) -> R<&'a sdm_schema::client::Table> {
    schema
        .tables
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| no_table(schema, &sdm_pql::pyrepr::str_repr(name)))
}

/// `schema[name]` with `name` possibly `None`, which names no table.
pub fn schema_table_or_none<'a>(
    schema: &'a Schema,
    name: Option<&str>,
) -> R<&'a sdm_schema::client::Table> {
    match name {
        Some(name) => schema_table(schema, name),
        None => Err(no_table(schema, "None")),
    }
}

/// What `schema[name]` refuses a name it has no table of with, `shown` the
/// name's `repr`.
fn no_table(schema: &Schema, shown: &str) -> Diagnostic {
    let names: Vec<String> = schema.tables.iter().map(|t| t.name.clone()).collect();
    invalid(format!(
        "no table {shown} in this schema; it has {}",
        sorted_texts(&names)
    ))
}

/// `table[name]`.
pub fn column_of<'a>(
    table: &'a sdm_schema::client::Table,
    name: &str,
) -> R<&'a sdm_schema::client::Col> {
    table.col(name).ok_or_else(|| {
        let names: Vec<String> = table.columns.iter().map(|c| c.name.clone()).collect();
        invalid(format!(
            "table {} has no column {}; it has {}",
            sdm_pql::pyrepr::str_repr(&table.name),
            sdm_pql::pyrepr::str_repr(name),
            sorted_texts(&names)
        ))
    })
}

/// A value standing for both halves of a request built from frames, which
/// sessions compare: equal frames give equal ones.
/// `_what_the_context_holds(context)`: a value standing for the labelled
/// rows, equal for equal frames.
fn fingerprint(context: &Table) -> String {
    let mut text = String::new();
    for table in [context] {
        text.push_str(&table.columns().join("\u{1}"));
        for name in table.columns() {
            if let Some(series) = table.series(&name) {
                for cell in &series.cells {
                    text.push_str(&encode(cell).to_string());
                }
            }
        }
        text.push('\u{2}');
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// `_judge_a_context_that_teaches_nothing(column, target, kind)`.
fn judge_a_context_that_teaches_nothing(
    column: &str,
    target: &[PyVal],
    kind: &str,
    said: &mut Warnings,
) {
    let answered: Vec<&PyVal> = target.iter().filter(|v| !is_na(v)).collect();
    if answered.is_empty() {
        return;
    }
    let mut seen: Vec<String> = Vec::new();
    for value in &answered {
        let key = value.eq_key();
        if !seen.contains(&key) {
            seen.push(key);
        }
    }
    if seen.len() >= 2 {
        return;
    }
    if !is_classification(kind) && kind != "regression" {
        return;
    }
    let what = if is_classification(kind) {
        "a classification needs at least two classes to learn from, and the answer will be that \
         one class at probability 1.0"
    } else {
        "nothing in the context distinguishes one answer from another, so the answer cannot say \
         much"
    };
    said.sdm(format!(
        "every context row has the same {column}, so {what}. If the rows being scored are the \
         ones that differ, score fewer of them; if {column} is constant in the whole table, \
         there is no context that answers this question."
    ));
}

/// `_check_scored_values_mean_what_the_schema_says(predict, schema, entity)`.
pub fn check_scored_values_mean_what_the_schema_says(
    predict: &Table,
    schema: &Schema,
    entity_table: &str,
) -> R<()> {
    let Some(table) = schema.tables.iter().find(|t| t.name == entity_table) else {
        return Ok(());
    };
    for name in predict.columns() {
        let Some(column) = table.col(&name) else {
            continue;
        };
        if column.untyped && !column.stype_declared {
            continue;
        }
        let Some(series) = predict.series(&name) else {
            continue;
        };
        let arriving = infer_tfm_dtype(&series);
        if arriving == UNTYPED || sdm_schema::client::supports_dtype(&column.stype, &arriving) {
            continue;
        }
        let n = sdm_pql::pyrepr::str_repr(&name);
        let e = sdm_pql::pyrepr::str_repr(entity_table);
        return Err(invalid(format!(
            "the rows being scored hold {} in column {n}, and the schema declares it {} meaning \
             {}, which {} cannot mean. Sent, the column would be described as both at once and \
             answered with numbers that are wrong without saying so. Convert it before scoring: \
             predict[{n}] = pd.to_numeric(predict[{n}]) or say what the column really is with \
             schema[{e}][{n}].stype = ...",
            sdm_pql::pyrepr::str_repr(&arriving),
            sdm_pql::pyrepr::str_repr(&column.dtype),
            sdm_pql::pyrepr::str_repr(&column.stype),
            sdm_pql::pyrepr::str_repr(&arriving)
        )));
    }
    Ok(())
}

/// A refusal of the schema core, as raised today.
pub fn from_schema(refusal: sdm_schema::client::Refusal) -> Diagnostic {
    schema_refusal(&refusal)
}
