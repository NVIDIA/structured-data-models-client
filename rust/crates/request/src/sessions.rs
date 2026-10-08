// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sessions (`client.py`, `SDMClient.fit` and `FittedModel`): a context
//! pinned once on the endpoint, and rows scored against it after.
//!
//! What `fit` returns is held by the host between calls as plain JSON
//! ([`Fitted::to_json`]), and handed back with each request scored.

use regex::Regex;
use sdm_protocol::{Asker, Endpoint};
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::client::{Asked, call, empty_answer, normalize, prepare, report_absent};
use crate::error::{R, Warnings, details, invalid, structured};
use crate::keys::Named;
use crate::models::{Profile, Spec};
use crate::request::Request;
use crate::result::{Result_, py_eq};
use crate::schemas::Tables;

/// `_SESSIONS_UNSUPPORTED`: the statuses of an endpoint without sessions.
const SESSIONS_UNSUPPORTED: [i32; 3] = [404, 405, 501];

/// `Question`: what a request asks, apart from the rows it asks about.
#[derive(Clone, Debug)]
pub struct Question {
    pub query: Option<String>,
    pub kind: PyVal,
    pub target: PyVal,
    pub dtype: PyVal,
    pub entity_table: Option<PyVal>,
    pub context: Option<String>,
}

impl PartialEq for Question {
    fn eq(&self, other: &Self) -> bool {
        self.query == other.query
            && py_eq(&self.kind, &other.kind)
            && py_eq(&self.target, &other.target)
            && py_eq(&self.dtype, &other.dtype)
            && match (&self.entity_table, &other.entity_table) {
                (Some(left), Some(right)) => py_eq(left, right),
                (None, None) => true,
                _ => false,
            }
            && self.context == other.context
    }
}

impl Question {
    pub fn to_json(&self) -> Value {
        json!({
            "query": self.query,
            "kind": crate::layout::encode_deep(&self.kind),
            "target": crate::layout::encode_deep(&self.target),
            "dtype": crate::layout::encode_deep(&self.dtype),
            "entity_table": self.entity_table.as_ref().map(crate::layout::encode_deep),
            "context": self.context,
        })
    }

    pub fn from_json(value: &Value) -> Option<Self> {
        let text = |name: &str| value.get(name).and_then(Value::as_str).map(str::to_string);
        Some(Question {
            query: text("query"),
            kind: PyVal::decode(value.get("kind")?).ok()?,
            target: PyVal::decode(value.get("target")?).ok()?,
            dtype: PyVal::decode(value.get("dtype")?).ok()?,
            entity_table: value
                .get("entity_table")
                .filter(|value| !value.is_null())
                .map(PyVal::decode)
                .transpose()
                .ok()?,
            context: text("context"),
        })
    }
}

/// `json.dumps(text)`: a JSON string as Python writes one, every character
/// outside ASCII escaped.
fn json_dumps_str(text: &str) -> String {
    let mut out = String::from("\"");
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
    out
}

/// `_one_spelling_of_each_literal(text)`: every quoted literal of a query
/// written one way.
fn one_spelling_of_each_literal(text: &str) -> String {
    let pattern = Regex::new(r#""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'"#).expect("pattern compiles");
    pattern
        .replace_all(text, |captures: &regex::Captures<'_>| {
            let literal = &captures[0];
            let mut inner = literal[1..literal.len() - 1].to_string();
            if literal.starts_with('\'') {
                inner = inner.replace("\\'", "'").replace('"', "\\\"");
            }
            match serde_json::from_str::<String>(&format!("\"{inner}\"")) {
                Ok(read) => json_dumps_str(&read),
                Err(_) => literal.to_string(),
            }
        })
        .into_owned()
}

/// `request._question`.
pub fn question_of(request: &Request) -> R<Question> {
    let query = match &request.query {
        None => None,
        Some(PyVal::Str(text)) => Some(one_spelling_of_each_literal(text)),
        Some(PyVal::Bytes(_)) => {
            return Err(crate::error::builtin(
                "TypeError",
                "cannot use a string pattern on a bytes-like object",
            ));
        }
        Some(other) => {
            return Err(crate::error::builtin(
                "TypeError",
                format!(
                    "expected string or bytes-like object, got '{}'",
                    crate::models::type_name(other)
                ),
            ));
        }
    };
    Ok(Question {
        query,
        kind: request.task.kind.clone(),
        target: request.task.target.column_name.clone(),
        dtype: request.task.target.dtype.clone(),
        entity_table: request.entity_table.clone(),
        context: request.context_fingerprint.clone(),
    })
}

/// What `SDMClient.fit` returns: a `FittedModel`, as the host keeps it.
#[derive(Clone, Debug)]
pub struct Fitted {
    pub session_id: String,
    pub model: String,
    pub outputs: Vec<String>,
    pub named_outputs: Option<Vec<String>>,
    /// What the session pinned, but the context.
    pub pinned: Value,
    pub pinned_keys: Vec<Named>,
    pub pinned_over_a_window: bool,
    pub held_back: usize,
    pub question: Option<Question>,
    /// Whether the host has released the session.
    pub closed: bool,
}

impl Fitted {
    pub fn to_json(&self) -> Value {
        json!({
            "session_id": self.session_id,
            "model": self.model,
            "outputs": self.outputs,
            "named_outputs": self.named_outputs,
            "pinned": self.pinned,
            "pinned_keys": self.pinned_keys,
            "pinned_over_a_window": self.pinned_over_a_window,
            "held_back": self.held_back,
            "question": self.question.as_ref().map(Question::to_json),
        })
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        let texts = |name: &str| -> Option<Vec<String>> {
            value.get(name).and_then(Value::as_array).map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
        };
        Ok(Fitted {
            session_id: value
                .get("session_id")
                .and_then(Value::as_str)
                .ok_or("a fitted model names its session")?
                .to_string(),
            model: value
                .get("model")
                .and_then(Value::as_str)
                .ok_or("a fitted model names its model")?
                .to_string(),
            outputs: texts("outputs").unwrap_or_default(),
            named_outputs: texts("named_outputs"),
            pinned: value.get("pinned").cloned().unwrap_or_else(|| json!({})),
            pinned_keys: texts("pinned_keys").unwrap_or_default(),
            pinned_over_a_window: value
                .get("pinned_over_a_window")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            held_back: value.get("held_back").and_then(Value::as_u64).unwrap_or(0) as usize,
            question: value.get("question").and_then(Question::from_json),
            closed: value
                .get("closed")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// `_refuse_to_pin_a_rewritten_question(request)`.
fn refuse_to_pin_a_rewritten_question(request: &Request) -> R<()> {
    let Some(rewritten) = &request.decomposition else {
        return Ok(());
    };
    Err(invalid(format!(
        "a {} is asked as {} questions, each needing the context its own labels come from, and \
         a session pins one context. Ask it with predict(), which sends each and puts the \
         answers back together.",
        rewritten.kind,
        rewritten.parts.len()
    )))
}

/// `_held_back(payload, holdout)`: the rows a session will score taken out
/// of the context it pins; how many.
fn held_back(
    payload: &mut Value,
    holdout: Option<&PyVal>,
    protocol: Option<&crate::materialize::SequenceProtocol>,
    said: &mut Warnings,
) -> R<usize> {
    let Some(holdout) = holdout.filter(|h| !matches!(h, PyVal::None)) else {
        return Ok(0);
    };
    let items = match (holdout, protocol) {
        (PyVal::Str(_) | PyVal::Bytes(_), _) => {
            let shown = crate::pyops::py_repr(holdout);
            return Err(invalid(format!(
                "holdout names the rows this session may score, and {shown} is one string rather \
                 than a list of keys; pass [{shown}] if that is the key you mean"
            )));
        }
        (_, Some(protocol)) => protocol.items("iter")?,
        (PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items), None) => {
            items.clone()
        }
        (other, None) => vec![other.clone()],
    };
    let mut keys: Vec<Named> = Vec::new();
    for key in &items {
        let named = match key {
            PyVal::List(parts) | PyVal::Tuple(parts) => crate::keys::named_by(parts),
            other => crate::keys::named_by(std::slice::from_ref(other)),
        };
        if !keys.contains(&named) {
            keys.push(named);
        }
    }
    if keys.is_empty() {
        return Ok(0);
    }
    let removed = crate::keys::without_the_keys(payload, &keys);
    if removed == 0 {
        said.sdm(format!(
            "none of the {} keys named as holdout matched a row of the context about to be \
             pinned, so nothing was held back. The keys are compared as the schema spells them, \
             so a key given as a different type matches nothing. Check they are the same kind \
             of value as the entity column holds.",
            keys.len()
        ));
    } else {
        said.sdm(format!(
            "{removed} of the rows named as holdout were in the context about to be pinned, \
             carrying the answers this session will be asked for. They have been taken out of \
             it, so the context is that many rows smaller than the request gathered."
        ));
    }
    Ok(removed)
}

/// What fitting takes beside the request.
pub struct Fit<'a> {
    pub asked: &'a Asked,
    pub holdout: Option<&'a PyVal>,
    pub holdout_protocol: Option<&'a crate::materialize::SequenceProtocol>,
}

fn session_plan(
    spec: &Spec,
    request: &Request,
    tables: &Tables,
    fields: &[String],
    host: &Host,
    said: &mut Warnings,
) -> R<Value> {
    match spec.profile {
        Profile::Relational => crate::relational_adapter::session_plan(request, fields, said),
        Profile::Tabular => {
            let mut schema = request.schema.build(tables, host)?;
            let envelope =
                crate::tabular_adapter::envelope(request, &mut schema, fields, None, said)?;
            Ok(crate::tabular_adapter::pinned_half(&envelope))
        }
    }
}

/// `SDMClient.fit(request, model=..., outputs=..., holdout=...)`.
pub async fn fit(
    asker: &Asker,
    request: &Request,
    tables: &Tables,
    how: Fit<'_>,
    host: &Host,
    said: &mut Warnings,
) -> R<Fitted> {
    refuse_to_pin_a_rewritten_question(request)?;
    let spec = crate::models::spec_for(&how.asked.model)?;
    let schema = request.schema.build(tables, host)?;
    let fields = prepare(request, &schema, spec, how.asked.outputs.as_deref())?;
    let mut payload = session_plan(spec, request, tables, &fields, host, said)?;
    let held = held_back(&mut payload, how.holdout, how.holdout_protocol, said)?;
    let created = match call(asker, Endpoint::SessionCreate, None, Some(&payload)).await {
        Ok(created) => created,
        Err(error) => {
            let explained = if crate::parts::is_structured(&error) {
                crate::client::explained(spec, error, &payload)
            } else {
                error
            };
            if explained
                .status
                .is_some_and(|s| SESSIONS_UNSUPPORTED.contains(&s))
            {
                // Raised from what the call raised: the host's own error,
                // or what the adapter made of it.
                let beneath = if explained.chain.is_empty() {
                    vec![crate::client::from_model()]
                } else {
                    explained.as_links()
                };
                return Err(structured(
                    "SESSIONS_UNSUPPORTED",
                    format!(
                        "this endpoint does not serve sessions (HTTP {}); use predict() instead, \
                         which sends the context with every call",
                        explained.status.unwrap_or(0)
                    ),
                )
                .caused_by(beneath));
            }
            return Err(explained);
        }
    };
    let session_id = ["session_id", "id"]
        .iter()
        .filter_map(|k| created.get(*k))
        .find(|v| crate::pyops::py_bool(&crate::keys::json_cell(v)))
        .map(|v| crate::pyops::py_str(&crate::keys::json_cell(v)));
    let Some(session_id) = session_id else {
        let mut keys = crate::client::object_keys(&created);
        keys.sort();
        return Err(structured(
            "INVALID_RESPONSE",
            "the endpoint created a session but named no session id",
        )
        .with_details(details([("response_keys", json!(keys))])));
    };
    let pinned: Map<String, Value> = payload
        .as_object()
        .map(|o| {
            o.iter()
                .filter(|(k, _)| k.as_str() != "context")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();
    Ok(Fitted {
        session_id,
        model: spec.model_id.to_string(),
        outputs: fields,
        named_outputs: how.asked.outputs.clone(),
        pinned: Value::Object(pinned),
        pinned_keys: crate::keys::keys_in(
            &payload,
            "context",
            crate::pyops::py_bool(&request.windowed),
            request.protocols.as_ref(),
        )?,
        pinned_over_a_window: crate::pyops::py_bool(&request.windowed),
        held_back: held,
        question: Some(question_of(request)?),
        closed: false,
    })
}

/// `_WHAT_DIFFERS`.
const WHAT_DIFFERS: [(&str, &str); 5] = [
    ("query", "the query"),
    ("kind", "the task kind"),
    ("target", "the target column"),
    ("dtype", "the target column's dtype"),
    ("entity_table", "the entity table"),
];

/// `_refuse_a_question_the_session_did_not_pin(pinned, asked)`.
fn refuse_a_question_the_session_did_not_pin(pinned: &Question, asked: &Question) -> R<()> {
    if pinned == asked {
        return Ok(());
    }
    let field = |q: &Question, name: &str| -> PyVal {
        match name {
            "query" => crate::keys::json_cell(&json!(q.query)),
            "kind" => q.kind.clone(),
            "target" => q.target.clone(),
            "dtype" => q.dtype.clone(),
            _ => q.entity_table.clone().unwrap_or(PyVal::None),
        }
    };
    let said = WHAT_DIFFERS
        .iter()
        .find(|(name, _)| !py_eq(&field(pinned, name), &field(asked, name)))
        .map_or_else(
            || {
                "the context pinned is not the one this request carries, and the context is \
                 what the answer is learnt from"
                    .to_string()
            },
            |(name, described)| {
                format!(
                    "{described} pinned is {} and this asks {}",
                    field(pinned, name).repr(),
                    field(asked, name).repr()
                )
            },
        );
    Err(invalid(format!(
        "this session pinned a different question: {said}. Only the rows travel on a scoring \
         call, so the pinned question is the one that would be answered. Fit again to ask \
         something else"
    )))
}

/// `_warn_scoring_what_was_pinned(request, pinned, over_a_window, dtypes)`.
fn warn_scoring_what_was_pinned(
    request: &Request,
    fitted: &Fitted,
    tables: &Tables,
    host: &Host,
    said: &mut Warnings,
) -> R<()> {
    if fitted.pinned_keys.is_empty() {
        return Ok(());
    }
    let scored: Vec<Named> = match request.payloads() {
        Some(payloads) => {
            let mut scored = Vec::new();
            for payload in payloads {
                scored.extend(crate::keys::rows_in(
                    &payload,
                    "predict",
                    fitted.pinned_over_a_window,
                    request.protocols.as_ref(),
                )?);
            }
            scored
        }
        None => {
            let schema = request.schema.build(tables, host)?;
            let dtypes = crate::tabular_adapter::declared_by(&fitted.pinned);
            scored_from_frames(request, &schema, dtypes)
        }
    };
    let shown = scored
        .iter()
        .filter(|named| fitted.pinned_keys.contains(named))
        .count();
    if shown == 0 {
        return Ok(());
    }
    said.sdm(format!(
        "{shown} of the {} rows being scored are in this session's pinned context, with their \
         target values, so the model is being asked about rows it has already been shown the \
         answers for. The context was chosen when fit() ran and cannot be changed now. Either \
         clear the target on every row this session may score before calling fit(), or use \
         predict(), which excludes the rows it scores from the context it sends.",
        scored.len()
    ));
    Ok(())
}

/// `scored_rows(request, dtypes=...)` of a request built from frames.
fn scored_from_frames(
    request: &Request,
    schema: &sdm_schema::client::Schema,
    dtypes: Option<Vec<(String, String)>>,
) -> Vec<Named> {
    let (Some(PyVal::Str(entity)), Some(halves)) = (&request.entity_table, &request.halves) else {
        return Vec::new();
    };
    let Ok(table) = crate::request::schema_table(schema, entity) else {
        return Vec::new();
    };
    let names = sdm_schema::client::key_names_of(&table.primary_key);
    if names.is_empty() || names.iter().any(|n| !halves.predict.has(n)) {
        return Vec::new();
    }
    let dtypes = dtypes.or_else(|| crate::keys::key_dtypes(table, &names, Some(&halves.predict)));
    crate::keys::keys_from_a_frame(&halves.predict, &names, dtypes.as_deref())
}

/// `FittedModel.predict(request)`.
pub async fn session_predict(
    asker: &Asker,
    fitted: &Fitted,
    request: &Request,
    tables: &Tables,
    host: &Host,
    said: &mut Warnings,
) -> R<Result_> {
    if fitted.closed {
        return Err(structured(
            "INVALID_CONFIGURATION",
            "this fitted model is closed; fit again to score more rows",
        ));
    }
    if let Some(parts) = &request.parts {
        return Err(structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "this request is asked as {} questions and a session pins one context, so \
                 scoring it here would answer the pinned question that many times. Ask it with \
                 predict()",
                parts.len()
            ),
        ));
    }
    if let Some(pinned) = &fitted.question {
        refuse_a_question_the_session_did_not_pin(pinned, &question_of(request)?)?;
    }
    let spec = crate::models::spec_for(&PyVal::Str(fitted.model.clone()))?;
    if request.nothing_to_score()? {
        let schema = request.schema.build(tables, host)?;
        return Ok(empty_answer(
            spec.model_id,
            &fitted.outputs,
            Some((request, &schema)),
        ));
    }
    warn_scoring_what_was_pinned(request, fitted, tables, host, said)?;
    let plan = match spec.profile {
        Profile::Relational => {
            crate::relational_adapter::session_prediction_plan(request, &fitted.outputs, said)?
        }
        Profile::Tabular => {
            let mut schema = request.schema.build(tables, host)?;
            let declared = crate::tabular_adapter::declared_by(&fitted.pinned);
            let envelope = crate::tabular_adapter::envelope(
                request,
                &mut schema,
                &fitted.outputs,
                declared.as_deref(),
                said,
            )
            .map_err(|error| {
                crate::tabular_adapter::a_row_the_pinned_schema_cannot_hold(
                    request,
                    tables,
                    declared.as_deref(),
                    error,
                )
            })?;
            vec![crate::tabular_adapter::scoring_half(&envelope)]
        }
    };
    let mut parts = Vec::new();
    for payload in &plan {
        let response = call(
            asker,
            Endpoint::SessionPredict,
            Some(fitted.session_id.clone()),
            Some(payload),
        )
        .await?;
        let mut merged = fitted.pinned.clone();
        if let (Some(into), Some(from)) = (merged.as_object_mut(), payload.as_object()) {
            for (k, v) in from {
                into.insert(k.clone(), v.clone());
            }
        }
        parts.push(normalize(
            spec,
            &response,
            &fitted.outputs,
            &merged,
            request.protocols.as_ref(),
        )?);
    }
    if parts.is_empty() {
        return Err(invalid(
            "the adapter produced no scoring call for this request",
        ));
    }
    let merged = Result_::merge(parts)?;
    report_absent(
        &merged,
        fitted.named_outputs.as_deref(),
        spec.model_id,
        said,
    );
    Ok(merged)
}

/// `FittedModel.close()`: the session released.
pub async fn close(asker: &Asker, fitted: &Fitted) -> R<()> {
    call(
        asker,
        Endpoint::SessionDelete,
        Some(fitted.session_id.clone()),
        None,
    )
    .await
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_equality_and_first_difference_use_the_same_python_semantics() {
        let pinned = Question {
            query: None,
            kind: PyVal::Bool(true),
            target: PyVal::decode(&json!({"nested": [false]})).unwrap(),
            dtype: PyVal::Int(1),
            entity_table: Some(PyVal::Str("users".into())),
            context: None,
        };
        let mut asked = Question {
            kind: PyVal::Int(1),
            target: PyVal::decode(&json!({"nested": [0]})).unwrap(),
            dtype: PyVal::Float(1.0),
            ..pinned.clone()
        };
        assert_eq!(pinned, asked);
        assert!(refuse_a_question_the_session_did_not_pin(&pinned, &asked).is_ok());
        asked.context = Some("changed".into());
        assert_ne!(pinned, asked);
        let refusal = refuse_a_question_the_session_did_not_pin(&pinned, &asked).unwrap_err();
        assert!(
            refusal
                .message
                .contains("the context pinned is not the one")
        );
        asked.dtype = PyVal::Int(2);
        let refusal = refuse_a_question_the_session_did_not_pin(&pinned, &asked).unwrap_err();
        assert!(
            refusal
                .message
                .contains("the target column's dtype pinned is 1 and this asks 2")
        );
        asked.target = PyVal::decode(&json!({"nested": [1]})).unwrap();
        let refusal = refuse_a_question_the_session_did_not_pin(&pinned, &asked).unwrap_err();
        assert!(refusal.message.contains("the target column pinned is"));
        asked.kind = PyVal::Bool(false);
        let refusal = refuse_a_question_the_session_did_not_pin(&pinned, &asked).unwrap_err();
        assert!(
            refusal
                .message
                .contains("the task kind pinned is True and this asks False")
        );
    }
}
