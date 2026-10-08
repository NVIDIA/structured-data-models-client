// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The run loop every language drives the core through, and the problems a
//! run reports.
//!
//! The core never talks to a database or to the model. A [`Run`] says which
//! queries it needs with [`Step::RunQueries`], or which call to the model
//! with [`Step::CallModel`]; the host makes them however its language does
//! and hands each [`Outcome`] back with [`Run::submit`], and the run carries
//! on until it is [`Step::Done`]. What goes wrong is a [`Diagnostic`], which
//! each language raises as its own exception, and what a run warns of on the
//! way is a [`Diagnostic`] too, which [`Run::take_warnings`] hands over in
//! the order the run gave them.
//!
//! The protocol, which every run keeps:
//! - [`Run::next`] asks for a batch of queries, all of whose outcomes it needs
//!   before it can go on, or for one call to the model; calling it again
//!   before submitting every outcome it asked for is an error.
//! - [`Run::submit`] takes the outcome of a query or call the run asked for
//!   and has no outcome for yet; any other id is an error.
//! - Once [`Run::next`] has returned [`Step::Done`] or an error, the run is
//!   finished, and every later call is an error.
//!
//! These types live in a crate of their own so that the crates the core
//! drives (schema, sampling, request) can use them without depending on the
//! core that drives them.

pub mod asking;

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};
use serde_json::{Map, Value};

pub use asking::{Ask, Asker, AsyncRun};

/// Identifies a query or a model call within one run, so its outcome can be
/// handed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryId(pub u64);

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A value a query's placeholder takes.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// SQL `NULL`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A 64-bit integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// Text.
    Text(String),
}

/// A query a run needs the result of.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// What [`Run::submit`] takes back with the query's outcome.
    pub id: QueryId,
    /// The statement, in the dialect of the source the run was made for.
    pub sql: String,
    /// The values of the statement's placeholders, in order.
    pub params: Vec<Param>,
    /// The columns the run reads from the result, by name, in order.
    pub columns: Vec<String>,
}

/// Which of the model's endpoints a call goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endpoint {
    /// `POST /v1/predictions`.
    Predict,
    /// `POST /v1/sessions`.
    SessionCreate,
    /// `POST /v1/sessions/{id}/predictions`.
    SessionPredict,
    /// `DELETE /v1/sessions/{id}`.
    SessionDelete,
}

impl Endpoint {
    /// The name hosts know the endpoint by.
    pub fn name(self) -> &'static str {
        match self {
            Endpoint::Predict => "predict",
            Endpoint::SessionCreate => "session_create",
            Endpoint::SessionPredict => "session_predict",
            Endpoint::SessionDelete => "session_delete",
        }
    }

    /// The endpoint named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "predict" => Endpoint::Predict,
            "session_create" => Endpoint::SessionCreate,
            "session_predict" => Endpoint::SessionPredict,
            "session_delete" => Endpoint::SessionDelete,
            _ => return None,
        })
    }
}

/// One call a run needs the model to answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCall {
    /// What [`Run::submit`] takes back with the call's outcome.
    pub id: QueryId,
    /// Where the call goes.
    pub endpoint: Endpoint,
    /// The session, for [`Endpoint::SessionPredict`] and
    /// [`Endpoint::SessionDelete`].
    pub session_id: Option<String>,
    /// The request body as JSON text; `None` for
    /// [`Endpoint::SessionDelete`].
    pub body: Option<String>,
}

/// What came of running a [`Query`] or making a [`ModelCall`].
#[derive(Debug, Clone)]
pub enum Outcome {
    /// The query's result.
    Data(QueryResult),
    /// The query failed in the database or its driver.
    Error {
        /// The SQLSTATE the driver reported, if any.
        sqlstate: Option<String>,
        /// The database's own error code, if any.
        vendor_code: Option<i64>,
        /// The error's text.
        message: String,
    },
    /// The model's answer to a call, as JSON text.
    Response(String),
    /// The call failed: a JSON object of the error the transport raised,
    /// with its `class`, `status`, `code`, `message` and `details`.
    ModelError(String),
}

/// The rows a query returned, as Arrow record batches of one schema.
#[derive(Debug, Clone)]
pub struct QueryResult {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl QueryResult {
    /// A result of `batches`, every one of which must have `schema`.
    pub fn try_new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Result<Self, ArrowError> {
        if let Some(batch) = batches.iter().find(|b| b.schema() != schema) {
            return Err(ArrowError::SchemaError(format!(
                "a batch has the schema {}, not the result's {}",
                batch.schema(),
                schema
            )));
        }
        Ok(Self { schema, batches })
    }

    /// A result of one batch.
    pub fn from_batch(batch: RecordBatch) -> Self {
        Self {
            schema: batch.schema(),
            batches: vec![batch],
        }
    }

    /// The result's schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The result's batches, in order.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The number of rows across every batch.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// What a run does next.
#[derive(Debug, Clone, PartialEq)]
pub enum Step<T> {
    /// Run these queries and submit each one's outcome, then call
    /// [`Run::next`] again.
    RunQueries(Vec<Query>),
    /// Make this call to the model and submit its outcome, then call
    /// [`Run::next`] again.
    CallModel(ModelCall),
    /// The run is finished, with this result.
    Done(T),
}

/// A problem a run reports, which each language raises as its own exception
/// or gives as its own warning.
///
/// A refusal names the `class` it is raised as, a warning the `category` it
/// is given as; the run loop's own problems name neither, and each language
/// raises those as its own core error.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Diagnostic {
    /// The error class a refusal is raised as, module-qualified, such as
    /// `sdm_client.errors.StructuredDataError`.
    pub class: Option<String>,
    /// The warning class a warning is given as, module-qualified, such as
    /// `sdm_client.core.warnings.SDMWarning`.
    pub category: Option<String>,
    /// What kind of problem it is, stable across releases and languages;
    /// see [`codes`]. Not every refusal has one.
    pub code: Option<String>,
    /// What happened, for a person to read.
    pub message: String,
    /// What callers branch on, by name. Boxed, so a diagnostic stays small
    /// enough to return by value.
    pub details: Box<Map<String, Value>>,
    /// The HTTP status of a model call that failed.
    pub status: Option<i32>,
    /// The errors beneath this one, outermost first, as today's code chains
    /// them (`raise ... from error`): each `{"class", "message"}` (with
    /// `code`, `details` and `status` for this package's own errors), or
    /// `{"from": "query" | "model"}`, the error the host met running the
    /// last statement or model call itself. Boxed, as `details` is.
    #[allow(clippy::box_collection)]
    pub chain: Box<Vec<Value>>,
}

impl Diagnostic {
    /// A problem of the run loop itself, with `code` and `message`.
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: Some(code.to_owned()),
            message: message.into(),
            ..Self::default()
        }
    }

    /// A refusal raised as `class`, with `code` (if it has one) and
    /// `message`.
    pub fn refusal(class: &str, code: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            class: Some(class.to_owned()),
            code: code.map(str::to_owned),
            message: message.into(),
            ..Self::default()
        }
    }

    /// A warning given as `category`.
    pub fn warning(category: &str, message: impl Into<String>) -> Self {
        Self {
            category: Some(category.to_owned()),
            message: message.into(),
            ..Self::default()
        }
    }

    /// The same diagnostic with `details`.
    #[must_use]
    pub fn with_details(mut self, details: Map<String, Value>) -> Self {
        self.details = Box::new(details);
        self
    }

    /// The same diagnostic with one more detail.
    #[must_use]
    pub fn with_detail(mut self, key: &str, value: Value) -> Self {
        self.details.insert(key.to_owned(), value);
        self
    }

    /// The same diagnostic, raised from `link` and then from what `link` was
    /// raised from (`links`, outermost first).
    #[must_use]
    pub fn caused_by(mut self, links: impl IntoIterator<Item = Value>) -> Self {
        self.chain.extend(links);
        self
    }

    /// This diagnostic as a link of another's chain, and its own chain.
    pub fn as_links(&self) -> Vec<Value> {
        let mut link = serde_json::Map::new();
        link.insert(
            "class".into(),
            Value::String(self.class.clone().unwrap_or_default()),
        );
        link.insert("message".into(), Value::String(self.message.clone()));
        if let Some(code) = &self.code {
            link.insert("code".into(), Value::String(code.clone()));
        }
        if !self.details.is_empty() {
            link.insert("details".into(), Value::Object((*self.details).clone()));
        }
        if let Some(status) = self.status {
            link.insert("status".into(), Value::from(status));
        }
        let mut out = vec![Value::Object(link)];
        out.extend(self.chain.iter().cloned());
        out
    }

    /// Whether this is a warning rather than a refusal.
    pub fn is_warning(&self) -> bool {
        self.category.is_some()
    }

    /// The code, or `""` where there is none.
    pub fn code_str(&self) -> &str {
        self.code.as_deref().unwrap_or("")
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(f, "{code}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for Diagnostic {}

/// The codes of the run loop's own diagnostics.
pub mod codes {
    /// A query the run asked for failed; the message carries the database's.
    pub const QUERY_FAILED: &str = "QUERY_FAILED";
    /// A query's result is not what the run asked for, so it cannot be read.
    pub const UNREADABLE_RESULT: &str = "UNREADABLE_RESULT";
    /// An outcome was submitted for a query the run is not waiting on.
    pub const UNKNOWN_QUERY: &str = "UNKNOWN_QUERY";
    /// The run was asked to go on before every outcome it needs was submitted.
    pub const OUTCOME_MISSING: &str = "OUTCOME_MISSING";
    /// The run was used after it finished.
    pub const RUN_FINISHED: &str = "RUN_FINISHED";
    /// An outcome of the wrong kind was submitted: a model's answer for a
    /// query, or a query's for a model call.
    pub const WRONG_OUTCOME: &str = "WRONG_OUTCOME";
}

/// A computation that needs query results or model answers it cannot get
/// itself.
///
/// The module documentation describes the protocol every run keeps.
pub trait Run<T> {
    /// What the run needs next: queries to run, a call to make, or its
    /// result.
    fn next(&mut self) -> Result<Step<T>, Diagnostic>;

    /// The outcome of the query or call `id`, one the run asked for.
    fn submit(&mut self, id: QueryId, outcome: Outcome) -> Result<(), Diagnostic>;

    /// The warnings the run gave since this was last called, in order. A
    /// host takes them after each [`Run::next`], whether it gave a step or
    /// an error.
    fn take_warnings(&mut self) -> Vec<Diagnostic> {
        Vec::new()
    }
}

impl<T, R: Run<T> + ?Sized> Run<T> for Box<R> {
    fn next(&mut self) -> Result<Step<T>, Diagnostic> {
        (**self).next()
    }

    fn submit(&mut self, id: QueryId, outcome: Outcome) -> Result<(), Diagnostic> {
        (**self).submit(id, outcome)
    }

    fn take_warnings(&mut self) -> Vec<Diagnostic> {
        (**self).take_warnings()
    }
}
