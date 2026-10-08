// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The sampling driver: `Sampler.sample_target` and
//! `Sampler.sample_subgraph` (`rfm/base/sampler.py`), over the in-memory
//! tables or a database, as runs of the core.
//!
//! A run starts from what the host describes of the call
//! (`scripts/parity/capture_sampling.py`, `describe`): the graph, the
//! sampler's state, the arguments, and, in memory, the tables themselves.
//! Over a database it asks the host for each statement Phase 6's dialects
//! write, in the order today's Python runs them, and reads the results as
//! pandas reads them. What today's method raises, the run ends with: its
//! Python class and text, and the warnings the method gave before it raised
//! ([`Raised`]). The run's own diagnostics are left to the run loop.
//!
//! - [`inputs`] reads the description;
//! - [`query`] is the sampler's view of the validated query;
//! - [`target`] is `sample_target`, [`subgraph`] `sample_subgraph`'s
//!   shaping;
//! - [`sql`] is `SQLSampler`'s traversal and the five databases' statements,
//!   [`local`] the in-memory backend.

pub mod init;
pub mod inputs;
pub mod lagged;
pub mod local;
pub mod output;
pub mod query;
pub mod sampler;
pub mod sql;
pub mod subgraph;
pub mod target;
pub mod task;

use std::fmt;
use std::future::Future;

use arrow_array::RecordBatch;
use sdm_protocol::{Asker, AsyncRun, Outcome, Param};
use sdm_schema::dt::Host;

pub use inputs::Inputs;
pub use output::{SubgraphDone, TargetDone};

use output::Warning;

/// What a step of the driver raised: the Python exception's class and
/// text, and the warnings today's method gave before it raised, in order.
/// A `KeyError`'s text is its argument, which Python shows as its `repr`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raised {
    pub class: String,
    pub message: String,
    pub warnings: Vec<Warning>,
    /// What it was raised from, outermost first: each `{"class",
    /// "message"}`, the class module-qualified.
    pub chain: Vec<serde_json::Value>,
}

impl Raised {
    pub fn new(class: &str, message: impl Into<String>) -> Self {
        Self {
            class: class.to_string(),
            message: message.into(),
            warnings: Vec::new(),
            chain: Vec::new(),
        }
    }

    /// The same error, raised from `class` with `message`.
    #[must_use]
    pub fn caused_by(mut self, class: &str, message: impl Into<String>) -> Self {
        self.chain
            .push(serde_json::json!({"class": class, "message": message.into()}));
        self
    }

    /// The same error, raised after `earlier` were given: they come before
    /// the warnings it already carries, which were given later.
    #[must_use]
    pub fn after(mut self, mut earlier: Vec<Warning>) -> Self {
        earlier.append(&mut self.warnings);
        self.warnings = earlier;
        self
    }

    /// `KeyError(key)`.
    pub fn key_error(key: &str) -> Self {
        Self::new("KeyError", key)
    }

    /// A state the core can't reach unless it is wrong, or input it was
    /// never meant to take.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            "RuntimeError",
            format!("internal error in the core: {}", message.into()),
        )
    }

    pub fn value_error(message: impl Into<String>) -> Self {
        Self::new("ValueError", message)
    }
}

impl fmt::Display for Raised {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl std::error::Error for Raised {}

impl From<sdm_labels::LabelError> for Raised {
    fn from(error: sdm_labels::LabelError) -> Self {
        match error.key {
            Some(key) => Raised::key_error(&key),
            None => Raised::new(&error.code, error.message),
        }
    }
}

impl From<sdm_labels::date_offset::OffsetError> for Raised {
    fn from(error: sdm_labels::date_offset::OffsetError) -> Self {
        Raised::new(error.code, error.message)
    }
}

impl From<sdm_dialects::Error> for Raised {
    fn from(error: sdm_dialects::Error) -> Self {
        Raised::new("ValueError", error.to_string())
    }
}

/// The database, through the host: each statement asked as a query of the
/// run, its result read back, a failure raised as the database's error.
#[derive(Clone)]
pub struct Db {
    asker: Asker,
}

/// What a statement returned: its columns, as Arrow gives them.
#[derive(Clone, Debug)]
pub struct Rows {
    pub schema: arrow_schema::SchemaRef,
    pub batches: Vec<RecordBatch>,
}

impl Rows {
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// The class a failed statement is raised as. The host submits a failure's
/// SQLSTATE, code and text but not its class, so the class is the driver's
/// own name for it.
pub const QUERY_FAILED: &str = "QueryFailed";

impl Db {
    pub fn new(asker: Asker) -> Self {
        Self { asker }
    }

    /// `self._run(statement)`.
    pub async fn run(&self, statement: (String, Option<Vec<Param>>)) -> Result<Rows, Raised> {
        let (sql, params) = statement;
        match self
            .asker
            .ask(sql, params.unwrap_or_default(), Vec::new())
            .await
        {
            Outcome::Data(result) => Ok(Rows {
                schema: result.schema(),
                batches: result.batches().to_vec(),
            }),
            Outcome::Error { message, .. } => Err(Raised::new(QUERY_FAILED, message)),
            Outcome::Response(_) | Outcome::ModelError(_) => Err(Raised::internal(
                "a model's answer was submitted for a query",
            )),
        }
    }
}

/// How a run of the driver ends: what today's method returns, or what it
/// raised, with the warnings it gave first.
pub type Ended<T> = Result<T, Raised>;

/// The run of `body`, given the database the run asks through and the
/// description `inputs_json` and the tables `tables` start it from.
fn driver_run<T, F, Fut>(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    body: F,
) -> AsyncRun<Ended<T>>
where
    T: 'static,
    F: FnOnce(Inputs, Db) -> Fut + Send + 'static,
    Fut: Future<Output = Ended<T>> + Send + 'static,
{
    let inputs = inputs::Inputs::parse(inputs_json, tables);
    AsyncRun::new(move |asker| async move {
        Ok(match inputs {
            Ok(inputs) => body(inputs, Db::new(asker)).await,
            Err(raised) => Err(raised),
        })
    })
}

/// The run of `sample_target` the description `inputs_json` and the tables
/// `tables` start.
pub fn sample_target_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<TargetDone>> {
    driver_run(inputs_json, tables, move |inputs, db| {
        target::sample_target(inputs, db, host)
    })
}

/// The run of `sample_lagged_target` the description `inputs_json` and the
/// tables `tables` start.
pub fn sample_lagged_target_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<lagged::LaggedDone>> {
    driver_run(inputs_json, tables, move |inputs, db| {
        lagged::sample_lagged_target(inputs, db, host)
    })
}

/// The run of `KumoRelational._get_task_table` the description
/// `inputs_json` and the tables `tables` start.
pub fn task_table_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<task::TaskDone>> {
    driver_run(inputs_json, tables, move |inputs, db| {
        task::get_task_table(inputs, db, host)
    })
}

/// The run of `KumoRelational._get_context` the description `inputs_json`
/// and the tables `tables` start.
pub fn context_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<task::ContextDone>> {
    driver_run(inputs_json, tables, move |inputs, db| {
        task::get_context(inputs, db, host)
    })
}

/// The run of the checks building a sampler makes (`init`): the keys no
/// index covers and the warning naming them, unless `optimize`.
pub fn missing_indices_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<(init::MissingIndices, Option<output::Warning>)>> {
    driver_run(inputs_json, tables, move |inputs, db| async move {
        let optimize = inputs
            .arguments
            .get("optimize")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let sampler = sampler::Sampler::new(&inputs, host)?;
        init::missing_indices(&sampler, &inputs, &db, optimize)
            .await
            .map_err(|raised| raised.after(sampler.take_warnings()))
    })
}

/// The run of `sample_subgraph` the description `inputs_json` and the
/// tables `tables` start.
pub fn sample_subgraph_run(
    inputs_json: &str,
    tables: Vec<(String, RecordBatch)>,
    host: Host,
) -> AsyncRun<Ended<SubgraphDone>> {
    driver_run(inputs_json, tables, move |inputs, db| {
        subgraph::sample_subgraph(inputs, db, host)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warnings_given_earlier_come_first() {
        let raised = Raised::new("ValueError", "no")
            .after(vec![Warning::user("second")])
            .after(vec![Warning::user("first")]);
        let messages: Vec<&str> = raised.warnings.iter().map(|w| w.message.as_str()).collect();
        assert_eq!(messages, ["first", "second"]);
        assert_eq!(raised.to_string(), "ValueError: no");
    }
}
