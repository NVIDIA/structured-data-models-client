// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A host that drives a [`Run`] from a recording instead of a database.
//!
//! The recording lists, in order, every query the run is expected to ask and
//! the outcome to give it. Replaying fails as soon as the run asks anything
//! else, so a replay proves both what the run computes from those outcomes
//! and exactly which statements it sends.

use std::collections::VecDeque;
use std::fmt;

use crate::{Diagnostic, Outcome, Param, Query, Run, Step};

/// One query the run is expected to ask, and the outcome it is given.
#[derive(Debug, Clone)]
pub struct Exchange {
    /// The statement the run must ask, byte for byte.
    pub sql: String,
    /// The values its placeholders must take.
    pub params: Vec<Param>,
    /// What the host submits for it.
    pub outcome: Outcome,
}

impl Exchange {
    /// The exchange of `sql`, without parameters, for `outcome`.
    pub fn new(sql: impl Into<String>, outcome: Outcome) -> Self {
        Self {
            sql: sql.into(),
            params: Vec::new(),
            outcome,
        }
    }
}

/// What a replay that went as recorded produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Replayed<T> {
    /// The run's result.
    pub value: T,
    /// Every query the run asked, in order.
    pub queries: Vec<Query>,
}

/// Why a replay did not go as recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    /// The run reported a problem.
    Run(Diagnostic),
    /// The run asked a query that is not the next one recorded.
    Unexpected {
        /// What the run asked.
        query: Box<Query>,
        /// The statement and parameters recorded next, if any.
        expected: Option<(String, Vec<Param>)>,
    },
    /// The run finished with these many recorded exchanges never asked.
    Unused(usize),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Run(diagnostic) => write!(f, "the run failed: {diagnostic}"),
            Self::Unexpected {
                query,
                expected: Some((sql, params)),
            } => write!(
                f,
                "the run asked {:?} with {:?}, and the recording has {sql:?} with {params:?}",
                query.sql, query.params
            ),
            Self::Unexpected {
                query,
                expected: None,
            } => write!(
                f,
                "the run asked {:?} after every recorded exchange",
                query.sql
            ),
            Self::Unused(n) => write!(f, "the run finished with {n} recorded exchanges unasked"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<Diagnostic> for ReplayError {
    fn from(diagnostic: Diagnostic) -> Self {
        Self::Run(diagnostic)
    }
}

/// Drive `run` to its end, answering each query from `exchanges`, in order.
pub fn replay<T, R>(
    run: &mut R,
    exchanges: impl IntoIterator<Item = Exchange>,
) -> Result<Replayed<T>, ReplayError>
where
    R: Run<T> + ?Sized,
{
    let mut exchanges: VecDeque<Exchange> = exchanges.into_iter().collect();
    let mut asked = Vec::new();
    loop {
        match run.next()? {
            Step::Done(value) => {
                if !exchanges.is_empty() {
                    return Err(ReplayError::Unused(exchanges.len()));
                }
                return Ok(Replayed {
                    value,
                    queries: asked,
                });
            }
            Step::CallModel(call) => {
                return Err(ReplayError::Run(Diagnostic::new(
                    crate::codes::WRONG_OUTCOME,
                    format!(
                        "the run asked to call the model's {} endpoint, and a replay of \
                         queries answers none",
                        call.endpoint.name()
                    ),
                )));
            }
            Step::RunQueries(queries) => {
                for query in queries {
                    let Some(next) = exchanges.pop_front() else {
                        return Err(ReplayError::Unexpected {
                            query: Box::new(query),
                            expected: None,
                        });
                    };
                    if next.sql != query.sql || next.params != query.params {
                        return Err(ReplayError::Unexpected {
                            query: Box::new(query),
                            expected: Some((next.sql, next.params)),
                        });
                    }
                    run.submit(query.id, next.outcome)?;
                    asked.push(query);
                }
            }
        }
    }
}
