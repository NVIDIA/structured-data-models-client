// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A [`Run`] written as plain `async` code.
//!
//! Logic that needs query results is an `async` body that awaits
//! [`Asker::ask`] for each one, and [`Asker::call_model`] for each answer of
//! the model it needs. [`AsyncRun`] drives the body: each [`Run::next`]
//! resumes it where it stopped, and the queries it stops on are the step's
//! queries, or the call it stops on is the step's call. So every part of the
//! body runs once, however many queries it asks, and queries the body awaits
//! together (with a join) are asked in one step. What the body warns of with
//! [`Asker::warn`] is handed over by [`Run::take_warnings`].
//!
//! No executor or thread is involved: the body only ever waits on its own
//! queries, so polling it once per step is enough.
//!
//! A body should await every [`Ask`] it creates to the end. One polled and
//! then dropped (say, the loser of a `select`) leaves its outcome stored
//! until the run is dropped.

use std::collections::HashMap;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use crate::{Diagnostic, Endpoint, ModelCall, Outcome, Param, Query, QueryId, Run, Step, codes};

/// What the body and the run share: the queries in flight and their
/// outcomes.
#[derive(Default)]
struct Shared {
    next_id: u64,
    /// Asked by the body since the last step, not yet handed to the host.
    asked: Vec<Query>,
    /// Model calls asked by the body since the last step.
    calls: Vec<ModelCall>,
    /// Warnings given by the body, not yet taken by the host.
    warnings: Vec<Diagnostic>,
    /// Handed to the host, with no outcome submitted yet.
    waiting: Vec<QueryId>,
    /// Submitted, and not yet taken by the body.
    outcomes: HashMap<QueryId, Outcome>,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    // A panic while holding the lock (`Ask::poll` panics on an outcome
    // awaited twice) panics the body, and that finishes the run, so a
    // poisoned lock is never read in a half-changed state.
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What an [`AsyncRun`]'s body asks the host through.
#[derive(Clone)]
pub struct Asker {
    shared: Arc<Mutex<Shared>>,
}

impl Asker {
    /// The outcome of running `sql` with `params`; `columns` are the columns
    /// the body reads from the result, as [`Query::columns`] says.
    pub fn ask(&self, sql: String, params: Vec<Param>, columns: Vec<String>) -> Ask {
        Ask {
            shared: Arc::clone(&self.shared),
            state: AskState::New(Asked::Query {
                sql,
                params,
                columns,
            }),
        }
    }

    /// The model's answer to a call to `endpoint` with `body` (JSON text),
    /// for `session_id` where the endpoint takes one.
    pub fn call_model(
        &self,
        endpoint: Endpoint,
        session_id: Option<String>,
        body: Option<String>,
    ) -> Ask {
        Ask {
            shared: Arc::clone(&self.shared),
            state: AskState::New(Asked::Call {
                endpoint,
                session_id,
                body,
            }),
        }
    }

    /// Give `warning`, after every warning given before it.
    pub fn warn(&self, warning: Diagnostic) {
        lock(&self.shared).warnings.push(warning);
    }
}

enum Asked {
    Query {
        sql: String,
        params: Vec<Param>,
        columns: Vec<String>,
    },
    Call {
        endpoint: Endpoint,
        session_id: Option<String>,
        body: Option<String>,
    },
}

/// The outcome of one query, once the host has submitted it.
#[must_use = "a query is only asked when it is awaited"]
pub struct Ask {
    shared: Arc<Mutex<Shared>>,
    state: AskState,
}

enum AskState {
    New(Asked),
    Asked(QueryId),
    Answered,
}

impl Future for Ask {
    type Output = Outcome;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Outcome> {
        let this = self.get_mut();
        let mut shared = lock(&this.shared);
        match mem::replace(&mut this.state, AskState::Answered) {
            AskState::New(asked) => {
                shared.next_id += 1;
                let id = QueryId(shared.next_id);
                match asked {
                    Asked::Query {
                        sql,
                        params,
                        columns,
                    } => shared.asked.push(Query {
                        id,
                        sql,
                        params,
                        columns,
                    }),
                    Asked::Call {
                        endpoint,
                        session_id,
                        body,
                    } => shared.calls.push(ModelCall {
                        id,
                        endpoint,
                        session_id,
                        body,
                    }),
                }
                this.state = AskState::Asked(id);
                Poll::Pending
            }
            AskState::Asked(id) => match shared.outcomes.remove(&id) {
                Some(outcome) => Poll::Ready(outcome),
                None => {
                    this.state = AskState::Asked(id);
                    Poll::Pending
                }
            },
            AskState::Answered => panic!("a query's outcome was awaited after it was taken"),
        }
    }
}

type Body<T> = Pin<Box<dyn Future<Output = Result<T, Diagnostic>> + Send>>;

/// A run whose logic is an `async` body; see the module documentation.
pub struct AsyncRun<T> {
    /// `None` once the run has finished, or panicked.
    body: Option<Body<T>>,
    shared: Arc<Mutex<Shared>>,
}

impl<T> AsyncRun<T> {
    /// The run of the body `make` returns, given the [`Asker`] it asks
    /// through. Nothing of the body runs before the first [`Run::next`].
    pub fn new<F>(make: impl FnOnce(Asker) -> F) -> Self
    where
        F: Future<Output = Result<T, Diagnostic>> + Send + 'static,
    {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let body = make(Asker {
            shared: Arc::clone(&shared),
        });
        Self {
            body: Some(Box::pin(body)),
            shared,
        }
    }
}

fn finished() -> Diagnostic {
    Diagnostic::new(codes::RUN_FINISHED, "the run has finished")
}

impl<T> Run<T> for AsyncRun<T> {
    fn next(&mut self) -> Result<Step<T>, Diagnostic> {
        let Some(mut body) = self.body.take() else {
            return Err(finished());
        };
        let waiting = lock(&self.shared).waiting.len();
        if waiting > 0 {
            self.body = Some(body);
            return Err(Diagnostic::new(
                codes::OUTCOME_MISSING,
                format!("{waiting} asked queries have no outcome yet"),
            ));
        }
        // The body is out of `self` while it runs, so a panic in it leaves
        // the run finished rather than resumable from a broken state.
        match body.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(result) => result.map(Step::Done),
            Poll::Pending => {
                let mut shared = lock(&self.shared);
                let asked = mem::take(&mut shared.asked);
                let mut calls = mem::take(&mut shared.calls);
                assert!(
                    calls.is_empty() || asked.is_empty(),
                    "an async run's body waited on a model call and queries at once"
                );
                assert!(
                    calls.len() <= 1,
                    "an async run's body waited on more than one model call at once"
                );
                if let Some(call) = calls.pop() {
                    shared.waiting = vec![call.id];
                    drop(shared);
                    self.body = Some(body);
                    return Ok(Step::CallModel(call));
                }
                assert!(
                    !asked.is_empty(),
                    "an async run's body waited on something other than its queries"
                );
                shared.waiting = asked.iter().map(|q| q.id).collect();
                drop(shared);
                self.body = Some(body);
                Ok(Step::RunQueries(asked))
            }
        }
    }

    fn submit(&mut self, id: QueryId, outcome: Outcome) -> Result<(), Diagnostic> {
        if self.body.is_none() {
            return Err(finished());
        }
        let mut shared = lock(&self.shared);
        let Some(i) = shared.waiting.iter().position(|q| *q == id) else {
            return Err(Diagnostic::new(
                codes::UNKNOWN_QUERY,
                format!("the run is not waiting on query {id}"),
            ));
        };
        shared.waiting.remove(i);
        shared.outcomes.insert(id, outcome);
        Ok(())
    }

    fn take_warnings(&mut self) -> Vec<Diagnostic> {
        mem::take(&mut lock(&self.shared).warnings)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn rows(outcome: &Outcome) -> usize {
        match outcome {
            Outcome::Data(result) => result.num_rows(),
            Outcome::Error { .. } | Outcome::Response(_) | Outcome::ModelError(_) => usize::MAX,
        }
    }

    fn empty() -> Outcome {
        let batch = arrow_array::RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty()));
        Outcome::Data(crate::QueryResult::from_batch(batch))
    }

    fn failed(message: &str) -> Outcome {
        Outcome::Error {
            sqlstate: None,
            vendor_code: None,
            message: message.to_owned(),
        }
    }

    #[test]
    fn each_part_of_the_body_runs_once_however_many_queries_it_asks() {
        let resumed = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&resumed);
        let mut run = AsyncRun::new(move |asker| async move {
            let mut seen = 0;
            for i in 0..500 {
                counter.fetch_add(1, Ordering::SeqCst);
                let outcome = asker
                    .ask(format!("SELECT {i}"), Vec::new(), Vec::new())
                    .await;
                seen += rows(&outcome);
            }
            Ok(seen)
        });
        let mut steps = 0;
        let value = loop {
            match run.next().unwrap() {
                Step::Done(value) => break value,
                Step::CallModel(_) => panic!("no model call was asked"),
                Step::RunQueries(queries) => {
                    steps += 1;
                    assert_eq!(queries.len(), 1);
                    assert_eq!(queries[0].sql, format!("SELECT {}", steps - 1));
                    run.submit(queries[0].id, empty()).unwrap();
                }
            }
        };
        assert_eq!((value, steps), (0, 500));
        assert_eq!(resumed.load(Ordering::SeqCst), 500);
    }

    #[test]
    fn queries_awaited_together_are_asked_in_one_step() {
        let mut run = AsyncRun::new(|asker| async move {
            let a = asker.ask("SELECT 'a'".into(), Vec::new(), Vec::new());
            let b = asker.ask("SELECT 'b'".into(), vec![Param::Int(1)], Vec::new());
            let mut both = (Box::pin(a), Box::pin(b));
            let (a, b) = std::future::poll_fn(|cx| {
                let a = both.0.as_mut().poll(cx);
                let b = both.1.as_mut().poll(cx);
                match (a, b) {
                    (Poll::Ready(a), Poll::Ready(b)) => Poll::Ready((a, b)),
                    _ => Poll::Pending,
                }
            })
            .await;
            Ok((rows(&a), rows(&b)))
        });
        let Step::RunQueries(queries) = run.next().unwrap() else {
            panic!("expected queries");
        };
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[1].params, [Param::Int(1)]);
        // Outcomes may come in any order.
        run.submit(queries[1].id, failed("no")).unwrap();
        run.submit(queries[0].id, empty()).unwrap();
        assert_eq!(run.next().unwrap(), Step::Done((0, usize::MAX)));
    }

    #[test]
    fn keeps_the_run_loops_protocol() {
        let mut run = AsyncRun::new(|asker| async move {
            asker.ask("SELECT 1".into(), Vec::new(), Vec::new()).await;
            Ok(())
        });
        let Step::RunQueries(queries) = run.next().unwrap() else {
            panic!("expected queries");
        };
        let id = queries[0].id;
        assert_eq!(run.next().unwrap_err().code_str(), codes::OUTCOME_MISSING);
        assert_eq!(
            run.submit(QueryId(99), empty()).unwrap_err().code_str(),
            codes::UNKNOWN_QUERY
        );
        run.submit(id, empty()).unwrap();
        assert_eq!(
            run.submit(id, empty()).unwrap_err().code_str(),
            codes::UNKNOWN_QUERY
        );
        assert_eq!(run.next().unwrap(), Step::Done(()));
        assert_eq!(run.next().unwrap_err().code_str(), codes::RUN_FINISHED);
        assert_eq!(
            run.submit(id, empty()).unwrap_err().code_str(),
            codes::RUN_FINISHED
        );
    }

    #[test]
    fn a_body_that_fails_or_panics_finishes_the_run() {
        let mut run: AsyncRun<()> =
            AsyncRun::new(|_| async { Err(Diagnostic::new(codes::QUERY_FAILED, "it failed")) });
        assert_eq!(run.next().unwrap_err().code_str(), codes::QUERY_FAILED);
        assert_eq!(run.next().unwrap_err().code_str(), codes::RUN_FINISHED);

        let mut run: AsyncRun<()> = AsyncRun::new(|asker| async move {
            asker.ask("SELECT 1".into(), Vec::new(), Vec::new()).await;
            panic!("a bug");
        });
        let Step::RunQueries(queries) = run.next().unwrap() else {
            panic!("expected queries");
        };
        run.submit(queries[0].id, empty()).unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.next()));
        assert!(panicked.is_err());
        assert_eq!(run.next().unwrap_err().code_str(), codes::RUN_FINISHED);
    }

    #[test]
    fn a_model_call_is_a_step_of_its_own_and_warnings_keep_their_order() {
        let mut run = AsyncRun::new(|asker| async move {
            asker.warn(Diagnostic::warning("builtins.UserWarning", "first"));
            asker.ask("SELECT 1".into(), Vec::new(), Vec::new()).await;
            asker.warn(Diagnostic::warning("builtins.UserWarning", "second"));
            let answer = asker
                .call_model(Endpoint::Predict, None, Some("{}".into()))
                .await;
            asker.warn(Diagnostic::warning("builtins.UserWarning", "third"));
            match answer {
                Outcome::Response(text) => Ok(text),
                _ => Err(Diagnostic::new(codes::WRONG_OUTCOME, "not an answer")),
            }
        });
        let Step::RunQueries(queries) = run.next().unwrap() else {
            panic!("expected queries");
        };
        let given: Vec<String> = run.take_warnings().into_iter().map(|w| w.message).collect();
        assert_eq!(given, ["first"]);
        run.submit(queries[0].id, empty()).unwrap();
        let Step::CallModel(call) = run.next().unwrap() else {
            panic!("expected a call");
        };
        assert_eq!(call.endpoint, Endpoint::Predict);
        assert_eq!(call.body.as_deref(), Some("{}"));
        let given: Vec<String> = run.take_warnings().into_iter().map(|w| w.message).collect();
        assert_eq!(given, ["second"]);
        assert_eq!(run.next().unwrap_err().code_str(), codes::OUTCOME_MISSING);
        run.submit(call.id, Outcome::Response("[1]".into()))
            .unwrap();
        assert_eq!(run.next().unwrap(), Step::Done("[1]".to_string()));
        let given: Vec<String> = run.take_warnings().into_iter().map(|w| w.message).collect();
        assert_eq!(given, ["third"]);
    }
}
