// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single Run state machine.
//!
//! An operation's body is plain `async` code over an [`Asker`]: it awaits
//! [`Asker::fetch`] for each data request and [`Asker::call_model`] for each
//! model call. [`Run::next`] polls the body once; the requests it stopped on
//! become the step, and requests awaited together (with a join) are asked in
//! one fetch step. No executor and no thread is involved: the body only ever
//! waits on its own requests, so one poll per step is enough.
//!
//! States are `Ready`, `AwaitingFetch`, `AwaitingModel`, `Finished`,
//! `Cancelled` and `Poisoned`. A call the state does not allow returns a
//! `protocol` error (`cancelled` once cancelled) and changes nothing; no call
//! panics because of the order the host makes them in. A run is poisoned
//! when its body panics: the state is set to `Poisoned` before every poll
//! and replaced only when the poll returns.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use sdm_contract::data::{AttachmentId, DataRequest, ExpectedSchema, ProviderError, SqlStatement};
use sdm_contract::ops::OperationResult;
use sdm_contract::run::{Endpoint, FetchStep, ModelCall, ReplyId};
use sdm_contract::{CoreError, CoreWarning, ErrorCode, ErrorKind, Step};

/// Data that travels beside a control message.
#[derive(Debug, Clone)]
pub enum Attachment {
    Bytes(Vec<u8>),
    Arrow(ArrowData),
}

/// The attachments one message carries, by the ids it names them with.
pub type Attachments = BTreeMap<AttachmentId, Attachment>;

/// Arrow record batches of one schema.
#[derive(Debug, Clone)]
pub struct ArrowData {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// The host's answer to one fetch.
#[derive(Debug, Clone)]
pub enum FetchResult {
    Data(ArrowData),
    Error(ProviderError),
}

/// The host's answer to one model call.
#[derive(Debug, Clone)]
pub enum ModelResult {
    Response {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    /// The transport failed before an answer; its message stays in the
    /// host.
    TransportError { kind: ErrorKind, retryable: bool },
}

/// What an awaited request resolves to.
#[derive(Debug, Clone)]
pub enum Reply {
    Fetch(FetchResult),
    Model(ModelResult),
}

/// Where a run is; see the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunState {
    Ready,
    /// The fetches still unanswered, in id order.
    AwaitingFetch(Vec<ReplyId>),
    AwaitingModel(ReplyId),
    Finished,
    Cancelled,
    Poisoned,
}

enum State {
    Ready,
    AwaitingFetch(BTreeSet<ReplyId>),
    AwaitingModel(ReplyId),
    Finished,
    Cancelled,
    Poisoned,
}

/// What the body and the run share.
#[derive(Default)]
struct Shared {
    /// The last ids given; ids start at 1 in each run.
    next_reply: u32,
    next_attachment: u32,
    /// Set when the body asked for more ids than a `u32` holds; the run
    /// then finishes with `INTERNAL_ERROR` at its next step.
    exhausted: Option<CoreError>,
    /// Asked by the body since the last step.
    fetches: Vec<FetchStep>,
    calls: Vec<ModelCall>,
    /// Attached by the body (key sets, model bodies), not yet taken by the
    /// host.
    attachments: BTreeMap<AttachmentId, Attachment>,
    warnings: Vec<CoreWarning>,
    /// Answered, not yet taken by the body.
    replies: BTreeMap<ReplyId, Reply>,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    // A panic while the lock is held panics the body, and that poisons the
    // run, so a poisoned lock is never read in a half-changed state that
    // matters.
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What an operation's body asks its host through.
#[derive(Clone)]
pub struct Asker {
    shared: Arc<Mutex<Shared>>,
}

impl Asker {
    /// The answer to one data request, with its compiled statement for a
    /// SQL provider and the schema it expects back.
    pub fn fetch(
        &self,
        request: DataRequest,
        sql: Option<SqlStatement>,
        expect: ExpectedSchema,
    ) -> Ask {
        Ask {
            shared: Arc::clone(&self.shared),
            state: AskState::New(Asked::Fetch {
                request,
                sql,
                expect,
            }),
        }
    }

    /// The model's answer to `body`, sent to `endpoint` as it is.
    pub fn call_model(&self, endpoint: Endpoint, content_type: String, body: Vec<u8>) -> Ask {
        Ask {
            shared: Arc::clone(&self.shared),
            state: AskState::New(Asked::Call {
                endpoint,
                content_type,
                body,
            }),
        }
    }

    /// Attaches data a request names, such as a key set, and returns its id,
    /// or `INTERNAL_ERROR` once the run's ids are used up.
    pub fn attach(&self, attachment: Attachment) -> Result<AttachmentId, CoreError> {
        let mut shared = lock(&self.shared);
        attach(&mut shared, attachment)
    }

    /// Gives a warning, after every warning given before it.
    pub fn warn(&self, warning: CoreWarning) {
        lock(&self.shared).warnings.push(warning);
    }
}

fn attach(shared: &mut Shared, attachment: Attachment) -> Result<AttachmentId, CoreError> {
    let id = reserve(shared)?;
    shared.attachments.insert(id, attachment);
    Ok(id)
}

/// The run's next attachment id, or `INTERNAL_ERROR` once they are used up.
fn reserve(shared: &mut Shared) -> Result<AttachmentId, CoreError> {
    let Some(next) = shared.next_attachment.checked_add(1) else {
        return Err(exhausted(&mut *shared, "attachment"));
    };
    shared.next_attachment = next;
    Ok(AttachmentId(next))
}

/// A run's attachment numbering, shared with its host: the host layer
/// numbers what it sends on the run's behalf (a message carried whole as an
/// attachment) from the same sequence, so ids stay unique within the run.
/// It can be used while another call holds the run.
#[derive(Clone)]
pub struct AttachmentIds {
    shared: Arc<Mutex<Shared>>,
}

impl AttachmentIds {
    /// The run's next attachment id, or `INTERNAL_ERROR` once they are
    /// used up, which also ends the run at its next step.
    pub fn reserve(&self) -> Result<AttachmentId, CoreError> {
        reserve(&mut lock(&self.shared))
    }
}

/// Records that the run's ids of one kind are used up, and the error.
fn exhausted(shared: &mut Shared, what: &str) -> CoreError {
    let err = internal(&format!("the run used up its {what} ids"));
    shared.exhausted.get_or_insert_with(|| err.clone());
    err
}

enum Asked {
    Fetch {
        request: DataRequest,
        sql: Option<SqlStatement>,
        expect: ExpectedSchema,
    },
    Call {
        endpoint: Endpoint,
        content_type: String,
        body: Vec<u8>,
    },
}

/// One request, answered once the host replies to it.
#[must_use = "a request is only asked when it is awaited"]
pub struct Ask {
    shared: Arc<Mutex<Shared>>,
    state: AskState,
}

enum AskState {
    New(Asked),
    Asked(ReplyId),
    Answered,
}

impl Future for Ask {
    type Output = Reply;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Reply> {
        let this = self.get_mut();
        let mut shared = lock(&this.shared);
        match mem::replace(&mut this.state, AskState::Answered) {
            AskState::New(asked) => {
                // Out of ids, the request is never asked; the run finishes
                // with the error at its next step.
                let Some(next) = shared.next_reply.checked_add(1) else {
                    exhausted(&mut shared, "reply");
                    return Poll::Pending;
                };
                shared.next_reply = next;
                let id = ReplyId(next);
                match asked {
                    Asked::Fetch {
                        request,
                        sql,
                        expect,
                    } => shared.fetches.push(FetchStep {
                        id,
                        request,
                        sql,
                        expect,
                    }),
                    Asked::Call {
                        endpoint,
                        content_type,
                        body,
                    } => {
                        let Ok(body) = attach(&mut shared, Attachment::Bytes(body)) else {
                            return Poll::Pending;
                        };
                        shared.calls.push(ModelCall {
                            id,
                            endpoint,
                            content_type,
                            body,
                        });
                    }
                }
                this.state = AskState::Asked(id);
                Poll::Pending
            }
            AskState::Asked(id) => match shared.replies.remove(&id) {
                Some(reply) => Poll::Ready(reply),
                None => {
                    this.state = AskState::Asked(id);
                    Poll::Pending
                }
            },
            AskState::Answered => panic!("a request's reply was awaited after it was taken"),
        }
    }
}

type Body = Pin<Box<dyn Future<Output = Result<OperationResult, CoreError>> + Send>>;

/// One run of one operation; see the module documentation.
pub struct Run {
    state: State,
    /// `None` once the run has finished, been cancelled or poisoned.
    body: Option<Body>,
    shared: Arc<Mutex<Shared>>,
}

fn protocol(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::protocol(code, message)
}

impl Run {
    /// The run of the body `make` returns, given the [`Asker`] it asks
    /// through. Nothing of the body runs before the first [`Run::next`].
    pub fn new<F>(make: impl FnOnce(Asker) -> F) -> Self
    where
        F: Future<Output = Result<OperationResult, CoreError>> + Send + 'static,
    {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let body = make(Asker {
            shared: Arc::clone(&shared),
        });
        Self {
            state: State::Ready,
            body: Some(Box::pin(body)),
            shared,
        }
    }

    /// The run's attachment numbering; see [`AttachmentIds`].
    pub fn attachment_ids(&self) -> AttachmentIds {
        AttachmentIds {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn state(&self) -> RunState {
        match &self.state {
            State::Ready => RunState::Ready,
            State::AwaitingFetch(ids) => RunState::AwaitingFetch(ids.iter().copied().collect()),
            State::AwaitingModel(id) => RunState::AwaitingModel(*id),
            State::Finished => RunState::Finished,
            State::Cancelled => RunState::Cancelled,
            State::Poisoned => RunState::Poisoned,
        }
    }

    /// The error a run in a terminal state answers every call with.
    fn terminal(&self) -> Option<CoreError> {
        match self.state {
            State::Finished => Some(protocol(ErrorCode::RunFinished, "the run has finished")),
            State::Cancelled => Some(CoreError::new(
                ErrorKind::Cancelled,
                ErrorCode::RunCancelled,
                "the run was cancelled",
            )),
            State::Poisoned => Some(protocol(
                ErrorCode::RunPoisoned,
                "the run panicked and can no longer be used",
            )),
            _ => None,
        }
    }

    /// What the run needs next: a batch of fetches, a model call, or its
    /// result. An error finishes the run.
    // The run protocol names this call `next`; a run is not an iterator.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Step, CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        match &self.state {
            State::AwaitingFetch(ids) => {
                return Err(protocol(
                    ErrorCode::ReplyMissing,
                    format!("{} fetches have no reply yet", ids.len()),
                ));
            }
            State::AwaitingModel(id) => {
                return Err(protocol(
                    ErrorCode::ReplyMissing,
                    format!("model call {} has no reply yet", id.0),
                ));
            }
            _ => {}
        }
        let Some(mut body) = self.body.take() else {
            self.state = State::Poisoned;
            return Err(internal("a ready run has no body"));
        };
        // Poisoned until the poll returns: a panic in the body leaves it so.
        self.state = State::Poisoned;
        let polled = body.as_mut().poll(&mut Context::from_waker(Waker::noop()));
        match polled {
            Poll::Ready(Ok(result)) => {
                self.state = State::Finished;
                Ok(Step::Done { result })
            }
            Poll::Ready(Err(err)) => {
                self.state = State::Finished;
                Err(err)
            }
            Poll::Pending => {
                let mut shared = lock(&self.shared);
                if let Some(err) = shared.exhausted.take() {
                    drop(shared);
                    self.state = State::Finished;
                    return Err(err);
                }
                let fetches = mem::take(&mut shared.fetches);
                let mut calls = mem::take(&mut shared.calls);
                drop(shared);
                let step = match (fetches.is_empty(), calls.len()) {
                    (false, 0) => {
                        self.state = State::AwaitingFetch(fetches.iter().map(|f| f.id).collect());
                        Step::Fetch { requests: fetches }
                    }
                    (true, 1) => {
                        let call = calls.pop().expect("one call");
                        self.state = State::AwaitingModel(call.id);
                        Step::CallModel(call)
                    }
                    (true, 0) => {
                        self.state = State::Finished;
                        return Err(internal("the operation waited on nothing it asked for"));
                    }
                    _ => {
                        self.state = State::Finished;
                        return Err(internal(
                            "the operation waited on a model call together with other requests",
                        ));
                    }
                };
                self.body = Some(body);
                Ok(step)
            }
        }
    }

    /// The host's answer to fetch `id`.
    pub fn submit_fetch(&mut self, id: ReplyId, result: FetchResult) -> Result<(), CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        match &mut self.state {
            State::AwaitingFetch(pending) if pending.contains(&id) => {
                pending.remove(&id);
                if pending.is_empty() {
                    self.state = State::Ready;
                }
                lock(&self.shared).replies.insert(id, Reply::Fetch(result));
                Ok(())
            }
            State::AwaitingModel(call) if *call == id => Err(protocol(
                ErrorCode::WrongReply,
                format!("{} is a model call, not a fetch", id.0),
            )),
            _ => Err(unknown(id)),
        }
    }

    /// The host's answer to model call `id`.
    pub fn submit_model(&mut self, id: ReplyId, result: ModelResult) -> Result<(), CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        match &self.state {
            State::AwaitingModel(call) if *call == id => {
                self.state = State::Ready;
                lock(&self.shared).replies.insert(id, Reply::Model(result));
                Ok(())
            }
            State::AwaitingFetch(pending) if pending.contains(&id) => Err(protocol(
                ErrorCode::WrongReply,
                format!("{} is a fetch, not a model call", id.0),
            )),
            _ => Err(unknown(id)),
        }
    }

    /// Cancels the run: its body is dropped and every later call is refused
    /// with `RUN_CANCELLED`.
    pub fn cancel(&mut self) -> Result<(), CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        self.body = None;
        self.state = State::Cancelled;
        let mut shared = lock(&self.shared);
        shared.replies.clear();
        shared.attachments.clear();
        Ok(())
    }

    /// Ends the run because the host cannot hand it what it asked for, such
    /// as data the admission table refuses: its body is dropped, the
    /// replies it holds are discarded, and every later call is refused with
    /// `RUN_FINISHED`. Attachments it made stay until the host takes them.
    pub fn end(&mut self) -> Result<(), CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        self.body = None;
        self.state = State::Finished;
        lock(&self.shared).replies.clear();
        Ok(())
    }

    /// Marks the run unusable after a panic the host contained outside it.
    pub fn poison(&mut self) {
        self.body = None;
        self.state = State::Poisoned;
    }

    /// The warnings given since this was last called, in order. A host takes
    /// them after every call, whether it succeeded or not.
    pub fn take_warnings(&mut self) -> Vec<CoreWarning> {
        mem::take(&mut lock(&self.shared).warnings)
    }

    /// Takes an attachment a step names (a key set, a model call's body).
    pub fn take_attachment(&mut self, id: AttachmentId) -> Option<Attachment> {
        lock(&self.shared).attachments.remove(&id)
    }
}

fn unknown(id: ReplyId) -> CoreError {
    protocol(
        ErrorCode::UnknownReply,
        format!(
            "the run is not waiting on {}: it is unknown, stale or already answered",
            id.0
        ),
    )
}

pub(crate) fn internal(message: &str) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
}

#[cfg(test)]
mod tests {
    use sdm_contract::ErrorCode;
    use sdm_contract::data::{DataRequest, ExpectedSchema};
    use sdm_contract::run::Endpoint;

    use super::{Attachment, Run, RunState, lock};

    fn describe() -> DataRequest {
        DataRequest::Describe { tables: vec![] }
    }

    #[test]
    fn running_out_of_reply_ids_is_an_internal_error() {
        let mut run = Run::new(|asker| async move {
            asker
                .fetch(describe(), None, ExpectedSchema::Discover {})
                .await;
            unreachable!("never answered")
        });
        lock(&run.shared).next_reply = u32::MAX;
        let err = run.next().unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(run.state(), RunState::Finished);
    }

    #[test]
    fn running_out_of_attachment_ids_is_an_internal_error() {
        let mut run = Run::new(|asker| async move {
            asker
                .call_model(Endpoint::Predict, "application/json".into(), vec![])
                .await;
            unreachable!("never answered")
        });
        lock(&run.shared).next_attachment = u32::MAX;
        let err = run.next().unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(run.state(), RunState::Finished);

        // Through attach, the body sees the error itself.
        let mut run = Run::new(|asker| async move {
            asker.attach(Attachment::Bytes(vec![1]))?;
            unreachable!("the attach fails")
        });
        lock(&run.shared).next_attachment = u32::MAX;
        let err = run.next().unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(run.state(), RunState::Finished);
    }
}
