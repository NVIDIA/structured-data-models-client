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

use std::collections::BTreeMap;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, DataRequest, ExpectedSchema, FetchPlan, ProviderCapabilities, ProviderError,
};
use sdm_contract::ops::{OperationResult, SAMPLER_STATE_VERSION, SamplerMemo, SamplerState};
use sdm_contract::run::{Endpoint, FetchStep, ModelCall, ReplyId};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Step};

use crate::compose::DataAccess;

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
    /// The schema as the source handed it over, before admission widened
    /// it, so inference sees what the source returned; the same as `schema`
    /// for data nothing admitted.
    pub received: SchemaRef,
}

impl ArrowData {
    /// Data of `schema`, as it was received.
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        ArrowData {
            received: Arc::clone(&schema),
            schema,
            batches,
        }
    }
}

/// The host's answer to one fetch.
#[derive(Debug, Clone)]
pub enum FetchResult {
    Data(ArrowData),
    /// One page of the answer, more following: the run asks again with the
    /// provider's `continuation` (design choice 1, paging) and gives the
    /// body the pages as one answer, so a body never sees a page.
    Page {
        data: ArrowData,
        continuation: String,
    },
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
    /// Each unanswered fetch, with the schema it expects back.
    AwaitingFetch(BTreeMap<ReplyId, ExpectedSchema>),
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
    /// Set when the body asked for more ids than a `u32` holds, or a
    /// provider's pages of one answer did not match; the run then finishes
    /// with that error at its next step.
    exhausted: Option<CoreError>,
    /// Asked by the body since the last step.
    fetches: Vec<FetchStep>,
    calls: Vec<ModelCall>,
    /// Attached by the body (key sets, model bodies), not yet taken by the
    /// host.
    attachments: BTreeMap<AttachmentId, Attachment>,
    warnings: Vec<CoreWarning>,
    /// The sampler memo the operation leaves for the host (decision E52),
    /// as it last kept it.
    memo: Option<SamplerMemo>,
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
    /// The run's data access, for an operation that names a provider.
    data: Option<DataAccess>,
}

impl Asker {
    /// The answer to one data request, planned by the run's data layer for
    /// its provider (decision H14); a request the plan refuses, or an
    /// operation that names no provider, is the error.
    pub fn fetch(&self, request: DataRequest) -> Result<Ask, CoreError> {
        let Some(data) = &self.data else {
            return Err(internal(
                "the operation asked for data and names no provider",
            ));
        };
        let plan = data.plan(&request)?;
        Ok(self.ask_planned(request, plan))
    }

    /// What the run's provider declared it answers, for an operation that
    /// names one.
    pub fn capabilities(&self) -> Result<&ProviderCapabilities, CoreError> {
        self.data
            .as_ref()
            .map(DataAccess::capabilities)
            .ok_or_else(|| internal("the operation asked what its provider answers and names none"))
    }

    /// Whether the run's provider compares a window's bounds with the
    /// stored text of the window's column (`DataLayer::compares_stored_times`);
    /// `false` for an operation that names no provider.
    pub fn compares_stored_times(&self) -> bool {
        self.data
            .as_ref()
            .is_some_and(DataAccess::compares_stored_times)
    }

    /// Whether the run's provider orders a seeded sample by the key's data
    /// type (`DataLayer::samples_by_key_dtype`); `false` for an operation
    /// that names no provider.
    pub fn samples_by_key_dtype(&self) -> bool {
        self.data
            .as_ref()
            .is_some_and(DataAccess::samples_by_key_dtype)
    }

    /// The catalog answer `data` holds, the admitted answer to `request`,
    /// a `describe`: read by the run's data layer into the contract's
    /// dialect-neutral shape (decision E54), so the operation reads no
    /// dialect's catalog.
    pub fn read(
        &self,
        request: &DataRequest,
        data: &ArrowData,
    ) -> Result<CatalogAnswer, CoreError> {
        let Some(access) = &self.data else {
            return Err(internal(
                "the operation read a catalog answer and names no provider",
            ));
        };
        access.read(request, &data.batches)
    }

    /// The answer to `request`, asked with `plan` in place of the data
    /// layer's: for the test-only operations, which drive plans of any size
    /// or shape.
    #[cfg(feature = "test-ops")]
    pub(crate) fn fetch_planned(&self, request: DataRequest, plan: FetchPlan) -> Ask {
        self.ask_planned(request, plan)
    }

    fn ask_planned(&self, request: DataRequest, plan: FetchPlan) -> Ask {
        Ask {
            shared: Arc::clone(&self.shared),
            state: AskState::New(Asked::Fetch(Box::new((request, plan)))),
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

    /// Keeps `memo` as what the run leaves the host, in place of any kept
    /// before: what a `materialize` run has read of its source, which the
    /// host takes once the run ends, done or failed (decision E52).
    pub fn keep(&self, memo: SamplerMemo) {
        lock(&self.shared).memo = Some(memo);
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
    /// Boxed: a request and its plan are far larger than a model call.
    Fetch(Box<(DataRequest, FetchPlan)>),
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
    /// Asked under the id; a fetch keeps its request and plan, to ask for
    /// the next page, and the pages read so far.
    Asked {
        id: ReplyId,
        fetch: Option<Box<(DataRequest, FetchPlan)>>,
        pages: Option<ArrowData>,
    },
    Answered,
}

/// The run's next reply id, or `None` once they are used up, which ends the
/// run with `INTERNAL_ERROR` at its next step.
fn next_reply(shared: &mut Shared) -> Option<ReplyId> {
    let Some(next) = shared.next_reply.checked_add(1) else {
        exhausted(shared, "reply");
        return None;
    };
    shared.next_reply = next;
    Some(ReplyId(next))
}

/// `page` added to the pages read before it, which it must match in
/// schema: a provider's pages are one answer (design choice 1, paging).
fn add_page(pages: Option<ArrowData>, page: ArrowData) -> Result<ArrowData, CoreError> {
    let Some(mut pages) = pages else {
        return Ok(page);
    };
    if pages.schema != page.schema || pages.received != page.received {
        return Err(schema_mismatch(
            "pages",
            None,
            "a page of the answer has another schema than the pages before it",
        ));
    }
    pages.batches.extend(page.batches);
    Ok(pages)
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
                let Some(id) = next_reply(&mut shared) else {
                    return Poll::Pending;
                };
                let mut fetch = None;
                match asked {
                    Asked::Fetch(asked) => {
                        let (request, plan) = (*asked).clone();
                        shared.fetches.push(FetchStep {
                            id,
                            request,
                            plan,
                            continuation: None,
                        });
                        fetch = Some(asked);
                    }
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
                this.state = AskState::Asked {
                    id,
                    fetch,
                    pages: None,
                };
                Poll::Pending
            }
            AskState::Asked { id, fetch, pages } => match shared.replies.remove(&id) {
                None => {
                    this.state = AskState::Asked { id, fetch, pages };
                    Poll::Pending
                }
                // A page: kept, and the next one asked for with the
                // provider's continuation, under a new id.
                Some(Reply::Fetch(FetchResult::Page { data, continuation })) => {
                    let pages = match add_page(pages, data) {
                        Ok(pages) => pages,
                        Err(err) => {
                            shared.exhausted.get_or_insert(err);
                            return Poll::Pending;
                        }
                    };
                    let Some(asked) = fetch else {
                        shared
                            .exhausted
                            .get_or_insert_with(|| internal("a page answered no fetch"));
                        return Poll::Pending;
                    };
                    let Some(next) = next_reply(&mut shared) else {
                        return Poll::Pending;
                    };
                    let (request, plan) = (*asked).clone();
                    shared.fetches.push(FetchStep {
                        id: next,
                        request,
                        plan,
                        continuation: Some(continuation),
                    });
                    this.state = AskState::Asked {
                        id: next,
                        fetch: Some(asked),
                        pages: Some(pages),
                    };
                    Poll::Pending
                }
                // The last page: the pages read as one answer.
                Some(Reply::Fetch(FetchResult::Data(data))) if pages.is_some() => {
                    match add_page(pages, data) {
                        Ok(whole) => Poll::Ready(Reply::Fetch(FetchResult::Data(whole))),
                        Err(err) => {
                            shared.exhausted.get_or_insert(err);
                            Poll::Pending
                        }
                    }
                }
                Some(reply) => Poll::Ready(reply),
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
    /// The opaque vendor the run's provider hands over, as its data layer
    /// declares it ([`DataAccess::opaque_vendor`]).
    opaque_vendor: Option<&'static str>,
}

fn protocol(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::protocol(code, message)
}

impl Run {
    /// The run of the body `make` returns, given the [`Asker`] it asks
    /// through, for an operation that names no provider. Nothing of the
    /// body runs before the first [`Run::next`].
    pub fn new<F>(make: impl FnOnce(Asker) -> F) -> Self
    where
        F: Future<Output = Result<OperationResult, CoreError>> + Send + 'static,
    {
        Run::build(None, make)
    }

    /// [`Run::new`], for an operation whose data requests `data` plans.
    pub fn with_data<F>(data: DataAccess, make: impl FnOnce(Asker) -> F) -> Self
    where
        F: Future<Output = Result<OperationResult, CoreError>> + Send + 'static,
    {
        Run::build(Some(data), make)
    }

    fn build<F>(data: Option<DataAccess>, make: impl FnOnce(Asker) -> F) -> Self
    where
        F: Future<Output = Result<OperationResult, CoreError>> + Send + 'static,
    {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let opaque_vendor = data.as_ref().and_then(DataAccess::opaque_vendor);
        let body = make(Asker {
            shared: Arc::clone(&shared),
            data,
        });
        Self {
            state: State::Ready,
            body: Some(Box::pin(body)),
            shared,
            opaque_vendor,
        }
    }

    /// The vendor of the opaque Arrow extension types the run's provider
    /// hands over, as its data layer declares it; `None` for a provider
    /// that declares none, or an operation that names no provider. The host
    /// layer reads an opaque column by its vendor's layout only when the
    /// column names this vendor (register P1B-43).
    pub fn opaque_vendor(&self) -> Option<&'static str> {
        self.opaque_vendor
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
            State::AwaitingFetch(ids) => RunState::AwaitingFetch(ids.keys().copied().collect()),
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
                        self.state = State::AwaitingFetch(
                            fetches
                                .iter()
                                .map(|f| (f.id, f.plan.expect().clone()))
                                .collect(),
                        );
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

    /// The host's answer to fetch `id`. Data must match the schema the
    /// request expects back, stored with the request (decision H16, check
    /// (b)): the same columns in the same order, by name, each of the
    /// expected closed-set type, holding the field metadata the request
    /// requires, and no nulls where it expects none. A mismatch is
    /// `PROVIDER_SCHEMA_MISMATCH`, and ends the run, as a refusal of data
    /// does; nothing is dropped, relabelled or filled. Each page of a paged
    /// answer is held to it as it arrives, as a whole answer is. A
    /// discovery request takes the schema the data declares.
    pub fn submit_fetch(&mut self, id: ReplyId, result: FetchResult) -> Result<(), CoreError> {
        if let Some(err) = self.terminal() {
            return Err(err);
        }
        match &mut self.state {
            State::AwaitingFetch(pending) if pending.contains_key(&id) => {
                if let FetchResult::Data(data) | FetchResult::Page { data, .. } = &result
                    && let Err(mismatch) = check_expected(&pending[&id], data)
                {
                    self.end()?;
                    return Err(mismatch);
                }
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
            State::AwaitingFetch(pending) if pending.contains_key(&id) => Err(protocol(
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

    /// Takes the sampler state the run left (decision E52), its memo
    /// attached as JSON under the run's next id, for the host to take; none
    /// when the run kept none, or it was taken. A host takes it once the
    /// run has ended, whether it finished or failed.
    pub fn take_sampler_state(&mut self) -> Result<Option<SamplerState>, CoreError> {
        let mut shared = lock(&self.shared);
        let Some(memo) = shared.memo.take() else {
            return Ok(None);
        };
        let bytes =
            serde_json::to_vec(&memo).map_err(|_| internal("a sampler memo does not serialize"))?;
        let id = attach(&mut shared, Attachment::Bytes(bytes))?;
        Ok(Some(SamplerState {
            version: SAMPLER_STATE_VERSION,
            memo: id,
        }))
    }
}

/// The refusal of fetched data whose schema is not the one asked for:
/// `check` names which check found it, `declared` (a batch against its
/// attachment's schema) or `expected` (the data against its request's).
pub fn schema_mismatch(check: &str, column: Option<&str>, reason: &str) -> CoreError {
    let mut error = CoreError::new(
        ErrorKind::DataSource,
        ErrorCode::ProviderSchemaMismatch,
        format!("the provider's data does not match the {check} schema: {reason}"),
    )
    .with_field("check", Detail::text(check))
    .with_field("reason", Detail::text(reason));
    if let Some(column) = column {
        error = error.with_field("column", Detail::text(column));
    }
    error
}

/// Check (b): `data` against the schema its request expects back.
fn check_expected(expect: &ExpectedSchema, data: &ArrowData) -> Result<(), CoreError> {
    let ExpectedSchema::Schema { fields } = expect else {
        return Ok(());
    };
    let mismatch = |column: Option<&str>, reason: &str| schema_mismatch("expected", column, reason);
    let got = data.schema.fields();
    if got.len() != fields.len() {
        return Err(mismatch(None, "the number of columns differs"));
    }
    for (index, (field, expected)) in got.iter().zip(fields).enumerate() {
        let name = expected.name.as_str();
        if field.name() != name {
            return Err(mismatch(Some(name), "a column's name differs"));
        }
        match ColumnType::from_arrow(name, field.data_type()) {
            Ok(column_type) if column_type == expected.column_type => {}
            _ => return Err(mismatch(Some(name), "a column's type differs")),
        }
        if expected
            .metadata
            .iter()
            .any(|(key, value)| field.metadata().get(key) != Some(value))
        {
            return Err(mismatch(
                Some(name),
                "a column lacks metadata the request requires",
            ));
        }
        if !expected.nullable
            && data
                .batches
                .iter()
                .any(|b| b.column(index).null_count() > 0)
        {
            return Err(mismatch(
                Some(name),
                "a column holds nulls the request does not take",
            ));
        }
    }
    Ok(())
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
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch};
    use sdm_contract::data::{DataRequest, DescribePart, ExpectedSchema, FetchPlan, field};
    use sdm_contract::run::Endpoint;
    use sdm_contract::{ColumnType, ErrorCode};

    use super::{ArrowData, Attachment, FetchResult, Run, RunState, Step, lock};

    fn describe() -> DataRequest {
        DataRequest::Describe {
            tables: vec![],
            part: DescribePart::Tables,
        }
    }

    #[test]
    fn running_out_of_reply_ids_is_an_internal_error() {
        let mut run = Run::new(|asker| async move {
            let plan = FetchPlan::Host {
                expect: ExpectedSchema::Discover {},
            };
            asker.ask_planned(describe(), plan).await;
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

    /// A page of one nullable `a` column holding `values`, more following.
    fn page(values: Vec<Option<i64>>) -> FetchResult {
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "a",
            arrow_schema::DataType::Int64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(values))],
        )
        .unwrap();
        FetchResult::Page {
            data: ArrowData::new(schema, vec![batch]),
            continuation: "more".into(),
        }
    }

    #[test]
    fn every_page_is_held_to_the_expected_schema() {
        // A null where the request takes none fails on the page that holds
        // it, first or later, as it fails in an answer of one page.
        for valid_before in [0, 1] {
            let mut run = Run::new(|asker| async move {
                let plan = FetchPlan::Host {
                    expect: ExpectedSchema::Schema {
                        fields: vec![field("a", ColumnType::Int64, false)],
                    },
                };
                asker.ask_planned(describe(), plan).await;
                unreachable!("the run ends on the page with a null")
            });
            for _ in 0..valid_before {
                let Step::Fetch { requests } = run.next().unwrap() else {
                    panic!("not a fetch")
                };
                run.submit_fetch(requests[0].id, page(vec![Some(1)]))
                    .unwrap();
            }
            let Step::Fetch { requests } = run.next().unwrap() else {
                panic!("not a fetch")
            };
            let err = run
                .submit_fetch(requests[0].id, page(vec![None]))
                .unwrap_err();
            assert_eq!(err.code, ErrorCode::ProviderSchemaMismatch, "{err:?}");
            assert_eq!(run.state(), RunState::Finished);
        }
    }
}
