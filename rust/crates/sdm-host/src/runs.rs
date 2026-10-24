// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run ownership, panic containment and busy handling: what every binding
//! calls.
//!
//! A [`HostRun`] owns one core run. A binding hands it control messages as
//! bytes, with the attachments each names, and takes control messages back
//! as bytes ([`Outgoing`]), errors included; it may share the run between
//! threads. Every call:
//!
//! - is contained: a panic anywhere below it poisons the run and comes back
//!   as `INTERNAL_ERROR` with `fields.panic = true` and the panic's
//!   location, never its message, and never as an unwind through the
//!   binding; [`contain`] reports the panic out of band, so no error a host
//!   hands in can poison a run (decision H17);
//! - is refused with `RUN_BUSY` (kind `protocol`) while another call is in
//!   the run, rather than waiting;
//! - reads control messages through the codecs, under the control cap, and
//!   writes them whole: one whose encoding would pass the cap travels as an
//!   attachment numbered in the run's sequence (decision H7).
//!
//! Attachments follow decision E2. The core numbers its own from 1 in each
//! run, and the host takes them by id ([`HostRun::take_attachment`]); an
//! attachment carrying a message whole travels with that message instead.
//! The host names the attachments of each message it sends with ids of its
//! choosing, scoped to that message: every id the message names must be
//! attached, as the kind it names, and every attachment named, and the ids
//! mean nothing once the message is read. A `start` carries the
//! operation's input-sized fields this way (decision H5). Fetched Arrow data passes through admission on the way
//! in; data admission refuses ends the run with that refusal.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, Once, PoisonError, TryLockError};

use arrow_schema::Schema;
use sdm_contract::data::AttachmentId;
use sdm_contract::data::ProviderError;
use sdm_contract::facts::LocalZone;
use sdm_contract::ops::AttachmentKind;
use sdm_contract::run::{FetchOutcome, ModelOutcome};
use sdm_contract::{ControlMessage, CoreError, Detail, ErrorKind, Scalar, Sentence};
use sdm_core::attached::check_attachments;
use sdm_core::run::schema_mismatch;
use sdm_core::{
    ArrowData, Attachment, AttachmentIds, Attachments, CompositionRoot, FetchResult, ModelResult,
    Run, RunState,
};

use crate::admission::admit_from;
use crate::codec::{Outgoing, decode, write, write_error};
use crate::zones::{ZoneResolver, ZoneRoot, ZoneSource};

/// One core run, owned by the host layer; see the module documentation.
pub struct HostRun {
    run: Mutex<Run>,
    ids: AttachmentIds,
    /// The zones of the run's host facts, which the core resolves with.
    zones: Arc<ZoneResolver>,
}

/// What a contained call failed with: an error, or a panic, which
/// [`contain`] reports out of band (decision H17).
#[derive(Debug, Clone)]
pub struct Failure {
    pub error: CoreError,
    /// Whether the call panicked; only [`contain`] sets it.
    pub panicked: bool,
}

impl From<CoreError> for Failure {
    fn from(error: CoreError) -> Self {
        Failure {
            error,
            panicked: false,
        }
    }
}

thread_local! {
    /// How many contained calls this thread is inside.
    static CONTAINED: Cell<usize> = const { Cell::new(0) };
    /// Where the last contained panic on this thread happened.
    static LOCATION: RefCell<Option<String>> = const { RefCell::new(None) };
}

static HOOK: Once = Once::new();

/// Installs, once, the panic hook that writes only the location of a panic
/// inside a contained call, never its message, which may quote input; any
/// other panic goes to the hook that was there before.
fn install_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if CONTAINED.with(Cell::get) == 0 {
                previous(info);
                return;
            }
            let at = info.location().map_or_else(
                || "an unknown location".to_owned(),
                |l| format!("{}:{}:{}", l.file(), l.line(), l.column()),
            );
            eprintln!("sdm core: a contained panic at {at}");
            LOCATION.with(|location| *location.borrow_mut() = Some(at));
        }));
    });
}

/// Runs `call`, turning a panic into `INTERNAL_ERROR` with
/// `fields.panic = true` and the panic's location, without its message,
/// and reporting it as a panic in the [`Failure`]. The panic hook is
/// installed at the first contained call, which for a binding is its first
/// `HostRun::start`.
pub fn contain<T>(call: impl FnOnce() -> Result<T, CoreError>) -> Result<T, Failure> {
    install_hook();
    CONTAINED.with(|depth| depth.set(depth.get() + 1));
    let outcome = catch_unwind(AssertUnwindSafe(call));
    CONTAINED.with(|depth| depth.set(depth.get() - 1));
    match outcome {
        Ok(result) => result.map_err(Failure::from),
        Err(_payload) => {
            let at = LOCATION
                .with(|location| location.borrow_mut().take())
                .unwrap_or_else(|| "an unknown location".to_owned());
            Err(Failure {
                error: CoreError::new(
                    Sentence::Internal,
                    "the core panicked; the run can no longer be used",
                )
                .with_field("panic", Detail::Scalar(Scalar::Bool(true)))
                .with_field("location", Detail::text(at)),
                panicked: true,
            })
        }
    }
}

/// Why a binding could not read an Arrow attachment it was handed into
/// Arrow data, before the core saw any of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unreadable {
    /// The producer failed while handing the data over: the provider
    /// failed.
    Producer,
    /// A batch broke the schema its own stream declares, such as nulls in
    /// a column declared without them: check (a) refuses it.
    Declared,
}

/// The provider error a producer's failure is, with nothing of its text,
/// which may quote values (decision H17).
fn producer_failed() -> ProviderError {
    ProviderError {
        error_class: None,
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: None,
        vendor_code: None,
    }
}

/// The Arrow attachments a binding could not read, by id, and why.
pub type Unreadables = BTreeMap<AttachmentId, Unreadable>;

/// `attachments` with each unreadable one standing in as empty Arrow data
/// of its id, so the message's attachments are checked as a whole; a
/// stand-in is never admitted.
fn with_stand_ins(mut attachments: Attachments, unreadable: &Unreadables) -> Attachments {
    for &id in unreadable.keys() {
        attachments.insert(
            id,
            Attachment::Arrow(ArrowData::new(Arc::new(Schema::empty()), vec![])),
        );
    }
    attachments
}

/// The error an unreadable attachment is, naming the attachment.
fn unreadable_error(id: AttachmentId, failure: Unreadable) -> CoreError {
    let error = match failure {
        Unreadable::Producer => CoreError::provider(&producer_failed()),
        Unreadable::Declared => schema_mismatch(
            "declared",
            None,
            "a batch does not hold to the schema its stream declares",
        ),
    };
    error.with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0)))
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The error of a `start` that made no run, as a control message; there is
/// no run to number its attachment, so it is the message's own first.
fn start_error(error: &CoreError) -> Outgoing {
    write_error(error, || Ok(AttachmentId(1)))
}

/// Whether `text` is a POSIX `TZ` string a run takes as its local zone:
/// the one parser [`HostRun::start`] checks with, so a binding can name a
/// local zone only when a run would read it, and pass none otherwise.
pub fn is_posix_zone(text: &str) -> bool {
    jiff::tz::TimeZone::posix(text).is_ok()
}

/// Refuses a local zone the host's facts name that cannot be read: a POSIX
/// `TZ` string must parse. An IANA name is resolved when a run first needs
/// it.
fn check_local_zone(zone: Option<&LocalZone>) -> Result<(), CoreError> {
    if let Some(LocalZone::Posix(text)) = zone
        && !is_posix_zone(text)
    {
        return Err(CoreError::new(
            Sentence::LocalZoneUnread,
            "the host's local zone is not a POSIX TZ string",
        )
        .with_field("field", Detail::text("facts.local_zone")));
    }
    Ok(())
}

impl HostRun {
    /// Starts the run a `start` message asks for, with the attachments it
    /// names; a refusal is an `error` control message.
    pub fn start(
        root: &CompositionRoot,
        message: &[u8],
        attachments: Attachments,
    ) -> Result<HostRun, Outgoing> {
        HostRun::start_reading(root, message, attachments, &Unreadables::new())
    }

    /// [`HostRun::start`], for a binding that could not read some of the
    /// Arrow attachments it was handed: `unreadable` names them and why.
    /// The `start` and its whole set of attachments are checked as any
    /// start's are, an unreadable one standing in as Arrow data of its id,
    /// so a message that names the wrong attachments is still
    /// `INVALID_MESSAGE`; only then does an unreadable attachment refuse
    /// the start, as a provider error or check (a)'s refusal.
    pub fn start_reading(
        root: &CompositionRoot,
        message: &[u8],
        attachments: Attachments,
        unreadable: &Unreadables,
    ) -> Result<HostRun, Outgoing> {
        contain(|| {
            let ControlMessage::Start(start) = decode(message)? else {
                return Err(invalid("a run starts with a start message"));
            };
            // The version first: no other field means anything under
            // another contract (H25). The core checks the attachments.
            sdm_core::dispatch::check_contract(start.contract)?;
            check_local_zone(start.facts.local_zone.as_ref())?;
            let zones = Arc::new(ZoneResolver::new(start.facts.zone_roots.iter().map(
                |root| ZoneRoot {
                    path: root.path.clone().into(),
                    version: root.version.clone(),
                },
            )));
            if let Some((&id, &failure)) = unreadable.first_key_value() {
                let envelope = with_stand_ins(attachments, unreadable);
                check_attachments(&start.operation.attachments(), &envelope)?;
                return Err(unreadable_error(id, failure));
            }
            // The core resolves zones through the host's own resolver, the
            // one whose sources the host records (decision H14).
            let core_zones: Arc<dyn sdm_core::Zones> = zones.clone();
            let run = sdm_core::start(root, *start, attachments, core_zones)?;
            Ok(HostRun::with(run, zones))
        })
        .map_err(|failure| start_error(&failure.error))
    }

    /// The host run of a core run built directly, as tests and the
    /// pure-Rust test host build one, with the zones it was built with.
    pub fn with(run: Run, zones: Arc<ZoneResolver>) -> HostRun {
        HostRun {
            ids: run.attachment_ids(),
            run: Mutex::new(run),
            zones,
        }
    }

    /// `error` as an `error` control message, whole.
    fn error(&self, error: &CoreError) -> Outgoing {
        write_error(error, || self.ids.reserve())
    }

    /// Runs `call` on the run: refused with `RUN_BUSY` while another call is
    /// in it, and contained, a panic poisoning the run; an error comes back
    /// as an `error` control message.
    fn enter<T>(&self, call: impl FnOnce(&mut Run) -> Result<T, CoreError>) -> Result<T, Outgoing> {
        let mut guard: MutexGuard<'_, Run> = match self.run.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => {
                return Err(self.error(&CoreError::protocol(
                    Sentence::RunBusy,
                    "another call is in this run; a run takes one call at a time",
                )));
            }
            // Only a contained panic leaves the lock poisoned, and the run
            // was poisoned with it.
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        let result = contain(|| call(&mut guard));
        // Only a panic poisons the run, as contain reports it; an error's
        // fields, which a host can set, never do.
        if result.as_ref().is_err_and(|failure| failure.panicked) {
            guard.poison();
        }
        drop(guard);
        result.map_err(|failure| self.error(&failure.error))
    }

    /// The run's next step, as a `step` control message, whole.
    pub fn next(&self) -> Result<Outgoing, Outgoing> {
        self.enter(|run| {
            let step = run.next()?;
            write(&ControlMessage::Step(step), || self.ids.reserve())
        })
    }

    /// Hands the run a `fetch_reply` or `model_reply`, with the attachments
    /// it names. Fetched data is admitted first; data admission refuses ends
    /// the run with that refusal.
    pub fn reply(&self, message: &[u8], attachments: Attachments) -> Result<(), Outgoing> {
        self.reply_reading(message, attachments, &Unreadables::new())
    }

    /// [`HostRun::reply`], for a binding that could not read some of the
    /// Arrow attachments it was handed: `unreadable` names them and why.
    /// The reply and its whole set of attachments are checked as any
    /// reply's are, an unreadable one standing in as Arrow data of its id,
    /// before the run changes. Then, for a fetch the run awaits whose data
    /// is unreadable, a producer's failure reaches the run as the
    /// provider's error, as a failed fetch the host reports does, and a
    /// batch that broke its own stream's schema ends the run with check
    /// (a)'s refusal, as admission does; a reply the run does not await
    /// gets the run's own refusal.
    pub fn reply_reading(
        &self,
        message: &[u8],
        attachments: Attachments,
        unreadable: &Unreadables,
    ) -> Result<(), Outgoing> {
        self.enter(|run| {
            let mut attachments = with_stand_ins(attachments, unreadable);
            match decode(message)? {
                ControlMessage::FetchReply(reply) => {
                    let result = match reply.outcome {
                        FetchOutcome::Data { data, continuation } => {
                            check_attachments(&[(data, AttachmentKind::Arrow)], &attachments)?;
                            if let Some(&failure) = unreadable.get(&data) {
                                // Not awaited: the run refuses it before it
                                // reads the outcome. Awaited: the producer's
                                // failure is the fetch's.
                                if !awaits(run, reply.id) || failure == Unreadable::Producer {
                                    return run.submit_fetch(
                                        reply.id,
                                        FetchResult::Error(producer_failed()),
                                    );
                                }
                                run.end()?;
                                return Err(unreadable_error(data, failure));
                            }
                            let Some(Attachment::Arrow(arrow)) = attachments.remove(&data) else {
                                return Err(invalid("fetched data is an Arrow attachment"));
                            };
                            if !awaits(run, reply.id) {
                                // The run's own refusal of a reply it does not
                                // await: finished, stale, or not a fetch.
                                return run.submit_fetch(reply.id, FetchResult::Data(arrow));
                            }
                            match admit_from(&arrow, run.opaque_vendor()) {
                                // A page, more following (design choice 1,
                                // paging): the run asks for the next.
                                Ok(admitted) => match continuation {
                                    Some(continuation) => FetchResult::Page {
                                        data: admitted,
                                        continuation,
                                    },
                                    None => FetchResult::Data(admitted),
                                },
                                Err(refusal) => {
                                    run.end()?;
                                    return Err(refusal);
                                }
                            }
                        }
                        FetchOutcome::Error { error } => {
                            check_attachments(&[], &attachments)?;
                            FetchResult::Error(error)
                        }
                    };
                    run.submit_fetch(reply.id, result)
                }
                ControlMessage::ModelReply(reply) => {
                    let result = match reply.outcome {
                        ModelOutcome::Response {
                            status,
                            content_type,
                            body,
                        } => {
                            check_attachments(&[(body, AttachmentKind::Bytes)], &attachments)?;
                            let Some(Attachment::Bytes(body_bytes)) = attachments.remove(&body)
                            else {
                                return Err(invalid("a model response's body is bytes"));
                            };
                            ModelResult::Response {
                                status,
                                content_type,
                                body: body_bytes,
                            }
                        }
                        ModelOutcome::TransportError { kind, retryable } => {
                            check_attachments(&[], &attachments)?;
                            ModelResult::TransportError { kind, retryable }
                        }
                        ModelOutcome::Failed { answer } => {
                            let detail = match answer.detail {
                                Some(detail) => {
                                    check_attachments(
                                        &[(detail, AttachmentKind::Bytes)],
                                        &attachments,
                                    )?;
                                    let Some(Attachment::Bytes(bytes)) =
                                        attachments.remove(&detail)
                                    else {
                                        return Err(invalid(
                                            "a failed model call's detail is bytes",
                                        ));
                                    };
                                    Some(bytes)
                                }
                                None => {
                                    check_attachments(&[], &attachments)?;
                                    None
                                }
                            };
                            ModelResult::Failed {
                                status: answer.status,
                                detail,
                                provider_code: answer.provider_code,
                                provider_error: answer.provider_error,
                            }
                        }
                    };
                    run.submit_model(reply.id, result)
                }
                _ => Err(invalid(
                    "a run takes replies here: a fetch_reply or a model_reply",
                )),
            }
        })
    }

    /// Cancels the run; its attachments are released.
    pub fn cancel(&self) -> Result<(), Outgoing> {
        self.enter(Run::cancel)
    }

    /// Takes an attachment the run made and a step named.
    pub fn take_attachment(&self, id: AttachmentId) -> Result<Option<Attachment>, Outgoing> {
        self.enter(|run| Ok(run.take_attachment(id)))
    }

    /// The warnings given since the last call, in order, as one `warnings`
    /// control message, whole; none when there are none. A warning the
    /// reference gives once in a process
    /// (`sdm_core::materialize::ONCE_PER_PROCESS`) is given here the first
    /// time a run of this process gives it, and dropped after (decision
    /// E52): every binding takes its warnings here, so each inherits it.
    pub fn take_warnings(&self) -> Result<Option<Outgoing>, Outgoing> {
        self.enter(|run| {
            let warnings: Vec<_> = run
                .take_warnings()
                .into_iter()
                .filter(|w| first_in_process(&w.code))
                .collect();
            if warnings.is_empty() {
                return Ok(None);
            }
            write(&ControlMessage::Warnings { warnings }, || {
                self.ids.reserve()
            })
            .map(Some)
        })
    }

    /// The sampler state the run left, as one `sampler_state` control
    /// message, its memo attached (decision E52); none when it left none.
    /// A host takes it once the run has ended, done or failed, keeps it on
    /// the materializer the run was for, and hands it back with that
    /// materializer's next `materialize`.
    pub fn take_sampler_state(&self) -> Result<Option<Outgoing>, Outgoing> {
        self.enter(|run| match run.take_sampler_state()? {
            Some(state) => write(&ControlMessage::SamplerState { state }, || {
                self.ids.reserve()
            })
            .map(Some),
            None => Ok(None),
        })
    }

    /// Where the run is.
    pub fn state(&self) -> Result<RunState, Outgoing> {
        self.enter(|run| Ok(run.state()))
    }

    /// The zones of the run's host facts: resolved through the host's
    /// lookup roots in order, then the bundled database.
    pub fn zones(&self) -> &ZoneResolver {
        &self.zones
    }

    /// Every zone the run resolved, by its canonical name, with its source.
    pub fn zone_sources(&self) -> BTreeMap<String, ZoneSource> {
        self.zones.sources()
    }
}

/// The codes of the once-a-process warnings this process has given.
static GIVEN_ONCE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether a warning of `code` is to be given: any warning but one given
/// once in a process, and that the first time.
fn first_in_process(code: &str) -> bool {
    if !sdm_core::materialize::ONCE_PER_PROCESS.contains(&code) {
        return true;
    }
    let mut given = GIVEN_ONCE.lock().unwrap_or_else(PoisonError::into_inner);
    if given.iter().any(|g| g == code) {
        return false;
    }
    given.push(code.to_owned());
    true
}

/// Whether `run` awaits fetch `id`.
fn awaits(run: &Run, id: sdm_contract::run::ReplyId) -> bool {
    matches!(run.state(), RunState::AwaitingFetch(ids) if ids.contains(&id))
}
