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
use std::sync::{Mutex, MutexGuard, Once, TryLockError};

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::LocalZone;
use sdm_contract::ops::AttachmentKind;
use sdm_contract::run::{FetchOutcome, ModelOutcome};
use sdm_contract::{ControlMessage, CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_core::attached::check_attachments;
use sdm_core::{
    Attachment, AttachmentIds, Attachments, CompositionRoot, FetchResult, ModelResult, Run,
    RunState,
};

use crate::admission::admit;
use crate::codec::{Outgoing, decode, write, write_error};
use crate::zones::{ZoneResolver, ZoneRoot, ZoneSource};

/// One core run, owned by the host layer; see the module documentation.
pub struct HostRun {
    run: Mutex<Run>,
    ids: AttachmentIds,
    zones: ZoneResolver,
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
                    ErrorKind::Internal,
                    ErrorCode::InternalError,
                    "the core panicked; the run can no longer be used",
                )
                .with_field("panic", Detail::Scalar(Scalar::Bool(true)))
                .with_field("location", Detail::text(at)),
                panicked: true,
            })
        }
    }
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::protocol(ErrorCode::InvalidMessage, message)
}

/// The error of a `start` that made no run, as a control message; there is
/// no run to number its attachment, so it is the message's own first.
fn start_error(error: &CoreError) -> Outgoing {
    write_error(error, || Ok(AttachmentId(1)))
}

/// Refuses a local zone the host's facts name that cannot be read: a POSIX
/// `TZ` string must parse. An IANA name is resolved when a run first needs
/// it.
fn check_local_zone(zone: Option<&LocalZone>) -> Result<(), CoreError> {
    if let Some(LocalZone::Posix(text)) = zone
        && jiff::tz::TimeZone::posix(text).is_err()
    {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
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
        contain(|| {
            let ControlMessage::Start(start) = decode(message)? else {
                return Err(invalid("a run starts with a start message"));
            };
            // The version first: no other field means anything under
            // another contract (H25). The core checks the attachments.
            sdm_core::dispatch::check_contract(start.contract)?;
            check_local_zone(start.facts.local_zone.as_ref())?;
            let zones = ZoneResolver::new(start.facts.zone_roots.iter().map(|root| ZoneRoot {
                path: root.path.clone().into(),
                version: root.version.clone(),
            }));
            let run = sdm_core::start(root, *start, attachments)?;
            Ok(HostRun::with(run, zones))
        })
        .map_err(|failure| start_error(&failure.error))
    }

    /// The host run of a core run built directly, as tests and the
    /// pure-Rust test host build one.
    pub fn with(run: Run, zones: ZoneResolver) -> HostRun {
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
                    ErrorCode::RunBusy,
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
        self.enter(|run| {
            let mut attachments = attachments;
            match decode(message)? {
                ControlMessage::FetchReply(reply) => {
                    let result = match reply.outcome {
                        FetchOutcome::Data { data } => {
                            check_attachments(&[(data, AttachmentKind::Arrow)], &attachments)?;
                            let Some(Attachment::Arrow(arrow)) = attachments.remove(&data) else {
                                return Err(invalid("fetched data is an Arrow attachment"));
                            };
                            if !awaits(run, reply.id) {
                                // The run's own refusal of a reply it does not
                                // await: finished, stale, or not a fetch.
                                return run.submit_fetch(reply.id, FetchResult::Data(arrow));
                            }
                            match admit(&arrow) {
                                Ok(admitted) => FetchResult::Data(admitted),
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
    /// control message, whole; none when there are none.
    pub fn take_warnings(&self) -> Result<Option<Outgoing>, Outgoing> {
        self.enter(|run| {
            let warnings = run.take_warnings();
            if warnings.is_empty() {
                return Ok(None);
            }
            write(&ControlMessage::Warnings { warnings }, || {
                self.ids.reserve()
            })
            .map(Some)
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

/// Whether `run` awaits fetch `id`.
fn awaits(run: &Run, id: sdm_contract::run::ReplyId) -> bool {
    matches!(run.state(), RunState::AwaitingFetch(ids) if ids.contains(&id))
}
