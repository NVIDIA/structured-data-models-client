// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run ownership, panic containment and busy handling: what every binding
//! calls.
//!
//! A [`HostRun`] owns one core run. A binding hands it control messages as
//! bytes, with the attachments each names, and takes control messages back
//! as bytes; it may share the run between threads. Every call:
//!
//! - is contained: a panic anywhere below it poisons the run and comes back
//!   as `INTERNAL_ERROR` with `fields.panic = true`, never as an unwind
//!   through the binding;
//! - is refused with `RUN_BUSY` (kind `protocol`) while another call is in
//!   the run, rather than waiting;
//! - reads and writes control messages through the codecs, under the
//!   control cap.
//!
//! Attachments follow decision E2. The core numbers its own from 1 in each
//! run, and the host takes them by id ([`HostRun::take_attachment`]). The
//! host names the attachments of each message it sends with ids of its
//! choosing, scoped to that message: every id the message names must be
//! attached and every attachment named, and the ids mean nothing once the
//! message is read. Fetched Arrow data passes through admission on the way
//! in; data admission refuses ends the run with that refusal.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, TryLockError};

use sdm_contract::data::AttachmentId;
use sdm_contract::ops::Operation;
use sdm_contract::run::{FetchOutcome, ModelOutcome, StartRun};
use sdm_contract::{ControlMessage, CoreError, Detail, ErrorKind, Scalar, codes};
use sdm_core::{Attachment, CompositionRoot, FetchResult, ModelResult, Run, RunState};

use crate::admission::admit;
use crate::codec::{decode, encode, encode_warnings};
use crate::zones::{ZoneResolver, ZoneRoot, ZoneSource};

/// The attachments a host sends with one control message, by the ids that
/// message names them with.
pub type Attachments = BTreeMap<AttachmentId, Attachment>;

/// One core run, owned by the host layer; see the module documentation.
pub struct HostRun {
    run: Mutex<Run>,
    zones: ZoneResolver,
}

/// Runs `call`, turning a panic into `INTERNAL_ERROR` with
/// `fields.panic = true`.
pub fn contain<T>(call: impl FnOnce() -> Result<T, CoreError>) -> Result<T, CoreError> {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or_else(|payload| {
        let what = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic without a message".to_owned());
        Err(CoreError::new(
            ErrorKind::Internal,
            codes::INTERNAL_ERROR,
            format!("the core panicked: {what}"),
        )
        .with_field("panic", Detail::Scalar(Scalar::Bool(true))))
    })
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::protocol(codes::INVALID_MESSAGE, message)
}

fn attachment_field(error: CoreError, id: AttachmentId) -> CoreError {
    error.with_field("attachment", Detail::Scalar(Scalar::UInt32(id.0)))
}

/// Checks that `attachments` holds exactly the ids `named`.
fn check_named(named: &BTreeSet<AttachmentId>, attachments: &Attachments) -> Result<(), CoreError> {
    if let Some(id) = named.iter().find(|id| !attachments.contains_key(id)) {
        return Err(attachment_field(
            invalid(format!(
                "the message names attachment {} and does not attach it",
                id.0
            )),
            *id,
        ));
    }
    if let Some(id) = attachments.keys().find(|id| !named.contains(id)) {
        return Err(attachment_field(
            invalid(format!("attachment {} is not named by its message", id.0)),
            *id,
        ));
    }
    Ok(())
}

/// The attachment ids a `start` names.
fn named_by_start(start: &StartRun) -> BTreeSet<AttachmentId> {
    match &start.operation {
        Operation::Materialize(m) => m.indices.into_iter().collect(),
        Operation::Predict(p) => BTreeSet::from([p.state.payload]),
        Operation::ValidateQuery(_) | Operation::DescribeSource(_) => BTreeSet::new(),
    }
}

impl HostRun {
    /// Starts the run a `start` message asks for, with the attachments it
    /// names.
    pub fn start(
        root: &CompositionRoot,
        message: &[u8],
        attachments: Attachments,
    ) -> Result<HostRun, CoreError> {
        contain(|| {
            let ControlMessage::Start(start) = decode(message)? else {
                return Err(invalid("a run starts with a start message"));
            };
            check_named(&named_by_start(&start), &attachments)?;
            if let Some(id) = attachments.keys().next() {
                // The predict route (T1.11) is the first to take state from
                // the host; until then no operation takes an attachment.
                return Err(attachment_field(
                    CoreError::new(
                        ErrorKind::Unsupported,
                        codes::UNSUPPORTED_REQUEST,
                        "this core takes no attachments with a start message yet",
                    ),
                    *id,
                ));
            }
            let zones = ZoneResolver::new(start.facts.zone_roots.iter().map(|root| ZoneRoot {
                path: root.path.clone().into(),
                version: root.version.clone(),
            }));
            let run = sdm_core::start(root, *start)?;
            Ok(HostRun::with(run, zones))
        })
    }

    /// The host run of a core run built directly, as tests and the
    /// pure-Rust test host build one.
    pub fn with(run: Run, zones: ZoneResolver) -> HostRun {
        HostRun {
            run: Mutex::new(run),
            zones,
        }
    }

    /// Runs `call` on the run: refused with `RUN_BUSY` while another call is
    /// in it, and contained, a panic poisoning the run.
    fn enter<T>(
        &self,
        call: impl FnOnce(&mut Run) -> Result<T, CoreError>,
    ) -> Result<T, CoreError> {
        let mut guard: MutexGuard<'_, Run> = match self.run.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => {
                return Err(CoreError::protocol(
                    codes::RUN_BUSY,
                    "another call is in this run; a run takes one call at a time",
                ));
            }
            // Only a contained panic leaves the lock poisoned, and the run
            // was poisoned with it.
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        let result = contain(|| call(&mut guard));
        if result
            .as_ref()
            .is_err_and(|e| e.fields.get("panic") == Some(&Detail::Scalar(Scalar::Bool(true))))
        {
            guard.poison();
        }
        result
    }

    /// The run's next step, as a `step` control message.
    pub fn next(&self) -> Result<Vec<u8>, CoreError> {
        self.enter(|run| encode(&ControlMessage::Step(run.next()?)))
    }

    /// Hands the run a `fetch_reply` or `model_reply`, with the attachments
    /// it names. Fetched data is admitted first; data admission refuses ends
    /// the run with that refusal.
    pub fn reply(&self, message: &[u8], attachments: Attachments) -> Result<(), CoreError> {
        self.enter(|run| {
            let mut attachments = attachments;
            match decode(message)? {
                ControlMessage::FetchReply(reply) => {
                    let result = match reply.outcome {
                        FetchOutcome::Data { data } => {
                            check_named(&BTreeSet::from([data]), &attachments)?;
                            let Some(Attachment::Arrow(arrow)) = attachments.remove(&data) else {
                                return Err(attachment_field(
                                    invalid("fetched data is an Arrow attachment, not bytes"),
                                    data,
                                ));
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
                            check_named(&BTreeSet::new(), &attachments)?;
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
                            check_named(&BTreeSet::from([body]), &attachments)?;
                            let Some(Attachment::Bytes(body_bytes)) = attachments.remove(&body)
                            else {
                                return Err(attachment_field(
                                    invalid("a model response's body is bytes, not Arrow data"),
                                    body,
                                ));
                            };
                            ModelResult::Response {
                                status,
                                content_type,
                                body: body_bytes,
                            }
                        }
                        ModelOutcome::TransportError {
                            kind,
                            retryable,
                            message,
                        } => {
                            check_named(&BTreeSet::new(), &attachments)?;
                            ModelResult::TransportError {
                                kind,
                                retryable,
                                message,
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
    pub fn cancel(&self) -> Result<(), CoreError> {
        self.enter(Run::cancel)
    }

    /// Takes an attachment the run made and a step named.
    pub fn take_attachment(&self, id: AttachmentId) -> Result<Option<Attachment>, CoreError> {
        self.enter(|run| Ok(run.take_attachment(id)))
    }

    /// The warnings given since the last call, as `warnings` control
    /// messages, in order.
    pub fn take_warnings(&self) -> Result<Vec<Vec<u8>>, CoreError> {
        self.enter(|run| encode_warnings(run.take_warnings()))
    }

    /// Where the run is.
    pub fn state(&self) -> Result<RunState, CoreError> {
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
