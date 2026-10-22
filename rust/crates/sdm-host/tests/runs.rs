// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run ownership through the host layer: starting from bytes, panic
//! containment, busy handling, replies with their attachments, admission on
//! the way in, and the attachment-id rules of decision E2.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Barrier};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch, new_null_array};
use arrow_schema::{DataType, IntervalUnit};
use sdm_contract::data::{AttachmentId, DataRequest, ProviderError};
use sdm_contract::data::{Determinism, Dialect, ProviderCapabilities, ProviderChoice};
use sdm_contract::facts::{PlatformFacts, ZoneRootFact};
use sdm_contract::ops::Dtype;
use sdm_contract::ops::{
    Materialize, Materialized, Operation, OperationResult, Predict, ProfileName, ProfileSpec,
    QueryColumn, QuerySchema, QueryTable, QueryValidation, RequestState, SourceDescription, Stype,
    ValidateQuery,
};
use sdm_contract::run::{
    Endpoint, FetchOutcome, FetchReply, ModelOutcome, ModelReply, ReplyId, StartRun,
};
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, CoreError, CoreWarning, Detail, ErrorCode, ErrorKind,
    HostFacts, Scalar, Step, Unit,
};
use sdm_core::{ArrowData, Attachment, FetchResult, ModelResult, Reply, Run, RunState};
use sdm_host::{
    Attachments, HostRun, Outgoing, ZoneResolver, ZoneSource, contain, decode, encode, read,
};

fn facts(zone_roots: Vec<ZoneRootFact>) -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots,
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

fn start_message(operation: Operation, zone_roots: Vec<ZoneRootFact>) -> Vec<u8> {
    encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts(zone_roots),
        operation,
    })))
    .unwrap()
}

fn users() -> QuerySchema {
    QuerySchema {
        tables: vec![QueryTable {
            name: "users".into(),
            columns: vec![
                QueryColumn {
                    name: "user_id".into(),
                    dtype: Dtype::Int,
                    stype: Stype::Id,
                },
                QueryColumn {
                    name: "age".into(),
                    dtype: Dtype::Int,
                    stype: Stype::Numerical,
                },
            ],
            primary_key: Some("user_id".into()),
            time_column: None,
            end_time_column: None,
        }],
        links: vec![],
    }
}

/// A bytes attachment of `value` as JSON.
fn json(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn validate_op() -> Operation {
    Operation::ValidateQuery(ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: None,
            capabilities: BTreeMap::new(),
        },
    })
}

/// The attachments of a `validate_query` start for `query` over `schema`.
fn validate_attachments(query: &str, schema: &QuerySchema) -> Attachments {
    BTreeMap::from([
        (AttachmentId(1), json(&query)),
        (AttachmentId(2), json(schema)),
    ])
}

/// Starts a `validate_query` run for `query` over the users table.
fn start_validate(query: &str, zone_roots: Vec<ZoneRootFact>) -> Result<HostRun, Outgoing> {
    HostRun::start(
        &sdm_host::default_root(),
        &start_message(validate_op(), zone_roots),
        validate_attachments(query, &users()),
    )
}

/// What a test reads a host call's error as: the core error its `error`
/// control message carries.
trait Core<T> {
    fn core(self) -> Result<T, CoreError>;
}

trait IntoCore {
    fn into_core(self) -> CoreError;
}

impl IntoCore for CoreError {
    fn into_core(self) -> CoreError {
        self
    }
}

impl IntoCore for sdm_host::Failure {
    fn into_core(self) -> CoreError {
        self.error
    }
}

impl IntoCore for Outgoing {
    fn into_core(self) -> CoreError {
        match read(&self).unwrap() {
            ControlMessage::Error { error } => error,
            other => panic!("not an error message: {other:?}"),
        }
    }
}

impl<T, E: IntoCore> Core<T> for Result<T, E> {
    fn core(self) -> Result<T, CoreError> {
        self.map_err(IntoCore::into_core)
    }
}

fn no_attachments() -> Attachments {
    BTreeMap::new()
}

fn step(out: &Outgoing) -> Step {
    match read(out).unwrap() {
        ControlMessage::Step(step) => step,
        other => panic!("not a step: {other:?}"),
    }
}

fn host_run(run: Run) -> HostRun {
    HostRun::with(run, Arc::new(ZoneResolver::new([])))
}

fn is_panic(error: &CoreError) -> bool {
    error.fields.get("panic") == Some(&Detail::Scalar(Scalar::Bool(true)))
}

#[test]
fn a_query_validates_from_bytes_to_bytes() {
    for (query, accepted) in [
        ("PREDICT users.age FOR EACH users.user_id", true),
        ("PREDICT users.height FOR EACH users.user_id", false),
    ] {
        let run = start_validate(query, vec![]).unwrap();
        let Step::Done {
            result: OperationResult::ValidateQuery(validation),
        } = step(&run.next().unwrap())
        else {
            panic!("validate_query ends with its result");
        };
        assert_eq!(validation.accepted, accepted, "{query}");
        assert_eq!(run.state().unwrap(), RunState::Finished);
        assert_eq!(run.next().core().unwrap_err().code, ErrorCode::RunFinished);
    }
}

#[test]
fn a_run_starts_only_from_a_start_message_the_core_takes() {
    let root = sdm_host::default_root();
    let cancel = encode(&ControlMessage::Cancel {}).unwrap();
    let err = HostRun::start(&root, &cancel, no_attachments())
        .core()
        .err()
        .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    let err = HostRun::start(&root, b"{", no_attachments())
        .core()
        .err()
        .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    let mut other = decode(&start_message(validate_op(), vec![])).unwrap();
    let ControlMessage::Start(start) = &mut other else {
        unreachable!()
    };
    start.contract = CONTRACT_VERSION + 1;
    let err = HostRun::start(
        &root,
        &encode(&other).unwrap(),
        validate_attachments("PREDICT users.age FOR EACH users.user_id", &users()),
    )
    .core()
    .err()
    .unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Protocol, ErrorCode::InvalidMessage)
    );
}

#[test]
fn a_panic_is_contained_and_poisons_the_run() {
    let run = host_run(Run::new(|_| async { panic!("boom in the body") }));
    let err = run.next().core().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Internal, ErrorCode::InternalError)
    );
    assert!(is_panic(&err));
    // The location, never the panic's message (decision H17).
    let written = serde_json::to_string(&err).unwrap();
    assert!(!written.contains("boom"), "{written}");
    assert!(matches!(
        err.fields.get("location"),
        Some(Detail::Scalar(Scalar::Utf8(at))) if at.contains("runs.rs")
    ));
    assert_eq!(run.state().unwrap(), RunState::Poisoned);
    for err in [
        run.next().core().unwrap_err(),
        run.cancel().core().unwrap_err(),
    ] {
        assert_eq!(err.code, ErrorCode::RunPoisoned);
    }

    // Outside any run, too.
    let failure = contain::<()>(|| panic!("{}", String::from("boom in the host"))).unwrap_err();
    assert!(failure.panicked);
    assert!(is_panic(&failure.error));
    assert!(
        !serde_json::to_string(&failure.error)
            .unwrap()
            .contains("boom")
    );
    // An error is not a panic, whatever its fields say.
    let failure = contain::<()>(|| {
        Err(CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "set by a host",
        )
        .with_field("panic", Detail::Scalar(Scalar::Bool(true))))
    })
    .unwrap_err();
    assert!(!failure.panicked);
}

#[test]
fn a_second_call_while_one_is_in_the_run_is_busy() {
    let inside = Arc::new(Barrier::new(2));
    let (release, wait) = mpsc::channel::<()>();
    let entered = Arc::clone(&inside);
    let run = Arc::new(host_run(Run::new(move |_| async move {
        entered.wait();
        wait.recv().unwrap();
        Ok(OperationResult::ValidateQuery(QueryValidation {
            accepted: true,
            diagnostics: vec![],
        }))
    })));
    let first = {
        let run = Arc::clone(&run);
        std::thread::spawn(move || run.next())
    };
    inside.wait();
    for err in [
        run.next().core().unwrap_err(),
        run.cancel().core().unwrap_err(),
        run.take_attachment(AttachmentId(1)).core().unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::RunBusy)
        );
    }
    release.send(()).unwrap();
    assert!(matches!(
        step(&first.join().unwrap().unwrap()),
        Step::Done { .. }
    ));
}

fn table(column: ArrayRef) -> ArrowData {
    let batch = RecordBatch::try_from_iter([("c", column)]).unwrap();
    ArrowData::new(batch.schema(), vec![batch])
}

/// A run that fetches twice and answers with the type of each column it
/// was handed, through a validation result's diagnostics.
fn fetching_run() -> Run {
    let host = ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            optional: Default::default(),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::Unseeded,
            fallback: None,
        },
    };
    let access = sdm_host::default_root().data_access(&host).unwrap();
    Run::with_data(access, |asker| async move {
        let mut seen = Vec::new();
        for _ in 0..2 {
            let reply = asker
                // A scan, whose answer's columns are the source's own.
                .fetch(DataRequest::Scan {
                    table: "t".into(),
                    columns: vec![],
                    filters: vec![],
                    limit: None,
                })?
                .await;
            seen.push(match reply {
                Reply::Fetch(FetchResult::Data(data)) => {
                    data.schema.field(0).data_type().to_string()
                }
                Reply::Fetch(FetchResult::Error(error)) => {
                    CoreError::provider(&error).code.as_str().to_owned()
                }
                Reply::Model(_) => "model".into(),
                Reply::Fetch(FetchResult::Page { .. }) => "page".into(),
            });
        }
        Ok(OperationResult::ValidateQuery(QueryValidation {
            accepted: true,
            diagnostics: seen
                .into_iter()
                .map(|code| sdm_contract::ops::PqlDiagnostic {
                    code,
                    message: String::new(),
                    severity: sdm_contract::ops::Severity::Warning,
                    span: None,
                })
                .collect(),
        }))
    })
}

fn fetch_reply(id: ReplyId, outcome: FetchOutcome) -> Vec<u8> {
    encode(&ControlMessage::FetchReply(FetchReply { id, outcome })).unwrap()
}

fn fetch_id(out: &Outgoing) -> ReplyId {
    match step(out) {
        Step::Fetch { requests } => requests[0].id,
        other => panic!("not a fetch: {other:?}"),
    }
}

#[test]
fn fetched_data_is_admitted_and_host_ids_are_scoped_to_their_message() {
    let run = host_run(fetching_run());
    let first = fetch_id(&run.next().unwrap());
    let data = Arc::new(LargeStringArray::from(vec!["a"])) as ArrayRef;
    // The host names the data 1 in this message...
    run.reply(
        &fetch_reply(
            first,
            FetchOutcome::Data {
                data: AttachmentId(1),
                continuation: None,
            },
        ),
        BTreeMap::from([(AttachmentId(1), Attachment::Arrow(table(data)))]),
    )
    .unwrap();
    let second = fetch_id(&run.next().unwrap());
    assert_ne!(first, second);
    // ...and may name other data 1 in the next; the first id means nothing
    // now.
    let error = ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: None,
        vendor_code: None,
    };
    run.reply(
        &fetch_reply(second, FetchOutcome::Error { error }),
        no_attachments(),
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::ValidateQuery(validation),
    } = step(&run.next().unwrap())
    else {
        panic!("done");
    };
    let seen: Vec<&str> = validation
        .diagnostics
        .iter()
        .map(|d| d.code.as_str())
        .collect();
    assert_eq!(
        seen,
        [
            DataType::Utf8.to_string().as_str(),
            ErrorCode::ProviderError.as_str()
        ]
    );
}

#[test]
fn a_reply_must_attach_exactly_what_it_names() {
    let run = host_run(fetching_run());
    let id = fetch_id(&run.next().unwrap());
    let data = || Attachment::Arrow(table(Arc::new(LargeStringArray::from(vec!["a"]))));
    let named_one = fetch_reply(
        id,
        FetchOutcome::Data {
            data: AttachmentId(1),
            continuation: None,
        },
    );
    for (attachments, why) in [
        (no_attachments(), "missing"),
        (BTreeMap::from([(AttachmentId(2), data())]), "other id"),
        (
            BTreeMap::from([(AttachmentId(1), data()), (AttachmentId(2), data())]),
            "extra",
        ),
        (
            BTreeMap::from([(AttachmentId(1), Attachment::Bytes(vec![1]))]),
            "bytes, not Arrow",
        ),
    ] {
        let err = run.reply(&named_one, attachments).core().unwrap_err();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::InvalidMessage),
            "{why}"
        );
        assert!(err.fields.contains_key("attachment"), "{why}");
    }
    // None of those reached the run, which still waits.
    assert_eq!(run.state().unwrap(), RunState::AwaitingFetch(vec![id]));
    // A step message is not a reply.
    let err = run
        .reply(
            &encode(&ControlMessage::Cancel {}).unwrap(),
            no_attachments(),
        )
        .core()
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
}

#[test]
fn data_admission_refuses_ends_the_run_with_the_refusal() {
    let run = host_run(fetching_run());
    let id = fetch_id(&run.next().unwrap());
    let half = new_null_array(&DataType::Interval(IntervalUnit::MonthDayNano), 1);
    let err = run
        .reply(
            &fetch_reply(
                id,
                FetchOutcome::Data {
                    data: AttachmentId(9),
                    continuation: None,
                },
            ),
            BTreeMap::from([(AttachmentId(9), Attachment::Arrow(table(half)))]),
        )
        .core()
        .unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedType)
    );
    assert_eq!(run.state().unwrap(), RunState::Finished);
    assert_eq!(run.next().core().unwrap_err().code, ErrorCode::RunFinished);

    // A reply the run does not await gets the run's own answer, before any
    // data is read.
    let other = host_run(fetching_run());
    let id = fetch_id(&other.next().unwrap());
    let half = new_null_array(&DataType::Interval(IntervalUnit::MonthDayNano), 1);
    let err = other
        .reply(
            &fetch_reply(
                ReplyId(id.0 + 7),
                FetchOutcome::Data {
                    data: AttachmentId(1),
                    continuation: None,
                },
            ),
            BTreeMap::from([(AttachmentId(1), Attachment::Arrow(table(half)))]),
        )
        .core()
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply);
}

fn model_run() -> Run {
    Run::new(|asker| async move {
        asker.warn(CoreWarning {
            code: "FIRST".into(),
            message: String::new(),
            fields: BTreeMap::new(),
        });
        let reply = asker
            .call_model(
                Endpoint::Predict,
                "application/json".into(),
                b"{\"q\":1}".to_vec(),
            )
            .await;
        asker.warn(CoreWarning {
            code: "SECOND".into(),
            message: String::new(),
            fields: BTreeMap::new(),
        });
        let Reply::Model(ModelResult::Response { body, .. }) = reply else {
            return Err(CoreError::new(
                ErrorKind::ModelEndpoint,
                ErrorCode::TransportError,
                "no",
            ));
        };
        Ok(OperationResult::ValidateQuery(QueryValidation {
            accepted: body == b"answer",
            diagnostics: vec![],
        }))
    })
}

#[test]
fn a_model_call_s_body_is_the_run_s_attachment_1_and_its_answer_the_host_s() {
    let run = host_run(model_run());
    let Step::CallModel(call) = step(&run.next().unwrap()) else {
        panic!("a model call");
    };
    // The core numbers its attachments from 1 in each run (E2).
    assert_eq!(call.body, AttachmentId(1));
    let Some(Attachment::Bytes(body)) = run.take_attachment(call.body).unwrap() else {
        panic!("the body is bytes");
    };
    assert_eq!(body, b"{\"q\":1}");
    assert!(
        run.take_attachment(call.body).unwrap().is_none(),
        "taken once"
    );
    // The host names the answer's body with an id of its own: the same
    // number, in its own message's scope.
    let reply = encode(&ControlMessage::ModelReply(ModelReply {
        id: call.id,
        outcome: ModelOutcome::Response {
            status: 200,
            content_type: "application/json".into(),
            body: AttachmentId(1),
        },
    }))
    .unwrap();
    let arrow = Attachment::Arrow(table(Arc::new(LargeStringArray::from(vec!["a"]))));
    let err = run
        .reply(&reply, BTreeMap::from([(AttachmentId(1), arrow)]))
        .core()
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    run.reply(
        &reply,
        BTreeMap::from([(AttachmentId(1), Attachment::Bytes(b"answer".to_vec()))]),
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::ValidateQuery(validation),
    } = step(&run.next().unwrap())
    else {
        panic!("done");
    };
    assert!(validation.accepted);
    // Both warnings, in order, in one message.
    let message = run.take_warnings().unwrap().expect("warnings");
    let ControlMessage::Warnings { warnings } = read(&message).unwrap() else {
        panic!("a warnings message");
    };
    let codes: Vec<&str> = warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["FIRST", "SECOND"]);
    assert!(run.take_warnings().unwrap().is_none());
}

#[test]
fn cancelling_releases_the_run_s_attachments() {
    let run = host_run(model_run());
    let Step::CallModel(call) = step(&run.next().unwrap()) else {
        panic!("a model call");
    };
    run.cancel().unwrap();
    assert!(run.take_attachment(call.body).unwrap().is_none());
    assert_eq!(run.next().core().unwrap_err().code, ErrorCode::RunCancelled);
}

#[test]
fn request_state_leaves_one_run_as_bytes_and_returns_under_the_host_s_id() {
    // A run finishes with request state whose payload and query it attached.
    let run = host_run(Run::new(|asker| async move {
        let payload = asker.attach(Attachment::Bytes(b"payload".to_vec()))?;
        let query = asker.attach(json(&"PREDICT users.age FOR EACH users.user_id"))?;
        Ok(OperationResult::Materialize(Materialized {
            state: RequestState {
                version: sdm_contract::ops::REQUEST_STATE_VERSION,
                payload,
                query,
                forecast: None,
                batches: None,
                ranking: None,
            },
            query: None,
        }))
    }));
    let Step::Done {
        result: OperationResult::Materialize(materialized),
    } = step(&run.next().unwrap())
    else {
        panic!("materialize ends with its state");
    };
    // The host takes the payload and query after the run finished.
    let payload = run.take_attachment(materialized.state.payload).unwrap();
    assert!(matches!(payload, Some(Attachment::Bytes(ref b)) if b == b"payload"));
    let query = run.take_attachment(materialized.state.query).unwrap();
    drop(run);

    // They come back with the predict run's start, under ids the host picks
    // for that message; the start must attach exactly what it names.
    let mut state = materialized.state;
    state.payload = AttachmentId(7);
    state.query = AttachmentId(1);
    let predict = start_message(
        Operation::Predict(Predict {
            state,
            model: "kumo-relational".into(),
            request_id: "sdm-000000000000".into(),
            outputs: vec!["prediction".into()],
            default_outputs: false,
            inference_asked: vec![],
        }),
        vec![],
    );
    let root = sdm_host::default_root();
    let err = HostRun::start(
        &root,
        &predict,
        BTreeMap::from([
            (AttachmentId(8), payload.clone().unwrap()),
            (AttachmentId(1), query.clone().unwrap()),
        ]),
    )
    .core()
    .err()
    .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    // Named and attached, the run starts, and reads the payload it was
    // handed back, which here is not JSON: refused by name, never quoted.
    let run = HostRun::start(
        &root,
        &predict,
        BTreeMap::from([
            (AttachmentId(7), payload.unwrap()),
            (AttachmentId(1), query.unwrap()),
        ]),
    )
    .core()
    .unwrap();
    let err = run.next().core().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
    );
    assert_eq!(err.fields["field"], Detail::text("state.payload"));
    // An index set a materialize names is checked the same way.
    let materialize = start_message(
        Operation::Materialize(Materialize {
            provider: ProviderChoice::Sql {
                dialect: Dialect::new("duckdb"),
                namespace: Vec::new(),
            },
            schema: AttachmentId(1),
            held: None,
            query: AttachmentId(2),
            indices: Some(AttachmentId(3)),
            anchor_time: None,
            seed: None,
            unseeded: false,
            anchor_entity: false,
            values: None,
            options: vec![],
            most_parts: None,
            most_entities: None,
            most_calls: None,
            candidates: None,
            sampler_state: None,
        }),
        vec![],
    );
    let err = HostRun::start(
        &root,
        &materialize,
        BTreeMap::from([
            (
                AttachmentId(1),
                json(&SourceDescription {
                    tables: vec![],
                    links: vec![],
                }),
            ),
            (AttachmentId(2), json(&"")),
        ]),
    )
    .core()
    .err()
    .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    assert_eq!(err.fields["attachment"], Detail::Scalar(Scalar::UInt32(3)));
}

#[test]
fn a_run_resolves_zones_through_its_host_s_roots() {
    let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tzdata-2026c");
    let run = start_validate(
        "PREDICT users.age FOR EACH users.user_id",
        vec![ZoneRootFact {
            path: vendored.to_string_lossy().into_owned(),
            version: Some("2026c".into()),
        }],
    )
    .core()
    .unwrap();
    run.zones().resolve("europe/berlin").unwrap();
    run.zones().resolve("UTC").unwrap();
    let sources = run.zone_sources();
    assert!(matches!(
        &sources["Europe/Berlin"],
        ZoneSource::HostRoot { index: 0, version: Some(v), .. } if v == "2026c"
    ));
    assert_eq!(sources["UTC"], ZoneSource::Fixed);
}

#[test]
fn a_host_run_can_be_shared_between_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<HostRun>();
}
