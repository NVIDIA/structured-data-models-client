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
use arrow_schema::DataType;
use sdm_contract::data::{AttachmentId, DataRequest, ExpectedSchema, ProviderError};
use sdm_contract::data::{Dialect, ProviderChoice};
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
    CONTRACT_VERSION, ControlMessage, CoreError, CoreWarning, Detail, ErrorKind, HostFacts, Scalar,
    Step, Unit, codes,
};
use sdm_core::{
    ArrowData, Attachment, CompositionRoot, FetchResult, ModelResult, Reply, Run, RunState,
};
use sdm_host::{Attachments, HostRun, ZoneResolver, ZoneSource, contain, decode, encode};

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

fn validate(query: &str) -> Operation {
    Operation::ValidateQuery(ValidateQuery {
        query: query.into(),
        schema: users(),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: None,
            capabilities: BTreeMap::new(),
        },
    })
}

fn no_attachments() -> Attachments {
    BTreeMap::new()
}

fn step(bytes: &[u8]) -> Step {
    match decode(bytes).unwrap() {
        ControlMessage::Step(step) => step,
        other => panic!("not a step: {other:?}"),
    }
}

fn host_run(run: Run) -> HostRun {
    HostRun::with(run, ZoneResolver::new([]))
}

fn is_panic(error: &CoreError) -> bool {
    error.fields.get("panic") == Some(&Detail::Scalar(Scalar::Bool(true)))
}

#[test]
fn a_query_validates_from_bytes_to_bytes() {
    let root = CompositionRoot::default();
    for (query, accepted) in [
        ("PREDICT users.age FOR EACH users.user_id", true),
        ("PREDICT users.height FOR EACH users.user_id", false),
    ] {
        let run = HostRun::start(
            &root,
            &start_message(validate(query), vec![]),
            no_attachments(),
        )
        .unwrap();
        let Step::Done {
            result: OperationResult::ValidateQuery(validation),
        } = step(&run.next().unwrap())
        else {
            panic!("validate_query ends with its result");
        };
        assert_eq!(validation.accepted, accepted, "{query}");
        assert_eq!(run.state().unwrap(), RunState::Finished);
        assert_eq!(run.next().unwrap_err().code, codes::RUN_FINISHED);
    }
}

#[test]
fn a_run_starts_only_from_a_start_message_the_core_takes() {
    let root = CompositionRoot::default();
    let cancel = encode(&ControlMessage::Cancel {}).unwrap();
    let err = HostRun::start(&root, &cancel, no_attachments())
        .err()
        .unwrap();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
    let err = HostRun::start(&root, b"{", no_attachments()).err().unwrap();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
    let mut other = decode(&start_message(
        validate("PREDICT users.age FOR EACH users.user_id"),
        vec![],
    ))
    .unwrap();
    let ControlMessage::Start(start) = &mut other else {
        unreachable!()
    };
    start.contract = CONTRACT_VERSION + 1;
    let err = HostRun::start(&root, &encode(&other).unwrap(), no_attachments())
        .err()
        .unwrap();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Protocol, codes::INVALID_MESSAGE)
    );
}

#[test]
fn a_panic_is_contained_and_poisons_the_run() {
    let run = host_run(Run::new(|_| async { panic!("boom in the body") }));
    let err = run.next().unwrap_err();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Internal, codes::INTERNAL_ERROR)
    );
    assert!(is_panic(&err));
    assert!(err.message.contains("boom in the body"));
    assert_eq!(run.state().unwrap(), RunState::Poisoned);
    for err in [run.next().unwrap_err(), run.cancel().unwrap_err()] {
        assert_eq!(err.code, codes::RUN_POISONED);
    }

    // Outside any run, too.
    let err = contain::<()>(|| panic!("{}", String::from("boom in the host"))).unwrap_err();
    assert!(is_panic(&err));
    assert!(err.message.contains("boom in the host"));
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
        run.next().unwrap_err(),
        run.cancel().unwrap_err(),
        run.take_attachment(AttachmentId(1)).unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code.as_str()),
            (ErrorKind::Protocol, codes::RUN_BUSY)
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
    ArrowData {
        schema: batch.schema(),
        batches: vec![batch],
    }
}

/// A run that fetches twice and answers with the type of each column it
/// was handed, through a validation result's diagnostics.
fn fetching_run() -> Run {
    Run::new(|asker| async move {
        let mut seen = Vec::new();
        for _ in 0..2 {
            let reply = asker
                .fetch(
                    DataRequest::Describe { tables: vec![] },
                    None,
                    ExpectedSchema::Discover {},
                )
                .await;
            seen.push(match reply {
                Reply::Fetch(FetchResult::Data(data)) => {
                    data.schema.field(0).data_type().to_string()
                }
                Reply::Fetch(FetchResult::Error(error)) => error.code,
                Reply::Model(_) => "model".into(),
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

fn fetch_id(bytes: &[u8]) -> ReplyId {
    match step(bytes) {
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
        code: codes::PROVIDER_ERROR.into(),
        message: "gone".into(),
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
        [DataType::Utf8.to_string().as_str(), codes::PROVIDER_ERROR]
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
        let err = run.reply(&named_one, attachments).unwrap_err();
        assert_eq!(
            (err.kind, err.code.as_str()),
            (ErrorKind::Protocol, codes::INVALID_MESSAGE),
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
        .unwrap_err();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
}

#[test]
fn data_admission_refuses_ends_the_run_with_the_refusal() {
    let run = host_run(fetching_run());
    let id = fetch_id(&run.next().unwrap());
    let half = new_null_array(&DataType::Float16, 1);
    let err = run
        .reply(
            &fetch_reply(
                id,
                FetchOutcome::Data {
                    data: AttachmentId(9),
                },
            ),
            BTreeMap::from([(AttachmentId(9), Attachment::Arrow(table(half)))]),
        )
        .unwrap_err();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Unsupported, codes::UNSUPPORTED_TYPE)
    );
    assert_eq!(run.state().unwrap(), RunState::Finished);
    assert_eq!(run.next().unwrap_err().code, codes::RUN_FINISHED);

    // A reply the run does not await gets the run's own answer, before any
    // data is read.
    let other = host_run(fetching_run());
    let id = fetch_id(&other.next().unwrap());
    let half = new_null_array(&DataType::Float16, 1);
    let err = other
        .reply(
            &fetch_reply(
                ReplyId(id.0 + 7),
                FetchOutcome::Data {
                    data: AttachmentId(1),
                },
            ),
            BTreeMap::from([(AttachmentId(1), Attachment::Arrow(table(half)))]),
        )
        .unwrap_err();
    assert_eq!(err.code, codes::UNKNOWN_REPLY);
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
                codes::TRANSPORT_ERROR,
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
        .unwrap_err();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
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
    let messages = run.take_warnings().unwrap();
    assert_eq!(messages.len(), 1);
    let ControlMessage::Warnings { warnings } = decode(&messages[0]).unwrap() else {
        panic!("a warnings message");
    };
    let codes: Vec<&str> = warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["FIRST", "SECOND"]);
    assert!(run.take_warnings().unwrap().is_empty());
}

#[test]
fn cancelling_releases_the_run_s_attachments() {
    let run = host_run(model_run());
    let Step::CallModel(call) = step(&run.next().unwrap()) else {
        panic!("a model call");
    };
    run.cancel().unwrap();
    assert!(run.take_attachment(call.body).unwrap().is_none());
    assert_eq!(run.next().unwrap_err().code, codes::RUN_CANCELLED);
}

#[test]
fn request_state_leaves_one_run_as_bytes_and_returns_under_the_host_s_id() {
    // A run finishes with request state whose payload it attached.
    let run = host_run(Run::new(|asker| async move {
        let payload = asker.attach(Attachment::Bytes(b"payload".to_vec()))?;
        Ok(OperationResult::Materialize(Materialized {
            state: RequestState {
                version: 1,
                payload,
                query: "PREDICT users.age FOR EACH users.user_id".into(),
            },
        }))
    }));
    let Step::Done {
        result: OperationResult::Materialize(materialized),
    } = step(&run.next().unwrap())
    else {
        panic!("materialize ends with its state");
    };
    // The host takes the payload after the run finished.
    let payload = run.take_attachment(materialized.state.payload).unwrap();
    assert!(matches!(payload, Some(Attachment::Bytes(ref b)) if b == b"payload"));
    drop(run);

    // It comes back with the predict run's start, under an id the host picks
    // for that message; the start must attach exactly what it names.
    let mut state = materialized.state;
    state.payload = AttachmentId(7);
    let predict = start_message(
        Operation::Predict(Predict {
            state,
            model: "relational".into(),
        }),
        vec![],
    );
    let root = CompositionRoot::default();
    let err = HostRun::start(
        &root,
        &predict,
        BTreeMap::from([(AttachmentId(8), payload.clone().unwrap())]),
    )
    .err()
    .unwrap();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
    // Named and attached; the core takes start attachments from T1.11.
    let err = HostRun::start(
        &root,
        &predict,
        BTreeMap::from([(AttachmentId(7), payload.unwrap())]),
    )
    .err()
    .unwrap();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Unsupported, codes::UNSUPPORTED_REQUEST)
    );
    // An index set a materialize names is checked the same way.
    let materialize = start_message(
        Operation::Materialize(Materialize {
            provider: ProviderChoice::Sql {
                dialect: Dialect::Duckdb,
            },
            schema: SourceDescription {
                tables: vec![],
                links: vec![],
            },
            query: String::new(),
            indices: Some(AttachmentId(3)),
            anchor_time: None,
            seed: None,
        }),
        vec![],
    );
    let err = HostRun::start(&root, &materialize, no_attachments())
        .err()
        .unwrap();
    assert_eq!(err.code, codes::INVALID_MESSAGE);
}

#[test]
fn a_run_resolves_zones_through_its_host_s_roots() {
    let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tzdata-2026c");
    let root = CompositionRoot::default();
    let run = HostRun::start(
        &root,
        &start_message(
            validate("PREDICT users.age FOR EACH users.user_id"),
            vec![ZoneRootFact {
                path: vendored.to_string_lossy().into_owned(),
                version: Some("2026c".into()),
            }],
        ),
        no_attachments(),
    )
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
