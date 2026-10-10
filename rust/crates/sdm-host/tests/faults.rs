// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Faults through `HostRun` (T1.7c): fetched data that does not match its
//! declared schema (check (a)) or the schema its request expects back
//! (check (b)) is refused with `PROVIDER_SCHEMA_MISMATCH` and never
//! admitted (decision H16); a test-only operation that panics while the run
//! starts, in its body, or on the PQL adapter's grown stack is contained,
//! and only a panic poisons a run (H17); and no input value, planted as a
//! canary in a malformed control message, a malformed attachment or a panic
//! payload, reaches an error's message, fields or cause, or stderr, which
//! the last test captures from a child process.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reruns its own binary for the abort test, and Arrow's field metadata is a HashMap"
)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Int32Array, Int64Array, RecordBatch, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use sdm_contract::data::{AttachmentId, DataRequest, ExpectedSchema, FetchPlan};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    Operation, Predict, ProfileName, ProfileSpec, QuerySchema, REQUEST_STATE_VERSION, RequestState,
    ValidateQuery,
};
use sdm_contract::run::{FetchOutcome, FetchReply, ModelOutcome, ModelReply, ReplyId, StartRun};
use sdm_contract::test_ops::{PanicAt, TestFetch, TestOperation};
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, Detail, ErrorCode, ErrorKind,
    Field as ContractField, HostFacts, Scalar, Unit,
};
use sdm_core::{ArrowData, Attachment, Attachments, RunState};
use sdm_host::{HostRun, Outgoing, Unreadable, Unreadables, encode, read};

fn start_message(operation: Operation) -> Vec<u8> {
    encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("UTC".into()),
                value: 1_767_225_600_000_000,
            },
            local_zone: None,
            zone_roots: vec![],
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
        },
        operation,
    })))
    .unwrap()
}

fn json(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn error(out: Outgoing) -> CoreError {
    match read(&out).unwrap() {
        ControlMessage::Error { error } => error,
        other => panic!("not an error: {other:?}"),
    }
}

fn text(error: &CoreError, field: &str) -> String {
    match error.fields.get(field) {
        Some(Detail::Scalar(Scalar::Utf8(s))) => s.clone(),
        other => panic!("{field} is {other:?} in {error:?}"),
    }
}

/// A run of the test-only fetch, waiting on its one fetch.
fn fetching(expect: ExpectedSchema) -> HostRun {
    let fetch = TestFetch {
        echo: false,
        request: DataRequest::Scan {
            table: "t".into(),
            columns: vec![],
            filters: vec![],
            limit: None,
        },
        plan: Some(FetchPlan::Host { expect }),
        provider: None,
    };
    let run = HostRun::start(
        &sdm_host::default_root(),
        &start_message(Operation::Test(TestOperation::Fetch {
            fetch: AttachmentId(1),
        })),
        Attachments::from([(AttachmentId(1), json(&fetch))]),
    )
    .map_err(error)
    .unwrap();
    run.next().map_err(error).unwrap();
    run
}

fn reply(run: &HostRun, data: ArrowData) -> Result<(), CoreError> {
    run.reply(
        &encode(&ControlMessage::FetchReply(FetchReply {
            id: ReplyId(1),
            outcome: FetchOutcome::Data {
                data: AttachmentId(1),
            },
        }))
        .unwrap(),
        Attachments::from([(AttachmentId(1), Attachment::Arrow(data))]),
    )
    .map_err(error)
}

/// Checks that `data` is refused by `check`, and that the run ends without
/// it.
fn refused_by(check: &str, expect: ExpectedSchema, data: ArrowData, fault: &str) {
    let run = fetching(expect);
    let err = reply(&run, data).expect_err(fault);
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::DataSource, ErrorCode::ProviderSchemaMismatch),
        "{fault}"
    );
    assert_eq!(text(&err, "check"), check, "{fault}");
    assert_eq!(run.state().unwrap(), RunState::Finished, "{fault}");
    assert_eq!(
        error(run.next().unwrap_err()).code,
        ErrorCode::RunFinished,
        "{fault}: never admitted"
    );
}

fn ints(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn schema(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(schema(fields), columns).unwrap()
}

fn int(name: &str) -> Field {
    Field::new(name, DataType::Int64, true)
}

#[test]
fn a_batch_that_does_not_match_its_declared_schema_is_refused_by_check_a() {
    let discover = || ExpectedSchema::Discover {};
    let declared = schema(vec![int("a"), int("b")]);
    // An extra column.
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            Arc::clone(&declared),
            vec![batch(
                vec![int("a"), int("b"), int("c")],
                vec![ints(&[1]), ints(&[2]), ints(&[3])],
            )],
        ),
        "an extra column",
    );
    // Two same-typed columns swapped.
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            Arc::clone(&declared),
            vec![batch(
                vec![int("b"), int("a")],
                vec![ints(&[2]), ints(&[1])],
            )],
        ),
        "swapped columns",
    );
    // Another decimal scale.
    let decimal = |scale| Field::new("d", DataType::Decimal128(10, scale), true);
    let decimals = |scale| -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![12_345])
                .with_precision_and_scale(10, scale)
                .unwrap(),
        )
    };
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            schema(vec![decimal(2)]),
            vec![batch(vec![decimal(3)], vec![decimals(3)])],
        ),
        "a decimal scale",
    );
    // Another timestamp unit.
    let micros = Field::new("t", DataType::Timestamp(TimeUnit::Microsecond, None), true);
    let nanos = Field::new("t", DataType::Timestamp(TimeUnit::Nanosecond, None), true);
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            schema(vec![micros.clone()]),
            vec![batch(
                vec![nanos],
                vec![Arc::new(TimestampNanosecondArray::from(vec![1]))],
            )],
        ),
        "a timestamp unit",
    );
    // Other field metadata.
    let tagged = |value: &str| {
        int("a").with_metadata(HashMap::from([("role".to_owned(), value.to_owned())]))
    };
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            schema(vec![tagged("key")]),
            vec![batch(vec![tagged("time")], vec![ints(&[1])])],
        ),
        "metadata",
    );
    // Nulls in a column its attachment declares without them.
    let nullable = batch(vec![int("a")], vec![Arc::new(Int64Array::from(vec![None]))]);
    refused_by(
        "declared",
        discover(),
        ArrowData::new(
            schema(vec![Field::new("a", DataType::Int64, false)]),
            vec![nullable],
        ),
        "nulls in a non-nullable column",
    );
    // Another schema in a later batch.
    let good = batch(
        vec![micros.clone()],
        vec![Arc::new(TimestampMicrosecondArray::from(vec![1]))],
    );
    let later = batch(
        vec![Field::new("t", DataType::Int32, true)],
        vec![Arc::new(Int32Array::from(vec![1]))],
    );
    refused_by(
        "declared",
        discover(),
        ArrowData::new(schema(vec![micros]), vec![good, later]),
        "a later batch",
    );
}

fn expected(names: &[&str]) -> ExpectedSchema {
    ExpectedSchema::Schema {
        fields: names
            .iter()
            .map(|name| ContractField {
                name: (*name).to_owned(),
                column_type: ColumnType::Int64,
                nullable: true,
                metadata: BTreeMap::new(),
            })
            .collect(),
    }
}

fn consistent(names: &[&str]) -> ArrowData {
    let fields: Vec<Field> = names.iter().map(|n| int(n)).collect();
    let columns = names.iter().map(|_| ints(&[1, 2])).collect();
    let batch = batch(fields, columns);
    ArrowData::new(batch.schema(), vec![batch])
}

#[test]
fn data_that_does_not_match_its_request_is_refused_by_check_b() {
    for (names, fault) in [
        (&["a"][..], "lacks a column"),
        (&["a", "b", "c"][..], "adds a column"),
        (&["a", "z"][..], "renames a column"),
        (&["b", "a"][..], "orders the columns otherwise"),
    ] {
        refused_by("expected", expected(&["a", "b"]), consistent(names), fault);
    }
    // A column of another type than the one asked for.
    let other = batch(
        vec![Field::new("a", DataType::Int32, true)],
        vec![Arc::new(Int32Array::from(vec![1]))],
    );
    refused_by(
        "expected",
        expected(&["a"]),
        ArrowData::new(other.schema(), vec![other]),
        "another type",
    );
    // A column without the field metadata the request requires.
    let mut required = expected(&["a"]);
    if let ExpectedSchema::Schema { fields } = &mut required {
        fields[0].metadata = BTreeMap::from([("role".to_owned(), "key".to_owned())]);
    }
    refused_by(
        "expected",
        required,
        consistent(&["a"]),
        "required metadata",
    );
    // Nulls where the request takes none.
    let mut non_null = expected(&["a"]);
    if let ExpectedSchema::Schema { fields } = &mut non_null {
        fields[0].nullable = false;
    }
    let nulls = batch(
        vec![int("a")],
        vec![Arc::new(Int64Array::from(vec![Some(1), None]))],
    );
    refused_by(
        "expected",
        non_null,
        ArrowData::new(nulls.schema(), vec![nulls]),
        "nulls the request does not take",
    );
    // The data asked for is taken.
    let run = fetching(expected(&["a", "b"]));
    reply(&run, consistent(&["a", "b"])).unwrap();
    assert!(run.next().is_ok());
}

fn fetch_reply(id: u32, data: u32) -> Vec<u8> {
    encode(&ControlMessage::FetchReply(FetchReply {
        id: ReplyId(id),
        outcome: FetchOutcome::Data {
            data: AttachmentId(data),
        },
    }))
    .unwrap()
}

fn unread(id: u32, failure: Unreadable) -> Unreadables {
    Unreadables::from([(AttachmentId(id), failure)])
}

fn extra() -> Attachments {
    Attachments::from([(AttachmentId(2), Attachment::Bytes(b"extra".to_vec()))])
}

#[test]
fn an_attachment_a_binding_could_not_read_ends_the_fetch_as_its_cause_says() {
    // A producer's failure reaches the run as the provider's error.
    let run = fetching(ExpectedSchema::Discover {});
    run.reply_reading(
        &fetch_reply(1, 1),
        Attachments::new(),
        &unread(1, Unreadable::Producer),
    )
    .map_err(error)
    .unwrap();
    let err = error(run.next().unwrap_err());
    assert_eq!(
        (err.kind, err.code, err.retryable),
        (ErrorKind::DataSource, ErrorCode::ProviderError, false)
    );
    assert_eq!(run.state().unwrap(), RunState::Finished);
    // A batch that broke its own stream's schema is check (a)'s refusal,
    // and ends the run.
    let run = fetching(ExpectedSchema::Discover {});
    let err = run
        .reply_reading(
            &fetch_reply(1, 1),
            Attachments::new(),
            &unread(1, Unreadable::Declared),
        )
        .map_err(error)
        .unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::DataSource, ErrorCode::ProviderSchemaMismatch)
    );
    assert_eq!(text(&err, "check"), "declared");
    assert_eq!(
        err.fields.get("attachment"),
        Some(&Detail::Scalar(Scalar::UInt32(1)))
    );
    assert_eq!(run.state().unwrap(), RunState::Finished);
}

#[test]
fn an_unreadable_attachment_s_message_is_checked_before_the_run_changes() {
    let run = fetching(ExpectedSchema::Discover {});
    for failure in [Unreadable::Producer, Unreadable::Declared] {
        // An attachment the reply does not name, or a reply that names
        // another, is the envelope's refusal; the run still awaits.
        for (message, attachments) in [
            (fetch_reply(1, 1), extra()),
            (fetch_reply(1, 2), Attachments::new()),
        ] {
            let err = run
                .reply_reading(&message, attachments, &unread(1, failure))
                .map_err(error)
                .unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidMessage, "{failure:?}");
            assert!(matches!(run.state().unwrap(), RunState::AwaitingFetch(_)));
        }
        // A reply the run does not await gets the run's own refusal.
        let err = run
            .reply_reading(&fetch_reply(7, 1), Attachments::new(), &unread(1, failure))
            .map_err(error)
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::UnknownReply, "{failure:?}");
        assert!(matches!(run.state().unwrap(), RunState::AwaitingFetch(_)));
    }
    // The run takes its fetch afterwards.
    let data = consistent(&["a"]);
    reply(&run, data).unwrap();
    assert!(run.next().is_ok());
    // A start: an unreadable attachment its operation does not name is the
    // envelope's refusal, not the producer's.
    let validate = start_message(Operation::ValidateQuery(ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: None,
            capabilities: BTreeMap::new(),
        },
    }));
    let attachments = Attachments::from([
        (AttachmentId(1), json(&"PREDICT t.c FOR EACH t.id")),
        (
            AttachmentId(2),
            json(&QuerySchema {
                tables: vec![],
                links: vec![],
            }),
        ),
    ]);
    let err = HostRun::start_reading(
        &sdm_host::default_root(),
        &validate,
        attachments,
        &unread(3, Unreadable::Producer),
    )
    .map_err(error)
    .err()
    .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
}

fn panicking(at: PanicAt, payload: &str) -> Result<HostRun, CoreError> {
    HostRun::start(
        &sdm_host::default_root(),
        &start_message(Operation::Test(TestOperation::Panic {
            at,
            payload: payload.into(),
        })),
        Attachments::new(),
    )
    .map_err(error)
}

fn is_panic(error: &CoreError) -> bool {
    error.fields.get("panic") == Some(&Detail::Scalar(Scalar::Bool(true)))
}

/// Everything an error carries, written out: its message, fields and
/// cause, as JSON and as text.
fn written(error: &CoreError) -> String {
    format!(
        "{} {error} {error:?}",
        serde_json::to_string(error).unwrap()
    )
}

#[test]
fn a_test_only_operation_that_panics_is_contained_wherever_it_panics() {
    // While the run starts: no run, the start's error.
    let err = panicking(PanicAt::Start, "at start").err().unwrap();
    assert!(is_panic(&err));
    assert_eq!(err.code, ErrorCode::InternalError);
    // In the body, and on the adapter's grown stack: the run is poisoned.
    for at in [PanicAt::Body, PanicAt::AdapterStack] {
        let run = panicking(at, "in the run").unwrap();
        let err = error(run.next().unwrap_err());
        assert!(is_panic(&err), "{at:?}");
        let location = text(&err, "location");
        if at == PanicAt::AdapterStack {
            // Inside the interim parser itself.
            assert!(location.contains("parser.rs"), "{location}");
        } else {
            assert!(!location.is_empty());
        }
        assert_eq!(run.state().unwrap(), RunState::Poisoned, "{at:?}");
        assert_eq!(error(run.next().unwrap_err()).code, ErrorCode::RunPoisoned);
    }
}

#[test]
fn an_error_carrying_fields_panic_does_not_poison_a_finished_run() {
    let claimed = CoreError::new(ErrorKind::DataSource, ErrorCode::ProviderError, "failed")
        .with_field("panic", Detail::Scalar(Scalar::Bool(true)));
    let run = HostRun::start(
        &sdm_host::default_root(),
        &start_message(Operation::Test(TestOperation::Fail {
            error: AttachmentId(1),
        })),
        Attachments::from([(AttachmentId(1), json(&claimed))]),
    )
    .map_err(error)
    .unwrap();
    assert_eq!(error(run.next().unwrap_err()), claimed);
    assert_eq!(run.state().unwrap(), RunState::Finished);
    assert_eq!(error(run.next().unwrap_err()).code, ErrorCode::RunFinished);
}

const CANARY: &str = "CANARY-7f3a";

/// Every error path a canary can reach through `HostRun` at T1.7c, each
/// error checked for the canary.
#[test]
fn canaries_reach_no_error() {
    let mut errors = Vec::new();
    let root = sdm_host::default_root();
    let validate = Operation::ValidateQuery(ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: None,
            capabilities: BTreeMap::new(),
        },
    });
    let good = String::from_utf8(start_message(validate.clone())).unwrap();
    let query_schema = json(&QuerySchema {
        tables: vec![],
        links: vec![],
    });
    let attachments = || {
        Attachments::from([
            (AttachmentId(1), json(&"PREDICT t.c FOR EACH t.id")),
            (AttachmentId(2), query_schema.clone()),
        ])
    };
    // Malformed control values: a version, a scalar, a tag.
    for bad in [
        good.replacen(
            &format!("\"contract\":{CONTRACT_VERSION}"),
            &format!("\"contract\":\"{CANARY}\""),
            1,
        ),
        good.replacen(
            "\"v\":\"1767225600000000\"",
            &format!("\"v\":\"{CANARY}\""),
            1,
        ),
        good.replacen(
            "\"op\":\"validate_query\"",
            &format!("\"op\":\"{CANARY}\""),
            1,
        ),
        format!("{{\"message\":\"{CANARY}\"}}"),
        format!("{good}{CANARY}"),
    ] {
        assert_ne!(bad, good);
        errors.push(error(
            HostRun::start(&root, bad.as_bytes(), attachments())
                .err()
                .unwrap(),
        ));
    }
    // Malformed attachments: not JSON, and JSON of another shape.
    for (query, schema) in [
        (
            Attachment::Bytes(format!("\"{CANARY}").into_bytes()),
            query_schema.clone(),
        ),
        (
            json(&"PREDICT t.c FOR EACH t.id"),
            Attachment::Bytes(format!("{{\"tables\":\"{CANARY}\",\"links\":[]}}").into_bytes()),
        ),
    ] {
        errors.push(error(
            HostRun::start(
                &root,
                good.as_bytes(),
                Attachments::from([(AttachmentId(1), query), (AttachmentId(2), schema)]),
            )
            .err()
            .unwrap(),
        ));
    }
    // A malformed reply.
    let run = fetching(ExpectedSchema::Discover {});
    errors.push(error(
        run.reply(
            format!(
                "{{\"message\":\"fetch_reply\",\"id\":1,\"outcome\":{{\"outcome\":\"{CANARY}\"}}}}"
            )
            .as_bytes(),
            Attachments::new(),
        )
        .unwrap_err(),
    ));
    // Arrow data values, a credential among them, refused by each check.
    let values = || -> ArrayRef {
        Arc::new(arrow_array::StringArray::from(vec![
            format!("{CANARY}-value"),
            format!("postgres://user:{CANARY}@db.example/sales"),
        ]))
    };
    let text_field = |name: &str| Field::new(name, DataType::Utf8, true);
    let run = fetching(ExpectedSchema::Discover {});
    errors.push(
        reply(
            &run,
            ArrowData::new(
                schema(vec![text_field("a")]),
                vec![batch(vec![text_field("b")], vec![values()])],
            ),
        )
        .unwrap_err(),
    );
    let run = fetching(expected(&["a"]));
    let wrong = batch(vec![text_field("a")], vec![values()]);
    errors.push(reply(&run, ArrowData::new(wrong.schema(), vec![wrong])).unwrap_err());
    // Panic payloads, wherever the panic happens.
    errors.push(panicking(PanicAt::Start, CANARY).err().unwrap());
    for at in [PanicAt::Body, PanicAt::AdapterStack] {
        let run = panicking(at, CANARY).unwrap();
        errors.push(error(run.next().unwrap_err()));
    }
    assert_eq!(errors.len(), 13);
    for err in &errors {
        let mut chain = Some(err);
        while let Some(e) = chain {
            assert!(!written(e).contains(CANARY), "{}", written(e));
            chain = e.cause.as_deref();
        }
    }
    println!("canary scenario done");
}

#[test]
fn canaries_reach_no_stderr() {
    // The scenario above, in a child process whose stderr is captured: the
    // panic hook writes the location of each contained panic there, and
    // nothing else.
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "canaries_reach_no_error",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("canary scenario done"), "{stdout}");
    assert_eq!(
        stderr.matches("sdm core: a contained panic at ").count(),
        3,
        "{stderr}"
    );
    assert!(!stderr.contains(CANARY), "{stderr}");
    assert!(!stdout.contains(CANARY), "{stdout}");
}

/// A predict run through `HostRun` (T1.11d), its request state's payload
/// `payload` at `version`.
fn predicting(version: u32, payload: &str) -> Result<HostRun, CoreError> {
    HostRun::start(
        &sdm_host::default_root(),
        &start_message(Operation::Predict(Predict {
            state: RequestState {
                version,
                payload: AttachmentId(1),
                query: AttachmentId(2),
            },
            model: "kumo-relational".into(),
            request_id: "sdm-000000000000".into(),
        })),
        Attachments::from([
            (
                AttachmentId(1),
                Attachment::Bytes(payload.as_bytes().to_vec()),
            ),
            (
                AttachmentId(2),
                json(&"PREDICT SUM(o.x, 0, 1, days) FOR EACH u.id"),
            ),
        ]),
    )
    .map_err(error)
}

/// A regression's payload whose data hold the canary, as a request's may.
fn planted_payload() -> String {
    format!(
        r#"{{"task":{{"kind":"regression"}},"schema":{{"related_tables":{{"u":{{"columns":{{"id":{{"dtype":"int64"}}}}}}}},"relationships":[{{"source_columns":["__kumo_entity_ref_0"],"target_columns":["id"],"target_table":"u"}}]}},"context":{{"instance_table":{{"format":"arrays","columns":["instance_id","c"],"rows":[[0,"{CANARY}"]]}}}},"predict":{{"instance_table":{{"format":"arrays","columns":["instance_id","__kumo_entity_ref_0"],"rows":[[1,5]]}}}},"metadata":{{"note":"postgres://user:{CANARY}@db.example/sales"}}}}"#
    )
}

/// The run's error once its model call is answered with `outcome`.
fn after_answer(outcome: ModelOutcome, body: Option<&str>) -> CoreError {
    let run = predicting(REQUEST_STATE_VERSION, &planted_payload()).unwrap();
    let ControlMessage::Step(sdm_contract::Step::CallModel(call)) =
        read(&run.next().unwrap()).unwrap()
    else {
        panic!("a model call");
    };
    // The call carries the planted values: they reach the model, and the
    // check below is of everything the errors after it write.
    let Some(Attachment::Bytes(sent)) = run.take_attachment(call.body).unwrap() else {
        panic!("a body");
    };
    assert!(String::from_utf8(sent).unwrap().contains(CANARY));
    let reply = encode(&ControlMessage::ModelReply(ModelReply {
        id: call.id,
        outcome,
    }))
    .unwrap();
    let attachments = match body {
        Some(body) => {
            Attachments::from([(AttachmentId(1), Attachment::Bytes(body.as_bytes().to_vec()))])
        }
        None => Attachments::new(),
    };
    run.reply(&reply, attachments).unwrap();
    error(run.next().unwrap_err())
}

/// Every error path `predict` adds at T1.11d, each checked for the canary:
/// a request state of another version, a payload that is not JSON, the
/// model's answers the core cannot read or that are not this request's,
/// an endpoint's refusal, and a transport that failed.
#[test]
fn predict_s_canaries_reach_no_error() {
    let response = |status: u16| ModelOutcome::Response {
        status,
        content_type: "application/json".into(),
        body: AttachmentId(1),
    };
    let mut errors = vec![
        predicting(REQUEST_STATE_VERSION + 1, &planted_payload())
            .err()
            .unwrap(),
        error(
            predicting(REQUEST_STATE_VERSION, &format!("{{\"a\": \"{CANARY}\" 1}}"))
                .unwrap()
                .next()
                .unwrap_err(),
        ),
    ];
    for (status, body) in [
        (200, format!("{{\"{CANARY}\"")),
        (200, format!(r#"{{"id":"{CANARY}","model":5}}"#)),
        (
            200,
            format!(
                r#"{{"id":"r","model":"m","metadata":{{}},"predictions":[{{"id":"{CANARY}","row_index":0,"prediction":1.0}}]}}"#
            ),
        ),
        (
            200,
            format!(
                r#"{{"id":"r","model":"m","metadata":{{}},"predictions":[{{"id":"1","row_index":0,"prediction":"{CANARY}"}}]}}"#
            ),
        ),
        (500, format!(r#"{{"detail":"no column {CANARY}"}}"#)),
    ] {
        errors.push(after_answer(response(status), Some(&body)));
    }
    errors.push(after_answer(
        ModelOutcome::TransportError {
            kind: ErrorKind::ModelEndpoint,
            retryable: true,
        },
        None,
    ));
    assert_eq!(errors.len(), 8);
    let codes: Vec<ErrorCode> = errors.iter().map(|e| e.code).collect();
    assert_eq!(
        codes,
        [
            ErrorCode::InvalidRequest,
            ErrorCode::InvalidRequest,
            ErrorCode::InvalidResponse,
            ErrorCode::InvalidResponse,
            ErrorCode::InvalidResponse,
            ErrorCode::UnsupportedRequest,
            ErrorCode::InvalidResponse,
            ErrorCode::TransportError,
        ]
    );
    for err in &errors {
        let mut chain = Some(err);
        while let Some(e) = chain {
            assert!(!written(e).contains(CANARY), "{}", written(e));
            chain = e.cause.as_deref();
        }
    }
}

/// The panic canary is not vacuous: outside the host's containment, the
/// test-only operation's body panics with the canary as its payload, which
/// `HostRun` keeps out of every error (`canaries_reach_no_error`).
#[test]
fn a_test_only_panic_s_payload_is_the_canary() {
    let mut run = sdm_core::start(
        &sdm_host::default_root(),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: HostFacts {
                now: Scalar::Timestamp {
                    unit: Unit::Us,
                    zone: Some("UTC".into()),
                    value: 0,
                },
                local_zone: None,
                zone_roots: vec![],
                platform: PlatformFacts {
                    os: "linux".into(),
                    strftime_pads_year: false,
                },
            },
            operation: Operation::Test(TestOperation::Panic {
                at: PanicAt::Body,
                payload: CANARY.into(),
            }),
        },
        Attachments::new(),
        Arc::new(NoZones),
    )
    .unwrap();
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.next()))
        .err()
        .unwrap();
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap();
    assert!(message.contains(CANARY), "{message}");
}

struct NoZones;

impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}
