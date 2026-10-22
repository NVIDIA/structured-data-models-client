// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract v3 through `HostRun` (T1.7b): input-sized fields travel as
//! attachments, so the 64 KB cap bounds only a message's envelope
//! (decision H5); every message the core writes arrives whole, an
//! attachment carrying one whose encoding would pass the cap (H7); a
//! provider's retryability reaches the error (H6); and the host's local zone
//! may be a POSIX `TZ` string.
//!
//! The fetch, result, error and warnings of exact sizes come from the
//! test-only operations (`test-ops`), which take their inputs as
//! attachments too.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use sdm_contract::data::{
    AttachmentId, DataRequest, Dialect, ExpectedSchema, FetchPlan, ProviderError, SqlStatement,
};
use sdm_contract::facts::{LocalZone, PlatformFacts};
use sdm_contract::ops::{
    Dtype, Operation, OperationResult, PqlDiagnostic, ProfileName, ProfileSpec, QueryColumn,
    QuerySchema, QueryTable, QueryValidation, Severity, Stype, ValidateQuery,
};
use sdm_contract::run::{FetchOutcome, FetchReply, FetchStep, ReplyId, StartRun};
use sdm_contract::test_ops::{TestFetch, TestOperation, TestResult};
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, CoreWarning, Detail, ErrorCode,
    ErrorKind, Field, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{ArrowData, Attachment, Attachments};
use sdm_host::{CONTROL_CAP, HostRun, Outgoing, decode, encode, read};

fn facts(local_zone: Option<LocalZone>) -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

fn start_message(operation: Operation, local_zone: Option<LocalZone>) -> ControlMessage {
    ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts(local_zone),
        operation,
    }))
}

/// A bytes attachment of `value` as JSON.
fn json(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn start(operation: Operation, attachments: Attachments) -> Result<HostRun, CoreError> {
    HostRun::start(
        &sdm_host::default_root(),
        &encode(&start_message(operation, None)).unwrap(),
        attachments,
    )
    .map_err(error)
}

/// The core error an `error` control message carries.
fn error(out: Outgoing) -> CoreError {
    match read(&out).unwrap() {
        ControlMessage::Error { error } => error,
        other => panic!("not an error: {other:?}"),
    }
}

/// Whether `out` travelled as an attachment, checking it is otherwise a
/// plain message within the cap.
fn attached(out: &Outgoing) -> bool {
    assert!(out.control.len() <= CONTROL_CAP);
    let is_attached = matches!(
        decode(&out.control).unwrap(),
        ControlMessage::Attached { .. }
    );
    assert_eq!(is_attached, !out.attachments.is_empty());
    is_attached
}

/// The message `make` builds, padded so its JSON is exactly `size` bytes.
fn exactly(size: usize, make: impl Fn(usize) -> ControlMessage) -> ControlMessage {
    let base = serde_json::to_vec(&make(0)).unwrap().len();
    let message = make(size - base);
    assert_eq!(serde_json::to_vec(&message).unwrap().len(), size);
    message
}

/// Either side of the cap: the size and whether it travels as an
/// attachment.
const EITHER_SIDE: [(usize, bool); 2] = [(CONTROL_CAP, false), (CONTROL_CAP + 1, true)];

fn validate(model: Option<String>) -> Operation {
    Operation::ValidateQuery(ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model,
            capabilities: BTreeMap::new(),
        },
    })
}

/// A users table with `extra` numeric columns `c0`, `c1`, ... beside its key.
fn wide_schema(extra: usize) -> QuerySchema {
    let mut columns = vec![QueryColumn {
        name: "user_id".into(),
        dtype: Dtype::Int,
        stype: Stype::Id,
    }];
    columns.extend((0..extra).map(|i| QueryColumn {
        name: format!("c{i}"),
        dtype: Dtype::Float,
        stype: Stype::Numerical,
    }));
    QuerySchema {
        tables: vec![QueryTable {
            name: "users".into(),
            columns,
            primary_key: Some("user_id".into()),
            time_column: None,
            end_time_column: None,
        }],
        links: vec![],
    }
}

fn validation(run: &HostRun) -> QueryValidation {
    match read(&run.next().unwrap()).unwrap() {
        ControlMessage::Step(Step::Done {
            result: OperationResult::ValidateQuery(validation),
        }) => validation,
        other => panic!("not validate_query's result: {other:?}"),
    }
}

#[test]
fn a_wide_schema_and_a_long_query_are_accepted_through_host_run() {
    let schema = wide_schema(5_000);
    let query = format!(
        "PREDICT users.c4999{}FOR EACH users.user_id",
        " ".repeat(70_000)
    );
    assert!(query.len() > CONTROL_CAP);
    assert!(serde_json::to_vec(&schema).unwrap().len() > CONTROL_CAP);
    let run = start(
        validate(None),
        Attachments::from([
            (AttachmentId(1), json(&query)),
            (AttachmentId(2), json(&schema)),
        ]),
    )
    .unwrap();
    let validation = validation(&run);
    assert!(validation.accepted, "{:?}", validation.diagnostics);
}

#[test]
fn a_start_envelope_is_capped_and_its_attachments_are_not() {
    let query = "PREDICT users.c0 FOR EACH users.user_id";
    let attachments = || {
        Attachments::from([
            (AttachmentId(1), json(&query)),
            (AttachmentId(2), json(&wide_schema(1))),
        ])
    };
    // The envelope: one byte within the cap starts; one past is refused.
    for (size, refused) in EITHER_SIDE {
        let message = exactly(size, |pad| {
            start_message(validate(Some("m".repeat(pad))), None)
        });
        let bytes = serde_json::to_vec(&message).unwrap();
        let started = HostRun::start(&sdm_host::default_root(), &bytes, attachments());
        match started {
            Ok(run) => {
                assert!(!refused);
                assert!(validation(&run).accepted);
            }
            Err(out) => {
                assert!(refused);
                let err = error(out);
                assert_eq!(
                    (err.kind, err.code),
                    (ErrorKind::Protocol, ErrorCode::InvalidMessage)
                );
            }
        }
    }
    // An attachment either side of the cap: both taken.
    for size in [CONTROL_CAP - 1, CONTROL_CAP + 1] {
        let base = serde_json::to_vec(query).unwrap().len();
        let padded = format!("{query}{}", " ".repeat(size - base));
        assert_eq!(serde_json::to_vec(&padded).unwrap().len(), size);
        let run = start(
            validate(None),
            Attachments::from([
                (AttachmentId(1), json(&padded)),
                (AttachmentId(2), json(&wide_schema(1))),
            ]),
        )
        .unwrap();
        assert!(validation(&run).accepted);
    }
}

/// A test-only operation whose input is `value`, as attachment 1.
fn test_op(operation: TestOperation, value: &impl serde::Serialize) -> HostRun {
    start(
        Operation::Test(operation),
        Attachments::from([(AttachmentId(1), json(value))]),
    )
    .unwrap()
}

fn wide_fetch(pad: usize) -> TestFetch {
    let fields = (0..200)
        .map(|i| Field {
            name: format!("column_{i:03}"),
            column_type: ColumnType::Int64,
            nullable: true,
            metadata: BTreeMap::new(),
        })
        .collect();
    TestFetch {
        echo: false,
        undeclared: vec![],
        fallback: None,
        request: DataRequest::Scan {
            table: "orders".into(),
            columns: (0..200).map(|i| format!("column_{i:03}")).collect(),
            filters: vec![],
            limit: None,
        },
        plan: Some(FetchPlan::Sql {
            statements: vec![SqlStatement {
                dialect: Dialect::new("duckdb"),
                text: format!("SELECT * FROM orders{}", " ".repeat(pad)),
                params: vec![],
            }],
            expect: ExpectedSchema::Schema { fields },
        }),
        provider: None,
    }
}

fn fetch_step(fetch: &TestFetch) -> ControlMessage {
    ControlMessage::Step(Step::Fetch {
        requests: vec![FetchStep {
            id: ReplyId(1),
            request: fetch.request.clone(),
            plan: fetch.plan.clone().expect("the test fetch's own plan"),
            continuation: None,
        }],
    })
}

#[test]
fn a_fetch_step_with_a_wide_plan_arrives_whole_either_side_of_the_cap() {
    for (size, past) in EITHER_SIDE {
        let base = serde_json::to_vec(&fetch_step(&wide_fetch(0)))
            .unwrap()
            .len();
        let fetch = wide_fetch(size - base);
        let expected = fetch_step(&fetch);
        assert_eq!(serde_json::to_vec(&expected).unwrap().len(), size);
        let run = test_op(
            TestOperation::Fetch {
                fetch: AttachmentId(1),
            },
            &fetch,
        );
        let out = run.next().unwrap();
        assert_eq!(attached(&out), past, "{size}");
        assert_eq!(read(&out).unwrap(), expected);
        // The run goes on as usual: it is answered, with the 200 columns
        // it expects back, and finishes.
        let rows = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
        let batch = RecordBatch::try_from_iter(
            (0..200).map(|i| (format!("column_{i:03}"), Arc::clone(&rows))),
        )
        .unwrap();
        run.reply(
            &encode(&ControlMessage::FetchReply(FetchReply {
                id: ReplyId(1),
                outcome: FetchOutcome::Data {
                    data: AttachmentId(1),
                    continuation: None,
                },
            }))
            .unwrap(),
            Attachments::from([(
                AttachmentId(1),
                Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
            )]),
        )
        .unwrap();
        assert_eq!(
            read(&run.next().unwrap()).unwrap(),
            ControlMessage::Step(Step::Done {
                result: OperationResult::Test(TestResult {
                    rows: 3,
                    data: None,
                    offset_seconds: None,
                })
            })
        );
    }
}

fn diagnostics_of(pad: usize) -> OperationResult {
    OperationResult::ValidateQuery(QueryValidation {
        accepted: false,
        diagnostics: (0..10)
            .map(|i| PqlDiagnostic {
                code: "PQL_SCHEMA_UNKNOWN_COLUMN".into(),
                message: format!(
                    "{i}{}",
                    if i == 0 {
                        "m".repeat(pad)
                    } else {
                        String::new()
                    }
                ),
                severity: Severity::Error,
                span: None,
            })
            .collect(),
    })
}

#[test]
fn a_validate_query_result_arrives_whole_either_side_of_the_cap() {
    for (size, past) in EITHER_SIDE {
        let message = exactly(size, |pad| {
            ControlMessage::Step(Step::Done {
                result: diagnostics_of(pad),
            })
        });
        let ControlMessage::Step(Step::Done { result }) = &message else {
            unreachable!()
        };
        let run = test_op(
            TestOperation::Finish {
                result: AttachmentId(1),
            },
            result,
        );
        let out = run.next().unwrap();
        assert_eq!(attached(&out), past, "{size}");
        assert_eq!(read(&out).unwrap(), message);
    }
}

#[test]
fn a_real_validation_whose_diagnostics_pass_the_cap_arrives_whole() {
    // An unknown column whose name passes 64 KB: the parser's diagnostic
    // names it, so the result passes the cap, and arrives whole.
    let name = "x".repeat(CONTROL_CAP + 10);
    let query = format!("PREDICT users.{name} FOR EACH users.user_id");
    let run = start(
        validate(None),
        Attachments::from([
            (AttachmentId(1), json(&query)),
            (AttachmentId(2), json(&wide_schema(1))),
        ]),
    )
    .unwrap();
    let out = run.next().unwrap();
    assert!(attached(&out));
    let ControlMessage::Step(Step::Done {
        result: OperationResult::ValidateQuery(validation),
    }) = read(&out).unwrap()
    else {
        panic!("validate_query's result")
    };
    assert!(!validation.accepted);
    let written: usize = validation.diagnostics.iter().map(|d| d.message.len()).sum();
    assert!(written > CONTROL_CAP, "{:?}", validation.diagnostics);
    // The run finished with its result: nothing was lost to the cap.
    assert_eq!(error(run.next().unwrap_err()).code, ErrorCode::RunFinished);
}

#[test]
fn an_error_arrives_whole_either_side_of_the_cap() {
    for (size, past) in EITHER_SIDE {
        let message = exactly(size, |pad| ControlMessage::Error {
            error: CoreError::new(ErrorKind::DataSource, ErrorCode::ProviderError, "failed")
                .with_field("table", Detail::text("t".repeat(pad)))
                .caused_by(CoreError::new(
                    ErrorKind::Internal,
                    ErrorCode::InternalError,
                    "cause",
                )),
        });
        let ControlMessage::Error { error: expected } = &message else {
            unreachable!()
        };
        let run = test_op(
            TestOperation::Fail {
                error: AttachmentId(1),
            },
            expected,
        );
        let out = run.next().unwrap_err();
        assert_eq!(attached(&out), past, "{size}");
        assert_eq!(read(&out).unwrap(), message);
    }
}

#[test]
fn warnings_arrive_whole_and_in_order_past_the_cap() {
    let warnings: Vec<CoreWarning> = (0..3)
        .map(|i| CoreWarning {
            code: "LONG".into(),
            message: format!("{i}{}", "w".repeat(CONTROL_CAP)),
            fields: BTreeMap::new(),
        })
        .collect();
    let run = test_op(
        TestOperation::Warn {
            warnings: AttachmentId(1),
        },
        &warnings,
    );
    run.next().unwrap();
    let out = run.take_warnings().unwrap().expect("warnings");
    assert!(attached(&out));
    assert_eq!(read(&out).unwrap(), ControlMessage::Warnings { warnings });
    assert!(run.take_warnings().unwrap().is_none());
}

#[test]
fn an_over_cap_message_s_attachment_is_numbered_in_the_run_s_sequence() {
    let fetch = wide_fetch(CONTROL_CAP);
    let run = test_op(
        TestOperation::Fetch {
            fetch: AttachmentId(1),
        },
        &fetch,
    );
    let out = run.next().unwrap();
    // The run's first attachment.
    assert_eq!(
        decode(&out.control).unwrap(),
        ControlMessage::Attached {
            attachment: AttachmentId(1)
        }
    );
    assert!(out.attachments.contains_key(&AttachmentId(1)));
    // It travelled with its message, so the run holds nothing under it.
    assert!(run.take_attachment(AttachmentId(1)).unwrap().is_none());
}

#[test]
fn a_provider_error_s_retryability_and_codes_reach_the_core_error() {
    for retryable in [true, false] {
        let run = test_op(
            TestOperation::Fetch {
                fetch: AttachmentId(1),
            },
            &wide_fetch(0),
        );
        run.next().unwrap();
        run.reply(
            &encode(&ControlMessage::FetchReply(FetchReply {
                id: ReplyId(1),
                outcome: FetchOutcome::Error {
                    error: ProviderError {
                        refuses_expression: false,
                        kind: ErrorKind::DataSource,
                        retryable,
                        sqlstate: Some("40001".into()),
                        vendor_code: Some("1213".into()),
                    },
                },
            }))
            .unwrap(),
            Attachments::new(),
        )
        .unwrap();
        let err = error(run.next().unwrap_err());
        assert_eq!(
            (err.kind, err.code, err.retryable),
            (ErrorKind::DataSource, ErrorCode::ProviderError, retryable)
        );
        assert_eq!(err.fields["sqlstate"], Detail::text("40001"));
        assert_eq!(err.fields["vendor_code"], Detail::text("1213"));
    }
}

#[test]
fn a_local_zone_may_be_an_iana_name_or_a_posix_tz_string() {
    let attachments = || {
        Attachments::from([
            (
                AttachmentId(1),
                json(&"PREDICT users.c0 FOR EACH users.user_id"),
            ),
            (AttachmentId(2), json(&wide_schema(1))),
        ])
    };
    let begin = |zone: LocalZone| {
        HostRun::start(
            &sdm_host::default_root(),
            &encode(&start_message(validate(None), Some(zone))).unwrap(),
            attachments(),
        )
        .map_err(error)
    };
    for zone in [
        LocalZone::Iana("America/New_York".into()),
        LocalZone::Posix("EST5EDT,M3.2.0,M11.1.0".into()),
        LocalZone::Posix("<+0530>-5:30".into()),
    ] {
        assert!(
            validation(&begin(zone.clone()).unwrap()).accepted,
            "{zone:?}"
        );
    }
    let err = begin(LocalZone::Posix("not a TZ string".into()))
        .err()
        .unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
    );
    assert_eq!(err.fields["field"], Detail::text("facts.local_zone"));
}

#[test]
fn a_start_from_another_contract_version_is_refused_for_its_version_first() {
    // Bad attachments (none of the two named) and a bad local zone too: the
    // version is what the refusal names (decision H25).
    let mut message = start_message(
        validate(None),
        Some(LocalZone::Posix("not a TZ string".into())),
    );
    let ControlMessage::Start(start) = &mut message else {
        unreachable!()
    };
    start.contract = CONTRACT_VERSION + 1;
    let err = error(
        HostRun::start(
            &sdm_host::default_root(),
            &encode(&message).unwrap(),
            Attachments::from([(AttachmentId(9), json(&"stray"))]),
        )
        .err()
        .unwrap(),
    );
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Protocol, ErrorCode::InvalidMessage)
    );
    assert_eq!(
        err.fields.get("contract"),
        Some(&Detail::Scalar(Scalar::UInt32(CONTRACT_VERSION + 1)))
    );
    assert!(!err.fields.contains_key("attachment"));
    assert!(!err.fields.contains_key("field"));
}
