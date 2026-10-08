// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A corpus of control messages, shared by the round-trip tests and the
//! byte-mutation fuzz property.

use std::collections::BTreeMap;

use sdm_contract::data::{
    AttachmentId, Bound, DataRequest, Dialect, ExpectedSchema, Filter, KeySet, ProviderError,
    SqlParam, SqlStatement,
};
use sdm_contract::facts::{LocalZone, PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{
    Capability, DescribeSource, DescribedSource, Materialize, Materialized, Operation,
    OperationResult, PqlDiagnostic, Predict, Predicted, ProfileName, ProfileSpec, QueryValidation,
    RequestState, Seed, Severity, Span, ValidateQuery,
};
use sdm_contract::run::{
    Endpoint, FetchOutcome, FetchReply, FetchStep, ModelCall, ModelOutcome, ModelReply, ReplyId,
    StartRun,
};
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, CoreWarning, Detail, ErrorCode,
    ErrorKind, Field, HostFacts, Scalar, Step, Unit,
};

pub fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_790_000_000_000_000,
        },
        local_zone: Some(LocalZone::Iana("Europe/Berlin".into())),
        zone_roots: vec![ZoneRootFact {
            path: "/usr/share/zoneinfo".into(),
            version: Some("2026c".into()),
        }],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
    }
}

pub fn keys() -> KeySet {
    KeySet {
        attachment: AttachmentId(3),
        columns: vec![Field {
            name: "id".into(),
            column_type: ColumnType::Uint64,
            nullable: false,
            metadata: BTreeMap::new(),
        }],
    }
}

/// One message of every kind, each shape the contract has.
pub fn messages() -> Vec<ControlMessage> {
    let validate = ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: Some("rfm".into()),
            capabilities: BTreeMap::from([(Capability::Temporal, false)]),
        },
    };
    let start = |operation| {
        let mut facts = facts();
        if matches!(operation, Operation::Materialize(_)) {
            facts.local_zone = Some(LocalZone::Posix("EST5EDT,M3.2.0,M11.1.0".into()));
        }
        ControlMessage::Start(Box::new(StartRun {
            contract: CONTRACT_VERSION,
            facts,
            operation,
        }))
    };
    let state = RequestState {
        version: 1,
        payload: AttachmentId(1),
        query: AttachmentId(2),
    };
    vec![
        start(Operation::ValidateQuery(validate)),
        start(Operation::DescribeSource(DescribeSource {
            provider: sdm_contract::data::ProviderChoice::Sql {
                dialect: Dialect::Duckdb,
            },
            source: AttachmentId(1),
            check_keys: true,
        })),
        start(Operation::Materialize(Materialize {
            provider: sdm_contract::data::ProviderChoice::Sql {
                dialect: Dialect::Duckdb,
            },
            schema: AttachmentId(1),
            query: AttachmentId(2),
            indices: Some(AttachmentId(3)),
            anchor_time: None,
            seed: Some(Seed::from(1_u128 << 70)),
        })),
        start(Operation::Predict(Predict {
            state: state.clone(),
            model: "rfm".into(),
        })),
        ControlMessage::Step(Step::Fetch {
            requests: vec![FetchStep {
                id: ReplyId(1),
                request: DataRequest::Scan {
                    table: "orders".into(),
                    columns: vec!["id".into(), "at".into()],
                    filters: vec![
                        Filter::KeyIn {
                            columns: vec!["id".into()],
                            keys: keys(),
                        },
                        Filter::TimeRange {
                            column: "at".into(),
                            lower: Some(Bound {
                                value: Scalar::Timestamp {
                                    unit: Unit::Us,
                                    zone: None,
                                    value: i64::MIN,
                                },
                                inclusive: true,
                            }),
                            upper: None,
                        },
                    ],
                    limit: None,
                },
                sql: Some(SqlStatement {
                    dialect: Dialect::Duckdb,
                    text: "SELECT id, at FROM orders WHERE id IN (SELECT id FROM keys)".into(),
                    params: vec![
                        SqlParam::KeySet { keys: keys() },
                        SqlParam::Value {
                            value: Scalar::UInt64(u64::MAX),
                        },
                    ],
                }),
                expect: ExpectedSchema::Discover {},
            }],
        }),
        ControlMessage::FetchReply(FetchReply {
            id: ReplyId(1),
            outcome: FetchOutcome::Data {
                data: AttachmentId(4),
            },
        }),
        ControlMessage::FetchReply(FetchReply {
            id: ReplyId(2),
            outcome: FetchOutcome::Error {
                error: ProviderError {
                    kind: ErrorKind::DataSource,
                    retryable: true,
                    sqlstate: Some("42P01".into()),
                    vendor_code: None,
                },
            },
        }),
        ControlMessage::Step(Step::CallModel(ModelCall {
            id: ReplyId(5),
            endpoint: Endpoint::Predict,
            content_type: "application/json".into(),
            body: AttachmentId(6),
        })),
        ControlMessage::ModelReply(ModelReply {
            id: ReplyId(5),
            outcome: ModelOutcome::Response {
                status: 200,
                content_type: "application/json".into(),
                body: AttachmentId(7),
            },
        }),
        ControlMessage::ModelReply(ModelReply {
            id: ReplyId(5),
            outcome: ModelOutcome::TransportError {
                kind: ErrorKind::ModelEndpoint,
                retryable: true,
            },
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::ValidateQuery(QueryValidation {
                accepted: false,
                diagnostics: vec![PqlDiagnostic {
                    code: "UNKNOWN_COLUMN".into(),
                    message: "no column".into(),
                    severity: Severity::Error,
                    span: Some(Span {
                        start_row: 1,
                        start_col: 8,
                        end_row: 1,
                        end_col: 17,
                    }),
                }],
            }),
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::DescribeSource(DescribedSource {
                description: AttachmentId(9),
            }),
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::Materialize(Materialized { state }),
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::Predict(Predicted {
                result: AttachmentId(10),
                columns: vec!["ENTITY".into(), "TARGET_PRED".into()],
            }),
        }),
        ControlMessage::Attached {
            attachment: AttachmentId(11),
        },
        ControlMessage::Cancel {},
        ControlMessage::Error {
            error: CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, "panicked")
                .with_field("panic", Detail::Scalar(Scalar::Bool(true)))
                .caused_by(CoreError::provider(&ProviderError {
                    kind: ErrorKind::DataSource,
                    retryable: true,
                    sqlstate: Some("08006".into()),
                    vendor_code: Some("57P01".into()),
                })),
        },
        ControlMessage::Warnings {
            warnings: vec![CoreWarning {
                code: "COMPOSITE_KEY_DROPPED".into(),
                message: "dropped".into(),
                fields: BTreeMap::from([(
                    "columns".into(),
                    Detail::List(vec![Scalar::Utf8("a".into()), Scalar::Utf8("b".into())]),
                )]),
            }],
        },
    ]
}
