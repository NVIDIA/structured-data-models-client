// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A corpus of control messages, shared by the round-trip tests and the
//! byte-mutation fuzz property.

use std::collections::BTreeMap;

use sdm_contract::data::{
    AttachmentId, Bound, DataRequest, DescribePart, Dialect, ExpectedSchema, FetchPlan, Filter,
    KeySet, ProviderError, SeedBound, SqlParam, SqlStatement, TableColumn, TableKey, TieBreak,
    TieTerm, Window,
};
use sdm_contract::facts::{LocalZone, PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{
    Capability, ColumnRef, DescribeSource, DescribedSource, EditSchema, Materialize, Materialized,
    Operation, OperationResult, PqlDiagnostic, Predict, Predicted, ProfileName, ProfileSpec,
    QueryValidation, RequestState, SamplerState, SchemaEdited, Seed, Severity, Span, ValidateQuery,
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

/// A seed key set: `key`, `batch`, and the seed time columns named.
pub fn seeds(times: &[&str]) -> KeySet {
    let field = |name: &str, column_type: ColumnType| Field {
        name: name.into(),
        column_type,
        nullable: false,
        metadata: BTreeMap::new(),
    };
    let mut columns = vec![
        field("key", ColumnType::Int64),
        field("batch", ColumnType::Int64),
    ];
    columns.extend(times.iter().map(|name| {
        field(
            name,
            ColumnType::Timestamp {
                unit: Unit::Ns,
                zone: None,
            },
        )
    }));
    KeySet {
        attachment: AttachmentId(4),
        columns,
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
                dialect: Dialect::new("duckdb"),
            },
            source: AttachmentId(1),
            check_keys: true,
        })),
        start(Operation::EditSchema(EditSchema {
            schema: AttachmentId(1),
            held: Some(AttachmentId(2)),
            edit: AttachmentId(3),
        })),
        start(Operation::Materialize(Materialize {
            provider: sdm_contract::data::ProviderChoice::Sql {
                dialect: Dialect::new("duckdb"),
            },
            schema: AttachmentId(1),
            held: Some(AttachmentId(5)),
            query: AttachmentId(2),
            indices: Some(AttachmentId(3)),
            anchor_time: None,
            seed: Some(Seed::from(1_u128 << 70)),
            unseeded: false,
            anchor_entity: false,
            values: None,
            options: vec!["num_neighbors".into()],
            sampler_state: Some(SamplerState {
                version: 1,
                memo: AttachmentId(4),
            }),
        })),
        start(Operation::Predict(Predict {
            state: state.clone(),
            model: "kumo-relational".into(),
            request_id: "sdm-000000000000".into(),
            outputs: vec!["prediction".into()],
            default_outputs: false,
            inference_asked: vec![],
        })),
        start(Operation::Predict(Predict {
            state: state.clone(),
            model: "kumo-relational".into(),
            request_id: "sdm-000000000001".into(),
            outputs: vec![],
            default_outputs: true,
            inference_asked: vec![],
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
                plan: FetchPlan::Sql {
                    statements: vec![SqlStatement {
                        dialect: Dialect::new("duckdb"),
                        text: "SELECT id, at FROM orders WHERE id IN (SELECT id FROM keys)".into(),
                        params: vec![
                            SqlParam::KeySet {
                                keys: keys(),
                                relation: "keys".into(),
                            },
                            SqlParam::Value {
                                value: Scalar::UInt64(u64::MAX),
                            },
                        ],
                    }],
                    expect: ExpectedSchema::Discover {},
                },
            }],
        }),
        ControlMessage::Step(Step::Fetch {
            requests: vec![
                FetchStep {
                    id: ReplyId(2),
                    request: DataRequest::Neighbors {
                        table: "orders".into(),
                        key: "user_id".into(),
                        columns: vec!["order_id".into(), "ts".into()],
                        seeds: seeds(&["anchor"]),
                        window: Some(Window {
                            column: "ts".into(),
                            lower: None,
                            upper: Some(SeedBound {
                                seed_column: "anchor".into(),
                                inclusive: true,
                            }),
                        }),
                        per_seed: Some(32),
                        order: Some(TieBreak(vec![
                            TieTerm {
                                column: "ts".into(),
                                as_timestamp: true,
                                descending: true,
                                nulls_last: true,
                            },
                            TieTerm {
                                column: "order_id".into(),
                                as_timestamp: false,
                                descending: false,
                                nulls_last: true,
                            },
                        ])),
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
                FetchStep {
                    id: ReplyId(3),
                    request: DataRequest::Lookup {
                        table: "users".into(),
                        key: "user_id".into(),
                        columns: vec!["age".into()],
                        seeds: seeds(&[]),
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
                FetchStep {
                    id: ReplyId(4),
                    request: DataRequest::SampleEntities {
                        table: "users".into(),
                        key: vec!["user_id".into()],
                        n: 10_000,
                        seed: Scalar::UInt64(42),
                        columns: vec!["user_id".into()],
                        filters: vec![Filter::NotNull {
                            column: "user_id".into(),
                        }],
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
                FetchStep {
                    id: ReplyId(5),
                    request: DataRequest::TimeBounds {
                        columns: vec![TableColumn {
                            table: "orders".into(),
                            column: "ts".into(),
                        }],
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
                FetchStep {
                    id: ReplyId(6),
                    request: DataRequest::KeyCounts {
                        keys: vec![TableKey {
                            table: "orders".into(),
                            columns: vec!["order_id".into()],
                        }],
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
                FetchStep {
                    id: ReplyId(7),
                    request: DataRequest::Describe {
                        tables: vec![],
                        part: DescribePart::Indexes,
                    },
                    plan: FetchPlan::Host {
                        expect: ExpectedSchema::Discover {},
                    },
                },
            ],
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
            result: OperationResult::EditSchema(SchemaEdited {
                schema: AttachmentId(9),
                declared: Some(ColumnRef {
                    table: "users".into(),
                    column: "signup".into(),
                }),
            }),
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::Materialize(Materialized { state }),
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::Predict(Predicted {
                result: AttachmentId(10),
                model: "kumo-relational".into(),
                request_id: "r1".into(),
                metadata: AttachmentId(12),
                outputs: vec!["prediction".into(), "probabilities".into()],
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
        ControlMessage::SamplerState {
            state: SamplerState {
                version: 1,
                memo: AttachmentId(7),
            },
        },
    ]
}
