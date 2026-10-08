// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control messages round-trip through JSON, and unknown fields are refused.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::data::{
    AttachmentId, Bound, DataRequest, Determinism, Dialect, ExpectedSchema, Filter, KeySet,
    OptionalRequest, ProviderCapabilities, ProviderChoice, ProviderError, SqlParam, SqlStatement,
};
use sdm_contract::facts::{PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{
    Capability, ColumnRef, Dtype, Operation, OperationResult, PqlDiagnostic, ProfileName,
    ProfileSpec, QueryColumn, QuerySchema, QueryTable, QueryValidation, Severity, Span, Stype,
    ValidateQuery,
};
use sdm_contract::run::{
    Endpoint, FetchOutcome, FetchReply, FetchStep, ModelCall, ModelOutcome, ModelReply, ReplyId,
    StartRun,
};
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, CoreWarning, Detail, ErrorKind, Field,
    HostFacts, Scalar, Step, Unit, codes,
};

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_790_000_000_000_000,
        },
        local_zone: Some("Europe/Berlin".into()),
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

fn keys() -> KeySet {
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

fn messages() -> Vec<ControlMessage> {
    let validate = ValidateQuery {
        query: "PREDICT users.age FOR EACH users.id".into(),
        schema: QuerySchema {
            tables: vec![QueryTable {
                name: "users".into(),
                columns: vec![QueryColumn {
                    name: "id".into(),
                    dtype: Dtype::Int,
                    stype: Stype::Id,
                }],
                primary_key: Some("id".into()),
                time_column: None,
                end_time_column: None,
            }],
            links: vec![vec![ColumnRef {
                table: "users".into(),
                column: "id".into(),
            }]],
        },
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: Some("rfm".into()),
            capabilities: BTreeMap::from([(Capability::Temporal, false)]),
        },
    };
    vec![
        ControlMessage::Start(Box::new(StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::ValidateQuery(validate),
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
                    code: codes::PROVIDER_ERROR.into(),
                    message: "Catalog Error".into(),
                    retryable: false,
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
                message: "timed out".into(),
            },
        }),
        ControlMessage::Step(Step::Done {
            result: OperationResult::ValidateQuery(QueryValidation {
                accepted: false,
                diagnostics: vec![PqlDiagnostic {
                    code: "UNKNOWN_COLUMN".into(),
                    message: "no column".into(),
                    severity: Severity::Error,
                    span: Some(Span { start: 8, end: 17 }),
                }],
            }),
        }),
        ControlMessage::Cancel {},
        ControlMessage::Error {
            error: CoreError::new(ErrorKind::Internal, codes::INTERNAL_ERROR, "panicked")
                .with_field("panic", Detail::Scalar(Scalar::Bool(true)))
                .caused_by(CoreError::protocol(codes::RUN_BUSY, "busy")),
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

#[test]
fn every_message_round_trips() {
    for message in messages() {
        let text = serde_json::to_string(&message).unwrap();
        let back: ControlMessage = serde_json::from_str(&text).unwrap();
        assert_eq!(back, message, "{text}");
    }
}

#[test]
fn provider_choices_round_trip() {
    for choice in [
        ProviderChoice::Sql {
            dialect: Dialect::Duckdb,
        },
        ProviderChoice::Host {
            capabilities: ProviderCapabilities {
                optional: BTreeSet::from([OptionalRequest::Preview, OptionalRequest::CountRows]),
                max_keys_per_request: Some(10_000),
                sampling: Determinism::SeededStable,
            },
        },
    ] {
        let text = serde_json::to_string(&choice).unwrap();
        assert_eq!(
            serde_json::from_str::<ProviderChoice>(&text).unwrap(),
            choice
        );
    }
}

#[test]
fn unknown_fields_and_tags_are_refused() {
    for text in [
        r#"{"message":"cancel","extra":1}"#,
        r#"{"message":"step","step":"ask_host","what":"truthiness"}"#,
        r#"{"message":"fetch_reply","id":1,"outcome":{"outcome":"data","data":2,"rows":3}}"#,
        r#"{"message":"error","error":{"kind":"python","code":"X","message":""}}"#,
        r#"{"message":"step","step":"fetch","requests":[{"id":1,"request":{"request":"describe","tables":[]},"sql":null,"expect":{"expect":"discover","x":1}}]}"#,
        r#"{"message":"warnings","warnings":[{"code":"W","message":"","fields":{"f":{"t":"null","v":"1"}}}]}"#,
    ] {
        assert!(
            serde_json::from_str::<ControlMessage>(text).is_err(),
            "{text} was accepted"
        );
    }
}
