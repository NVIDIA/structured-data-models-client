// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The control codecs: a message round-trips, one past the 64 KB cap or not
//! of the contract is refused before it is read, and the core never writes
//! one past the cap.

use std::collections::BTreeMap;

use sdm_contract::facts::{PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{
    Operation, OperationResult, PqlDiagnostic, ProfileName, ProfileSpec, QuerySchema,
    QueryValidation, Severity, ValidateQuery,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, CoreError, CoreWarning, Detail, ErrorKind, HostFacts, Scalar,
    Step, Unit, codes,
};
use sdm_host::{CONTROL_CAP, decode, encode, encode_error, encode_warnings};

fn start(query: String) -> ControlMessage {
    ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("UTC".into()),
                value: 0,
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
        },
        operation: Operation::ValidateQuery(ValidateQuery {
            query,
            schema: QuerySchema {
                tables: vec![],
                links: vec![],
            },
            profile: ProfileSpec {
                name: ProfileName::Relational,
                model: None,
                capabilities: BTreeMap::new(),
            },
        }),
    }))
}

fn count(error: &CoreError, field: &str) -> u64 {
    match error.fields[field] {
        Detail::Scalar(Scalar::UInt64(n)) => n,
        ref other => panic!("{field} is {other:?}"),
    }
}

#[test]
fn a_message_round_trips() {
    let message = start("PREDICT users.age FOR EACH users.user_id".into());
    assert_eq!(decode(&encode(&message).unwrap()).unwrap(), message);
}

/// A start message of exactly `size` bytes, as a host might write one.
fn start_of(size: usize) -> Vec<u8> {
    let written = |query: String| serde_json::to_vec(&start(query)).unwrap();
    let empty = written(String::new()).len();
    let bytes = written("x".repeat(size - empty));
    assert_eq!(bytes.len(), size);
    bytes
}

#[test]
fn a_message_up_to_the_cap_is_read_and_one_past_it_is_refused() {
    assert!(decode(&start_of(CONTROL_CAP)).is_ok());
    let err = decode(&start_of(CONTROL_CAP + 1)).unwrap_err();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Protocol, codes::INVALID_MESSAGE)
    );
    assert_eq!(count(&err, "bytes"), (CONTROL_CAP + 1) as u64);
    assert_eq!(count(&err, "cap"), CONTROL_CAP as u64);
}

#[test]
fn a_message_not_of_the_contract_is_refused() {
    for text in [
        &b"not json"[..],
        b"{}",
        br#"{"message":"cancel","extra":1}"#,
        br#"{"message":"step","step":"ask_host","what":"sort"}"#,
        b"\xff\xfe",
    ] {
        let err = decode(text).unwrap_err();
        assert_eq!(
            (err.kind, err.code.as_str()),
            (ErrorKind::Protocol, codes::INVALID_MESSAGE),
            "{}",
            String::from_utf8_lossy(text)
        );
    }
}

fn done_with_message_of(size: usize) -> ControlMessage {
    ControlMessage::Step(Step::Done {
        result: OperationResult::ValidateQuery(QueryValidation {
            accepted: false,
            diagnostics: vec![PqlDiagnostic {
                code: "PQL_PARSE_SYNTAX".into(),
                message: "m".repeat(size),
                severity: Severity::Error,
                span: None,
            }],
        }),
    })
}

#[test]
fn the_core_never_writes_past_the_cap() {
    assert!(encode(&done_with_message_of(1_000)).is_ok());
    let err = encode(&done_with_message_of(CONTROL_CAP)).unwrap_err();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Internal, codes::INTERNAL_ERROR)
    );
    assert!(count(&err, "bytes") > CONTROL_CAP as u64);
}

#[test]
fn the_core_never_writes_what_its_contract_refuses() {
    // A decimal of precision 0 is outside the contract (decision E4).
    let warning = CoreWarning {
        code: "W".into(),
        message: String::new(),
        fields: BTreeMap::from([(
            "d".to_owned(),
            Detail::Scalar(Scalar::Decimal128 {
                precision: 0,
                scale: 0,
                value: 1,
            }),
        )]),
    };
    let err = encode(&ControlMessage::Warnings {
        warnings: vec![warning],
    })
    .unwrap_err();
    assert_eq!(err.code, codes::INTERNAL_ERROR);
}

#[test]
fn an_error_is_always_written_within_the_cap() {
    let big = CoreError::new(
        ErrorKind::DataSource,
        "PROVIDER_ERROR",
        "x".repeat(CONTROL_CAP),
    )
    .with_field("table", Detail::text("orders"));
    let caused =
        CoreError::new(ErrorKind::Internal, codes::INTERNAL_ERROR, "small").caused_by(big.clone());
    for (error, keeps_fields) in [(caused, true), (big, false)] {
        let bytes = encode_error(&error);
        assert!(bytes.len() <= CONTROL_CAP);
        let ControlMessage::Error { error: written } = decode(&bytes).unwrap() else {
            panic!("not an error message");
        };
        assert_eq!((written.kind, &written.code), (error.kind, &error.code));
        assert_eq!(written.fields == error.fields, keeps_fields);
    }
}

#[test]
fn warnings_are_written_in_order_within_the_cap() {
    assert!(encode_warnings(vec![]).unwrap().is_empty());
    let warnings: Vec<CoreWarning> = (0..40)
        .map(|i| CoreWarning {
            code: "LONG".into(),
            message: format!("{i:02}{}", "w".repeat(4_000)),
            fields: BTreeMap::new(),
        })
        .collect();
    let messages = encode_warnings(warnings.clone()).unwrap();
    assert!(messages.len() > 1);
    let mut read = Vec::new();
    for bytes in messages {
        assert!(bytes.len() <= CONTROL_CAP);
        let ControlMessage::Warnings { warnings } = decode(&bytes).unwrap() else {
            panic!("not a warnings message");
        };
        read.extend(warnings);
    }
    assert_eq!(read, warnings);
    // One warning alone past the cap is the core's defect.
    let huge = CoreWarning {
        code: "HUGE".into(),
        message: "w".repeat(CONTROL_CAP),
        fields: BTreeMap::new(),
    };
    assert_eq!(
        encode_warnings(vec![huge]).unwrap_err().code,
        codes::INTERNAL_ERROR
    );
}
