// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The control codecs: a message round-trips; one from the host past the
//! 64 KB cap, or not of the contract, is refused before it is read; one the
//! core writes past the cap travels whole as an attachment (decision H7).

use std::collections::BTreeMap;

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::{LocalZone, PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedSource, DescribedTable, Dtype, Operation, OperationResult,
    PqlDiagnostic, ProfileName, ProfileSpec, QueryValidation, Severity, SourceDescription, Stype,
    ValidateQuery,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, CoreWarning, Detail, ErrorCode,
    ErrorKind, HostFacts, Scalar, Step, Unit,
};
use sdm_core::Attachment;
use sdm_host::{CONTROL_CAP, Outgoing, decode, encode, read, write, write_error};

/// A start whose profile names `model`, the one free-text field of its
/// envelope.
fn start(model: String) -> ControlMessage {
    ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("UTC".into()),
                value: 0,
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
        },
        operation: Operation::ValidateQuery(ValidateQuery {
            query: AttachmentId(1),
            schema: AttachmentId(2),
            profile: ProfileSpec {
                name: ProfileName::Relational,
                model: Some(model),
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
    let message = start("rfm".into());
    assert_eq!(decode(&encode(&message).unwrap()).unwrap(), message);
}

/// A start message of exactly `size` bytes, as a host might write one.
fn start_of(size: usize) -> Vec<u8> {
    let written = |model: String| serde_json::to_vec(&start(model)).unwrap();
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
        (err.kind, err.code),
        (ErrorKind::Protocol, ErrorCode::InvalidMessage)
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
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::InvalidMessage),
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

/// The message `make` builds, padded so its JSON is exactly `size` bytes.
fn exactly(size: usize, make: impl Fn(usize) -> ControlMessage) -> ControlMessage {
    let base = serde_json::to_vec(&make(0)).unwrap().len();
    let message = make(size - base);
    assert_eq!(serde_json::to_vec(&message).unwrap().len(), size);
    message
}

fn no_id() -> Result<AttachmentId, CoreError> {
    panic!("a message within the cap needs no attachment")
}

/// Writes `message` as the core does and reads it back as a host does,
/// checking whether it travelled in the control message or as an
/// attachment.
fn written_whole(message: &ControlMessage, attached: bool) -> Outgoing {
    let out = if attached {
        write(message, || Ok(AttachmentId(5))).unwrap()
    } else {
        write(message, no_id).unwrap()
    };
    assert!(out.control.len() <= CONTROL_CAP);
    assert_eq!(out.attachments.is_empty(), !attached);
    if attached {
        assert_eq!(
            decode(&out.control).unwrap(),
            ControlMessage::Attached {
                attachment: AttachmentId(5)
            }
        );
    }
    assert_eq!(&read(&out).unwrap(), message);
    out
}

#[test]
fn the_core_writes_a_message_whole_within_the_cap_or_as_an_attachment() {
    let at_cap = exactly(CONTROL_CAP, done_with_message_of);
    written_whole(&at_cap, false);
    let past_cap = exactly(CONTROL_CAP + 1, done_with_message_of);
    written_whole(&past_cap, true);
    written_whole(&done_with_message_of(4 * CONTROL_CAP), true);
}

/// A source description whose JSON is exactly `size` bytes.
fn description_of(size: usize) -> Vec<u8> {
    let make = |pad: usize| SourceDescription {
        tables: vec![DescribedTable {
            name: "t".repeat(pad + 1),
            columns: vec![DescribedColumn {
                name: "c".into(),
                received: ColumnType::Int64,
                dtype: Dtype::Int,
                stype: Stype::Numerical,
                stype_before_key: None,
                nullable: true,
                unique: false,
            }],
            primary_key: vec![],
            time_column: None,
            edit_stamp_refused: None,
            duplicate_keys: None,
        }],
        links: vec![vec![ColumnRef {
            table: "t".into(),
            column: "c".into(),
        }]],
    };
    let base = serde_json::to_vec(&make(0)).unwrap().len();
    let bytes = serde_json::to_vec(&make(size - base)).unwrap();
    assert_eq!(bytes.len(), size);
    bytes
}

#[test]
fn a_done_carrying_a_source_description_arrives_whole_either_side_of_the_cap() {
    // The description is input-sized, so it is an attachment of its JSON
    // beside a Done that names it (decision H5); the Done stays small.
    let mut lengths = std::collections::BTreeSet::new();
    for size in [
        CONTROL_CAP - 1,
        CONTROL_CAP,
        CONTROL_CAP + 1,
        3 * CONTROL_CAP,
    ] {
        let description = description_of(size);
        let done = ControlMessage::Step(Step::Done {
            result: OperationResult::DescribeSource(DescribedSource {
                description: AttachmentId(4),
            }),
        });
        let mut out = written_whole(&done, false);
        lengths.insert(out.control.len());
        out.attachments
            .insert(AttachmentId(4), Attachment::Bytes(description.clone()));
        let ControlMessage::Step(Step::Done {
            result: OperationResult::DescribeSource(described),
        }) = read(&out).unwrap()
        else {
            panic!("a done")
        };
        let Attachment::Bytes(bytes) = &out.attachments[&described.description] else {
            panic!("bytes")
        };
        let back: SourceDescription = serde_json::from_slice(bytes).unwrap();
        assert_eq!(serde_json::to_vec(&back).unwrap(), description);
    }
    // The control message is the same whatever the description's size.
    assert_eq!(lengths.len(), 1, "{lengths:?}");
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
    let err = write(
        &ControlMessage::Warnings {
            warnings: vec![warning],
        },
        no_id,
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::InternalError);
}

fn error_of(size: usize) -> ControlMessage {
    exactly(size, |pad| ControlMessage::Error {
        error: CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, "small")
            .with_field("table", Detail::text("x".repeat(pad)))
            .caused_by(CoreError::new(
                ErrorKind::DataSource,
                ErrorCode::ProviderError,
                "cause",
            )),
    })
}

#[test]
fn an_error_is_written_whole_with_its_fields_and_cause_either_side_of_the_cap() {
    for (size, attached) in [(CONTROL_CAP, false), (CONTROL_CAP + 1, true)] {
        let ControlMessage::Error { error } = error_of(size) else {
            unreachable!()
        };
        let out = write_error(&error, || Ok(AttachmentId(2)));
        assert!(out.control.len() <= CONTROL_CAP);
        assert_eq!(out.attachments.is_empty(), !attached);
        assert_eq!(read(&out).unwrap(), ControlMessage::Error { error });
    }
}

#[test]
fn an_error_the_encoder_refuses_becomes_an_internal_error() {
    let bad = CoreError::new(ErrorKind::DataSource, ErrorCode::ProviderError, "bad").with_field(
        "d",
        Detail::Scalar(Scalar::Decimal128 {
            precision: 0,
            scale: 0,
            value: 1,
        }),
    );
    let out = write_error(&bad, no_id);
    let ControlMessage::Error { error } = read(&out).unwrap() else {
        panic!("an error message")
    };
    assert_eq!(error.code, ErrorCode::InternalError);
    // And a run out of ids cannot attach: that is said, within the cap.
    let ControlMessage::Error { error: big } = error_of(CONTROL_CAP + 1) else {
        unreachable!()
    };
    let out = write_error(&big, || {
        Err(CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "out of ids",
        ))
    });
    let ControlMessage::Error { error } = read(&out).unwrap() else {
        panic!("an error message")
    };
    assert_eq!(
        (error.code, error.message.as_str()),
        (ErrorCode::InternalError, "out of ids")
    );
}

#[test]
fn warnings_are_written_in_order_in_one_message_of_any_size() {
    let warnings: Vec<CoreWarning> = (0..40)
        .map(|i| CoreWarning {
            code: "LONG".into(),
            message: format!("{i:02}{}", "w".repeat(4_000)),
            fields: BTreeMap::new(),
        })
        .collect();
    written_whole(&ControlMessage::Warnings { warnings }, true);
}
