// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kinds each code may carry, checked where an error crosses the
//! boundary (decision E22): an outgoing error outside the table becomes
//! `INTERNAL_ERROR` naming the pair, a host-to-core message carrying one is
//! refused with `INVALID_MESSAGE`, and a provider's kind outside its
//! code's set ends that fetch as `INTERNAL_ERROR`, never as a refusal of
//! the reply. No Phase 1a operation calls the model, so the transport's
//! kind check is `CoreError::transport`'s, tested in `sdm-contract`.

use std::collections::BTreeMap;
use std::sync::Arc;

use sdm_contract::data::{AttachmentId, DataRequest, ExpectedSchema, FetchPlan, ProviderError};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::run::{FetchOutcome, FetchReply, ReplyId, StartRun};
use sdm_contract::test_ops::{TestFetch, TestOperation};
use sdm_contract::{
    CONTRACT_VERSION, ControlMessage, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar,
    Unit,
};
use sdm_core::{Attachment, Run};
use sdm_host::{Attachments, HostRun, Outgoing, ZoneResolver, encode, read};

fn start_message(operation: TestOperation) -> Vec<u8> {
    encode(&ControlMessage::Start(Box::new(StartRun {
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
            date_literal_range: None,
        },
        operation: sdm_contract::ops::Operation::Test(operation),
    })))
    .unwrap()
}

fn error_of(out: &Outgoing) -> CoreError {
    match read(out).unwrap() {
        ControlMessage::Error { error } => error,
        other => panic!("not an error: {other:?}"),
    }
}

/// A `CoreError` of any pair, as a host could write one.
fn any_pair(kind: ErrorKind, code: ErrorCode) -> CoreError {
    serde_json::from_value(serde_json::json!({
        "kind": kind, "code": code, "message": "m",
    }))
    .unwrap()
}

#[test]
fn an_outgoing_error_outside_the_table_is_internal_naming_the_pair() {
    for (outside, wrap) in [
        (any_pair(ErrorKind::DataSource, ErrorCode::RunBusy), false),
        (
            any_pair(ErrorKind::Protocol, ErrorCode::UnsupportedType),
            true,
        ),
    ] {
        let (kind, code) = (outside.kind, outside.code);
        let error = if wrap {
            CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, "m").caused_by(outside)
        } else {
            outside
        };
        let run = HostRun::with(
            Run::new(move |_| async move { Err(error) }),
            Arc::new(ZoneResolver::new([])),
        );
        let written = error_of(&run.next().unwrap_err());
        assert_eq!(
            (written.kind, written.code),
            (ErrorKind::Internal, ErrorCode::InternalError)
        );
        assert_eq!(written.fields["pair_kind"], Detail::text(kind.as_str()));
        assert_eq!(written.fields["pair_code"], Detail::text(code.as_str()));
        assert!(written.cause.is_none());
    }
}

#[test]
fn an_outgoing_error_in_the_table_is_written_as_it_is() {
    for code in ErrorCode::ALL {
        for &kind in code.kinds() {
            let error = CoreError::new(kind, code, "m");
            let run = HostRun::with(
                Run::new(move |_| async move { Err(error) }),
                Arc::new(ZoneResolver::new([])),
            );
            let written = error_of(&run.next().unwrap_err());
            assert_eq!((written.kind, written.code), (kind, code));
        }
    }
}

fn fail_start(error: &CoreError) -> Result<HostRun, Outgoing> {
    HostRun::start(
        &sdm_host::default_root(),
        &start_message(TestOperation::Fail {
            error: AttachmentId(1),
        }),
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(error).unwrap()),
        )]),
    )
}

#[test]
fn a_host_message_carrying_an_error_outside_the_table_is_refused() {
    for code in ErrorCode::ALL {
        for kind in ErrorKind::ALL {
            let started = fail_start(&any_pair(kind, code));
            if code.allows(kind) {
                let error = error_of(&started.unwrap().next().unwrap_err());
                assert_eq!((error.kind, error.code), (kind, code));
            } else {
                let refused = error_of(&started.err().expect("refused"));
                assert_eq!(refused.code, ErrorCode::InvalidMessage, "{code}");
                assert_eq!(
                    refused.fields["attachment"],
                    Detail::Scalar(Scalar::UInt32(1))
                );
            }
        }
    }
}

#[test]
fn an_error_message_sent_to_the_core_is_refused() {
    let message = encode(&ControlMessage::Error {
        error: any_pair(ErrorKind::DataSource, ErrorCode::RunBusy),
    })
    .unwrap();
    let refused = HostRun::start(&sdm_host::default_root(), &message, Attachments::new());
    assert_eq!(
        error_of(&refused.err().expect("refused")).code,
        ErrorCode::InvalidMessage
    );
    let run = fetch_run();
    let refused = run.reply(&message, Attachments::new()).unwrap_err();
    assert_eq!(error_of(&refused).code, ErrorCode::InvalidMessage);
}

fn fetch_run() -> HostRun {
    let fetch = TestFetch {
        request: DataRequest::Scan {
            table: "t".into(),
            columns: vec![],
            filters: vec![],
            limit: None,
        },
        plan: Some(FetchPlan::Host {
            expect: ExpectedSchema::Discover {},
        }),
        provider: None,
        echo: false,
        undeclared: vec![],
        fallback: None,
    };
    let run = HostRun::start(
        &sdm_host::default_root(),
        &start_message(TestOperation::Fetch {
            fetch: AttachmentId(1),
        }),
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&fetch).unwrap()),
        )]),
    )
    .unwrap();
    run.next().unwrap();
    run
}

#[test]
fn a_provider_s_kind_outside_its_code_s_set_ends_the_fetch_as_internal() {
    for kind in ErrorKind::ALL {
        let run = fetch_run();
        let reply = encode(&ControlMessage::FetchReply(FetchReply {
            id: ReplyId(1),
            outcome: FetchOutcome::Error {
                error: ProviderError {
                    refuses_expression: false,
                    kind,
                    retryable: true,
                    sqlstate: None,
                    vendor_code: None,
                },
            },
        }))
        .unwrap();
        // The reply is the fetch's outcome, never refused.
        run.reply(&reply, Attachments::new()).unwrap();
        let error = error_of(&run.next().unwrap_err());
        if ErrorCode::ProviderError.allows(kind) {
            assert_eq!((error.kind, error.code), (kind, ErrorCode::ProviderError));
            assert!(error.retryable);
        } else {
            assert_eq!(error.code, ErrorCode::InternalError, "{}", kind.as_str());
            assert!(!error.retryable);
        }
    }
}
