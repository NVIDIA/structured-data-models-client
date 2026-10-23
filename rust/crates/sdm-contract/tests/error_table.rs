// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The table of the kinds each code may carry (decision E22): every code
//! and every kind against it, through each public constructor.

use std::panic::catch_unwind;

use sdm_contract::data::ProviderError;
use sdm_contract::{Code, CoreError, Detail, ErrorCode, ErrorKind, Sentence};

/// The table as this test states it, one row per code; the crate's must
/// agree with it pair by pair.
fn expected(code: ErrorCode) -> &'static [ErrorKind] {
    use ErrorCode as C;
    use ErrorKind as K;
    match code {
        C::InvalidRequest | C::InvalidConfiguration | C::InvalidSchema => &[K::InvalidRequest],
        C::UnsupportedRequest
        | C::MissingExtra
        | C::SessionsUnsupported
        | C::UnsupportedType
        | C::ValueOutOfRange => &[K::Unsupported],
        C::UnknownModel => &[K::NotFound],
        C::InvalidResponse | C::TransportError => &[K::ModelEndpoint],
        C::InternalError => &[K::Internal],
        C::ProviderSchemaMismatch => &[K::DataSource],
        C::ProviderError => &[K::DataSource, K::NotFound],
        C::RunCancelled => &[K::Cancelled],
        C::UnknownReply
        | C::ReplyMissing
        | C::WrongReply
        | C::RunFinished
        | C::RunPoisoned
        | C::RunBusy
        | C::InvalidMessage => &[K::Protocol],
    }
}

#[test]
fn every_code_and_kind_is_held_to_the_table() {
    for code in ErrorCode::ALL {
        assert!(!code.kinds().is_empty(), "{code} carries no kind");
        for kind in ErrorKind::ALL {
            assert_eq!(
                code.allows(kind),
                expected(code).contains(&kind),
                "{code} and {}",
                kind.as_str()
            );
        }
    }
}

#[test]
fn the_lists_of_codes_and_kinds_are_whole_and_spelled_as_serde_spells_them() {
    for code in ErrorCode::ALL {
        let spelled = serde_json::to_string(&code).unwrap();
        assert_eq!(spelled, format!("\"{}\"", code.as_str()));
    }
    for kind in ErrorKind::ALL {
        let spelled = serde_json::to_string(&kind).unwrap();
        assert_eq!(spelled, format!("\"{}\"", kind.as_str()));
    }
    let mut codes = ErrorCode::ALL.to_vec();
    codes.sort();
    codes.dedup();
    assert_eq!(codes.len(), ErrorCode::ALL.len());
    let mut kinds = ErrorKind::ALL.to_vec();
    kinds.sort();
    kinds.dedup();
    assert_eq!(kinds.len(), ErrorKind::ALL.len());
}

/// Every sentence through `new` and `protocol` (T3.17): an error's
/// sentence builds with its code and the code's kind; a warning's, and
/// through `protocol` a sentence of another kind, fail the assertion in a
/// debug build.
#[test]
fn new_and_protocol_build_only_the_table_s_pairs() {
    for &sentence in Sentence::ALL {
        let built = catch_unwind(|| CoreError::new(sentence, "m"));
        match sentence.code() {
            Code::Error(code) => {
                let error = built.expect("an error's sentence builds");
                assert!(error.holds_to_table());
                assert_eq!((error.kind, error.code), (code.kinds()[0], code));
                let built = catch_unwind(|| CoreError::protocol(sentence, "m"));
                if code.allows(ErrorKind::Protocol) {
                    assert_eq!(built.unwrap().kind, ErrorKind::Protocol);
                } else if cfg!(debug_assertions) {
                    assert!(built.is_err(), "protocol {code} built");
                }
            }
            Code::Warning(_) if cfg!(debug_assertions) => {
                assert!(built.is_err(), "{} built", sentence.as_str());
            }
            Code::Warning(_) => assert_eq!(built.unwrap().code, ErrorCode::InternalError),
        }
    }
}

/// A provider's or a transport's kind comes from the host: one its code
/// does not carry gives `INTERNAL_ERROR` naming the pair.
#[test]
fn provider_and_transport_errors_outside_the_table_are_internal() {
    for kind in ErrorKind::ALL {
        let provided = CoreError::provider(&ProviderError {
            kind,
            retryable: true,
            sqlstate: Some("42P01".into()),
            vendor_code: None,
        });
        let carried = CoreError::transport(kind, true);
        for (error, code) in [
            (provided, ErrorCode::ProviderError),
            (carried, ErrorCode::TransportError),
        ] {
            assert!(error.holds_to_table());
            if code.allows(kind) {
                assert_eq!((error.kind, error.code), (kind, code));
                assert!(error.retryable);
            } else {
                assert_eq!(
                    (error.kind, error.code),
                    (ErrorKind::Internal, ErrorCode::InternalError)
                );
                assert!(!error.retryable);
                assert_eq!(error.fields["pair_kind"], Detail::text(kind.as_str()));
                assert_eq!(error.fields["pair_code"], Detail::text(code.as_str()));
            }
        }
    }
}

#[test]
fn outside_table_names_any_pair_and_holds_to_the_table_itself() {
    for code in ErrorCode::ALL {
        for kind in ErrorKind::ALL {
            let error = CoreError::outside_table(kind, code);
            assert_eq!(error.code, ErrorCode::InternalError);
            assert!(error.holds_to_table());
        }
    }
}

/// A deserialized error can carry any pair; `holds_to_table` finds one
/// outside the table, in the error or anywhere in its cause chain.
#[test]
fn a_pair_outside_the_table_is_found_in_the_cause_chain() {
    let outside: CoreError = serde_json::from_str(
        r#"{"kind":"internal","code":"RUN_BUSY","sentence":"run_busy","message":"m"}"#,
    )
    .unwrap();
    assert!(!outside.holds_to_table());
    // A sentence of another code than the error's is outside it too.
    let crossed: CoreError = serde_json::from_str(
        r#"{"kind":"internal","code":"INTERNAL_ERROR","sentence":"run_busy","message":"m"}"#,
    )
    .unwrap();
    assert!(!crossed.holds_to_table());
    let wrapped = CoreError::new(Sentence::Internal, "m").caused_by(outside);
    assert!(!wrapped.holds_to_table());
    let unsupported = sdm_contract::types::unsupported_type(
        "c",
        &arrow_schema::DataType::Interval(arrow_schema::IntervalUnit::DayTime),
    );
    assert!(unsupported.holds_to_table());
}
