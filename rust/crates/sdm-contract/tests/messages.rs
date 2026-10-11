// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control messages round-trip through JSON, and unknown fields are refused.

mod corpus;

use std::collections::BTreeSet;

use corpus::messages;
use sdm_contract::ControlMessage;
use sdm_contract::data::{
    Determinism, Dialect, OptionalRequest, ProviderCapabilities, ProviderChoice,
};

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
            dialect: Dialect::new("duckdb"),
        },
        ProviderChoice::Host {
            capabilities: ProviderCapabilities {
                optional: BTreeSet::from([OptionalRequest::Preview, OptionalRequest::CountRows]),
                max_keys_per_request: Some(10_000),
                max_seed: None,
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

#[test]
fn a_seed_of_any_size_round_trips_exactly() {
    use sdm_contract::ops::Seed;
    let seed = Seed::from(1_u128 << 70);
    let text = serde_json::to_string(&seed).unwrap();
    assert_eq!(text, r#""0x400000000000000000""#);
    let back: Seed = serde_json::from_str(&text).unwrap();
    assert_eq!(back, seed);
    assert_eq!(back.to_u128(), Some(1_u128 << 70));
    // Past any fixed width, and negative: the decoder takes both; the core
    // decides which seeds a run takes.
    let wide = format!(r#""0x{}""#, "f".repeat(200));
    let wide_seed: Seed = serde_json::from_str(&wide).unwrap();
    assert_eq!(serde_json::to_string(&wide_seed).unwrap(), wide);
    assert_eq!(wide_seed.to_u128(), None);
    let negative: Seed = serde_json::from_str(r#""-0x2a""#).unwrap();
    assert!(negative.is_negative());
    assert_eq!(negative, Seed::from(-42_i128));
    assert_eq!(
        serde_json::to_string(&Seed::from(0_u128)).unwrap(),
        r#""0x0""#
    );
    for refused in [
        r#""42""#,
        r#""0x""#,
        r#""0x02a""#,
        r#""0x2A""#,
        r#""-0x0""#,
        r#""+0x1""#,
        r#""0X2a""#,
        r#""0x 1""#,
        "42",
        r#""-0x00""#,
    ] {
        assert!(
            serde_json::from_str::<Seed>(refused).is_err(),
            "{refused} was accepted"
        );
    }
}

#[test]
fn rows_count_from_one() {
    use sdm_contract::ops::Span;
    let span = |row: u32| format!(r#"{{"start_row":{row},"start_col":0,"end_row":1,"end_col":0}}"#);
    assert!(serde_json::from_str::<Span>(&span(1)).is_ok());
    assert!(serde_json::from_str::<Span>(&span(0)).is_err());
    let schema: serde_json::Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    for row in ["start_row", "end_row"] {
        assert_eq!(
            schema["$defs"]["Span"]["properties"][row]["minimum"], 1,
            "{row}"
        );
    }
}

#[test]
fn every_error_code_is_listed_once_and_spelled_as_the_contract_spells_it() {
    use sdm_contract::ErrorCode;
    // A code added to the enum and not to ALL fails to compile here.
    let listed = |code: ErrorCode| match code {
        ErrorCode::InvalidRequest
        | ErrorCode::UnsupportedRequest
        | ErrorCode::InvalidResponse
        | ErrorCode::InvalidConfiguration
        | ErrorCode::TransportError
        | ErrorCode::InternalError
        | ErrorCode::UnknownModel
        | ErrorCode::MissingExtra
        | ErrorCode::InvalidSchema
        | ErrorCode::SessionsUnsupported
        | ErrorCode::UnsupportedType
        | ErrorCode::ValueOutOfRange
        | ErrorCode::ProviderSchemaMismatch
        | ErrorCode::ProviderError
        | ErrorCode::UnknownReply
        | ErrorCode::ReplyMissing
        | ErrorCode::WrongReply
        | ErrorCode::RunFinished
        | ErrorCode::RunCancelled
        | ErrorCode::RunPoisoned
        | ErrorCode::RunBusy
        | ErrorCode::InvalidMessage => ErrorCode::ALL.contains(&code),
    };
    let mut names = BTreeSet::new();
    for code in ErrorCode::ALL {
        assert!(listed(code));
        let text = serde_json::to_string(&code).unwrap();
        assert_eq!(text, format!("\"{}\"", code.as_str()));
        assert!(
            code.as_str()
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_')
        );
        assert_eq!(serde_json::from_str::<ErrorCode>(&text).unwrap(), code);
        assert!(names.insert(code.as_str()), "{code} twice");
    }
    assert!(serde_json::from_str::<ErrorCode>(r#""NOT_A_CODE""#).is_err());
}

#[test]
fn each_error_kind_s_name_is_its_contract_spelling() {
    use sdm_contract::ErrorKind;
    for kind in [
        ErrorKind::InvalidRequest,
        ErrorKind::Unsupported,
        ErrorKind::NotFound,
        ErrorKind::DataSource,
        ErrorKind::ModelEndpoint,
        ErrorKind::Cancelled,
        ErrorKind::Protocol,
        ErrorKind::Internal,
    ] {
        // A match with no catch-all arm, so a new kind fails to compile
        // here until it is listed.
        match kind {
            ErrorKind::InvalidRequest
            | ErrorKind::Unsupported
            | ErrorKind::NotFound
            | ErrorKind::DataSource
            | ErrorKind::ModelEndpoint
            | ErrorKind::Cancelled
            | ErrorKind::Protocol
            | ErrorKind::Internal => {}
        }
        let text = serde_json::to_string(&kind).unwrap();
        assert_eq!(text, format!("\"{}\"", kind.as_str()));
    }
}

#[test]
fn provider_and_transport_errors_carry_no_message_text() {
    for text in [
        r#"{"message":"fetch_reply","id":1,"outcome":{"outcome":"error","error":{"kind":"data_source","retryable":false,"sqlstate":null,"vendor_code":null,"message":"value 'secret' is bad"}}}"#,
        r#"{"message":"fetch_reply","id":1,"outcome":{"outcome":"error","error":{"kind":"data_source","code":"X","retryable":false,"sqlstate":null,"vendor_code":null}}}"#,
        r#"{"message":"model_reply","id":1,"outcome":{"outcome":"transport_error","kind":"model_endpoint","retryable":true,"message":"timed out"}}"#,
    ] {
        assert!(
            serde_json::from_str::<ControlMessage>(text).is_err(),
            "{text} was accepted"
        );
    }
}

#[test]
fn a_provider_error_keeps_its_retryability_and_codes_in_the_core_error() {
    use sdm_contract::data::ProviderError;
    use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};
    for retryable in [true, false] {
        let error = CoreError::provider(&ProviderError {
            kind: ErrorKind::DataSource,
            retryable,
            sqlstate: Some("40001".into()),
            vendor_code: Some("1213".into()),
        });
        assert_eq!(error.code, ErrorCode::ProviderError);
        assert_eq!(error.kind, ErrorKind::DataSource);
        assert_eq!(error.retryable, retryable);
        assert_eq!(error.fields["sqlstate"], Detail::text("40001"));
        assert_eq!(error.fields["vendor_code"], Detail::text("1213"));
    }
    // An error written without the field is not retryable.
    let error: CoreError =
        serde_json::from_str(r#"{"kind":"internal","code":"INTERNAL_ERROR","message":""}"#)
            .unwrap();
    assert!(!error.retryable);
}
