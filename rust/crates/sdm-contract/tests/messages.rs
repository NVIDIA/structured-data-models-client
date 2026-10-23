// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control messages round-trip through JSON, and unknown fields are refused.

mod corpus;

use std::collections::BTreeSet;

use corpus::messages;
use sdm_contract::ControlMessage;
use sdm_contract::data::{
    Determinism, Dialect, IndexAdvice, NullOrder, OptionalRequest, ProviderCapabilities,
    ProviderChoice,
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
            namespace: Vec::new(),
        },
        ProviderChoice::Host {
            capabilities: ProviderCapabilities {
                opaque_vendor: None,
                compares_stored_times: false,
                samples_by_key_dtype: false,
                reads_nan_as_missing: false,
                derived_relations: false,
                neighbor_lookback: false,
                sample_rows: false,
                optional: BTreeSet::from([OptionalRequest::Preview, OptionalRequest::CountRows]),
                max_keys_per_request: Some(10_000),
                max_seed: u64::MAX,
                text_times: None,
                null_order: Some(NullOrder::First),
                index_advice: Some(IndexAdvice::SingleColumn),
                sampling: Determinism::SeededStable,
                fallback: None,
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

/// A SQL provider's namespace (contract v28): left out, it reads as none
/// and is written back left out, so a v27 choice reads as it did; each
/// part a name or `null`, the source's current one (a Databricks source
/// given its catalog and not its schema); anything else refused.
#[test]
fn a_provider_namespace_takes_names_and_null_parts() {
    for (text, namespace) in [
        (r#"{"provider":"sql","dialect":"postgres"}"#, vec![]),
        (
            r#"{"provider":"sql","dialect":"postgres","namespace":["sales"]}"#,
            vec![Some("sales".to_owned())],
        ),
        (
            r#"{"provider":"sql","dialect":"databricks","namespace":["main",null]}"#,
            vec![Some("main".to_owned()), None],
        ),
    ] {
        let read: ProviderChoice = serde_json::from_str(text).unwrap();
        let ProviderChoice::Sql {
            namespace: held, ..
        } = &read
        else {
            panic!("not a SQL provider: {text}");
        };
        assert_eq!(held, &namespace, "{text}");
        let written: serde_json::Value = serde_json::to_value(&read).unwrap();
        let given: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(written, given, "{text}");
    }
    for refused in [
        r#"{"provider":"sql","dialect":"postgres","namespace":"sales"}"#,
        r#"{"provider":"sql","dialect":"postgres","namespace":[1]}"#,
        r#"{"provider":"host","capabilities":{"optional":[],"max_keys_per_request":null,
            "sampling":"seeded_stable","max_seed":1,"text_times":null},"namespace":["sales"]}"#,
    ] {
        assert!(
            serde_json::from_str::<ProviderChoice>(refused).is_err(),
            "{refused}"
        );
    }
}

/// A table named for describe_source keeps only the columns it lists
/// (contract v28): left out, it reads as none and is written back left out,
/// so a v27 table reads as it did; a list of names reads as given; anything
/// else is refused.
#[test]
fn a_named_table_lists_the_columns_it_keeps_or_none() {
    use sdm_contract::ops::NamedTable;
    for (text, kept) in [
        (
            r#"{"name":"users","source":{"source":["app_users"]}}"#,
            None,
        ),
        (
            r#"{"name":"users","source":{"source":["app_users"]},"columns":["id","target"]}"#,
            Some(vec!["id".to_owned(), "target".to_owned()]),
        ),
    ] {
        let read: NamedTable = serde_json::from_str(text).unwrap();
        assert_eq!(read.columns, kept, "{text}");
        let written = serde_json::to_value(&read).unwrap();
        let given: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(written, given, "{text}");
    }
    for refused in [
        r#"{"name":"users","source":{"source":["app_users"]},"columns":"id"}"#,
        r#"{"name":"users","source":{"source":["app_users"]},"columns":[1]}"#,
        r#"{"name":"users","source":{"source":["app_users"]},"columns":[null]}"#,
    ] {
        assert!(
            serde_json::from_str::<NamedTable>(refused).is_err(),
            "{refused}"
        );
    }
}

/// The sampling policies a host provider declares (T1.26a), by their wire
/// spelling; a provider leaving them out declares none, which the core
/// reads as no null order (an ordered read refused, `null_order`) and no
/// index advice (no index read, no warning), never as DuckDB's.
#[test]
fn the_sampling_policies_are_spelled_and_default_to_none() {
    let declared: ProviderChoice = serde_json::from_str(
        r#"{"provider":"host","capabilities":{"optional":[],"max_keys_per_request":null,
            "sampling":"seeded_stable","max_seed":2147483647,"text_times":null,
            "null_order":"first","index_advice":"single_column"}}"#,
    )
    .unwrap();
    let ProviderChoice::Host { capabilities } = declared else {
        panic!("not a host provider");
    };
    assert_eq!(capabilities.null_order, Some(NullOrder::First));
    assert_eq!(capabilities.index_advice, Some(IndexAdvice::SingleColumn));
    let silent: ProviderChoice = serde_json::from_str(
        r#"{"provider":"host","capabilities":{"optional":[],"max_keys_per_request":null,
            "sampling":"seeded_stable","max_seed":2147483647}}"#,
    )
    .unwrap();
    let ProviderChoice::Host { capabilities } = silent else {
        panic!("not a host provider");
    };
    assert_eq!(capabilities.null_order, None);
    assert_eq!(capabilities.index_advice, None);
    for wrong in [
        r#""null_order":"nulls_first""#,
        r#""null_order":true"#,
        r#""index_advice":"multi_column""#,
        r#""index_advice":true"#,
    ] {
        let text = format!(
            r#"{{"provider":"host","capabilities":{{"optional":[],"max_keys_per_request":null,"sampling":"seeded_stable","max_seed":2147483647,{wrong}}}}}"#
        );
        assert!(
            serde_json::from_str::<ProviderChoice>(&text).is_err(),
            "{text}"
        );
    }
    // Every provider declares its largest seed (contract v26, design
    // choice 1): no "any seed" default, so null or left out is refused.
    for seed in [r#","max_seed":null"#, ""] {
        let text = format!(
            r#"{{"provider":"host","capabilities":{{"optional":[],"max_keys_per_request":null,"sampling":"seeded_stable"{seed}}}}}"#
        );
        assert!(
            serde_json::from_str::<ProviderChoice>(&text).is_err(),
            "{text}"
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
            refuses_expression: false,
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
    let error: CoreError = serde_json::from_str(
        r#"{"kind":"internal","code":"INTERNAL_ERROR","sentence":"internal","message":""}"#,
    )
    .unwrap();
    assert!(!error.retryable);
}
