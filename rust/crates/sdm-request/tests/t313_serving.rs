// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks Model Serving endpoint's envelope and answers, on the core
//! (T3.13), as the old client's serving transport writes and reads them
//! (`CL/core/databricks_serving.py`): the call wrapped as one
//! `request_json` row, written as Python's `json.dumps` writes it and
//! weighed against the caller's limit; the answer read from its one
//! `response_json` row; a failure the served model answered with, or the
//! endpoint's HTTP or SDK failure, named by its status and codes alone. The
//! expected texts were written by the reference's own functions at
//! 43e55f4f (`_envelope`, `_decoded`, `_error_from`) and the Databricks
//! SDK's error parser, on the same inputs.

use sdm_contract::error::{REFUSED_CODE, REFUSED_STATUS};
use sdm_contract::{CoreError, Detail, ErrorCode, Scalar, Sentence};
use sdm_request::serving::{self, Failure, REQUEST_COLUMN};
use sdm_wire::json::Value;

const ENDPOINT: &str = "kumo-rfm-serving";
const SECRET: &str = "Sup3rSecret-CustomerValue-9f2a";

fn envelope(call: &str, limit: u64) -> Result<String, CoreError> {
    serving::envelope(ENDPOINT, limit, call.as_bytes())
        .map(|bytes| String::from_utf8(bytes).unwrap())
}

fn call_in(envelope: &str) -> Value {
    let read = sdm_wire::json::read(envelope.as_bytes()).unwrap();
    let Value::Array(rows) = read.get("dataframe_records").unwrap() else {
        panic!("one request row");
    };
    sdm_wire::json::read(
        rows[0]
            .get(REQUEST_COLUMN)
            .unwrap()
            .as_text()
            .unwrap()
            .as_bytes(),
    )
    .unwrap()
}

fn text(value: &str) -> Detail {
    Detail::text(value)
}

fn count(value: u64) -> Detail {
    Detail::Scalar(Scalar::UInt64(value))
}

/// Every text an error holds.
fn written(error: &CoreError) -> String {
    let mut out = serde_json::to_string(error).unwrap();
    out.push_str(&error.to_string());
    out
}

/// The refusal `error` is: its sentence, the status the host raises it
/// under and the code, as the old client's `NimRequestError(status,
/// code=...)`; it never repeats what the endpoint said.
fn refusal(error: &CoreError) -> (Sentence, u16, Option<String>) {
    assert_eq!(error.code, ErrorCode::EndpointRefused, "{error:?}");
    assert!(error.holds_to_table());
    assert_eq!(error.fields["endpoint"], text(ENDPOINT));
    for said in [SECRET, "NVDA_SECRET", "schema is invalid", "Jane", "123-45"] {
        assert!(!written(error).contains(said), "{}", written(error));
    }
    let Detail::Scalar(Scalar::UInt16(status)) = error.fields[REFUSED_STATUS] else {
        panic!("a status: {error:?}");
    };
    let code = error.fields.get(REFUSED_CODE).map(|code| match code {
        Detail::Scalar(Scalar::Utf8(code)) => code.clone(),
        other => panic!("a code: {other:?}"),
    });
    (error.sentence, status, code)
}

/// A refusal expected: its sentence, status and code.
type Expected<'a> = (Sentence, u16, Option<&'a str>);

fn field<'a>(error: &'a CoreError, name: &str) -> Option<&'a Detail> {
    error.fields.get(name)
}

// --- The envelope -------------------------------------------------------------

#[test]
fn the_shared_codec_preserves_the_call_and_the_recorded_reference_logical_payload() {
    // Python's json.dumps: every character past ASCII and DEL escaped, a
    // character past the BMP as its surrogate pair, a float as its repr
    // (1E5 as 100000.0, 12345678.900000001 as 12345678.9), -0 as 0, then
    // the whole wrapped again, every quote and backslash escaped once more.
    let call = concat!(
        r#"{"a":"é😀"#,
        "\u{7f}",
        r#"","b":[1,2.5,1e-05,-0,1E5,1.0e+20,12345678.900000001],"q":"\"\\/","c":"\n\u0001"#,
        "\u{2028}",
        r#"\t"}"#
    );
    let expected = r#"{"dataframe_records":[{"request_json":"{\"a\":\"\\u00e9\\ud83d\\ude00\\u007f\",\"b\":[1,2.5,1e-05,0,100000.0,1e+20,12345678.9],\"q\":\"\\\"\\\\/\",\"c\":\"\\n\\u0001\\u2028\\t\"}"}]}"#;
    assert_eq!(expected.len(), 182);
    let wrapped = envelope(call, 16_000_000).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&wrapped).unwrap();
    let reference: serde_json::Value = serde_json::from_str(expected).unwrap();
    let actual_call = actual["dataframe_records"][0][REQUEST_COLUMN]
        .as_str()
        .unwrap();
    let reference_call = reference["dataframe_records"][0][REQUEST_COLUMN]
        .as_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(actual_call).unwrap(),
        serde_json::from_str::<serde_json::Value>(reference_call).unwrap()
    );
    assert_eq!(
        sdm_wire::json::read(actual_call.as_bytes()).unwrap(),
        sdm_wire::json::read(call.as_bytes()).unwrap()
    );
    assert!(!wrapped.is_ascii());
    // Members in order, a name past ASCII escaped too, an integer past
    // every fixed width as its digits.
    let call = r#"{"kü":null,"t":true,"f":false,"n":-12345678901234567890123}"#;
    let reference = r#"{"dataframe_records":[{"request_json":"{\"k\\u00fc\":null,\"t\":true,\"f\":false,\"n\":-12345678901234567890123}"}]}"#;
    assert_eq!(
        call_in(&envelope(call, 16_000_000).unwrap()),
        call_in(reference)
    );
}

#[test]
fn the_caller_s_limit_counts_the_wrapped_envelope_at_its_boundary() {
    // A limit other than the gateway's 16,000,000, met exactly and missed
    // by one, on a call whose escaped Unicode grows sevenfold once wrapped.
    let call = concat!(
        r#"{"rows":["é😀\""#,
        "Sup3rSecret-CustomerValue-9f2a",
        r#"\""]}"#
    );
    let wrapped = envelope(call, 16_000_000).unwrap();
    let size = wrapped.len() as u64;
    let reference = concat!(
        r#"{"dataframe_records":[{"request_json":"{\"rows\":[\"\\u00e9\\ud83d\\ude00\\\""#,
        "Sup3rSecret-CustomerValue-9f2a",
        r#"\\\"\"]}"}]}"#
    );
    assert_eq!(reference.len(), 119);
    assert_eq!(call_in(&wrapped), call_in(reference));
    assert!(size < reference.len() as u64);
    assert_eq!(envelope(call, size).unwrap(), wrapped);
    let refused = envelope(call, size - 1).unwrap_err();
    assert_eq!(refused.sentence, Sentence::ServingPastLimit);
    assert_eq!(refused.code, ErrorCode::InvalidRequest);
    assert_eq!(refused.fields["bytes"], count(size));
    assert_eq!(refused.fields["limit"], count(size - 1));
    assert_eq!(refused.fields["endpoint"], text(ENDPOINT));
    assert!(!written(&refused).contains(SECRET));
}

#[test]
fn the_utf8_envelope_keeps_number_tokens_and_exactly_counts_the_sent_bytes() {
    let call = r#"{"x":"é😀","n":-0,"i":1844674407370955161600000,"f":1e+10}"#;
    let expected = r#"{"dataframe_records":[{"request_json":"{\"x\":\"é😀\",\"n\":-0,\"i\":1844674407370955161600000,\"f\":1e+10}"}]}"#;
    let limit = expected.len() as u64;
    assert_eq!(envelope(call, limit).unwrap(), expected);
    let error = envelope(call, limit - 1).unwrap_err();
    assert_eq!(error.fields["bytes"], count(limit));
    assert_eq!(error.fields["limit"], count(limit - 1));
    assert_eq!(
        call_in(expected),
        sdm_wire::json::read(call.as_bytes()).unwrap()
    );
    for call in [r#"{"x":1e400}"#, r#"{"x":[NaN]}"#, r#"{"x":Infinity}"#] {
        let error = envelope(call, u64::MAX).unwrap_err();
        assert_eq!(error.sentence, Sentence::ServingValueNotJson);
        assert_eq!(error.code, ErrorCode::InvalidRequest);
    }
}

#[test]
fn diagnostic_fallback_uses_nonblank_text_and_ascii_phrases_only() {
    for detail in ["null", "false", "0", "1", "[]", "{}", r#""  \t ""#] {
        let body = format!(
            r#"{{"status":503,"detail":{detail},"title":"FAILED TO COLLECT ANY CONTEXT EXAMPLES"}}"#
        );
        let error = serving::answer(ENDPOINT, &served(&body)).unwrap_err();
        assert_eq!(error.sentence, Sentence::ServingCallersFault, "{detail}");
        assert_eq!(error.fields["endpoint_code"], Detail::Scalar(Scalar::Null));
    }
    let error = serving::answer(
        ENDPOINT,
        &served(r#"{"status":503,"detail":"failed to collect any context exampleſ"}"#),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::ServingFailed);
    let error = serving::answer(
        ENDPOINT,
        &served(r#"{"status":500,"code":"ABC\n","detail":"private data"}"#),
    )
    .unwrap_err();
    assert_eq!(error.sentence, Sentence::ServingFailed);
}

#[test]
fn known_ascii_codes_are_canonical_and_arbitrary_lowercase_text_is_not_a_code() {
    for (body, expected) in [
        (
            r#"{"status":422,"code":"invalid_request"}"#,
            Some("INVALID_REQUEST"),
        ),
        (
            r#"{"status":500,"detail":"invalid_schema: x"}"#,
            Some("INVALID_SCHEMA"),
        ),
        (r#"{"status":422,"code":"private_text"}"#, None),
    ] {
        let error = serving::answer(ENDPOINT, &served(body)).unwrap_err();
        assert_eq!(refusal(&error).2.as_deref(), expected);
    }
}

// --- A success, and the answers that are not one ------------------------------

fn served(inner: &str) -> Vec<u8> {
    let row = Value::Object(vec![("response_json".into(), Value::text(inner))]);
    Value::Object(vec![("predictions".into(), Value::Array(vec![row]))]).to_bytes()
}

#[test]
fn the_answer_is_the_served_model_s_one_response_json_row() {
    let inner = r#"{"object":"prediction.response","id":"r1","predictions":[{"id":"7","row_index":0,"prediction":NaN}],"status":"500"}"#;
    let read = serving::answer(ENDPOINT, &served(inner)).unwrap();
    assert_eq!(read.get("id"), Some(&Value::text("r1")));
    // A status that is no integer is no failure: text, a float, a boolean.
    for status in [r#""500""#, "500.0", "true", "399", "null"] {
        let inner = format!(r#"{{"status":{status},"predictions":[]}}"#);
        assert!(
            serving::answer(ENDPOINT, &served(&inner)).is_ok(),
            "{status}"
        );
    }
    // The row may carry other columns.
    let body = format!(
        r#"{{"predictions":[{{"other":1,"response_json":{}}}],"served_model_name":"m"}}"#,
        serde_json::to_string(inner).unwrap()
    );
    assert!(serving::answer(ENDPOINT, body.as_bytes()).is_ok());
}

#[test]
fn a_reply_that_is_not_one_row_holding_text_is_a_transport_error() {
    for body in [
        "",
        "{}",
        "null",
        "[]",
        r#""x""#,
        "3",
        r#"{"predictions":null}"#,
        r#"{"predictions":{}}"#,
        r#"{"predictions":[]}"#,
        r#"{"predictions":[{"response_json":"{}"},{"response_json":"{}"}]}"#,
    ] {
        let error = serving::answer(ENDPOINT, body.as_bytes()).unwrap_err();
        assert_eq!(
            error.sentence,
            Sentence::ServingPredictionRowCount,
            "{body}"
        );
        assert_eq!(error.code, ErrorCode::TransportError);
        assert_eq!(error.fields["endpoint"], text(ENDPOINT));
    }
    for body in [
        r#"{"predictions":[{}]}"#,
        r#"{"predictions":[{"response_json":1}]}"#,
        r#"{"predictions":[{"response_json":null}]}"#,
        r#"{"predictions":["{}"]}"#,
    ] {
        let error = serving::answer(ENDPOINT, body.as_bytes()).unwrap_err();
        assert_eq!(
            error.sentence,
            Sentence::ServingResponseColumnMissing,
            "{body}"
        );
        assert_eq!(error.code, ErrorCode::TransportError);
    }
    // A reply the SDK could not read as JSON is its failure, as the old
    // client reports one: a 503 with no code.
    for body in ["{", "<html>bad gateway</html>", " "] {
        let error = serving::answer(ENDPOINT, body.as_bytes()).unwrap_err();
        assert_eq!(
            refusal(&error),
            (Sentence::ServingFailed, 503, None),
            "{body}"
        );
    }
}

#[test]
fn a_non_object_uses_json_type_names_and_keeps_the_reference_type_vectors() {
    for (inner, got) in [
        ("[1]", "list"),
        (r#""s""#, "str"),
        ("1", "int"),
        ("-12345678901234567890", "int"),
        ("1.5", "float"),
        ("1e5", "float"),
        ("NaN", "float"),
        ("true", "bool"),
        ("null", "NoneType"),
    ] {
        let error = serving::answer(ENDPOINT, &served(inner)).unwrap_err();
        assert_eq!(error.sentence, Sentence::ServingNotAnObject, "{inner}");
        assert_eq!(error.code, ErrorCode::TransportError);
        let json_type = match got {
            "list" => "array",
            "str" => "string",
            "int" | "float" => "number",
            "bool" => "boolean",
            "NoneType" => "null",
            _ => unreachable!(),
        };
        assert_eq!(error.fields["got"], text(json_type), "{inner}");
    }
    for inner in ["{", "", "\u{feff}{}", &format!("{{\"a\": {SECRET}}}")] {
        let error = serving::answer(ENDPOINT, &served(inner)).unwrap_err();
        assert_eq!(error.sentence, Sentence::ServingNotJson, "{inner}");
        assert_eq!(error.code, ErrorCode::TransportError);
        assert!(!written(&error).contains(SECRET));
    }
    // Nested past the reader's depth: refused as too deep, as every answer
    // is (register P1B-22).
    let deep = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
    let error = serving::answer(ENDPOINT, &served(&deep)).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidResponse);
    assert_eq!(error.fields["reason"], text("too_deep"));
}

#[test]
fn a_failure_the_model_answered_with_is_named_by_its_code_alone() {
    // `_refused_in_band`, each case the old client's own test's.
    let cases: Vec<(String, Expected<'_>, Option<&str>)> = vec![
        (
            format!(
                r#"{{"status":422,"code":"UNSUPPORTED_TASK","detail":"the requested task is not supported for {SECRET}"}}"#
            ),
            (Sentence::ServingRefused, 422, Some("UNSUPPORTED_TASK")),
            None,
        ),
        (
            format!(r#"{{"status":500,"title":"the model failed on {SECRET}"}}"#),
            (Sentence::ServingFailed, 500, None),
            None,
        ),
        (
            format!(
                r#"{{"status":500,"code":"INTERNAL_ERROR","detail":"INVALID_SCHEMA: could not parse row {SECRET}"}}"#
            ),
            (Sentence::ServingRefused, 500, Some("INVALID_SCHEMA")),
            None,
        ),
        (
            format!(
                r#"{{"status":500,"code":"INTERNAL_ERROR","detail":"prediction failed: Failed To Collect Any Context examples for {SECRET}"}}"#
            ),
            (Sentence::ServingCallersFault, 500, Some("INVALID_REQUEST")),
            Some("INTERNAL_ERROR"),
        ),
        (
            r#"{"status":500,"detail":"could not parse row NVDA_SECRET: 42"}"#.to_owned(),
            (Sentence::ServingFailed, 500, None),
            None,
        ),
        // A detail that says nothing: the title is read for the diagnosis.
        (
            r#"{"status":503,"detail":"","title":"failed to collect any context examples"}"#
                .to_owned(),
            (Sentence::ServingCallersFault, 503, Some("INVALID_REQUEST")),
            None,
        ),
        (
            r#"{"status":503,"detail":0.0,"title":"failed to collect any context examples"}"#
                .to_owned(),
            (Sentence::ServingCallersFault, 503, Some("INVALID_REQUEST")),
            None,
        ),
        // A detail that is no text is not read, nor is the title then.
        (
            r#"{"status":503,"code":"BUSY","detail":{"x":1},"title":"failed to collect any context examples"}"#
                .to_owned(),
            (Sentence::ServingRefused, 503, Some("BUSY")),
            None,
        ),
        // Below 500, the detail's code is not read.
        (
            r#"{"status":400,"detail":"INVALID_SCHEMA: x"}"#.to_owned(),
            (Sentence::ServingFailed, 400, None),
            None,
        ),
        // A code that is not shaped as one is no code.
        (
            r#"{"status":400,"code":"not a code"}"#.to_owned(),
            (Sentence::ServingFailed, 400, None),
            None,
        ),
        (
            format!(r#"{{"status":400,"code":"{}"}}"#, "A".repeat(65)),
            (Sentence::ServingFailed, 400, None),
            None,
        ),
        (
            format!(r#"{{"status":400,"code":"{}"}}"#, "A".repeat(64)),
            (Sentence::ServingRefused, 400, Some("A".repeat(64).leak())),
            None,
        ),
        // A detail's code is three to 64 characters, then ": " and a
        // character that is not a space.
        (
            r#"{"status":500,"detail":"AB: x"}"#.to_owned(),
            (Sentence::ServingFailed, 500, None),
            None,
        ),
        (
            r#"{"status":500,"detail":"ABC:  x"}"#.to_owned(),
            (Sentence::ServingFailed, 500, None),
            None,
        ),
        (
            r#"{"status":500,"detail":"ABC: \u001fx"}"#.to_owned(),
            (Sentence::ServingFailed, 500, None),
            None,
        ),
        (
            r#"{"status":500,"code":"X_1","detail":"A_9: é"}"#.to_owned(),
            (Sentence::ServingRefused, 500, Some("A_9")),
            None,
        ),
        // The last of a member named twice, as Python's reader keeps it.
        (
            r#"{"status":200,"status":409,"code":"CONFLICT"}"#.to_owned(),
            (Sentence::ServingRefused, 409, Some("CONFLICT")),
            None,
        ),
    ];
    for (inner, reference, reference_endpoint_code) in cases {
        // Retain the recorded reference vector while naming both deliberate policy changes.
        let (expected, endpoint_code) = match inner.as_str() {
            r#"{"status":503,"code":"BUSY","detail":{"x":1},"title":"failed to collect any context examples"}"# => {
                (
                    (Sentence::ServingCallersFault, 503, Some("INVALID_REQUEST")),
                    Some("BUSY"),
                )
            }
            r#"{"status":500,"detail":"ABC: \u001fx"}"# => {
                ((Sentence::ServingRefused, 500, Some("ABC")), None)
            }
            _ => (reference, reference_endpoint_code),
        };
        let error = serving::answer(ENDPOINT, &served(&inner)).unwrap_err();
        let (sentence, status, code) = refusal(&error);
        assert_eq!(
            (sentence, status, code.as_deref()),
            (expected.0, expected.1, expected.2),
            "{inner}"
        );
        // No SDK carried it: no provider's code or class.
        assert_eq!(field(&error, "provider_code"), None);
        assert_eq!(field(&error, "provider_error"), None);
        if sentence == Sentence::ServingCallersFault {
            assert_eq!(error.fields["diagnosis"], text("EMPTY_CONTEXT"));
            assert_eq!(
                field(&error, "endpoint_code"),
                Some(&endpoint_code.map_or(Detail::Scalar(Scalar::Null), text))
            );
        } else {
            assert_eq!(field(&error, "diagnosis"), None);
        }
        if sentence == Sentence::ServingRefused {
            assert_eq!(error.fields["endpoint_code"], text(expected.2.unwrap()));
        }
    }
}

// --- The endpoint's or the SDK's failure ----------------------------------------

fn sdk(status: u16, message: &str, code: Option<&str>, class: &str) -> CoreError {
    serving::failure(
        ENDPOINT,
        &Failure {
            status,
            detail: Some(message.as_bytes()),
            provider_code: code,
            provider_error: Some(class),
        },
    )
}

fn rest(status: u16, body: &str) -> CoreError {
    serving::failure(
        ENDPOINT,
        &Failure {
            status,
            detail: Some(body.as_bytes()),
            provider_code: None,
            provider_error: None,
        },
    )
}

#[test]
fn an_sdk_failure_is_named_by_the_served_model_s_code_where_its_message_begins() {
    // The old client's own cases (`_error_from`): the status from the SDK's
    // error class, the code the model opened its message with, else the
    // SDK's own.
    for (message, code) in [
        (
            "Encountered an unexpected error while evaluating the model. Error 'INVALID_SCHEMA: the request schema is invalid'",
            "INVALID_SCHEMA",
        ),
        (
            "Encountered an unexpected error while evaluating the model. It failed. Twice. Error 'INVALID_SCHEMA: the request schema is invalid'",
            "INVALID_SCHEMA",
        ),
        (
            "INVALID_SCHEMA: Failed to parse input: row 3 customer_name=Jane Doe ssn=123-45-6789",
            "INVALID_SCHEMA",
        ),
        (
            "BAD_REQUEST: could not parse row NVDA_SECRET: 42",
            "BAD_REQUEST",
        ),
        (
            "BAD_REQUEST: could not parse row Error 'NVDA_SECRET: 42'",
            "BAD_REQUEST",
        ),
        (
            "BAD_REQUEST: BAD_REQUEST: could not parse row NVDA_SECRET: 42",
            "BAD_REQUEST",
        ),
    ] {
        let error = sdk(400, message, Some("BAD_REQUEST"), "BadRequest");
        let (sentence, status, said) = refusal(&error);
        assert_eq!((status, said.as_deref()), (400, Some(code)), "{message}");
        assert_eq!(
            sentence,
            if code == "BAD_REQUEST" {
                Sentence::ServingFailedCoded
            } else {
                Sentence::ServingRefused
            },
            "{message}"
        );
        assert_eq!(error.fields["provider_code"], text("BAD_REQUEST"));
        assert_eq!(error.fields["provider_error"], text("BadRequest"));
    }
}

#[test]
fn a_server_failure_that_is_the_caller_s_to_fix_is_reported_as_such() {
    let error = sdk(
        503,
        &format!("failed to collect any context examples for {SECRET}"),
        Some("INTERNAL_ERROR"),
        "InternalError",
    );
    assert_eq!(
        refusal(&error),
        (
            Sentence::ServingCallersFault,
            503,
            Some("INVALID_REQUEST".into())
        )
    );
    assert_eq!(error.fields["diagnosis"], text("EMPTY_CONTEXT"));
    assert_eq!(error.fields["endpoint_code"], text("INTERNAL_ERROR"));
    assert_eq!(error.fields["provider_code"], text("INTERNAL_ERROR"));
    assert_eq!(error.fields["provider_error"], text("InternalError"));
    // The model's own code, where it opened with one, is the endpoint's.
    let error = sdk(
        503,
        "EMPTY: failed to collect any context examples",
        Some("INTERNAL_ERROR"),
        "InternalError",
    );
    assert_eq!(error.fields["endpoint_code"], text("EMPTY"));
    // Below 500 the diagnosis is not read.
    let error = sdk(
        400,
        "failed to collect any context examples",
        None,
        "BadRequest",
    );
    assert_eq!(refusal(&error), (Sentence::ServingFailed, 400, None));
}

#[test]
fn a_gateway_failure_names_the_endpoint_and_nothing_it_said() {
    let error = sdk(
        403,
        &format!("PERMISSION_DENIED: {SECRET} may not query this endpoint"),
        Some("PERMISSION_DENIED"),
        "PermissionDenied",
    );
    assert_eq!(
        refusal(&error),
        (
            Sentence::ServingFailedCoded,
            403,
            Some("PERMISSION_DENIED".into())
        )
    );
    // A failure with no code: the status alone, and the SDK's class.
    let error = sdk(503, "socket closed", None, "ConnectionError");
    assert_eq!(refusal(&error), (Sentence::ServingFailed, 503, None));
    assert_eq!(error.fields["provider_error"], text("ConnectionError"));
    assert_eq!(field(&error, "provider_code"), None);
    // An SDK code that is not shaped as one is left out.
    let error = sdk(400, "bad", Some("not a code"), "BadRequest");
    assert_eq!(refusal(&error), (Sentence::ServingFailed, 400, None));
    assert_eq!(field(&error, "provider_code"), None);
    // No detail at all.
    let error = serving::failure(
        ENDPOINT,
        &Failure {
            status: 504,
            detail: None,
            provider_code: None,
            provider_error: Some("TimeoutError"),
        },
    );
    assert_eq!(refusal(&error), (Sentence::ServingFailed, 504, None));
}

#[test]
fn an_http_failure_reads_as_the_sdk_reads_the_same_response() {
    // The REST body a host without the SDK hands over (Java's), read as the
    // Databricks SDK reads it and reported under the status its error
    // class gives the old client: the error code before the HTTP status,
    // and a status the SDK has no class for reported as 503.
    let cases: Vec<(u16, String, Expected<'_>)> = vec![
        (
            400,
            r#"{"error_code":"BAD_REQUEST","message":"INVALID_SCHEMA: Failed to parse input: row 3 customer_name=Jane"}"#.to_owned(),
            (Sentence::ServingRefused, 400, Some("INVALID_SCHEMA")),
        ),
        (
            400,
            r#"{"error_code":"BAD_REQUEST","message":"BAD_REQUEST: could not parse row NVDA_SECRET: 42"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 400, Some("BAD_REQUEST")),
        ),
        (
            404,
            r#"{"error_code":"RESOURCE_DOES_NOT_EXIST","message":"Endpoint does not exist"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 404, Some("RESOURCE_DOES_NOT_EXIST")),
        ),
        (
            400,
            r#"{"error_code":"RESOURCE_DOES_NOT_EXIST","message":"gone"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 404, Some("RESOURCE_DOES_NOT_EXIST")),
        ),
        (
            400,
            r#"{"error_code":"REQUEST_LIMIT_EXCEEDED","message":"slow down"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 429, Some("REQUEST_LIMIT_EXCEEDED")),
        ),
        (
            500,
            r#"{"error_code":"INVALID_STATE","message":"x"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 400, Some("INVALID_STATE")),
        ),
        (
            400,
            r#"{"error_code":"ALREADY_EXISTS","message":"x"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 409, Some("ALREADY_EXISTS")),
        ),
        (
            400,
            r#"{"error_code":"DATA_LOSS","message":"x"}"#.to_owned(),
            (Sentence::ServingFailedCoded, 503, Some("DATA_LOSS")),
        ),
        (
            500,
            format!(r#"{{"error_code":"INTERNAL_ERROR","message":"failed to collect any context examples for {SECRET}"}}"#),
            (Sentence::ServingCallersFault, 503, Some("INVALID_REQUEST")),
        ),
        // Registered, proposed (T3.13): the SDK names an HTML page's
        // failure by the HTTP reason phrase (`BAD_GATEWAY`), which a host
        // without the SDK is not given; the core names it by its status.
        (
            502,
            "<html><title>Bad Gateway</title></html>".to_owned(),
            (Sentence::ServingFailed, 503, None),
        ),
        (
            500,
            format!("upstream failure quoting {SECRET}"),
            (Sentence::ServingFailed, 503, None),
        ),
        (401, String::new(), (Sentence::ServingFailed, 401, None)),
        (
            429,
            r#"{"message":"too many"}"#.to_owned(),
            (Sentence::ServingFailed, 429, None),
        ),
        (504, "{}".to_owned(), (Sentence::ServingFailed, 504, None)),
        (
            409,
            r#"{"error_code":"lower","message":"ABC: x"}"#.to_owned(),
            (Sentence::ServingRefused, 409, Some("ABC")),
        ),
        // The SDK's `error` member is its message.
        (
            400,
            r#"{"error_code":"BAD_REQUEST","message":"ignored","error":"INVALID_SCHEMA: x"}"#.to_owned(),
            (Sentence::ServingRefused, 400, Some("INVALID_SCHEMA")),
        ),
        // A body in the SDK's `CODE: message` form.
        (
            400,
            "BAD_REQUEST: INVALID_SCHEMA: could not parse row".to_owned(),
            (Sentence::ServingRefused, 400, Some("INVALID_SCHEMA")),
        ),
        (
            403,
            format!("PERMISSION_DENIED: {SECRET} may not query"),
            (Sentence::ServingFailedCoded, 403, Some("PERMISSION_DENIED")),
        ),
    ];
    for (status, body, reference) in cases {
        let expected = if status == 502 && body.starts_with("<html>") {
            (Sentence::ServingFailedCoded, 503, Some("BAD_GATEWAY"))
        } else {
            reference
        };
        let error = rest(status, &body);
        let (sentence, reported, code) = refusal(&error);
        assert_eq!(
            (sentence, reported, code.as_deref()),
            (expected.0, expected.1, expected.2),
            "{status} {body}"
        );
        assert_eq!(field(&error, "provider_error"), None, "{body}");
    }
}

#[test]
fn an_invalid_status_is_a_malformed_response_not_a_clamped_status() {
    // Registered, proposed (T3.13): Python's in-band status is any integer
    // from 400; the core's field holds a 16-bit status.
    let error = serving::answer(ENDPOINT, &served(r#"{"status":70000}"#)).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidResponse);
    assert_eq!(error.fields["reason"], text("invalid_serving_status"));
}
