// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Databricks Model Serving's envelope and responses, shared by every host.
//!
//! The existing wire JSON codec writes both layers and preserves the call's
//! member order and number tokens. The request limit counts the final UTF-8
//! bytes posted. No runtime's JSON formatting or object protocols are copied.
//!
//! Response policy is explicit: one response_json row holding an object;
//! integer failure statuses within the contract's range; JSON type names;
//! nonblank diagnostic text with a title fallback; bounded ASCII error codes
//! and ASCII-insensitive known phrases. Endpoint text is read for codes and
//! never repeated, because it can quote the submitted rows.

use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::{self, Value, is_integer_token};

use crate::predict::bad_response;

/// The column the served model reads a request from. Renaming it breaks
/// every endpoint already deployed.
pub const REQUEST_COLUMN: &str = "request_json";

/// The column the served model writes its answer to.
pub const RESPONSE_COLUMN: &str = "response_json";

/// The gateway's own limit, in decimal megabytes (`_MAX_REQUEST_BYTES`),
/// a host's default for the caller's.
pub const MAX_REQUEST_BYTES: u64 = 16 * 1000 * 1000;

/// What the old client calls a failure the caller can act on, though the
/// endpoint reports it as its own (`_THE_CALLERS_TO_FIX`): the wording,
/// matched case-insensitively, and the code it is the caller's under.
const THE_CALLERS_TO_FIX: &[(&str, &str)] =
    &[("failed to collect any context examples", "EMPTY_CONTEXT")];

/// The gateway's fixed sentence before the served model's own exception
/// is quoted (`_QUOTED_MODEL_ERROR`).
const MODEL_ERROR: &str = "Encountered an unexpected error while evaluating the model.";

/// The SDK's error classes that name a status by the error's code before
/// its HTTP status (`ERROR_CODE_MAPPING`), each as the status the old
/// client reports for the class (`_status_of`).
const STATUS_BY_CODE: &[(&str, u16)] = &[
    ("INVALID_STATE", 400),
    ("INVALID_PARAMETER_VALUE", 400),
    ("RESOURCE_DOES_NOT_EXIST", 404),
    ("ABORTED", 409),
    ("ALREADY_EXISTS", 409),
    ("RESOURCE_ALREADY_EXISTS", 409),
    ("RESOURCE_EXHAUSTED", 429),
    ("REQUEST_LIMIT_EXCEEDED", 429),
    ("UNKNOWN", 503),
    ("DATA_LOSS", 503),
];

/// The HTTP statuses whose SDK error class the old client reports as the
/// status itself; any other it reports as 503, which callers retry.
const STATUS_KEPT: &[u16] = &[400, 401, 403, 404, 409, 429, 504];

/// The call `call` (the core's own JSON) as the endpoint's envelope, the
/// bytes a host posts, or `ServingPastLimit` when it is more than
/// `max_request_bytes`, counted on the actual UTF-8 body.
pub fn envelope(endpoint: &str, max_request_bytes: u64, call: &[u8]) -> Result<Vec<u8>, CoreError> {
    let read = json::read(call).map_err(|e| {
        CoreError::new(
            Sentence::ServingValueNotJson,
            "the call has no JSON form to send",
        )
        .with_field("reason", Detail::text(e.category))
    })?;
    if !read.all_finite() {
        return Err(CoreError::new(
            Sentence::ServingValueNotJson,
            "the call carries a value with no JSON form",
        )
        .with_field("reason", Detail::text("out_of_range_float")));
    }
    // Value's writer always produces UTF-8; this conversion performs no
    // number decoding, integer narrowing or host-dependent formatting.
    let payload = String::from_utf8(read.to_bytes()).map_err(|_| {
        CoreError::new(
            Sentence::Internal,
            "the wire JSON writer returned invalid UTF-8",
        )
    })?;
    let out = Value::Object(vec![(
        "dataframe_records".into(),
        Value::Array(vec![Value::Object(vec![(
            REQUEST_COLUMN.into(),
            Value::Text(payload),
        )])]),
    )])
    .to_bytes();
    let size = out.len() as u64;
    if size > max_request_bytes {
        return Err(CoreError::new(
            Sentence::ServingPastLimit,
            "the call is larger than the serving endpoint takes",
        )
        .with_field("endpoint", Detail::text(endpoint))
        .with_field("bytes", count(size))
        .with_field("limit", count(max_request_bytes)));
    }
    Ok(out)
}

/// The served model's answer in the endpoint's reply `body` (a 2xx), a JSON
/// object, or the error the old client raises for it.
pub fn answer(endpoint: &str, body: &[u8]) -> Result<Value, CoreError> {
    let at = |sentence: Sentence, message: &str| {
        CoreError::new(sentence, message.to_owned()).with_field("endpoint", Detail::text(endpoint))
    };
    // The SDK reads the reply first: an empty one as no predictions, one
    // that is not JSON as its failure.
    let reply = if body.is_empty() {
        Value::Null
    } else {
        json::read_answer(body).map_err(|_| refusal(endpoint, 503, None, None, None))?
    };
    let predictions = match &reply {
        Value::Object(_) => reply.get("predictions"),
        _ => None,
    };
    let row = match predictions {
        Some(Value::Array(rows)) if rows.len() == 1 => &rows[0],
        _ => {
            return Err(at(
                Sentence::ServingPredictionRowCount,
                "the serving endpoint did not answer one prediction row",
            ));
        }
    };
    let Some(Value::Text(text)) = row.get(RESPONSE_COLUMN) else {
        return Err(at(
            Sentence::ServingResponseColumnMissing,
            "the serving endpoint's row holds no answer text",
        ));
    };
    let not_json = || {
        at(
            Sentence::ServingNotJson,
            "the serving endpoint's answer is not JSON",
        )
    };
    // Python reads text, not bytes: a byte order mark is not JSON to it.
    if text.starts_with('\u{feff}') {
        return Err(not_json());
    }
    let read = json::read_answer(text.as_bytes()).map_err(|e| match e.category {
        // Nested past the reader's depth, as every answer is refused
        // (register P1B-22).
        "depth" => {
            bad_response("response", "too_deep").with_field("category", Detail::text("depth"))
        }
        _ => not_json(),
    })?;
    if !matches!(read, Value::Object(_)) {
        return Err(at(
            Sentence::ServingNotAnObject,
            "the serving endpoint's answer is not a JSON object",
        )
        .with_field("got", Detail::text(value_type(&read))));
    }
    match read.get("status") {
        Some(Value::Number(token)) if is_integer_token(token) => {
            match status_at_least_400(token)
                .map_err(|()| bad_response("response", "invalid_serving_status"))?
            {
                Some(status) => Err(in_band(endpoint, status, &read)),
                None => Ok(read),
            }
        }
        _ => Ok(read),
    }
}

/// An endpoint's failure as a host's transport reported it: the HTTP
/// status (an SDK's, the status its error class gives), and what it said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Failure<'a> {
    pub status: u16,
    /// The SDK error's message, or the HTTP response's body.
    pub detail: Option<&'a [u8]>,
    /// The SDK error's code, carried apart from its message.
    pub provider_code: Option<&'a str>,
    /// The SDK error's class's name. With neither it nor a code, `detail`
    /// is the HTTP response's body.
    pub provider_error: Option<&'a str>,
}

/// The refusal an endpoint's failure is, as the old client raises it.
pub fn failure(endpoint: &str, failed: &Failure<'_>) -> CoreError {
    let detail = failed
        .detail
        .map(|d| String::from_utf8_lossy(d).into_owned())
        .unwrap_or_default();
    let (text, raw_code) = if failed.provider_error.is_some() || failed.provider_code.is_some() {
        (detail, failed.provider_code.map(str::to_owned))
    } else {
        rest_body(&detail, failed.status)
    };
    let raw_code = raw_code.map(|code| canonical_code(&code));
    let status = reported_status(failed.status, raw_code.as_deref());
    let provider_code = raw_code.as_deref().filter(|c| code_shaped(c));
    let code = model_diagnosis(&text, provider_code);
    if status >= 500
        && let Some(diagnosis) = diagnosis_in(&text)
    {
        return callers_fault(
            endpoint,
            status,
            diagnosis,
            code.as_deref().or(provider_code),
            provider_code,
            failed.provider_error,
        );
    }
    refusal(
        endpoint,
        status,
        code.as_deref(),
        provider_code,
        failed.provider_error,
    )
}

// --- Reading the endpoint's failures -------------------------------------------

/// A failure the served model answered with (`_refused_in_band`), named by
/// its code alone.
fn in_band(endpoint: &str, status: u16, answer: &Value) -> CoreError {
    let mut code = answer
        .get("code")
        .and_then(Value::as_text)
        .filter(|said| code_shaped(said))
        .map(canonical_code);
    if status >= 500 {
        let reported =
            diagnostic_text(answer.get("detail")).or_else(|| diagnostic_text(answer.get("title")));
        if let Some(diagnosis) = reported.and_then(diagnosis_in) {
            return callers_fault(endpoint, status, diagnosis, code.as_deref(), None, None);
        }
        if let Some(leading) = diagnostic_text(answer.get("detail")).and_then(leading_code) {
            code = Some(canonical_code(leading));
        }
    }
    refusal(endpoint, status, code.as_deref(), None, None)
}

/// A failure the endpoint called its own, reported as the caller's
/// (`_callers_fault`), with what the endpoint called it.
fn callers_fault(
    endpoint: &str,
    status: u16,
    diagnosis: &str,
    endpoint_code: Option<&str>,
    provider_code: Option<&str>,
    provider_error: Option<&str>,
) -> CoreError {
    let mut out = CoreError::refused_by_endpoint(
        Sentence::ServingCallersFault,
        status,
        Some("INVALID_REQUEST"),
        "the serving endpoint refused the request as the caller's to fix",
    )
    .with_field("endpoint", Detail::text(endpoint))
    .with_field("diagnosis", Detail::text(diagnosis));
    out = out.with_field(
        "endpoint_code",
        endpoint_code.map_or(Detail::Scalar(Scalar::Null), Detail::text),
    );
    with_provider(out, provider_code, provider_error)
}

/// A failure known only by its status and codes (`_refusal`): under the
/// code it was refused with where there is one, else under the SDK's.
fn refusal(
    endpoint: &str,
    status: u16,
    code: Option<&str>,
    provider_code: Option<&str>,
    provider_error: Option<&str>,
) -> CoreError {
    let out = match (code, provider_code) {
        (Some(code), _) => CoreError::refused_by_endpoint(
            Sentence::ServingRefused,
            status,
            Some(code),
            "the serving endpoint refused the request",
        )
        .with_field("endpoint_code", Detail::text(code)),
        (None, Some(provider)) => CoreError::refused_by_endpoint(
            Sentence::ServingFailedCoded,
            status,
            Some(provider),
            "the serving endpoint's request failed",
        ),
        (None, None) => CoreError::refused_by_endpoint(
            Sentence::ServingFailed,
            status,
            None,
            "the serving endpoint's request failed",
        ),
    };
    with_provider(
        out.with_field("endpoint", Detail::text(endpoint)),
        provider_code,
        provider_error,
    )
}

fn with_provider(
    mut out: CoreError,
    provider_code: Option<&str>,
    provider_error: Option<&str>,
) -> CoreError {
    if let Some(code) = provider_code {
        out = out.with_field("provider_code", Detail::text(code));
    }
    if let Some(class) = provider_error {
        out = out.with_field("provider_error", Detail::text(class));
    }
    out
}

/// The message and error code the Databricks SDK reads out of an HTTP
/// failure's body (its error parsers, in order): a JSON object's `message`
/// (its `error`, where it has one) and `error_code`; a body that is
/// `CODE: message`, its first line; any other, as it is, with no code.
fn rest_body(body: &str, status: u16) -> (String, Option<String>) {
    if let Ok(read @ Value::Object(_)) = json::read(body.as_bytes()) {
        let message = diagnostic_text(read.get("error"))
            .or_else(|| diagnostic_text(read.get("message")))
            .unwrap_or("request failed")
            .to_owned();
        let code = match read.get("error_code") {
            Some(Value::Text(code)) => Some(code.clone()),
            _ => None,
        };
        return (message, code);
    }
    // A bounded code prefix, followed by the first line of diagnostic text.
    let run = body
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
        .count();
    if run > 0
        && code_shaped(&body[..run])
        && let Some(rest) = body[run..].strip_prefix(": ")
    {
        let line = rest.split('\n').next().unwrap_or_default();
        return (line.to_owned(), Some(body[..run].to_owned()));
    }
    // HTTP gives these gateway failures stable codes even when the body
    // is an HTML page rather than a model/SDK error object. No host
    // exception class is invented or needed for this common policy.
    let html = body.trim_start().to_ascii_lowercase();
    let gateway = if html.starts_with("<html") || html.starts_with("<!doctype html") {
        match status {
            502 => Some("BAD_GATEWAY"),
            503 => Some("SERVICE_UNAVAILABLE"),
            504 => Some("GATEWAY_TIMEOUT"),
            _ => None,
        }
    } else {
        None
    };
    (body.to_owned(), gateway.map(str::to_owned))
}

/// The status the old client reports for an SDK error of `status` and
/// `code` (`_status_of`, over the SDK's `_error_mapper`): the class the
/// code names, else the class the status names, each reported as its own
/// status where the old client names one, else 503.
fn reported_status(status: u16, code: Option<&str>) -> u16 {
    if let Some(code) = code
        && let Some((_, by_code)) = STATUS_BY_CODE
            .iter()
            .find(|(c, _)| c.eq_ignore_ascii_case(code))
    {
        return *by_code;
    }
    if STATUS_KEPT.contains(&status) {
        status
    } else {
        503
    }
}

/// The code the served model opened its message with, if the message
/// carried one (`_model_diagnosis`): read where the model's message
/// begins, at the start, after the SDK's own `CODE: ` prefix, or in the
/// `Error '...'` right after the gateway's fixed wrapper; never the SDK's
/// own code.
fn model_diagnosis(text: &str, provider_code: Option<&str>) -> Option<String> {
    let text = match provider_code {
        Some(code) => text
            .get(..code.len())
            .filter(|head| head.eq_ignore_ascii_case(code))
            .and_then(|_| text[code.len()..].strip_prefix(": "))
            .unwrap_or(text),
        None => text,
    };
    let quoted = quoted_model_error(text);
    [Some(text), quoted]
        .into_iter()
        .flatten()
        .filter_map(leading_code)
        .find(|code| !provider_code.is_some_and(|provider| provider.eq_ignore_ascii_case(code)))
        .map(canonical_code)
}

/// The text after `Encountered an unexpected error while evaluating the
/// model.( [^'.]*\.)* Error '`, where `text` begins with it.
fn quoted_model_error(text: &str) -> Option<&str> {
    let mut rest = text.strip_prefix(MODEL_ERROR)?;
    loop {
        if let Some(quoted) = rest.strip_prefix(" Error '") {
            return Some(quoted);
        }
        let sentence = rest.strip_prefix(' ')?;
        let end = sentence.find(['\'', '.'])?;
        if sentence.as_bytes()[end] != b'.' {
            return None;
        }
        rest = &sentence[end + 1..];
    }
}

/// The code `text` opens with as `CODE: detail` (`_leading_code`): an
/// bounded enum code (known names without ASCII case), then `: ` and a
/// character outside standard whitespace.
fn leading_code(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    if !bytes.first()?.is_ascii_alphabetic() {
        return None;
    }
    let run = 1 + bytes[1..]
        .iter()
        .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
        .count();
    if !(3..=64).contains(&run) || !code_shaped(&text[..run]) {
        return None;
    }
    let next = text[run..].strip_prefix(": ")?.chars().next()?;
    (!next.is_whitespace()).then(|| &text[..run])
}

/// Whether `code` is an enum-like name, never arbitrary diagnostic text.
/// No trailing whitespace or regular-expression end-anchor quirk is admitted.
fn code_shaped(code: &str) -> bool {
    if known_code(code).is_some() {
        return true;
    }
    let bytes = code.as_bytes();
    (1..=64).contains(&bytes.len())
        && bytes[0].is_ascii_uppercase()
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
}

/// These published code names alone are recognized without ASCII case.
/// Unknown codes must already have the bounded upper-case enum form.
fn known_code(code: &str) -> Option<&'static str> {
    STATUS_BY_CODE
        .iter()
        .map(|(name, _)| *name)
        .chain([
            "BAD_REQUEST",
            "BAD_GATEWAY",
            "SERVICE_UNAVAILABLE",
            "GATEWAY_TIMEOUT",
            "INVALID_REQUEST",
            "INVALID_SCHEMA",
            "INTERNAL_ERROR",
            "UNSUPPORTED_TASK",
            "EMPTY_CONTEXT",
            "PERMISSION_DENIED",
        ])
        .find(|name| name.eq_ignore_ascii_case(code))
}

fn canonical_code(code: &str) -> String {
    known_code(code).unwrap_or(code).to_owned()
}

/// A diagnostic is text with something besides standard whitespace.
fn diagnostic_text(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_text)
        .filter(|text| !text.trim().is_empty())
}

fn diagnosis_in(reported: &str) -> Option<&'static str> {
    let folded = reported.to_ascii_lowercase();
    THE_CALLERS_TO_FIX
        .iter()
        .find(|(wording, _)| folded.contains(wording))
        .map(|(_, code)| *code)
}

/// A failure status must fit the shared status field; never clamp it.
fn status_at_least_400(token: &str) -> Result<Option<u16>, ()> {
    if token.starts_with('-') {
        return Ok(None);
    }
    let status = token.parse::<u16>().map_err(|_| ())?;
    Ok((status >= 400).then_some(status))
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::Text(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn count(n: u64) -> Detail {
    Detail::Scalar(Scalar::UInt64(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leading_code_is_three_to_64_characters_then_a_word() {
        assert_eq!(leading_code("ABC: x"), Some("ABC"));
        assert_eq!(leading_code("AB: x"), None);
        assert_eq!(
            leading_code(&format!("{}: x", "A".repeat(64))).map(str::len),
            Some(64)
        );
        assert_eq!(leading_code(&format!("{}: x", "A".repeat(65))), None);
        assert_eq!(leading_code("ABC: \u{a0}x"), None);
        assert_eq!(leading_code("ABC: "), None);
        assert_eq!(leading_code("aBC: x"), None);
    }

    #[test]
    fn a_code_is_bounded_and_contains_no_trailing_whitespace() {
        assert!(code_shaped("A"));
        assert!(!code_shaped("ABC\n"));
        assert!(!code_shaped("ABC\n\n"));
        assert!(!code_shaped(""));
        assert!(!code_shaped("A-B"));
        assert!(!code_shaped(&"A".repeat(65)));
    }

    #[test]
    fn a_quoted_model_error_is_read_only_after_the_gateway_s_sentences() {
        let text = "Encountered an unexpected error while evaluating the model. Error 'X'";
        assert_eq!(quoted_model_error(text), Some("X'"));
        let text =
            "Encountered an unexpected error while evaluating the model. One. Two. Error 'X'";
        assert_eq!(quoted_model_error(text), Some("X'"));
        let text =
            "Encountered an unexpected error while evaluating the model. It's bad. Error 'X'";
        assert_eq!(quoted_model_error(text), None);
        assert_eq!(quoted_model_error("Error 'X'"), None);
    }

    #[test]
    fn a_status_is_reported_as_the_sdk_s_class_gives_it() {
        assert_eq!(reported_status(500, None), 503);
        assert_eq!(reported_status(502, None), 503);
        assert_eq!(reported_status(429, None), 429);
        assert_eq!(reported_status(400, Some("RESOURCE_DOES_NOT_EXIST")), 404);
        assert_eq!(reported_status(400, Some("BAD_REQUEST")), 400);
        assert_eq!(reported_status(503, Some("UNKNOWN")), 503);
    }
}
