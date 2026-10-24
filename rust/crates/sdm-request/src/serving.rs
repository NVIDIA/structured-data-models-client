// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks Model Serving endpoint's envelope and answers (T3.13), as
//! the old client's serving transport writes and reads them
//! (`CL/core/databricks_serving.py`), for every host alike: a host posts
//! the bytes [`envelope`] writes to the endpoint's invocations and hands
//! back what it was answered with, which [`answer`] reads, or the
//! endpoint's failure, which [`failure`] names. No host reads a code out of
//! what the endpoint said.
//!
//! - [`envelope`] is `_envelope` with the size check of `predict`: the call
//!   as the one `request_json` row of `dataframe_records`, written as
//!   Python's `json.dumps` writes it (every character past ASCII escaped, a
//!   float as Python's `repr`), and refused past the caller's limit
//!   (`max_request_bytes`), counted on the wrapped envelope, which escapes
//!   every quote of the call again;
//! - [`answer`] is `_response_body` then `_decoded`: the reply's one
//!   prediction row and its `response_json` text, read as JSON, an object,
//!   and a failure the served model answered with by its `status`
//!   (`_refused_in_band`); a reply the SDK could not read as JSON is the
//!   SDK's failure, a 503 with no code, as the old client reports it;
//! - [`failure`] is `_error_from`: the status the old client reports (the
//!   SDK's error class's), the code the served model opened its message
//!   with (`_model_diagnosis`), else the SDK's own, and a server failure
//!   that is the caller's to fix (`_diagnosis_in`) reported as such. It
//!   reads either form a host has: an SDK's error (its message and its
//!   code apart, with its class's name, Python's) or the HTTP response's
//!   body (a host without the SDK, Java's), read as the SDK reads one.
//!
//! What the endpoint said is read for codes only, never repeated: it can
//! quote the rows it was sent.

use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::{self, Value, is_integer_token};

use crate::predict::bad_response;
use crate::serving_compat::{fold, is_present, is_space, value_type, write_json, write_text};

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
/// `max_request_bytes`. `float_repr` writes a float as Python's `repr`.
pub fn envelope(
    endpoint: &str,
    max_request_bytes: u64,
    call: &[u8],
    float_repr: fn(f64) -> String,
) -> Result<Vec<u8>, CoreError> {
    let read = json::read(call).map_err(|e| {
        CoreError::new(
            Sentence::ServingValueNotJson,
            "the call has no JSON form to send",
        )
        .with_field("reason", Detail::text(e.category))
    })?;
    let mut payload = String::new();
    write_json(&read, float_repr, &mut payload).map_err(|reason| {
        CoreError::new(
            Sentence::ServingValueNotJson,
            "the call carries a value with no JSON form",
        )
        .with_field("reason", Detail::text(reason))
    })?;
    let mut out = String::with_capacity(payload.len() + payload.len() / 8 + 48);
    out.push_str(r#"{"dataframe_records":[{""#);
    out.push_str(REQUEST_COLUMN);
    out.push_str(r#"":"#);
    write_text(&payload, &mut out);
    out.push_str("}]}");
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
    Ok(out.into_bytes())
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
        Some(Value::Number(token)) if is_integer_token(token) => match status_at_least_400(token) {
            Some(status) => Err(in_band(endpoint, status, &read)),
            None => Ok(read),
        },
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
        rest_body(&detail)
    };
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
            code.or(provider_code),
            provider_code,
            failed.provider_error,
        );
    }
    refusal(endpoint, status, code, provider_code, failed.provider_error)
}

// --- Reading the endpoint's failures -------------------------------------------

/// A failure the served model answered with (`_refused_in_band`), named by
/// its code alone.
fn in_band(endpoint: &str, status: u16, answer: &Value) -> CoreError {
    let mut code = match answer.get("code") {
        Some(Value::Text(said)) if code_shaped(said) => Some(said.as_str()),
        _ => None,
    };
    let detail = answer.get("detail");
    if status >= 500 {
        // `detail or title`, as Python's truthiness reads it.
        let reported = match detail {
            Some(d) if is_present(d) => Some(d),
            _ => answer.get("title"),
        };
        if let Some(diagnosis) = reported.and_then(Value::as_text).and_then(diagnosis_in) {
            return callers_fault(endpoint, status, diagnosis, code, None, None);
        }
        // A 5xx can be the model naming what was wrong with the request.
        if let Some(Value::Text(said)) = detail
            && let Some(leading) = leading_code(said)
        {
            code = Some(leading);
        }
    }
    refusal(endpoint, status, code, None, None)
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
    if let Some(code) = endpoint_code {
        out = out.with_field("endpoint_code", Detail::text(code));
    }
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
fn rest_body(body: &str) -> (String, Option<String>) {
    if let Ok(read @ Value::Object(_)) = json::read(body.as_bytes()) {
        let text = |value: Option<&Value>| match value {
            Some(Value::Text(text)) => text.clone(),
            Some(other) => String::from_utf8_lossy(&other.to_bytes()).into_owned(),
            None => "request failed".to_owned(),
        };
        let message = match read.get("error") {
            Some(error) => text(Some(error)),
            None => text(read.get("message")),
        };
        let code = match read.get("error_code") {
            Some(Value::Text(code)) => Some(code.clone()),
            _ => None,
        };
        return (message, code);
    }
    // `([A-Z_]+): (.*)`, matched where the body begins.
    let run = body
        .bytes()
        .take_while(|b| b.is_ascii_uppercase() || *b == b'_')
        .count();
    if run > 0
        && let Some(rest) = body[run..].strip_prefix(": ")
    {
        let line = rest.split('\n').next().unwrap_or_default();
        return (line.to_owned(), Some(body[..run].to_owned()));
    }
    (body.to_owned(), None)
}

/// The status the old client reports for an SDK error of `status` and
/// `code` (`_status_of`, over the SDK's `_error_mapper`): the class the
/// code names, else the class the status names, each reported as its own
/// status where the old client names one, else 503.
fn reported_status(status: u16, code: Option<&str>) -> u16 {
    if let Some(code) = code
        && let Some((_, by_code)) = STATUS_BY_CODE.iter().find(|(c, _)| *c == code)
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
fn model_diagnosis<'a>(text: &'a str, provider_code: Option<&str>) -> Option<&'a str> {
    let text = match provider_code {
        Some(code) => text
            .strip_prefix(code)
            .and_then(|rest| rest.strip_prefix(": "))
            .unwrap_or(text),
        None => text,
    };
    let quoted = quoted_model_error(text);
    [Some(text), quoted]
        .into_iter()
        .flatten()
        .filter_map(leading_code)
        .find(|code| Some(*code) != provider_code)
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
/// upper-case letter, then two to 63 of upper-case letters, digits and
/// underscores, then `: ` and a character that is not a space.
fn leading_code(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    if !bytes.first()?.is_ascii_uppercase() {
        return None;
    }
    let run = 1 + bytes[1..]
        .iter()
        .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || **b == b'_')
        .count();
    if !(3..=64).contains(&run) {
        return None;
    }
    let next = text[run..].strip_prefix(": ")?.chars().next()?;
    (!is_space(next)).then(|| &text[..run])
}

/// Whether `code` is shaped as an error code, enum-like enough to be no
/// secret (`_ERROR_CODE_RE`, `^[A-Z][A-Z0-9_]{0,63}$`, whose `$` also
/// takes one trailing newline).
fn code_shaped(code: &str) -> bool {
    let bare = code.strip_suffix('\n').unwrap_or(code);
    let bytes = bare.as_bytes();
    (1..=64).contains(&bytes.len())
        && bytes[0].is_ascii_uppercase()
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
}

/// The code of a failure the caller can act on, where `reported` says one
/// (`_diagnosis_in`): its wording found in the text, compared as Python's
/// `casefold` compares.
fn diagnosis_in(reported: &str) -> Option<&'static str> {
    let folded = fold(reported);
    THE_CALLERS_TO_FIX
        .iter()
        .find(|(wording, _)| folded.contains(wording))
        .map(|(_, code)| *code)
}

/// An integer status token's value where it is 400 or more (an in-band
/// failure), at most the largest status the field holds.
fn status_at_least_400(token: &str) -> Option<u16> {
    let negative = token.starts_with('-');
    let digits = token.trim_start_matches('-');
    if negative {
        return None;
    }
    match digits.parse::<u64>() {
        Ok(n) if n < 400 => None,
        Ok(n) => Some(u16::try_from(n).unwrap_or(u16::MAX)),
        // Past every fixed width: past 400, and past a status.
        Err(_) => Some(u16::MAX),
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
    fn a_code_may_end_in_one_newline_as_python_s_pattern_takes_it() {
        assert!(code_shaped("A"));
        assert!(code_shaped("ABC\n"));
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

    #[test]
    fn python_s_truthiness() {
        for (value, truth) in [
            (Value::Null, false),
            (Value::Number("0".into()), false),
            (Value::Number("-0.0".into()), false),
            (Value::Number("0e5".into()), false),
            (Value::Number("1e-400".into()), false),
            (Value::Number("NaN".into()), true),
            (Value::Number("2".into()), true),
            (Value::text(""), false),
            (Value::text("x"), true),
            (Value::Array(vec![]), false),
            (Value::Object(vec![]), false),
        ] {
            assert_eq!(is_present(&value), truth, "{value:?}");
        }
    }
}
