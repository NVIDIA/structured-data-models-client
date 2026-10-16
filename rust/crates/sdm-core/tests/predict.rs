// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict` (T1.11d, T1.28b): the request state handed back is read only
//! at its own version, refused by name otherwise (H23); the model and the
//! outputs are checked as the client's `_prepare` checks them, the task's
//! default outputs asked where the caller named none; a request with no row
//! to score is answered without a call; the context is checked as the
//! adapter's `_preflight` checks it; the call is the reference's adapter's,
//! every number as the payload wrote it; the answer is read as the bindings
//! read it, a non-finite number as Python's reader takes it, and laid out
//! under each row's entity, ordered by `row_index`, a class as its target's
//! type; and every answer the result does not carry, or a call that
//! failed, ends the run with a named error that holds no value of the
//! request or the answer (H17). The refusals of what the route does not
//! port are diagnostic, register entry P1B-4.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reruns its own binary for the deeply nested answer"
)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_schema::DataType;
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult, Predict, REQUEST_STATE_VERSION, RequestState};
use sdm_contract::run::{Endpoint, StartRun};
use sdm_contract::{
    CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{
    Attachment, Attachments, CompositionRoot, ModelResult, Run, UnknownZone, Zones, start,
};
use serde_json::{Value, json};

struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}

const CANARY: &str = "CANARY-pr3d";

fn facts() -> HostFacts {
    HostFacts {
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
    }
}

/// A gathered payload as materialize writes one, with numbers whose text a
/// float would change, an entity past the doubles' exact range as its
/// digits, and the engine's inference options the model does not read.
fn payload_text() -> String {
    r#"{"model":"kumo-relational","task":{"kind":"regression","target":{"column_name":"TARGET","dtype":"float32"},"entity_table_names":["users"]},"schema":{"instance_table":{"columns":{"instance_id":{"dtype":"int64"},"__kumo_entity_ref_0":{"dtype":"int64"}}},"related_tables":{"users":{"columns":{"user_id":{"dtype":"int64"}}}},"relationships":[{"source_columns":["__kumo_entity_ref_0"],"target_columns":["user_id"],"target_table":"users"}]},"context":{"instance_table":{"format":"arrays","columns":["instance_id","TARGET"],"rows":[[0,1e-05],[1,12345678.900000001]]}},"predict":{"instance_table":{"format":"arrays","columns":["instance_id","__kumo_entity_ref_0"],"rows":[[2,7],[3,"9007199254740993"]]}},"output":{"fields":["prediction"]},"inference":{"inference_config":{"kind":"regression","num_estimators":1,"output_type":"median","target_transforms":["quantile"]},"run_mode":"fast"},"metadata":{"num_context_examples":2,"source_query":"PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id"}}"#
        .to_owned()
}

fn predict_run(version: u32, payload: String, model: &str) -> Result<Run, CoreError> {
    predict_run_asking(version, payload, model, &["prediction"])
}

fn predict_run_asking(
    version: u32,
    payload: String,
    model: &str,
    outputs: &[&str],
) -> Result<Run, CoreError> {
    predict_run_with(version, payload, model, outputs, false)
}

fn predict_run_with(
    version: u32,
    payload: String,
    model: &str,
    outputs: &[&str],
    default_outputs: bool,
) -> Result<Run, CoreError> {
    predict_run_setting(version, payload, model, outputs, default_outputs, &[])
}

/// A run whose caller set the inference options `inference_asked`.
fn predict_run_setting(
    version: u32,
    payload: String,
    model: &str,
    outputs: &[&str],
    default_outputs: bool,
    inference_asked: &[&str],
) -> Result<Run, CoreError> {
    let operation = Operation::Predict(Predict {
        state: RequestState {
            version,
            payload: AttachmentId(1),
            query: AttachmentId(2),
            forecast: None,
        },
        model: model.into(),
        request_id: "sdm-0123456789ab".into(),
        outputs: outputs.iter().map(|o| (*o).to_owned()).collect(),
        default_outputs,
        inference_asked: inference_asked.iter().map(|o| (*o).to_owned()).collect(),
    });
    let attachments = Attachments::from([
        (AttachmentId(1), Attachment::Bytes(payload.into_bytes())),
        (
            AttachmentId(2),
            Attachment::Bytes(
                serde_json::to_vec("PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id")
                    .unwrap(),
            ),
        ),
    ]);
    start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
        Arc::new(NoZones),
    )
}

/// The call a run asks for: its body, as the bytes the core wrote.
fn call(run: &mut Run) -> (sdm_contract::run::ReplyId, Vec<u8>) {
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!("a model call");
    };
    assert_eq!(call.endpoint, Endpoint::Predict);
    assert_eq!(call.content_type, "application/json");
    let Some(Attachment::Bytes(body)) = run.take_attachment(call.body) else {
        panic!("the body is attached");
    };
    (call.id, body)
}

/// The run's outcome once the model answered `status` with `answer`.
fn answered(status: u16, answer: &[u8]) -> (Run, Result<Step, CoreError>) {
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-relational").unwrap();
    let (id, _) = call(&mut run);
    run.submit_model(
        id,
        ModelResult::Response {
            status,
            content_type: "application/json".into(),
            body: answer.to_vec(),
        },
    )
    .unwrap();
    let step = run.next();
    (run, step)
}

fn rule(error: &CoreError) -> Option<&Detail> {
    error.fields.get("rule")
}

fn text(value: &str) -> Detail {
    Detail::text(value)
}

/// Every text an error holds, its cause chain's included.
fn written(error: &CoreError) -> String {
    let mut out = serde_json::to_string(error).unwrap();
    out.push_str(&error.to_string());
    out
}

#[test]
fn a_request_state_of_another_version_is_refused_by_name() {
    for version in [0, REQUEST_STATE_VERSION + 1, u32::MAX] {
        let refused = predict_run(version, payload_text(), "kumo-relational")
            .err()
            .unwrap();
        assert_eq!(
            (refused.kind, refused.code),
            (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
        );
        assert_eq!(refused.fields["field"], text("state.version"));
        assert_eq!(
            refused.fields["version"],
            Detail::Scalar(Scalar::UInt32(version))
        );
    }
    // Its own version starts a run.
    assert!(predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-relational").is_ok());
}

#[test]
fn the_call_is_the_adapter_s_with_every_number_as_written() {
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-relational").unwrap();
    let (_, body) = call(&mut run);
    let text = String::from_utf8(body).unwrap();
    // Numbers keep the payload's tokens, an integer past 2**53 its digits.
    assert!(text.contains("[0,1e-05],[1,12345678.900000001]"), "{text}");
    assert!(text.contains(r#"[3,"9007199254740993"]"#), "{text}");
    let sent: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(sent["model"], "kumo-relational");
    assert_eq!(
        sent["output"],
        json!({"fields": ["prediction"], "prediction_statistic": "median"})
    );
    // The payload's own metadata, with the call's id.
    assert_eq!(sent["metadata"]["request_id"], "sdm-0123456789ab");
    assert_eq!(sent["metadata"]["num_context_examples"], 2);
    // No inference option the caller set, so no config: the engine's
    // defaults are left out (the reference's 26ecdd28); the rest of the
    // section as the payload had it.
    assert_eq!(sent["inference"], json!({"run_mode": "fast"}));
    // The members in the payload's order, the model where the payload had
    // it.
    let keys: Vec<&str> = sent
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "model",
            "task",
            "schema",
            "context",
            "predict",
            "output",
            "inference",
            "metadata"
        ]
    );
}

#[test]
fn only_the_inference_options_the_caller_set_are_sent() {
    let sent = |asked: &[&str]| -> Value {
        let mut run = predict_run_setting(
            REQUEST_STATE_VERSION,
            payload_text(),
            "kumo-relational",
            &["prediction"],
            false,
            asked,
        )
        .unwrap();
        let (_, body) = call(&mut run);
        serde_json::from_slice::<Value>(&body).unwrap()["inference"].clone()
    };
    // One option set is sent alone, with the value the payload holds.
    assert_eq!(
        sent(&["num_estimators"]),
        json!({"inference_config": {"num_estimators": 1}, "run_mode": "fast"})
    );
    // Two, in the payload's order.
    assert_eq!(
        sent(&["num_estimators", "kind"]),
        json!({"inference_config": {"kind": "regression", "num_estimators": 1}, "run_mode": "fast"})
    );
    // An option the model reads, set but not in the payload's config, adds
    // nothing; with nothing else set there is no config.
    assert_eq!(sent(&["column_shuffle"]), json!({"run_mode": "fast"}));
    // An option set that the model does not read is still refused.
    let error = match predict_run_setting(
        REQUEST_STATE_VERSION,
        payload_text(),
        "kumo-relational",
        &["prediction"],
        false,
        &["output_type"],
    ) {
        Err(error) => error,
        Ok(mut run) => run
            .next()
            .expect_err("an option the model does not read was sent"),
    };
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
    assert_eq!(
        error.fields["unsupported"],
        Detail::List(vec![Scalar::Utf8("output_type".into())])
    );
}

#[test]
fn a_chosen_statistic_is_sent_and_an_inference_section_not_an_object_is_left_out() {
    let payload = payload_text()
        .replace(
            r#""output":{"fields":["prediction"]}"#,
            r#""output":{"fields":["prediction"],"prediction_statistic":"mean"}"#,
        )
        .replace(
            r#""inference":{"inference_config":{"kind":"regression","num_estimators":1,"output_type":"median","target_transforms":["quantile"]},"run_mode":"fast"}"#,
            r#""inference":null"#,
        );
    let mut run = predict_run(REQUEST_STATE_VERSION, payload, "kumo-relational").unwrap();
    let (_, body) = call(&mut run);
    let sent: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(sent["output"]["prediction_statistic"], "mean");
    assert!(sent.get("inference").is_none());
}

#[test]
fn the_answer_is_laid_out_under_each_row_s_entity_ordered_by_row_index() {
    let answer = br#"{"object":"prediction.response","id":"r9","model":"kumo-relational","metadata":{"n":12345678901234567890},"predictions":[{"id":"3","row_index":1,"prediction":2.5},{"id":"2","row_index":0.0,"prediction":1.0}]}"#;
    let (mut run, step) = answered(200, answer);
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = step.unwrap()
    else {
        panic!("done");
    };
    assert_eq!(predicted.model, "kumo-relational");
    assert_eq!(predicted.request_id, "r9");
    let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
        panic!("the metadata is attached");
    };
    // The endpoint's metadata as it wrote it, its integer's digits kept.
    assert_eq!(metadata, br#"{"n":12345678901234567890}"#);
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached");
    };
    let [batch] = result.batches.as_slice() else {
        panic!("one batch");
    };
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["row_index", "id", "prediction", "answer_index"]);
    // Each answer's place in the endpoint's order, which the host's items
    // keep.
    assert_eq!(
        batch
            .column(3)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [1, 0]
    );
    assert_eq!(batch.column(1).data_type(), &DataType::Int64);
    // Ordered by row_index, as the reference's frame orders them, whatever
    // the endpoint's order (T1.28b).
    assert_eq!(
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [0, 1]
    );
    // The second row's entity was written as digits past 2**53.
    assert_eq!(
        batch
            .column(1)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [7, 9_007_199_254_740_993]
    );
    assert_eq!(
        batch
            .column(2)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec(),
        [1.0, 2.5]
    );
}

#[test]
fn an_answer_that_is_not_this_request_s_is_an_invalid_response_naming_the_field() {
    let item =
        |id: &str, row: &str| format!(r#"{{"id":"{id}","row_index":{row},"prediction":1.0}}"#);
    let response = |items: &[String]| {
        format!(
            r#"{{"id":"r1","model":"m","metadata":{{}},"predictions":[{}]}}"#,
            items.join(",")
        )
    };
    let cases = [
        (
            response(&[item("2", "2"), item("3", "1")]),
            "prediction.row_index",
            "out_of_range",
        ),
        (
            response(&[item("2", "-1"), item("3", "1")]),
            "prediction.row_index",
            "out_of_range",
        ),
        (
            response(&[item("2", "0"), item("2", "0")]),
            "prediction.row_index",
            "repeated",
        ),
        (
            response(&[item("2", "0")]),
            "prediction.row_index",
            "unanswered",
        ),
        (
            response(&[item(CANARY, "0"), item("3", "1")]),
            "prediction.id",
            "not_the_row_s",
        ),
        (
            response(&[
                r#"{"row_index":0,"prediction":1.0}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.id",
            "missing",
        ),
        (
            response(&[
                r#"{"id":"2","row_index":0,"forecast_step":1,"prediction":1.0}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.forecast_step",
            "not_a_forecast",
        ),
        (
            format!(r#"{{"id":"{CANARY}","model":1}}"#),
            "model",
            "not_text",
        ),
        (format!("{{\"{CANARY}\""), "response", "not_json"),
        ("[1, NaN]".to_owned(), "response", "not_an_object"),
        (
            response(&[
                r#"{"id":"2","row_index":0,"prediction":"1.0"}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.prediction",
            "not_a_number",
        ),
        (
            response(&[
                r#"{"id":"2","row_index":0,"prediction":123456789012345678901}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.prediction",
            "not_a_number",
        ),
        (
            response(&[
                r#"{"id":"2","row_index":0,"prediction":1.0,"probabilities":[1]}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.probabilities",
            "not_an_object",
        ),
        (
            response(&[
                r#"{"id":"2","row_index":0,"prediction":1.0,"probabilities":{"a":true}}"#
                    .to_owned(),
                item("3", "1"),
            ]),
            "prediction.probabilities",
            "not_a_number",
        ),
        (
            response(&[
                r#"{"id":"2","row_index":0,"prediction":1.0,"probabilities":{"a":NaN}}"#.to_owned(),
                item("3", "1"),
            ]),
            "prediction.probabilities",
            "not_finite",
        ),
    ];
    for (answer, field, reason) in cases {
        let (_, step) = answered(200, answer.as_bytes());
        let error = step.unwrap_err();
        assert_eq!(
            (error.kind, error.code),
            (ErrorKind::ModelEndpoint, ErrorCode::InvalidResponse),
            "{answer}"
        );
        assert_eq!(error.fields["field"], text(field), "{answer}");
        assert_eq!(error.fields["reason"], text(reason), "{answer}");
        assert!(!written(&error).contains(CANARY), "{}", written(&error));
    }
}

#[test]
fn an_answer_the_result_does_not_carry_yet_is_refused_by_name() {
    let base = r#"{"id":"r1","model":"m","metadata":{},"predictions":[ITEM,{"id":"3","row_index":1,"prediction":1.0}]}"#;
    for (item, wanted, field) in [
        (
            r#"{"id":"2","row_index":0,"prediction":1.0,"scores":[0.5]}"#,
            "response_field",
            Some("scores"),
        ),
        (
            r#"{"id":"2","row_index":0,"prediction":1.0,"explanation":{}}"#,
            "response_field",
            Some("explanation"),
        ),
        (
            r#"{"id":"2","row_index":0,"prediction":1.0,"quantiles":{"0.5":1.0}}"#,
            "response_field",
            Some("quantiles"),
        ),
    ] {
        let (_, step) = answered(200, base.replace("ITEM", item).as_bytes());
        let error = step.unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedRequest, "{item}");
        assert_eq!(rule(&error), Some(&text(wanted)), "{item}");
        assert_eq!(error.fields["operation"], text("predict"));
        if let Some(field) = field {
            assert_eq!(error.fields["field"], text(field), "{item}");
        }
    }
    // A null field is no field, as the bindings read it.
    let (_, step) = answered(
        200,
        base.replace(
            "ITEM",
            r#"{"id":"2","row_index":0,"prediction":1.0,"scores":null}"#,
        )
        .as_bytes(),
    );
    assert!(step.is_ok());
    // An integer prediction is a regression's number, and a missing one is
    // none, laid out as NaN, as the reference's frame gives it (T1.28b).
    for item in [
        r#"{"id":"2","row_index":0,"prediction":1}"#,
        r#"{"id":"2","row_index":0}"#,
        r#"{"id":"2","row_index":0,"prediction":null}"#,
    ] {
        let (_, step) = answered(200, base.replace("ITEM", item).as_bytes());
        assert!(step.is_ok(), "{item}");
    }
}

#[test]
fn a_refusal_and_a_failed_transport_end_the_run_with_named_errors() {
    let (_, step) = answered(503, format!("{{\"detail\":\"{CANARY}\"}}").as_bytes());
    let error = step.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidResponse);
    assert_eq!(error.fields["status"], Detail::Scalar(Scalar::UInt16(503)));
    assert!(!written(&error).contains(CANARY));

    for retryable in [false, true] {
        let mut run =
            predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-relational").unwrap();
        let (id, _) = call(&mut run);
        run.submit_model(
            id,
            ModelResult::TransportError {
                kind: ErrorKind::ModelEndpoint,
                retryable,
            },
        )
        .unwrap();
        let error = run.next().unwrap_err();
        assert_eq!(
            (error.kind, error.code, error.retryable),
            (
                ErrorKind::ModelEndpoint,
                ErrorCode::TransportError,
                retryable
            )
        );
    }
}

#[test]
fn what_predict_does_not_port_yet_is_refused_by_name() {
    // The tabular model cannot answer a request of this shape, as the
    // client's check refuses it; one that is no model is unknown.
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-tabular").unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedRequest);
    assert_eq!(error.fields["model"], text("kumo-tabular"));
    assert_eq!(
        error.fields["unsupported"],
        Detail::List(vec![
            Scalar::Utf8("multiple_tables".into()),
            Scalar::Utf8("relationships".into())
        ])
    );
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), CANARY).unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::NotFound, ErrorCode::UnknownModel)
    );
    assert!(!written(&error).contains(CANARY));
    // A task this build writes no call for, before any call.
    let forecast = payload_text().replace(
        r#""kind":"regression","target""#,
        r#""kind":"forecasting","target""#,
    );
    let mut run = predict_run(REQUEST_STATE_VERSION, forecast, "kumo-relational").unwrap();
    assert_eq!(rule(&run.next().unwrap_err()), Some(&text("problem_type")));
    // An entity of a float key is read as the reference's `as_declared`
    // reads it, a double, and given as the JSON text of that value (T1.36
    // fix 8; until then refused by rule id_type).
    let payload = payload_text().replace(
        r#""user_id":{"dtype":"int64"}"#,
        r#""user_id":{"dtype":"float64"}"#,
    );
    let mut run = predict_run(REQUEST_STATE_VERSION, payload, "kumo-relational").unwrap();
    let (id, _) = call(&mut run);
    run.submit_model(
        id,
        ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: br#"{"id":"r1","model":"m","metadata":{},"predictions":[{"id":"2","row_index":0,"prediction":1.0},{"id":"3","row_index":1,"prediction":2.0}]}"#.to_vec(),
        },
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = run.next().unwrap()
    else {
        panic!("done");
    };
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached");
    };
    let [batch] = result.batches.as_slice() else {
        panic!("one batch");
    };
    let written = batch
        .column_by_name("id_as_written")
        .expect("each id as written")
        .as_string::<i32>();
    assert_eq!(
        (written.value(0), written.value(1)),
        ("7.0", "9007199254740992.0")
    );
}

#[test]
fn a_payload_that_is_not_json_is_refused_by_where_never_by_what() {
    let mut run = predict_run(
        REQUEST_STATE_VERSION,
        format!("{{\"a\": \"{CANARY}\" \"b\"}}"),
        "kumo-relational",
    )
    .unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(error.fields["field"], text("state.payload"));
    assert_eq!(error.fields["category"], text("syntax"));
    assert!(!written(&error).contains(CANARY));
}

/// An answer nested far past the reader's depth, about 20 KB, as an
/// endpoint might send: refused as an invalid response, before anything of
/// it is built.
#[test]
fn a_deeply_nested_answer_is_an_invalid_response() {
    let deep = format!(
        r#"{{"id":"r1","model":"m","metadata":{{"nested":{}0{}}},"predictions":[]}}"#,
        "[".repeat(10_000),
        "]".repeat(10_000)
    );
    let (_, step) = answered(200, deep.as_bytes());
    let error = step.unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidResponse);
    assert_eq!(error.fields["field"], text("response"));
    assert_eq!(error.fields["reason"], text("too_deep"));
    println!("deep answer refused");
}

/// The same answer in a process of its own: it must end the run with that
/// error, not abort the process on an overflowed stack (T1.11d review
/// round 1).
#[test]
fn a_deeply_nested_answer_ends_the_run_in_a_process_of_its_own() {
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "a_deeply_nested_answer_is_an_invalid_response",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child.status.success(),
        "{:?}\n{stdout}\n{}",
        child.status,
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(stdout.contains("deep answer refused"), "{stdout}");
}

/// A non-finite number in the answer is read as the reference's reader
/// reads it (owner decision of 2026-10-12, register P1B-4): a prediction
/// written as a token past a double's range is an infinity, the `NaN` and
/// `Infinity` tokens a NaN and an infinity, laid out as such; and the
/// endpoint's metadata keeps each such token as written, which the Python
/// host's reader reads back as the same number.
#[test]
fn a_non_finite_prediction_is_read_as_the_reference_reads_it() {
    for (token, read) in [
        ("1e400", f64::INFINITY),
        ("-1e400", f64::NEG_INFINITY),
        ("1.7976931348623159e308", f64::INFINITY),
        ("Infinity", f64::INFINITY),
        ("-Infinity", f64::NEG_INFINITY),
        ("NaN", f64::NAN),
    ] {
        let answer = format!(
            r#"{{"id":"r1","model":"m","metadata":{{"x":[1,{{"y":{token}}}],"w":"NaN Infinity"}},"predictions":[{{"id":"2","row_index":0,"prediction":{token}}},{{"id":"3","row_index":1,"prediction":1.0}}]}}"#
        );
        let (mut run, step) = answered(200, answer.as_bytes());
        let Step::Done {
            result: OperationResult::Predict(predicted),
        } = step.unwrap()
        else {
            panic!("done");
        };
        let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
            panic!("the metadata is attached");
        };
        assert_eq!(
            String::from_utf8(metadata).unwrap(),
            format!(r#"{{"x":[1,{{"y":{token}}}],"w":"NaN Infinity"}}"#)
        );
        let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
            panic!("the result is attached");
        };
        let got = result.batches[0]
            .column(2)
            .as_primitive::<Float64Type>()
            .value(0);
        assert_eq!(got.to_bits(), read.to_bits(), "{token}");
    }
}

/// The outputs are checked as the client's `_prepare` checks them (T1.28b):
/// none named refused `INVALID_REQUEST`; one the model does not produce
/// refused naming the model; an explanation over two rows refused as the
/// adapter's `_preflight` refuses it; any the model produces asked as named,
/// a name repeated kept.
#[test]
fn the_outputs_are_checked_as_the_client_checks_them() {
    let refused = |outputs: &[&str]| {
        let mut run = predict_run_asking(
            REQUEST_STATE_VERSION,
            payload_text(),
            "kumo-relational",
            outputs,
        )
        .unwrap();
        run.next().unwrap_err()
    };
    let none = refused(&[]);
    assert_eq!(none.code, ErrorCode::InvalidRequest);
    assert_eq!(none.fields["field"], text("outputs"));
    let unknown = refused(&["prediction", "bogus"]);
    assert_eq!(unknown.code, ErrorCode::UnsupportedRequest);
    assert_eq!(
        unknown.fields["unsupported_outputs"],
        Detail::List(vec![Scalar::Utf8("bogus".into())])
    );
    for outputs in [&["prediction", "explanation"][..], &["explanation"]] {
        let error = refused(outputs);
        assert_eq!(error.code, ErrorCode::UnsupportedRequest);
        assert_eq!(error.fields["scored"], Detail::Scalar(Scalar::UInt64(2)));
    }
    for outputs in [
        &["prediction", "prediction"][..],
        &["quantiles"],
        &["probabilities"],
    ] {
        let mut run = predict_run_asking(
            REQUEST_STATE_VERSION,
            payload_text(),
            "kumo-relational",
            outputs,
        )
        .unwrap();
        let (_, body) = call(&mut run);
        let sent: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(sent["output"]["fields"], json!(outputs));
    }
}

/// The caller naming no outputs asks the task's default ones, a
/// classification's prediction and probabilities, laid out as its target's
/// type: a boolean target's classes as booleans, read from the wire's
/// spellings, and its probabilities keyed as the column spells the classes;
/// the result says the outputs asked.
#[test]
fn a_classification_s_default_outputs_and_its_answer_as_its_target_s_type() {
    let binary = payload_text()
        .replace(
            r#""kind":"regression","target":{"column_name":"TARGET","dtype":"float32"}"#,
            r#""kind":"binary_classification","target":{"column_name":"TARGET","dtype":"bool","classes":["false","true"],"positive_class":"true"}"#,
        )
        .replace("[[0,1e-05],[1,12345678.900000001]]", r#"[[0,true],[1,false]]"#);
    let mut run =
        predict_run_with(REQUEST_STATE_VERSION, binary, "kumo-relational", &[], true).unwrap();
    let (id, body) = call(&mut run);
    let sent: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        sent["output"],
        json!({"fields": ["prediction", "probabilities"]})
    );
    run.submit_model(
        id,
        ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: br#"{"id":"r1","model":"m","metadata":{},"predictions":[{"id":"3","row_index":1,"prediction":"0"},{"id":"2","row_index":0,"prediction":"TRUE","probabilities":{"false":0.25,"true":0.75}}]}"#.to_vec(),
        },
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = run.next().unwrap()
    else {
        panic!("done");
    };
    assert_eq!(predicted.outputs, ["prediction", "probabilities"]);
    // No notice for an output the caller did not name.
    assert!(run.take_warnings().is_empty());
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached");
    };
    let batch = &result.batches[0];
    assert_eq!(batch.column(2).data_type(), &DataType::Boolean);
    let predictions: Vec<Option<bool>> = batch.column(2).as_boolean().iter().collect();
    assert_eq!(predictions, [Some(true), Some(false)]);
    let probabilities = batch.column(3).as_map();
    assert!(probabilities.is_valid(0) && !probabilities.is_valid(1));
    let keys: Vec<&str> = probabilities
        .keys()
        .as_string::<i32>()
        .iter()
        .flatten()
        .collect();
    assert_eq!(keys, ["False", "True"]);
    // A class the target's type cannot hold is refused by name.
    let multiclass_text = payload_text().replace(
        r#""kind":"regression","target":{"column_name":"TARGET","dtype":"float32"}"#,
        r#""kind":"multiclass_classification","target":{"column_name":"TARGET","dtype":"string","classes":["a","b"]}"#,
    );
    let mut run = predict_run_with(
        REQUEST_STATE_VERSION,
        multiclass_text,
        "kumo-relational",
        &[],
        true,
    )
    .unwrap();
    let (id, _) = call(&mut run);
    run.submit_model(
        id,
        ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: br#"{"id":"r1","model":"m","metadata":{},"predictions":[{"id":"2","row_index":0,"prediction":1},{"id":"3","row_index":1,"prediction":"a"}]}"#.to_vec(),
        },
    )
    .unwrap();
    assert_eq!(
        rule(&run.next().unwrap_err()),
        Some(&text("prediction_type"))
    );
}

/// The adapter's checks of the context (`_preflight`), before any call: a
/// context with no rows, a classification with more classes than the model
/// takes or a single one refused; a regression's constant context given as
/// a notice naming the target as the host words it, and answered.
#[test]
fn the_context_is_checked_before_the_call() {
    let with_context = |kind_and_target: &str, rows: &str| {
        payload_text()
            .replace(
                r#""kind":"regression","target":{"column_name":"TARGET","dtype":"float32"}"#,
                kind_and_target,
            )
            .replace("[[0,1e-05],[1,12345678.900000001]]", rows)
    };
    const MULTICLASS: &str = r#""kind":"multiclass_classification","target":{"column_name":"TARGET","dtype":"string","classes":["a"]}"#;
    let refused = |payload: String| {
        let mut run =
            predict_run_with(REQUEST_STATE_VERSION, payload, "kumo-relational", &[], true).unwrap();
        run.next().unwrap_err()
    };
    let empty = refused(with_context(MULTICLASS, "[]"));
    assert_eq!(empty.code, ErrorCode::UnsupportedRequest);
    assert_eq!(
        empty.fields["context_rows"],
        Detail::Scalar(Scalar::UInt64(0))
    );
    let eleven: Vec<String> = (0..11).map(|i| format!(r#"[{i},"c{i}"]"#)).collect();
    let many = refused(with_context(MULTICLASS, &format!("[{}]", eleven.join(","))));
    assert_eq!(many.fields["classes"], Detail::Scalar(Scalar::UInt64(11)));
    assert_eq!(many.fields["limit"], Detail::Scalar(Scalar::UInt64(10)));
    let one = refused(with_context(MULTICLASS, r#"[[0,"a"],[1,null],[2,"a"]]"#));
    assert_eq!(
        one.fields["classes_in_context"],
        Detail::Scalar(Scalar::UInt64(1))
    );
    // A regression over a constant: noted, then sent.
    let constant = payload_text().replace(
        "[[0,1e-05],[1,12345678.900000001]]",
        "[[0,3],[1,3.0],[2,null]]",
    );
    let mut run = predict_run(REQUEST_STATE_VERSION, constant, "kumo-relational").unwrap();
    let _ = call(&mut run);
    let warnings = run.take_warnings();
    let [warning] = warnings.as_slice() else {
        panic!("one notice: {warnings:?}");
    };
    assert_eq!(warning.code, "CONTEXT_CONSTANT");
    // The wire calls the target TARGET and the question names no column, so
    // the question itself names it.
    assert_eq!(warning.fields["variant"], text("question"));
    assert_eq!(
        warning.fields["question"],
        text("PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id")
    );
}

/// A request with no row to score is answered without a call
/// (`_empty_answer`): no rows, no request id, empty metadata, the outputs
/// asked.
#[test]
fn a_request_with_no_rows_is_answered_without_a_call() {
    let nothing = payload_text().replace(r#"[[2,7],[3,"9007199254740993"]]"#, "[]");
    let mut run = predict_run(REQUEST_STATE_VERSION, nothing, "kumo-relational").unwrap();
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = run.next().unwrap()
    else {
        panic!("answered without a call");
    };
    assert_eq!(predicted.request_id, "");
    assert_eq!(predicted.outputs, ["prediction"]);
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached");
    };
    assert_eq!(result.batches[0].num_rows(), 0);
    let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
        panic!("the metadata is attached");
    };
    assert_eq!(metadata, b"{}");
}

/// An output the caller named that the answer leaves out is given as a
/// notice naming the model asked and the outputs (`_report_absent`), and
/// the answer is laid out.
#[test]
fn an_output_named_and_left_out_is_noted() {
    let mut run = predict_run_asking(
        REQUEST_STATE_VERSION,
        payload_text(),
        "kumo-relational",
        &["prediction", "quantiles", "prediction"],
    )
    .unwrap();
    let (id, _) = call(&mut run);
    run.submit_model(
        id,
        ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: br#"{"id":"r1","model":"m","metadata":{},"predictions":[{"id":"2","row_index":0,"prediction":1.0},{"id":"3","row_index":1,"prediction":2.0}]}"#.to_vec(),
        },
    )
    .unwrap();
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    let warnings = run.take_warnings();
    let [warning] = warnings.as_slice() else {
        panic!("one notice: {warnings:?}");
    };
    assert_eq!(warning.code, "OUTPUTS_ABSENT");
    assert_eq!(warning.fields["model"], text("kumo-relational"));
    assert_eq!(
        warning.fields["absent"],
        Detail::List(vec![Scalar::Utf8("quantiles".into())])
    );
}

/// The call's size: the core writes the call's bytes, which the host sends
/// unchanged, and refuses a call past the endpoint's limit before the model
/// is called (register PD-2, T1.28a's fix round 1); counted to the byte,
/// with characters past ASCII counted as the bytes written.
#[test]
fn a_call_past_the_limit_is_refused_before_it_is_sent() {
    // The payload with the query padded by `padding`, and the call's length.
    let padded = |padding: &str| {
        payload_text().replace(
            "\"source_query\":\"PREDICT",
            &format!("\"source_query\":\"{padding}PREDICT"),
        )
    };
    let called = |payload: String| -> Result<usize, CoreError> {
        let mut run = predict_run(REQUEST_STATE_VERSION, payload, "kumo-relational").unwrap();
        match run.next()? {
            Step::CallModel(call) => match run.take_attachment(call.body) {
                Some(Attachment::Bytes(body)) => Ok(body.len()),
                _ => panic!("the body is attached"),
            },
            other => panic!("not a call: {other:?}"),
        }
    };
    let limit = sdm_request::payload::PAYLOAD_BYTES;
    let base = called(padded("")).unwrap();
    // ASCII to the limit, exactly: taken; one byte more: refused.
    assert_eq!(called(padded(&"a".repeat(limit - base))).unwrap(), limit);
    let past = called(padded(&"a".repeat(limit - base + 1))).unwrap_err();
    assert_eq!(past.code, ErrorCode::InvalidRequest);
    assert_eq!(past.fields["field"], text("call"));
    // Three-byte characters, each counted as its three bytes written.
    let euros = (limit - base) / 3;
    let rest = limit - base - 3 * euros;
    let at = format!("{}{}", "\u{20ac}".repeat(euros), "a".repeat(rest));
    assert_eq!(called(padded(&at)).unwrap(), limit);
    let over = format!("{}{}", "\u{20ac}".repeat(euros + 1), "a".repeat(rest));
    assert!(called(padded(&over)).is_err());
}
