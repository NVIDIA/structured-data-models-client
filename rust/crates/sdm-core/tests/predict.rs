// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `predict` (T1.11d): the request state handed back is read only at its
//! own version, refused by name otherwise (H23); the call is the
//! reference's adapter's, every number as the payload wrote it; the answer
//! is read as the bindings read it and laid out under each row's entity;
//! and every answer the result does not carry, or a call that failed, ends
//! the run with a named error that holds no value of the request or the
//! answer (H17). The refusals of what the thin route does not port are
//! diagnostic, register entry P1B-4.

use std::sync::Arc;

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
    let operation = Operation::Predict(Predict {
        state: RequestState {
            version,
            payload: AttachmentId(1),
            query: AttachmentId(2),
        },
        model: model.into(),
        request_id: "sdm-0123456789ab".into(),
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
    // Only the inference options the model reads; the rest of the section
    // as the payload had it.
    assert_eq!(
        sent["inference"],
        json!({"inference_config": {"kind": "regression", "num_estimators": 1}, "run_mode": "fast"})
    );
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
fn the_answer_is_laid_out_under_each_row_s_entity_in_the_endpoint_s_order() {
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
    assert_eq!(names, ["row_index", "id", "prediction"]);
    assert_eq!(batch.column(1).data_type(), &DataType::Int64);
    assert_eq!(
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [1, 0]
    );
    // The second row's entity was written as digits past 2**53.
    assert_eq!(
        batch
            .column(1)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [9_007_199_254_740_993, 7]
    );
    assert_eq!(
        batch
            .column(2)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec(),
        [2.5, 1.0]
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
        ("[1, NaN]".to_owned(), "response", "not_json"),
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
            r#"{"id":"2","row_index":0,"prediction":1}"#,
            "prediction_type",
            None,
        ),
        (
            r#"{"id":"2","row_index":0,"prediction":"1.0"}"#,
            "prediction_type",
            None,
        ),
        (r#"{"id":"2","row_index":0}"#, "prediction_type", None),
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
    // Another model than the relational one, and one that is no model.
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), "kumo-tabular").unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(rule(&error), Some(&text("model")));
    let mut run = predict_run(REQUEST_STATE_VERSION, payload_text(), CANARY).unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::NotFound, ErrorCode::UnknownModel)
    );
    assert!(!written(&error).contains(CANARY));
    // A task other than a regression.
    let payload = payload_text().replace(
        r#""kind":"regression","target""#,
        r#""kind":"binary_classification","target""#,
    );
    let mut run = predict_run(REQUEST_STATE_VERSION, payload, "kumo-relational").unwrap();
    assert_eq!(rule(&run.next().unwrap_err()), Some(&text("problem_type")));
    // An entity of a key whose type the result does not lay out yet.
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
            body: br#"{"id":"r1","model":"m","metadata":{},"predictions":[]}"#.to_vec(),
        },
    )
    .unwrap();
    assert_eq!(rule(&run.next().unwrap_err()), Some(&text("id_type")));
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

/// A prediction written as a finite token past a double's range reads as
/// infinity: refused, as NaN and Infinity are, on every host (register
/// P1B-4), never laid out.
#[test]
fn an_overflowing_prediction_is_an_invalid_response() {
    for token in ["1e400", "-1e400", "1.7976931348623159e308"] {
        let answer = format!(
            r#"{{"id":"r1","model":"m","metadata":{{}},"predictions":[{{"id":"2","row_index":0,"prediction":{token}}},{{"id":"3","row_index":1,"prediction":1.0}}]}}"#
        );
        let (_, step) = answered(200, answer.as_bytes());
        let error = step.unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidResponse, "{token}");
        assert_eq!(error.fields["field"], text("prediction.prediction"));
        assert_eq!(error.fields["reason"], text("not_finite"));
    }
    // The largest double is a prediction like any other.
    let answer = r#"{"id":"r1","model":"m","metadata":{},"predictions":[{"id":"2","row_index":0,"prediction":1.7976931348623157e308},{"id":"3","row_index":1,"prediction":1.0}]}"#;
    assert!(answered(200, answer.as_bytes()).1.is_ok());
}
