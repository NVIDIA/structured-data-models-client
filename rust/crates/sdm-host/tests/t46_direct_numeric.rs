// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain-data direct numeric preparation and tabular prediction through HostRun.
//! Frozen reference: request.py:372-488, base.py:218-310 and tabular.py:171-333.

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::{ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde_json::{Value, json};
use std::sync::Arc;

fn start(operation: Value, attached: Attachments) -> HostRun {
    let message = sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: TimestampUnit::Us,
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
        operation: serde_json::from_value::<Operation>(operation).unwrap(),
    })))
    .unwrap();
    HostRun::start(&sdm_host::default_root(), &message, attached)
        .unwrap_or_else(|error| panic!("start: {:?}", sdm_host::read(&error)))
}
fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}
fn next(run: &HostRun) -> ControlMessage {
    let out = match run.next() {
        Ok(out) | Err(out) => out,
    };
    sdm_host::read(&out).unwrap()
}

fn prepare(value: Value, more: Attachments) -> Result<Value, sdm_contract::CoreError> {
    prepare_trace(value, more).0
}

fn prepare_trace(
    value: Value,
    more: Attachments,
) -> (
    Result<Value, sdm_contract::CoreError>,
    Vec<sdm_contract::Sentence>,
) {
    let mut warnings = Vec::new();
    let mut attached = more;
    attached.insert(AttachmentId(1), bytes(value));
    let run = start(json!({"op":"prepare_request","input":1}), attached);
    loop {
        match next(&run) {
            ControlMessage::Warnings { warnings: given } => {
                warnings.extend(given.into_iter().map(|w| w.sentence));
            }
            ControlMessage::Step(Step::Warnings) => {
                let out = run.take_warnings().unwrap().expect("pending warnings");
                let ControlMessage::Warnings { warnings: given } = sdm_host::read(&out).unwrap()
                else {
                    panic!("warning packet")
                };
                warnings.extend(given.into_iter().map(|w| w.sentence));
            }
            ControlMessage::Error { error } => return (Err(error), warnings),
            ControlMessage::Step(Step::Done {
                result: OperationResult::PrepareRequest(result),
                ..
            }) => {
                let Attachment::Bytes(value) = run.take_attachment(result.value).unwrap().unwrap()
                else {
                    panic!("preparation bytes")
                };
                return (Ok(serde_json::from_slice(&value).unwrap()), warnings);
            }
            other => panic!("preparation must not fetch or call model: {other:?}"),
        }
    }
}
fn response_bytes(run: &HostRun, id: AttachmentId) -> Vec<u8> {
    let Attachment::Bytes(bytes) = run.take_attachment(id).unwrap().unwrap() else {
        panic!("response bytes")
    };
    bytes
}

fn finish_response(
    run: &HostRun,
    value: AttachmentId,
    native_member: Option<(&str, bool)>,
) -> Result<HostRun, sdm_contract::CoreError> {
    let mut data: Value = serde_json::from_slice(&response_bytes(run, value)).unwrap();
    let mut buffers = Attachments::new();
    for (n, name) in ["payload", "response", "rows"].iter().enumerate() {
        let old = AttachmentId(data[*name].as_u64().unwrap() as u32);
        buffers.insert(
            AttachmentId(n as u32 + 2),
            Attachment::Bytes(response_bytes(run, old)),
        );
        data[*name] = json!(n + 2);
    }
    let Attachment::Bytes(rows) = &buffers[&AttachmentId(4)] else {
        panic!("rows")
    };
    let rows: Vec<Value> = serde_json::from_slice(rows).unwrap();
    assert_eq!(data["scope"]["call_request_id"], "sdm-fixed");
    assert_eq!(data["scope"]["request_id"], "direct-fixture");
    let mut accepted = Vec::new();
    let mut at = 0;
    while at < rows.len() {
        let checked = prepare(
            json!({"stage":"check_response_rows","scope":data["scope"],"rows":rows[at..],"decimal_limit":4300}),
            Attachments::new(),
        )?;
        let ready = checked["rows"].as_array().unwrap();
        accepted.extend(ready.iter().cloned());
        at += ready.len();
        if checked["row"].is_null() {
            continue;
        }
        let row = &checked["row"];
        let literal_value = match native_member {
            Some((member, text)) => {
                let member = if text {
                    json!({"kind":"unicode","points":member.chars().map(u32::from).collect::<Vec<_>>()})
                } else {
                    json!({"kind":"integer","digits":format!("0x{member}"),"radix":"hexadecimal"})
                };
                json!({"outcome":"value","value":{"kind":"list"},"members":[member]})
            }
            None => json!({"outcome":"failed","failure":{"ancestry":["syntax_error"]}}),
        };
        let literal = json!({"scope":data["scope"],"ordinal":row["row"]["ordinal"],"text":row["id"]["value"],"value":literal_value});
        let selected = prepare(
            json!({"stage":"select_response_member","checked":row,"literal":literal}),
            Attachments::new(),
        )
        .unwrap();
        let string = native_member.map(|member| json!({"scope":data["scope"],"ordinal":row["row"]["ordinal"],"value":{"outcome":"value","points":member.chars().map(u32::from).collect::<Vec<_>>()}}));
        accepted.push(prepare(
            json!({"stage":"attribute_response_row","selected":selected,"string":string}),
            Attachments::new(),
        )?);
        at += 1;
    }
    buffers.insert(AttachmentId(5), bytes(json!(accepted)));
    buffers.insert(
        AttachmentId(1),
        bytes(json!({"stage":"layout_response","data":data,"attributed":5})),
    );
    let layout = start(json!({"op":"prepare_request","input":1}), buffers);
    Ok(layout)
}

fn arguments() -> Value {
    prepare(json!({"stage":"arguments", "context":{"kind":"frame"}, "predict":{"kind":"frame"}, "related":[], "task":{"value":"regression"}}), Attachments::new()).unwrap()
}
fn header(names: &[&str], rows: usize) -> Value {
    json!({"rows":rows,"columns":names.iter().enumerate().map(|(ordinal,name)|json!({"ordinal":ordinal,"name":{"kind":"text","value":name},"physical":{"type":"float64"},"float_storage":"nan_missing"})).collect::<Vec<_>>(),"duplicates":[]})
}
fn headers() -> Value {
    prepare(json!({"stage":"headers", "arguments":arguments(), "context":header(&["feature","target"],3), "predict":header(&["feature"],2), "target_context":{"positions":[1]}, "target_predict":{"positions":[]}, "predict_context":[[0]]}), Attachments::new()).unwrap()
}
fn rows(names: &[(&str, Vec<f64>)]) -> Attachment {
    let batch = RecordBatch::try_from_iter(names.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
        )
    }))
    .unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}
fn halves() -> Attachments {
    Attachments::from([
        (
            AttachmentId(2),
            rows(&[("0", vec![1., 2., 3.]), ("1", vec![2., 4., 6.])]),
        ),
        (AttachmentId(3), rows(&[("0", vec![4., 5.])])),
    ])
}

#[test]
fn early_arguments_need_no_headers_or_values_and_reject_unknown_task() {
    let error=prepare(json!({"stage":"arguments","context":{"kind":"frame"},"predict":{"kind":"frame"},"related":[],"task":{"value":"unknown"}}),Attachments::new()).unwrap_err();
    assert_eq!(error.sentence.as_str(), "direct_task");
}

#[test]
fn constructor_header_and_arrow_coherence_checks_each_axis() {
    for axis in [
        "row_count",
        "field_count",
        "physical",
        "ordinal",
        "float_storage",
    ] {
        let mut header = headers();
        match axis {
            "row_count" => header["context"]["rows"] = json!(4),
            "field_count" => {
                header["context"]["columns"].as_array_mut().unwrap().pop();
            }
            "physical" => header["context"]["columns"][0]["physical"] = json!({"type":"int64"}),
            "float_storage" => header["context"]["columns"][0]["float_storage"] = Value::Null,
            _ => header["context"]["columns"][0]["ordinal"] = json!(1),
        }
        let error=prepare(json!({"stage":"schema","headers":header,"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap_err();
        assert_eq!(
            error.sentence,
            sdm_contract::Sentence::InvalidMessage,
            "{axis}"
        );
    }
}

#[test]
fn direct_numeric_preparation_is_model_neutral_and_binds_one_tabular_call() {
    for after_advice in [false, true] {
        let prepared=prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
        assert_eq!(prepared["task"], "regression");
        assert_eq!(prepared["target"], "target");
        assert_eq!(
            prepared["description"]["tables"][0]["columns"][0]["dtype"],
            "float64"
        );
        assert!(prepared.get("model").is_none());
        let mut call = prepare(
            json!({"stage":"model","model":"kumo-tabular"}),
            Attachments::new(),
        )
        .unwrap();
        call = prepare(
            json!({"stage":"outputs","call":call,"kind":"regression"}),
            Attachments::new(),
        )
        .unwrap();
        call=prepare(json!({"stage":"capabilities","call":call,"kind":"regression","anchored":false,"relationships":false,"tables":1}),Attachments::new()).unwrap();
        call = prepare(
            json!({"stage":"task","call":call,"kind":"regression"}),
            Attachments::new(),
        )
        .unwrap();
        call = prepare(
            json!({"stage":"capacity","call":call,"rows":3}),
            Attachments::new(),
        )
        .unwrap();
        assert_eq!(
            call["shape"],
            json!({"kind":"regression","anchored":false,"relationships":false,"tables":1})
        );
        let mut schema = prepared["schema"].clone();
        if after_advice {
            schema["tables"][0]["columns"][0]["dtype"] = json!("string");
            schema["tables"][0]["columns"][0]["stype"] = json!("text");
        }
        let input = json!({"bound":call,"context":2,"predict":3,"context_header":header(&["feature","target"],3),"predict_header":header(&["feature"],2),"schema":schema,"entity_table":"instance","target":"target","task":"regression"});
        let overlap = prepare(json!({"stage":"overlap","input":input}), halves()).unwrap();
        let mut input = input;
        input.as_object_mut().unwrap().remove("bound");
        input["overlap"] = overlap;
        let envelope = prepare(json!({"stage":"envelope","input":input}), halves()).unwrap();
        let mut attached = halves();
        let count = if after_advice { 4 } else { 3 };
        let task = if after_advice {
            "forecasting"
        } else {
            "regression"
        };
        if after_advice {
            attached.insert(
                AttachmentId(2),
                rows(&[("0", vec![1., 2., 3., 7.]), ("1", vec![2., 4., 6., 8.])]),
            );
        }
        attached.insert(AttachmentId(1),bytes(json!({"stage":"body","envelope":envelope,"context":2,"predict":3,"context_header":header(&["feature","target"],count),"predict_header":header(&["feature"],2),"task":task,"primary_key":[],"inference":{}})));
        let run = start(
            json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":call["outputs"],"default_outputs":true}),
            attached,
        );
        loop {
            match next(&run) {
                ControlMessage::Warnings { .. } => {}
                ControlMessage::Step(Step::Warnings) => {
                    let out = run.take_warnings().unwrap().expect("pending warnings");
                    assert!(matches!(
                        sdm_host::read(&out).unwrap(),
                        ControlMessage::Warnings { .. }
                    ));
                }
                ControlMessage::Step(Step::CallModel(call)) => {
                    let Attachment::Bytes(body) = run.take_attachment(call.body).unwrap().unwrap()
                    else {
                        panic!("body bytes")
                    };
                    let value: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(value["model"], "kumo-tabular");
                    assert_eq!(
                        value["task"],
                        json!({"kind":task,"target":{"column_name":"target","dtype":"float64"}})
                    );
                    assert_eq!(
                        value["context"]["instance_table"]["rows"],
                        if after_advice {
                            json!([["1.0", 2.], ["2.0", 4.], ["3.0", 6.], ["7.0", 8.]])
                        } else {
                            json!([[1., 2.], [2., 4.], [3., 6.]])
                        }
                    );
                    assert_eq!(
                        value["predict"]["instance_table"]["rows"],
                        if after_advice {
                            json!([["4.0"], ["5.0"]])
                        } else {
                            json!([[4.], [5.]])
                        }
                    );
                    assert_eq!(value["metadata"]["request_id"], "sdm-fixed");
                    assert!(value.get("inference").is_none());
                    assert!(value["output"].get("prediction_statistic").is_none());
                    let answer=br#"{"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":[{"row_index":0,"id":"0","prediction":10},{"row_index":1,"id":"1","prediction":11}]}"#;
                    let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
                        id: call.id,
                        outcome: ModelOutcome::Response {
                            status: 200,
                            body: AttachmentId(100),
                            content_type: "application/json".into(),
                        },
                    }))
                    .unwrap();
                    run.reply(
                        &reply,
                        Attachments::from([(
                            AttachmentId(100),
                            Attachment::Bytes(answer.to_vec()),
                        )]),
                    )
                    .unwrap();
                }
                ControlMessage::Step(Step::Done {
                    result: OperationResult::DecodedDirectResponse(decoded),
                    ..
                }) => {
                    let laid = finish_response(&run, decoded.value, None).unwrap();
                    let message = next(&laid);
                    let ControlMessage::Step(Step::Done {
                        result: OperationResult::Predict(result),
                        ..
                    }) = message
                    else {
                        panic!("layout: {message:?}")
                    };
                    assert_eq!(result.model, "kumo-tabular");
                    assert_eq!(result.request_id, "direct-fixture");
                    let Attachment::Arrow(rows) =
                        laid.take_attachment(result.result).unwrap().unwrap()
                    else {
                        panic!("result Arrow")
                    };
                    assert_eq!(
                        rows.batches
                            .iter()
                            .map(RecordBatch::num_rows)
                            .sum::<usize>(),
                        2
                    );
                    break;
                }
                other => panic!("expected tabular model call/result, got {other:?}"),
            }
        }
    }
}

#[test]
fn direct_preparation_rejects_wrong_versions_and_impossible_call_phases() {
    let mut arguments = arguments();
    arguments["version"] = json!(u32::MAX);
    let error=prepare(json!({"stage":"headers","arguments":arguments,"context":header(&["target"],1),"predict":header(&[],0),"target_context":{"positions":[0]},"target_predict":{"positions":[]},"predict_context":[]}),Attachments::new()).unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::InvalidMessage);
    let model = prepare(
        json!({"stage":"model","model":"kumo-tabular"}),
        Attachments::new(),
    )
    .unwrap();
    let error = prepare(
        json!({"stage":"capacity","call":model,"rows":3}),
        Attachments::new(),
    )
    .unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::InvalidMessage);
}

fn bound(context_rows: usize) -> Value {
    let mut call = prepare(
        json!({"stage":"model","model":"kumo-tabular"}),
        Attachments::new(),
    )
    .unwrap();
    for stage in [
        json!({"stage":"outputs","kind":"regression"}),
        json!({"stage":"capabilities","kind":"regression","anchored":false,"relationships":false,"tables":1}),
        json!({"stage":"task","kind":"regression"}),
        json!({"stage":"capacity","rows":context_rows}),
    ] {
        let mut input = stage;
        input["call"] = call;
        call = prepare(input, Attachments::new()).unwrap();
    }
    call
}

#[test]
fn direct_prelude_must_agree_with_the_legitimately_checked_bound_call() {
    let prepared = prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
    for axis in [
        "task-ranking",
        "task-forecasting",
        "rows-4",
        "rows-10001",
        "schema-tables",
        "schema-links",
    ] {
        let call = bound(3);
        let mut input = json!({"bound":call,"context":2,"predict":3,"context_header":header(&["feature","target"],3),"predict_header":header(&["feature"],2),"schema":prepared["schema"],"entity_table":"instance","target":"target","task":"regression"});
        let mut attached = halves();
        match axis {
            "task-ranking" => input["task"] = json!("ranking"),
            "task-forecasting" => input["task"] = json!("forecasting"),
            "schema-tables" => {
                let mut extra = input["schema"]["tables"][0].clone();
                extra["name"] = json!("extra");
                input["schema"]["tables"]
                    .as_array_mut()
                    .unwrap()
                    .push(extra);
            }
            "schema-links" => {
                input["schema"]["links"] = json!([[{"table":"instance","column":"feature"},{"table":"instance","column":"target"}]])
            }
            _ => {
                let count = if axis == "rows-4" { 4 } else { 10_001 };
                input["context_header"] = header(&["feature", "target"], count);
                let feature = (0..count).map(|n| n as f64).collect();
                let target = (0..count).map(|n| (n + 1) as f64).collect();
                attached.insert(AttachmentId(2), rows(&[("0", feature), ("1", target)]));
            }
        }
        let error = prepare(json!({"stage":"overlap","input":input}), attached).unwrap_err();
        assert_eq!(
            error.sentence,
            sdm_contract::Sentence::InvalidMessage,
            "{axis}"
        );
    }
}

#[test]
fn nullable_cast_fault_is_core_selected_from_native_storage_and_values() {
    for storage in ["nan_missing", "validity_masked"] {
        for null in [false, true] {
            for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
                let call = bound(3);
                let prepared = prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
                let mut schema = prepared["schema"].clone();
                schema["tables"][0]["columns"][0]["dtype"] = json!("int64");
                let mut context_header = header(&["feature", "target"], 3);
                context_header["columns"][0]["physical"] = json!({"type":"int64"});
                context_header["columns"][0]["float_storage"] = Value::Null;
                let mut values = vec![Some(1.), Some(infinity)];
                if null {
                    values.push(None);
                }
                let mut predict_header = header(&["feature"], values.len());
                predict_header["columns"][0]["float_storage"] = json!(storage);
                let context = RecordBatch::try_from_iter([
                    (
                        "0",
                        Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                    ),
                    (
                        "1",
                        Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
                    ),
                ])
                .unwrap();
                let scoring = RecordBatch::try_from_iter([(
                    "0",
                    Arc::new(Float64Array::from(values)) as ArrayRef,
                )])
                .unwrap();
                let data = || {
                    Attachments::from([
                        (
                            AttachmentId(2),
                            Attachment::Arrow(ArrowData::new(
                                context.schema(),
                                vec![context.clone()],
                            )),
                        ),
                        (
                            AttachmentId(3),
                            Attachment::Arrow(ArrowData::new(
                                scoring.schema(),
                                vec![scoring.clone()],
                            )),
                        ),
                    ])
                };
                let input = json!({"bound":call,"context":2,"predict":3,"context_header":context_header,"predict_header":predict_header,"schema":schema,"entity_table":"instance","target":"target","task":"regression"});
                let overlap = prepare(json!({"stage":"overlap","input":input}), data()).unwrap();
                let mut input = input;
                input.as_object_mut().unwrap().remove("bound");
                input["overlap"] = overlap;
                let planned = prepare(json!({"stage":"envelope","input":input}), data());
                if storage == "nan_missing" && null {
                    assert_eq!(
                        planned.unwrap_err().sentence,
                        sdm_contract::Sentence::DirectNullableCast
                    );
                    continue;
                }
                let envelope = planned.unwrap();
                let mut attached = data();
                attached.insert(AttachmentId(1),bytes(json!({"stage":"body","envelope":envelope,"context":2,"predict":3,"context_header":context_header,"predict_header":predict_header,"task":"regression","primary_key":[],"inference":{}})));
                let run = start(
                    json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":call["outputs"],"default_outputs":true}),
                    attached,
                );
                match next(&run) {
                    ControlMessage::Error { error } => {
                        assert_eq!(error.sentence, sdm_contract::Sentence::DirectWireNonFinite)
                    }
                    other => panic!("non-finite rows must fail before transport: {other:?}"),
                }
            }
        }
    }
}

fn attached_batches(context: &[RecordBatch], scoring: &[RecordBatch]) -> Attachments {
    Attachments::from([
        (
            AttachmentId(2),
            Attachment::Arrow(ArrowData::new(context[0].schema(), context.to_vec())),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(scoring[0].schema(), scoring.to_vec())),
        ),
    ])
}

fn native_header(names: &[&str], batches: &[RecordBatch]) -> Value {
    let mut value = header(names, batches.iter().map(RecordBatch::num_rows).sum());
    for (at, field) in batches[0].schema().fields().iter().enumerate() {
        if field.data_type() == &arrow_schema::DataType::Int64 {
            value["columns"][at]["physical"] = json!({"type":"int64"});
            value["columns"][at]
                .as_object_mut()
                .unwrap()
                .remove("float_storage");
        }
    }
    value
}

fn prelude(
    context_names: &[&str],
    scoring_names: &[&str],
    context: &[RecordBatch],
    scoring: &[RecordBatch],
    key: &[&str],
) -> Value {
    let ch = native_header(context_names, context);
    let ph = native_header(scoring_names, scoring);
    let target = context_names.iter().position(|n| *n == "target").unwrap();
    let headers = prepare(json!({"stage":"headers","arguments":arguments(),"context":ch,"predict":ph,"target_context":{"positions":[target]},"target_predict":{"positions":[]},"predict_context":scoring_names.iter().map(|name|vec![context_names.iter().position(|n| n == name).unwrap()]).collect::<Vec<_>>()}), Attachments::new()).unwrap();
    let prepared = prepare(json!({"stage":"schema","headers":headers,"context":2,"predict":3,"entity_table":"instance"}), attached_batches(context, scoring)).unwrap();
    let mut schema = prepared["schema"].clone();
    schema["tables"][0]["primary_key"] = json!(key);
    for column in schema["tables"][0]["columns"].as_array_mut().unwrap() {
        if key.iter().any(|name| column["name"] == *name) {
            column["stype"] = json!("ID");
        }
    }
    schema["tables"][0]["frame_key_form"] =
        json!(if key.len() > 1 { "sequence" } else { "scalar" });
    json!({"bound":bound(ch["rows"].as_u64().unwrap() as usize),"context":2,"predict":3,"context_header":ch,"predict_header":ph,"schema":schema,"entity_table":"instance","target":"target","task":"regression"})
}

fn plan(mut input: Value, context: &[RecordBatch], scoring: &[RecordBatch]) -> Value {
    let overlap = prepare(
        json!({"stage":"overlap","input":input}),
        attached_batches(context, scoring),
    )
    .unwrap();
    input.as_object_mut().unwrap().remove("bound");
    input["overlap"] = overlap;
    prepare(
        json!({"stage":"envelope","input":input}),
        attached_batches(context, scoring),
    )
    .unwrap()
}

fn direct_run(
    input: &Value,
    envelope: Value,
    key: &[&str],
    context: &[RecordBatch],
    scoring: &[RecordBatch],
) -> HostRun {
    let mut attached = attached_batches(context, scoring);
    attached.insert(AttachmentId(1), bytes(json!({"stage":"body","envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"primary_key":key,"task":"regression","inference":{}})));
    start(
        json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":["prediction"],"default_outputs":true}),
        attached,
    )
}

#[test]
fn overlap_warns_before_unrelated_feature_cast_and_uses_numeric_key_identity() {
    for unrelated_cast in [false, true] {
        for composite in [false, true] {
            let context = RecordBatch::try_from_iter([
                (
                    "0",
                    Arc::new(Float64Array::from(vec![-0., 1., 2.])) as ArrayRef,
                ),
                (
                    "1",
                    Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                ),
                (
                    "2",
                    Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
                ),
            ])
            .unwrap();
            let scoring = RecordBatch::try_from_iter([
                ("0", Arc::new(Float64Array::from(vec![0., 3.])) as ArrayRef),
                (
                    "1",
                    Arc::new(Float64Array::from(if unrelated_cast {
                        vec![Some(f64::INFINITY), None]
                    } else {
                        vec![Some(1.), Some(4.)]
                    })) as ArrayRef,
                ),
            ])
            .unwrap();
            let context = [context];
            let scoring = [scoring];
            let key = if composite {
                vec!["user_id", "feature"]
            } else {
                vec!["user_id"]
            };
            let input = prelude(
                &["user_id", "feature", "target"],
                &["user_id", "feature"],
                &context,
                &scoring,
                &key,
            );
            let (answer, warnings) = prepare_trace(
                json!({"stage":"overlap","input":input}),
                attached_batches(&context, &scoring),
            );
            // A failure in a key is advice-only; an unrelated column is not consumed here.
            if composite && unrelated_cast {
                assert!(warnings.is_empty());
                assert_eq!(answer.unwrap()["shown"], 0);
                continue;
            }
            assert_eq!(warnings, vec![sdm_contract::Sentence::DirectAnswersShown]);
            assert_eq!(answer.unwrap()["shown"], 1);
            if unrelated_cast {
                let mut input = input;
                input.as_object_mut().unwrap().remove("bound");
                input["overlap"] = json!({"version":1,"bound":bound(3),"shown":1,"predict_rows":2});
                let error = prepare(
                    json!({"stage":"envelope","input":input}),
                    attached_batches(&context, &scoring),
                )
                .unwrap_err();
                assert_eq!(error.sentence, sdm_contract::Sentence::DirectNullableCast);
                assert_eq!(
                    error.fields["ordinal"],
                    sdm_contract::Detail::Scalar(Scalar::UInt64(1))
                );
            }
        }
    }
}

#[test]
fn direct_column_order_is_preserved_across_arrow_batches() {
    for first_precision in [false, true] {
        let make = |a: f64, b: i64, target: i64| {
            let mut cols = vec![
                ("0", Arc::new(Float64Array::from(vec![a])) as ArrayRef),
                (
                    "1",
                    Arc::new(arrow_array::Int64Array::from(vec![b])) as ArrayRef,
                ),
            ];
            if first_precision {
                cols.swap(0, 1);
                cols[0].0 = "0";
                cols[1].0 = "1";
            }
            cols.push((
                "2",
                Arc::new(arrow_array::Int64Array::from(vec![target])) as ArrayRef,
            ));
            RecordBatch::try_from_iter(cols).unwrap()
        };
        let context = [
            make(1., 9_007_199_254_740_993, 1),
            make(f64::INFINITY, 1, 2),
        ];
        let scoring = [RecordBatch::try_from_iter([
            ("0", Arc::new(Float64Array::from(vec![3., 4.])) as ArrayRef),
            ("1", Arc::new(Float64Array::from(vec![1., 2.])) as ArrayRef),
        ])
        .unwrap()];
        let cn = if first_precision {
            vec!["b", "a", "target"]
        } else {
            vec!["a", "b", "target"]
        };
        let pn = if first_precision {
            vec!["b", "a"]
        } else {
            vec!["a", "b"]
        };
        let input = prelude(&cn, &pn, &context, &scoring, &[]);
        let envelope = plan(input.clone(), &context, &scoring);
        let run = direct_run(&input, envelope, &[], &context, &scoring);
        let ControlMessage::Error { error } = next(&run) else {
            panic!("body must fail before transport")
        };
        assert_eq!(
            error.sentence,
            if first_precision {
                sdm_contract::Sentence::DirectFloatPrecision
            } else {
                sdm_contract::Sentence::DirectWireNonFinite
            }
        );
    }
}

#[test]
fn direct_attribution_starts_as_written_and_reads_later_declared_keys() {
    use arrow_array::cast::AsArray;
    for later_key in [None, Some("instance_id"), Some("__kumo_entity_ref_feature")] {
        let context = [RecordBatch::try_from_iter([
            (
                "0",
                Arc::new(Float64Array::from(vec![1., 2., 3.])) as ArrayRef,
            ),
            (
                "1",
                Arc::new(Float64Array::from(vec![11., 12., 13.])) as ArrayRef,
            ),
            (
                "2",
                Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
            ),
        ])
        .unwrap()];
        let scoring = [RecordBatch::try_from_iter([
            ("0", Arc::new(Float64Array::from(vec![4., 5.])) as ArrayRef),
            (
                "1",
                Arc::new(Float64Array::from(vec![14., 15.])) as ArrayRef,
            ),
        ])
        .unwrap()];
        let mut input = prelude(
            &["__kumo_entity_ref_feature", "instance_id", "target"],
            &["__kumo_entity_ref_feature", "instance_id"],
            &context,
            &scoring,
            &["instance_id"],
        );
        // Text advice freezes the column plan, but not the captured table's key.
        input["schema"]["tables"][0]["columns"][0]["dtype"] = json!("string");
        input["schema"]["tables"][0]["columns"][0]["stype"] = json!("text");
        let envelope = plan(input.clone(), &context, &scoring);
        let key = later_key.into_iter().collect::<Vec<_>>();
        let run = direct_run(&input, envelope, &key, &context, &scoring);
        let ControlMessage::Step(Step::CallModel(call)) = next(&run) else {
            panic!("model call")
        };
        let Attachment::Bytes(body) = run.take_attachment(call.body).unwrap().unwrap() else {
            panic!("body")
        };
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["schema"]["instance_table"].get("primary_key").cloned(),
            if key.is_empty() {
                None
            } else {
                Some(json!(key))
            }
        );
        assert_eq!(
            body["schema"]["instance_table"]["columns"]["__kumo_entity_ref_feature"]["stype"],
            "categorical"
        );
        let answer = json!({"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":[{"row_index":0,"id":"opaque-a","prediction":10},{"row_index":1,"id":"opaque-b","prediction":11}]});
        let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
            id: call.id,
            outcome: ModelOutcome::Response {
                status: 200,
                body: AttachmentId(100),
                content_type: "application/json".into(),
            },
        }))
        .unwrap();
        run.reply(
            &reply,
            Attachments::from([(AttachmentId(100), bytes(answer))]),
        )
        .unwrap();
        let ControlMessage::Step(Step::Done {
            result: OperationResult::DecodedDirectResponse(decoded),
            ..
        }) = next(&run)
        else {
            panic!("result")
        };
        let laid = finish_response(&run, decoded.value, None).unwrap();
        let message = next(&laid);
        let ControlMessage::Step(Step::Done {
            result: OperationResult::Predict(result),
            ..
        }) = message
        else {
            panic!("layout: {message:?}")
        };
        let Attachment::Arrow(result) = laid.take_attachment(result.result).unwrap().unwrap()
        else {
            panic!("result Arrow")
        };
        let batch = &result.batches[0];
        let id = batch.column_by_name("id").unwrap().as_string::<i32>();
        assert_eq!(
            id.value(0),
            match later_key {
                None => "opaque-a",
                Some("instance_id") => "14.0",
                _ => "4.0",
            }
        );
    }
}

#[test]
fn direct_primitive_list_ids_cannot_hide_a_wrong_key() {
    for first in [
        "[(4),]",
        "[(5),]",
        r"['\x34',]",
        r"['\x35',]",
        "[+4,]",
        "[+5,]",
        "[0x4,]",
        "[0x5,]",
        "[0o4,]",
        "[0o5,]",
        "[0b100,]",
        "[0b101,]",
        "[4]",
        "[5]",
        "[4,]",
        "[5,]",
        " [ 4, ] ",
        " [ 5, ] ",
        "['4',]",
        "['5',]",
    ] {
        let context = [RecordBatch::try_from_iter([
            (
                "0",
                Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            ),
            (
                "1",
                Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
            ),
        ])
        .unwrap()];
        let scoring = [RecordBatch::try_from_iter([(
            "0",
            Arc::new(arrow_array::Int64Array::from(vec![4, 5])) as ArrayRef,
        )])
        .unwrap()];
        let input = prelude(
            &["user_id", "target"],
            &["user_id"],
            &context,
            &scoring,
            &["user_id"],
        );
        let envelope = plan(input.clone(), &context, &scoring);
        let run = direct_run(&input, envelope, &["user_id"], &context, &scoring);
        let ControlMessage::Step(Step::CallModel(call)) = next(&run) else {
            panic!("model call")
        };
        let answer = json!({"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":[{"row_index":0,"id":first,"prediction":10},{"row_index":1,"id":"[5]","prediction":11}]});
        let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
            id: call.id,
            outcome: ModelOutcome::Response {
                status: 200,
                body: AttachmentId(100),
                content_type: "application/json".into(),
            },
        }))
        .unwrap();
        run.reply(
            &reply,
            Attachments::from([(AttachmentId(100), bytes(answer))]),
        )
        .unwrap();
        let ControlMessage::Step(Step::Done {
            result: OperationResult::DecodedDirectResponse(decoded),
            ..
        }) = next(&run)
        else {
            panic!("decoded response")
        };
        let wrong = first.contains('5') || first == "[0b101,]";
        let laid = finish_response(&run, decoded.value, Some(if wrong { "5" } else { "4" }));
        if wrong {
            assert_eq!(
                laid.err().expect("wrong key must fail").sentence,
                sdm_contract::Sentence::TabularIdMismatch
            );
        } else {
            let laid = laid.unwrap();
            let message = next(&laid);
            assert!(
                matches!(
                    message,
                    ControlMessage::Step(Step::Done {
                        result: OperationResult::Predict(_),
                        ..
                    })
                ),
                "{message:?}"
            );
        }
    }
}

fn checked_native_row(text: &str) -> Value {
    let scope = json!({"version":1,"model":"kumo-tabular","request_id":"server-nonce","call_request_id":"sdm-fixed","response_rows":2,"scoring_rows":2});
    let rows = json!([{"ordinal":0,"row_index":"0","id":text,"key":{"t":"i64","v":"4"},"raw_in_batch":false,"prior_ordinal":null},{"ordinal":1,"row_index":"0","id":"[4]","key":{"t":"i64","v":"4"},"raw_in_batch":false,"prior_ordinal":0}]);
    let checked = prepare(
        json!({"stage":"check_response_rows","scope":scope,"rows":rows,"decimal_limit":640}),
        Attachments::new(),
    )
    .unwrap();
    assert_eq!(checked["rows"], json!([]));
    assert_eq!(checked["row"]["row"]["ordinal"], 0);
    checked["row"].clone()
}
#[test]
fn reached_native_builtin_facts_control_caught_failure_and_attribution() {
    // The frozen warning promoted to SyntaxError is caught; the successful
    // string literal is checked. No core warning or language callback exists.
    for promoted in [false, true] {
        let checked = checked_native_row(r"['\q']");
        let literal = json!({"scope":checked["scope"],"ordinal":0,"text":r"['\q']","value":if promoted {json!({"outcome":"failed","failure":{"ancestry":["syntax_error"]}})} else {json!({"outcome":"value","value":{"kind":"list"},"members":[{"kind":"unicode","points":[92,113]}]})}});
        let selected = prepare(
            json!({"stage":"select_response_member","checked":checked,"literal":literal}),
            Attachments::new(),
        )
        .unwrap();
        assert_eq!(selected["member"].is_null(), promoted);
        let string = if promoted {
            Value::Null
        } else {
            json!({"scope":checked["scope"],"ordinal":0,"value":{"outcome":"value","points":[92,113]}})
        };
        let attributed = prepare(
            json!({"stage":"attribute_response_row","selected":selected,"string":string}),
            Attachments::new(),
        );
        if promoted {
            assert_eq!(attributed.unwrap()["id"], json!({"t":"i64","v":"4"}));
        } else {
            assert_eq!(
                attributed.unwrap_err().sentence,
                sdm_contract::Sentence::TabularIdMismatch
            );
        }
    }
}
#[test]
fn native_value_association_and_uncaught_string_failure_are_explicit() {
    let checked = checked_native_row("[0x4,]");
    let literal = json!({"scope":checked["scope"],"ordinal":0,"text":"[0x4,]","value":{"outcome":"value","value":{"kind":"list"},"members":[{"kind":"integer","digits":"0x4","radix":"hexadecimal"}]}});
    let mut wrong = literal.clone();
    wrong["scope"]["call_request_id"] = json!("other-call");
    assert_eq!(
        prepare(
            json!({"stage":"select_response_member","checked":checked,"literal":wrong}),
            Attachments::new()
        )
        .unwrap_err()
        .sentence,
        sdm_contract::Sentence::InvalidMessage
    );
    let selected = prepare(
        json!({"stage":"select_response_member","checked":checked,"literal":literal}),
        Attachments::new(),
    )
    .unwrap();
    let string = json!({"scope":checked["scope"],"ordinal":0,"value":{"outcome":"failed","failure":{"ancestry":["value_error"]}}});
    let error = prepare(
        json!({"stage":"attribute_response_row","selected":selected,"string":string}),
        Attachments::new(),
    )
    .unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::DirectNativeFailure);
    assert_eq!(error.fields["stage"], sdm_contract::Detail::text("string"));
}
