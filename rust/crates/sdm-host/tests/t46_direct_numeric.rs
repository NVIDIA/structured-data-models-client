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
    finish_response_limit(run, value, native_member, 4300)
}

fn finish_response_limit(
    run: &HostRun,
    value: AttachmentId,
    native_member: Option<(&str, bool)>,
    decimal_limit: u32,
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
            None => json!({"outcome":"failed","failure":{"ancestry":["value_error"]}}),
        };
        let literal = json!({"scope":data["scope"],"ordinal":row["row"]["ordinal"],"text":row["id"]["value"],"value":literal_value});
        let selected = prepare(
            json!({"stage":"select_response_member","checked":row,"literal":literal}),
            Attachments::new(),
        )
        .unwrap();
        let string = native_member.map(|(member, _)| json!({"scope":data["scope"],"ordinal":row["row"]["ordinal"],"value":{"outcome":"value","points":member.chars().map(u32::from).collect::<Vec<_>>()}}));
        accepted.push(prepare(
            json!({"stage":"attribute_response_row","selected":selected,"string":string}),
            Attachments::new(),
        )?);
        at += 1;
    }
    buffers.insert(AttachmentId(5), bytes(json!(accepted)));
    buffers.insert(
        AttachmentId(1),
        bytes(json!({"stage":"layout_response","data":data,"attributed":5,"decimal_limit":decimal_limit})),
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
fn current_schema_key_metadata_is_published_before_the_later_consistency_fault() {
    let inferred = prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}), halves()).unwrap();
    let mut held = inferred["schema"].clone();
    held["tables"][0]["primary_key"] = json!(["feature"]);
    held["tables"][0]["columns"][0]["stype"] = json!("numerical");
    held["tables"][0]["columns"][1]["stype"] = json!("text");
    let adopted = prepare(
        json!({"stage":"adopt_schema","schema":held}),
        Attachments::new(),
    )
    .unwrap();
    assert_eq!(adopted["tables"][0]["columns"][0]["stype"], "ID");
    assert_eq!(
        adopted["tables"][0]["columns"][0]["stype_before_key"],
        "numerical"
    );
    let error = prepare(
        json!({"stage":"consistent_schema","schema":adopted}),
        Attachments::new(),
    )
    .unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::MeaningNotCarried);
    let mut releasable = inferred["schema"].clone();
    releasable["tables"][0]["columns"][0]["stype"] = json!("ID");
    releasable["tables"][0]["columns"][0]["stype_before_key"] = json!("numerical");
    releasable["tables"][0]["columns"][1]["stype"] = json!("ID");
    let released = prepare(
        json!({"stage":"adopt_schema","schema":releasable}),
        Attachments::new(),
    )
    .unwrap();
    assert_eq!(released["tables"][0]["columns"][0]["stype"], "numerical");
    assert!(released["tables"][0]["columns"][0]["stype_before_key"].is_null());
    assert_eq!(released["tables"][0]["columns"][1]["stype"], "ID");
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
        let inferred=prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
        let prepared=prepare(json!({"stage":"finish_constructor","schema":inferred,"context":2,"predict":3,"context_header":header(&["feature","target"],3),"predict_header":header(&["feature"],2),"target_context":{"positions":[1]},"context_predict":[{"positions":[0]},{"positions":[]}]}),halves()).unwrap();
        let task=prepare(json!({"stage":"constructor_task","prepared":prepared,"entity_table":"instance","tables":[{"kind":"text","value":"instance"}],"table":{"columns":[{"kind":"text","value":"feature"},{"kind":"text","value":"target"}],"target_dtype":"float64"}}),Attachments::new()).unwrap();
        assert_eq!(
            task,
            json!({"kind":"regression","target":"target","table":"instance","dtype":"float64"})
        );
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
        let run = encoded_prediction(
            json!({"envelope":envelope,"context":2,"predict":3,"context_header":header(&["feature","target"],count),"predict_header":header(&["feature"],2),"task":task,"primary_key":[],"inference":{}}),
            attached,
            json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":call["outputs"],"default_outputs":true}),
        ).unwrap();
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
    bound_as(context_rows, "regression")
}

#[test]
fn direct_generated_optional_owners_and_native_mapping_faults_remain_selected_data() {
    let context = [RecordBatch::try_from_iter([
        (
            "0",
            Arc::new(Float64Array::from(vec![1., 2., 3.])) as ArrayRef,
        ),
        (
            "1",
            Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(Float64Array::from(vec![4., 5.])) as ArrayRef,
    )])
    .unwrap()];
    for (field, value, before, native) in [
        ("id", json!(4), "prediction response has 'id' = ", None),
        (
            "forecast_step",
            json!("bad"),
            "prediction response has 'forecast_step' = ",
            None,
        ),
        (
            "row_index",
            json!("bad"),
            "PredictionItem response has 'row_index' = ",
            None,
        ),
        ("explanation", json!(4), "", Some("dict")),
        ("metadata", json!([["x"]]), "", Some("dict")),
        ("rankings", json!(4), "", Some("iter")),
        (
            "probabilities",
            json!({"x":"bad"}),
            "response carries ",
            Some("float"),
        ),
        ("scores", json!([null]), "response carries ", Some("float")),
        (
            "embeddings",
            json!(["bad"]),
            "response carries ",
            Some("float"),
        ),
        (
            "quantiles",
            json!({"x":"bad"}),
            "response carries ",
            Some("float"),
        ),
    ] {
        let input = prelude(
            &["feature", "target"],
            &["feature"],
            &context,
            &scoring,
            &[],
        );
        let envelope = plan(input.clone(), &context, &scoring);
        let run = direct_run(&input, envelope, &[], &context, &scoring).unwrap();
        let ControlMessage::Step(Step::CallModel(call)) = next(&run) else {
            panic!("model call")
        };
        let mut first = json!({"row_index":0,"id":"0","prediction":10});
        first[field] = value.clone();
        let answer = json!({"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":[first,{"row_index":1,"id":"1","prediction":11}]});
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
        let ControlMessage::Error { error } = next(&run) else {
            panic!("selected generated fault")
        };
        assert_eq!(
            error.sentence,
            sdm_contract::Sentence::DirectResponseContract
        );
        assert_eq!(error.fields["before"], sdm_contract::Detail::text(before));
        let offending = if native == Some("float") {
            if value.is_array() {
                &value[0]
            } else {
                &value["x"]
            }
        } else {
            &value
        };
        assert_eq!(
            error.fields["offending"],
            sdm_contract::Detail::text(serde_json::to_string(offending).unwrap())
        );
        if native == Some("float") {
            assert_eq!(
                error.fields["after"],
                sdm_contract::Detail::text(" where the contract requires a number")
            );
            let shown = if offending.is_null() { "None" } else { "'bad'" };
            assert_eq!(
                error.fields["diagnostic"],
                sdm_contract::Detail::text(format!(
                    "response carries {shown} where the contract requires a number"
                ))
            );
        }
        assert_eq!(
            error.fields.get("native"),
            native.map(sdm_contract::Detail::text).as_ref()
        );
    }
}

#[test]
fn explicit_text_target_returns_before_the_regression_meaning_override() {
    let context = [RecordBatch::try_from_iter([
        (
            "0",
            Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        ),
        (
            "1",
            Arc::new(arrow_array::Int64Array::from(vec![2, 4, 6])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(arrow_array::Int64Array::from(vec![4, 5])) as ArrayRef,
    )])
    .unwrap()];
    let mut input = prelude(
        &["feature", "target"],
        &["feature"],
        &context,
        &scoring,
        &[],
    );
    let target = &mut input["schema"]["tables"][0]["columns"][1];
    target["dtype"] = json!("string");
    target["stype"] = json!("text");
    target["stype_declared"] = json!(true);
    let envelope = plan(input.clone(), &context, &scoring);
    let (_, payload) = encoded_body(json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"regression","primary_key":[]}), attached_batches(&context, &scoring)).unwrap();
    let payload: Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(
        payload["schema"]["instance_table"]["columns"]["target"],
        json!({"dtype":"string","stype":"categorical"})
    );
    assert_eq!(
        payload["context"]["instance_table"]["rows"],
        json!([[1, "2"], [2, "4"], [3, "6"]])
    );
}

#[test]
fn direct_regression_keeps_the_complete_generated_prediction_value_family() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    let context = [RecordBatch::try_from_iter([
        (
            "0",
            Arc::new(Float64Array::from(vec![1., 2., 3.])) as ArrayRef,
        ),
        (
            "1",
            Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(Float64Array::from(vec![4., 5.])) as ArrayRef,
    )])
    .unwrap()];
    let wide = "1".to_owned() + &"0".repeat(660);
    let cases = [
        None,
        Some("null".to_owned()),
        Some("false".to_owned()),
        Some("true".to_owned()),
        Some("2".to_owned()),
        Some(wide.clone()),
        Some(format!("-{wide}")),
        Some("2.0".to_owned()),
        Some("1.5".to_owned()),
        Some(r#""0_2""#.to_owned()),
        Some(r#""1_0.5""#.to_owned()),
        Some(r#""unknown""#.to_owned()),
        Some(format!("\"{}2\"", "0".repeat(650))),
        Some("[]".to_owned()),
        Some("[1,2]".to_owned()),
        Some("{}".to_owned()),
        Some(r#"{"a":2}"#.to_owned()),
        Some(format!(r#"{{"a":[{wide},{{"b":-{wide}}}]}}"#)),
    ];
    for prediction in cases {
        let input = prelude(
            &["feature", "target"],
            &["feature"],
            &context,
            &scoring,
            &[],
        );
        let envelope = plan(input.clone(), &context, &scoring);
        let run = direct_run(&input, envelope, &[], &context, &scoring).unwrap();
        let ControlMessage::Step(Step::CallModel(call)) = next(&run) else {
            panic!("model call")
        };
        let field = prediction
            .as_ref()
            .map(|v| format!(",\"prediction\":{v}"))
            .unwrap_or_default();
        let answer = format!(r#"{{"id":"direct-fixture","model":"kumo-tabular","metadata":{{}},"predictions":[{{"row_index":0,"id":"0"{field}}},{{"row_index":1,"id":"1"{field}}}]}}"#).into_bytes();
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
            Attachments::from([(AttachmentId(100), Attachment::Bytes(answer))]),
        )
        .unwrap();
        let ControlMessage::Step(Step::Done {
            result: OperationResult::DecodedDirectResponse(decoded),
            ..
        }) = next(&run)
        else {
            panic!("decoded response")
        };
        let laid = finish_response_limit(&run, decoded.value, None, 640).unwrap();
        let ControlMessage::Step(Step::Done {
            result: OperationResult::Predict(result),
            ..
        }) = next(&laid)
        else {
            panic!("prediction result")
        };
        let Attachment::Arrow(result) = laid.take_attachment(result.result).unwrap().unwrap()
        else {
            panic!("result table")
        };
        let batch = &result.batches[0];
        let expected = prediction.as_deref().unwrap_or("null");
        for at in 0..2 {
            let actual = if let Some(text) = batch.column_by_name("prediction_as_answered") {
                let values = text.as_string::<i32>();
                if values.is_null(at) {
                    "null".to_owned()
                } else {
                    values.value(at).to_owned()
                }
            } else if let Some(column) = batch.column_by_name("prediction") {
                match column.data_type() {
                    arrow_schema::DataType::Boolean => column.as_boolean().value(at).to_string(),
                    arrow_schema::DataType::Int64 => {
                        column.as_primitive::<Int64Type>().value(at).to_string()
                    }
                    arrow_schema::DataType::Float64 => {
                        serde_json::to_string(&column.as_primitive::<Float64Type>().value(at))
                            .unwrap()
                    }
                    arrow_schema::DataType::Utf8 => {
                        serde_json::to_string(column.as_string::<i32>().value(at)).unwrap()
                    }
                    other => panic!("unexpected carrier {other:?}"),
                }
            } else {
                "null".to_owned()
            };
            assert_eq!(actual, expected, "{prediction:?}");
        }
    }
}

fn bound_as(context_rows: usize, kind: &str) -> Value {
    let mut call = prepare(
        json!({"stage":"model","model":"kumo-tabular"}),
        Attachments::new(),
    )
    .unwrap();
    for stage in [
        json!({"stage":"outputs","kind":kind}),
        json!({"stage":"capabilities","kind":kind,"anchored":false,"relationships":false,"tables":1}),
        json!({"stage":"task","kind":kind}),
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
                let attached = data();
                let error = encoded_prediction(
                    json!({"envelope":envelope,"context":2,"predict":3,"context_header":context_header,"predict_header":predict_header,"task":"regression","primary_key":[],"inference":{}}),
                    attached,
                    json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":call["outputs"],"default_outputs":true}),
                ).err().expect("nonfinite rows fail before transport");
                assert_eq!(error.sentence, sdm_contract::Sentence::DirectWireNonFinite);
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
        value["columns"][at]["physical"] = serde_json::to_value(
            sdm_contract::ColumnType::from_arrow(field.name(), field.data_type()).unwrap(),
        )
        .unwrap();
        if !matches!(
            field.data_type(),
            arrow_schema::DataType::Float32 | arrow_schema::DataType::Float64
        ) {
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
) -> Result<HostRun, sdm_contract::CoreError> {
    encoded_prediction(
        json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"primary_key":key,"task":"regression","inference":{}}),
        attached_batches(context, scoring),
        json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":["prediction"],"default_outputs":true}),
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
        let error = direct_run(&input, envelope, &[], &context, &scoring)
            .err()
            .expect("body must fail before transport");
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
        let run = direct_run(&input, envelope, &key, &context, &scoring).unwrap();
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
        let run = direct_run(&input, envelope, &["user_id"], &context, &scoring).unwrap();
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
        let laid = finish_response(
            &run,
            decoded.value,
            Some((if wrong { "5" } else { "4" }, first.contains('\''))),
        );
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

fn encoded_prediction(
    mut input: Value,
    attached: Attachments,
    operation: Value,
) -> Result<HostRun, sdm_contract::CoreError> {
    let inference = input
        .as_object_mut()
        .unwrap()
        .remove("inference")
        .unwrap_or_else(|| json!({}));
    let (mut body, payload) = encoded_body(input, attached)?;
    body["payload"] = json!(2);
    Ok(start(
        operation,
        Attachments::from([
            (
                AttachmentId(1),
                bytes(json!({"stage":"encoded","body":body,"inference":inference})),
            ),
            (AttachmentId(2), Attachment::Bytes(payload)),
        ]),
    ))
}

fn encoded_body(
    input: Value,
    mut attached: Attachments,
) -> Result<(Value, Vec<u8>), sdm_contract::CoreError> {
    attached.insert(
        AttachmentId(1),
        bytes(json!({"stage":"encode_body","input":input})),
    );
    let run = start(json!({"op":"prepare_request","input":1}), attached);
    loop {
        match next(&run) {
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                run.take_warnings().unwrap().expect("warnings");
            }
            ControlMessage::Error { error } => return Err(error),
            ControlMessage::Step(Step::Done {
                result: OperationResult::PrepareRequest(value),
                ..
            }) => {
                let body: Value =
                    serde_json::from_slice(&response_bytes(&run, value.value)).unwrap();
                let payload =
                    response_bytes(&run, AttachmentId(body["payload"].as_u64().unwrap() as u32));
                return Ok((body, payload));
            }
            other => panic!("row encoding performs no model I/O: {other:?}"),
        }
    }
}

#[test]
fn encoded_numeric_body_is_useful_before_native_inference_is_observed() {
    for nonfinite in [false, true] {
        let context = [RecordBatch::try_from_iter([
            (
                "0",
                Arc::new(Float64Array::from(vec![
                    1.,
                    if nonfinite { f64::INFINITY } else { 2. },
                    3.,
                ])) as ArrayRef,
            ),
            (
                "1",
                Arc::new(arrow_array::Int64Array::from(vec![2, 4, 6])) as ArrayRef,
            ),
        ])
        .unwrap()];
        let scoring = [RecordBatch::try_from_iter([(
            "0",
            Arc::new(Float64Array::from(vec![4., 5.])) as ArrayRef,
        )])
        .unwrap()];
        let input = prelude(
            &["feature", "target"],
            &["feature"],
            &context,
            &scoring,
            &[],
        );
        let envelope = plan(input.clone(), &context, &scoring);
        let encoded = encoded_body(
            json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"regression","primary_key":[]}),
            attached_batches(&context, &scoring),
        );
        if nonfinite {
            assert_eq!(
                encoded.unwrap_err().sentence,
                sdm_contract::Sentence::DirectWireNonFinite
            );
            continue;
        }
        let (body, payload) = encoded.unwrap();
        assert_eq!(body["version"], 1);
        assert_eq!(body["context_shape"]["rows"], 3);
        assert_eq!(body["predict_shape"]["rows"], 2);
        let payload: Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(
            payload["context"]["instance_table"]["rows"],
            json!([[1., 2], [2., 4], [3., 6]])
        );
        assert_eq!(
            payload["predict"]["instance_table"]["rows"],
            json!([[4.], [5.]])
        );
        assert!(payload.get("metadata").is_none());
        assert!(payload.get("inference").is_none());
        assert_eq!(payload["model"], "kumo-tabular");
        assert_eq!(payload["output"]["fields"], json!(["prediction"]));
    }
}

#[test]
fn argument_forms_select_errors_without_serializing_unused_native_names() {
    for (context, predict, related, sentence, ordinal) in [
        (
            "dict",
            "frame",
            vec![json!({"kind":"frame"})],
            sdm_contract::Sentence::DirectFrame,
            None,
        ),
        (
            "frame",
            "dict",
            vec![json!({"kind":"frame"})],
            sdm_contract::Sentence::DirectFrame,
            None,
        ),
        (
            "frame",
            "frame",
            vec![json!({"kind":"frame"}), json!({"kind":"dict"})],
            sdm_contract::Sentence::DirectFrame,
            Some(1),
        ),
        (
            "frame",
            "frame",
            vec![json!({"kind":"frame"})],
            sdm_contract::Sentence::DirectRelated,
            None,
        ),
    ] {
        let error = prepare(json!({"stage":"arguments","context":{"kind":context},"predict":{"kind":predict},"related":related,"task":{"value":"unknown"}}), Attachments::new()).unwrap_err();
        assert_eq!(error.sentence, sentence);
        assert_eq!(
            error.fields.get("related_ordinal"),
            ordinal
                .map(|n| sdm_contract::Detail::Scalar(Scalar::UInt64(n)))
                .as_ref()
        );
        // Names remain native values and are needed only by the selected presenter.
        assert!(!error.fields.contains_key("related_tables"));
    }
}

#[test]
fn encoded_int64_cells_reach_final_predict_only_in_the_shared_canonical_form() {
    for large in [
        -(1_i64 << 53) - 1,
        -(1_i64 << 53) + 1,
        (1_i64 << 53) - 1,
        (1_i64 << 53) + 1,
        i64::MIN,
        i64::MAX,
    ] {
        for key in [false, true] {
            let context = [RecordBatch::try_from_iter([
                (
                    "0",
                    Arc::new(arrow_array::Int64Array::from(vec![large, 2, 3])) as ArrayRef,
                ),
                (
                    "1",
                    Arc::new(arrow_array::Int64Array::from(vec![large, 4, 6])) as ArrayRef,
                ),
            ])
            .unwrap()];
            let scoring = [RecordBatch::try_from_iter([(
                "0",
                Arc::new(arrow_array::Int64Array::from(vec![large, 5])) as ArrayRef,
            )])
            .unwrap()];
            let keys = if key { vec!["feature"] } else { vec![] };
            let input = prelude(
                &["feature", "target"],
                &["feature"],
                &context,
                &scoring,
                &keys,
            );
            let envelope = plan(input.clone(), &context, &scoring);
            let (body, encoded) = encoded_body(json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"regression","primary_key":keys}), attached_batches(&context, &scoring)).unwrap();
            let original: Value = serde_json::from_slice(&encoded).unwrap();
            let expected = if large.unsigned_abs() > 9_007_199_254_740_991 {
                json!(large.to_string())
            } else {
                json!(large)
            };
            assert_eq!(
                original["context"]["instance_table"]["rows"][0],
                json!([expected, expected])
            );
            assert_eq!(
                original["predict"]["instance_table"]["rows"][0],
                json!([expected])
            );
            for replacement in [
                None,
                Some(json!("+9007199254740993")),
                Some(json!("09007199254740993")),
                Some(json!("9223372036854775808")),
                Some(json!("2")),
                Some(json!("1.5")),
            ] {
                let mut body = body.clone();
                body["payload"] = json!(2);
                let mut payload = original.clone();
                if let Some(value) = &replacement {
                    payload["predict"]["instance_table"]["rows"][0][0] = value.clone();
                }
                let run = start(
                    json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":["prediction"],"default_outputs":true}),
                    Attachments::from([
                        (
                            AttachmentId(1),
                            bytes(json!({"stage":"encoded","body":body,"inference":{}})),
                        ),
                        (AttachmentId(2), bytes(payload)),
                    ]),
                );
                match next(&run) {
                    ControlMessage::Step(Step::CallModel(_)) if replacement.is_none() => {}
                    ControlMessage::Error { error } if replacement.is_some() => {
                        assert_eq!(error.sentence, sdm_contract::Sentence::InvalidMessage)
                    }
                    other => panic!("integer coherence ({large}, {replacement:?}): {other:?}"),
                }
            }
        }
    }
}

fn classified_result(
    integer: bool,
    label: Value,
    probabilities: serde_json::Map<String, Value>,
    decimal_limit: u32,
) -> ArrowData {
    let feature = Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])) as ArrayRef;
    let target = if integer {
        Arc::new(arrow_array::Int64Array::from(vec![2, 4, 6])) as ArrayRef
    } else {
        Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef
    };
    let context = [RecordBatch::try_from_iter([("0", feature), ("1", target)]).unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(arrow_array::Int64Array::from(vec![4, 5])) as ArrayRef,
    )])
    .unwrap()];
    let mut input = prelude(
        &["feature", "target"],
        &["feature"],
        &context,
        &scoring,
        &[],
    );
    input["task"] = json!("classification");
    input["bound"] = bound_as(3, "classification");
    let envelope = plan(input.clone(), &context, &scoring);
    let run = encoded_prediction(json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"classification","primary_key":[],"inference":{}}), attached_batches(&context,&scoring), json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":["prediction","probabilities"],"default_outputs":true})).unwrap();
    let ControlMessage::Step(Step::CallModel(call)) = next(&run) else {
        panic!("model call")
    };
    let answer = json!({"id":"direct-fixture","model":"kumo-tabular","metadata":{},"predictions":(0..2).map(|i|json!({"row_index":i,"id":i.to_string(),"prediction":label,"probabilities":probabilities})).collect::<Vec<_>>()});
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
        panic!("decoded")
    };
    let laid = finish_response_limit(&run, decoded.value, None, decimal_limit).unwrap();
    let message = next(&laid);
    let ControlMessage::Step(Step::Done {
        result: OperationResult::Predict(result),
        ..
    }) = message
    else {
        panic!("layout: {message:?}")
    };
    let Attachment::Arrow(result) = laid.take_attachment(result.result).unwrap().unwrap() else {
        panic!("result")
    };
    result
}

#[test]
fn numeric_class_layout_uses_the_existing_numeric_text_grammar_and_collision_rule() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    for (integer, label, second_key, expected_key, expected_number) in [
        (true, "0_2", "4", "2", Some(2.)),
        (true, "٢", "٤", "2", Some(2.)),
        (true, "\u{2003}+0_2\u{2003}", "4", "2", Some(2.)),
        (true, "0__2", "4", "0__2", None),
        (true, "0_2", "2", "0_2", Some(2.)),
        (false, "1_0.5", "4", "10.5", Some(10.5)),
        (false, "١٠.٥", "٤", "10.5", Some(10.5)),
        (false, "1_0e-1", "4", "1.0", Some(1.)),
        (false, "1_.5", "4", "1_.5", None),
        (false, "1_0.5", "10.5", "1_0.5", Some(10.5)),
    ] {
        let mut probabilities = serde_json::Map::new();
        probabilities.insert(label.to_owned(), json!(0.75));
        probabilities.insert(second_key.to_owned(), json!(0.25));
        let result = classified_result(integer, json!(label), probabilities, 4300);
        let batch = &result.batches[0];
        let prediction = batch.column_by_name("prediction").unwrap();
        match expected_number {
            Some(expected) if integer => assert_eq!(
                prediction.as_primitive::<Int64Type>().value(0),
                expected as i64
            ),
            Some(expected) => {
                assert_eq!(prediction.as_primitive::<Float64Type>().value(0), expected)
            }
            None => assert_eq!(prediction.as_string::<i32>().value(0), label),
        }
        let classes = batch
            .column_by_name("probabilities")
            .unwrap()
            .as_map()
            .value(0);
        let keys = classes.column(0).as_string::<i32>();
        assert!((0..keys.len()).any(|at| keys.value(at) == expected_key));
        assert_eq!(keys.len(), 2);
    }
}

#[test]
fn final_integer_class_conversion_counts_original_digits_with_the_fresh_limit() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    for (zeros, limit, restored) in [
        (639, 640, true),
        (640, 640, false),
        (650, 0, true),
        (650, 640, false),
    ] {
        let label = format!("{}2", "0".repeat(zeros));
        let probabilities =
            serde_json::Map::from_iter([(label.clone(), json!(0.75)), ("4".into(), json!(0.25))]);
        // Row-ID checks run under 4300 in the helper. Final conversion receives
        // the explicitly later process observation, not any row's earlier limit.
        let result = classified_result(true, json!(label), probabilities, limit);
        let batch = &result.batches[0];
        let prediction = batch.column_by_name("prediction").unwrap();
        if restored {
            assert_eq!(prediction.as_primitive::<Int64Type>().value(0), 2);
        } else {
            assert_eq!(prediction.as_string::<i32>().value(0), label);
        }
        let classes = batch
            .column_by_name("probabilities")
            .unwrap()
            .as_map()
            .value(0);
        let keys = classes.column(0).as_string::<i32>();
        assert!((0..keys.len()).any(|at| keys.value(at) == if restored { "2" } else { &label }));
    }
}

#[test]
fn whole_float_classes_use_exact_integer_conversion_and_existing_written_carrier() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    for (value, digits, narrow) in [
        (-9223372036854775808., "-9223372036854775808", true),
        (9.21e18, "9210000000000000000", true),
        (9223372036854774784., "9223372036854774784", true),
        (9223372036854775808., "9223372036854775808", false),
        (-9223372036854777856., "-9223372036854777856", false),
    ] {
        let result = classified_result(true, json!(value), serde_json::Map::new(), 4300);
        let batch = &result.batches[0];
        if narrow {
            assert_eq!(
                batch
                    .column_by_name("prediction")
                    .unwrap()
                    .as_primitive::<Int64Type>()
                    .value(0),
                digits.parse::<i64>().unwrap()
            );
        } else {
            assert_eq!(
                batch
                    .column_by_name("prediction_as_answered")
                    .unwrap()
                    .as_string::<i32>()
                    .value(0),
                digits
            );
        }
    }
    let result = classified_result(true, json!(1.5), serde_json::Map::new(), 4300);
    assert_eq!(
        result.batches[0]
            .column_by_name("prediction")
            .unwrap()
            .as_primitive::<Float64Type>()
            .value(0),
        1.5
    );
}

#[test]
fn inferred_constructor_schema_precedes_fresh_remaining_guard_observations() {
    fn batch(columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_from_iter(
            columns
                .into_iter()
                .enumerate()
                .map(|(at, array)| (at.to_string(), array)),
        )
        .unwrap()
    }
    let initial_context = batch(vec![
        Arc::new(arrow_array::Int64Array::from(
            (0..100)
                .map(|at| if at == 0 { None } else { Some(at) })
                .collect::<Vec<_>>(),
        )),
        Arc::new(arrow_array::Int64Array::from_iter_values(0..100)),
        Arc::new(arrow_array::Int64Array::from_iter_values(100..200)),
    ]);
    let initial_predict = batch(vec![
        Arc::new(arrow_array::Int64Array::from(vec![100, 101])),
        Arc::new(arrow_array::Int64Array::from(vec![100, 101])),
    ]);
    let context_header = native_header(
        &["user_id", "feature", "target"],
        std::slice::from_ref(&initial_context),
    );
    let predict_header = native_header(
        &["user_id", "feature"],
        std::slice::from_ref(&initial_predict),
    );
    let headers = prepare(json!({"stage":"headers","arguments":arguments(),"context":context_header,"predict":predict_header,"target_context":{"positions":[2]},"target_predict":{"positions":[]},"predict_context":[[0],[1]]}),Attachments::new()).unwrap();
    let (inferred, notices) = prepare_trace(
        json!({"stage":"schema","headers":headers,"context":2,"predict":3,"entity_table":"instance"}),
        attached_batches(
            std::slice::from_ref(&initial_context),
            std::slice::from_ref(&initial_predict),
        ),
    );
    assert_eq!(notices, vec![sdm_contract::Sentence::KeyNoValueColumn]);
    let inferred = inferred.unwrap();
    assert_eq!(inferred["entity_table"], "instance");
    assert_eq!(
        inferred["description"]["tables"][0]["primary_key"],
        json!(["user_id"])
    );
    for axis in ["version", "physical", "target-position", "membership"] {
        let mut input = json!({"stage":"finish_constructor","schema":inferred,"context":2,"predict":3,"context_header":context_header,"predict_header":predict_header,"target_context":{"positions":[2]},"context_predict":[{"positions":[0]},{"positions":[1]},{"positions":[]}]});
        match axis {
            "version" => input["schema"]["version"] = json!(u32::MAX),
            "physical" => {
                input["context_header"]["columns"][0]["physical"] = json!({"type":"float64"})
            }
            "target-position" => input["target_context"]["positions"] = json!([1]),
            _ => input["context_predict"][0]["positions"] = json!([2]),
        }
        let error = prepare(
            input,
            attached_batches(
                std::slice::from_ref(&initial_context),
                std::slice::from_ref(&initial_predict),
            ),
        )
        .unwrap_err();
        assert_eq!(
            error.sentence,
            sdm_contract::Sentence::InvalidMessage,
            "{axis}"
        );
    }
    for change in [
        "baseline",
        "drop-predict",
        "drop-context",
        "add-context",
        "add-predict",
        "scored-values",
        "constant-target",
        "null-target",
        "drop-target",
        "empty-context",
        "duplicate-present",
        "duplicate-null",
    ] {
        let mut context_names = vec!["user_id", "feature", "target"];
        let mut predict_names = vec!["user_id", "feature"];
        let mut context_arrays = initial_context.columns().to_vec();
        let mut predict_arrays = initial_predict.columns().to_vec();
        match change {
            "drop-predict" => {
                predict_names.remove(1);
                predict_arrays.remove(1);
            }
            "drop-context" => {
                context_names.remove(1);
                context_arrays.remove(1);
            }
            "add-context" => {
                context_names.push("new_feature");
                context_arrays.push(Arc::new(arrow_array::Int64Array::from_iter_values(0..100)));
            }
            "add-predict" => {
                predict_names.push("new_feature");
                predict_arrays.push(Arc::new(arrow_array::Int64Array::from(vec![2, 3])));
            }
            "scored-values" => {
                predict_arrays[1] = Arc::new(arrow_array::UInt64Array::from(vec![
                    1_u64 << 63,
                    (1_u64 << 63) + 1,
                ]))
            }
            "constant-target" => {
                context_arrays[2] = Arc::new(arrow_array::Int64Array::from(vec![2; 100]))
            }
            "null-target" => {
                context_arrays[2] = Arc::new(arrow_array::Int64Array::from(vec![None; 100]))
            }
            "drop-target" => {
                context_names.pop();
                context_arrays.pop();
            }
            "duplicate-present" | "duplicate-null" => {
                context_names.push("target");
                context_arrays.push(if change == "duplicate-present" {
                    Arc::new(arrow_array::Int64Array::from_iter_values(200..300)) as ArrayRef
                } else {
                    Arc::new(arrow_array::Int64Array::from(vec![None; 100])) as ArrayRef
                });
            }
            _ => {}
        }
        let context = if change == "empty-context" {
            batch(context_arrays).slice(0, 0)
        } else {
            batch(context_arrays)
        };
        let scoring = batch(predict_arrays);
        let context_header = native_header(&context_names, std::slice::from_ref(&context));
        let predict_header = native_header(&predict_names, std::slice::from_ref(&scoring));
        let target = context_names
            .iter()
            .enumerate()
            .filter_map(|(at, name)| (*name == "target").then_some(at))
            .collect::<Vec<_>>();
        let context_predict = context_names.iter().map(|name|json!({"positions":predict_names.iter().enumerate().filter_map(|(at,p)|(*p==*name).then_some(at)).collect::<Vec<_>>()})).collect::<Vec<_>>();
        let (finished, warnings) = prepare_trace(
            json!({"stage":"finish_constructor","schema":inferred,"context":2,"predict":3,"context_header":context_header,"predict_header":predict_header,"target_context":{"positions":target},"context_predict":context_predict}),
            attached_batches(
                std::slice::from_ref(&context),
                std::slice::from_ref(&scoring),
            ),
        );
        match change {
            "drop-predict" | "add-context" => assert_eq!(
                finished.unwrap_err().sentence,
                sdm_contract::Sentence::DirectMissing
            ),
            "scored-values" => assert_eq!(
                finished.unwrap_err().sentence,
                sdm_contract::Sentence::DirectScored
            ),
            "drop-target" => assert_eq!(
                finished.unwrap_err().sentence,
                sdm_contract::Sentence::DirectSchemaMissing
            ),
            "duplicate-present" => assert_eq!(
                finished.unwrap_err().sentence,
                sdm_contract::Sentence::DirectAmbiguous
            ),
            _ => {
                let finished = finished.unwrap();
                assert_eq!(finished["schema"], inferred["schema"]);
                assert_eq!(finished["target"], "target");
            }
        }
        assert_eq!(
            warnings,
            if change == "constant-target" {
                vec![sdm_contract::Sentence::DirectConstantRegression]
            } else {
                vec![]
            },
            "{change}"
        );
    }
    // A host that promotes the first schema advice stops here. No later
    // constructor guard is consumed before it can make that native choice.
    let mut attached = attached_batches(
        std::slice::from_ref(&initial_context),
        std::slice::from_ref(&initial_predict),
    );
    attached.insert(AttachmentId(1),bytes(json!({"stage":"schema","headers":headers,"context":2,"predict":3,"entity_table":"instance"})));
    let run = start(json!({"op":"prepare_request","input":1}), attached);
    assert!(matches!(
        next(&run),
        ControlMessage::Warnings { .. } | ControlMessage::Step(Step::Warnings)
    ));
}

#[test]
fn constructor_task_metadata_observes_only_fresh_lookup_after_constant_advice() {
    let inferred=prepare(json!({"stage":"schema","headers":headers(),"context":2,"predict":3,"entity_table":"instance"}),halves()).unwrap();
    for restored in [false, true] {
        let mut supplied = inferred.clone();
        if restored {
            supplied["description"]["tables"][0]["columns"]
                .as_array_mut()
                .unwrap()
                .remove(1);
            supplied["schema"]["tables"][0]["columns"]
                .as_array_mut()
                .unwrap()
                .remove(1);
        }
        let mut attached = halves();
        attached.insert(
            AttachmentId(2),
            rows(&[("0", vec![1., 2., 3.]), ("1", vec![2., 2., 2.])]),
        );
        let (prepared, notices) = prepare_trace(
            json!({"stage":"finish_constructor","schema":supplied,"context":2,"predict":3,"context_header":header(&["feature","target"],3),"predict_header":header(&["feature"],2),"target_context":{"positions":[1]},"context_predict":[{"positions":[0]},{"positions":[]}]}),
            attached,
        );
        assert_eq!(
            notices,
            vec![sdm_contract::Sentence::DirectConstantRegression]
        );
        let prepared = prepared.unwrap();
        let columns = if restored {
            vec![
                json!({"kind":"text","value":"feature"}),
                json!({"kind":"text","value":"target"}),
            ]
        } else {
            vec![json!({"kind":"text","value":"feature"})]
        };
        let result = prepare(
            json!({"stage":"constructor_task","prepared":prepared,"entity_table":"instance","tables":[{"kind":"text","value":"instance"}],"table":{"columns":columns,"target_dtype":if restored {Some("float64")} else {None}}}),
            Attachments::new(),
        );
        if restored {
            assert_eq!(
                result.unwrap(),
                json!({"kind":"regression","target":"target","table":"instance","dtype":"float64"})
            );
        } else {
            assert_eq!(
                result.unwrap_err().sentence,
                sdm_contract::Sentence::NoColumn
            );
        }
    }
}

#[test]
fn boolean_and_numeric_halves_share_numeric_wire_categories() {
    let context = [RecordBatch::try_from_iter([
        (
            "0",
            Arc::new(arrow_array::BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
            ])) as ArrayRef,
        ),
        (
            "1",
            Arc::new(arrow_array::Int64Array::from(vec![2, 4, 6])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(arrow_array::UInt64Array::from(vec![0, 1])) as ArrayRef,
    )])
    .unwrap()];
    let input = prelude(
        &["feature", "target"],
        &["feature"],
        &context,
        &scoring,
        &[],
    );
    let envelope = plan(input.clone(), &context, &scoring);
    let (_,payload)=encoded_body(json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"regression","primary_key":[]}),attached_batches(&context,&scoring)).unwrap();
    let payload: Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(
        payload["schema"]["instance_table"]["columns"]["feature"],
        json!({"dtype":"int64","stype":"categorical"})
    );
    assert_eq!(
        payload["context"]["instance_table"]["rows"],
        json!([[1, 2], [0, 4], [null, 6]])
    );
    assert_eq!(
        payload["predict"]["instance_table"]["rows"],
        json!([[0], [1]])
    );
}

#[test]
fn direct_row_bounds_use_exact_decoded_digits_before_machine_width() {
    let digits = "1606938044258990275541962092341162602522202993782792835301376";
    for index in [digits.to_owned(), format!("-{digits}")] {
        let scope = json!({"version":1,"model":"kumo-tabular","request_id":"response-nonce","call_request_id":"sdm-fixed","response_rows":1,"scoring_rows":2});
        let row = json!({"ordinal":0,"row_index":index,"id":null,"key":null,"raw_in_batch":false,"prior_ordinal":null});
        let error = prepare(
            json!({"stage":"check_response_rows","scope":scope,"rows":[row],"decimal_limit":4300}),
            Attachments::new(),
        )
        .unwrap_err();
        assert_eq!(error.sentence, sdm_contract::Sentence::TabularRowBounds);
        assert_eq!(error.fields["index"], sdm_contract::Detail::text(index));
        assert_eq!(
            error.fields["rows"],
            sdm_contract::Detail::Scalar(Scalar::UInt64(2))
        );
    }
}

#[test]
fn wide_bounds_are_reached_after_earlier_native_conversion_outcome() {
    let checked = checked_native_row(r"[4+1, '\q']");
    let literal = json!({"scope":checked["scope"],"ordinal":0,"text":r"[4+1, '\q']","value":{"outcome":"failed","failure":{"ancestry":["value_error"]}}});
    let selected = prepare(
        json!({"stage":"select_response_member","checked":checked,"literal":literal}),
        Attachments::new(),
    )
    .unwrap();
    let accepted = prepare(
        json!({"stage":"attribute_response_row","selected":selected,"string":null}),
        Attachments::new(),
    )
    .unwrap();
    assert_eq!(accepted["id"], json!({"t":"i64","v":"4"}));
    let index = "9".repeat(663);
    let row = json!({"ordinal":1,"row_index":index,"id":"opaque-b","key":null,"raw_in_batch":false,"prior_ordinal":null});
    let error=prepare(json!({"stage":"check_response_rows","scope":checked["scope"],"rows":[row],"decimal_limit":640}),Attachments::new()).unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::TabularRowBounds);
    assert_eq!(error.fields["index"], sdm_contract::Detail::text(index));
}

#[test]
fn encoded_body_facts_and_rows_must_agree_before_transport() {
    let context = [RecordBatch::try_from_iter([
        (
            "0",
            Arc::new(Float64Array::from(vec![1., 2., 3.])) as ArrayRef,
        ),
        (
            "1",
            Arc::new(Float64Array::from(vec![2., 4., 6.])) as ArrayRef,
        ),
    ])
    .unwrap()];
    let scoring = [RecordBatch::try_from_iter([(
        "0",
        Arc::new(Float64Array::from(vec![4., 5.])) as ArrayRef,
    )])
    .unwrap()];
    let input = prelude(
        &["feature", "target"],
        &["feature"],
        &context,
        &scoring,
        &[],
    );
    let envelope = plan(input.clone(), &context, &scoring);
    let (body,payload)=encoded_body(json!({"envelope":envelope,"context":2,"predict":3,"context_header":input["context_header"],"predict_header":input["predict_header"],"task":"regression","primary_key":[]}),attached_batches(&context,&scoring)).unwrap();
    for rows_incoherent in [false, true] {
        let mut body = body.clone();
        body["payload"] = json!(2);
        let mut payload: Value = serde_json::from_slice(&payload).unwrap();
        if rows_incoherent {
            payload["context"]["instance_table"]["rows"][0] = json!([1.]);
        } else {
            body["context_shape"]["rows"] = json!(4);
        }
        let run = start(
            json!({"op":"predict","direct":1,"model":"kumo-tabular","request_id":"sdm-fixed","outputs":["prediction"],"default_outputs":true}),
            Attachments::from([
                (
                    AttachmentId(1),
                    bytes(json!({"stage":"encoded","body":body,"inference":{}})),
                ),
                (AttachmentId(2), bytes(payload)),
            ]),
        );
        let ControlMessage::Error { error } = next(&run) else {
            panic!("incoherent encoded rows must not call a model")
        };
        assert_eq!(error.sentence, sdm_contract::Sentence::InvalidMessage);
    }
}
