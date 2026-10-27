// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual HostRun model exchange and fixed response products for decomposed
//! frame forecasts/rankings. Request bodies are the existing frozen public
//! materialization fixtures; model answers are explicitly scripted here.
//! This covers response ownership, attribution and final composition, not
//! native Python callbacks, graph construction or live model accuracy.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::prepare::*;
use sdm_contract::run::{Endpoint, ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn bytes<T: serde::Serialize>(value: &T) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn start(input: Preparation, mut attached: Attachments) -> HostRun {
    assert!(attached.insert(AttachmentId(1), bytes(&input)).is_none());
    let message = sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                value: 0,
                unit: TimestampUnit::Ns,
                zone: Some("UTC".into()),
            },
            local_zone: None,
            zone_roots: vec![],
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
            date_literal_range: None,
        },
        operation: Operation::PrepareRequest(PrepareRequest {
            input: AttachmentId(1),
        }),
    })))
    .unwrap();
    HostRun::start(&sdm_host::default_root(), &message, attached)
        .unwrap_or_else(|out| panic!("start: {:?}", sdm_host::read(&out)))
}

fn take_bytes(run: &HostRun, id: AttachmentId) -> Vec<u8> {
    match run.take_attachment(id).unwrap().unwrap() {
        Attachment::Bytes(value) => value,
        _ => panic!("expected bytes at {id:?}"),
    }
}

fn done<T: DeserializeOwned>(run: &HostRun) -> T {
    let out = run
        .next()
        .unwrap_or_else(|out| panic!("next: {:?}", sdm_host::read(&out)));
    let ControlMessage::Step(Step::Done {
        result: OperationResult::PrepareRequest(result),
    }) = sdm_host::read(&out).unwrap()
    else {
        panic!("unexpected preparation step: {:?}", sdm_host::read(&out));
    };
    assert!(run.take_warnings().unwrap().is_none());
    serde_json::from_slice(&take_bytes(run, result.value)).unwrap()
}

fn prepare<T: DeserializeOwned>(input: Preparation, attached: Attachments) -> T {
    done(&start(input, attached))
}

fn bound(kind: &str) -> CallPreparation {
    let model = prepare(
        Preparation::Model {
            model: "kumo-relational".into(),
        },
        Attachments::new(),
    );
    let outputs = prepare(
        Preparation::Outputs {
            call: model,
            outputs: None,
            kind: Some(kind.into()),
            native_default: None,
        },
        Attachments::new(),
    );
    let shape = prepare(
        Preparation::Capabilities {
            call: outputs,
            kind: kind.into(),
            anchored: true,
            relationships: true,
            tables: 3,
            native: None,
        },
        Attachments::new(),
    );
    let task = prepare(
        Preparation::Task {
            call: shape,
            kind: kind.into(),
            admitted: None,
        },
        Attachments::new(),
    );
    prepare(
        Preparation::Capacity {
            call: task,
            rows: 3,
        },
        Attachments::new(),
    )
}

fn payloads(family: &str) -> Vec<Value> {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../../packages/structured-data-models-client/tests/t44_decomposition_reference.json"
    ))
    .unwrap();
    serde_json::from_value(reference["families"][family]["seed7"].clone()).unwrap()
}

struct OwnedPart {
    part: RelationalResponsePart,
    buffers: [Vec<u8>; 4],
}

impl OwnedPart {
    fn take(run: &HostRun, part: RelationalResponsePart) -> Self {
        let buffers = [
            take_bytes(run, part.data.response),
            take_bytes(run, part.data.rows),
            take_bytes(run, part.rows),
            take_bytes(run, part.target_rows),
        ];
        Self { part, buffers }
    }

    fn attach(&self, attached: &mut Attachments) -> RelationalResponsePart {
        let first = u32::try_from(attached.len()).unwrap() + 2;
        let mut part = self.part.clone();
        for (offset, buffer) in self.buffers.iter().enumerate() {
            let id = AttachmentId(first + u32::try_from(offset).unwrap());
            assert!(
                attached
                    .insert(id, Attachment::Bytes(buffer.clone()))
                    .is_none()
            );
        }
        part.data.response = AttachmentId(first);
        part.data.rows = AttachmentId(first + 1);
        part.rows = AttachmentId(first + 2);
        part.target_rows = AttachmentId(first + 3);
        part
    }
}

fn exchange(payload: &Value, child: usize, classification: bool) -> (CallPreparation, OwnedPart) {
    let kind = payload["task"]["kind"].as_str().unwrap();
    let bound = bound(kind);
    let request_id = format!("host-call-{child}");
    let body = serde_json::to_vec(payload).unwrap();
    let run = start(
        Preparation::CallRelational {
            input: RelationalEncodedCall {
                call: bound.clone(),
                body: AttachmentId(2),
                request_id: request_id.clone(),
                model_endpoint: None,
            },
        },
        Attachments::from([(AttachmentId(2), Attachment::Bytes(body.clone()))]),
    );
    let out = run.next().unwrap();
    let ControlMessage::Step(Step::CallModel(call)) = sdm_host::read(&out).unwrap() else {
        panic!("encoded relational request did not suspend for its model call");
    };
    assert_eq!(call.endpoint, Endpoint::Predict);
    assert_eq!(call.content_type, "application/json");
    assert_eq!(take_bytes(&run, call.body), body);

    let source = payload["predict"]["instance_table"]["rows"]
        .as_array()
        .unwrap();
    assert_eq!(source.len(), 2);
    // Deliberately answer in reverse source order. Final Arrow sorting and
    // answer_index must retain the separate original response lineage.
    let predictions = [1usize, 0]
        .into_iter()
        .map(|row| {
            let mut answer = json!({
                "row_index":row,
                "id":source[row][0].as_i64().unwrap().to_string(),
                "prediction":if classification { json!(true) } else { json!([[0.0,1.0],[10.0,11.0]][child][row]) }
            });
            if classification {
                let positive = [[0.8, 0.2], [0.6, 0.9]][child][row];
                answer["probabilities"] = json!({"False":1.0-positive,"True":positive});
            }
            answer
        })
        .collect::<Vec<_>>();
    let response = json!({
        "id":format!("scripted-answer-{child}"),
        "model":"kumo-relational", "metadata":{}, "predictions":predictions
    });
    let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
        id: call.id,
        outcome: ModelOutcome::Response {
            status: 200,
            content_type: "application/json".into(),
            body: AttachmentId(99),
        },
    }))
    .unwrap();
    run.reply(
        &reply,
        Attachments::from([(AttachmentId(99), bytes(&response))]),
    )
    .unwrap();
    let mut data: RelationalResponseData = done(&run);
    assert_eq!(data.scope.call_request_id, request_id);
    assert_eq!(data.scope.request_id, format!("scripted-answer-{child}"));
    assert_eq!(data.scope.response_rows, 2);
    let response_bytes = take_bytes(&run, data.response);
    let decoded_bytes = take_bytes(&run, data.rows);
    // The later native facts retain their own buffers after this model run
    // is dropped, just as a different language host must do.
    drop(run);
    let decoded: Vec<DecodedRelationalRow> = serde_json::from_slice(&decoded_bytes).unwrap();
    assert_eq!(decoded[0].row_index, "1");
    assert_eq!(decoded[1].row_index, "0");
    let reference: RelationalReferenceColumn = prepare(
        Preparation::RelationalReferenceColumn {
            facts: RelationalReferenceColumns {
                prefix_matches: vec![false, true],
                exhausted: false,
            },
        },
        Attachments::new(),
    );
    assert_eq!(reference.ordinal, Some(1));
    let attribution = RelationalAttribution {
        scope: data.scope.clone(),
        source_rows: Some(source.len()),
        instances: true,
    };
    let mut attributed = Vec::new();
    for row in &decoded {
        let rows: CheckedRelationalRows = prepare(
            Preparation::CheckRelationalRows {
                attribution: attribution.clone(),
                rows: vec![row.clone()],
            },
            Attachments::new(),
        );
        assert!(rows.rows.is_empty());
        let checked: CheckedRelationalRow = prepare(
            Preparation::CheckRelationalForecast {
                bound: rows.forecast.unwrap(),
                // Forecast decomposition's model calls are regression
                // children, not a direct forecasting wire task.
                forecasting: false,
            },
            Attachments::new(),
        );
        let index = checked.bound.source_row.unwrap();
        let displayed = source[index][0].as_i64().unwrap().to_string();
        let accepted: AttributedRelationalRow = prepare(
            Preparation::AttributeRelationalInstance {
                checked,
                instance: RelationalInstanceText {
                    ordinal: row.ordinal,
                    points: displayed.chars().map(u32::from).collect(),
                    different: row.id.as_deref() != Some(displayed.as_str()),
                },
            },
            Attachments::new(),
        );
        assert_eq!(accepted.source_row, Some(index));
        attributed.push(accepted);
    }
    data.response = AttachmentId(2);
    data.rows = AttachmentId(3);
    let mut attached = Attachments::from([
        (AttachmentId(2), Attachment::Bytes(response_bytes)),
        (AttachmentId(3), Attachment::Bytes(decoded_bytes)),
        (AttachmentId(4), bytes(&attributed)),
    ]);
    let covered: RelationalAttribution = prepare(
        Preparation::CheckRelationalCoverage {
            data: data.clone(),
            attribution: attribution.clone(),
            rows: AttachmentId(4),
        },
        attached.clone(),
    );
    assert_eq!(covered, attribution);
    let meaning: RelationalTargetMeaning = prepare(
        Preparation::RelationalTargetMeaning {
            kind: kind.chars().map(u32::from).collect(),
            classification,
        },
        Attachments::new(),
    );
    assert_eq!(meaning.classification, classification);
    let converted = decoded
        .iter()
        .map(|row| RelationalTargetRow {
            ordinal: row.ordinal,
            prediction: row.prediction.clone(),
            probabilities: row.probabilities.clone(),
        })
        .collect::<Vec<_>>();
    attached.insert(AttachmentId(5), bytes(&converted));
    let part = RelationalResponsePart {
        data,
        attribution,
        rows: AttachmentId(4),
        target: RelationalTargetFacts {
            kind: kind.chars().map(u32::from).collect(),
            classification,
            dtype_truth: classification.then_some(true),
            numeric_prefix: classification.then_some(true),
        },
        target_rows: AttachmentId(5),
    };
    let completion = start(Preparation::CompleteRelationalResponse { part }, attached);
    let part = done(&completion);
    (bound, OwnedPart::take(&completion, part))
}

fn children(calls: &[(CallPreparation, OwnedPart)]) -> (Vec<RelationalCallResult>, Attachments) {
    let mut attached = Attachments::new();
    let children = calls
        .iter()
        .map(|(bound, part)| RelationalCallResult {
            bound: bound.clone(),
            parts: vec![part.attach(&mut attached)],
            counted: vec![Some(2)],
            known_empty: false,
        })
        .collect();
    (children, attached)
}

fn finish(
    bound: CallPreparation,
    aggregation: RelationalAggregation,
    attached: Attachments,
) -> (RelationalResultData, ArrowData) {
    let run = start(
        Preparation::FinishRelationalResponse {
            input: RelationalResultInput {
                bound,
                aggregation,
                report_absent: true,
            },
        },
        attached,
    );
    let result: RelationalResultData = done(&run);
    assert_eq!(result.predicted.model, "kumo-relational");
    assert_eq!(result.predicted.request_id, "scripted-answer-0");
    assert_eq!(
        take_bytes(&run, result.predicted.metadata).as_slice(),
        b"{}"
    );
    let Attachment::Arrow(rows) = run
        .take_attachment(result.predicted.result)
        .unwrap()
        .unwrap()
    else {
        panic!("finished relational result was not Arrow");
    };
    (result, rows)
}

#[test]
fn forecast_children_exchange_decode_attribute_and_finish_with_step_lineage() {
    let calls = payloads("forecast")
        .iter()
        .enumerate()
        .map(|(child, payload)| exchange(payload, child, false))
        .collect::<Vec<_>>();
    let (children, attached) = children(&calls);
    let (result, rows) = finish(
        calls[0].0.clone(),
        RelationalAggregation::Steps {
            children,
            steps: vec![1, 2],
        },
        attached,
    );
    assert_eq!(result.answer_details.get("parts_answered"), Some(&2));
    assert_eq!(result.lineage.len(), 4);
    let batch = &rows.batches[0];
    let positions = batch
        .column_by_name("row_index")
        .unwrap()
        .as_primitive::<Int64Type>();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int64Type>();
    let steps = batch
        .column_by_name("forecast_step")
        .unwrap()
        .as_primitive::<Int64Type>();
    let answers = batch
        .column_by_name("answer_index")
        .unwrap()
        .as_primitive::<Int64Type>();
    let values = batch
        .column_by_name("prediction")
        .unwrap()
        .as_primitive::<Float64Type>();
    assert_eq!(batch.num_rows(), 4);
    for (row, (source, step, value, answer)) in [
        (0, 1, 0.0, 1),
        (0, 2, 10.0, 3),
        (1, 1, 1.0, 0),
        (1, 2, 11.0, 2),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(positions.value(row), source);
        assert_eq!(ids.value(row), source);
        assert_eq!(steps.value(row), step);
        assert_eq!(values.value(row), value);
        assert_eq!(answers.value(row), answer);
        let lineage = &result.lineage[usize::try_from(answer).unwrap()];
        assert_eq!(lineage.source_row, Some(usize::try_from(source).unwrap()));
        assert_eq!(lineage.answer_ordinal, usize::try_from(1 - source).unwrap());
        assert_eq!(lineage.call_request_id, format!("host-call-{}", step - 1));
    }
}

#[test]
fn ranking_children_exchange_then_core_confidence_ordering_retains_group_owners() {
    let calls = payloads("ranking")
        .iter()
        .enumerate()
        .map(|(child, payload)| exchange(payload, child, true))
        .collect::<Vec<_>>();
    let (children, attached) = children(&calls);
    let population: RelationalRankPopulation = prepare(
        Preparation::PrepareRelationalRankPopulation {
            source: RelationalRankingSource {
                children: children.clone(),
                positions: vec![0, 1],
            },
        },
        attached.clone(),
    );
    assert_eq!(population.items.len(), 4);
    for (item, expected) in population.items.iter().zip([0.2, 0.8, 0.9, 0.6]) {
        let confidence: RelationalRankConfidence = prepare(
            Preparation::RankRelationalScore { item: item.clone() },
            Attachments::new(),
        );
        assert_eq!(confidence.score, Scalar::Float64(expected));
    }
    assert_eq!(
        population
            .items
            .iter()
            .map(|item| item.row_index)
            .collect::<Vec<_>>(),
        [1, 0, 1, 0]
    );
    // These are the ordinary integer source-key dictionary observations.
    // The host reports membership only; confidence and top-k stay in Rust.
    let groups = RelationalRankGroups {
        membership: (0..4)
            .map(|ordinal| RelationalRankMembership {
                ordinal,
                score_first: ordinal % 2,
                asked_first: ordinal % 2,
            })
            .collect(),
        groups: vec![
            RelationalRankGroup {
                members: vec![0, 2],
                representative: 0,
            },
            RelationalRankGroup {
                members: vec![1, 3],
                representative: 1,
            },
        ],
    };
    let mut attached = attached;
    assert!(attached.insert(AttachmentId(10), bytes(&groups)).is_none());
    let (result, rows) = finish(
        calls[0].0.clone(),
        RelationalAggregation::Ranking {
            children,
            positions: vec![0, 1],
            candidates: vec!["0".into(), "1".into()],
            top_k: Some(1),
            outputs: vec!["rankings".into()],
            unscored_candidates: vec![],
            groups: AttachmentId(10),
        },
        attached,
    );
    assert_eq!(result.answer_details.get("parts_answered"), Some(&2));
    assert_eq!(result.predicted.outputs, ["rankings"]);
    let batch = &rows.batches[0];
    assert_eq!(batch.num_rows(), 2);
    for (row, (source, candidate, score, answer)) in
        [(0, "0", 0.8, 1), (1, "1", 0.9, 0)].into_iter().enumerate()
    {
        assert_eq!(
            batch
                .column_by_name("row_index")
                .unwrap()
                .as_primitive::<Int64Type>()
                .value(row),
            source
        );
        assert_eq!(
            batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<Int64Type>()
                .value(row),
            source
        );
        assert_eq!(
            batch
                .column_by_name("answer_index")
                .unwrap()
                .as_primitive::<Int64Type>()
                .value(row),
            answer
        );
        let ranked = batch
            .column_by_name("rankings")
            .unwrap()
            .as_list::<i32>()
            .value(row);
        let scores = batch
            .column_by_name("scores")
            .unwrap()
            .as_list::<i32>()
            .value(row);
        let candidates = batch
            .column_by_name("candidates")
            .unwrap()
            .as_list::<i32>()
            .value(row);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked.as_string::<i32>().value(0), candidate);
        assert_eq!(scores.as_primitive::<Float64Type>().value(0), score);
        assert_eq!(candidates.as_primitive::<Int64Type>().value(0), source);
        let lineage = &result.lineage[usize::try_from(answer).unwrap()];
        assert_eq!(lineage.call_request_id, "host-call-0");
        assert_eq!(lineage.source_row, Some(usize::try_from(source).unwrap()));
        assert_eq!(lineage.answer_ordinal, usize::try_from(1 - source).unwrap());
    }
}
