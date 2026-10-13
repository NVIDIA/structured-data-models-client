// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The predict-side rules no public call reaches on the core route, held
//! here (T1.28b): the parts of one request merged into one result
//! (`PredictionResult.merge`), which a request state of one payload never
//! needs, and the outputs an answer leaves out (`PredictionResult.absent`),
//! with the result's layout of merged parts ordered by `row_index` as the
//! reference's frame orders them.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use sdm_contract::ErrorCode;
use sdm_request::predict::{Answer, AnswerType, Entity, Part, Row, absent, merge, table};
use sdm_wire::json::Value;

fn row(row_index: i64, id: i64, prediction: Answer) -> Row {
    Row {
        row_index,
        id: Entity::Integer(id),
        prediction,
        probabilities: None,
    }
}

fn part(request_id: &str, rows: Vec<Row>) -> Part {
    Part {
        model: "kumo-relational".into(),
        request_id: request_id.into(),
        metadata: Value::Object(vec![("served_by".into(), Value::text("a"))]),
        rows,
        answers: AnswerType::Float,
        text_ids: false,
    }
}

#[test]
fn parts_are_merged_as_the_reference_merges_them() {
    // None: refused, as the reference's merge refuses nothing to merge.
    assert_eq!(merge(vec![]).unwrap_err().code, ErrorCode::InternalError);
    // One: as it is, no batch fields.
    let one = part("r1", vec![row(0, 7, Answer::Float(1.0))]);
    assert_eq!(merge(vec![one.clone()]).unwrap(), one);
    // Several: every part's answers, the first's model and id, its metadata
    // with the number of parts and each part's id.
    let merged = merge(vec![
        part(
            "r1",
            vec![row(1, 8, Answer::Float(2.0)), row(0, 7, Answer::Float(1.0))],
        ),
        part("r2", vec![row(0, 9, Answer::Float(3.0))]),
    ])
    .unwrap();
    assert_eq!(merged.request_id, "r1");
    assert_eq!(merged.rows.len(), 3);
    assert_eq!(
        String::from_utf8(merged.metadata.to_bytes()).unwrap(),
        r#"{"served_by":"a","batches":2,"batch_request_ids":["r1","r2"]}"#
    );
    // Laid out by row_index, stable among the parts' equal ones, as the
    // reference's frame sorts the merged items.
    let batch = table(&merged).unwrap();
    let indexes: Vec<i64> = batch
        .column(0)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    let ids: Vec<i64> = batch
        .column(1)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    let predictions: Vec<f64> = batch
        .column(2)
        .as_primitive::<Float64Type>()
        .values()
        .to_vec();
    assert_eq!(indexes, [0, 0, 1]);
    assert_eq!(ids, [7, 9, 8]);
    assert_eq!(predictions, [1.0, 3.0, 2.0]);
    // Parts of different models are not one request's.
    let mut other = part("r3", vec![]);
    other.model = "kumo-tabular".into();
    let error = merge(vec![part("r1", vec![]), other]).unwrap_err();
    assert_eq!(error.code, ErrorCode::InternalError);
}

#[test]
fn an_absent_output_is_one_no_answer_carries() {
    let named = |names: &[&str]| names.iter().map(|n| (*n).to_owned()).collect::<Vec<_>>();
    let answered = part(
        "r1",
        vec![row(0, 7, Answer::None), row(1, 8, Answer::Float(f64::NAN))],
    );
    // A NaN is carried, as the reference's float is; a null is not.
    assert!(absent(&answered, &named(&["prediction"])).is_empty());
    let unanswered = part("r1", vec![row(0, 7, Answer::None)]);
    assert_eq!(
        absent(
            &unanswered,
            &named(&["prediction", "probabilities", "prediction", "bogus"])
        ),
        ["prediction", "probabilities"]
    );
    // An answer with no rows leaves nothing out.
    assert!(absent(&part("", vec![]), &named(&["prediction"])).is_empty());
    // The prediction column is left out when no answer carries one, and a
    // null among them is NaN, as the reference's frame gives them.
    let batch = table(&unanswered).unwrap();
    assert_eq!(batch.num_columns(), 2);
    let batch = table(&answered).unwrap();
    let column = batch.column(2).as_primitive::<Float64Type>();
    assert!(column.value(0).is_nan() && column.value(1).is_nan());
    assert_eq!(column.null_count(), 0);
}
