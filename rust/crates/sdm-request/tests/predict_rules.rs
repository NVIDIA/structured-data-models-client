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
        forecast_step: None,
        forecast_step_as_answered: None,
        answer_index: 0,
        id: Entity::Integer(id),
        prediction,
        probabilities: None,
        scores: None,
        rankings: None,
        explanation: None,
        metadata: None,
        quantiles: None,
        embeddings: None,
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
        explanation: None,
    }
}

#[test]
fn gathered_regression_retains_its_existing_prediction_admission() {
    use sdm_wire::json;
    let payload = json::read(br#"{"model":"kumo-relational","task":{"kind":"regression","target":{"column_name":"target","dtype":"int64"}},"predict":{"instance_table":{"format":"arrays","columns":[],"rows":[[]]}}}"#).unwrap();
    for (prediction, accepted) in [
        ("2".to_owned(), true),
        ("2.0".to_owned(), true),
        ("1.5".to_owned(), true),
        ("1".to_owned() + &"0".repeat(660), false),
        ("false".to_owned(), false),
        (r#""2""#.to_owned(), false),
        ("[]".to_owned(), false),
        ("{}".to_owned(), false),
    ] {
        let answer = format!(
            r#"{{"id":"r","model":"kumo-relational","metadata":{{}},"predictions":[{{"row_index":0,"prediction":{prediction}}}]}}"#
        );
        let response = sdm_wire::response::PredictionResponse::read(
            &json::read_answer(answer.as_bytes()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            sdm_request::predict::layout(&payload, &response, |v| v.to_string()).is_ok(),
            accepted,
            "{prediction}"
        );
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
    // Each answer's place in the endpoint's order, the parts in order.
    let order: Vec<i64> = batch
        .column(batch.num_columns() - 1)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    assert_eq!(order, [1, 2, 0]);
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
    // missing one among them is null.
    let batch = table(&unanswered).unwrap();
    let names: Vec<String> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["row_index", "id", "answer_index"]);
    let batch = table(&answered).unwrap();
    let column = batch.column(2).as_primitive::<Float64Type>();
    // A missing prediction is null, as the reference's item is None; a NaN
    // is a NaN (fix round 1).
    assert!(column.is_null(0) && column.value(1).is_nan());
    assert_eq!(column.null_count(), 1);
}

#[test]
fn a_regression_s_column_is_typed_as_pandas_types_it() {
    let names = |part: &Part| -> Vec<String> {
        table(part)
            .unwrap()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };
    // Every answer an integer: int64, exact past 2**53 and at int64's ends.
    let ints = part(
        "r1",
        vec![
            row(0, 7, Answer::Int(9_007_199_254_740_993)),
            row(1, 8, Answer::Int(i64::MIN)),
            row(2, 9, Answer::Int(i64::MAX)),
        ],
    );
    let batch = table(&ints).unwrap();
    assert_eq!(
        batch
            .column(2)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        [9_007_199_254_740_993, i64::MIN, i64::MAX]
    );
    assert_eq!(
        names(&ints),
        ["row_index", "id", "prediction", "answer_index"]
    );
    // Mixed with a float or a missing one: float64, and each answer as
    // written beside it, so a host's items keep the integer exact.
    let mixed = part(
        "r1",
        vec![
            row(0, 7, Answer::Int(9_007_199_254_740_993)),
            row(1, 8, Answer::Float(1.5)),
            row(2, 9, Answer::None),
        ],
    );
    let batch = table(&mixed).unwrap();
    assert_eq!(
        batch.column(2).data_type(),
        &arrow_schema::DataType::Float64
    );
    let written: Vec<Option<&str>> = batch.column(3).as_string::<i32>().iter().collect();
    assert_eq!(written, [Some("9007199254740993"), Some("1.5"), None]);
    // A class written as a non-finite number: no value of the target's
    // type, the item's own the number.
    let mut nan_class = part(
        "r1",
        vec![
            row(0, 7, Answer::NonFinite(f64::NAN)),
            row(1, 8, Answer::Bool(true)),
        ],
    );
    nan_class.answers = AnswerType::Bool;
    let batch = table(&nan_class).unwrap();
    assert!(batch.column(2).is_null(0));
    let written: Vec<Option<&str>> = batch.column(3).as_string::<i32>().iter().collect();
    assert_eq!(written, [Some("NaN"), Some("true")]);
}

#[test]
fn gathered_class_predictions_and_probability_keys_retain_the_original_grammar() {
    use sdm_wire::json;
    for (dtype, token, key, prediction, spelled) in [
        ("int64", r#""10""#, "10", Answer::Int(10), "10"),
        (
            "int64",
            r#""1_0""#,
            "1_0",
            Answer::Written(Value::text("1_0")),
            "1_0",
        ),
        (
            "int64",
            r#""١٠""#,
            "١٠",
            Answer::Written(Value::text("١٠")),
            "١٠",
        ),
        (
            "int64",
            "9.19e18",
            "1",
            Answer::Int(9_190_000_000_000_000_000),
            "1",
        ),
        (
            "int64",
            "9.21e18",
            "1",
            Answer::Written(Value::Number("9.21e18".into())),
            "1",
        ),
        (
            "int64",
            "-9.21e18",
            "1",
            Answer::Written(Value::Number("-9.21e18".into())),
            "1",
        ),
        ("float64", r#""10.5""#, "10.5", Answer::Float(10.5), "10.5"),
        (
            "float64",
            r#""1_0.5""#,
            "1_0.5",
            Answer::Written(Value::text("1_0.5")),
            "1_0.5",
        ),
        (
            "float64",
            r#""١٠.5""#,
            "١٠.5",
            Answer::Written(Value::text("١٠.5")),
            "١٠.5",
        ),
    ] {
        let payload = json::read(format!(r#"{{"model":"kumo-relational","task":{{"kind":"multiclass_classification","target":{{"column_name":"target","dtype":"{dtype}"}}}},"predict":{{"instance_table":{{"format":"arrays","columns":[],"rows":[[]]}}}}}}"#).as_bytes()).unwrap();
        let value = json::read_answer(format!(r#"{{"id":"server","model":"kumo-relational","metadata":{{}},"predictions":[{{"row_index":0,"id":"0","prediction":{token},"probabilities":{{"{key}":0.75}}}}]}}"#).as_bytes()).unwrap();
        let response = sdm_wire::response::PredictionResponse::read(&value).unwrap();
        let part = sdm_request::predict::layout(&payload, &response, |v| v.to_string()).unwrap();
        assert_eq!(part.rows[0].prediction, prediction, "{dtype}/{token}");
        assert_eq!(
            part.rows[0].probabilities.as_deref(),
            Some([(spelled.to_owned(), 0.75)].as_slice())
        );
    }
}
