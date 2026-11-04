// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapter from owned normalized Arrow results to the common result merge policy.
use crate::attached::take_json;
use crate::run::{Asker, Attachment, Attachments, internal};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, MapArray, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{
    CoreError,
    ops::OperationResult,
    result::{MergeInput, MergeResults, ResultPart},
};
use sdm_request::predict::{self, Answer, AnswerType, Entity, Part, Row};
use sdm_wire::json::{self, Value};
use std::sync::Arc;

fn json(text: &str) -> Result<Value, CoreError> {
    json::read_answer(text.as_bytes())
        .map_err(|_| internal("normalized result contains invalid JSON"))
}
fn scalar(array: &dyn Array, row: usize) -> Result<Value, CoreError> {
    if array.is_null(row) {
        return Ok(Value::Null);
    }
    Ok(match array.data_type() {
        DataType::Int64 => Value::Number(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row)
                .to_string(),
        ),
        DataType::Float64 => {
            let value = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row);
            Value::float(value).unwrap_or_else(|| {
                Value::Number(
                    if value.is_nan() {
                        "NaN"
                    } else if value.is_sign_negative() {
                        "-Infinity"
                    } else {
                        "Infinity"
                    }
                    .into(),
                )
            })
        }
        DataType::Boolean => Value::Bool(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row),
        ),
        DataType::Utf8 => Value::Text(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .into(),
        ),
        DataType::List(_) => {
            let values = array
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap()
                .value(row);
            Value::Array(
                (0..values.len())
                    .map(|i| scalar(values.as_ref(), i))
                    .collect::<Result<_, _>>()?,
            )
        }
        DataType::Map(_, _) => {
            let entries = array
                .as_any()
                .downcast_ref::<MapArray>()
                .unwrap()
                .value(row);
            Value::Object(
                (0..entries.len())
                    .map(|i| {
                        let Value::Text(key) = scalar(entries.column(0).as_ref(), i)? else {
                            return Err(internal("result mapping has non-text key"));
                        };
                        Ok((key, scalar(entries.column(1).as_ref(), i)?))
                    })
                    .collect::<Result<_, CoreError>>()?,
            )
        }
        _ => return Err(internal("normalized result column has unsupported storage")),
    })
}
fn cell(batch: &RecordBatch, name: &str, row: usize) -> Result<Value, CoreError> {
    batch
        .column_by_name(name)
        .map_or(Ok(Value::Null), |array| scalar(array.as_ref(), row))
}
fn restored(batch: &RecordBatch, name: &str, raw: &str, row: usize) -> Result<Value, CoreError> {
    match cell(batch, raw, row)? {
        Value::Text(text) => json(&text),
        Value::Null => cell(batch, name, row),
        _ => Err(internal("result rescue column is not JSON text")),
    }
}
fn number(value: Value) -> Result<Option<i64>, CoreError> {
    match value {
        Value::Null => Ok(None),
        Value::Number(n) => n
            .parse()
            .map(Some)
            .map_err(|_| internal("result row index is not int64")),
        _ => Err(internal("result row index is not numeric")),
    }
}
fn floats(value: Value) -> Result<Option<Vec<f64>>, CoreError> {
    match value {
        Value::Null => Ok(None),
        Value::Array(values) => values
            .into_iter()
            .map(|v| match v {
                Value::Number(n) => n
                    .parse()
                    .map_err(|_| internal("result score is not numeric")),
                _ => Err(internal("result score is not numeric")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => Err(internal("result scores are not an array")),
    }
}
fn mapping(value: Value) -> Result<Option<Value>, CoreError> {
    match value {
        Value::Null => Ok(None),
        Value::Text(text) => json(&text).map(Some),
        _ => Err(internal("result mapping is not JSON text")),
    }
}
fn answer(value: Value) -> Answer {
    match value {
        Value::Null => Answer::None,
        Value::Bool(v) => Answer::Bool(v),
        Value::Text(v) => Answer::Text(v),
        Value::Number(ref n) => {
            if let Ok(v) = n.parse::<i64>() {
                Answer::Int(v)
            } else if !n.contains(['.', 'e', 'E'])
                && n.bytes().all(|b| b.is_ascii_digit() || b == b'-')
            {
                Answer::Written(value)
            } else if let Ok(v) = n.parse::<f64>() {
                if v.is_finite() {
                    Answer::Float(v)
                } else {
                    Answer::NonFinite(v)
                }
            } else {
                Answer::Written(value)
            }
        }
        _ => Answer::Written(value),
    }
}
fn prediction(value: Value, kind: AnswerType) -> Answer {
    let decoded = answer(value.clone());
    match (&decoded, kind) {
        (Answer::None | Answer::NonFinite(_), _) => decoded,
        (Answer::Int(_) | Answer::Float(_), AnswerType::Float)
        | (Answer::Float(_), AnswerType::FloatClass)
        | (Answer::Int(_), AnswerType::Int)
        | (Answer::Bool(_), AnswerType::Bool)
        | (Answer::Text(_), AnswerType::Text) => decoded,
        _ => Answer::Written(value),
    }
}

fn entity(value: Value) -> Entity {
    match value {
        Value::Null => Entity::None,
        Value::Text(v) => Entity::Text(v),
        Value::Number(ref n) => n
            .parse()
            .map(Entity::Integer)
            .unwrap_or(Entity::Written(value)),
        _ => Entity::Written(value),
    }
}
type CandidatePositions = Vec<Option<Vec<i64>>>;

fn decode(
    description: &ResultPart,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(Part, CandidatePositions), CoreError> {
    let answers = match schema.metadata().get("sdm.answer_type").map(String::as_str) {
        Some("Float") => AnswerType::Float,
        Some("FloatClass") => AnswerType::FloatClass,
        Some("Bool") => AnswerType::Bool,
        Some("Int") => AnswerType::Int,
        Some("Text") => AnswerType::Text,
        None if schema.field_with_name("candidates").is_ok()
            && schema.field_with_name("prediction").is_err() =>
        {
            AnswerType::Float
        }
        _ => return Err(internal("result lacks native answer-type metadata")),
    };
    let mut rows = Vec::new();
    for batch in batches {
        for i in 0..batch.num_rows() {
            let probabilities = match cell(batch, "probabilities", i)? {
                Value::Null => None,
                Value::Object(values) => Some(
                    values
                        .into_iter()
                        .map(|(key, value)| match value {
                            Value::Number(n) => n
                                .parse()
                                .map(|v| (key, v))
                                .map_err(|_| internal("invalid probability")),
                            _ => Err(internal("invalid probability")),
                        })
                        .collect::<Result<_, _>>()?,
                ),
                _ => return Err(internal("invalid probability mapping")),
            };
            let rankings = match cell(batch, "rankings", i)? {
                Value::Null => None,
                Value::Array(values) => Some(
                    values
                        .into_iter()
                        .map(|v| match v {
                            Value::Text(s) => json(&s),
                            _ => Err(internal("invalid ranking value")),
                        })
                        .collect::<Result<_, _>>()?,
                ),
                _ => return Err(internal("invalid rankings")),
            };
            let candidates = match cell(batch, "candidates", i)? {
                Value::Null => None,
                Value::Array(values) => Some(
                    values
                        .into_iter()
                        .map(|v| number(v)?.ok_or_else(|| internal("missing candidate position")))
                        .collect::<Result<_, _>>()?,
                ),
                _ => return Err(internal("invalid candidate positions")),
            };
            rows.push((
                Row {
                    row_index: number(cell(batch, "row_index", i)?)?
                        .ok_or_else(|| internal("missing result row index"))?,
                    answer_index: number(cell(batch, "answer_index", i)?)?
                        .ok_or_else(|| internal("missing answer index"))?,
                    forecast_step: number(cell(batch, "forecast_step", i)?)?,
                    id: entity(restored(batch, "id", "id_as_written", i)?),
                    prediction: prediction(
                        restored(batch, "prediction", "prediction_as_answered", i)?,
                        answers,
                    ),
                    probabilities,
                    rankings,
                    scores: floats(cell(batch, "scores", i)?)?,
                    embeddings: floats(cell(batch, "embeddings", i)?)?,
                    metadata: mapping(cell(batch, "metadata", i)?)?,
                    explanation: mapping(cell(batch, "explanation", i)?)?,
                    gradient: mapping(cell(batch, "sdm.gradient", i)?)?,
                    quantiles: mapping(cell(batch, "quantiles", i)?)?,
                },
                candidates,
            ));
        }
    }
    rows.sort_by_key(|(row, _)| row.answer_index);
    let (rows, candidates) = rows.into_iter().unzip();
    Ok((
        Part {
            model: description.model.clone(),
            request_id: description.request_id.clone(),
            metadata: json(&description.metadata)?,
            rows,
            answers,
            text_ids: schema
                .metadata()
                .get("sdm.text_ids")
                .is_some_and(|v| v == "true"),
            explanation: schema
                .metadata()
                .get("explanation")
                .map(|v| json(v))
                .transpose()?,
        },
        candidates,
    ))
}
fn written(answer: Answer) -> Answer {
    Answer::Written(match answer {
        Answer::None => return Answer::None,
        Answer::Bool(v) => Value::Bool(v),
        Answer::Int(v) => Value::Number(v.to_string()),
        Answer::Text(v) => Value::Text(v),
        Answer::NonFinite(v) => return Answer::NonFinite(v),
        Answer::Float(v) => Value::float(v).unwrap_or_else(|| {
            Value::Number(
                if v.is_nan() {
                    "NaN"
                } else if v.is_sign_negative() {
                    "-Infinity"
                } else {
                    "Infinity"
                }
                .into(),
            )
        }),
        Answer::Written(v) => v,
    })
}
pub(crate) fn items(
    asker: &Asker,
    request: sdm_contract::result::ReadResultItems,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: ResultPart = take_json(&mut attached, request.input)?;
    let Some(Attachment::Arrow(data)) = attached.remove(&request.table) else {
        return Err(internal("item projection lacks normalized Arrow result"));
    };
    let empty_ranking = data.schema.field_with_name("candidates").is_ok();
    let (mut part, positions) = decode(&input, &data.schema, &data.batches)?;
    let rows = std::mem::take(&mut part.rows);
    let mut groups: Vec<(bool, Vec<Row>, CandidatePositions)> = Vec::new();
    for (row, positions) in rows.into_iter().zip(positions) {
        let ranked = positions.is_some();
        if groups.last().is_none_or(|group| group.0 != ranked) {
            groups.push((ranked, Vec::new(), Vec::new()));
        }
        let group = groups.last_mut().unwrap();
        group.1.push(row);
        group.2.push(positions);
    }
    if groups.is_empty() {
        groups.push((empty_ranking, Vec::new(), Vec::new()));
    }
    let mut segments = Vec::new();
    for (ranking, rows, positions) in groups {
        part.rows = rows;
        let projected = sdm_request::items::project(&part, &positions, ranking, &input.outputs)?;
        let table = asker.attach(Attachment::Arrow(crate::run::ArrowData::new(
            projected.table.schema(),
            vec![projected.table],
        )))?;
        segments.push(sdm_contract::result::ResultItemSegment {
            table,
            fields: projected.fields,
        });
    }
    Ok(OperationResult::ResultItems(
        sdm_contract::result::ResultItems { segments },
    ))
}

pub(crate) fn merge(
    asker: &Asker,
    request: MergeResults,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: MergeInput = take_json(&mut attached, request.input)?;
    if input.parts.len() != request.tables.len() {
        return Err(internal("merge result descriptions and tables differ"));
    }
    let outputs = input
        .parts
        .first()
        .map(|v| v.outputs.clone())
        .unwrap_or_default();
    let mut parts = Vec::new();
    let mut positions = Vec::new();
    let mut descriptors = Vec::new();
    for (description, id) in input.parts.iter().zip(request.tables) {
        let Some(Attachment::Arrow(data)) = attached.remove(&id) else {
            return Err(internal("merge lacks Arrow result"));
        };
        let (part, held) = decode(description, &data.schema, &data.batches)?;
        descriptors.push(sdm_contract::result::MergeDescriptor {
            model: description.model.clone(),
            request_id: description.request_id.clone(),
            outputs: description.outputs.clone(),
            has_explanation: part
                .explanation
                .as_ref()
                .is_some_and(sdm_request::result::truth),
            names_rows: true,
            names_step: part.rows.iter().any(|r| r.forecast_step.is_some()),
            gradient_rows: sdm_request::gradients::records(&part.rows),
            candidates: description
                .unscored_candidates
                .iter()
                .map(|s| json(s).map(sdm_request::result::candidate))
                .collect::<Result<_, _>>()?,
        });
        parts.push(part);
        positions.extend(held);
    }
    let plan = sdm_request::result::plan(&descriptors)?;

    if parts
        .first()
        .is_some_and(|first| parts.iter().any(|part| part.answers != first.answers))
    {
        for part in &mut parts {
            for row in &mut part.rows {
                row.prediction = written(std::mem::replace(&mut row.prediction, Answer::None));
            }
        }
    }
    let merged = predict::merge_planned(parts, &plan)?;
    let mut laid = predict::table(&merged)?;
    if let Some(collection)=&plan.gradients {
        let mut metadata=laid.schema().metadata().clone();
        metadata.insert("sdm.gradients".into(),serde_json::to_string(collection).map_err(|_|internal("gradient metadata invalid"))?);
        laid=RecordBatch::try_new(Arc::new(laid.schema().as_ref().clone().with_metadata(metadata)),laid.columns().to_vec()).map_err(|_|internal("gradient result layout invalid"))?;
    }
    if positions.iter().any(Option::is_some) {
        let mut order: Vec<usize> = (0..merged.rows.len()).collect();
        order.sort_by_key(|&i| merged.rows[i].row_index);
        let mut builder =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::Int64Builder::new());
        for i in order {
            if let Some(values) = &positions[i] {
                builder.values().append_slice(values);
                builder.append(true);
            } else {
                builder.append(false);
            }
        }
        let values: ArrayRef = Arc::new(builder.finish());
        let mut fields = laid
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>();
        fields.push(Field::new("candidates", values.data_type().clone(), true));
        let schema = Arc::new(Schema::new(fields).with_metadata(laid.schema().metadata().clone()));
        let mut columns = laid.columns().to_vec();
        columns.push(values);
        laid = RecordBatch::try_new(schema, columns)
            .map_err(|_| internal("merged candidate layout failed"))?;
    }
    let mut result = crate::predict::finish_table(asker, merged, outputs, laid)?;
    if let OperationResult::Predict(value) = &mut result {
        value.unscored_candidates = plan.candidate_positions;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn part(prediction: Answer) -> Part {
        Part {
            model: "kumo-tabular".into(),
            request_id: "one".into(),
            metadata: Value::Object(vec![]),
            answers: AnswerType::Text,
            text_ids: true,
            explanation: None,
            rows: vec![Row {
                row_index: 0,
                answer_index: 0,
                forecast_step: None,
                id: Entity::None,
                prediction,
                probabilities: None,
                scores: None,
                rankings: None,
                explanation: None,
                gradient: None,
                metadata: None,
                quantiles: None,
                embeddings: None,
            }],
        }
    }
    fn roundtrip(original: &Part) -> Part {
        let table = predict::table(original).unwrap();
        decode(
            &ResultPart {
                model: original.model.clone(),
                request_id: original.request_id.clone(),
                metadata: "{}".into(),
                outputs: vec!["prediction".into()],
                unscored_candidates: vec![],
            },
            &table.schema(),
            std::slice::from_ref(&table),
        )
        .unwrap()
        .0
    }
    #[test]
    fn undeclared_prediction_values_survive_same_type_merges() {
        for value in [
            Value::Bool(true),
            Value::Number("17".into()),
            Value::Number("18446744073709551617".into()),
        ] {
            let original = part(Answer::Written(value));
            let read = roundtrip(&original);
            assert_eq!(read.rows[0].prediction, original.rows[0].prediction);
            let merged = predict::merge(vec![read.clone(), read]).unwrap();
            let again = roundtrip(&merged);
            assert_eq!(again.rows.len(), 2);
            assert_eq!(again.rows[0].prediction, original.rows[0].prediction);
        }
    }
    #[test]
    fn mixed_numeric_result_layout_preserves_original_wide_integer() {
        let mut integer = part(Answer::Int(9_007_199_254_740_993));
        integer.answers = AnswerType::Int;
        let mut fraction = part(Answer::Float(3.5));
        fraction.answers = AnswerType::Float;
        let mut parts = vec![roundtrip(&integer), roundtrip(&fraction)];
        for part in &mut parts {
            for row in &mut part.rows {
                row.prediction = written(std::mem::replace(&mut row.prediction, Answer::None));
            }
        }
        let merged = predict::merge(parts).unwrap();
        let decoded = roundtrip(&merged);
        assert_eq!(
            decoded.rows[0].prediction,
            Answer::Int(9_007_199_254_740_993)
        );
        assert_eq!(
            decoded.rows[1].prediction,
            Answer::Written(Value::Number("3.5".into()))
        );
    }
}
