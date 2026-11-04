// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic prediction items, without a language-specific ranking policy.
use crate::predict::{self, Answer, Entity, Part};
use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StructBuilder};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{
    CoreError, Scalar, Sentence,
    direct::PrimitiveOption,
    result::{ItemEncoding, ItemField},
};
use sdm_wire::json::Value;
use std::sync::Arc;

fn invalid(message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}

/// The existing lossless value grammar, not JSON's machine-number interpretation.
pub fn value(v: &Value) -> Result<PrimitiveOption, CoreError> {
    Ok(match v {
        Value::Null => PrimitiveOption::Value(Scalar::Null),
        Value::Bool(v) => PrimitiveOption::Value(Scalar::Bool(*v)),
        Value::Text(v) => PrimitiveOption::Value(Scalar::Utf8(v.clone())),
        Value::Number(v) if sdm_wire::json::is_integer_token(v) => {
            PrimitiveOption::Integer(v.clone())
        }
        Value::Number(v) => PrimitiveOption::Value(Scalar::Float64(
            v.parse()
                .map_err(|_| invalid("invalid numeric result value"))?,
        )),
        Value::Array(v) => PrimitiveOption::Array(v.iter().map(value).collect::<Result<_, _>>()?),
        Value::Object(v) => PrimitiveOption::Object(
            v.iter()
                .map(|(k, v)| Ok((k.clone(), value(v)?)))
                .collect::<Result<_, CoreError>>()?,
        ),
    })
}
fn answer(v: &Answer) -> Result<PrimitiveOption, CoreError> {
    Ok(match v {
        Answer::None => PrimitiveOption::Value(Scalar::Null),
        Answer::Float(v) | Answer::NonFinite(v) => PrimitiveOption::Value(Scalar::Float64(*v)),
        Answer::Int(v) => PrimitiveOption::Integer(v.to_string()),
        Answer::Bool(v) => PrimitiveOption::Value(Scalar::Bool(*v)),
        Answer::Text(v) => PrimitiveOption::Value(Scalar::Utf8(v.clone())),
        Answer::Written(v) => value(v)?,
    })
}
fn entity(v: &Entity) -> Result<PrimitiveOption, CoreError> {
    Ok(match v {
        Entity::None => PrimitiveOption::Value(Scalar::Null),
        Entity::Integer(v) => PrimitiveOption::Integer(v.to_string()),
        Entity::Float(v) => PrimitiveOption::Value(Scalar::Float64(*v)),
        Entity::Text(v) => PrimitiveOption::Value(Scalar::Utf8(v.clone())),
        Entity::Written(v) => value(v)?,
    })
}
fn encoded(
    values: impl Iterator<Item = Result<Option<PrimitiveOption>, CoreError>>,
) -> Result<ArrayRef, CoreError> {
    let values = values
        .map(|v| {
            v?.map(|v| serde_json::to_string(&v).map_err(|_| invalid("cannot encode result value")))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(StringArray::from(values)))
}

pub struct Projected {
    pub table: RecordBatch,
    pub fields: Vec<ItemField>,
}

/// `positions` and rows share answer order; positions always refer to original host values.
pub fn project(
    part: &Part,
    positions: &[Option<Vec<i64>>],
    ranking: bool,
    outputs: &[String],
) -> Result<Projected, CoreError> {
    if positions.len() != part.rows.len() {
        return Err(invalid("candidate owners differ from result rows"));
    }
    let base = predict::table(part)?;
    let answers = base
        .column_by_name("answer_index")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| invalid("result lacks answer order"))?;
    let mut order = (0..base.num_rows()).collect::<Vec<_>>();
    order.sort_by_key(|i| answers.value(*i));
    let indices = UInt64Array::from_iter_values(order.iter().map(|i| *i as u64));
    let base = arrow_select::take::take_record_batch(&base, &indices)
        .map_err(|_| invalid("cannot order semantic result items"))?;
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    let mut descriptions = Vec::new();
    let mut add = |name: &str, encoding: ItemEncoding, array: ArrayRef| {
        fields.push(Field::new(name, array.data_type().clone(), true));
        columns.push(array);
        descriptions.push(ItemField {
            name: name.to_owned(),
            encoding,
        });
    };
    for name in [
        "row_index",
        "id",
        "forecast_step",
        "prediction",
        "probabilities",
        "scores",
        "embeddings",
    ] {
        if ranking && !matches!(name, "row_index" | "id") {
            continue;
        }
        let Some(column) = base.column_by_name(name) else {
            continue;
        };
        if name == "id" && base.column_by_name("id_as_written").is_some() {
            add(
                name,
                ItemEncoding::Value,
                encoded(part.rows.iter().map(|r| entity(&r.id).map(Some)))?,
            );
        } else if name == "prediction" && base.column_by_name("prediction_as_answered").is_some() {
            add(
                name,
                ItemEncoding::Value,
                encoded(part.rows.iter().map(|r| answer(&r.prediction).map(Some)))?,
            );
        } else {
            add(name, ItemEncoding::Native, column.clone());
        }
    }
    if ranking {
        if outputs.iter().any(|s| s == "prediction") {
            let mut values = ListBuilder::new(Int64Builder::new());
            for row in positions {
                let row = row
                    .as_ref()
                    .ok_or_else(|| invalid("ranking lacks candidate positions"))?;
                for position in row {
                    if *position < 0 {
                        return Err(invalid("negative original candidate position"));
                    }
                    values.values().append_value(*position);
                }
                values.append(true);
            }
            add(
                "prediction",
                ItemEncoding::CandidateList,
                Arc::new(values.finish()),
            );
        }
        if outputs.iter().any(|s| s == "rankings") {
            let mut values = ListBuilder::new(StructBuilder::new(
                vec![
                    Field::new("position", DataType::Int64, false),
                    Field::new("score", DataType::Float64, false),
                ],
                vec![
                    Box::new(Int64Builder::new()),
                    Box::new(Float64Builder::new()),
                ],
            ));
            for (row, positions) in part.rows.iter().zip(positions) {
                let positions = positions
                    .as_ref()
                    .ok_or_else(|| invalid("ranking lacks candidate positions"))?;
                let scores = row
                    .scores
                    .as_ref()
                    .ok_or_else(|| invalid("ranking lacks scores"))?;
                if positions.len() != scores.len() {
                    return Err(invalid("ranking scores differ from candidate positions"));
                }
                for (position, score) in positions.iter().zip(scores) {
                    if *position < 0 {
                        return Err(invalid("negative original candidate position"));
                    }
                    values
                        .values()
                        .field_builder::<Int64Builder>(0)
                        .unwrap()
                        .append_value(*position);
                    values
                        .values()
                        .field_builder::<Float64Builder>(1)
                        .unwrap()
                        .append_value(*score);
                    values.values().append(true);
                }
                values.append(true);
            }
            add(
                "rankings",
                ItemEncoding::CandidateRankings,
                Arc::new(values.finish()),
            );
        }
    } else if base.column_by_name("rankings").is_some() {
        add(
            "rankings",
            ItemEncoding::Mappings,
            encoded(part.rows.iter().map(|r| {
                r.rankings
                    .as_ref()
                    .map(|v| value(&Value::Array(v.clone())))
                    .transpose()
            }))?,
        );
    }
    for name in ["metadata", "explanation", "quantiles"] {
        if base.column_by_name(name).is_none() {
            continue;
        }
        add(
            name,
            ItemEncoding::Mapping,
            encoded(part.rows.iter().map(|r| {
                match name {
                    "metadata" => r.metadata.as_ref(),
                    "explanation" => r.explanation.as_ref(),
                    _ => r.quantiles.as_ref(),
                }
                .map(value)
                .transpose()
            }))?,
        );
    }
    let table = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|_| invalid("cannot construct semantic items"))?;
    Ok(Projected {
        table,
        fields: descriptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::{AnswerType, Row};
    use arrow_array::{Array, ListArray};
    fn row(index: i64) -> Row {
        Row {
            row_index: index,
            answer_index: 1 - index,
            forecast_step: None,
            id: Entity::Integer(index),
            prediction: Answer::Int(index),
            probabilities: None,
            scores: None,
            rankings: None,
            explanation: None,
            gradient: None,
            gradient_position: None,
            metadata: None,
            quantiles: None,
            embeddings: None,
        }
    }
    fn part(rows: Vec<Row>) -> Part {
        Part {
            model: "kumo-relational".into(),
            request_id: "r".into(),
            metadata: Value::Object(vec![]),
            rows,
            answers: AnswerType::Float,
            text_ids: false,
            explanation: None,
        }
    }
    #[test]
    fn lossless_values_use_existing_scalar_grammar() {
        let large = Value::Number("1208925819614629174706177".into());
        assert_eq!(
            value(&large).unwrap(),
            PrimitiveOption::Integer("1208925819614629174706177".into())
        );
        assert_eq!(
            answer(&Answer::NonFinite(f64::NAN)).unwrap(),
            PrimitiveOption::Value(Scalar::Float64(f64::NAN))
        );
        let map = Value::Array(vec![Value::Array(vec![Value::int(7), large])]);
        let encoded = serde_json::to_string(&value(&map).unwrap()).unwrap();
        assert!(encoded.contains("1208925819614629174706177"));
    }
    #[test]
    fn answer_order_and_rescue_values_are_kept() {
        let mut first = row(1);
        first.prediction = Answer::Int(9_007_199_254_740_993);
        first.metadata = Some(Value::Object(vec![("x".into(), Value::int(4))]));
        let mut second = row(0);
        second.prediction = Answer::Float(3.5);
        let projected = project(
            &part(vec![first, second]),
            &[None, None],
            false,
            &["prediction".into()],
        )
        .unwrap();
        assert_eq!(
            projected
                .table
                .column_by_name("row_index")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1, 0]
        );
        assert!(
            projected
                .fields
                .iter()
                .any(|f| f.name == "prediction" && f.encoding == ItemEncoding::Value)
        );
        let data = projected
            .table
            .column_by_name("prediction")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<PrimitiveOption>(data.value(0)).unwrap(),
            PrimitiveOption::Integer("9007199254740993".into())
        );
        let metadata = projected.table.column_by_name("metadata").unwrap();
        assert!(metadata.is_null(1));
    }
    #[test]
    fn ranking_uses_only_original_positions_and_requested_outputs() {
        let mut r = row(1);
        r.prediction = Answer::None;
        r.rankings = Some(vec![Value::text("deliberately not the original")]);
        r.scores = Some(vec![0.75]);
        let projected = project(
            &part(vec![r]),
            &[Some(vec![7])],
            true,
            &["prediction".into()],
        )
        .unwrap();
        assert!(projected.table.column_by_name("rankings").is_none());
        assert!(projected.table.column_by_name("scores").is_none());
        let p = projected
            .table
            .column_by_name("prediction")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        assert_eq!(p.as_any().downcast_ref::<Int64Array>().unwrap().value(0), 7);
        assert_eq!(
            projected.fields.last().unwrap().encoding,
            ItemEncoding::CandidateList
        );
    }
    #[test]
    fn empty_items_keep_an_explicit_empty_arrow_result() {
        let projected = project(&part(vec![]), &[], false, &[]).unwrap();
        assert_eq!(projected.table.num_rows(), 0);
        assert!(projected.table.column_by_name("row_index").is_some());
    }
}
