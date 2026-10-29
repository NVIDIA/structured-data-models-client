// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn positive(column: &str) -> Shape {
    Shape::Condition {
        target: Box::new(Shape::Column(column.into())),
        test: Test::Compare(labels::RelOp::Gt, Some(labels::Constant::Int(0))),
    }
}

#[test]
fn actual_entity_presence_enters_a_real_predicate_before_optional_where() {
    use crate::materialize::native_conditions::entity_condition;
    for where_clause in [None, Some(positive("age"))] {
        let shape = entity_condition("user_id".into(), where_clause.clone());
        let first = ConditionInputs::default()
            .complete_target(
                &shape,
                labels::Target::Column(labels::Column::Bool(vec![
                    Some(true),
                    Some(false),
                    Some(true),
                ])),
            )
            .unwrap();
        if where_clause.is_some() {
            assert!(first.truth().is_none());
            let complete = first
                .complete(
                    &shape,
                    labels::Column::Int(vec![Some(1), Some(2), Some(-1)]),
                )
                .unwrap();
            assert_eq!(complete.truth().unwrap().values, [true, false, false]);
        } else {
            assert_eq!(first.truth().unwrap().values, [true, false, true]);
        }
    }
    assert!(
        labels::truth(&labels::Target::Column(labels::Column::Bool(vec![Some(
            true
        )])))
        .is_err()
    );
}

#[test]
fn repeated_native_occurrences_remain_ordered_and_actual_values_choose_the_rows() {
    let filter = Shape::Both {
        op: labels::BoolOp::And,
        left: Box::new(positive("amount")),
        right: Box::new(positive("amount")),
    };
    assert_eq!(operand_columns(&filter).unwrap(), ["amount", "amount"]);
    let left = ConditionInputs::default()
        .complete(&filter, labels::Column::Float(vec![Some(-1.0), Some(2.0)]))
        .unwrap();
    assert!(left.truth().is_none());
    assert_eq!(left.completed(), 1);
    let both = left
        .complete(&filter, labels::Column::Float(vec![Some(9.0), Some(3.0)]))
        .unwrap();
    assert_eq!(both.truth().unwrap().values, [false, true]);
    assert_eq!(both.completed(), 2);
    assert!(
        both.complete(&filter, labels::Column::Float(vec![Some(1.0), Some(1.0)]))
            .is_err()
    );
    assert!(left.truth().is_none());
}

#[test]
fn a_false_left_side_still_requires_the_right_native_value_and_its_failure() {
    let filter = Shape::Both {
        op: labels::BoolOp::And,
        left: Box::new(positive("left")),
        right: Box::new(Shape::Condition {
            target: Box::new(Shape::Column("right".into())),
            test: Test::Compare(labels::RelOp::Gt, Some(labels::Constant::Int(i128::MAX))),
        }),
    };
    let left = ConditionInputs::default()
        .complete(&filter, labels::Column::Int(vec![Some(-1)]))
        .unwrap();
    assert!(left.truth().is_none());
    assert!(
        left.complete(&filter, labels::Column::Int(vec![Some(1)]))
            .is_err()
    );
}

#[test]
fn a_left_comparison_refusal_prevents_later_native_conversion() {
    let filter = Shape::Both {
        op: labels::BoolOp::Or,
        left: Box::new(Shape::Condition {
            target: Box::new(Shape::Column("left".into())),
            test: Test::Compare(labels::RelOp::Gt, Some(labels::Constant::Int(i128::MAX))),
        }),
        right: Box::new(positive("right")),
    };
    let mut events = Vec::new();
    let result = (|| {
        let mut completed = ConditionInputs::default();
        for column in operand_columns(&filter)? {
            events.push(column);
            completed = completed.complete(&filter, labels::Column::Int(vec![Some(1)]))?;
        }
        Ok::<_, CoreError>(completed)
    })();
    assert!(result.is_err());
    assert_eq!(events, ["left"]);
}

#[test]
fn nested_not_and_null_keep_the_existing_truth_and_presence_masks() {
    let filter = Shape::Not(Box::new(Shape::Condition {
        target: Box::new(Shape::Column("amount".into())),
        test: Test::Compare(labels::RelOp::Eq, None),
    }));
    let values = labels::Column::Int(vec![None, Some(1)]);
    let expected = labels::truth(&labels::Target::Not(Box::new(tested(
        labels::Target::Column(values.clone()),
        &Test::Compare(labels::RelOp::Eq, None),
    ))))
    .unwrap();
    let completed = ConditionInputs::default()
        .complete(&filter, values)
        .unwrap();
    assert_eq!(completed.truth().unwrap().as_ref(), &expected);
    assert_eq!(expected.values, [false, true]);
    assert_eq!(expected.mask, [false, true]);
}

#[test]
fn aggregated_condition_uses_completed_values_at_candidate_extent_not_source_extent() {
    let target = Shape::Aggregated(Windowed {
        aggregation: labels::Aggregation::Sum,
        table: "orders".into(),
        column: Some("amount".into()),
        filter: None,
        start: sdm_engine::time::offset::Offset::of(0, sdm_contract::pql::TimeUnit::Days),
        end: sdm_engine::time::offset::Offset::of(30, sdm_contract::pql::TimeUnit::Days),
    });
    let condition = Shape::Condition {
        target: Box::new(target),
        test: Test::Compare(labels::RelOp::Gt, Some(labels::Constant::Int(0))),
    };
    let completed = ConditionInputs::default()
        .complete_target(
            &condition,
            labels::Target::Aggregated(labels::Aggregated {
                values: vec![-2.0, 5.0],
                present: vec![true, true],
            }),
        )
        .unwrap();
    assert_eq!(completed.truth().unwrap().values, [false, true]);
    assert_eq!(completed.completed(), 1);
    assert!(operand_columns(&condition).is_err());
}

#[test]
fn completed_aggregated_left_is_evaluated_before_a_later_operand_can_complete() {
    let filter = Shape::Both {
        op: labels::BoolOp::And,
        left: Box::new(Shape::Condition {
            target: Box::new(Shape::Column("left".into())),
            test: Test::Compare(labels::RelOp::Gt, Some(labels::Constant::Int(i128::MAX))),
        }),
        right: Box::new(positive("right")),
    };
    assert!(
        ConditionInputs::default()
            .complete_target(
                &filter,
                labels::Target::Column(labels::Column::Int(vec![Some(1)])),
            )
            .is_err()
    );
}

#[test]
fn already_selected_native_numeric_values_use_existing_aggregate_kernels() {
    let batch = [0, 0, 1, 1];
    let values = labels::Values::Float(vec![Some(4.0), Some(-1.0), Some(9.0), None]);
    for (aggregation, expected) in [
        (labels::Aggregation::Sum, vec![3.0, 9.0, 0.0]),
        (labels::Aggregation::Count, vec![2.0, 1.0, 0.0]),
        (labels::Aggregation::Avg, vec![1.5, 9.0, 0.0]),
        (labels::Aggregation::Min, vec![-1.0, 9.0, 0.0]),
        (labels::Aggregation::Max, vec![4.0, 9.0, 0.0]),
    ] {
        let actual = labels::aggregate_selected(aggregation, 3, &batch, &values).unwrap();
        assert_eq!(actual.values, expected);
        assert_eq!(
            actual.present,
            if matches!(
                aggregation,
                labels::Aggregation::Sum | labels::Aggregation::Count
            ) {
                vec![true; 3]
            } else {
                vec![true, true, false]
            }
        );
    }
    assert!(labels::aggregate_selected(labels::Aggregation::Sum, 1, &batch, &values).is_err());
    assert!(labels::aggregate_selected(labels::Aggregation::Sum, 3, &[0], &values).is_err());
}
