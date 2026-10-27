// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Completed native time facts retain the shared policy and strict bindings.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, TimestampMicrosecondArray};
use sdm_contract::data::Relation;
use sdm_contract::frames::{
    FrameDecimalExponent, FrameScalar, FrameTimeFacts, FrameTimeFirst, PassedOver,
};
use sdm_contract::Scalar;
use sdm_schema::frames::{self, Columns};

fn rows() -> RecordBatch {
    RecordBatch::try_from_iter([(
        "observed_at",
        Arc::new(TimestampMicrosecondArray::from_iter_values([1, 2, 3, 4])) as ArrayRef,
    )]).unwrap()
}

fn facts() -> FrameTimeFacts {
    let batch = rows();
    let (nomination, _) = frames::nominate(
        "rows", &Relation { source: vec!["rows".into()] },
        batch.schema().as_ref(), std::slice::from_ref(&batch),
    ).unwrap();
    let (adoption, _) = frames::adopt(
        nomination, batch.schema().as_ref(), std::slice::from_ref(&batch),
    ).unwrap();
    FrameTimeFacts {
        plan: frames::time_plan(adoption),
        firsts: vec![FrameTimeFirst {
            name: "observed_at".into(), rows: 4, held: 4, same: false,
            first: Some(FrameScalar::Decimal {
                negative: false, digits: "1".into(),
                exponent: FrameDecimalExponent::Finite { value: 2_147_483_648 },
            }),
        }],
    }
}

#[test]
fn native_first_tuple_does_not_narrow_to_arrow_i8_or_infer_a_new_meaning() {
    let batch = rows();
    let batches = [batch.clone()];
    let schema = batch.schema();
    let input = Columns::new(schema.as_ref(), &batches, &[]).unwrap();
    let result = frames::choose_time_from_facts(facts(), &input).unwrap();
    assert_eq!(result.table.time_column.as_deref(), Some("observed_at"));
}

#[test]
fn native_counts_share_empty_sparse_and_constant_precedence() {
    let batch = rows();
    let batches = [batch.clone()];
    let schema = batch.schema();
    let input = Columns::new(schema.as_ref(), &batches, &[]).unwrap();
    for (rows, held, same, expected) in [
        (0, 0, false, "constant"),
        (4, 0, false, "constant"),
        (4, 1, true, "sparse"),
        (4, 2, true, "constant"),
        (1, 1, true, "only"),
    ] {
        let mut facts = facts();
        let first = &mut facts.firsts[0];
        first.rows = rows;
        first.held = held;
        first.same = same;
        if held == 0 { first.first = None; }
        let result = frames::choose_time_from_facts(facts, &input).unwrap();
        if expected == "only" {
            assert_eq!(result.table.time_column.as_deref(), Some("observed_at"));
        } else {
            assert_eq!(result.time.refused[0].1.reason(), expected);
            if held == 0 {
                assert!(matches!(&result.time.refused[0].1,
                    PassedOver::FrameConstant { first: None, .. }));
            }
        }
    }
}

#[test]
fn native_facts_cannot_forge_order_counts_or_a_decimal_tuple() {
    let batch = rows();
    let batches = [batch.clone()];
    let schema = batch.schema();
    let input = Columns::new(schema.as_ref(), &batches, &[]).unwrap();
    for variant in 0..8 {
        let mut facts = facts();
        match variant {
            0 => facts.plan.columns.clear(),
            1 => facts.firsts.clear(),
            2 => facts.firsts[0].name = "other".into(),
            3 => facts.firsts[0].held = 5,
            4 => facts.firsts[0].first = None,
            5 => facts.firsts[0].first = Some(FrameScalar::Value { value: Scalar::Null }),
            6 | 7 => {
                let Some(FrameScalar::Decimal { digits, .. }) = &mut facts.firsts[0].first else {
                    panic!("fixture Decimal tuple")
                };
                *digits = if variant == 6 { "01".into() } else { "x".into() };
            }
            _ => unreachable!(),
        }
        assert!(frames::choose_time_from_facts(facts, &input).is_err());
    }
}
