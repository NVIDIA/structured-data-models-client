// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_engine::labels::{self, BoolOp, Truth};

#[test]
fn completed_truth_combination_retains_values_independently_of_missingness() {
    let left = Truth {
        values: vec![true, false],
        mask: vec![false, true],
    };
    let right = Truth {
        values: vec![false, true],
        mask: vec![true, false],
    };
    assert_eq!(
        labels::negate_truth(&left),
        Truth {
            values: vec![false, true],
            mask: vec![false, true]
        }
    );
    assert_eq!(
        labels::combine_truth(BoolOp::Or, &left, &right).unwrap(),
        Truth {
            values: vec![true, true],
            mask: vec![false, false]
        }
    );
    assert_eq!(
        labels::combine_truth(BoolOp::And, &left, &right).unwrap(),
        Truth {
            values: vec![false, false],
            mask: vec![false, false]
        }
    );
}

#[test]
fn completed_truth_rejects_short_or_unaligned_value_and_presence_inputs() {
    let full = Truth {
        values: vec![true, false],
        mask: vec![true, true],
    };
    for short in [
        Truth {
            values: vec![true],
            mask: vec![true],
        },
        Truth {
            values: vec![true, false],
            mask: vec![true],
        },
    ] {
        assert!(labels::combine_truth(BoolOp::And, &full, &short).is_err());
        assert!(labels::combine_truth(BoolOp::Or, &short, &full).is_err());
    }
}
