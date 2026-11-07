// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::limits::Limits;
use sdm_request::options::{BatchTask, rows_one_call_scores, rows_one_task_call_scores};

#[test]
fn temporal_links_cannot_raise_the_model_limit() {
    let limits = Limits::default();
    for (asked, expected) in [
        (None, 200),
        (Some(100), 100),
        (Some(199), 199),
        (Some(200), 200),
        (Some(201), 200),
        (Some(1000), 200),
        (Some(u64::MAX), 200),
    ] {
        assert_eq!(
            rows_one_task_call_scores(asked, BatchTask::TemporalLink, &limits),
            expected
        );
    }
}

#[test]
fn ordinary_and_decomposed_tasks_keep_their_existing_limit() {
    let limits = Limits::default();
    for (asked, expected) in [
        (None, 10_000),
        (Some(100), 100),
        (Some(201), 201),
        (Some(1000), 1000),
        (Some(u64::MAX), 10_000),
    ] {
        assert_eq!(
            rows_one_task_call_scores(asked, BatchTask::Ordinary, &limits),
            expected
        );
        assert_eq!(rows_one_call_scores(asked, &limits), expected);
    }
}

#[test]
fn a_raised_limit_raises_each_task_s_call_and_rows_per_call_still_lowers_it() {
    let limits = Limits {
        max_relational_predict_rows: 50_000,
        max_temporal_link_predict_rows: 1_000,
        ..Limits::default()
    };
    assert_eq!(
        rows_one_task_call_scores(None, BatchTask::Ordinary, &limits),
        50_000
    );
    assert_eq!(
        rows_one_task_call_scores(Some(20_000), BatchTask::Ordinary, &limits),
        20_000
    );
    assert_eq!(
        rows_one_task_call_scores(None, BatchTask::TemporalLink, &limits),
        1_000
    );
    assert_eq!(
        rows_one_task_call_scores(Some(u64::MAX), BatchTask::TemporalLink, &limits),
        1_000
    );
}
