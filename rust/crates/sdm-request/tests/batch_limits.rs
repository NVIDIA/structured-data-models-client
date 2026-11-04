// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_request::options::{BatchTask, rows_one_call_scores, rows_one_task_call_scores};

#[test]
fn temporal_links_cannot_raise_the_model_limit() {
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
            rows_one_task_call_scores(asked, BatchTask::TemporalLink),
            expected
        );
    }
}

#[test]
fn ordinary_and_decomposed_tasks_keep_their_existing_limit() {
    for (asked, expected) in [
        (None, 10_000),
        (Some(100), 100),
        (Some(201), 201),
        (Some(1000), 1000),
        (Some(u64::MAX), 10_000),
    ] {
        assert_eq!(
            rows_one_task_call_scores(asked, BatchTask::Ordinary),
            expected
        );
        assert_eq!(rows_one_call_scores(asked), expected);
    }
}
