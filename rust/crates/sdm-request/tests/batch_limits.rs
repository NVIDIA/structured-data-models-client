// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_request::options::{BatchTask, rows_one_call_scores, rows_one_task_call_scores};

#[test]
fn each_task_batches_by_its_default_and_the_size_asked_is_taken_as_is() {
    for (task, default) in [
        (BatchTask::TemporalLink, 200),
        (BatchTask::Ordinary, 10_000),
    ] {
        assert_eq!(rows_one_task_call_scores(None, task), default);
        for asked in [1, 100, 20_000, u64::MAX] {
            assert_eq!(rows_one_task_call_scores(Some(asked), task), asked);
        }
    }
    assert_eq!(rows_one_call_scores(None), 10_000);
    assert_eq!(rows_one_call_scores(Some(50_000)), 50_000);
}
