// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The toy run, "count the rows of table T", driven by the Rust replay host.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use sdm_core::replay::{Exchange, ReplayError, replay};
use sdm_core::{Outcome, Param, QueryResult, codes, count_rows};

const USERS: &str = "SELECT COUNT(*) AS n FROM \"users\"";

fn count(n: i64) -> Outcome {
    let batch = RecordBatch::try_from_iter([("n", Arc::new(Int64Array::from(vec![n])) as _)])
        .expect("a batch");
    Outcome::Data(QueryResult::from_batch(batch))
}

#[test]
fn replay_counts_the_rows_with_one_query() {
    for n in [0, 3, 12_345] {
        let replayed = replay(&mut count_rows("users"), [Exchange::new(USERS, count(n))])
            .expect("the replay goes as recorded");
        assert_eq!(replayed.value, u64::try_from(n).expect("a count"));
        assert_eq!(replayed.queries.len(), 1);
        assert_eq!(replayed.queries[0].columns, ["n"]);
    }
}

#[test]
fn replay_reports_a_failed_query_with_its_message() {
    let failed = Outcome::Error {
        sqlstate: None,
        vendor_code: Some(1),
        message: "no such table: users".to_owned(),
    };
    let error = replay(&mut count_rows("users"), [Exchange::new(USERS, failed)]).unwrap_err();
    let ReplayError::Run(diagnostic) = error else {
        panic!("expected the run's diagnostic, got {error:?}");
    };
    assert_eq!(diagnostic.code_str(), codes::QUERY_FAILED);
    assert!(diagnostic.message.contains("no such table: users"));
}

#[test]
fn replay_refuses_a_query_that_was_not_recorded() {
    let error = replay(&mut count_rows("orders"), [Exchange::new(USERS, count(1))]).unwrap_err();
    assert!(
        matches!(&error, ReplayError::Unexpected { query, expected: Some((sql, _)) }
            if query.sql.ends_with("\"orders\"") && sql == USERS),
        "{error}"
    );
    let error = replay(
        &mut count_rows("users"),
        [Exchange {
            sql: USERS.to_owned(),
            params: vec![Param::Int(1)],
            outcome: count(1),
        }],
    )
    .unwrap_err();
    assert!(matches!(error, ReplayError::Unexpected { .. }), "{error}");
}

#[test]
fn replay_refuses_a_recording_the_run_does_not_use_up() {
    let error = replay(
        &mut count_rows("users"),
        [
            Exchange::new(USERS, count(1)),
            Exchange::new(USERS, count(2)),
        ],
    )
    .unwrap_err();
    assert_eq!(error, ReplayError::Unused(1));
    let error = replay(&mut count_rows("users"), []).unwrap_err();
    assert!(
        matches!(error, ReplayError::Unexpected { expected: None, .. }),
        "{error}"
    );
}
