// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual bound reads carry complete partitions without changing reply policy.

use super::*;
use sdm_contract::data::DescribePart;
use sdm_contract::ops::QueryValidation;
use std::future::poll_fn;

fn request() -> DataRequest {
    DataRequest::Describe {
        tables: vec![],
        part: DescribePart::Tables,
    }
}

fn plans(count: usize) -> Vec<FetchPlan> {
    (0..count)
        .map(|_| FetchPlan::Host {
            expect: ExpectedSchema::Discover {},
        })
        .collect()
}

fn done() -> OperationResult {
    OperationResult::ValidateQuery(QueryValidation {
        accepted: true,
        diagnostics: vec![],
        query: None,
    })
}

#[test]
fn independent_same_request_reads_have_separate_complete_partitions() {
    let mut run = Run::new(|asker| async move {
        asker.describe_fetch_partitions();
        let mut left = Box::pin(asker.ask_bound(request(), plans(2)));
        let mut right = Box::pin(asker.ask_bound(request(), plans(3)));
        poll_fn(move |cx| {
            let _ = left.as_mut().poll(cx);
            let _ = right.as_mut().poll(cx);
            Poll::<()>::Pending
        })
        .await;
        Ok(done())
    });
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("expected actual requests")
    };
    assert_eq!(requests.len(), 5);
    let first = requests[0].id;
    let second = requests[2].id;
    assert_ne!(first, second);
    for (at, request) in requests.iter().enumerate() {
        let group = request.partition.as_ref().unwrap();
        let (head, ordinal, parts) = if at < 2 {
            (first, at, 2)
        } else {
            (second, at - 2, 3)
        };
        assert_eq!(group.first, head);
        assert_eq!(group.ordinal, ordinal as u32);
        assert_eq!(group.parts, parts);
    }
    // This description does not introduce native-only fail-fast policy in
    // the generic Ask. An unanswered earlier chunk remains unanswered.
    run.submit_fetch(
        requests[1].id,
        FetchResult::Error(ProviderError {
            kind: ErrorKind::DataSource,
            retryable: false,
            sqlstate: None,
            vendor_code: None,
            error_class: None,
            driver_message: Some("private".into()),
        }),
    )
    .unwrap();
    assert!(matches!(run.state(), RunState::AwaitingFetch(ids) if ids.len() == 4));
    run.cancel().unwrap();
    assert_eq!(run.state(), RunState::Cancelled);
}

#[test]
fn one_part_is_explicit_only_when_its_consumer_requests_partition_data() {
    for described in [false, true] {
        let mut run = Run::new(move |asker| async move {
            if described {
                asker.describe_fetch_partitions();
            }
            asker.ask_bound(request(), plans(1)).await;
            Ok(done())
        });
        let Step::Fetch { requests } = run.next().unwrap() else {
            panic!("expected actual request")
        };
        assert_eq!(requests.len(), 1);
        if described {
            let partition = requests[0].partition.as_ref().unwrap();
            assert_eq!(partition.first, requests[0].id);
            assert_eq!(partition.ordinal, 0);
            assert_eq!(partition.parts, 1);
        } else {
            assert!(requests[0].partition.is_none());
            assert!(
                serde_json::to_value(&requests[0])
                    .unwrap()
                    .get("partition")
                    .is_none()
            );
        }
    }
}
