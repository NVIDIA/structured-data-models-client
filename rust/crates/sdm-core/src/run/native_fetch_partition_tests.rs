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
fn counted_native_lookup_chunks_share_only_the_actual_read_and_task_owner() {
    use crate::materialize::native_lookup::LookupContext;
    use arrow_array::UInt64Array;
    use sdm_contract::data::{KeySet, Relation};
    use sdm_engine::graph::Table;
    let mut run = Run::new(|asker| async move {
        asker.describe_fetch_partitions();
        let context = Arc::new(LookupContext {
            entity: Table {
                name: "people".into(),
                source: Relation::named("people"),
                columns: vec![],
                primary_key: Some("id".into()),
                time_column: None,
            },
            key: "id".into(),
            scored: Arc::new(UInt64Array::from(vec![u64::MAX])),
            num_train: 2,
            range: sdm_contract::frames::materialize::FrameBatchRange { start: 4, stop: 5 },
        });
        let request = DataRequest::Lookup {
            table: Relation::named("pinned_people"),
            key: vec!["id".into()],
            columns: vec!["id".into()],
            seeds: KeySet {
                attachment: AttachmentId(99),
                columns: vec![],
            },
        };
        let mut first = Box::pin(
            asker
                .ask_bound(request.clone(), plans(2))
                .with_lookup(Arc::clone(&context)),
        );
        let mut second = Box::pin(asker.ask_bound(request, plans(1)).with_lookup(context));
        poll_fn(move |cx| {
            let _ = first.as_mut().poll(cx);
            let _ = second.as_mut().poll(cx);
            Poll::<()>::Pending
        })
        .await;
        Ok(done())
    });
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("lookup data")
    };
    let first = requests[0].lookup.as_ref().unwrap();
    assert_eq!(requests[1].lookup.as_ref(), Some(first));
    let second = requests[2].lookup.as_ref().unwrap();
    assert_ne!(first.owner, second.owner);
    assert_eq!(first.read, requests[0].id);
    assert_eq!(second.read, requests[2].id);
    assert_eq!(first.source, Relation::named("pinned_people"));
    assert_eq!(first.context_rows, 2);
    assert_eq!(first.scored_rows, 1);
    assert_eq!(first.range.start, 4);
    let Some(Attachment::LookupRows(owner)) = run.take_attachment(first.owner) else {
        panic!("lookup owner")
    };
    assert!(owner.matches(first));
    assert!(!owner.matches(second));
    run.cancel().unwrap();
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

#[test]
fn result_metadata_has_its_own_purpose_and_does_not_claim_the_future_data_owner() {
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use sdm_contract::data::ProviderChoice;
    use sdm_contract::run::FetchPurpose;
    // The core consumes a declared metadata port, independently of rendering.
    // Provider-specific projection remains covered by the data integration tests.
    struct ProjectedInput;
    impl sdm_contract::data::DataLayer for ProjectedInput {
        fn capabilities(
            &self,
            provider: &ProviderChoice,
        ) -> Result<sdm_contract::data::ProviderCapabilities, CoreError> {
            let ProviderChoice::Host { capabilities } = provider else {
                panic!("declared fixture")
            };
            Ok(capabilities.clone())
        }
        fn plan(&self, _: &DataRequest, _: &ProviderChoice) -> Result<FetchPlan, CoreError> {
            Ok(FetchPlan::Host {
                expect: ExpectedSchema::Discover {},
            })
        }
        fn result_metadata(
            &self,
            _: &ProviderChoice,
            _: &DataRequest,
            plan: &FetchPlan,
        ) -> Result<Option<FetchPlan>, CoreError> {
            Ok(Some(plan.clone()))
        }
        fn read(
            &self,
            _: &DataRequest,
            _: &ProviderChoice,
            _: &[RecordBatch],
        ) -> Result<sdm_contract::catalog::CatalogAnswer, CoreError> {
            Err(internal("fixture never reads a catalog"))
        }
    }
    let provider: ProviderChoice = serde_json::from_value(serde_json::json!({
        "provider":"host","capabilities":{"optional":[],"sampling":"seeded_stable","max_seed":u64::MAX}
    })).unwrap();
    let root = crate::CompositionRoot::new(Arc::new(ProjectedInput));
    let observed = Arc::new(Mutex::new(None));
    let returned = Arc::clone(&observed);
    let mut run = Run::with_data(
        root.data_access(&provider).unwrap(),
        move |asker| async move {
            asker.describe_fetch_partitions();
            let pending = asker.fetch(DataRequest::Scan {
                table: Relation::named("events"),
                columns: vec!["id".into(), "ts".into()],
                filters: vec![],
                limit: None,
            })?;
            let read = pending.source_read().expect("real future data identity");
            assert_eq!(read.first(), None);
            *returned.lock().unwrap() = Some(read);
            assert!(matches!(pending.await, Reply::Fetch(FetchResult::Data(_))));
            Ok(done())
        },
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("metadata fetch")
    };
    assert_eq!(requests.len(), 1);
    let metadata = &requests[0];
    assert_eq!(metadata.purpose, Some(FetchPurpose::DescribeResult));
    assert_eq!(observed.lock().unwrap().as_ref().unwrap().first(), None);
    let batch = RecordBatch::try_from_iter([
        (
            "name",
            Arc::new(StringArray::from(vec!["id", "ts"])) as ArrayRef,
        ),
        (
            "type",
            Arc::new(StringArray::from(vec!["BIGINT", "TIMESTAMP"])) as ArrayRef,
        ),
    ])
    .unwrap();
    run.submit_fetch(
        metadata.id,
        FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])),
    )
    .unwrap();
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("actual data fetch")
    };
    assert_eq!(requests.len(), 1);
    let data = &requests[0];
    assert_eq!(data.purpose, Some(FetchPurpose::SourceRows));
    assert_ne!(data.id, metadata.id);
    assert_eq!(data.partition.as_ref().unwrap().first, data.id);
    assert_eq!(
        observed.lock().unwrap().as_ref().unwrap().first(),
        Some(data.id)
    );
    let batch = RecordBatch::try_from_iter([
        ("id", Arc::new(Int64Array::from(vec![1])) as ArrayRef),
        ("ts", Arc::new(Int64Array::from(vec![2])) as ArrayRef),
    ])
    .unwrap();
    run.submit_fetch(
        data.id,
        FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])),
    )
    .unwrap();
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
}
