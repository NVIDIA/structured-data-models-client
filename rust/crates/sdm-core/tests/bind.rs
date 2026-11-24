// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core sends a request as its provider takes it (design S4): whole,
//! the key set as the core made it, or, past the keys the provider takes
//! in one request (`max_keys_per_request`), as parts of its seeds asked in
//! one step, each part's seeds installed as one of the run's attachments
//! and named by the run's id, and the answers joined in order; any failed
//! part fails the request. A built-in source's key sets written as text
//! are its SQL kit's (`sdm-sql`'s `kit_bind` tests).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::data::{
    AttachmentId, DataRequest, Determinism, FetchPlan, OptionalRequest, ProviderCapabilities,
    ProviderChoice,
};
use sdm_contract::ops::OperationResult;
use sdm_contract::{ErrorCode, Step};
use sdm_core::{ArrowData, Asker, Attachment, CompositionRoot, FetchResult, Reply, Run};

/// What the body saw: the key set's attachment id and the reply.
#[derive(Default)]
struct Seen {
    keys: Option<AttachmentId>,
    reply: Option<Reply>,
}

fn done() -> OperationResult {
    OperationResult::ValidateQuery(sdm_contract::ops::QueryValidation {
        accepted: true,
        diagnostics: vec![],
        query: None,
    })
}

/// Seeds of `keys`, batches from 0.
fn seeds(keys: Vec<i64>) -> ArrowData {
    let rows = keys.len() as i64;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from_iter_values(0..rows)),
        ],
    )
    .unwrap();
    ArrowData::new(schema, vec![batch])
}

fn lookup(source: &[&str], seeds: AttachmentId) -> DataRequest {
    let mut request: DataRequest = serde_json::from_str(&format!(
        r#"{{"columns":["user_id"],"key":["user_id"],"request":"lookup","seeds":{{"attachment":0,"columns":[{{"name":"key","nullable":true,"type":{{"type":"int64"}}}},{{"name":"batch","nullable":false,"type":{{"type":"int64"}}}}]}},"table":{{"source":{}}}}}"#,
        serde_json::to_string(source).unwrap()
    ))
    .unwrap();
    let DataRequest::Lookup { seeds: set, .. } = &mut request else {
        unreachable!("a lookup")
    };
    set.attachment = seeds;
    request
}

/// A run on `provider` whose body attaches `keys` and looks them up in
/// `users`, keeping what it saw in `seen`.
fn keyed_run(provider: &ProviderChoice, keys: ArrowData, seen: Arc<Mutex<Seen>>) -> Run {
    let root = CompositionRoot::default();
    Run::with_data(
        root.data_access(provider).unwrap(),
        move |asker: Asker| async move {
            let id = asker.attach(Attachment::Arrow(keys))?;
            seen.lock().unwrap().keys = Some(id);
            let reply = asker.fetch(lookup(&["users"], id))?.await;
            seen.lock().unwrap().reply = Some(reply);
            Ok(done())
        },
    )
}

/// A provider declaring lookups, taking at most `max` keys a request.
fn host(max: Option<u64>) -> ProviderChoice {
    ProviderChoice::new(ProviderCapabilities {
        opaque_vendor: None,
        compares_stored_times: false,
        samples_by_key_dtype: false,
        reads_nan_as_missing: false,
        derived_relations: false,
        neighbor_lookback: false,
        keys_optional: false,
        key_enforcement: false,
        sample_rows: false,
        optional: BTreeSet::from([OptionalRequest::Lookup]),
        filters: Default::default(),
        max_keys_per_request: max,
        max_seed: u64::MAX,
        text_times: None,
        null_order: None,
        index_advice: None,
        sampling: Determinism::Unseeded,
        fallback_text_order: None,
        fallback: None,
        database: None,
        column_affinities: false,
        reads_tables_as_they_stand: false,
        namespace: Vec::new(),
        tries_upper_case_names: false,
        namespace_first: false,
        view: None,
    })
}

/// A provider's answer to a lookup of `seeds`: each seed's batch position
/// and the seed, its user.
fn answer(seeds: &ArrowData) -> FetchResult {
    let mut batches: Vec<i64> = Vec::new();
    let mut keys: Vec<i64> = Vec::new();
    for batch in &seeds.batches {
        keys.extend(batch.column(0).as_primitive::<Int64Type>().values().iter());
        batches.extend(batch.column(1).as_primitive::<Int64Type>().values().iter());
    }
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(batches)),
            Arc::new(Int64Array::from(keys)),
        ],
    )
    .unwrap();
    FetchResult::Data(ArrowData::new(schema, vec![batch]))
}

fn int64s(data: &ArrowData, column: usize) -> Vec<i64> {
    data.batches
        .iter()
        .flat_map(|b| {
            b.column(column)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

#[test]
fn a_host_provider_is_handed_the_core_s_key_set_as_it_was_made() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let made = seeds(vec![9, 8]);
    let mut run = keyed_run(&host(None), made.clone(), Arc::clone(&seen));
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    let [step] = &requests[..] else {
        panic!("one fetch")
    };
    let original = seen.lock().unwrap().keys.unwrap();
    assert_eq!(step.plan.key_sets, [original]);
    let DataRequest::Lookup { seeds, .. } = &step.request else {
        panic!("not the lookup")
    };
    assert_eq!(seeds.attachment, original);
    let Some(Attachment::Arrow(handed)) = run.take_attachment(original) else {
        panic!("the key set is not attached")
    };
    assert_eq!(handed.batches, made.batches);
    assert_eq!(handed.schema, made.schema);
}

#[test]
fn seeds_past_the_provider_s_limit_are_asked_as_parts_in_one_hop_and_joined_in_order() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let keys: Vec<i64> = (100..107).collect();
    let mut run = keyed_run(&host(Some(3)), seeds(keys.clone()), Arc::clone(&seen));
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    assert_eq!(requests.len(), 3, "parts of at most three seeds");
    let original = seen.lock().unwrap().keys.unwrap();
    let mut sent = Vec::new();
    let mut answers = Vec::new();
    for step in &requests {
        let DataRequest::Lookup { seeds, .. } = &step.request else {
            panic!("not the lookup")
        };
        // Each part's request and plan name its own seeds, never the
        // core's key set.
        assert_ne!(seeds.attachment, original);
        assert_eq!(step.plan.key_sets, [seeds.attachment]);
        let Some(Attachment::Arrow(part)) = run.take_attachment(seeds.attachment) else {
            panic!("a part's seeds are not attached")
        };
        assert!(
            part.batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
                <= 3
        );
        sent.extend(int64s(&part, 0));
        answers.push(answer(&part));
    }
    assert_eq!(sent, keys, "the seeds in order, each once");
    assert!(
        run.take_attachment(original).is_none(),
        "the core's key set is not left behind"
    );
    for (step, answer) in requests.iter().zip(answers) {
        run.submit_fetch(step.id, answer).unwrap();
    }
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    let reply = seen.lock().unwrap().reply.take().unwrap();
    let Reply::Fetch(FetchResult::Data(data)) = reply else {
        panic!("not data: {reply:?}")
    };
    assert_eq!(int64s(&data, 0), (0..7).collect::<Vec<i64>>());
    assert_eq!(int64s(&data, 1), keys);
}

#[test]
fn parts_the_provider_answers_in_two_shapes_are_refused() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut run = keyed_run(&host(Some(2)), seeds(vec![1, 2, 3]), Arc::clone(&seen));
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    assert_eq!(requests.len(), 2);
    let other: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int64, false),
        Field::new("m", DataType::Int64, true),
    ]));
    for (at, step) in requests.iter().enumerate() {
        let DataRequest::Lookup { seeds, .. } = &step.request else {
            panic!("not the lookup")
        };
        let Some(Attachment::Arrow(part)) = run.take_attachment(seeds.attachment) else {
            panic!("a part's seeds are not attached")
        };
        let result = if at == 0 {
            answer(&part)
        } else {
            let batch = RecordBatch::try_new(
                Arc::clone(&other),
                vec![
                    Arc::new(Int64Array::from(vec![2])),
                    Arc::new(Int64Array::from(vec![3])),
                ],
            )
            .unwrap();
            FetchResult::Data(ArrowData::new(Arc::clone(&other), vec![batch]))
        };
        run.submit_fetch(step.id, result).unwrap();
    }
    let error = run.next().unwrap_err();
    assert_eq!(error.code, ErrorCode::ProviderSchemaMismatch, "{error:?}");
}

#[test]
fn a_plan_names_the_request_s_key_sets() {
    let request = lookup(&["users"], AttachmentId(4));
    assert_eq!(
        FetchPlan::of(request.expected(), &request).key_sets,
        [AttachmentId(4)]
    );
}
