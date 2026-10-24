// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core sends a request as the data layer binds it (T3.28, design S4):
//! each fetch of `DataLayer::bind`'s answer in one step, each batch it
//! names installed as one of the run's attachments and named by the run's
//! id, and the answers joined in order; any failed fetch fails the
//! request. A key set's text travels as an attachment, never in the
//! control message.
//!
//! The provider here is a fake Databricks warehouse: it reads the one
//! text cell each fetch's attachment holds, as the old client's statement
//! reads its parameter (`posexplode(from_json(?, ...))`), and numbers each
//! seed by its position plus the offset the statement adds.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, Determinism, FetchPlan, KeySet, OptionalRequest,
    ProviderCapabilities, ProviderChoice, ProviderError, SqlParam,
};
use sdm_contract::ops::OperationResult;
use sdm_contract::run::FetchStep;
use sdm_contract::{CoreError, ErrorKind, Step};
use sdm_core::{ArrowData, Asker, Attachment, CompositionRoot, FetchResult, Reply, Run};

/// The budget of one key set's text on Databricks, in bytes.
const BUDGET: usize = 900_000;

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

/// A run on `provider` over `layer` whose body attaches `keys` and looks
/// them up in `source`, keeping what it saw in `seen`.
fn keyed_run(
    layer: Arc<dyn DataLayer>,
    provider: &ProviderChoice,
    source: &'static [&'static str],
    keys: ArrowData,
    seen: Arc<Mutex<Seen>>,
) -> Run {
    let root = CompositionRoot::new(layer);
    Run::with_data(
        root.data_access(provider).unwrap(),
        move |asker: Asker| async move {
            let id = asker.attach(Attachment::Arrow(keys))?;
            seen.lock().unwrap().keys = Some(id);
            let reply = asker.fetch(lookup(source, id))?.await;
            seen.lock().unwrap().reply = Some(reply);
            Ok(done())
        },
    )
}

fn databricks() -> ProviderChoice {
    ProviderChoice::sql("databricks")
}

const USERS: &[&str] = &["main", "shop", "users"];

/// The one text cell an attachment holds.
fn cell(attachment: Option<Attachment>) -> String {
    let Some(Attachment::Arrow(data)) = attachment else {
        panic!("a text parameter's attachment is not Arrow")
    };
    let rows: usize = data.batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 1, "one row");
    assert_eq!(data.schema.fields().len(), 1, "one column");
    let batch = data.batches.iter().find(|b| b.num_rows() == 1).unwrap();
    batch.column(0).as_string::<i32>().value(0).to_owned()
}

/// The offset a chunk's statement adds to each seed's position: none, or
/// the `pos + <n>` it writes.
fn offset(text: &str) -> usize {
    text.split_once("pos + ")
        .map(|(_, rest)| {
            rest.split_once(' ')
                .map(|(n, _)| n.parse().unwrap())
                .unwrap()
        })
        .unwrap_or(0)
}

/// A fake warehouse's answer to one fetch of `step`, its text `payload`:
/// each seed's position plus the statement's offset, and the seed.
fn answer(step: &FetchStep, payload: &str) -> FetchResult {
    let FetchPlan::Sql { statements, .. } = &step.plan else {
        panic!("not SQL")
    };
    let moved = offset(&statements.last().unwrap().text);
    let keys: Vec<i64> = serde_json::from_str(payload).unwrap();
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int32, false),
        Field::new("seed", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(
                (0..keys.len()).map(|at| i32::try_from(at + moved).unwrap()),
            )),
            Arc::new(Int64Array::from(keys)),
        ],
    )
    .unwrap();
    FetchResult::Data(ArrowData::new(schema, vec![batch]))
}

/// The text attachment the last statement of `step` binds.
fn text_param(step: &FetchStep) -> AttachmentId {
    let FetchPlan::Sql { statements, .. } = &step.plan else {
        panic!("not SQL")
    };
    let params = &statements.last().unwrap().params;
    let [SqlParam::Text { attachment }] = &params[..] else {
        panic!("not one text parameter: {params:?}")
    };
    *attachment
}

/// Keys whose text is past the budget: six-digit keys, about 2.1 MB.
fn many() -> Vec<i64> {
    (0..300_000).map(|i| 100_000 + i).collect()
}

#[test]
fn a_key_set_past_the_budget_is_asked_as_chunks_in_one_hop_and_joined_in_order() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let keys = many();
    let mut run = keyed_run(
        Arc::new(sdm_data::Data::new()),
        &databricks(),
        USERS,
        seeds(keys.clone()),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    assert!(requests.len() >= 3, "one fetch per chunk: {}", requests.len());
    // The core's own key set is not what the provider is handed.
    let original = seen.lock().unwrap().keys.unwrap();
    let named: BTreeSet<AttachmentId> = requests.iter().map(text_param).collect();
    assert_eq!(named.len(), requests.len(), "each chunk its own text");
    assert!(!named.contains(&original));
    let mut payloads = Vec::new();
    for step in &requests {
        let payload = cell(run.take_attachment(text_param(step)));
        assert!(payload.len() <= BUDGET, "{}", payload.len());
        payloads.push(payload);
    }
    assert!(
        run.take_attachment(original).is_none(),
        "the core's key set is not left behind"
    );
    // The chunks are the seeds in order, each once.
    let sent: Vec<i64> = payloads
        .iter()
        .flat_map(|p| serde_json::from_str::<Vec<i64>>(p).unwrap())
        .collect();
    assert_eq!(sent, keys);
    for (step, payload) in requests.iter().zip(&payloads) {
        run.submit_fetch(step.id, answer(step, payload)).unwrap();
    }
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    let reply = seen.lock().unwrap().reply.take().unwrap();
    let Reply::Fetch(FetchResult::Data(data)) = reply else {
        panic!("not data: {reply:?}")
    };
    let batches: Vec<i32> = data
        .batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    let expected: Vec<i32> = (0..i32::try_from(keys.len()).unwrap()).collect();
    assert_eq!(batches, expected, "each seed at its place in the key set");
    let found: Vec<i64> = data
        .batches
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(found, keys);
}

#[test]
fn a_later_chunk_failing_fails_the_request() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut run = keyed_run(
        Arc::new(sdm_data::Data::new()),
        &databricks(),
        USERS,
        seeds(many()),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    assert!(requests.len() >= 2);
    let failing = requests.len() - 1;
    for (at, step) in requests.iter().enumerate() {
        let payload = cell(run.take_attachment(text_param(step)));
        let result = if at == failing {
            FetchResult::Error(ProviderError {
                error_class: Some("INSUFFICIENT_PERMISSIONS".into()),
                kind: ErrorKind::DataSource,
                retryable: false,
                sqlstate: Some("42501".into()),
                vendor_code: None,
            })
        } else {
            answer(step, &payload)
        };
        run.submit_fetch(step.id, result).unwrap();
    }
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    let reply = seen.lock().unwrap().reply.take().unwrap();
    let Reply::Fetch(FetchResult::Error(error)) = reply else {
        panic!("the request did not fail: {reply:?}")
    };
    assert_eq!(error.sqlstate.as_deref(), Some("42501"));
}

#[test]
fn a_key_set_within_the_budget_is_one_fetch_of_one_text() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut run = keyed_run(
        Arc::new(sdm_data::Data::new()),
        &databricks(),
        USERS,
        seeds(vec![5, 6]),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    let [step] = &requests[..] else {
        panic!("one fetch")
    };
    let payload = cell(run.take_attachment(text_param(step)));
    assert_eq!(payload, "[5,6]");
    run.submit_fetch(step.id, answer(step, &payload)).unwrap();
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    assert!(matches!(
        seen.lock().unwrap().reply,
        Some(Reply::Fetch(FetchResult::Data(_)))
    ));
}

// --- bind's default, round-tripped through the core ------------------------------

/// `sdm-data`'s layer, its key sets rebound by the contract's default
/// `bind` through `bound_keys`: each key doubled, so the batch the provider
/// is handed is told apart from the core's.
struct Doubling(sdm_data::Data);

impl DataLayer for Doubling {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        self.0.capabilities(provider)
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        self.0.plan(request, provider)
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        self.0.read(request, provider, batches)
    }

    fn bound_keys(
        &self,
        _provider: &ProviderChoice,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<Option<(SchemaRef, Vec<RecordBatch>)>, CoreError> {
        let doubled = batches
            .iter()
            .map(|batch| {
                let keys = batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int64Type>();
                let keys: Int64Array = keys.iter().map(|k| k.map(|k| k * 2)).collect();
                RecordBatch::try_new(
                    Arc::clone(schema),
                    vec![Arc::new(keys), Arc::clone(batch.column(1))],
                )
                .unwrap()
            })
            .collect();
        Ok(Some((Arc::clone(schema), doubled)))
    }
}

#[test]
fn bind_s_default_round_trips_one_fetch_its_key_set_rebound_once() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut run = keyed_run(
        Arc::new(Doubling(sdm_data::Data::new())),
        &ProviderChoice::sql("duckdb"),
        &["users"],
        seeds(vec![1, 2, 3]),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    let [step] = &requests[..] else {
        panic!("one fetch")
    };
    // The plan is the layer's, its key set named by the id it is handed
    // under; that attachment holds the key set as the layer rebound it.
    let FetchPlan::Sql { statements, .. } = &step.plan else {
        panic!("not SQL")
    };
    let sets: Vec<&KeySet> = statements
        .iter()
        .flat_map(|s| &s.params)
        .filter_map(|p| match p {
            SqlParam::KeySet { keys, .. } => Some(keys),
            _ => None,
        })
        .collect();
    let [set] = &sets[..] else {
        panic!("one key set: {sets:?}")
    };
    let mut planned = sdm_data::Data::new()
        .plan(
            &lookup(&["users"], seen.lock().unwrap().keys.unwrap()),
            &ProviderChoice::sql("duckdb"),
        )
        .unwrap();
    if let FetchPlan::Sql { statements, .. } = &mut planned {
        for param in statements.iter_mut().flat_map(|s| &mut s.params) {
            if let SqlParam::KeySet { keys, .. } = param {
                keys.attachment = set.attachment;
            }
        }
    }
    assert_eq!(step.plan, planned);
    let Some(Attachment::Arrow(handed)) = run.take_attachment(set.attachment) else {
        panic!("the key set is not attached")
    };
    let keys: Vec<i64> = handed
        .batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(keys, [2, 4, 6], "rebound once");
    // Nothing else is left attached for the host.
    for id in 1..=8 {
        assert!(run.take_attachment(AttachmentId(id)).is_none(), "{id}");
    }
}

#[test]
fn a_host_provider_is_handed_the_core_s_key_set_as_it_was_made() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let host = ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            derived_relations: false,
            neighbor_lookback: false,
            keys_optional: false,
            sample_rows: false,
            optional: BTreeSet::from([OptionalRequest::Lookup]),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::Unseeded,
            fallback: None,
        },
    };
    let made = seeds(vec![9, 8]);
    let mut run = keyed_run(
        Arc::new(sdm_data::Data::new()),
        &host,
        &["users"],
        made.clone(),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    let [step] = &requests[..] else {
        panic!("one fetch")
    };
    assert!(matches!(step.plan, FetchPlan::Host { .. }));
    let original = seen.lock().unwrap().keys.unwrap();
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
