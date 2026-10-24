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
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
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

/// Postgres naming its schema, which a source must name or read from its
/// session (T3.7).
fn postgres() -> ProviderChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"postgres","namespace":["public"]}"#)
        .unwrap()
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
    assert!(
        requests.len() >= 3,
        "one fetch per chunk: {}",
        requests.len()
    );
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

                driver_message: None,
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

/// `sdm-data`'s plans and catalog reads, bound by the contract's default
/// `bind`: one fetch, the plan as it is, the key sets as the core made them.
struct Unbound(sdm_data::Data);

impl DataLayer for Unbound {
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
}

#[test]
fn bind_s_default_round_trips_one_fetch_of_the_plan_and_the_core_s_key_set() {
    // Postgres, whose own binding writes text: the default writes none.
    let seen = Arc::new(Mutex::new(Seen::default()));
    let made = seeds(vec![1, 2, 3]);
    let mut run = keyed_run(
        Arc::new(Unbound(sdm_data::Data::new())),
        &postgres(),
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
    let original = seen.lock().unwrap().keys.unwrap();
    let planned = sdm_data::Data::new()
        .plan(&lookup(&["users"], original), &postgres())
        .unwrap();
    assert_eq!(step.plan, planned, "the plan as it is");
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
    assert!(sets.iter().all(|set| set.attachment == original));
    let Some(Attachment::Arrow(handed)) = run.take_attachment(original) else {
        panic!("the key set is not attached")
    };
    assert_eq!(handed.batches, made.batches);
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
            key_enforcement: false,
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

// --- one conversion over the whole answer (fix 4) --------------------------------

/// A Postgres answer of `n`, a declared scale-zero `numeric` past Arrow's
/// decimals, as the provider hands it over: each value's exact text, its
/// declared scale marked, after the batch of each seed answered.
fn wide(batches: &[i64], values: &[&str]) -> FetchResult {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int64, false),
        Field::new("n", DataType::Utf8, true).with_metadata(
            [(
                sdm_contract::data::DECLARED_SCALE_MARKER.to_owned(),
                "0".to_owned(),
            )]
            .into_iter()
            .collect(),
        ),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(batches.to_vec())),
            Arc::new(arrow_array::StringArray::from(values.to_vec())),
        ],
    )
    .unwrap();
    FetchResult::Data(ArrowData::new(schema, vec![batch]))
}

/// A Postgres lookup past the budget, its chunks answered by `answers`.
fn chunked_postgres(answers: Vec<FetchResult>) -> Option<Reply> {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut run = keyed_run(
        Arc::new(sdm_data::Data::new()),
        &postgres(),
        &["wide_numbers"],
        seeds(many()),
        Arc::clone(&seen),
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("not a fetch")
    };
    assert_eq!(requests.len(), answers.len(), "one fetch per chunk");
    for (step, answer) in requests.iter().zip(answers) {
        run.submit_fetch(step.id, answer).unwrap();
    }
    match run.next() {
        Ok(Step::Done { .. }) => seen.lock().unwrap().reply.take(),
        Ok(other) => panic!("not done: {other:?}"),
        Err(error) => Some(Reply::Fetch(FetchResult::Error(ProviderError {
            error_class: Some(format!("{:?}", error.code)),
            kind: ErrorKind::DataSource,
            retryable: false,
            sqlstate: None,
            vendor_code: None,
            driver_message: None,
        }))),
    }
}

#[test]
fn chunks_of_one_answer_are_read_as_one_answer_small_wide_and_empty() {
    // A small value in one chunk, a wide one in another, none in a third:
    // the column is read as the whole answer says (each value's exact
    // text), as one fetch of all three would read it.
    let reply = chunked_postgres(vec![
        wide(&[0], &["1"]),
        wide(&[1], &["100000000000000000000"]),
        wide(&[], &[]),
    ]);
    let Some(Reply::Fetch(FetchResult::Data(data))) = reply else {
        panic!("the answer was refused: {reply:?}")
    };
    let column = data.schema.field_with_name("n").unwrap();
    assert_eq!(column.data_type(), &DataType::Utf8);
    let values: Vec<String> = data
        .batches
        .iter()
        .flat_map(|b| {
            let n = b.column(1).as_string::<i32>();
            (0..n.len())
                .map(|at| n.value(at).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(values, ["1", "100000000000000000000"]);
    // Every value small: the whole answer reads as integers.
    let reply = chunked_postgres(vec![wide(&[0], &["1"]), wide(&[], &[]), wide(&[2], &["7"])]);
    let Some(Reply::Fetch(FetchResult::Data(data))) = reply else {
        panic!("the answer was refused: {reply:?}")
    };
    assert_eq!(
        data.schema.field_with_name("n").unwrap().data_type(),
        &DataType::Int64
    );
}

#[test]
fn chunks_the_provider_answers_in_two_shapes_are_still_refused() {
    // A source schema that truly differs between chunks is the provider's
    // fault, refused as before.
    let other: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int64, false),
        Field::new("m", DataType::Int64, true),
    ]));
    let differs = FetchResult::Data(ArrowData::new(
        Arc::clone(&other),
        vec![
            RecordBatch::try_new(
                other,
                vec![
                    Arc::new(Int64Array::from(vec![1])),
                    Arc::new(Int64Array::from(vec![2])),
                ],
            )
            .unwrap(),
        ],
    ));
    let reply = chunked_postgres(vec![wide(&[0], &["1"]), differs, wide(&[], &[])]);
    let Some(Reply::Fetch(FetchResult::Error(error))) = reply else {
        panic!("a schema that differs was joined: {reply:?}")
    };
    assert_eq!(
        error.error_class.as_deref(),
        Some("ProviderSchemaMismatch"),
        "{error:?}"
    );
}

#[test]
fn timestamp_preflight_binds_empty_keys_then_preserves_the_original_lookup() {
    use arrow_array::{StringArray, TimestampMicrosecondArray};
    use sdm_contract::data::SqlResultConversion;
    let provider: ProviderChoice = serde_json::from_str(
        r#"{"provider":"sql","dialect":"databricks","timestamp_encoding":"naive_microseconds"}"#,
    )
    .unwrap();
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::new()));
    let seen = Arc::new(Mutex::new(Seen::default()));
    let saved = Arc::clone(&seen);
    let mut run = Run::with_data(
        root.data_access(&provider).unwrap(),
        move |asker| async move {
            let id = asker.attach(Attachment::Arrow(seeds(vec![7, 8])))?;
            let mut request = lookup(USERS, id);
            if let DataRequest::Lookup { columns, .. } = &mut request {
                columns.push("at".into());
            }
            saved.lock().unwrap().reply = Some(asker.fetch(request)?.await);
            Ok(done())
        },
    );
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("metadata fetch")
    };
    assert_eq!(requests.len(), 1);
    let metadata = &requests[0];
    let FetchPlan::Sql { statements, .. } = &metadata.plan else {
        panic!("SQL")
    };
    assert!(
        statements
            .last()
            .unwrap()
            .text
            .starts_with("DESCRIBE QUERY WITH ")
    );
    assert_eq!(cell(run.take_attachment(text_param(metadata))), "[]");
    let batch = RecordBatch::try_from_iter(vec![
        (
            "col_name",
            Arc::new(StringArray::from(vec![
                "__kumo_batch__",
                "user_id",
                "at",
                "__kumo_matches__",
            ])) as arrow_array::ArrayRef,
        ),
        (
            "data_type",
            Arc::new(StringArray::from(vec![
                "int",
                "bigint",
                "timestamp_ntz",
                "bigint",
            ])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();
    run.submit_fetch(
        metadata.id,
        FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])),
    )
    .unwrap();
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("data fetch")
    };
    assert_eq!(requests.len(), 1);
    let actual = &requests[0];
    assert_eq!(cell(run.take_attachment(text_param(actual))), "[7, 8]");
    let FetchPlan::Sql {
        statements,
        result_conversions,
        ..
    } = &actual.plan
    else {
        panic!("SQL")
    };
    assert_eq!(
        result_conversions,
        &[SqlResultConversion::TimestampMicros {
            column: 2,
            name: "at".into()
        }]
    );
    assert!(
        !statements
            .last()
            .unwrap()
            .text
            .starts_with("DESCRIBE QUERY")
    );
    assert!(
        statements
            .last()
            .unwrap()
            .text
            .contains("CAST(t.`at` AS TIMESTAMP_NTZ)")
    );
    let batch = RecordBatch::try_from_iter(vec![
        (
            "__kumo_batch__",
            Arc::new(Int64Array::from(vec![0, 1])) as arrow_array::ArrayRef,
        ),
        (
            "user_id",
            Arc::new(Int64Array::from(vec![7, 8])) as arrow_array::ArrayRef,
        ),
        (
            "at",
            Arc::new(TimestampMicrosecondArray::from(vec![
                1710037800123456_i64,
                -1,
            ])) as arrow_array::ArrayRef,
        ),
        (
            "__kumo_matches__",
            Arc::new(Int64Array::from(vec![1, 1])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();
    run.submit_fetch(
        actual.id,
        FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])),
    )
    .unwrap();
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    assert!(matches!(
        seen.lock().unwrap().reply,
        Some(Reply::Fetch(FetchResult::Data(_)))
    ));
}
