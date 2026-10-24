// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The in-memory fallback (T2.6), the implementer's own tests: a host
//! provider that declares no optional request and the fallback's limits,
//! each scan the fallback asks answered here from one in-memory table, and
//! the answers held to what each request's pushdown gives on that table,
//! worked out by hand. The conformance kit holds the fallback to the
//! pushdowns on every provider.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, Determinism, FallbackLimits, Filter, KeySet,
    MATCHES_COLUMN, OptionalRequest, ProviderCapabilities, ProviderChoice, ProviderError,
    SeedBound, TableColumn, TableKey, TieBreak, TieTerm, Window,
};
use sdm_contract::ops::{OperationResult, QueryValidation};
use sdm_contract::{ColumnType, CoreError, Detail, ErrorCode, ErrorKind, Scalar, Step, Unit};
use sdm_core::{ArrowData, Asker, Attachment, CompositionRoot, FetchResult, Reply, Run};

const US: i64 = 1_000_000;

/// The one table every scan reads: `user`, `at` (microseconds), `v`, `day`
/// (text, which the fallback reads no time from).
fn table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("user", DataType::Int64, true),
        Field::new("at", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("v", DataType::Utf8, true),
        Field::new("day", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                None,
                Some(3),
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(10 * US),
                Some(30 * US),
                Some(20 * US),
                Some(5 * US),
                None,
                Some(40 * US),
                Some(50 * US),
            ])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("c"),
                Some("b"),
                Some("d"),
                Some("e"),
                Some("f"),
                None,
            ])),
            Arc::new(StringArray::from(vec![Some("2024-01-01"); 7])),
        ],
    )
    .unwrap()
}

fn limits(max_rows: u64, max_bytes: u64) -> FallbackLimits {
    FallbackLimits {
        max_rows,
        max_bytes,
    }
}

fn host(optional: &[OptionalRequest], fallback: Option<FallbackLimits>) -> ProviderChoice {
    ProviderChoice::Host {
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
            optional: optional.iter().copied().collect::<BTreeSet<_>>(),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::SeededStable,
            fallback,
        },
    }
}

/// What a run gave: its answer, or its error, and the requests it asked.
struct Ran {
    answer: Result<RecordBatch, CoreError>,
    asked: Vec<DataRequest>,
}

/// Runs one fetch of `request` (with `seeds`, for a request that takes
/// them) on `provider`, each scan answered from `rows` by `answer`.
fn run_with(
    provider: ProviderChoice,
    request: DataRequest,
    seeds: Option<RecordBatch>,
    answer: impl Fn(&DataRequest) -> FetchResult,
) -> Ran {
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::new()));
    let kept: Arc<Mutex<Option<ArrowData>>> = Arc::default();
    let into = Arc::clone(&kept);
    let mut run = Run::with_data(
        root.data_access(&provider).unwrap(),
        move |asker: Asker| async move {
            let mut request = request;
            if let Some(seeds) = seeds {
                let id = asker.attach(Attachment::Arrow(ArrowData::new(
                    seeds.schema(),
                    vec![seeds],
                )))?;
                match &mut request {
                    DataRequest::Neighbors { seeds, .. } | DataRequest::Lookup { seeds, .. } => {
                        seeds.attachment = id;
                    }
                    _ => unreachable!(),
                }
            }
            match asker.fetch(request)?.await {
                Reply::Fetch(FetchResult::Data(data)) => {
                    *into.lock().unwrap() = Some(data);
                    Ok(OperationResult::ValidateQuery(QueryValidation {
                        accepted: true,
                        diagnostics: vec![],
                        query: None,
                    }))
                }
                Reply::Fetch(FetchResult::Error(error)) => Err(CoreError::provider(&error)),
                _ => unreachable!(),
            }
        },
    );
    let mut asked = vec![];
    let ended = loop {
        match run.next() {
            Ok(Step::Fetch { requests }) => {
                for step in requests {
                    asked.push(step.request.clone());
                    run.submit_fetch(step.id, answer(&step.request)).unwrap();
                }
            }
            Ok(Step::Done { .. }) => break Ok(()),
            Ok(other) => panic!("{other:?}"),
            Err(error) => break Err(error),
        }
    };
    let answer = ended.map(|()| {
        let data = kept.lock().unwrap().take().unwrap();
        arrow_select::concat::concat_batches(&data.schema, &data.batches).unwrap()
    });
    Ran { answer, asked }
}

/// The fallback's answer on the table, its limits ample.
fn fallback(request: DataRequest, seeds: Option<RecordBatch>) -> Ran {
    run_with(
        host(&[], Some(limits(1_000, 1 << 30))),
        request,
        seeds,
        scan_of(table()),
    )
}

/// A host that answers a scan from `rows`: its columns by name, its
/// not-null filters, then its limit.
fn scan_of(rows: RecordBatch) -> impl Fn(&DataRequest) -> FetchResult {
    move |request| {
        let DataRequest::Scan {
            columns,
            filters,
            limit,
            ..
        } = request
        else {
            panic!("the fallback asked for {request:?}")
        };
        let mut kept: Vec<u32> = (0..rows.num_rows() as u32).collect();
        for filter in filters {
            let Filter::NotNull { column } = filter else {
                panic!("{filter:?}")
            };
            let column = rows.column_by_name(column).unwrap();
            kept.retain(|&row| column.is_valid(row as usize));
        }
        if let Some(limit) = limit {
            kept.truncate(*limit as usize);
        }
        let at = arrow_array::UInt32Array::from(kept);
        let fields: Vec<Field> = columns
            .iter()
            .map(|c| rows.schema().field_with_name(c).unwrap().clone())
            .collect();
        let arrays: Vec<ArrayRef> = columns
            .iter()
            .map(|c| arrow_select::take::take(rows.column_by_name(c).unwrap(), &at, None).unwrap())
            .collect();
        let schema = Arc::new(Schema::new(fields));
        FetchResult::Data(ArrowData::new(
            Arc::clone(&schema),
            vec![RecordBatch::try_new(schema, arrays).unwrap()],
        ))
    }
}

fn int64(batch: &RecordBatch, name: &str) -> Vec<Option<i64>> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_primitive::<Int64Type>()
        .iter()
        .collect()
}

fn text(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect()
}

fn keys(columns: &[(&str, Vec<Option<i64>>)]) -> Vec<sdm_contract::Field> {
    columns
        .iter()
        .map(|(name, _)| sdm_contract::data::field(name, ColumnType::Int64, true))
        .collect()
}

/// Seeds of `key` values, batch positions 0.., and a window's upper bound
/// in seconds.
fn seeds(key: &[Option<i64>], upper: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, true),
            Field::new("batch", DataType::Int64, false),
            Field::new(
                "upper",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
        ])),
        vec![
            Arc::new(Int64Array::from(key.to_vec())),
            Arc::new(Int64Array::from((0..key.len() as i64).collect::<Vec<_>>())),
            Arc::new(TimestampNanosecondArray::from(
                upper.iter().map(|s| s * 1_000_000_000).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn key_set(batch: &RecordBatch) -> KeySet {
    let mut columns = keys(&[("key", vec![]), ("batch", vec![])]);
    columns.push(sdm_contract::data::field(
        "upper",
        ColumnType::Timestamp {
            unit: Unit::Ns,
            zone: None,
        },
        true,
    ));
    let _ = batch;
    KeySet {
        attachment: AttachmentId(0),
        columns,
    }
}

fn neighbors(window: bool, per_seed: Option<u64>, seeds: &RecordBatch) -> DataRequest {
    DataRequest::Neighbors {
        table: "events".into(),
        key: vec!["user".into()],
        columns: vec!["v".into(), "at".into()],
        seeds: key_set(seeds),
        window: window.then(|| Window {
            source_type: None,
            column: "at".into(),
            lower: None,
            upper: Some(SeedBound {
                seed_column: "upper".into(),
                inclusive: true,
            }),
        }),
        per_seed,
        order: per_seed.map(|_| {
            TieBreak(vec![TieTerm {
                column: "at".into(),
                as_timestamp: true,
                descending: true,
                nulls_last: true,
            }])
        }),
    }
}

fn refusal(ran: &Ran) -> &CoreError {
    ran.answer.as_ref().expect_err("an error")
}

fn field<'a>(error: &'a CoreError, name: &str) -> Option<&'a Detail> {
    error.fields.get(name)
}

#[test]
fn neighbors_keep_each_seeds_latest_rows_within_its_window() {
    // Seed 0 is user 1 up to 25 s, seed 1 user 2 up to 100 s, seed 2 a
    // user no row holds, seed 3 user 1 up to 100 s: the latest two rows of
    // each within its window, by `at` descending, in batch order.
    let seeds = seeds(&[Some(1), Some(2), Some(9), Some(1)], &[25, 100, 100, 100]);
    let ran = fallback(neighbors(true, Some(2), &seeds), Some(seeds.clone()));
    let answer = ran.answer.unwrap();
    assert_eq!(answer.schema().field(0).name(), BATCH_COLUMN);
    assert_eq!(
        int64(&answer, BATCH_COLUMN),
        [Some(0), Some(0), Some(1), Some(3), Some(3)]
    );
    assert_eq!(
        text(&answer, "v"),
        ["b", "a", "d", "c", "b"].map(|v| Some(v.to_owned()))
    );
    // One scan of the columns the request reads, each once.
    let [DataRequest::Scan { columns, .. }] = ran.asked.as_slice() else {
        panic!("{:?}", ran.asked)
    };
    assert_eq!(columns, &["user", "v", "at"]);
}

#[test]
fn neighbors_without_a_limit_give_every_row_of_each_seed() {
    let seeds = seeds(&[Some(2), Some(1), None], &[0, 0, 0]);
    let answer = fallback(neighbors(false, None, &seeds), Some(seeds.clone()))
        .answer
        .unwrap();
    // The null seed matches nothing; the null key no seed.
    assert_eq!(
        int64(&answer, BATCH_COLUMN),
        [Some(0), Some(0), Some(1), Some(1), Some(1)]
    );
    assert_eq!(
        text(&answer, "v"),
        ["d", "e", "a", "c", "b"].map(|v| Some(v.to_owned()))
    );
}

#[test]
fn a_key_of_another_integer_width_joins_its_seeds_exactly() {
    // The table's key as int32, the seeds int64, as an INTEGER key meets
    // the core's seeds.
    let mut rows = table();
    let narrow: ArrayRef = Arc::new(Int32Array::from(vec![
        Some(1),
        Some(1),
        Some(1),
        Some(2),
        Some(2),
        None,
        Some(3),
    ]));
    let mut fields: Vec<Field> = rows
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields[0] = Field::new("user", DataType::Int32, true);
    let mut columns = rows.columns().to_vec();
    columns[0] = narrow;
    rows = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let seeds = seeds(&[Some(3), Some(9), Some(1)], &[0, 0, 0]);
    let request = DataRequest::Lookup {
        table: "events".into(),
        key: vec!["user".into()],
        columns: vec!["v".into()],
        seeds: key_set(&seeds),
    };
    let ran = run_with(
        host(&[], Some(limits(1_000, 1 << 30))),
        request,
        Some(seeds),
        scan_of(rows),
    );
    let answer = ran.answer.unwrap();
    assert_eq!(int64(&answer, BATCH_COLUMN), [Some(0), Some(2)]);
    assert_eq!(int64(&answer, MATCHES_COLUMN), [Some(1), Some(3)]);
}

#[test]
fn a_lookup_gives_one_row_per_seed_found_with_its_count() {
    let seeds = seeds(&[Some(3), Some(9), Some(1)], &[0, 0, 0]);
    let request = DataRequest::Lookup {
        table: "events".into(),
        key: vec!["user".into()],
        columns: vec!["user".into()],
        seeds: key_set(&seeds),
    };
    let answer = fallback(request, Some(seeds)).answer.unwrap();
    let names: Vec<String> = answer
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, [BATCH_COLUMN, MATCHES_COLUMN, "user"]);
    assert_eq!(int64(&answer, BATCH_COLUMN), [Some(0), Some(2)]);
    assert_eq!(int64(&answer, MATCHES_COLUMN), [Some(1), Some(3)]);
    assert_eq!(int64(&answer, "user"), [Some(3), Some(1)]);
}

#[test]
fn key_counts_count_rows_distinct_keys_and_rows_missing_one() {
    let request = DataRequest::KeyCounts {
        keys: vec![
            TableKey {
                unique: false,
                table: "events".into(),
                columns: vec!["user".into()],
                label: None,
            },
            TableKey {
                unique: false,
                table: "events".into(),
                columns: vec!["user".into(), "v".into()],
                label: None,
            },
        ],
    };
    let answer = fallback(request, None).answer.unwrap();
    let column = |i: usize| answer.column(i).clone();
    assert_eq!(
        column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[0, 1]
    );
    assert_eq!(int64(&answer, "n"), [Some(7), Some(7)]);
    assert_eq!(int64(&answer, "d"), [Some(3), Some(5)]);
    assert_eq!(int64(&answer, "z"), [Some(1), Some(2)]);
}

#[test]
fn time_bounds_are_each_columns_smallest_and_largest_time() {
    let request = DataRequest::TimeBounds {
        columns: vec![TableColumn {
            table: "events".into(),
            column: "at".into(),
        }],
    };
    let answer = fallback(request, None).answer.unwrap();
    let times = |name: &str| -> Vec<Option<i64>> {
        answer
            .column_by_name(name)
            .unwrap()
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect()
    };
    assert_eq!(times("min_date"), [Some(5 * US)]);
    assert_eq!(times("max_date"), [Some(50 * US)]);
}

#[test]
fn column_ranges_and_a_repeated_key_read_every_row() {
    let ranges = fallback(
        DataRequest::ColumnRanges {
            table: "events".into(),
            columns: vec!["user".into(), "v".into()],
        },
        None,
    )
    .answer
    .unwrap();
    assert_eq!(ranges.num_columns(), 7);
    assert_eq!(ranges.num_rows(), 1);
    let count = |i: usize| ranges.column(i).as_primitive::<Int64Type>().value(0);
    assert_eq!((count(0), count(3), count(6)), (7, 6, 6));
    assert_eq!(ranges.column(1).as_primitive::<Int64Type>().value(0), 1);
    assert_eq!(ranges.column(2).as_primitive::<Int64Type>().value(0), 3);
    assert_eq!(ranges.column(4).as_string::<i32>().value(0), "a");
    assert_eq!(ranges.column(5).as_string::<i32>().value(0), "f");

    let repeated = fallback(
        DataRequest::RepeatedKey {
            table: "events".into(),
            key: vec!["user".into()],
        },
        None,
    )
    .answer
    .unwrap();
    assert_eq!(int64(&repeated, "user"), [Some(1)]);
    let none = fallback(
        DataRequest::RepeatedKey {
            table: "events".into(),
            key: vec!["v".into()],
        },
        None,
    )
    .answer
    .unwrap();
    assert_eq!(none.num_rows(), 0);
}

#[test]
fn a_preview_is_the_first_rows() {
    let answer = fallback(
        DataRequest::Preview {
            table: "events".into(),
            columns: vec!["v".into()],
            limit: 2,
        },
        None,
    )
    .answer
    .unwrap();
    assert_eq!(text(&answer, "v"), [Some("a".into()), Some("c".into())]);
}

#[test]
fn a_count_reads_the_rows_its_filters_keep() {
    let ran = fallback(
        DataRequest::CountRows {
            table: "events".into(),
            filters: vec![Filter::NotNull {
                column: "at".into(),
            }],
        },
        None,
    );
    assert_eq!(int64(&ran.answer.unwrap(), "rows"), [Some(6)]);
    let [DataRequest::Scan { columns, .. }] = ran.asked.as_slice() else {
        panic!("{:?}", ran.asked)
    };
    assert_eq!(columns, &["at"]);
}

fn sample(n: u64, seed: u64) -> DataRequest {
    DataRequest::SampleEntities {
        rows: None,
        table: "events".into(),
        key: vec!["user".into()],
        n,
        seed: Scalar::UInt64(seed),
        columns: vec!["user".into(), "v".into()],
        filters: vec![Filter::NotNull {
            column: "user".into(),
        }],
        key_dtype: None,
    }
}

#[test]
fn a_sample_is_the_same_rows_whatever_order_the_provider_scans_in() {
    let drawn = |rows: RecordBatch| -> Vec<Option<String>> {
        let ran = run_with(
            host(&[], Some(limits(1_000, 1 << 30))),
            sample(3, 7),
            None,
            scan_of(rows),
        );
        text(&ran.answer.unwrap(), "v")
    };
    let forward = drawn(table());
    let reversed: Vec<u32> = (0..table().num_rows() as u32).rev().collect();
    let at = arrow_array::UInt32Array::from(reversed);
    let backward = arrow_select::take::take_record_batch(&table(), &at).unwrap();
    assert_eq!(forward.len(), 3);
    assert_eq!(forward, drawn(backward));
    // Another seed, another draw of the six rows holding a key.
    assert_ne!(
        forward,
        text(&fallback(sample(3, 8), None).answer.unwrap(), "v")
    );
    assert_eq!(
        text(&fallback(sample(10, 7), None).answer.unwrap(), "v").len(),
        6
    );
}

#[test]
fn rows_past_the_limit_fail_by_name_and_no_scan_moves_more() {
    let ran = run_with(
        host(&[], Some(limits(4, 1 << 30))),
        DataRequest::RepeatedKey {
            table: "events".into(),
            key: vec!["user".into()],
        },
        None,
        scan_of(table()),
    );
    let error = refusal(&ran);
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
    assert_eq!(field(error, "request"), Some(&Detail::text("repeated_key")));
    assert_eq!(field(error, "rule"), Some(&Detail::text("fallback_limit")));
    assert_eq!(field(error, "limit"), Some(&Detail::text("max_rows")));
    assert_eq!(
        field(error, "most"),
        Some(&Detail::Scalar(Scalar::UInt64(4)))
    );
    let [DataRequest::Scan { limit, .. }] = ran.asked.as_slice() else {
        panic!("{:?}", ran.asked)
    };
    assert_eq!(*limit, Some(5));
    // At the limit, it answers.
    let ran = run_with(
        host(&[], Some(limits(7, 1 << 30))),
        DataRequest::RepeatedKey {
            table: "events".into(),
            key: vec!["user".into()],
        },
        None,
        scan_of(table()),
    );
    assert!(ran.answer.is_ok());
}

#[test]
fn bytes_past_the_limit_fail_by_name() {
    let ran = run_with(
        host(&[], Some(limits(1_000, 64))),
        DataRequest::Preview {
            table: "events".into(),
            columns: vec!["v".into()],
            limit: 5,
        },
        None,
        scan_of(table()),
    );
    let error = refusal(&ran);
    assert_eq!(field(error, "limit"), Some(&Detail::text("max_bytes")));
    assert_eq!(
        field(error, "most"),
        Some(&Detail::Scalar(Scalar::UInt64(64)))
    );
}

#[test]
fn a_time_the_fallback_does_not_read_fails_by_name() {
    let request = DataRequest::TimeBounds {
        columns: vec![TableColumn {
            table: "events".into(),
            column: "day".into(),
        }],
    };
    let ran = fallback(request, None);
    let error = refusal(&ran);
    assert_eq!(error.code, ErrorCode::UnsupportedRequest);
    assert_eq!(field(error, "rule"), Some(&Detail::text("fallback")));
    assert_eq!(field(error, "field"), Some(&Detail::text("columns")));
}

#[test]
fn a_scan_the_provider_fails_is_the_requests_failure() {
    let ran = run_with(
        host(&[], Some(limits(1_000, 1 << 30))),
        DataRequest::Preview {
            table: "events".into(),
            columns: vec!["v".into()],
            limit: 5,
        },
        None,
        |_| {
            FetchResult::Error(ProviderError {
                driver_message: None,
                error_class: None,
                driver_message: None,
                kind: ErrorKind::DataSource,
                retryable: true,
                sqlstate: Some("08006".into()),
                vendor_code: None,
            })
        },
    );
    let error = refusal(&ran);
    assert_eq!(error.code, ErrorCode::ProviderError);
    assert!(error.retryable);
    assert_eq!(field(error, "sqlstate"), Some(&Detail::text("08006")));
}

#[test]
fn a_declared_request_is_the_providers_and_none_undeclared_without_limits() {
    // Declared: planned for the provider, never the fallback's scans.
    let ran = run_with(
        host(&[OptionalRequest::Preview], Some(limits(1_000, 1 << 30))),
        DataRequest::Preview {
            table: "events".into(),
            columns: vec!["v".into()],
            limit: 2,
        },
        None,
        |_| FetchResult::Data(ArrowData::new(Arc::new(Schema::empty()), vec![])),
    );
    assert!(matches!(
        ran.asked.as_slice(),
        [DataRequest::Preview { .. }]
    ));
    // Undeclared with no limits: refused, as before.
    let ran = run_with(
        host(&[], None),
        DataRequest::Preview {
            table: "events".into(),
            columns: vec!["v".into()],
            limit: 2,
        },
        None,
        scan_of(table()),
    );
    let error = refusal(&ran);
    assert_eq!(error.code, ErrorCode::UnsupportedRequest);
    assert_eq!(field(error, "rule"), None);
    assert!(ran.asked.is_empty());
}
