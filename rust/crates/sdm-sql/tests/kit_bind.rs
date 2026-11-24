// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A built-in source's request as its SQL kit binds it (design S4): a key
//! set written as text past the dialect's budget is asked as one fetch per
//! chunk, each chunk's text a one-cell batch the fetch binds, the answers
//! joined in order and read as one answer; any failed fetch fails the
//! request. A source asking for lossless timestamps reads the result's
//! schema first, with empty key sets, and every chunk keeps the projection
//! that read chose.
//!
//! The driver here is a fake Databricks warehouse: it reads the one text
//! cell each fetch binds, as the old client's statement reads its
//! parameter (`posexplode(from_json(?,...))`), and numbers each seed by its
//! position plus the offset the statement adds.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::data::{AttachmentId, DataRequest, KeyData, ProviderError};
use sdm_contract::{ErrorCode, ErrorKind};
use sdm_sql::kit::{Answered, KitError, KitFetch, SqlKit};
use sdm_sql::plan::{SqlChoice, SqlParam, SqlPlan, SqlResultConversion, SqlTimestampKind};

/// The budget of one key set's text on Databricks, in bytes.
const BUDGET: usize = 900_000;

/// The id the request names its seeds by.
const SEEDS: AttachmentId = AttachmentId(2);

const USERS: &[&str] = &["main", "shop", "users"];

/// Seeds of `keys`, batches from 0, as the core made them.
fn seeds(keys: Vec<i64>) -> BTreeMap<AttachmentId, KeyData> {
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
    BTreeMap::from([(
        SEEDS,
        KeyData {
            schema,
            batches: vec![batch],
        },
    )])
}

/// A lookup of `columns` in `source` by `user_id`, against the seeds.
fn lookup(source: &[&str], columns: &[&str]) -> DataRequest {
    serde_json::from_str(&format!(
        r#"{{"columns":{},"key":["user_id"],"request":"lookup","seeds":{{"attachment":{},"columns":[{{"name":"key","nullable":true,"type":{{"type":"int64"}}}},{{"name":"batch","nullable":false,"type":{{"type":"int64"}}}}]}},"table":{{"source":{}}}}}"#,
        serde_json::to_string(columns).unwrap(),
        SEEDS.0,
        serde_json::to_string(source).unwrap()
    ))
    .unwrap()
}

fn databricks() -> SqlChoice {
    SqlChoice::new("databricks")
}

/// Databricks, asking for timestamps as microseconds.
fn databricks_micros() -> SqlChoice {
    serde_json::from_str(
        r#"{"provider":"sql","dialect":"databricks","timestamp_encoding":"microseconds"}"#,
    )
    .unwrap()
}

/// Postgres naming its schema.
fn postgres() -> SqlChoice {
    serde_json::from_str(r#"{"provider":"sql","dialect":"postgres","namespace":["public"]}"#)
        .unwrap()
}

fn float_text(x: f64) -> String {
    format!("{x:?}")
}

/// Keys whose text is past the budget: six-digit keys, about 2.1 MB.
fn many() -> Vec<i64> {
    (0..300_000).map(|i| 100_000 + i).collect()
}

/// The text batch the last statement of `plan` binds.
fn text_param(plan: &SqlPlan) -> AttachmentId {
    let params = &plan.statements.last().unwrap().params;
    let [SqlParam::Text { attachment }] = &params[..] else {
        panic!("not one text parameter: {params:?}")
    };
    *attachment
}

/// The one text cell the fetch binds.
fn cell(fetch: &KitFetch<'_>) -> String {
    let data = &fetch.keys[&text_param(fetch.plan)];
    let rows: usize = data.batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 1, "one row");
    assert_eq!(data.schema.fields().len(), 1, "one column");
    let batch = data.batches.iter().find(|b| b.num_rows() == 1).unwrap();
    batch.column(0).as_string::<i32>().value(0).to_owned()
}

/// The offset a chunk's statement adds to each seed's position: none, or
/// the `CAST(pos AS BIGINT) + <n>` it writes.
fn offset(text: &str) -> usize {
    text.split_once("CAST(pos AS BIGINT) + ")
        .map(|(_, rest)| {
            rest.split_once(' ')
                .map(|(n, _)| n.parse().unwrap())
                .unwrap()
        })
        .unwrap_or(0)
}

/// A fake warehouse's answer to `plan`, its text `payload`: each seed's
/// position plus the statement's offset, and the seed.
fn answer(plan: &SqlPlan, payload: &str) -> Answered {
    let text = &plan.statements.last().unwrap().text;
    assert!(text.contains("CAST(pos AS BIGINT)"), "{text}");
    let moved = offset(text);
    let keys: Vec<i64> = serde_json::from_str(payload).unwrap();
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("__kumo_batch__", DataType::Int64, false),
        Field::new("seed", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..keys.len()).map(|at| i64::try_from(at + moved).unwrap()),
            )),
            Arc::new(Int64Array::from(keys)),
        ],
    )
    .unwrap();
    Answered {
        schema,
        batches: vec![batch],
    }
}

fn denied() -> ProviderError {
    ProviderError {
        error_class: Some("INSUFFICIENT_PERMISSIONS".into()),
        driver_message: None,
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: Some("42501".into()),
        vendor_code: None,
        facts: None,
    }
}

/// An answer of the given columns, each named and typed as given.
fn batch_of(columns: Vec<(&str, ArrayRef)>) -> Answered {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    Answered {
        schema: batch.schema(),
        batches: vec![batch],
    }
}

fn int64s(answered: &Answered, column: usize) -> Vec<i64> {
    answered
        .batches
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
fn a_key_set_past_the_budget_is_asked_as_chunks_and_joined_in_order() {
    let keys = many();
    let kit = SqlKit::new(databricks(), None).unwrap();
    let mut payloads = Vec::new();
    let answered = kit
        .answer(
            &lookup(USERS, &["user_id"]),
            &seeds(keys.clone()),
            float_text,
            &mut |fetch| {
                // The core's own key set is not what the driver is handed.
                assert!(!fetch.keys.contains_key(&SEEDS));
                let payload = cell(&fetch);
                assert!(payload.len() <= BUDGET, "{}", payload.len());
                let answered = answer(fetch.plan, &payload);
                payloads.push(payload);
                Ok(answered)
            },
        )
        .unwrap();
    assert!(
        payloads.len() >= 3,
        "one fetch per chunk: {}",
        payloads.len()
    );
    // The chunks are the seeds in order, each once.
    let sent: Vec<i64> = payloads
        .iter()
        .flat_map(|p| serde_json::from_str::<Vec<i64>>(p).unwrap())
        .collect();
    assert_eq!(sent, keys);
    let expected: Vec<i64> = (0..i64::try_from(keys.len()).unwrap()).collect();
    assert_eq!(
        int64s(&answered, 0),
        expected,
        "each seed at its place in the key set"
    );
    assert_eq!(int64s(&answered, 1), keys);
}

#[test]
fn a_later_chunk_failing_fails_the_request() {
    let kit = SqlKit::new(databricks(), None).unwrap();
    let mut asked = 0;
    let result = kit.answer(
        &lookup(USERS, &["user_id"]),
        &seeds(many()),
        float_text,
        &mut |fetch| {
            asked += 1;
            if asked == 2 {
                return Err(denied());
            }
            Ok(answer(fetch.plan, &cell(&fetch)))
        },
    );
    let Err(KitError::Provider(error)) = result else {
        panic!("the request did not fail: {result:?}")
    };
    assert_eq!(error.sqlstate.as_deref(), Some("42501"));
    // The kit reads what the failure means for the core.
    assert!(error.facts.unwrap().denied);
}

#[test]
fn a_key_set_within_the_budget_is_one_fetch_of_one_text() {
    let kit = SqlKit::new(databricks(), None).unwrap();
    let mut payloads = Vec::new();
    kit.answer(
        &lookup(USERS, &["user_id"]),
        &seeds(vec![5, 6]),
        float_text,
        &mut |fetch| {
            let payload = cell(&fetch);
            let answered = answer(fetch.plan, &payload);
            payloads.push(payload);
            Ok(answered)
        },
    )
    .unwrap();
    assert_eq!(payloads, ["[5,6]"]);
}

// --- one conversion over the whole answer --------------------------------

/// A Postgres answer of `n`, a declared scale-zero `numeric` past Arrow's
/// decimals, as the driver hands it over: each value's exact text, its
/// declared scale marked, after the batch of each seed answered.
fn wide(batches: &[i64], values: &[&str]) -> Answered {
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
            Arc::new(StringArray::from(values.to_vec())),
        ],
    )
    .unwrap();
    Answered {
        schema,
        batches: vec![batch],
    }
}

/// A Postgres lookup past the budget, its chunks answered by `answers`.
fn chunked_postgres(answers: Vec<Answered>) -> Result<Answered, KitError> {
    let kit = SqlKit::new(postgres(), None).unwrap();
    let count = answers.len();
    let mut answers = answers.into_iter();
    let mut asked = 0;
    let result = kit.answer(
        &lookup(&["wide_numbers"], &["user_id"]),
        &seeds(many()),
        float_text,
        &mut |_| {
            asked += 1;
            Ok(answers.next().expect("one answer per chunk"))
        },
    );
    // Every chunk is asked unless one is refused first.
    if result.is_ok() {
        assert_eq!(asked, count, "one fetch per chunk");
    }
    result
}

#[test]
fn chunks_of_one_answer_are_read_as_one_answer_small_wide_and_empty() {
    // A small value in one chunk, a wide one in another, none in a third:
    // the column is read as the whole answer says (each value's exact
    // text), as one fetch of all three would read it.
    let data = chunked_postgres(vec![
        wide(&[0], &["1"]),
        wide(&[1], &["100000000000000000000"]),
        wide(&[], &[]),
    ])
    .unwrap();
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
    let data =
        chunked_postgres(vec![wide(&[0], &["1"]), wide(&[], &[]), wide(&[2], &["7"])]).unwrap();
    assert_eq!(
        data.schema.field_with_name("n").unwrap().data_type(),
        &DataType::Int64
    );
}

#[test]
fn chunks_the_driver_answers_in_two_shapes_are_still_refused() {
    // A source schema that truly differs between chunks is the driver's
    // fault, refused by name.
    let differs = batch_of(vec![
        (
            "__kumo_batch__",
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        ),
        ("m", Arc::new(Int64Array::from(vec![2])) as ArrayRef),
    ]);
    let result = chunked_postgres(vec![wide(&[0], &["1"]), differs, wide(&[], &[])]);
    let Err(KitError::Refused(error)) = result else {
        panic!("a schema that differs was joined: {result:?}")
    };
    assert_eq!(error.code, ErrorCode::ProviderSchemaMismatch, "{error:?}");
}

// --- lossless timestamps ------------------------------------------------------

/// A warehouse's description of a result: each column's name and type.
fn described(names: &[&str], types: &[&str]) -> Answered {
    batch_of(vec![
        (
            "col_name",
            Arc::new(StringArray::from(names.to_vec())) as ArrayRef,
        ),
        (
            "data_type",
            Arc::new(StringArray::from(types.to_vec())) as ArrayRef,
        ),
    ])
}

#[test]
fn timestamp_preflight_binds_empty_keys_then_preserves_the_original_lookup() {
    use arrow_array::TimestampMicrosecondArray;
    let kit = SqlKit::new(databricks_micros(), None).unwrap();
    let mut asked = 0;
    let answered = kit.answer(
        &lookup(USERS, &["user_id", "at"]),
        &seeds(vec![7, 8]),
        float_text,
        &mut |fetch| {
            asked += 1;
            let text = &fetch.plan.statements.last().unwrap().text;
            if asked == 1 {
                // The schema-only read, of no seed.
                assert!(text.starts_with("DESCRIBE QUERY WITH "), "{text}");
                assert_eq!(cell(&fetch), "[]");
                return Ok(described(
                    &["__kumo_batch__", "user_id", "at", "__kumo_matches__"],
                    &["bigint", "bigint", "timestamp_ntz", "bigint"],
                ));
            }
            assert_eq!(
                serde_json::from_str::<Vec<i64>>(&cell(&fetch)).unwrap(),
                vec![7, 8]
            );
            assert_eq!(
                fetch.plan.result_conversions,
                [SqlResultConversion::TimestampMicros {
                    column: 2,
                    name: "at".into(),
                    source_kind: SqlTimestampKind::WallTime,
                }]
            );
            assert!(!text.starts_with("DESCRIBE QUERY"));
            assert!(text.contains("DATEDIFF(CAST(t.`at` AS DATE)"), "{text}");
            Ok(batch_of(vec![
                (
                    "__kumo_batch__",
                    Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
                ),
                (
                    "user_id",
                    Arc::new(Int64Array::from(vec![7, 8])) as ArrayRef,
                ),
                (
                    "at",
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        1_710_037_800_123_456_i64,
                        -1,
                    ])) as ArrayRef,
                ),
                (
                    "__kumo_matches__",
                    Arc::new(Int64Array::from(vec![1, 1])) as ArrayRef,
                ),
            ]))
        },
    );
    assert_eq!(asked, 2);
    assert!(answered.is_ok(), "{answered:?}");
}

#[test]
fn both_timestamp_kinds_survive_first_and_later_key_text_chunks_without_reprojection() {
    let kit = SqlKit::new(databricks_micros(), None).unwrap();
    let mut offsets = Vec::new();
    let mut asked = 0;
    let _ = kit.answer(
        &lookup(USERS, &["user_id", "wall", "instant"]),
        &seeds(many()),
        float_text,
        &mut |fetch| {
            asked += 1;
            if asked == 1 {
                assert_eq!(cell(&fetch), "[]");
                return Ok(described(
                    &[
                        "__kumo_batch__",
                        "user_id",
                        "wall",
                        "instant",
                        "__kumo_matches__",
                    ],
                    &["bigint", "bigint", "timestamp_ntz", "timestamp", "bigint"],
                ));
            }
            assert_eq!(
                serde_json::to_value(&fetch.plan.result_conversions).unwrap(),
                serde_json::json!([
                    {"conversion":"timestamp_micros","column":2,"name":"wall","source_kind":"wall_time"},
                    {"conversion":"timestamp_micros","column":3,"name":"instant","source_kind":"instant"},
                ])
            );
            let text = &fetch.plan.statements.last().unwrap().text;
            assert_eq!(
                text.matches("UNIX_MICROS(t.`instant`)").count(),
                1,
                "{text}"
            );
            assert_eq!(
                text.matches("DATEDIFF(CAST(t.`wall` AS DATE)").count(),
                1,
                "{text}"
            );
            assert!(!text.starts_with("DESCRIBE QUERY"));
            assert!(!text.contains("TIMESTAMPDIFF(MICROSECOND"));
            offsets.push(offset(text));
            assert!(cell(&fetch).len() <= BUDGET);
            // Every chunk alike; the answer itself is not this test's.
            Ok(described(&[], &[]))
        },
    );
    assert!(offsets.len() >= 3, "{offsets:?}");
    assert_eq!(offsets[0], 0);
    assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
}
