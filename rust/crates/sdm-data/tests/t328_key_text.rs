// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Key sets bound by each dialect (T3.28, design S4): `DataLayer::bind`
//! writes a Databricks or Postgres key set as the JSON text its statement
//! reads, one text cell per fetch, split past the dialect's budget of
//! 900,000 bytes as the old client splits it (`_chunk_rows`, at
//! `DB/sampler.py:32-44` and `PG/sampler.py:125-182`), each chunk's batch
//! offset in its SQL where the text does not carry it; DuckDB and SQLite
//! keep binding the core's Arrow key set as a relation.
//!
//! The JSON is the old client's spelling per dialect: Databricks's a JSON
//! array of each key (one column) or of an object per seed keyed by column
//! name (several), the batch left out, ASCII only (`json.dumps`' default);
//! Postgres's an array of rows, each an array of its cells' texts, the
//! batch included, as the core route has bound it since T2.3 (its
//! characters as they are).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, Decimal128Array, Float64Array, Int64Array, RecordBatch,
    StringArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::data::{
    AttachmentId, BoundFetch, BoundRequest, DataLayer, DataRequest, FetchPlan, KeyData,
    ProviderChoice, SqlParam,
};
use sdm_data::Data;

/// The budget of one key set's text, in bytes, on Databricks and Postgres.
const BUDGET: usize = 900_000;

fn choice(dialect: &str) -> ProviderChoice {
    serde_json::from_str(&format!(r#"{{"provider":"sql","dialect":"{dialect}"}}"#)).unwrap()
}

/// A lookup of `users` against seeds of a key of `key_type` (as the
/// contract writes a type) and the batch, on a relation of `parts`.
fn lookup(key_type: &str, parts: &str) -> DataRequest {
    serde_json::from_str(&format!(
        r#"{{"columns":["user_id","age"],"key":["user_id"],"request":"lookup","seeds":{{"attachment":2,"columns":[{{"name":"key","nullable":true,"type":{key_type}}},{{"name":"batch","nullable":false,"type":{{"type":"int64"}}}}]}},"table":{{"source":[{parts}]}}}}"#
    ))
    .unwrap()
}

/// A neighbor read of `orders` against seeds of an `int64` key, the batch,
/// and the window's two bounds, zone-less nanosecond times.
fn neighbors(parts: &str) -> DataRequest {
    serde_json::from_str(&format!(
        r#"{{"columns":["ts"],"key":["user_id"],"order":null,"per_seed":null,"request":"neighbors","seeds":{{"attachment":2,"columns":[{{"name":"key","nullable":true,"type":{{"type":"int64"}}}},{{"name":"batch","nullable":false,"type":{{"type":"int64"}}}},{{"name":"end","nullable":false,"type":{{"type":"timestamp","unit":"ns","zone":null}}}},{{"name":"start","nullable":false,"type":{{"type":"timestamp","unit":"ns","zone":null}}}}]}},"table":{{"source":[{parts}]}},"window":{{"column":"ts","lower":{{"inclusive":false,"seed_column":"start"}},"upper":{{"inclusive":true,"seed_column":"end"}}}}}}"#
    ))
    .unwrap()
}

const DATABRICKS_USERS: &str = r#""main","shop","users""#;
const DATABRICKS_ORDERS: &str = r#""main","shop","orders""#;

/// Seeds of `keys` and their batch positions, from 0.
fn seeds(keys: ArrayRef) -> KeyData {
    let rows = keys.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", keys.data_type().clone(), true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![keys, Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )
    .unwrap();
    KeyData {
        schema,
        batches: vec![batch],
    }
}

/// `request` planned and bound on `dialect` with `keys` as its seeds.
fn bind_on(dialect: &str, request: &DataRequest, keys: KeyData) -> (FetchPlan, BoundRequest) {
    let data = Data::new();
    let provider = choice(dialect);
    let plan = data.plan(request, &provider).unwrap();
    let mut handed = BTreeMap::new();
    handed.insert(AttachmentId(2), keys);
    let bound = data
        .bind(&provider, request, plan.clone(), &handed, reference_float)
        .unwrap();
    (plan, bound)
}

/// The text the old client's language gives each double these tests bind
/// (its `repr`), the oracle, a table of the values used; the core passes in
/// the parser crate's own (`sdm_pql::float_repr`).
fn reference_float(x: f64) -> String {
    const WRITTEN: &[(f64, &str)] = &[
        (1.5, "1.5"),
        (0.1, "0.1"),
        (-2.0, "-2.0"),
        (2.5, "2.5"),
        (-0.05, "-0.05"),
        (1e16, "1e+16"),
        (1e-5, "1e-05"),
        (123456789.0, "123456789.0"),
    ];
    WRITTEN
        .iter()
        .find(|(value, _)| value.to_bits() == x.to_bits())
        .map_or_else(
            || panic!("no reference text for {x:?}"),
            |(_, text)| (*text).to_owned(),
        )
}

/// The statements of a SQL plan.
fn statements(plan: &FetchPlan) -> &[sdm_contract::data::SqlStatement] {
    match plan {
        FetchPlan::Sql { statements, .. } => statements,
        FetchPlan::Host { .. } => panic!("not a SQL plan"),
    }
}

/// The one text a fetch binds: its last statement's one parameter, a
/// text parameter, and the one cell of the batch its local reference
/// names, a one-row, one-column text batch.
fn text(fetch: &BoundFetch) -> String {
    let last = statements(&fetch.plan).last().unwrap();
    let [SqlParam::Text { attachment }] = &last.params[..] else {
        panic!("not one text parameter: {:?}", last.params);
    };
    let data = &fetch.batches[attachment];
    assert_eq!(data.schema.fields().len(), 1);
    assert_eq!(data.schema.field(0).data_type(), &DataType::Utf8);
    let cells: Vec<String> = data
        .batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_string::<i32>();
            (0..column.len())
                .map(|row| column.value(row).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    let [cell] = &cells[..] else {
        panic!("not one cell: {} cells", cells.len());
    };
    assert_eq!(fetch.batches.len(), 1, "a fetch binds one text");
    cell.clone()
}

/// The texts of every fetch, in order.
fn texts(bound: &BoundRequest) -> Vec<String> {
    bound.fetches.iter().map(text).collect()
}

/// The old client's chunking (`_chunk_rows`), transcribed as the oracle
/// over each row's JSON text: each row its text and one separator, the
/// brackets two, never an empty chunk unless there is no row.
fn reference_chunks(rows: &[String], budget: usize) -> Vec<Vec<String>> {
    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut size = 2;
    for row in rows {
        let more = row.len() + 1;
        if !current.is_empty() && size + more > budget {
            chunks.push(std::mem::take(&mut current));
            size = 2;
        }
        current.push(row.clone());
        size += more;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(Vec::new());
    }
    chunks
}

fn joined(rows: &[String]) -> String {
    format!("[{}]", rows.join(","))
}

// --- Databricks ------------------------------------------------------------------

#[test]
fn a_databricks_key_set_is_one_text_of_the_old_client_s_json() {
    let request = lookup(r#"{"type":"int64"}"#, DATABRICKS_USERS);
    let keys = seeds(Arc::new(Int64Array::from(vec![Some(3), None, Some(-5)])));
    let (plan, bound) = bind_on("databricks", &request, keys);
    let [fetch] = &bound.fetches[..] else {
        panic!("one fetch: {}", bound.fetches.len());
    };
    // The batch left out: its position in the array is the seed's batch.
    assert_eq!(text(fetch), "[3,null,-5]");
    // The statements are the plan's, the key set's parameter now the text.
    let before: Vec<&str> = statements(&plan).iter().map(|s| s.text.as_str()).collect();
    let after: Vec<&str> = statements(&fetch.plan)
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert_eq!(before, after);
    assert!(
        statements(&fetch.plan)
            .iter()
            .flat_map(|s| &s.params)
            .all(|p| !matches!(p, SqlParam::KeySet { .. })),
        "no key set travels as Arrow on Databricks"
    );
}

#[test]
fn a_databricks_text_key_is_written_ascii_only() {
    let request = lookup(r#"{"type":"utf8"}"#, DATABRICKS_USERS);
    let keys = seeds(Arc::new(StringArray::from(vec![
        "é",
        "a\"b\\c",
        "x\ny\t\u{1}",
        "\u{1F600}",
        "\u{7f}",
    ])));
    let (_, bound) = bind_on("databricks", &request, keys);
    let [only] = &texts(&bound)[..] else {
        panic!("one fetch")
    };
    assert_eq!(
        only,
        r#"["\u00e9","a\"b\\c","x\ny\t\u0001","\ud83d\ude00","\u007f"]"#
    );
    assert!(only.is_ascii());
}

#[test]
fn a_databricks_float_or_decimal_key_is_a_double_and_a_whole_decimal_an_integer() {
    let request = lookup(r#"{"type":"float64"}"#, DATABRICKS_USERS);
    let keys = seeds(Arc::new(Float64Array::from(vec![
        1.5,
        0.1,
        -2.0,
        1e16,
        1e-5,
        f64::NAN,
        f64::NEG_INFINITY,
    ])));
    let (_, bound) = bind_on("databricks", &request, keys);
    // Each double as the old client's language writes it (`repr`), a NaN
    // and an infinity as its JSON does.
    assert_eq!(texts(&bound), ["[1.5,0.1,-2.0,1e+16,1e-05,NaN,-Infinity]"]);

    let request = lookup(
        r#"{"type":"decimal128","precision":10,"scale":2}"#,
        DATABRICKS_USERS,
    );
    let decimals = Decimal128Array::from(vec![150, -5])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let (_, bound) = bind_on("databricks", &request, seeds(Arc::new(decimals)));
    assert_eq!(texts(&bound), ["[1.5,-0.05]"]);

    let request = lookup(
        r#"{"type":"decimal128","precision":38,"scale":0}"#,
        DATABRICKS_USERS,
    );
    let whole = Decimal128Array::from(vec![12_345_678_901_i128])
        .with_precision_and_scale(38, 0)
        .unwrap();
    let (_, bound) = bind_on("databricks", &request, seeds(Arc::new(whole)));
    assert_eq!(texts(&bound), ["[12345678901]"]);
}

#[test]
fn a_databricks_windowed_key_set_is_an_object_per_seed_its_times_to_the_second() {
    let request = neighbors(DATABRICKS_ORDERS);
    let time = DataType::Timestamp(TimeUnit::Nanosecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
        Field::new("end", time.clone(), false),
        Field::new("start", time, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7, 8])),
            Arc::new(Int64Array::from(vec![0, 1])),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_709_251_200_500_000_000,
                0,
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_709_164_800_000_000_001,
                -1,
            ])),
        ],
    )
    .unwrap();
    let (_, bound) = bind_on(
        "databricks",
        &request,
        KeyData {
            schema,
            batches: vec![batch],
        },
    );
    assert_eq!(
        texts(&bound),
        [concat!(
            r#"[{"id":7,"e":"2024-03-01 00:00:00","s":"2024-02-29 00:00:00"},"#,
            r#"{"id":8,"e":"1970-01-01 00:00:00","s":"1969-12-31 23:59:59"}]"#
        )]
    );
}

#[test]
fn an_empty_databricks_key_set_is_one_fetch_of_an_empty_array() {
    let request = lookup(r#"{"type":"int64"}"#, DATABRICKS_USERS);
    let (_, bound) = bind_on(
        "databricks",
        &request,
        seeds(Arc::new(Int64Array::from(Vec::<i64>::new()))),
    );
    assert_eq!(texts(&bound), ["[]"]);
}

/// Many seeds of six-digit keys, past the budget.
fn many_keys(count: i64) -> Int64Array {
    Int64Array::from_iter_values((0..count).map(|i| 100_000 + i % 900_000))
}

#[test]
fn a_databricks_key_set_past_the_budget_is_split_as_the_old_client_splits_it() {
    let request = lookup(r#"{"type":"int64"}"#, DATABRICKS_USERS);
    let keys = many_keys(300_000);
    let rows: Vec<String> = keys.values().iter().map(i64::to_string).collect();
    let expected = reference_chunks(&rows, BUDGET);
    assert!(expected.len() >= 3, "{}", expected.len());
    let (plan, bound) = bind_on("databricks", &request, seeds(Arc::new(keys)));
    let sent = texts(&bound);
    assert_eq!(
        sent,
        expected.iter().map(|c| joined(c)).collect::<Vec<_>>(),
        "the old client's chunks, byte for byte"
    );
    assert!(sent.iter().all(|t| t.len() <= BUDGET));
    // Each chunk's batches continue from the one before: the first reads
    // the array's positions as they are, each later one moved by the
    // seeds before it.
    let original = &statements(&plan).last().unwrap().text;
    let mut offset = 0;
    for (fetch, chunk) in bound.fetches.iter().zip(&expected) {
        let last = &statements(&fetch.plan).last().unwrap().text;
        if offset == 0 {
            assert_eq!(last, original);
        } else {
            assert!(
                last.contains(&format!("pos + {offset} AS `batch`")),
                "chunk at {offset}: {last}"
            );
            assert_eq!(
                last.replacen(&format!("pos + {offset} AS"), "pos AS", 1),
                *original
            );
        }
        offset += chunk.len();
    }
}

#[test]
fn one_databricks_key_past_the_budget_alone_is_a_chunk_of_its_own() {
    let request = lookup(r#"{"type":"utf8"}"#, DATABRICKS_USERS);
    let huge = "k".repeat(BUDGET + 10);
    let keys = seeds(Arc::new(StringArray::from(vec!["a", huge.as_str(), "b"])));
    let (_, bound) = bind_on("databricks", &request, keys);
    assert_eq!(
        texts(&bound),
        [
            r#"["a"]"#.to_owned(),
            format!(r#"["{huge}"]"#),
            r#"["b"]"#.to_owned()
        ]
    );
    let offsets: Vec<bool> = bound
        .fetches
        .iter()
        .map(|f| {
            statements(&f.plan)
                .last()
                .unwrap()
                .text
                .contains("pos + 2 AS")
        })
        .collect();
    assert_eq!(offsets, [false, false, true]);
}

// --- Postgres --------------------------------------------------------------------

#[test]
fn a_postgres_key_set_is_one_text_of_its_rows_cells_its_batch_carried() {
    let request = lookup(r#"{"type":"int64"}"#, r#""users""#);
    let keys = seeds(Arc::new(Int64Array::from(vec![Some(3), None])));
    let (plan, bound) = bind_on("postgres", &request, keys);
    // The old client's: each key alone, the batch its position.
    assert_eq!(texts(&bound), ["[3,null]"]);
    let [fetch] = &bound.fetches[..] else {
        panic!("one fetch")
    };
    let before: Vec<&str> = statements(&plan).iter().map(|s| s.text.as_str()).collect();
    let after: Vec<&str> = statements(&fetch.plan)
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn a_postgres_cell_is_written_as_its_text_its_characters_as_they_are() {
    let request = lookup(r#"{"type":"utf8"}"#, r#""users""#);
    let keys = seeds(Arc::new(StringArray::from(vec!["é\"\n", "\u{7f}"])));
    let (_, bound) = bind_on("postgres", &request, keys);
    assert_eq!(texts(&bound), [r#"["\u00e9\"\n","\u007f"]"#]);

    let request = lookup(
        r#"{"type":"decimal128","precision":10,"scale":2}"#,
        r#""users""#,
    );
    let decimals = Decimal128Array::from(vec![150, -5])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let (_, bound) = bind_on("postgres", &request, seeds(Arc::new(decimals)));
    // As the old client writes a decimal of a scale: its double's text.
    assert_eq!(texts(&bound), [r#"["1.5","-0.05"]"#]);

    let request = lookup(r#"{"type":"binary"}"#, r#""users""#);
    let binary = BinaryArray::from(vec![&[1_u8, 0xab][..]]);
    let (_, bound) = bind_on("postgres", &request, seeds(Arc::new(binary)));
    assert_eq!(texts(&bound), [r#"["\\x01ab"]"#]);

    let request = lookup(r#"{"type":"float64"}"#, r#""users""#);
    let floats = Float64Array::from(vec![2.5, 0.1]);
    let (_, bound) = bind_on("postgres", &request, seeds(Arc::new(floats)));
    assert_eq!(texts(&bound), ["[2.5,0.1]"]);
}

#[test]
fn a_postgres_seed_time_is_written_as_the_old_client_writes_it() {
    let request = neighbors(r#""orders""#);
    let time = DataType::Timestamp(TimeUnit::Nanosecond, None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("batch", DataType::Int64, false),
        Field::new("end", time.clone(), false),
        Field::new("start", time, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_709_251_200_500_000_000,
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                1_709_164_800_000_000_000,
            ])),
        ],
    )
    .unwrap();
    let (_, bound) = bind_on(
        "postgres",
        &request,
        KeyData {
            schema,
            batches: vec![batch],
        },
    );
    assert_eq!(
        texts(&bound),
        [r#"[{"id":7,"e":"2024-03-01T00:00:00.500000","s":"2024-02-29T00:00:00"}]"#]
    );
}

#[test]
fn a_postgres_key_set_past_the_budget_is_split_as_the_old_client_splits_it() {
    let request = lookup(r#"{"type":"int64"}"#, r#""users""#);
    let keys = many_keys(300_000);
    let rows: Vec<String> = keys.values().iter().map(i64::to_string).collect();
    let expected = reference_chunks(&rows, BUDGET);
    assert!(expected.len() >= 2, "{}", expected.len());
    let (plan, bound) = bind_on("postgres", &request, seeds(Arc::new(keys)));
    assert_eq!(
        texts(&bound),
        expected.iter().map(|c| joined(c)).collect::<Vec<_>>()
    );
    // Each later chunk numbers its seeds from the seeds before it.
    let original = &statements(&plan).last().unwrap().text;
    let mut offset = 0;
    for (fetch, chunk) in bound.fetches.iter().zip(&expected) {
        let last = &statements(&fetch.plan).last().unwrap().text;
        if offset == 0 {
            assert_eq!(last, original);
        } else {
            assert_eq!(
                *last,
                original.replacen(
                    "(r.ordinality - 1)::bigint",
                    &format!("(r.ordinality - 1 + {offset})::bigint"),
                    1
                )
            );
        }
        offset += chunk.len();
    }
}

#[test]
fn a_postgres_key_of_several_columns_is_an_array_of_its_values() {
    let request: DataRequest = serde_json::from_str(
        r#"{"columns":["v"],"key":["a","b"],"request":"lookup","seeds":{"attachment":2,"columns":[{"name":"key_1","nullable":true,"type":{"type":"int64"}},{"name":"key_2","nullable":true,"type":{"type":"utf8"}},{"name":"batch","nullable":false,"type":{"type":"int64"}}]},"table":{"source":["pairs"]}}"#,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key_1", DataType::Int64, true),
        Field::new("key_2", DataType::Utf8, true),
        Field::new("batch", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["x", "é"])),
            Arc::new(Int64Array::from(vec![0, 1])),
        ],
    )
    .unwrap();
    let (_, bound) = bind_on(
        "postgres",
        &request,
        KeyData {
            schema,
            batches: vec![batch],
        },
    );
    assert_eq!(texts(&bound), [r#"[[1,"x"],[2,"\u00e9"]]"#]);
}

#[test]
fn a_key_set_whose_batches_are_not_its_positions_is_a_named_defect() {
    // The JSON numbers each seed by its position, as the old client's does.
    for dialect in ["postgres", "databricks"] {
        let parts = if dialect == "postgres" {
            r#""users""#
        } else {
            DATABRICKS_USERS
        };
        let request = lookup(r#"{"type":"int64"}"#, parts);
        let mut keys = seeds(Arc::new(Int64Array::from(vec![1, 2])));
        keys.batches = vec![
            RecordBatch::try_new(
                Arc::clone(&keys.schema),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2])),
                    Arc::new(Int64Array::from(vec![0, 0])),
                ],
            )
            .unwrap(),
        ];
        let data = Data::new();
        let provider = choice(dialect);
        let plan = data.plan(&request, &provider).unwrap();
        let refused = data
            .bind(
                &provider,
                &request,
                plan,
                &BTreeMap::from([(AttachmentId(2), keys)]),
                reference_float,
            )
            .unwrap_err();
        assert_eq!(
            refused.code,
            sdm_contract::ErrorCode::InternalError,
            "{dialect}"
        );
    }
}

// --- DuckDB and SQLite -----------------------------------------------------------

#[test]
fn duckdb_and_sqlite_bind_the_core_s_arrow_key_set_as_it_is() {
    for dialect in ["duckdb", "sqlite"] {
        let request = lookup(r#"{"type":"int64"}"#, r#""users""#);
        let keys = seeds(Arc::new(Int64Array::from(vec![1, 2])));
        let (plan, bound) = bind_on(dialect, &request, keys);
        let [fetch] = &bound.fetches[..] else {
            panic!("{dialect}: one fetch")
        };
        assert_eq!(fetch.plan, plan, "{dialect}");
        assert!(fetch.batches.is_empty(), "{dialect}");
    }
}

#[test]
fn a_request_binding_no_key_set_is_its_plan() {
    let request: DataRequest = serde_json::from_str(
        r#"{"request":"preview","table":{"source":["main","shop","users"]},"columns":["user_id"],"limit":5}"#,
    )
    .unwrap();
    let data = Data::new();
    let provider = choice("databricks");
    let plan = data.plan(&request, &provider).unwrap();
    let bound = data
        .bind(
            &provider,
            &request,
            plan.clone(),
            &BTreeMap::new(),
            reference_float,
        )
        .unwrap();
    assert_eq!(
        bound,
        BoundRequest {
            fetches: vec![BoundFetch {
                plan,
                batches: BTreeMap::new()
            }]
        }
    );
}
