// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L1 vectors for the core's two `neighbors` reads on DuckDB (T1.26b,
//! register ORD-2, decision F2): `neighbors.json`, recorded from the
//! reference's SQL sampler (`_by_fkey`, `_by_time`) by
//! `neighbors_record.py` (a protected path, never recorded again to pass a
//! dependency bump, F4), holds the rows each read returned for its seeds;
//! `neighbor-requests.json` holds the requests the core makes for the same
//! reads, which `sdm-engine`'s `neighbor_requests.rs` holds
//! `sdm_engine::sample::by_fkey` and `by_time` to. Each request is planned
//! by the root's data layer for DuckDB, its seeds bound as the provider
//! binds a key set, and the answer the core admitted compared with the
//! reference's rows.
//!
//! What the vectors record, the SQL sampler's policy: the latest rows up to
//! each seed's anchor, the anchor itself included, newest first, ties by
//! the key, at most `per_seed` of each seed, in `(seed, order)`; a window
//! after its start and up to its end, both to the microsecond and before
//! 1970 alike; a row whose time is null never returned; a row reached by
//! several seeds returned for each. A `TIMESTAMPTZ` column is compared in
//! the wall clock of the database's own zone (`SET GLOBAL TimeZone`, the
//! machine's zone on the reference's host), not the source session's
//! (`SET TimeZone='America/New_York'`), which no read's cursor carries, as
//! the reference's cursors do not (register DT-13): the vectors' zoned
//! case puts a row on either side of its anchor by which of the two is
//! read, and the core reads it as the reference does. [`check`] fails
//! closed: a perturbed row, and evidence missing, empty or retyped, each
//! fail (seeded below).

use std::collections::BTreeMap;
use std::sync::Arc;

use duckdb::Connection;
use duckdb::arrow::array::{Array, ArrayRef, Float64Array, Int64Array, TimestampNanosecondArray};
use duckdb::arrow::compute::{cast, concat_batches};
use duckdb::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use duckdb::arrow::record_batch::RecordBatch;
use sdm_contract::data::{AttachmentId, DataRequest, Dialect, FetchPlan, ProviderChoice, SqlParam};
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::{FetchOutcome, FetchReply, StartRun};
use sdm_contract::test_ops::{TestFetch, TestOperation, TestResult};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{ArrowData, Attachment, HostRun, default_root, encode, read};
use sdm_testkit::{duck, host};
use serde_json::{Value, json};

const VECTORS: &str = include_str!("neighbors.json");
const REQUESTS: &str = include_str!("neighbor-requests.json");

/// A database holding the vectors' tables, with the zones they set.
fn database(vectors: &Value) -> Result<Connection, String> {
    let connection = duck::open().map_err(|e| e.to_string())?;
    let statements = vectors["setup"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(vectors["zones"].as_array().into_iter().flatten());
    for statement in statements {
        let text = statement.as_str().ok_or("a setup statement is not text")?;
        connection.execute_batch(text).map_err(|e| e.to_string())?;
    }
    Ok(connection)
}

fn ns_column(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(TimestampNanosecondArray::from(values))
}

/// The seeds of `case`: each key, its position, and its anchor (`time`),
/// or its window's start and end, as the core's seeds carry them.
fn seeds(case: &Value) -> Result<RecordBatch, String> {
    let rows = case["seeds"]
        .as_array()
        .filter(|s| !s.is_empty())
        .ok_or("no seeds")?;
    let field = |row: &Value, at: usize| -> Result<Option<i64>, String> {
        match &row[at] {
            Value::Null => Ok(None),
            Value::Number(n) => n
                .as_i64()
                .map(Some)
                .ok_or_else(|| "a seed is not an integer".into()),
            _ => Err("a seed is not an integer".into()),
        }
    };
    let keys = rows
        .iter()
        .map(|r| field(r, 0)?.ok_or_else(|| "a seed's key is missing".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut fields = vec![
        Field::new("key", DataType::Int64, false),
        Field::new("batch", DataType::Int64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(keys)),
        Arc::new(Int64Array::from_iter_values(0..rows.len() as i64)),
    ];
    let timestamp = DataType::Timestamp(TimeUnit::Nanosecond, None);
    let times = |at: usize| -> Result<Vec<Option<i64>>, String> {
        rows.iter().map(|r| field(r, at)).collect()
    };
    match case["form"].as_str() {
        Some("fkey") => {
            fields.push(Field::new("time", timestamp, false));
            columns.push(ns_column(times(1)?));
        }
        Some("time") => {
            let start = times(1)?;
            if start.iter().any(Option::is_some) {
                fields.push(Field::new("start", timestamp.clone(), false));
                columns.push(ns_column(start));
            }
            fields.push(Field::new("end", timestamp, false));
            columns.push(ns_column(times(2)?));
        }
        _ => return Err("a case's form is neither fkey nor time".into()),
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| e.to_string())
}

/// The request `case` makes, from `neighbor-requests.json`.
fn request(case: &Value, requests: &Value, seeds: &RecordBatch) -> Result<DataRequest, String> {
    let table = case["table"].as_str().ok_or("a case names no table")?;
    let name = match case["form"].as_str() {
        Some("fkey") => format!("fkey/{table}"),
        _ if seeds.schema().field_with_name("start").is_err() => format!("time/{table}/no-start"),
        _ => format!("time/{table}"),
    };
    let mut request = requests["requests"][&name].clone();
    if request.is_null() {
        return Err(format!("no request for {name}"));
    }
    if case["form"] == "fkey" {
        request["per_seed"] = case["per_seed"].clone();
    }
    request["seeds"]["columns"] = serde_json::to_value(
        seeds
            .schema()
            .fields()
            .iter()
            .map(|f| sdm_contract::Field {
                name: f.name().clone(),
                column_type: sdm_contract::ColumnType::from_arrow(f.name(), f.data_type()).unwrap(),
                nullable: f.is_nullable(),
                metadata: Default::default(),
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    serde_json::from_value(request).map_err(|e| e.to_string())
}

/// The answer the core admitted to `request` with `seeds`, through
/// `HostRun`, the DuckDB provider binding the seeds as it binds any key
/// set.
fn answered(connection: &Connection, request: DataRequest, seeds: RecordBatch) -> RecordBatch {
    let fetch = TestFetch {
        request,
        plan: None,
        provider: Some(ProviderChoice::Sql {
            dialect: Dialect::new("duckdb"),
        }),
        echo: true,
    };
    let message = encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: host::facts().unwrap(),
        operation: Operation::Test(TestOperation::Keyed {
            fetch: AttachmentId(1),
            keys: AttachmentId(2),
        }),
    })))
    .unwrap();
    let run = HostRun::start(
        &default_root(),
        &message,
        BTreeMap::from([
            (
                AttachmentId(1),
                Attachment::Bytes(serde_json::to_vec(&fetch).unwrap()),
            ),
            (
                AttachmentId(2),
                Attachment::Arrow(ArrowData::new(seeds.schema(), vec![seeds])),
            ),
        ]),
    )
    .map_err(|out| read(&out).unwrap())
    .unwrap();
    let ControlMessage::Step(Step::Fetch { requests }) = read(&run.next().unwrap()).unwrap() else {
        panic!("not a fetch");
    };
    let step = &requests[0];
    let mut sets = BTreeMap::new();
    if let FetchPlan::Sql { statements, .. } = &step.plan {
        for param in statements.iter().flat_map(|s| &s.params) {
            if let SqlParam::KeySet { keys, .. } = param {
                let Ok(Some(Attachment::Arrow(data))) = run.take_attachment(keys.attachment) else {
                    panic!("the seeds are not attached");
                };
                sets.insert(
                    keys.attachment,
                    duck::Fetched {
                        schema: data.schema,
                        batches: data.batches,
                    },
                );
            }
        }
    }
    let fetched = duck::answer_with(connection, step, &sets).unwrap();
    run.reply(
        &encode(&ControlMessage::FetchReply(FetchReply {
            id: step.id,
            outcome: FetchOutcome::Data {
                data: AttachmentId(1),
                continuation: None,
            },
        }))
        .unwrap(),
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Arrow(ArrowData::new(fetched.schema, fetched.batches)),
        )]),
    )
    .map_err(|out| read(&out).unwrap())
    .unwrap();
    let ControlMessage::Step(Step::Done {
        result: OperationResult::Test(TestResult { data: Some(id), .. }),
    }) = read(&run.next().unwrap()).unwrap()
    else {
        panic!("the fetch echoes its data");
    };
    let Ok(Some(Attachment::Arrow(data))) = run.take_attachment(id) else {
        panic!("the echo is Arrow");
    };
    concat_batches(&data.schema, &data.batches).unwrap()
}

/// The answer's rows as the vectors write them: the seed's position, then
/// each column, a time as nanoseconds since the epoch, its instant.
fn rows_of(answer: &RecordBatch, columns: &[&str]) -> Vec<Value> {
    let int = |name: &str| -> Vec<Option<i64>> {
        let column = answer.column_by_name(name).unwrap();
        cast(column, &DataType::Int64)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    let mut read: Vec<Vec<Value>> = vec![Vec::new(); answer.num_rows()];
    for name in std::iter::once("__kumo_batch__").chain(columns.iter().copied()) {
        let column = answer.column_by_name(name).unwrap();
        let values: Vec<Value> = match column.data_type() {
            DataType::Timestamp(_, zone) => {
                let target = DataType::Timestamp(TimeUnit::Nanosecond, zone.clone());
                let times = cast(column, &target).unwrap();
                let times = cast(&times, &DataType::Int64).unwrap();
                let times = times.as_any().downcast_ref::<Int64Array>().unwrap();
                times.iter().map(|v| json!(v)).collect()
            }
            DataType::Float64 => column
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .map(|v| json!(v))
                .collect(),
            _ => int(name).into_iter().map(|v| json!(v)).collect(),
        };
        for (row, value) in values.into_iter().enumerate() {
            read[row].push(value);
        }
    }
    read.into_iter().map(Value::Array).collect()
}

/// The canonical text of each row, sorted: a multiset's order.
fn sorted(rows: &[Value]) -> Vec<String> {
    let mut texts: Vec<String> = rows.iter().map(Value::to_string).collect();
    texts.sort();
    texts
}

/// What a run compared: the cases, and the rows in them.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    cases: usize,
    rows: usize,
}

fn check(
    vectors: &Value,
    found: &dyn Fn(&Value) -> Result<Vec<Value>, String>,
) -> Result<Counts, String> {
    let cases = vectors["cases"]
        .as_array()
        .filter(|c| !c.is_empty())
        .ok_or("no cases recorded")?;
    let mut rows = 0;
    for case in cases {
        let id = case["id"].as_str().ok_or("a case has no id")?;
        let wanted = case["rows"]
            .as_array()
            .ok_or_else(|| format!("{id}: no rows recorded"))?;
        let ordered = case["ordered"]
            .as_bool()
            .ok_or_else(|| format!("{id}: no order recorded"))?;
        let got = found(case).map_err(|e| format!("{id}: {e}"))?;
        let same = if ordered {
            got == *wanted
        } else {
            sorted(&got) == sorted(wanted)
        };
        if !same {
            return Err(format!("{id}: rows otherwise than the reference's"));
        }
        rows += wanted.len();
    }
    Ok(Counts {
        cases: cases.len(),
        rows,
    })
}

/// A seeded fault: what it changes in the vectors.
type Seed = Box<dyn Fn(&mut Value)>;

fn vectors() -> Value {
    serde_json::from_str(VECTORS).unwrap()
}

#[test]
fn both_neighbor_reads_return_the_reference_s_rows_on_duckdb() {
    let vectors = vectors();
    let requests: Value = serde_json::from_str(REQUESTS).unwrap();
    let connection = database(&vectors).unwrap();
    let columns: Vec<&str> = vectors["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    let found = |case: &Value| -> Result<Vec<Value>, String> {
        let seeds = seeds(case)?;
        let request = request(case, &requests, &seeds)?;
        Ok(rows_of(&answered(&connection, request, seeds), &columns))
    };
    assert_eq!(
        check(&vectors, &found).unwrap(),
        Counts {
            cases: 14,
            rows: 73
        }
    );
    // The zoned case's rows are the database zone's reading: the source
    // session's zone, set on its connection, reaches no read.
    let session_free = {
        let mut free = vectors.clone();
        free["zones"] = json!(["SET GLOBAL TimeZone='UTC'"]);
        free
    };
    let other = database(&session_free).unwrap();
    let case = vectors["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "fkey/timestamptz")
        .unwrap();
    let seeds = seeds(case).unwrap();
    let asked = request(case, &requests, &seeds).unwrap();
    assert_eq!(
        rows_of(&answered(&other, asked, seeds), &columns),
        *case["rows"].as_array().unwrap()
    );
}

#[test]
fn the_check_fails_closed() {
    let vectors = vectors();
    // The reference's own rows, as found: every seeded fault must fail.
    let recorded = |case: &Value| -> Result<Vec<Value>, String> {
        Ok(case["rows"].as_array().cloned().unwrap_or_default())
    };
    assert!(check(&vectors, &recorded).is_ok());
    let perturbed = |case: &Value| -> Result<Vec<Value>, String> {
        let mut rows = case["rows"].as_array().cloned().unwrap_or_default();
        if case["id"] == "fkey/ties-kept-by-key" {
            rows.swap(1, 2);
        }
        Ok(rows)
    };
    assert!(
        check(&vectors, &perturbed).is_err(),
        "a reordered tie passed"
    );
    let dropped = |case: &Value| -> Result<Vec<Value>, String> {
        let mut rows = case["rows"].as_array().cloned().unwrap_or_default();
        if case["id"] == "time/window-ends" {
            rows.pop();
        }
        Ok(rows)
    };
    assert!(check(&vectors, &dropped).is_err(), "a dropped row passed");
    let seeded: Vec<(&str, Seed)> = vec![
        ("no cases", Box::new(|v| v["cases"] = json!([]))),
        ("cases missing", Box::new(|v| v["cases"] = Value::Null)),
        (
            "rows retyped",
            Box::new(|v| v["cases"][0]["rows"] = json!("rows")),
        ),
        (
            "order missing",
            Box::new(|v| v["cases"][0]["ordered"] = Value::Null),
        ),
        (
            "id missing",
            Box::new(|v| v["cases"][0]["id"] = Value::Null),
        ),
    ];
    for (fault, seed) in seeded {
        let mut faulty = vectors.clone();
        seed(&mut faulty);
        assert!(check(&faulty, &recorded).is_err(), "{fault} passed");
    }
}
