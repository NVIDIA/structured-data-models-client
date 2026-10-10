// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust host's DuckDB provider (T1.11b): `base.regression`'s
//! schema completes through `HostRun`, and, by `contract/provider-vectors.json`,
//! which the Python host's provider tests read too, a fetch the driver hands
//! over in several batches and an empty one are admitted as the vectors
//! say, and a `TIMESTAMPTZ` column reads in the database (`GLOBAL`) zone's wall
//! clock, a session zone of the source's connection being seen by no read.
//! Each fetch is planned by the root's data layer for DuckDB and echoed back
//! by the test-only fetch, so what is compared is what the core admitted.

use std::collections::BTreeMap;
use std::path::PathBuf;

use duckdb::Connection;
use duckdb::arrow::array::{Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray};
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::datatypes::DataType;
use duckdb::arrow::record_batch::RecordBatch;
use sdm_contract::data::{AttachmentId, DataRequest, Dialect, ProviderChoice};
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::{FetchOutcome, FetchReply, StartRun};
use sdm_contract::test_ops::{TestFetch, TestOperation, TestResult};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{ArrowData, Attachment, HostRun, default_root, encode, read};
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{bases, case, duck};
use serde_json::Value;

fn contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract")
}

fn vectors() -> Value {
    serde_json::from_slice(&std::fs::read(contract().join("provider-vectors.json")).unwrap())
        .unwrap()
}

/// A database holding the vectors' tables, and the zone `zone` names set.
fn database(zone: Option<&str>) -> Connection {
    let connection = duck::open().unwrap();
    let vectors = vectors();
    for statement in vectors["setup"].as_array().unwrap() {
        connection
            .execute_batch(statement.as_str().unwrap())
            .unwrap();
    }
    if let Some(zone) = zone {
        connection
            .execute_batch(vectors[zone].as_str().unwrap())
            .unwrap();
    }
    connection
}

/// The fetch of `request`, planned by the root's data layer for DuckDB.
fn start(request: &Value) -> HostRun {
    let fetch = TestFetch {
        request: serde_json::from_value::<DataRequest>(request.clone()).unwrap(),
        plan: None,
        provider: Some(ProviderChoice::Sql {
            dialect: Dialect::Duckdb,
        }),
        echo: true,
    };
    let message = encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: host::facts().unwrap(),
        operation: Operation::Test(TestOperation::Fetch {
            fetch: AttachmentId(1),
        }),
    })))
    .unwrap();
    HostRun::start(
        &default_root(),
        &message,
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&fetch).unwrap()),
        )]),
    )
    .map_err(|out| read(&out).unwrap())
    .unwrap()
}

/// The fetch step the run asks, answered with `batches` of `schema`; the
/// data the core admitted, echoed back.
fn admitted(
    run: &HostRun,
    answer: impl FnOnce(&sdm_contract::run::FetchStep) -> duck::Fetched,
) -> ArrowData {
    let ControlMessage::Step(Step::Fetch { requests }) = read(&run.next().unwrap()).unwrap() else {
        panic!("not a fetch");
    };
    let fetched = answer(&requests[0]);
    run.reply(
        &encode(&ControlMessage::FetchReply(FetchReply {
            id: requests[0].id,
            outcome: FetchOutcome::Data {
                data: AttachmentId(1),
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
    data
}

fn rows(data: &ArrowData) -> RecordBatch {
    concat_batches(&data.schema, &data.batches).unwrap()
}

/// A table as the test reads it: its name, each column's name, data and
/// semantic types, its key and its time column.
type Summary = (String, Vec<(String, String, String)>, Value, Value);

#[test]
fn base_regression_s_schema_completes_on_the_pure_rust_host() {
    let loaded = case::load(&[contract().join("cases/s1/base.regression.json")]).unwrap();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&loaded[0].case, &bases).unwrap();
    assert_eq!(observed.operations, ["describe_source"]);
    let Answer::Described(described) = observed.answer else {
        panic!("not described: {:?}", observed.answer);
    };
    // The reference's schema of base.regression: keys and the time column
    // inferred, each column's type and meaning.
    let tables = described["tables"].as_array().unwrap();
    let summary: Vec<Summary> = tables
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_owned(),
                t["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        (
                            c["name"].as_str().unwrap().to_owned(),
                            c["dtype"].as_str().unwrap().to_owned(),
                            c["stype"].as_str().unwrap().to_owned(),
                        )
                    })
                    .collect(),
                t["primary_key"].clone(),
                t["time_column"].clone(),
            )
        })
        .collect();
    let column = |n: &str, d: &str, s: &str| (n.to_owned(), d.to_owned(), s.to_owned());
    assert_eq!(
        summary,
        [
            (
                "orders".to_owned(),
                vec![
                    column("order_id", "int", "ID"),
                    column("user_id", "int", "ID"),
                    column("price", "float", "numerical"),
                    column("ts", "date", "timestamp"),
                ],
                serde_json::json!(["order_id"]),
                serde_json::json!("ts"),
            ),
            (
                "users".to_owned(),
                vec![
                    column("user_id", "int", "ID"),
                    column("age", "int", "numerical"),
                    column("country", "string", "categorical"),
                ],
                serde_json::json!(["user_id"]),
                Value::Null,
            ),
        ]
    );
    assert!(tables.iter().all(|t| t["duplicate_keys"]["v"] == "0"));
    // One fetch per statement the reference runs in Schema.from_source.
    assert_eq!(observed.fetches.len(), 10);
    // The whole description, the inputs and the step trace are the ones
    // python-core records (contract/pinned, which contract/tests holds
    // python-core to), so CI gates the two hosts' equality.
    let pinned: Value = serde_json::from_slice(
        &std::fs::read(contract().join("pinned/base.regression.core.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(described, pinned["described"]);
    assert_eq!(Value::from(observed.fetches), pinned["fetches"]);
    assert_eq!(Value::from(observed.inputs), pinned["inputs"]);
}

#[test]
fn a_time_of_day_is_refused_as_a_time_column_by_name() {
    let mut loaded = case::load(&[contract().join("cases/s1/base.regression.json")])
        .unwrap()
        .remove(0)
        .case;
    loaded.input.setup = Some(vec![
        "CREATE TABLE things (thing_id BIGINT, tod TIME)".into(),
        "INSERT INTO things SELECT i, TIME '01:00:00' + INTERVAL (i) MINUTE FROM range(30) t(i)"
            .into(),
    ]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&loaded, &bases).unwrap();
    assert_eq!(
        observed.answer,
        Answer::Error {
            kind: "unsupported".into(),
            code: "UNSUPPORTED_REQUEST".into()
        }
    );
}

#[test]
fn a_fetch_in_several_batches_is_admitted_as_one_attachment_of_its_rows() {
    let connection = database(None);
    let several = &vectors()["several"];
    let mut handed = 0;
    let data = admitted(&start(&several["request"]), |step| {
        let fetched = duck::answer(&connection, step).unwrap();
        handed = fetched.batches.len();
        fetched
    });
    assert!(handed > 1, "the driver handed over {handed} batch");
    let whole = rows(&data);
    assert_eq!(whole.num_rows() as u64, several["rows"].as_u64().unwrap());
    // The same rows, handed over as one batch, are admitted the same.
    let one = admitted(&start(&several["request"]), |step| {
        let fetched = duck::answer(&connection, step).unwrap();
        let batch = concat_batches(&fetched.schema, &fetched.batches).unwrap();
        duck::Fetched {
            schema: fetched.schema,
            batches: vec![batch],
        }
    });
    assert_eq!(rows(&one), whole);
    assert_eq!(one.schema, data.schema);
    // Every row, as the vectors write them: n from 0 and name r<n>.
    let n = whole
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let name = whole
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut seen: Vec<i64> = (0..whole.num_rows()).map(|i| n.value(i)).collect();
    for i in 0..whole.num_rows() {
        assert_eq!(name.value(i), format!("r{}", n.value(i)));
    }
    seen.sort_unstable();
    assert_eq!(seen, (0..5000).collect::<Vec<i64>>());
}

#[test]
fn an_empty_fetch_keeps_its_schema() {
    let connection = database(None);
    let empty = &vectors()["empty"];
    let data = admitted(&start(&empty["request"]), |step| {
        duck::answer(&connection, step).unwrap()
    });
    assert_eq!(rows(&data).num_rows(), 0);
    let fields: Vec<(String, DataType)> = data
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    assert_eq!(
        fields,
        [
            ("n".to_owned(), DataType::Int64),
            ("name".to_owned(), DataType::Utf8)
        ]
    );
    assert_eq!(empty["fields"][1][1], "utf8");
}

/// The time bounds' rows as counts, read under `zone`.
fn bounds(zone: Option<&str>) -> (Vec<Vec<i64>>, ArrowData) {
    let connection = database(zone);
    let data = admitted(&start(&vectors()["time_bounds"]["request"]), |step| {
        duck::answer(&connection, step).unwrap()
    });
    let whole = rows(&data);
    let position = whole
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let low = whole
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let high = whole
        .column(2)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert!(low.timezone().is_none());
    let found = (0..whole.num_rows())
        .map(|i| vec![i64::from(position.value(i)), low.value(i), high.value(i)])
        .collect();
    (found, data)
}

#[test]
fn a_timestamptz_column_reads_in_the_database_zone_s_wall_clock() {
    let wanted: Vec<Vec<i64>> =
        serde_json::from_value(vectors()["time_bounds"]["rows"].clone()).unwrap();
    assert_eq!(bounds(Some("zone")).0, wanted);
    // A zone set for the source's own session is not one its reads see, as
    // in the reference, whose reads are on cursors too.
    assert_eq!(bounds(Some("session_zone")).0, bounds(None).0);
}

#[test]
fn the_host_s_duckdb_links_the_reference_version_with_icu() {
    let connection = duck::open().unwrap();
    let autoload: String = connection
        .query_row(
            "SELECT current_setting('autoload_known_extensions')::VARCHAR",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(autoload, "false");
    let autoinstall: String = connection
        .query_row(
            "SELECT current_setting('autoinstall_known_extensions')::VARCHAR",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(autoinstall, "false");
}
