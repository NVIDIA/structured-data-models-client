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
use std::sync::Arc;

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
            dialect: Dialect::new("duckdb"),
            namespace: Vec::new(),
        }),
        echo: true,
        undeclared: vec![],
        fallback: None,
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
    assert_eq!(
        observed.operations,
        ["describe_source", "materialize", "predict"]
    );
    let Some(described) = observed.described.clone() else {
        panic!("not described: {:?}", observed.answer);
    };
    let Some(materialized) = observed.materialized.clone() else {
        panic!("not materialized: {:?}", observed.answer);
    };
    let Answer::Predicted(predicted) = observed.answer.clone() else {
        panic!("not predicted: {:?}", observed.answer);
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
    // One fetch per statement the reference runs: ten in Schema.from_source,
    // six in materialize (T1.11c); and the six of the two tables' catalogs
    // and previews each materialize reads again, as the source holds them
    // at that run (the owner's plan B, T3.4).
    assert_eq!(observed.fetches.len(), 22);
    // The whole description, the request state with its payload, the
    // inputs and the step trace are the ones python-core records
    // (contract/pinned, which contract/tests holds python-core to), so CI
    // gates the two hosts' equality.
    let pinned: Value = serde_json::from_slice(
        &std::fs::read(contract().join("pinned/base.regression.core.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(described, pinned["described"]);
    assert_eq!(materialized, pinned["materialized"]);
    assert_eq!(Value::from(observed.fetches), pinned["fetches"]);
    assert_eq!(Value::from(observed.inputs.clone()), pinned["inputs"]);
    // predict's call and result (T1.11d): the call's request id compared by
    // presence and type only, as capture_check.masked compares it (decision
    // E35, approved by the owner on 2026-10-10, the one exclusion of
    // contract/compare-policy.yaml), since each host draws its own;
    // everything else exactly.
    let mut masked = predicted.clone();
    let id = &mut masked["body"]["metadata"]["request_id"];
    assert!(id.is_string(), "{id}");
    *id = serde_json::json!({"excluded value of type": "string"});
    assert_eq!(masked, pinned["predicted"]);
    // The state predict started with is the one materialize answered, as
    // this host held it between the runs.
    assert_eq!(observed.inputs[2]["state"], materialized);
    // Not vacuous: two rows, each its entity, a prediction of 1.0 and its
    // place in the endpoint's order (answer_index, T1.28b).
    let rows = pinned["predicted"]["result"]["rows"].as_array().unwrap();
    assert_eq!(
        rows,
        &[
            serde_json::json!([{"i64": "0"}, {"i64": "0"}, {"f64_bits": "3ff0000000000000"}, {"i64": "0"}]),
            serde_json::json!([{"i64": "1"}, {"i64": "1"}, {"f64_bits": "3ff0000000000000"}, {"i64": "1"}]),
        ]
    );
    assert_eq!(
        pinned["predicted"]["body"]["context"]["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        112
    );
    // The pin is not vacuous: it holds the payload of 112 context examples
    // and 2 to predict, over both tables.
    let payload = &pinned["materialized"]["payload"];
    assert_eq!(
        payload["context"]["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        112
    );
    assert_eq!(
        payload["predict"]["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        payload["context"]["related_tables"]["orders"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        408
    );
}

#[test]
fn a_time_of_day_is_passed_over_as_holding_no_time_and_said() {
    // T1.23: as the reference's engine reads a time of day as no time at
    // all, it is passed over as holding no value, and the run says so.
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
    let described = observed.described.expect("described");
    let things = described["tables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "things")
        .unwrap();
    assert_eq!(things["time_column"], Value::Null);
    let codes: Vec<&str> = observed
        .warnings
        .iter()
        .map(|w| w["code"].as_str().unwrap())
        .collect();
    assert!(codes.contains(&"TIME_VALUES_UNREAD"), "{codes:?}");
    assert!(codes.contains(&"TIME_COLUMNS_GUESSED"), "{codes:?}");
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

/// The seeds of a key-set case: its keys, of the type named, as Arrow
/// built from their exact text, and their batch positions.
fn seeds_of(column_type: &Value, values: &[&str]) -> RecordBatch {
    use duckdb::arrow::array::{ArrayRef, Decimal128Array, UInt64Array};
    use duckdb::arrow::datatypes::{Field, Schema};
    let keys: ArrayRef = match column_type["type"].as_str().unwrap() {
        "uint64" => Arc::new(UInt64Array::from_iter_values(
            values.iter().map(|v| v.parse::<u64>().unwrap()),
        )),
        "int64" => Arc::new(Int64Array::from_iter_values(
            values.iter().map(|v| v.parse::<i64>().unwrap()),
        )),
        "decimal128" => {
            let precision = column_type["precision"].as_u64().unwrap() as u8;
            let scale = column_type["scale"].as_i64().unwrap() as i8;
            // The value's digits, the point dropped: its unscaled integer.
            let unscaled = values.iter().map(|v| {
                let (whole, fraction) = v.split_once('.').unwrap_or((v, ""));
                assert_eq!(fraction.len(), scale as usize);
                format!("{whole}{fraction}").parse::<i128>().unwrap()
            });
            Arc::new(
                Decimal128Array::from_iter_values(unscaled)
                    .with_precision_and_scale(precision, scale)
                    .unwrap(),
            )
        }
        other => panic!("no keys of {other}"),
    };
    let rows = keys.len();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", keys.data_type().clone(), false),
            Field::new("batch", DataType::Int64, false),
        ])),
        vec![keys, Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )
    .unwrap()
}

/// A lookup with seeds `keys`, through `HostRun`: the core attaches the
/// keys to the step as its own key set, the DuckDB provider binds them and
/// answers, and the core echoes what it admitted. Also the step's control
/// message, which the key set does not travel in.
fn keyed(connection: &Connection, request: &Value, keys: RecordBatch) -> (RecordBatch, Vec<u8>) {
    let mut request = request.clone();
    // The seeds' columns as the keys' own schema declares them.
    request["seeds"]["columns"] = serde_json::to_value(
        keys.schema()
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
    .unwrap();
    let fetch = TestFetch {
        request: serde_json::from_value::<DataRequest>(request).unwrap(),
        plan: None,
        provider: Some(ProviderChoice::Sql {
            dialect: Dialect::new("duckdb"),
            namespace: Vec::new(),
        }),
        echo: true,
        undeclared: vec![],
        fallback: None,
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
                Attachment::Arrow(ArrowData::new(keys.schema(), vec![keys])),
            ),
        ]),
    )
    .map_err(|out| read(&out).unwrap())
    .unwrap();
    let out = run.next().unwrap();
    let control = out.control.clone();
    let ControlMessage::Step(Step::Fetch { requests }) = read(&out).unwrap() else {
        panic!("not a fetch");
    };
    let step = &requests[0];
    let mut sets = BTreeMap::new();
    if let sdm_contract::data::FetchPlan::Sql { statements, .. } = &step.plan {
        for param in statements.iter().flat_map(|s| &s.params) {
            if let sdm_contract::data::SqlParam::KeySet { keys, .. } = param {
                let Ok(Some(Attachment::Arrow(data))) = run.take_attachment(keys.attachment) else {
                    panic!("the key set is not attached");
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
    assert_eq!(sets.len(), 1, "the plan binds one key set");
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
    (rows(&data), control)
}

/// Whether each seed found exactly its own row, its label `labels`' in
/// batch order: the check a key that passed through a float fails.
fn each_seed_finds_its_own_row(result: &RecordBatch, labels: &[&str]) -> Result<(), String> {
    let batch = result
        .column_by_name("__kumo_batch__")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let matches = result
        .column_by_name("__kumo_matches__")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let label = result
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut found = vec![None; labels.len()];
    for row in 0..result.num_rows() {
        let at = batch.value(row) as usize;
        if matches.value(row) != 1 {
            return Err(format!("seed {at} matched {} rows", matches.value(row)));
        }
        if found[at].replace(label.value(row).to_owned()).is_some() {
            return Err(format!("seed {at} answered twice"));
        }
    }
    let found: Vec<Option<String>> = found;
    let wanted: Vec<Option<String>> = labels.iter().map(|l| Some((*l).to_owned())).collect();
    if found != wanted {
        return Err(format!("labels {found:?}, not {wanted:?}"));
    }
    Ok(())
}

#[test]
fn keys_that_collide_as_float64_bind_join_and_answer_exactly() {
    let vectors = vectors();
    let connection = database(None);
    let cases = vectors["keys"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let values: Vec<&str> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let labels: Vec<&str> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let exact = seeds_of(&case["type"], &values);
        // The values do collide as float64: the check is not vacuous.
        let forced = {
            use duckdb::arrow::compute::cast;
            let key = cast(exact.column(0), &DataType::Float64).unwrap();
            let floats = key
                .as_any()
                .downcast_ref::<duckdb::arrow::array::Float64Array>()
                .unwrap();
            assert!(
                (1..floats.len()).any(|i| (0..i).any(|j| floats.value(i) == floats.value(j))),
                "{}: the values do not collide as float64",
                case["column"]
            );
            RecordBatch::try_new(
                Arc::new(duckdb::arrow::datatypes::Schema::new(vec![
                    duckdb::arrow::datatypes::Field::new("key", DataType::Float64, false),
                    duckdb::arrow::datatypes::Field::new("batch", DataType::Int64, false),
                ])),
                vec![key, Arc::clone(exact.column(1))],
            )
            .unwrap()
        };
        let (result, _) = keyed(&connection, &case["request"], exact);
        each_seed_finds_its_own_row(&result, &labels)
            .unwrap_or_else(|e| panic!("{}: {e}", case["column"]));
        // A forced conversion to float64 on the way fails the same check.
        let (result, _) = keyed(&connection, &case["request"], forced);
        assert!(
            each_seed_finds_its_own_row(&result, &labels).is_err(),
            "{}: keys forced to float64 still found their rows",
            case["column"]
        );
    }
}

#[test]
fn a_key_set_past_the_control_cap_travels_as_an_attachment() {
    let vectors = vectors();
    let large = &vectors["large"];
    let rows = large["rows"].as_u64().unwrap() as usize;
    let modulo = large["modulo"].as_u64().unwrap() as usize;
    let values: Vec<String> = (0..rows).map(|i| (i % modulo).to_string()).collect();
    let keys = seeds_of(
        &serde_json::json!({"type": "int64"}),
        &values.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert!(keys.get_array_memory_size() > 64 * 1024);
    let connection = database(None);
    let (result, control) = keyed(&connection, &large["request"], keys);
    // The step itself, not an attached one, is under the cap: the keys
    // travel beside it.
    assert!(control.len() < 64 * 1024);
    let step: Value = serde_json::from_slice(&control).unwrap();
    assert_eq!(step["message"], "step");
    assert_eq!(result.num_rows(), rows);
    let batch = result
        .column_by_name("__kumo_batch__")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let name = result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for row in 0..rows {
        let seed = batch.value(row) as usize;
        assert_eq!(name.value(row), format!("r{}", seed % modulo));
    }
}

#[test]
fn two_key_sets_bound_to_one_name_are_refused() {
    use sdm_contract::data::{FetchPlan, KeySet, SqlParam, SqlStatement};
    use sdm_contract::run::FetchStep;
    let connection = database(None);
    let set = |id| SqlParam::KeySet {
        keys: KeySet {
            attachment: AttachmentId(id),
            columns: vec![],
        },
        relation: "seeds".into(),
    };
    let step = FetchStep {
        id: sdm_contract::run::ReplyId(1),
        request: DataRequest::Describe {
            tables: vec![],
            part: sdm_contract::data::DescribePart::Tables,
        },
        plan: FetchPlan::Sql {
            statements: vec![SqlStatement {
                dialect: Dialect::new("duckdb"),
                text: "SELECT 1".into(),
                params: vec![set(1), set(2)],
            }],
            expect: sdm_contract::data::ExpectedSchema::Discover {},
        },
        continuation: None,
        purpose: None,
    };
    let keys = |values: &[&str]| {
        let batch = seeds_of(&serde_json::json!({"type": "int64"}), values);
        duck::Fetched {
            schema: batch.schema(),
            batches: vec![batch],
        }
    };
    let both = BTreeMap::from([
        (AttachmentId(1), keys(&["1"])),
        (AttachmentId(2), keys(&["2"])),
    ]);
    assert!(duck::answer_with(&connection, &step, &both).is_err());
}

/// The Arrow type (pyarrow's spelling) DuckDB reads a column of
/// `duckdb_type` as, for the types `base.regression`'s catalog reads hold;
/// any other is a test failure, never a match.
fn arrow_name(duckdb_type: &str) -> String {
    if let Some(inner) = duckdb_type.strip_suffix("[]") {
        return format!("list<l: {}>", arrow_name(inner));
    }
    match duckdb_type {
        "BOOLEAN" => "bool",
        "INTEGER" => "int32",
        "BIGINT" => "int64",
        "DOUBLE" => "double",
        "VARCHAR" => "string",
        "TIMESTAMP" => "timestamp[us]",
        other => panic!("no Arrow name for {other} here"),
    }
    .to_owned()
}

/// The committed reference observation of `case`.
fn reference_of(case: &str) -> Value {
    let directory = contract().join("reference");
    let mut files: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    assert_eq!(files.len(), 1, "one reference observation file");
    let text = std::fs::read_to_string(files.remove(0)).unwrap();
    text.lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|observation| observation["case"] == case)
        .unwrap()
}

/// Decision H9, measured at T1.11e: on `base.regression` this host's DuckDB
/// provider runs the reference's number of statements in each public call
/// (10 in `Schema.from_source`, 6 in `materialize`, none in `predict`), and
/// each returns the reference statement's result: its columns, its Arrow
/// schema where the reference read Arrow, and its rows in canonical form,
/// in order for the one statement whose order is total (`_by_fkey`,
/// `contract/compare-policy.yaml`'s `ordered`), as a multiset otherwise.
/// Its record is complete: DuckDB's own query log counts the same
/// statements, the three key sets bound, and the same texts.
#[test]
fn base_regression_s_statements_return_the_reference_s_results() {
    let loaded = case::load(&[contract().join("cases/s1/base.regression.json")]).unwrap();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&loaded[0].case, &bases).unwrap();
    let driver = observed.driver.expect("a driver record");
    let reference = reference_of("base.regression");
    let theirs = reference["driver"]["capture"]["statements"]
        .as_array()
        .unwrap()
        .clone();
    let ours = driver["capture"]["statements"].as_array().unwrap().clone();
    let per_stage = |statements: &[Value]| -> Vec<(String, usize)> {
        ["from_source", "materialize", "predict"]
            .iter()
            .map(|stage| {
                let n = statements.iter().filter(|s| s["stage"] == *stage).count();
                ((*stage).to_owned(), n)
            })
            .collect()
    };
    let counts = vec![
        ("from_source".to_owned(), 10),
        ("materialize".to_owned(), 6),
        ("predict".to_owned(), 0),
    ];
    // The reads the core marks as plan B's taken out (register P3A-1): two
    // tables' catalog, keys and preview; every statement after them is the
    // reference's.
    let fresh = ours.len() - sdm_testkit::plan_b::as_the_old_client_ran(&ours).len();
    assert_eq!(fresh, 6);
    let ours = sdm_testkit::plan_b::as_the_old_client_ran(&ours);
    assert_eq!(per_stage(&ours), counts);
    assert_eq!(per_stage(&theirs), counts);
    // The record is complete: DuckDB logged the same statements, and the
    // three bindings, and every text it logs as written is the one run.
    let logged = &driver["query_log"];
    assert_eq!(logged["statements"], 16 + fresh as u64);
    assert_eq!(logged["bindings"], 3);
    // Of the six read again, the four whose text DuckDB logs as written.
    assert_eq!(logged["compared"], 14 + 4);
    assert_eq!(logged["differing"], serde_json::json!([]));
    assert_eq!(driver["capture"]["bindings"].as_array().unwrap().len(), 3);
    for (position, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(a["stage"], b["stage"], "{position}");
        let [ra] = a["results"].as_array().unwrap().as_slice() else {
            panic!("{position}: one result read");
        };
        let [rb] = b["results"].as_array().unwrap().as_slice() else {
            panic!("{position}: one result read by the reference");
        };
        let (va, vb) = (&ra["values"], &rb["values"]);
        assert_eq!(va["unencodable"], serde_json::json!([]), "{position}");
        assert_eq!(vb["unencodable"], serde_json::json!([]), "{position}");
        assert_eq!(va["columns"], vb["columns"], "{position}: columns");
        assert_eq!(ra["rows"], rb["rows"], "{position}: rows");
        if rb.get("schema").is_some() {
            assert_eq!(ra["schema"], rb["schema"], "{position}: Arrow schema");
        } else {
            // The reference read its catalog as rows: its DuckDB type
            // names, as the Arrow types DuckDB reads them as, are this
            // host's Arrow schema (compare.py's `arrow_name`).
            let names: Vec<Value> = vb["types"]
                .as_array()
                .expect("the reference's DuckDB types")
                .iter()
                .map(|kind| Value::from(arrow_name(kind.as_str().unwrap())))
                .collect();
            let ours: Vec<Value> = ra["schema"]
                .as_array()
                .unwrap()
                .iter()
                .map(|field| field["type"].clone())
                .collect();
            assert_eq!(ours, names, "{position}: Arrow types");
        }
        assert_eq!(va["unordered"], vb["unordered"], "{position}: rows");
        // _by_fkey, the sixth of materialize's six, ordered totally.
        if position == 15 {
            assert_eq!(va["ordered"], vb["ordered"], "{position}: order");
        }
    }
}

/// The answer to the one statement the shipped data layer plans for the
/// columns of `table` on DuckDB, and that statement.
fn columns_of(connection: &Connection, table: &str) -> (String, Vec<RecordBatch>) {
    use sdm_contract::data::{DescribePart, FetchPlan};
    let request = DataRequest::Describe {
        tables: vec![table.into()],
        part: DescribePart::Columns,
    };
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
        namespace: Vec::new(),
    };
    let FetchPlan::Sql { statements, .. } =
        default_root().data_layer().plan(&request, &duckdb).unwrap()
    else {
        panic!("not SQL");
    };
    let text = statements[0].text.clone();
    let fetched = duck::run(connection, &text).unwrap();
    (text, fetched.batches)
}

#[test]
fn a_table_the_pragma_reads_as_another_name_is_read_by_its_own() {
    // Register entry P1B-5: PRAGMA table_info reads its argument as a
    // qualified name, so a name holding a double quote or a dot is read as
    // another; the data layer reads such a table by its own name, in the
    // pragma's columns and rows.
    let connection = duck::open().unwrap();
    let columns = "(id INTEGER, \"o\"\"dd\" VARCHAR NOT NULL, b DOUBLE DEFAULT 1.5, \
                   t TIMESTAMP, d DECIMAL(18, 3)[], s STRUCT(x INTEGER), PRIMARY KEY (id, b))";
    connection
        .execute_batch(&format!(
            "CREATE TABLE twin {columns};
             CREATE TABLE \"double\"\"quote\" {columns};
             CREATE TABLE \"a.b\" {columns};
             CREATE TABLE \"main.x\" {columns};
             CREATE TABLE \"\"\"lead\" {columns};
             CREATE TABLE x (other BIGINT);"
        ))
        .unwrap();
    let (text, twin) = columns_of(&connection, "twin");
    assert_eq!(text, "PRAGMA table_info(\"twin\")");
    let twin = concat_batches(&twin[0].schema(), &twin).unwrap();
    assert_eq!(twin.num_rows(), 6);
    for name in ["double\"quote", "a.b", "main.x", "\"lead"] {
        // The pragma itself fails, or reads `x` in `main.x`'s place.
        let pragma = format!("PRAGMA table_info(\"{}\")", name.replace('"', "\"\""));
        match duck::run(&connection, &pragma) {
            Err(_) => {}
            Ok(read) => assert_ne!(
                concat_batches(&read.schema, &read.batches).unwrap(),
                twin,
                "{name}"
            ),
        }
        let (text, answer) = columns_of(&connection, name);
        assert!(!text.starts_with("PRAGMA"), "{text}");
        let answer = concat_batches(&answer[0].schema(), &answer).unwrap();
        assert_eq!(answer, twin, "{name}");
    }
}

/// `value` with each `{"number": text}` the observation writes for a
/// number read back as that number (`host::numbers_as_text` undone).
fn numbers_again(value: Value) -> Value {
    match value {
        Value::Object(members) => match members.get("number") {
            Some(Value::String(text)) if members.len() == 1 => {
                serde_json::from_str(text).expect("a number's text")
            }
            _ => Value::Object(
                members
                    .into_iter()
                    .map(|(k, v)| (k, numbers_again(v)))
                    .collect(),
            ),
        },
        Value::Array(items) => Value::Array(items.into_iter().map(numbers_again).collect()),
        other => other,
    }
}

#[test]
fn every_payload_rust_host_materializes_is_the_locked_definition_s() {
    // T1.22: the request state's payload of each case the pure-Rust host
    // materializes, base.regression and the family cases among them,
    // against the contract's locked definition (`$defs/StatePayload`).
    let loaded = case::load_for(&[contract().join("cases")], case::Database::Duckdb).unwrap();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut held = Vec::new();
    for one in &loaded {
        let observed = host::run(&one.case, &bases).unwrap();
        let Some(state) = observed.materialized else {
            continue;
        };
        let payload = numbers_again(state["payload"].clone());
        assert_eq!(
            sdm_contract::schema_check::check("StatePayload", &payload),
            Ok(()),
            "{}",
            one.case.id
        );
        held.push(one.case.id.clone());
    }
    assert!(held.contains(&"base.regression".to_owned()), "{held:?}");
    // At least one case of each family that reaches materialize.
    for family in ["key.", "null.", "time.", "empty."] {
        assert!(
            held.iter().any(|id| id.starts_with(family)),
            "{family} {held:?}"
        );
    }
}

/// A type as pyarrow spells it, which `contract/request-kinds.json`
/// records.
fn spelled(data_type: &DataType) -> String {
    match data_type {
        DataType::Boolean => "bool".into(),
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 => "string".into(),
        DataType::Date32 => "date32[day]".into(),
        DataType::Timestamp(unit, zone) => {
            let unit = match unit {
                duckdb::arrow::datatypes::TimeUnit::Second => "s",
                duckdb::arrow::datatypes::TimeUnit::Millisecond => "ms",
                duckdb::arrow::datatypes::TimeUnit::Microsecond => "us",
                duckdb::arrow::datatypes::TimeUnit::Nanosecond => "ns",
            };
            match zone {
                None => format!("timestamp[{unit}]"),
                Some(zone) => format!("timestamp[{unit}, tz={zone}]"),
            }
        }
        DataType::Decimal128(precision, scale) => format!("decimal128({precision}, {scale})"),
        DataType::List(item) => format!("list<{}: {}>", item.name(), spelled(item.data_type())),
        other => panic!("no spelling for {other}"),
    }
}

/// Cell `i` of `array` as `contract/request-kinds.json` writes one: null,
/// a boolean, text, an integer as its decimal text, a float as its
/// hexadecimal form (Python's `float.hex`), a timestamp or date as its
/// count, a list of such.
fn cell(array: &dyn Array, i: usize) -> Value {
    use duckdb::arrow::array::AsArray;
    use duckdb::arrow::datatypes as t;
    if array.is_null(i) {
        return Value::Null;
    }
    let text = |v: String| Value::String(v);
    match array.data_type() {
        DataType::Boolean => Value::Bool(array.as_boolean().value(i)),
        DataType::Int8 => text(array.as_primitive::<t::Int8Type>().value(i).to_string()),
        DataType::Int16 => text(array.as_primitive::<t::Int16Type>().value(i).to_string()),
        DataType::Int32 => text(array.as_primitive::<t::Int32Type>().value(i).to_string()),
        DataType::Int64 => text(array.as_primitive::<t::Int64Type>().value(i).to_string()),
        DataType::UInt8 => text(array.as_primitive::<t::UInt8Type>().value(i).to_string()),
        DataType::UInt16 => text(array.as_primitive::<t::UInt16Type>().value(i).to_string()),
        DataType::UInt32 => text(array.as_primitive::<t::UInt32Type>().value(i).to_string()),
        DataType::UInt64 => text(array.as_primitive::<t::UInt64Type>().value(i).to_string()),
        DataType::Float64 => text(hex_float(array.as_primitive::<t::Float64Type>().value(i))),
        DataType::Float32 => text(hex_float(f64::from(
            array.as_primitive::<t::Float32Type>().value(i),
        ))),
        DataType::Utf8 => text(array.as_string::<i32>().value(i).to_owned()),
        DataType::Date32 => text(array.as_primitive::<t::Date32Type>().value(i).to_string()),
        DataType::Timestamp(duckdb::arrow::datatypes::TimeUnit::Microsecond, _) => text(
            array
                .as_primitive::<t::TimestampMicrosecondType>()
                .value(i)
                .to_string(),
        ),
        DataType::Timestamp(duckdb::arrow::datatypes::TimeUnit::Nanosecond, _) => text(
            array
                .as_primitive::<t::TimestampNanosecondType>()
                .value(i)
                .to_string(),
        ),
        DataType::List(_) => {
            let items = array.as_list::<i32>().value(i);
            Value::Array((0..items.len()).map(|j| cell(items.as_ref(), j)).collect())
        }
        other => panic!("no cell of {other}"),
    }
}

/// `value` as Python's `float.hex` writes it.
fn hex_float(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0x0.0p+0"
        } else {
            "0x0.0p+0"
        }
        .into();
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & ((1 << 52) - 1);
    let (lead, exponent) = if exponent == 0 {
        (0, -1022)
    } else {
        (1, exponent - 1023)
    };
    format!("{sign}0x{lead}.{mantissa:013x}p{exponent:+}")
}

/// The rows of `data` as `contract/request-kinds.json` writes them, in
/// sorted order.
fn canonical(data: &ArrowData) -> Vec<Value> {
    let mut rows = Vec::new();
    for batch in &data.batches {
        for i in 0..batch.num_rows() {
            rows.push(Value::Array(
                batch
                    .columns()
                    .iter()
                    .map(|column| cell(column.as_ref(), i))
                    .collect(),
            ));
        }
    }
    // Sorted by compact JSON, as the Python host sorts them.
    rows.sort_by_key(|row| serde_json::to_string(row).unwrap());
    rows
}

#[test]
fn every_request_kind_runs_on_every_base_as_the_vectors_say() {
    // T1.23: every data request kind, planned by the root's data layer for
    // DuckDB, answered by this host's provider and admitted, is the Arrow
    // schema and rows contract/request-kinds.json records, which the
    // Python host's provider is held to as well: the same fetch gives the
    // same admitted Arrow on both hosts, on every base.
    let kinds: Value =
        serde_json::from_slice(&std::fs::read(contract().join("request-kinds.json")).unwrap())
            .unwrap();
    let fetches = kinds["fetches"].as_array().unwrap();
    let names: std::collections::BTreeSet<&str> = fetches
        .iter()
        .map(|f| f["request"]["request"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 9, "{names:?}");
    // Every kind on every base, not only every kind and every base.
    let pairs: std::collections::BTreeSet<(&str, &str)> = fetches
        .iter()
        .map(|f| {
            (
                f["request"]["request"].as_str().unwrap(),
                f["base"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(pairs.len(), names.len() * 3, "{pairs:?}");
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let mut held = 0;
    for fetch in fetches {
        let base = fetch["base"].as_str().unwrap();
        let connection = duck::open().unwrap();
        for statement in &bases[base] {
            connection.execute_batch(statement).unwrap();
        }
        let data = match fetch.get("keys") {
            Some(keys) => {
                let keys: Vec<i64> = keys
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|k| k.as_i64().unwrap())
                    .collect();
                let rows = keys.len() as i64;
                let seeds = RecordBatch::try_new(
                    Arc::new(duckdb::arrow::datatypes::Schema::new(vec![
                        duckdb::arrow::datatypes::Field::new("key", DataType::Int64, true),
                        duckdb::arrow::datatypes::Field::new("batch", DataType::Int64, true),
                    ])),
                    vec![
                        Arc::new(Int64Array::from(keys)),
                        Arc::new(Int64Array::from_iter_values(0..rows)),
                    ],
                )
                .unwrap();
                // The key set the seeds travel as, which keyed names in full.
                let mut request = fetch["request"].clone();
                request["seeds"] = serde_json::json!({"attachment": 0, "columns": []});
                let (batch, _) = keyed(&connection, &request, seeds);
                ArrowData::new(batch.schema(), vec![batch])
            }
            None => {
                let run = start(&fetch["request"]);
                admitted(&run, |step| duck::answer(&connection, step).unwrap())
            }
        };
        let place = format!("{base}: {}", fetch["name"]);
        let schema: Vec<Value> = data
            .schema
            .fields()
            .iter()
            .map(|f| serde_json::json!([f.name(), spelled(f.data_type())]))
            .collect();
        assert_eq!(Value::from(schema), fetch["schema"], "{place}");
        assert_eq!(Value::from(canonical(&data)), fetch["rows"], "{place}");
        held += 1;
    }
    assert_eq!(held, fetches.len());
    assert!(held >= 30);
}

#[test]
fn a_source_of_5000_columns_is_described_and_materialized_through_host_run() {
    // T1.23's exit: describe_source and materialize on a generated source
    // of 5,000 columns run to Done through HostRun on this core runner
    // (the client's tests run them on python-core): base.regression with
    // its users widened to 5,000 columns, each a feature.
    let mut loaded = case::load(&[contract().join("cases/s1/base.regression.json")])
        .unwrap()
        .remove(0)
        .case;
    let features: Vec<String> = (3..5_000)
        .map(|i| format!("(user_id * {i}) % 7 AS f{i}"))
        .collect();
    loaded.input.setup = Some(vec![format!(
        "CREATE OR REPLACE TABLE users AS SELECT *, {} FROM users",
        features.join(", ")
    )]);
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let observed = host::run(&loaded, &bases).unwrap();
    let described = observed.described.expect("described");
    let users = described["tables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "users")
        .unwrap();
    assert_eq!(users["columns"].as_array().unwrap().len(), 5_000);
    assert_eq!(
        observed.operations,
        ["describe_source", "materialize", "predict"]
    );
    let state = observed.materialized.expect("materialized");
    // The payload's users carry every column, beside the instance's id.
    let sent = state["payload"]["context"]["related_tables"]["users"]["columns"]
        .as_array()
        .map(Vec::len);
    assert_eq!(sent, Some(5_001));
    assert!(
        matches!(observed.answer, Answer::Predicted(_)),
        "{:?}",
        observed.answer
    );
}

/// `base.regression` described with `source` as its core's SourceSpec,
/// over `setup`: what this host observed.
fn described_as(source: Value, setup: Vec<String>) -> host::Observed {
    let mut loaded = case::load(&[contract().join("cases/s1/base.regression.json")])
        .unwrap()
        .remove(0)
        .case;
    loaded.input.setup = Some(setup);
    let core = loaded.core.as_mut().unwrap();
    core.materialize = None;
    core.predict = None;
    core.describe_source.as_mut().unwrap().source = source;
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    host::run(&loaded, &bases).unwrap()
}

#[test]
fn a_named_table_the_source_lacks_is_the_caller_s_naming_refused() {
    // T1.23 fix 1 (subagent P2-2): DuckDB fails reading a missing table's
    // columns; the core asks the table list then and refuses the naming
    // with the reference's INVALID_REQUEST, on this host as on python-core.
    let observed = described_as(
        serde_json::json!({"tables": [{"name": "x", "source": {"source": ["nope"]}}], "primary_keys": {}, "time_columns": {}}),
        vec![],
    );
    let Answer::Error { code, fields, .. } = &observed.answer else {
        panic!("{:?}", observed.answer);
    };
    assert_eq!(code, "INVALID_REQUEST");
    assert_eq!(fields["field"], "tables");
}

#[test]
fn a_source_table_named_twice_is_joined_under_its_last_name() {
    // T1.23 fix 1 (Codex P2): as the reference maps source names to the
    // schema's (`_declared_links`), a declared foreign key to a table named
    // twice joins the last name, by its exact source name or the one equal
    // in any case.
    for (parent, referenced) in [("parent", "parent"), ("\"Parent\"", "Parent")] {
        let observed = described_as(
            serde_json::json!({
                "tables": [
                    {"name": "first", "source": {"source": [referenced]}},
                    {"name": "children", "source": {"source": ["child"]}},
                    {"name": "second", "source": {"source": [referenced]}},
                ],
                "primary_keys": {},
                "time_columns": {},
            }),
            vec![
                format!("CREATE TABLE {parent} (id BIGINT PRIMARY KEY)"),
                format!("INSERT INTO {parent} SELECT i FROM range(4) t(i)"),
                "CREATE TABLE child (child_id BIGINT PRIMARY KEY, parent_id BIGINT REFERENCES parent (id))".into(),
                "INSERT INTO child SELECT i, i % 4 FROM range(9) t(i)".into(),
            ],
        );
        let described = observed.described.expect("described");
        assert_eq!(
            described["links"],
            serde_json::json!([[
                {"table": "children", "column": "parent_id"},
                {"table": "second", "column": "id"}
            ]]),
            "{parent}"
        );
    }
}

#[test]
fn the_source_says_which_relations_it_holds_under_a_name() {
    // T1.23 fix 2 (Codex P2): asked only once reading a named table's
    // columns has failed, the table list with the name asked answers the
    // relations, tables or views, DuckDB would read under it, matched as
    // DuckDB folds a name (ASCII letters only): the core matches no name
    // itself.
    let connection = duck::open().unwrap();
    connection
        .execute_batch(
            "CREATE TABLE parent (id BIGINT); CREATE VIEW parent_view AS SELECT * FROM parent; \
             CREATE TABLE \"Äpfel\" (id BIGINT); CREATE SCHEMA other; \
             CREATE TABLE other.elsewhere (id BIGINT); \
             CREATE VIEW other.elsewhere_view AS SELECT * FROM other.elsewhere; \
             ATTACH ':memory:' AS att; CREATE TABLE att.attached (id BIGINT)",
        )
        .unwrap();
    for (name, held) in [
        ("parent", 1),
        ("PARENT", 1),
        ("parent_view", 1),
        ("Parent_View", 1),
        ("Äpfel", 1),
        ("äpfel", 0),
        ("nope", 0),
        // Not what an unqualified name reads: another schema, another
        // attached database (T1.23 fix 3).
        ("elsewhere", 0),
        ("elsewhere_view", 0),
        ("attached", 0),
    ] {
        // The name asked as a relation of one part (contract v26).
        let run = start(&serde_json::json!(
            {"request": "describe", "tables": [{"source": [name]}], "part": "tables"}
        ));
        let data = admitted(&run, |step| duck::answer(&connection, step).unwrap());
        let rows: usize = data.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, held, "{name}");
    }
}

#[test]
fn a_listed_table_the_schema_may_not_name_is_refused() {
    // T1.23 fix 1, the listed path: a table the source lists under a name
    // the schema may not give (here a backtick) is refused, INVALID_REQUEST
    // naming the field, as the reference refuses it.
    let observed = described_as(
        serde_json::json!({"tables": [], "primary_keys": {}, "time_columns": {}}),
        vec![
            "CREATE TABLE \"a`b\" (id BIGINT PRIMARY KEY)".into(),
            "INSERT INTO \"a`b\" VALUES (1), (2)".into(),
        ],
    );
    let Answer::Error { code, fields, .. } = &observed.answer else {
        panic!("{:?}", observed.answer);
    };
    assert_eq!(code, "INVALID_REQUEST");
    assert_eq!(fields["field"], "tables");
}

/// Decision E52 on this host, the parity of the Python host's
/// `test_a_second_call_runs_the_reference_s_statements`: one materializer's
/// three calls on `base.regression`, each handed the sampler state the one
/// before it left, run the reference's statements (6, then 4: the index
/// read and the time bounds the first made are skipped), warn of the
/// missing indexes once, and answer the same request each time.
#[test]
fn a_materializer_s_later_calls_skip_what_its_first_read() {
    let loaded = case::load(&[contract().join("cases/s1/base.regression.json")]).unwrap();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    let calls = host::materialize_repeatedly(&loaded[0].case, &bases, 3).unwrap();
    // The reads plan B adds taken out (register P3A-1).
    let kinds: Vec<Vec<String>> = calls
        .iter()
        .map(|call| {
            sdm_testkit::plan_b::requests_as_the_old_client_made(&call.fetches)
                .iter()
                .map(|f| f["request"]["request"].as_str().unwrap().to_owned())
                .collect()
        })
        .collect();
    assert_eq!(
        kinds[0],
        [
            "describe",
            "time_bounds",
            "sample_entities",
            "neighbors",
            "lookup",
            "neighbors"
        ]
    );
    for later in &kinds[1..] {
        assert_eq!(
            later,
            &["sample_entities", "neighbors", "lookup", "neighbors"]
        );
    }
    let warned: Vec<Vec<String>> = calls
        .iter()
        .map(|call| {
            call.warnings
                .iter()
                .map(|w| w["code"].as_str().unwrap().to_owned())
                .collect()
        })
        .collect();
    assert_eq!(warned, [vec!["MISSING_INDICES".to_owned()], vec![], vec![]]);
    let answered: Vec<&Option<Value>> = calls.iter().map(|call| &call.materialized).collect();
    assert!(answered[0].is_some());
    assert!(answered.iter().all(|a| *a == answered[0]));
}

/// Decision E74 on this host, the parity of the Python host's
/// `test_a_state_the_core_refuses_is_not_handed_back_again`: a sampler state
/// the core refuses at the start (of another version, or a memo that does
/// not read) is one the core says to drop, and the host follows it alone:
/// the next call starts afresh, reading what a materializer's first call
/// reads, rather than being refused the same way.
#[test]
fn a_sampler_state_the_core_refuses_is_dropped_on_its_word() {
    let loaded = case::load(&[contract().join("cases/s1/base.regression.json")]).unwrap();
    let bases = bases::load(&contract().join("bases.yaml")).unwrap();
    for (held, field) in [
        ((99, b"{}".to_vec()), "sampler_state.version"),
        ((1, b"not a memo".to_vec()), "sampler_state"),
    ] {
        let calls =
            host::materialize_repeatedly_holding(&loaded[0].case, &bases, 2, Some(held)).unwrap();
        let Answer::Error { fields, .. } = &calls[0].answer else {
            panic!("{field}: {:?}", calls[0].answer);
        };
        assert_eq!(fields["field"], field);
        assert!(calls[0].materialized.is_none(), "{field}");
        // The reads plan B adds taken out (register P3A-1).
        let fetches = sdm_testkit::plan_b::requests_as_the_old_client_made(&calls[1].fetches);
        let reads: Vec<&str> = fetches
            .iter()
            .map(|f| f["request"]["request"].as_str().unwrap())
            .collect();
        assert_eq!(reads[..2], ["describe", "time_bounds"], "{field}");
        assert!(calls[1].materialized.is_some(), "{field}");
    }
}
