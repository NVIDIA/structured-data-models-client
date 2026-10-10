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
    assert_eq!(observed.operations, ["describe_source", "materialize"]);
    let Some(described) = observed.described.clone() else {
        panic!("not described: {:?}", observed.answer);
    };
    let Answer::Materialized(materialized) = observed.answer.clone() else {
        panic!("not materialized: {:?}", observed.answer);
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
    // six in materialize (T1.11c).
    assert_eq!(observed.fetches.len(), 16);
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
    assert_eq!(Value::from(observed.inputs), pinned["inputs"]);
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
            dialect: Dialect::Duckdb,
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
                dialect: Dialect::Duckdb,
                text: "SELECT 1".into(),
                params: vec![set(1), set(2)],
            }],
            expect: sdm_contract::data::ExpectedSchema::Discover {},
        },
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
