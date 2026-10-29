// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual SQL-plan HostRuns with explicit fixture replies, without a database
//! or a claim about pandas effects. Public Python twins exercise those effects.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::builder::{Int64Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray, TimestampNanosecondArray, UInt32Array, UInt64Array,
};
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataRequest, DescribePart, ExpectedSchema, FetchPlan,
    MATCHES_COLUMN, Relation, SEED_BATCH, SEED_KEY,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::materialize::{
    MaterializePreparation, NativeTargetValues, PreparedMaterializeValue,
};
use sdm_contract::ops::{Materialized, Operation, OperationResult};
use sdm_contract::run::{FetchOutcome, FetchReply, FetchStep, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

const DAY: i64 = 86_400_000_000;
const FIRST: i64 = 1_672_531_200_000_000;

fn ints(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
}

fn data(batch: RecordBatch) -> ArrowData {
    ArrowData::new(batch.schema(), vec![batch])
}

fn json_bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn control(out: &Outgoing) -> Value {
    serde_json::to_value(sdm_host::read(out).unwrap()).unwrap()
}

fn start(operation: Value, attachments: Attachments) -> Result<HostRun, Value> {
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: TimestampUnit::Ns,
                zone: Some("UTC".into()),
                value: 0,
            },
            local_zone: None,
            zone_roots: vec![],
            date_literal_range: None,
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
        },
        operation: serde_json::from_value::<Operation>(operation).unwrap(),
    }));
    HostRun::start(
        &sdm_host::default_root(),
        &sdm_host::encode(&start).unwrap(),
        attachments,
    )
    .map_err(|out| control(&out))
}

fn table(name: &str) -> RecordBatch {
    let columns: Vec<(&str, ArrayRef)> = match name {
        "users" => vec![("user_id", ints(0..8)), ("age", ints(20..28))],
        "items" => vec![
            ("item_id", ints(0..4)),
            (
                "amount",
                Arc::new(Float64Array::from(vec![3.0, 4.0, 5.0, 6.0])),
            ),
        ],
        "buys" => vec![
            ("buy_id", ints(0..120)),
            ("user_id", ints((0..120).map(|r| r % 8))),
            ("item_id", ints((0..120).map(|r| (r * 3) % 4))),
            (
                "ts",
                Arc::new(TimestampMicrosecondArray::from(
                    (0..120).map(|r| FIRST + r * 5 * DAY).collect::<Vec<_>>(),
                )),
            ),
        ],
        _ => panic!("fixture has no relation {name}"),
    };
    RecordBatch::try_from_iter(columns).unwrap()
}

fn project(batch: &RecordBatch, names: &Value) -> RecordBatch {
    batch
        .project(
            &names
                .as_array()
                .unwrap()
                .iter()
                .map(|name| batch.schema().index_of(name.as_str().unwrap()).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
}

fn integer(column: &ArrayRef, row: usize) -> i64 {
    if let Some(array) = column.as_primitive_opt::<Int64Type>() {
        array.value(row)
    } else if let Some(array) = column.as_primitive_opt::<Int32Type>() {
        i64::from(array.value(row))
    } else if let Some(array) = column.as_primitive_opt::<UInt32Type>() {
        i64::from(array.value(row))
    } else if let Some(array) = column.as_primitive_opt::<UInt64Type>() {
        i64::try_from(array.value(row)).unwrap()
    } else {
        panic!(
            "fixture key/batch is not an integer: {:?}",
            column.data_type()
        )
    }
}

fn nanos(column: &ArrayRef, row: usize) -> i64 {
    if let Some(array) = column.as_any().downcast_ref::<TimestampNanosecondArray>() {
        array.value(row)
    } else if let Some(array) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        array.value(row) * 1000
    } else {
        panic!("fixture time is not a timestamp")
    }
}

/// This finite fixture answers typed requests, never parses or executes SQL.
/// Its keyed replies retain the actual seed batch, window and projection.
fn answer(
    asked: &FetchStep,
    run: &HostRun,
    seeds: &mut BTreeMap<AttachmentId, ArrowData>,
) -> ArrowData {
    assert!(matches!(&asked.plan, FetchPlan::Sql { statements, .. } if !statements.is_empty()));
    let request = &asked.request;
    let value = serde_json::to_value(request).unwrap();
    if let DataRequest::Describe {
        tables,
        part: DescribePart::Columns,
    } = request
    {
        let source = table(tables[0].source.last().unwrap());
        let FetchPlan::Sql {
            expect: ExpectedSchema::Schema { fields },
            ..
        } = &asked.plan
        else {
            panic!("catalog schema")
        };
        let schema = Arc::new(arrow_schema::Schema::new(
            fields
                .iter()
                .map(sdm_contract::types::Field::to_arrow)
                .collect::<Vec<_>>(),
        ));
        let names = source
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        let types = names
            .iter()
            .map(|name| match name.as_str() {
                "ts" => "TIMESTAMP",
                "amount" => "DOUBLE",
                _ => "BIGINT",
            })
            .collect::<Vec<_>>();
        let keys = names
            .iter()
            .map(|name| name == &names[0])
            .collect::<Vec<_>>();
        return data(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from_iter_values(0..source.num_columns() as i32)),
                    Arc::new(StringArray::from(names)),
                    Arc::new(StringArray::from(types)),
                    Arc::new(BooleanArray::from(vec![true; source.num_columns()])),
                    Arc::new(StringArray::from(vec![None::<&str>; source.num_columns()])),
                    Arc::new(BooleanArray::from(keys)),
                ],
            )
            .unwrap(),
        );
    }
    if matches!(request, DataRequest::Describe { .. }) {
        let FetchPlan::Sql {
            expect: ExpectedSchema::Schema { fields },
            ..
        } = &asked.plan
        else {
            panic!("catalog schema")
        };
        let schema = Arc::new(arrow_schema::Schema::new(
            fields
                .iter()
                .map(sdm_contract::types::Field::to_arrow)
                .collect::<Vec<_>>(),
        ));
        return data(RecordBatch::new_empty(schema));
    }
    if let DataRequest::TimeBounds { columns } = request {
        return data(
            RecordBatch::try_from_iter([
                (
                    "table_index",
                    Arc::new(Int32Array::from_iter_values(0..columns.len() as i32)) as ArrayRef,
                ),
                (
                    "min_date",
                    Arc::new(TimestampMicrosecondArray::from(vec![FIRST; columns.len()]))
                        as ArrayRef,
                ),
                (
                    "max_date",
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        FIRST + 119 * 5 * DAY;
                        columns.len()
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
        );
    }
    let relation: Relation = serde_json::from_value(value["table"].clone()).unwrap();
    let source = table(relation.source.last().unwrap());
    match value["request"].as_str().unwrap() {
        "preview" | "scan" => data(project(&source, &value["columns"])),
        "sample_entities" => {
            // The fixture contains eight non-null entity rows, all requested.
            assert!(value["n"].as_u64().unwrap() >= source.num_rows() as u64);
            assert!(
                value["filters"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|f| f["filter"] == "not_null")
            );
            data(project(&source, &value["columns"]))
        }
        "count_rows" => {
            data(RecordBatch::try_from_iter([("n", ints([source.num_rows() as i64]))]).unwrap())
        }
        "neighbors" | "lookup" => {
            let id = AttachmentId(value["seeds"]["attachment"].as_u64().unwrap() as u32);
            let seed = seeds
                .entry(id)
                .or_insert_with(|| {
                    let Attachment::Arrow(rows) = run.take_attachment(id).unwrap().unwrap() else {
                        panic!("actual seed rows")
                    };
                    rows
                })
                .batches[0]
                .clone();
            let key = value["key"].as_array().unwrap();
            assert_eq!(key.len(), 1);
            let source_key = source.column_by_name(key[0].as_str().unwrap()).unwrap();
            let seed_key = seed.column_by_name(SEED_KEY).unwrap();
            let mut positions = Vec::new();
            let mut batches = Vec::new();
            let mut counts = Vec::new();
            for at in 0..seed.num_rows() {
                let mut matched = Vec::new();
                for row in 0..source.num_rows() {
                    if integer(source_key, row) != integer(seed_key, at) {
                        continue;
                    }
                    let window = &value["window"];
                    if !window.is_null() {
                        let time = nanos(
                            source
                                .column_by_name(window["column"].as_str().unwrap())
                                .unwrap(),
                            row,
                        );
                        let outside =
                            [("lower", true), ("upper", false)]
                                .into_iter()
                                .any(|(side, lower)| {
                                    let bound = &window[side];
                                    if bound.is_null() {
                                        return false;
                                    }
                                    let limit = nanos(
                                        seed.column_by_name(bound["seed_column"].as_str().unwrap())
                                            .unwrap(),
                                        at,
                                    );
                                    if lower {
                                        if bound["inclusive"] == true {
                                            time < limit
                                        } else {
                                            time <= limit
                                        }
                                    } else if bound["inclusive"] == true {
                                        time > limit
                                    } else {
                                        time >= limit
                                    }
                                });
                        if outside {
                            continue;
                        }
                    }
                    matched.push(row as u32);
                }
                let total = matched.len() as i64;
                if let Some(order) = value["order"].as_array() {
                    matched.sort_by(|left, right| {
                        for term in order {
                            let column = source
                                .column_by_name(term["column"].as_str().unwrap())
                                .unwrap();
                            let order = if term["as_timestamp"] == true {
                                nanos(column, *left as usize).cmp(&nanos(column, *right as usize))
                            } else {
                                integer(column, *left as usize)
                                    .cmp(&integer(column, *right as usize))
                            };
                            let order = if term["descending"] == true {
                                order.reverse()
                            } else {
                                order
                            };
                            if !order.is_eq() {
                                return order;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                }
                if value["request"] == "lookup" {
                    matched.truncate(1);
                } else if let Some(limit) = value["per_seed"].as_u64() {
                    matched.truncate(limit as usize);
                }
                for row in matched {
                    positions.push(row);
                    batches.push(integer(seed.column_by_name(SEED_BATCH).unwrap(), at));
                    counts.push(total);
                }
            }
            let selection = UInt32Array::from(positions);
            let projected = project(&source, &value["columns"]);
            let mut columns = vec![(BATCH_COLUMN.to_owned(), ints(batches))];
            if value["request"] == "lookup" {
                columns.push((MATCHES_COLUMN.to_owned(), ints(counts)));
            }
            for (field, column) in projected.schema().fields().iter().zip(projected.columns()) {
                columns.push((
                    field.name().clone(),
                    arrow_select::take::take(column.as_ref(), &selection, None).unwrap(),
                ));
            }
            data(RecordBatch::try_from_iter(columns).unwrap())
        }
        other => panic!("fixture did not authorize request {other}"),
    }
}

struct Stages {
    held: Attachments,
    issued: Vec<Value>,
}

impl Stages {
    fn stage(&mut self, value: Value) -> Result<PreparedMaterializeValue, Value> {
        let input: MaterializePreparation = serde_json::from_value(value).unwrap();
        let mut attachments: Attachments = input
            .attachments()
            .into_iter()
            .map(|(id, _)| (id, self.held[&id].clone()))
            .collect();
        let id = AttachmentId(self.held.keys().map(|id| id.0).max().unwrap_or(0) + 100);
        attachments.insert(id, Attachment::Bytes(serde_json::to_vec(&input).unwrap()));
        let run = start(json!({"op":"prepare_materialize","input":id}), attachments)?;
        let mut seeds = BTreeMap::new();
        let mut result = loop {
            let out = run.next().map_err(|out| control(&out))?;
            match sdm_host::read(&out).unwrap() {
                ControlMessage::Step(Step::Fetch { requests }) => {
                    for request in requests {
                        self.issued.push(serde_json::to_value(&request).unwrap());
                        let rows = answer(&request, &run, &mut seeds);
                        let reply = ControlMessage::FetchReply(FetchReply {
                            id: request.id,
                            outcome: FetchOutcome::Data {
                                data: AttachmentId(1),
                                continuation: None,
                            },
                        });
                        run.reply(
                            &sdm_host::encode(&reply).unwrap(),
                            Attachments::from([(AttachmentId(1), Attachment::Arrow(rows))]),
                        )
                        .map_err(|out| control(&out))?;
                    }
                }
                ControlMessage::Step(Step::Warnings) | ControlMessage::Warnings { .. } => {
                    let _ = run.take_warnings().unwrap();
                }
                ControlMessage::Step(Step::Done {
                    result: OperationResult::PrepareMaterialize(result),
                }) => {
                    let Attachment::Bytes(bytes) =
                        run.take_attachment(result.value).unwrap().unwrap()
                    else {
                        panic!("actual prepared bytes")
                    };
                    break serde_json::from_slice::<PreparedMaterializeValue>(&bytes).unwrap();
                }
                other => panic!("unexpected prepared SQL step {other:?}"),
            }
        };
        let mut aliases = BTreeMap::new();
        rebind(&mut result, &mut |id| {
            let old = *id;
            if let Some(new) = aliases.get(&old) {
                *id = *new;
                return;
            }
            let new = if let Some(owner) = run.take_attachment(old).unwrap() {
                let new = AttachmentId(
                    (self.held.keys().map(|id| id.0).max().unwrap_or(0) + 1).max(1000),
                );
                self.held.insert(new, owner);
                new
            } else {
                assert!(self.held.contains_key(&old));
                old
            };
            aliases.insert(old, new);
            *id = new;
        });
        Ok(result)
    }
}

fn rebind(value: &mut PreparedMaterializeValue, keep: &mut impl FnMut(&mut AttachmentId)) {
    match value {
        PreparedMaterializeValue::Source(value) => keep(&mut value.owner),
        PreparedMaterializeValue::Population(value) => keep(&mut value.owner),
        PreparedMaterializeValue::ColumnMeaning(_) => {}
        PreparedMaterializeValue::LabelInputs(value) => keep(&mut value.owner),
        PreparedMaterializeValue::TargetColumn(value) => {
            keep(&mut value.input.owner);
            keep(&mut value.owner);
            for operand in &mut value.condition_values {
                keep(&mut operand.converted.rows);
                if let Some(id) = &mut operand.converted.presence {
                    keep(id);
                }
            }
        }
        PreparedMaterializeValue::LabelGroups(value) => keep(&mut value.owner),
        PreparedMaterializeValue::LabelBatch(value) => keep(&mut value.owner),
        PreparedMaterializeValue::Request(value) => {
            keep(&mut value.state.payload);
            keep(&mut value.state.query);
            for step in value.state.forecast.iter_mut().flatten() {
                keep(&mut step.payload);
                keep(&mut step.query);
            }
            for batch in value.state.batches.iter_mut().flatten() {
                keep(&mut batch.payload);
            }
            if let Some(ranking) = &mut value.state.ranking {
                for part in &mut ranking.parts {
                    keep(&mut part.payload);
                    keep(&mut part.query);
                }
            }
        }
    }
}

fn fixture() -> (Stages, Value) {
    let column = |name: &str, dtype: &str, stype: &str| json!({"name":name,"dtype":dtype,"stype":stype,"stype_before_key":null});
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["main","users"]},"columns":[column("user_id","int64","ID"),column("age","int64","numerical")],"primary_key":["user_id"],"time_column":null},
        {"name":"items","source":{"source":["main","items"]},"columns":[column("item_id","int64","ID"),column("amount","float64","numerical")],"primary_key":["item_id"],"time_column":null},
        {"name":"buys","source":{"source":["main","buys"]},"columns":[column("buy_id","int64","ID"),column("user_id","int64","ID"),column("item_id","int64","ID"),column("ts","date","timestamp")],"primary_key":["buy_id"],"time_column":"ts"}],
        "links":[[{"table":"buys","column":"user_id"},{"table":"users","column":"user_id"}],[{"table":"buys","column":"item_id"},{"table":"items","column":"item_id"}]]});
    let held = Attachments::from([
        (AttachmentId(1), json_bytes(schema)),
        (
            AttachmentId(2),
            json_bytes(json!(
                "PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) FOR EACH users.user_id"
            )),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(data(
                RecordBatch::try_from_iter([("index", ints([0, 1]))]).unwrap(),
            )),
        ),
        (
            AttachmentId(5),
            json_bytes(json!({"num_context_samples":3,"num_neighbors":[3,3],"random_seed":7})),
        ),
    ]);
    (
        Stages {
            held,
            issued: vec![],
        },
        json!({"stage":"task","request":{"provider":{"provider":"sql","dialect":"duckdb"},"schema":1,"query":2,"indices":3,"seed":"0x7","values":5,"options":[],"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"}}}),
    )
}

fn native(
    stages: &mut Stages,
    positions: Vec<u64>,
    values: Vec<i64>,
    presence: Vec<bool>,
) -> NativeTargetValues {
    let id = AttachmentId(stages.held.keys().map(|id| id.0).max().unwrap() + 1);
    let rows = RecordBatch::try_from_iter([
        (
            "position",
            Arc::new(UInt64Array::from(positions)) as ArrayRef,
        ),
        ("target", ints(values)),
    ])
    .unwrap();
    stages.held.insert(id, Attachment::Arrow(data(rows)));
    let presence_id = AttachmentId(id.0 + 1);
    let present = RecordBatch::try_from_iter([(
        "present",
        Arc::new(BooleanArray::from(presence)) as ArrayRef,
    )])
    .unwrap();
    stages
        .held
        .insert(presence_id, Attachment::Arrow(data(present)));
    NativeTargetValues {
        rows: id,
        presence: Some(presence_id),
        encoding: None,
    }
}

#[test]
fn sql_native_target_and_changed_group_completion_reach_the_actual_request() {
    let (mut stages, task) = fixture();
    let PreparedMaterializeValue::Source(source) = stages.stage(task).unwrap() else {
        panic!("source")
    };
    let mut foreign = source.clone();
    foreign.identity.push_str("-foreign");
    assert_eq!(
        stages
            .stage(json!({"stage":"population","source":foreign}))
            .unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let PreparedMaterializeValue::Population(population) = stages
        .stage(json!({"stage":"population","source":source}))
        .unwrap()
    else {
        panic!("population")
    };
    assert!(population.rows > 0 && population.rows <= u64::from(population.batch_rows));
    let request = json!({"stage":"label_inputs","population":population,"range":{"start":0,"stop":population.rows},"previous":null});
    let PreparedMaterializeValue::LabelInputs(inputs) = stages.stage(request.clone()).unwrap()
    else {
        panic!("actual SQL inputs")
    };
    assert!(inputs.source_read.is_some());
    assert!(stages.issued.iter().any(|fetch| fetch["partition"]["first"]
        == json!(inputs.source_read.unwrap())
        && fetch["request"]["request"] == "neighbors"
        && fetch["request"]["table"]["source"] == json!(["main", "buys"])));
    let Attachment::LabelInputs(owner) = &stages.held[&inputs.owner] else {
        panic!("input owner")
    };
    let source_rows = owner.rows().unwrap().batches[0].clone();
    assert!(source_rows.num_rows() > 1);
    let positions = (0..source_rows.num_rows() as u64).collect::<Vec<_>>();
    let short = native(
        &mut stages,
        positions[1..].to_vec(),
        vec![3; positions.len() - 1],
        vec![true; positions.len()],
    );
    assert_eq!(
        stages
            .stage(json!({"stage":"target_column","inputs":inputs,"converted":short}))
            .unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let converted = native(
        &mut stages,
        positions.clone(),
        vec![3; positions.len()],
        vec![true; positions.len()],
    );
    let PreparedMaterializeValue::TargetColumn(column) = stages
        .stage(json!({"stage":"target_column","inputs":inputs,"converted":converted}))
        .unwrap()
    else {
        panic!("target")
    };
    let PreparedMaterializeValue::LabelInputs(other) = stages.stage(request).unwrap() else {
        panic!("other inputs")
    };
    let mut stale = column.clone();
    stale.input = other;
    assert_eq!(
        stages
            .stage(json!({"stage":"native_label_groups","column":stale}))
            .unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let Attachment::TargetColumn(owner) = &stages.held[&column.owner] else {
        panic!("target owner")
    };
    let selected = owner.selection().unwrap().batches[0]
        .column(0)
        .as_boolean()
        .iter()
        .enumerate()
        .filter_map(|(i, selected)| (selected == Some(true)).then_some(i as u64))
        .collect::<Vec<_>>();
    assert!(!selected.is_empty());
    let converted = native(
        &mut stages,
        selected.clone(),
        vec![3; selected.len()],
        vec![true; selected.len()],
    );
    let PreparedMaterializeValue::TargetColumn(column) = stages
        .stage(json!({"stage":"windowed_target","column":column,"converted":converted}))
        .unwrap()
    else {
        panic!("windowed target")
    };
    let PreparedMaterializeValue::LabelGroups(groups) = stages
        .stage(json!({"stage":"native_label_groups","column":column}))
        .unwrap()
    else {
        panic!("groups")
    };
    let Attachment::MaterializeGroups(owner) = &stages.held[&groups.owner] else {
        panic!("group owner")
    };
    let rows = owner.rows().unwrap().batches[0].clone();
    let arrow_schema::DataType::List(field) = rows.column(2).data_type() else {
        panic!("actual groups must retain their scalar list type")
    };
    let mut labels = ListBuilder::new(Int64Builder::new()).with_field(Arc::clone(field));
    for _ in 0..rows.num_rows() {
        labels.values().append_value(1);
        labels.append(true);
    }
    let changed = RecordBatch::try_from_iter([
        ("entity", Arc::clone(rows.column(0))),
        ("anchor", Arc::clone(rows.column(1))),
        ("target", Arc::new(labels.finish()) as ArrayRef),
    ])
    .unwrap();
    assert_eq!(changed.column(2).data_type(), rows.column(2).data_type());
    assert_ne!(changed.column(2).to_data(), rows.column(2).to_data());
    stages.held.insert(
        AttachmentId(50_000),
        Attachment::Arrow(data(changed.clone())),
    );
    stages.held.insert(
        AttachmentId(50_001),
        Attachment::Arrow(data(changed.slice(0, changed.num_rows() - 1))),
    );
    let complete = |rows| json!({"stage":"complete_label_batch","population":population,"groups":groups,"restored":{"rows":rows,"encoding":null}});
    assert_eq!(
        stages.stage(complete(50_001)).unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let PreparedMaterializeValue::LabelBatch(batch) = stages.stage(complete(50_000)).unwrap()
    else {
        panic!("complete batch")
    };
    let PreparedMaterializeValue::Request(Materialized { state, query }) = stages
        .stage(json!({"stage":"request","population":population,"batches":[batch],"values":null}))
        .unwrap()
    else {
        panic!("actual nested request")
    };
    assert_eq!(query.unwrap().entity_table, "users");
    let Attachment::Bytes(bytes) = &stages.held[&state.payload] else {
        panic!("payload")
    };
    let payload: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(payload["task"]["kind"], "temporal_link_prediction");
    let context = &payload["context"]["instance_table"];
    let target = context["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|name| name == "TARGET")
        .unwrap();
    assert_eq!(context["rows"].as_array().unwrap().len(), 3);
    assert!(
        context["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row[target] == json!([1]))
    );
    assert_eq!(
        payload["predict"]["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}
