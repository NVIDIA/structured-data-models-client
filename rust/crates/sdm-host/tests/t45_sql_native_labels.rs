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
    returned: Vec<(FetchStep, ArrowData)>,
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
                        self.returned.push((request.clone(), rows.clone()));
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

fn complete_entity_inputs(
    stages: &mut Stages,
    mut inputs: sdm_contract::materialize::PreparedLabelInputs,
) -> sdm_contract::materialize::PreparedLabelInputs {
    let views = inputs
        .eligibility
        .as_ref()
        .expect("required entity history")
        .columns
        .clone();
    for view in views {
        assert_eq!(
            view.form,
            sdm_contract::materialize::TargetColumnForm::Condition
        );
        let Attachment::LabelInputs(owner) = &stages.held[&view.owner] else {
            panic!("actual entity column view")
        };
        let rows = owner.rows().unwrap().batches[0].clone();
        let target = rows.column(2).clone();
        let presence = (0..target.len())
            .map(|row| target.is_valid(row))
            .collect::<Vec<_>>();
        let values =
            RecordBatch::try_from_iter([("position", rows.column(0).clone()), ("target", target)])
                .unwrap();
        let present = RecordBatch::try_from_iter([(
            "present",
            Arc::new(arrow_array::BooleanArray::from(presence)) as ArrayRef,
        )])
        .unwrap();
        let next = stages
            .held
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .unwrap();
        stages.held.insert(
            AttachmentId(next),
            Attachment::Arrow(ArrowData::new(values.schema(), vec![values])),
        );
        stages.held.insert(
            AttachmentId(next + 1),
            Attachment::Arrow(ArrowData::new(present.schema(), vec![present])),
        );
        let PreparedMaterializeValue::TargetColumn(column)=stages.stage(json!({"stage":"target_column","inputs":view,"converted":{"rows":next,"presence":next+1,"encoding":null}})).unwrap() else {panic!("completed entity values")};
        let PreparedMaterializeValue::LabelInputs(completed) = stages
            .stage(json!({"stage":"eligibility_column","inputs":inputs,"column":column}))
            .unwrap()
        else {
            panic!("completed entity history")
        };
        inputs = completed;
    }
    inputs
}

fn rebind_inputs(
    value: &mut sdm_contract::materialize::PreparedLabelInputs,
    keep: &mut impl FnMut(&mut AttachmentId),
) {
    keep(&mut value.owner);
    if let Some(history) = &mut value.eligibility {
        for column in &mut history.columns {
            rebind_inputs(column, keep);
        }
        for column in &mut history.values {
            keep(&mut column.owner);
        }
    }
}

fn rebind_groups(
    value: &mut sdm_contract::frames::materialize::PreparedFrameLabelGroups,
    keep: &mut impl FnMut(&mut AttachmentId),
) {
    keep(&mut value.owner);
    if let Some(restored) = &mut value.restored {
        keep(&mut restored.rows);
    }
    if let Some(history) = &mut value.what_if {
        for column in &mut history.columns {
            rebind_inputs(column, keep);
        }
        for column in &mut history.values {
            keep(&mut column.owner);
        }
    }
}

fn rebind(value: &mut PreparedMaterializeValue, keep: &mut impl FnMut(&mut AttachmentId)) {
    match value {
        PreparedMaterializeValue::LookupRows { positions } => keep(positions),
        PreparedMaterializeValue::Source(value) => keep(&mut value.owner),
        PreparedMaterializeValue::Population(value) => keep(&mut value.owner),
        PreparedMaterializeValue::ColumnMeaning(_) => {}
        PreparedMaterializeValue::LabelInputs(value) => rebind_inputs(value, keep),
        PreparedMaterializeValue::TargetColumn(value) => {
            rebind_inputs(&mut value.input, keep);
            keep(&mut value.owner);
            for operand in &mut value.condition_values {
                keep(&mut operand.converted.rows);
                if let Some(id) = &mut operand.converted.presence {
                    keep(id);
                }
            }
        }
        PreparedMaterializeValue::LabelGroups(value) => rebind_groups(value, keep),
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
            returned: vec![],
        },
        json!({"stage":"task","request":{"provider":{"provider":"sql","dialect":"duckdb"},"schema":1,"query":2,"indices":3,"seed":"0x7","values":5,"options":[],"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"}}}),
    )
}

#[test]
fn stored_time_spelling_probe_is_not_a_retained_native_source_read() {
    let (mut stages, mut task) = fixture();
    task["request"]["provider"]["dialect"] = json!("sqlite");
    let PreparedMaterializeValue::Source(source) = stages.stage(task).unwrap() else {
        panic!("actual source")
    };
    let PreparedMaterializeValue::Population(population) = stages
        .stage(json!({"stage":"population","source":source}))
        .unwrap()
    else {
        panic!("actual population")
    };
    let PreparedMaterializeValue::LabelInputs(inputs) = stages
        .stage(json!({"stage":"label_inputs","population":population,
            "range":{"start":0,"stop":population.rows},"previous":null}))
        .unwrap()
    else {
        panic!("actual label inputs")
    };
    let probe = stages.issued.iter().find(|step| {
        step["request"]["request"] == "scan" && step["request"]["limit"] == 1
    }).expect("the actual SQLite stored-time spelling probe");
    assert_eq!(probe["purpose"], "locate");
    let Attachment::LabelInputs(owner) = &stages.held[&inputs.owner] else {
        panic!("actual retained input owner")
    };
    let views = owner.temporal_sources().unwrap();
    assert_eq!(views.len(), 2);
    for (descriptor, _) in views {
        let value: Value = serde_json::from_slice(&descriptor).unwrap();
        assert_ne!(value["source_read"], probe["partition"]["first"]);
    }
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
        times: None,
    }
}

fn native_times(stages: &mut Stages, values: ArrayRef) -> AttachmentId {
    let id = AttachmentId(stages.held.keys().map(|id| id.0).max().unwrap() + 1);
    let rows = RecordBatch::try_from_iter([("time", values)]).unwrap();
    stages.held.insert(id, Attachment::Arrow(data(rows)));
    id
}

#[test]
fn sql_temporal_sources_retain_whole_issued_reads_before_native_completion() {
    let (mut stages, task) = fixture();
    let PreparedMaterializeValue::Source(source) = stages.stage(task).unwrap() else {
        panic!("actual source")
    };
    let tables = source.tables.clone();
    let PreparedMaterializeValue::Population(population) = stages
        .stage(json!({"stage":"population","source":source}))
        .unwrap()
    else {
        panic!("actual population")
    };
    let PreparedMaterializeValue::LabelInputs(inputs) = stages
        .stage(json!({"stage":"label_inputs","population":population,
            "range":{"start":0,"stop":population.rows},"previous":null}))
        .unwrap()
    else {
        panic!("actual inputs before native completion")
    };
    assert!(inputs.eligibility.as_ref().unwrap().values.is_empty());
    let Attachment::LabelInputs(owner) = &stages.held[&inputs.owner] else {
        panic!("actual retained inputs")
    };
    let fetched_before = stages.issued.len();
    let views = owner.temporal_sources().unwrap();
    let repeated = owner.temporal_sources().unwrap();
    assert_eq!(stages.issued.len(), fetched_before);
    assert_eq!(views.len(), 2);
    assert_eq!(repeated.len(), views.len());
    let mut seen = Vec::new();
    for ((descriptor, rows), (again, repeated_rows)) in views.iter().zip(&repeated) {
        assert_eq!(descriptor, again);
        let descriptor: Value = serde_json::from_slice(descriptor).unwrap();
        let role = descriptor["source_role"].as_str().unwrap();
        let read = u32::try_from(descriptor["source_read"].as_u64().unwrap()).unwrap();
        let (name, request) = match role {
            "entity" => ("users", "sample_entities"),
            "windowed" => ("buys", "neighbors"),
            other => panic!("unexpected source role {other}"),
        };
        let logical = json!({"source":["main",name]});
        assert_eq!(descriptor["source"], logical);
        assert!(!seen.contains(&role.to_owned()));
        seen.push(role.to_owned());
        let table = tables
            .iter()
            .find(|table| json!(table.source) == logical)
            .unwrap();
        let physical = serde_json::to_value(&table.read).unwrap();
        let batches = stages
            .returned
            .iter()
            .filter(|(fetch, _)| {
                let asked = serde_json::to_value(&fetch.request).unwrap();
                fetch
                    .partition
                    .as_ref()
                    .is_some_and(|part| part.first.0 == read)
                    && asked["request"] == request
                    && asked["table"] == physical
            })
            .flat_map(|(_, rows)| rows.batches.iter())
            .collect::<Vec<_>>();
        assert!(!batches.is_empty(), "view must name a real issued read");
        let fetched = arrow_select::concat::concat_batches(&batches[0].schema(), batches).unwrap();
        let columns = fetched
            .schema()
            .fields()
            .iter()
            .filter(|field| {
                table.temporal_columns.contains(field.name())
                    && matches!(field.data_type(), arrow_schema::DataType::Timestamp(_, _))
            })
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            descriptor,
            json!({"source_role":role,"source_read":read,
            "source":logical,"columns":columns,"rows":fetched.num_rows()})
        );
        let view = arrow_select::concat::concat_batches(&rows.schema, &rows.batches).unwrap();
        let again =
            arrow_select::concat::concat_batches(&repeated_rows.schema, &repeated_rows.batches)
                .unwrap();
        assert_eq!(view, again);
        assert_eq!(view.num_rows(), fetched.num_rows());
        assert_eq!(
            view.schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect::<Vec<_>>(),
            [vec!["source_row".to_owned()], columns.clone()].concat()
        );
        assert_eq!(
            view.column(0)
                .as_primitive::<UInt64Type>()
                .iter()
                .collect::<Vec<_>>(),
            (0..fetched.num_rows())
                .map(|row| Some(row as u64))
                .collect::<Vec<_>>()
        );
        for column in &columns {
            assert_eq!(
                view.column_by_name(column).unwrap().to_data(),
                fetched.column_by_name(column).unwrap().to_data()
            );
        }
        if role == "entity" {
            // All eight fetched rows precede candidate selection and the
            // requested three-context limit. No time column was requested.
            assert_eq!(fetched.num_rows(), 8);
            assert!(columns.is_empty());
        } else {
            assert_eq!(Some(read), inputs.source_read.map(|id| id.0));
            assert_eq!(columns, ["ts"]);
        }
    }
    seen.sort();
    assert_eq!(seen, ["entity", "windowed"]);
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
    let inputs = complete_entity_inputs(&mut stages, inputs);
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
    let mut short = native(
        &mut stages,
        positions[1..].to_vec(),
        vec![3; positions.len() - 1],
        vec![true; positions.len()],
    );
    short.times = Some(native_times(
        &mut stages,
        Arc::clone(source_rows.column_by_name("time").unwrap()),
    ));
    assert_eq!(
        stages
            .stage(json!({"stage":"target_column","inputs":inputs,"converted":short}))
            .unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let fetched_before_time = stages.issued.len();
    for times in [
        None,
        Some(
            source_rows
                .column_by_name("time")
                .unwrap()
                .slice(1, positions.len() - 1),
        ),
        Some(ints(vec![1; positions.len()])),
    ] {
        let mut bad = native(
            &mut stages,
            positions.clone(),
            vec![3; positions.len()],
            vec![true; positions.len()],
        );
        bad.times = times.map(|values| native_times(&mut stages, values));
        assert_eq!(
            stages
                .stage(json!({"stage":"target_column","inputs":inputs,"converted":bad}))
                .unwrap_err()["error"]["code"],
            "INVALID_MESSAGE"
        );
    }
    assert_eq!(stages.issued.len(), fetched_before_time);
    let mut missing_times = native(
        &mut stages,
        positions.clone(),
        vec![3; positions.len()],
        vec![true; positions.len()],
    );
    missing_times.times = Some(native_times(
        &mut stages,
        arrow_array::new_null_array(
            source_rows.column_by_name("time").unwrap().data_type(),
            positions.len(),
        ),
    ));
    let PreparedMaterializeValue::TargetColumn(changed_time) = stages
        .stage(json!({"stage":"target_column","inputs":inputs,"converted":missing_times}))
        .unwrap()
    else {
        panic!("changed captured times")
    };
    let Attachment::TargetColumn(changed_time) = &stages.held[&changed_time.owner] else {
        panic!("changed time owner")
    };
    assert!(
        changed_time.selection().unwrap().batches[0]
            .column(0)
            .as_boolean()
            .iter()
            .all(|value| value == Some(false))
    );
    let mut converted = native(
        &mut stages,
        positions.clone(),
        vec![3; positions.len()],
        vec![true; positions.len()],
    );
    converted.times = Some(native_times(
        &mut stages,
        Arc::clone(source_rows.column_by_name("time").unwrap()),
    ));
    let PreparedMaterializeValue::TargetColumn(column) = stages
        .stage(json!({"stage":"target_column","inputs":inputs,"converted":converted}))
        .unwrap()
    else {
        panic!("target")
    };
    let PreparedMaterializeValue::LabelInputs(other) = stages.stage(request).unwrap() else {
        panic!("other inputs")
    };
    let other = complete_entity_inputs(&mut stages, other);
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
    let complete = |rows| json!({"stage":"restore_label_groups","groups":groups,"restored":{"rows":rows,"encoding":null}});
    assert_eq!(
        stages.stage(complete(50_001)).unwrap_err()["error"]["code"],
        "INVALID_MESSAGE"
    );
    let PreparedMaterializeValue::LabelGroups(restored) = stages.stage(complete(50_000)).unwrap()
    else {
        panic!("actual restored groups")
    };
    assert!(restored.restored.is_some());
    assert!(restored.what_if.is_none());
    let PreparedMaterializeValue::LabelBatch(batch)=stages.stage(json!({"stage":"complete_label_batch","population":population,"groups":restored,"restored":null})).unwrap() else {panic!("completed batch")};
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
            .all(|row| row[target] == json!(["1"]))
    );
    assert_eq!(
        payload["predict"]["instance_table"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}
