// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `materialize` on the core, through `sdm_core::start`, with a host
//! provider whose answers the tests give: a seed of any size reaches the
//! sample it seeds as that exact integer, and DuckDB's data layer refuses
//! one its parser refuses (RS-7); the sampler's index check warns of the
//! keys no index covers; and what the core does not materialize yet is
//! refused by the rule's name. `sdm-testkit`'s `duckdb_provider.rs` runs
//! the whole of `base.regression` on DuckDB.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{
    AttachmentId, DataRequest, DescribePart, Determinism, Dialect, OptionalRequest,
    ProviderCapabilities, ProviderChoice,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, Materialize, Operation, Seed,
    SourceDescription, Stype,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, CoreError, Detail, ErrorCode, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{
    ArrowData, Attachment, Attachments, CompositionRoot, FetchResult, Run, UnknownZone, Zones,
    start,
};

struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 0,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
    }
}

fn column(
    name: &str,
    received: ColumnType,
    dtype: Dtype,
    stype: Stype,
    unique: bool,
) -> DescribedColumn {
    DescribedColumn {
        name: name.into(),
        received,
        dtype,
        stype,
        nullable: !unique,
        unique,
    }
}

/// `base.regression`'s shape: users, and orders referring to them by
/// `user_id`; each key unique, the foreign key not.
fn shop() -> SourceDescription {
    let ts = ColumnType::Timestamp {
        unit: Unit::Us,
        zone: None,
    };
    SourceDescription {
        tables: vec![
            DescribedTable {
                name: "orders".into(),
                columns: vec![
                    column("order_id", ColumnType::Int64, Dtype::Int, Stype::Id, true),
                    column("user_id", ColumnType::Int64, Dtype::Int, Stype::Id, false),
                    column(
                        "price",
                        ColumnType::Float64,
                        Dtype::Float,
                        Stype::Numerical,
                        false,
                    ),
                    column("ts", ts, Dtype::Date, Stype::Timestamp, false),
                ],
                primary_key: vec!["order_id".into()],
                time_column: Some("ts".into()),
                duplicate_keys: None,
            },
            DescribedTable {
                name: "users".into(),
                columns: vec![
                    column("user_id", ColumnType::Int64, Dtype::Int, Stype::Id, true),
                    column(
                        "country",
                        ColumnType::Utf8,
                        Dtype::String,
                        Stype::Categorical,
                        false,
                    ),
                ],
                primary_key: vec!["user_id".into()],
                time_column: None,
                duplicate_keys: None,
            },
        ],
        links: vec![vec![
            ColumnRef {
                table: "orders".into(),
                column: "user_id".into(),
            },
            ColumnRef {
                table: "users".into(),
                column: "user_id".into(),
            },
        ]],
    }
}

/// A host provider declaring the requests materialize asks.
fn host() -> ProviderChoice {
    ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            optional: BTreeSet::from([
                OptionalRequest::TimeBounds,
                OptionalRequest::SampleEntities,
                OptionalRequest::Neighbors,
                OptionalRequest::Lookup,
            ]),
            max_keys_per_request: None,
            sampling: Determinism::SeededStable,
        },
    }
}

const QUERY: &str = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";

/// 2024-03-01, nanoseconds since the epoch.
const ANCHOR: i64 = 1_709_251_200_000_000_000;

fn materialize(
    provider: ProviderChoice,
    description: &SourceDescription,
    query: &str,
    seed: Option<Seed>,
) -> Result<Run, CoreError> {
    let operation = Operation::Materialize(Materialize {
        provider,
        schema: AttachmentId(1),
        query: AttachmentId(2),
        indices: Some(AttachmentId(3)),
        anchor_time: Some(Scalar::Timestamp {
            unit: Unit::Ns,
            zone: None,
            value: ANCHOR,
        }),
        seed,
    });
    let indices = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "indices",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef],
    )
    .unwrap();
    let attachments = Attachments::from([
        (
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(description).unwrap()),
        ),
        (
            AttachmentId(2),
            Attachment::Bytes(serde_json::to_vec(query).unwrap()),
        ),
        (
            AttachmentId(3),
            Attachment::Arrow(ArrowData::new(indices.schema(), vec![indices])),
        ),
    ]);
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::new()));
    start(
        &root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
        Arc::new(NoZones),
    )
}

fn data(columns: Vec<(&str, ArrayRef)>) -> FetchResult {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap();
    FetchResult::Data(ArrowData::new(batch.schema(), vec![batch]))
}

/// Answers the catalog requests materialize asks before it samples: no
/// index, and orders' times from 2023-01-01 to 2024-07-07.
fn catalog(request: &DataRequest) -> Option<FetchResult> {
    let us =
        |text: i64| -> ArrayRef { Arc::new(TimestampMicrosecondArray::from(vec![Some(text)])) };
    match request {
        DataRequest::Describe {
            part: DescribePart::Indexes,
            ..
        } => Some(data(vec![
            (
                "table_name",
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ),
            (
                "expressions",
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ),
        ])),
        DataRequest::TimeBounds { .. } => Some(data(vec![
            (
                "table_index",
                Arc::new(arrow_array::Int32Array::from(vec![0])) as ArrayRef,
            ),
            ("min_date", us(1_672_531_200_000_000)),
            ("max_date", us(1_720_310_400_000_000)),
        ])),
        _ => None,
    }
}

/// Runs `run`, answering what `catalog` answers, to the first request it
/// does not, which it returns, with the warnings given on the way.
fn to_the_sample(run: &mut Run) -> (DataRequest, Vec<String>) {
    let mut warnings = Vec::new();
    loop {
        let step = run.next().unwrap();
        warnings.extend(run.take_warnings().into_iter().map(|w| w.code));
        let Step::Fetch { requests } = step else {
            panic!("not a fetch: {step:?}");
        };
        for asked in requests {
            match catalog(&asked.request) {
                Some(answer) => run.submit_fetch(asked.id, answer).unwrap(),
                None => return (asked.request, warnings),
            }
        }
    }
}

#[test]
fn a_seed_of_2_to_the_70_reaches_the_sample_as_that_exact_integer() {
    let seed = Seed::from(1_u128 << 70);
    assert_eq!(String::from(seed.clone()), "0x400000000000000000");
    let mut run = materialize(host(), &shop(), QUERY, Some(seed)).unwrap();
    let (request, warnings) = to_the_sample(&mut run);
    let DataRequest::SampleEntities { seed, n, .. } = request else {
        panic!("not the sample: {request:?}");
    };
    assert_eq!(
        seed,
        Scalar::Decimal128 {
            precision: 38,
            scale: 0,
            value: 1_i128 << 70,
        }
    );
    assert_eq!(n, 10_000);
    // The foreign key is not one the source keys, and no index covers it.
    assert_eq!(warnings, ["MISSING_INDICES"]);

    // A seed a u64 holds is one; none is the reference's default, 42.
    for (given, taken) in [
        (Some(Seed::from(7_u128)), Scalar::UInt64(7)),
        (None, Scalar::UInt64(42)),
    ] {
        let mut run = materialize(host(), &shop(), QUERY, given).unwrap();
        let (request, _) = to_the_sample(&mut run);
        let DataRequest::SampleEntities { seed, .. } = request else {
            panic!("not the sample");
        };
        assert_eq!(seed, taken);
    }
}

#[test]
fn duckdb_refuses_a_seed_its_parser_refuses_by_name() {
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::Duckdb,
    };
    for (seed, refused) in [
        ((1_u128 << 31) - 1, false),
        (1 << 31, true),
        (1 << 70, true),
    ] {
        let mut run = materialize(duckdb.clone(), &shop(), QUERY, Some(Seed::from(seed))).unwrap();
        let mut outcome = None;
        for _ in 0..4 {
            match run.next() {
                Err(err) => {
                    outcome = Some(Err(err));
                    break;
                }
                Ok(Step::Fetch { requests }) => {
                    let mut sample = false;
                    for asked in requests {
                        if let DataRequest::SampleEntities { .. } = asked.request {
                            sample = true;
                            break;
                        }
                        let answer = catalog(&asked.request).unwrap();
                        run.submit_fetch(asked.id, answer).unwrap();
                    }
                    if sample {
                        outcome = Some(Ok(()));
                        break;
                    }
                }
                Ok(other) => panic!("{other:?}"),
            }
        }
        match outcome.unwrap() {
            Ok(()) => assert!(!refused, "{seed} was sampled"),
            Err(err) => {
                assert!(refused, "{seed}: {err:?}");
                assert_eq!(err.code, ErrorCode::InvalidRequest);
                assert_eq!(err.fields["request"], Detail::text("sample_entities"));
                assert_eq!(err.fields["field"], Detail::text("seed"));
            }
        }
    }
}

/// The error `run` ends with, answering the catalog requests it asks
/// first: the index check comes before the task is read, as the
/// reference builds its sampler before its task table.
fn refusal(run: &mut Run) -> CoreError {
    loop {
        match run.next() {
            Err(err) => return err,
            Ok(Step::Fetch { requests }) => {
                for asked in requests {
                    let answer = catalog(&asked.request).expect("a catalog request");
                    run.submit_fetch(asked.id, answer).unwrap();
                }
            }
            Ok(other) => panic!("{other:?}"),
        }
    }
}

fn rule(err: &CoreError) -> String {
    assert_eq!(err.code, ErrorCode::UnsupportedRequest, "{err:?}");
    assert_eq!(err.fields["operation"], Detail::text("materialize"));
    match &err.fields["rule"] {
        Detail::Scalar(Scalar::Utf8(rule)) => rule.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn what_the_core_does_not_materialize_yet_is_refused_by_name() {
    for (query, wanted) in [
        (
            "PREDICT AVG(orders.price, 0, 30, days) FOR EACH users.user_id",
            "aggregation",
        ),
        (
            "PREDICT SUM(orders.price, 0, 2, months) FOR EACH users.user_id",
            "month_window",
        ),
        (
            "PREDICT users.country FOR EACH users.user_id",
            "static_target",
        ),
        (
            "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id WHERE users.country = 'US'",
            "entity_filter",
        ),
    ] {
        let mut run = materialize(host(), &shop(), query, None).unwrap();
        let err = refusal(&mut run);
        assert_eq!(err.code, ErrorCode::UnsupportedRequest, "{query}: {err:?}");
        assert_eq!(rule(&err), wanted, "{query}");
    }
    // A table the entity table refers to, which this build does not sample.
    let mut parent = shop();
    parent.tables[1].columns.push(column(
        "country_code",
        ColumnType::Utf8,
        Dtype::String,
        Stype::Id,
        false,
    ));
    parent.tables.push(DescribedTable {
        name: "countries".into(),
        columns: vec![column(
            "code",
            ColumnType::Utf8,
            Dtype::String,
            Stype::Id,
            true,
        )],
        primary_key: vec!["code".into()],
        time_column: None,
        duplicate_keys: None,
    });
    parent.links.push(vec![
        ColumnRef {
            table: "users".into(),
            column: "country_code".into(),
        },
        ColumnRef {
            table: "countries".into(),
            column: "code".into(),
        },
    ]);
    let mut run = materialize(host(), &parent, QUERY, None).unwrap();
    assert_eq!(rule(&refusal(&mut run)), "parent_table");
    // A key of several columns.
    let mut composite = shop();
    composite.tables[1].primary_key = vec!["user_id".into(), "country".into()];
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    assert_eq!(rule(&refusal(&mut run)), "composite_key");
}
