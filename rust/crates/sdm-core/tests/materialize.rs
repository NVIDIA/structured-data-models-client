// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `materialize` on the core, through `sdm_core::start`, with a host
//! provider whose answers the tests give: a seed of any size reaches the
//! sample it seeds as that exact integer, and the core refuses one past the
//! largest the provider declares, DuckDB's from 2**31 among them (RS-7);
//! the sampler's index check warns of the
//! keys no index covers; and what the core does not materialize yet is
//! refused by the rule's name. `sdm-testkit`'s `duckdb_provider.rs` runs
//! the whole of `base.regression` on DuckDB.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Int64Array, ListArray, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    AttachmentId, BATCH_COLUMN, DataLayer, DataRequest, DescribePart, Determinism, Dialect,
    FetchPlan, IndexAdvice, MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderCapabilities,
    ProviderChoice, TextTimes, catalog_columns,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, Materialize, Operation,
    SAMPLER_STATE_VERSION, SamplerMemo, SamplerState, Seed, SourceDescription, Stype,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, CoreError, Detail, ErrorCode, HeldState, HostFacts, Scalar, Step,
    Unit,
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
        date_literal_range: None,
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
        stype_before_key: None,
        nullable: !unique,
        unique,
        declared_type: None,
        exact_number: None,
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
                source: "orders".into(),
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
                time_column_inferred: None,
                duplicate_keys: None,
            },
            DescribedTable {
                name: "users".into(),
                source: "users".into(),
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
                time_column_inferred: None,
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
        namespace: None,
    }
}

/// `description` as a host holds it, each column with its facts (T3.16).
/// `description` as a host holds it once described: each table kept as the
/// description read it, its columns with what the source said of each, as
/// a described schema keeps every table its engine read, so a run reads
/// none of them anew (T3.4 fixes 12 and 13).
fn held_of(description: &SourceDescription) -> sdm_contract::ops::HeldSchema {
    let mut held = sdm_schema::held::held(
        description,
        &BTreeSet::new(),
        &std::collections::BTreeMap::new(),
    );
    held.read_under = held
        .tables
        .iter()
        .map(|table| sdm_contract::ops::ReadUnder {
            relation: table.source.clone(),
            namespace: Vec::new(),
            columns: table
                .columns
                .iter()
                .filter_map(|column| {
                    Some(sdm_contract::ops::KeptColumn {
                        name: column.name.clone(),
                        source: column.source.clone()?,
                    })
                })
                .collect(),
        })
        .collect();
    held
}

/// A host provider declaring the requests materialize asks.
fn host() -> ProviderChoice {
    ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            derived_relations: false,
            neighbor_lookback: false,
            keys_optional: false,
            sample_rows: false,
            optional: BTreeSet::from([
                OptionalRequest::TimeBounds,
                OptionalRequest::SampleEntities,
                OptionalRequest::Neighbors,
                OptionalRequest::Lookup,
            ]),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: Some(NullOrder::Last),
            index_advice: Some(IndexAdvice::SingleColumn),
            sampling: Determinism::SeededStable,
            fallback: None,
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
    let (operation, attachments) = materialize_start(provider, description, query, seed);
    materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments)
}

/// The operation and attachments of a `materialize` start.
fn materialize_start(
    provider: ProviderChoice,
    description: &SourceDescription,
    query: &str,
    seed: Option<Seed>,
) -> (Operation, Attachments) {
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
        unseeded: false,
        anchor_entity: false,
        values: None,
        options: vec![],
        most_parts: None,
        most_entities: None,
        most_calls: None,
        candidates: None,
        sampler_state: None,
        progress: false,
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
            Attachment::Bytes(serde_json::to_vec(&held_of(description)).unwrap()),
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
    (operation, attachments)
}

fn materialize_on(
    layer: Arc<dyn DataLayer>,
    operation: Operation,
    attachments: Attachments,
) -> Result<Run, CoreError> {
    let root = CompositionRoot::new(layer);
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
            // A host provider answers in the contract's catalog shape, in
            // no dialect's (decision E54).
            (
                catalog_columns::TABLE,
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ),
            (
                catalog_columns::COLUMNS,
                Arc::new(ListArray::new_null(
                    Arc::new(Field::new(catalog_columns::LIST_ITEM, DataType::Utf8, true)),
                    0,
                )),
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

/// What [`catalog`] answers, but the indexes in DuckDB's own catalog
/// shape, which a DuckDB plan expects and `sdm-data` reads (decision E54).
fn duckdb_catalog(request: &DataRequest) -> Option<FetchResult> {
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
        _ => catalog(request),
    }
}

/// Runs `run`, answering what `catalog` answers, to the first request it
/// does not, which it returns, with the warnings given on the way.
fn to_the_sample(run: &mut Run) -> (DataRequest, Vec<String>) {
    let mut warnings = Vec::new();
    loop {
        let step = run.next().unwrap();
        warnings.extend(
            run.take_warnings()
                .into_iter()
                .map(|w| w.code)
                .filter(|c| !REQUEST_NOTICES.contains(&c.as_str())),
        );
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
fn a_seed_of_2_to_the_70_is_past_every_range_a_provider_declares() {
    // Contract v26 (design choice 1): every provider declares its largest
    // seed, a u64, with no "any seed" default, so a seed of 2**70 is past
    // the widest range one can declare, refused by RS-7's rule before the
    // sample, by name, never sampled under another seed.
    let seed = Seed::from(1_u128 << 70);
    assert_eq!(String::from(seed.clone()), "0x400000000000000000");
    let run = materialize(host_up_to(u64::MAX), &shop(), QUERY, Some(seed)).unwrap();
    let err = at_the_sample(run, catalog).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("seed"));
    assert_eq!(
        err.fields["max_seed"],
        Detail::Scalar(Scalar::UInt64(u64::MAX))
    );

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

/// How `run` ends at the sample: `Ok` when the sample is asked for, the
/// error otherwise, answering the catalog requests before it with
/// `answer`.
fn at_the_sample(
    mut run: Run,
    answer: fn(&DataRequest) -> Option<FetchResult>,
) -> Result<DataRequest, CoreError> {
    for _ in 0..4 {
        match run.next() {
            Err(err) => return Err(err),
            Ok(Step::Fetch { requests }) => {
                for asked in requests {
                    if let DataRequest::SampleEntities { .. } = asked.request {
                        return Ok(asked.request);
                    }
                    let answer = answer(&asked.request).unwrap();
                    run.submit_fetch(asked.id, answer).unwrap();
                }
            }
            Ok(other) => panic!("{other:?}"),
        }
    }
    panic!("the run never reached the sample");
}

fn duckdb() -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
        namespace: Vec::new(),
    }
}

#[test]
fn the_core_refuses_a_seed_past_the_largest_duckdb_declares() {
    for (seed, refused) in [
        ((1_u128 << 31) - 1, false),
        (1 << 31, true),
        (1 << 70, true),
        // Past what a seed's scalar carries, RS-7's refusal all the same
        // (T1.36 fix 9).
        (10_u128.pow(38), true),
        (u128::MAX, true),
    ] {
        let run = materialize(duckdb(), &shop(), QUERY, Some(Seed::from(seed))).unwrap();
        match at_the_sample(run, duckdb_catalog) {
            Ok(_) => assert!(!refused, "{seed} was sampled"),
            Err(err) => {
                assert!(refused, "{seed}: {err:?}");
                assert_eq!(err.code, ErrorCode::InvalidRequest);
                assert_eq!(err.fields["request"], Detail::text("sample_entities"));
                assert_eq!(err.fields["field"], Detail::text("seed"));
                // The core's refusal, read from the capability, naming the
                // largest seed and no provider.
                assert_eq!(
                    err.fields["max_seed"],
                    Detail::Scalar(Scalar::UInt64((1 << 31) - 1))
                );
                assert!(!err.message.to_lowercase().contains("duckdb"), "{err:?}");
            }
        }
    }
}

/// A host provider declaring the requests materialize asks, and `max_seed`.
fn host_up_to(max_seed: u64) -> ProviderChoice {
    let ProviderChoice::Host { mut capabilities } = host() else {
        unreachable!()
    };
    capabilities.max_seed = max_seed;
    ProviderChoice::Host { capabilities }
}

#[test]
fn a_host_provider_s_declared_range_is_the_one_the_core_holds_it_to() {
    for (max_seed, seed, refused) in [
        (100, 100_u128, false),
        (100, 101, true),
        (0, 0, false),
        (0, 1, true),
        (u64::MAX, u128::from(u64::MAX), false),
        (u64::MAX, 1 << 64, true),
        (u64::MAX, 10_u128.pow(38), true),
    ] {
        let run =
            materialize(host_up_to(max_seed), &shop(), QUERY, Some(Seed::from(seed))).unwrap();
        match at_the_sample(run, catalog) {
            Ok(_) => assert!(!refused, "{seed} was sampled under {max_seed:?}"),
            Err(err) => {
                assert!(refused, "{seed} under {max_seed:?}: {err:?}");
                assert_eq!(err.code, ErrorCode::InvalidRequest);
                assert_eq!(err.fields["field"], Detail::text("seed"));
                assert_eq!(
                    err.fields["max_seed"],
                    Detail::Scalar(Scalar::UInt64(max_seed))
                );
            }
        }
    }
}

/// The shipped data layer, but for the seed range it gives DuckDB, which
/// it widens to every seed: a seeded fault in the capability.
struct Forgetful(sdm_data::Data);

impl DataLayer for Forgetful {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        let mut capabilities = self.0.capabilities(provider)?;
        capabilities.max_seed = u64::MAX;
        Ok(capabilities)
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        self.0.plan(request, provider)
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        self.0.read(request, provider, batches)
    }
}

#[test]
fn without_the_capability_no_invalid_request_is_given() {
    // Seeded: DuckDB's declared range widened, a seed of 2**31 is no
    // longer the caller's error: it reaches the dialect's rendering, which
    // never renders a seed its parser refuses and calls that the core's
    // error. So the refusal above is the capability's, not a fallback.
    let (operation, attachments) =
        materialize_start(duckdb(), &shop(), QUERY, Some(Seed::from(1_u128 << 31)));
    let layer: Arc<dyn DataLayer> = Arc::new(Forgetful(sdm_data::Data::new()));
    let forgetful = materialize_on(layer, operation, attachments).unwrap();
    let err = at_the_sample(forgetful, duckdb_catalog).unwrap_err();
    assert_eq!(err.code, ErrorCode::InternalError, "{err:?}");
    assert!(!err.fields.contains_key("max_seed"));
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
    // An entity filter and an assumption are lifted (T2.13b): the run
    // reaches its first read.
    for query in [
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id WHERE users.country = 'US'",
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id ASSUMING COUNT(orders.*, 0, 7, days) > 1",
    ] {
        let mut run = materialize(host(), &shop(), query, None).unwrap();
        assert!(matches!(run.next(), Ok(Step::Fetch { .. })), "{query}");
    }
    // A table the entity table refers to is sampled from T2.11: the run
    // reaches its first read.
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
        source: "countries".into(),
        columns: vec![column(
            "code",
            ColumnType::Utf8,
            Dtype::String,
            Stype::Id,
            true,
        )],
        primary_key: vec!["code".into()],
        time_column: None,
        time_column_inferred: None,
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
    assert!(matches!(run.next(), Ok(Step::Fetch { .. })));
    // A key of several columns, of a table no link refers to: refused as
    // the reference's SQL samplers refuse every such table, before anything
    // is read, where the reference first reads the source's fold of it
    // (KO2-68), an invalid request naming the table and its key's columns.
    let mut composite = shop();
    composite.tables[0].primary_key = vec!["order_id".into(), "user_id".into()];
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("schema"));
    assert_eq!(err.fields["table"], Detail::text("orders"));
    assert_eq!(
        err.fields["columns"],
        Detail::List(vec![
            Scalar::Utf8("order_id".into()),
            Scalar::Utf8("user_id".into())
        ])
    );
    assert!(!err.fields.contains_key("rule"), "{err:?}");
    // A float among the key's columns is refused before anything is read.
    let mut composite = shop();
    composite.tables[0].primary_key = vec!["order_id".into(), "price".into()];
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    // A query whose entity's table is keyed so is refused as the
    // reference's language refuses it, with no key, before anything is
    // read.
    let mut composite = shop();
    composite.tables[1].primary_key = vec!["user_id".into(), "age".into()];
    composite.links.clear();
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    // One a link refers to is the reference's refusal of a relationship
    // that cannot join, before anything is read (decision E55).
    let mut composite = shop();
    composite.tables[1].primary_key = vec!["user_id".into(), "country".into()];
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("relationships"));
}

/// The schema `materialize` is handed is held as the reference holds a
/// schema it makes a request of (`Schema.check_consistent`, decision E55),
/// before anything is read: a meaning the column's values cannot carry is
/// refused; the columns joined on are marked again from what the host
/// holds, so a column the schema no longer joins on means again what it
/// meant before.
#[test]
fn the_schema_is_held_to_the_reference_s_consistency_check() {
    fn price(schema: &mut SourceDescription) -> &mut DescribedColumn {
        schema.tables[0]
            .columns
            .iter_mut()
            .find(|c| c.name == "price")
            .unwrap()
    }
    // Text a float cannot be.
    let mut text = shop();
    price(&mut text).stype = Stype::Text;
    let mut run = materialize(host(), &text, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("schema"));
    assert_eq!(err.fields["table"], Detail::text("orders"));
    assert_eq!(err.fields["column"], Detail::text("price"));
    // An identifier the schema does not join on, which meant a number
    // before it was joined on: a number again, so the sum resolves and the
    // run goes on to its first read.
    let mut released = shop();
    price(&mut released).stype = Stype::Id;
    price(&mut released).stype_before_key = Some(Stype::Numerical);
    let mut run = materialize(host(), &released, QUERY, None).unwrap();
    assert!(matches!(run.next(), Ok(Step::Fetch { .. })));
    // The control: an identifier with nothing to give back stays one, and
    // a sum over it does not resolve.
    let mut kept = shop();
    price(&mut kept).stype = Stype::Id;
    let mut run = materialize(host(), &kept, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    // A time column the schema joins on, which keeps its identifier's
    // meaning: the engine's table refuses it, and so does the core, before
    // anything is read (it read the source's CAST of the key before).
    let mut joined = shop();
    joined.tables[0].time_column = Some("user_id".into());
    let mut run = materialize(host(), &joined, QUERY, None).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("time_column"));
    assert_eq!(err.fields["table"], Detail::text("orders"));
    assert_eq!(err.fields["column"], Detail::text("user_id"));
}

/// The tables of `description` named otherwise than the source names them:
/// orders read from `app_orders`, users from `app_users`.
fn renamed() -> SourceDescription {
    let mut renamed = shop();
    renamed.tables[0].source = "app_orders".into();
    renamed.tables[1].source = "app_users".into();
    renamed
}

/// The index answer: one index over `column` of each source table named.
fn indexes(on: &[(&str, &str)]) -> FetchResult {
    let tables: Vec<&str> = on.iter().map(|(t, _)| *t).collect();
    let mut builder =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new())
            .with_field(Arc::new(Field::new(
                catalog_columns::LIST_ITEM,
                DataType::Utf8,
                true,
            )));
    for (_, column) in on {
        builder.values().append_value(column);
        builder.append(true);
    }
    data(vec![
        (catalog_columns::TABLE, Arc::new(StringArray::from(tables))),
        (catalog_columns::COLUMNS, Arc::new(builder.finish())),
    ])
}

/// The notices of what the request leaves out (T1.28a), which these
/// sampler tests leave to the client's route tests: their sources leave
/// tables unreached.
const REQUEST_NOTICES: [&str; 5] = [
    sdm_core::materialize::CONTEXT_CAPPED,
    sdm_core::materialize::TABLES_UNLINKED,
    sdm_core::materialize::TABLES_UNREACHED,
    sdm_core::materialize::COLUMNS_UNREAD,
    sdm_core::materialize::CONTEXT_ANSWERS,
];

/// Every request `run` asks up to the sample, each answered as
/// [`catalog`] answers it, the indexes as `index_answer`, and the warnings
/// given on the way.
fn requests_to_the_sample(
    run: &mut Run,
    index_answer: &dyn Fn() -> FetchResult,
) -> (Vec<DataRequest>, Vec<String>) {
    let mut asked_for = Vec::new();
    let mut warnings = Vec::new();
    loop {
        let step = run.next().unwrap();
        warnings.extend(
            run.take_warnings()
                .into_iter()
                .map(|w| w.code)
                .filter(|c| !REQUEST_NOTICES.contains(&c.as_str())),
        );
        let Step::Fetch { requests } = step else {
            panic!("not a fetch: {step:?}");
        };
        for asked in requests {
            asked_for.push(asked.request.clone());
            let answer = match &asked.request {
                DataRequest::Describe {
                    part: DescribePart::Indexes,
                    ..
                } => index_answer(),
                other => match catalog(other) {
                    Some(answer) => answer,
                    None => return (asked_for, warnings),
                },
            };
            run.submit_fetch(asked.id, answer).unwrap();
        }
    }
}

/// A column of `n` rows of the type the shop's column `name` holds, its
/// values `values` where given (the keys), else nulls.
fn shop_column(name: &str, n: usize, values: Option<ArrayRef>) -> ArrayRef {
    if let Some(values) = values {
        return values;
    }
    match name {
        "order_id" | "user_id" => Arc::new(Int64Array::from(vec![None::<i64>; n])),
        "price" => Arc::new(arrow_array::Float64Array::from(vec![None::<f64>; n])),
        "ts" | "born" => Arc::new(TimestampMicrosecondArray::from(vec![None::<i64>; n])),
        _ => Arc::new(StringArray::from(vec![None::<&str>; n])),
    }
}

/// Runs `run` to its result over the renamed shop, answering every request:
/// the catalog as [`catalog`] does, the sample with four users, the labels'
/// and the first hop's neighbors with no row, the lookup with one row per
/// seed; each request's kind and the table it names, in order.
fn tables_to_the_result(run: &mut Run) -> Vec<(&'static str, String)> {
    steps_to_the_result(run, &|| indexes(&[]))
        .0
        .into_iter()
        .filter_map(|(request, _)| match request {
            DataRequest::Describe { .. } => None,
            DataRequest::TimeBounds { columns } => Some(("time_bounds", name(&columns[0].table))),
            DataRequest::SampleEntities { table, .. } => Some(("sample_entities", name(&table))),
            DataRequest::Neighbors { table, .. } => Some(("neighbors", name(&table))),
            DataRequest::Lookup { table, .. } => Some(("lookup", name(&table))),
            other => panic!("{other:?}"),
        })
        .collect()
}

/// The name of a relation of one part, as every request of a host giving
/// one part names its table (design choice 12).
fn name(relation: &sdm_contract::data::Relation) -> String {
    relation
        .single()
        .expect("a relation of one part")
        .to_owned()
}

/// Every request a run asked with its plan, in order, and the warnings it
/// gave.
type Steps = (Vec<(DataRequest, FetchPlan)>, Vec<String>);

/// Runs `run` to its result as [`tables_to_the_result`] does, the indexes
/// answered as `index_answer`: every request with its plan, in order, and
/// the warnings given on the way.
fn steps_to_the_result(run: &mut Run, index_answer: &dyn Fn() -> FetchResult) -> Steps {
    try_steps(run, index_answer).unwrap()
}

/// What [`steps_to_the_result`] gives, or the error the run ends with.
fn try_steps(run: &mut Run, index_answer: &dyn Fn() -> FetchResult) -> Result<Steps, CoreError> {
    try_steps_sampling(run, index_answer, &|columns| {
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2, 3]));
        data(
            columns
                .iter()
                .map(|c| {
                    let given = (c == "user_id").then(|| Arc::clone(&keys));
                    (c.as_str(), shop_column(c, 4, given))
                })
                .collect(),
        )
    })
}

/// [`try_steps`], the sample answered by `sample`, given the columns
/// asked for.
fn try_steps_sampling(
    run: &mut Run,
    index_answer: &dyn Fn() -> FetchResult,
    sample: &dyn Fn(&[String]) -> FetchResult,
) -> Result<Steps, CoreError> {
    let mut seen = Vec::new();
    let mut warnings = Vec::new();
    loop {
        let step = run.next()?;
        warnings.extend(
            run.take_warnings()
                .into_iter()
                .map(|w| w.code)
                .filter(|c| !REQUEST_NOTICES.contains(&c.as_str())),
        );
        let requests = match step {
            Step::Fetch { requests } => requests,
            Step::Done { .. } => return Ok((seen, warnings)),
            other => panic!("{other:?}"),
        };
        for asked in requests {
            seen.push((asked.request.clone(), asked.plan.clone()));
            let answer = match &asked.request {
                DataRequest::Describe { .. } => index_answer(),
                DataRequest::TimeBounds { .. } => catalog(&asked.request).unwrap(),
                DataRequest::SampleEntities { columns, .. } => sample(columns),
                DataRequest::Neighbors { columns, .. } => {
                    let mut answer: Vec<(&str, ArrayRef)> = columns
                        .iter()
                        .map(|c| (c.as_str(), shop_column(c, 0, None)))
                        .collect();
                    answer.push((BATCH_COLUMN, Arc::new(Int64Array::from(Vec::<i64>::new()))));
                    data(answer)
                }
                DataRequest::Lookup { columns, seeds, .. } => {
                    let Some(Attachment::Arrow(seeds)) = run.take_attachment(seeds.attachment)
                    else {
                        panic!("the lookup's seeds are attached");
                    };
                    let seeds = &seeds.batches[0];
                    let n = seeds.num_rows();
                    let mut answer: Vec<(&str, ArrayRef)> = columns
                        .iter()
                        .map(|c| {
                            let given = (c == "user_id").then(|| Arc::clone(seeds.column(0)));
                            (c.as_str(), shop_column(c, n, given))
                        })
                        .collect();
                    let rows: Vec<i64> = (0..n as i64).collect();
                    answer.push((BATCH_COLUMN, Arc::new(Int64Array::from(rows))));
                    answer.push((MATCHES_COLUMN, Arc::new(Int64Array::from(vec![1_i64; n]))));
                    data(answer)
                }
                other => panic!("{other:?}"),
            };
            run.submit_fetch(asked.id, answer).unwrap();
        }
    }
}

#[test]
fn a_table_named_otherwise_is_read_by_its_source_name() {
    // T1.24 (decision E61): a description may name a table otherwise than
    // its source does (contract version 12); every data request names the
    // source's table, as the reference's engine reads it, while the query
    // names the schema's. Until T1.24 it was refused by name
    // (`table_source`).
    let mut run = materialize(host(), &renamed(), QUERY, None).unwrap();
    let (asked, warnings) = requests_to_the_sample(&mut run, &|| indexes(&[]));
    let tables: Vec<String> = asked
        .iter()
        .filter_map(|request| match request {
            DataRequest::TimeBounds { columns } => Some(name(&columns[0].table)),
            DataRequest::SampleEntities { table, .. } => Some(name(table)),
            _ => None,
        })
        .collect();
    assert_eq!(tables, ["app_orders", "app_users"]);
    // Orders' foreign key, which no index covers.
    assert_eq!(warnings, ["MISSING_INDICES"]);
    // Every request kind of the run to its result, each answered, names
    // the source's table (T1.24 fix 1, review P3-1): the time bounds, the
    // sample, the labels' and the first hop's neighbors, and the lookup.
    let mut run = materialize(host(), &renamed(), QUERY, None).unwrap();
    let kinds = tables_to_the_result(&mut run);
    let kinds: Vec<(&str, &str)> = kinds.iter().map(|(k, t)| (*k, t.as_str())).collect();
    assert_eq!(
        kinds,
        [
            ("time_bounds", "app_orders"),
            ("sample_entities", "app_users"),
            ("neighbors", "app_orders"),
            ("lookup", "app_users"),
            ("neighbors", "app_orders"),
        ]
    );
    // An index is the source's: one over the source table's column covers
    // the schema's table read from it, and none is warned of; seeded, the
    // same index under the schema's name of the table covers nothing.
    let mut run = materialize(host(), &renamed(), QUERY, None).unwrap();
    let (_, warnings) = requests_to_the_sample(&mut run, &|| indexes(&[("app_orders", "user_id")]));
    assert!(warnings.is_empty(), "{warnings:?}");
    let mut run = materialize(host(), &renamed(), QUERY, None).unwrap();
    let (_, warnings) = requests_to_the_sample(&mut run, &|| indexes(&[("orders", "user_id")]));
    assert_eq!(warnings, ["MISSING_INDICES"]);
}

#[test]
fn the_options_the_host_passes_through_are_refused_by_name() {
    // T1.24 (decision E61): the host's bridge refuses none of the call's
    // options; the core refuses by name each it does not take, before
    // anything is fetched: the other options, the first in name order,
    // then an unseeded sample, a context size and an anchor at each
    // entity's own time. Seeded: the call with none of them reaches its
    // first fetch.
    let with = |change: &dyn Fn(&mut Materialize)| {
        let (mut operation, attachments) = materialize_start(host(), &shop(), QUERY, None);
        let Operation::Materialize(request) = &mut operation else {
            unreachable!()
        };
        change(request);
        let mut run =
            materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments).unwrap();
        run.next()
    };
    // With the call's option values (T1.28a): `values`, and the options
    // named.
    let valued = |values: &str, names: &[&str]| {
        let (mut operation, mut attachments) = materialize_start(host(), &shop(), QUERY, None);
        let Operation::Materialize(request) = &mut operation else {
            unreachable!()
        };
        request.values = Some(AttachmentId(9));
        request.options = names.iter().map(|n| (*n).to_owned()).collect();
        attachments.insert(
            AttachmentId(9),
            Attachment::Bytes(values.as_bytes().to_vec()),
        );
        let mut run =
            materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments).unwrap();
        run.next()
    };
    // An unseeded sample and an anchor at each entity's own time are taken
    // from T2.11: the run reaches its first read.
    assert!(matches!(
        with(&|m| m.unseeded = true),
        Ok(Step::Fetch { .. })
    ));
    assert!(matches!(
        with(&|m| m.anchor_entity = true),
        Ok(Step::Fetch { .. })
    ));
    // From T1.28a the context size is the core's: bounded first, before
    // the options, as the client bounds it before anything else.
    for (asked, code) in [
        ("0", ErrorCode::InvalidRequest),
        ("-3", ErrorCode::InvalidRequest),
        ("10001", ErrorCode::UnsupportedRequest),
        (
            "1267650600228229401496703205376",
            ErrorCode::UnsupportedRequest,
        ),
        ("2.5", ErrorCode::InvalidRequest),
    ] {
        let err = valued(
            &format!(r#"{{"num_context_samples": {asked}}}"#),
            &["return_embeddings"],
        )
        .unwrap_err();
        assert_eq!(err.code, code, "{asked}");
        assert_eq!(err.fields["field"], Detail::text("num_context_samples"));
    }
    // The hops the options ask, bounded as the client bounds them; a
    // negative seed refused.
    for (values, field) in [
        (r#"{"num_hops": 0}"#, "num_hops"),
        (r#"{"num_neighbors": []}"#, "num_neighbors"),
        (r#"{"random_seed": -1}"#, "seed"),
    ] {
        let err = valued(values, &[]).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest, "{values}");
        assert_eq!(err.fields["field"], Detail::text(field), "{values}");
    }
    let past = valued(r#"{"num_hops": 7}"#, &[]).unwrap_err();
    assert_eq!(past.code, ErrorCode::UnsupportedRequest);
    assert_eq!(past.fields["field"], Detail::text("num_hops"));
    assert!(!past.fields.contains_key("rule"), "the bound, not the rule");
    // The sampling options taken from T2.11 reach the first read.
    for values in [
        r#"{"num_hops": 3}"#,
        r#"{"num_neighbors": [4, 0]}"#,
        r#"{"random_seed": null}"#,
        r#"{"max_pq_iterations": 2, "use_prediction_time": true}"#,
        r#"{"return_embeddings": "yes", "exclude_cols_dict": {"orders": ["price"]}}"#,
        r#"{"context_anchor_time": {"t": "ts", "v": "1704067200000000000", "unit": "ns", "zone": null}}"#,
    ] {
        assert!(
            matches!(valued(values, &[]), Ok(Step::Fetch { .. })),
            "{values}"
        );
    }
    assert!(matches!(
        valued(r#"{"random_seed": 7, "num_context_samples": 5}"#, &[]),
        Ok(Step::Fetch { .. })
    ));

    // A name that is not one of the reference's options is the host's
    // fault.
    let err = with(&|m| m.options = vec!["no_such_option".into()]).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("options"));
    // Every sampling option's value travels in `values` from T2.11 (with
    // T2.8's top_k and T2.9's lag_timesteps): none is a name the host
    // passes.
    for name in [
        "lag_timesteps",
        "num_hops",
        "use_prediction_time",
        "context_anchor_time",
    ] {
        let err = with(&|m| m.options = vec![name.into()]).unwrap_err();
        assert_eq!(err.fields["field"], Detail::text("options"), "{name}");
    }
    assert!(matches!(with(&|_| {}), Ok(Step::Fetch { .. })));
}

/// The run of `materialize` over the shop with the call's `values`.
fn valued_run(values: &str) -> Run {
    let (mut operation, mut attachments) = materialize_start(host(), &shop(), QUERY, None);
    let Operation::Materialize(request) = &mut operation else {
        unreachable!()
    };
    request.values = Some(AttachmentId(9));
    attachments.insert(
        AttachmentId(9),
        Attachment::Bytes(values.as_bytes().to_vec()),
    );
    materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments).unwrap()
}

#[test]
fn the_sampling_options_the_engine_reads_late_are_refused_where_it_reads_them() {
    // T2.11: the options the reference's engine reads after its sampler is
    // built are refused after the index check and its warning, as there:
    // the context's anchor and max_pq_iterations with the task, the
    // excluded columns and a num_hops of none as the subgraph is sampled,
    // the two booleans as the request is built.
    for (values, field, last) in [
        (r#"{"max_pq_iterations": 0}"#, "max_pq_iterations", "task"),
        (r#"{"max_pq_iterations": "3"}"#, "max_pq_iterations", "task"),
        (
            r#"{"context_anchor_time": "unreadable"}"#,
            "context_anchor_time",
            "task",
        ),
        (
            r#"{"context_anchor_time": "entity"}"#,
            "context_anchor_time",
            "task",
        ),
        (
            r#"{"exclude_cols_dict": ["orders"]}"#,
            "exclude_cols_dict",
            "labels",
        ),
        (
            r#"{"exclude_cols_dict": {"orders": ["order_id"]}}"#,
            "exclude_cols_dict",
            "labels",
        ),
        (
            r#"{"exclude_cols_dict": {"nowhere": []}}"#,
            "exclude_cols_dict",
            "labels",
        ),
        (r#"{"num_hops": null}"#, "num_hops", "labels"),
        (
            r#"{"use_prediction_time": null}"#,
            "use_prediction_time",
            "request",
        ),
        (
            r#"{"return_embeddings": "x"}"#,
            "return_embeddings",
            "request",
        ),
    ] {
        let mut run = valued_run(values);
        let mut seen = Vec::new();
        let mut warned = Vec::new();
        let err = loop {
            let step = match run.next() {
                Ok(step) => step,
                Err(err) => break err,
            };
            warned.extend(run.take_warnings().into_iter().map(|w| w.code));
            let Step::Fetch { requests } = step else {
                panic!("{values}: {step:?}")
            };
            for asked in requests {
                seen.push(asked.request.name());
                let answer = match &asked.request {
                    DataRequest::Describe { .. } => indexes(&[]),
                    DataRequest::TimeBounds { .. } => catalog(&asked.request).unwrap(),
                    DataRequest::SampleEntities { columns, .. } => {
                        let keys: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2, 3]));
                        data(
                            columns
                                .iter()
                                .map(|c| {
                                    let given = (c == "user_id").then(|| Arc::clone(&keys));
                                    (c.as_str(), shop_column(c, 4, given))
                                })
                                .collect(),
                        )
                    }
                    DataRequest::Neighbors { columns, .. } => {
                        let mut answer: Vec<(&str, ArrayRef)> = columns
                            .iter()
                            .map(|c| (c.as_str(), shop_column(c, 0, None)))
                            .collect();
                        answer.push((BATCH_COLUMN, Arc::new(Int64Array::from(Vec::<i64>::new()))));
                        data(answer)
                    }
                    DataRequest::Lookup { columns, seeds, .. } => {
                        let Some(Attachment::Arrow(seeds)) = run.take_attachment(seeds.attachment)
                        else {
                            panic!("the lookup's seeds are attached");
                        };
                        let seeds = &seeds.batches[0];
                        let n = seeds.num_rows();
                        let mut answer: Vec<(&str, ArrayRef)> = columns
                            .iter()
                            .map(|c| {
                                let given = (c == "user_id").then(|| Arc::clone(seeds.column(0)));
                                (c.as_str(), shop_column(c, n, given))
                            })
                            .collect();
                        let rows: Vec<i64> = (0..n as i64).collect();
                        answer.push((BATCH_COLUMN, Arc::new(Int64Array::from(rows))));
                        answer.push((MATCHES_COLUMN, Arc::new(Int64Array::from(vec![1_i64; n]))));
                        data(answer)
                    }
                    other => panic!("{other:?}"),
                };
                run.submit_fetch(asked.id, answer).unwrap();
            }
        };
        warned.extend(run.take_warnings().into_iter().map(|w| w.code));
        assert_eq!(err.code, ErrorCode::InvalidRequest, "{values}: {err:?}");
        assert_eq!(err.fields["field"], Detail::text(field), "{values}");
        assert_eq!(seen.first().copied(), Some("describe"), "{values}");
        // Refused with the task: no entity sampled; as the subgraph is
        // sampled: the labels read, no entity row looked up; as the request
        // is built: the subgraph walked.
        let sampled = seen.contains(&"sample_entities");
        let walked = seen.contains(&"lookup");
        let at = match (sampled, walked) {
            (false, false) => "task",
            (true, false) => "labels",
            (true, true) => "request",
            _ => "neither",
        };
        assert_eq!(at, last, "{values}: {seen:?}");
        assert!(warned.contains(&"MISSING_INDICES".to_owned()), "{values}");
    }
}

// --- the times a source returns (T1.25) -------------------------------------

/// The shop with orders' time column received as `received` (`dtype`), and
/// a host provider declaring `text_times`.
fn shop_with_times(received: ColumnType, dtype: Dtype) -> SourceDescription {
    let mut shop = shop();
    let ts = shop.tables[0]
        .columns
        .iter_mut()
        .find(|c| c.name == "ts")
        .unwrap();
    ts.received = received;
    ts.dtype = dtype;
    shop
}

fn host_reading(text_times: Option<TextTimes>) -> ProviderChoice {
    let ProviderChoice::Host { mut capabilities } = host() else {
        unreachable!()
    };
    capabilities.text_times = text_times;
    ProviderChoice::Host { capabilities }
}

/// Runs `run` to its end over the shop, answering every request: the
/// catalog as [`catalog`] does, four users, each label window and first
/// hop with orders 0 and 1 of the first seed's entity, their times
/// `ts(2)`, and the lookup with one row per seed; the result or the
/// error, and the warnings given.
fn times_to_the_end(
    run: &mut Run,
    ts: &dyn Fn(usize) -> ArrayRef,
) -> (Result<Vec<u8>, CoreError>, Vec<sdm_contract::CoreWarning>) {
    let mut warnings = Vec::new();
    loop {
        let step = match run.next() {
            Ok(step) => step,
            Err(err) => return (Err(err), warnings),
        };
        warnings.extend(run.take_warnings());
        let requests = match step {
            Step::Fetch { requests } => requests,
            Step::Done { result } => {
                let sdm_contract::ops::OperationResult::Materialize(done) = result else {
                    panic!("not a materialize result");
                };
                let Some(Attachment::Bytes(payload)) = run.take_attachment(done.state.payload)
                else {
                    panic!("the payload is attached");
                };
                return (Ok(payload), warnings);
            }
            other => panic!("{other:?}"),
        };
        for asked in requests {
            // Two orders of the first seed's entity, 0 and 1, in its batch.
            let rows = |columns: &[String], key: ArrayRef| -> FetchResult {
                let n = 2;
                let mut answer: Vec<(&str, ArrayRef)> = columns
                    .iter()
                    .map(|c| {
                        let column: ArrayRef = match c.as_str() {
                            "order_id" => Arc::new(Int64Array::from(vec![0_i64, 1])),
                            "user_id" => Arc::clone(&key),
                            "price" => Arc::new(arrow_array::Float64Array::from(vec![1.5; n])),
                            "ts" => ts(n),
                            _ => shop_column(c, n, None),
                        };
                        (c.as_str(), column)
                    })
                    .collect();
                answer.push((
                    BATCH_COLUMN,
                    Arc::new(Int64Array::from(vec![0_i64; n])) as ArrayRef,
                ));
                data(answer)
            };
            let answer = match &asked.request {
                DataRequest::Describe { .. } => indexes(&[]),
                DataRequest::TimeBounds { .. } => catalog(&asked.request).unwrap(),
                DataRequest::SampleEntities { columns, .. } => {
                    let keys: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2, 3]));
                    data(
                        columns
                            .iter()
                            .map(|c| {
                                let given = (c == "user_id").then(|| Arc::clone(&keys));
                                (c.as_str(), shop_column(c, 4, given))
                            })
                            .collect(),
                    )
                }
                DataRequest::Neighbors { columns, seeds, .. } => {
                    let Some(Attachment::Arrow(seeds)) = run.take_attachment(seeds.attachment)
                    else {
                        panic!("the neighbors' seeds are attached");
                    };
                    let first = seeds.batches[0].column(0).slice(0, 1);
                    let key =
                        arrow_select::concat::concat(&[first.as_ref(), first.as_ref()]).unwrap();
                    rows(columns, key)
                }
                DataRequest::Lookup { columns, seeds, .. } => {
                    let Some(Attachment::Arrow(seeds)) = run.take_attachment(seeds.attachment)
                    else {
                        panic!("the lookup's seeds are attached");
                    };
                    let seeds = &seeds.batches[0];
                    let n = seeds.num_rows();
                    let mut answer: Vec<(&str, ArrayRef)> = columns
                        .iter()
                        .map(|c| {
                            let given = (c == "user_id").then(|| Arc::clone(seeds.column(0)));
                            (c.as_str(), shop_column(c, n, given))
                        })
                        .collect();
                    answer.push((
                        BATCH_COLUMN,
                        Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
                    ));
                    answer.push((MATCHES_COLUMN, Arc::new(Int64Array::from(vec![1_i64; n]))));
                    data(answer)
                }
                other => panic!("{other:?}"),
            };
            run.submit_fetch(asked.id, answer).unwrap();
        }
    }
}

/// A column of `n` rows a fetch answers with.
type Column = Box<dyn Fn(usize) -> ArrayRef>;

/// The wire text of 2024-01-05T00:00Z, which the payload writes its
/// order's time as.
const SENT: &str = "2024-01-05T00:00:00.000000Z";

fn sends(payload: &[u8], text: &str) -> bool {
    String::from_utf8_lossy(payload).contains(text)
}

#[test]
fn a_date_a_zoned_timestamp_and_text_are_read_to_their_instants() {
    // Until T1.25, materialize read a zone-less timestamp time column only
    // (`time_type`); the reference's engine reads every column meaning a
    // timestamp to a zone-less UTC one as it fetches it.
    let day = 19_727; // 2024-01-05
    let cases: Vec<(ColumnType, Dtype, Column)> = vec![
        (
            ColumnType::Date32,
            Dtype::Date,
            Box::new(move |n| Arc::new(arrow_array::Date32Array::from(vec![day; n])) as ArrayRef),
        ),
        (
            ColumnType::Timestamp {
                unit: Unit::Us,
                zone: Some("America/New_York".into()),
            },
            Dtype::Date,
            Box::new(|n| {
                Arc::new(
                    TimestampMicrosecondArray::from(vec![1_704_412_800_000_000; n])
                        .with_timezone("America/New_York"),
                ) as ArrayRef
            }),
        ),
        (
            ColumnType::Utf8,
            Dtype::String,
            Box::new(|n| Arc::new(StringArray::from(vec!["2024-01-05"; n])) as ArrayRef),
        ),
    ];
    for (received, dtype, ts) in cases {
        let mut run = materialize(
            host_reading(Some(TextTimes::ValueByValue)),
            &shop_with_times(received.clone(), dtype),
            QUERY,
            None,
        )
        .unwrap();
        let (payload, warnings) = times_to_the_end(&mut run, ts.as_ref());
        let payload = payload.unwrap_or_else(|err| panic!("{received:?}: {err:?}"));
        assert!(sends(&payload, SENT), "{received:?}");
        assert!(!sends(&payload, "America/New_York"), "{received:?}");
        assert!(
            warnings.iter().all(|w| w.code != "TIME_VALUES_UNREAD"),
            "{received:?}"
        );
    }
}

#[test]
fn text_times_are_read_as_the_provider_declares_and_refused_by_name_otherwise() {
    let text = || shop_with_times(ColumnType::Utf8, Dtype::String);
    let days = |n: usize| Arc::new(StringArray::from(vec!["2024-01-05"; n])) as ArrayRef;
    // No reading declared, the client's rules (not ported), a value outside
    // the core's list of formats (T1.25b), day and month values the fetch
    // puts in no order, and offsets mixed: each by its rule's name, before
    // any text is read as a time.
    for (provider, ts, wanted) in [
        (host_reading(None), Box::new(days) as Column, "text_times"),
        (
            host_reading(Some(TextTimes::ClientRules)),
            Box::new(days),
            "text_time_rules",
        ),
        (
            host_reading(Some(TextTimes::ValueByValue)),
            Box::new(|n| Arc::new(StringArray::from(vec!["March 5 2024"; n])) as ArrayRef),
            "text_time",
        ),
        (
            host_reading(Some(TextTimes::ValueByValue)),
            Box::new(|n| Arc::new(StringArray::from(vec!["not a date"; n])) as ArrayRef),
            "text_time",
        ),
        (
            host_reading(Some(TextTimes::ValueByValue)),
            Box::new(|n| Arc::new(StringArray::from(vec!["01/05/2024"; n])) as ArrayRef),
            "text_time_order",
        ),
        (
            host_reading(Some(TextTimes::ValueByValue)),
            Box::new(|n| Arc::new(StringArray::from(vec!["2024-02-30"; n])) as ArrayRef),
            "text_time",
        ),
        (
            host_reading(Some(TextTimes::ValueByValue)),
            Box::new(|n| {
                let values: Vec<&str> = (0..n)
                    .map(|i| {
                        if i % 2 == 0 {
                            "2024-01-05 10:00+01:00"
                        } else {
                            "2024-01-05 10:00"
                        }
                    })
                    .collect();
                Arc::new(StringArray::from(values)) as ArrayRef
            }),
            "text_time_zones",
        ),
    ] {
        let mut run = materialize(provider, &text(), QUERY, None).unwrap();
        let (result, _) = times_to_the_end(&mut run, ts.as_ref());
        let err = result.unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedRequest, "{wanted}");
        assert_eq!(rule(&err), wanted);
    }
}

#[test]
fn a_text_time_read_as_no_time_is_dropped_and_warned_of() {
    // A blank value holds no time: dropped and warned of, as the
    // reference's engine drops a value it reads as none, with how many of
    // how many and three examples. (A value naming no date, `2024-02-30`,
    // is outside the core's list and refused, T1.25b.)
    let mut run = materialize(
        host_reading(Some(TextTimes::ValueByValue)),
        &shop_with_times(ColumnType::Utf8, Dtype::String),
        QUERY,
        None,
    )
    .unwrap();
    let (payload, warnings) = times_to_the_end(&mut run, &|n| {
        Arc::new(StringArray::from(vec!["  "; n])) as ArrayRef
    });
    payload.unwrap();
    let unread: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == "TIME_VALUES_UNREAD")
        .collect();
    assert!(!unread.is_empty());
    for warning in unread {
        assert_eq!(warning.fields["column"], Detail::text("ts"));
        assert_eq!(warning.fields["table"], Detail::text("orders"));
        assert_eq!(warning.fields["unread"], warning.fields["rows"]);
        assert_eq!(
            warning.fields["examples"],
            Detail::List(vec![Scalar::Utf8("  ".into())])
        );
    }
}

/// The shop's host provider, its sampling policies as given (T1.26a).
fn host_with(null_order: Option<NullOrder>, index_advice: Option<IndexAdvice>) -> ProviderChoice {
    let ProviderChoice::Host { mut capabilities } = host() else {
        unreachable!()
    };
    capabilities.null_order = null_order;
    capabilities.index_advice = index_advice;
    ProviderChoice::Host { capabilities }
}

/// The shipped data layer, its DuckDB capabilities changed by `change`: a
/// test provider declaring other sampling policies than DuckDB's, or, as a
/// seeded fault, forgetting one.
struct Changed(fn(&mut ProviderCapabilities));

impl DataLayer for Changed {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        let mut capabilities = sdm_data::Data::new().capabilities(provider)?;
        (self.0)(&mut capabilities);
        Ok(capabilities)
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        sdm_data::Data::new().plan(request, provider)
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        sdm_data::Data::new().read(request, provider, batches)
    }
}

/// DuckDB's index answer, in its own catalog's shape: one row per index
/// of `orders`, its `expressions` as DuckDB writes them.
fn duckdb_indexes(expressions: &[&str]) -> FetchResult {
    data(vec![
        (
            "table_name",
            Arc::new(StringArray::from(vec!["orders"; expressions.len()])),
        ),
        (
            "expressions",
            Arc::new(StringArray::from(expressions.to_vec())),
        ),
    ])
}

/// The order and the statements of every neighbor read `run` asks with
/// a row limit per seed, run to its result.
fn ordered_reads(steps: &[(DataRequest, FetchPlan)]) -> Vec<(Vec<bool>, Vec<String>)> {
    steps
        .iter()
        .filter_map(|(request, plan)| match request {
            DataRequest::Neighbors {
                order: Some(order), ..
            } => {
                let texts = match plan {
                    FetchPlan::Sql { statements, .. } => {
                        statements.iter().map(|s| s.text.clone()).collect()
                    }
                    FetchPlan::Host { .. } => vec![],
                };
                Some((order.0.iter().map(|t| t.nulls_last).collect(), texts))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn the_sampler_s_null_order_is_the_one_the_provider_declares() {
    // A host provider declaring nulls first is asked for its neighbors
    // with every term nulls first; one declaring nulls last, every term
    // nulls last; the order is the provider's, never a default.
    for (declared, nulls_last) in [(NullOrder::First, false), (NullOrder::Last, true)] {
        let provider = host_with(Some(declared), Some(IndexAdvice::SingleColumn));
        let mut run = materialize(provider, &renamed(), QUERY, None).unwrap();
        let (steps, _) = steps_to_the_result(&mut run, &|| indexes(&[]));
        let reads = ordered_reads(&steps);
        assert_eq!(reads.len(), 1, "{declared:?}");
        // The time column, latest first, then the key.
        assert_eq!(reads[0].0, [nulls_last, nulls_last], "{declared:?}");
    }
    // A SQL provider: DuckDB as `sdm-data` declares it orders nulls last,
    // its statement unchanged (H9's traces hold its text and rows); a test
    // provider rendering DuckDB's SQL but declaring nulls first is asked
    // for nulls first in both its window's order and the result's.
    let duckdb_order = |layer: Arc<dyn DataLayer>| -> Vec<(Vec<bool>, Vec<String>)> {
        let (operation, attachments) = materialize_start(duckdb(), &renamed(), QUERY, None);
        let mut run = materialize_on(layer, operation, attachments).unwrap();
        let (steps, _) = steps_to_the_result(&mut run, &|| duckdb_indexes(&[]));
        ordered_reads(&steps)
    };
    let shipped = duckdb_order(Arc::new(sdm_data::Data::new()));
    assert_eq!(shipped.len(), 1);
    assert_eq!(shipped[0].0, [true, true]);
    let text = &shipped[0].1[0];
    assert_eq!(text.matches("NULLS LAST").count(), 4, "{text}");
    assert!(!text.contains("NULLS FIRST"), "{text}");
    let first = duckdb_order(Arc::new(Changed(|c| {
        c.null_order = Some(NullOrder::First);
    })));
    assert_eq!(first[0].0, [false, false]);
    let text = &first[0].1[0];
    assert_eq!(text.matches("NULLS FIRST").count(), 4, "{text}");
    assert!(!text.contains("NULLS LAST"), "{text}");
    // The statements differ in their null order only.
    assert_eq!(
        text.replace("NULLS FIRST", "NULLS LAST"),
        shipped[0].1[0],
        "{text}"
    );
}

#[test]
fn a_provider_declaring_no_null_order_is_not_asked_for_an_order() {
    // Seeded fault, failing closed: a provider declaring no null order is
    // refused by name, never given DuckDB's, before anything is fetched
    // (T1.26b, T1.26a's review P3); the same for DuckDB's rendering with
    // its declaration forgotten.
    let mut run = materialize(
        host_with(None, Some(IndexAdvice::SingleColumn)),
        &renamed(),
        QUERY,
        None,
    )
    .unwrap();
    assert_eq!(rule(&run.next().unwrap_err()), "null_order");
    let (operation, attachments) = materialize_start(duckdb(), &renamed(), QUERY, None);
    let layer: Arc<dyn DataLayer> = Arc::new(Changed(|c| c.null_order = None));
    let mut run = materialize_on(layer, operation, attachments).unwrap();
    assert_eq!(rule(&run.next().unwrap_err()), "null_order");
}

#[test]
fn a_provider_not_declaring_a_request_the_sampler_asks_is_refused_by_name() {
    // A provider declaring no `neighbors` (or any other request the sampler
    // asks) is refused naming the request, before anything is fetched, as
    // `DataAccess::plan` refuses an undeclared request.
    for missing in [
        OptionalRequest::Neighbors,
        OptionalRequest::Lookup,
        OptionalRequest::SampleEntities,
        OptionalRequest::TimeBounds,
    ] {
        let ProviderChoice::Host { mut capabilities } = host() else {
            unreachable!()
        };
        capabilities.optional.remove(&missing);
        let mut run =
            materialize(ProviderChoice::Host { capabilities }, &shop(), QUERY, None).unwrap();
        let err = run.next().unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedRequest);
        assert_eq!(
            err.fields.get("request"),
            Some(&Detail::text(missing.name())),
            "{missing:?}"
        );
    }
}

/// Runs `run` as [`steps_to_the_result`] does, to the error it ends with.
fn run_to_error(run: &mut Run, index_answer: &dyn Fn() -> FetchResult) -> CoreError {
    try_steps(run, index_answer).expect_err("the run reached its result")
}

#[test]
fn a_provider_without_index_advice_is_neither_asked_for_indexes_nor_warned() {
    // With the advice, the run reads the indexes once, first, and warns
    // of the foreign key no index covers; without it, it asks one request
    // fewer, none of them the index read, and warns of nothing.
    let mut run = materialize(
        host_with(Some(NullOrder::Last), Some(IndexAdvice::SingleColumn)),
        &renamed(),
        QUERY,
        None,
    )
    .unwrap();
    let (advised, warnings) = steps_to_the_result(&mut run, &|| indexes(&[]));
    assert!(matches!(
        advised[0].0,
        DataRequest::Describe {
            part: DescribePart::Indexes,
            ..
        }
    ));
    assert_eq!(warnings, ["MISSING_INDICES"]);
    let mut run = materialize(
        host_with(Some(NullOrder::Last), None),
        &renamed(),
        QUERY,
        None,
    )
    .unwrap();
    let (silent, warnings) =
        steps_to_the_result(&mut run, &|| panic!("the indexes were asked for"));
    assert_eq!(silent.len(), advised.len() - 1);
    assert!(
        !silent
            .iter()
            .any(|(r, _)| matches!(r, DataRequest::Describe { .. })),
        "{silent:?}"
    );
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        silent.iter().map(|(r, _)| r).collect::<Vec<_>>(),
        advised[1..].iter().map(|(r, _)| r).collect::<Vec<_>>()
    );
    // DuckDB as `sdm-data` declares it reads its indexes, in one statement;
    // with the declaration forgotten (seeded), no statement reads them and
    // nothing is warned of: the read and the warning are the capability's.
    let duckdb_run = |layer: Arc<dyn DataLayer>| {
        let (operation, attachments) = materialize_start(duckdb(), &renamed(), QUERY, None);
        let mut run = materialize_on(layer, operation, attachments).unwrap();
        steps_to_the_result(&mut run, &|| duckdb_indexes(&[]))
    };
    let statements = |steps: &[(DataRequest, FetchPlan)]| -> Vec<String> {
        steps
            .iter()
            .flat_map(|(_, plan)| match plan {
                FetchPlan::Sql { statements, .. } => {
                    statements.iter().map(|s| s.text.clone()).collect()
                }
                FetchPlan::Host { .. } => vec![],
            })
            .collect()
    };
    let (shipped, warnings) = duckdb_run(Arc::new(sdm_data::Data::new()));
    assert_eq!(warnings, ["MISSING_INDICES"]);
    let shipped = statements(&shipped);
    assert!(shipped[0].contains("duckdb_indexes()"), "{shipped:?}");
    let (forgotten, warnings) = duckdb_run(Arc::new(Changed(|c| c.index_advice = None)));
    let forgotten = statements(&forgotten);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(forgotten, shipped[1..]);
}

#[test]
fn an_index_the_reader_cannot_resolve_covers_nothing() {
    // DuckDB's index over the foreign key alone covers it, and nothing is
    // warned of; an index whose expressions hold a comma, between columns
    // or inside one expression, is unresolved and covers nothing, as the
    // reference reads no column from it: the foreign key is warned of.
    let warned = |expressions: &[&str]| -> Vec<String> {
        let (operation, attachments) = materialize_start(duckdb(), &shop(), QUERY, None);
        let mut run =
            materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments).unwrap();
        let answer = duckdb_indexes(expressions);
        steps_to_the_result(&mut run, &|| answer.clone()).1
    };
    assert!(warned(&["[user_id]"]).is_empty());
    assert!(warned(&["['\"user_id\"']"]).is_empty());
    for unresolved in [
        "[user_id, ts]",
        "[ts, user_id]",
        "['(COALESCE(user_id, 0))', '(lower(\\',\\'))']",
        "['(COALESCE(user_id, 0))']",
    ] {
        assert_eq!(warned(&[unresolved]), ["MISSING_INDICES"], "{unresolved}");
    }
    // A host provider's list naming a null column is unresolved too.
    let mut builder =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new())
            .with_field(Arc::new(Field::new(
                catalog_columns::LIST_ITEM,
                DataType::Utf8,
                true,
            )));
    builder.values().append_value("user_id");
    builder.values().append_null();
    builder.append(true);
    let with_null = data(vec![
        (
            catalog_columns::TABLE,
            Arc::new(StringArray::from(vec!["orders"])),
        ),
        (catalog_columns::COLUMNS, Arc::new(builder.finish())),
    ]);
    let mut run = materialize(host(), &shop(), QUERY, None).unwrap();
    let (_, warnings) = steps_to_the_result(&mut run, &|| with_null.clone());
    assert_eq!(warnings, ["MISSING_INDICES"]);
}

// --- the sampler state (T1.26b, decision E52) ---------------------------------

/// The sampler state a run left: its version, and its memo's bytes.
type Left = (u32, Vec<u8>);

/// Takes the sampler state `run` left, as a host takes it.
fn left(run: &mut Run) -> Option<Left> {
    let state = run.take_sampler_state().unwrap()?;
    let Some(Attachment::Bytes(bytes)) = run.take_attachment(state.memo) else {
        panic!("the memo is attached");
    };
    Some((state.version, bytes))
}

fn memo(left: &Left) -> SamplerMemo {
    serde_json::from_slice(&left.1).unwrap()
}

/// A `materialize` start over `description` handing back `held`, with
/// `options` passed through and `indices` the entities.
fn handing_back(
    description: &SourceDescription,
    held: Option<&Left>,
    options: &[&str],
    indices: Vec<Option<i64>>,
) -> Result<Run, CoreError> {
    handing_back_changed(description, held, options, indices, |_| ())
}

/// [`handing_back`], with `change` made to the start's attachments.
fn handing_back_changed(
    description: &SourceDescription,
    held: Option<&Left>,
    options: &[&str],
    indices: Vec<Option<i64>>,
    change: impl FnOnce(&mut Attachments),
) -> Result<Run, CoreError> {
    let (mut operation, mut attachments) = materialize_start(host(), description, QUERY, None);
    let Operation::Materialize(start) = &mut operation else {
        unreachable!()
    };
    start.options = options.iter().map(|o| (*o).to_owned()).collect();
    if let Some((version, bytes)) = held {
        start.sampler_state = Some(SamplerState {
            version: *version,
            memo: AttachmentId(4),
        });
        attachments.insert(AttachmentId(4), Attachment::Bytes(bytes.clone()));
    }
    let entities = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "indices",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(indices)) as ArrayRef],
    )
    .unwrap();
    attachments.insert(
        AttachmentId(3),
        Attachment::Arrow(ArrowData::new(entities.schema(), vec![entities])),
    );
    change(&mut attachments);
    materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments)
}

/// Each request's kind, in order.
fn kinds(steps: &Steps) -> Vec<&'static str> {
    steps.0.iter().map(|(request, _)| request.name()).collect()
}

#[test]
fn a_second_call_skips_the_reads_the_first_made() {
    // The reference checks the indexes once, building its sampler, and
    // keeps the time bounds it reads (H9's counts on a second call: 6, then
    // 4); the core skips what its memo says a run of this materializer read,
    // and the request is the same.
    let mut first = handing_back(&shop(), None, &[], vec![Some(0), Some(1)]).unwrap();
    let steps = steps_to_the_result(&mut first, &|| indexes(&[]));
    assert_eq!(
        kinds(&steps),
        [
            "describe",
            "time_bounds",
            "sample_entities",
            "neighbors",
            "lookup",
            "neighbors"
        ]
    );
    assert_eq!(steps.1, ["MISSING_INDICES"]);
    let first_left = left(&mut first).unwrap();
    assert_eq!(first_left.0, SAMPLER_STATE_VERSION);
    let kept = memo(&first_left);
    assert!(kept.indexes_checked);
    assert_eq!(kept.bounds.len(), 1);
    assert_eq!(
        (
            kept.bounds[0].table.as_str(),
            kept.bounds[0].column.as_str()
        ),
        ("orders", "ts")
    );
    let mut second = handing_back(&shop(), Some(&first_left), &[], vec![Some(0), Some(1)]).unwrap();
    let again = steps_to_the_result(&mut second, &|| indexes(&[]));
    assert_eq!(
        kinds(&again),
        ["sample_entities", "neighbors", "lookup", "neighbors"]
    );
    assert!(again.1.is_empty(), "{:?}", again.1);
    assert_eq!(memo(&left(&mut second).unwrap()), kept);
    // The same requests from the sample on, the time bounds the memo kept
    // standing for those read.
    assert_eq!(steps.0[2..], again.0[..]);
}

#[test]
fn each_materializer_checks_once() {
    // Two materializers, each with its own state: each checks its indexes
    // once and warns once, as each reference materializer builds its own
    // sampler.
    let mut states = Vec::new();
    for _ in 0..2 {
        let mut run = handing_back(&shop(), None, &[], vec![Some(0), Some(1)]).unwrap();
        let steps = steps_to_the_result(&mut run, &|| indexes(&[]));
        assert_eq!(steps.0[0].0.name(), "describe");
        assert_eq!(steps.1, ["MISSING_INDICES"]);
        states.push(left(&mut run).unwrap());
    }
    for held in &states {
        let mut run = handing_back(&shop(), Some(held), &[], vec![Some(0), Some(1)]).unwrap();
        let steps = steps_to_the_result(&mut run, &|| indexes(&[]));
        assert_eq!(steps.0[0].0.name(), "sample_entities");
        assert!(steps.1.is_empty());
    }
}

#[test]
fn a_failure_before_the_check_leaves_it_to_the_next_run() {
    // Refused before anything is read (a name that is no option the core
    // reads): the state left says nothing was read, and the next run
    // checks and warns.
    let mut refused =
        handing_back(&shop(), None, &["no_such_option"], vec![Some(0), Some(1)]).unwrap();
    let err = refused.next().unwrap_err();
    assert_eq!(err.fields["field"], Detail::text("options"), "{err:?}");
    let held = left(&mut refused).unwrap();
    assert!(!memo(&held).indexes_checked);
    assert!(memo(&held).bounds.is_empty());
    let mut next = handing_back(&shop(), Some(&held), &[], vec![Some(0), Some(1)]).unwrap();
    let steps = steps_to_the_result(&mut next, &|| indexes(&[]));
    assert_eq!(steps.0[0].0.name(), "describe");
    assert_eq!(steps.1, ["MISSING_INDICES"]);
}

#[test]
fn a_failure_after_the_check_keeps_it() {
    // Refused after the index read (an entity missing), as the reference
    // refuses it after building its sampler: the check is kept, and the
    // next run neither reads the indexes nor warns again; it reads the time
    // bounds, which the refused run did not.
    let mut refused = handing_back(&shop(), None, &[], vec![None, Some(1)]).unwrap();
    let err = run_to_error(&mut refused, &|| indexes(&[]));
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    let held = left(&mut refused).unwrap();
    assert!(memo(&held).indexes_checked);
    assert!(memo(&held).bounds.is_empty());
    let mut next = handing_back(&shop(), Some(&held), &[], vec![Some(0), Some(1)]).unwrap();
    let steps = steps_to_the_result(&mut next, &|| indexes(&[]));
    assert_eq!(steps.0[0].0.name(), "time_bounds");
    assert!(steps.1.is_empty());
    // And a run refused after the time bounds (an anchor before the data)
    // keeps them too.
    let (mut operation, attachments) = materialize_start(host(), &shop(), QUERY, None);
    let Operation::Materialize(start) = &mut operation else {
        unreachable!()
    };
    start.anchor_time = Some(Scalar::Timestamp {
        unit: Unit::Ns,
        zone: None,
        value: 0,
    });
    let mut early =
        materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments).unwrap();
    let err = run_to_error(&mut early, &|| indexes(&[]));
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    let kept = memo(&left(&mut early).unwrap());
    assert!(kept.indexes_checked);
    assert_eq!(kept.bounds.len(), 1);
}

#[test]
fn a_state_of_another_version_or_unreadable_is_refused_by_name() {
    let mut first = handing_back(&shop(), None, &[], vec![Some(0), Some(1)]).unwrap();
    steps_to_the_result(&mut first, &|| indexes(&[]));
    let (version, bytes) = left(&mut first).unwrap();
    let field = |err: &CoreError| match err.fields.get("field") {
        Some(Detail::Scalar(Scalar::Utf8(f))) => f.clone(),
        other => panic!("{other:?}"),
    };
    // Another version, before anything of it is read.
    let Err(err) = handing_back(
        &shop(),
        Some(&(version + 1, bytes.clone())),
        &[],
        vec![Some(0)],
    ) else {
        panic!("another version was read");
    };
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    assert_eq!(field(&err), "sampler_state.version");
    // Each refusal of the state says, in so many words, to drop it
    // (decision E74); the hosts follow that, never the field.
    assert_eq!(err.drop_held, [HeldState::SamplerState]);
    // A memo that is not one: not JSON, of another shape, a field more or
    // retyped, a table's bounds twice, nested past the reader's depth, or
    // wide past any memo's fields.
    let mut kept: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut twice = kept.clone();
    let bound = twice["bounds"][0].clone();
    twice["bounds"].as_array_mut().unwrap().push(bound);
    let mut extra = kept.clone();
    extra["trusted"] = serde_json::json!(true);
    // Bounds no read gives: the smallest time after the largest, and one
    // missing without the other.
    let mut swapped = kept.clone();
    let (min, max) = (
        swapped["bounds"][0]["min"].clone(),
        swapped["bounds"][0]["max"].clone(),
    );
    assert!(min.as_i64().unwrap() < max.as_i64().unwrap());
    swapped["bounds"][0]["min"] = max;
    swapped["bounds"][0]["max"] = min;
    let mut half = kept.clone();
    half["bounds"][0]["max"] = serde_json::Value::Null;
    kept["indexes_checked"] = serde_json::json!("yes");
    let deep = format!("{}{}", "[".repeat(5_000), "]".repeat(5_000));
    let wide = serde_json::to_vec(&serde_json::Value::Object(
        (0..100_000)
            .map(|i| (format!("k{i}"), serde_json::json!(i)))
            .collect(),
    ))
    .unwrap();
    for (what, corrupt) in [
        ("not JSON", b"\xffnot json".to_vec()),
        ("another shape", b"{}".to_vec()),
        ("retyped", serde_json::to_vec(&kept).unwrap()),
        ("a field more", serde_json::to_vec(&extra).unwrap()),
        ("a table twice", serde_json::to_vec(&twice).unwrap()),
        ("bounds swapped", serde_json::to_vec(&swapped).unwrap()),
        ("half the bounds", serde_json::to_vec(&half).unwrap()),
        ("deep", deep.into_bytes()),
        ("wide", wide),
    ] {
        let Err(err) = handing_back(&shop(), Some(&(version, corrupt)), &[], vec![Some(0)]) else {
            panic!("{what}: read");
        };
        assert_eq!(field(&err), "sampler_state", "{what}");
        assert_eq!(err.drop_held, [HeldState::SamplerState], "{what}");
    }
}

#[test]
fn only_a_refusal_of_the_held_state_says_to_drop_it() {
    // A refusal of anything else of the start, with a sound state handed,
    // leaves the state to the host (decision E74).
    let mut first = handing_back(&shop(), None, &[], vec![Some(0), Some(1)]).unwrap();
    steps_to_the_result(&mut first, &|| indexes(&[]));
    let held = left(&mut first).unwrap();
    let Err(err) = handing_back_changed(&shop(), Some(&held), &[], vec![Some(0)], |a| {
        a.insert(
            AttachmentId(2),
            Attachment::Bytes(b"\xffnot a query".to_vec()),
        );
    }) else {
        panic!("a query that is not JSON was read");
    };
    assert!(err.drop_held.is_empty(), "{err:?}");
}

#[test]
fn a_state_made_under_another_source_is_not_read() {
    // A memo of the renamed source, handed to a run over the shop: made
    // under another description, it is not read; the run checks and warns
    // as a first run does, and leaves a memo of its own source.
    let mut other = materialize(host(), &renamed(), QUERY, None).unwrap();
    steps_to_the_result(&mut other, &|| indexes(&[]));
    let foreign = left(&mut other).unwrap();
    assert!(memo(&foreign).indexes_checked);
    let mut run = handing_back(&shop(), Some(&foreign), &[], vec![Some(0), Some(1)]).unwrap();
    let steps = steps_to_the_result(&mut run, &|| indexes(&[]));
    assert_eq!(steps.0[0].0.name(), "describe");
    assert_eq!(steps.0[1].0.name(), "time_bounds");
    assert_eq!(steps.1, ["MISSING_INDICES"]);
    let own = left(&mut run).unwrap();
    assert_ne!(memo(&own).source, memo(&foreign).source);
    // A column's meaning edited since is not another source: the memo is
    // read, as the reference's sampler, built once, keeps what it read.
    let mut edited = shop();
    edited.tables[0].columns[2].stype = Stype::Categorical;
    let mut run = handing_back(&edited, Some(&own), &[], vec![Some(0), Some(1)]).unwrap();
    let steps = steps_to_the_result(&mut run, &|| indexes(&[]));
    assert_eq!(steps.0[0].0.name(), "sample_entities");
}

/// The shop with the users born: 30 at the epoch, every other one in
/// 2100, so few are entities at any anchor of the walk back.
fn shop_born() -> SourceDescription {
    let mut born = shop();
    born.tables[1].columns.push(column(
        "born",
        ColumnType::Timestamp {
            unit: Unit::Us,
            zone: None,
        },
        Dtype::Date,
        Stype::Timestamp,
        false,
    ));
    born.tables[1].time_column = Some("born".into());
    born
}

#[test]
fn the_coverage_warning_is_given_on_every_run_that_meets_it() {
    // A sample of 10,000 entities giving fewer than 500 context examples:
    // the reference warns once in a process, which the core, holding no
    // state across runs, leaves to its host (`sdm-host`); it gives the
    // warning on every run that meets it, the second included.
    let sample = |columns: &[String]| -> FetchResult {
        let n = 10_000_usize;
        data(
            columns
                .iter()
                .map(|c| {
                    let column: ArrayRef = match c.as_str() {
                        "user_id" => Arc::new(Int64Array::from_iter_values(0..n as i64)),
                        "born" => Arc::new(TimestampMicrosecondArray::from_iter_values(
                            (0..n).map(|i| if i < 30 { 0 } else { 4_102_444_800_000_000 }),
                        )),
                        other => shop_column(other, n, None),
                    };
                    (c.as_str(), column)
                })
                .collect(),
        )
    };
    for _ in 0..2 {
        let mut run = materialize(host(), &shop_born(), QUERY, None).unwrap();
        let (_, warnings) = try_steps_sampling(&mut run, &|| indexes(&[]), &sample).unwrap();
        assert_eq!(warnings, ["MISSING_INDICES", "CONTEXT_COVERAGE"]);
    }
}

// --- the caller's inference options (T1.28a) ---------------------------------

#[test]
fn inference_options_nested_past_the_reader_s_depth_are_refused_by_name() {
    // E49's probe of the one JSON reader T1.28a adds: the caller's
    // inference options, read with the wire's bounded reader before the run
    // exists, so ten thousand nested levels are refused by name, never read
    // recursively, and one level under the bound is read (and refused for
    // what it is, an option the engine does not read, when the run reaches
    // it).
    let start = |text: String| {
        let (mut operation, mut attachments) = materialize_start(duckdb(), &shop(), QUERY, None);
        let Operation::Materialize(request) = &mut operation else {
            unreachable!()
        };
        request.values = Some(AttachmentId(9));
        attachments.insert(AttachmentId(9), Attachment::Bytes(text.into_bytes()));
        materialize_on(Arc::new(sdm_data::Data::new()), operation, attachments)
    };
    let deep = format!(
        r#"{{"inference_config": {{"num_estimators": {}1{}}}}}"#,
        "[".repeat(10_000),
        "]".repeat(10_000)
    );
    let Err(err) = start(deep) else {
        panic!("refused before the run exists");
    };
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    assert_eq!(err.fields["field"], Detail::text("values"));
    assert_eq!(err.fields["category"], Detail::text("depth"));
    let within = format!(
        r#"{{"inference_config": {{"num_estimators": {}1{}}}}}"#,
        "[".repeat(100),
        "]".repeat(100)
    );
    assert!(start(within).is_ok(), "read within the bound");
    let Err(not_json) = start("{".into()) else {
        panic!("not JSON");
    };
    assert_eq!(not_json.fields["field"], Detail::text("values"));
}
