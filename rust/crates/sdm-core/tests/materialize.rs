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
    FetchPlan, MATCHES_COLUMN, OptionalRequest, ProviderCapabilities, ProviderChoice, TextTimes,
    catalog_columns,
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
        stype_before_key: None,
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
            max_seed: None,
            text_times: None,
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
        num_context_samples: None,
        options: vec![],
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
    }
}

#[test]
fn the_core_refuses_a_seed_past_the_largest_duckdb_declares() {
    for (seed, refused) in [
        ((1_u128 << 31) - 1, false),
        (1 << 31, true),
        (1 << 70, true),
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
fn host_up_to(max_seed: Option<u64>) -> ProviderChoice {
    let ProviderChoice::Host { mut capabilities } = host() else {
        unreachable!()
    };
    capabilities.max_seed = max_seed;
    ProviderChoice::Host { capabilities }
}

#[test]
fn a_host_provider_s_declared_range_is_the_one_the_core_holds_it_to() {
    for (max_seed, seed, refused) in [
        (Some(100), 100_u128, false),
        (Some(100), 101, true),
        (Some(0), 0, false),
        (Some(0), 1, true),
        (Some(u64::MAX), u128::from(u64::MAX), false),
        (Some(u64::MAX), 1 << 64, true),
        (None, 1 << 70, false),
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
                    Detail::Scalar(Scalar::UInt64(max_seed.unwrap()))
                );
            }
        }
    }
}

/// The shipped data layer, but for the seed range it gives DuckDB, which
/// it forgets: a seeded fault in the capability.
struct Forgetful(sdm_data::Data);

impl DataLayer for Forgetful {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        let mut capabilities = self.0.capabilities(provider)?;
        capabilities.max_seed = None;
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
    // Seeded: DuckDB's declared range forgotten, a seed of 2**31 is no
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
    assert_eq!(rule(&refusal(&mut run)), "parent_table");
    // A key of several columns, of a table no link refers to.
    let mut composite = shop();
    composite.tables[0].primary_key = vec!["order_id".into(), "user_id".into()];
    let mut run = materialize(host(), &composite, QUERY, None).unwrap();
    assert_eq!(rule(&refusal(&mut run)), "composite_key");
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
        warnings.extend(run.take_warnings().into_iter().map(|w| w.code));
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
        "ts" => Arc::new(TimestampMicrosecondArray::from(vec![None::<i64>; n])),
        _ => Arc::new(StringArray::from(vec![None::<&str>; n])),
    }
}

/// Runs `run` to its result over the renamed shop, answering every request:
/// the catalog as [`catalog`] does, the sample with four users, the labels'
/// and the first hop's neighbors with no row, the lookup with one row per
/// seed; each request's kind and the table it names, in order.
fn tables_to_the_result(run: &mut Run) -> Vec<(&'static str, String)> {
    let mut seen = Vec::new();
    loop {
        let step = run.next().unwrap();
        let _ = run.take_warnings();
        let requests = match step {
            Step::Fetch { requests } => requests,
            Step::Done { .. } => return seen,
            other => panic!("{other:?}"),
        };
        for asked in requests {
            let answer = match &asked.request {
                DataRequest::Describe { .. } => indexes(&[]),
                DataRequest::TimeBounds { columns } => {
                    seen.push(("time_bounds", columns[0].table.clone()));
                    catalog(&asked.request).unwrap()
                }
                DataRequest::SampleEntities { table, columns, .. } => {
                    seen.push(("sample_entities", table.clone()));
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
                DataRequest::Neighbors { table, columns, .. } => {
                    seen.push(("neighbors", table.clone()));
                    let mut answer: Vec<(&str, ArrayRef)> = columns
                        .iter()
                        .map(|c| (c.as_str(), shop_column(c, 0, None)))
                        .collect();
                    answer.push((BATCH_COLUMN, Arc::new(Int64Array::from(Vec::<i64>::new()))));
                    data(answer)
                }
                DataRequest::Lookup {
                    table,
                    columns,
                    seeds,
                    ..
                } => {
                    seen.push(("lookup", table.clone()));
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
            DataRequest::TimeBounds { columns } => Some(columns[0].table.clone()),
            DataRequest::SampleEntities { table, .. } => Some(table.clone()),
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
    let refused = |change: &dyn Fn(&mut Materialize)| rule(&with(change).unwrap_err());
    assert_eq!(
        refused(&|m| m.options = vec!["num_neighbors".into(), "inference_config".into()]),
        "inference_config"
    );
    assert_eq!(refused(&|m| m.unseeded = true), "unseeded");
    assert_eq!(
        refused(&|m| m.num_context_samples = Some(10)),
        "num_context_samples"
    );
    assert_eq!(refused(&|m| m.anchor_entity = true), "anchor_entity");
    // The options before the seed, the seed before the context size, as
    // the bridge refused them.
    assert_eq!(
        refused(&|m| {
            m.unseeded = true;
            m.options = vec!["top_k".into()];
        }),
        "top_k"
    );
    assert_eq!(
        refused(&|m| {
            m.unseeded = true;
            m.num_context_samples = Some(10);
        }),
        "unseeded"
    );
    // A name that is not one of the reference's options is the host's
    // fault.
    let err = with(&|m| m.options = vec!["no_such_option".into()]).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest, "{err:?}");
    assert_eq!(err.fields["field"], Detail::text("options"));
    for name in [
        "context_anchor_time",
        "exclude_cols_dict",
        "lag_timesteps",
        "max_pq_iterations",
        "num_hops",
        "return_embeddings",
        "use_prediction_time",
    ] {
        assert_eq!(refused(&|m| m.options = vec![name.into()]), name);
    }
    assert!(matches!(with(&|_| {}), Ok(Step::Fetch { .. })));
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
    // No reading declared, the client's rules (not ported), a value whose
    // reading takes a rule outside the shapes, and offsets mixed: each by
    // its rule's name, before any text is read as a time.
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
    // The reference's engine drops a value its reading takes as no time and
    // warns of it, with how many of how many and three examples.
    let mut run = materialize(
        host_reading(Some(TextTimes::ValueByValue)),
        &shop_with_times(ColumnType::Utf8, Dtype::String),
        QUERY,
        None,
    )
    .unwrap();
    let (payload, warnings) = times_to_the_end(&mut run, &|n| {
        Arc::new(StringArray::from(vec!["not a date"; n])) as ArrayRef
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
            Detail::List(vec![Scalar::Utf8("not a date".into())])
        );
    }
}
