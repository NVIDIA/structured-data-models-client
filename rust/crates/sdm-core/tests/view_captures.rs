// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric-view description and refresh behavior over small authored warehouse replies.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_contract::data::{DataRequest, ProviderChoice, ProviderError};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    DescribeSource, Given, NamedTable, Operation, OperationResult, SourceDescription, SourceSpec,
};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ColumnType, ErrorCode, HostFacts, Scalar, Step, Unit};
use sdm_core::{
    ArrowData, Attachment, Attachments, CompositionRoot, FetchResult, UnknownZone, Zones, start,
};
use serde_json::{Value as Json, json};

struct Utc;

impl Zones for Utc {
    fn zone(&self, name: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        match name {
            "UTC" | "Etc/UTC" => Ok(jiff::tz::TimeZone::UTC),
            _ => Err(UnknownZone),
        }
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_761_264_000_000_000,
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

fn view_inputs() -> Json {
    json!({"cases": [
        {"case": "view.metric.from-source", "leg": "databricks-sql", "input": {"view": {"trust_expressions": false}}},
        {"case": "view.metric.wildcards-nested-join", "leg": "databricks-sql", "input": {"view": {"trust_expressions": false}}}
    ]})
}

// Small authored warehouse inputs for the ordinary metric-view regression tests.
// Types come from the simulated catalog, never from the core's expected schema.
fn ordinary_rows(request: &DataRequest, edits: &Edits, nested: bool) -> ArrowData {
    use arrow_array::{
        ArrayRef, Float64Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
    };
    use arrow_schema::{Field, Schema};
    use sdm_contract::data::DescribePart;
    let strings = |names: &[&str], rows: Vec<Vec<String>>| {
        let arrays: Vec<ArrayRef> = (0..names.len())
            .map(|i| {
                Arc::new(StringArray::from(
                    rows.iter().map(|r| r[i].as_str()).collect::<Vec<_>>(),
                )) as ArrayRef
            })
            .collect();
        let fields = names
            .iter()
            .zip(&arrays)
            .map(|(name, a)| Field::new(*name, a.data_type().clone(), true))
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        ArrowData::new(schema, vec![batch])
    };
    match request {
        DataRequest::Describe {
            part: DescribePart::ViewDefinition,
            ..
        } => {
            let mut yaml = r#"version: 0.1
source: ph_catalog.ph_schema.orders
filter: source.price > 50
joins:
  - name: users
    source: ph_catalog.ph_schema.users
    on: source.user_id = users.user_id
dimensions:
  - name: order_id
    expr: source.order_id
  - name: user_id
    expr: source.user_id
  - name: ts
    expr: source.ts
  - name: PRICE
    expr: source.price
  - name: price_band
    expr: CASE WHEN source.price > 100 THEN 'high' ELSE 'low' END
  - name: country
    expr: users.country
  - name: age
    expr: users.age
measures:
  - name: revenue
    expr: SUM(source.price)
"#
            .to_owned();
            if nested {
                yaml = yaml.replace(
                    "    on: source.user_id = users.user_id",
                    "    on: source.user_id = users.user_id\n    joins:\n      - name: countries\n        source: ph_catalog.ph_schema.countries\n        on: users.country = countries.country",
                ).replace("measures:", "  - name: region\n    expr: users.countries.region\nmeasures:");
            }
            if nested {
                let start = yaml.find("dimensions:").unwrap();
                let end = yaml.find("measures:").unwrap();
                yaml.replace_range(start..end, "dimensions:\n  - expr: source.* EXCEPT (price)\n  - expr: users.* EXCEPT (age, user_id)\n  - name: region\n    expr: users.countries.region\n");
            }
            if let Some((from, to)) = edits.definition {
                yaml = yaml.replace(from, to);
            }
            strings(
                &["col_name", "data_type", "comment", "metadata"],
                [
                    ("# Detailed Table Information", ""),
                    ("Catalog", "ph_catalog"),
                    ("Database", "ph_schema"),
                    ("Type", "METRIC_VIEW"),
                    ("View Text", &yaml),
                ]
                .into_iter()
                .map(|(a, b)| vec![a.into(), b.into(), String::new(), String::new()])
                .collect(),
            )
        }
        DataRequest::Describe {
            tables,
            part: DescribePart::Columns,
        } => {
            let mut rows = Vec::new();
            for table in tables {
                let name = table.source.last().unwrap().as_str();
                let columns: &[(&str, &str)] = match name {
                    "orders" => &[
                        ("order_id", "bigint"),
                        ("country", "string"),
                        ("user_id", "bigint"),
                        ("price", "double"),
                        ("ts", "timestamp_ntz"),
                    ],
                    "countries" => &[("country", "string"), ("region", "string")],
                    "users" => &[("user_id", "bigint"), ("age", "int"), ("country", "string")],
                    _ => panic!("unexpected physical table {name}"),
                };
                for (column, dtype) in columns {
                    rows.push(vec![
                        (*column).into(),
                        (*dtype).into(),
                        "YES".into(),
                        "ph_catalog".into(),
                        "ph_schema".into(),
                        name.into(),
                        "MANAGED".into(),
                    ]);
                }
            }
            strings(
                &[
                    "column_name",
                    "full_data_type",
                    "is_nullable",
                    "table_catalog",
                    "table_schema",
                    "table_name",
                    "table_type",
                ],
                rows,
            )
        }
        DataRequest::Describe {
            part: DescribePart::Keys,
            ..
        } => strings(
            &["constraint_type", "constraint_name", "column_name"],
            vec![],
        ),
        DataRequest::Preview { table, columns, .. } => {
            let physical = table.source.last().unwrap().as_str();
            let arrays: Vec<ArrayRef> = columns
                .iter()
                .map(|name| match (physical, name.as_str()) {
                    ("orders", "order_id") => {
                        Arc::new(Int64Array::from(vec![10, 11, 12])) as ArrayRef
                    }
                    ("orders" | "users", "user_id") => Arc::new(Int64Array::from(vec![0, 1, 2])),
                    ("orders", "PRICE" | "price") => {
                        Arc::new(Float64Array::from(vec![60.0, 90.0, 120.0]))
                    }
                    ("orders", "ts")
                        if edits
                            .definition
                            .is_some_and(|(_, to)| to.contains("DATE_TRUNC")) =>
                    {
                        Arc::new(TimestampMicrosecondArray::from(vec![
                            1_698_796_800_000_000,
                            1_698_796_800_000_000,
                            1_701_388_800_000_000,
                        ]))
                    }
                    ("orders", "ts" | "ts_raw") => Arc::new(TimestampMicrosecondArray::from(vec![
                        1_700_000_000_000_000,
                        1_701_000_000_000_000,
                        1_702_000_000_000_000,
                    ])),
                    ("orders", "price_band") => {
                        Arc::new(StringArray::from(vec!["low", "low", "high"]))
                    }
                    ("users", "age") => Arc::new(Int32Array::from(vec![20, 30, 40])),
                    ("countries", "region") => {
                        Arc::new(StringArray::from(vec!["north", "north", "south"]))
                    }
                    ("countries", "country") => Arc::new(StringArray::from(vec!["US", "CA", "MX"])),
                    ("users" | "orders", "country") => {
                        Arc::new(StringArray::from(vec!["US", "CA", "US"]))
                    }
                    _ => panic!("unexpected physical preview {physical}.{name}"),
                })
                .collect();
            let schema = Arc::new(Schema::new(
                columns
                    .iter()
                    .zip(&arrays)
                    .map(|(name, a)| Field::new(name, a.data_type().clone(), true))
                    .collect::<Vec<_>>(),
            ));
            let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
            ArrowData::new(schema, vec![batch])
        }
        DataRequest::KeyCounts { keys } => {
            for key in keys {
                let table = key.table.source.last().unwrap().as_str();
                assert!(
                    (table == "users" && key.columns == ["user_id"])
                        || (table == "countries" && key.columns == ["country"])
                        || (table == "orders" && key.columns == ["order_id"]),
                    "unexpected authored key: {key:?}"
                );
            }
            let arrays: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from_iter_values(0..keys.len() as i32)),
                Arc::new(Int64Array::from(vec![3; keys.len()])),
                Arc::new(Int64Array::from(vec![3; keys.len()])),
                Arc::new(Int64Array::from(vec![0; keys.len()])),
            ];
            let schema = Arc::new(Schema::new(
                ["k", "n", "d", "z"]
                    .iter()
                    .zip(&arrays)
                    .map(|(name, a)| Field::new(*name, a.data_type().clone(), false))
                    .collect::<Vec<_>>(),
            ));
            let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
            ArrowData::new(schema, vec![batch])
        }
        _ => panic!("unhandled ordinary warehouse request: {request:?}"),
    }
}

fn view_name(_case: &Json) -> String {
    "ph_catalog.ph_schema.metric_view".into()
}

/// Observed description, requests and diagnostics.
struct DescribedRun {
    ended: Result<SourceDescription, sdm_contract::CoreError>,
    requests: Vec<DataRequest>,
    warnings: Vec<sdm_contract::CoreWarning>,
    notes: Vec<sdm_contract::ops::ViewNote>,
}

/// An answer of a test's own to the requests it names.
type Answer = Box<dyn Fn(&DataRequest) -> Option<FetchResult>>;

/// Per-test definition changes and explicit provider outcomes.
#[derive(Default)]
struct Edits {
    definition: Option<(&'static str, &'static str)>,
    answer: Option<Answer>,
}

/// A provider's failure under `sqlstate`.
fn refusal(sqlstate: &str) -> FetchResult {
    FetchResult::Error(ProviderError {
        driver_message: None,
        kind: sdm_contract::ErrorKind::DataSource,
        retryable: false,
        sqlstate: Some(sqlstate.to_owned()),
        vendor_code: None,
        error_class: None,
    })
}

/// Build protocol attachments from the contract types, as the hosts do.
/// In particular, each explicit override is a tagged `Given` value.
fn source_spec() -> SourceSpec {
    SourceSpec {
        tables: Vec::new(),
        primary_keys: BTreeMap::new(),
        time_columns: BTreeMap::new(),
        relationships: None,
    }
}

/// Describe the authored view through the real core operation.
fn describe_with(document: &Json, case: &Json, edits: &Edits, strict: bool) -> DescribedRun {
    describe_spec(document, case, edits, strict, source_spec())
}

fn describe_spec(
    _document: &Json,
    case: &Json,
    edits: &Edits,
    _strict: bool,
    spec: SourceSpec,
) -> DescribedRun {
    let trust = case["input"]["view"]["trust_expressions"]
        .as_bool()
        .unwrap_or(false);
    let provider: ProviderChoice = serde_json::from_value(json!({
        "provider": "sql", "dialect": "databricks",
        "view": {"name": view_name(case), "trust_expressions": trust},
    }))
    .unwrap();
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::reading_views(
        sdm_core::pql::spaces(),
    )));
    let mut run = start(
        &root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::DescribeSource(DescribeSource {
                provider,
                source: sdm_contract::data::AttachmentId(1),
                check_keys: false,
            }),
        },
        Attachments::from([(
            sdm_contract::data::AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec).unwrap()),
        )]),
        Arc::new(Utc),
    )
    .unwrap();
    let mut requests = Vec::new();
    let ended = |run: &mut sdm_core::Run, requests, ended| DescribedRun {
        ended,
        requests,
        warnings: run.take_warnings(),
        notes: Vec::new(),
    };
    loop {
        let step = match run.next() {
            Ok(step) => step,
            Err(error) => return ended(&mut run, requests, Err(error)),
        };
        match step {
            Step::Done { result } => {
                let OperationResult::DescribeSource(described) = result else {
                    panic!("not a description");
                };
                let Some(Attachment::Bytes(bytes)) = run.take_attachment(described.description)
                else {
                    panic!("no description attached");
                };
                let description = serde_json::from_slice(&bytes).unwrap();
                let mut described_run = ended(&mut run, requests, Ok(description));
                described_run.notes = described.notes;
                return described_run;
            }
            Step::Fetch {
                requests: asked_now,
            } => {
                for asked in asked_now {
                    requests.push(asked.request.clone());
                    if let Some(answer) = edits.answer.as_ref().and_then(|f| f(&asked.request)) {
                        run.submit_fetch(asked.id, answer).unwrap();
                        continue;
                    }
                    let answer = FetchResult::Data(ordinary_rows(
                        &asked.request,
                        edits,
                        case["case"] == "view.metric.wildcards-nested-join",
                    ));
                    run.submit_fetch(asked.id, answer).unwrap();
                }
            }
            Step::CallModel(_) => panic!("a description called a model"),
        }
    }
}

/// Select an authored view variant.
fn case<'d>(document: &'d Json, id: &str) -> &'d Json {
    document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["case"] == id && c["leg"] == "databricks-sql")
        .unwrap()
}

/// Whether `request` reads the base table `table` (its last name part).
fn reads(request: &DataRequest, table: &str) -> bool {
    request
        .relations()
        .iter()
        .any(|(_, relation)| relation.source.last().is_some_and(|t| t == table))
}

/// Whether `request` samples an expression.
fn samples(request: &DataRequest) -> bool {
    matches!(request, DataRequest::Preview { table, .. } if table.columns.iter().any(|c| c.expression.is_some()))
}

/// S5's first sequence: a filter the view cannot apply refuses it before
/// any joined table's columns are read (`RE/graph.py:1641-1646`).
#[test]
fn a_filter_is_refused_before_any_joined_table_is_read() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "filter: source.price > 50",
            "filter: my_udf(source.price) > 50",
        )),
        answer: None,
    };
    let described_run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let Err(refused) = &described_run.ended else {
        panic!("a filter calling an untrusted function was applied");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewFilterRefusedUntrusted,
        "{refused:?}"
    );
    assert!(
        described_run.requests.iter().any(|r| reads(r, "orders")),
        "the source table's columns were not read: {:?}",
        described_run.requests
    );
    assert!(
        !described_run.requests.iter().any(|r| reads(r, "users")),
        "a joined table was read before the filter was refused: {:?}",
        described_run.requests
    );
}

/// Dropping a parent join also drops its descendants before expression sampling.
#[test]
fn a_join_dropped_takes_its_children_before_their_samples() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "  - name: region\n    expr: users.countries.region",
            "  - name: region\n    expr: users.countries.region\n  - name: area\n    expr: UPPER(users.countries.region)",
        )),
        answer: Some(Box::new(|request: &DataRequest| match request {
            // `users`' key repeats, so its join is dropped.
            DataRequest::KeyCounts { keys } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("k", arrow_schema::DataType::Int32, false),
                    arrow_schema::Field::new("n", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("d", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("z", arrow_schema::DataType::Int64, false),
                ]));
                let n = keys.len();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32)),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![5; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![0; n])),
                    ],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            DataRequest::RepeatedKey { key, .. } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    &key[0],
                    arrow_schema::DataType::Int64,
                    true,
                )]));
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(arrow_array::Int64Array::from(vec![1]))],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            _ => None,
        })),
    };
    let described_run = describe_with(
        &document,
        case(&document, "view.metric.wildcards-nested-join"),
        &edits,
        false,
    );

    let Ok(description) = &described_run.ended else {
        panic!("{:?}", described_run.ended.err());
    };
    let tables: Vec<&str> = description.tables.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tables, ["orders"]);
    assert!(
        !described_run
            .requests
            .iter()
            .any(|r| samples(r) && reads(r, "countries")),
        "a dropped join's child was sampled: {:?}",
        described_run.requests
    );
    let said: Vec<&str> = described_run
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .collect();
    assert!(
        said.iter()
            .any(|w| w.starts_with("Failed to add join 'users'") && w.contains("is not unique")),
        "{said:?}"
    );
    assert!(
        said.iter()
            .any(|w| w.starts_with("Failed to add join 'countries'")
                && w.contains("'area'")
                && w.contains("its parent join 'users' was not added")),
        "{said:?}"
    );
}

/// S5's fourth sequence: an expression's sample the warehouse refuses with
/// `42501` (the caller may not read what it reads) leaves that column out,
/// retried alone first, as any class 22 or 42 refusal does (row 45).
#[test]
fn a_42501_on_an_expression_sample_leaves_that_column_out() {
    let document = view_inputs();
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request: &DataRequest| {
            samples(request).then(|| refusal("42501"))
        })),
    };
    let described_run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );

    let Ok(description) = &described_run.ended else {
        panic!("{:?}", described_run.ended.err());
    };
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    assert!(
        !orders.columns.iter().any(|c| c.name == "price_band"),
        "{:?}",
        orders.columns
    );
    // Sampled with the batch, then alone.
    assert_eq!(
        described_run.requests.iter().filter(|r| samples(r)).count(),
        2
    );
    let refused = described_run
        .warnings
        .iter()
        .find(|w| w.sentence == sdm_contract::Sentence::ViewColumnRefused)
        .expect("no warning of the column left out");
    assert_eq!(
        refused.message,
        "Failed to add column 'price_band' to table 'orders': permission was denied while evaluating the expression (SQLSTATE: 42501)"
    );
}

#[test]
fn metric_view_description_preserves_columns_sources_and_join_identity() {
    let document = view_inputs();
    let description = described(&document, "view.metric.from-source");
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    let users = description
        .tables
        .iter()
        .find(|t| t.name == "users")
        .unwrap();
    assert_eq!(description.tables.len(), 2);
    assert_eq!(orders.source.source, ["orders"]);
    assert_eq!(users.source.source, ["users"]);
    assert!(orders.primary_key.is_empty());
    for (name, dtype, stype) in [
        ("user_id", "int64", sdm_contract::ops::Stype::Id),
        ("PRICE", "float64", sdm_contract::ops::Stype::Numerical),
        (
            "price_band",
            "string",
            sdm_contract::ops::Stype::Categorical,
        ),
        ("ts", "timestamp[us]", sdm_contract::ops::Stype::Timestamp),
    ] {
        let column = orders.columns.iter().find(|c| c.name == name).unwrap();
        assert_eq!(column.dtype.wire_name(), dtype, "{name}");
        assert_eq!(column.stype, stype, "{name}");
    }

    assert_eq!(
        orders
            .columns
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["order_id", "user_id", "ts", "PRICE", "price_band"]
    );
    assert_eq!(
        users
            .columns
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["country", "age", "user_id"]
    );
    assert_eq!(orders.time_column.as_deref(), Some("ts"));
    assert_eq!(users.primary_key, ["user_id"]);
    assert_eq!(description.links.len(), 1);
    let link = &description.links[0];
    assert_eq!(
        (
            &*link[0].table,
            &*link[0].column,
            &*link[1].table,
            &*link[1].column
        ),
        ("orders", "user_id", "users", "user_id")
    );
    assert!(!orders.columns.iter().any(|c| c.name == "revenue"));
}

#[test]
fn metric_query_sources_and_join_filters_have_named_refusals() {
    let document = view_inputs();
    for (from, to, sentence) in [
        (
            "source: ph_catalog.ph_schema.orders",
            "source: SELECT * FROM ph_catalog.ph_schema.orders",
            sdm_contract::Sentence::ViewSourceUnsupported,
        ),
        (
            "filter: source.price > 50",
            "filter: users.country = 'US'",
            sdm_contract::Sentence::ViewFilterRefused,
        ),
    ] {
        let edits = Edits {
            definition: Some((from, to)),
            answer: None,
        };
        let run = describe_with(
            &document,
            case(&document, "view.metric.from-source"),
            &edits,
            false,
        );
        let error = run.ended.unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
        assert_eq!(error.sentence, sentence);
        assert!(!run.requests.iter().any(samples));
    }
}

#[test]
fn unsupported_join_condition_omits_only_its_table() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "on: source.user_id = users.user_id",
            "on: source.user_id = users.user_id AND users.age > 20",
        )),
        answer: None,
    };
    let run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let description = run.ended.unwrap();
    assert_eq!(
        description
            .tables
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["orders"]
    );
    assert!(description.links.is_empty());
    assert!(
        run.warnings
            .iter()
            .any(|w| w.message.contains("users") && w.message.contains("join"))
    );
}

#[test]
fn nested_metric_join_preserves_both_relationships() {
    let document = view_inputs();
    let run = describe_with(
        &document,
        case(&document, "view.metric.wildcards-nested-join"),
        &Edits::default(),
        false,
    );
    let description = run.ended.unwrap();
    assert_eq!(
        description
            .tables
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["orders", "users", "countries"]
    );
    let countries = description
        .tables
        .iter()
        .find(|t| t.name == "countries")
        .unwrap();
    assert_eq!(countries.primary_key, ["country"]);
    let users = description
        .tables
        .iter()
        .find(|t| t.name == "users")
        .unwrap();
    assert!(!users.columns.iter().any(|c| c.name == "age"));
    assert!(users.columns.iter().any(|c| c.name == "user_id"));
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    assert!(orders.columns.iter().any(|c| c.name == "price")); // Revenue depends on it.

    assert!(countries.columns.iter().any(|c| c.name == "region"));
    let mut links = description
        .links
        .iter()
        .map(|link| {
            (
                link[0].table.as_str(),
                link[0].column.as_str(),
                link[1].table.as_str(),
                link[1].column.as_str(),
            )
        })
        .collect::<Vec<_>>();
    links.sort();
    assert_eq!(
        links,
        [
            ("orders", "user_id", "users", "user_id"),
            ("users", "country", "countries", "country")
        ]
    );
}

#[test]
fn parameter_defaults_are_substituted_before_view_reads() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "filter: source.price > 50",
            "parameters:\n  - name: min_price\n    data_type: DOUBLE\n    default: 50\nfilter: source.price >= min_price",
        )),
        answer: None,
    };
    let run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    assert!(run.ended.is_ok(), "{:?}", run.ended);
    let filters = run
        .requests
        .iter()
        .flat_map(|r| r.relations())
        .filter_map(|(_, r)| r.filter.as_deref())
        .collect::<Vec<_>>();
    assert!(!filters.is_empty());
    assert!(
        filters
            .iter()
            .all(|f| *f == "source.price >= CAST('50' AS DOUBLE)"),
        "{filters:?}"
    );
}

#[test]
fn derived_month_keeps_raw_timestamp_as_sampling_time() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some(("expr: source.ts", "expr: DATE_TRUNC('MONTH', source.ts)")),
        answer: None,
    };
    let run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let description = run.ended.unwrap();
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    assert_eq!(orders.time_column.as_deref(), Some("ts_raw"));
    let projections = run
        .requests
        .iter()
        .flat_map(|r| r.relations())
        .flat_map(|(_, r)| r.columns.iter())
        .collect::<Vec<_>>();
    assert!(
        projections
            .iter()
            .any(|c| c.name == "ts" && c.expression.as_deref() == Some("DATE_TRUNC('MONTH', ts)")),
        "{projections:?}"
    );
    assert!(
        projections.iter().any(|c| c.name == "ts_raw"
            && c.column.as_deref() == Some("ts")
            && c.expression.is_none()),
        "{projections:?}"
    );

    for name in ["ts", "ts_raw"] {
        let column = orders.columns.iter().find(|c| c.name == name).unwrap();
        assert_eq!(column.dtype.wire_name(), "timestamp[us]");
        assert_eq!(
            column.received,
            ColumnType::Timestamp {
                unit: Unit::Us,
                zone: None
            }
        );
    }
}

#[test]
fn untrusted_dimension_is_omitted_without_losing_other_columns() {
    let document = view_inputs();
    for (expression, trust) in [
        ("bitmap_bucket_number(source.order_id)", false),
        ("trim(FROM source.price)", true),
    ] {
        let mut input = case(&document, "view.metric.from-source").clone();
        input["input"]["view"]["trust_expressions"] = json!(trust);
        let edits = Edits {
            definition: Some((
                "CASE WHEN source.price > 100 THEN 'high' ELSE 'low' END",
                expression,
            )),
            answer: None,
        };
        let run = describe_with(&document, &input, &edits, false);
        let description = run.ended.unwrap();
        let orders = description
            .tables
            .iter()
            .find(|t| t.name == "orders")
            .unwrap();
        assert_eq!(
            orders
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["order_id", "user_id", "ts", "PRICE"]
        );
        assert!(
            run.warnings
                .iter()
                .any(|w| w.message.contains("price_band"))
        );
        assert!(!run.requests.iter().any(|r| matches!(r, DataRequest::Preview { columns, .. } if columns.iter().any(|c| c == "price_band"))));
    }
}

#[test]
fn sibling_joins_keep_distinct_keys_and_column_ownership() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "dimensions:",
            "  - name: countries\n    source: ph_catalog.ph_schema.countries\n    on: source.country = countries.country\ndimensions:\n  - name: region\n    expr: countries.region",
        )),
        answer: None,
    };
    let run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let description = run.ended.unwrap();
    assert_eq!(description.tables.len(), 3);
    for (table, key, attribute, absent) in [
        ("users", "user_id", "age", "region"),
        ("countries", "country", "region", "age"),
    ] {
        let child = description.tables.iter().find(|t| t.name == table).unwrap();
        assert_eq!(child.primary_key, [key]);
        assert!(child.columns.iter().any(|c| c.name == attribute));
        assert!(!child.columns.iter().any(|c| c.name == absent));
    }
    let mut links = description
        .links
        .iter()
        .map(|link| {
            (
                link[0].table.as_str(),
                link[0].column.as_str(),
                link[1].table.as_str(),
                link[1].column.as_str(),
            )
        })
        .collect::<Vec<_>>();
    links.sort();
    assert_eq!(
        links,
        [
            ("orders", "country", "countries", "country"),
            ("orders", "user_id", "users", "user_id")
        ]
    );
}

/// The schema a host holds of `description`, each column with the facts
/// the core gave it.
fn held(description: &SourceDescription) -> sdm_contract::ops::HeldSchema {
    use sdm_contract::ops::{HeldColumn, HeldSchema, HeldTable, SourceFacts};
    HeldSchema {
        inference: None,
        tables: description
            .tables
            .iter()
            .map(|table| HeldTable {
                calculated_columns: vec![],
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table
                    .columns
                    .iter()
                    .map(|c| HeldColumn {
                        name: c.name.clone(),
                        dtype: Some(c.dtype.wire_name().into()),
                        stype: c.stype,
                        stype_before_key: c.stype_before_key,
                        source: Some(SourceFacts {
                            relation: table.source.clone(),
                            dtype: c.dtype,
                            received: c.received.clone(),
                            nullable: c.nullable,
                            unique: c.unique,
                            declared_type: c.declared_type.clone(),
                            catalog: None,
                            exact_number: c.exact_number,
                        }),
                    })
                    .collect(),
                primary_key: table.primary_key.clone(),
                time_column: table.time_column.clone(),
                end_time_column: table.end_time_column.clone(),
                duplicate_keys: None,
            })
            .collect(),
        links: description.links.clone(),
    }
}

/// How a `materialize` over a view's schema read the view: the first
/// request past plan B's reads (the view read again), or how it ended
/// before.
enum Located {
    Past(Box<DataRequest>),
    Ended(sdm_contract::CoreError),
}

/// `materialize` of `held` over the view of `case`, its view read as the
/// recording (edited by `edits`) answers, to the first request past plan
/// B's reads.
fn materialize_over(
    document: &Json,
    case: &Json,
    held: &sdm_contract::ops::HeldSchema,
    edits: &Edits,
) -> Located {
    materialize_query(
        document,
        case,
        held,
        edits,
        "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id",
    )
}

fn materialize_query(
    _document: &Json,
    case: &Json,
    held: &sdm_contract::ops::HeldSchema,
    edits: &Edits,
    query: &str,
) -> Located {
    use sdm_contract::data::AttachmentId;
    use sdm_contract::ops::Materialize;
    let provider: ProviderChoice = serde_json::from_value(json!({
        "provider": "sql", "dialect": "databricks",
        "view": {"name": view_name(case), "trust_expressions": false},
    }))
    .unwrap();
    let indices = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "indices",
            arrow_schema::DataType::Int64,
            false,
        )])),
        vec![Arc::new(arrow_array::Int64Array::from(vec![0, 1])) as arrow_array::ArrayRef],
    )
    .unwrap();
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::reading_views(
        sdm_core::pql::spaces(),
    )));
    let mut run = start(
        &root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::Materialize(Materialize {
                provider,
                schema: AttachmentId(1),
                query: AttachmentId(2),
                indices: Some(AttachmentId(3)),
                anchor_time: Some(Scalar::Timestamp {
                    unit: Unit::Ns,
                    zone: None,
                    value: 1_709_251_200_000_000_000,
                }),
                seed: None,
                unseeded: false,
                anchor_entity: false,
                values: None,
                options: vec![],
                most_parts: None,
                most_entities: None,
                most_calls: None,
                candidates: None,
                sampler_state: None,
            }),
        },
        Attachments::from([
            (
                AttachmentId(1),
                Attachment::Bytes(serde_json::to_vec(held).unwrap()),
            ),
            (
                AttachmentId(2),
                Attachment::Bytes(serde_json::to_vec(query).unwrap()),
            ),
            (
                AttachmentId(3),
                Attachment::Arrow(ArrowData::new(indices.schema(), vec![indices])),
            ),
        ]),
        Arc::new(Utc),
    )
    .unwrap();
    loop {
        let step = match run.next() {
            Ok(step) => step,
            Err(error) => return Located::Ended(error),
        };
        let Step::Fetch { requests } = step else {
            panic!("materialize ended before it sampled: {step:?}");
        };
        for asked in requests {
            if asked.purpose.is_none() {
                return Located::Past(Box::new(asked.request));
            }
            if let Some(answer) = edits.answer.as_ref().and_then(|f| f(&asked.request)) {
                run.submit_fetch(asked.id, answer).unwrap();
                continue;
            }
            run.submit_fetch(
                asked.id,
                FetchResult::Data(ordinary_rows(
                    &asked.request,
                    edits,
                    case["case"] == "view.metric.wildcards-nested-join",
                )),
            )
            .unwrap();
        }
    }
}

/// The schema `describe_source` gives of the view of `case`.
fn described(document: &Json, id: &str) -> SourceDescription {
    let described_run = describe_with(
        document,
        case(document, id),
        &Edits::default(),
        id != "view.metric.from-source",
    );
    described_run.ended.unwrap()
}

/// Plan B: each `materialize` reads the view again, every probe included,
/// each table read as the relation the view gives it at this run: the
/// source table's rows under the view's filter, with its expressions.
#[test]
fn materialize_reads_each_table_as_the_view_gives_it_now() {
    let document = view_inputs();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    let Located::Past(request) =
        materialize_over(&document, case(&document, id), &schema, &Edits::default())
    else {
        panic!("materialize ended reading the view");
    };
    let orders = request
        .relations()
        .into_iter()
        .map(|(_, r)| r.clone())
        .find(|r| r.source.last().is_some_and(|t| t == "orders"));
    let orders = orders.unwrap_or_else(|| panic!("the first read is not of orders: {request:?}"));
    assert_eq!(orders.filter.as_deref(), Some("source.price > 50"));
    assert!(
        orders
            .columns
            .iter()
            .any(|c| c.name == "PRICE" && c.column.as_deref() == Some("price")),
        "{orders:?}"
    );
}

/// Plan B: a table the schema holds that the view no longer gives is
/// refused by name, with why.
#[test]
fn a_join_the_view_no_longer_gives_is_refused_by_name() {
    let document = view_inputs();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    // The join's key now repeats, so the view drops it at this run.
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request: &DataRequest| match request {
            DataRequest::KeyCounts { keys } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("k", arrow_schema::DataType::Int32, false),
                    arrow_schema::Field::new("n", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("d", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("z", arrow_schema::DataType::Int64, false),
                ]));
                let n = keys.len();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32)),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![0; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                    ],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            _ => None,
        })),
    };
    let Located::Ended(refused) = materialize_over(&document, case(&document, id), &schema, &edits)
    else {
        panic!("a table the view no longer gives was read");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewTableOmitted,
        "{refused:?}"
    );
    assert_eq!(
        refused.fields.get("reason"),
        Some(&sdm_contract::Detail::text(
            "its key 'user_id' holds no value in table 'users'"
        ))
    );
}

/// Plan B: a column the schema holds that the view no longer gives is
/// refused by name; an expression the view changed is read as it now
/// reads.
#[test]
fn a_dimension_removed_is_refused_and_one_changed_read_as_it_now_reads() {
    let document = view_inputs();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    let removed = Edits {
        definition: Some((
            "  - name: price_band\n    expr: CASE WHEN source.price > 100 THEN 'high' ELSE 'low' END\n",
            "",
        )),
        answer: None,
    };
    let Located::Ended(refused) =
        materialize_over(&document, case(&document, id), &schema, &removed)
    else {
        panic!("a column the view no longer gives was read");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewColumnOmitted,
        "{refused:?}"
    );
    assert_eq!(
        refused.fields.get("column"),
        Some(&sdm_contract::Detail::text("price_band"))
    );

    let changed = Edits {
        definition: Some(("source.price > 100 THEN", "source.price > 200 THEN")),
        answer: None,
    };
    let Located::Past(request) =
        materialize_over(&document, case(&document, id), &schema, &changed)
    else {
        panic!("materialize ended reading the view");
    };
    let expressions: Vec<Option<String>> = request
        .relations()
        .iter()
        .flat_map(|(_, relation)| relation.columns.iter())
        .filter(|column| column.name == "price_band")
        .map(|column| column.expression.clone())
        .collect();
    assert_eq!(
        expressions,
        [Some(
            "CASE WHEN price > 200 THEN 'high' ELSE 'low' END".to_owned()
        )],
        "{request:?}"
    );
}

#[test]
fn logical_table_aliases_and_swapped_names_survive_describe_and_materialize() {
    let document = view_inputs();
    let recorded = case(&document, "view.metric.from-source");
    for (buyers, transactions) in [("buyers", "transactions"), ("orders", "users")] {
        let spec = SourceSpec {
            tables: vec![
                NamedTable {
                    name: buyers.into(),
                    source: sdm_contract::data::Relation::named("users"),
                    columns: None,
                },
                NamedTable {
                    name: transactions.into(),
                    source: sdm_contract::data::Relation::named("orders"),
                    columns: None,
                },
            ],
            ..source_spec()
        };
        let described_run = describe_spec(&document, recorded, &Edits::default(), false, spec);
        assert!(
            described_run
                .notes
                .iter()
                .any(|note| note.table.as_deref() == Some(transactions))
        );
        assert!(described_run.notes.iter().all(|note| {
            note.table
                .as_deref()
                .is_none_or(|table| table == buyers || table == transactions)
        }));
        let description = described_run.ended.unwrap();
        assert_eq!(
            description
                .tables
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            [buyers, transactions]
        );
        assert_eq!(
            description.tables[0].source,
            sdm_contract::data::Relation::named("users")
        );
        assert_eq!(description.tables[0].primary_key, ["user_id"]);
        assert_eq!(
            description.tables[1].source,
            sdm_contract::data::Relation::named("orders")
        );
        assert_eq!(description.links[0][0].table, transactions);
        assert_eq!(description.links[0][1].table, buyers);
        let query =
            format!("PREDICT COUNT({transactions}.*, 0, 30, days) > 0 FOR EACH {buyers}.user_id");
        let mut schema = held(&description);
        // Sampler setup counts tables in schema order. Observe each physical
        // source at the first post-refresh read, including after reordering.
        for expected in ["users", "orders"] {
            let Located::Past(request) =
                materialize_query(&document, recorded, &schema, &Edits::default(), &query)
            else {
                panic!("aliased view schema did not reach sampling setup");
            };
            assert!(
                matches!(*request, DataRequest::CountRows { .. }),
                "{request:?}"
            );
            assert!(
                request
                    .relations()
                    .iter()
                    .any(|(_, r)| r.source.last().is_some_and(|s| s == expected)),
                "{request:?}"
            );
            schema.tables.reverse();
        }
    }
}

#[test]
fn a_views_caller_overrides_use_common_key_time_and_relationship_rules() {
    let document = view_inputs();
    let recorded = case(&document, "view.metric.from-source");
    let spec = SourceSpec {
        primary_keys: BTreeMap::from([(
            "orders".into(),
            Given::Value(Some(vec!["order_id".into()])),
        )]),
        time_columns: BTreeMap::from([("orders".into(), Given::Value(None))]),
        relationships: Some(Vec::new()),
        ..source_spec()
    };
    let description = describe_spec(&document, recorded, &Edits::default(), false, spec)
        .ended
        .unwrap();
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    assert_eq!(orders.primary_key, ["order_id"]);
    assert_eq!(orders.time_column, None);
    assert_eq!(orders.duplicate_keys, None);
    assert!(description.links.is_empty());
    let original = described(&document, "view.metric.from-source");
    let source = original
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap()
        .columns
        .iter()
        .find(|c| c.name == "user_id")
        .unwrap();
    assert_eq!(
        orders
            .columns
            .iter()
            .find(|c| c.name == "user_id")
            .unwrap()
            .stype,
        source.stype_before_key.unwrap_or(source.stype)
    );

    for spec in [
        SourceSpec {
            primary_keys: BTreeMap::from([(
                "absent".into(),
                Given::Value(Some(vec!["id".into()])),
            )]),
            ..source_spec()
        },
        SourceSpec {
            primary_keys: BTreeMap::from([(
                "orders".into(),
                Given::Value(Some(vec!["absent".into()])),
            )]),
            ..source_spec()
        },
        SourceSpec {
            primary_keys: BTreeMap::from([("orders".into(), Given::Value(None))]),
            ..source_spec()
        },
        SourceSpec {
            time_columns: BTreeMap::from([("orders".into(), Given::Value(Some("absent".into())))]),
            ..source_spec()
        },
    ] {
        let error = describe_spec(&document, recorded, &Edits::default(), false, spec)
            .ended
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest, "{error:?}");
    }
}

#[test]
fn view_refresh_checks_held_types_and_accepts_compatible_numeric_widths() {
    let document = view_inputs();
    let recorded = case(&document, "view.metric.from-source");
    let original = held(&described(&document, "view.metric.from-source"));
    for dtype in ["string", "int32", "float32"] {
        let mut schema = original.clone();
        schema
            .tables
            .iter_mut()
            .find(|t| t.name == "orders")
            .unwrap()
            .columns
            .iter_mut()
            .find(|c| c.name == "order_id")
            .unwrap()
            .dtype = Some(dtype.into());
        let result = materialize_over(&document, recorded, &schema, &Edits::default());
        if dtype == "string" {
            let Located::Ended(error) = result else {
                panic!("incompatible type sampled");
            };
            assert_eq!(error.sentence, sdm_contract::Sentence::ColumnTypeUnmatched);
            assert_eq!(
                error.fields.get("dtype"),
                Some(&sdm_contract::Detail::text("string"))
            );
        } else {
            assert!(
                matches!(result, Located::Past(_)),
                "compatible numeric type refused"
            );
        }
    }
    // The same logical dimension becomes numeric when the view's expression changes.
    let edits = Edits {
        definition: Some(("THEN 'high' ELSE 'low' END", "THEN 1 ELSE 0 END")),
        answer: Some(Box::new(move |request| match request {
            DataRequest::Preview { columns, .. }
                if columns.len() == 1 && columns[0] == "price_band" =>
            {
                let batch = RecordBatch::try_from_iter([(
                    "price_band",
                    Arc::new(arrow_array::Int64Array::from(vec![1, 0])) as arrow_array::ArrayRef,
                )])
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(
                    batch.schema(),
                    vec![batch],
                )))
            }
            _ => None,
        })),
    };
    let Located::Ended(error) = materialize_over(&document, recorded, &original, &edits) else {
        panic!("changed incompatible dimension reached sampling");
    };
    assert_eq!(error.sentence, sdm_contract::Sentence::ColumnTypeUnmatched);
    assert_eq!(
        error.fields.get("column"),
        Some(&sdm_contract::Detail::text("price_band"))
    );
}

/// A driver may include row values and its full statement in ordinary cast
/// failures. Only safe classification and the caller's logical names may escape.
#[test]
fn refused_expression_diagnostics_keep_context_and_never_echo_driver_text() {
    let document = view_inputs();
    let recorded = case(&document, "view.metric.from-source");
    let schema = held(&described(&document, "view.metric.from-source"));
    const ROW_SECRET: &str = "customer-cell-canary-93841";
    const RAW_SQL: &str = "SELECT raw-driver-sql-canary FROM private_table";
    for (sqlstate, error_class, explanation) in [
        (
            "42000",
            Some("CAST_INVALID_INPUT"),
            "a value could not be converted",
        ),
        ("22018", None, "a value could not be converted"),
        ("22012", Some("DIVIDE_BY_ZERO"), "division by zero"),
        (
            "42000",
            Some("unknown-driver-class-canary"),
            "the warehouse could not evaluate the expression",
        ),
    ] {
        let driver_text = format!("Cannot cast '{ROW_SECRET}'; statement: {RAW_SQL}");
        let edits =
            Edits {
                definition: None,
                answer: Some(Box::new(move |request| {
                    samples(request).then(|| {
                        FetchResult::Error(serde_json::from_value(json!({
                    "kind": "data_source", "retryable": false,
                    "sqlstate": sqlstate, "error_class": error_class,
                    "vendor_code": "vendor-secret-canary", "driver_message": driver_text,
                })).unwrap())
                    })
                })),
            };
        let described_run = describe_with(&document, recorded, &edits, false);
        assert!(described_run.ended.is_ok(), "{:?}", described_run.ended);
        let warning = described_run
            .warnings
            .iter()
            .find(|w| w.sentence == sdm_contract::Sentence::ViewColumnRefused)
            .unwrap();
        let warning_json = serde_json::to_string(warning).unwrap();
        assert!(warning_json.contains(explanation), "{warning_json}");
        assert!(
            warning_json.contains("price_band") && warning_json.contains("orders"),
            "{warning_json}"
        );
        assert!(warning_json.contains(sqlstate), "{warning_json}");
        for forbidden in [
            ROW_SECRET,
            RAW_SQL,
            "Cannot cast",
            "unknown-driver-class-canary",
            "vendor-secret-canary",
        ] {
            assert!(!warning_json.contains(forbidden), "{warning_json}");
        }

        // Re-conversion at materialize omits this held logical column, and its
        // public refusal must keep the same useful safe reason, without data.
        let Located::Ended(error) = materialize_over(&document, recorded, &schema, &edits) else {
            panic!("a refused held expression reached sampling");
        };
        assert_eq!(error.sentence, sdm_contract::Sentence::ViewColumnOmitted);
        let error_json = serde_json::to_string(&error).unwrap();
        assert!(error_json.contains(explanation), "{error_json}");
        assert!(
            error_json.contains("price_band") && error_json.contains("orders"),
            "{error_json}"
        );
        assert!(error_json.contains(sqlstate), "{error_json}");
        for forbidden in [
            ROW_SECRET,
            RAW_SQL,
            "Cannot cast",
            "unknown-driver-class-canary",
            "vendor-secret-canary",
        ] {
            assert!(!error_json.contains(forbidden), "{error_json}");
        }
    }
}

#[test]
fn a_failed_duplicate_example_keeps_joins_unchecked() {
    let document = view_inputs();
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request| match request {
            DataRequest::KeyCounts { keys } => {
                let n = keys.len();
                let batch = RecordBatch::try_from_iter([
                    (
                        "k",
                        Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32))
                            as arrow_array::ArrayRef,
                    ),
                    (
                        "n",
                        Arc::new(arrow_array::Int64Array::from(vec![10; n]))
                            as arrow_array::ArrayRef,
                    ),
                    (
                        "d",
                        Arc::new(arrow_array::Int64Array::from(vec![9; n]))
                            as arrow_array::ArrayRef,
                    ),
                    (
                        "z",
                        Arc::new(arrow_array::Int64Array::from(vec![0; n]))
                            as arrow_array::ArrayRef,
                    ),
                ])
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(
                    batch.schema(),
                    vec![batch],
                )))
            }
            DataRequest::RepeatedKey { .. } => Some(FetchResult::Error(
                serde_json::from_value(json!({"kind": "data_source", "retryable": false,
                    "sqlstate": "42501", "vendor_code": "2003",
                    "driver_message": "key-cell-canary in SELECT key-driver-sql-canary"}))
                .unwrap(),
            )),
            _ => None,
        })),
    };
    let described_run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let description = described_run.ended.unwrap();
    assert_eq!(description.links.len(), 1);
    let warnings = serde_json::to_string(&described_run.warnings).unwrap();
    assert!(
        warnings.contains("counting the join keys")
            && warnings.contains("42501")
            && warnings.contains("2003"),
        "{warnings}"
    );
    assert!(
        !warnings.contains("key-cell-canary") && !warnings.contains("key-driver-sql-canary"),
        "{warnings}"
    );
    assert!(
        described_run
            .warnings
            .iter()
            .any(|w| w.sentence == sdm_contract::Sentence::ViewJoinUnchecked)
    );
}

#[test]
fn metric_view_physical_time_columns_keep_their_declared_window_kind() {
    let document = view_inputs();
    let description = described(&document, "view.metric.from-source");
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    let time = orders.columns.iter().find(|c| c.name == "ts").unwrap();
    // The genuine metric-view catalog declares this column timestamp_ntz.
    // Keep its wall-time kind, rather than treating it as a zoned instant.
    assert_eq!(time.declared_type.as_deref(), Some("timestamp_ntz"));
    let graph = sdm_engine::graph::Graph::new(&description).unwrap();
    let request = sdm_engine::sample::by_time(
        graph.table("orders").unwrap(),
        "user_id",
        vec!["ts".into()],
        sdm_contract::data::KeySet {
            attachment: sdm_contract::data::AttachmentId(9),
            columns: Vec::new(),
        },
        false,
    )
    .unwrap();
    let DataRequest::Neighbors {
        window: Some(window),
        ..
    } = request
    else {
        panic!("no window");
    };
    assert_eq!(window.source_type.as_deref(), Some("timestamp_ntz"));
}

#[test]
fn physical_source_column_shadows_same_named_metric_parameter() {
    let document = view_inputs();
    let edits = Edits {
        definition: Some((
            "filter: source.price > 50",
            "parameters:\n  - name: PRICE\n    data_type: DOUBLE\n    default: 999\nfilter: price > 50",
        )),
        answer: None,
    };
    let run = describe_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    assert!(run.ended.is_ok(), "{:?}", run.ended);
    let filters: Vec<_> = run
        .requests
        .iter()
        .flat_map(|r| r.relations())
        .filter_map(|(_, r)| r.filter.as_deref())
        .collect();
    assert!(!filters.is_empty());
    assert!(filters.iter().all(|filter| *filter == "price > 50"));
}
