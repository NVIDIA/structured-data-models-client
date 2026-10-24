// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Snowflake semantic view's conversion (T3.5, V1 to V7) over a fake data
//! layer, the Snowflake dialect being T3.6's: the layer answers the view's
//! text in Snowflake's rules (`sdm-data`'s), every read as a host provider
//! would, and the test answers each read from tables it holds. S5's third
//! sequence: each table is finished, its columns read, its expressions
//! sampled and its rows read, before the next table's first read.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn, ViewDefinition};
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, DescribePart, ExpectedSchema, FetchPlan,
    ProviderCapabilities, ProviderChoice, Relation,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    DescribeSource, NamedTable, Operation, OperationResult, SourceDescription, SourceSpec,
};
use sdm_contract::run::StartRun;
use sdm_contract::view::{DeclaredType, ViewKind, ViewTextAnswer, ViewTextAsk};
use sdm_contract::{CONTRACT_VERSION, CoreError, HostFacts, Scalar, Step, Unit};
use sdm_core::{
    ArrowData, Attachment, Attachments, CompositionRoot, FetchResult, UnknownZone, Zones, start,
};
use sdm_data::view_expr::ExprDialect;
use serde_json::json;

struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}

/// A semantic view's two tables, `USERS` before `ORDERS`, as Snowflake
/// writes the view out: upper-case names, each column an expression.
const VIEW: &str = r#"
name: shop
tables:
  - name: USERS
    base_table: {database: DB, schema: SHOP, table: USERS}
    primary_key: {columns: [USER_ID]}
    dimensions:
      - {name: USER_ID, expr: user_id, data_type: NUMBER(38,0)}
      - {name: COUNTRY, expr: country, data_type: TEXT}
  - name: ORDERS
    base_table: {database: DB, schema: SHOP, table: ORDERS}
    primary_key: {columns: [ORDER_ID]}
    dimensions:
      - {name: ORDER_ID, expr: order_id, data_type: NUMBER(38,0)}
      - {name: USER_ID, expr: user_id, data_type: NUMBER(38,0)}
    facts:
      - {name: PRICE_WITH_TAX, expr: price * 1.2, data_type: FLOAT}
relationships:
  - name: ORDERS_TO_USERS
    left_table: ORDERS
    right_table: USERS
    relationship_columns: [{left_column: USER_ID, right_column: USER_ID}]
"#;

/// The fake layer: Snowflake's reading of the view's text, every read
/// passed to the test as a host provider's request, each catalog answer
/// the test's tables'.
struct Fake {
    definition: String,
    catalogs: BTreeMap<String, Vec<&'static str>>,
}

impl DataLayer for Fake {
    fn capabilities(&self, _: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        // A warehouse's: it reads a derived relation.
        let mut capabilities =
            sdm_data::Data::new().capabilities(&ProviderChoice::sql("duckdb"))?;
        capabilities.derived_relations = true;
        Ok(capabilities)
    }

    fn plan(&self, _: &DataRequest, _: &ProviderChoice) -> Result<FetchPlan, CoreError> {
        Ok(FetchPlan::Host {
            expect: ExpectedSchema::Discover {},
        })
    }

    fn read(
        &self,
        request: &DataRequest,
        _: &ProviderChoice,
        _: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { tables, part } = request else {
            panic!("a read of no catalog");
        };
        Ok(match part {
            DescribePart::ViewDefinition => CatalogAnswer::ViewDefinition(ViewDefinition {
                kind: ViewKind::Semantic,
                text: self.definition.clone(),
                parameters: vec![],
                located: vec![],
                columns: vec![],
            }),
            DescribePart::Columns => {
                let table = tables[0].source.last().unwrap();
                CatalogAnswer::Columns(
                    self.catalogs[table]
                        .iter()
                        .map(|name| CatalogColumn {
                            name: (*name).to_owned(),
                            declared_type: if *name == "TS" {
                                "TIMESTAMP_LTZ(6)".into()
                            } else {
                                String::new()
                            },
                            not_null: false,
                            primary_key: false,
                            key_nullable: false,
                            unique: false,
                            dtype: None,
                        })
                        .collect(),
                )
            }
            DescribePart::Keys => CatalogAnswer::Keys(vec![]),
            other => panic!("a read of {other:?}"),
        })
    }

    fn view_kind(&self, _: &ProviderChoice) -> Option<ViewKind> {
        Some(ViewKind::Semantic)
    }

    fn view_text(&self, _: &ProviderChoice, ask: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        Ok(sdm_data::view_expr::answer(
            ask,
            ExprDialect::Snowflake,
            sdm_core::pql::spaces(),
            &|_| DeclaredType::Unknown,
        ))
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

/// The rows of `columns` of a preview: each its name's values in the test's
/// tables, an expression's as a float.
fn rows(relation: &Relation, columns: &[String]) -> ArrowData {
    let fields: Vec<Field> = columns
        .iter()
        .map(|name| {
            let expression = relation
                .columns
                .iter()
                .any(|c| c.name == *name && c.expression.is_some());
            let data_type = match name.to_lowercase().as_str() {
                "country" => DataType::Utf8,
                "ts" | "day_ts" => DataType::Timestamp(TimeUnit::Microsecond, None),
                "user_id" | "order_id" | "age" => DataType::Int64,
                _ if expression => DataType::Float64,
                _ => DataType::Int64,
            };
            Field::new(name.clone(), data_type, true)
        })
        .collect();
    let arrays: Vec<ArrayRef> = fields
        .iter()
        .map(|field| -> ArrayRef {
            match field.data_type() {
                DataType::Utf8 => Arc::new(StringArray::from(vec!["US", "DE", "US"])),
                DataType::Float64 => Arc::new(Float64Array::from(vec![1.2, 2.4, 3.6])),
                DataType::Timestamp(..) => Arc::new(TimestampMicrosecondArray::from(vec![
                    1_704_067_200_000_000,
                    1_704_153_600_000_000,
                    1_704_240_000_000_000,
                ])),
                _ => Arc::new(Int64Array::from(vec![1, 2, 3])),
            }
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
    ArrowData::new(schema, vec![batch])
}

/// The base table a request reads, by its last name part.
fn table_of(request: &DataRequest) -> Option<String> {
    request
        .relations()
        .first()
        .and_then(|(_, relation)| relation.source.last().cloned())
}

fn source_spec() -> SourceSpec {
    SourceSpec {
        tables: Vec::new(),
        primary_keys: BTreeMap::new(),
        time_columns: BTreeMap::new(),
        relationships: None,
    }
}

fn semantic_run(definition: &str, spec: SourceSpec) -> sdm_core::Run {
    let fake = Fake {
        definition: definition.to_owned(),
        catalogs: BTreeMap::from([
            ("USERS".to_owned(), vec!["USER_ID", "COUNTRY"]),
            (
                "ORDERS".to_owned(),
                vec!["ORDER_ID", "USER_ID", "PRICE", "TS"],
            ),
        ]),
    };
    let provider: ProviderChoice = serde_json::from_value(json!({
        "provider": "sql", "dialect": "snowflake",
        "view": {"name": "DB.SHOP.SHOP_VIEW", "trust_expressions": false},
    }))
    .unwrap();
    start(
        &CompositionRoot::new(Arc::new(fake)),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::DescribeSource(DescribeSource {
                provider,
                source: AttachmentId(1),
                check_keys: false,
            }),
        },
        Attachments::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec).unwrap()),
        )]),
        Arc::new(NoZones),
    )
    .unwrap()
}

fn describe_view(definition: &str, spec: SourceSpec) -> (SourceDescription, Vec<(String, String)>) {
    let mut run = semantic_run(definition, spec);
    let mut read: Vec<(String, String)> = Vec::new();
    let description: SourceDescription = loop {
        match run.next().unwrap() {
            Step::Done { result } => {
                let OperationResult::DescribeSource(described) = result else {
                    panic!("not a description");
                };
                let Some(Attachment::Bytes(bytes)) = run.take_attachment(described.description)
                else {
                    panic!("no description");
                };
                break serde_json::from_slice(&bytes).unwrap();
            }
            Step::Fetch { requests } => {
                for asked in requests {
                    let answer = match &asked.request {
                        DataRequest::Preview { table, columns, .. } => rows(table, columns),
                        _ => ArrowData::new(Arc::new(Schema::empty()), vec![]),
                    };
                    if let Some(table) = table_of(&asked.request) {
                        read.push((table, asked.request.name().to_owned()));
                    }
                    run.submit_fetch(asked.id, FetchResult::Data(answer))
                        .unwrap();
                }
            }
            Step::CallModel(_) => panic!("a description called a model"),
        }
    };
    (description, read)
}

#[test]
fn each_table_is_finished_before_the_next_is_read() {
    let (description, read) = describe_view(VIEW, source_spec());
    // Every read of USERS before the first of ORDERS.
    let first_orders = read.iter().position(|(t, _)| t == "ORDERS").unwrap();
    assert!(
        read[first_orders..].iter().all(|(t, _)| t != "USERS"),
        "{read:?}"
    );
    // Each table: its columns, then its expressions' sample, then its rows.
    let users: Vec<&str> = read
        .iter()
        .filter(|(t, _)| t == "USERS")
        .map(|(_, r)| r.as_str())
        .collect();
    assert_eq!(users, ["describe", "describe", "preview"], "{read:?}");
    let tables: Vec<&str> = description.tables.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tables, ["USERS", "ORDERS"]);
    assert_eq!(description.links.len(), 1, "{:?}", description.links);
    let orders = &description.tables[1];
    assert_eq!(orders.primary_key, ["ORDER_ID"]);
    assert!(orders.columns.iter().any(|c| c.name == "PRICE_WITH_TAX"));
}

#[test]
fn a_semantic_table_with_no_columns_or_key_does_not_read_or_break_other_tables() {
    let definition = VIEW.replace("tables:\n", "tables:\n  - name: UNUSED\n    base_table: {database: DB, schema: SHOP, table: NEVER_READ}\n");
    let (description, read) = describe_view(
        &definition,
        SourceSpec {
            tables: vec![NamedTable {
                name: "people".into(),
                source: Relation::named("USERS"),
                columns: None,
            }],
            ..source_spec()
        },
    );
    assert_eq!(description.tables.len(), 1);
    assert_eq!(description.tables[0].name, "people");
    assert_eq!(description.tables[0].source, Relation::named("USERS"));
    assert!(!read.iter().any(|(name, _)| name == "NEVER_READ"));
}

#[test]
fn a_failed_semantic_definition_fetch_uses_its_prefetch_kind() {
    let mut run = semantic_run(VIEW, source_spec());
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!("definition was not fetched");
    };
    assert_eq!(requests.len(), 1);
    assert!(matches!(
        requests[0].request,
        DataRequest::Describe {
            part: DescribePart::ViewDefinition,
            ..
        }
    ));
    run.submit_fetch(
        requests[0].id,
        FetchResult::Error(
            serde_json::from_value(json!({
                "kind": "data_source", "retryable": false, "sqlstate": "42501",
                "driver_message": "No permission to read this semantic view"
            }))
            .unwrap(),
        ),
    )
    .unwrap();
    let error = run.next().unwrap_err();
    assert_eq!(error.sentence, sdm_contract::Sentence::ViewSemanticNotFound);
    assert_eq!(
        error.fields.get("view"),
        Some(&sdm_contract::Detail::text("DB.SHOP.SHOP_VIEW"))
    );
}

#[test]
fn semantic_models_never_infer_additional_links_from_column_names() {
    let definition = VIEW.split("relationships:").next().unwrap();
    let (description, _) = describe_view(definition, source_spec());
    assert_eq!(description.tables.len(), 2);
    assert!(description.links.is_empty(), "{:?}", description.links);
}

#[test]
fn converted_temporal_facts_reach_the_existing_zoned_window_planner() {
    let definition = VIEW.replace("relationships:\n", "    time_dimensions:\n      - {name: DAY_TS, expr: DATE_TRUNC('DAY', TS), data_type: TIMESTAMP_LTZ(6)}\nrelationships:\n");
    let (description, _) = describe_view(&definition, source_spec());
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "ORDERS")
        .unwrap();
    let time = orders.columns.iter().find(|c| c.name == "DAY_TS").unwrap();
    assert_eq!(time.declared_type.as_deref(), Some("TIMESTAMP_LTZ(6)"));
    let graph = sdm_engine::graph::Graph::new(&description).unwrap();
    let mut table = graph.table("ORDERS").unwrap().clone();
    table.source = Relation::of(vec!["DB".into(), "SHOP".into(), "ORDERS".into()]);
    let seeds = sdm_contract::data::KeySet {
        attachment: AttachmentId(9),
        columns: vec![
            sdm_contract::data::field("key", sdm_contract::ColumnType::Int64, false),
            sdm_contract::data::field("batch", sdm_contract::ColumnType::Int64, false),
            sdm_contract::data::field(
                "end",
                sdm_contract::ColumnType::Timestamp {
                    unit: Unit::Ns,
                    zone: None,
                },
                false,
            ),
        ],
    };
    let request =
        sdm_engine::sample::by_time(&table, "USER_ID", vec!["DAY_TS".into()], seeds, false)
            .unwrap();
    let DataRequest::Neighbors {
        window: Some(window),
        ..
    } = &request
    else {
        panic!("no temporal window");
    };
    assert_eq!(window.source_type.as_deref(), Some("TIMESTAMP_LTZ(6)"));
    let FetchPlan::Sql { statements, .. } = sdm_data::Data::new()
        .plan(&request, &ProviderChoice::sql("snowflake"))
        .unwrap()
    else {
        panic!("no Snowflake SQL");
    };
    assert!(
        statements
            .iter()
            .any(|s| s.text.contains("CONVERT_TIMEZONE('UTC'")),
        "{statements:?}"
    );
}
