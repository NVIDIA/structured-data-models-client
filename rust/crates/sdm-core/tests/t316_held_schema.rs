// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema as plain data (T3.16, the structure design's fix A), through
//! `sdm_core::start`, over a host provider whose answers the tests give:
//! `materialize` takes the schema as the host holds it, each column with
//! what the source said of it; a schema whose columns all carry their facts
//! asks the source no catalog question and no preview, and a column with
//! none is read as `describe_source` reads it, only that column previewed;
//! a column the source lacks is left unread, as the reference's engine
//! leaves it. `edit_schema` takes the schema alone and answers it, each
//! column's facts kept, a column with none edited by its data type, and an
//! exact number's facts retyped by its role (T2.21).

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{
    AttachmentId, DataRequest, DescribePart, Determinism, IndexAdvice, NullOrder, OptionalRequest,
    ProviderCapabilities, ProviderChoice, catalog_columns as c,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ColumnRef, Dtype, EditSchema, ExactNumber, Given, HeldColumn, HeldSchema, HeldTable,
    Materialize, Operation, OperationResult, SchemaEdit, SourceFacts, Stype,
};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ColumnType, CoreError, HostFacts, Scalar, Step, Unit};
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

fn source(relation: &str, dtype: Dtype, received: ColumnType, unique: bool) -> Option<SourceFacts> {
    Some(SourceFacts {
        relation: relation.into(),
        dtype,
        received,
        nullable: !unique,
        unique,
        declared_type: None,
        exact_number: None,
    })
}

fn held_column(name: &str, dtype: Dtype, stype: Stype, facts: Option<SourceFacts>) -> HeldColumn {
    HeldColumn {
        name: name.into(),
        dtype: dtype.wire_name().into(),
        stype,
        stype_before_key: None,
        source: facts,
    }
}

fn link(from: (&str, &str), to: (&str, &str)) -> Vec<ColumnRef> {
    vec![
        ColumnRef {
            table: from.0.into(),
            column: from.1.into(),
        },
        ColumnRef {
            table: to.0.into(),
            column: to.1.into(),
        },
    ]
}

/// `base.regression`'s shape as the host holds it, each column with its
/// facts but those of `users.country` when `country_facts` is false, and
/// `users.ghost`, which the source lacks, when `ghost` is true.
fn shop(country_facts: bool, ghost: bool) -> HeldSchema {
    let ts = ColumnType::Timestamp {
        unit: Unit::Us,
        zone: None,
    };
    let mut users = vec![
        held_column(
            "user_id",
            Dtype::Int,
            Stype::Id,
            source("users", Dtype::Int, ColumnType::Int64, true),
        ),
        held_column(
            "country",
            Dtype::String,
            Stype::Categorical,
            country_facts
                .then(|| source("users", Dtype::String, ColumnType::Utf8, false))
                .flatten(),
        ),
    ];
    if ghost {
        users.push(held_column("ghost", Dtype::Float, Stype::Numerical, None));
    }
    HeldSchema {
        tables: vec![
            HeldTable {
                name: "orders".into(),
                source: "orders".into(),
                columns: vec![
                    held_column(
                        "order_id",
                        Dtype::Int,
                        Stype::Id,
                        source("orders", Dtype::Int, ColumnType::Int64, true),
                    ),
                    held_column(
                        "user_id",
                        Dtype::Int,
                        Stype::Id,
                        source("orders", Dtype::Int, ColumnType::Int64, false),
                    ),
                    held_column(
                        "price",
                        Dtype::Float,
                        Stype::Numerical,
                        source("orders", Dtype::Float, ColumnType::Float64, false),
                    ),
                    held_column(
                        "ts",
                        Dtype::Date,
                        Stype::Timestamp,
                        source("orders", Dtype::Date, ts, false),
                    ),
                ],
                primary_key: vec!["order_id".into()],
                time_column: Some("ts".into()),
                duplicate_keys: None,
            },
            HeldTable {
                name: "users".into(),
                source: "users".into(),
                columns: users,
                primary_key: vec!["user_id".into()],
                time_column: None,
                duplicate_keys: None,
            },
        ],
        links: vec![link(("orders", "user_id"), ("users", "user_id"))],
    }
}

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
                OptionalRequest::Preview,
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

fn run_of(operation: Operation, attachments: Attachments) -> Result<Run, CoreError> {
    start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
        Arc::new(NoZones),
    )
}

fn materialize(held: &HeldSchema) -> Run {
    let operation = Operation::Materialize(Materialize {
        provider: host(),
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
    run_of(
        operation,
        Attachments::from([
            (
                AttachmentId(1),
                Attachment::Bytes(serde_json::to_vec(held).unwrap()),
            ),
            (
                AttachmentId(2),
                Attachment::Bytes(serde_json::to_vec(QUERY).unwrap()),
            ),
            (
                AttachmentId(3),
                Attachment::Arrow(ArrowData::new(indices.schema(), vec![indices])),
            ),
        ]),
    )
    .unwrap()
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

fn names(lists: &[&[&str]]) -> ArrayRef {
    let mut builder =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new())
            .with_field(Arc::new(Field::new(c::LIST_ITEM, DataType::Utf8, true)));
    for list in lists {
        for name in *list {
            builder.values().append_value(name);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// The source's answers, in the contract's catalog shape: `users` holds
/// `user_id` (its primary key) and `country`, no index, and the orders'
/// times; `None` for a request the sample asks.
fn answer(request: &DataRequest) -> Option<FetchResult> {
    let utf8 = |values: Vec<&str>| -> ArrayRef { Arc::new(StringArray::from(values)) };
    let flags = |values: Vec<bool>| -> ArrayRef { Arc::new(BooleanArray::from(values)) };
    match request {
        DataRequest::Describe { part, .. } => Some(match part {
            DescribePart::Indexes => data(vec![
                (c::TABLE, utf8(vec![])),
                (
                    c::COLUMNS,
                    Arc::new(ListArray::new_null(
                        Arc::new(Field::new(c::LIST_ITEM, DataType::Utf8, true)),
                        0,
                    )),
                ),
            ]),
            DescribePart::Columns => data(vec![
                (c::NAME, utf8(vec!["user_id", "country"])),
                (c::TYPE, utf8(vec!["BIGINT", "VARCHAR"])),
                (c::NOT_NULL, flags(vec![true, false])),
                (c::PRIMARY_KEY, flags(vec![true, false])),
            ]),
            DescribePart::Keys => data(vec![
                (c::KIND, utf8(vec![c::KIND_PRIMARY])),
                (c::COLUMNS, names(&[&["user_id"]])),
            ]),
            _ => return None,
        }),
        DataRequest::Preview { columns, .. } => Some(data(
            columns
                .iter()
                .map(|name| -> (&str, ArrayRef) {
                    match name.as_str() {
                        "country" => ("country", utf8(vec!["US", "DE"])),
                        _ => ("user_id", Arc::new(Int64Array::from(vec![0, 1]))),
                    }
                })
                .collect(),
        )),
        DataRequest::TimeBounds { .. } => Some(data(vec![
            (
                "table_index",
                Arc::new(arrow_array::Int32Array::from(vec![0])) as ArrayRef,
            ),
            (
                "min_date",
                Arc::new(TimestampMicrosecondArray::from(vec![Some(
                    1_672_531_200_000_000,
                )])),
            ),
            (
                "max_date",
                Arc::new(TimestampMicrosecondArray::from(vec![Some(
                    1_720_310_400_000_000,
                )])),
            ),
        ])),
        _ => None,
    }
}

/// Every request `run` asks, step by step, answered by [`answer`], to the
/// first the sample asks, which is the last.
fn to_the_sample(run: &mut Run) -> Vec<Vec<DataRequest>> {
    let mut steps = Vec::new();
    loop {
        let step = run.next().unwrap();
        let Step::Fetch { requests } = step else {
            panic!("not a fetch: {step:?}");
        };
        steps.push(requests.iter().map(|r| r.request.clone()).collect());
        for asked in requests {
            match answer(&asked.request) {
                Some(answered) => run.submit_fetch(asked.id, answered).unwrap(),
                None => return steps,
            }
        }
    }
}

fn reads_the_catalog(request: &DataRequest) -> bool {
    matches!(
        request,
        DataRequest::Describe {
            part: DescribePart::Columns | DescribePart::Keys | DescribePart::IndexColumns,
            ..
        } | DataRequest::Preview { .. }
            | DataRequest::ColumnRanges { .. }
    )
}

#[test]
fn a_schema_whose_columns_carry_their_facts_asks_the_source_nothing_more() {
    let steps = to_the_sample(&mut materialize(&shop(true, false)));
    assert!(!steps.iter().flatten().any(reads_the_catalog), "{steps:#?}");
}

#[test]
fn a_column_with_no_facts_is_read_as_describe_reads_it_and_only_it_previewed() {
    let steps = to_the_sample(&mut materialize(&shop(false, false)));
    // The catalog's columns and keys of its table, asked together in one
    // step, before anything is sampled.
    let first = &steps[0];
    assert!(
        first.iter().any(|r| matches!(
            r,
            DataRequest::Describe {
                part: DescribePart::Columns,
                tables
            } if tables[0].single() == Some("users")
        )),
        "{steps:#?}"
    );
    assert!(first.iter().any(|r| matches!(
        r,
        DataRequest::Describe {
            part: DescribePart::Keys,
            ..
        }
    )));
    let previews: Vec<&Vec<String>> = steps
        .iter()
        .flatten()
        .filter_map(|r| match r {
            DataRequest::Preview { columns, .. } => Some(columns),
            _ => None,
        })
        .collect();
    assert_eq!(previews, [&vec!["country".to_owned()]]);
    // Only the table holding such a column is read.
    assert!(!steps.iter().flatten().any(|r| matches!(
        r,
        DataRequest::Describe { tables, part: DescribePart::Columns }
            if tables[0].single() == Some("orders")
    )));
}

#[test]
fn a_column_the_source_lacks_is_left_unread_as_the_reference_leaves_it() {
    let steps = to_the_sample(&mut materialize(&shop(true, true)));
    // `ghost` was asked of the source, which does not hold it, and the run
    // went on to its sample, refusing nothing.
    assert!(steps.iter().flatten().any(|r| matches!(
        r,
        DataRequest::Describe {
            part: DescribePart::Columns,
            ..
        }
    )));
    assert!(!steps.iter().flatten().any(|r| matches!(
        r,
        DataRequest::Preview { columns, .. } if columns.iter().any(|c| c == "ghost")
    )));
}

fn edit(
    held: &HeldSchema,
    edit: &SchemaEdit,
) -> Result<(HeldSchema, Option<ColumnRef>), CoreError> {
    let mut run = run_of(
        Operation::EditSchema(EditSchema {
            schema: AttachmentId(1),
            edit: AttachmentId(2),
        }),
        Attachments::from([
            (
                AttachmentId(1),
                Attachment::Bytes(serde_json::to_vec(held).unwrap()),
            ),
            (
                AttachmentId(2),
                Attachment::Bytes(serde_json::to_vec(edit).unwrap()),
            ),
        ]),
    )?;
    let Step::Done { result } = run.next()? else {
        panic!("an edit asks for data");
    };
    let OperationResult::EditSchema(edited) = result else {
        panic!("another result");
    };
    let Some(Attachment::Bytes(bytes)) = run.take_attachment(edited.schema) else {
        panic!("the schema is not attached");
    };
    Ok((serde_json::from_slice(&bytes).unwrap(), edited.declared))
}

fn column<'a>(held: &'a HeldSchema, table: &str, name: &str) -> &'a HeldColumn {
    held.tables
        .iter()
        .find(|t| t.name == table)
        .and_then(|t| t.columns.iter().find(|c| c.name == name))
        .unwrap()
}

#[test]
fn a_column_with_no_facts_is_edited_by_its_data_type_and_keeps_none() {
    let held = shop(false, false);
    let (answered, declared) = edit(
        &held,
        &SchemaEdit::Stype {
            table: "users".into(),
            column: "country".into(),
            stype: Given::Value("text".into()),
        },
    )
    .unwrap();
    let country = column(&answered, "users", "country");
    assert_eq!(country.stype, Stype::Text);
    assert_eq!(country.dtype, "string");
    assert_eq!(country.source, None);
    assert_eq!(declared.unwrap().column, "country");
    // Every other column keeps the facts it was handed, as handed.
    assert_eq!(
        column(&answered, "orders", "ts").source,
        column(&held, "orders", "ts").source
    );
    // A meaning its data type cannot carry is refused, as the reference's
    // setter refuses it.
    assert!(
        edit(
            &held,
            &SchemaEdit::Stype {
                table: "users".into(),
                column: "country".into(),
                stype: Given::Value("numerical".into()),
            },
        )
        .is_err()
    );
}

/// A NUMERIC key and the column referring to it, each handed over as exact
/// text, held with their facts: the key's catalog declares it, the other
/// is read by its role.
fn numeric() -> HeldSchema {
    let mut held = shop(true, false);
    let exact = |relation: &str, exact: ExactNumber, text: bool| SourceFacts {
        relation: relation.into(),
        dtype: if text { Dtype::String } else { Dtype::Float64 },
        received: if text {
            ColumnType::Utf8
        } else {
            ColumnType::Float64
        },
        nullable: exact == ExactNumber::Other,
        unique: exact == ExactNumber::DeclaredKey,
        declared_type: None,
        exact_number: Some(exact),
    };
    for table in &mut held.tables {
        for column in &mut table.columns {
            if column.name == "user_id" {
                let (kind, text) = if table.name == "users" {
                    (ExactNumber::DeclaredKey, true)
                } else {
                    (ExactNumber::Other, false)
                };
                column.source = Some(exact(&table.name, kind, text));
                column.dtype = if text { "string" } else { "float64" }.into();
            }
        }
    }
    held.links = vec![];
    held
}

#[test]
fn an_exact_number_s_facts_follow_its_role_through_edits() {
    let held = numeric();
    let linked = SchemaEdit::Relationships {
        links: vec![Given::Value(link(
            ("orders", "user_id"),
            ("users", "user_id"),
        ))],
    };
    let (answered, _) = edit(&held, &linked).unwrap();
    // Joined to the key's text, the column is read as text too (T2.21);
    // the key keeps its digits.
    let foreign = column(&answered, "orders", "user_id");
    assert_eq!(foreign.dtype, "string");
    assert_eq!(
        foreign.source.as_ref().map(|f| (&f.received, f.dtype)),
        Some((&ColumnType::Utf8, Dtype::String))
    );
    assert_eq!(column(&answered, "users", "user_id").dtype, "string");
    // Cleared, it is a double again, the key still text.
    let (cleared, _) = edit(&answered, &SchemaEdit::Relationships { links: vec![] }).unwrap();
    let foreign = column(&cleared, "orders", "user_id");
    assert_eq!(foreign.dtype, "float64");
    assert_eq!(
        foreign.source.as_ref().map(|f| &f.received),
        Some(&ColumnType::Float64)
    );
    assert_eq!(column(&cleared, "users", "user_id").dtype, "string");
    // A key edit keeps the key's digits, and sets its repeats as the
    // reference's key setter does.
    let (unkeyed, _) = edit(
        &cleared,
        &SchemaEdit::PrimaryKey {
            table: "users".into(),
            key: Given::Value(None),
        },
    )
    .unwrap();
    assert_eq!(column(&unkeyed, "users", "user_id").dtype, "string");
    let users = unkeyed.tables.iter().find(|t| t.name == "users").unwrap();
    assert_eq!(users.duplicate_keys, Some(Scalar::UInt64(0)));
}

#[test]
fn a_table_pointed_at_another_relation_has_its_columns_read_afresh() {
    // Facts are of the relation they were read from: the table the host
    // points at another holds none of it, and every column it keeps is read
    // from the relation it names now, as the reference reads a table by its
    // source name.
    let mut held = shop(true, false);
    let users = held.tables.iter_mut().find(|t| t.name == "users").unwrap();
    users.source = "other_users".into();
    let steps = to_the_sample(&mut materialize(&held));
    assert!(steps.iter().flatten().any(|r| matches!(
        r,
        DataRequest::Describe { tables, part: DescribePart::Columns }
            if tables[0].single() == Some("other_users")
    )));
    let previews: Vec<&Vec<String>> = steps
        .iter()
        .flatten()
        .filter_map(|r| match r {
            DataRequest::Preview { columns, .. } => Some(columns),
            _ => None,
        })
        .collect();
    assert_eq!(
        previews,
        [&vec!["user_id".to_owned(), "country".to_owned()]]
    );
}
