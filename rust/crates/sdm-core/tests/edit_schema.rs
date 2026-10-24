// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `edit_schema` through dispatch (T1.24b, decision E67): an edit to a
//! schema the core described is checked and made by `sdm-schema` over the
//! schema the host holds, answered as the schema with the edit made and
//! the column whose meaning it chose, or refused at the start of nothing
//! but the run; the schema is the host's alone, as plain data, each
//! column with its facts (T3.16); and the edit's attachments, which a caller's
//! values fill, are refused by name, never a crash, however deep or wide
//! (E49's probe).

use std::sync::Arc;

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, EditSchema, Given, HeldColumn, HeldSchema,
    HeldTable, Operation, OperationResult, SchemaEdit, SourceDescription, SourceFacts, Stype,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, CoreError, Detail, ErrorCode, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{Attachment, Attachments, CompositionRoot, UnknownZone, Zones};

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

fn start(operation: Operation, attachments: Attachments) -> Result<sdm_core::Run, CoreError> {
    sdm_core::start(
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

fn json(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

fn column(name: &str, dtype: Dtype, stype: Stype) -> DescribedColumn {
    DescribedColumn {
        name: name.into(),
        received: match dtype {
            Dtype::String => ColumnType::Utf8,
            _ => ColumnType::Int64,
        },
        dtype,
        stype,
        stype_before_key: None,
        nullable: true,
        unique: false,
        declared_type: None,
        exact_number: None,
    }
}

fn described() -> SourceDescription {
    SourceDescription {
        tables: vec![DescribedTable {
            name: "users".into(),
            source: "users".into(),
            columns: vec![
                column("user_id", Dtype::Int, Stype::Id),
                column("age", Dtype::Int, Stype::Numerical),
                column("country", Dtype::String, Stype::Categorical),
            ],
            primary_key: vec!["user_id".into()],
            time_column: None,
            time_column_inferred: None,
            duplicate_keys: None,
        }],
        links: vec![],
        namespace: None,
    }
}

fn held(description: &SourceDescription) -> HeldSchema {
    HeldSchema {
        tables: description
            .tables
            .iter()
            .map(|t| HeldTable {
                name: t.name.clone(),
                source: t.source.clone(),
                columns: t
                    .columns
                    .iter()
                    .map(|c| HeldColumn {
                        name: c.name.clone(),
                        dtype: c.dtype.wire_name().into(),
                        stype: c.stype,
                        stype_before_key: c.stype_before_key,
                        source: Some(SourceFacts {
                            relation: t.source.clone(),
                            dtype: c.dtype,
                            received: c.received.clone(),
                            nullable: c.nullable,
                            unique: c.unique,
                            declared_type: c.declared_type.clone(),
                            exact_number: c.exact_number,
                        }),
                    })
                    .collect(),
                primary_key: t.primary_key.clone(),
                time_column: t.time_column.clone(),
                duplicate_keys: t.duplicate_keys.clone(),
            })
            .collect(),
        links: description.links.clone(),
        read_under: Vec::new(),
    }
}

fn edit_op() -> Operation {
    Operation::EditSchema(EditSchema {
        schema: AttachmentId(1),
        edit: AttachmentId(2),
    })
}

fn edit_with(held: Attachment, edit: Attachment) -> Result<Step, CoreError> {
    let attachments = Attachments::from([(AttachmentId(1), held), (AttachmentId(2), edit)]);
    start(edit_op(), attachments)?.next()
}

fn stype(text: &str) -> SchemaEdit {
    SchemaEdit::Stype {
        table: "users".into(),
        column: "age".into(),
        stype: Given::Value(text.into()),
    }
}

#[test]
fn an_edit_is_made_and_answered_as_the_schema_with_it() {
    let mut run = start(
        edit_op(),
        Attachments::from([
            (AttachmentId(1), json(&held(&described()))),
            (AttachmentId(2), json(&stype("categorical"))),
        ]),
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::EditSchema(edited),
    } = run.next().unwrap()
    else {
        panic!("edit_schema did not finish with its result");
    };
    assert_eq!(
        edited.declared,
        Some(ColumnRef {
            table: "users".into(),
            column: "age".into(),
        })
    );
    let Attachment::Bytes(bytes) = run.take_attachment(edited.schema).unwrap() else {
        panic!("the schema edited is not attached");
    };
    let answer: HeldSchema = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(answer.tables[0].columns[1].stype, Stype::Categorical);
    // What the source said of each column comes back as it was handed.
    assert_eq!(answer.tables[0].columns, {
        let mut columns = held(&described()).tables[0].columns.clone();
        columns[1].stype = Stype::Categorical;
        columns
    });
}

#[test]
fn a_refused_edit_ends_the_run_with_the_reference_s_refusal() {
    let error = edit_with(json(&held(&described())), json(&stype("banana"))).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(error.fields.get("field"), Some(&Detail::text("stype")));
}

#[test]
fn a_column_the_host_retyped_is_edited_by_what_its_source_holds() {
    // T3.16: no description to be changed past; a data type written by
    // hand is no edit, as the reference reads a table in place.
    let mut retyped = held(&described());
    retyped.tables[0].columns[1].dtype = "float64".into();
    assert!(matches!(
        edit_with(json(&retyped), json(&stype("categorical"))).unwrap(),
        Step::Done { .. }
    ));
}

fn attachment_named(err: &CoreError) -> u32 {
    match err.fields.get("attachment") {
        Some(Detail::Scalar(Scalar::UInt32(id))) => *id,
        other => panic!("no attachment named: {other:?} in {err:?}"),
    }
}

#[test]
fn an_edit_s_attachments_deep_wide_or_retyped_are_refused_by_name() {
    // Thousands of levels deep, in the edit and in the schema held.
    let deep = |depth: usize| {
        let mut text = String::from("{\"edit\":\"relationships\",\"links\":");
        text.push_str(&"[".repeat(depth));
        text.push_str(&"]".repeat(depth));
        text.push('}');
        Attachment::Bytes(text.into_bytes())
    };
    for depth in [200, 5_000, 100_000] {
        let error = edit_with(json(&held(&described())), deep(depth)).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidMessage, "{depth}");
        assert_eq!(attachment_named(&error), 2);
        let mut text = String::from("{\"tables\":");
        text.push_str(&"[".repeat(depth));
        text.push_str(&"]".repeat(depth));
        text.push_str(",\"links\":[]}");
        let error =
            edit_with(Attachment::Bytes(text.into_bytes()), json(&stype("ID"))).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidMessage, "{depth}");
        assert_eq!(attachment_named(&error), 1);
    }
    // Very wide: a hundred thousand relationships, each naming a column the
    // schema lacks, read and refused by the edit's rule; a hundred thousand
    // keys of an object refused as fields the edit does not have.
    let wide = SchemaEdit::Relationships {
        links: (0..100_000)
            .map(|i| {
                Given::Value(vec![
                    ColumnRef {
                        table: "users".into(),
                        column: format!("c{i}"),
                    },
                    ColumnRef {
                        table: "users".into(),
                        column: "user_id".into(),
                    },
                ])
            })
            .collect(),
    };
    let error = edit_with(json(&held(&described())), json(&wide)).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(
        error.fields.get("field"),
        Some(&Detail::text("relationships"))
    );
    let mut object = serde_json::Map::new();
    object.insert("edit".into(), "stype".into());
    for i in 0..100_000 {
        object.insert(format!("k{i}"), i.into());
    }
    let error = edit_with(
        json(&held(&described())),
        json(&serde_json::Value::Object(object)),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidMessage);
    assert_eq!(attachment_named(&error), 2);
    // Retyped, missing or empty: refused naming the attachment, no text.
    for edit in [
        Attachment::Bytes(
            br#"{"edit":"stype","table":7,"column":"age","stype":{"value":"ID"}}"#.to_vec(),
        ),
        Attachment::Bytes(br#"{"edit":"stype","table":"users","column":"age"}"#.to_vec()),
        Attachment::Bytes(b"".to_vec()),
        Attachment::Bytes(br#"{"edit":"rename","table":"users"}"#.to_vec()),
    ] {
        let error = edit_with(json(&held(&described())), edit).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidMessage);
        assert_eq!(attachment_named(&error), 2);
    }
}

/// Where the schema read each relation (`HeldSchema::read_under`) comes back
/// from an edit as it was handed, a relation the schema never read left
/// out, so a host holding the schema the edit answers reads the same tables
/// (round 5's Codex P1).
#[test]
fn an_edit_answers_where_the_schema_read_each_relation_as_it_was() {
    let mut given = held(&described());
    given.read_under = vec![sdm_contract::ops::ReadUnder {
        relation: given.tables[0].source.clone(),
        namespace: vec!["main".into(), "shop".into()],
        columns: Vec::new(),
    }];
    let mut run = start(
        edit_op(),
        Attachments::from([
            (AttachmentId(1), json(&given)),
            (AttachmentId(2), json(&stype("categorical"))),
        ]),
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::EditSchema(edited),
    } = run.next().unwrap()
    else {
        panic!("edit_schema did not finish with its result");
    };
    let Attachment::Bytes(bytes) = run.take_attachment(edited.schema).unwrap() else {
        panic!("the schema edited is not attached");
    };
    let answer: HeldSchema = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(answer.read_under, given.read_under);
}
