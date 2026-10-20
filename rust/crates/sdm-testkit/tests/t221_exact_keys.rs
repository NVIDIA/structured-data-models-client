// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.21's rule on `rust-host` (the package's test author's): a key's
//! representation is decided by the effective keys and relationships
//! (catalog, caller-supplied, edited), not by the catalog alone (learnings
//! lesson 19, design choice 5, KO2-35). `contract/tests/test_t221_exact_keys.py`
//! holds `python-core` to the same on a real Postgres server.
//!
//! A column of numbers a host provider hands over as their exact text,
//! its field marked `sdm:exact_number` (`sdm_contract::data::
//! EXACT_NUMBER_MARKER`), is how Postgres's unconstrained `numeric`
//! reaches the core; the contract loader makes no such column, so these
//! tests make it: each case's tables, built on DuckDB from its base and
//! setup and copied into the non-SQL provider (`sdm_testkit::memory`,
//! T2.7), have `users.user_id` and `orders.user_id` (a `DECIMAL(38, 0)`
//! past 2**53) handed over as that marked text, declared `numeric`. Then:
//!
//! - `the_catalog_key_control_keeps_the_key_exact`: with the key and the
//!   link declared by the catalog (`key.past-2-53.catalog`), the run
//!   answers through predict, each `user_id` described as its text: the
//!   control, which holds on the base;
//! - `a_given_or_edited_key_is_read_as_a_catalog_key_is`: with the key
//!   given (`key.past-2-53.given-key`), the key and the link given
//!   (`.given-link`) or the link set by an edit (`.edited-link`), and
//!   nothing declared, the run answers as the control does: each
//!   `user_id` in the schema `materialize` was given (the description, the
//!   edit made) held as the control holds it (what it was received as, its
//!   dtype and meaning; whether it is nullable or unique is the
//!   database's fact, which a SQL key declares and a given key does not),
//!   and the call the
//!   core wrote the control's (the `payload` rule: but its request id).
//!   On the base the key is read as doubles, and 2**53 + 1 is refused
//!   as inexact (the mid-phase design check's reproduction).
//!
//! Gating: `contract/tests/t221-pending.json`'s `pending` names, as
//! `rust:<case id>@memory-marked`, the keys that failed when written. Each
//! must still fail (strict); once one passes it fails, naming itself,
//! until it is taken out.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Decimal128Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::catalog_columns::KIND_FOREIGN;
use sdm_contract::data::{EXACT_NUMBER_MARKED, EXACT_NUMBER_MARKER};
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use sdm_testkit::memory::Memory;
use sdm_testkit::{bases, duck, layer};
use serde_json::Value;

const PENDING: &str = "contract/tests/t221-pending.json";
const CONTROL: &str = "key.past-2-53.catalog";
const GIVEN: [&str; 3] = [
    "key.past-2-53.given-key",
    "key.past-2-53.given-link",
    "key.past-2-53.edited-link",
];
/// The column made exact text in each table.
const KEY: &str = "user_id";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The pending file's `list` names of this test, as `rust:` keys.
fn named(list: &str) -> BTreeSet<String> {
    read_json(&root().join(PENDING))[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .filter(|n| n.ends_with("@memory-marked"))
        .map(str::to_owned)
        .collect()
}

fn gating_key(id: &str) -> String {
    format!("{id}@memory-marked")
}

/// Runs `check`; a pending key must fail, any other must pass.
fn gated(id: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let key = gating_key(id);
    let outcome = std::panic::catch_unwind(check);
    match (outcome, named("pending").contains(&key)) {
        (Ok(()), true) => panic!("{key} passes now: take rust:{key} out of {PENDING}'s pending"),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let authored: BTreeSet<String> = GIVEN
        .iter()
        .chain([&CONTROL])
        .map(|id| gating_key(id))
        .collect();
    assert!(named("pending").is_subset(&named("authored")));
    assert_eq!(named("authored"), authored);
}

fn case(id: &str) -> Case {
    let path = root().join(format!("contract/cases/s1/{id}.json"));
    case::load(&[path]).unwrap().remove(0).case
}

/// `field`'s column, a `DECIMAL(p, 0)`, as the marked text a host provider
/// hands Postgres's unconstrained `numeric` over as: each value's digits.
fn marked(field: &Field, array: &ArrayRef) -> (Field, ArrayRef) {
    assert!(
        matches!(field.data_type(), DataType::Decimal128(_, 0)),
        "{}: {} is not a whole decimal",
        field.name(),
        field.data_type()
    );
    let decimals = array.as_primitive::<Decimal128Type>();
    let text: StringArray = (0..decimals.len())
        .map(|i| (!decimals.is_null(i)).then(|| decimals.value(i).to_string()))
        .collect();
    let field = Field::new(field.name(), DataType::Utf8, field.is_nullable()).with_metadata(
        [(
            EXACT_NUMBER_MARKER.to_owned(),
            EXACT_NUMBER_MARKED.to_owned(),
        )]
        .into(),
    );
    (field, Arc::new(text))
}

/// The case's tables, built on DuckDB from its base and setup, in the
/// non-SQL provider, each `user_id` handed over as marked text and
/// declared `numeric`, as Postgres's provider hands an unconstrained
/// `numeric` over; its catalog's keys as DuckDB declares them.
fn source(case: &Case) -> Memory {
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    let connection = duck::open().unwrap();
    for statement in bases[&case.input.base]
        .iter()
        .chain(case.input.setup.iter().flatten())
    {
        connection.execute_batch(statement).unwrap();
    }
    let mut memory = Memory::from_duckdb(&connection).unwrap();
    let mut made = 0;
    for table in &mut memory.tables {
        // A foreign key named as Postgres's catalog names it (`foreign`),
        // where DuckDB's copy leaves its kind unnamed.
        for key in &mut table.keys {
            if key.kind == "other"
                && table
                    .foreign_keys
                    .iter()
                    .any(|foreign| foreign.columns == key.columns)
            {
                key.kind = KIND_FOREIGN.to_owned();
            }
        }
        let schema = table.rows.schema();
        let Some((at, _)) = schema.column_with_name(KEY) else {
            continue;
        };
        let (field, array) = marked(schema.field(at), table.rows.column(at));
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut columns = table.rows.columns().to_vec();
        fields[at] = field;
        columns[at] = array;
        table.rows = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        for column in &mut table.columns {
            if column.name == KEY {
                column.declared_type = "numeric".to_owned();
            }
        }
        made += 1;
    }
    assert_eq!(made, 2, "{}: users and orders each hold {KEY}", case.id);
    memory
}

/// The case run over its marked source on the swapped data layer.
fn observe(id: &str) -> Observed {
    let case = case(id);
    host::run_memory_over(&case, &source(&case), &layer::root()).unwrap()
}

/// The parts of a column's description that are its representation: what
/// it was received as, its dtype and its meaning. Whether it is nullable or
/// unique is the database's fact (a SQL key declares both, a given key
/// neither), so it is left out.
const REPRESENTATION: [&str; 3] = ["received", "dtype", "stype"];

/// Each `user_id` column's representation, by table, in the schema
/// `materialize` was given: the description with the case's edits made
/// (an edited link's schema as it stands after the edit).
fn key_columns(observed: &Observed) -> Vec<(String, Value)> {
    let schema = observed
        .inputs
        .iter()
        .find(|input| input["op"] == "materialize")
        .map(|input| &input["schema"])
        .unwrap_or_else(|| panic!("materialize not started: {:?}", observed.answer));
    let mut found = Vec::new();
    for table in schema["tables"].as_array().unwrap() {
        for column in table["columns"].as_array().unwrap() {
            if column["name"] == KEY {
                let held: serde_json::Map<String, Value> = REPRESENTATION
                    .iter()
                    .map(|part| ((*part).to_owned(), column[*part].clone()))
                    .collect();
                found.push((
                    table["name"].as_str().unwrap().to_owned(),
                    Value::Object(held),
                ));
            }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(found.len(), 2, "{KEY} not described in both tables");
    found
}

/// The call the core wrote, its request id left out.
fn call(observed: &Observed) -> Value {
    let Answer::Predicted(answer) = &observed.answer else {
        panic!("not answered through predict: {:?}", observed.answer);
    };
    let mut body = answer["body"].clone();
    if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    body
}

/// Whether `column`, as described, is the key's exact text.
fn is_text(column: &Value) -> bool {
    column.to_string().contains("utf8") || column.to_string().contains("string")
}

#[test]
fn the_catalog_key_control_keeps_the_key_exact() {
    gated(CONTROL, || {
        let control = observe(CONTROL);
        assert_eq!(
            control.answer.outcome(),
            ("success", None),
            "{:?} {:?}",
            control.answer,
            control.described.as_ref().map(|d| d.to_string())
        );
        for (table, column) in key_columns(&control) {
            assert!(is_text(&column), "{table}.{KEY} described as {column}");
        }
        call(&control);
    });
}

#[test]
fn a_given_or_edited_key_is_read_as_a_catalog_key_is() {
    let control = observe(CONTROL);
    let mut problems = Vec::new();
    for id in GIVEN {
        let key = gating_key(id);
        let observed = observe(id);
        let found = if observed.answer.outcome() != ("success", None) {
            Some(format!("{:?} where the control answers", observed.answer))
        } else if key_columns(&observed) != key_columns(&control) {
            Some(format!(
                "{KEY} described as {:?} where the control describes {:?}",
                key_columns(&observed),
                key_columns(&control)
            ))
        } else if call(&observed) != call(&control) {
            Some("the call written is not the control's".to_owned())
        } else {
            None
        };
        match (found, named("pending").contains(&key)) {
            (Some(why), false) => problems.push(format!("{id}: {why}")),
            (None, true) => problems.push(format!(
                "{key} passes now: take rust:{key} out of {PENDING}'s pending"
            )),
            _ => {}
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
