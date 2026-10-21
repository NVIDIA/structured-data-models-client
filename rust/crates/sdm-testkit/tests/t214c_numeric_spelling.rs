// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.14c's graph key identity on `rust-host` (the package's test
//! author's): one number spelled two ways is one key, and two numbers are
//! two keys, however close a double holds them (the whole-phase review's
//! Postgres finding; the rides, item 6: exact numeric equality, no float
//! rounding, per P2G-2 and P2C-3).
//! `contract/tests/test_t214c_numeric_spelling.py` holds `python-core` to
//! the same on a real Postgres server.
//!
//! Postgres keeps a `numeric`'s scale, so `1.0` and `1.00` are one number
//! with two texts; its provider hands an unconstrained `numeric` over as
//! that text, its field marked `sdm:exact_number`
//! (`sdm_contract::data::EXACT_NUMBER_MARKER`), which the core reads as text
//! where the schema keys or links it (P2G-2). These tests make such a
//! source as T2.21's do (`t221_exact_keys.rs`): `key.past-2-53.given-link`'s
//! tables (ids from 2**53, the keys and the link from `orders.user_id`
//! given, nothing declared), built on DuckDB and copied into the non-SQL
//! provider (`sdm_testkit::memory`), `users.user_id` and `orders.user_id`
//! handed over as marked text: each user's id written `<digits>.0`, each
//! order's link as the test names, and the entities given as the users'
//! text (`<digits>.0`). The run goes through predict at the case's default
//! number of hops (two: a user, its orders, and the orders' user again).
//!
//! - `spelling.same`: each order's link written `<digits>.0`, as its user's
//!   key is: answered (the control);
//! - `spelling.equal`: each link written `<digits>.00`: one number, so one
//!   key: answered, and the call the core wrote is `spelling.same`'s (the
//!   `payload` rule: but its request id). When written, the non-SQL
//!   provider, which looks keys up by the core's key identity, found no
//!   order for any user (the text `<digits>.0` is not `<digits>.00`), and
//!   the call carried no order; on Postgres, whose SQL joins the two by
//!   value, the second hop took `<digits>.00`'s user for another node and
//!   failed (`INTERNAL_ERROR`, "a row looked up was already reached"), which
//!   `test_t214c_numeric_spelling.py` holds;
//! - `spelling.different`: each link written `<digits>.01`, another number,
//!   which no user's key is, though a double rounds every one of them to a
//!   user's (ids past 2**53 are a double's every other integer): answered,
//!   with no order linked to any user, so the call is not `spelling.same`'s.
//!
//! Gating: `contract/tests/t214c-pending.json`'s `pending` names, as
//! `rust:spelling.<name>@memory-marked`, the keys that failed when written.
//! Each must still fail (strict); once one passes it fails, naming itself,
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
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t214c-pending.json";
const CASE: &str = "key.past-2-53.given-link";
/// The column made exact text in each table.
const KEY: &str = "user_id";
/// How each user's key is written after its digits.
const KEY_SPELLING: &str = ".0";
/// Each test's name and how it writes each order's link after its digits.
const SPELLINGS: [(&str, &str); 3] = [
    ("spelling.same", ".0"),
    ("spelling.equal", ".00"),
    ("spelling.different", ".01"),
];

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
        .map(str::to_owned)
        .collect()
}

fn gating_key(name: &str) -> String {
    format!("{name}@memory-marked")
}

/// Runs `check`; a pending key must fail, any other must pass.
fn gated(name: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let key = gating_key(name);
    let outcome = std::panic::catch_unwind(check);
    match (outcome, named("pending").contains(&key)) {
        (Ok(()), true) => panic!("{key} passes now: take rust:{key} out of {PENDING}'s pending"),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let authored: BTreeSet<String> = SPELLINGS.iter().map(|(n, _)| gating_key(n)).collect();
    let mine = |list: &str| -> BTreeSet<String> {
        named(list)
            .into_iter()
            .filter(|k| k.starts_with("spelling."))
            .collect()
    };
    assert!(mine("pending").is_subset(&mine("authored")));
    assert_eq!(mine("authored"), authored);
}

/// The case, its entities given as the users' keys are written: the text
/// `<digits>.0`, where the case gives each as an integer.
fn case() -> Case {
    let path = root().join(format!("contract/cases/s1/{CASE}.json"));
    let mut case = case::load(&[path]).unwrap().remove(0).case;
    let materialize = case
        .core
        .as_mut()
        .and_then(|core| core.materialize.as_mut())
        .expect("the case materializes");
    for entity in &mut materialize.indices {
        assert_eq!(entity["t"], "i64", "{CASE}: an entity {entity}");
        let digits = entity["v"].as_str().unwrap().to_owned();
        *entity = json!({"t": "utf8", "v": format!("{digits}{KEY_SPELLING}")});
    }
    case
}

/// `field`'s column, a `DECIMAL(p, 0)`, as the marked text a host provider
/// hands Postgres's unconstrained `numeric` over as: each value's digits,
/// then `spelling` (a scale Postgres keeps, `.0` or `.00`, or other digits).
fn marked(field: &Field, array: &ArrayRef, spelling: &str) -> (Field, ArrayRef) {
    assert!(
        matches!(field.data_type(), DataType::Decimal128(_, 0)),
        "{}: {} is not a whole decimal",
        field.name(),
        field.data_type()
    );
    let decimals = array.as_primitive::<Decimal128Type>();
    let text: StringArray = (0..decimals.len())
        .map(|i| (!decimals.is_null(i)).then(|| format!("{}{spelling}", decimals.value(i))))
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
/// `numeric` over: the users' written `<digits>.0`, the orders'
/// `<digits><link>`.
fn source(case: &Case, link: &str) -> Memory {
    let bases = bases::load(&root().join("contract/bases.yaml")).unwrap();
    let connection = duck::open().unwrap();
    for statement in bases[&case.input.base]
        .iter()
        .chain(case.input.setup.iter().flatten())
    {
        connection.execute_batch(statement).unwrap();
    }
    let mut memory = Memory::from_duckdb(&connection).unwrap();
    let mut made = BTreeSet::new();
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
        let spelling = match table.name.as_str() {
            "users" => KEY_SPELLING,
            "orders" => link,
            _ => continue,
        };
        let schema = table.rows.schema();
        let (at, _) = schema
            .column_with_name(KEY)
            .unwrap_or_else(|| panic!("{}: no {KEY}", table.name));
        let (field, array) = marked(schema.field(at), table.rows.column(at), spelling);
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
        made.insert(table.name.clone());
    }
    assert_eq!(made.len(), 2, "users and orders each hold {KEY}");
    memory
}

/// The case run over its marked source, each order's link written
/// `<digits><link>`, on the swapped data layer.
fn observe(link: &str) -> Observed {
    let case = case();
    host::run_memory_over(&case, &source(&case, link), &layer::root()).unwrap()
}

/// The call the core wrote, its request id left out.
fn call(observed: &Observed) -> Value {
    let Answer::Predicted(answer) = &observed.answer else {
        panic!(
            "not answered through predict: {:?} {:?}",
            observed.answer,
            observed.described.as_ref().map(|d| d.to_string())
        );
    };
    let mut body = answer["body"].clone();
    if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.remove("request_id");
    }
    body
}

/// How many rows of `table` the call carries, in every part of it
/// (`context`, `predict`); none where no part samples the table.
fn rows_of(call: &Value, table: &str) -> usize {
    ["context", "predict"]
        .iter()
        .filter_map(|part| call[*part]["related_tables"][table]["rows"].as_array())
        .map(Vec::len)
        .sum()
}

/// The users' text: whether `call` names each user's key as the users'
/// table writes it, `<digits>.0`, wherever it carries one.
fn users_written_as_keys(call: &Value) -> bool {
    call.to_string()
        .contains(&format!("{}{KEY_SPELLING}", 9_007_199_254_740_993_u64))
}

fn spelling(name: &str) -> &'static str {
    SPELLINGS.iter().find(|(n, _)| *n == name).unwrap().1
}

#[test]
fn a_link_written_as_its_key_is_answered() {
    gated("spelling.same", || {
        let same = call(&observe(spelling("spelling.same")));
        assert!(rows_of(&same, "orders") > 0, "no order reached: {same}");
        assert!(rows_of(&same, "users") > 0, "no user reached: {same}");
        assert!(users_written_as_keys(&same), "{same}");
    });
}

#[test]
fn one_number_written_two_ways_is_one_key() {
    let same = call(&observe(spelling("spelling.same")));
    gated("spelling.equal", || {
        let equal = call(&observe(spelling("spelling.equal")));
        assert!(
            equal == same,
            "the call for links written <digits>.00 is not the call for \
             links written <digits>.0, as the users' keys are"
        );
    });
}

#[test]
fn two_numbers_are_two_keys_however_close_a_double_holds_them() {
    let same = call(&observe(spelling("spelling.same")));
    gated("spelling.different", || {
        // Not vacuous: every link is a number no user's key is, which a
        // double rounds to a user's key.
        let first = 9_007_199_254_740_992_u64;
        for id in first..first + 4 {
            let link: f64 = format!("{id}.01").parse().unwrap();
            assert_ne!(format!("{id}.01"), format!("{id}"));
            assert!((first..first + 6).any(|user| user as f64 == link));
        }
        let different = call(&observe(spelling("spelling.different")));
        assert_eq!(
            rows_of(&different, "orders"),
            0,
            "an order whose link is no user's key was reached: {different}"
        );
        assert!(
            rows_of(&different, "users") > 0,
            "no user reached: {different}"
        );
        assert!(different != same);
    });
}
