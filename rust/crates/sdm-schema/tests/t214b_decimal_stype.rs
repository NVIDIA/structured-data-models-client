// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A scaled decimal's semantic type counted as the reference counts it
//! (T2.14b's test author; the decimal stype fix riding along in T2.14b).
//!
//! The reference reads a decimal of scale > 0 as float64 before it infers
//! anything (DuckDB's `Dtype.float`, `_sanitize`'s cast to `pa.float64()`),
//! so values that differ only past float64's precision are one value to it,
//! and the column is categorical. The case `decimal.stype-past-float-precision`
//! holds the whole route to the reference's line on DuckDB, SQLite and
//! Postgres; this holds `sdm-schema`'s inference to the same column read
//! from Arrow: 80 values `1.25 + i * 1e-18` as `DECIMAL(38, 18)`, distinct as
//! decimals and all 1.25 as float64. Two controls keep it honest: values
//! that differ within float64's precision stay numerical, and a decimal of
//! scale 0, which the reference reads as int64, is counted as it is.
//!
//! Gating: `contract/tests/t214b-pending.json`'s `pending` names
//! `stype:rust-unit` while the fix is not in; a pending test must still
//! fail (strict), and once it passes it fails, naming itself, until it is
//! taken out.

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::CatalogColumn;
use sdm_contract::ops::Stype;
use sdm_schema::infer::{self, TextShare, TextTimes, TextUnread};

/// No column here is text: a reading of text as times is never asked.
struct NoText;

impl TextTimes for NoText {
    fn read(&self, _: &[Option<&str>], _: bool) -> Result<Vec<Option<i128>>, TextUnread> {
        Err(TextUnread::Undecided)
    }

    fn share(&self, _: &[Option<&str>]) -> Result<TextShare, TextUnread> {
        Err(TextUnread::Undecided)
    }

    fn day_first_format(&self, _: &[Option<&str>]) -> Option<String> {
        None
    }
}

const ROWS: i128 = 80;

/// The table `orders`, its key `order_id` and the decimal column `d` of
/// `values` at `precision` and `scale`; its stype of `d`.
fn stype_of_d(values: Vec<i128>, precision: u8, scale: i8) -> Stype {
    let rows = values.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, true),
        Field::new("d", DataType::Decimal128(precision, scale), true),
    ]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(
            0..i64::try_from(rows).unwrap(),
        )),
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(precision, scale)
                .unwrap(),
        ),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
    let catalog: Vec<CatalogColumn> = schema
        .fields()
        .iter()
        .map(|f| CatalogColumn {
            canonical: None,
            name: f.name().clone(),
            declared_type: String::new(),
            not_null: false,
            primary_key: false,
            key_nullable: false,
            unique: false,
            dtype: None,
        })
        .collect();
    let table = infer::table(
        "orders",
        &"orders".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &NoText,
    )
    .unwrap();
    table.column("d").unwrap().stype
}

/// `1.25 + i * step` at scale 18, as its unscaled integers.
fn scaled(step: i128) -> Vec<i128> {
    (0..ROWS)
        .map(|i| 1_250_000_000_000_000_000 + i * step)
        .collect()
}

/// The pending file, read as the other vector tests read theirs.
const PENDING: &str = include_str!("../../../../contract/tests/t214b-pending.json");

fn pending(name: &str) -> bool {
    let gating: serde_json::Value = serde_json::from_str(PENDING).unwrap();
    let listed = |list: &str| {
        gating[list]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key.as_str() == Some(name))
    };
    assert!(listed("authored"), "{name} is not among the authored keys");
    listed("pending")
}

/// Runs `check`: it must pass, or, while `name` is pending, fail.
fn held(name: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome.is_ok(), pending(name)) {
        (true, true) => {
            panic!("{name} passes now: take it out of contract/tests/t214b-pending.json's pending")
        }
        (false, false) => std::panic::resume_unwind(outcome.unwrap_err()),
        _ => {}
    }
}

#[test]
fn a_scaled_decimal_that_differs_only_past_float64_is_categorical() {
    // The rides-along fix: the 80 values are one float64 to the
    // reference, so the column is categorical, as the case's line has it.
    held("stype:rust-unit", || {
        assert_eq!(stype_of_d(scaled(1), 38, 18), Stype::Categorical);
    });
}

#[test]
fn a_scaled_decimal_that_differs_within_float64_stays_numerical() {
    // Control: 1e-4 apart, 80 float64 values, so numerical on both.
    assert_eq!(
        stype_of_d(scaled(100_000_000_000_000), 38, 18),
        Stype::Numerical
    );
}

#[test]
fn a_decimal_of_scale_zero_is_counted_as_its_integers() {
    // Control: scale 0 is not cast (the reference reads it as int64, or
    // exact where it does not fit), so 80 consecutive integers from 2**60,
    // which are one float64, still count as 80: numerical.
    let values = (0..ROWS).map(|i| (1_i128 << 60) + i).collect();
    assert_eq!(stype_of_d(values, 38, 0), Stype::Numerical);
}
