// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.33 part 3, fix round 1 (the package's independent test author): an
//! `arrow.opaque` field is decoded by its vendor's layout only when its
//! `vendor_name` is the vendor the source declared. Codex's P1: a
//! `fixed_size_binary(16)` field of `type_name` `hugeint` and
//! `vendor_name` `other-vendor`, holding the little-endian bytes of 7,
//! came out of `admit` as an `int64` 7 at 08c531b5, read in DuckDB's
//! layout although nothing says it is DuckDB's.
//!
//! Driven through the public `sdm_host::admit` with no vendor declared:
//! a field of another vendor, or of none, never enters the decoders. It
//! gets what admission gave it before part 3 (00f1889c, measured): what
//! admission gives the same storage with no extension on it, a
//! `fixed_size_binary` widened to `binary` and binary kept, each value's
//! bytes the same, never a number, text or a time of day. The positive control, a real DuckDB
//! HUGEINT (`vendor_name` `DuckDB`) still read as a number, is the type
//! set's `lossless.hugeint` and `lossless.hugeint-wide` rows, which
//! `sdm-testkit`'s `t133p3_reads.rs` runs through real DuckDB.
//!
//! Gated: `contract/tests/t133p3-pending.json`'s `pending["host-vendor"]`
//! lists the tests here that failed at 08c531b5. Each listed one must
//! still fail; the implementer takes an id out once it passes. If the fix
//! adds the declared vendor as an argument of `admit`, the implementer
//! passes "no vendor declared" here and changes nothing else.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads the pending list, and Arrow's field metadata is a HashMap"
)]

use std::collections::{BTreeSet, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BinaryArray, FixedSizeBinaryArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_core::ArrowData;
use sdm_host::{admit, admit_from};

const PENDING: &str = "contract/tests/t133p3-pending.json";
const GROUP: &str = "host-vendor";

fn pending() -> BTreeSet<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(PENDING);
    let loaded: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    loaded["pending"][GROUP]
        .as_array()
        .map(|ids| ids.iter().map(|v| v.as_str().unwrap().to_owned()).collect())
        .unwrap_or_default()
}

/// Runs `check` under the gate: it must pass, or, while `name` is pending,
/// fail.
fn held(name: &str, check: impl FnOnce()) {
    let failed = catch_unwind(AssertUnwindSafe(check)).is_err();
    let listed = pending().contains(name);
    assert!(
        !(listed && !failed),
        "{name} passes now: take it out of {PENDING}'s pending[\"{GROUP}\"]"
    );
    assert!(listed || !failed, "{name} failed (above)");
}

/// One column `f` of `values`, its field carrying `arrow.opaque` with
/// `type_name` and, where given, `vendor_name`.
fn opaque(type_name: &str, vendor: Option<&str>, values: ArrayRef) -> ArrowData {
    let mut extension = serde_json::json!({"type_name": type_name});
    if let Some(vendor) = vendor {
        extension["vendor_name"] = serde_json::json!(vendor);
    }
    let metadata = HashMap::from([
        ("ARROW:extension:name".to_owned(), "arrow.opaque".to_owned()),
        ("ARROW:extension:metadata".to_owned(), extension.to_string()),
    ]);
    let field = Field::new("f", values.data_type().clone(), true).with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).unwrap();
    ArrowData::new(schema, vec![batch])
}

fn fixed(size: i32, values: &[&[u8]]) -> ArrayRef {
    Arc::new(
        FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            values.iter().map(|v| Some(v.to_vec())),
            size,
        )
        .unwrap(),
    )
}

fn binary(values: &[&[u8]]) -> ArrayRef {
    Arc::new(BinaryArray::from_iter_values(values.iter().copied()))
}

/// The bytes of each opaque type's layout in DuckDB's, each a value the
/// decoders at 08c531b5 read: 7 as a HUGEINT and a UHUGEINT, 12:34:56.789
/// +01:30 as a TIMETZ, 7 as a BIGNUM, '101' as a BIT.
fn cases() -> Vec<(&'static str, ArrayRef)> {
    let seven = 7i128.to_le_bytes();
    let tz = [0xe7u8, 0xcb, 0x00, 0x08, 0x26, 0xe6, 0x8b, 0x0a];
    vec![
        ("hugeint", fixed(16, &[&seven])),
        ("uhugeint", fixed(16, &[&seven])),
        ("time_tz", fixed(8, &[&tz])),
        ("bignum", binary(&[&[0x80, 0, 1, 7]])),
        ("bit", binary(&[&[5, 0xfd]])),
    ]
}

/// What admission makes of `values`' storage with no extension on it:
/// what it made of the opaque field before part 3 (00f1889c, measured:
/// a `fixed_size_binary` widened to `binary`, binary kept, each value's
/// bytes the same).
fn as_storage(values: &ArrayRef) -> ArrowData {
    let field = Field::new("f", values.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(values)]).unwrap();
    admit(&ArrowData::new(schema, vec![batch])).expect("the storage is admitted")
}

/// `type_name` of `vendor`, holding `values`, through `admit`: admitted
/// as its storage is, never decoded: the storage's admitted type, the
/// same values.
///
/// `declared` is the vendor the source declares at the data boundary:
/// `None` admits through `admit` (which declares none), `Some` through
/// `admit_from`, as `HostRun` admits a fetch (runs.rs).
fn admitted_as_storage(
    type_name: &str,
    vendor: Option<&str>,
    declared: Option<&str>,
    values: &ArrayRef,
) {
    let data = opaque(type_name, vendor, Arc::clone(values));
    let result = match declared {
        None => admit(&data),
        Some(declared) => admit_from(&data, Some(declared)),
    };
    let admitted = match result {
        Ok(admitted) => admitted,
        Err(refused) => panic!("{type_name} of vendor {vendor:?}: refused {refused:?}"),
    };
    let storage = as_storage(values);
    assert_ne!(
        admitted.schema.field(0).data_type(),
        &DataType::Int64,
        "{type_name} of vendor {vendor:?} was read as a number"
    );
    assert_eq!(
        admitted.schema.field(0).data_type(),
        storage.schema.field(0).data_type(),
        "{type_name} of vendor {vendor:?} was decoded"
    );
    assert_eq!(
        admitted.batches[0].column(0).to_data(),
        storage.batches[0].column(0).to_data(),
        "{type_name} of vendor {vendor:?}: its values changed"
    );
}

fn not_decoded(vendor: Option<&str>, declared: Option<&str>) {
    for (type_name, values) in cases() {
        admitted_as_storage(type_name, vendor, declared, &values);
    }
}

/// Codex's reproduction exactly: a foreign vendor's `hugeint` is never an
/// `int64`.
#[test]
fn another_vendor_s_hugeint_is_not_read_as_a_number() {
    held("another_vendor_s_hugeint_is_not_read_as_a_number", || {
        let seven = 7i128.to_le_bytes();
        admitted_as_storage("hugeint", Some("other-vendor"), None, &fixed(16, &[&seven]));
    });
}

#[test]
fn a_hugeint_naming_no_vendor_is_not_read_as_a_number() {
    held("a_hugeint_naming_no_vendor_is_not_read_as_a_number", || {
        let seven = 7i128.to_le_bytes();
        admitted_as_storage("hugeint", None, None, &fixed(16, &[&seven]));
    });
}

#[test]
fn another_vendor_s_opaque_types_are_admitted_as_before_part_3() {
    held(
        "another_vendor_s_opaque_types_are_admitted_as_before_part_3",
        || not_decoded(Some("other-vendor"), None),
    );
}

#[test]
fn opaque_types_naming_no_vendor_are_admitted_as_before_part_3() {
    held(
        "opaque_types_naming_no_vendor_are_admitted_as_before_part_3",
        || not_decoded(None, None),
    );
}

// Round 2 (the subagent's P3-2): the tests above go through `admit`, which
// declares no vendor and so never reaches the comparison of a column's
// `vendor_name` with the declared one. These go through `admit_from` with
// "DuckDB" declared, as `HostRun` admits a DuckDB source's fetch: the
// comparison decides each. Each failed with the comparison removed
// (checked in a scratch edit of extensions.rs at 59a4efef).

const DUCKDB: &str = "DuckDB";

#[test]
fn duckdb_declared_another_vendor_s_opaque_types_are_admitted_as_before_part_3() {
    held(
        "duckdb_declared_another_vendor_s_opaque_types_are_admitted_as_before_part_3",
        || not_decoded(Some("other-vendor"), Some(DUCKDB)),
    );
}

#[test]
fn duckdb_declared_opaque_types_naming_no_vendor_are_admitted_as_before_part_3() {
    held(
        "duckdb_declared_opaque_types_naming_no_vendor_are_admitted_as_before_part_3",
        || not_decoded(None, Some(DUCKDB)),
    );
}

/// The positive control: DuckDB declared and named, each type is read by
/// DuckDB's layout, never admitted as its storage.
#[test]
fn duckdb_declared_duckdb_s_opaque_types_are_read() {
    held("duckdb_declared_duckdb_s_opaque_types_are_read", || {
        for (type_name, values) in cases() {
            let data = opaque(type_name, Some(DUCKDB), Arc::clone(&values));
            let admitted = admit_from(&data, Some(DUCKDB))
                .unwrap_or_else(|e| panic!("{type_name}: refused {e:?}"));
            let storage = as_storage(&values);
            let read = admitted.schema.field(0).data_type();
            assert_ne!(
                read,
                storage.schema.field(0).data_type(),
                "{type_name} of DuckDB, DuckDB declared, was not read"
            );
            let column = admitted.batches[0].column(0);
            match type_name {
                "hugeint" | "uhugeint" | "bignum" => {
                    assert_eq!(read, &DataType::Int64, "{type_name}");
                    assert_eq!(column.as_primitive::<Int64Type>().value(0), 7);
                }
                "bit" => {
                    assert_eq!(read, &DataType::Utf8, "{type_name}");
                    assert_eq!(column.as_string::<i32>().value(0), "101");
                }
                _ => assert!(
                    matches!(read, DataType::Time64(_) | DataType::Timestamp(..)),
                    "{type_name} read as {read}"
                ),
            }
        }
    });
}
