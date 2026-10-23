// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.33's type set, held from outside the implementation (the package's
//! independent test author) to `contract/type-set-vectors.json`, which
//! `contract/tests/test_t133_type_set.py` holds to the register and to its
//! own shape.
//!
//! - Exit "each arrow-rs `DataType` classified as accepted, admitted or
//!   refused by name": every variant of the `DataType` of the arrow-schema
//!   version in `rust/Cargo.lock` has at least one vector ([`variant`] is
//!   an exhaustive match, so a new variant fails to compile here until it
//!   is placed), each parameterized family with its parameters (every time
//!   unit, with and without a zone; every interval kind; both union modes;
//!   every integer dictionary index; each decimal width; each list form;
//!   each run-end width), and no type twice. Each vector's `class` is what
//!   admission (`sdm_host::admitted_type` and `admit`, the table at
//!   `sdm-host/src/admission.rs:9-22`) does with the type: `accepted` kept
//!   as received, `admitted` widened to `admitted_as`, or `refused` with
//!   `UNSUPPORTED_TYPE` naming the column (with its path through nested
//!   types) and the Arrow type received there.
//! - "With its received-type inference vector": each buildable vector's
//!   three-row column `f` (a value, a null, a value), beside an `id`
//!   column the catalog declares the primary key, arrives from the driver
//!   in a real `describe_source` run of this host's core (`sdm_core::start`
//!   with `default_root`) through a host provider, which answers the
//!   catalog in the contract's own shape and the preview with the column,
//!   each fetch reply admitted as `HostRun` admits it (`admit`, `runs.rs`).
//!   `inference` is what that gives `f`: its data type in the description
//!   (`dtype`), the data type its `COLUMNS_LEFT_OUT` warning names
//!   (`left_out`), or the refusal by name: the
//!   code and, for `UNSUPPORTED_REQUEST`, the rule, for `UNSUPPORTED_TYPE`
//!   the column and the Arrow type. A panic, or a refusal naming no rule,
//!   column or type, never passes: where the core gives one, the vector
//!   expects the named refusal and is pending.
//!
//! The expected values are derived from the admission table, the closed
//! set (`sdm-contract/src/types.rs` `from_arrow`) and inference's
//! `dtype_of` (`sdm-schema/src/infer.rs`), never recorded from a run.
//!
//! Gating: `contract/tests/t133p2-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written. Each
//! must still fail (strict: one that passes must be taken out of the
//! list), and the list may only shrink: it stays within
//! `authored["rust-host"]`. The implementer removes ids as they pass and
//! changes nothing else of this file or the vectors.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads the vectors, and Arrow's field metadata is a HashMap"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::types::{
    ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, IntervalDayTimeType,
    IntervalMonthDayNanoType, IntervalYearMonthType, RunEndIndexType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array, Decimal32Array,
    Decimal64Array, Decimal128Array, Decimal256Array, DictionaryArray, DurationMicrosecondArray,
    DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray,
    FixedSizeListArray, Float16Array, Float32Array, Float64Array, Int64Array, LargeBinaryArray,
    LargeListArray, LargeListViewArray, LargeStringArray, ListArray, ListViewArray, MapArray,
    NullArray, PrimitiveArray, RecordBatch, RunArray, StringArray, StringViewArray, StructArray,
    Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UnionArray,
};
use arrow_buffer::{
    IntervalDayTime, IntervalMonthDayNano, NullBuffer, OffsetBuffer, ScalarBuffer, i256,
};
use arrow_schema::{
    DataType, Field, FieldRef, Fields, IntervalUnit, Schema, TimeUnit, UnionFields, UnionMode,
};
use half::f16;
use sdm_contract::data::{
    AttachmentId, DataRequest, DescribePart, Determinism, ExpectedSchema, OptionalRequest,
    ProviderCapabilities, ProviderChoice,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{DescribeSource, Operation, OperationResult, SourceDescription};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, ErrorCode, HostFacts, Scalar, Step, Unit};
use sdm_core::{ArrowData, Attachment, FetchResult};
use sdm_host::{ZoneResolver, admit, admitted_type};
use serde_json::{Value, json};

const PENDING: &str = "contract/tests/t133p2-pending.json";
const VECTORS: &str = "contract/type-set-vectors.json";
/// The column every vector is.
const COLUMN: &str = "f";
/// The rows of each vector's column: a value, a null, a value.
const ROWS: usize = 3;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn file() -> Value {
    let all = read_json(VECTORS);
    assert_eq!(all["format"], "sdm-type-set-vectors/1");
    all
}

fn vectors() -> Vec<Value> {
    file()["vectors"].as_array().unwrap().clone()
}

fn names(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn pending() -> BTreeSet<String> {
    names(&read_json(PENDING)["pending"]["rust-host"])
}

// --- the variants ----------------------------------------------------------

/// The variant of `data_type`, by name: every variant of `DataType`, so a
/// new one fails to compile here until a vector places it.
fn variant(data_type: &DataType) -> &'static str {
    match data_type {
        DataType::Null => "Null",
        DataType::Boolean => "Boolean",
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Float16 => "Float16",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Timestamp(..) => "Timestamp",
        DataType::Date32 => "Date32",
        DataType::Date64 => "Date64",
        DataType::Time32(_) => "Time32",
        DataType::Time64(_) => "Time64",
        DataType::Duration(_) => "Duration",
        DataType::Interval(_) => "Interval",
        DataType::Binary => "Binary",
        DataType::FixedSizeBinary(_) => "FixedSizeBinary",
        DataType::LargeBinary => "LargeBinary",
        DataType::BinaryView => "BinaryView",
        DataType::Utf8 => "Utf8",
        DataType::LargeUtf8 => "LargeUtf8",
        DataType::Utf8View => "Utf8View",
        DataType::List(_) => "List",
        DataType::ListView(_) => "ListView",
        DataType::FixedSizeList(..) => "FixedSizeList",
        DataType::LargeList(_) => "LargeList",
        DataType::LargeListView(_) => "LargeListView",
        DataType::Struct(_) => "Struct",
        DataType::Union(..) => "Union",
        DataType::Dictionary(..) => "Dictionary",
        DataType::Decimal32(..) => "Decimal32",
        DataType::Decimal64(..) => "Decimal64",
        DataType::Decimal128(..) => "Decimal128",
        DataType::Decimal256(..) => "Decimal256",
        DataType::Map(..) => "Map",
        DataType::RunEndEncoded(..) => "RunEndEncoded",
    }
}

/// One value of each variant, for the count: the match above is what
/// keeps it whole.
const VARIANTS: [&str; 41] = [
    "Null",
    "Boolean",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float16",
    "Float32",
    "Float64",
    "Timestamp",
    "Date32",
    "Date64",
    "Time32",
    "Time64",
    "Duration",
    "Interval",
    "Binary",
    "FixedSizeBinary",
    "LargeBinary",
    "BinaryView",
    "Utf8",
    "LargeUtf8",
    "Utf8View",
    "List",
    "ListView",
    "FixedSizeList",
    "LargeList",
    "LargeListView",
    "Struct",
    "Union",
    "Dictionary",
    "Decimal32",
    "Decimal64",
    "Decimal128",
    "Decimal256",
    "Map",
    "RunEndEncoded",
];

// --- a vector's type, read -------------------------------------------------

fn unit(text: &str) -> TimeUnit {
    match text {
        "s" => TimeUnit::Second,
        "ms" => TimeUnit::Millisecond,
        "us" => TimeUnit::Microsecond,
        "ns" => TimeUnit::Nanosecond,
        other => panic!("no time unit {other:?}"),
    }
}

fn item(spec: &Value) -> FieldRef {
    Arc::new(Field::new("item", data_type(spec), true))
}

fn int(value: &Value) -> i64 {
    value.as_i64().unwrap()
}

/// The Arrow type a vector's `type` spells: a variant's name for one with
/// no parameters, else `{variant: parameters}`.
fn data_type(spec: &Value) -> DataType {
    if let Some(name) = spec.as_str() {
        return match name {
            "Null" => DataType::Null,
            "Boolean" => DataType::Boolean,
            "Int8" => DataType::Int8,
            "Int16" => DataType::Int16,
            "Int32" => DataType::Int32,
            "Int64" => DataType::Int64,
            "UInt8" => DataType::UInt8,
            "UInt16" => DataType::UInt16,
            "UInt32" => DataType::UInt32,
            "UInt64" => DataType::UInt64,
            "Float16" => DataType::Float16,
            "Float32" => DataType::Float32,
            "Float64" => DataType::Float64,
            "Date32" => DataType::Date32,
            "Date64" => DataType::Date64,
            "Binary" => DataType::Binary,
            "LargeBinary" => DataType::LargeBinary,
            "BinaryView" => DataType::BinaryView,
            "Utf8" => DataType::Utf8,
            "LargeUtf8" => DataType::LargeUtf8,
            "Utf8View" => DataType::Utf8View,
            other => panic!("no type {other:?}"),
        };
    }
    let (name, p) = spec.as_object().unwrap().iter().next().unwrap();
    let pair = |i: usize| &p[i];
    let precision = |i: usize| u8::try_from(int(pair(i))).unwrap();
    let scale = |i: usize| i8::try_from(int(pair(i))).unwrap();
    match name.as_str() {
        "Timestamp" => DataType::Timestamp(
            unit(pair(0).as_str().unwrap()),
            pair(1).as_str().map(Arc::from),
        ),
        "Time32" => DataType::Time32(unit(p.as_str().unwrap())),
        "Time64" => DataType::Time64(unit(p.as_str().unwrap())),
        "Duration" => DataType::Duration(unit(p.as_str().unwrap())),
        "Interval" => DataType::Interval(match p.as_str().unwrap() {
            "YearMonth" => IntervalUnit::YearMonth,
            "DayTime" => IntervalUnit::DayTime,
            "MonthDayNano" => IntervalUnit::MonthDayNano,
            other => panic!("no interval {other:?}"),
        }),
        "FixedSizeBinary" => DataType::FixedSizeBinary(i32::try_from(int(p)).unwrap()),
        "Decimal32" => DataType::Decimal32(precision(0), scale(1)),
        "Decimal64" => DataType::Decimal64(precision(0), scale(1)),
        "Decimal128" => DataType::Decimal128(precision(0), scale(1)),
        "Decimal256" => DataType::Decimal256(precision(0), scale(1)),
        "List" => DataType::List(item(p)),
        "LargeList" => DataType::LargeList(item(p)),
        "ListView" => DataType::ListView(item(p)),
        "LargeListView" => DataType::LargeListView(item(p)),
        "FixedSizeList" => {
            DataType::FixedSizeList(item(pair(0)), i32::try_from(int(pair(1))).unwrap())
        }
        "Struct" => DataType::Struct(
            p.as_array()
                .unwrap()
                .iter()
                .map(|f| Field::new(f[0].as_str().unwrap(), data_type(&f[1]), true))
                .collect::<Fields>(),
        ),
        "Map" => DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", data_type(pair(0)), false),
                    Field::new("value", data_type(pair(1)), true),
                ])),
                false,
            )),
            false,
        ),
        "Union" => {
            let mode = match pair(0).as_str().unwrap() {
                "Sparse" => UnionMode::Sparse,
                "Dense" => UnionMode::Dense,
                other => panic!("no union mode {other:?}"),
            };
            let members = pair(1).as_array().unwrap();
            let codes = members.iter().map(|m| i8::try_from(int(&m[0])).unwrap());
            let fields = members
                .iter()
                .map(|m| Field::new(m[1].as_str().unwrap(), data_type(&m[2]), true));
            DataType::Union(UnionFields::try_new(codes, fields).unwrap(), mode)
        }
        "Dictionary" => {
            DataType::Dictionary(Box::new(data_type(pair(0))), Box::new(data_type(pair(1))))
        }
        "RunEndEncoded" => DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", data_type(pair(0)), false)),
            Arc::new(Field::new("values", data_type(pair(1)), true)),
        ),
        other => panic!("no type {other:?}"),
    }
}

// --- a column of a type ----------------------------------------------------

/// Row `i` is null: the second of every three.
fn null_at(i: usize) -> bool {
    i % 3 == 1
}

fn nulls(n: usize, nullable: bool) -> Option<NullBuffer> {
    nullable.then(|| NullBuffer::from((0..n).map(|i| !null_at(i)).collect::<Vec<_>>()))
}

fn opts<T>(n: usize, nullable: bool, value: impl Fn(usize) -> T) -> Vec<Option<T>> {
    (0..n)
        .map(|i| (!(nullable && null_at(i))).then(|| value(i)))
        .collect()
}

fn small(i: usize) -> i64 {
    i64::try_from(i).unwrap() + 1
}

fn primitive<T: arrow_array::ArrowPrimitiveType>(
    n: usize,
    nullable: bool,
    value: impl Fn(usize) -> T::Native,
) -> ArrayRef {
    Arc::new(
        opts(n, nullable, value)
            .into_iter()
            .collect::<PrimitiveArray<T>>(),
    )
}

fn keys<K: ArrowDictionaryKeyType>(n: usize, values: ArrayRef) -> ArrayRef
where
    K::Native: TryFrom<usize>,
{
    let indices: PrimitiveArray<K> = opts(n, true, |i| K::Native::try_from(i % 2).ok().unwrap())
        .into_iter()
        .collect();
    Arc::new(DictionaryArray::<K>::try_new(indices, values).unwrap())
}

fn runs<R: RunEndIndexType>(n: usize, values: &ArrayRef) -> ArrayRef
where
    R::Native: TryFrom<usize>,
{
    let ends = PrimitiveArray::<R>::from_iter_values(
        (1..=n).map(|i| R::Native::try_from(i).ok().unwrap()),
    );
    Arc::new(RunArray::<R>::try_new(&ends, values.as_ref()).unwrap())
}

/// `n` values of `data_type`, every third from the second null where
/// `nullable`: nested types hold two child values a row.
#[allow(clippy::too_many_lines, reason = "one arm a variant")]
fn column(data_type: &DataType, n: usize, nullable: bool) -> ArrayRef {
    let seconds = |i: usize| 1_700_000_000 + 86_400 * small(i);
    let words = |i: usize| format!("value {i}, a string longer than twelve bytes");
    match data_type {
        DataType::Null => Arc::new(NullArray::new(n)),
        DataType::Boolean => Arc::new(BooleanArray::from(opts(n, nullable, |i| i % 2 == 0))),
        DataType::Int8 => primitive::<Int8Type>(n, nullable, |i| i8::try_from(small(i)).unwrap()),
        DataType::Int16 => {
            primitive::<Int16Type>(n, nullable, |i| i16::try_from(small(i)).unwrap())
        }
        DataType::Int32 => {
            primitive::<Int32Type>(n, nullable, |i| i32::try_from(small(i)).unwrap())
        }
        DataType::Int64 => primitive::<Int64Type>(n, nullable, small),
        DataType::UInt8 => primitive::<UInt8Type>(n, nullable, |i| u8::try_from(small(i)).unwrap()),
        DataType::UInt16 => {
            primitive::<UInt16Type>(n, nullable, |i| u16::try_from(small(i)).unwrap())
        }
        DataType::UInt32 => {
            primitive::<UInt32Type>(n, nullable, |i| u32::try_from(small(i)).unwrap())
        }
        DataType::UInt64 => {
            primitive::<UInt64Type>(n, nullable, |i| u64::try_from(small(i)).unwrap())
        }
        DataType::Float16 => Arc::new(Float16Array::from(opts(n, nullable, |i| {
            f16::from_f32(0.5 + i as f32)
        }))),
        DataType::Float32 => Arc::new(Float32Array::from(opts(n, nullable, |i| 0.5 + i as f32))),
        DataType::Float64 => Arc::new(Float64Array::from(opts(n, nullable, |i| 0.5 + i as f64))),
        DataType::Timestamp(u, zone) => {
            let scale = match u {
                TimeUnit::Second => 1,
                TimeUnit::Millisecond => 1_000,
                TimeUnit::Microsecond => 1_000_000,
                TimeUnit::Nanosecond => 1_000_000_000,
            };
            let values = opts(n, nullable, |i| seconds(i) * scale);
            let zone = zone.clone();
            match u {
                TimeUnit::Second => {
                    Arc::new(TimestampSecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(zone))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(TimestampNanosecondArray::from(values).with_timezone_opt(zone))
                }
            }
        }
        DataType::Date32 => Arc::new(Date32Array::from(opts(n, nullable, |i| {
            19_700 + i32::try_from(i).unwrap()
        }))),
        DataType::Date64 => Arc::new(Date64Array::from(opts(n, nullable, |i| seconds(i) * 1_000))),
        DataType::Time32(TimeUnit::Second) => {
            Arc::new(Time32SecondArray::from(opts(n, nullable, |i| {
                3_600 + i32::try_from(i).unwrap()
            })))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            Arc::new(Time32MillisecondArray::from(opts(n, nullable, |i| {
                3_600_000 + i32::try_from(i).unwrap()
            })))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            Arc::new(Time64MicrosecondArray::from(opts(n, nullable, |i| {
                3_600_000_000 + small(i)
            })))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            Arc::new(Time64NanosecondArray::from(opts(n, nullable, |i| {
                3_600_000_000_000 + small(i)
            })))
        }
        DataType::Duration(u) => {
            let values = opts(n, nullable, small);
            match u {
                TimeUnit::Second => Arc::new(DurationSecondArray::from(values)),
                TimeUnit::Millisecond => Arc::new(DurationMillisecondArray::from(values)),
                TimeUnit::Microsecond => Arc::new(DurationMicrosecondArray::from(values)),
                TimeUnit::Nanosecond => Arc::new(DurationNanosecondArray::from(values)),
            }
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            primitive::<IntervalYearMonthType>(n, nullable, |i| i32::try_from(i).unwrap())
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            primitive::<IntervalDayTimeType>(n, nullable, |i| {
                IntervalDayTime::new(i32::try_from(i).unwrap(), 0)
            })
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            primitive::<IntervalMonthDayNanoType>(n, nullable, |i| {
                IntervalMonthDayNano::new(0, i32::try_from(i).unwrap(), 0)
            })
        }
        DataType::Binary => Arc::new(BinaryArray::from(
            opts(n, nullable, |i| words(i).into_bytes())
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        )),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from(
            opts(n, nullable, |i| words(i).into_bytes())
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        )),
        DataType::BinaryView => Arc::new(BinaryViewArray::from(
            opts(n, nullable, |i| words(i).into_bytes())
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        )),
        DataType::FixedSizeBinary(width) => {
            let width_usize = usize::try_from(*width).unwrap();
            Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    opts(n, nullable, |i| {
                        vec![u8::try_from(i % 256).unwrap(); width_usize]
                    })
                    .into_iter(),
                    *width,
                )
                .unwrap(),
            )
        }
        DataType::Utf8 => Arc::new(StringArray::from(opts(n, nullable, words))),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(opts(n, nullable, words))),
        DataType::Utf8View => Arc::new(StringViewArray::from(opts(n, nullable, words))),
        DataType::List(field) => Arc::new(ListArray::new(
            Arc::clone(field),
            OffsetBuffer::from_lengths(vec![2; n]),
            column(field.data_type(), 2 * n, field.is_nullable()),
            nulls(n, nullable),
        )),
        DataType::LargeList(field) => Arc::new(LargeListArray::new(
            Arc::clone(field),
            OffsetBuffer::from_lengths(vec![2; n]),
            column(field.data_type(), 2 * n, field.is_nullable()),
            nulls(n, nullable),
        )),
        DataType::ListView(field) => Arc::new(ListViewArray::new(
            Arc::clone(field),
            ScalarBuffer::from(
                (0..n)
                    .map(|i| i32::try_from(2 * i).unwrap())
                    .collect::<Vec<_>>(),
            ),
            ScalarBuffer::from(vec![2; n]),
            column(field.data_type(), 2 * n, field.is_nullable()),
            nulls(n, nullable),
        )),
        DataType::LargeListView(field) => Arc::new(LargeListViewArray::new(
            Arc::clone(field),
            ScalarBuffer::from(
                (0..n)
                    .map(|i| i64::try_from(2 * i).unwrap())
                    .collect::<Vec<_>>(),
            ),
            ScalarBuffer::from(vec![2_i64; n]),
            column(field.data_type(), 2 * n, field.is_nullable()),
            nulls(n, nullable),
        )),
        DataType::FixedSizeList(field, size) => Arc::new(FixedSizeListArray::new(
            Arc::clone(field),
            *size,
            column(
                field.data_type(),
                n * usize::try_from(*size).unwrap(),
                field.is_nullable(),
            ),
            nulls(n, nullable),
        )),
        DataType::Struct(fields) if fields.is_empty() => {
            Arc::new(StructArray::new_empty_fields(n, nulls(n, nullable)))
        }
        DataType::Struct(fields) => Arc::new(StructArray::new(
            fields.clone(),
            fields
                .iter()
                .map(|f| column(f.data_type(), n, f.is_nullable()))
                .collect(),
            nulls(n, nullable),
        )),
        DataType::Map(entries, sorted) => {
            let DataType::Struct(children) = entries.data_type() else {
                panic!("a map's entries are a struct");
            };
            let pairs = StructArray::new(
                children.clone(),
                children
                    .iter()
                    .map(|f| column(f.data_type(), 2 * n, f.is_nullable()))
                    .collect(),
                None,
            );
            Arc::new(MapArray::new(
                Arc::clone(entries),
                OffsetBuffer::from_lengths(vec![2; n]),
                pairs,
                nulls(n, nullable),
                *sorted,
            ))
        }
        DataType::Union(fields, mode) => {
            let codes: Vec<i8> = fields.iter().map(|(code, _)| code).collect();
            let ids: Vec<i8> = (0..n).map(|i| codes[i % codes.len()]).collect();
            match mode {
                UnionMode::Sparse => Arc::new(
                    UnionArray::try_new(
                        fields.clone(),
                        ScalarBuffer::from(ids),
                        None,
                        fields
                            .iter()
                            .map(|(_, f)| column(f.data_type(), n, true))
                            .collect(),
                    )
                    .unwrap(),
                ),
                UnionMode::Dense => {
                    let mut seen = vec![0_i32; codes.len()];
                    let offsets: Vec<i32> = ids
                        .iter()
                        .map(|id| {
                            let at = codes.iter().position(|c| c == id).unwrap();
                            seen[at] += 1;
                            seen[at] - 1
                        })
                        .collect();
                    Arc::new(
                        UnionArray::try_new(
                            fields.clone(),
                            ScalarBuffer::from(ids),
                            Some(ScalarBuffer::from(offsets)),
                            fields
                                .iter()
                                .zip(&seen)
                                .map(|((_, f), count)| {
                                    column(f.data_type(), usize::try_from(*count).unwrap(), true)
                                })
                                .collect(),
                        )
                        .unwrap(),
                    )
                }
            }
        }
        DataType::Dictionary(index, values) => {
            let values = column(values, 2, false);
            match **index {
                DataType::Int8 => keys::<Int8Type>(n, values),
                DataType::Int16 => keys::<Int16Type>(n, values),
                DataType::Int32 => keys::<Int32Type>(n, values),
                DataType::Int64 => keys::<Int64Type>(n, values),
                DataType::UInt8 => keys::<UInt8Type>(n, values),
                DataType::UInt16 => keys::<UInt16Type>(n, values),
                DataType::UInt32 => keys::<UInt32Type>(n, values),
                DataType::UInt64 => keys::<UInt64Type>(n, values),
                ref other => panic!("no dictionary index {other}"),
            }
        }
        DataType::Decimal32(p, s) => Arc::new(
            Decimal32Array::from(opts(n, nullable, |i| i32::try_from(small(i)).unwrap()))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Decimal64(p, s) => Arc::new(
            Decimal64Array::from(opts(n, nullable, small))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from(opts(n, nullable, |i| i128::from(small(i))))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Decimal256(p, s) => Arc::new(
            Decimal256Array::from(opts(n, nullable, |i| i256::from_i128(i128::from(small(i)))))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::RunEndEncoded(ends, values) => {
            let values = column(values.data_type(), n, nullable);
            match ends.data_type() {
                DataType::Int16 => runs::<Int16Type>(n, &values),
                DataType::Int32 => runs::<Int32Type>(n, &values),
                DataType::Int64 => runs::<Int64Type>(n, &values),
                other => panic!("no run end type {other}"),
            }
        }
        other => panic!("{other} is not built: a vector of it is not buildable"),
    }
}

// --- what the core does with it ---------------------------------------------

fn detail(error: &CoreError, field: &str) -> Option<String> {
    detail_of(&error.fields, field)
}

fn code(error: &CoreError) -> String {
    serde_json::to_value(error.code)
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned()
}

/// A refusal as a vector states it: its code and what it names. One that
/// names nothing a vector can state reads as unnamed.
fn refusal(error: &CoreError) -> Value {
    match error.code {
        ErrorCode::UnsupportedType => {
            match (detail(error, "column"), detail(error, "arrow_type")) {
                (Some(column), Some(arrow_type)) => json!({"refusal": {
                    "code": code(error), "column": column, "arrow_type": arrow_type,
                }}),
                _ => json!({"unnamed": format!("{error:?}")}),
            }
        }
        ErrorCode::UnsupportedRequest => match detail(error, "rule") {
            Some(rule) => json!({"refusal": {"code": code(error), "rule": rule}}),
            None => json!({"unnamed": format!("{error:?}")}),
        },
        ErrorCode::InternalError => json!({"unnamed": format!("{error:?}")}),
        _ => json!({"refusal": {"code": code(error), "message": error.message.clone()}}),
    }
}

/// What a panic said.
fn panicked(payload: &(dyn std::any::Any + Send)) -> Value {
    let text = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    json!({"panic": text})
}

/// The class admission gives `received` as a type, and the type admitted.
fn classify(received: &DataType) -> (String, Option<DataType>, Option<Value>) {
    match catch_unwind(|| admitted_type(COLUMN, received)) {
        Ok(Ok(admitted)) if &admitted == received => ("accepted".into(), Some(admitted), None),
        Ok(Ok(admitted)) => ("admitted".into(), Some(admitted), None),
        Ok(Err(error)) => ("refused".into(), None, Some(refusal(&error))),
        Err(payload) => ("panicked".into(), None, Some(panicked(payload.as_ref()))),
    }
}

/// `f` of `received` beside the key `id`, as a fetch reply hands it over.
fn table(received: &DataType) -> ArrowData {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(COLUMN, received.clone(), true),
    ]));
    let ids: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![ids, column(received, ROWS, true)]).unwrap();
    ArrowData::new(schema, vec![batch])
}

/// A host provider, which answers the catalog in the contract's own shape
/// and declares the preview, the one optional request describing a table
/// of three rows asks.
fn host_provider() -> ProviderChoice {
    ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            derived_relations: false,
            neighbor_lookback: false,
            sample_rows: false,
            optional: BTreeSet::from([OptionalRequest::Preview]),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::Unseeded,
            fallback: None,
        },
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
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

/// The schema a request's answer must have, where the contract fixes it.
fn expected_schema(request: &DataRequest) -> Option<Arc<Schema>> {
    match request.expected() {
        ExpectedSchema::Schema { fields } => Some(Arc::new(Schema::new(
            fields
                .iter()
                .map(sdm_contract::types::Field::to_arrow)
                .collect::<Vec<_>>(),
        ))),
        ExpectedSchema::Discover {} => None,
    }
}

/// The source's answer to `request`: table `t`'s columns `id` (its
/// declared primary key) and `f`, of `received`, no other key, foreign key
/// or index, and its rows.
fn answer(request: &DataRequest, received: &DataType) -> ArrowData {
    match request {
        DataRequest::Describe {
            part: DescribePart::Columns,
            ..
        } => {
            let schema = expected_schema(request).unwrap();
            let columns: Vec<ArrayRef> = vec![
                Arc::new(StringArray::from(vec!["id", COLUMN])),
                Arc::new(StringArray::from(vec![
                    "BIGINT".to_owned(),
                    received.to_string(),
                ])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            ArrowData::new(schema, vec![batch])
        }
        DataRequest::Describe { .. } => {
            let schema = expected_schema(request).unwrap();
            ArrowData::new(Arc::clone(&schema), vec![RecordBatch::new_empty(schema)])
        }
        DataRequest::Preview { .. } => table(received),
        other => panic!("describe_source asked what this source does not answer: {other:?}"),
    }
}

/// What the core makes of a column of `received` arriving from the
/// driver: `describe_source` of table `t` through a host provider, each
/// fetch reply admitted as `HostRun` admits it (`runs.rs`), and `f`'s data
/// type in the description, the data type its `COLUMNS_LEFT_OUT` warning
/// names, or the refusal. (`HostRun` itself says each warning code once a
/// process, so the run is driven here as `HostRun` drives it.)
fn infer_column(received: &DataType) -> Value {
    let run = || -> Result<Value, CoreError> {
        let spec = json!({
            "tables": [{"name": "t", "source": {"source": ["t"]}}],
            "primary_keys": {},
            "time_columns": {},
        });
        let start = StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::DescribeSource(DescribeSource {
                provider: host_provider(),
                source: AttachmentId(1),
                check_keys: false,
            }),
        };
        let attachments = BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec).unwrap()),
        )]);
        let zones: Arc<dyn sdm_core::Zones> = Arc::new(ZoneResolver::new([]));
        let mut run = sdm_core::start(&sdm_host::default_root(), start, attachments, zones)?;
        for _ in 0..16 {
            match run.next()? {
                Step::Fetch { requests } => {
                    for step in requests {
                        let admitted = admit(&answer(&step.request, received))?;
                        run.submit_fetch(step.id, FetchResult::Data(admitted))?;
                    }
                }
                Step::Done {
                    result: OperationResult::DescribeSource(described),
                } => {
                    let Some(Attachment::Bytes(bytes)) = run.take_attachment(described.description)
                    else {
                        panic!("a description is a bytes attachment");
                    };
                    let description: SourceDescription = serde_json::from_slice(&bytes).unwrap();
                    let table = &description.tables[0];
                    if let Some(column) = table.columns.iter().find(|c| c.name == COLUMN) {
                        return Ok(json!({"dtype": serde_json::to_value(column.dtype).unwrap()}));
                    }
                    let left = run
                        .take_warnings()
                        .into_iter()
                        .find(|w| {
                            w.code == "COLUMNS_LEFT_OUT"
                                && w.fields.get("column") == Some(&Detail::text(COLUMN))
                        })
                        .expect("f is described or left out with the warning naming it");
                    return Ok(json!({"left_out": detail_of(&left.fields, "dtype")}));
                }
                other => panic!("describe_source took another step: {other:?}"),
            }
        }
        panic!("describe_source did not finish in 16 steps");
    };
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(answer)) => answer,
        Ok(Err(error)) => refusal(&error),
        Err(payload) => panicked(payload.as_ref()),
    }
}

fn detail_of(fields: &BTreeMap<String, Detail>, field: &str) -> Option<String> {
    match fields.get(field) {
        Some(Detail::Scalar(Scalar::Utf8(s))) => Some(s.clone()),
        _ => None,
    }
}

/// A vector's expected inference, with each Arrow type it names in
/// arrow-rs's spelling.
fn expected_inference(vector: &Value) -> Value {
    let mut want = vector["inference"].clone();
    if let Some(arrow_type) = want.pointer_mut("/refusal/arrow_type") {
        *arrow_type = Value::String(data_type(arrow_type).to_string());
    }
    want
}

/// What the core does with the vector's type, as the vector states it.
fn observe(vector: &Value) -> BTreeMap<&'static str, Value> {
    let received = data_type(&vector["type"]);
    let (class, admitted, refused) = classify(&received);
    let mut got = BTreeMap::from([
        ("arrow", Value::String(received.to_string())),
        ("variant", Value::String(variant(&received).into())),
        ("class", Value::String(class.clone())),
        (
            "admitted_as",
            match (&admitted, class.as_str()) {
                (Some(admitted), "admitted") => Value::String(admitted.to_string()),
                _ => Value::Null,
            },
        ),
    ]);
    let buildable = vector["buildable"].as_bool().unwrap();
    let inference = match (refused, buildable) {
        // Refused as a type: no array of it gets further, whatever it holds.
        (Some(refused), _) => refused,
        (None, true) => infer_column(&received),
        (None, false) => Value::Null,
    };
    got.insert("inference", inference);
    got
}

fn want(vector: &Value) -> BTreeMap<&'static str, Value> {
    BTreeMap::from([
        ("arrow", vector["arrow"].clone()),
        ("variant", vector["variant"].clone()),
        ("class", vector["class"].clone()),
        (
            "admitted_as",
            match &vector["admitted_as"] {
                Value::Null => Value::Null,
                spec => Value::String(data_type(spec).to_string()),
            },
        ),
        ("inference", expected_inference(vector)),
    ])
}

// --- the tests ----------------------------------------------------------------

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json(PENDING);
    let now = names(&pending["pending"]["rust-host"]);
    let authored = names(&pending["authored"]["rust-host"]);
    assert!(
        now.is_subset(&authored),
        "pending ids added: {:?}",
        now.difference(&authored)
    );
    let all: BTreeSet<String> = vectors()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(
        now.is_subset(&all),
        "pending ids that are no vector: {:?}",
        now.difference(&all)
    );
}

#[test]
fn the_vectors_are_of_the_arrow_schema_the_workspace_locks() {
    let lock = std::fs::read_to_string(root().join("rust/Cargo.lock")).unwrap();
    let at = lock
        .find("name = \"arrow-schema\"\nversion = \"")
        .expect("arrow-schema in Cargo.lock");
    let version = lock[at..].lines().nth(1).unwrap();
    let version = version
        .trim_start_matches("version = \"")
        .trim_end_matches('"');
    assert_eq!(file()["arrow_schema"], version);
}

#[test]
fn every_variant_of_data_type_has_a_vector_and_no_type_has_two() {
    let all = vectors();
    let named: BTreeSet<&str> = all
        .iter()
        .map(|v| variant(&data_type(&v["type"])))
        .collect();
    let every: BTreeSet<&str> = VARIANTS.into_iter().collect();
    assert_eq!(every.len(), 41);
    assert_eq!(named, every, "the variants the vectors place");
    let listed: BTreeSet<String> = names(&file()["variants"]);
    assert_eq!(
        listed,
        every
            .iter()
            .map(|s| (*s).to_owned())
            .collect::<BTreeSet<_>>()
    );
    let mut seen = BTreeMap::new();
    for vector in &all {
        let id = vector["id"].as_str().unwrap();
        let received = data_type(&vector["type"]);
        if let Some(other) = seen.insert(received.clone(), id) {
            panic!("{id} and {other} are both {received}");
        }
    }
}

/// The parameters of each family, each with a vector.
#[test]
fn each_parameterized_family_has_each_parameter() {
    let types: Vec<DataType> = vectors().iter().map(|v| data_type(&v["type"])).collect();
    let has = |f: &dyn Fn(&DataType) -> bool| types.iter().any(f);
    let units = [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ];
    for u in &units {
        assert!(
            has(&|t| matches!(t, DataType::Timestamp(x, None) if x == u)),
            "{u:?}"
        );
        assert!(
            has(&|t| matches!(t, DataType::Timestamp(x, Some(_)) if x == u)),
            "{u:?} zoned"
        );
        for (name, f) in [
            (
                "time32",
                &(|t: &DataType| matches!(t, DataType::Time32(x) if x == u))
                    as &dyn Fn(&DataType) -> bool,
            ),
            (
                "time64",
                &|t: &DataType| matches!(t, DataType::Time64(x) if x == u),
            ),
            (
                "duration",
                &|t: &DataType| matches!(t, DataType::Duration(x) if x == u),
            ),
        ] {
            assert!(has(f), "{name} {u:?}");
        }
    }
    for kind in [
        IntervalUnit::YearMonth,
        IntervalUnit::DayTime,
        IntervalUnit::MonthDayNano,
    ] {
        assert!(
            has(&|t| matches!(t, DataType::Interval(x) if *x == kind)),
            "{kind:?}"
        );
    }
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        assert!(
            has(&|t| matches!(t, DataType::Union(_, x) if *x == mode)),
            "{mode:?}"
        );
    }
    for index in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ] {
        assert!(
            has(
                &|t| matches!(t, DataType::Dictionary(k, v) if **k == index && **v == DataType::Utf8)
            ),
            "dictionary<{index}, utf8>"
        );
    }
    assert!(has(
        &|t| matches!(t, DataType::Dictionary(_, v) if **v != DataType::Utf8)
    ));
    for ends in [DataType::Int16, DataType::Int32, DataType::Int64] {
        assert!(
            has(&|t| matches!(t, DataType::RunEndEncoded(r, _) if *r.data_type() == ends)),
            "run ends {ends}"
        );
    }
    // Each decimal width, in range and out of it where Arrow lets a type be.
    assert!(has(
        &|t| matches!(t, DataType::Decimal128(p, 0) if *p <= 38)
    ));
    assert!(has(
        &|t| matches!(t, DataType::Decimal128(p, s) if *p <= 38 && *s > 0)
    ));
    assert!(has(
        &|t| matches!(t, DataType::Decimal128(p, _) if *p > 38 || *p == 0)
    ));
    assert!(has(
        &|t| matches!(t, DataType::Decimal256(p, _) if *p <= 76)
    ));
    assert!(has(&|t| matches!(t, DataType::Decimal256(p, _) if *p > 76)));
    // Each list form, of an item admission takes and of one it widens.
    for form in [
        "List",
        "LargeList",
        "ListView",
        "LargeListView",
        "FixedSizeList",
    ] {
        assert!(has(&|t| variant(t) == form), "{form}");
    }
}

#[test]
fn each_type_is_admitted_and_inferred_as_its_vector_says() {
    let pending = pending();
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    for vector in vectors() {
        let id = vector["id"].as_str().unwrap();
        let got = observe(&vector);
        let want = want(&vector);
        let ok = got == want;
        match (ok, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => {
                let differ: BTreeMap<_, _> = want
                    .iter()
                    .filter(|(k, v)| got.get(*k) != Some(*v))
                    .map(|(k, v)| (*k, json!({"want": v, "got": got.get(k)})))
                    .collect();
                failures.push(format!(
                    "{id}: {}",
                    Value::from_iter(differ.into_iter().map(|(k, v)| (k.to_owned(), v)))
                ));
            }
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of {PENDING}'s pending[\"rust-host\"]: \
         {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} vectors differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// No vector expects a panic or an unnamed refusal: a refusal names its
/// code and, by the code, its rule or its column and Arrow type.
#[test]
fn every_expected_refusal_is_named() {
    for vector in vectors() {
        let id = vector["id"].as_str().unwrap();
        let inference = &vector["inference"];
        if inference.is_null() {
            assert_eq!(vector["buildable"], false, "{id}: no inference");
            continue;
        }
        let keys: Vec<&String> = inference.as_object().unwrap().keys().collect();
        assert!(
            matches!(keys.as_slice(), [k] if ["dtype", "left_out", "refusal"].contains(&k.as_str())),
            "{id}: {inference}"
        );
        if let Some(refused) = inference.get("refusal") {
            match refused["code"].as_str().unwrap() {
                "UNSUPPORTED_TYPE" => {
                    assert!(
                        refused["column"].is_string() && !refused["arrow_type"].is_null(),
                        "{id}"
                    );
                }
                "UNSUPPORTED_REQUEST" => assert!(refused["rule"].is_string(), "{id}"),
                other => panic!("{id}: a refusal {other} names no rule"),
            }
        }
    }
}
