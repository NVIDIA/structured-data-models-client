// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow admission (decisions B-D6 and H16): the one table that turns the
//! Arrow forms a host may hand over into the closed type set, widening
//! without loss, or refuses them by name.
//!
//! | Received | Admitted as |
//! |---|---|
//! | `int8` carrying the canonical extension `arrow.bool8` | `boolean`, 0 false and any other value true |
//! | `large_utf8`, `utf8_view` | `utf8` |
//! | `large_binary`, `binary_view`, `fixed_size_binary` | `binary` |
//! | `float16` | `float32` |
//! | `decimal32(p, s)`, `decimal64(p, s)` | `decimal128(p, s)` |
//! | `timestamp[s]`, `timestamp[ms]`, with or without a zone | `timestamp[us]`, the same zone |
//! | `duration[s]`, `duration[ms]` | `duration[us]` |
//! | `time32[s]`, `time32[ms]` | `time64[us]` |
//! | `date64` | `timestamp[us]`, whatever its values |
//! | `large_list<T>`, `fixed_size_list<T, n>`, `list_view<T>`, `large_list_view<T>` | `list<T>` |
//! | `dictionary<K, large_utf8>`, `dictionary<K, utf8_view>` | `dictionary<K, utf8>` |
//! | `dictionary<K, V>`, any other `V` | `V`, admitted, the dictionary decoded |
//! | `run_end_encoded<R, V>` | `V`, admitted (values tagged `arrow.bool8`, `boolean`), the runs decoded |
//!
//! Before anything is admitted, each batch must match the schema its
//! attachment declares (check (a) of decision H16): the same number of
//! columns, in the same order, with the same names, types and field
//! metadata, and no nulls in a column declared non-nullable; a mismatch,
//! such as an extra column, two same-typed columns swapped, another decimal
//! scale or timestamp unit, other metadata, or another schema in a later
//! batch, is `PROVIDER_SCHEMA_MISMATCH`, never a dropped, relabelled or
//! filled column. Check (b), against the schema the request expects back,
//! is the run's, in `sdm-core`.
//!
//! Nested types are admitted child by child, keeping each field's name,
//! nullability and metadata, and every value and null. The one exception
//! to keeping metadata: a field read as what its extension type means
//! (`arrow.bool8` as `boolean`) drops the extension's two keys, which would
//! name a storage the admitted field no longer has. DuckDB hands its
//! booleans over as `arrow.bool8` under `arrow_lossless_conversion`, its
//! catalog's included (T1.33). The received schema
//! is kept beside the admitted one ([`ArrowData::received`]). Anything the
//! table does not take into the set is refused with `UNSUPPORTED_TYPE`,
//! naming the column (with its path through nested types) and the Arrow
//! type received there. A `timestamp[s]` or `timestamp[ms]` value past
//! what an int64 count of microseconds reaches (DuckDB's `'infinity'` at
//! those units, or a finite value an Arrow import kept) is admitted as the
//! microsecond count furthest from the epoch on its side, DuckDB's own
//! microsecond infinity, which `materialize` refuses with
//! `VALUE_OUT_OF_RANGE` as a time past the nanosecond range (register
//! DT-14), as the reference describes such a value and refuses it there.
//! Any other value the admitted type cannot hold, such as a `duration[s]`
//! past int64 microseconds, is refused with `VALUE_OUT_OF_RANGE`, never
//! cut. Where the values of a batch would pass the int32 offsets of the
//! admitted type, the batch is split between rows; only a single row past
//! them is refused, with `VALUE_OUT_OF_RANGE` naming the column and the
//! type received (registers P1B-79 and P1B-80).
//!
//! Native-to-Arrow conversion (pandas dtypes, JDBC types) stays in each host.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date64Type, Decimal32Type, Decimal64Type, DurationMillisecondType, DurationSecondType,
    Float16Type, Int8Type, Int16Type, Int32Type, Int64Type, RunEndIndexType, Time32MillisecondType,
    Time32SecondType, TimestampMillisecondType, TimestampSecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Decimal128Array, DurationMicrosecondArray,
    Float32Array, ListArray, MapArray, RecordBatch, RunArray, StringArray, StructArray,
    Time64MicrosecondArray, TimestampMicrosecondArray, UInt64Array,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef, TimeUnit};
use sdm_contract::{ColumnType, CoreError, Detail, Sentence};
use sdm_core::ArrowData;
use sdm_core::run::schema_mismatch;

/// The largest offset an admitted variable-size array holds: int32's.
pub const OFFSET_CAPACITY: usize = i32::MAX as usize;

/// The refusal of a type outside the set, naming what was received.
fn unsupported(column: &str, received: &DataType) -> CoreError {
    CoreError::new(
        Sentence::UnsupportedType,
        format!("column {column:?} has the Arrow type {received}, which the core does not take"),
    )
    .with_field("column", Detail::text(column))
    .with_field("arrow_type", Detail::text(received.to_string()))
}

/// The closed-set type `received` is admitted as, or its refusal by name,
/// naming the type received.
pub fn admitted_type(column: &str, received: &DataType) -> Result<DataType, CoreError> {
    let admitted = convert_type(column, received)?;
    // The whole, nested, type: a map's shape, say.
    ColumnType::from_arrow(column, &admitted).map_err(|_| unsupported(column, received))?;
    Ok(admitted)
}

fn convert_field(parent: &str, field: &Field) -> Result<Field, CoreError> {
    let path = format!("{parent}.{}", field.name());
    admitted_field(&path, field, convert_type)
}

/// The canonical extension type naming an `int8` that holds booleans.
const BOOL8: &str = "arrow.bool8";

/// The keys an extension type sets on its field.
const EXTENSION_KEYS: [&str; 2] = ["ARROW:extension:name", "ARROW:extension:metadata"];

/// Whether `field` is an `int8` carrying `arrow.bool8`, read as booleans.
fn is_bool8(field: &Field) -> bool {
    field.data_type() == &DataType::Int8 && field.extension_type_name() == Some(BOOL8)
}

/// `field` as admitted at `path`: its type through `convert`, its name,
/// nullability and metadata kept, but an `arrow.bool8` field's, which is
/// `boolean` without the extension's keys. The extension on any storage but
/// `int8` (another integer, text, a dictionary of `int8`) is refused by name:
/// read by its storage it would keep a tag naming a type it does not hold,
/// and a dictionary decoded to `int8` would read true as 1.
fn admitted_field(
    path: &str,
    field: &Field,
    convert: fn(&str, &DataType) -> Result<DataType, CoreError>,
) -> Result<Field, CoreError> {
    if field.extension_type_name() == Some(BOOL8) && !is_bool8(field) {
        return Err(unsupported(path, field.data_type()));
    }
    if is_bool8(field) {
        let mut metadata = field.metadata().clone();
        for key in EXTENSION_KEYS {
            metadata.remove(key);
        }
        return Ok(
            Field::new(field.name(), DataType::Boolean, field.is_nullable())
                .with_metadata(metadata),
        );
    }
    Ok(Field::new(
        field.name(),
        convert(path, field.data_type())?,
        field.is_nullable(),
    )
    .with_metadata(field.metadata().clone()))
}

fn is_nested(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(_) | DataType::Struct(_) | DataType::Map(..)
    )
}

fn convert_type(column: &str, received: &DataType) -> Result<DataType, CoreError> {
    use DataType as D;
    let admitted = match received {
        D::LargeUtf8 | D::Utf8View => D::Utf8,
        D::LargeBinary | D::BinaryView | D::FixedSizeBinary(_) => D::Binary,
        D::Float16 => D::Float32,
        D::Decimal32(precision, scale) | D::Decimal64(precision, scale) => {
            D::Decimal128(*precision, *scale)
        }
        D::Timestamp(TimeUnit::Second | TimeUnit::Millisecond, zone) => {
            D::Timestamp(TimeUnit::Microsecond, zone.clone())
        }
        D::Duration(TimeUnit::Second | TimeUnit::Millisecond) => D::Duration(TimeUnit::Microsecond),
        D::Time32(_) => D::Time64(TimeUnit::Microsecond),
        D::Date64 => D::Timestamp(TimeUnit::Microsecond, None),
        D::List(item)
        | D::LargeList(item)
        | D::FixedSizeList(item, _)
        | D::ListView(item)
        | D::LargeListView(item) => D::List(Arc::new(convert_field(column, item)?)),
        D::Struct(fields) => D::Struct(
            fields
                .iter()
                .map(|f| convert_field(column, f))
                .collect::<Result<Fields, _>>()?,
        ),
        D::Map(entries, sorted) => D::Map(Arc::new(convert_field(column, entries)?), *sorted),
        D::Dictionary(index, values)
            if matches!(**values, D::Utf8 | D::LargeUtf8 | D::Utf8View) =>
        {
            D::Dictionary(index.clone(), Box::new(D::Utf8))
        }
        // Any other dictionary is decoded to its values.
        D::Dictionary(_, values) => convert_type(column, values)?,
        // A run-end array is decoded to its values (E17); values tagged
        // `arrow.bool8` are booleans, as the same column unencoded is
        // (admitted_field), the tag on any other storage refused (T2.12, FR
        // P3-8).
        D::RunEndEncoded(_, values) if values.extension_type_name() == Some(BOOL8) => {
            if !is_bool8(values) {
                return Err(unsupported(column, received));
            }
            D::Boolean
        }
        D::RunEndEncoded(_, values) => convert_type(column, values.data_type())?,
        other => other.clone(),
    };
    if !is_nested(&admitted) {
        ColumnType::from_arrow(column, &admitted).map_err(|_| unsupported(column, received))?;
    }
    Ok(admitted)
}

/// The refusal of a value the admitted type cannot hold.
fn out_of_range(column: &str, received: &DataType, admitted: &DataType) -> CoreError {
    CoreError::new(
        Sentence::AdmittedPastType,
        format!(
            "column {column:?} holds a value of Arrow type {received} that {admitted} cannot \
             hold"
        ),
    )
    .with_field("column", Detail::text(column))
    .with_field("arrow_type", Detail::text(received.to_string()))
    .with_field("admitted_type", Detail::text(admitted.to_string()))
}

/// Admits `data`: checks each batch against its declared schema, then
/// admits the schema and every batch, column by column. No opaque column
/// is read by a vendor's layout ([`admit_from`]).
pub fn admit(data: &ArrowData) -> Result<ArrowData, CoreError> {
    admit_from(data, None)
}

/// [`admit`], for data a provider whose opaque Arrow extension types are
/// `vendor`'s handed over (`Run::opaque_vendor`): a top-level opaque
/// column naming that vendor is read as what it means (crate::extensions,
/// register P1B-43); one naming another vendor or none is admitted by its
/// storage, as before.
pub fn admit_from(data: &ArrowData, vendor: Option<&str>) -> Result<ArrowData, CoreError> {
    admitted(data, OFFSET_CAPACITY, vendor)
}

/// [`admit`], with offsets bounded by `capacity` instead of int32's, so
/// tests can split and refuse without gigabytes of data.
pub fn admit_with_capacity(data: &ArrowData, capacity: usize) -> Result<ArrowData, CoreError> {
    admitted(data, capacity, None)
}

fn admitted(
    data: &ArrowData,
    capacity: usize,
    vendor: Option<&str>,
) -> Result<ArrowData, CoreError> {
    let capacity = capacity.min(OFFSET_CAPACITY);
    for batch in &data.batches {
        check_declared(&data.schema, batch)?;
    }
    // After check (a), before the table: a column of one of DuckDB's
    // extension types (a UUID, a BIT, a TIMETZ, a HUGEINT, a UHUGEINT or a
    // BIGNUM) is read as what it means (crate::extensions, P1B-43).
    let (read, read_batches) = crate::extensions::read(&data.schema, &data.batches, vendor)?;
    let fields: Vec<Field> = read
        .fields()
        .iter()
        .map(|f| admitted_field(f.name(), f, admitted_type))
        .collect::<Result<_, CoreError>>()?;
    let schema: SchemaRef = Arc::new(Schema::new_with_metadata(fields, read.metadata().clone()));
    let admit = Admit { capacity };
    let mut batches = Vec::new();
    for batch in &read_batches {
        admit.batch(batch, &schema, &mut batches)?;
    }
    Ok(ArrowData {
        schema,
        batches,
        received: Arc::clone(&data.schema),
    })
}

/// Check (a): a batch against the schema its attachment declares.
fn check_declared(declared: &Schema, batch: &RecordBatch) -> Result<(), CoreError> {
    let mismatch = |column: Option<&str>, reason: &str| schema_mismatch("declared", column, reason);
    let got = batch.schema();
    if got.fields().len() != declared.fields().len() {
        return Err(mismatch(None, "the number of columns differs"));
    }
    for (index, (field, wanted)) in got.fields().iter().zip(declared.fields()).enumerate() {
        let name = wanted.name().as_str();
        if field.name() != wanted.name() {
            return Err(mismatch(Some(name), "a column's name differs"));
        }
        if field.data_type() != wanted.data_type() {
            return Err(mismatch(Some(name), "a column's type differs"));
        }
        if field.metadata() != wanted.metadata() {
            return Err(mismatch(Some(name), "a column's metadata differs"));
        }
        // Logical nulls: a run-end array's nulls are its values'.
        if !wanted.is_nullable() && batch.column(index).logical_null_count() > 0 {
            return Err(mismatch(
                Some(name),
                "a column declared without nulls holds nulls",
            ));
        }
    }
    Ok(())
}

/// Why a conversion failed: values past the offset capacity, which a split
/// between rows may cure, or a refusal no split cures.
enum Fail {
    Capacity(CoreError),
    Refused(CoreError),
}

impl From<CoreError> for Fail {
    fn from(error: CoreError) -> Self {
        Fail::Refused(error)
    }
}

type Converted<T> = Result<T, Fail>;

struct Admit {
    capacity: usize,
}

/// The rows `offsets` span: where they start, and the offsets rebased to
/// start at 0, if the span fits `capacity`.
fn rebase(offsets: &[i64], capacity: usize) -> Option<(usize, OffsetBuffer<i32>)> {
    let first = *offsets.first()?;
    let last = *offsets.last()?;
    let span = usize::try_from(last.checked_sub(first)?).ok()?;
    if span > capacity {
        return None;
    }
    let start = usize::try_from(first).ok()?;
    let rebased = offsets
        .iter()
        .map(|o| i32::try_from(o - first).ok())
        .collect::<Option<Vec<i32>>>()?;
    // The offsets of a valid array rise; rebasing keeps their order.
    Some((start, OffsetBuffer::new(ScalarBuffer::from(rebased))))
}

/// The runs of a run-end array the array's slice covers, in row order:
/// each run's physical index and how many of its rows the slice holds, or
/// `None` if the run ends do not cover those rows. Only the runs inside
/// the slice are read.
fn runs<R: RunEndIndexType>(run: &RunArray<R>) -> Option<(&ArrayRef, Vec<(usize, usize)>)> {
    let ends = run.run_ends();
    let mut picks = Vec::new();
    if ends.is_empty() {
        return Some((run.values(), picks));
    }
    let stop = ends.offset().checked_add(ends.len())?;
    let first = run.get_start_physical_index();
    let last = run.get_end_physical_index();
    let mut row = ends.offset();
    for (physical, end) in (first..).zip(ends.values().get(first..=last)?) {
        let end = end.to_usize()?.min(stop);
        if end > row {
            picks.push((physical, end.checked_sub(row)?));
            row = end;
        }
    }
    (row == stop).then_some((run.values(), picks))
}

/// Adds `times` the size the elements `start..end` of `array` take at each
/// level with int32 offsets (string or binary bytes, list or map items) to
/// that level's total, numbering the levels in the order the type is
/// walked; `None` if a size cannot be read or a total passes usize.
fn tally(
    array: &dyn Array,
    (start, end): (usize, usize),
    times: usize,
    totals: &mut Vec<usize>,
    level: &mut usize,
) -> Option<()> {
    fn span<O: ArrowNativeType>(offsets: &[O], start: usize, end: usize) -> Option<(usize, usize)> {
        Some((
            offsets.get(start)?.to_usize()?,
            offsets.get(end)?.to_usize()?,
        ))
    }
    fn add(
        totals: &mut Vec<usize>,
        level: &mut usize,
        (from, to): (usize, usize),
        times: usize,
    ) -> Option<()> {
        if totals.len() <= *level {
            totals.resize(*level + 1, 0);
        }
        let size = to.checked_sub(from)?.checked_mul(times)?;
        totals[*level] = totals[*level].checked_add(size)?;
        *level += 1;
        Some(())
    }
    // Below the top level a null value's bytes count too: `take` skips
    // them in a struct's string child but not where it copies ranges, and
    // counting too much only splits sooner.
    match array.data_type() {
        DataType::Utf8 => {
            let offsets = array.as_string::<i32>().value_offsets();
            add(totals, level, span(offsets, start, end)?, times)
        }
        DataType::Binary => {
            let offsets = array.as_binary::<i32>().value_offsets();
            add(totals, level, span(offsets, start, end)?, times)
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let items = span(list.value_offsets(), start, end)?;
            add(totals, level, items, times)?;
            tally(list.values().as_ref(), items, times, totals, level)
        }
        DataType::Map(..) => {
            let map = array.as_map();
            let items = span(map.value_offsets(), start, end)?;
            add(totals, level, items, times)?;
            tally(map.entries(), items, times, totals, level)
        }
        // Its own offsets cannot pass int32's; its items' can.
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let items = span(list.value_offsets(), start, end)?;
            tally(list.values().as_ref(), items, times, totals, level)
        }
        DataType::FixedSizeList(_, size) => {
            let size = usize::try_from(*size).ok()?;
            let items = (start.checked_mul(size)?, end.checked_mul(size)?);
            let values = array.as_fixed_size_list().values();
            tally(values.as_ref(), items, times, totals, level)
        }
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .try_for_each(|c| tally(c.as_ref(), (start, end), times, totals, level)),
        // `take` copies no values of any other type into int32 offsets.
        _ => Some(()),
    }
}

/// A failed `take`: values past int32 offsets are `past`, which a split
/// between rows may cure; anything else is internal.
fn taken(column: &str, error: arrow_schema::ArrowError, past: Fail) -> Fail {
    match error {
        arrow_schema::ArrowError::OffsetOverflowError(_) => past,
        other => Fail::Refused(internal(column, &other)),
    }
}

fn widen_offsets(offsets: &[i32]) -> Vec<i64> {
    offsets.iter().map(|o| i64::from(*o)).collect()
}

fn internal(column: &str, e: &arrow_schema::ArrowError) -> CoreError {
    CoreError::new(
        Sentence::Internal,
        format!("admitting column {column:?}: {e}"),
    )
}

fn child(column: &str, field: &FieldRef) -> String {
    format!("{column}.{}", field.name())
}

/// Scales every valid value by `factor`, or refuses with `overflow`; a null
/// slot's value is not a value, and is written as 0.
fn scale(
    values: impl Iterator<Item = i64>,
    nulls: Option<&NullBuffer>,
    factor: i64,
    overflow: impl Fn() -> CoreError,
) -> Result<Vec<i64>, CoreError> {
    values
        .enumerate()
        .map(|(i, v)| match nulls {
            Some(n) if n.is_null(i) => Some(0),
            _ => v.checked_mul(factor),
        })
        .collect::<Option<Vec<i64>>>()
        .ok_or_else(overflow)
}

/// The microsecond count furthest from the epoch on each side: DuckDB's
/// `'infinity'` and `'-infinity'` as a `TIMESTAMP`.
const MICROS_PAST: i64 = i64::MAX;

/// Scales every valid timestamp by `factor`, a value past what an int64
/// count of microseconds holds kept as [`MICROS_PAST`] on its side, as
/// DuckDB hands over a microsecond infinity (register DT-14, T1.36 fix 3):
/// every such value is past what a nanosecond count holds, which
/// `materialize` refuses with `VALUE_OUT_OF_RANGE` wherever it reads one,
/// and kept so it orders after (or before) every value the type holds. A
/// null slot's value is not a value, and is written as 0.
fn saturate(
    values: impl Iterator<Item = i64>,
    nulls: Option<&NullBuffer>,
    factor: i64,
) -> Vec<i64> {
    values
        .enumerate()
        .map(|(i, v)| match nulls {
            Some(n) if n.is_null(i) => 0,
            _ => v
                .checked_mul(factor)
                .unwrap_or(if v < 0 { -MICROS_PAST } else { MICROS_PAST }),
        })
        .collect()
}

impl Admit {
    /// Admits `batch` into `out`, split between rows wherever its values
    /// would pass the offset capacity.
    fn batch(
        &self,
        batch: &RecordBatch,
        schema: &SchemaRef,
        out: &mut Vec<RecordBatch>,
    ) -> Result<(), CoreError> {
        let rows = batch.num_rows();
        if rows == 0 {
            return match self.rows(batch, schema) {
                Ok(admitted) => {
                    out.push(admitted);
                    Ok(())
                }
                Err(Fail::Capacity(e) | Fail::Refused(e)) => Err(e),
            };
        }
        let mut start = 0;
        while start < rows {
            let mut len = rows - start;
            loop {
                match self.rows(&batch.slice(start, len), schema) {
                    Ok(admitted) => {
                        out.push(admitted);
                        start += len;
                        break;
                    }
                    Err(Fail::Refused(e)) => return Err(e),
                    // One row past the capacity cannot be split.
                    Err(Fail::Capacity(e)) if len == 1 => return Err(e),
                    Err(Fail::Capacity(_)) => len = len.div_ceil(2),
                }
            }
        }
        Ok(())
    }

    fn rows(&self, batch: &RecordBatch, schema: &SchemaRef) -> Converted<RecordBatch> {
        let columns = batch
            .columns()
            .iter()
            .zip(schema.fields())
            .map(|(array, admitted)| self.array(admitted.name(), array, admitted.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new(Arc::clone(schema), columns)
            .map_err(|e| Fail::Refused(internal("", &e)))
    }

    /// `array`, of a type [`admitted_type`] took, as `admitted`.
    fn array(&self, column: &str, array: &ArrayRef, admitted: &DataType) -> Converted<ArrayRef> {
        let received = array.data_type();
        if received == admitted {
            return Ok(Arc::clone(array));
        }
        let past = || Fail::Capacity(out_of_range(column, received, admitted));
        let overflow = || out_of_range(column, received, admitted);
        let rebuilt = |e: arrow_schema::ArrowError| Fail::Refused(internal(column, &e));
        let nulls = array.nulls().cloned();
        Ok(match (received, admitted) {
            // An `arrow.bool8` field's values (admitted_field).
            (DataType::Int8, DataType::Boolean) => Arc::new(BooleanArray::new(
                array
                    .as_primitive::<Int8Type>()
                    .values()
                    .iter()
                    .map(|v| *v != 0)
                    .collect(),
                nulls,
            )),
            (DataType::LargeUtf8, DataType::Utf8) => {
                let large = array.as_string::<i64>();
                let (start, offsets) = rebase(large.offsets(), self.capacity).ok_or_else(past)?;
                let len = usize::try_from(*offsets.last().unwrap_or(&0)).unwrap_or(0);
                Arc::new(
                    StringArray::try_new(
                        offsets,
                        large.values().slice_with_length(start, len),
                        nulls,
                    )
                    .map_err(rebuilt)?,
                )
            }
            (DataType::LargeBinary, DataType::Binary) => {
                let large = array.as_binary::<i64>();
                let (start, offsets) = rebase(large.offsets(), self.capacity).ok_or_else(past)?;
                let len = usize::try_from(*offsets.last().unwrap_or(&0)).unwrap_or(0);
                Arc::new(
                    BinaryArray::try_new(
                        offsets,
                        large.values().slice_with_length(start, len),
                        nulls,
                    )
                    .map_err(rebuilt)?,
                )
            }
            (DataType::Utf8View, DataType::Utf8) => {
                let view = array.as_string_view();
                let total: usize = view.iter().map(|v| v.map_or(0, str::len)).sum();
                if total > self.capacity {
                    return Err(past());
                }
                Arc::new(view.iter().collect::<StringArray>())
            }
            (DataType::BinaryView, DataType::Binary) => {
                let view = array.as_binary_view();
                let total: usize = view.iter().map(|v| v.map_or(0, <[u8]>::len)).sum();
                if total > self.capacity {
                    return Err(past());
                }
                Arc::new(view.iter().collect::<BinaryArray>())
            }
            (DataType::FixedSizeBinary(width), DataType::Binary) => {
                let fixed = array.as_fixed_size_binary();
                let total = usize::try_from(*width)
                    .ok()
                    .and_then(|w| w.checked_mul(fixed.len()));
                if total.is_none_or(|t| t > self.capacity) {
                    return Err(past());
                }
                Arc::new(fixed.iter().collect::<BinaryArray>())
            }
            (DataType::Float16, DataType::Float32) => {
                let half = array.as_primitive::<Float16Type>();
                let values: Vec<f32> = half.values().iter().map(|v| v.to_f32()).collect();
                Arc::new(Float32Array::new(ScalarBuffer::from(values), nulls))
            }
            (DataType::Decimal32(..), DataType::Decimal128(precision, scale)) => {
                let narrow = array.as_primitive::<Decimal32Type>();
                let values: Vec<i128> = narrow.values().iter().map(|v| i128::from(*v)).collect();
                Arc::new(
                    Decimal128Array::new(ScalarBuffer::from(values), nulls)
                        .with_precision_and_scale(*precision, *scale)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::Decimal64(..), DataType::Decimal128(precision, scale)) => {
                let narrow = array.as_primitive::<Decimal64Type>();
                let values: Vec<i128> = narrow.values().iter().map(|v| i128::from(*v)).collect();
                Arc::new(
                    Decimal128Array::new(ScalarBuffer::from(values), nulls)
                        .with_precision_and_scale(*precision, *scale)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::Timestamp(unit, zone), DataType::Timestamp(TimeUnit::Microsecond, _)) => {
                let (values, factor) = match unit {
                    TimeUnit::Second => (
                        array.as_primitive::<TimestampSecondType>().values().clone(),
                        1_000_000,
                    ),
                    TimeUnit::Millisecond => (
                        array
                            .as_primitive::<TimestampMillisecondType>()
                            .values()
                            .clone(),
                        1_000,
                    ),
                    _ => {
                        return Err(rebuilt(arrow_schema::ArrowError::CastError(
                            unit.to_string(),
                        )));
                    }
                };
                let scaled = saturate(values.iter().copied(), nulls.as_ref(), factor);
                Arc::new(
                    TimestampMicrosecondArray::try_new(ScalarBuffer::from(scaled), nulls)
                        .map_err(rebuilt)?
                        .with_timezone_opt(zone.clone()),
                )
            }
            (DataType::Date64, DataType::Timestamp(TimeUnit::Microsecond, None)) => {
                let millis = array.as_primitive::<Date64Type>().values().clone();
                let scaled = scale(millis.iter().copied(), nulls.as_ref(), 1_000, overflow)?;
                Arc::new(
                    TimestampMicrosecondArray::try_new(ScalarBuffer::from(scaled), nulls)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::Duration(unit), DataType::Duration(TimeUnit::Microsecond)) => {
                let (values, factor) = match unit {
                    TimeUnit::Second => (
                        array.as_primitive::<DurationSecondType>().values().clone(),
                        1_000_000,
                    ),
                    TimeUnit::Millisecond => (
                        array
                            .as_primitive::<DurationMillisecondType>()
                            .values()
                            .clone(),
                        1_000,
                    ),
                    _ => {
                        return Err(rebuilt(arrow_schema::ArrowError::CastError(
                            unit.to_string(),
                        )));
                    }
                };
                let scaled = scale(values.iter().copied(), nulls.as_ref(), factor, overflow)?;
                Arc::new(
                    DurationMicrosecondArray::try_new(ScalarBuffer::from(scaled), nulls)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::Time32(unit), DataType::Time64(TimeUnit::Microsecond)) => {
                let values: Vec<i64> = match unit {
                    TimeUnit::Second => array
                        .as_primitive::<Time32SecondType>()
                        .values()
                        .iter()
                        .map(|v| i64::from(*v))
                        .collect(),
                    TimeUnit::Millisecond => array
                        .as_primitive::<Time32MillisecondType>()
                        .values()
                        .iter()
                        .map(|v| i64::from(*v))
                        .collect(),
                    _ => {
                        return Err(rebuilt(arrow_schema::ArrowError::CastError(
                            unit.to_string(),
                        )));
                    }
                };
                let factor = if *unit == TimeUnit::Second {
                    1_000_000
                } else {
                    1_000
                };
                let scaled = scale(values.into_iter(), nulls.as_ref(), factor, overflow)?;
                Arc::new(
                    Time64MicrosecondArray::try_new(ScalarBuffer::from(scaled), nulls)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::List(_), DataType::List(item)) => {
                let list = array.as_list::<i32>();
                let offsets = widen_offsets(list.offsets());
                self.list(column, received, &offsets, list.values(), item, nulls)?
            }
            (DataType::LargeList(_), DataType::List(item)) => {
                let list = array.as_list::<i64>();
                self.list(column, received, list.offsets(), list.values(), item, nulls)?
            }
            (DataType::FixedSizeList(_, size), DataType::List(item)) => {
                let list = array.as_fixed_size_list();
                let size = i64::from(*size);
                let offsets = (0..=list.len())
                    .map(|i| i64::try_from(i).ok().and_then(|i| i.checked_mul(size)))
                    .collect::<Option<Vec<i64>>>()
                    .ok_or_else(past)?;
                self.list(column, received, &offsets, list.values(), item, nulls)?
            }
            (DataType::ListView(_), DataType::List(item)) => {
                let view = array.as_list_view::<i32>();
                let spans = view
                    .offsets()
                    .iter()
                    .zip(view.sizes().iter())
                    .map(|(o, s)| (i64::from(*o), i64::from(*s)));
                self.view(column, received, spans, view.values(), item, nulls)?
            }
            (DataType::LargeListView(_), DataType::List(item)) => {
                let view = array.as_list_view::<i64>();
                let spans = view
                    .offsets()
                    .iter()
                    .zip(view.sizes().iter())
                    .map(|(o, s)| (*o, *s));
                self.view(column, received, spans, view.values(), item, nulls)?
            }
            (DataType::Struct(_), DataType::Struct(fields)) => {
                Arc::new(self.r#struct(column, array.as_struct(), fields)?)
            }
            (DataType::Map(_, _), DataType::Map(entries, sorted)) => {
                let map = array.as_map();
                let DataType::Struct(fields) = entries.data_type() else {
                    return Err(rebuilt(arrow_schema::ArrowError::CastError(
                        "a map's entries are a struct".into(),
                    )));
                };
                let (start, offsets) =
                    rebase(&widen_offsets(map.offsets()), self.capacity).ok_or_else(past)?;
                let len = usize::try_from(*offsets.last().unwrap_or(&0)).unwrap_or(0);
                let path = child(column, entries);
                let converted = self.r#struct(&path, &map.entries().slice(start, len), fields)?;
                Arc::new(
                    MapArray::try_new(Arc::clone(entries), offsets, converted, nulls, *sorted)
                        .map_err(rebuilt)?,
                )
            }
            (DataType::Dictionary(_, _), DataType::Dictionary(_, values)) => {
                let dictionary = array.as_any_dictionary();
                let converted = self.array(column, dictionary.values(), values)?;
                dictionary.with_values(converted)
            }
            (DataType::Dictionary(_, _), _) => {
                // Decoded: each key's value, a null key a null.
                let dictionary = array.as_any_dictionary();
                let (values, keys) = (dictionary.values(), dictionary.keys());
                let picks: Vec<(usize, usize)> = if values.is_empty() {
                    // Every key is null.
                    Vec::new()
                } else {
                    let normalized = dictionary.normalized_keys();
                    (0..keys.len())
                        .filter(|row| keys.is_valid(*row))
                        .map(|row| (normalized[row], 1))
                        .collect()
                };
                if !self.fits_taken(values, &picks) {
                    return Err(past());
                }
                let decoded = arrow_select::take::take(values.as_ref(), keys, None)
                    .map_err(|e| taken(column, e, past()))?;
                self.array(column, &decoded, admitted)?
            }
            (DataType::RunEndEncoded(run_ends, _), _) => {
                let (values, picks) = match run_ends.data_type() {
                    DataType::Int16 => runs(array.as_run::<Int16Type>()),
                    DataType::Int32 => runs(array.as_run::<Int32Type>()),
                    DataType::Int64 => runs(array.as_run::<Int64Type>()),
                    _ => None,
                }
                .ok_or_else(|| {
                    rebuilt(arrow_schema::ArrowError::InvalidArgumentError(
                        "a run-end array's run ends do not cover its rows".into(),
                    ))
                })?;
                if !self.fits_taken(values, &picks) {
                    return Err(past());
                }
                // Decoded: each row its run's value, a null run's rows null.
                let indices = picks
                    .iter()
                    .map(|(physical, rows)| u64::try_from(*physical).map(|p| (p, *rows)))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| past())?
                    .into_iter()
                    .flat_map(|(physical, rows)| std::iter::repeat_n(physical, rows))
                    .collect::<UInt64Array>();
                let decoded = arrow_select::take::take(values.as_ref(), &indices, None)
                    .map_err(|e| taken(column, e, past()))?;
                self.array(column, &decoded, admitted)?
            }
            _ => {
                return Err(rebuilt(arrow_schema::ArrowError::CastError(format!(
                    "no admission from {received} to {admitted}"
                ))));
            }
        })
    }

    /// Whether taking `values` at `picks`, each a physical index and how
    /// many rows repeat it, keeps every level with int32 offsets within the
    /// capacity; checked before `take` builds anything. A null string,
    /// binary, list or map value takes nothing; `take` copies a null struct
    /// or fixed-size list slot's children all the same, so they count.
    fn fits_taken(&self, values: &ArrayRef, picks: &[(usize, usize)]) -> bool {
        let skips_nulls = matches!(
            values.data_type(),
            DataType::Utf8
                | DataType::Binary
                | DataType::List(_)
                | DataType::LargeList(_)
                | DataType::Map(..)
        );
        let mut totals = Vec::new();
        picks
            .iter()
            .filter(|(physical, _)| !skips_nulls || values.is_valid(*physical))
            .try_for_each(|(physical, rows)| {
                let range = (*physical, physical.checked_add(1)?);
                tally(values.as_ref(), range, *rows, &mut totals, &mut 0)
            })
            .is_some_and(|()| totals.iter().all(|total| *total <= self.capacity))
    }

    /// A list of `item` over the rows `offsets` span of `values`.
    fn list(
        &self,
        column: &str,
        received: &DataType,
        offsets: &[i64],
        values: &ArrayRef,
        item: &FieldRef,
        nulls: Option<NullBuffer>,
    ) -> Converted<ArrayRef> {
        let past = || {
            Fail::Capacity(out_of_range(
                column,
                received,
                &DataType::List(Arc::clone(item)),
            ))
        };
        let (start, offsets) = rebase(offsets, self.capacity).ok_or_else(past)?;
        let len = usize::try_from(*offsets.last().unwrap_or(&0)).unwrap_or(0);
        let values = self.array(
            &child(column, item),
            &values.slice(start, len),
            item.data_type(),
        )?;
        Ok(Arc::new(
            ListArray::try_new(Arc::clone(item), offsets, values, nulls)
                .map_err(|e| Fail::Refused(internal(column, &e)))?,
        ))
    }

    /// A list of `item` from a list view's `(offset, size)` spans: each
    /// row's values in order, a null row empty.
    fn view(
        &self,
        column: &str,
        received: &DataType,
        spans: impl Iterator<Item = (i64, i64)>,
        values: &ArrayRef,
        item: &FieldRef,
        nulls: Option<NullBuffer>,
    ) -> Converted<ArrayRef> {
        let past = || {
            Fail::Capacity(out_of_range(
                column,
                received,
                &DataType::List(Arc::clone(item)),
            ))
        };
        let mut indices: Vec<u64> = Vec::new();
        let mut offsets: Vec<i64> = vec![0];
        for (row, (offset, size)) in spans.enumerate() {
            if nulls.as_ref().is_none_or(|n| n.is_valid(row)) {
                let from = u64::try_from(offset).map_err(|_| past())?;
                let count = u64::try_from(size).map_err(|_| past())?;
                indices.extend(from..from + count);
            }
            let end = i64::try_from(indices.len()).map_err(|_| past())?;
            if usize::try_from(end).map_or(true, |e| e > self.capacity) {
                return Err(past());
            }
            offsets.push(end);
        }
        let gathered = arrow_select::take::take(values.as_ref(), &UInt64Array::from(indices), None)
            .map_err(|e| Fail::Refused(internal(column, &e)))?;
        self.list(column, received, &offsets, &gathered, item, nulls)
    }

    fn r#struct(
        &self,
        column: &str,
        array: &StructArray,
        fields: &Fields,
    ) -> Converted<StructArray> {
        let columns = array
            .columns()
            .iter()
            .zip(fields)
            .map(|(c, f)| self.array(&child(column, f), c, f.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        StructArray::try_new(fields.clone(), columns, array.nulls().cloned())
            .map_err(|e| Fail::Refused(internal(column, &e)))
    }
}
