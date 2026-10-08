// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow admission (decision B-D6): the one table that turns the Arrow forms
//! a host may hand over into the closed type set, without loss, or refuses
//! them by name.
//!
//! | Received | Admitted as |
//! |---|---|
//! | `large_utf8` | `utf8` |
//! | `large_binary` | `binary` |
//! | `timestamp[s]`, `timestamp[ms]`, with or without a zone | `timestamp[us]`, the same zone |
//! | `large_list<T>`, `fixed_size_list<T, n>` | `list<T>` |
//! | `dictionary<K, large_utf8>` | `dictionary<K, utf8>` |
//!
//! Nested types are admitted child by child, keeping each field's name,
//! nullability and metadata, and every value and null. Anything the table
//! does not take into the set is refused with `UNSUPPORTED_TYPE`, naming the
//! column (with its path through nested types) and the Arrow type received;
//! a value the admitted type cannot hold, such as a `timestamp[s]` past what
//! an int64 count of microseconds reaches or a `large_list` past an int32
//! offset, is refused with `VALUE_OUT_OF_RANGE`, never cut.
//!
//! Native-to-Arrow conversion (pandas dtypes, JDBC types) stays in each host.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{TimestampMillisecondType, TimestampSecondType};
use arrow_array::{
    Array, ArrayRef, BinaryArray, ListArray, MapArray, RecordBatch, StringArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef, TimeUnit};
use sdm_contract::{ColumnType, CoreError, Detail, ErrorKind, codes};
use sdm_core::ArrowData;

/// The closed-set type `received` is admitted as, or its refusal by name.
pub fn admitted_type(column: &str, received: &DataType) -> Result<DataType, CoreError> {
    let admitted = convert_type(column, received);
    // A part the table leaves as it is and the set does not hold is
    // refused here, named by its path and the type received there.
    ColumnType::from_arrow(column, &admitted)?;
    Ok(admitted)
}

fn convert_field(parent: &str, field: &Field) -> Field {
    let path = format!("{parent}.{}", field.name());
    Field::new(
        field.name(),
        convert_type(&path, field.data_type()),
        field.is_nullable(),
    )
    .with_metadata(field.metadata().clone())
}

fn convert_type(column: &str, received: &DataType) -> DataType {
    match received {
        DataType::LargeUtf8 => DataType::Utf8,
        DataType::LargeBinary => DataType::Binary,
        DataType::Timestamp(TimeUnit::Second | TimeUnit::Millisecond, zone) => {
            DataType::Timestamp(TimeUnit::Microsecond, zone.clone())
        }
        DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
            DataType::List(Arc::new(convert_field(column, item)))
        }
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| convert_field(column, f))
                .collect::<Fields>(),
        ),
        DataType::Map(entries, sorted) => {
            DataType::Map(Arc::new(convert_field(column, entries)), *sorted)
        }
        DataType::Dictionary(index, values) if **values == DataType::LargeUtf8 => {
            DataType::Dictionary(index.clone(), Box::new(DataType::Utf8))
        }
        other => other.clone(),
    }
}

/// The refusal of a value the admitted type cannot hold.
fn out_of_range(column: &str, received: &DataType, admitted: &DataType) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        codes::VALUE_OUT_OF_RANGE,
        format!(
            "column {column:?} holds a value of Arrow type {received} that {admitted} cannot \
             hold"
        ),
    )
    .with_field("column", Detail::text(column))
    .with_field("arrow_type", Detail::text(received.to_string()))
    .with_field("admitted_type", Detail::text(admitted.to_string()))
}

/// Admits `data`: its schema, then every batch, column by column.
pub fn admit(data: &ArrowData) -> Result<ArrowData, CoreError> {
    let fields: Vec<Field> = data
        .schema
        .fields()
        .iter()
        .map(|f| {
            Ok(Field::new(
                f.name(),
                admitted_type(f.name(), f.data_type())?,
                f.is_nullable(),
            )
            .with_metadata(f.metadata().clone()))
        })
        .collect::<Result<_, CoreError>>()?;
    let schema: SchemaRef = Arc::new(Schema::new_with_metadata(
        fields,
        data.schema.metadata().clone(),
    ));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(data.schema.fields())
                .zip(schema.fields())
                .map(|((array, received), admitted)| {
                    convert_array(received.name(), array, admitted.data_type())
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(Arc::clone(&schema), columns)
                .map_err(|e| invalid_data(&format!("a batch does not fit its schema: {e}")))
        })
        .collect::<Result<_, CoreError>>()?;
    Ok(ArrowData { schema, batches })
}

fn invalid_data(message: &str) -> CoreError {
    CoreError::protocol(codes::INVALID_MESSAGE, message)
}

/// `array`, of a type [`admitted_type`] took, as `admitted`.
fn convert_array(
    column: &str,
    array: &ArrayRef,
    admitted: &DataType,
) -> Result<ArrayRef, CoreError> {
    let received = array.data_type();
    if received == admitted {
        return Ok(Arc::clone(array));
    }
    let overflow = || out_of_range(column, received, admitted);
    let rebuilt = |e: arrow_schema::ArrowError| {
        CoreError::new(
            ErrorKind::Internal,
            codes::INTERNAL_ERROR,
            format!("admitting column {column:?}: {e}"),
        )
    };
    Ok(match (received, admitted) {
        (DataType::LargeUtf8, DataType::Utf8) => {
            let large = array.as_string::<i64>();
            let offsets = narrow_offsets(large.offsets()).ok_or_else(overflow)?;
            Arc::new(
                StringArray::try_new(offsets, large.values().clone(), large.nulls().cloned())
                    .map_err(rebuilt)?,
            )
        }
        (DataType::LargeBinary, DataType::Binary) => {
            let large = array.as_binary::<i64>();
            let offsets = narrow_offsets(large.offsets()).ok_or_else(overflow)?;
            Arc::new(
                BinaryArray::try_new(offsets, large.values().clone(), large.nulls().cloned())
                    .map_err(rebuilt)?,
            )
        }
        (DataType::Timestamp(unit, zone), DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            let (values, nulls, factor) = match unit {
                TimeUnit::Second => {
                    let a = array.as_primitive::<TimestampSecondType>();
                    (a.values().clone(), a.nulls().cloned(), 1_000_000)
                }
                TimeUnit::Millisecond => {
                    let a = array.as_primitive::<TimestampMillisecondType>();
                    (a.values().clone(), a.nulls().cloned(), 1_000)
                }
                _ => {
                    return Err(rebuilt(arrow_schema::ArrowError::CastError(
                        unit.to_string(),
                    )));
                }
            };
            // A null slot's value is not a value: it is written as 0.
            let scaled = values
                .iter()
                .enumerate()
                .map(|(i, v)| match &nulls {
                    Some(n) if n.is_null(i) => Some(0),
                    _ => v.checked_mul(factor),
                })
                .collect::<Option<Vec<i64>>>()
                .ok_or_else(overflow)?;
            Arc::new(
                TimestampMicrosecondArray::try_new(ScalarBuffer::from(scaled), nulls)
                    .map_err(rebuilt)?
                    .with_timezone_opt(zone.clone()),
            )
        }
        (DataType::List(_), DataType::List(item)) => {
            let list = array.as_list::<i32>();
            let values = convert_array(&child(column, item), list.values(), item.data_type())?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(item),
                    list.offsets().clone(),
                    values,
                    list.nulls().cloned(),
                )
                .map_err(rebuilt)?,
            )
        }
        (DataType::LargeList(_), DataType::List(item)) => {
            let list = array.as_list::<i64>();
            let offsets = narrow_offsets(list.offsets()).ok_or_else(overflow)?;
            let values = convert_array(&child(column, item), list.values(), item.data_type())?;
            Arc::new(
                ListArray::try_new(Arc::clone(item), offsets, values, list.nulls().cloned())
                    .map_err(rebuilt)?,
            )
        }
        (DataType::FixedSizeList(_, size), DataType::List(item)) => {
            let list = array.as_fixed_size_list();
            let offsets = (0..=list.len())
                .map(|i| i32::try_from(i).ok().and_then(|i| i.checked_mul(*size)))
                .collect::<Option<Vec<i32>>>()
                .ok_or_else(overflow)?;
            let values = convert_array(&child(column, item), list.values(), item.data_type())?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(item),
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    values,
                    list.nulls().cloned(),
                )
                .map_err(rebuilt)?,
            )
        }
        (DataType::Struct(_), DataType::Struct(fields)) => {
            Arc::new(convert_struct(column, array.as_struct(), fields)?)
        }
        (DataType::Map(_, _), DataType::Map(entries, sorted)) => {
            let map = array.as_map();
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(rebuilt(arrow_schema::ArrowError::CastError(
                    "a map's entries are a struct".into(),
                )));
            };
            let path = child(column, entries);
            let converted = convert_struct(&path, map.entries(), fields)?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    map.offsets().clone(),
                    converted,
                    map.nulls().cloned(),
                    *sorted,
                )
                .map_err(rebuilt)?,
            )
        }
        (DataType::Dictionary(_, _), DataType::Dictionary(_, values)) => {
            let dictionary = array.as_any_dictionary();
            let converted = convert_array(column, dictionary.values(), values)?;
            dictionary.with_values(converted)
        }
        _ => {
            return Err(rebuilt(arrow_schema::ArrowError::CastError(format!(
                "no admission from {received} to {admitted}"
            ))));
        }
    })
}

fn child(column: &str, field: &FieldRef) -> String {
    format!("{column}.{}", field.name())
}

fn convert_struct(
    column: &str,
    array: &StructArray,
    fields: &Fields,
) -> Result<StructArray, CoreError> {
    let columns = array
        .columns()
        .iter()
        .zip(fields)
        .map(|(c, f)| convert_array(&child(column, f), c, f.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    StructArray::try_new(fields.clone(), columns, array.nulls().cloned()).map_err(|e| {
        CoreError::new(
            ErrorKind::Internal,
            codes::INTERNAL_ERROR,
            format!("admitting column {column:?}: {e}"),
        )
    })
}

/// int64 offsets as int32, if every one fits.
fn narrow_offsets(offsets: &OffsetBuffer<i64>) -> Option<OffsetBuffer<i32>> {
    let narrow = offsets
        .iter()
        .map(|o| i32::try_from(*o).ok())
        .collect::<Option<Vec<i32>>>()?;
    // The offsets of a valid array rise; narrowing keeps their order.
    Some(OffsetBuffer::new(ScalarBuffer::from(narrow)))
}
