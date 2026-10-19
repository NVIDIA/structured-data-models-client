// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A column of numbers handed over as their exact decimal text, its field
//! marked [`MARKER`] by the host provider (a Postgres `numeric` declared
//! without a precision, which fixes no Arrow decimal), read as the
//! reference's reads such a column, by its role: a key, the one column of
//! a primary, unique or foreign key the table declares, as that text, so
//! no two keys a double cannot tell apart merge; any other column as a
//! double. The role is settled once, from the preview and the table's
//! declared keys, when the table is described ([`settle`]), and every later
//! fetch of the column is held to the type described ([`conform`]). A
//! source that marks no column is read as before.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::{CatalogKey, KeyKind};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

use crate::ArrowData;

/// The field metadata key a host provider sets on such a column, and its
/// value.
pub const MARKER: &str = sdm_contract::data::EXACT_NUMBER_MARKER;
pub const MARKED: &str = sdm_contract::data::EXACT_NUMBER_MARKED;

/// Whether `field` is a column of numbers handed over as their text.
pub fn is_marked(field: &Field) -> bool {
    field.metadata().get(MARKER).map(String::as_str) == Some(MARKED)
        && field.data_type() == &DataType::Utf8
}

/// Whether `name` is a key of a table declaring `keys`: the one column of
/// a primary, unique or foreign key, as the reference reads a column's
/// role.
fn is_key(name: &str, keys: &[CatalogKey]) -> bool {
    keys.iter().any(|key| {
        matches!(
            key.kind,
            KeyKind::Primary | KeyKind::Unique | KeyKind::Foreign
        ) && key.columns.len() == 1
            && key.columns[0] == name
    })
}

/// `array`, text of numbers, as doubles; a text that is no number is the
/// source's answer not reading, refused by name.
fn doubles(name: &str, array: &ArrayRef) -> Result<ArrayRef, CoreError> {
    let read: Result<Float64Array, CoreError> = array
        .as_string::<i32>()
        .iter()
        .map(|value| {
            value
                .map(|text| {
                    text.parse::<f64>().map_err(|_| {
                        CoreError::new(
                            ErrorKind::DataSource,
                            ErrorCode::ProviderSchemaMismatch,
                            format!("column {name:?} holds a value that is not a number"),
                        )
                        .with_field("column", Detail::text(name))
                    })
                })
                .transpose()
        })
        .collect();
    Ok(Arc::new(read?))
}

/// `field` of the type `to`, its name, nullability and metadata kept.
fn retyped(field: &Field, to: DataType) -> Field {
    Field::new(field.name(), to, field.is_nullable()).with_metadata(field.metadata().clone())
}

/// `preview`, a table declaring `keys`, with each marked column that is no
/// key read as doubles; a key stays its text.
pub fn settle(preview: ArrowData, keys: &[CatalogKey]) -> Result<ArrowData, CoreError> {
    let doubled: Vec<String> = preview
        .schema
        .fields()
        .iter()
        .filter(|f| is_marked(f) && !is_key(f.name(), keys))
        .map(|f| f.name().clone())
        .collect();
    if doubled.is_empty() {
        return Ok(preview);
    }
    let fields: Vec<Field> = preview
        .schema
        .fields()
        .iter()
        .map(|f| {
            if doubled.contains(f.name()) {
                retyped(f, DataType::Float64)
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        preview.schema.metadata().clone(),
    ));
    let batches = preview
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(preview.schema.fields())
                .map(|(array, field)| {
                    if doubled.contains(field.name()) {
                        doubles(field.name(), array)
                    } else {
                        Ok(Arc::clone(array))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(Arc::clone(&schema), columns).map_err(|_| internal())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowData {
        schema,
        batches,
        received: Arc::clone(&preview.received),
    })
}

fn internal() -> CoreError {
    CoreError::new(
        ErrorKind::Internal,
        ErrorCode::InternalError,
        "a column of numbers read as doubles does not fit its batch",
    )
}

/// A fetched column (`field`, `array`) in the type `described` for it: a
/// marked column described as a double read as doubles. `None` where the
/// column is not marked or is described as its text, so it is left as it
/// came.
pub fn conform(
    field: &Field,
    array: &ArrayRef,
    described: &DataType,
) -> Result<Option<(Field, ArrayRef)>, CoreError> {
    if !is_marked(field) || described != &DataType::Float64 {
        return Ok(None);
    }
    Ok(Some((
        retyped(field, DataType::Float64),
        doubles(field.name(), array)?,
    )))
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, StringArray};

    use super::*;

    fn marked(name: &str) -> Field {
        let mut metadata = Field::new(name, DataType::Utf8, true).metadata().clone();
        metadata.insert(MARKER.to_owned(), MARKED.to_owned());
        Field::new(name, DataType::Utf8, true).with_metadata(metadata)
    }

    fn key(kind: KeyKind, column: &str) -> CatalogKey {
        CatalogKey {
            kind,
            columns: vec![column.to_owned()],
            index: None,
        }
    }

    #[test]
    fn a_key_stays_its_text_and_any_other_column_is_read_as_doubles() {
        let schema = Arc::new(Schema::new(vec![
            marked("id"),
            marked("ref"),
            marked("amount"),
            Field::new("note", DataType::Utf8, true),
        ]));
        let text = |values: Vec<Option<&str>>| Arc::new(StringArray::from(values)) as ArrayRef;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                text(vec![Some("100000000000000001"), Some("2")]),
                text(vec![Some("7"), None]),
                text(vec![Some("1.25"), Some("NaN")]),
                text(vec![Some("x"), Some("y")]),
            ],
        )
        .unwrap();
        let preview = ArrowData {
            schema: Arc::clone(&schema),
            batches: vec![batch],
            received: schema,
        };
        let keys = [
            key(KeyKind::Primary, "id"),
            key(KeyKind::Foreign, "ref"),
            CatalogKey {
                kind: KeyKind::Unique,
                columns: vec!["amount".to_owned(), "note".to_owned()],
                index: None,
            },
        ];
        let settled = settle(preview, &keys).unwrap();
        let types: Vec<&DataType> = settled
            .schema
            .fields()
            .iter()
            .map(|f| f.data_type())
            .collect();
        assert_eq!(
            types,
            [
                &DataType::Utf8,
                &DataType::Utf8,
                &DataType::Float64,
                &DataType::Utf8
            ]
        );
        let amounts = settled.batches[0]
            .column(2)
            .as_primitive::<arrow_array::types::Float64Type>();
        assert!((amounts.value(0) - 1.25).abs() < f64::EPSILON);
        assert!(amounts.value(1).is_nan());
        assert_eq!(
            settled.batches[0].column(0).as_string::<i32>().value(0),
            "100000000000000001"
        );
    }

    #[test]
    fn a_later_fetch_follows_the_type_described() {
        let array: ArrayRef = Arc::new(StringArray::from(vec![Some("3.5"), None]));
        let (field, read) = conform(&marked("amount"), &array, &DataType::Float64)
            .unwrap()
            .unwrap();
        assert_eq!(field.data_type(), &DataType::Float64);
        assert!(read.is_null(1));
        assert!(
            conform(&marked("id"), &array, &DataType::Utf8)
                .unwrap()
                .is_none()
        );
        let plain = Field::new("amount", DataType::Utf8, true);
        assert!(
            conform(&plain, &array, &DataType::Float64)
                .unwrap()
                .is_none()
        );
        let bad: ArrayRef = Arc::new(StringArray::from(vec![Some("seven")]));
        let refused = conform(&marked("amount"), &bad, &DataType::Float64).unwrap_err();
        assert_eq!(refused.code, ErrorCode::ProviderSchemaMismatch);
    }
}
