// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One big-integer tier per column (register P1B-43; the orchestrator's
//! decision of 2026-10-14): a HUGEINT, UHUGEINT or VARINT column the host
//! layer read (`sdm-host`'s `extensions.rs`) arrives in the tier its fetch's
//! values need, `int64`, else `decimal128(38, 0)`, else text, its field
//! marked [`MARKER`]. The tier is decided for the whole column, never per
//! fetch: `describe_source` widens the preview to the tier of the column's
//! smallest and largest values over the whole table, and every later fetch
//! is widened to the tier described ([`conform`]), or refused by name
//! (`VALUE_OUT_OF_RANGE`) where it holds a value past it, never delivered
//! in another tier.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{Array, ArrayRef, Decimal128Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::{CoreError, Detail, Sentence};

use crate::ArrowData;

/// The field metadata key the host layer sets on a big-integer column it
/// read, and its value.
pub const MARKER: &str = sdm_contract::data::BIG_INTEGER_MARKER;
pub const MARKED: &str = sdm_contract::data::BIG_INTEGER_MARKED;

/// Whether `field` is a big-integer column the host layer read.
pub fn is_marked(field: &Field) -> bool {
    field.metadata().get(MARKER).map(String::as_str) == Some(MARKED)
}

/// Where `data_type` stands among the tiers, narrowest first, if it is one.
fn rank(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int64 => Some(0),
        DataType::Decimal128(38, 0) => Some(1),
        DataType::Utf8 => Some(2),
        _ => None,
    }
}

/// The wider of two tiers; `None` if either is none.
pub fn wider(a: &DataType, b: &DataType) -> Option<DataType> {
    Some(if rank(a)? >= rank(b)? { a } else { b }.clone())
}

fn internal(message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}

/// `array`, of a tier, in the tier `to`, no narrower; `None` where `to` is
/// narrower than the array's tier or either is no tier.
fn widen(array: &ArrayRef, to: &DataType) -> Option<ArrayRef> {
    let from = array.data_type();
    if rank(from)? > rank(to)? {
        return None;
    }
    if from == to {
        return Some(Arc::clone(array));
    }
    Some(match (from, to) {
        (DataType::Int64, DataType::Decimal128(..)) => Arc::new(
            array
                .as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(i128::from))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(38, 0)
                .ok()?,
        ),
        (DataType::Int64, DataType::Utf8) => Arc::new(
            array
                .as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(|v| v.to_string()))
                .collect::<StringArray>(),
        ),
        (DataType::Decimal128(..), DataType::Utf8) => Arc::new(
            array
                .as_primitive::<Decimal128Type>()
                .iter()
                .map(|v| v.map(|v| v.to_string()))
                .collect::<StringArray>(),
        ),
        _ => return None,
    })
}

/// `field` of the type `to`, its name, nullability and metadata kept.
fn retyped(field: &Field, to: &DataType) -> Field {
    Field::new(field.name(), to.clone(), field.is_nullable())
        .with_metadata(field.metadata().clone())
}

/// `data` with each column `tiers` names widened to its tier.
pub fn widen_data(data: &ArrowData, tiers: &[(String, DataType)]) -> Result<ArrowData, CoreError> {
    let tier = |name: &str| tiers.iter().find(|(n, _)| n == name).map(|(_, t)| t);
    let fields: Vec<Field> = data
        .schema
        .fields()
        .iter()
        .map(|f| match tier(f.name()) {
            Some(to) => retyped(f, to),
            None => f.as_ref().clone(),
        })
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(
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
                .map(|(array, field)| match tier(field.name()) {
                    Some(to) => widen(array, to)
                        .ok_or_else(|| internal("a big integer's tier does not widen")),
                    None => Ok(Arc::clone(array)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(Arc::clone(&schema), columns)
                .map_err(|_| internal("a widened big integer does not fit its batch"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowData {
        schema,
        batches,
        received: Arc::clone(&data.received),
    })
}

/// A fetched big-integer column (`field`, `array`) in the tier `described`
/// for it: widened where it is narrower, refused by name
/// (`VALUE_OUT_OF_RANGE`) where it holds a value past it. `None` where the
/// column is not a marked big integer or `described` is no tier, so it is
/// left as it came.
pub fn conform(
    field: &Field,
    array: &ArrayRef,
    described: &DataType,
) -> Result<Option<(Field, ArrayRef)>, CoreError> {
    if !is_marked(field) || rank(described).is_none() || rank(array.data_type()).is_none() {
        return Ok(None);
    }
    match widen(array, described) {
        Some(widened) => Ok(Some((retyped(field, described), widened))),
        None => Err(CoreError::new(
            Sentence::BigIntegerPastType,
            format!(
                "column {:?} holds a value that the type it was described as, {described}, \
                 cannot hold",
                field.name()
            ),
        )
        .with_field("column", Detail::text(field.name()))
        .with_field("arrow_type", Detail::text(array.data_type().to_string()))
        .with_field("admitted_type", Detail::text(described.to_string()))),
    }
}

/// The tier of `field`'s values in a whole-table answer: its own where it
/// is a marked big integer of a tier.
pub fn tier(field: &Field) -> Option<DataType> {
    (is_marked(field) && rank(field.data_type()).is_some()).then(|| field.data_type().clone())
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use sdm_contract::ErrorCode;

    use super::*;

    const DECIMAL: DataType = DataType::Decimal128(38, 0);

    fn marked(name: &str, data_type: DataType) -> Field {
        let mut metadata = Field::new(name, DataType::Null, true).metadata().clone();
        metadata.insert(MARKER.to_owned(), MARKED.to_owned());
        Field::new(name, data_type, true).with_metadata(metadata)
    }

    #[test]
    fn a_narrower_fetch_widens_and_a_wider_one_is_refused() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![Some(-7), None]));
        let field = marked("f", DataType::Int64);
        let (as_decimal, widened) = conform(&field, &ints, &DataType::Decimal128(38, 0))
            .unwrap()
            .unwrap();
        assert_eq!(as_decimal.data_type(), &DataType::Decimal128(38, 0));
        assert_eq!(widened.as_primitive::<Decimal128Type>().value(0), -7);
        assert!(widened.is_null(1));
        let (_, text) = conform(&field, &ints, &DataType::Utf8).unwrap().unwrap();
        assert_eq!(text.as_string::<i32>().value(0), "-7");
        let wide: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(10i128.pow(30))])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let refused = conform(&marked("f", DECIMAL), &wide, &DataType::Int64).unwrap_err();
        assert_eq!(refused.code, ErrorCode::ValueOutOfRange);
        let (_, text) = conform(&marked("f", DECIMAL), &wide, &DataType::Utf8)
            .unwrap()
            .unwrap();
        assert_eq!(
            text.as_string::<i32>().value(0),
            format!("1{}", "0".repeat(30))
        );
        // Unmarked, or described as no tier: left as it came.
        let plain = Field::new("f", DataType::Int64, true);
        assert!(conform(&plain, &ints, &DataType::Utf8).unwrap().is_none());
        assert!(
            conform(&field, &ints, &DataType::Float64)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn the_wider_tier_is_taken() {
        assert_eq!(wider(&DataType::Int64, &DECIMAL), Some(DECIMAL));
        assert_eq!(wider(&DataType::Utf8, &DECIMAL), Some(DataType::Utf8));
        assert_eq!(wider(&DataType::Int64, &DataType::Float64), None);
    }
}
