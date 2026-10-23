// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `edit_schema`'s run (decision E67, T1.24b): one edit to a schema,
//! checked and made by `sdm-schema` as the reference's setters check and
//! make it, over the schema as the host holds it, as plain data (T3.16);
//! the schema with the edit made, as the host then holds it, is the run's
//! result, with the warnings the reference gives as it makes it, or the
//! refusal it gives at the assignment. Every host's edit reaches this one function: the Python
//! host's setters and `sdm-testkit`'s edits alike.

use sdm_contract::CoreError;
use std::collections::BTreeMap;

use sdm_contract::ops::{Dtype, HeldSchema, OperationResult, SchemaEdit, SchemaEdited, Stype};

use crate::run::{Asker, Attachment, internal};

/// Whether a column of `dtype` can mean `stype`, as the language answers
/// it (`Stype.supports_dtype`).
pub(crate) fn supports(stype: Stype, dtype: Dtype) -> bool {
    crate::pql::parser_stype(stype).supports_dtype(crate::pql::parser_dtype(dtype))
}

/// Every data type name a host may give a column, and the contract's name
/// of the data type it names (`sdm_schema::held::DTYPE_NAMES`): the one
/// table of them, which the Python client reads for its own checks (T3.16).
pub fn dtype_names() -> Vec<(&'static str, String)> {
    sdm_schema::held::DTYPE_NAMES
        .iter()
        .map(|(name, dtype)| {
            let named = serde_json::to_value(dtype).unwrap_or_default();
            (*name, named.as_str().unwrap_or_default().to_owned())
        })
        .collect()
}

/// Every meaning a column of each data type can carry, as the language
/// answers it ([`supports`]), each pair by the contract's names (`Stype`'s
/// and `Dtype`'s): plain data a host reads in place of the old PQL
/// package's `Stype.supports_dtype` (T3.16, the structure design's rule
/// 11).
pub fn meanings() -> Vec<(String, String)> {
    const STYPES: [Stype; 9] = [
        Stype::Numerical,
        Stype::Categorical,
        Stype::Multicategorical,
        Stype::Id,
        Stype::Text,
        Stype::Timestamp,
        Stype::Sequence,
        Stype::Image,
        Stype::Unsupported,
    ];
    const DTYPES: [Dtype; 18] = [
        Dtype::Bool,
        Dtype::Int,
        Dtype::Byte,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float,
        Dtype::Float32,
        Dtype::Float64,
        Dtype::String,
        Dtype::Binary,
        Dtype::Date,
        Dtype::Time,
        Dtype::Timedelta,
        Dtype::Floatlist,
        Dtype::Intlist,
        Dtype::Stringlist,
        Dtype::Unsupported,
    ];
    // Every variant listed: a new one fails these matches, so the arrays
    // above are held to the enums by the compiler (Claude r2 P3-2).
    let _every_stype = |stype: Stype| match stype {
        Stype::Numerical
        | Stype::Categorical
        | Stype::Multicategorical
        | Stype::Id
        | Stype::Text
        | Stype::Timestamp
        | Stype::Sequence
        | Stype::Image
        | Stype::Unsupported => (),
    };
    let _every_dtype = |dtype: Dtype| match dtype {
        Dtype::Bool
        | Dtype::Int
        | Dtype::Byte
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Float
        | Dtype::Float32
        | Dtype::Float64
        | Dtype::String
        | Dtype::Binary
        | Dtype::Date
        | Dtype::Time
        | Dtype::Timedelta
        | Dtype::Floatlist
        | Dtype::Intlist
        | Dtype::Stringlist
        | Dtype::Unsupported => (),
    };
    let name = |value: serde_json::Value| value.as_str().unwrap_or_default().to_owned();
    let mut pairs = Vec::new();
    for stype in STYPES {
        for dtype in DTYPES {
            if supports(stype, dtype) {
                pairs.push((
                    name(serde_json::to_value(stype).unwrap_or_default()),
                    name(serde_json::to_value(dtype).unwrap_or_default()),
                ));
            }
        }
    }
    pairs
}

/// The body of an `edit_schema` run.
pub(crate) async fn run(
    asker: Asker,
    held: HeldSchema,
    edit: SchemaEdit,
) -> Result<OperationResult, CoreError> {
    // Edits read no data: a column the source was never asked of is edited
    // by its data type, as the reference's setters edit any column.
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let edited = sdm_schema::edit::edit_named(read.description, &edit, &supports, &read.types)?;
    for warning in edited.warnings {
        asker.warn(warning);
    }
    let answer = sdm_schema::held::held(&edited.description, &read.unfacted, &read.retyped);
    let bytes = serde_json::to_vec(&answer)
        .map_err(|_| internal("the schema edited does not write as JSON"))?;
    let schema = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::EditSchema(SchemaEdited {
        schema,
        declared: edited.declared,
    }))
}
