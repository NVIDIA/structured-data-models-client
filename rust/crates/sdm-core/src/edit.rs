// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `edit_schema`'s run: one edit to a schema,
//! checked and made by `sdm-schema` as the reference's setters check and
//! make it, over the schema as the host holds it, as plain data;
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
/// table of them, which the Python client reads for its own checks.
pub fn dtype_names() -> Vec<(&'static str, String)> {
    sdm_schema::held::DTYPE_NAMES
        .iter()
        .map(|(name, dtype)| {
            let named = serde_json::to_value(dtype).unwrap_or_default();
            (*name, named.as_str().unwrap_or_default().to_owned())
        })
        .collect()
}

/// Every data type the contract names, in its order.
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

/// Every data type's name on the client's wire ([`Dtype::wire_name`]), by
/// the contract's name of it: the one table of them, which a host reads to
/// name a described column's type as the schema holds it.
pub fn wire_names() -> Vec<(String, &'static str)> {
    DTYPES
        .iter()
        .map(|&dtype| {
            let named = serde_json::to_value(dtype).unwrap_or_default();
            (
                named.as_str().unwrap_or_default().to_owned(),
                dtype.wire_name(),
            )
        })
        .collect()
}

/// Every meaning a column of each data type can carry, as the language
/// answers it ([`supports`]), each pair by the contract's names (`Stype`'s
/// and `Dtype`'s): plain data a host reads in place of the old PQL
/// package's `Stype.supports_dtype` (the structure design's rule
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
    context: Option<sdm_contract::ops::SchemaEditContext>,
    frame: Option<crate::ArrowData>,
) -> Result<OperationResult, CoreError> {
    // Source edits perform no reads. Frame edits may receive a whole Arrow table
    // or native observations, but never ask the host to execute schema policy.
    if frame.is_some() && (context.is_none() || !matches!(&edit, SchemaEdit::PrimaryKey { .. })) {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "Arrow edit data requires a frame primary-key edit",
        ));
    }
    if frame.is_some()
        && let Some(sdm_contract::ops::SchemaEditContext::Frames { key_counts }) = &context
        && !key_counts.is_empty()
    {
        return Err(CoreError::protocol(
            sdm_contract::Sentence::InvalidMessage,
            "frame edit provides both Arrow data and native counts",
        ));
    }
    let read = if matches!(
        &context,
        Some(sdm_contract::ops::SchemaEditContext::Frames { .. })
    ) {
        sdm_schema::held::read_frame(held, true)?
    } else {
        sdm_schema::held::read(held, &BTreeMap::new(), true)?
    };
    let mut edited =
        sdm_schema::edit::edit(read.description.clone(), &edit, &supports, &read.types)?;
    if let Some(sdm_contract::ops::SchemaEditContext::Frames { key_counts }) = context
        && let SchemaEdit::PrimaryKey { table, .. } = &edit
    {
        let shape = edited
            .description
            .tables
            .iter_mut()
            .find(|shape| &shape.name == table)
            .ok_or_else(|| internal("edited frame table is absent"))?;
        if !shape.primary_key.is_empty() {
            let observed;
            let count = if let Some(frame) = frame {
                let frame = crate::arrow_logical::logical(frame)?;
                observed = sdm_schema::keys::count_frame(
                    &frame.schema,
                    &frame.batches,
                    &shape.primary_key,
                )?;
                &observed
            } else {
                key_counts
                    .get(table)
                    .ok_or_else(|| internal("frame key edit lacks fresh observations"))?
            };
            edited
                .warnings
                .extend(sdm_schema::keys::hold_frame(shape, count)?);
        }
    }
    for warning in edited.warnings {
        asker.warn(warning);
    }
    let answer = read.hold(&edited.description);
    let bytes = serde_json::to_vec(&answer)
        .map_err(|_| internal("the schema edited does not write as JSON"))?;
    let schema = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::EditSchema(SchemaEdited {
        schema,
        declared: edited.declared,
    }))
}
