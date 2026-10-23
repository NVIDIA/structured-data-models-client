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

/// The body of an `edit_schema` run.
pub(crate) async fn run(
    asker: Asker,
    held: HeldSchema,
    edit: SchemaEdit,
) -> Result<OperationResult, CoreError> {
    // Edits read no data: a column the source was never asked of is edited
    // by its data type, as the reference's setters edit any column.
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let edited = sdm_schema::edit::edit(read.description, &edit, &supports)?;
    for warning in edited.warnings {
        asker.warn(warning);
    }
    let answer = sdm_schema::held::held(&edited.description, &read.unfacted);
    let bytes = serde_json::to_vec(&answer)
        .map_err(|_| internal("the schema edited does not write as JSON"))?;
    let schema = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::EditSchema(SchemaEdited {
        schema,
        declared: edited.declared,
    }))
}
