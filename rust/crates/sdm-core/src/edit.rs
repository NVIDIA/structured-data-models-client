// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `edit_schema`'s run (decision E67, T1.24b): one edit to a schema the
//! core described, checked and made by `sdm-schema` as the reference's
//! setters check and make it, over the schema as the host holds it; the
//! schema with the edit made is the run's result, with the warnings the
//! reference gives as it makes it, or the refusal it gives at the
//! assignment. Every host's edit reaches this one function: the Python
//! host's setters and `sdm-testkit`'s edits alike.

use sdm_contract::CoreError;
use sdm_contract::ops::{
    Dtype, HeldSchema, OperationResult, SchemaEdit, SchemaEdited, SourceDescription, Stype,
};

use crate::run::{Asker, Attachment, internal};

/// Whether a column of `dtype` can mean `stype`, as the language answers
/// it (`Stype.supports_dtype`).
pub(crate) fn supports(stype: Stype, dtype: Dtype) -> bool {
    crate::pql::interim_stype(stype).supports_dtype(crate::pql::interim_dtype(dtype))
}

/// The body of an `edit_schema` run.
pub(crate) async fn run(
    asker: Asker,
    described: SourceDescription,
    held: Option<HeldSchema>,
    edit: SchemaEdit,
) -> Result<OperationResult, CoreError> {
    let description = sdm_schema::edit::read_held(described, held, "edit_schema")?;
    let edited = sdm_schema::edit::edit(description, &edit, &supports)?;
    for warning in edited.warnings {
        asker.warn(warning);
    }
    let bytes = serde_json::to_vec(&edited.description)
        .map_err(|_| internal("the schema edited does not write as JSON"))?;
    let schema = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::EditSchema(SchemaEdited {
        schema,
        declared: edited.declared,
    }))
}
