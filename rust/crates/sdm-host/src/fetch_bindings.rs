// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-observation witnesses for actual bound fetch owners, without I/O.

use std::collections::{BTreeMap, BTreeSet};

use arrow_schema::SchemaRef;
use sdm_contract::data::{AttachmentId, DataRequest, Dialect, FetchPlan, ProviderChoice, SqlParam};
use sdm_contract::types::Field;
use sdm_contract::{CoreError, Sentence};
use serde_json::{Value, json};

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "fetch binding witness has inconsistent owners",
    )
}

/// Test-only bytes adapter, keeping JSON serialization in the host codec
/// layer while language bindings supply only actual Arrow schemas.
pub fn fetch_binding_bytes(
    request: &[u8],
    plan: &[u8],
    schemas: &BTreeMap<AttachmentId, SchemaRef>,
) -> Result<Vec<u8>, CoreError> {
    let request: DataRequest = serde_json::from_slice(request).map_err(|_| invalid())?;
    let plan: FetchPlan = serde_json::from_slice(plan).map_err(|_| invalid())?;
    let value = json!({
        "owners": fetch_binding_schemas(&plan, schemas)?,
        "witness": fetch_plan_witness(&request, &plan)?,
    });
    serde_json::to_vec(&value)
        .map_err(|_| CoreError::new(Sentence::Internal, "fetch binding witness does not encode"))
}

/// Actual Arrow schemas are read from retained bound owners, not inferred
/// from the request. The plan must name every owner, with the same schema.
pub fn fetch_binding_schemas(
    plan: &FetchPlan,
    schemas: &BTreeMap<AttachmentId, SchemaRef>,
) -> Result<Value, CoreError> {
    let named: BTreeSet<_> = plan.key_sets().into_iter().collect();
    if named != schemas.keys().copied().collect() {
        return Err(invalid());
    }
    let mut owners = Vec::new();
    for (id, schema) in schemas {
        let fields = schema
            .fields()
            .iter()
            .map(|field| Field::from_arrow(field))
            .collect::<Result<Vec<_>, _>>()?;
        if let FetchPlan::Sql { statements, .. } = plan {
            for param in statements.iter().flat_map(|statement| &statement.params) {
                match param {
                    SqlParam::KeySet { keys, .. } if keys.attachment == *id => {
                        if keys.columns != fields {
                            return Err(invalid());
                        }
                    }
                    SqlParam::Text { attachment } if *attachment == *id => {
                        if fields.len() != 1
                            || fields[0].name != "keys"
                            || fields[0].column_type != sdm_contract::types::ColumnType::Utf8
                            || fields[0].nullable
                        {
                            return Err(invalid());
                        }
                    }
                    _ => {}
                }
            }
        }
        let metadata: BTreeMap<_, _> = schema
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        owners.push(json!({"attachment":id,"fields":fields,"metadata":metadata}));
    }
    Ok(Value::Array(owners))
}

/// Only a complete independently rendered Arrow-key plan receives a
/// canonical spelling. Bound text/chunks/projections remain literal SQL.
pub fn fetch_plan_witness(
    request: &DataRequest,
    plan: &FetchPlan,
) -> Result<Option<Value>, CoreError> {
    let FetchPlan::Sql { statements, .. } = plan else {
        return Ok(None);
    };
    let Some(first) = statements.first() else {
        return Ok(None);
    };
    let dialect = first.dialect.name();
    if !matches!(dialect, "sqlite" | "duckdb")
        || statements
            .iter()
            .any(|statement| statement.dialect.name() != dialect)
        || statements
            .iter()
            .flat_map(|statement| &statement.params)
            .any(|param| matches!(param, SqlParam::Text { .. }))
    {
        return Ok(None);
    }
    let provider = ProviderChoice::Sql {
        dialect: Dialect::new(dialect),
        namespace: vec![],
        view: None,
        timestamp_encoding: None,
    };
    let access = crate::default_root().data_access(&provider)?;
    let Ok(rendered) = access.plan(request) else {
        return Ok(None);
    };
    if rendered != *plan {
        return Ok(None);
    }
    let mut canonical = request.clone();
    match &mut canonical {
        DataRequest::Lookup { seeds, .. } | DataRequest::Neighbors { seeds, .. } => {
            seeds.attachment = AttachmentId(0)
        }
        _ => return Ok(None),
    }
    let planned = access.plan(&canonical)?;
    Ok(Some(json!({"issued_request":request,"issued_plan":plan,
        "canonical_request":canonical,"canonical_plan":planned})))
}
