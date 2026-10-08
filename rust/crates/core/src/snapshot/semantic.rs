// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use arrow_array::{Array, LargeStringArray, StringArray, StringViewArray};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    Error, Result, array, decode_typed, encode_typed, field, fields, object, require, text,
    typed_observation, unique_json,
};

fn source_inputs(
    input: &Value,
    external: &Value,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<()> {
    let given = decode_typed(field(input, "generator_input")?)?;
    fields(&given, &["definition", "tables", "trust_expressions"])?;
    let definition = text(field(&given, "definition")?)?;
    let trust = field(&given, "trust_expressions")?
        .as_bool()
        .ok_or(Error("invalid semantic expression trust"))?;
    for table in array(field(&given, "tables")?)? {
        fields(table, &["identity", "columns"])?;
        let identity = array(field(table, "identity")?)?;
        require(
            identity.len() == 3 && identity.iter().all(Value::is_string),
            "invalid semantic table identity",
        )?;
        require(
            object(field(table, "columns")?)?
                .values()
                .all(Value::is_string),
            "invalid semantic table columns",
        )?;
    }
    let initial = field(external, "initial_schema")?;
    fields(initial, &["function", "args"])?;
    let arguments = array(field(initial, "args")?)?;
    require(
        field(initial, "function")? == "schema_source_run"
            && arguments.len() == 3
            && arguments[0] == "snow",
        "invalid original semantic source call",
    )?;
    let source = unique_json(text(&arguments[1])?.as_bytes())?;
    let options = unique_json(text(&arguments[2])?.as_bytes())?;
    require(
        source
            == json!({"backend":"snow", "namespace":{"database":"TEST", "schema":"PARITY_SHOP"}, "semantic_model":"SV", "trust_expressions":trust})
            && options
                == json!({"tables":null, "primary_keys":null, "time_columns":null, "relationships":null, "check_keys":true}),
        "semantic source arguments differ from original public inputs",
    )?;
    let first = array(field(external, "queries")?)?
        .first()
        .ok_or(Error("missing semantic definition query"))?;
    let outcome = field(first, "outcome")?;
    if let Some(data) = outcome.get("data") {
        let decoded = crate::public_replay::decode(data, blobs, directory)
            .map_err(|_| Error("invalid semantic definition Arrow input"))?;
        let batch = match decoded {
            crate::plain::Plain::Table(schema, batches) => {
                arrow_select::concat::concat_batches(&schema, &batches)
                    .map_err(|_| Error("invalid semantic definition table"))?
            }
            crate::plain::Plain::Batch(batch) => batch,
            _ => return Err(Error("semantic definition requires an Arrow table")),
        };
        require(
            batch.num_rows() > 0 && batch.num_columns() > 0,
            "missing semantic definition value",
        )?;
        let values = batch.column(0);
        require(!values.is_null(0), "missing semantic definition value")?;
        let found = if let Some(values) = values.as_any().downcast_ref::<StringArray>() {
            values.value(0)
        } else if let Some(values) = values.as_any().downcast_ref::<LargeStringArray>() {
            values.value(0)
        } else if let Some(values) = values.as_any().downcast_ref::<StringViewArray>() {
            values.value(0)
        } else {
            return Err(Error("semantic definition is not text"));
        };
        require(
            found == definition,
            "semantic definition differs from original generator input",
        )?;
    }
    Ok(())
}

pub(super) fn replay(
    input: &Value,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<Value> {
    fields(input, &["generator", "generator_input", "host_inputs"])?;
    let coordinate = field(input, "generator")?;
    fields(coordinate, &["name", "stream_seed", "ordinal"])?;
    require(
        field(coordinate, "name")? == "schema.semantic_views"
            && field(coordinate, "stream_seed")?.as_u64().is_some()
            && field(coordinate, "ordinal")?.as_u64().is_some(),
        "invalid semantic generator coordinate",
    )?;
    let facts = field(input, "host_inputs")?;
    fields(facts, &["host", "external"])?;
    for (name, bytes) in blobs {
        let prefix = format!("{directory}/tables/");
        let digest = name
            .strip_prefix(&prefix)
            .and_then(|name| name.strip_suffix(".arrow"))
            .ok_or(Error("semantic Arrow input has another owner"))?;
        require(
            format!("{:x}", Sha256::digest(bytes)) == digest,
            "semantic Arrow input digest differs",
        )?;
    }
    let host = super::host::descriptor(field(facts, "host")?)?;
    let external = field(facts, "external")?;
    fields(
        external,
        &["original_input_sha256", "initial_schema", "queries"],
    )?;
    let mut original = field(input, "generator_input")?.clone();
    original.sort_all_objects();
    let original = serde_json::to_vec(&original)
        .map_err(|_| Error("invalid original semantic generator input"))?;
    require(
        field(external, "original_input_sha256")?
            == &json!(format!("{:x}", Sha256::digest(original))),
        "semantic external inputs belong to another generator input",
    )?;
    source_inputs(input, external, blobs, directory)?;
    let external = json!({"initial_schema": field(external, "initial_schema")?, "queries": field(external, "queries")?});
    let (observation, queries) = crate::semantic_replay::replay(&external, host, blobs, directory)
        .map_err(|_| Error("semantic replay refused its input"))?;
    let queries = array(&queries)?.iter().map(|query| {
        Ok(json!({"tuple": array(query)?.iter().cloned().map(encode_typed).collect::<Result<Vec<_>>>()?}))
    }).collect::<Result<Vec<_>>>()?;
    typed_observation(
        json!({"tuple": [encode_typed(observation)?, {"list": queries}]}),
        Value::Null,
    )
}
