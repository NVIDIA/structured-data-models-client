// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test evidence for actual bound owners, never reconstructed seed data.

use crate::{duck::Fetched, values};
use anyhow::{Result, anyhow, bail};
use sdm_contract::data::{AttachmentId, DataRequest, FetchPlan, SqlParam};
use sdm_contract::run::FetchStep;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Default)]
pub struct State {
    owners: BTreeMap<(u64, u64), Value>,
    fetches: BTreeSet<(u64, u64)>,
}

fn members(value: &Value, names: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name))
    })
}

pub fn observe(
    step: &FetchStep,
    run: usize,
    keys: &BTreeMap<AttachmentId, Fetched>,
) -> Result<Value> {
    if !matches!(step.plan, FetchPlan::Sql { .. }) {
        return Ok(json!({"available":false,"run":run,"fetch":step.id,"owners":[],"witness":null}));
    }
    let schemas = keys
        .iter()
        .map(|(id, value)| (*id, value.schema.clone()))
        .collect();
    let mut owners = sdm_host::fetch_binding_schemas(&step.plan, &schemas)
        .map_err(|_| anyhow!("actual bound seed schema differs from issued plan"))?;
    for owner in owners
        .as_array_mut()
        .ok_or_else(|| anyhow!("no bound owner list"))?
    {
        let id: AttachmentId = serde_json::from_value(owner["attachment"].clone())?;
        let actual = &keys[&id];
        let fields = actual
            .schema
            .fields()
            .iter()
            .map(|field| field.as_ref())
            .collect::<Vec<_>>();
        owner["rows"] = json!(
            actual
                .batches
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>()
        );
        owner["values"] = values::values(&fields, &actual.batches);
    }
    Ok(
        json!({"available":true,"run":run,"fetch":step.id,"owners":owners,
        "witness":sdm_host::fetch_plan_witness(&step.request,&step.plan)
            .map_err(|_|anyhow!("actual fetch plan witness failed"))?}),
    )
}

fn canonical(fetch: &Value, state: &mut State) -> Result<Option<Value>> {
    let Some(facts) = fetch.get("bindings") else {
        return Ok(None);
    };
    if !members(facts, &["available", "run", "fetch", "owners", "witness"]) {
        bail!("fetch binding has no closed issuing observation");
    }
    let run = facts["run"]
        .as_u64()
        .ok_or_else(|| anyhow!("fetch binding has no issuing Run"))?;
    let id = facts["fetch"]
        .as_u64()
        .ok_or_else(|| anyhow!("fetch binding has no issuing reply"))?;
    if id == 0 || u32::try_from(id).is_err() {
        bail!("fetch binding reply is outside its actual ID domain");
    }
    if !state.fetches.insert((run, id)) {
        bail!("a fetch binding repeats its issuing identity");
    }
    if facts["available"] == false {
        if facts["owners"] != json!([]) || !facts["witness"].is_null() {
            bail!("unavailable fetch bindings claim unobserved owners");
        }
        return Ok(None);
    }
    if facts["available"] != true {
        bail!("fetch binding capability is not observed");
    }
    let request: DataRequest = serde_json::from_value(fetch["request"].clone())?;
    let plan: FetchPlan = serde_json::from_value(fetch["plan"].clone())?;
    let ids = plan.key_sets().into_iter().fold(Vec::new(), |mut all, id| {
        if !all.contains(&id) {
            all.push(id);
        }
        all
    });
    let owners = facts["owners"]
        .as_array()
        .ok_or_else(|| anyhow!("fetch bindings have no owner facts"))?;
    if owners.len() != ids.len() {
        bail!("fetch bindings do not cover every bound owner");
    }
    let mut content = BTreeMap::new();
    let mut schemas = BTreeMap::new();
    for owner in owners {
        if !members(
            owner,
            &["attachment", "fields", "metadata", "rows", "values"],
        ) {
            bail!("bound seed owner has an unknown shape");
        }
        let attachment: AttachmentId = serde_json::from_value(owner["attachment"].clone())?;
        if !ids.contains(&attachment) || content.contains_key(&attachment) {
            bail!("foreign or repeated bound seed owner");
        }
        let fields: Vec<sdm_contract::types::Field> =
            serde_json::from_value(owner["fields"].clone())?;
        let metadata: BTreeMap<String, String> = serde_json::from_value(owner["metadata"].clone())?;
        let schema = arrow_schema::Schema::new_with_metadata(
            fields
                .iter()
                .map(|field| field.to_arrow())
                .collect::<Vec<_>>(),
            metadata.into_iter().collect(),
        );
        schemas.insert(attachment, std::sync::Arc::new(schema));
        let value = &owner["values"];
        if !members(
            value,
            &["columns", "types", "ordered", "unordered", "unencodable"],
        ) || owner["rows"].as_u64().is_none()
            || value["unencodable"] != json!([])
            || value["types"] != Value::Null
            || value["columns"] != json!(fields.iter().map(|field| &field.name).collect::<Vec<_>>())
        {
            bail!("bound seed values have no exact complete reading");
        }
        for field in ["ordered", "unordered"] {
            if !value[field].as_str().is_some_and(|text| {
                text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            }) {
                bail!("bound seed values have no digest");
            }
        }
        let mut semantic = owner.clone();
        semantic
            .as_object_mut()
            .ok_or_else(|| anyhow!("invalid seed owner facts"))?
            .remove("attachment");
        if let Some(prior) = state
            .owners
            .insert((run, u64::from(attachment.0)), semantic.clone())
            && prior != semantic
        {
            bail!("one Run attachment changes its observed bound seed content");
        }
        content.insert(attachment, semantic);
    }
    sdm_host::fetch_binding_schemas(&plan, &schemas)
        .map_err(|_| anyhow!("bound schema is not the issued parameter schema"))?;
    if let FetchPlan::Sql { statements, .. } = &plan {
        for param in statements.iter().flat_map(|statement| &statement.params) {
            if let SqlParam::Text { attachment } = param
                && content[attachment]["rows"].as_u64() != Some(1)
            {
                bail!("bound text is not its actual single-row parameter");
            }
        }
    }
    let expected = sdm_host::fetch_plan_witness(&request, &plan)
        .map_err(|_| anyhow!("bound plan witness cannot be checked"))?;
    if facts["witness"] != serde_json::to_value(&expected)? {
        bail!("bound plan witness differs from independent rendering");
    }
    let mut canonical_request = request.clone();
    match &mut canonical_request {
        DataRequest::Lookup { seeds, .. } | DataRequest::Neighbors { seeds, .. } => {
            seeds.attachment = AttachmentId(0)
        }
        _ => {}
    }
    let mut canonical_plan = plan.clone();
    let names = ids
        .iter()
        .enumerate()
        .map(|(at, id)| Ok((*id, AttachmentId(u32::try_from(at)?))))
        .collect::<Result<BTreeMap<_, _>>>()?;
    canonical_plan.rename_attachments(&names);
    let canonical_plan = if let Some(witness) = expected {
        witness["canonical_plan"].clone()
    } else {
        serde_json::to_value(canonical_plan)?
    };
    Ok(Some(
        json!({"request":canonical_request,"plan":canonical_plan,
        "owners":ids.iter().map(|id|&content[id]).collect::<Vec<_>>()}),
    ))
}

pub fn same(left: &Value, right: &Value, a: &mut State, b: &mut State) -> Result<bool> {
    // A refused replay does not consume either issuing identity or content.
    let mut next_a = a.clone();
    let mut next_b = b.clone();
    let one = canonical(left, &mut next_a)?;
    let two = canonical(right, &mut next_b)?;
    let agrees = match (one, two) {
        (Some(one), Some(two)) => one == two,
        _ => left["request"] == right["request"] && left["plan"] == right["plan"],
    };
    if agrees {
        *a = next_a;
        *b = next_b;
    }
    Ok(agrees)
}
