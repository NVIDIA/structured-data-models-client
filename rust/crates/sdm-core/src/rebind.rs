// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connection handles stay in hosts; existing source admission stays in the engine.
use crate::run::{Attachment, Attachments, Run, internal};
use sdm_contract::ops::{DescribedSource, HeldSchema, OperationResult};
use sdm_contract::rebind::RebindSource;
use sdm_contract::{CoreError, Detail, HostFacts, Sentence};

pub(crate) fn start(
    root: &crate::compose::CompositionRoot,
    request: RebindSource,
    mut attachments: Attachments,
    facts: HostFacts,
) -> Result<Run, CoreError> {
    if !matches!(
        request.provider,
        sdm_contract::data::ProviderChoice::Sql { .. }
    ) || request.provider != request.previous_provider
        || request.source_kind != request.previous_source_kind
    {
        let reason = "rebinding requires the same backend, namespace, source kind and semantic model/trust settings";
        return Err(CoreError::new(Sentence::SchemaRefinement, reason)
            .with_field("reason", Detail::text(reason)));
    }
    let held: HeldSchema = crate::attached::take_json(&mut attachments, request.schema)?;
    let data = root.data_access(&request.provider)?;
    Ok(Run::with_data(data, move |asker| async move {
        let mut description = crate::describe::initialize(&asker, Some(&held), &facts)
            .await?
            .ok_or_else(|| internal("rebound source was not initialized"))?;
        // Keep durable logical declarations, never refresh-time SQL projections.
        for table in &mut description.tables {
            table.source = held
                .tables
                .iter()
                .find(|t| t.name == table.name)
                .ok_or_else(|| internal("rebinding lost an original table"))?
                .source
                .clone();
        }
        let bytes = serde_json::to_vec(&description)
            .map_err(|_| internal("rebound schema serialization"))?;
        Ok(OperationResult::DescribeSource(DescribedSource {
            description: asker.attach(Attachment::Bytes(bytes))?,
            notes: vec![],
        }))
    }))
}
