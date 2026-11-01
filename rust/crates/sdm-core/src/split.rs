// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attachment adapter for request inspection. All decoding policy is shared.
use crate::run::{ArrowData, Asker, Attachment, Attachments, internal};
use sdm_contract::{
    CoreError,
    ops::OperationResult,
    split::{ReadRequestSplit, RelatedTable, RequestSplit},
};

pub fn read(
    asker: &Asker,
    input: ReadRequestSplit,
    attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let bytes = input
        .payloads
        .iter()
        .map(|id| match attached.get(id) {
            Some(Attachment::Bytes(bytes)) => Ok(bytes.as_ref()),
            _ => Err(internal("request split payload attachment is missing")),
        })
        .collect::<Result<Vec<&[u8]>, _>>()?;
    let out = sdm_request::split::read(&bytes, input.half, &input.selection)?;
    let attach = |batch: arrow_array::RecordBatch| {
        asker.attach(Attachment::Arrow(ArrowData::new(
            batch.schema(),
            vec![batch],
        )))
    };
    Ok(OperationResult::RequestSplit(RequestSplit {
        instance: out.instance.map(&attach).transpose()?,
        related_tables: out
            .related
            .into_iter()
            .map(|(name, data)| {
                Ok(RelatedTable {
                    name,
                    data: data.map(&attach).transpose()?,
                })
            })
            .collect::<Result<_, CoreError>>()?,
    }))
}
