// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context-analysis operation and lifecycle adapter; no host computations.
use crate::run::{Asker, Attachment, Attachments, internal};
use sdm_contract::{CoreError, columns::AnalyzeColumns, ops::OperationResult};

pub(crate) fn run(
    asker: &Asker,
    input: AnalyzeColumns,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    if input.payloads.len() != 1 {
        return Err(internal("column analysis expects one normalized context"));
    }
    let Some(Attachment::Bytes(bytes)) = attached.remove(&input.payloads[0]) else {
        return Err(internal("column analysis payload is not bytes"));
    };
    let payload = crate::predict::payload(&bytes)?;
    let result = sdm_request::columns::analyze(&sdm_request::columns::restored(&payload)?)?;
    crate::direct::value(asker, &result)
}

pub(crate) fn compute(payloads: &[sdm_wire::json::Value]) -> Result<String, CoreError> {
    let restored = payloads
        .iter()
        .map(sdm_request::columns::restored)
        .collect::<Result<Vec<_>, _>>()?;
    let payload = if restored.len() == 1 {
        restored[0].clone()
    } else {
        sdm_request::columns::forecast(&restored)?
    };
    serde_json::to_string(&sdm_request::columns::analyze(&payload)?)
        .map_err(|_| internal("analysis result cannot be encoded"))
}
pub(crate) fn requested(outputs: &[String]) -> bool {
    outputs.iter().any(|o| o == "explanation")
}
pub(crate) fn expose(
    asker: &Asker,
    analysis: Option<&str>,
    outputs: &[String],
) -> Result<Option<String>, CoreError> {
    if !requested(outputs) {
        return Ok(None);
    }
    let Some(raw) = analysis else {
        return Ok(None);
    };
    let analysis = sdm_request::columns::validate(raw)?;
    if let sdm_contract::columns::ColumnAnalysis::Failed { reason }
    | sdm_contract::columns::ColumnAnalysis::Unsupported { reason } = &analysis
    {
        asker.warn(
            sdm_contract::CoreWarning::new(
                sdm_contract::Sentence::ColumnAnalysisUnavailable,
                "column analysis is unavailable; prediction is unaffected",
            )
            .with_field("reason", sdm_contract::Detail::text(reason)),
        );
    }
    Ok(Some(raw.to_owned()))
}
