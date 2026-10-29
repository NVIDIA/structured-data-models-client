// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned relational response data and its completed native attribution.

use crate::attached::take_json;
use crate::prepare::invalid;
use crate::run::{Asker, Attachment, Attachments};
use sdm_contract::CoreError;
use sdm_contract::data::AttachmentId;
use sdm_contract::prepare::relational_response::*;
use sdm_request::predict::{Part, relational_response as response_rows};
use sdm_wire::json::{self, Value};
use sdm_wire::response::PredictionResponse;

fn encoded<T: serde::Serialize>(value: &T) -> Result<Attachment, CoreError> {
    serde_json::to_vec(value)
        .map(Attachment::Bytes)
        .map_err(|_| invalid("relational response data cannot serialize"))
}

fn bytes(attached: &mut Attachments, id: AttachmentId) -> Result<Vec<u8>, CoreError> {
    match attached.remove(&id) {
        Some(Attachment::Bytes(bytes)) => Ok(bytes),
        _ => Err(invalid("relational response owner is not attached bytes")),
    }
}

/// Called immediately after the model operation's complete generated decode.
/// No mutable request field is inspected or snapshotted here.
pub(crate) fn decode(
    asker: &Asker,
    value: Value,
    response: &PredictionResponse,
    call_request_id: &str,
) -> Result<RelationalResponseData, CoreError> {
    let (scope, rows) = response_rows::decode(response, call_request_id)?;
    Ok(RelationalResponseData {
        scope,
        response: asker.attach(Attachment::Bytes(value.to_bytes()))?,
        rows: asker.attach(encoded(&rows)?)?,
        entity_reference_prefix: response_rows::reference_prefix().to_owned(),
        forecast_kind: "forecasting".to_owned(),
    })
}

fn decoded(
    data: &RelationalResponseData,
    attached: &mut Attachments,
) -> Result<(Value, PredictionResponse, Vec<DecodedRelationalRow>), CoreError> {
    if data.entity_reference_prefix != response_rows::reference_prefix()
        || data.forecast_kind != "forecasting"
    {
        return Err(invalid(
            "relational response names another native table meaning",
        ));
    }
    let value = json::read_answer(&bytes(attached, data.response)?)
        .map_err(|_| invalid("owned relational response is unreadable"))?;
    let response = PredictionResponse::read(&value)
        .map_err(|_| invalid("owned relational response has another generated shape"))?;
    let rows: Vec<DecodedRelationalRow> = take_json(attached, data.rows)?;
    let (scope, actual) = response_rows::decode(&response, &data.scope.call_request_id)?;
    if scope != data.scope || actual != rows {
        return Err(invalid(
            "relational response owners disagree with decoded rows",
        ));
    }
    Ok((value, response, rows))
}

/// Reached after native per-row matching and before target dtype reads.
pub(crate) fn coverage(
    data: RelationalResponseData,
    attribution: RelationalAttribution,
    rows: AttachmentId,
    mut attached: Attachments,
) -> Result<RelationalAttribution, CoreError> {
    let (_, _, decoded) = decoded(&data, &mut attached)?;
    let accepted: Vec<AttributedRelationalRow> = take_json(&mut attached, rows)?;
    if attribution.scope != data.scope || accepted.iter().map(|row| &row.row).ne(decoded.iter()) {
        return Err(invalid(
            "native relational attribution names another response",
        ));
    }
    response_rows::coverage(&attribution, &accepted)?;
    Ok(attribution)
}

/// Pure replay of already reached conversion facts for merge composition.
/// Native objects are not read, encoded or converted by this operation.
pub(crate) fn layout(
    part: &RelationalResponsePart,
    attached: &mut Attachments,
) -> Result<(Part, Vec<RelationalRowLineage>), CoreError> {
    let (_, response, _) = decoded(&part.data, attached)?;
    if part.attribution.scope != part.data.scope {
        return Err(invalid(
            "native relational part names another response scope",
        ));
    }
    let rows: Vec<AttributedRelationalRow> = take_json(attached, part.rows)?;
    let target: Vec<RelationalTargetRow> = take_json(attached, part.target_rows)?;
    response_rows::layout(
        &response,
        &part.attribution,
        &rows,
        &part.target,
        &target,
        crate::pql::float_text,
    )
}

/// Validate one completed response at its original boundary and return
/// independently owned complete data for a later numbered merge/ranking.
pub(crate) fn complete(
    asker: &Asker,
    mut part: RelationalResponsePart,
    mut attached: Attachments,
) -> Result<RelationalResponsePart, CoreError> {
    let (value, response, decoded) = decoded(&part.data, &mut attached)?;
    let rows: Vec<AttributedRelationalRow> = take_json(&mut attached, part.rows)?;
    let target: Vec<RelationalTargetRow> = take_json(&mut attached, part.target_rows)?;
    if part.attribution.scope != part.data.scope {
        return Err(invalid(
            "native relational part names another response scope",
        ));
    }
    response_rows::layout(
        &response,
        &part.attribution,
        &rows,
        &part.target,
        &target,
        crate::pql::float_text,
    )?;
    part.data.response = asker.attach(Attachment::Bytes(value.to_bytes()))?;
    part.data.rows = asker.attach(encoded(&decoded)?)?;
    part.rows = asker.attach(encoded(&rows)?)?;
    part.target_rows = asker.attach(encoded(&target)?)?;
    Ok(part)
}
