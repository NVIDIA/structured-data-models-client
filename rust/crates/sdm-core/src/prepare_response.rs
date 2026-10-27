// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned response buffers and the fixed native-object conversion boundary.

use crate::attached::take_json;
use crate::prepare::invalid;
use crate::run::{Asker, Attachment, Attachments};
use sdm_contract::data::AttachmentId;
use sdm_contract::ops::OperationResult;
use sdm_contract::prepare::{
    AttributedDirectRow, DecodedDirectResponse, DecodedDirectRow, DirectResponseData,
};
use sdm_contract::{CoreError, CoreWarning};
use sdm_request::{predict as policy, tabular::response as rows};
use sdm_wire::json::{self, Value};
use sdm_wire::response::PredictionResponse;

fn encoded<T: serde::Serialize>(value: &T) -> Result<Attachment, CoreError> {
    serde_json::to_vec(value)
        .map(Attachment::Bytes)
        .map_err(|_| invalid("direct response data cannot serialize"))
}

pub(crate) fn decode(
    asker: &Asker,
    payload: &Value,
    response_value: Value,
    response: &PredictionResponse,
    outputs: Vec<String>,
    named: bool,
) -> Result<OperationResult, CoreError> {
    let (scope, decoded) = rows::decode(payload, response, crate::pql::float_text)?;
    let data = DirectResponseData {
        scope,
        outputs,
        named,
        payload: asker.attach(Attachment::Bytes(payload.to_bytes()))?,
        response: asker.attach(Attachment::Bytes(response_value.to_bytes()))?,
        rows: asker.attach(encoded(&decoded)?)?,
    };
    let value = asker.attach(encoded(&data)?)?;
    Ok(OperationResult::DecodedDirectResponse(
        DecodedDirectResponse { value },
    ))
}

fn bytes(attached: &mut Attachments, id: AttachmentId) -> Result<Vec<u8>, CoreError> {
    match attached.remove(&id) {
        Some(Attachment::Bytes(bytes)) => Ok(bytes),
        _ => Err(invalid("direct response buffer is not attached bytes")),
    }
}

pub(crate) fn layout(
    asker: &Asker,
    data: DirectResponseData,
    attributed: AttachmentId,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let payload = json::read(&bytes(&mut attached, data.payload)?)
        .map_err(|_| invalid("direct response payload is unreadable"))?;
    if payload.get("output").and_then(|v| v.get("fields"))
        != Some(&Value::texts(data.outputs.iter().map(String::as_str)))
    {
        return Err(invalid(
            "direct response outputs disagree with the encoded call",
        ));
    }
    let value = json::read_answer(&bytes(&mut attached, data.response)?)
        .map_err(|_| invalid("direct decoded response is unreadable"))?;
    let response = PredictionResponse::read(&value)
        .map_err(|_| invalid("direct decoded response has another shape"))?;
    let decoded: Vec<DecodedDirectRow> = take_json(&mut attached, data.rows)?;
    let (scope, actual) = rows::decode(&payload, &response, crate::pql::float_text)?;
    if scope != data.scope || decoded != actual {
        return Err(invalid(
            "direct response buffers disagree with their typed rows",
        ));
    }
    let accepted: Vec<AttributedDirectRow> = take_json(&mut attached, attributed)?;
    let part = rows::layout(
        &payload,
        &response,
        &scope,
        accepted,
        crate::pql::float_text,
    )?;
    if data.named {
        let absent = policy::absent(&part, &data.outputs);
        if !absent.is_empty() {
            let model = payload
                .get("model")
                .and_then(Value::as_text)
                .ok_or_else(|| invalid("direct payload has no model"))?;
            let notice = policy::absent_notice(model, &absent);
            asker
                .warn(CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields));
        }
    }
    crate::predict::finish_direct(asker, part, data.outputs, None)
}
