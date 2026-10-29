// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed constructor domain product codecs shared by native bindings.
//! Source observations and domain decisions belong to their existing owners.

use sdm_contract::frames::FrameInference;
use sdm_contract::{CoreError, Sentence};

pub fn key_advice(
    bytes: &[u8],
    dates: std::collections::BTreeMap<String, String>,
) -> Result<sdm_core::FrameKeyAdvice, CoreError> {
    Ok(sdm_core::FrameKeyAdvice::new(
        serde_json::from_slice(bytes)
            .map_err(|_| invalid("native key advice requires a completed schema population"))?,
        dates,
    ))
}

pub fn notices_bytes(notices: &[sdm_contract::CoreWarning]) -> Result<Vec<u8>, CoreError> {
    serde_json::to_vec(notices).map_err(|_| invalid("native schema notices cannot be serialized"))
}

pub fn native_sample(bytes: &[u8]) -> Result<Vec<u8>, CoreError> {
    let sample = serde_json::from_slice(bytes)
        .map_err(|_| invalid("native sample observations are not a typed population"))?;
    notices_bytes(&sdm_core::native_frame_sample(sample)?)
}

pub fn key_adoption(bytes: &[u8]) -> Result<sdm_core::FrameKeyAdoption, CoreError> {
    sdm_core::FrameKeyAdoption::new(
        serde_json::from_slice(bytes)
            .map_err(|_| invalid("native key adoption requires a completed constructor product"))?,
    )
}

pub fn key_quality(bytes: &[u8]) -> Result<sdm_contract::frames::NativeKeyQuality, CoreError> {
    serde_json::from_slice(bytes)
        .map_err(|_| invalid("native key quality is not a typed observation"))
}

pub fn finish_key_adoption(
    adoption: sdm_core::FrameKeyAdoption,
    rows: Option<u64>,
) -> Result<(Vec<u8>, Vec<u8>), CoreError> {
    let (product, notices) = adoption.finish(rows)?;
    Ok((
        serde_json::to_vec(&product)
            .map_err(|_| invalid("native key adoption serialization failed"))?,
        serde_json::to_vec(&notices)
            .map_err(|_| invalid("native key adoption notice serialization failed"))?,
    ))
}

pub fn object_meaning_bytes(
    meaning: sdm_contract::frames::NativeObjectMeaning,
) -> Result<Vec<u8>, CoreError> {
    serde_json::to_vec(&meaning).map_err(|_| invalid("native column meaning serialization failed"))
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

pub fn key_candidates(bytes: &[u8], rows: u64) -> Result<sdm_core::FrameKeyCandidates, CoreError> {
    let FrameInference::ColumnMeanings(table) = serde_json::from_slice(bytes)
        .map_err(|_| invalid("native key candidates require completed column meanings"))?
    else {
        return Err(invalid(
            "native key candidates require the column-meanings product",
        ));
    };
    Ok(sdm_core::FrameKeyCandidates::new(*table, rows))
}

pub fn finish_key_candidates(
    candidates: sdm_core::FrameKeyCandidates,
    missing: Option<(bool, u64)>,
) -> Result<(Vec<u8>, Vec<u8>), CoreError> {
    let (nomination, notices) = candidates.finish(missing)?;
    Ok((
        serde_json::to_vec(&FrameInference::Nomination(Box::new(nomination)))
            .map_err(|_| invalid("native key nomination serialization failed"))?,
        serde_json::to_vec(&notices)
            .map_err(|_| invalid("native key notice serialization failed"))?,
    ))
}

/// Neutral constructor facts enter the same column meaning for every binding.
pub fn native_column(
    table: String,
    name: String,
    bytes: &[u8],
    declared: &[u8],
) -> Result<crate::FrameNativeConstructor, CoreError> {
    let values = serde_json::from_slice(bytes)
        .map_err(|_| invalid("native column constructor requires typed source facts"))?;
    let declared: Option<sdm_contract::frames::materialize::FramePhysicalDeclaration> =
        serde_json::from_slice(declared)
            .map_err(|_| invalid("native column constructor requires an original declaration"))?;
    sdm_core::native_column_constructor(&table, name, &values, declared.as_ref())
}

pub fn native_column_bytes(value: &crate::FrameNativeConstructor) -> Result<Vec<u8>, CoreError> {
    serde_json::to_vec(&value.complete()?)
        .map_err(|_| invalid("native constructed column cannot be serialized"))
}

pub fn native_column_dtype_bytes(
    value: &crate::FrameNativeConstructor,
) -> Result<Vec<u8>, CoreError> {
    serde_json::to_vec(&value.dtype())
        .map_err(|_| invalid("native constructed dtype cannot be serialized"))
}
