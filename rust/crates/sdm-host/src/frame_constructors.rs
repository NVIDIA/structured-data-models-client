// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed constructor domain product codecs shared by native bindings.
//! Source observations and domain decisions belong to their existing owners.

use sdm_contract::frames::FrameInference;
use sdm_contract::{CoreError, Sentence};

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
