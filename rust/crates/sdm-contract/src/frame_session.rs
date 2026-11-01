// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned metadata for materialized in-memory relational sessions.
use crate::{
    data::AttachmentId,
    session::{SessionKeys, SessionQuestion},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameSessionOperation {
    Fit {
        input: AttachmentId,
        payloads: Vec<AttachmentId>,
    },
    Predict {
        input: AttachmentId,
        payloads: Vec<AttachmentId>,
        pinned: AttachmentId,
    },
    Close {
        input: AttachmentId,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameSessionState {
    pub version: u32,
    pub session_id: String,
    pub closed: bool,
    pub model: String,
    pub outputs: Vec<String>,
    pub default_outputs: bool,
    pub question: Option<SessionQuestion>,
    pub keys: SessionKeys,
    pub windowed: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameSessionInput {
    Fit {
        model: String,
        outputs: Option<Vec<String>>,
        question: Option<SessionQuestion>,
        holdout: SessionKeys,
        windowed: bool,
        request_id: String,
        inference_asked: Vec<String>,
    },
    Predict {
        state: FrameSessionState,
        question: Option<SessionQuestion>,
        request_ids: Vec<String>,
        inference_asked: Vec<String>,
    },
    Close {
        state: FrameSessionState,
    },
}
