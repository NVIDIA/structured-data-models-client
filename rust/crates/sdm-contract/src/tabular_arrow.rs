// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow-native Tabular calls. Data-sized metadata is always an attachment.

use crate::{
    data::AttachmentId,
    ops::{HeldSchema, ModelEndpoint},
    session::{SessionKeys, TabularSessionState},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArrowTabular {
    Describe {
        input: AttachmentId,
        table: AttachmentId,
    },
    Prepare {
        input: AttachmentId,
        context: AttachmentId,
        predict: AttachmentId,
    },
    Predict {
        input: AttachmentId,
        context: AttachmentId,
        predict: AttachmentId,
    },
    Materialize {
        input: AttachmentId,
        table: AttachmentId,
    },
    MaterializePredict {
        input: AttachmentId,
        table: AttachmentId,
        predict: AttachmentId,
    },
    Fit {
        input: AttachmentId,
        context: AttachmentId,
        predict: AttachmentId,
    },
    Score {
        input: AttachmentId,
        predict: AttachmentId,
    },
    Close {
        input: AttachmentId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowDescription {
    pub entity_table: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowRequest {
    pub target: String,
    pub task: String,
    pub entity_table: String,
    pub schema: Option<HeldSchema>,
    pub query: Option<String>,
    pub context_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowPrediction {
    pub request: ArrowRequest,
    pub model: String,
    pub outputs: Option<Vec<String>>,
    pub request_id: String,
    pub endpoint: Option<ModelEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowFit {
    pub prediction: ArrowPrediction,
    pub holdout: SessionKeys,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowSession {
    pub state: TabularSessionState,
    pub keys: SessionKeys,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowScore {
    pub session: ArrowSession,
    pub request: ArrowRequest,
    pub request_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowMaterialize {
    pub schema: HeldSchema,
    pub query: String,
    pub indices: Vec<crate::session::SessionKeyPart>,
    pub num_context_samples: Option<usize>,
}
