// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit native string data, pending coordinated contract35 linking.
//! Each attachment is one owned Arrow column. Its logical interpretation is
//! bound to this descriptor, never inferred from arbitrary Binary values.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::AttachmentId;

/// Closed value encodings, independently of native storage or casing.
/// Both Arrow offset widths retain their original physical buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StringEncoding {
    /// Arrow Utf8 or LargeUtf8, whose values are ordinary Unicode scalars.
    Utf8,
    /// Arrow Binary or LargeBinary, containing canonical UTF8 with
    /// surrogatepass. Adjacent surrogate points remain distinct points.
    Utf8Surrogatepass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StringData {
    pub values: AttachmentId,
    pub encoding: StringEncoding,
}
