// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL adapter: the one module that names a PQL implementation. Until
//! the interim parser is ported behind it, validation is unsupported.

use sdm_contract::ops::{QuerySchema, QueryValidation};
use sdm_contract::{CoreError, ErrorKind, HostFacts, codes};

use crate::profiles::Profile;

/// Validates `query` against `schema` for `profile`.
pub fn validate(
    _query: &str,
    _schema: &QuerySchema,
    _profile: &Profile,
    _facts: &HostFacts,
) -> Result<QueryValidation, CoreError> {
    Err(CoreError::new(
        ErrorKind::Unsupported,
        codes::UNSUPPORTED_REQUEST,
        "this build has no PQL parser behind the adapter yet",
    ))
}
