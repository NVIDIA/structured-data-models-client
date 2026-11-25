// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a warehouse's failure means, read from its SQLSTATE and the
//! engine's error class (Databricks's closed list), as the old client
//! names the classes of its errors: the neutral facts the core reads every
//! source's failure by ([`sdm_contract::data::FailureFacts`]).

use sdm_contract::data::{FailureReason, ProviderError};

/// The engine's error class, without its subclass (`A.B` is `A`).
fn class(error: &ProviderError) -> Option<&str> {
    error
        .error_class
        .as_deref()
        .map(|name| name.split('.').next().unwrap_or(name))
}

/// What went wrong, where the failure's codes say.
pub(crate) fn reason(error: &ProviderError) -> Option<FailureReason> {
    use FailureReason as R;
    let state = error.sqlstate.as_deref();
    match (class(error), state) {
        (Some("CAST_INVALID_INPUT" | "CANNOT_PARSE_TIMESTAMP"), _)
        | (_, Some("22018" | "22007")) => Some(R::ValueNotConverted),
        (Some("DIVIDE_BY_ZERO" | "INTERVAL_DIVIDED_BY_ZERO"), _) | (_, Some("22012")) => {
            Some(R::DivisionByZero)
        }
        (Some("CAST_OVERFLOW" | "NUMERIC_VALUE_OUT_OF_RANGE" | "ARITHMETIC_OVERFLOW"), _)
        | (_, Some("22003" | "22008")) => Some(R::OutOfRange),
        (Some("UNRESOLVED_COLUMN" | "COLUMN_NOT_FOUND" | "UNRESOLVED_FIELD"), _)
        | (_, Some("42703" | "42S22")) => Some(R::ColumnUnresolved),
        (Some("TABLE_OR_VIEW_NOT_FOUND"), _) | (_, Some("42P01" | "42S02")) => {
            Some(R::TableUnresolved)
        }
        (Some("INSUFFICIENT_PERMISSIONS" | "PERMISSION_DENIED"), _) | (_, Some("42501")) => {
            Some(R::PermissionDenied)
        }
        (Some("PARSE_SYNTAX_ERROR"), _) | (_, Some("42601")) => Some(R::SyntaxRejected),
        (Some("DATATYPE_MISMATCH"), _) | (_, Some("42804")) => Some(R::TypesIncompatible),
        _ => None,
    }
}

/// Whether the source denied the caller access: SQLSTATE `42501`.
pub(crate) fn denied(error: &ProviderError) -> bool {
    error.sqlstate.as_deref() == Some("42501")
}
