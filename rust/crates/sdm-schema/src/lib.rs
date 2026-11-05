// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema inference, validation and editing over connector-neutral facts and Arrow.
//! This crate contains no SQL or connector-specific dispatch.
//!
//! [`infer`] determines column types, keys and time columns using catalog facts
//! and preview values, requesting whole-table evidence when needed.
//! [`assemble`] combines tables and relationships; [`edit`] validates schema edits.
//! [`held`] converts host-held schema data, and [`view`] parses semantic-view definitions.
//! Unsupported inputs return structured errors rather than guessed metadata.

#![forbid(unsafe_code)]

pub mod assemble;
pub mod catalog;
pub mod direct;
pub mod edit;
pub mod held;
pub mod infer;
pub mod keys;
mod names;
mod values;
pub mod view;

use sdm_contract::{CoreError, Detail, Sentence};

/// The refusal of an input whose inference takes a rule this build does not
/// port yet, naming the rule, never a value or a name the input holds
///.
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        Sentence::NotYetDescribe,
        "describing this source takes an inference rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("describe_source"))
    .with_field("rule", Detail::text(rule))
}

pub mod refine;
