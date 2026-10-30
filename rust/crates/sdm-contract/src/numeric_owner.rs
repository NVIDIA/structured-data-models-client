// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed native numeric observations, scoped to one suspended ordering run.
use crate::data::AttachmentId;
use crate::run::ReplyId;
use crate::tabular::{NumericComparisonProfile, TabularIndexScalar, TabularSortProfile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NumericSourceRules {
    CommonTypeOnly,
    MutableFixedIntegerPairs,
    MutableExactUnboundedIntegers,
    FixedWeakScalars,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProbeExceptionPolicy {
    TypeAndValue,
    TypeValueAndDecimalInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OrderNumericOwners {
    pub length: usize,
    pub sort_profile: TabularSortProfile,
    pub source_rules: NumericSourceRules,
    pub probe_exception_policy: ProbeExceptionPolicy,
}

/// Opaque host-assigned owner identity, positive and unique within this run.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct OwnerId(pub u64);
/// Opaque identity of a retained original exception within this run.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ExceptionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericDestination {
    /// Width in bytes, one/two/four/eight.
    SignedInteger { width: u8 },
    /// Width in bytes, one/two/four/eight.
    UnsignedInteger { width: u8 },
    /// IEEE width in bytes, two/four/eight.
    BinaryFloat { width: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericOwnerRequest {
    LoadProbeOwner {
        position: usize,
    },
    ObserveComparisonEnvironment,
    CaptureSortOwners,
    ConvertOwner {
        owner: OwnerId,
        destination: NumericDestination,
        /// Selected narrowing can dispatch the native overflow handler.
        overflow_possible: bool,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericOwnerStep {
    pub id: ReplyId,
    pub request: NumericOwnerRequest,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericOwnerFact {
    pub owner: OwnerId,
    pub fact: TabularIndexScalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NumericExceptionClass {
    TypeError,
    ValueError,
    DecimalInvalidOperation,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericOwnerOutcome {
    ProbeOwner {
        owner: OwnerId,
        fact: TabularIndexScalar,
    },
    ComparisonEnvironment {
        policy: NumericComparisonProfile,
    },
    /// JSON bytes attachments: Vec<NumericOwnerFact>, Vec<usize>, Vec<usize>.
    /// Source and missing positions form a disjoint complete input partition.
    SortOwners {
        owners_and_facts: AttachmentId,
        source_positions: AttachmentId,
        missing_positions: AttachmentId,
    },
    /// Exact destination storage bits, unused high bits must be zero.
    Converted {
        bits: u64,
    },
    Raised {
        exception: ExceptionId,
        class: NumericExceptionClass,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericOwnerReply {
    pub id: ReplyId,
    pub outcome: NumericOwnerOutcome,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OrderedNumericOwners {
    Sorted {
        positions: Vec<usize>,
        already_ordered: bool,
    },
    Propagate {
        exception: ExceptionId,
    },
}
