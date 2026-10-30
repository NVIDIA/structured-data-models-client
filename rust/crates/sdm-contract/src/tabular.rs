// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reached frame-backed Tabular selection facts, carried in byte attachments.
//! Native owners, missing-key diagnostics and sampled index labels stay separate.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Only admitted native-equivalent primitive keys. Null and custom keys are not
/// represented here. Signed/unsigned native integers share one exact domain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularKey {
    Text {
        value: String,
    },
    Integer {
        #[serde(with = "integer_text")]
        #[schemars(with = "String")]
        value: i128,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExpandTabularKeys {
    pub source: Vec<TabularKey>,
    pub requested: Vec<TabularKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExpandedTabularKeys {
    pub rows: Vec<usize>,
    /// Ordinals in the original requested sequence, including repeated misses.
    pub missing: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FilterTabularRows {
    pub known: Vec<bool>,
    pub excluded: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FilteredTabularRows {
    pub positions: Vec<usize>,
}

/// Count admission is separate from native custom count operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecideTabularContext {
    pub population: usize,
    pub requested: Option<usize>,
}

/// Domain results let the host render the original reached native diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TabularCountError {
    Zero,
    AboveLimit,
    AbovePopulation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularContextDecision {
    Selected {
        wanted: usize,
        reduced: bool,
        advise_default_reduction: bool,
    },
    Refused {
        reason: TabularCountError,
    },
}

/// Observe this population after any default-reduction warning was delivered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SampleTabularRows {
    pub population: usize,
    pub wanted: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SampledTabularRows {
    /// Candidate-frame offsets in draw order, not sorted source positions.
    Sampled {
        offsets: Vec<usize>,
    },
    Refused {
        reason: TabularCountError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObserveTabularMembership {
    pub present: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObservedTabularMembership {
    pub missing: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecideTabularTask {
    Target {
        target_stype: String,
    },
    Kind {
        target_stype: String,
        /// None means absent; Some is reached native type/membership validity.
        stated: Option<bool>,
        binary_whole: Option<bool>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TabularTaskError {
    UnsupportedTarget,
    UnsupportedTask,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularTaskDecision {
    TargetSupported,
    Explicit,
    Task { kind: String, warn_binary: bool },
    InspectTarget,
    Refused { reason: TabularTaskError },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TabularSortProfile {
    /// Resolve only ordering outcomes independent of producer tie policy.
    Unbound,
    Scalar,
    Avx2Gnu14,
    Avx512Gnu14,
    Stable,
}

/// Only ordinary native index producers admitted by the host are represented.
/// Missing values have inert payload slots. Unicode carries Python code points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabularIndexFacts {
    pub values: TabularIndexValues,
    pub missing: Vec<bool>,
    pub profile: TabularSortProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularIndexValues {
    Signed {
        values: Vec<i64>,
        width: u8,
    },
    Unsigned {
        values: Vec<u64>,
        width: u8,
    },
    Floating {
        bits: Vec<u64>,
        width: u8,
    },
    /// Component widths are four or eight bytes; bits preserve native values.
    Complex {
        real_bits: Vec<u64>,
        imag_bits: Vec<u64>,
        width: u8,
    },
    Temporal {
        values: Vec<i64>,
    },
    Unicode {
        values: Vec<Vec<u32>>,
    },
    Categorical {
        codes: Vec<i64>,
        width: u8,
    },
    /// Each codes vector corresponds to one level, in sampled row order.
    Multi {
        levels: Vec<TabularIndexFacts>,
        codes: Vec<Vec<i64>>,
    },
    Object {
        values: Vec<TabularIndexScalar>,
    },
    ScalarObject {
        values: Vec<TabularIndexScalar>,
        semantics: NumericComparisonProfile,
    },
}

/// Built-in missing-owner self-comparison behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NativeMissingOwner {
    None,
    Ambiguous,
    Temporal,
}

/// Closed numeric-owner comparison policies, independent of sorting's tie policy.
/// The host binds a verified producer and its actual promotion mode to a policy;
/// neither a sorting profile nor a producer major version establishes this choice.
///
/// All policies preserve fixed floating widths and use the wider width for two
/// fixed floating owners. Fixed integer/floating pairs compare at the floating
/// width for one-byte integers, at least binary32 for two-byte integers, and
/// binary64 for four/eight-byte integers. Integer signedness does not alter that
/// table. Boolean owners contribute exact zero or one. Plain binary64 pairs and
/// unbounded-integer/plain-binary64 pairs retain exact scalar comparisons.
/// Unordered floating comparisons still perform reached conversions first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NumericComparisonProfile {
    /// Use common fixed types for mixed scalar owners. A signed integer paired
    /// with an unsigned eight-byte integer compares through binary64, including
    /// its rounding. An unbounded integer that fits signed64 or unsigned64
    /// contributes that type; one outside both ranges takes the object fallback,
    /// comparing exactly against another integer or the floating owner's value.
    /// Plain binary64/fixed-floating pairs use binary64, as do representable
    /// unbounded-integer/fixed-floating pairs. This mode does not weakly narrow
    /// plain binary64 operands to the other owner's floating width.
    CommonType,
    /// Use CommonType floating rules and its out-of-range unbounded-integer
    /// fallback, but compare every integer/integer pair exactly, including the
    /// signed64/unsigned64 boundary. This policy must be bound to the producer's
    /// common-type promotion mode, not a weak or warning-producing mode.
    ExactIntegerPairs,
    /// Compare integer pairs exactly. A plain binary64 operand paired with a
    /// fixed floating owner is narrowed to that owner's width; an unbounded
    /// integer paired with a fixed floating owner converts through binary64 and
    /// then that width. Failure to convert the integer to finite binary64 yields
    /// IntegerToFloatOverflow. Finite narrowing to infinity yields the explicit
    /// NativeEffectsPending outcome because warning/handler effects are not yet
    /// represented; already-infinite input does not take that overflow branch.
    /// Fixed-integer/plain-binary64 pairs compare in binary64, including rounding.
    WeakScalars,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularIndexScalar {
    /// A masked built-in missing owner whose reached comparison stops probing.
    NativeMissing {
        owner: NativeMissingOwner,
    },
    /// Canonical integer value; width is one, two, four or eight bytes.
    FixedInteger {
        signed: bool,
        width: u8,
        value: String,
    },
    /// Raw IEEE bits; width is two (binary16), four or eight bytes.
    FixedFloating {
        width: u8,
        bits: u64,
    },
    Integer {
        value: String,
    },
    Floating {
        bits: u64,
    },
    Text {
        points: Vec<u32>,
    },
    Boolean {
        value: bool,
    },
    /// Finite decimal coefficient and exponent, without floating conversion.
    Decimal {
        negative: bool,
        digits: String,
        exponent: i64,
    },
    Date {
        ordinal: i32,
    },
    /// Exact civil datetime with either no timezone or a fixed UTC offset.
    DateTime {
        ordinal: i32,
        micros: u64,
        offset_micros: Option<i64>,
    },
    /// Object complex labels retain their unordered native comparison behavior.
    Complex {
        real_bits: u64,
        imag_bits: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OrderSampledTabularRows {
    pub index: TabularIndexFacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OrderedSampledTabularRows {
    /// Reuse the same facts after binding their remaining producer profiles.
    NeedsProfile,
    Sorted {
        positions: Vec<usize>,
        /// Native sort_index may copy/finalize without taking a permutation.
        already_ordered: bool,
    },
    Refused {
        reason: TabularOrderError,
        left: usize,
        right: usize,
    },
}

/// A comparison outcome, not a new public exception or diagnostic wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TabularOrderError {
    Incomparable,
    DateTimeMismatch,
    NaiveAwareDatetime,
    IntegerToFloatOverflow,
    NativeEffectsPending,
}

/// Reached native comparison against the model cap, before frame length reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckTabularCap {
    pub over_limit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckedTabularCap {
    pub accepted: bool,
}

/// One ordinary selection composition. Only proven-passive native producers
/// may supply the complete population before earlier semantic guards finish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectTabularRows {
    pub mode: TabularSelectionMode,
    pub known: Vec<bool>,
    pub requested_context: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TabularSelectionMode {
    Indices {
        source: Vec<TabularKey>,
        requested: Vec<TabularKey>,
    },
    Supplied {
        source: Vec<TabularKey>,
        scoring: Vec<TabularKey>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SelectedTabularRows {
    Selected {
        /// Source positions in requested order; empty for supplied frames.
        scored: Vec<usize>,
        /// Source positions before any context draw or sampled-index ordering.
        candidates: Vec<usize>,
        context: TabularContextDecision,
    },
    Missing {
        ordinals: Vec<usize>,
    },
}

// Integer observations use canonical decimal text like the other exact scalars.
// Internally tagged enum deserialization cannot carry a bare i128 JSON number.
mod integer_text {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &i128, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
        let text = String::deserialize(deserializer)?;
        let value = text.parse::<i128>().map_err(serde::de::Error::custom)?;
        if value.to_string() != text {
            return Err(serde::de::Error::custom("noncanonical tabular integer"));
        }
        Ok(value)
    }
}
