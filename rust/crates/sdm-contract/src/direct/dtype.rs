// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native dtype facts and exact range observations for a direct column.

use crate::ops::Dtype;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DtypeStorage {
    Boolean,
    Timestamp,
    Integer { may_exceed_i64: bool },
    Floating { bytes: usize },
    Decimal { precision: Option<u8>, scale: i8 },
    Object { content: ObjectContent },
    Other,
}

/// Closed observed scalar families, independent of the host library's labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ObjectContent {
    Empty,
    Decimal,
    Integer,
    Floating,
    MixedIntegerFloat,
    Boolean,
    Date,
    DateTime,
    Timestamp,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DtypePopulation {
    Declared,
    WholeNumbers,
    Decimals,
}

/// An inferred type and the population whose remaining values can change it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DtypeFinding {
    pub dtype: Dtype,
    pub untyped: bool,
    pub population: DtypePopulation,
    pub decimal_safe_precision: i64,
    pub integer_min: String,
    pub integer_max: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DtypeObservation {
    Storage {
        storage: DtypeStorage,
    },
    WholeNumber {
        previous: DtypeFinding,
        within_range: Option<bool>,
    },
    Decimal {
        previous: DtypeFinding,
        is_decimal: bool,
        finite: bool,
        fractional: bool,
        outside_integer_range: bool,
    },
    Exhausted {
        finding: DtypeFinding,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DtypeResolutionObservation {
    NullableInteger {
        declared: Dtype,
        equal: bool,
    },
    Widen {
        before: Option<Dtype>,
        arriving: Dtype,
        untyped: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DtypeResolution {
    pub dtype: Option<Dtype>,
}
