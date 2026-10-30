// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema facts observed in native consumption order for direct frames.

use super::{ColumnFlags, DirectShape, dtype::DtypeFinding};
use crate::ops::{Dtype, HeldSchema, Stype};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObservedColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
    pub untyped: bool,
}

/// Separately reached native population facts for one eligible timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimePopulation {
    pub name: String,
    pub rows: usize,
    pub held: usize,
    pub same: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimeStart {
    /// Exact UTC nanoseconds, absent when native conversion found no real date.
    pub nanos: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SchemaObservation {
    TableNames {
        table: String,
        columns: Vec<String>,
    },
    Column {
        name: String,
        finding: DtypeFinding,
        distinct: Option<usize>,
    },
    KeyCandidates {
        columns: Vec<ObservedColumn>,
        rows: usize,
    },
    KeyCount {
        rows: usize,
        distinct: usize,
    },
    KeyCompanions {
        columns: Vec<ObservedColumn>,
        distinguishing: Vec<String>,
    },
    KeyValidity {
        table: String,
        column: String,
        nulls: u64,
        duplicates: u64,
        rows: usize,
        empty: Vec<String>,
    },
    KeyRepeated {
        table: String,
        column: String,
        duplicates: u64,
        distinguishing: Vec<String>,
        usable: Vec<String>,
    },
    TimeCandidates {
        columns: Vec<ObservedColumn>,
        primary_key: Option<String>,
    },
    TimeChoice {
        columns: Vec<ObservedColumn>,
        primary_key: Option<String>,
        rows: usize,
        populations: Vec<TimePopulation>,
        starts: Option<Vec<(String, TimeStart)>>,
    },
    CategoricalAdvice {
        table: String,
        column: String,
        distinct: usize,
        present: usize,
    },
    DateAdviceCandidates {
        columns: Vec<ObservedColumn>,
        primary_key: Option<String>,
        time_column: Option<String>,
        empty: bool,
    },
    DateMinimumYear {
        year: i64,
    },
    DateMaximumYear {
        table: String,
        column: String,
        minimum: i64,
        year: i64,
    },
    Assemble {
        table: String,
        columns: Vec<ObservedColumn>,
        primary_key: Option<String>,
        duplicates: u64,
        time_column: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservedSchema {
    Column {
        column: ObservedColumn,
    },
    Names {
        names: Vec<String>,
    },
    Key {
        accepted: bool,
    },
    Time {
        column: Option<String>,
        earliest: Vec<String>,
    },
    Dates {
        names: Vec<String>,
        utc_readings: Vec<bool>,
    },
    Schema {
        schema: HeldSchema,
        flags: Vec<ColumnFlags>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparationObservation {
    TabularMeaning { inferred: Stype, different: Option<bool>, classification: Option<bool> },
    CoherentMeaning { inferred: Stype, dtype: Dtype, declared: Option<bool> },
    ConstancyPopulation { empty: Option<bool>, diverse: Option<bool> },
    ConstancyKind { classification: bool, regression: Option<bool> },
    ConstancyNotice { classification: bool },
    Admission { observation: AdmissionObservation },
    Arguments {
        shape: DirectShape,
    },
    Consistent {
        schema: HeldSchema,
    },
    Declarations {
        columns: Vec<(String, String, bool)>,
    },
    Population {
        shape: DirectShape,
        known: usize,
    },
    Names {
        table: String,
        columns: Vec<String>,
    },
    ScoredColumn {
        table: String,
        schema: HeldSchema,
        flags: Vec<ColumnFlags>,
        name: String,
        finding: DtypeFinding,
    },
    ScoredComplete {
        shape: DirectShape,
        schema: HeldSchema,
    },
    Constancy {
        target: String,
        task: String,
        known: usize,
        distinct: usize,
    },
    MissingKey {
        table: String,
        column: String,
        count: usize,
    },
}

/// Reached native constructor facts, before later frame/schema observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdmissionObservation {
    SchemaColumns { missing: bool },
    TableEntryName { valid: bool },
    NameKind { valid: bool },
    NameCharacters { present: Vec<bool> },
    Frame { valid: bool, row_hint: bool, parameter: String, native_type: String },
    Related { present: bool },
    Task { accepted: bool },
    TargetCells { containers: bool },
    ContextRows { empty: bool },
    ContextTarget { present: bool },
    PredictTarget { present: bool },
    KnownTarget { present: bool },
    ExtraColumns { present: bool },
    DuplicateColumns { present: bool },
    ColumnTypes { non_strings: bool },
    MissingFeatures { present: bool },
}

/// A selected meaning, or a request for the reached native declaration flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MeaningResult { pub stype: Option<Stype> }
