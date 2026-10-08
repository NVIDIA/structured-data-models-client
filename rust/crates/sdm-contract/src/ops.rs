// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operations: typed requests and their results, not functions looked up by
//! name.
//!
//! Phase 1 needs four: `validate_query` (S0), and `describe_source`,
//! `materialize` and `predict` for the Phase 1b journey. The journey's three
//! are the shapes the journey is known to need; Phase 1b refines them, each
//! change with a contract version bump.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, ProviderChoice};
use crate::scalar::Scalar;
use crate::types::ColumnType;

/// One operation's request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    ValidateQuery(ValidateQuery),
    DescribeSource(DescribeSource),
    Materialize(Materialize),
    Predict(Predict),
}

impl Operation {
    /// The operation's name, as the contract spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Operation::ValidateQuery(_) => "validate_query",
            Operation::DescribeSource(_) => "describe_source",
            Operation::Materialize(_) => "materialize",
            Operation::Predict(_) => "predict",
        }
    }
}

/// One operation's result, the payload of the run's last step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperationResult {
    ValidateQuery(QueryValidation),
    DescribeSource(SourceDescription),
    Materialize(Materialized),
    Predict(Predicted),
}

// --- validate_query -------------------------------------------------------------

/// Validate a predictive query against a schema, for a profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidateQuery {
    pub query: String,
    pub schema: QuerySchema,
    pub profile: ProfileSpec,
}

/// The schema a query is resolved against: names and types, no rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QuerySchema {
    /// The tables, in order.
    pub tables: Vec<QueryTable>,
    /// Groups of linked columns, such as a primary key and its foreign keys.
    pub links: Vec<Vec<ColumnRef>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryTable {
    pub name: String,
    pub columns: Vec<QueryColumn>,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
    pub end_time_column: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ColumnRef {
    pub table: String,
    pub column: String,
}

/// PQL's data types.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bool,
    Int,
    Byte,
    Int16,
    Int32,
    Int64,
    Float,
    Float32,
    Float64,
    String,
    Binary,
    Date,
    Time,
    Timedelta,
    Floatlist,
    Intlist,
    Stringlist,
    Unsupported,
}

/// PQL's semantic types.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub enum Stype {
    #[serde(rename = "numerical")]
    Numerical,
    #[serde(rename = "categorical")]
    Categorical,
    #[serde(rename = "multicategorical")]
    Multicategorical,
    #[serde(rename = "ID")]
    Id,
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "timestamp")]
    Timestamp,
    #[serde(rename = "sequence")]
    Sequence,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "unsupported")]
    Unsupported,
}

/// A named profile, optionally naming the model in diagnostics and
/// changing named capabilities, as the language's `Profile.named` and
/// `Capabilities.also` do. The profiles themselves are the core's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    pub name: ProfileName,
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capabilities: BTreeMap<Capability, bool>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProfileName {
    FineTuning,
    Relational,
    Tabular,
    Timeseries,
    Anomaly,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Relationships,
    MultipleTables,
    Temporal,
    Static,
    EntityFilters,
    Counterfactuals,
    DerivedTargets,
    LinkPrediction,
    Forecasting,
    KeyColumnsInFilters,
    TimestampColumnsInFilters,
}

/// A query's validation: accepted, or refused with PQL's diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryValidation {
    pub accepted: bool,
    pub diagnostics: Vec<PqlDiagnostic>,
}

/// One PQL diagnostic: its code and where in the query text it points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PqlDiagnostic {
    pub code: String,
    pub message: String,
    pub severity: Severity,
    pub span: Option<Span>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

/// Where in the query text a diagnostic or node is, as PQL counts it: rows
/// from 1, columns from 0 in code points, from the first character of the
/// first token to the first character of the last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start_row: u32,
    pub start_col: u32,
    pub end_row: u32,
    pub end_col: u32,
}

// --- describe_source --------------------------------------------------------------

/// Describe the tables of a source: types, keys and time columns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeSource {
    pub provider: ProviderChoice,
    /// The tables to describe, in order.
    pub tables: Vec<String>,
    /// Declared primary keys, by table.
    pub primary_keys: BTreeMap<String, Vec<String>>,
    /// Declared time columns, by table.
    pub time_columns: BTreeMap<String, String>,
    /// Whether to count duplicate keys.
    pub check_keys: bool,
}

/// The described schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceDescription {
    pub tables: Vec<DescribedTable>,
    pub links: Vec<Vec<ColumnRef>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribedTable {
    pub name: String,
    pub columns: Vec<DescribedColumn>,
    pub primary_key: Vec<String>,
    pub time_column: Option<String>,
    /// Rows sharing a key value, when keys were checked: a `u64` scalar.
    pub duplicate_keys: Option<Scalar>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribedColumn {
    pub name: String,
    /// The type the provider handed over, after admission.
    #[serde(
        serialize_with = "crate::types::checked_column_type",
        deserialize_with = "crate::types::strict_column_type"
    )]
    pub received: ColumnType,
    pub dtype: Dtype,
    pub stype: Stype,
}

// --- materialize and predict --------------------------------------------------

/// Build a request for a query over a described source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Materialize {
    pub provider: ProviderChoice,
    pub schema: SourceDescription,
    pub query: String,
    /// The entities to predict for: an Arrow attachment of one column.
    pub indices: Option<AttachmentId>,
    /// The anchor time, a `ts` scalar.
    pub anchor_time: Option<Scalar>,
    /// The sampling seed, a `u64` scalar.
    pub seed: Option<Scalar>,
}

/// A materialized request, held by the host and handed back to `predict`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Materialized {
    pub state: RequestState,
}

/// What `predict` needs of a materialized request; the payload bytes travel
/// as an attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestState {
    /// The version of this state's format, independent of the contract's.
    pub version: u32,
    pub payload: AttachmentId,
    pub query: String,
}

/// Send a materialized request to the model and lay out its answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Predict {
    pub state: RequestState,
    pub model: String,
}

/// The prediction: an Arrow attachment and how its columns are laid out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Predicted {
    pub result: AttachmentId,
    /// The result's columns, in order.
    pub columns: Vec<String>,
}
