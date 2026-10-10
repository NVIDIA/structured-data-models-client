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
//!
//! # Input-sized fields
//!
//! The control cap bounds a message's envelope only (decision H5). A field
//! whose size grows with what the user hands in (query text, a
//! [`QuerySchema`], a [`SourceDescription`], the tables a source is
//! described by) travels as a bytes attachment holding the UTF-8 JSON of the
//! same serde type, which the message names by [`AttachmentId`]; the field's
//! documentation names the type. No inline field grows with column, table or
//! row count.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{AttachmentId, ProviderChoice};
use crate::scalar::Scalar;
#[cfg(feature = "test-ops")]
use crate::test_ops::{TestOperation, TestResult};
use crate::types::ColumnType;

/// One operation's request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    ValidateQuery(ValidateQuery),
    DescribeSource(DescribeSource),
    Materialize(Materialize),
    Predict(Predict),
    /// Operations only tests reach, built only with the `test-ops` feature,
    /// which a shipped build leaves off; not part of the contract's schema.
    #[cfg(feature = "test-ops")]
    #[schemars(skip)]
    Test(TestOperation),
}

/// What an attachment a message names holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttachmentKind {
    /// Bytes, such as the UTF-8 JSON of an input-sized field or a model
    /// body.
    Bytes,
    /// Arrow data.
    Arrow,
}

impl Operation {
    /// The operation's name, as the contract spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Operation::ValidateQuery(_) => "validate_query",
            Operation::DescribeSource(_) => "describe_source",
            Operation::Materialize(_) => "materialize",
            Operation::Predict(_) => "predict",
            #[cfg(feature = "test-ops")]
            Operation::Test(_) => "test",
        }
    }

    /// The attachments a `start` of this operation names, with the kind
    /// each must be, in the order the operation's fields name them.
    pub fn attachments(&self) -> Vec<(AttachmentId, AttachmentKind)> {
        use AttachmentKind::{Arrow, Bytes};
        match self {
            Operation::ValidateQuery(v) => vec![(v.query, Bytes), (v.schema, Bytes)],
            Operation::DescribeSource(d) => vec![(d.source, Bytes)],
            Operation::Materialize(m) => {
                let mut named = vec![(m.schema, Bytes), (m.query, Bytes)];
                named.extend(m.indices.map(|id| (id, Arrow)));
                named
            }
            Operation::Predict(p) => vec![(p.state.payload, Bytes), (p.state.query, Bytes)],
            #[cfg(feature = "test-ops")]
            Operation::Test(t) => t.attachments(),
        }
    }
}

/// One operation's result, the payload of the run's last step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperationResult {
    ValidateQuery(QueryValidation),
    DescribeSource(DescribedSource),
    Materialize(Materialized),
    Predict(Predicted),
    #[cfg(feature = "test-ops")]
    #[schemars(skip)]
    Test(TestResult),
}

// --- validate_query -------------------------------------------------------------

/// Validate a predictive query against a schema, for a profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidateQuery {
    /// The query text: a bytes attachment of a JSON string.
    pub query: AttachmentId,
    /// A bytes attachment of a [`QuerySchema`] as JSON.
    pub schema: AttachmentId,
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
    #[serde(deserialize_with = "row")]
    #[schemars(range(min = 1))]
    pub start_row: u32,
    pub start_col: u32,
    #[serde(deserialize_with = "row")]
    #[schemars(range(min = 1))]
    pub end_row: u32,
    pub end_col: u32,
}

/// A row, which PQL counts from 1.
fn row<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let row = u32::deserialize(deserializer)?;
    if row == 0 {
        return Err(serde::de::Error::custom("rows count from 1"));
    }
    Ok(row)
}

// --- describe_source --------------------------------------------------------------

/// Describe the tables of a source: types, keys and time columns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeSource {
    pub provider: ProviderChoice,
    /// A bytes attachment of a [`SourceSpec`] as JSON.
    pub source: AttachmentId,
    /// Whether to count duplicate keys.
    pub check_keys: bool,
}

/// The tables a source is described by, and what the caller declared of
/// them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceSpec {
    /// The tables to describe, in order.
    pub tables: Vec<String>,
    /// Declared primary keys, by table.
    pub primary_keys: BTreeMap<String, Vec<String>>,
    /// Declared time columns, by table.
    pub time_columns: BTreeMap<String, String>,
}

/// `describe_source`'s result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribedSource {
    /// A bytes attachment of a [`SourceDescription`] as JSON.
    pub description: AttachmentId,
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
    /// Whether the column may hold no value, as the catalog declares it:
    /// neither declared `NOT NULL` nor part of the declared primary key.
    pub nullable: bool,
    /// Whether the catalog declares a primary or unique key of this one
    /// column, which the source refuses to repeat.
    pub unique: bool,
}

// --- materialize and predict --------------------------------------------------

/// Build a request for a query over a described source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Materialize {
    pub provider: ProviderChoice,
    /// A bytes attachment of a [`SourceDescription`] as JSON.
    pub schema: AttachmentId,
    /// The query text: a bytes attachment of a JSON string.
    pub query: AttachmentId,
    /// The entities to predict for: an Arrow attachment of one column.
    pub indices: Option<AttachmentId>,
    /// The anchor time, a `ts` scalar.
    pub anchor_time: Option<Scalar>,
    /// The sampling seed, an integer of any size.
    pub seed: Option<Seed>,
}

/// An integer of any size, as signed hexadecimal text with no leading
/// zeros (`0x0`, `0x2a`, `-0x400000000000000000`), so no language's integer
/// parser meets a digit limit; how large a seed the core takes is the
/// core's rule, not the decoder's (decision H7).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Seed {
    negative: bool,
    /// Lowercase hexadecimal digits with no leading zero, `0` for zero.
    magnitude: String,
}

impl Seed {
    /// Whether the seed is below zero.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// The seed's magnitude as lowercase hexadecimal digits, `0` for zero.
    pub fn magnitude_hex(&self) -> &str {
        &self.magnitude
    }

    /// The seed, when it is not negative and a `u128` holds it.
    pub fn to_u128(&self) -> Option<u128> {
        if self.negative {
            return None;
        }
        u128::from_str_radix(&self.magnitude, 16).ok()
    }
}

impl From<u128> for Seed {
    fn from(value: u128) -> Self {
        Seed {
            negative: false,
            magnitude: format!("{value:x}"),
        }
    }
}

impl From<i128> for Seed {
    fn from(value: i128) -> Self {
        Seed {
            negative: value < 0,
            magnitude: format!("{:x}", value.unsigned_abs()),
        }
    }
}

impl TryFrom<String> for Seed {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        let (negative, rest) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.as_str()),
        };
        let refused = || "a seed is signed hexadecimal text with no leading zeros".to_owned();
        let digits = rest.strip_prefix("0x").ok_or_else(refused)?;
        let lower = |c: char| c.is_ascii_digit() || ('a'..='f').contains(&c);
        if digits.is_empty()
            || !digits.chars().all(lower)
            || (digits.len() > 1 && digits.starts_with('0'))
            || (negative && digits == "0")
        {
            return Err(refused());
        }
        Ok(Seed {
            negative,
            magnitude: digits.to_owned(),
        })
    }
}

impl From<Seed> for String {
    fn from(seed: Seed) -> String {
        let sign = if seed.negative { "-" } else { "" };
        format!("{sign}0x{}", seed.magnitude)
    }
}

impl JsonSchema for Seed {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Seed".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": "^(0x0|-?0x[1-9a-f][0-9a-f]*)$"
        })
    }
}

/// A materialized request, held by the host and handed back to `predict`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Materialized {
    pub state: RequestState,
}

/// The version of [`RequestState`]'s format this build writes and reads
/// (decision H23); it is independent of the contract's.
pub const REQUEST_STATE_VERSION: u32 = 1;

/// What `predict` needs of a materialized request: state that outlives the
/// run that made it. `materialize` answers it; the host takes its
/// attachments when that run finishes, holds them, and attaches them again
/// to the `start` of the `predict` run, under ids it picks for that
/// message. A version other than [`REQUEST_STATE_VERSION`] is refused by
/// name before anything of the state is read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestState {
    /// The version of this state's format, independent of the contract's.
    pub version: u32,
    /// A bytes attachment of the request's payload as JSON, as
    /// `materialize` wrote it or as the host holds it since.
    pub payload: AttachmentId,
    /// The query text: a bytes attachment of a JSON string.
    pub query: AttachmentId,
}

/// Send a materialized request to the model and lay out its answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Predict {
    pub state: RequestState,
    /// The model id the request is sent to, as the wire names it.
    pub model: String,
    /// The id the request carries in its `metadata.request_id`. The host
    /// draws it, as the reference's client draws one for every request; the
    /// core draws nothing at random but under a seed.
    pub request_id: String,
}

/// The prediction. Nothing inline grows with the result's width or the
/// endpoint's answer (decision H5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Predicted {
    /// An Arrow attachment, one row per answer in the order the endpoint
    /// gave them, whose schema names its columns in order.
    pub result: AttachmentId,
    /// The model that answered, as the endpoint names it.
    pub model: String,
    /// The endpoint's id for the call.
    pub request_id: String,
    /// A bytes attachment of the endpoint's `metadata` object as JSON.
    pub metadata: AttachmentId,
}
