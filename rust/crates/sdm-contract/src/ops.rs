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

use crate::data::{AttachmentId, ProviderChoice, Relation};
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
    EditSchema(EditSchema),
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
            Operation::EditSchema(_) => "edit_schema",
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
            Operation::EditSchema(e) => vec![(e.schema, Bytes), (e.edit, Bytes)],
            Operation::Materialize(m) => {
                let mut named = vec![(m.schema, Bytes), (m.query, Bytes)];
                named.extend(m.indices.map(|id| (id, Arrow)));
                named.extend(m.candidates.map(|id| (id, Bytes)));
                named.extend(m.sampler_state.as_ref().map(|s| (s.memo, Bytes)));
                named.extend(m.values.map(|id| (id, Bytes)));
                named
            }
            Operation::Predict(p) => {
                let mut named = vec![(p.state.payload, Bytes), (p.state.query, Bytes)];
                for step in p.state.forecast.iter().flatten() {
                    named.push((step.payload, Bytes));
                    named.push((step.query, Bytes));
                }
                for batch in p.state.batches.iter().flatten() {
                    named.push((batch.payload, Bytes));
                }
                for part in p.state.ranking.iter().flat_map(|r| &r.parts) {
                    named.push((part.payload, Bytes));
                    named.push((part.query, Bytes));
                }
                named
            }
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
    EditSchema(SchemaEdited),
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

impl Dtype {
    /// The data type's name on the client's wire, as a host's schema
    /// names a column's data type (`int64`, `string`, `timestamp[us]`):
    /// a date and a time are both `timestamp[us]`.
    pub fn wire_name(self) -> &'static str {
        match self {
            Dtype::Bool => "bool",
            Dtype::Int | Dtype::Int64 => "int64",
            Dtype::Byte | Dtype::Int16 | Dtype::Int32 => "int32",
            Dtype::Float | Dtype::Float64 => "float64",
            Dtype::Float32 => "float32",
            Dtype::String => "string",
            Dtype::Date | Dtype::Time => "timestamp[us]",
            Dtype::Binary => "binary",
            Dtype::Timedelta => "timedelta",
            Dtype::Floatlist => "floatlist",
            Dtype::Intlist => "intlist",
            Dtype::Stringlist => "stringlist",
            Dtype::Unsupported => "unsupported",
        }
    }
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
    /// What the core resolved of an accepted query (contract v28, T3.8),
    /// which a host reads instead of resolving the query again: its facts
    /// and its text as the core writes it back. Absent for a refused query
    /// and from a core that does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<QueryFacts>,
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
    /// The tables to describe, in order, each by the name the schema gives
    /// it and the name the source holds it under; empty, every table the
    /// source lists, each under its own name (decision E59).
    pub tables: Vec<NamedTable>,
    /// Declared primary keys, by the schema's name of the table: each its
    /// columns in order, `null` naming none (refused as an empty key), or
    /// `unreadable` for a value the host could not write as an ordered
    /// sequence of names (the conformance rule on [`Given`]), which the
    /// core refuses (decision E101).
    pub primary_keys: BTreeMap<String, Given<Option<Vec<String>>>>,
    /// Declared time columns, by the schema's name of the table: a name,
    /// `null` saying the table has none, as the caller's `None` does
    /// (decision E61), or `unreadable` for a value that is not a name.
    pub time_columns: BTreeMap<String, Given<Option<String>>>,
    /// The relationships given, each `[foreign key, referenced key]` by the
    /// schema's names, in place of the declared and inferred ones; an empty
    /// list says there are none. Left out, they are the source's declared
    /// foreign keys and the links inferred by name (decision E59).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationships: Option<Vec<Vec<ColumnRef>>>,
}

/// A table named for the schema: the name a query uses, and the relation
/// of the source it reads (design choice 12), which a host naming a table
/// by a string gives as one part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NamedTable {
    pub name: String,
    pub source: Relation,
    /// The columns the schema keeps of the table (contract v28, T3.8), as a
    /// schema built column by column names them: only these are read and
    /// described, as the reference reads only the columns a schema
    /// describes (`_copy_for_one_gather`). Left out, every column the
    /// source holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<Vec<String>>,
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
    /// The relationships, each `[foreign key, referenced key]`, once: in a
    /// description `describe_source` answers, the declared foreign keys and
    /// the links inferred by name that can join, in that order (decision
    /// E55); in `materialize`'s schema, the host's.
    pub links: Vec<Vec<ColumnRef>>,
    /// The namespace the tables were read under, each part given, where
    /// the run read parts the provider left to the connection (a
    /// Databricks source naming no catalog or schema: the session's
    /// current ones; contract v28, T3.4): `materialize` reads the same
    /// tables under it (a host keeps it for each table it described,
    /// [`HeldSchema::read_under`]), whatever the session's current parts
    /// are by then, as the reference keeps the tables it described. Left
    /// out, none was read, and each table is read under the provider's
    /// own namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribedTable {
    pub name: String,
    /// The relation of the source this table reads (design choice 12): its
    /// own name, one part, unless the caller named it otherwise (decision
    /// E59).
    pub source: Relation,
    pub columns: Vec<DescribedColumn>,
    pub primary_key: Vec<String>,
    pub time_column: Option<String>,
    /// The column inference took as the time column, when the table's
    /// time column is now another or none: a lone stamp of the last edit a
    /// link then refused (decision E55), or the column inference took where
    /// the caller named another (decision E59). `None` when the time column
    /// is inference's, or inference took none. The host's schema was
    /// ordered by it first, as the reference's was; `materialize` does not
    /// read it.
    pub time_column_inferred: Option<String>,
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
    /// What the column means in the schema as it stands: `ID` for a column
    /// the schema joins on, its primary key or a link's foreign key
    /// (decision E55).
    pub stype: Stype,
    /// What the column meant before the schema's joining on it made it an
    /// identifier, given back when the schema stops joining on it; `None`
    /// when joining on it changed nothing, or it is not joined on. In a
    /// description `describe_source` answers, the core's; in `materialize`'s
    /// schema, what the host holds, from which the core marks the columns
    /// joined on again (decision E55).
    pub stype_before_key: Option<Stype>,
    /// Whether the column may hold no value, as the catalog declares it:
    /// neither declared `NOT NULL` nor part of the declared primary key.
    pub nullable: bool,
    /// Whether the catalog declares a primary or unique key of this one
    /// column, which the source refuses to repeat.
    pub unique: bool,
    /// The column's type as the source's catalog declares it (`describe`
    /// columns' `type`, as the source spells it), kept only for a
    /// big-integer column the host layer read from an opaque extension type
    /// (a HUGEINT, UHUGEINT or VARINT, marked [`crate::data::BIG_INTEGER_MARKER`]):
    /// a key the data layer joins a `decimal128(38, 0)` seed to in the key's
    /// own type, so `materialize` asks the source nothing more (register
    /// P1B-43). `None` for every other column, which joins as before part
    /// 3, and absent from the JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_type: Option<String>,
    /// Set for a column of numbers the host provider handed over as their
    /// exact decimal text (marked [`crate::data::EXACT_NUMBER_MARKER`]):
    /// whether its catalog declares it a key. The schema reads such a column
    /// by its role in the schema as it stands (T2.21): as its text
    /// (`received` `utf8`, `dtype` `string`) where the catalog declares it a
    /// key, or where the schema keys it or joins it and every column it
    /// joins to is such a column too; as a double (`float64`) otherwise.
    /// `None` for every other column, and absent from the JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_number: Option<ExactNumber>,
}

/// A column of numbers handed over as their exact text
/// ([`DescribedColumn::exact_number`]), by what its catalog declares of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExactNumber {
    /// The catalog declares it a key, the one column of a primary, unique
    /// or foreign key: always read as its text, as the reference reads it.
    DeclaredKey,
    /// Any other: its text while the schema keys or joins on it, as above.
    Other,
}

// --- edit_schema ------------------------------------------------------------------

/// Check one edit to a schema and make it, as the reference client's
/// setters check and make it (decision E67): the schema the host holds, as
/// plain data, and the edit; the result is the schema with the edit made,
/// or the refusal the reference gives at the assignment (contract v28,
/// T3.16: the schema alone, with what the source said of each column on
/// the column, never a description the core answered earlier).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditSchema {
    /// A bytes attachment of a [`HeldSchema`] as JSON: the host's schema as
    /// it stands.
    pub schema: AttachmentId,
    /// A bytes attachment of a [`SchemaEdit`] as JSON.
    pub edit: AttachmentId,
}

/// `edit_schema`'s result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SchemaEdited {
    /// A bytes attachment of a [`HeldSchema`] as JSON: the schema with the
    /// edit made, which the host holds from now on: each column's data
    /// type, meanings and source facts, each table's key, time column and
    /// repeated keys, and the links.
    pub schema: AttachmentId,
    /// The column the edit gave a meaning the caller chose (the reference's
    /// `Column.stype_declared`): the column whose meaning was assigned, or
    /// the time column named, which is assigned `timestamp`; none for any
    /// other edit.
    pub declared: Option<ColumnRef>,
}

/// A schema as a host holds it, as plain data (contract v28, T3.16): its
/// tables, columns, meanings, keys, time columns and links as they stand,
/// however the host came to hold them (described, built by hand, edited,
/// copied or unpickled), and, on each column the source was asked of,
/// what the source said of it ([`SourceFacts`]). `edit_schema` and
/// `materialize` read it as it is; `materialize` asks the source for the
/// facts of a column that carries none, as `describe_source` reads them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeldSchema {
    /// The host's tables as they stand, in its order, each by the name the
    /// schema gives it.
    pub tables: Vec<HeldTable>,
    /// The host's relationships as they stand, each
    /// `[foreign key, referenced key]`.
    pub links: Vec<Vec<ColumnRef>>,
    /// Where the schema read each relation it read under parts its provider
    /// left to the connection (a Databricks source naming no catalog or
    /// schema: [`SourceDescription::namespace`] of the description that read
    /// it), by relation, once each: `materialize` reads a table of that
    /// relation there again while the source's own parts agree, whatever
    /// the session's current parts are by then, as the reference keeps each
    /// table it read by its source name. A fact of the schema that read the
    /// relations, not of any table or column: a table adopted from another
    /// schema, or edited, is read where this schema read its relation, and
    /// a relation it never read where the source and its session say
    /// (contract v28, T3.4). Absent from the JSON when none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_under: Vec<ReadUnder>,
}

/// One relation a schema read, and the namespace it read it under
/// ([`HeldSchema::read_under`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadUnder {
    pub relation: Relation,
    /// Each part, outermost first, as the source answered it.
    pub namespace: Vec<String>,
}

/// One table as the host holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeldTable {
    pub name: String,
    /// The relation of the source it reads, as the host holds it (design
    /// choice 12).
    pub source: Relation,
    pub columns: Vec<HeldColumn>,
    pub primary_key: Vec<String>,
    pub time_column: Option<String>,
    /// Rows sharing a key value, as the host holds it
    /// ([`DescribedTable::duplicate_keys`]): a `u64` scalar, or none.
    /// `edit_schema` sets it as a key edit does; `materialize` does not
    /// read it. Absent from the JSON when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duplicate_keys: Option<Scalar>,
}

/// One column as the host holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeldColumn {
    pub name: String,
    /// The column's data type as the host names it, on the client's wire
    /// ([`Dtype::wire_name`]).
    pub dtype: String,
    pub stype: Stype,
    /// What the column meant before the schema joined on it, as the host
    /// holds it ([`DescribedColumn::stype_before_key`]).
    pub stype_before_key: Option<Stype>,
    /// What the source said of the column when it was described or last
    /// edited, as the core answered it; absent from the JSON for a column
    /// the source was never asked of (built by hand). A column with none,
    /// or with facts of another relation than its table's, is one whose
    /// facts `materialize` reads from the source, and which `edit_schema`
    /// edits by its data type, as the reference's setters do.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceFacts>,
}

/// What the source said of one column ([`DescribedColumn`]'s facts), which
/// a host keeps on the column and hands back whole (contract v28, T3.16).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceFacts {
    /// The relation of the source the facts were read from: a column of a
    /// table the host points at another relation since holds facts of none
    /// of it, and is read afresh, as one with none is.
    pub relation: Relation,
    /// The data type the source's catalog and values give the column
    /// ([`DescribedColumn::dtype`]), which tells a date from a time of day
    /// where the wire's name does not.
    pub dtype: Dtype,
    /// The type the provider handed over, after admission
    /// ([`DescribedColumn::received`]).
    #[serde(
        serialize_with = "crate::types::checked_column_type",
        deserialize_with = "crate::types::strict_column_type"
    )]
    pub received: ColumnType,
    /// [`DescribedColumn::nullable`].
    pub nullable: bool,
    /// [`DescribedColumn::unique`].
    pub unique: bool,
    /// [`DescribedColumn::declared_type`]; absent from the JSON when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_type: Option<String>,
    /// [`DescribedColumn::exact_number`]; absent from the JSON when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_number: Option<ExactNumber>,
}

/// One edit to a schema, as the caller made it to the host's schema, each
/// value as the contract writes it or [`Given::Unreadable`]: the host
/// passes every edit it is given, and the core takes or refuses it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "edit", rename_all = "snake_case", deny_unknown_fields)]
pub enum SchemaEdit {
    /// A table's primary key set (`schema[table].primary_key = key`), its
    /// columns in order, or cleared (`null`).
    PrimaryKey {
        table: String,
        key: Given<Option<Vec<String>>>,
    },
    /// A table's time column set (`schema[table].time_column = column`),
    /// or cleared (`null`).
    TimeColumn {
        table: String,
        column: Given<Option<String>>,
    },
    /// A column's meaning set (`schema[table][column].stype = stype`), as
    /// the text the caller gave.
    Stype {
        table: String,
        column: String,
        stype: Given<String>,
    },
    /// A column left out (`del schema[table][column]`).
    LeaveOut {
        table: String,
        column: Given<String>,
    },
    /// The relationships replaced (`schema.relationships = links`), or
    /// edited as a list, as the list stands after the edit; each
    /// `[foreign key, referenced key]`.
    Relationships { links: Vec<Given<Vec<ColumnRef>>> },
}

/// A value a caller gave an edit or `describe_source`: the value, as the
/// contract writes it, or a value of the host's own no contract type holds
/// (a key named by a number, a set of names, a relationship that is not
/// one), which the core refuses as the reference refuses it.
///
/// The conformance rule every host holds to (decision E101): the host's
/// null or none is `{"value": null}`; text is the value; a finite, ordered
/// collection read once without error, every member non-null text, is a
/// list of those texts; a collection with a null, non-text or nested
/// member, a collection with no order, a mapping, binary data, a number, a
/// boolean, a temporal value or anything else is `unreadable`. Text the
/// host cannot encode as UTF-8 is the host's encoding refusal, and an
/// error raised while reading a collection is the caller's, never
/// `unreadable`. Whether a value is valid (an empty key, a column the
/// table lacks) is the core's call, never the host's.
///
/// A value no contract type holds may instead be given `{"spelled":
/// "<text>"}`, the text the host's language shows it as (in Python, its
/// `repr`): the core takes it as `unreadable` and names it in its refusal
/// (`given`), as the reference names it, a value of the caller's own call
/// (KO2-86, T2.14b).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Given<T> {
    Value(T),
    Unreadable,
    Spelled(String),
}

impl<T> Given<T> {
    /// The value, where the host could write it as the contract's type.
    pub fn value(&self) -> Option<&T> {
        match self {
            Given::Value(value) => Some(value),
            Given::Unreadable | Given::Spelled(_) => None,
        }
    }

    /// The host's text of a value no contract type holds, where it gave
    /// one.
    pub fn spelled(&self) -> Option<&str> {
        match self {
            Given::Spelled(text) => Some(text),
            Given::Value(_) | Given::Unreadable => None,
        }
    }
}

// --- materialize and predict --------------------------------------------------

/// One candidate of a ranking as a host gives it (`Materialize.candidates`,
/// T2.8), each in one of these forms:
///
/// - `{"value": <scalar>}`: the host's value as the contract's scalar of
///   its own type, a missing value `null`.
/// - `{"integer": "<digits>"}`: an integer wider than an integer scalar
///   holds, by its digits (`-` before a negative one).
/// - `{"decimal": "<digits>"}`: an exact decimal of any precision, by its
///   fixed-point digits (`-12.50`), its scale the digits after the point.
/// - `{"spelled": "<text>"}`: a value of a type no scalar holds, by the
///   text the reference writes into its question (in Python, `str` of it);
///   it is no text candidate, and the core holds it equal to no other.
/// - `{"typed": {"text": "<text>", "type": "<name>"}}`: a value of a type
///   no scalar holds, written into its question by its text as `spelled`
///   is, with the name of the kind of value it is (`date`, `datetime` for a
///   timestamp with no zone, `datetime_zoned` for one with one, `time`).
///   Two of one kind and one text are one candidate; a host whose language
///   holds others equal (two zones of one instant) says so with `repeat`
///   (T2.14b).
/// - `repeat`: a value the host's own language holds equal to one given
///   before it, left out as the reference's `dict.fromkeys` leaves it out.
///   A host whose candidates are all scalars sends none, the core keeping
///   each distinct value once by its own equality.
/// - `unreadable`: a value the host cannot look up by value (in Python,
///   one it cannot hash), which names no row and which the core refuses by
///   name, as the reference fails on it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum GivenCandidate {
    Value(Scalar),
    Integer(String),
    Decimal(String),
    Spelled(String),
    Typed(TypedCandidate),
    Repeat,
    Unreadable,
}

/// A candidate of a type no scalar holds, by its text and the name of its
/// kind (`GivenCandidate::Typed`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TypedCandidate {
    pub text: String,
    #[serde(rename = "type")]
    pub kind: TypedKind,
}

/// The kinds a typed candidate is of, a closed set: a kind no host may
/// name otherwise, an unknown one refused by the contract's reader by
/// name.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TypedKind {
    Date,
    Datetime,
    DatetimeZoned,
    Time,
}

/// Build a request for a query over a described source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Materialize {
    pub provider: ProviderChoice,
    /// A bytes attachment of a [`HeldSchema`] as JSON: the schema to
    /// sample, as the host holds it (contract v28, T3.16).
    pub schema: AttachmentId,
    /// The query text: a bytes attachment of a JSON string.
    pub query: AttachmentId,
    /// The entities to predict for: an Arrow attachment of one column.
    pub indices: Option<AttachmentId>,
    /// The rows a ranking (`RANK TOP k`) is over, as the caller named them
    /// (contract v27, optional, T2.8): a bytes attachment of a JSON list of
    /// [`GivenCandidate`]s, each candidate as the host's value is written,
    /// so values of several types keep each its own. The core asks a
    /// ranking as one question per distinct candidate, writing each into
    /// its question by the type of the column ranked, and refuses
    /// candidates on any other question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<AttachmentId>,
    /// The anchor time, a `ts` scalar.
    pub anchor_time: Option<Scalar>,
    /// The sampling seed, an integer of any size.
    pub seed: Option<Seed>,
    /// Whether the caller asked for an unseeded sample, a new one each
    /// call (`random_seed=None`), with `seed` left out; from contract v27
    /// the core draws it under a seed of its own, new each call (T2.11).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unseeded: bool,
    /// Whether the caller anchored each entity at its own time
    /// (`anchor_time='entity'`), with `anchor_time` left out; from contract
    /// v27 taken for a static query (T2.11).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub anchor_entity: bool,
    /// A bytes attachment of a JSON object of the call's option values the
    /// core reads (T1.28a), each as the caller gave it and the host's JSON
    /// writes it, a member only where the call names the option:
    /// `num_context_samples`, the context size, whose run mode the core
    /// chooses and bounds; `num_hops` and `num_neighbors`, whose hops it
    /// bounds before it refuses them by name; `prediction_statistic`, which
    /// it checks and records; `inference_config`, which it reads by the
    /// engine's rules and writes into the payload (decision E45(a)); and
    /// `random_seed`, in place of `seed` and `unseeded` where it is given
    /// (`null`, an unseeded sample); from contract v26, `rows_per_call`,
    /// the most rows one model call scores, which only lowers the profile's
    /// (`main`'s multi-call `predict`, read from T2.8c); `top_k`, how
    /// many candidates a ranking keeps, in place of the query's `TOP k`,
    /// read from T2.8; from contract v27, `lag_timesteps`, the past steps
    /// of a temporal target added as features (T2.9), any value, which the
    /// core refuses where the reference's engine refuses it, once the
    /// context is labelled; and from contract v27 the sampling options
    /// (T2.11): `num_hops` and `num_neighbors`, the hops walked and each
    /// one's size; `max_pq_iterations`; `use_prediction_time`;
    /// `return_embeddings`, which names `embeddings` among the payload's
    /// outputs; `exclude_cols_dict`, as given; and `context_anchor_time`,
    /// written as `anchor_time` is (a `ts` scalar), or as the text
    /// `entity`, `null`, or the text `unreadable` for any other value. The core reads it with `sdm-wire`'s
    /// bounded reader and holds each value's type to what the reference's
    /// client takes, so the host checks none of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<AttachmentId>,
    /// The names of the call's other sampling options, which this message
    /// carries no value of (none from contract v27: every sampling option
    /// travels in `values`): the core refuses any name it is given as
    /// `INVALID_REQUEST` (decision E61). The host
    /// passes each the call gave, so it refuses none of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    /// The most steps a forecast is asked as, one call each, which the host
    /// lets its caller set (the client's `decompose.MOST_PARTS`); 200 where
    /// it names none. A forecast of more is refused, `INVALID_REQUEST`, rule
    /// `too_many_parts` (T1.36 fix 21).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub most_parts: Option<u32>,
    /// The most entities one `materialize` gathers, across the calls it is
    /// asked in, which the host lets its caller set (`main`'s
    /// `MOST_ENTITIES`); 100,000 where it names none (design choice 1,
    /// multi-call `predict`, read from T2.8c, which asks a request of more
    /// rows than one call scores in several calls; until then the core
    /// refuses more than one call's rows, a lower bound than either).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub most_entities: Option<u64>,
    /// The most model calls one request is asked in (`main`'s
    /// `MOST_CALLS`); 100 where the host names none (read from T2.8c, as
    /// `most_entities`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub most_calls: Option<u32>,
    /// The sampler state the materializer's last run left, which the host
    /// hands back as it took it (decision E52); none for a materializer's
    /// first run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler_state: Option<SamplerState>,
    /// Whether the host takes the run's steps that ask nothing
    /// ([`crate::run::Step::GraphBuilt`], [`crate::run::Step::TablesInitialized`]),
    /// to keep the schema's engine reading as the reference keeps it
    /// (contract v28, T3.4). Left out, `false`: the run says neither.
    #[serde(default, skip_serializing_if = "crate::data::is_false")]
    pub progress: bool,
}

/// The version of [`SamplerState`]'s format this build writes and reads;
/// it is independent of the contract's. A state of another version is
/// refused by name before anything of it is read.
pub const SAMPLER_STATE_VERSION: u32 = 1;

/// What a materializer's runs have read of its source, so a later run
/// skips a read the reference's sampler makes once (decision E52): the
/// core holds no state across runs. A `materialize` run leaves it, done or
/// failed, and the host takes it after the run ends (a `sampler_state`
/// message), keeps it on its materializer as it is, opaque, and hands it
/// back with that materializer's next `materialize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SamplerState {
    /// The version of this state's format, independent of the contract's.
    pub version: u32,
    /// A bytes attachment of a [`SamplerMemo`] as JSON.
    pub memo: AttachmentId,
}

/// [`SamplerState`]'s memo: what the runs read, and of what.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SamplerMemo {
    /// A fingerprint of the provider and source description the memo was
    /// made under; a memo of another is not read.
    pub source: String,
    /// Whether the sampler's check of the indexes its reads would use was
    /// made (and its warning given, if any).
    pub indexes_checked: bool,
    /// The time bounds read, at most one entry per table.
    pub bounds: Vec<MemoBounds>,
    /// One stored value of each windowed table's time column, read once
    /// where the provider compares a window's bounds with its stored text
    /// (SQLite's `_time_spelling`), at most one entry per table; left out
    /// where none was read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stored_times: Vec<MemoStoredTime>,
    /// Each table's rows, read once where the provider's sampling is sized
    /// by them ([`crate::data::ProviderCapabilities::sample_rows`],
    /// `neighbor_lookback`; contract v28), as the reference's warehouse
    /// samplers count every table's rows as they are built; left out where
    /// none was read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<MemoRows>,
}

/// One table's rows, as counted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoRows {
    /// The table, as the schema names it.
    pub table: String,
    pub rows: u64,
}

/// One table's stored time value, as read: the text of one non-null value
/// of its time column, or none where that value was not text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoStoredTime {
    /// The table, as the schema names it.
    pub table: String,
    pub value: Option<String>,
}

/// The smallest and largest time of one table's time column, as read, in
/// nanoseconds since the epoch; none where no row holds a time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MemoBounds {
    /// The table, as the schema names it.
    pub table: String,
    pub column: String,
    pub min: Option<i64>,
    pub max: Option<i64>,
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
    /// What the core resolved of the query it gathered (contract v26,
    /// T2.18), which a host reads instead of resolving the query itself:
    /// the core resolves a query naming any column (KO2-36), which a host's
    /// own validator may not. Absent from a core that does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<QueryFacts>,
}

/// The facts of a resolved query a host reads: plain values, no tree. Its
/// text, as the core writes it back, is the state's `query`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryFacts {
    /// The table whose rows are the entities.
    pub entity_table: String,
    pub problem_type: Option<crate::pql::ProblemType>,
    /// Whether the target carries a time range, so its answer moves with
    /// the moment it is asked at.
    pub windowed: bool,
    /// How many timeframes `FORECAST n TIMEFRAMES` asks for; 1 otherwise.
    pub num_forecasts: u32,
    /// The query as the core writes it back, the reference's
    /// `ValidatedPredictiveQuery.to_string()` (contract v28, T3.8): given
    /// by `validate_query`; absent from `materialize`'s, whose state holds
    /// each question's text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// The version of [`RequestState`]'s format this build writes and reads
/// (decision H23); it is independent of the contract's.
/// Version 2 (contract v26, design choice 1): the state of a request asked
/// in several model calls ([`RequestState::batches`]).
pub const REQUEST_STATE_VERSION: u32 = 2;

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
    /// A forecast (`FORECAST n TIMEFRAMES`), which the core asks as one
    /// regression per step, the window moved a span along at each (T1.36
    /// fix 21): `payload` and `query` are step 1's, and this holds every
    /// later step's, in order, empty for a forecast of one step. Absent for
    /// any other request. `predict` sends each step's payload and gives the
    /// answers back as one, each row carrying its step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forecast: Option<Vec<ForecastStep>>,
    /// A request of more rows than one model call scores, asked in several
    /// calls (`main`'s multi-call `predict`, design choice 1, T2.8c):
    /// `payload` is the first call's, and this holds every later call's, in
    /// order, each scoring the rows after the last; `predict` numbers each
    /// call's rows on from where the previous call's ended and gives the
    /// answers back as one. Absent for a request of one call; a forecast or
    /// ranking, already one call per step or candidate, is never batched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batches: Option<Vec<RequestBatch>>,
    /// A ranking (`RANK TOP k`), which the core asks as one binary question
    /// per distinct candidate, whether the entity reaches it in the window
    /// (the client's `_per_candidate`, contract v27, optional, T2.8):
    /// `payload` and `query` are the first candidate's question, and this
    /// holds every later one's, in order. Absent for any other request.
    /// `predict` asks each and gives the answers back as one ranking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranking: Option<Ranking>,
}

/// A ranking's request state ([`RequestState::ranking`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Ranking {
    /// How many candidates each row keeps, best first (the call's `top_k`,
    /// else the query's `TOP k`); none keeps every candidate scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The first candidate's position among the candidates `materialize`
    /// was given (0, but where the host held the first a repeat); its
    /// question is the state's own. Each later part's position is past it,
    /// in increasing order, or `predict` refuses the state.
    pub candidate: u32,
    /// The first candidate as the result's `rankings` names it, the JSON
    /// text of its value.
    pub value: String,
    /// The question of each later candidate, in the order the candidates
    /// were named.
    pub parts: Vec<RankingPart>,
}

/// One later candidate's question of a ranking's request state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RankingPart {
    /// The candidate this question asks after: its position among the
    /// candidates `materialize` was given, its first where it was named
    /// more than once.
    pub candidate: u32,
    /// The candidate as the result's `rankings` names it: the JSON text of
    /// its value.
    pub value: String,
    /// A bytes attachment of this question's payload as JSON, as
    /// [`RequestState::payload`].
    pub payload: AttachmentId,
    /// This question's text: a bytes attachment of a JSON string.
    pub query: AttachmentId,
    /// The id this question's call carries, as [`ForecastStep::request_id`];
    /// `materialize` writes none, and `predict` refuses a later candidate's
    /// question without one (`INVALID_REQUEST`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// One later call of a request asked in several ([`RequestState::batches`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestBatch {
    /// A bytes attachment of this call's payload as JSON, as
    /// [`RequestState::payload`].
    pub payload: AttachmentId,
    /// The rows this call scores.
    pub rows: u64,
    /// The id this call carries in its `metadata.request_id`, which the
    /// host draws for each call, as [`ForecastStep::request_id`];
    /// `materialize` writes none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// One later step of a forecast's request state: its number, from 2, and
/// its payload and query, as [`RequestState`]'s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastStep {
    pub step: u32,
    pub payload: AttachmentId,
    pub query: AttachmentId,
    /// The id this step's call carries in its `metadata.request_id`, which
    /// the host draws for each step as it draws [`Predict::request_id`] for
    /// step 1's, one per call as the reference's client draws them (T1.36
    /// fix 24). `materialize` writes none; `predict` refuses a later step
    /// without one (`INVALID_REQUEST`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
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
    /// The outputs the caller asked for, by the wire's names, in order,
    /// as given (a name repeated, kept), which the core checks against the
    /// model as the client's `_prepare` does: an empty list refused
    /// (`INVALID_REQUEST`), a name the model does not produce refused
    /// (`UNSUPPORTED_REQUEST`). Empty when [`Predict::default_outputs`].
    pub outputs: Vec<String>,
    /// The caller named no outputs: the core asks the task's default ones
    /// (a regression's `prediction`, a classification's `prediction` and
    /// `probabilities`), and gives no notice of one the answer leaves out,
    /// which it gives only for outputs the caller named (T1.28b).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub default_outputs: bool,
    /// The inference-config keys the caller set at `materialize`, rather
    /// than left to their defaults, as the host holds them: a key the model
    /// does not read among them is refused before the call is made, the
    /// others' defaults left out of it silently (T1.28a).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inference_asked: Vec<String>,
}

/// The prediction. Nothing inline grows with the result's width or the
/// endpoint's answer (decision H5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Predicted {
    /// An Arrow attachment, one row per answer, ordered by `row_index` as
    /// the reference's result frame orders them (T1.28b), whose schema
    /// names its columns in order: `row_index`, `id`, `id_as_written` where
    /// the ids are not all integers or all text (utf8: the JSON text of the
    /// reference's item value, null where there was none), `forecast_step`
    /// where an answer given as written carries one, `prediction` where an
    /// answer carries one, `prediction_as_answered` where the prediction
    /// column changed an answer (utf8: the JSON text of the reference's item
    /// value, null where there was none; present only for a regression's
    /// integer widened among doubles or a class written as a non-finite
    /// number), `probabilities` where an answer carries them, and
    /// `answer_index`, each answer's place in the endpoint's order
    /// (`sdm_request::predict::table`).
    ///
    /// Contract v26 fixes the columns ranking and forecasting add (design
    /// choice 1; written from T2.8 and T2.9), after `probabilities` and
    /// before `answer_index`, each where an answer carries it: `scores`
    /// (list of float64, an item's scores in the endpoint's order),
    /// `rankings` (list of utf8, an item's rankings, each a mapping), and
    /// `quantiles` (a regression's quantiles, each named as the endpoint
    /// named it, in the order written; from contract v27, T2.9, a
    /// mapping-valued field, laid out as the next paragraph says, an
    /// answer's `{}` the text `{}`, none null).
    ///
    /// A mapping-valued field (T2.8, contract v27) is laid out as JSON
    /// object text in a Utf8 column, and a list of mappings as a list of
    /// those texts: a JSON object when every key is text, otherwise (a key
    /// that is a number, a boolean or null) the JSON list of its `[key,
    /// value]` pairs, as a JSON object holds no other key. Each is read
    /// from the answer as Python's `dict` reads it, keys equal as Python's
    /// are, by the one reader `sdm_wire::response::mapping`.
    ///
    /// A ranking's result (T2.8, contract v27), one row per entity asked
    /// about, in place
    /// of `prediction` and `probabilities`: `rankings` (the candidates the
    /// row ranks, best first, at most `top_k`, each the JSON text of its
    /// value, as the state names it), `scores` (how sure the endpoint was
    /// of each, in the same order) and `candidates` (list of int64, each
    /// one's position among the candidates `materialize` was given, its
    /// first where it was given more than once, in the same order), which a
    /// host gives back as the outputs named in `outputs`, each candidate as
    /// the value it was given as, found by its position.
    ///
    /// Contract v27 adds, after those, `embeddings` (list of float64, an
    /// answer's embedding in the order written, null where it has none;
    /// T2.11).
    pub result: AttachmentId,
    /// The model that answered, as the endpoint names it.
    pub model: String,
    /// The endpoint's id for the call; empty for a request with no row to
    /// score, answered without a call.
    pub request_id: String,
    /// A bytes attachment of the endpoint's `metadata` object as JSON.
    pub metadata: AttachmentId,
    /// The outputs the call asked for, the task's default ones where the
    /// caller named none.
    pub outputs: Vec<String>,
    /// A ranking's candidates no question was sent for, the context sampled
    /// for each holding one answer only (T2.8): each its position among the
    /// candidates `materialize` was given, in the order named. Empty for
    /// any other result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unscored_candidates: Vec<u32>,
}

// --- attachments ----------------------------------------------------------------

/// What each bytes attachment holds that a control message names and the
/// core reads or writes as JSON, by `<type>.<field>` of the field naming
/// it: part of the locked schema (`$defs/Attachments`, put there by
/// [`crate::schema_json`]), so a change to an attachment's shape needs a
/// version bump as a change to a control message's does. A model call's
/// body and the endpoint's answer are the endpoint's bytes, and an Arrow
/// attachment carries its own schema; neither is listed.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Attachments {
    #[schemars(rename = "ValidateQuery.query")]
    pub validate_query_query: String,
    #[schemars(rename = "ValidateQuery.schema")]
    pub validate_query_schema: QuerySchema,
    #[schemars(rename = "DescribeSource.source")]
    pub describe_source_source: SourceSpec,
    #[schemars(rename = "DescribedSource.description")]
    pub described_source_description: SourceDescription,
    #[schemars(rename = "EditSchema.schema")]
    pub edit_schema_schema: HeldSchema,
    #[schemars(rename = "EditSchema.edit")]
    pub edit_schema_edit: SchemaEdit,
    #[schemars(rename = "SchemaEdited.schema")]
    pub schema_edited_schema: HeldSchema,
    #[schemars(rename = "Materialize.schema")]
    pub materialize_schema: HeldSchema,
    #[schemars(rename = "Materialize.query")]
    pub materialize_query: String,
    #[schemars(rename = "Materialize.values")]
    pub materialize_values: CallValues,
    #[schemars(rename = "Materialize.candidates")]
    pub materialize_candidates: Vec<GivenCandidate>,
    #[schemars(rename = "SamplerState.memo")]
    pub sampler_state_memo: SamplerMemo,
    #[schemars(rename = "RequestState.payload")]
    pub request_state_payload: StatePayload,
    #[schemars(rename = "RequestState.query")]
    pub request_state_query: String,
    #[schemars(rename = "ForecastStep.payload")]
    pub forecast_step_payload: StatePayload,
    #[schemars(rename = "ForecastStep.query")]
    pub forecast_step_query: String,
    #[schemars(rename = "RequestBatch.payload")]
    pub request_batch_payload: StatePayload,
    #[schemars(rename = "RankingPart.payload")]
    pub ranking_part_payload: StatePayload,
    #[schemars(rename = "RankingPart.query")]
    pub ranking_part_query: String,
    #[schemars(rename = "Predicted.metadata")]
    pub predicted_metadata: EndpointMetadata,
}

/// [`RequestState`]'s payload: the prediction request as JSON, as
/// `materialize` writes it (`sdm_request::payload`) and `predict` reads it
/// (`sdm_request::predict`), at the state's version, which the schema
/// records as `x-request-state-version`.
///
/// Its definition is the schema `sdm-wire`'s typed payload
/// (`sdm_wire::payload::StatePayload`, which the writer serializes)
/// derives, committed as `schema/state-payload.json` and embedded here
/// ([`STATE_PAYLOAD_SCHEMA`]); `sdm-wire`'s `tests/state_payload_schema.rs`
/// holds the file to what the types generate, so the payload has one
/// definition and a change to it fails the lock until the version moves
/// (T1.28w, decision E74). What is fixed is closed (every object whose keys
/// the format names refuses a key it does not name) and what is open is the
/// data (the tables named by the source, and each row's cells).
/// `sdm-request`'s tests hold the payload its producer builds, and
/// `sdm-testkit`'s the payload of every case `rust-host` materializes, to
/// this definition.
pub struct StatePayload;

/// The schema `sdm-wire`'s typed payload derives, as committed.
pub const STATE_PAYLOAD_SCHEMA: &str = include_str!("../schema/state-payload.json");

impl JsonSchema for StatePayload {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "StatePayload".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut payload: serde_json::Value = serde_json::from_str(STATE_PAYLOAD_SCHEMA)
            .expect("the committed state payload schema is JSON");
        payload["x-request-state-version"] = serde_json::json!(REQUEST_STATE_VERSION);
        schemars::Schema::try_from(payload).expect("the state payload's schema is an object")
    }
}

/// [`Materialize`]'s `values`: the call's option values, a JSON object of
/// the options' names and the values given, read with `sdm-wire`'s bounded
/// reader.
pub struct CallValues;

impl JsonSchema for CallValues {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "CallValues".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "object" })
    }
}

/// [`Predicted`]'s metadata: the endpoint's `metadata` object, as it gave
/// it.
pub struct EndpointMetadata;

impl JsonSchema for EndpointMetadata {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "EndpointMetadata".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "object" })
    }
}
