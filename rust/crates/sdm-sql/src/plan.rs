// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A built-in source's SQL as its kit writes it: the source a host names
//! ([`SqlChoice`]), and the statements the kit hands the host's driver to
//! run for one fetch ([`SqlPlan`]). Both are written as JSON to a binding:
//! a host names its source by a [`SqlChoice`] and runs each [`SqlPlan`] it
//! is handed. The core sees neither: it sends a built-in source the typed
//! requests it sends every source.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use sdm_contract::Scalar;
use sdm_contract::data::{AttachmentId, ExpectedSchema, KeySet, ViewChoice};

/// A built-in SQL dialect, by the name the host gives it. Which names a
/// build serves is the kit's, which refuses one it does not serve when it
/// is made.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Dialect(pub String);

impl Dialect {
    /// The dialect named `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Dialect(name.into())
    }

    /// The dialect's name, as the host gave it.
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// A built-in source, as its host names it: its dialect, where its tables
/// are, the semantic model it is read through, and the timestamp transport
/// its driver needs.
///
/// `namespace` is where the source's tables are, as the source was given
/// it: the name parts before a table's own, outermost first, a Postgres
/// source's schema (`PostgresSource(schema=...)`), each `null` where the
/// source names none, which is then the connection's current one. Every
/// relation of one part a request names is read under it, and the source's
/// tables are listed in it, as the reference qualifies each read by the
/// schema it was given (`PostgresTable`). Left out, empty or all `null`,
/// the connection's current one. A dialect takes as many parts as a
/// relation's parts before its table's, and refuses by name more, or a
/// part given inside one left `null`.
///
/// `view` names the semantic model the source is read through, a
/// Databricks metric view or a Snowflake semantic view by its dialect, as
/// the caller named it, with whether the caller trusts its expressions: its
/// logical tables are then the relations of one part a request names, each
/// the logical table's name. Left out, the source's own tables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "ChoiceWire", into = "ChoiceWire")]
pub struct SqlChoice {
    pub dialect: Dialect,
    pub namespace: Vec<Option<String>>,
    pub view: Option<ViewChoice>,
    /// A host's required lossless transport for timestamps that its
    /// driver cannot preserve through native getters. The kit owns the SQL
    /// projection; the host performs only the conversion explicitly named
    /// by the resulting plan.
    pub timestamp_encoding: Option<SqlTimestampEncoding>,
}

/// [`SqlChoice`] as JSON writes it, tagged `"provider": "sql"`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
enum ChoiceWire {
    Sql {
        dialect: Dialect,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        namespace: Vec<Option<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        view: Option<ViewChoice>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timestamp_encoding: Option<SqlTimestampEncoding>,
    },
}

impl From<ChoiceWire> for SqlChoice {
    fn from(wire: ChoiceWire) -> Self {
        let ChoiceWire::Sql {
            dialect,
            namespace,
            view,
            timestamp_encoding,
        } = wire;
        SqlChoice {
            dialect,
            namespace,
            view,
            timestamp_encoding,
        }
    }
}

impl From<SqlChoice> for ChoiceWire {
    fn from(choice: SqlChoice) -> Self {
        ChoiceWire::Sql {
            dialect: choice.dialect,
            namespace: choice.namespace,
            view: choice.view,
            timestamp_encoding: choice.timestamp_encoding,
        }
    }
}

impl SqlChoice {
    /// The built-in source of `dialect`, over the connection's current
    /// namespace.
    pub fn new(dialect: impl Into<String>) -> Self {
        SqlChoice {
            dialect: Dialect::new(dialect),
            namespace: Vec::new(),
            view: None,
            timestamp_encoding: None,
        }
    }
}

/// A driver's requested lossless SQL timestamp transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqlTimestampEncoding {
    /// Timestamps as signed microseconds: timezone-free wall fields from
    /// the epoch's wall fields, or a zoned timestamp's physical instant
    /// from the Unix epoch. The kit selects the source kind from actual
    /// warehouse result metadata, independent of the session or JVM zone.
    Microseconds,
}

/// The timestamp meaning used by the kit's SQL projection, retained
/// through key-set binding and chunk re-rendering. A host transfers the
/// projected integer unchanged for either kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqlTimestampKind {
    /// A timestamp's physical instant, measured from the Unix epoch.
    Instant,
    /// A timezone-free timestamp's civil fields, measured from the
    /// epoch's wall fields without assigning any local timezone.
    WallTime,
}

/// A native result conversion the kit selects.
/// Column positions are zero-based and names remain exactly the
/// warehouse's original output names, including duplicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "conversion", rename_all = "snake_case", deny_unknown_fields)]
pub enum SqlResultConversion {
    /// PostgreSQL zoned values read as text, including array elements and
    /// range bounds, use canonical UTC without changing the session.
    /// This policy applies to the whole result, not one projection.
    ZonedTextUtc,
    TimestampMicros {
        column: u32,
        name: String,
        source_kind: SqlTimestampKind,
    },
}

/// A statement's parameter: a scalar, or a key set the statement's text
/// reads under `relation`. How the keys come to be read by that name is
/// the dialect's: the DuckDB driver registers the attachment as a relation
/// of that name; a PostgreSQL dialect binds the keys as JSON text that a
/// common table expression of that name unpacks, as the reference's
/// PostgreSQL sampler does. `relation` names what the text reads, not an
/// instruction to register an Arrow relation. A dialect that binds its key
/// sets as JSON text names none: the kit turns each into a `text`
/// parameter when it binds the plan.
///
/// A `text` parameter binds the one text cell of a one-row, one-column
/// Arrow batch: a key set the dialect writes as JSON text itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "param", rename_all = "snake_case", deny_unknown_fields)]
pub enum SqlParam {
    Value { value: Scalar },
    KeySet { keys: KeySet, relation: String },
    Text { attachment: AttachmentId },
}

/// One statement a built-in source's driver runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SqlStatement {
    pub dialect: Dialect,
    pub text: String,
    pub params: Vec<SqlParam>,
}

/// The statements one fetch runs, in order on one session, the last one's
/// rows the answer, with the Arrow schema that answer has, and the result
/// conversions the driver applies to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "PlanWire", into = "PlanWire")]
pub struct SqlPlan {
    pub statements: Vec<SqlStatement>,
    pub expect: ExpectedSchema,
    pub result_conversions: Vec<SqlResultConversion>,
}

/// [`SqlPlan`] as JSON writes it, tagged `"plan": "sql"`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "plan", rename_all = "snake_case", deny_unknown_fields)]
enum PlanWire {
    Sql {
        statements: Vec<SqlStatement>,
        expect: ExpectedSchema,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        result_conversions: Vec<SqlResultConversion>,
    },
}

impl From<PlanWire> for SqlPlan {
    fn from(wire: PlanWire) -> Self {
        let PlanWire::Sql {
            statements,
            expect,
            result_conversions,
        } = wire;
        SqlPlan {
            statements,
            expect,
            result_conversions,
        }
    }
}

impl From<SqlPlan> for PlanWire {
    fn from(plan: SqlPlan) -> Self {
        PlanWire::Sql {
            statements: plan.statements,
            expect: plan.expect,
            result_conversions: plan.result_conversions,
        }
    }
}

impl SqlPlan {
    /// Each batch the plan's statements bind, a key set's or a text's, that
    /// `to` names, named by what `to` maps it to.
    pub fn rename_attachments(&mut self, to: &BTreeMap<AttachmentId, AttachmentId>) {
        for param in self.statements.iter_mut().flat_map(|s| &mut s.params) {
            let named = match param {
                SqlParam::KeySet { keys, .. } => &mut keys.attachment,
                SqlParam::Text { attachment } => attachment,
                SqlParam::Value { .. } => continue,
            };
            if let Some(renamed) = to.get(named) {
                *named = *renamed;
            }
        }
    }

    /// The batches the plan's statements bind, key sets and texts, in
    /// order, each as often as it is bound.
    pub fn key_sets(&self) -> Vec<AttachmentId> {
        self.statements
            .iter()
            .flat_map(|statement| &statement.params)
            .filter_map(|param| match param {
                SqlParam::KeySet { keys, .. } => Some(keys.attachment),
                SqlParam::Text { attachment } => Some(*attachment),
                SqlParam::Value { .. } => None,
            })
            .collect()
    }
}
