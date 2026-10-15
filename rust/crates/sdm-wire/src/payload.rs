// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request state's payload as typed structures: the prediction request
//! as `materialize` writes it (`sdm_request::payload`), held in the request
//! state, and read by `predict`.
//!
//! These types are the one definition of the payload's shape. The writer
//! serializes them; the contract's locked definition of the payload
//! (`$defs/StatePayload`) is the schema they derive ([`schema_json`]),
//! committed as `sdm-contract`'s `schema/state-payload.json`, which
//! `sdm-contract` embeds and `tests/state_payload_schema.rs` holds equal to
//! what these types generate (T1.28w, decision E74). A field added to,
//! dropped from or retyped in them therefore changes the contract's schema,
//! which fails the contract lock until the version moves with it.
//!
//! What is fixed is closed: every object whose keys the format names
//! refuses a key it does not name. What is open is the data: the tables
//! named by the source (`related_tables`, a table's `columns`), kept in the
//! order written, and the cells of each row, any JSON value, each number
//! the token written ([`Value`]).
//!
//! The task and the inference options take one shape per kind of task
//! (T1.28a): a regression's ([`Task`] of a [`Target`], [`InferenceConfig`],
//! the defaults of [`StatePayload`]'s parameters) and a classification's
//! ([`Task`] of a [`ClassTarget`], with its classes and a binary task's
//! positive class, and [`ClassificationConfig`]). The schema is derived
//! once, over each kind ([`AnyTask`], each kind's task with its kind's name
//! and target; [`AnyConfig`]), so it names every member each writes and
//! refuses a member one kind writes missing from it.

use schemars::JsonSchema;
use sdm_contract::ops::Stype;
use serde::Serialize;
use serde::ser::SerializeMap;

use crate::json::Value;

/// Tables or columns by name, an object in the order written: each entry
/// written as the writer gave it, as the payload's objects were written
/// before they were typed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Named<T>(pub Vec<(String, T)>);

impl<T: Serialize> Serialize for Named<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, value) in &self.0 {
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

impl<T: JsonSchema> JsonSchema for Named<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Named_{}", T::schema_name()).into()
    }

    fn inline_schema() -> bool {
        true
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "additionalProperties": generator.subschema_for::<T>(),
        })
    }
}

/// The payload of one prediction request: a regression's by default, a
/// classification's with [`ClassTarget`] and [`ClassificationConfig`].
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields, rename = "StatePayload")]
pub struct StatePayload<K = Task, C = InferenceConfig> {
    /// The model the request was laid out for.
    pub model: String,
    pub task: K,
    pub schema: PayloadSchema,
    /// The examples with a label, and the rows sampled for them.
    pub context: Half,
    /// The examples to predict, and the rows sampled for them.
    pub predict: Half,
    pub output: Output,
    pub inference: Inference<C>,
    pub metadata: Metadata,
}

/// What is predicted, of which entity.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields, rename = "Task")]
pub struct Task<T = Target> {
    /// The task's kind, as the engine names it (`regression`,
    /// `binary_classification`, `multiclass_classification`).
    pub kind: String,
    pub target: T,
    pub entity_table_names: Vec<String>,
    /// The instance table's column holding each example's anchor time,
    /// where the examples have one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub anchor_time_column: Option<String>,
}

/// The label column of the instance table: a regression's.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Target {
    pub column_name: String,
    pub dtype: Dtype,
}

/// A classification's label column, with its classes in the order the
/// engine writes them (`_task_spec`): a binary task's `false` and `true`,
/// `true` its positive class; a multiclass task's labels in the order the
/// context first holds them, as text.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ClassTarget {
    pub column_name: String,
    pub dtype: Dtype,
    pub classes: Vec<String>,
    /// A binary task's positive class; a multiclass task names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub positive_class: Option<String>,
}

/// A column's type on the wire, as the reference's engine names it
/// (`_dtype_name`, `_list_dtype_name`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub enum Dtype {
    // Every integer type.
    #[serde(rename = "int64")]
    Int64,
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float64")]
    Float64,
    #[serde(rename = "string")]
    String,
    // A column of lists of text: a text feature sent as its words.
    #[serde(rename = "stringlist")]
    StringList,
    #[serde(rename = "bool")]
    Bool,
    // A time with no zone, written to the microsecond.
    #[serde(rename = "timestamp[us]")]
    TimestampMicros,
}

impl Dtype {
    /// The dtype's name on the wire.
    pub fn wire_name(self) -> &'static str {
        match self {
            Dtype::Int64 => "int64",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::String => "string",
            Dtype::StringList => "stringlist",
            Dtype::Bool => "bool",
            Dtype::TimestampMicros => "timestamp[us]",
        }
    }
}

/// The schema of the request's tables and the links between them.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PayloadSchema {
    pub instance_table: TableSchema,
    pub related_tables: Named<TableSchema>,
    pub relationships: Vec<Relationship>,
}

/// One table's schema: its columns, in order, and its key.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct TableSchema {
    pub columns: Named<ColumnSpec>,
    pub primary_key: PrimaryKey,
}

/// One column's spec.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ColumnSpec {
    pub dtype: Dtype,
    /// The column's semantic type, where the schema gives one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Stype")]
    pub stype: Option<Stype>,
    /// `false` for a column of the key; never written otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub nullable: Option<bool>,
}

/// A table's key: one column, or several.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum PrimaryKey {
    One(String),
    Several(Vec<String>),
}

/// A link from columns of one table to columns of another; the link from
/// the instance table names no source table.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Relationship {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub source_table: Option<String>,
    pub source_columns: Vec<String>,
    pub target_table: String,
    pub target_columns: Vec<String>,
}

/// One half of the examples: its instance table's rows and the rows
/// sampled for them, table by table.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Half {
    pub instance_table: Rows,
    pub related_tables: Named<Rows>,
}

/// A table's rows as `arrays` JSON: its columns, then one array of cells
/// per row, each cell any JSON value its column's dtype gives.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Rows {
    pub format: Format,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// How rows are written: `arrays` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub enum Format {
    #[serde(rename = "arrays")]
    Arrays,
}

/// The outputs asked for.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Output {
    pub fields: Vec<String>,
    /// A regression's statistic, where the caller chose one (T1.28a).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub prediction_statistic: Option<String>,
}

/// How the model is run.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields, rename = "Inference")]
pub struct Inference<C = InferenceConfig> {
    pub run_mode: String,
    pub use_prediction_time: bool,
    pub inference_config: C,
}

/// A regression's inference options, as the engine writes them
/// (`_inference_config_to_json`).
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct InferenceConfig {
    /// `regression`, the schema holds.
    #[schemars(with = "RegressionConfigKind")]
    pub kind: String,
    pub num_estimators: u64,
    pub column_shuffle: bool,
    pub category_shuffle: bool,
    pub hop_shuffle: bool,
    /// The reader's, not the writer's: a regression's options hold none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub class_shuffle: Option<bool>,
    pub target_transforms: Vec<String>,
    pub output_type: String,
}

/// A classification's inference options, as the engine writes them.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ClassificationConfig {
    /// `classification`, the schema holds.
    #[schemars(with = "ClassificationConfigKind")]
    pub kind: String,
    pub num_estimators: u64,
    pub column_shuffle: bool,
    pub category_shuffle: bool,
    pub hop_shuffle: bool,
    /// Always written; a reader may meet a state without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub class_shuffle: Option<bool>,
}

/// What the request was made from.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct Metadata {
    /// The query, as the reference writes it back; none when it is empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub source_query: Option<String>,
    pub num_context_examples: u64,
    pub num_prediction_examples: u64,
}

/// A task's kind on the wire, one name, for the schema only.
macro_rules! kind_name {
    ($name:ident, $schema:literal, [$($kind:literal),+]) => {
        #[doc = concat!("The kind `", $schema, "` names, for the schema only.")]
        pub struct $name;

        impl JsonSchema for $name {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                $schema.into()
            }

            fn inline_schema() -> bool {
                true
            }

            fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
                schemars::json_schema!({ "type": "string", "enum": [$($kind),+] })
            }
        }
    };
}

kind_name!(RegressionKind, "RegressionKind", ["regression"]);
kind_name!(BinaryKind, "BinaryKind", ["binary_classification"]);
kind_name!(RegressionConfigKind, "RegressionConfigKind", ["regression"]);
kind_name!(
    ClassificationConfigKind,
    "ClassificationConfigKind",
    ["classification"]
);
kind_name!(
    MulticlassKind,
    "MulticlassKind",
    ["multiclass_classification"]
);

/// A regression's task, for the schema only.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct RegressionTask {
    #[schemars(with = "RegressionKind")]
    pub kind: String,
    pub target: Target,
    pub entity_table_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub anchor_time_column: Option<String>,
}

/// A binary task's label column: its classes and positive class.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct BinaryTarget {
    pub column_name: String,
    pub dtype: Dtype,
    pub classes: Vec<String>,
    pub positive_class: String,
}

/// A multiclass task's label column: its classes, no positive class.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct MulticlassTarget {
    pub column_name: String,
    pub dtype: Dtype,
    pub classes: Vec<String>,
}

/// A binary task, for the schema only.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct BinaryTask {
    #[schemars(with = "BinaryKind")]
    pub kind: String,
    pub target: BinaryTarget,
    pub entity_table_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub anchor_time_column: Option<String>,
}

/// A multiclass task, for the schema only.
#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct MulticlassTask {
    #[schemars(with = "MulticlassKind")]
    pub kind: String,
    pub target: MulticlassTarget,
    pub entity_table_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub anchor_time_column: Option<String>,
}

/// What is predicted, of which entity: a task of each kind, its kind's
/// name and its target. The schema of [`Task`] of each target, for the
/// schema only.
#[derive(JsonSchema)]
#[serde(untagged)]
#[schemars(rename = "Task")]
pub enum AnyTask {
    Regression(RegressionTask),
    Binary(BinaryTask),
    Multiclass(MulticlassTask),
}

/// The engine's inference options, a regression's or a classification's,
/// each its own type, told apart by its kind: the schema of both, for the
/// schema only.
#[derive(JsonSchema)]
#[serde(untagged)]
#[schemars(rename = "InferenceOptions")]
pub enum AnyConfig {
    Regression(InferenceConfig),
    Classification(ClassificationConfig),
}

impl<K: Serialize, C: Serialize> StatePayload<K, C> {
    /// The payload's UTF-8 JSON text: compact, members in the order the
    /// types declare them, open maps in the order written, each cell's
    /// number its token.
    pub fn to_bytes(&self) -> Vec<u8> {
        match serde_json::to_vec(self) {
            Ok(bytes) => bytes,
            // Every member is text, a bool, an integer or a cell, each of
            // which serializes to memory.
            Err(_) => unreachable!("the payload serializes"),
        }
    }
}

/// The JSON Schema of [`StatePayload`] of every kind of task (over
/// [`AnyTask`] and [`AnyConfig`]), every definition inlined, as
/// committed in `sdm-contract`'s `schema/state-payload.json`:
/// pretty-printed, with a trailing newline, with no `$schema`, since the
/// contract holds it as one of its own definitions.
pub fn schema_json() -> String {
    let generator = schemars::generate::SchemaSettings::draft2020_12()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator();
    let mut schema = generator.into_root_schema_for::<StatePayload<AnyTask, AnyConfig>>();
    schema.remove("$schema");
    let mut text = serde_json::to_string_pretty(&schema).unwrap_or_default();
    text.push('\n');
    text
}
