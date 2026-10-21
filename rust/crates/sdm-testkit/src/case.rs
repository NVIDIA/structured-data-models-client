// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The case loader: `contract/cases/*.json`, in the format of
//! `contract/case.schema.json` (`sdm-case/1`). Every key of the format is
//! read, an unknown one refused, so the loader cannot drift from the schema
//! the Python runner validates against: what is required is required, null
//! only where the schema takes it, and the schema's `minItems`,
//! `minProperties` and `minLength` and `expect`'s `if`/`then`/`else` held
//! (T1.21b, T1.21c; `contract/tests/malformed-cases.json` holds both loaders
//! to the same changes). Its patterns and enumerations of text (an id, a
//! slice, a seam, a path or cell, a seed's digits) are the Python
//! validator's alone.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

/// The case format this loader reads.
pub const FORMAT: &str = "sdm-case/1";

/// One contract case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub format: String,
    pub id: String,
    pub status: Status,
    pub slice: String,
    /// The databases the case runs on when not all three (design choice
    /// 2, KO2-19); only beside `exclusion`, the owner's decision row.
    #[serde(default, deserialize_with = "some_databases")]
    pub databases: Option<Vec<Database>>,
    #[serde(default, deserialize_with = "some_text")]
    pub exclusion: Option<String>,
    pub paths: Vec<String>,
    pub routes: Vec<String>,
    #[serde(default, deserialize_with = "present")]
    pub seam: Option<String>,
    pub cells: Vec<String>,
    pub input: Input,
    pub until: Until,
    #[serde(default, deserialize_with = "some_core")]
    pub core: Option<CoreInputs>,
    pub expect: Expect,
    #[serde(default, deserialize_with = "some_text")]
    pub register: Option<String>,
    #[serde(deserialize_with = "text")]
    pub why: String,
    #[serde(deserialize_with = "text")]
    pub source: String,
}

/// A database the corpus runs on, in the order every case runs on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Database {
    Duckdb,
    Sqlite,
    Postgres,
}

impl Database {
    /// Every database, in order: a case without `databases` runs on each.
    pub const ALL: [Database; 3] = [Database::Duckdb, Database::Sqlite, Database::Postgres];

    /// The name the case format and the observations' keys use.
    pub fn name(self) -> &'static str {
        match self {
            Database::Duckdb => "duckdb",
            Database::Sqlite => "sqlite",
            Database::Postgres => "postgres",
        }
    }
}

impl Case {
    /// The databases the case runs on, in [`Database::ALL`]'s order.
    pub fn databases(&self) -> Vec<Database> {
        Database::ALL
            .into_iter()
            .filter(|database| {
                self.databases
                    .as_ref()
                    .is_none_or(|named| named.contains(database))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Acceptance,
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Until {
    FromSource,
    Materialize,
    Predict,
}

/// The public call's inputs. The pure-Rust host reads `pql` for S0, and the
/// base and the setup for an operation that reads data (from T1.11b); the
/// public call's arguments are the Python runner's.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub base: String,
    #[serde(default, deserialize_with = "some_statements")]
    pub setup: Option<Vec<String>>,
    #[serde(default, deserialize_with = "some_from_source")]
    pub from_source: Option<FromSource>,
    /// The user's edits to the schema `Schema.from_source` returned, in
    /// order, at least one; the Python runner makes them (T1.21b).
    #[serde(default, deserialize_with = "some_edits")]
    pub edits: Option<Vec<Edit>>,
    #[serde(deserialize_with = "text")]
    pub pql: String,
    pub materialize: Materialize,
    /// The call the scripted endpoint fails, numbered from zero in the
    /// order received (T2.1a, for `main`'s multi-call `predict`, T2.8c); a
    /// JSON integer written as one, `1.0` refused as the Python loader
    /// refuses it.
    #[serde(default, deserialize_with = "present")]
    pub fail_call: Option<u64>,
    /// Whether the scripted endpoint answers a classification call asking
    /// for probabilities with them (T2.8's ranking cases), a JSON boolean.
    #[serde(default, deserialize_with = "present")]
    pub probabilities: Option<bool>,
    /// Whether the scripted endpoint answers a regression call asking for
    /// quantiles with them (T2.9's forecast cases), a JSON boolean.
    #[serde(default, deserialize_with = "present")]
    pub quantiles: Option<bool>,
    /// The output fields `predict` asks for (the public call's
    /// `outputs=`), as written (T2.9's quantile cases); absent, the
    /// model's defaults for the task.
    #[serde(default, deserialize_with = "present")]
    pub outputs: Option<Vec<String>>,
    /// Whether the scripted endpoint answers a call asking for an
    /// explanation with one (T2.13's cases), a JSON boolean.
    #[serde(default, deserialize_with = "present")]
    pub explanation: Option<bool>,
    /// Whether the scripted endpoint gives each item it answers its
    /// metadata (T2.13's cases), a JSON boolean.
    #[serde(default, deserialize_with = "present")]
    pub item_metadata: Option<bool>,
    /// Whether the scripted endpoint answers a call asking for embeddings
    /// with one on each item (T2.11's cases), a JSON boolean.
    #[serde(default, deserialize_with = "present")]
    pub embeddings: Option<bool>,
}

/// Keyword arguments to `Schema.from_source`, each as the case gives it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FromSource {
    #[serde(default, deserialize_with = "present")]
    pub tables: Option<Tables>,
    #[serde(default, deserialize_with = "present")]
    pub primary_keys: Option<serde_json::Map<String, Value>>,
    #[serde(default, deserialize_with = "present")]
    pub time_columns: Option<serde_json::Map<String, Value>>,
    /// The relationships passed in (T1.21b).
    #[serde(default, deserialize_with = "present")]
    pub relationships: Option<Vec<Relationship>>,
}

/// Text the case format requires to hold something: a table or column
/// name of an edit or a relationship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name(pub String);

impl<'de> Deserialize<'de> for Name {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Name, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.is_empty() {
            return Err(D::Error::custom("a name is empty"));
        }
        Ok(Name(text))
    }
}

/// A relationship passed to `Schema.from_source`: `source_table.source_column`
/// refers to `target_table.target_column` (T1.21b). Read only from an object
/// ([`object_only`]), never from a list of its fields.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "Object")]
pub struct Relationship {
    pub source_table: Name,
    pub source_column: Name,
    pub target_table: Name,
    pub target_column: Name,
}

/// One edit to the schema `Schema.from_source` returned (T1.21b; from
/// T1.24b a key, a time column, a meaning and the relationships set). Read
/// only from an object ([`object_only`]), never from a list of its fields.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "Object")]
pub enum Edit {
    /// `schema[table].time_column = None`.
    ClearTimeColumn { table: Name },
    /// `del schema[table][column]`.
    DeleteColumn { table: Name, column: Name },
    /// `schema[table].primary_key = column`.
    SetPrimaryKey { table: Name, column: Name },
    /// `schema[table].time_column = column`.
    SetTimeColumn { table: Name, column: Name },
    /// `schema[table][column].stype = stype`.
    SetStype {
        table: Name,
        column: Name,
        stype: Name,
    },
    /// `schema.relationships = [Relationship(**link) for link in ...]`.
    SetRelationships { relationships: Vec<Relationship> },
}

/// A JSON object, which serde would otherwise let a struct or a tagged enum
/// read from a list of its fields; the case schema takes objects only.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct Object(serde_json::Map<String, Value>);

/// What `T` reads from `object`, as the object it is.
fn object_only<T: serde::de::DeserializeOwned>(object: Object) -> Result<T, String> {
    serde_json::from_value(Value::Object(object.0)).map_err(|error| error.to_string())
}

impl TryFrom<Object> for Relationship {
    type Error = String;
    fn try_from(object: Object) -> Result<Relationship, String> {
        object_only::<RelationshipFields>(object).map(|f| Relationship {
            source_table: f.source_table,
            source_column: f.source_column,
            target_table: f.target_table,
            target_column: f.target_column,
        })
    }
}

impl TryFrom<Object> for Edit {
    type Error = String;
    fn try_from(object: Object) -> Result<Edit, String> {
        object_only::<EditFields>(object).map(|edit| match edit {
            EditFields::ClearTimeColumn { table } => Edit::ClearTimeColumn { table },
            EditFields::DeleteColumn { table, column } => Edit::DeleteColumn { table, column },
            EditFields::SetPrimaryKey { table, column } => Edit::SetPrimaryKey { table, column },
            EditFields::SetTimeColumn { table, column } => Edit::SetTimeColumn { table, column },
            EditFields::SetStype {
                table,
                column,
                stype,
            } => Edit::SetStype {
                table,
                column,
                stype,
            },
            EditFields::SetRelationships { relationships } => {
                Edit::SetRelationships { relationships }
            }
        })
    }
}

/// [`Relationship`]'s fields, read from an object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationshipFields {
    source_table: Name,
    source_column: Name,
    target_table: Name,
    target_column: Name,
}

/// [`Edit`]'s fields, read from an object.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum EditFields {
    ClearTimeColumn {
        table: Name,
    },
    DeleteColumn {
        table: Name,
        column: Name,
    },
    SetPrimaryKey {
        table: Name,
        column: Name,
    },
    SetTimeColumn {
        table: Name,
        column: Name,
    },
    SetStype {
        table: Name,
        column: Name,
        stype: Name,
    },
    SetRelationships {
        relationships: Vec<Relationship>,
    },
}

/// An optional field the case schema does not let be null: absent is
/// `None` (`serde(default)`), and present is read as `T`, so a null is
/// refused as the Python validator refuses it, never taken for absent.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

/// Text the case schema requires to hold something (`minLength: 1`).
fn text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let text = String::deserialize(deserializer)?;
    if text.is_empty() {
        return Err(D::Error::custom("a text the case format requires is empty"));
    }
    Ok(text)
}

/// Optional text that, when given, holds something (`minLength: 1`).
fn some_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    text(deserializer).map(Some)
}

/// `databases`, when given: at least one, none twice, and not all three,
/// since an exclusion leaves one out (`minItems`, `uniqueItems`,
/// `maxItems`).
fn some_databases<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<Database>>, D::Error> {
    let databases = Vec::<Database>::deserialize(deserializer)?;
    let distinct: BTreeSet<Database> = databases.iter().copied().collect();
    if databases.is_empty()
        || distinct.len() != databases.len()
        || distinct.len() == Database::ALL.len()
    {
        return Err(D::Error::custom(
            "databases is empty, names a database twice or leaves none out",
        ));
    }
    Ok(Some(databases))
}

/// Whether `text` names a decision row, `KO<phase>-<n>`.
fn is_decision_row(text: &str) -> bool {
    let numbers = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
    text.strip_prefix("KO")
        .and_then(|rest| rest.split_once('-'))
        .is_some_and(|(phase, row)| numbers(phase) && numbers(row))
}

/// `input.setup`, when given: at least one statement, none empty, as the
/// schema's `statements` says (`minItems`, `minLength`).
fn some_statements<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error> {
    let statements = Vec::<String>::deserialize(deserializer)?;
    if statements.is_empty() || statements.iter().any(String::is_empty) {
        return Err(D::Error::custom(
            "input.setup is empty or holds an empty statement",
        ));
    }
    Ok(Some(statements))
}

/// `input.from_source`, when given, names at least one argument
/// (`minProperties`).
fn some_from_source<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<FromSource>, D::Error> {
    let from_source = FromSource::deserialize(deserializer)?;
    if from_source.tables.is_none()
        && from_source.primary_keys.is_none()
        && from_source.time_columns.is_none()
        && from_source.relationships.is_none()
    {
        return Err(D::Error::custom("input.from_source names no argument"));
    }
    Ok(Some(from_source))
}

/// `core`, when given, names at least one operation (`minProperties`).
fn some_core<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<CoreInputs>, D::Error> {
    let core = CoreInputs::deserialize(deserializer)?;
    if core.validate_query.is_none()
        && core.describe_source.is_none()
        && core.materialize.is_none()
        && core.predict.is_none()
    {
        return Err(D::Error::custom("core names no operation"));
    }
    Ok(Some(core))
}

/// A field the case schema requires and lets be null: absent is refused,
/// null is `None` (no `serde(default)`, so serde does not take a missing
/// `Option` for `None`).
fn nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

/// `input.edits`, when given, holds at least one edit, as the case schema's
/// `minItems` says.
fn some_edits<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Vec<Edit>>, D::Error> {
    let edits = Vec::<Edit>::deserialize(deserializer)?;
    if edits.is_empty() {
        return Err(D::Error::custom("input.edits is empty"));
    }
    Ok(Some(edits))
}

/// `materialize.inference_config`, when given, is an object of at least one
/// key, as the case schema's `minProperties` says; what it holds is free.
fn some_options<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Map<String, Value>>, D::Error> {
    let options = serde_json::Map::<String, Value>::deserialize(deserializer)?;
    if options.is_empty() {
        return Err(D::Error::custom("inference_config is empty"));
    }
    Ok(Some(options))
}

/// `from_source`'s tables: a list of names, or names by alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Tables {
    List(Vec<Value>),
    Named(serde_json::Map<String, Value>),
}

/// Keyword arguments to `materialize`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Materialize {
    pub indices: Vec<Index>,
    #[serde(default, deserialize_with = "present")]
    pub indices_as: Option<IndicesAs>,
    /// An ISO timestamp, or `entity`: each entity at its own time (T2.11).
    #[serde(default, deserialize_with = "present")]
    pub anchor_time: Option<String>,
    /// The context's anchor time (T2.11), written as `anchor_time` is.
    #[serde(default, deserialize_with = "present")]
    pub context_anchor_time: Option<String>,
    /// The inference options passed, kept as written, free-form (T1.21b).
    #[serde(default, deserialize_with = "some_options")]
    pub inference_config: Option<serde_json::Map<String, Value>>,
    /// The rows a ranking is over, each read as an index is (E153).
    #[serde(default, deserialize_with = "present")]
    pub candidates: Option<Vec<Index>>,
    /// A JSON integer of any size, written as one, as the case schema's
    /// `integer` and the Python loader take it; `0`, which the reference
    /// refuses, is a case's input.
    #[serde(default, deserialize_with = "present")]
    pub top_k: Option<Integer>,
    /// How many entities one call scores (`main`'s 1fc5a686, T2.8c), kept
    /// as written, whatever JSON scalar it is, so a value the reference
    /// refuses is the case's input.
    #[serde(default, deserialize_with = "present")]
    pub rows_per_call: Option<Written>,
    /// How many past steps of the target are added as lagged features
    /// (T2.9), kept as written, whatever JSON scalar it is, as
    /// `rows_per_call` is, so a value the reference refuses is the case's
    /// input.
    #[serde(default, deserialize_with = "present")]
    pub lag_timesteps: Option<Written>,
    /// The sampling options (T2.11), each kept as the text the case wrote,
    /// any JSON value, `null` included (`Some(AsWritten("null"))`, the
    /// public call's `None`: an unseeded sample for `random_seed`), so a
    /// value the reference refuses is the case's input and no number of
    /// any size passes through a double; absent, `None`.
    #[serde(default, deserialize_with = "present")]
    pub random_seed: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub num_hops: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub num_neighbors: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub max_pq_iterations: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub use_prediction_time: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub return_embeddings: Option<AsWritten>,
    #[serde(default, deserialize_with = "present")]
    pub exclude_cols_dict: Option<AsWritten>,
}

/// Any JSON value, `null`, a list and an object included, kept as the
/// exact text the case wrote (whitespace around it trimmed): a number of
/// any size or form, nested or not, passes through nothing that would
/// change it (T2.11's sampling options).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsWritten(pub String);

impl<'de> Deserialize<'de> for AsWritten {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<AsWritten, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        Ok(AsWritten(raw.get().trim().to_owned()))
    }
}

/// A JSON scalar other than `null` (a number, a bool or a string), kept as
/// the text the case wrote: a number of any size or form passes through
/// nothing that would change it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written(pub String);

impl<'de> Deserialize<'de> for Written {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Written, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get().trim();
        match text.as_bytes().first() {
            Some(b'"' | b'-' | b'0'..=b'9') => Ok(Written(text.to_owned())),
            _ if text == "true" || text == "false" => Ok(Written(text.to_owned())),
            _ => Err(D::Error::custom("not a number, a bool or a string")),
        }
    }
}

/// One entity key: a JSON integer literal of any size, kept as its
/// digits, or text. A number with a fraction or an exponent, `1.0`
/// included, is refused, as the Python loader refuses it; nothing passes
/// through a double.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    Integer(String),
    Text(String),
}

/// Whether `text` is a JSON integer literal: `-?(0|[1-9][0-9]*)`.
fn is_integer_literal(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
}

impl<'de> Deserialize<'de> for Index {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Index, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get().trim();
        if text.starts_with('"') {
            serde_json::from_str(text)
                .map(Index::Text)
                .map_err(D::Error::custom)
        } else if is_integer_literal(text) {
            Ok(Index::Integer(text.to_string()))
        } else {
            Err(D::Error::custom(
                "an index is a JSON integer literal or a string",
            ))
        }
    }
}

/// A JSON integer literal of any size, kept as its digits: a number with a
/// fraction or an exponent, `3.0` included, is refused, and nothing passes
/// through a fixed width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Integer(pub String);

impl<'de> Deserialize<'de> for Integer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Integer, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let text = raw.get().trim();
        if is_integer_literal(text) {
            Ok(Integer(text.to_owned()))
        } else {
            Err(D::Error::custom("not a JSON integer literal"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicesAs {
    Decimal,
}

/// The core operations a case's call starts, with what the host passes.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreInputs {
    #[serde(default, deserialize_with = "present")]
    pub validate_query: Option<ValidateQueryInput>,
    #[serde(default, deserialize_with = "present")]
    pub describe_source: Option<DescribeSourceInput>,
    #[serde(default, deserialize_with = "present")]
    pub materialize: Option<MaterializeInput>,
    #[serde(default, deserialize_with = "present")]
    pub predict: Option<PredictInput>,
}

/// `predict`'s input beside the request state, which is `materialize`'s
/// answer, handed back as the host held it: the model the call goes to.
/// The request id is the host's to draw (decision E35 for its value).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictInput {
    pub model: String,
}

/// `materialize`'s inputs beside the query text, which is `input.pql`,
/// kept as written: the provider, the entities to predict for (each a
/// contract scalar, one Arrow column), the anchor time (a contract `ts`
/// scalar or null) and the seed (the contract's signed hexadecimal text, or
/// null for the default). Not the schema it samples: that is the
/// description `describe_source` answered, with `input.edits` made, and a
/// case naming one is refused (`deny_unknown_fields`; decision E50, the
/// rust-host journey guard).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterializeInput {
    pub provider: Value,
    /// None at all is a call the core refuses, as the reference does
    /// (`empty.indices`, from T1.32d), which this host carries.
    pub indices: Vec<Value>,
    pub anchor_time: Value,
    /// Required, and null for the default.
    #[serde(deserialize_with = "nullable")]
    pub seed: Option<String>,
}

/// `describe_source`'s inputs, kept as written: the provider, the source's
/// tables and overrides (the contract's `SourceSpec`) and whether keys are
/// checked. The data are the case's base and setup.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescribeSourceInput {
    pub provider: Value,
    pub source: Value,
    pub check_keys: bool,
}

/// `validate_query`'s inputs beside the query text, which is `input.pql`:
/// kept as written, so an observation records exactly what the case gave.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateQueryInput {
    pub profile: Value,
    pub schema: Value,
}

/// The reference's outcome. As the schema's `if`/`then`/`else` says, a
/// refusal names its error and its code (null for an error without one),
/// and a success names none of error, code and message.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "ExpectFields")]
pub struct Expect {
    pub outcome: String,
    pub stage: String,
    pub error: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
    pub today: Option<String>,
}

/// [`Expect`]'s fields as written: `code` is `None` when absent and
/// `Some(None)` when null, which the schema tells apart.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectFields {
    outcome: String,
    stage: String,
    #[serde(default, deserialize_with = "present")]
    error: Option<String>,
    #[serde(default, deserialize_with = "present")]
    code: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    message: Option<String>,
    #[serde(default, deserialize_with = "present")]
    today: Option<String>,
}

impl TryFrom<ExpectFields> for Expect {
    type Error = String;
    fn try_from(fields: ExpectFields) -> Result<Expect, String> {
        if fields.outcome == "refusal" {
            if fields.error.is_none() || fields.code.is_none() {
                return Err("a refusal names its error and its code".into());
            }
        } else if fields.error.is_some() || fields.code.is_some() || fields.message.is_some() {
            return Err("a success names no error, code or message".into());
        }
        Ok(Expect {
            outcome: fields.outcome,
            stage: fields.stage,
            error: fields.error,
            code: fields.code.flatten(),
            message: fields.message,
            today: fields.today,
        })
    }
}

/// A case with the SHA-256 of its file's bytes.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub case: Case,
    pub digest: String,
}

/// The SHA-256 of `bytes`, as lowercase hexadecimal.
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn json_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    let entries =
        std::fs::read_dir(directory).with_context(|| format!("reading {}", directory.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            json_files(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            found.push(path);
        }
    }
    Ok(())
}

/// The case files named, a directory standing for every `*.json` under it,
/// in path order.
pub fn case_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found = Vec::new();
            json_files(path, &mut found)?;
            found.sort();
            files.extend(found);
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}

/// Every case named, each read whole, its ids unique. The error names the
/// file and the position, never a value read.
pub fn load(paths: &[PathBuf]) -> Result<Vec<Loaded>> {
    let mut loaded = Vec::new();
    let mut ids = BTreeSet::new();
    for path in case_files(paths)? {
        let name = path.display().to_string();
        let bytes = std::fs::read(&path).with_context(|| format!("reading {name}"))?;
        let case: Case = serde_json::from_slice(&bytes).map_err(|error| {
            anyhow::anyhow!(
                "{name}: not a case (line {}, column {})",
                error.line(),
                error.column()
            )
        })?;
        if case.format != FORMAT {
            bail!("{name}: not {FORMAT}");
        }
        // `dependentRequired`: a case leaves a database out only under the
        // owner's exclusion, and an exclusion only leaves one out.
        if case.databases.is_some() != case.exclusion.is_some() {
            bail!("{name}: not a case: databases and exclusion come together");
        }
        if case
            .exclusion
            .as_deref()
            .is_some_and(|row| !is_decision_row(row))
        {
            bail!("{name}: not a case: exclusion is not a decision row");
        }
        if !ids.insert(case.id.clone()) {
            bail!("{name}: case id {} repeats", case.id);
        }
        loaded.push(Loaded {
            case,
            digest: sha256(&bytes),
        });
    }
    Ok(loaded)
}
