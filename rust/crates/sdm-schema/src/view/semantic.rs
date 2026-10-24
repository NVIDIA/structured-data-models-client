// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic view's definition, as the warehouse writes it out in YAML,
//! read into its structure (design S5, T3.5's V2, V3 and V6): its logical
//! tables, each with its base table, primary key and columns, and its
//! relationships, expressions kept as written. The old client reads these
//! by the paths of `RE/graph.py:2072-2252`, and fails with a raw `KeyError`
//! or `TypeError` where one is missing or of another kind; here each such
//! is refused by name, saying what is missing. Which table an expression
//! reads, and whether it may run, the data layer says.

use std::fmt;

use super::yaml::{self, Value, YamlError};

/// A semantic view's definition.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticView {
    pub tables: Vec<LogicalTable>,
    pub relationships: Vec<Relationship>,
}

/// A logical table.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalTable {
    pub name: String,
    pub base: BaseTable,
    pub primary_key: PrimaryKey,
    /// The columns of its `dimensions`, `time_dimensions` and `facts`, in
    /// that order (`:2105-2113`).
    pub columns: Vec<ViewColumn>,
}

/// The table a logical table reads, named in full as the definition says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseTable {
    pub database: String,
    pub schema: String,
    pub table: String,
}

/// A logical table's `primary_key` (`:2081-2090`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimaryKey {
    /// None declared, or one of no columns.
    None,
    One(String),
    /// More than one column, which the old client drops with a message.
    Composite(Vec<String>),
}

/// A column of a logical table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewColumn {
    pub name: String,
    /// Its `expr` as written; none for a column of the base table of the
    /// same name.
    pub expr: Option<String>,
    pub data_type: Option<String>,
    pub section: Section,
}

/// The section a column is declared in; a time dimension is a time column's
/// first candidate (`:2192-2205`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Dimension,
    TimeDimension,
    Fact,
}

/// A relationship between two logical tables (`:2209-2222`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relationship {
    pub name: String,
    pub left_table: String,
    pub right_table: String,
    pub keys: RelationshipKeys,
}

/// A relationship's `relationship_columns`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationshipKeys {
    /// One pair of columns.
    One { left: String, right: String },
    /// Any other number, which the old client drops with a message.
    Composite,
}

/// Why a semantic view's definition cannot be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticRefusal {
    /// The text is no YAML.
    Yaml(YamlError),
    /// An entry the old client reads is missing.
    Misses { entry: String },
    /// An entry is of a kind the old client cannot read: a name that is
    /// no text, or a list that is no list. With its `repr`.
    Unreadable { entry: String, repr: String },
}

impl fmt::Display for SemanticRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticRefusal::Yaml(error) => {
                write!(
                    f,
                    "The semantic view definition is not readable YAML: {error}"
                )
            }
            SemanticRefusal::Misses { entry } => {
                write!(f, "The semantic view definition misses '{entry}'")
            }
            SemanticRefusal::Unreadable { entry, repr } => write!(
                f,
                "The semantic view definition gives '{entry}' as {repr}, which cannot be read"
            ),
        }
    }
}

impl std::error::Error for SemanticRefusal {}

/// Reads a semantic view's definition.
pub fn parse(definition: &str) -> Result<SemanticView, SemanticRefusal> {
    let cfg = yaml::load(definition).map_err(SemanticRefusal::Yaml)?;
    let tables = list(required(&cfg, "tables")?, "tables")?;
    // The old client reads every table's name first (`:2069`).
    for table in tables {
        text(required(table, "name")?, "name")?;
    }
    let tables = tables
        .iter()
        .map(logical_table)
        .collect::<Result<Vec<_>, _>>()?;
    let relationships = match cfg.get("relationships") {
        None => Vec::new(),
        Some(section) => list(section, "relationships")?
            .iter()
            .map(relationship)
            .collect::<Result<_, _>>()?,
    };
    Ok(SemanticView {
        tables,
        relationships,
    })
}

fn logical_table(cfg: &Value) -> Result<LogicalTable, SemanticRefusal> {
    let name = text(required(cfg, "name")?, "name")?;
    let base = required(cfg, "base_table")?;
    let base = BaseTable {
        table: text(required(base, "table")?, "table")?,
        database: text(required(base, "database")?, "database")?,
        schema: text(required(base, "schema")?, "schema")?,
    };
    let primary_key = match cfg.get("primary_key") {
        None => PrimaryKey::None,
        Some(key) => {
            let mut columns = list(required(key, "columns")?, "columns")?
                .iter()
                .map(|column| text(column, "columns"))
                .collect::<Result<Vec<_>, _>>()?;
            match columns.len() {
                0 => PrimaryKey::None,
                1 => PrimaryKey::One(columns.remove(0)),
                _ => PrimaryKey::Composite(columns),
            }
        }
    };
    let mut columns = Vec::new();
    for (section, key) in [
        (Section::Dimension, "dimensions"),
        (Section::TimeDimension, "time_dimensions"),
        (Section::Fact, "facts"),
    ] {
        let Some(entries) = cfg.get(key) else {
            continue;
        };
        for column in list(entries, key)? {
            columns.push(ViewColumn {
                name: text(required(column, "name")?, "name")?,
                expr: optional_text(column, "expr")?,
                data_type: optional_text(column, "data_type")?,
                section,
            });
        }
    }
    Ok(LogicalTable {
        name,
        base,
        primary_key,
        columns,
    })
}

fn relationship(cfg: &Value) -> Result<Relationship, SemanticRefusal> {
    let name = text(required(cfg, "name")?, "name")?;
    let pairs = list(
        required(cfg, "relationship_columns")?,
        "relationship_columns",
    )?;
    let keys = match pairs {
        [pair] => RelationshipKeys::One {
            left: text(required(pair, "left_column")?, "left_column")?,
            right: text(required(pair, "right_column")?, "right_column")?,
        },
        _ => RelationshipKeys::Composite,
    };
    Ok(Relationship {
        name,
        left_table: text(required(cfg, "left_table")?, "left_table")?,
        right_table: text(required(cfg, "right_table")?, "right_table")?,
        keys,
    })
}

/// `cfg[entry]`, a mapping's entry that must be there.
fn required<'a>(cfg: &'a Value, entry: &str) -> Result<&'a Value, SemanticRefusal> {
    cfg.get(entry).ok_or_else(|| SemanticRefusal::Misses {
        entry: entry.to_owned(),
    })
}

fn list<'a>(value: &'a Value, entry: &str) -> Result<&'a [Value], SemanticRefusal> {
    match value {
        Value::Seq(items) => Ok(items),
        other => Err(unreadable(entry, other)),
    }
}

fn text(value: &Value, entry: &str) -> Result<String, SemanticRefusal> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| unreadable(entry, value))
}

/// `cfg.get(entry, None)`: text, or none when absent or null.
fn optional_text(cfg: &Value, entry: &str) -> Result<Option<String>, SemanticRefusal> {
    match cfg.get(entry) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => text(value, entry).map(Some),
    }
}

fn unreadable(entry: &str, value: &Value) -> SemanticRefusal {
    SemanticRefusal::Unreadable {
        entry: entry.to_owned(),
        repr: value.repr(),
    }
}
