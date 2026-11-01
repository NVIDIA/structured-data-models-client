// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The graph a request samples: the described source's tables, each with
//! its key and time column, and its links as edges from a foreign key to
//! the table it refers to, as the reference's materializer builds the
//! engine's graph from the client's schema (`_build_graph`) and its
//! sampler reads it.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::ops::{ColumnRef, DescribedColumn, SourceDescription, Stype};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};

/// One table of the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    /// The name the schema gives the table, which the query and the
    /// payload use.
    pub name: String,
    /// The relation of the source the table reads, which every data request
    /// names (design choice 12), as the reference's engine reads a table
    /// named otherwise by its source name (decision E61).
    pub source: sdm_contract::data::Relation,
    pub columns: Vec<DescribedColumn>,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
}

impl Table {
    /// The column `name`, if the table has it.
    pub fn column(&self, name: &str) -> Option<&DescribedColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// `names` in the order the table holds them, a name it does not hold
    /// last, by name, as the reference's SQL sampler orders a projection
    /// (`_ordered_columns`).
    pub fn ordered(&self, names: &BTreeSet<String>) -> Vec<String> {
        let mut known: Vec<String> = self
            .columns
            .iter()
            .filter(|c| names.contains(&c.name))
            .map(|c| c.name.clone())
            .collect();
        known.extend(names.iter().filter(|n| self.column(n).is_none()).cloned());
        known
    }
}

/// An edge: `src.fkey` refers to the primary key of `dst`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge {
    pub src: String,
    pub fkey: String,
    pub dst: String,
}

/// The tables and edges, in the order the description gives them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    pub tables: Vec<Table>,
    pub edges: Vec<Edge>,
}

/// The prefix the SQL samplers' own columns carry (`__kumo_`), which no
/// source column may start with.
pub const RESERVED_PREFIX: &str = "__kumo_";

/// The prefix a key of several columns is folded under (`DERIVED_PREFIX`).
pub const FOLDED_PREFIX: &str = "__kumo_key_";

/// Maximum complete encoded native/warehouse identity column name.
pub const FOLDED_NAME_MOST: usize = 200;

/// The column a key of several columns is folded into: the prefix, then
/// each column's name as the hexadecimal of its UTF-8, joined by `_`
/// (`composite_key.encode_identity`).
pub fn folded_name(key: &[String]) -> String {
    format!(
        "{FOLDED_PREFIX}{}",
        key.iter()
            .map(|c| c.bytes().map(|b| format!("{b:02x}")).collect::<String>())
            .collect::<Vec<_>>()
            .join("_")
    )
}

/// `error`, naming every column of `description` the SQL samplers' check of
/// the schema's names refuses (`_reject_reserved_columns`), as
/// `table.column`, sorted: each column named with [`RESERVED_PREFIX`], and
/// each key of several columns as the column it is folded into; with the
/// prefix and the reason, `reserved_name` or `reserved_names` by their
/// count, for the host's words.
pub fn with_reserved(error: CoreError, description: &SourceDescription) -> CoreError {
    let mut reserved: Vec<String> = Vec::new();
    for table in &description.tables {
        if table.primary_key.len() > 1 {
            reserved.push(format!(
                "{}.{}",
                table.name,
                folded_name(&table.primary_key)
            ));
        }
        for column in &table.columns {
            if column.name.starts_with(RESERVED_PREFIX) {
                reserved.push(format!("{}.{}", table.name, column.name));
            }
        }
    }
    reserved.sort_unstable();
    reserved.dedup();
    let reason = if reserved.len() > 1 {
        "reserved_names"
    } else {
        "reserved_name"
    };
    let sentence = if reserved.len() > 1 {
        Sentence::ReservedNames
    } else {
        Sentence::ReservedName
    };
    error
        .saying(sentence)
        .with_field("reason", Detail::text(reason))
        .with_field("prefix", Detail::text(RESERVED_PREFIX))
        .with_field(
            "reserved",
            Detail::List(reserved.into_iter().map(Scalar::Utf8).collect()),
        )
}

fn invalid(message: &str, field: &str) -> CoreError {
    CoreError::new(Sentence::SchemaLinkUnread, message).with_field("field", Detail::text(field))
}

impl Graph {
    /// The graph of `description`. A link names a foreign key and the
    /// primary key it refers to, `[foreign, referenced]`; one repeated is
    /// kept once, and one whose referenced column is not its table's key is
    /// refused, as the reference's materializer refuses it; a column named
    /// with the samplers' reserved prefix is refused, as their sampler
    /// refuses it. So is a table keyed by several columns, as the
    /// reference's SQL samplers refuse every such table: its engine folds
    /// the key into one column named with that prefix
    /// (`__kumo_key_<columns>`), which their check of the schema's names
    /// then refuses (`_reject_reserved_columns`), whatever the query.
    pub fn new(description: &SourceDescription) -> Result<Graph, CoreError> {
        let mut tables = Vec::with_capacity(description.tables.len());
        for table in &description.tables {
            let primary_key = match table.primary_key.as_slice() {
                [] => None,
                [one] => Some(one.clone()),
                several => {
                    return Err(with_reserved(invalid(
                        "a table keyed by several columns cannot be sampled: the SQL samplers fold its key into a column of the name they reserve",
                        "schema",
                    )
                    .with_field("table", Detail::text(&table.name))
                    .with_field(
                        "columns",
                        Detail::List(several.iter().map(|c| Scalar::Utf8(c.clone())).collect()),
                    ), description));
                }
            };
            if table
                .columns
                .iter()
                .any(|c| c.name.starts_with(RESERVED_PREFIX))
            {
                return Err(with_reserved(
                    invalid(
                        "a column's name starts with the prefix the SQL samplers reserve",
                        "schema",
                    ),
                    description,
                ));
            }
            tables.push(Table {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table.columns.clone(),
                primary_key,
                time_column: table.time_column.clone(),
            });
        }
        let mut graph = Graph {
            tables,
            edges: Vec::new(),
        };
        graph.add_links(description)?;
        Ok(graph)
    }

    fn add_links(&mut self, description: &SourceDescription) -> Result<(), CoreError> {
        for link in &description.links {
            let [foreign, referenced]: &[ColumnRef; 2] = link
                .as_slice()
                .try_into()
                .map_err(|_| invalid("a link is not a foreign key and its key", "schema"))?;
            let (Some(src), Some(dst)) =
                (self.table(&foreign.table), self.table(&referenced.table))
            else {
                return Err(invalid("a link names a table the schema lacks", "schema"));
            };
            if src.column(&foreign.column).is_none() {
                return Err(invalid("a link names a column its table lacks", "schema"));
            }
            if dst.primary_key.as_deref() != Some(referenced.column.as_str()) {
                return Err(invalid(
                    "a link refers to a column that is not its table's primary key",
                    "schema",
                ));
            }
            let edge = Edge {
                src: foreign.table.clone(),
                fkey: foreign.column.clone(),
                dst: referenced.table.clone(),
            };
            if !self.edges.contains(&edge) {
                self.edges.push(edge);
            }
        }
        Ok(())
    }

    /// The table `name`, if the graph has it.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables.iter().find(|t| t.name == name)
    }

    /// `table`'s foreign keys, each with the table it refers to.
    pub fn foreign_keys(&self, table: &str) -> Vec<(&str, &str)> {
        self.edges
            .iter()
            .filter(|e| e.src == table)
            .map(|e| (e.fkey.as_str(), e.dst.as_str()))
            .collect()
    }

    /// The foreign keys that refer to `table`, each with its table.
    pub fn referring(&self, table: &str) -> Vec<(&str, &str)> {
        self.edges
            .iter()
            .filter(|e| e.dst == table)
            .map(|e| (e.src.as_str(), e.fkey.as_str()))
            .collect()
    }

    /// `table`'s feature columns, in its order, with their semantic types:
    /// every column but its key and its foreign keys (the reference's
    /// `table_stype_dict`).
    pub fn features(&self, table: &Table) -> Vec<(String, Stype)> {
        let foreign: BTreeSet<&str> = self
            .foreign_keys(&table.name)
            .into_iter()
            .map(|(fkey, _)| fkey)
            .collect();
        table
            .columns
            .iter()
            .filter(|c| table.primary_key.as_deref() != Some(c.name.as_str()))
            .filter(|c| !foreign.contains(c.name.as_str()))
            .map(|c| (c.name.clone(), c.stype))
            .collect()
    }
}

/// Native source semantics alongside the scalar operational traversal graph.
/// Full ordered identities are never discarded to make a table keyless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeGraph {
    pub topology: Graph,
    pub primary_keys: BTreeMap<String, Vec<String>>,
}

impl NativeGraph {
    pub fn new(description: &SourceDescription) -> Result<Self, CoreError> {
        let mut tables = Vec::with_capacity(description.tables.len());
        let mut primary_keys = BTreeMap::new();
        for table in &description.tables {
            if primary_keys
                .insert(table.name.clone(), table.primary_key.clone())
                .is_some()
            {
                return Err(invalid("native graph repeats a table identity", "schema"));
            }
            for name in &table.primary_key {
                if !table.columns.iter().any(|column| &column.name == name) {
                    return Err(invalid(
                        "native row identity names an absent source column",
                        "schema",
                    ));
                }
            }
            let primary_key = match table.primary_key.as_slice() {
                [] => None,
                [one] => Some(one.clone()),
                many => Some(folded_name(many)),
            };
            tables.push(Table {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table.columns.clone(),
                primary_key,
                time_column: table.time_column.clone(),
            });
        }
        let mut topology = Graph {
            tables,
            edges: Vec::new(),
        };
        topology.add_links(description)?;
        let native = Self {
            topology,
            primary_keys,
        };
        native.validate()?;
        Ok(native)
    }
    /// Validate a typed native product before reading any row or adjacency.
    pub fn validate(&self) -> Result<(), CoreError> {
        let invalid_product = || {
            CoreError::protocol(
                Sentence::InvalidMessage,
                "native graph topology contradicts its complete ordered identities",
            )
        };
        if self.primary_keys.len() != self.topology.tables.len() {
            return Err(invalid_product());
        }
        let mut names = BTreeSet::new();
        for table in &self.topology.tables {
            if !names.insert(&table.name) {
                return Err(invalid_product());
            }
            let key = self
                .primary_keys
                .get(&table.name)
                .ok_or_else(invalid_product)?;
            if key.iter().collect::<BTreeSet<_>>().len() != key.len()
                || key.iter().any(|name| table.column(name).is_none())
            {
                return Err(invalid_product());
            }
            let operational = match key.as_slice() {
                [] => None,
                [one] => Some(one.clone()),
                many => Some(folded_name(many)),
            };
            if table.primary_key != operational {
                return Err(invalid_product());
            }
        }
        let mut edges = BTreeSet::new();
        for edge in &self.topology.edges {
            let source = self.topology.table(&edge.src).ok_or_else(invalid_product)?;
            let target = self.topology.table(&edge.dst).ok_or_else(invalid_product)?;
            if !edges.insert(edge)
                || source.column(&edge.fkey).is_none()
                || self.primary_keys[&target.name].len() != 1
            {
                return Err(invalid_product());
            }
        }
        Ok(())
    }
}

impl std::ops::Deref for NativeGraph {
    type Target = Graph;
    fn deref(&self) -> &Graph {
        &self.topology
    }
}
