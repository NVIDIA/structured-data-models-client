// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The graph a request samples: the described source's tables, each with
//! its key and time column, and its links as edges from a foreign key to
//! the table it refers to, as the reference's materializer builds the
//! engine's graph from the client's schema (`_build_graph`) and its
//! sampler reads it.

use std::collections::BTreeSet;

use sdm_contract::ops::{ColumnRef, DescribedColumn, SourceDescription, Stype};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

use crate::not_yet;

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

fn invalid(message: &str, field: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("field", Detail::text(field))
}

impl Graph {
    /// The graph of `description`. A link names a foreign key and the
    /// primary key it refers to, `[foreign, referenced]`; one repeated is
    /// kept once, and one whose referenced column is not its table's key is
    /// refused, as the reference's materializer refuses it; a column named
    /// with the samplers' reserved prefix is refused, as their sampler
    /// refuses it. A key of several columns is not sampled yet.
    pub fn new(description: &SourceDescription) -> Result<Graph, CoreError> {
        let mut tables = Vec::with_capacity(description.tables.len());
        for table in &description.tables {
            let primary_key = match table.primary_key.as_slice() {
                [] => None,
                [one] => Some(one.clone()),
                _ => return Err(not_yet("composite_key")),
            };
            if table
                .columns
                .iter()
                .any(|c| c.name.starts_with(RESERVED_PREFIX))
            {
                return Err(invalid(
                    "a column's name starts with the prefix the SQL samplers reserve",
                    "schema",
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
        for link in &description.links {
            let [foreign, referenced]: &[ColumnRef; 2] = link
                .as_slice()
                .try_into()
                .map_err(|_| invalid("a link is not a foreign key and its key", "schema"))?;
            let (Some(src), Some(dst)) =
                (graph.table(&foreign.table), graph.table(&referenced.table))
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
            if !graph.edges.contains(&edge) {
                graph.edges.push(edge);
            }
        }
        Ok(graph)
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
