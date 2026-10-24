// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic model's definition as the conversion reads it (design S5):
//! one [`ViewSpec`] for a metric view and a semantic view alike, with no
//! dialect in it. It holds the definition's [`Document`] and, for each
//! logical table, column, relationship, parameter and measure, the node it
//! was written as.
//!
//! The parse reads only what the old client read before its first warehouse
//! read, and refuses only where it failed there. Everything else is kept as
//! its node and read when the conversion comes to it, through the
//! accessors here, so an entry the old client never looked at (a nameless
//! join's children, a column's type it never used, the members of a
//! composite key, the tables of a composite relationship) can never refuse
//! the view. Expressions stay opaque text, the data layer's to read.

use std::fmt;

use super::yaml::{Document, NodeId, Value, YamlError};

/// The kind of view a definition is, as the contract names it.
pub use sdm_contract::view::ViewKind;

/// A view's definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewSpec {
    pub kind: ViewKind,
    /// The definition's values, which a message may quote.
    pub document: Document,
    /// The logical tables: a metric view's source table (first, under the
    /// alias `source`) and its joins, each before the joins nested in it; a
    /// semantic view's tables, in order.
    pub tables: Vec<TableItem>,
    /// The columns: a metric view's dimensions, on no table until their
    /// expressions say which; a semantic view's, each on its table, by
    /// section.
    pub columns: Vec<ColumnItem>,
    /// A semantic view's relationships.
    pub relationships: Vec<RelationshipItem>,
    /// A metric view's `filter`, unless absent or null.
    pub filter: Option<NodeId>,
    pub parameters: Vec<ParameterItem>,
    pub measures: Vec<MeasureItem>,
    /// The sections the conversion leaves out, which it says.
    pub ignored: Vec<Ignored>,
    /// The `order` of each window of each measure, as written, in order;
    /// the data layer unquotes them and keeps each once.
    pub window_orders: Vec<String>,
}

/// A section the conversion leaves out, saying so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ignored {
    /// A metric view's `materialization`.
    Materialization,
    /// A metric view's `fields`, passed over for its `dimensions`.
    Fields,
}

/// An entry of the logical tables.
#[derive(Clone, Debug, PartialEq)]
pub enum TableItem {
    /// A semantic view's table, its name read and the rest of it not yet
    /// ([`super::semantic::table`] reads it, in the conversion's order).
    Declared {
        name: NodeId,
        cfg: NodeId,
    },
    /// A join that is no mapping, declared on the table at `parent`.
    NotAMapping {
        cfg: NodeId,
        parent: usize,
    },
    /// A join with no text name, declared on the table at `parent`; the
    /// joins nested in it are read only when a skipped join above it names
    /// them ([`ViewSpec::nested_names`]).
    Nameless {
        cfg: NodeId,
        parent: usize,
    },
    Table(TableSpec),
}

/// A logical table.
#[derive(Clone, Debug, PartialEq)]
pub struct TableSpec {
    /// Its name in the view: a metric view's `source`, a join's name, a
    /// semantic view's table's name.
    pub alias: String,
    pub source: Source,
    /// The table a join is declared on.
    pub parent: Option<usize>,
    /// How a join joins.
    pub join: Option<JoinSpec>,
    /// A semantic view's table's `primary_key`.
    pub primary_key: PrimaryKey,
    /// The entry as written.
    pub cfg: NodeId,
}

/// The relation a logical table reads.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// A metric view's or a join's `source`, a name for the data layer to
    /// read, if written.
    Name(Option<NodeId>),
    /// A semantic view's `base_table`, its parts as written.
    Parts {
        database: NodeId,
        schema: NodeId,
        table: NodeId,
    },
}

/// How a join joins.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinSpec {
    pub cardinality: Cardinality,
    pub keys: JoinKeys,
}

/// A join's `cardinality`, `many_to_one` when absent.
#[derive(Clone, Debug, PartialEq)]
pub enum Cardinality {
    ManyToOne,
    OneToMany,
    /// Any other value Python can look up in a set.
    Unsupported(NodeId),
    /// A list or a mapping, which the old client failed on looking it up
    /// in its set of cardinalities (`TypeError`); refused by name.
    Unhashable,
}

/// What a join is joined on.
#[derive(Clone, Debug, PartialEq)]
pub enum JoinKeys {
    /// `using` with one column, as given (Python's `str` of it names it).
    Using(NodeId),
    /// `using` with any other number of columns, or no list.
    UsingSeveral,
    /// No `using`: the `on`, or the value of the key `true` a bare `on:` is
    /// in YAML 1.1, if either is there.
    On(Option<NodeId>),
}

/// A semantic view's table's `primary_key`.
#[derive(Clone, Debug, PartialEq)]
pub enum PrimaryKey {
    /// None declared, or one of no columns.
    None,
    /// One column, `columns[0]` as Python indexes it: a list's first item,
    /// a mapping's entry keyed 0, a text's first character.
    One(NodeId),
    /// More than one, which the conversion leaves out without reading them.
    Composite,
}

/// A column of the view.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnItem {
    /// A metric view's dimension that is no mapping.
    NotAMapping {
        cfg: NodeId,
    },
    Column(ColumnSpec),
}

/// A column: its table if known, its section, and its entries as written.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnSpec {
    pub table: Option<usize>,
    pub section: Section,
    pub name: Option<NodeId>,
    pub expr: Option<NodeId>,
    pub data_type: Option<NodeId>,
    pub cfg: NodeId,
}

/// The section a column is declared in; a time dimension is a time column's
/// first candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Dimension,
    TimeDimension,
    Fact,
}

/// A semantic view's relationship: its name and its column pairs as
/// written; its tables and keys are read only for one pair
/// ([`ViewSpec::relationship`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RelationshipItem {
    pub name: NodeId,
    /// How many column pairs it has: `len(relationship_columns)`, a
    /// mapping's entries or a text's characters as a list's items.
    pub pairs: usize,
    pub cfg: NodeId,
}

/// A relationship of one column pair, read.
#[derive(Clone, Debug, PartialEq)]
pub struct Relationship {
    pub left_table: NodeId,
    pub left_column: NodeId,
    pub right_table: NodeId,
    pub right_column: NodeId,
}

/// A parameter of the view: its name, and its `data_type` and `default`
/// unless absent or null.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterItem {
    pub name: String,
    pub data_type: Option<NodeId>,
    pub default: Option<NodeId>,
}

/// A measure: its name, its `expr` unless absent or null, and whether it
/// has a `window`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureItem {
    pub name: String,
    pub expr: Option<NodeId>,
    pub window: bool,
}

/// Why a definition is refused before anything of it is read further: where
/// the old client failed, a refusal by name.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewRefusal {
    pub kind: ViewKind,
    pub reason: RefusalReason,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RefusalReason {
    /// The text is no YAML (the old client let PyYAML's error through).
    Yaml(YamlError),
    /// The definition is no mapping.
    NotAMapping,
    /// A metric view's definition misses its `source`.
    MissesSource,
    /// An entry the old client reads is missing (its `KeyError`).
    Misses(String),
    /// A section the old client iterates cannot be iterated (its
    /// `TypeError`).
    NotAList(String),
    /// A join nests itself, which the old client follows until Python
    /// stops it.
    NestsItself,
    /// An entry the old client looked up in a set or used as a key is a
    /// list or a mapping (its `TypeError`).
    Unhashable(String),
    /// A name the old client wrote into a pattern is not text (its
    /// `TypeError`).
    NotText(String),
}

impl fmt::Display for ViewRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = self.kind.noun();
        match &self.reason {
            RefusalReason::Yaml(error) => {
                write!(f, "The {noun} definition is not readable YAML: {error}")
            }
            RefusalReason::NotAMapping => write!(f, "The {noun} definition is not a YAML mapping"),
            RefusalReason::MissesSource => write!(f, "The {noun} definition misses a 'source'"),
            RefusalReason::Misses(entry) => write!(f, "The {noun} definition misses '{entry}'"),
            RefusalReason::NotAList(section) => {
                write!(f, "The {noun} definition's '{section}' is not a list")
            }
            RefusalReason::NestsItself => {
                write!(f, "The {noun} definition has a join nested in itself")
            }
            RefusalReason::Unhashable(entry) => {
                write!(
                    f,
                    "The {noun} definition's '{entry}' is a list or a mapping"
                )
            }
            RefusalReason::NotText(entry) => {
                write!(f, "The {noun} definition's '{entry}' is not text")
            }
        }
    }
}

impl std::error::Error for ViewRefusal {}

impl ViewSpec {
    pub(crate) fn refusal(&self, reason: RefusalReason) -> ViewRefusal {
        ViewRefusal {
            kind: self.kind,
            reason,
        }
    }

    /// The value of `node`.
    pub fn value(&self, node: NodeId) -> &Value {
        self.document.value(node)
    }

    /// The text, for a text value.
    pub fn text(&self, node: NodeId) -> Option<&str> {
        self.document.text(node)
    }

    /// The names of every join nested in the entry `cfg`, at any depth, in
    /// order, as the old client lists those a skipped join takes with it
    /// (`_collect_aliases`, `DB/metric_view.py:817-823`), through nested
    /// joins with no name. Reading them may meet what cannot be iterated,
    /// where the old client failed.
    pub fn nested_names(&mut self, cfg: NodeId) -> Result<Vec<String>, ViewRefusal> {
        let mut names = Vec::new();
        let mut visiting = Vec::new();
        self.collect(cfg, &mut names, &mut visiting)?;
        Ok(names)
    }

    fn collect(
        &mut self,
        cfg: NodeId,
        names: &mut Vec<String>,
        visiting: &mut Vec<NodeId>,
    ) -> Result<(), ViewRefusal> {
        if visiting.contains(&cfg) {
            return Err(self.refusal(RefusalReason::NestsItself));
        }
        visiting.push(cfg);
        for child in self.entries(self.document.get(cfg, "joins"), "joins")? {
            if !matches!(self.value(child), Value::Map(_)) {
                continue;
            }
            if let Some(name) = self.document.get(child, "name").and_then(|n| self.text(n)) {
                names.push(name.to_owned());
            }
            self.collect(child, names, visiting)?;
        }
        visiting.pop();
        Ok(())
    }

    /// The members `for x in section or []` visits: none for an absent or
    /// empty value, else as Python iterates it.
    pub(crate) fn entries(
        &mut self,
        section: Option<NodeId>,
        name: &str,
    ) -> Result<Vec<NodeId>, ViewRefusal> {
        let Some(node) = section.filter(|&n| self.document.is_set(n)) else {
            return Ok(Vec::new());
        };
        self.document
            .iterate(node)
            .ok_or_else(|| self.refusal(RefusalReason::NotAList(name.to_owned())))
    }

    /// `cfg[entry]`, a mapping's entry that must be there.
    pub(crate) fn required(&self, cfg: NodeId, entry: &str) -> Result<NodeId, ViewRefusal> {
        self.document
            .get(cfg, entry)
            .ok_or_else(|| self.refusal(RefusalReason::Misses(entry.to_owned())))
    }

    /// A relationship's tables and keys, read only for one column pair, in
    /// the old client's order (`RE/graph.py:2215-2218`): its
    /// `relationship_columns[0]` as Python indexes it.
    pub fn relationship(&mut self, item: &RelationshipItem) -> Result<Relationship, ViewRefusal> {
        let left_table = self.required(item.cfg, "left_table")?;
        let pairs = self.required(item.cfg, "relationship_columns")?;
        let pair = self
            .document
            .first(pairs)
            .ok_or_else(|| self.refusal(RefusalReason::Misses("relationship_columns".into())))?;
        let left_column = self.required(pair, "left_column")?;
        let right_table = self.required(item.cfg, "right_table")?;
        let right_column = self.required(pair, "right_column")?;
        Ok(Relationship {
            left_table,
            left_column,
            right_table,
            right_column,
        })
    }
}
