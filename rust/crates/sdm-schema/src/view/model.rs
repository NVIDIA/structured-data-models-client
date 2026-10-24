// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The logical model a view converts into (design S5): its tables, each
//! with its columns, as the source holds them or as expressions, and its
//! primary key, and the links between them. Its steps are the graph's part
//! of the old client's conversion (`RE/graph.py`):
//!
//! - [`ViewModel::add_table`], and [`ViewModel::add_join`], which keys and
//!   links a joined table and, refused, undoes every change it made to the
//!   tables already there (`:1820-1925`);
//! - [`ViewModel::add_relationship`], a semantic view's link (`:2209-2252`);
//! - the finish: [`measure_notes`] (`:376-421`), [`time_column`]
//!   (`:274-373`), [`ViewModel::drop_invalid_edges`] and
//!   [`ViewModel::validate`] (`:3256-3371`).
//!
//! Each step decides from facts the core read before it (a key's count, the
//! columns of a table as its source holds them, which column an expression
//! names) and answers with typed facts the core words. A step that needs
//! a read on the way, the time column's, asks for it through the closures
//! it is given.

use sdm_contract::ops::{Dtype, Stype};

/// A column of a table of the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelColumn {
    pub name: String,
    /// Its expression, as the view writes it; none for a column of the
    /// source.
    pub expr: Option<String>,
    pub dtype: Dtype,
    pub stype: Stype,
}

/// A table of the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTable {
    pub name: String,
    pub columns: Vec<ModelColumn>,
    pub primary_key: Option<String>,
}

impl ModelTable {
    /// The column named `name`, exactly.
    pub fn column(&self, name: &str) -> Option<&ModelColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    fn column_mut(&mut self, name: &str) -> Option<&mut ModelColumn> {
        self.columns.iter_mut().find(|c| c.name == name)
    }

    fn remove(&mut self, name: &str) -> Option<ModelColumn> {
        let at = self.columns.iter().position(|c| c.name == name)?;
        if self.primary_key.as_deref() == Some(name) {
            self.primary_key = None;
        }
        Some(self.columns.remove(at))
    }
}

/// A link from a foreign key of `source` to the primary key of `target`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub source: String,
    pub key: String,
    pub target: String,
}

/// The model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewModel {
    /// Its tables, in the order they were added.
    pub tables: Vec<ModelTable>,
    pub edges: Vec<Edge>,
}

/// Why a join key, counted, cannot be the primary key a join gives its
/// table (`_join_key_reason`, `:250-271`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyReason {
    /// No row holds a value of it.
    NoValue,
    /// `repeats` of `rows` rows repeat a value.
    NotUnique { repeats: u64, rows: u64 },
}

/// Why a join key counted `rows` rows, `distinct` distinct values among
/// those that hold one, and `repeats` rows repeating a value, cannot key
/// its table, or `None` if it can. Nothing counted (`None`) passes; a row
/// with no key is one no join reaches, so it does not count against it.
pub fn join_key_reason(
    rows: Option<u64>,
    distinct: Option<u64>,
    repeats: u64,
) -> Option<KeyReason> {
    let (rows, distinct) = (rows?, distinct?);
    if distinct == 0 {
        return Some(KeyReason::NoValue);
    }
    (repeats > 0).then_some(KeyReason::NotUnique { repeats, rows })
}

/// What a join's key was counted as, before the join is added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyCheck {
    Counted {
        rows: Option<u64>,
        distinct: Option<u64>,
        repeats: u64,
    },
    /// The count failed, so the key is kept unchecked.
    Failed,
}

/// A metric view's join, ready to add: its table (named by the join's
/// alias, the view's columns on it added), the table it keys and its key,
/// the table linking to that one and its key, how the key was counted, and
/// each key column as the source holds it, for a key a table lacks or holds
/// as an expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinStep {
    pub table: ModelTable,
    pub one: (String, String),
    pub many: (String, String),
    pub count: KeyCheck,
    pub source_columns: Vec<(String, ModelColumn)>,
}

/// What an added join says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinNote {
    /// Its key was kept unchecked, since counting the keys failed.
    KeptUnchecked { key: String },
    /// A derived column of `table` used as a key was replaced by the source
    /// column of its name.
    ReplacedDerived { key: String, table: String },
}

/// Why a join is left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinRefusal {
    /// The table it keys already has another primary key.
    AlreadyKeyed { table: String, key: String },
    /// Its key's count refuses it.
    Key {
        table: String,
        key: String,
        reason: KeyReason,
    },
    /// Its keys could not be linked.
    NotLinked(LinkFailure),
}

/// Why two keys could not be linked: the old client's `ValueError`s on the
/// way, typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkFailure {
    /// The source holds no column of the key's name.
    NoSourceColumn { key: String },
    /// The source column's type is not supported.
    UnsupportedType { table: String, key: String },
    /// The key's type cannot be an ID (`Column.stype`'s refusal).
    KeyType { key: String, dtype: Dtype },
    /// The link is there already.
    EdgeExists(Edge),
    /// The foreign key's type cannot link (`Graph.link`).
    ForeignKeyType {
        key: String,
        table: String,
        dtype: Dtype,
    },
}

/// A semantic view's relationship, ready to add: the referencing table and
/// key, the referenced table and key, and the referencing key's column as
/// its base table holds it, if it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationshipStep {
    pub left_table: String,
    pub left_key: String,
    pub right_table: String,
    pub right_key: String,
    pub left_key_column: Option<ModelColumn>,
}

/// Why a relationship is left out (`:2224-2252`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationshipRefusal {
    /// The referenced key is not its table's primary key.
    NotPrimaryKey {
        key: String,
        table: String,
    },
    /// The referencing key is its table's primary key.
    IsPrimaryKey {
        key: String,
        table: String,
    },
    /// A table it names is not in the model.
    NoTable(String),
    NotLinked(LinkFailure),
}

/// Why a model cannot be used (`Graph.validate`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRefusal {
    /// It has no table.
    NoTable,
    /// A table of that name is there already.
    TableTaken(String),
    /// An edge is invalid.
    Edge(Edge, EdgeReason),
}

fn can_be_id(dtype: Dtype) -> bool {
    is_int(dtype) || is_float(dtype) || dtype == Dtype::String || dtype == Dtype::Binary
}

fn is_int(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Int | Dtype::Byte | Dtype::Int16 | Dtype::Int32 | Dtype::Int64
    )
}

fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Float | Dtype::Float32 | Dtype::Float64)
}

/// One change a join or relationship made to a table already in the model,
/// to undo on a refusal.
enum Change {
    /// A column added.
    Added { table: String, key: String },
    /// A column replaced, with what it was.
    Replaced { table: String, column: ModelColumn },
}

impl ViewModel {
    pub fn new() -> ViewModel {
        ViewModel::default()
    }

    /// The table named `name`.
    pub fn table(&self, name: &str) -> Option<&ModelTable> {
        self.tables.iter().find(|t| t.name == name)
    }

    fn table_mut(&mut self, name: &str) -> Option<&mut ModelTable> {
        self.tables.iter_mut().find(|t| t.name == name)
    }

    /// Adds a table; a name taken refuses it (`Graph.add_table`).
    pub fn add_table(&mut self, table: ModelTable) -> Result<(), ModelRefusal> {
        if self.table(&table.name).is_some() {
            return Err(ModelRefusal::TableTaken(table.name));
        }
        self.tables.push(table);
        Ok(())
    }

    /// Whether an edge points at `table`, which settles a lone edit stamp's
    /// time column (`_settle_time_columns`).
    pub fn linked(&self, table: &str) -> bool {
        self.edges.iter().any(|e| e.target == table)
    }

    /// Adds a metric view's join (`:1833-1925`): checks that the table it
    /// keys has no other key and that its key's count allows it, adds each
    /// key its table lacks or holds derived, from the source, keys the one
    /// side, adds the joined table and links the many side to it. Refused at
    /// any step, it undoes every change, so the tables already in the model
    /// are as they were; the notes of a refused join are its caller's to
    /// drop, with the column messages of its table.
    pub fn add_join(&mut self, step: JoinStep) -> Result<Vec<JoinNote>, JoinRefusal> {
        let JoinStep {
            table,
            one,
            many,
            count,
            source_columns,
        } = step;
        let alias = table.name.clone();
        let mut child = Some(table);
        let mut notes = Vec::new();
        let mut changes: Vec<Change> = Vec::new();
        let mut keyed: Option<(String, String, Stype)> = None;
        let outcome = (|| -> Result<(), JoinRefusal> {
            let one_key_now = self.key_of(&child, &one.0);
            if let Some(key) = &one_key_now
                && *key != one.1
            {
                return Err(JoinRefusal::AlreadyKeyed {
                    table: one.0.clone(),
                    key: key.clone(),
                });
            }
            if one_key_now.is_none() {
                match count {
                    KeyCheck::Failed => notes.push(JoinNote::KeptUnchecked { key: one.1.clone() }),
                    KeyCheck::Counted {
                        rows,
                        distinct,
                        repeats,
                    } => {
                        if let Some(reason) = join_key_reason(rows, distinct, repeats) {
                            return Err(JoinRefusal::Key {
                                table: one.0.clone(),
                                key: one.1.clone(),
                                reason,
                            });
                        }
                    }
                }
            }
            for (table, key) in [&one, &many] {
                let source = source_columns
                    .iter()
                    .find(|(t, c)| t == table && c.name == *key)
                    .map(|(_, c)| c.clone());
                let target = self.table_or_child(&mut child, table);
                match target.column(key).map(|c| c.expr.is_some()) {
                    None => {
                        target.columns.push(
                            source_column(table, key, source).map_err(JoinRefusal::NotLinked)?,
                        );
                        changes.push(Change::Added {
                            table: table.clone(),
                            key: key.clone(),
                        });
                    }
                    Some(true) => {
                        let column =
                            source_column(table, key, source).map_err(JoinRefusal::NotLinked)?;
                        let before = target.remove(key).unwrap();
                        target.columns.push(column);
                        changes.push(Change::Replaced {
                            table: table.clone(),
                            column: before,
                        });
                        notes.push(JoinNote::ReplacedDerived {
                            key: key.clone(),
                            table: table.clone(),
                        });
                    }
                    Some(false) => {}
                }
            }
            if one_key_now.is_none() {
                let target = self.table_or_child(&mut child, &one.0);
                let column = target.column_mut(&one.1).unwrap();
                if !can_be_id(column.dtype) {
                    return Err(JoinRefusal::NotLinked(LinkFailure::KeyType {
                        key: one.1.clone(),
                        dtype: column.dtype,
                    }));
                }
                keyed = Some((one.0.clone(), one.1.clone(), column.stype));
                column.stype = Stype::Id;
                target.primary_key = Some(one.1.clone());
            }
            self.tables.push(child.take().unwrap());
            self.link(&many.0, &many.1, &one.0)
                .map_err(JoinRefusal::NotLinked)
        })();
        if let Err(refusal) = outcome {
            if child.is_none() {
                self.tables.retain(|t| t.name != alias);
            }
            if let Some((table, key, stype)) = keyed
                && let Some(target) = self.table_mut(&table)
            {
                target.primary_key = None;
                if let Some(column) = target.column_mut(&key) {
                    column.stype = stype;
                }
            }
            for change in changes.into_iter().rev() {
                let (table, key, before) = match change {
                    Change::Added { table, key } => (table, key, None),
                    Change::Replaced { table, column } => {
                        (table.clone(), column.name.clone(), Some(column))
                    }
                };
                if let Some(target) = self.table_mut(&table) {
                    target.remove(&key);
                    if let Some(column) = before {
                        target.columns.push(column);
                    }
                }
            }
            return Err(refusal);
        }
        Ok(notes)
    }

    /// The primary key of `name`, the joined table's while it is not in
    /// the model yet.
    fn key_of(&self, child: &Option<ModelTable>, name: &str) -> Option<String> {
        match child {
            Some(child) if child.name == name => child.primary_key.clone(),
            _ => self.table(name).and_then(|t| t.primary_key.clone()),
        }
    }

    fn table_or_child<'a>(
        &'a mut self,
        child: &'a mut Option<ModelTable>,
        name: &str,
    ) -> &'a mut ModelTable {
        match child {
            Some(child) if child.name == name => child,
            _ => self
                .table_mut(name)
                .expect("a join's tables are in the model or the join"),
        }
    }

    /// Links `key` of `source` to the primary key of `target`
    /// (`Graph.link`, `:2877-2950`): the link not there already, both tables
    /// there, the key a column of `source` whose type can link; the key
    /// becomes an ID.
    fn link(&mut self, source: &str, key: &str, target: &str) -> Result<(), LinkFailure> {
        let edge = Edge {
            source: source.to_owned(),
            key: key.to_owned(),
            target: target.to_owned(),
        };
        if self.edges.contains(&edge) {
            return Err(LinkFailure::EdgeExists(edge));
        }
        let column = self
            .table_mut(source)
            .and_then(|t| t.column_mut(key))
            .ok_or_else(|| LinkFailure::NoSourceColumn {
                key: key.to_owned(),
            })?;
        if !can_be_id(column.dtype) {
            return Err(LinkFailure::ForeignKeyType {
                key: key.to_owned(),
                table: source.to_owned(),
                dtype: column.dtype,
            });
        }
        column.stype = Stype::Id;
        self.edges.push(edge);
        Ok(())
    }

    /// Adds a semantic view's relationship (`:2224-2252`): the referenced
    /// key must be its table's primary key and the referencing key not its
    /// own; a referencing key its table lacks is added from the base
    /// table, and taken back if the link fails.
    pub fn add_relationship(&mut self, step: RelationshipStep) -> Result<(), RelationshipRefusal> {
        let right = self
            .table(&step.right_table)
            .ok_or_else(|| RelationshipRefusal::NoTable(step.right_table.clone()))?;
        if right.primary_key.as_deref() != Some(step.right_key.as_str()) {
            return Err(RelationshipRefusal::NotPrimaryKey {
                key: step.right_key,
                table: step.right_table,
            });
        }
        let left = self
            .table_mut(&step.left_table)
            .ok_or_else(|| RelationshipRefusal::NoTable(step.left_table.clone()))?;
        if left.primary_key.as_deref() == Some(step.left_key.as_str()) {
            return Err(RelationshipRefusal::IsPrimaryKey {
                key: step.left_key,
                table: step.left_table,
            });
        }
        let mut added = false;
        if left.column(&step.left_key).is_none() {
            let column = source_column(
                &step.left_table,
                &step.left_key,
                step.left_key_column.clone(),
            )
            .map_err(RelationshipRefusal::NotLinked)?;
            left.columns.push(column);
            added = true;
        }
        if let Err(failure) = self.link(&step.left_table, &step.left_key, &step.right_table) {
            if added && let Some(left) = self.table_mut(&step.left_table) {
                left.remove(&step.left_key);
            }
            return Err(RelationshipRefusal::NotLinked(failure));
        }
        Ok(())
    }

    /// Takes out every edge [`edge_reason`] refuses, saying why
    /// (`_drop_invalid_edges`).
    pub fn drop_invalid_edges(&mut self) -> Vec<(Edge, EdgeReason)> {
        let mut dropped = Vec::new();
        for edge in self.edges.clone() {
            if let Some(reason) = edge_reason(&self.edge_facts(&edge)) {
                self.edges.retain(|e| *e != edge);
                dropped.push((edge, reason));
            }
        }
        dropped
    }

    /// The model is refused if it has no table, or an invalid edge
    /// (`Graph.validate`).
    pub fn validate(&self) -> Result<(), ModelRefusal> {
        if self.tables.is_empty() {
            return Err(ModelRefusal::NoTable);
        }
        for edge in &self.edges {
            if let Some(reason) = edge_reason(&self.edge_facts(edge)) {
                return Err(ModelRefusal::Edge(edge.clone(), reason));
            }
        }
        Ok(())
    }

    fn edge_facts(&self, edge: &Edge) -> EdgeFacts {
        let source = self.table(&edge.source);
        let target = self.table(&edge.target);
        EdgeFacts {
            fkey: edge.key.clone(),
            fkey_dtype: source.and_then(|t| t.column(&edge.key)).map(|c| c.dtype),
            target_key: target.and_then(|t| {
                let key = t.primary_key.as_ref()?;
                Some((key.clone(), t.column(key)?.dtype))
            }),
            source_key: source.and_then(|t| t.primary_key.clone()),
        }
    }
}

/// The key column as the source holds it, or why it cannot be added
/// (`Table.add_column`'s refusals).
fn source_column(
    table: &str,
    key: &str,
    column: Option<ModelColumn>,
) -> Result<ModelColumn, LinkFailure> {
    let column = column.ok_or_else(|| LinkFailure::NoSourceColumn {
        key: key.to_owned(),
    })?;
    if column.dtype == Dtype::Unsupported {
        return Err(LinkFailure::UnsupportedType {
            table: table.to_owned(),
            key: key.to_owned(),
        });
    }
    Ok(column)
}

/// What an edge is judged by: its foreign key's type (`None` if its table
/// no longer has the column), the target's primary key and type, and the
/// source's primary key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeFacts {
    pub fkey: String,
    pub fkey_dtype: Option<Dtype>,
    pub target_key: Option<(String, Dtype)>,
    pub source_key: Option<String>,
}

/// Why an edge cannot be used (`_edge_error`, `:3283-3348`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdgeReason {
    /// Its table no longer has the foreign key's column.
    NoColumn,
    /// The target has no primary key.
    NoPrimaryKey,
    /// The foreign key is its own table's primary key.
    KeyIsPrimary,
    /// The foreign key is neither a number nor a string.
    NotNumberOrString { dtype: Dtype },
    /// The two keys' types do not match: number with number, string with
    /// string.
    Incompatible {
        fkey_dtype: Dtype,
        pkey: String,
        pkey_dtype: Dtype,
    },
}

/// Why an edge cannot be used, or `None`.
pub fn edge_reason(facts: &EdgeFacts) -> Option<EdgeReason> {
    let Some(fkey_dtype) = facts.fkey_dtype else {
        return Some(EdgeReason::NoColumn);
    };
    let Some((pkey, pkey_dtype)) = &facts.target_key else {
        return Some(EdgeReason::NoPrimaryKey);
    };
    if facts.source_key.as_deref() == Some(facts.fkey.as_str()) {
        return Some(EdgeReason::KeyIsPrimary);
    }
    let number = |d: Dtype| is_int(d) || is_float(d);
    let string = |d: Dtype| matches!(d, Dtype::String | Dtype::Binary);
    if !number(fkey_dtype) && !string(fkey_dtype) {
        return Some(EdgeReason::NotNumberOrString { dtype: fkey_dtype });
    }
    if number(fkey_dtype) != number(*pkey_dtype) || string(fkey_dtype) != string(*pkey_dtype) {
        return Some(EdgeReason::Incompatible {
            fkey_dtype,
            pkey: pkey.clone(),
            pkey_dtype: *pkey_dtype,
        });
    }
    None
}

/// A column of the fact table as the measure notes read it: its name,
/// whether it is an expression, and the column it is if it is one column
/// and nothing else (`_renamed_column`, the data layer's to read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureColumn {
    pub name: String,
    pub expression: bool,
    pub renames: Option<String>,
}

/// A measure's predictive query aggregation: its function, and the column
/// it aggregates, none for a count of rows.
pub type Aggregation = (String, Option<String>);

/// A measure of the view and the predictive query aggregation it is
/// (`MeasureNote`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureNote {
    pub measure: String,
    pub table: String,
    pub function: String,
    /// The column it aggregates; none for a count of rows.
    pub column: Option<String>,
}

/// The note of each simple measure over the fact table `table`
/// (`_measure_notes`, `:376-421`): a measure names a column in any case,
/// and a dimension renaming a column stands for it; a dimension that has a
/// source column's name but holds something else shadows it, and gets no
/// note. `measures` are each measure's name and aggregation, if any.
pub fn measure_notes(
    table: &str,
    columns: &[MeasureColumn],
    source_columns: &[String],
    measures: &[(String, Option<Aggregation>)],
) -> Vec<MeasureNote> {
    let folded = |name: &str| name.to_lowercase();
    let column = |name: &str| columns.iter().find(|c| folded(&c.name) == folded(name));
    let in_source = |name: &str| source_columns.iter().any(|s| folded(s) == folded(name));
    let renamed_as = |name: &str| {
        columns
            .iter()
            .find(|c| {
                c.expression
                    && c.renames
                        .as_deref()
                        .is_some_and(|r| folded(r) == folded(name))
            })
            .map(|c| c.name.clone())
    };
    let mut notes = Vec::new();
    for (measure, aggregation) in measures {
        let Some((function, name)) = aggregation else {
            continue;
        };
        let target = match name {
            None => None,
            Some(name) => match column(name) {
                None => match renamed_as(name) {
                    Some(target) => Some(target),
                    None => continue,
                },
                Some(c)
                    if !c.expression
                        || !in_source(name)
                        || c.renames
                            .as_deref()
                            .is_some_and(|r| folded(r) == folded(name)) =>
                {
                    Some(c.name.clone())
                }
                Some(_) => continue,
            },
        };
        notes.push(MeasureNote {
            measure: measure.clone(),
            table: table.to_owned(),
            function: function.clone(),
            column: target,
        });
    }
    notes
}

/// A column of a table, as its time column is taken: whether it holds
/// timestamps, whether it is the source's, and the source column it
/// truncates or casts to a date or time, if its expression does one
/// (`derived_column`, the data layer's to read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeInput {
    pub name: String,
    pub timestamp: bool,
    pub source: bool,
    pub derived: Option<String>,
}

/// A source column to add to take as the time column: `raw`, under `name`,
/// its own unless a dimension has it (an expression reading `raw` then).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawAdd {
    pub name: String,
    pub raw: String,
}

/// What adding a column gave: whether it was added, and whether it holds
/// timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Added {
    pub added: bool,
    pub timestamp: bool,
}

/// A time column chosen among candidates, and the column it falls back on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Choice {
    pub column: Option<String>,
    pub fallback: Option<String>,
}

/// The time column taken in place of the dimensions that truncate or cast
/// it, or because a window measure orders by it (`TimeColumnNote`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeColumnNote {
    pub table: String,
    pub column: String,
    pub instead_of: Vec<String>,
    /// The source column it is, added under a name of its own because a
    /// dimension has its name.
    pub source_column: Option<String>,
    /// Whether it was added as a column, under its own name.
    pub added: bool,
    pub window_order: bool,
}

/// What taking the time column did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeOutcome {
    pub chosen: Option<String>,
    /// The columns added and kept, in order.
    pub kept: Vec<String>,
    /// The columns added and taken out again.
    pub removed: Vec<String>,
    /// The dimensions the chosen column was taken in place of.
    pub instead_of: Vec<String>,
    pub note: Option<TimeColumnNote>,
}

/// Takes the time column of a view's table, and notes it where the view's
/// own columns would have led elsewhere (`_infer_view_time_column`,
/// `:274-373`). A truncated or cast date starts at or before the column it
/// derives from, so that column is offered in its place, added under a
/// name of its own (`{raw}_raw`, `{raw}_raw_2`, ...) where a dimension has
/// its name; a dimension a window measure orders by is offered alone
/// first. `add` adds a column, as the core reads it; `choose` takes a time
/// column among candidates from the rows the core read. `source_columns`
/// are each source column with whether it holds dates or times.
pub fn time_column(
    table: &str,
    columns: &[TimeInput],
    source_columns: &[(String, bool)],
    window_orders: &[String],
    mut add: impl FnMut(&RawAdd) -> Added,
    mut choose: impl FnMut(&[String]) -> Choice,
) -> TimeOutcome {
    let timestamps: Vec<&TimeInput> = columns.iter().filter(|c| c.timestamp).collect();
    let source = |name: &str| {
        let folded = name.to_lowercase();
        source_columns
            .iter()
            .find(|(s, _)| s.to_lowercase() == folded)
    };
    // Each timestamp derived from a source date or time, and that column.
    let mut derived_from: Vec<(String, String)> = timestamps
        .iter()
        .filter_map(|c| {
            let (column, timelike) = source(c.derived.as_deref()?)?;
            timelike.then(|| (c.name.clone(), column.clone()))
        })
        .collect();
    let mut names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
    let mut added: Vec<String> = Vec::new();
    let mut renamed: Vec<(String, String)> = Vec::new();
    let mut raws: Vec<String> = Vec::new();
    for (_, raw) in &derived_from {
        if !raws.contains(raw) {
            raws.push(raw.clone());
        }
    }
    for raw in raws {
        let taken: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        let mut name = raw.clone();
        let usable = match columns.iter().find(|c| c.name == raw) {
            Some(existing) if existing.source => existing.timestamp,
            _ => {
                if taken.contains(&raw.to_lowercase()) {
                    name = format!("{raw}_raw");
                    let mut suffix = 2;
                    while taken.contains(&name.to_lowercase()) {
                        name = format!("{raw}_raw_{suffix}");
                        suffix += 1;
                    }
                }
                let answer = add(&RawAdd {
                    name: name.clone(),
                    raw: raw.clone(),
                });
                if answer.added {
                    if name != raw {
                        renamed.push((name.clone(), raw.clone()));
                    }
                    added.push(name.clone());
                    names.push(name.clone());
                }
                answer.added && answer.timestamp
            }
        };
        derived_from = derived_from
            .into_iter()
            .filter(|(_, v)| *v != raw || usable)
            .map(|(k, v)| if v == raw { (k, name.clone()) } else { (k, v) })
            .collect();
    }
    let from = |name: &str| {
        derived_from
            .iter()
            .find(|(k, _)| k == name)
            .map_or(name.to_owned(), |(_, v)| v.clone())
    };
    let unique = |items: Vec<String>| {
        let mut out: Vec<String> = Vec::new();
        for item in items {
            if !out.contains(&item) {
                out.push(item);
            }
        }
        out
    };
    let candidates = unique(timestamps.iter().map(|c| from(&c.name)).collect());
    let axes = unique(
        window_orders
            .iter()
            .filter(|n| timestamps.iter().any(|c| c.name == **n))
            .map(|n| from(n))
            .collect(),
    );
    let mut choice: Option<Choice> = None;
    let mut chosen = None;
    if !axes.is_empty() && candidates.len() > 1 {
        let answer = choose(&axes);
        chosen = answer.column.clone();
        choice = Some(answer);
    }
    if chosen.is_none() && !candidates.is_empty() {
        let answer = choose(&candidates);
        chosen = answer.column.clone();
        choice = Some(answer);
    }
    let kept_set: Vec<Option<String>> = choice
        .as_ref()
        .map_or_else(Vec::new, |c| vec![c.column.clone(), c.fallback.clone()]);
    let (kept, removed): (Vec<String>, Vec<String>) = added
        .iter()
        .cloned()
        .partition(|name| kept_set.contains(&Some(name.clone())));
    let window_order = chosen.as_ref().is_some_and(|c| axes.contains(c)) && candidates.len() > 1;
    let instead_of: Vec<String> = derived_from
        .iter()
        .filter(|(_, raw)| Some(raw) == chosen.as_ref())
        .map(|(k, _)| k.clone())
        .collect();
    let note = chosen
        .as_ref()
        .filter(|_| window_order || !instead_of.is_empty())
        .map(|column| {
            let source_column = renamed
                .iter()
                .find(|(n, _)| n == column)
                .map(|(_, r)| r.clone());
            TimeColumnNote {
                table: table.to_owned(),
                column: column.clone(),
                instead_of: instead_of.clone(),
                added: added.contains(column) && source_column.is_none(),
                source_column,
                window_order,
            }
        });
    TimeOutcome {
        chosen,
        kept,
        removed,
        instead_of,
        note,
    }
}
