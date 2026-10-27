// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema as a host holds it, as plain data (contract v28, T3.16): read
//! into the schema the core works on, and written back.
//!
//! A host hands `edit_schema` and `materialize` its schema as it stands
//! ([`HeldSchema`]), whatever made it: described on the core, built by
//! hand, edited, copied or unpickled. Each column the source was asked of
//! carries what the source said of it ([`SourceFacts`]), which the host
//! keeps on the column and hands back whole; nothing the core answered
//! earlier is merged in. A column with no facts is one the source was never
//! asked of: `materialize` reads its facts from the source first, as
//! `describe_source` reads them (`sdm_core::discover`), and `edit_schema`,
//! which reads no data, edits it by its data type, as the reference's
//! setters edit any column.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::data::Relation;
use sdm_contract::ops::{
    DescribedColumn, DescribedTable, Dtype, HeldColumn, HeldSchema, HeldTable, SourceDescription,
    SourceFacts,
};
use sdm_contract::{ColumnType, CoreError, Detail, Sentence};

use crate::assemble::invalid;

/// A column by its table's name in the schema and its own name.
pub type ColumnAt = (String, String);

/// The schema the core works on, read from the schema as the host holds it,
/// and the columns that held no facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    pub description: SourceDescription,
    /// The columns the host held no facts of and `found` gave none for,
    /// each read by its data type ([`by_data_type`]); [`held`] writes them
    /// back with none.
    pub unfacted: BTreeSet<ColumnAt>,
    /// The data type the source gave each column the host holds as another
    /// (a column the caller retyped): the schema is read by the host's, as
    /// the old client's checks read it, and the source's is what its data
    /// is read as ([`Read::by_the_source`]).
    pub retyped: BTreeMap<ColumnAt, Dtype>,
    /// The data type name the host gave each column, as the reference's
    /// checks of the caller's schema read it (a meaning, a key, a join's
    /// family, [`crate::assemble::join_family`]).
    pub types: crate::assemble::HostTypes,
}

impl Read {
    /// The schema with each column of the data type the source gave it:
    /// the one the reference's engine reads a table in place by.
    pub fn by_the_source(&self) -> SourceDescription {
        let mut description = self.description.clone();
        for table in &mut description.tables {
            for column in &mut table.columns {
                if let Some(dtype) = self.retyped.get(&(table.name.clone(), column.name.clone())) {
                    column.dtype = *dtype;
                }
            }
        }
        description
    }
}

/// The facts `column` of `table` holds of its table's relation: none where
/// it holds none, or holds those of another relation (a table the host
/// pointed elsewhere since), which are not this column's.
pub fn facts_held<'a>(table: &HeldTable, column: &'a HeldColumn) -> Option<&'a SourceFacts> {
    column
        .source
        .as_ref()
        .filter(|facts| facts.relation == table.source)
}

/// The columns of `held` that carry no facts of their table's relation
/// ([`facts_held`]), by table, in the host's order: what `materialize` asks
/// the source for.
pub fn lacking_facts(held: &HeldSchema) -> Vec<(&HeldTable, Vec<&str>)> {
    held.tables
        .iter()
        .filter_map(|table| {
            let lacking: Vec<&str> = table
                .columns
                .iter()
                .filter(|c| facts_held(table, c).is_none())
                .map(|c| c.name.as_str())
                .collect();
            (!lacking.is_empty()).then_some((table, lacking))
        })
        .collect()
}

/// What the edits of the schema read of a column the source was never asked
/// of: its data type as the host names it, as the reference's setters read
/// a column's `dtype` ([`wire_dtype`]; a name no data type has is
/// [`Dtype::Unsupported`], which an edit lets through as the old client's
/// checks do), held nullable and not unique, of no exact numbers or
/// declared type. `received` is not read for such a column: only a column
/// of exact numbers is retyped by its role ([`crate::assemble::read_exact_numbers`]).
pub fn by_data_type(relation: &Relation, dtype: &str) -> SourceFacts {
    SourceFacts {
        relation: relation.clone(),
        dtype: wire_dtype(dtype),
        received: ColumnType::Null,
        nullable: true,
        unique: false,
        declared_type: None,
        exact_number: None,
    }
}

/// Every data type name a host may give a column, and the data type it
/// names (T3.16): the client's wire names, as the old client maps them to
/// the language's (`int64` and `int32` an integer, `float64` and `float32`
/// a float, a timestamp of any unit a time), then the language's own names,
/// which the old client passes through. The one table of them: the Python
/// client reads it from the core (`sdm_client._native.DTYPE_NAMES`) for its
/// own checks of a column's meaning.
pub const DTYPE_NAMES: [(&str, Dtype); 22] = [
    ("int64", Dtype::Int),
    ("int32", Dtype::Int),
    ("float64", Dtype::Float),
    ("float32", Dtype::Float),
    ("timestamp[s]", Dtype::Time),
    ("timestamp[ms]", Dtype::Time),
    ("timestamp[us]", Dtype::Time),
    ("timestamp[ns]", Dtype::Time),
    ("bool", Dtype::Bool),
    ("int", Dtype::Int),
    ("byte", Dtype::Byte),
    ("int16", Dtype::Int16),
    ("float", Dtype::Float),
    ("string", Dtype::String),
    ("binary", Dtype::Binary),
    ("date", Dtype::Date),
    ("time", Dtype::Time),
    ("timedelta", Dtype::Timedelta),
    ("floatlist", Dtype::Floatlist),
    ("intlist", Dtype::Intlist),
    ("stringlist", Dtype::Stringlist),
    ("unsupported", Dtype::Unsupported),
];

/// The data type a host names a column by ([`DTYPE_NAMES`]); `None` for a
/// name no data type has, which the old client's checks let through.
pub fn named_dtype(name: &str) -> Option<Dtype> {
    DTYPE_NAMES
        .iter()
        .find(|(given, _)| *given == name)
        .map(|(_, dtype)| *dtype)
}

/// Whether a column the caller gave the data type named `name` can be the
/// source's column of type `source` (the owner's plan B, item 3, T3.4): the
/// two of one kind ([`kind`]); a name no data type has, or the language's
/// `unsupported`, taking any.
pub fn type_matches(name: &str, source: Dtype) -> bool {
    match named_dtype(name) {
        None | Some(Dtype::Unsupported) => true,
        Some(given) => kind(given) == kind(source),
    }
}

/// Check caller types against fresh source facts before applying held meanings.
/// Exact numeric columns keep their existing role-dependent handling.
pub fn check_source_types(
    held: &HeldSchema,
    found: &BTreeMap<ColumnAt, SourceFacts>,
) -> Result<(), CoreError> {
    for table in &held.tables {
        for column in &table.columns {
            let Some(source) = found.get(&(table.name.clone(), column.name.clone())) else {
                continue;
            };
            if source.exact_number.is_none() && !type_matches(&column.dtype, source.dtype) {
                return Err(CoreError::new(
                    Sentence::ColumnTypeUnmatched,
                    "a column's type the caller gave is not the one the source holds",
                )
                .with_field("field", Detail::text("schema"))
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name))
                .with_field("dtype", Detail::text(&column.dtype))
                .with_field("source_dtype", Detail::text(source.dtype.wire_name())));
            }
        }
    }
    Ok(())
}

/// The kind of value a column of `dtype` holds, which a caller's type and
/// the source's must share ([`type_matches`]): a number of any width, a
/// list of numbers, a list of strings, a time or a date, or each other type
/// its own.
fn kind(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Int
        | Dtype::Byte
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Float
        | Dtype::Float32
        | Dtype::Float64 => "number",
        Dtype::Intlist | Dtype::Floatlist => "numbers",
        Dtype::Stringlist => "strings",
        Dtype::Date | Dtype::Time => "time",
        Dtype::String => "string",
        Dtype::Bool => "bool",
        Dtype::Binary => "binary",
        Dtype::Timedelta => "timedelta",
        Dtype::Unsupported => "unsupported",
    }
}

/// [`named_dtype`], a name no data type has read as [`Dtype::Unsupported`],
/// which an edit lets through as the old client does
/// ([`crate::edit::edit`]).
pub fn wire_dtype(name: &str) -> Dtype {
    named_dtype(name).unwrap_or(Dtype::Unsupported)
}

/// The data type the schema reads `column` by, of which the source said
/// `facts`: the one the host holds, as the old client's checks read the
/// caller's schema, where it names another than the source's (a column the
/// caller retyped, or one named by no data type, which the old client's
/// checks let through, [`Dtype::Unsupported`]); the source's, which tells a
/// date from a time of day, where the two name one type on the wire.
fn held_dtype(column: &HeldColumn, facts: &SourceFacts) -> Dtype {
    let held = wire_dtype(&column.dtype);
    if held.wire_name() == facts.dtype.wire_name() {
        facts.dtype
    } else {
        held
    }
}

/// `held` as the schema the core works on: each table the host holds, in its
/// order, its source, key, time column, repeated keys and links the host's;
/// each column's meanings the host's, and what the source said of it: the
/// facts `found` gives (`materialize`'s reading of the source, or of the
/// table as it was kept), else those the host holds, else, with `by_type`,
/// those its data type gives
/// ([`by_data_type`], `edit_schema`'s). A column with neither is left out,
/// as the reference's engine reads only the columns its source holds. A
/// table or column the host holds twice is the host's fault,
/// `INVALID_REQUEST` naming `schema`.
pub fn read(
    held: HeldSchema,
    found: &BTreeMap<ColumnAt, SourceFacts>,
    by_type: bool,
) -> Result<Read, CoreError> {
    let mut tables = Vec::with_capacity(held.tables.len());
    let mut unfacted = BTreeSet::new();
    let mut retyped = BTreeMap::new();
    let mut types = crate::assemble::HostTypes::default();
    let links = held.links;
    let mut names = BTreeSet::new();
    for table in held.tables {
        if !names.insert(table.name.clone()) {
            return Err(invalid(
                Sentence::SchemaTableTwice,
                "the schema holds a table twice",
                "schema",
            )
            .with_field("table", Detail::text(&table.name)));
        }
        let mut seen = BTreeSet::new();
        let mut columns = Vec::with_capacity(table.columns.len());
        for column in &table.columns {
            if !seen.insert(column.name.clone()) {
                return Err(invalid(
                    Sentence::SchemaColumnTwice,
                    "a table holds a column twice",
                    "schema",
                )
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name)));
            }
            let at = (table.name.clone(), column.name.clone());
            let facts = match found.get(&at) {
                Some(facts) => facts.clone(),
                None => match facts_held(&table, column) {
                    Some(facts) => facts.clone(),
                    None if by_type => {
                        unfacted.insert(at.clone());
                        by_data_type(&table.source, &column.dtype)
                    }
                    None => continue,
                },
            };
            // The host's name of the type, which a join reads; but for a
            // column of exact numbers, whose type its role decides (T2.21).
            if facts.exact_number.is_none() {
                types.names.insert(at.clone(), column.dtype.clone());
            }
            if named_dtype(&column.dtype).is_none() {
                types.unknown.insert(at.clone());
            }
            let dtype = held_dtype(column, &facts);
            if dtype != facts.dtype && !unfacted.contains(&at) {
                retyped.insert(at, facts.dtype);
            }
            columns.push(DescribedColumn {
                name: column.name.clone(),
                dtype,
                received: facts.received,
                stype: column.stype,
                stype_before_key: column.stype_before_key,
                nullable: facts.nullable,
                unique: facts.unique,
                declared_type: facts.declared_type,
                exact_number: facts.exact_number,
            });
        }
        tables.push(DescribedTable {
            name: table.name,
            source: table.source,
            columns,
            frame_key_form: table.frame_key_form,
            primary_key: table.primary_key,
            time_column: table.time_column,
            time_column_inferred: None,
            duplicate_keys: table.duplicate_keys,
        });
    }
    Ok(Read {
        description: SourceDescription { tables, links },
        unfacted,
        retyped,
        types,
    })
}

/// `description` as a host holds it: each column's data type by its wire
/// name, its meanings, and its facts, but for the columns of `unfacted`,
/// which carry none still, the data type of those of `retyped` the source's
/// ([`Read::retyped`]); each table's key, time column and repeated
/// keys; the links.
pub fn held(
    description: &SourceDescription,
    unfacted: &BTreeSet<ColumnAt>,
    retyped: &BTreeMap<ColumnAt, Dtype>,
) -> HeldSchema {
    HeldSchema {
        tables: description
            .tables
            .iter()
            .map(|table| HeldTable {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table
                    .columns
                    .iter()
                    .map(|column| held_column(table, column, unfacted, retyped))
                    .collect(),
                frame_key_form: table.frame_key_form,
                primary_key: table.primary_key.clone(),
                time_column: table.time_column.clone(),
                duplicate_keys: table.duplicate_keys.clone(),
            })
            .collect(),
        links: description.links.clone(),
    }
}

fn held_column(
    table: &DescribedTable,
    column: &DescribedColumn,
    unfacted: &BTreeSet<ColumnAt>,
    retyped: &BTreeMap<ColumnAt, Dtype>,
) -> HeldColumn {
    let at = (table.name.clone(), column.name.clone());
    let lacking = unfacted.contains(&at);
    HeldColumn {
        untyped: false,
        stype_declared: false,
        name: column.name.clone(),
        dtype: column.dtype.wire_name().to_owned(),
        stype: column.stype,
        stype_before_key: column.stype_before_key,
        source: (!lacking).then(|| {
            // The source's data type of a column the caller retyped, which
            // the facts keep as the source gave it.
            let mut facts = facts_of(&table.source, column);
            if let Some(dtype) = retyped.get(&at) {
                facts.dtype = *dtype;
            }
            facts
        }),
    }
}

/// What the source said of `column`, read from `relation`, as a host keeps
/// it.
pub fn facts_of(relation: &Relation, column: &DescribedColumn) -> SourceFacts {
    SourceFacts {
        relation: relation.clone(),
        dtype: column.dtype,
        received: column.received.clone(),
        nullable: column.nullable,
        unique: column.unique,
        declared_type: column.declared_type.clone(),
        exact_number: column.exact_number,
    }
}
