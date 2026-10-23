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
use sdm_contract::{ColumnType, CoreError};

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
/// a column's `dtype` (an unknown name is a type no meaning other than its
/// own supports), held nullable and not unique, of no exact numbers or
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

/// The data type a wire name names, as inference gives it
/// ([`Dtype::wire_name`]'s inverse, a date for `timestamp[us]`, an integer
/// for `int64` and a float for `float64`); [`Dtype::Unsupported`] for a name
/// no data type has.
pub fn wire_dtype(name: &str) -> Dtype {
    match name {
        "bool" => Dtype::Bool,
        "int64" => Dtype::Int,
        "int32" => Dtype::Int32,
        "float64" => Dtype::Float,
        "float32" => Dtype::Float32,
        "string" => Dtype::String,
        "timestamp[us]" => Dtype::Date,
        "binary" => Dtype::Binary,
        "timedelta" => Dtype::Timedelta,
        "floatlist" => Dtype::Floatlist,
        "intlist" => Dtype::Intlist,
        "stringlist" => Dtype::Stringlist,
        _ => Dtype::Unsupported,
    }
}

/// `held` as the schema the core works on: each table the host holds, in its
/// order, its source, key, time column, repeated keys and links the host's;
/// each column's meanings the host's, and what the source said of it: the
/// facts the host holds, else those `found` gives (`materialize`'s reading
/// of the source), else, with `by_type`, those its data type gives
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
    let links = held.links;
    let mut names = BTreeSet::new();
    for table in held.tables {
        if !names.insert(table.name.clone()) {
            return Err(invalid("the schema holds a table twice", "schema"));
        }
        let mut seen = BTreeSet::new();
        let mut columns = Vec::with_capacity(table.columns.len());
        for column in &table.columns {
            if !seen.insert(column.name.clone()) {
                return Err(invalid("a table holds a column twice", "schema"));
            }
            let at = (table.name.clone(), column.name.clone());
            let facts = match facts_held(&table, column) {
                Some(facts) => facts.clone(),
                None => match found.get(&at) {
                    Some(facts) => facts.clone(),
                    None if by_type => {
                        unfacted.insert(at);
                        by_data_type(&table.source, &column.dtype)
                    }
                    None => continue,
                },
            };
            columns.push(DescribedColumn {
                name: column.name.clone(),
                received: facts.received,
                dtype: facts.dtype,
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
            primary_key: table.primary_key,
            time_column: table.time_column,
            time_column_inferred: None,
            duplicate_keys: table.duplicate_keys,
        });
    }
    Ok(Read {
        description: SourceDescription { tables, links },
        unfacted,
    })
}

/// `description` as a host holds it: each column's data type by its wire
/// name, its meanings, and its facts, but for the columns of `unfacted`,
/// which carry none still; each table's key, time column and repeated
/// keys; the links.
pub fn held(description: &SourceDescription, unfacted: &BTreeSet<ColumnAt>) -> HeldSchema {
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
                    .map(|column| held_column(table, column, unfacted))
                    .collect(),
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
) -> HeldColumn {
    let lacking = unfacted.contains(&(table.name.clone(), column.name.clone()));
    HeldColumn {
        name: column.name.clone(),
        dtype: column.dtype.wire_name().to_owned(),
        stype: column.stype,
        stype_before_key: column.stype_before_key,
        source: (!lacking).then(|| facts_of(&table.source, column)),
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
