// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's edit semantics (decision E67, T1.24b): an edit to a schema
//! the core described, checked and made as the reference client's setters
//! check and make it on a table that stays in its source
//! (`Column.__setattr__`, `_TableShape.__setattr__`,
//! `SourceTable.__delitem__`, `Schema.__setattr__` and the relationships'
//! list), with the helpers they call (`_mark_as_key`, `_release_as_key`,
//! `_adopt_as_time_column`, `_stay_a_key_if_the_schema_joins_on_this`,
//! `_check_key`, `_adopt_key`, `_check_time_column`, `_settle`,
//! `_check_relationships`, `_require_joinable`, `_mark_key_columns`).
//!
//! A refusal is the reference's at the assignment: `INVALID_REQUEST`
//! naming the field, before anything of the schema changes. The host
//! passes every edit it is given, a value no contract type holds as
//! [`Given::Unreadable`], which is refused as the reference refuses a
//! value it cannot read.
//!
//! The schema edited is the one the host holds, as plain data
//! ([`crate::held`], T3.16): a column the source was never asked of is
//! edited by its data type, as the reference's setters edit any column.

use sdm_contract::ops::{
    ColumnRef, DescribedTable, Dtype, Given, SchemaEdit, SourceDescription, Stype,
};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar};

use crate::assemble::{
    HostTypes, Link, can_identify, checked, empty_key, invalid, key_cannot_identify,
    key_is_time_column, key_not_names, mark_keys, meaning_not_carried, name_each_other, no_column,
    no_table, once, read_exact_numbers, sorted_names, time_column_not_a_name, time_is_key,
    two_ways, with_given,
};

/// The meanings a caller may assign a column, by the text it gives
/// (`SEMANTIC_TYPES`): the language's, but `image` and `unsupported`.
const SEMANTIC_TYPES: [(&str, Stype); 7] = [
    ("ID", Stype::Id),
    ("categorical", Stype::Categorical),
    ("multicategorical", Stype::Multicategorical),
    ("numerical", Stype::Numerical),
    ("sequence", Stype::Sequence),
    ("text", Stype::Text),
    ("timestamp", Stype::Timestamp),
];

/// Whether a column of a data type can mean a semantic type, as the
/// language answers it (`Stype.supports_dtype`); the caller, which holds
/// the language, passes it.
pub type Supports<'a> = &'a dyn Fn(Stype, Dtype) -> bool;

/// An edit made: the schema with it, the column whose meaning it assigned
/// (`stype_declared`), and the warnings the reference gives as it makes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edited {
    pub description: SourceDescription,
    pub declared: Option<ColumnRef>,
    pub warnings: Vec<CoreWarning>,
}

/// `description` with `edit` made, as the reference's setters make it, or
/// the reference's refusal of it at the assignment (`INVALID_REQUEST`,
/// naming the field, and the table and column where it names them).
/// `description` with `edit` made, as the reference's setters make it, or
/// the reference's refusal of it at the assignment (`INVALID_REQUEST`,
/// naming the field, and the table and column where it names them); each
/// column's data type read by the name the host gave it (`types`, T3.16),
/// as the setters read a column's `dtype`: a meaning or key checked against
/// it, a name no data type has let through, a link's two columns joined by
/// their names' families, and a refusal quoting the name.
pub fn edit(
    mut description: SourceDescription,
    edit: &SchemaEdit,
    supports: Supports<'_>,
    types: &HostTypes,
) -> Result<Edited, CoreError> {
    let mut declared = None;
    let mut warnings = Vec::new();
    match edit {
        SchemaEdit::PrimaryKey { table, key } => {
            primary_key(&mut description, table, key, types)?;
        }
        SchemaEdit::TimeColumn { table, column } => {
            declared = time_column(&mut description, table, column, supports, types)?;
        }
        SchemaEdit::Stype {
            table,
            column,
            stype,
        } => {
            stype_of(&mut description, table, column, stype, supports, types)?;
            declared = Some(ColumnRef {
                table: table.clone(),
                column: column.clone(),
            });
        }
        SchemaEdit::LeaveOut { table, column } => leave_out(&mut description, table, column)?,
        SchemaEdit::Relationships { links } => {
            warnings = relationships(&mut description, links, types)?;
        }
    }
    Ok(Edited {
        description,
        declared,
        warnings,
    })
}

/// The table the schema names `name`, or the reference's refusal of a
/// table it does not hold (`schema[name]`).
fn table_at(description: &SourceDescription, name: &str, field: &str) -> Result<usize, CoreError> {
    description
        .tables
        .iter()
        .position(|t| t.name == name)
        .ok_or_else(|| {
            no_table(
                name,
                field,
                description.tables.iter().map(|t| t.name.as_str()),
            )
        })
}

/// The names of `table`'s columns, in its order.
fn columns_of(table: &DescribedTable) -> Vec<&str> {
    table.columns.iter().map(|c| c.name.as_str()).collect()
}

/// The column of `table` named `name`, or the reference's refusal of a
/// column the table does not hold (`_check_column`).
fn column_at(table: &DescribedTable, name: &str, field: &str) -> Result<usize, CoreError> {
    table
        .columns
        .iter()
        .position(|c| c.name == name)
        .ok_or_else(|| no_column(&table.name, name, field, &columns_of(table)))
}

fn named(refused: CoreError, table: &str, column: &str) -> CoreError {
    refused
        .with_field("table", Detail::text(table))
        .with_field("column", Detail::text(column))
}

/// `schema[table].primary_key = key` (`_TableShape.__setattr__`): `None`
/// clears it, the table's repeated keys counted as none, and the schema
/// gives the columns it no longer joins on back what they meant; a key is
/// held to `_check_key` (not empty; each column the table's, able to
/// identify a row, and not its time column), its columns marked as
/// identifiers, its repeats uncounted (a table in its source is counted by
/// `Schema.from_source` only), and the schema's marks derived again.
fn primary_key(
    description: &mut SourceDescription,
    table: &str,
    key: &Given<Option<Vec<String>>>,
    types: &HostTypes,
) -> Result<(), CoreError> {
    let at = table_at(description, table, "primary_key")?;
    let names = match key {
        Given::Unreadable | Given::Spelled(_) => {
            return Err(key_not_names(table, "primary_key", key.spelled()));
        }
        Given::Value(None) => {
            let shape = &mut description.tables[at];
            shape.duplicate_keys = Some(Scalar::UInt64(0));
            shape.primary_key = Vec::new();
            mark_keys(description);
            return Ok(());
        }
        Given::Value(Some(names)) => names,
    };
    let shape = &description.tables[at];
    if names.is_empty() {
        return Err(empty_key(table, "primary_key"));
    }
    for name in names {
        let column = &shape.columns[column_at(shape, name, "primary_key")?];
        // A column named by no data type the old client knows, which its
        // checks let through (`supports_dtype`, T3.16).
        if !types.is_unknown(table, name) && !can_identify(column.dtype) {
            return Err(key_cannot_identify(
                table,
                name,
                types.name(table, name, column.dtype),
                "primary_key",
            ));
        }
        if shape.time_column.as_deref() == Some(name.as_str()) {
            return Err(key_is_time_column(table, name, "primary_key"));
        }
    }
    let shape = &mut description.tables[at];
    shape.primary_key.clone_from(names);
    shape.duplicate_keys = None;
    mark_keys(description);
    Ok(())
}

/// `schema[table].time_column = column` (`_TableShape.__setattr__`):
/// `None` clears it and changes nothing else; a column is held to
/// `_check_time_column` (the table's, not part of its key) and assigned
/// `timestamp` as `schema[table][column].stype = 'timestamp'` assigns it
/// (its meaning one its values can carry, then the caller's choice,
/// adopted as the time column where the table has none, the schema's marks
/// derived again), then named. The column assigned its meaning, if any.
fn time_column(
    description: &mut SourceDescription,
    table: &str,
    column: &Given<Option<String>>,
    supports: Supports<'_>,
    types: &HostTypes,
) -> Result<Option<ColumnRef>, CoreError> {
    let at = table_at(description, table, "time_column")?;
    let name = match column {
        Given::Unreadable | Given::Spelled(_) => {
            return Err(time_column_not_a_name(
                table,
                "time_column",
                column.spelled(),
                &columns_of(&description.tables[at]),
            ));
        }
        Given::Value(None) => {
            description.tables[at].time_column = None;
            return Ok(None);
        }
        Given::Value(Some(name)) => name,
    };
    let shape = &description.tables[at];
    column_at(shape, name, "time_column")?;
    if shape.primary_key.contains(name) {
        return Err(time_is_key(table, name, "time_column"));
    }
    assign(
        description,
        at,
        name,
        Stype::Timestamp,
        supports,
        "time_column",
        types,
    )?;
    description.tables[at].time_column = Some(name.clone());
    Ok(Some(ColumnRef {
        table: table.to_owned(),
        column: name.clone(),
    }))
}

/// `schema[table][column].stype = stype` (`Column.__setattr__`): the text
/// one of the meanings a caller may assign (`_check_stype`), then
/// [`assign`]ed.
fn stype_of(
    description: &mut SourceDescription,
    table: &str,
    column: &str,
    stype: &Given<String>,
    supports: Supports<'_>,
    types: &HostTypes,
) -> Result<(), CoreError> {
    let at = table_at(description, table, "stype")?;
    column_at(&description.tables[at], column, "stype")?;
    let meaning = stype.value().and_then(|text| {
        SEMANTIC_TYPES
            .iter()
            .find(|(name, _)| name == text)
            .map(|(_, stype)| *stype)
    });
    let Some(meaning) = meaning else {
        // Not a semantic type (`_check_stype`): the text given, or the
        // host's text of another value, and the types a caller may name.
        let refused = named(
            invalid("a column's meaning given is not a semantic type", "stype"),
            table,
            column,
        )
        .with_field(
            "stypes",
            sorted_names(SEMANTIC_TYPES.iter().map(|(n, _)| *n)),
        );
        return Err(match stype {
            Given::Value(text) => refused
                .with_field("reason", Detail::text("not_a_semantic_type"))
                .with_field("given_text", Detail::text(text)),
            Given::Spelled(text) => with_given(refused, Some(text))
                .with_field("reason", Detail::text("not_a_semantic_type_spelled")),
            Given::Unreadable => refused,
        });
    };
    assign(description, at, column, meaning, supports, "stype", types)
}

/// A meaning assigned to a column of the table at `at`
/// (`Column.__setattr__` after `_check_stype`): one its values can carry
/// (`_check_supports_dtype`), then held; a timestamp orders the table if
/// nothing does and the column is not part of its key
/// (`_adopt_as_time_column`); then `ID` forgets what the column meant
/// before it was a key, and any other meaning has the schema mark its
/// keys again (`_stay_a_key_if_the_schema_joins_on_this`).
fn assign(
    description: &mut SourceDescription,
    at: usize,
    column: &str,
    meaning: Stype,
    supports: Supports<'_>,
    field: &str,
    types: &HostTypes,
) -> Result<(), CoreError> {
    let shape = &mut description.tables[at];
    let index = column_at(shape, column, field)?;
    // A column named by no data type the old client knows, which its checks
    // let through (`supports_dtype`, T3.16); any other by its type.
    if !types.is_unknown(&shape.name, column) && !supports(meaning, shape.columns[index].dtype) {
        let said = SEMANTIC_TYPES
            .iter()
            .find(|(_, stype)| *stype == meaning)
            .map_or("unsupported", |(name, _)| *name);
        return Err(meaning_not_carried(
            &shape.name,
            column,
            types.name(&shape.name, column, shape.columns[index].dtype),
            said,
            field,
        ));
    }
    shape.columns[index].stype = meaning;
    if meaning == Stype::Timestamp
        && shape.time_column.is_none()
        && !shape.primary_key.iter().any(|k| k == column)
    {
        shape.time_column = Some(column.to_owned());
    }
    if meaning == Stype::Id {
        shape.columns[index].stype_before_key = None;
        return Ok(());
    }
    mark_keys(description);
    Ok(())
}

/// `del schema[table][column]` (`SourceTable.__delitem__`): the table's
/// column, and neither its key nor its time column; the links are left as
/// they are.
fn leave_out(
    description: &mut SourceDescription,
    table: &str,
    column: &Given<String>,
) -> Result<(), CoreError> {
    let at = table_at(description, table, "column")?;
    let Given::Value(name) = column else {
        let refused = invalid("a column named is not its table's", "column")
            .with_field("table", Detail::text(table));
        return Err(match column.spelled() {
            Some(_) => with_given(refused, column.spelled())
                .with_field("reason", Detail::text("column_not_a_name"))
                .with_field("columns", sorted_names(columns_of(&description.tables[at]))),
            None => refused,
        });
    };
    let shape = &mut description.tables[at];
    let index = column_at(shape, name, "column")?;
    if shape.primary_key.contains(name) {
        return Err(named(
            invalid("a column left out is its table's primary key", "column"),
            table,
            name,
        )
        .with_field("reason", Detail::text("left_out_key")));
    }
    if shape.time_column.as_deref() == Some(name.as_str()) {
        return Err(named(
            invalid("a column left out is its table's time column", "column"),
            table,
            name,
        )
        .with_field("reason", Detail::text("left_out_time_column")));
    }
    shape.columns.remove(index);
    Ok(())
}

/// `schema.relationships = links`, or the list edited
/// (`Schema.__setattr__`, `_Links`): each link one the schema can join
/// (`_check_relationships`), a column of exact numbers read as the links
/// read it ([`read_exact_numbers`]), the links each once in their order, the
/// warnings of two tables naming each other and of one table reaching
/// another through several columns, and the schema's marks derived again.
fn relationships(
    description: &mut SourceDescription,
    links: &[Given<Vec<ColumnRef>>],
    types: &HostTypes,
) -> Result<Vec<CoreWarning>, CoreError> {
    // Each column of exact numbers read as the links given read it, before
    // they are checked (T2.21); a refusal leaves the schema as it was.
    let readable: Vec<Link> = links
        .iter()
        .filter_map(|link| link.value().cloned())
        .collect();
    read_exact_numbers(&mut description.tables, &readable);
    let mut given = Vec::with_capacity(links.len());
    for link in links {
        let Given::Value(link) = link else {
            let refused = invalid(
                "a relationship given is not a relationship",
                "relationships",
            );
            return Err(match link.spelled() {
                Some(_) => with_given(refused, link.spelled())
                    .with_field("reason", Detail::text("not_a_relationship")),
                None => refused,
            });
        };
        checked(&description.tables, link, types)?;
        given.push(link.clone());
    }
    let settled = once(given);
    let mut warnings = name_each_other(&settled);
    warnings.extend(two_ways(&settled));
    description.links = settled;
    mark_keys(description);
    Ok(warnings)
}
