// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema's assembly from the tables inferred and the foreign keys the
//! catalog declares, by the reference client's rules (`Schema.from_source`
//! after the engine has read the source, decision E50, T1.21d): a keyless
//! table keyed by the one column declared foreign keys refer to; links
//! inferred by name, kept only where they can join, each once; a lone edit
//! stamp settled by the links; the columns the schema joins on marked as
//! identifiers; and the warnings the reference gives on the way. The same
//! marking and link checks hold the schema `materialize` is handed
//! (`Schema.check_consistent`), whatever the host did to it.
//!
//! The warnings are codes and fields, the names and values a warning
//! speaks of; the words, the reference client's with its Python remedies,
//! are each host's to give from them (decision E59, the design's
//! presentation rule). A warning's `message` is a sentence that names
//! nothing, for a host with no words of its own.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::frames::{FrameKeyForm, FrameRelationship};
use sdm_contract::ops::{
    self, ColumnRef, DescribedColumn, DescribedTable, Dtype, ExactNumber, SourceDescription, Stype,
};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, Scalar, Sentence};

use crate::infer::{InferredTable, PassedOver, Taken, TimeChoice};

/// A link, `[foreign key, referenced key]`, as the contract holds it.
pub type Link = Vec<ColumnRef>;

/// A schema assembled: its description and the warnings given, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembled {
    pub description: SourceDescription,
    pub warnings: Vec<CoreWarning>,
}

pub(crate) fn invalid(sentence: Sentence, message: &str, field: &str) -> CoreError {
    CoreError::new(sentence, message).with_field("field", Detail::text(field))
}

/// Whether a column of `dtype` can identify a row: the integer, float and
/// text types, as the language's `Stype.ID.supports_dtype` takes them;
/// the language also takes binary, which no description holds
/// (`infer::table` refuses it as `column_type`).
pub(crate) fn can_identify(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Int
            | Dtype::Byte
            | Dtype::Int16
            | Dtype::Int32
            | Dtype::Int64
            | Dtype::Float
            | Dtype::Float32
            | Dtype::Float64
            | Dtype::String
    )
}

/// The characters a table's name in the schema may not hold, as the
/// reference's `_TableShape._check_name` refuses them: a backtick, a
/// newline, a carriage return and a dot.
pub const NOT_IN_A_NAME: [char; 4] = ['`', '\n', '\r', '.'];

/// Refuses a name the schema may not give a table (`_check_name`): empty,
/// or holding a character of [`NOT_IN_A_NAME`], `INVALID_REQUEST` naming
/// the field. A rule on the schema's names, never on the source's, which
/// may hold any character its catalog does.
pub fn check_table_name(name: &str) -> Result<(), CoreError> {
    if name.is_empty() {
        return Err(invalid(
            Sentence::TableNameEmpty,
            "a table's name is empty",
            "tables",
        )
        .with_field("table", Detail::text(name)));
    }
    if name.contains(NOT_IN_A_NAME) {
        let held: Vec<String> = NOT_IN_A_NAME
            .iter()
            .filter(|c| name.contains(**c))
            .map(char::to_string)
            .collect();
        return Err(invalid(
            Sentence::TableNameCharacters,
            "a table's name holds a character a name may not: a backtick, a newline, a carriage return or a dot",
            "tables",
        )
        .with_field("table", Detail::text(name))
        .with_field("characters", texts(&held)));
    }
    Ok(())
}

/// Keys each keyless table by the column declared foreign keys refer to,
/// when they all refer to one (`_key_by_the_referenced_column`): a catalog
/// lets a foreign key refer only to a primary key or a unique column, so
/// that column identifies a row. Before the keys are counted, so this one
/// is too. The key's column is marked an identifier, what it meant kept.
/// A column that cannot identify a row, or that is the table's time
/// column, is refused (`INVALID_REQUEST`), as the reference's key setter
/// refuses it.
pub fn key_by_reference(tables: &mut [InferredTable], declared: &[Link]) -> Result<(), CoreError> {
    let mut referenced: Vec<(&str, BTreeSet<&str>)> = Vec::new();
    for link in declared {
        let [_, target] = link.as_slice() else {
            continue;
        };
        match referenced.iter_mut().find(|(t, _)| *t == target.table) {
            Some((_, columns)) => {
                columns.insert(&target.column);
            }
            None => referenced.push((&target.table, BTreeSet::from([target.column.as_str()]))),
        }
    }
    let referenced: Vec<(String, String)> = referenced
        .into_iter()
        .filter_map(
            |(table, columns)| match columns.into_iter().collect::<Vec<_>>()[..] {
                [one] => Some((table.to_owned(), one.to_owned())),
                _ => None,
            },
        )
        .collect();
    for (name, column) in referenced {
        let Some(table) = tables.iter_mut().find(|t| t.name == name) else {
            continue;
        };
        if !table.primary_key.is_empty() {
            continue;
        }
        let time = table.time_column.clone();
        let Some(keyed) = table.columns.iter_mut().find(|c| c.name == column) else {
            continue;
        };
        if !can_identify(keyed.dtype) {
            return Err(key_cannot_identify(
                &name,
                &column,
                keyed.dtype.wire_name(),
                "primary_key",
            ));
        }
        if time.as_deref() == Some(column.as_str()) {
            return Err(key_is_time_column(&name, &column, "primary_key"));
        }
        if keyed.stype != Stype::Id {
            keyed.stype_before_key = Some(keyed.stype);
            keyed.stype = Stype::Id;
        }
        table.primary_key = vec![column];
    }
    Ok(())
}

/// What the caller gave beside the tables: the tables whose time column
/// was named or cleared, which the choice's settling and its warning pass
/// over, and the relationships, in place of the declared and inferred ones.
#[derive(Debug, Clone, Copy, Default)]
pub struct Given<'a> {
    pub time_columns: Option<&'a BTreeSet<String>>,
    pub relationships: Option<&'a [Link]>,
}

impl Given<'_> {
    fn names_time_of(&self, table: &str) -> bool {
        self.time_columns.is_some_and(|given| given.contains(table))
    }
}

/// The schema of `tables`, counted, with the `declared` foreign keys: the
/// links inferred by name beside them, those that cannot join dropped with
/// a warning, or the relationships `given`, each checked
/// (`_checked_links`); a lone edit stamp settled by the links, the links
/// each once, the columns joined on marked, and the reference's warnings
/// of a guessed time column and of two ways between tables. `duplicates`
/// is each table's count of repeated keys, as the contract's scalar.
pub fn assemble(
    tables: Vec<InferredTable>,
    declared: Vec<Link>,
    given: Given<'_>,
    duplicates: Option<Scalar>,
) -> Result<Assembled, CoreError> {
    let choices = tables.iter().map(|t| t.time.clone()).collect();
    let described = tables
        .into_iter()
        .map(|t| described_table(t, duplicates.clone()))
        .collect();
    assemble_described(described, choices, declared, given, true, None)
}

/// Frame construction emits link validation notices before its final frame
/// inference notices, then time/two-way notices. Its Schema constructor's
/// internal pair notice was deliberately not presented by the native path.
pub fn assemble_frames(
    tables: Vec<InferredTable>,
    given: Given<'_>,
    duplicates: Option<Scalar>,
    key_forms: &BTreeMap<String, FrameKeyForm>,
    slots: Option<&[FrameRelationship]>,
) -> Result<(Assembled, Vec<CoreWarning>), CoreError> {
    let choices = tables.iter().map(|t| t.time.clone()).collect();
    let mut described: Vec<_> = tables
        .into_iter()
        .map(|t| described_table(t, duplicates.clone()))
        .collect();
    for table in &mut described {
        if !table.primary_key.is_empty() {
            table.frame_key_form = Some(
                key_forms
                    .get(&table.name)
                    .copied()
                    .unwrap_or(FrameKeyForm::Scalar),
            );
        }
    }
    let mut explicit = Vec::new();
    if let Some(slots) = slots {
        for slot in slots {
            match slot {
                FrameRelationship::Link { source, target } => {
                    let link = vec![source.clone(), target.clone()];
                    checked(&described, &link, &HostTypes::default())?;
                    explicit.push(link);
                }
                FrameRelationship::Invalid {
                    native_type,
                    native_repr,
                } => {
                    return Err(CoreError::new(
                        Sentence::FrameRelationshipInvalid,
                        "a native relationship slot is not a Relationship",
                    )
                    .with_field("native_type", Detail::text(native_type))
                    .with_field("native_repr", Detail::text(native_repr)));
                }
            }
        }
    }
    let given = Given {
        relationships: slots.map(|_| explicit.as_slice()),
        ..given
    };
    let mut links = Vec::new();
    let assembled = assemble_described(described, choices, vec![], given, true, Some(&mut links))?;
    Ok((assembled, links))
}

/// Assemble logical source facts already converted by a semantic model.
/// Caller overrides use the same rules as ordinary sources, and model links
/// are the only inferred links: column names do not imply additional ones.
pub fn assemble_converted(
    mut description: SourceDescription,
    choices: Vec<TimeChoice>,
    primary_keys: &BTreeMap<String, ops::Given<Option<Vec<String>>>>,
    time_columns: &BTreeMap<String, ops::Given<Option<String>>>,
    relationships: Option<&[Link]>,
) -> Result<Assembled, CoreError> {
    apply_described_keys_and_times(&mut description.tables, primary_keys, time_columns, false)?;
    let timed = time_columns.keys().cloned().collect();
    assemble_described(
        description.tables,
        choices,
        description.links,
        Given {
            time_columns: Some(&timed),
            relationships,
        },
        false,
        None,
    )
}

fn assemble_described(
    mut described: Vec<DescribedTable>,
    mut choices: Vec<TimeChoice>,
    declared: Vec<Link>,
    given: Given<'_>,
    infer_by_name: bool,
    mut frame_links: Option<&mut Vec<CoreWarning>>,
) -> Result<Assembled, CoreError> {
    let mut warnings = Vec::new();
    let links = if let Some(given) = given.relationships {
        // The caller's, each held to the columns and the join as given,
        // each column of exact numbers read as the links given read it.
        read_exact_numbers(&mut described, given);
        for link in given {
            checked(&described, link, &HostTypes::default())?;
        }
        given.to_vec()
    } else {
        // The declared, then those inferred by name whose foreign key no
        // declared one covers, each kept only where it can join.
        let covered: BTreeSet<(&str, &str)> = declared
            .iter()
            .filter_map(|link| link.first())
            .map(|f| (f.table.as_str(), f.column.as_str()))
            .collect();
        let by_name: Vec<Link> = if infer_by_name {
            by_name(&described)
        } else {
            Vec::new()
        }
        .into_iter()
        .filter(|link| !covered.contains(&(link[0].table.as_str(), link[0].column.as_str())))
        .collect();
        let mut links = Vec::new();
        for link in declared.iter().cloned().chain(by_name) {
            match joins(&described, &link, &HostTypes::default()) {
                Ok(()) => links.push(link),
                // A link inferred or declared names columns the tables hold.
                Err(Unjoinable::Missing) => {}
                Err(why) => warnings.push(cannot_join(&described, &link, why)),
            }
        }
        links
    };

    let frames = frame_links.is_some();
    if let Some(link_notices) = &mut frame_links {
        link_notices.append(&mut warnings);
    }
    // A lone edit stamp, settled by the links, unless the caller named the
    // table's time column.
    let inferred: Vec<Option<String>> = choices.iter().map(|c| c.column.clone()).collect();
    let linked: BTreeSet<&str> = links.iter().map(|l| l[1].table.as_str()).collect();
    for (table, choice) in described.iter_mut().zip(choices.iter_mut()) {
        if given.names_time_of(&table.name) {
            continue;
        }
        let settled = choice.for_links(linked.contains(table.name.as_str()));
        if settled != *choice {
            table.time_column.clone_from(&settled.column);
            *choice = settled;
        }
    }
    // What inference took, where the time column is now another or none.
    for (table, inferred) in described.iter_mut().zip(inferred) {
        if inferred.is_some() && inferred != table.time_column {
            table.time_column_inferred = inferred;
        }
    }

    let mut description = SourceDescription {
        tables: described,
        links: once(links),
    };
    mark_keys(&mut description);
    warnings.extend(guessed_time_columns(&description, &choices, given));
    if !frames {
        warnings.extend(name_each_other(&description.links));
    }
    warnings.extend(two_ways(&description.links));
    Ok(Assembled {
        description,
        warnings,
    })
}

/// A table's inferred source facts as the common schema description.
pub fn described_table(table: InferredTable, duplicates: Option<Scalar>) -> DescribedTable {
    DescribedTable {
        name: table.name,
        source: table.source,
        columns: table
            .columns
            .into_iter()
            .map(|c| DescribedColumn {
                name: c.name,
                received: c.received,
                dtype: c.dtype,
                stype: c.stype,
                stype_before_key: c.stype_before_key,
                nullable: c.nullable,
                unique: c.unique,
                declared_type: None,
                exact_number: c.exact_number,
            })
            .collect(),
        frame_key_form: None,
        primary_key: table.primary_key,
        time_column: table.time_column,
        time_column_inferred: None,
        duplicate_keys: duplicates,
    }
}

/// The refusal of a relationship given (`_checked_links`): it names a
/// table and column the schema holds at each end, and it can join
/// (`_require_joinable`), each refused as `INVALID_REQUEST` naming the
/// link.
pub(crate) fn checked(
    tables: &[DescribedTable],
    link: &[ColumnRef],
    types: &HostTypes,
) -> Result<(), CoreError> {
    let field = "relationships";
    let refused = match joins(tables, link, types) {
        Ok(()) => return Ok(()),
        // The first end the schema does not hold, in the reference's order
        // (`_require_column` of the source, then of the target), named as
        // `_require_column` names it.
        Err(Unjoinable::Missing) => link
            .iter()
            .find_map(|end| {
                let Some(table) = tables.iter().find(|t| t.name == end.table) else {
                    return Some(no_table(
                        &end.table,
                        field,
                        tables.iter().map(|t| t.name.as_str()),
                    ));
                };
                (!table.columns.iter().any(|c| c.name == end.column)).then(|| {
                    let columns: Vec<&str> =
                        table.columns.iter().map(|c| c.name.as_str()).collect();
                    no_column(&end.table, &end.column, field, &columns)
                })
            })
            .unwrap_or_else(|| {
                invalid(
                    Sentence::LinkUnheld,
                    "a relationship given names a table or column the schema does not hold",
                    field,
                )
            }),
        // What the target's key is (`_require_joinable`): several columns,
        // or one or none other than the column named.
        Err(Unjoinable::NotTheKey) => {
            let key: &[String] = link
                .get(1)
                .and_then(|end| tables.iter().find(|t| t.name == end.table))
                .map_or(&[], |t| t.primary_key.as_slice());
            let (sentence, reason) = match key.len() {
                0 => (Sentence::LinkNoKey, "no_key"),
                1 => (Sentence::LinkNotTheKey, "not_the_key"),
                _ => (Sentence::LinkCompositeKey, "composite_key"),
            };
            invalid(
                sentence,
                "a relationship given does not join to its table's primary key of one column",
                field,
            )
            .with_field("reason", Detail::text(reason))
            .with_field(
                "key",
                Detail::List(key.iter().cloned().map(Scalar::Utf8).collect()),
            )
        }
        Err(Unjoinable::Types) => {
            let dtype = |at: Option<&ColumnRef>| {
                at.and_then(|at| column(tables, at).map(|c| named(types, at, c)))
                    .unwrap_or("unsupported")
            };
            invalid(
                Sentence::LinkTypes,
                "a relationship given joins two kinds of value that cannot match",
                field,
            )
            .with_field("reason", Detail::text("types"))
            .with_field("source_dtype", Detail::text(dtype(link.first())))
            .with_field("target_dtype", Detail::text(dtype(link.get(1))))
        }
    };
    Err(named_link(refused, link))
}

/// `refused` naming `link` by its four names, when it has both ends.
fn named_link(refused: CoreError, link: &[ColumnRef]) -> CoreError {
    let [foreign, referenced] = link else {
        return refused;
    };
    refused
        .with_field("source_table", Detail::text(&foreign.table))
        .with_field("source_column", Detail::text(&foreign.column))
        .with_field("target_table", Detail::text(&referenced.table))
        .with_field("target_column", Detail::text(&referenced.column))
}

/// The keys and time columns the caller named, applied to the tables
/// inferred, as the reference's `_apply_keys_and_times` applies them,
/// between the reading and the key count: each key's table and columns
/// must be the schema's, each column able to identify a row and not the
/// table's time column; each time column's table and column the schema's,
/// not part of its key, and of a type that can hold a time. A key may name
/// several columns (a composite key, held column by column), but not one
/// twice (`key_column_twice`, P1B-73). Naming a key marks its columns
/// identifiers, what each meant kept, and gives the table's other columns
/// back what they meant; naming a time column makes it a timestamp; a time
/// column cleared (`None`) leaves the table none, its columns meaning what
/// they meant, as the reference's setter takes `None` (decision E61). A key
/// or time column the host could not write as names (`unreadable`) is
/// refused by name, after its table is found, as `edit_schema` refuses it;
/// a key of `null` names no column and is refused as an empty key, as the
/// reference's `_as_key_tuple` reads `None` (decision E101). Refusals are
/// `INVALID_REQUEST`, naming the field.
pub fn apply_keys_and_times(
    tables: &mut [InferredTable],
    primary_keys: &BTreeMap<String, ops::Given<Option<Vec<String>>>>,
    time_columns: &BTreeMap<String, ops::Given<Option<String>>>,
) -> Result<(), CoreError> {
    apply_keys_and_times_mode(tables, primary_keys, time_columns, false)
}

/// Frame construction follows native setter key sequences. P1B-73's
/// duplicate-column refusal belongs only to ordinary source description.
pub fn apply_frame_keys_and_times(
    tables: &mut [InferredTable],
    primary_keys: &BTreeMap<String, ops::Given<Option<Vec<String>>>>,
    time_columns: &BTreeMap<String, ops::Given<Option<String>>>,
) -> Result<(), CoreError> {
    apply_keys_and_times_mode(tables, primary_keys, time_columns, true)
}

fn apply_keys_and_times_mode(
    tables: &mut [InferredTable],
    primary_keys: &BTreeMap<String, ops::Given<Option<Vec<String>>>>,
    time_columns: &BTreeMap<String, ops::Given<Option<String>>>,
    frames: bool,
) -> Result<(), CoreError> {
    if primary_keys.is_empty() && time_columns.is_empty() {
        return Ok(());
    }
    let mut described: Vec<_> = tables
        .iter()
        .cloned()
        .map(|t| described_table(t, None))
        .collect();
    apply_described_keys_and_times(&mut described, primary_keys, time_columns, frames)?;
    for (table, updated) in tables.iter_mut().zip(described) {
        table.primary_key = updated.primary_key;
        table.time_column = updated.time_column;
        for (column, updated) in table.columns.iter_mut().zip(updated.columns) {
            column.stype = updated.stype;
            column.stype_before_key = updated.stype_before_key;
        }
    }
    Ok(())
}

fn apply_described_keys_and_times(
    tables: &mut [DescribedTable],
    primary_keys: &BTreeMap<String, ops::Given<Option<Vec<String>>>>,
    time_columns: &BTreeMap<String, ops::Given<Option<String>>>,
    frames: bool,
) -> Result<(), CoreError> {
    let find =
        |tables: &mut [DescribedTable], name: &str, field: &str| -> Result<usize, CoreError> {
            tables
                .iter()
                .position(|t| t.name == name)
                .ok_or_else(|| no_table(name, field, tables.iter().map(|t| t.name.as_str())))
        };
    let columns_of = |table: &DescribedTable| -> Vec<String> {
        table.columns.iter().map(|c| c.name.clone()).collect()
    };
    for (name, key) in primary_keys {
        let at = find(tables, name, "primary_keys")?;
        let key = match key {
            ops::Given::Unreadable | ops::Given::Spelled(_) => {
                return Err(key_not_names(name, "primary_keys", key.spelled()));
            }
            ops::Given::Value(None) => return Err(empty_key(name, "primary_keys")),
            ops::Given::Value(Some(key)) => key,
        };
        let table = &mut tables[at];
        for column in key {
            if !table.columns.iter().any(|c| c.name == *column) {
                return Err(no_column(name, column, "primary_keys", &columns_of(table)));
            }
        }
        // A column named twice is refused by name (P1B-73, the owner's;
        // `describe_source` only).
        let mut named = BTreeSet::new();
        if !frames && let Some(twice) = key.iter().find(|column| !named.insert(column.as_str())) {
            return Err(invalid(
                Sentence::KeyColumnTwice,
                "a primary key names a column more than once",
                "primary_keys",
            )
            .with_field("table", Detail::text(name))
            .with_field("column", Detail::text(twice)));
        }
        if key.is_empty() {
            return Err(empty_key(name, "primary_keys"));
        }
        // Each column of the key, in order, as the reference's `_check_key`
        // holds each: able to identify a row, and not the time column.
        let time = table.time_column.clone();
        for column in key {
            let keyed = table
                .columns
                .iter()
                .find(|c| c.name == *column)
                .expect("the column was found");
            if !can_identify(keyed.dtype) {
                return Err(key_cannot_identify(
                    name,
                    column,
                    keyed.dtype.wire_name(),
                    "primary_keys",
                ));
            }
            if time.as_deref() == Some(column.as_str()) {
                return Err(key_is_time_column(name, column, "primary_keys"));
            }
        }
        for keyed in table.columns.iter_mut().filter(|c| key.contains(&c.name)) {
            if keyed.stype != Stype::Id {
                keyed.stype_before_key = Some(keyed.stype);
                keyed.stype = Stype::Id;
            }
        }
        table.primary_key.clone_from(key);
        // The table's other columns, no longer its key, mean again what
        // they meant; one that meant a timestamp orders the table if
        // nothing does.
        for other in &mut table.columns {
            if key.contains(&other.name) {
                continue;
            }
            let Some(before) = other.stype_before_key.take() else {
                continue;
            };
            other.stype = before;
            if before == Stype::Timestamp && table.time_column.is_none() {
                table.time_column = Some(other.name.clone());
            }
        }
    }
    for (name, column) in time_columns {
        let at = find(tables, name, "time_columns")?;
        let table = &mut tables[at];
        let given = column;
        let column = match column {
            ops::Given::Unreadable | ops::Given::Spelled(_) => {
                return Err(time_column_not_a_name(
                    name,
                    "time_columns",
                    given.spelled(),
                    &columns_of(table),
                ));
            }
            ops::Given::Value(None) => {
                table.time_column = None;
                continue;
            }
            ops::Given::Value(Some(column)) => column,
        };
        if !table.columns.iter().any(|c| c.name == *column) {
            return Err(no_column(name, column, "time_columns", &columns_of(table)));
        }
        if table.primary_key.contains(column) {
            return Err(time_is_key(name, column, "time_columns"));
        }
        let timed = table
            .columns
            .iter_mut()
            .find(|c| &c.name == column)
            .expect("the column was found");
        if !matches!(timed.dtype, Dtype::Date | Dtype::Time | Dtype::String) {
            return Err(meaning_not_carried(
                name,
                column,
                timed.dtype.wire_name(),
                "timestamp",
                "time_columns",
            ));
        }
        timed.stype = Stype::Timestamp;
        table.time_column = Some(column.clone());
    }
    Ok(())
}

/// The names given, in Python's sorted order (by code point, as Rust
/// orders text), as a refusal lists what a schema or table has.
pub(crate) fn sorted_names<S: AsRef<str>>(names: impl IntoIterator<Item = S>) -> Detail {
    let mut names: Vec<String> = names.into_iter().map(|n| n.as_ref().to_owned()).collect();
    names.sort();
    Detail::List(names.into_iter().map(Scalar::Utf8).collect())
}

/// A table the schema does not hold (`_require_table`, `schema[name]`),
/// naming the tables it has (`no_table`).
pub(crate) fn no_table<'a>(
    name: &str,
    field: &str,
    tables: impl IntoIterator<Item = &'a str>,
) -> CoreError {
    invalid(
        Sentence::NoTable,
        "a table named is not the schema's",
        field,
    )
    .with_field("table", Detail::text(name))
    .with_field("reason", Detail::text("no_table"))
    .with_field("tables", sorted_names(tables))
}

/// A key the host could not write as a column or a sequence of them
/// (`unreadable`), refused alike by `describe_source` and `edit_schema`,
/// `field` each operation's own (decision E101), naming the value as the
/// host spelled it, where it did (`given`, KO2-86).
pub(crate) fn key_not_names(table: &str, field: &str, spelled: Option<&str>) -> CoreError {
    let refused = |sentence| {
        invalid(
            sentence,
            "a primary key names a column or a sequence of them",
            field,
        )
        .with_field("table", Detail::text(table))
    };
    match spelled {
        Some(_) => with_given(refused(Sentence::KeyNotNamesGiven), spelled),
        None => refused(Sentence::KeyNotNames),
    }
}

/// `refused`, naming the host's text of a value the caller gave (`given`),
/// where the host gave one (KO2-86); `refused` says the sentence that
/// names it.
pub(crate) fn with_given(refused: CoreError, spelled: Option<&str>) -> CoreError {
    match spelled {
        Some(text) => refused
            .with_field("given", Detail::text(text))
            .with_field("variant", Detail::text("given")),
        None => refused,
    }
}

/// A time column the host could not write as a name (`unreadable`),
/// refused alike by `describe_source` and `edit_schema` (decision E101) as
/// a column the table does not have: where the host spelled the value, it
/// is named (`given`, KO2-86), with the table's columns.
pub(crate) fn time_column_not_a_name<S: AsRef<str>>(
    table: &str,
    field: &str,
    spelled: Option<&str>,
    columns: &[S],
) -> CoreError {
    let refused = |sentence| {
        invalid(sentence, "a column named is not its table's", field)
            .with_field("table", Detail::text(table))
    };
    match spelled {
        Some(_) => with_given(refused(Sentence::TimeColumnNotANameGiven), spelled)
            .with_field("columns", sorted_names(columns)),
        None => refused(Sentence::TimeColumnNotAName),
    }
}

/// A key naming no column, refused alike by `describe_source` (an empty
/// list, or `null`) and `edit_schema` (an empty list; its `null` clears).
pub(crate) fn empty_key(table: &str, field: &str) -> CoreError {
    invalid(
        Sentence::EmptyKey,
        "a table was given an empty primary key",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("reason", Detail::text("empty_key"))
}

/// A column the table does not have (`_check_column`), naming the table's
/// columns.
pub(crate) fn no_column<S: AsRef<str>>(
    table: &str,
    column: &str,
    field: &str,
    columns: &[S],
) -> CoreError {
    invalid(
        Sentence::NoColumn,
        "a column named is not its table's",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text("no_column"))
    .with_field("columns", sorted_names(columns))
}

/// A column named as a key whose values cannot identify a row
/// (`_check_key`), naming its data type on the wire.
pub(crate) fn key_cannot_identify(
    table: &str,
    column: &str,
    dtype: &str,
    field: &str,
) -> CoreError {
    invalid(
        Sentence::KeyCannotIdentify,
        "a column named as a primary key cannot identify a row",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text("key_cannot_identify"))
    .with_field("dtype", Detail::text(dtype))
}

/// A column named as a key that is its table's time column (`_check_key`).
pub(crate) fn key_is_time_column(table: &str, column: &str, field: &str) -> CoreError {
    invalid(
        Sentence::KeyIsTimeColumn,
        "a column named as a primary key is its table's time column",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text("key_is_time_column"))
}

/// A column named as a time column that is part of its table's key
/// (`_check_time_column`).
pub(crate) fn time_is_key(table: &str, column: &str, field: &str) -> CoreError {
    invalid(
        Sentence::TimeIsKey,
        "a column named as a time column is its table's primary key",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text("time_is_key"))
}

/// A meaning a column's values cannot carry (`_check_supports_dtype`),
/// naming the column's data type on the wire and the meaning.
pub(crate) fn meaning_not_carried(
    table: &str,
    column: &str,
    dtype: &str,
    meaning: &str,
    field: &str,
) -> CoreError {
    invalid(
        Sentence::MeaningNotCarried,
        "a column's meaning is not one its values can carry",
        field,
    )
    .with_field("table", Detail::text(table))
    .with_field("column", Detail::text(column))
    .with_field("reason", Detail::text("meaning_not_carried"))
    .with_field("dtype", Detail::text(dtype))
    .with_field("stype", Detail::text(meaning))
}

/// Links each column to another table's primary key of the same name
/// (`_infer_relationships`): only an exact name match against a key of one
/// column counts, and a name more than one table is keyed by links
/// nothing.
fn by_name(tables: &[DescribedTable]) -> Vec<Link> {
    let mut ambiguous = BTreeSet::new();
    let mut keys: BTreeMap<&str, &str> = BTreeMap::new();
    for table in tables {
        if table.frame_key_form == Some(FrameKeyForm::Sequence) {
            continue;
        }
        let [key] = table.primary_key.as_slice() else {
            continue;
        };
        if keys.contains_key(key.as_str()) {
            ambiguous.insert(key.as_str());
        }
        keys.insert(key, &table.name);
    }
    let mut links = Vec::new();
    for table in tables {
        for column in &table.columns {
            let Some(target) = keys.get(column.name.as_str()) else {
                continue;
            };
            if *target == table.name || ambiguous.contains(column.name.as_str()) {
                continue;
            }
            links.push(vec![
                ColumnRef {
                    table: table.name.clone(),
                    column: column.name.clone(),
                },
                ColumnRef {
                    table: (*target).to_owned(),
                    column: column.name.clone(),
                },
            ]);
        }
    }
    links
}

/// Why a link cannot join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unjoinable {
    /// A table or column it names is not the schema's.
    Missing,
    /// It does not refer to its table's primary key of one column.
    NotTheKey,
    /// Its two columns hold values that cannot match.
    Types,
}

/// The data type names a host gave its columns (T3.16), which the
/// reference's checks of the caller's schema read: each by table and
/// column, but for a column of exact numbers, whose type its role decides
/// (T2.21); and those of them no data type has, which those checks let
/// through. A column with no name here is named by its type's wire name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostTypes {
    pub names: BTreeMap<(String, String), String>,
    pub unknown: BTreeSet<(String, String)>,
}

impl HostTypes {
    /// The name of the data type of `table`'s `column`, of type `dtype`.
    pub fn name<'a>(&'a self, table: &str, column: &str, dtype: Dtype) -> &'a str {
        self.names
            .get(&(table.to_owned(), column.to_owned()))
            .map_or(dtype.wire_name(), String::as_str)
    }

    /// Whether the host named `table`'s `column` by no data type, which the
    /// reference's checks let through (`supports_dtype`).
    pub fn is_unknown(&self, table: &str, column: &str) -> bool {
        self.unknown
            .contains(&(table.to_owned(), column.to_owned()))
    }
}

/// What has to agree across a join, as the client's `_join_family` reads a
/// data type's name: any name starting `int` or `float` a number, any
/// starting `timestamp` a time, any other itself.
pub fn join_family(name: &str) -> &str {
    if name.starts_with("int") || name.starts_with("float") {
        "number"
    } else if name.starts_with("timestamp") {
        "timestamp"
    } else {
        name
    }
}

fn column<'a>(tables: &'a [DescribedTable], at: &ColumnRef) -> Option<&'a DescribedColumn> {
    tables
        .iter()
        .find(|t| t.name == at.table)?
        .columns
        .iter()
        .find(|c| c.name == at.column)
}

fn named<'a>(types: &'a HostTypes, at: &ColumnRef, column: &DescribedColumn) -> &'a str {
    types.name(&at.table, &at.column, column.dtype)
}

/// Whether `link` can join (`_require_joinable`): it names columns the
/// schema holds, refers to its table's primary key of one column, and its
/// two columns hold values of one family, each column's type read by the
/// name `types` gives it (the host's, T3.16).
pub fn joins(
    tables: &[DescribedTable],
    link: &[ColumnRef],
    types: &HostTypes,
) -> Result<(), Unjoinable> {
    let [foreign, referenced] = link else {
        return Err(Unjoinable::Missing);
    };
    let (Some(source), Some(target)) = (column(tables, foreign), column(tables, referenced)) else {
        return Err(Unjoinable::Missing);
    };
    let key: &[String] = tables
        .iter()
        .find(|t| t.name == referenced.table)
        .map_or(&[], |t| t.primary_key.as_slice());
    if key != [referenced.column.clone()] {
        return Err(Unjoinable::NotTheKey);
    }
    if join_family(named(types, foreign, source)) != join_family(named(types, referenced, target)) {
        return Err(Unjoinable::Types);
    }
    Ok(())
}

/// The links, each once, in their order (`_each_link_once`).
pub(crate) fn once(links: Vec<Link>) -> Vec<Link> {
    let mut seen = BTreeSet::new();
    links
        .into_iter()
        .filter(|link| {
            seen.insert(
                link.iter()
                    .map(|c| (c.table.clone(), c.column.clone()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Each column of numbers handed over as their exact text
/// ([`DescribedColumn::exact_number`]) in the type its role in `tables`
/// keyed as they stand and joined by `links` gives it, both ends of a link
/// decided together (learnings lesson 19, T2.21): its text (`string`,
/// received as `utf8`) where its catalog declares it a key, as the
/// reference reads it, or where the schema keys it (a primary key's
/// column) or joins on it and every column it joins to is read as text
/// too; a double (`float64`) otherwise, as a feature is read. So a key the
/// caller names or edits is as exact as one the catalog declares, a column
/// joined to a number of another type is read as the reference reads it,
/// and a key no longer keyed or joined is a double again.
pub fn read_exact_numbers(tables: &mut [DescribedTable], links: &[Link]) {
    let exact =
        |tables: &[DescribedTable], at: &ColumnRef| column(tables, at).and_then(|c| c.exact_number);
    // The columns read as text, to start: every exact column the catalog
    // declares a key, and every other the schema keys or joins on.
    let mut text: BTreeSet<(String, String)> = BTreeSet::new();
    for table in tables.iter() {
        for c in &table.columns {
            let keyed = table.primary_key.contains(&c.name)
                || links
                    .iter()
                    .flatten()
                    .any(|end| end.table == table.name && end.column == c.name);
            match c.exact_number {
                Some(ExactNumber::DeclaredKey) => {}
                Some(ExactNumber::Other) if keyed => {}
                _ => continue,
            }
            text.insert((table.name.clone(), c.name.clone()));
        }
    }
    // Then, until nothing changes, a link with an end not read as text
    // takes its other end out of the text too, unless the catalog declares
    // that end a key.
    loop {
        let mut changed = false;
        for link in links {
            let [foreign, referenced] = link.as_slice() else {
                continue;
            };
            let held = |at: &ColumnRef| text.contains(&(at.table.clone(), at.column.clone()));
            if held(foreign) && held(referenced) {
                continue;
            }
            for end in [foreign, referenced] {
                if exact(tables, end) == Some(ExactNumber::Other)
                    && text.remove(&(end.table.clone(), end.column.clone()))
                {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for table in tables.iter_mut() {
        for c in &mut table.columns {
            if c.exact_number.is_none() {
                continue;
            }
            if text.contains(&(table.name.clone(), c.name.clone())) {
                c.dtype = Dtype::String;
                c.received = ColumnType::Utf8;
            } else {
                c.dtype = Dtype::Float64;
                c.received = ColumnType::Float64;
            }
        }
    }
}

/// Marks the columns the schema joins on, each table's primary key and
/// each link's foreign key, as identifiers, keeping what each meant, and
/// gives a column the schema no longer joins on back the meaning it had
/// (`Schema._mark_key_columns`); a column given back `timestamp` becomes
/// its table's time column when the table has none and it is not the key
/// (`Column._adopt_as_time_column`); first, each column of exact numbers
/// in the type its role gives it ([`read_exact_numbers`]).
pub fn mark_keys(description: &mut SourceDescription) {
    read_exact_numbers(&mut description.tables, &description.links);
    let mut joined: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for table in &description.tables {
        joined.insert(
            table.name.clone(),
            table
                .primary_key
                .iter()
                .filter(|k| table.columns.iter().any(|c| &c.name == *k))
                .cloned()
                .collect(),
        );
    }
    for link in &description.links {
        let Some(foreign) = link.first() else {
            continue;
        };
        let held = description
            .tables
            .iter()
            .any(|t| t.name == foreign.table && t.columns.iter().any(|c| c.name == foreign.column));
        if held && let Some(columns) = joined.get_mut(&foreign.table) {
            columns.insert(foreign.column.clone());
        }
    }
    for table in &mut description.tables {
        let keys = joined.remove(&table.name).unwrap_or_default();
        for column in &mut table.columns {
            if keys.contains(&column.name) {
                if column.stype != Stype::Id {
                    column.stype_before_key = Some(column.stype);
                    column.stype = Stype::Id;
                }
                continue;
            }
            let Some(before) = column.stype_before_key.take() else {
                continue;
            };
            column.stype = before;
            if before == Stype::Timestamp
                && table.time_column.is_none()
                && !table.primary_key.contains(&column.name)
            {
                table.time_column = Some(column.name.clone());
            }
        }
    }
}

/// Refuses a schema whose links cannot join, as the reference's
/// `Schema.check_consistent` refuses the caller's schema before its engine
/// reads anything (`_check_relationships`, `schema.py:971`): each link names
/// a table and column the schema holds at each end, joins its target's
/// primary key of one column, and its two columns' types, read by the name
/// `types` gives them (the host's, T3.16), are of one family
/// (`_require_joinable`), each refusal naming the link as the reference's
/// does (T3.16 review round 3).
pub fn check_links(description: &SourceDescription, types: &HostTypes) -> Result<(), CoreError> {
    for link in &description.links {
        checked(&description.tables, link, types)?;
    }
    Ok(())
}

/// Refuses the edges the reference's engine refuses when it validates the
/// graph it built over the tables it read (`Graph.validate`'s
/// `_edge_error`, run as `KumoRelational` is built, `relational.py:418`),
/// each in its order: a foreign key that is its own table's primary key
/// (T1.24b: found by the E49 differential), then a foreign key and the key
/// it refers to holding values of two kinds, a number and a string, by the
/// types the source holds (`description`'s, read after the tables are
/// kept). The rest of `_edge_error` (an edge whose column or target key is
/// gone) is refused earlier, by [`check_links`] and the graph's build.
pub fn check_edges(description: &SourceDescription) -> Result<(), CoreError> {
    for link in &description.links {
        let [foreign, referenced] = link.as_slice() else {
            continue;
        };
        let keyed = description
            .tables
            .iter()
            .find(|t| t.name == foreign.table)
            .is_some_and(|t| t.primary_key == [foreign.column.clone()]);
        if keyed {
            return Err(named_link(
                invalid(
                    Sentence::PrimaryKeyAsForeignKey,
                    "a relationship's foreign key is its table's primary key",
                    "relationships",
                ),
                link,
            ));
        }
        let (Some(source), Some(target)) = (
            column(&description.tables, foreign),
            column(&description.tables, referenced),
        ) else {
            continue;
        };
        // The engine's kinds: a number (any integer or float) or a string;
        // the join family of each type's wire name tells them apart.
        let kind = |dtype: Dtype| join_family(dtype.wire_name()).to_owned();
        if kind(source.dtype) != kind(target.dtype) {
            let name = |dtype: Dtype| {
                serde_json::to_value(dtype)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default()
            };
            return Err(named_link(
                invalid(
                    Sentence::EdgeKeyTypes,
                    "a relationship's two keys hold values of two kinds in the source",
                    "relationships",
                ),
                link,
            )
            .with_field("reason", Detail::text("edge_key_types"))
            .with_field("source_dtype", Detail::text(name(source.dtype)))
            .with_field("target_dtype", Detail::text(name(target.dtype))));
        }
    }
    Ok(())
}

fn texts<S: AsRef<str>>(names: &[S]) -> Detail {
    Detail::List(
        names
            .iter()
            .map(|n| Scalar::Utf8(n.as_ref().to_owned()))
            .collect(),
    )
}

/// A warning of `code`, saying `message`, a sentence that names nothing,
/// with `fields`, whose names and values the warning speaks of: the host
/// words it from these (decision E59).
fn warning(sentence: Sentence, message: &str, fields: Vec<(&str, Detail)>) -> CoreWarning {
    CoreWarning::new(sentence, message).with_fields(fields)
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(n as u64))
}

/// The warnings of the columns a table leaves out (`_describe_source_table`),
/// one a column, in table order, naming the table, the column and its
/// type (`dtype`, as the contract names it).
pub fn left_out(table: &InferredTable) -> Vec<CoreWarning> {
    table
        .left_out
        .iter()
        .map(|column| {
            warning(
                Sentence::ColumnsLeftOut,
                "a column whose values have no scalar form was left out of its table",
                vec![
                    ("table", Detail::text(&table.name)),
                    ("column", Detail::text(&column.name)),
                    ("dtype", Detail::text(dtype_name(column.dtype))),
                ],
            )
        })
        .collect()
}

/// The warnings of the times of day a table's preview holds, which the
/// reference's engine reads as no time and drops (its `to_datetime`): one
/// a column, naming the table, the column, how many values were dropped
/// (`unread`) of how many rows (`rows`), and the first three distinct
/// (`examples`, the contract's `time64` scalars).
pub fn unread_times(table: &InferredTable) -> Vec<CoreWarning> {
    table
        .unread_times
        .iter()
        .map(|unread| {
            warning(
                Sentence::TimeValuesUnread,
                "a time of day's values were read as no time and dropped",
                vec![
                    ("table", Detail::text(&table.name)),
                    ("column", Detail::text(&unread.column)),
                    ("unread", count(unread.unread)),
                    ("rows", count(unread.rows)),
                    ("examples", Detail::List(unread.examples.clone())),
                ],
            )
        })
        .collect()
}

/// The warnings of the text columns whose dates the client's rule reads
/// day first (`date-rules.md`, "The core's text formats"), as the
/// reference warns of a format that puts the day first: one a column,
/// naming the table, the column and the format, as `strftime` writes it.
pub fn day_first(table: &InferredTable) -> Vec<CoreWarning> {
    table
        .day_first
        .iter()
        .map(|(column, format)| {
            warning(
                Sentence::DatesReadDayFirst,
                "text was read as dates by a format that puts the day first",
                vec![
                    ("table", Detail::text(&table.name)),
                    ("column", Detail::text(column)),
                    ("format", Detail::text(format)),
                ],
            )
        })
        .collect()
}

/// The warning that the columns of a table that tied for its key left it
/// without one (`infer_primary_key`'s), naming the table and the tied
/// columns (`columns`), in table order; none where no tie was met.
pub fn key_tied(table: &InferredTable) -> Vec<CoreWarning> {
    if table.tied_keys.is_empty() {
        return Vec::new();
    }
    vec![warning(
        Sentence::KeyCandidatesTied,
        "several columns tied for a table's primary key, so it was given none",
        vec![
            ("table", Detail::text(&table.name)),
            ("columns", texts(&table.tied_keys)),
        ],
    )]
}

/// The warnings given once every table is read, in the reference's order:
/// its engine types the columns of every table before it infers any
/// table's key, so each table's tied candidates are said after every
/// table's unread times and day-first dates, in table order, and the
/// columns each table leaves out after them, as the schema describes the
/// tables (T2.12 fix 1).
pub fn after_reading(tables: &[InferredTable]) -> Vec<CoreWarning> {
    tables
        .iter()
        .flat_map(key_tied)
        .chain(tables.iter().flat_map(left_out))
        .collect()
}

/// A data type as the contract names it (`Dtype`'s serialized name).
fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Bool => "bool",
        Dtype::Int => "int",
        Dtype::Byte => "byte",
        Dtype::Int16 => "int16",
        Dtype::Int32 => "int32",
        Dtype::Int64 => "int64",
        Dtype::Float => "float",
        Dtype::Float32 => "float32",
        Dtype::Float64 => "float64",
        Dtype::String => "string",
        Dtype::Date => "date",
        Dtype::Time => "time",
        Dtype::Binary => "binary",
        Dtype::Timedelta => "timedelta",
        Dtype::Floatlist => "floatlist",
        Dtype::Intlist => "intlist",
        Dtype::Stringlist => "stringlist",
        Dtype::Unsupported => "unsupported",
    }
}

/// The warning of a link dropped because it cannot join
/// (`_warn_of_a_link_that_cannot_join`): the link by its four names, the
/// reason, and either the target table's key (`key`, empty when it has
/// none) or the two columns' wire types (`source_dtype`, `target_dtype`).
/// `variant` says which of the reference's sentences: `not_the_key`,
/// `no_key` or `types`.
fn cannot_join(tables: &[DescribedTable], link: &[ColumnRef], why: Unjoinable) -> CoreWarning {
    let (foreign, referenced) = (&link[0], &link[1]);
    let key: &[String] = tables
        .iter()
        .find(|t| t.name == referenced.table)
        .map_or(&[], |t| t.primary_key.as_slice());
    let mut fields = vec![
        ("source_table", Detail::text(&foreign.table)),
        ("source_column", Detail::text(&foreign.column)),
        ("target_table", Detail::text(&referenced.table)),
        ("target_column", Detail::text(&referenced.column)),
    ];
    let sentence = if why == Unjoinable::NotTheKey {
        // A declared foreign key may name a unique column that is not the
        // key, which is no fault of the types.
        fields.push(("reason", Detail::text("not_the_key")));
        let (sentence, variant) = if key.is_empty() {
            (Sentence::LinkCannotJoinNoKey, "no_key")
        } else {
            (Sentence::LinkCannotJoinNotTheKey, "not_the_key")
        };
        fields.push(("variant", Detail::text(variant)));
        fields.push(("key", texts(key)));
        sentence
    } else {
        let dtype = |at: &ColumnRef| {
            column(tables, at)
                .map(|c| Dtype::wire_name(c.dtype))
                .unwrap_or("")
        };
        fields.push(("reason", Detail::text("types")));
        fields.push(("variant", Detail::text("types")));
        fields.push(("source_dtype", Detail::text(dtype(foreign))));
        fields.push(("target_dtype", Detail::text(dtype(referenced))));
        Sentence::LinkCannotJoinTypes
    };
    warning(
        sentence,
        "a relationship declared or inferred cannot join, so none was made from it",
        fields,
    )
}

/// The warnings of time columns taken, or refused, on a guess about what a
/// column holds (`_warn_how_time_columns_were_inferred`), one a line of the
/// reference's warning, tables in name order: a column taken as the
/// earliest of several (`taken_earliest`, with `among` and the first of
/// the others, `other`) or as a lone edit stamp (`taken_edit_stamp`); and,
/// for a table left without a time column, each timestamp passed over,
/// with its reason (`refused_<reason>`, and for its values `missing`,
/// `rows` and the one value every row holds, `value`, a `refused_constant`
/// said differently when no row holds one, `refused_empty`, or when some
/// hold none, `refused_constant_where_held`). Each names its `table` and
/// `column`. Only where something could be windowed: two tables or more,
/// or a link; never for a table whose time column the caller named.
/// The lines saying `column`, taken, was among the columns the rows read
/// would pass over, kept because the whole table's answer failed
/// (`taken_unchecked`), each with the reason the rows read give.
fn unchecked(
    choice: &TimeChoice,
    column: &str,
    at: &dyn Fn(&str) -> Vec<(&'static str, Detail)>,
) -> Vec<CoreWarning> {
    let message = "a time column was taken, or refused, on a guess about what a column holds";
    choice
        .unconfirmed
        .iter()
        .filter(|(name, _)| name == column)
        .map(|(_, why)| {
            let mut fields = at(column);
            let (variant, sentence) = match why {
                PassedOver::Constant {
                    value,
                    missing,
                    rows,
                } => {
                    fields.push(("missing", count(*missing)));
                    fields.push(("rows", count(*rows)));
                    match value {
                        None => (
                            "taken_unchecked_empty",
                            Sentence::TimeColumnsTakenUncheckedEmpty,
                        ),
                        Some(value) => {
                            fields.push(("value", Detail::Scalar(value.clone())));
                            if *missing > 0 {
                                (
                                    "taken_unchecked_constant_where_held",
                                    Sentence::TimeColumnsTakenUncheckedConstantWhereHeld,
                                )
                            } else {
                                (
                                    "taken_unchecked_constant",
                                    Sentence::TimeColumnsTakenUncheckedConstant,
                                )
                            }
                        }
                    }
                }
                PassedOver::Sparse { missing, rows } => {
                    fields.push(("missing", count(*missing)));
                    fields.push(("rows", count(*rows)));
                    (
                        "taken_unchecked_sparse",
                        Sentence::TimeColumnsTakenUncheckedSparse,
                    )
                }
                // Only a reason of the values is unconfirmed.
                _ => (
                    "taken_unchecked_empty",
                    Sentence::TimeColumnsTakenUncheckedEmpty,
                ),
            };
            fields.push(("variant", Detail::text(variant)));
            warning(sentence, message, fields)
        })
        .collect()
}

fn guessed_time_columns(
    description: &SourceDescription,
    choices: &[TimeChoice],
    given: Given<'_>,
) -> Vec<CoreWarning> {
    if description.tables.len() < 2 && description.links.is_empty() {
        return vec![];
    }
    let mut order: Vec<(&DescribedTable, &TimeChoice)> =
        description.tables.iter().zip(choices).collect();
    order.sort_by(|a, b| a.0.name.cmp(&b.0.name));
    let message = "a time column was taken, or refused, on a guess about what a column holds";
    let mut said = Vec::new();
    for (table, choice) in order {
        let name = &table.name;
        if given.names_time_of(name) {
            continue;
        }
        let at = |column: &str| {
            vec![
                ("table", Detail::text(name)),
                ("column", Detail::text(column)),
            ]
        };
        match &table.time_column {
            Some(column) if Some(column) == choice.column.as_ref() => match choice.how {
                Some(Taken::Earliest) => {
                    let other = choice.among.iter().find(|c| *c != column);
                    let mut fields = at(column);
                    fields.push(("variant", Detail::text("taken_earliest")));
                    fields.push(("among", texts(&choice.among)));
                    fields.push(("other", Detail::text(other.map_or("", String::as_str))));
                    said.push(warning(Sentence::TimeColumnsTakenEarliest, message, fields));
                    said.extend(unchecked(choice, column, &at));
                }
                Some(Taken::EditStamp) => {
                    let mut fields = at(column);
                    fields.push(("variant", Detail::text("taken_edit_stamp")));
                    said.push(warning(
                        Sentence::TimeColumnsTakenEditStamp,
                        message,
                        fields,
                    ));
                    said.extend(unchecked(choice, column, &at));
                }
                _ => said.extend(unchecked(choice, column, &at)),
            },
            Some(_) => {}
            None => {
                for (column, why) in choice.rejected() {
                    if !table.columns.iter().any(|c| &c.name == column) {
                        continue;
                    }
                    let mut fields = at(column);
                    fields.push(("reason", Detail::text(why.reason())));
                    let (variant, sentence) = match why {
                        PassedOver::Constant {
                            value,
                            missing,
                            rows,
                        } => {
                            fields.push(("missing", count(*missing)));
                            fields.push(("rows", count(*rows)));
                            match value {
                                None => ("refused_empty", Sentence::TimeColumnsRefusedEmpty),
                                Some(value) => {
                                    fields.push(("value", Detail::Scalar(value.clone())));
                                    if *missing > 0 {
                                        (
                                            "refused_constant_where_held",
                                            Sentence::TimeColumnsRefusedConstantWhereHeld,
                                        )
                                    } else {
                                        ("refused_constant", Sentence::TimeColumnsRefusedConstant)
                                    }
                                }
                            }
                        }
                        PassedOver::Sparse { missing, rows } => {
                            fields.push(("missing", count(*missing)));
                            fields.push(("rows", count(*rows)));
                            ("refused_sparse", Sentence::TimeColumnsRefusedSparse)
                        }
                        PassedOver::LoadStamp => {
                            ("refused_load_stamp", Sentence::TimeColumnsRefusedLoadStamp)
                        }
                        PassedOver::EditStamp => {
                            ("refused_edit_stamp", Sentence::TimeColumnsRefusedEditStamp)
                        }
                        PassedOver::BirthDate => {
                            ("refused_birth_date", Sentence::TimeColumnsRefusedBirthDate)
                        }
                    };
                    fields.push(("variant", Detail::text(variant)));
                    said.push(warning(sentence, message, fields));
                }
            }
        }
    }
    said
}

/// The warnings of two tables naming each other, once a pair
/// (`_warn_of_a_pair_joined_both_ways`); a table pointing at itself is one
/// link, not two.
pub(crate) fn name_each_other(links: &[Link]) -> Vec<CoreWarning> {
    let pairs: BTreeSet<(&str, &str)> = links
        .iter()
        .map(|l| (l[0].table.as_str(), l[1].table.as_str()))
        .collect();
    let mut said = BTreeSet::new();
    let mut warnings = Vec::new();
    for &(source, target) in &pairs {
        if source == target
            || !pairs.contains(&(target, source))
            || said.contains(&(target, source))
        {
            continue;
        }
        said.insert((source, target));
        warnings.push(warning(
            Sentence::TablesNameEachOther,
            "two tables name each other, so a relationship is crossed in both directions",
            vec![
                ("source_table", Detail::text(source)),
                ("target_table", Detail::text(target)),
            ],
        ));
    }
    warnings
}

/// The warnings of one table reaching another through more than one column
/// (`_warn_two_ways_to_reach_one_table`), each pair once, in order: the
/// columns in name order, the one a remedy keeps (`kept`, the first) and
/// the others (`others`).
pub(crate) fn two_ways(links: &[Link]) -> Vec<CoreWarning> {
    let mut ways: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
    for link in links {
        let columns = ways
            .entry((link[0].table.as_str(), link[1].table.as_str()))
            .or_default();
        if !columns.contains(&link[0].column.as_str()) {
            columns.push(&link[0].column);
        }
    }
    let mut warnings = Vec::new();
    for ((source, target), mut columns) in ways {
        if columns.len() < 2 {
            continue;
        }
        columns.sort_unstable();
        warnings.push(warning(
            Sentence::TwoWaysToATable,
            "one table reaches another through more than one column",
            vec![
                ("source_table", Detail::text(source)),
                ("target_table", Detail::text(target)),
                ("columns", texts(&columns)),
                ("kept", Detail::text(columns[0])),
                ("others", texts(&columns[1..])),
            ],
        ));
    }
    warnings
}
