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
//! The warnings' texts are the reference client's, word for word, its
//! overrides included; their codes and fields say the same for a host that
//! words them itself.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedTable, Dtype, SourceDescription, Stype,
};
use sdm_contract::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Scalar};

use crate::infer::{InferredTable, Taken, TimeChoice};

/// A link, `[foreign key, referenced key]`, as the contract holds it.
pub type Link = Vec<ColumnRef>;

/// The warning that time columns were inferred, or refused, on a guess.
pub const TIME_COLUMNS_GUESSED: &str = "TIME_COLUMNS_GUESSED";
/// The warning that a link inferred or declared cannot join.
pub const LINK_CANNOT_JOIN: &str = "LINK_CANNOT_JOIN";
/// The warning that two tables name each other.
pub const TABLES_NAME_EACH_OTHER: &str = "TABLES_NAME_EACH_OTHER";
/// The warning that one table reaches another through several columns.
pub const TWO_WAYS_TO_A_TABLE: &str = "TWO_WAYS_TO_A_TABLE";

/// A schema assembled: its description and the warnings given, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembled {
    pub description: SourceDescription,
    pub warnings: Vec<CoreWarning>,
}

fn invalid(message: &str, field: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("field", Detail::text(field))
}

/// Whether a column of `dtype` can identify a row: the integer, float and
/// text types, as the language's `Stype.ID.supports_dtype` takes them;
/// the language also takes binary, which no description holds
/// (`infer::table` refuses it as `column_type`).
fn can_identify(dtype: Dtype) -> bool {
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
        if table.primary_key.is_some() {
            continue;
        }
        let time = table.time.column.clone();
        let Some(keyed) = table.columns.iter_mut().find(|c| c.name == column) else {
            continue;
        };
        if !can_identify(keyed.dtype) {
            return Err(invalid(
                "a column a foreign key refers to cannot identify a row, so it cannot be a primary key",
                "primary_key",
            ));
        }
        if time.as_deref() == Some(column.as_str()) {
            return Err(invalid(
                "a column a foreign key refers to is its table's time column, so it cannot also be its primary key",
                "primary_key",
            ));
        }
        if keyed.stype != Stype::Id {
            keyed.stype_before_key = Some(keyed.stype);
            keyed.stype = Stype::Id;
        }
        table.primary_key = Some(column);
    }
    Ok(())
}

/// The schema of `tables`, counted, with the `declared` foreign keys: the
/// links inferred by name beside them, those that cannot join dropped with
/// a warning, a lone edit stamp settled by the links, the links each once,
/// the columns joined on marked, and the reference's warnings of a guessed
/// time column and of two ways between tables. `duplicates` is each
/// table's count of repeated keys, as the contract's scalar.
pub fn assemble(
    tables: Vec<InferredTable>,
    declared: Vec<Link>,
    duplicates: Option<Scalar>,
) -> Assembled {
    let mut warnings = Vec::new();
    let mut choices: Vec<TimeChoice> = tables.iter().map(|t| t.time.clone()).collect();
    let mut described: Vec<DescribedTable> = tables
        .into_iter()
        .map(|table| DescribedTable {
            name: table.name,
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
                })
                .collect(),
            primary_key: table.primary_key.into_iter().collect(),
            time_column: table.time.column,
            edit_stamp_refused: None,
            duplicate_keys: duplicates.clone(),
        })
        .collect();

    // Links: the declared, then those inferred by name whose foreign key no
    // declared one covers, each kept only where it can join.
    let covered: BTreeSet<(&str, &str)> = declared
        .iter()
        .filter_map(|link| link.first())
        .map(|f| (f.table.as_str(), f.column.as_str()))
        .collect();
    let by_name: Vec<Link> = by_name(&described)
        .into_iter()
        .filter(|link| !covered.contains(&(link[0].table.as_str(), link[0].column.as_str())))
        .collect();
    let mut links = Vec::new();
    for link in declared.iter().cloned().chain(by_name) {
        match joins(&described, &link) {
            Ok(()) => links.push(link),
            // A link inferred or declared names columns the tables hold.
            Err(Unjoinable::Missing) => {}
            Err(why) => warnings.push(cannot_join(&described, &link, why)),
        }
    }

    // A lone edit stamp, settled by the links.
    let linked: BTreeSet<&str> = links.iter().map(|l| l[1].table.as_str()).collect();
    for (table, choice) in described.iter_mut().zip(choices.iter_mut()) {
        let settled = choice.for_links(linked.contains(table.name.as_str()));
        if settled != *choice {
            table.time_column.clone_from(&settled.column);
            if settled.column.is_none() {
                table.edit_stamp_refused.clone_from(&choice.fallback);
            }
            *choice = settled;
        }
    }

    let mut description = SourceDescription {
        tables: described,
        links: once(links),
    };
    mark_keys(&mut description);
    if let Some(warning) = guessed_time_columns(&description, &choices) {
        warnings.push(warning);
    }
    warnings.extend(name_each_other(&description.links));
    warnings.extend(two_ways(&description.links));
    Assembled {
        description,
        warnings,
    }
}

/// Links each column to another table's primary key of the same name
/// (`_infer_relationships`): only an exact name match against a key of one
/// column counts, and a name more than one table is keyed by links
/// nothing.
fn by_name(tables: &[DescribedTable]) -> Vec<Link> {
    let mut ambiguous = BTreeSet::new();
    let mut keys: BTreeMap<&str, &str> = BTreeMap::new();
    for table in tables {
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

/// What has to agree across a join: a number matches a number, a time a
/// time (`_join_family`).
fn family(dtype: Dtype) -> u8 {
    match dtype {
        Dtype::Int
        | Dtype::Byte
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Float
        | Dtype::Float32
        | Dtype::Float64 => 0,
        Dtype::Date | Dtype::Time => 1,
        Dtype::String => 2,
        Dtype::Bool => 3,
        Dtype::Binary => 4,
        Dtype::Timedelta => 5,
        Dtype::Floatlist => 6,
        Dtype::Intlist => 7,
        Dtype::Stringlist => 8,
        Dtype::Unsupported => 9,
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

/// Whether `link` can join (`_require_joinable`): it names columns the
/// schema holds, refers to its table's primary key of one column, and its
/// two columns hold values of one family.
pub fn joins(tables: &[DescribedTable], link: &[ColumnRef]) -> Result<(), Unjoinable> {
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
    if family(source.dtype) != family(target.dtype) {
        return Err(Unjoinable::Types);
    }
    Ok(())
}

/// The links, each once, in their order (`_each_link_once`).
fn once(links: Vec<Link>) -> Vec<Link> {
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

/// Marks the columns the schema joins on, each table's primary key and
/// each link's foreign key, as identifiers, keeping what each meant, and
/// gives a column the schema no longer joins on back the meaning it had
/// (`Schema._mark_key_columns`); a column given back `timestamp` becomes
/// its table's time column when the table has none and it is not the key
/// (`Column._adopt_as_time_column`).
pub fn mark_keys(description: &mut SourceDescription) {
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

/// Refuses a schema whose links cannot join (`Schema.check_consistent`'s
/// relationships), as `INVALID_REQUEST`.
pub fn check_links(description: &SourceDescription) -> Result<(), CoreError> {
    for link in &description.links {
        let message = match joins(&description.tables, link) {
            Ok(()) => continue,
            Err(Unjoinable::Missing) => {
                "a relationship names a table or column the schema does not hold"
            }
            Err(Unjoinable::NotTheKey) => {
                "a relationship does not join to its table's primary key of one column"
            }
            Err(Unjoinable::Types) => "a relationship joins two kinds of value that cannot match",
        };
        let mut refused = invalid(message, "relationships");
        // The link by its names, as the warnings name one, when it has
        // both ends.
        if let [foreign, referenced] = link.as_slice() {
            refused = refused
                .with_field("source_table", Detail::text(&foreign.table))
                .with_field("source_column", Detail::text(&foreign.column))
                .with_field("target_table", Detail::text(&referenced.table))
                .with_field("target_column", Detail::text(&referenced.column));
        }
        return Err(refused);
    }
    Ok(())
}

/// A name as the reference's messages write it, a Python string literal
/// (its `repr`), so a remedy the message suggests is valid Python: in
/// single quotes, or double when the name holds a single quote and no
/// double one; a backslash and the quote chosen escaped; tab, newline and
/// carriage return as `\t`, `\n`, `\r`; any other character Python does
/// not print as itself as `\xhh`, `\uhhhh` or `\Uhhhhhhhh` ([`printed`]).
pub fn quoted(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !printed(c) => {
                let code = u32::from(c);
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x1_0000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Whether Python's `str.isprintable` holds of `c`, which its `repr`
/// writes as itself: every character but the controls (Cc), the format
/// characters (Cf), the separators other than the space (Zs, Zl, Zp) and
/// the private use ones (Co), as Unicode 15 assigns them. A code point no
/// version assigns (Cn), which Python escapes too, is written as itself
/// here: telling one apart takes the whole character database.
fn printed(c: char) -> bool {
    const NOT_PRINTED: &[(u32, u32)] = &[
        (0x0000, 0x001f),
        (0x007f, 0x00a0),
        (0x00ad, 0x00ad),
        (0x0600, 0x0605),
        (0x061c, 0x061c),
        (0x06dd, 0x06dd),
        (0x070f, 0x070f),
        (0x0890, 0x0891),
        (0x08e2, 0x08e2),
        (0x1680, 0x1680),
        (0x180e, 0x180e),
        (0x2000, 0x200f),
        (0x2028, 0x202f),
        (0x205f, 0x2064),
        (0x2066, 0x206f),
        (0x3000, 0x3000),
        (0xe000, 0xf8ff),
        (0xfeff, 0xfeff),
        (0xfff9, 0xfffb),
        (0x110bd, 0x110bd),
        (0x110cd, 0x110cd),
        (0x13430, 0x1343f),
        (0x1bca0, 0x1bca3),
        (0x1d173, 0x1d17a),
        (0xe0001, 0xe0001),
        (0xe0020, 0xe007f),
        (0xf0000, 0xffffd),
        (0x100000, 0x10fffd),
    ];
    let code = u32::from(c);
    !NOT_PRINTED
        .iter()
        .any(|&(low, high)| (low..=high).contains(&code))
}

/// Names joined as the reference's `_and_list` joins them: each quoted,
/// the last after "and".
fn and_list<S: AsRef<str>>(names: &[S]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| quoted(n.as_ref())).collect();
    match quoted.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// A list of names as the reference's messages write one.
fn listed<S: AsRef<str>>(names: &[S]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| quoted(n.as_ref())).collect();
    format!("[{}]", quoted.join(", "))
}

/// The data type a message names, as the client's wire names it.
fn wire_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Bool => "bool",
        Dtype::Int | Dtype::Int64 => "int64",
        Dtype::Byte | Dtype::Int16 | Dtype::Int32 => "int32",
        Dtype::Float | Dtype::Float64 => "float64",
        Dtype::Float32 => "float32",
        Dtype::String => "string",
        Dtype::Date | Dtype::Time => "timestamp[us]",
        Dtype::Binary => "binary",
        Dtype::Timedelta => "timedelta",
        Dtype::Floatlist => "floatlist",
        Dtype::Intlist => "intlist",
        Dtype::Stringlist => "stringlist",
        Dtype::Unsupported => "unsupported",
    }
}

fn texts<S: AsRef<str>>(names: &[S]) -> Detail {
    Detail::List(
        names
            .iter()
            .map(|n| Scalar::Utf8(n.as_ref().to_owned()))
            .collect(),
    )
}

/// The warning of a link dropped because it cannot join
/// (`_warn_of_a_link_that_cannot_join`).
fn cannot_join(tables: &[DescribedTable], link: &[ColumnRef], why: Unjoinable) -> CoreWarning {
    let (foreign, referenced) = (&link[0], &link[1]);
    let key: &[String] = tables
        .iter()
        .find(|t| t.name == referenced.table)
        .map_or(&[], |t| t.primary_key.as_slice());
    let fields = BTreeMap::from([
        ("source_table".to_owned(), Detail::text(&foreign.table)),
        ("source_column".to_owned(), Detail::text(&foreign.column)),
        ("target_table".to_owned(), Detail::text(&referenced.table)),
        ("target_column".to_owned(), Detail::text(&referenced.column)),
    ]);
    let message = if why == Unjoinable::NotTheKey {
        // A declared foreign key may name a unique column that is not the
        // key, which is no fault of the types.
        let has = if key.is_empty() {
            format!("{} has no primary key", quoted(&referenced.table))
        } else {
            format!(
                "the primary key of {} is {}",
                quoted(&referenced.table),
                and_list(key)
            )
        };
        format!(
            "{}.{} refers to {}.{}, but {has}. A relationship joins to a primary key, so none \
             was made from it. Pass primary_keys={{{}: {}}} if {} identifies a row of {}.",
            foreign.table,
            foreign.column,
            referenced.table,
            referenced.column,
            quoted(&referenced.table),
            quoted(&referenced.column),
            quoted(&referenced.column),
            quoted(&referenced.table),
        )
    } else {
        let dtype = |at: &ColumnRef| column(tables, at).map(|c| wire_name(c.dtype)).unwrap_or("");
        format!(
            "{}.{} names the key of {} and holds {} where that key holds {}, which cannot \
             match, so no relationship was made from it. Store the two as one type if they are \
             the same identifier.",
            foreign.table,
            foreign.column,
            quoted(&referenced.table),
            quoted(dtype(foreign)),
            quoted(dtype(referenced)),
        )
    };
    CoreWarning {
        code: LINK_CANNOT_JOIN.to_owned(),
        message,
        fields: fields
            .into_iter()
            .chain([(
                "reason".to_owned(),
                Detail::text(if why == Unjoinable::NotTheKey {
                    "not_the_key"
                } else {
                    "types"
                }),
            )])
            .collect(),
    }
}

/// The warning of time columns taken, or refused, on a guess about what a
/// column holds (`_warn_how_time_columns_were_inferred`), for the paths
/// this build takes: a lone edit stamp taken, and, for a table left
/// without a time column, each timestamp passed over by its name. Only
/// where something could be windowed: two tables or more, or a link.
fn guessed_time_columns(
    description: &SourceDescription,
    choices: &[TimeChoice],
) -> Option<CoreWarning> {
    if description.tables.len() < 2 && description.links.is_empty() {
        return None;
    }
    let mut order: Vec<(&DescribedTable, &TimeChoice)> =
        description.tables.iter().zip(choices).collect();
    order.sort_by(|a, b| a.0.name.cmp(&b.0.name));
    let mut lines = Vec::new();
    let mut tables = Vec::new();
    for (table, choice) in order {
        let name = &table.name;
        match &table.time_column {
            Some(column) if Some(column) == choice.column.as_ref() => {
                if choice.how == Some(Taken::EditStamp) {
                    tables.push(name.clone());
                    lines.push(format!(
                        "{name}.{column} was taken as the time column although it looks like \
                         when a row was last changed, because the table has no other timestamp \
                         and no table links to it, so it reads as a log of changes. If it does \
                         not say when a row became true, clear it: time_columns={{{}: None}}.",
                        quoted(name)
                    ));
                }
            }
            Some(_) => {}
            None => {
                let before = lines.len();
                lines.extend(
                    choice
                        .rejected()
                        .filter(|(column, _)| table.columns.iter().any(|c| &c.name == column))
                        .map(|(column, why)| {
                            format!(
                                "{name}.{column} was not taken as the time column because {}. If \
                                 it does say when a row became true, name it: \
                                 time_columns={{{}: {}}}.",
                                why.why(),
                                quoted(name),
                                quoted(column)
                            )
                        }),
                );
                if lines.len() > before {
                    tables.push(name.clone());
                }
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    let listed: Vec<String> = lines.iter().map(|line| format!("- {line}")).collect();
    Some(CoreWarning {
        code: TIME_COLUMNS_GUESSED.to_owned(),
        message: format!(
            "Some time columns were inferred, or refused, on a guess about what a column \
             holds:\n{}",
            listed.join("\n")
        ),
        fields: BTreeMap::from([("tables".to_owned(), texts(&tables))]),
    })
}

/// The warnings of two tables naming each other, once a pair
/// (`_warn_of_a_pair_joined_both_ways`); a table pointing at itself is one
/// link, not two.
fn name_each_other(links: &[Link]) -> Vec<CoreWarning> {
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
        warnings.push(CoreWarning {
            code: TABLES_NAME_EACH_OTHER.to_owned(),
            message: format!(
                "{} and {} name each other, and a relationship is crossed in both directions, \
                 so this is two ways over one pair and an aggregation between them is refused \
                 for not having a single one. Declare whichever of the two a question is about \
                 and leave the other out.",
                quoted(source),
                quoted(target)
            ),
            fields: BTreeMap::from([
                ("source_table".to_owned(), Detail::text(source)),
                ("target_table".to_owned(), Detail::text(target)),
            ]),
        });
    }
    warnings
}

/// The warnings of one table reaching another through more than one column
/// (`_warn_two_ways_to_reach_one_table`), each pair once, in order.
fn two_ways(links: &[Link]) -> Vec<CoreWarning> {
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
        let (kept, rest) = (columns[0], &columns[1..]);
        warnings.push(CoreWarning {
            code: TWO_WAYS_TO_A_TABLE.to_owned(),
            message: format!(
                "{} reaches {} through {}, and an aggregation across the two tables has to be \
                 resolved to one of them, so a query that crosses this pair is refused until it \
                 is. They are all kept because they are all real, and which one a question means \
                 is not something this can decide. To keep {}, drop the others: \
                 schema.relationships = [link for link in schema.relationships if not \
                 (link.source_table == {} and link.target_table == {} and link.source_column in \
                 {})].",
                quoted(source),
                quoted(target),
                and_list(&columns),
                quoted(kept),
                quoted(source),
                quoted(target),
                listed(rest),
            ),
            fields: BTreeMap::from([
                ("source_table".to_owned(), Detail::text(source)),
                ("target_table".to_owned(), Detail::text(target)),
                ("columns".to_owned(), texts(&columns)),
            ]),
        });
    }
    warnings
}
