// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One table's inference, by the reference's rules: each column's data type
//! from the Arrow type its preview returned, its semantic type from its name
//! and the preview's values, the primary key from the catalog or, where it
//! declares none, from names and values, and the time column among the
//! timestamps, those named as a load, an edit or a birth passed over
//! (T1.21d), and from T1.23 those whose values say nothing, then the one
//! left, the creation stamp or the earliest; a list column left out. From
//! T1.32e a preview that may not hold every row has the timestamps' values
//! read over the whole table ([`Previewed::whole_table_columns`],
//! [`whole_table`], [`finish`]), as the reference asks its source.
//! Every path these rules take that this build does not port is refused
//! by name ([`crate::not_yet`]).

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::data::{EXACT_NUMBER_MARKED, EXACT_NUMBER_MARKER, Relation};
use sdm_contract::ops::{Dtype, ExactNumber, SourceFacts, Stype};
use sdm_contract::{ColumnType, CoreError, Detail, Scalar, Sentence, Unit};

use crate::names;
use crate::not_yet;
pub use crate::values::Floats;
use crate::values::{self, Value};
use sdm_contract::catalog::{CatalogColumn, CatalogKey, KeyKind};

/// The rows a preview reads: a preview that returns this many may have left
/// rows unread, which the reference then reads the whole table for.
pub const PREVIEW_ROWS: u64 = 1_000;

/// How inference reads text as times, by the core's list of formats
/// (`date-rules.md`, "The core's text formats"): each value's instant in
/// nanoseconds since the epoch, in UTC, or `None`. `sdm-engine::time::text`
/// holds it (T1.25, T1.25b), above this crate; `sdm-core` hands it in.
pub trait TextTimes {
    /// The values read, to UTC where `utc`, or why not ([`TextUnread`]).
    fn read(&self, values: &[Option<&str>], utc: bool) -> Result<Vec<Option<i128>>, TextUnread>;

    /// How many of `values` read as times ([`TextShare`]): the score's.
    fn share(&self, values: &[Option<&str>]) -> Result<TextShare, TextUnread>;

    /// The format of `values`' day-first dates when the values decide that
    /// the day comes first, which inference warns of.
    fn day_first_format(&self, values: &[Option<&str>]) -> Option<String>;
}

/// Why text does not read as times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextUnread {
    /// A value outside the reader's formats.
    Undecided,
    /// An order of day and month the values leave open (`text_time_order`).
    Order,
    /// Offsets mixed where the reading is not to UTC: the reference's
    /// reading raises, and its inference refuses the source.
    MixedZones,
}

/// One column, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
    /// What the column meant before a key named after inference made it an
    /// identifier (a keyless table keyed by the column its declared foreign
    /// keys refer to), as the reference's key setter records it.
    pub stype_before_key: Option<Stype>,
    /// The column's type as admitted from the preview.
    pub received: ColumnType,
    /// Whether the catalog lets the column hold no value.
    pub nullable: bool,
    /// Whether the catalog declares a primary or unique key of this one
    /// column.
    pub unique: bool,
    /// For a column of numbers handed over as their exact text, whether
    /// the catalog declares it a key ([`exact_number`]); `None` otherwise.
    pub exact_number: Option<ExactNumber>,
}

/// What `field`, of a table whose catalog declares `constraints`, is as a
/// column of numbers the host provider handed over as their exact text
/// (marked [`EXACT_NUMBER_MARKER`], whatever type it was read in since): a
/// key its catalog declares, the one column of a primary, unique or
/// foreign key, as the reference's Postgres table reads a column's role
/// (`relational_keys`), or any other; `None` for an unmarked column.
pub fn exact_number(field: &Field, constraints: &[CatalogKey]) -> Option<ExactNumber> {
    if field
        .metadata()
        .get(EXACT_NUMBER_MARKER)
        .map(String::as_str)
        != Some(EXACT_NUMBER_MARKED)
    {
        return None;
    }
    let declared = constraints.iter().any(|key| {
        matches!(
            key.kind,
            KeyKind::Primary | KeyKind::Unique | KeyKind::Foreign
        ) && key.columns.len() == 1
            && key.columns[0] == *field.name()
    });
    Some(if declared {
        ExactNumber::DeclaredKey
    } else {
        ExactNumber::Other
    })
}

/// Why inference passed a timestamp over for the time column, as the
/// reference's `RejectedTimeColumn` holds it: its values (no row holds
/// one, or every row that holds one holds the same; more than half the
/// rows hold none), or its name (a stamp of a row's load, of its last edit,
/// or a date of birth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassedOver {
    /// No row holds a value (`value` `None`), or the table has more than
    /// one row and every row that holds one holds `value`; `missing` of
    /// its `rows` rows hold none.
    Constant {
        value: Option<Scalar>,
        missing: usize,
        rows: usize,
    },
    /// `missing` of its `rows` rows, more than half, hold no value.
    Sparse {
        missing: usize,
        rows: usize,
    },
    LoadStamp,
    EditStamp,
    BirthDate,
}

impl PassedOver {
    /// The reason's name, as the reference's `RejectedTimeColumn.reason`
    /// gives it.
    pub fn reason(&self) -> &'static str {
        match self {
            PassedOver::Constant { .. } => "constant",
            PassedOver::Sparse { .. } => "sparse",
            PassedOver::LoadStamp => "load_stamp",
            PassedOver::EditStamp => "edit_stamp",
            PassedOver::BirthDate => "birth_date",
        }
    }
}

/// How the time column was taken (`TimeColumnChoice.how`): the one
/// timestamp left, the one creation stamp among several, the one whose
/// values start earliest, or a lone stamp of the last edit in a table no
/// other links to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taken {
    Only,
    Created,
    Earliest,
    EditStamp,
}

/// The time column inference took and how, and the timestamps it passed
/// over, as the reference's `TimeColumnChoice` holds them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimeChoice {
    pub column: Option<String>,
    /// The view dimensions that truncate or cast the chosen raw column.
    pub instead_of: Vec<String>,
    pub how: Option<Taken>,
    /// The columns it was chosen from, itself included.
    pub among: Vec<String>,
    /// Every timestamp passed over, in table order, the lone edit stamp
    /// taken as the last resort included.
    pub refused: Vec<(String, PassedOver)>,
    /// The lone edit stamp left, when no other timestamp was.
    pub fallback: Option<String>,
    /// Where the whole table was asked about and its answer failed, each
    /// timestamp the preview's rows would pass over for its values, kept
    /// (behind every column they do not) and named, its counts the rows
    /// read (`TimeColumnChoice.unconfirmed`).
    pub unconfirmed: Vec<(String, PassedOver)>,
}

impl TimeChoice {
    /// This choice for a table another table links to, or none does
    /// (`TimeColumnChoice.for_links`): a lone edit stamp is refused in a
    /// table something links to, an entity edited in place, and taken in
    /// one nothing links to, a log of changes.
    pub fn for_links(&self, linked: bool) -> TimeChoice {
        let Some(fallback) = &self.fallback else {
            return self.clone();
        };
        TimeChoice {
            column: (!linked).then(|| fallback.clone()),
            instead_of: if !linked && self.column.as_ref() == Some(fallback) {
                self.instead_of.clone()
            } else {
                vec![]
            },
            how: (!linked).then_some(Taken::EditStamp),
            among: if linked {
                vec![]
            } else {
                vec![fallback.clone()]
            },
            refused: self.refused.clone(),
            fallback: self.fallback.clone(),
            unconfirmed: self.unconfirmed.clone(),
        }
    }

    /// The timestamps passed over, each with its reason, in table order:
    /// the refused but the one taken.
    pub fn rejected(&self) -> impl Iterator<Item = &(String, PassedOver)> {
        self.refused
            .iter()
            .filter(move |(name, _)| Some(name) != self.column.as_ref())
    }
}

/// A column whose type has no scalar form on the wire, which the schema
/// leaves out and names (`_describe_source_table`): a list, binary or a duration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeftOut {
    pub name: String,
    pub dtype: Dtype,
}

/// A time of day whose values the reference's engine reads as no time at
/// all (its `to_datetime` over the preview), which it says it dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadTimes {
    pub column: String,
    /// The values not read: every one the preview holds.
    pub unread: usize,
    /// The rows the preview holds.
    pub rows: usize,
    /// The first three distinct values, in the order they appear.
    pub examples: Vec<Scalar>,
}

/// One table, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredTable {
    pub name: String,
    /// The relation of the source the table is read from.
    pub source: Relation,
    pub columns: Vec<InferredColumn>,
    /// The columns left out, in table order.
    pub left_out: Vec<LeftOut>,
    /// The times of day the engine's reading drops, in table order.
    pub unread_times: Vec<UnreadTimes>,
    /// The text columns whose dates the engine's reading guessed a format
    /// for that puts the day first, with the format, in table order.
    pub day_first: Vec<(String, String)>,
    /// The columns that tied for the best score when the key was guessed,
    /// in table order, which left the table without one (empty otherwise).
    pub tied_keys: Vec<String>,
    /// The key's columns, in order: none, one, or several a caller named
    /// (`crate::assemble::apply_keys_and_times`).
    pub primary_key: Vec<String>,
    /// How inference chose the time column, before the links settle a
    /// lone edit stamp (`crate::assemble`).
    pub time: TimeChoice,
    /// The table's time column: inference's choice until the caller names
    /// one (`crate::assemble::apply_keys_and_times`).
    pub time_column: Option<String>,
    /// The column sets the catalog declares a primary or unique key over
    /// that the source enforces, refusing to repeat them
    /// ([`CatalogKey::enforced`]).
    pub unique: Vec<BTreeSet<String>>,
    /// The columns the catalog declares a primary or unique key of alone,
    /// enforced or not (a Databricks key is declared and not enforced): a
    /// key over one of them is the catalog's, held to its count as a
    /// declared key is (`crate::keys::hold`).
    pub declared: BTreeSet<String>,
}

impl InferredTable {
    /// Whether the source refuses both a repeated value of `key` and a row
    /// missing part of it, which leaves nothing to count.
    pub fn enforces(&self, key: &[String]) -> bool {
        let unique = self.keeps_unique(key);
        let never_null = key
            .iter()
            .all(|name| self.columns.iter().any(|c| &c.name == name && !c.nullable));
        unique && never_null
    }

    /// Whether the source keeps `key` unique: a declared primary or unique
    /// key within it, whether or not its columns may hold a null.
    pub fn keeps_unique(&self, key: &[String]) -> bool {
        let names: BTreeSet<&String> = key.iter().collect();
        self.unique
            .iter()
            .any(|set| set.iter().all(|column| names.contains(column)))
    }

    /// The column `name`, if the table has it.
    pub fn column(&self, name: &str) -> Option<&InferredColumn> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// The owner's switch of P1B-11 (2026-10-13): an empty table is refused as
/// an invalid request, which the reference's unnamed error says it is, not
/// as a rule the core lacks; the operation and rule stay named, and the
/// table is named by its source, as the reference's error names it.
pub fn empty_table(source: &Relation) -> CoreError {
    let refused = not_yet("empty_table");
    CoreError::new(Sentence::EmptyTable, "a table of the source holds no rows")
        .with_field("operation", refused.fields["operation"].clone())
        .with_field("rule", refused.fields["rule"].clone())
        .with_field(
            "source",
            Detail::List(source.source.iter().cloned().map(Scalar::Utf8).collect()),
        )
}

/// Infers table `name`, read as `source` in the source, from its catalog
/// columns and constraints and its preview (`schema` and `batches`, as
/// admitted), which holds the catalog's columns in order, reading text as
/// times by `times` and floats as Arrow-backed pandas holds them:
/// [`preview`], then [`finish`] with no whole-table read,
/// for a preview that holds every row. A preview of [`PREVIEW_ROWS`] rows
/// that leaves the choice of the time column to the whole table
/// ([`Previewed::whole_table_columns`]) is an internal error here: the
/// caller asks the source and calls [`finish`].
pub fn table(
    name: &str,
    source: &Relation,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
    batches: &[RecordBatch],
    times: &dyn TextTimes,
) -> Result<InferredTable, CoreError> {
    finish(
        preview(
            name,
            source,
            catalog,
            constraints,
            schema,
            batches,
            times,
            Floats::NanValue,
        )?,
        None,
    )
}

/// A table inferred from its preview, its time column not chosen yet, with
/// what the choice reads of the preview.
#[derive(Debug, Clone)]
pub struct Previewed {
    table: InferredTable,
    rows: usize,
    instants: Vec<Result<Vec<Option<i128>>, TextUnread>>,
    presence: Vec<Vec<Option<Value>>>,
    firsts: Vec<Option<Scalar>>,
}

impl Previewed {
    /// The table as inferred so far: its columns, key and what the engine
    /// said of its values, with no time column.
    pub fn table(&self) -> &InferredTable {
        &self.table
    }

    /// The columns the time column's choice reads over every row of the
    /// table, as the reference's choice asks its source (`full_range`,
    /// asked only of a preview that returned [`PREVIEW_ROWS`] rows, which
    /// may have left rows unread): each timestamp that is not the key and
    /// is not passed over by its name as a load stamp or a date of birth,
    /// in table order. Empty where the preview holds every row or no
    /// column is such a timestamp, and nothing is asked.
    pub fn whole_table_columns(&self) -> Vec<String> {
        if (self.rows as u64) < PREVIEW_ROWS {
            return vec![];
        }
        candidates(&self.table.columns, &self.table.primary_key)
            .into_iter()
            .map(|i| &self.table.columns[i].name)
            .filter(|name| !names::is_load_stamp(name) && !names::is_birth_date(name))
            .cloned()
            .collect()
    }
}

impl Previewed {
    /// The columns of `among` the time column's choice reads over every
    /// row of the table ([`Previewed::whole_table_columns`]), for a choice
    /// among them alone ([`choose_among`]).
    pub fn whole_table_columns_among(&self, among: &[String]) -> Vec<String> {
        if (self.rows as u64) < PREVIEW_ROWS {
            return vec![];
        }
        among
            .iter()
            .filter(|name| {
                self.table
                    .columns
                    .iter()
                    .any(|c| c.name == **name && c.stype == Stype::Timestamp)
                    && !self.table.primary_key.contains(*name)
                    && !names::is_load_stamp(name)
                    && !names::is_birth_date(name)
            })
            .cloned()
            .collect()
    }
}

/// The time column of a semantic model's table chosen among `among` alone,
/// as the old client's `_infer_time_column_among` chooses it
/// (`RE/base/table.py:859-893`): the choice of [`finish`] over those
/// columns, `whole` the whole table's reading of
/// [`Previewed::whole_table_columns_among`] where one was asked, `failed`
/// where asking it failed. A column of `among` that holds no timestamp is
/// no candidate.
pub fn choose_among(
    previewed: &Previewed,
    among: &[String],
    whole: Option<&WholeTable>,
    failed: bool,
) -> Result<TimeChoice, CoreError> {
    let table = &previewed.table;
    let mut columns = table.columns.clone();
    for column in &mut columns {
        if !among.contains(&column.name) && column.stype == Stype::Timestamp {
            // Not offered: read as no timestamp, so never a candidate.
            column.stype = Stype::Numerical;
        }
    }
    choose_time_column(
        &columns,
        &Read {
            instants: &previewed.instants,
            presence: &previewed.presence,
            firsts: &previewed.firsts,
        },
        previewed.rows,
        &table.primary_key,
        whole,
        failed,
    )
}

/// Chooses the time column of a table inferred from its preview
/// ([`choose_time_column`]), reading `whole`, the answer to the source
/// about [`Previewed::whole_table_columns`], where those are not empty. A
/// `whole` missing where they are, or naming other columns, is an internal
/// error: the caller asked the source otherwise than the choice reads it.
pub fn finish(
    previewed: Previewed,
    whole: Option<&WholeTable>,
) -> Result<InferredTable, CoreError> {
    choose(previewed, whole, false)
}

/// Chooses the time column of a table inferred from its preview where the
/// whole table was asked about ([`Previewed::whole_table_columns`]) and
/// its answer failed, as the reference's choice does when its `full_range`
/// raises: from the preview's rows, a column they would pass over for its
/// values kept, behind every column they do not, and named in
/// [`TimeChoice::unconfirmed`]. Where nothing was asked it is [`finish`].
pub fn finish_unchecked(previewed: Previewed) -> Result<InferredTable, CoreError> {
    let failed = !previewed.whole_table_columns().is_empty();
    choose(previewed, None, failed)
}

fn choose(
    previewed: Previewed,
    whole: Option<&WholeTable>,
    failed: bool,
) -> Result<InferredTable, CoreError> {
    let asked = previewed.whole_table_columns();
    let whole = match whole {
        None if asked.is_empty() || failed => None,
        Some(read)
            if !asked.is_empty() && read.columns.iter().map(|(name, _)| name).eq(asked.iter()) =>
        {
            Some(read)
        }
        _ => {
            return Err(CoreError::new(
                Sentence::Internal,
                "the time column's choice was given another reading of the whole table than it asks",
            ));
        }
    };
    let Previewed {
        mut table,
        rows,
        instants,
        presence,
        firsts,
    } = previewed;
    let time = choose_time_column(
        &table.columns,
        &Read {
            instants: &instants,
            presence: &presence,
            firsts: &firsts,
        },
        rows,
        &table.primary_key,
        whole,
        failed,
    )?;
    table.time_column = time.column.clone();
    table.time = time;
    Ok(table)
}

/// Infers table `name`, read as `source` in the source, from its catalog
/// columns and constraints and its preview (`schema` and `batches`, as
/// admitted), which holds the catalog's columns in order, reading text as
/// times by `times` and floats as the reference's frame of this source
/// holds them (`floats`), all but its time column ([`finish`] chooses it).
/// Inference reads the table by the name the schema gives it, as the
/// reference's engine does (its key's guess is by that name).
#[allow(
    clippy::too_many_arguments,
    reason = "the table, its catalog, its preview and the source's two readings"
)]
pub fn preview(
    name: &str,
    source: &Relation,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
    batches: &[RecordBatch],
    times: &dyn TextTimes,
    floats: Floats,
) -> Result<Previewed, CoreError> {
    preview_keyed(
        name,
        source,
        catalog,
        constraints,
        schema,
        batches,
        times,
        floats,
        false,
    )
}

/// The single-column keys a table's catalog declares, as inference reads
/// them: its primary key's, and the columns the source keeps unique.
struct Keyed {
    /// The one column of each declared primary key of one column.
    primary: BTreeSet<String>,
    /// The one column of each primary, unique or partial unique key of one
    /// column, enforced or not, and each column the source keeps unique by
    /// itself: the columns the catalog makes a key of.
    declared: BTreeSet<String>,
    /// The one column of each primary, unique or partial unique key of one
    /// column, and each column the source keeps unique by itself.
    unique: BTreeSet<String>,
}

impl Keyed {
    fn of(catalog: &[CatalogColumn], constraints: &[CatalogKey]) -> Keyed {
        let single = |kinds: &[KeyKind], enforced: bool| -> BTreeSet<String> {
            constraints
                .iter()
                .filter(|c| kinds.contains(&c.kind) && c.columns.len() == 1)
                .filter(|c| c.enforced || !enforced)
                .map(|c| c.columns[0].clone())
                .collect()
        };
        // A column the source keeps unique by itself (a SQLite rowid alias)
        // is unique as a key of its own is; a partial unique index keeps its
        // one column unique as the reference reads it (SQLite's
        // `unique_keys`), but makes no key of several. A key the source
        // declares and does not enforce (Databricks') keeps nothing unique.
        let mut unique = single(
            &[KeyKind::Primary, KeyKind::Unique, KeyKind::PartialUnique],
            true,
        );
        unique.extend(catalog.iter().filter(|c| c.unique).map(|c| c.name.clone()));
        let mut declared = single(
            &[KeyKind::Primary, KeyKind::Unique, KeyKind::PartialUnique],
            false,
        );
        declared.extend(catalog.iter().filter(|c| c.unique).map(|c| c.name.clone()));
        Keyed {
            primary: single(&[KeyKind::Primary], false),
            declared,
            unique,
        }
    }

    /// What the source says of `declared`, whose preview returned `field`
    /// (T3.16: one reading for `describe_source` and `materialize`): the type
    /// its declaration fixes, where the catalog reads one (SQLite's, as the
    /// reference types a column by it first, whatever its values), else its
    /// values' type; the type admitted; whether it may hold no value (a
    /// primary key holds none, but where the source lets one, SQLite's not
    /// its rowid alias, as its catalog says); whether a key of it alone
    /// keeps it unique; and whether it is a column of exact numbers.
    fn facts(
        &self,
        relation: &Relation,
        declared: &CatalogColumn,
        field: &Field,
        constraints: &[CatalogKey],
    ) -> Result<SourceFacts, CoreError> {
        let dtype = match declared.dtype {
            // A declaration the reference reads as unsupported (Databricks'
            // struct, map or interval, `_to_dtype`) is refused by name, as
            // a struct or map the values give is (register P1B-10).
            Some(Dtype::Unsupported) => return Err(not_yet("column_type")),
            Some(declared) => declared,
            None => dtype_of(field.data_type())?,
        };
        let is_primary = declared.primary_key || self.primary.contains(&declared.name);
        Ok(SourceFacts {
            relation: relation.clone(),
            dtype,
            received: ColumnType::from_arrow(&declared.name, field.data_type())?,
            nullable: !declared.not_null && (!is_primary || declared.key_nullable),
            unique: self.unique.contains(&declared.name),
            declared_type: None,
            exact_number: exact_number(field, constraints),
        })
    }
}

/// What the source says of each column of a table whose catalog declares
/// `catalog` and `constraints` and whose preview returned `schema`, in the
/// catalog's order, as [`preview_keyed`] reads it, with no inference: the
/// facts `materialize` reads of a column the host's schema holds none of
/// (T3.16). The declared type of a big integer is the caller's to set.
pub fn source_facts(
    relation: &Relation,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
) -> Result<Vec<(String, SourceFacts)>, CoreError> {
    if schema.fields().len() != catalog.len() {
        return Err(CoreError::new(
            Sentence::Internal,
            "a preview holds another number of columns than its catalog",
        ));
    }
    let keyed = Keyed::of(catalog, constraints);
    catalog
        .iter()
        .zip(schema.fields())
        .map(|(declared, field)| {
            Ok((
                declared.name.clone(),
                keyed.facts(relation, declared, field, constraints)?,
            ))
        })
        .collect()
}

/// [`preview`], where `key_stated` says whether the caller names the
/// table's key (`primary_keys=`, or a schema's own key): the key inference
/// would take then never decides anything, so one of a type no key takes is
/// left unset rather than refused (T3.8 fix 3), and the caller's key is
/// applied after, as the reference applies a named key over any it read.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, its catalog, its preview and the source's two readings"
)]
pub fn preview_keyed(
    name: &str,
    source: &Relation,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
    batches: &[RecordBatch],
    times: &dyn TextTimes,
    floats: Floats,
    key_stated: bool,
) -> Result<Previewed, CoreError> {
    preview_with(
        name,
        source,
        catalog,
        constraints,
        schema,
        batches,
        times,
        floats,
        Keying::Inferred { stated: key_stated },
    )
}

/// How a preview's table is keyed.
#[derive(Clone, Copy)]
enum Keying {
    /// By the catalog's key, else its one unique column, else a guess;
    /// `stated` where the caller names the key.
    Inferred { stated: bool },
    /// By nothing the preview reads: a semantic model's table, which its
    /// view keys (`RE/graph.py:1820-1925`, `:2083-2096`).
    Never,
}

/// The columns of a semantic model's table inferred from its rows, as the
/// old client's `Table.add_columns` infers each (`RE/base/table.py:225-318`):
/// `columns` the columns the conversion adds, each with the data type the
/// source's catalog or the view declares where either does
/// ([`CatalogColumn::dtype`]), else its values', and its meaning by its name
/// and values; `schema` and `batches` the rows sampled of them, in that
/// order. No key is read or guessed: the view's joins and keys set it.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, its columns, its rows and the source's two readings"
)]
pub fn view_table(
    name: &str,
    source: &Relation,
    columns: &[CatalogColumn],
    schema: &Schema,
    batches: &[RecordBatch],
    times: &dyn TextTimes,
    floats: Floats,
) -> Result<Previewed, CoreError> {
    preview_with(
        name,
        source,
        columns,
        &[],
        schema,
        batches,
        times,
        floats,
        Keying::Never,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the table, its catalog, its preview, the source's two readings and its keying"
)]
fn preview_with(
    name: &str,
    source: &Relation,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
    batches: &[RecordBatch],
    times: &dyn TextTimes,
    floats: Floats,
    keying: Keying,
) -> Result<Previewed, CoreError> {
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if rows == 0 {
        // The reference refuses an empty table as a RuntimeError.
        return Err(empty_table(source));
    }
    let keyed = Keyed::of(catalog, constraints);
    let declared_primary = &keyed.primary;
    let kept_unique = catalog.iter().filter(|c| c.unique).map(|c| c.name.clone());
    let unique = constraints
        .iter()
        .filter(|c| matches!(c.kind, KeyKind::Primary | KeyKind::Unique) && c.enforced)
        .map(|c| c.columns.iter().cloned().collect::<BTreeSet<_>>())
        .chain(kept_unique.map(|name| BTreeSet::from([name])))
        .collect::<Vec<_>>();

    let mut columns = Vec::with_capacity(catalog.len());
    let mut samples = Vec::with_capacity(catalog.len());
    let mut instants = Vec::with_capacity(catalog.len());
    let mut presence = Vec::with_capacity(catalog.len());
    let mut firsts = Vec::with_capacity(catalog.len());
    let mut left_out = Vec::new();
    let mut unread_times = Vec::new();
    let mut day_first = Vec::new();
    let mut primary = Vec::new();
    for (index, declared) in catalog.iter().enumerate() {
        let field = schema.field(index);
        let facts = keyed.facts(source, declared, field, constraints)?;
        let dtype = facts.dtype;
        if matches!(
            dtype,
            Dtype::Intlist
                | Dtype::Floatlist
                | Dtype::Stringlist
                | Dtype::Binary
                | Dtype::Timedelta
        ) {
            // No scalar form on the wire (a list, binary or a duration):
            // left out and named, as the reference's schema leaves it out.
            left_out.push(LeftOut {
                name: declared.name.clone(),
                dtype,
            });
            continue;
        }
        let sample = values::column(batches, index, floats)?;
        if dtype == Dtype::Time {
            // The engine reads a time of day's values as times, which none
            // is, and says how many it dropped.
            let unread = sample.iter().flatten().count();
            if unread > 0 {
                unread_times.push(UnreadTimes {
                    column: declared.name.clone(),
                    unread,
                    rows,
                    examples: values::times_of_day(batches, index, 3),
                });
            }
        }
        let mut guessed = None;
        let stype = stype_of(&declared.name, dtype, &sample, times, &mut guessed)?;
        if let Some(format) = guessed {
            day_first.push((declared.name.clone(), format));
        }
        let is_primary = declared.primary_key || declared_primary.contains(&declared.name);
        if is_primary {
            primary.push(declared.name.clone());
        }
        columns.push(InferredColumn {
            name: declared.name.clone(),
            dtype,
            stype,
            stype_before_key: None,
            received: facts.received,
            nullable: facts.nullable,
            unique: facts.unique,
            exact_number: facts.exact_number,
        });
        let text_time = dtype == Dtype::String && stype == Stype::Timestamp;
        instants.push(if dtype == Dtype::Time {
            // Dropped by the engine's reading: none is held.
            Ok(vec![None; rows])
        } else if text_time {
            // Read as the time column's choice reads them, to UTC; only
            // where they start is asked of them, and a reading that does
            // not decide is refused there.
            let text: Vec<Option<&str>> = sample
                .iter()
                .map(|v| match v {
                    Some(Value::Text(t)) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            times.read(&text, true)
        } else {
            Ok(values::instants(batches, index))
        });
        // What a timestamp held in each row: text as it is written, any
        // other as its instant (a time of day as none, as above).
        presence.push(if text_time {
            sample.clone()
        } else {
            match instants.last() {
                Some(Ok(read)) => read.iter().map(|v| v.map(Value::Int)).collect(),
                _ => vec![None; rows],
            }
        });
        firsts.push(if text_time {
            sample.iter().flatten().next().and_then(|v| match v {
                Value::Text(t) => Some(Scalar::Utf8(t.clone())),
                _ => None,
            })
        } else {
            values::first_temporal(batches, index)
        });
        samples.push(sample);
    }

    // The catalog's key, when it declares exactly one single-column one,
    // then its one unique column, then a guess from names and values; each
    // makes its column an identifier. A column left out is never the key.
    let held = |n: &&String| columns.iter().any(|c| &c.name == *n);
    let primary: Vec<String> = primary.iter().filter(held).cloned().collect();
    let unique_columns: Vec<&String> = catalog
        .iter()
        .map(|c| &c.name)
        .filter(|n| keyed.declared.contains(*n))
        .filter(held)
        .collect();
    let key_stated = match keying {
        Keying::Inferred { stated } => stated,
        Keying::Never => true,
    };
    let (key, tied_keys) = match (keying, primary.as_slice(), unique_columns.as_slice()) {
        (Keying::Never, _, _) => (None, Vec::new()),
        (_, [one], _) => (Some(one.clone()), Vec::new()),
        (_, _, [one]) => (Some((*one).clone()), Vec::new()),
        _ => match guess_key(name, &columns, &samples) {
            Guess::Key(key) => (Some(key), Vec::new()),
            Guess::None => (None, Vec::new()),
            Guess::Tied(tied) => (None, tied),
        },
    };
    let mut key = key;
    if let Some(found) = key.clone() {
        let column = columns
            .iter_mut()
            .find(|c| c.name == found)
            .expect("the key is a column of the table");
        if !matches!(column.dtype, Dtype::Int | Dtype::Float | Dtype::String) {
            if !key_stated {
                return Err(not_yet("key_type"));
            }
            // The caller's key replaces it: none is inferred.
            key = None;
        } else {
            column.stype = Stype::Id;
        }
    }

    Ok(Previewed {
        table: InferredTable {
            name: name.to_owned(),
            source: source.clone(),
            columns,
            left_out,
            unread_times,
            day_first,
            tied_keys,
            primary_key: key.into_iter().collect(),
            time: TimeChoice::default(),
            time_column: None,
            unique,
            declared: keyed.declared.clone(),
        },
        rows,
        instants,
        presence,
        firsts,
    })
}

/// A column's data type from the Arrow type its preview returned, as the
/// reference's reading of the catalog's types gives it for the plain types,
/// binary and the lists of them; a duration is `timedelta`, which the
/// reference's `_WIRE_DTYPE` has no wire dtype for, so it is left out (no
/// DuckDB type hands one over). Any other is refused by name (`column_type`).
/// A list is read as the reference reads a catalog type ending `[]`: of
/// integers, floats or text (binary is text to it, so a list of binary is
/// a list of text), refused otherwise where the reference fails unnamed
/// (P1B-51). Admission hands every list over as a `list`, so a fixed-size
/// list (P1B-9) and a list view (P1B-50) are read as lists too. A struct
/// and a map are refused (P1B-10).
fn dtype_of(data_type: &DataType) -> Result<Dtype, CoreError> {
    Ok(match data_type {
        DataType::Boolean => Dtype::Bool,
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Decimal128(_, 0) => Dtype::Int,
        DataType::Float32 | DataType::Float64 | DataType::Decimal128(_, _) => Dtype::Float,
        DataType::Utf8 => Dtype::String,
        DataType::Binary => Dtype::Binary,
        DataType::Date32 | DataType::Timestamp(_, _) => Dtype::Date,
        DataType::Time64(_) => Dtype::Time,
        DataType::Duration(_) => Dtype::Timedelta,
        DataType::List(element) | DataType::LargeList(element) => {
            match dtype_of(element.data_type()) {
                Ok(Dtype::Int) => Dtype::Intlist,
                Ok(Dtype::Float) => Dtype::Floatlist,
                Ok(Dtype::String | Dtype::Binary) => Dtype::Stringlist,
                _ => return Err(not_yet("column_type")),
            }
        }
        _ => return Err(not_yet("column_type")),
    })
}

fn is_numeric(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Int | Dtype::Float)
}

/// The semantic type: an identifier by its name, a timestamp by its type,
/// then a category by its name and how many distinct values the first
/// 1,000 rows hold, else the type's default.
fn stype_of(
    name: &str,
    dtype: Dtype,
    sample: &[Option<Value>],
    times: &dyn TextTimes,
    day_first: &mut Option<String>,
) -> Result<Stype, CoreError> {
    if matches!(dtype, Dtype::Int | Dtype::Float | Dtype::String) && names::says_id(name) {
        return Ok(Stype::Id);
    }
    match dtype {
        Dtype::Date | Dtype::Time => return Ok(Stype::Timestamp),
        Dtype::String => {
            if text_holds_dates(name, sample, times, day_first)? {
                return Ok(Stype::Timestamp);
            }
            if text_holds_categories(sample) {
                return Ok(Stype::Multicategorical);
            }
        }
        _ => {}
    }
    if dtype == Dtype::Bool {
        return Ok(Stype::Categorical);
    }
    let measured = is_numeric(dtype) && names::says_measure(name);
    if !measured {
        let first: Vec<Option<Value>> = sample.iter().take(1_000).cloned().collect();
        let held = first.iter().flatten().count();
        let distinct = values::distinct(&first);
        let share = |limit: f64| held > 0 && (distinct as f64 / held as f64) <= limit;
        let categorical = distinct < 20
            || if dtype == Dtype::String {
                share(0.5)
            } else {
                share(0.05)
            };
        if categorical {
            return Ok(Stype::Categorical);
        }
    }
    Ok(match dtype {
        Dtype::String => Stype::Text,
        _ => Stype::Numerical,
    })
}

/// How many values read as times: `fewest`, the fewer of the two orders'
/// counts where the values decide none; `listed`, those that read in
/// either order; `most`, those and the values in the client's shapes the
/// list does not read in either; `unshaped`, the values outside the shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextShare {
    pub fewest: usize,
    pub listed: usize,
    pub most: usize,
    pub unshaped: usize,
}

/// Text is a timestamp by the reference's score (`contains_timestamp`)
/// over the values the core's list reads (`times`): its first 100 values
/// hold one, and a name that says time is worth 0.3 and the share of those
/// values read as times the rest, scoring 1.0 or more, summed as the
/// reference sums them in floating point. The share is counted at fewest
/// and at most: where the values the list reads decide otherwise in the
/// two orders of day and month, the text is refused `text_time_order`
/// (register DT-16), or `text_time` where a value outside the list is
/// among them, as a source's column is refused (`Reader::column`: a value
/// outside the list before an open order). Otherwise the values the list
/// reads decide it: a value it does not read is no time under any name,
/// so such a column is text, never refused for its name (the owner's item
/// 24 of 2026-10-14; register DT-15 and DT-17).
/// Offsets mixed, which the reference refuses unnamed, are refused by name
/// (`text_time_zones`, register DT-11).
fn text_holds_dates(
    name: &str,
    sample: &[Option<Value>],
    times: &dyn TextTimes,
    day_first: &mut Option<String>,
) -> Result<bool, CoreError> {
    let first: Vec<Option<&str>> = sample
        .iter()
        .take(100)
        .flatten()
        .filter_map(|v| match v {
            Value::Text(text) if !text.is_empty() => Some(Some(text.as_str())),
            _ => None,
        })
        .collect();
    if first.is_empty() {
        return Ok(false);
    }
    // The format of its day-first dates, which it warns of.
    *day_first = times.day_first_format(&first);
    let named = if names::says_time(name) { 0.3 } else { 0.0 };
    let score = |held: usize| named + held as f64 / first.len() as f64 >= 1.0;
    match times.share(&first) {
        Ok(share) if score(share.fewest) != score(share.listed) => {
            let outside = share.most > share.listed || share.unshaped > 0;
            Err(not_yet(if outside {
                "text_time"
            } else {
                "text_time_order"
            }))
        }
        // The values the list reads decide it: one it does not read is no
        // time, under any name (the owner's item 24 of 2026-10-14; DT-15,
        // DT-17), so the column is text rather than refused.
        Ok(share) => Ok(score(share.fewest)),
        Err(TextUnread::Undecided) => Err(not_yet("text_time")),
        Err(TextUnread::Order) => Err(not_yet("text_time_order")),
        Err(TextUnread::MixedZones) => Err(not_yet("text_time_zones")),
    }
}

/// The most categories a multi-category column holds in its first 500
/// values (the reference's `MAX_CAT`).
const MOST_CATEGORIES: usize = 100;

/// Text holds several categories per value, as the reference reads it
/// (`contains_multicategorical`): over its first 500 values, nulls left
/// out, the separator is whichever of `;`, `:`, `|` and a tab occurs most
/// often, the first met on a tie (none of them, no categories); split by
/// it, the values hold at most 100 distinct parts, and the values
/// themselves are more than one and a half times as many distinct as the
/// parts. An empty part counts as a part.
fn text_holds_categories(sample: &[Option<Value>]) -> bool {
    let first: Vec<&str> = sample
        .iter()
        .take(500)
        .flatten()
        .filter_map(|v| match v {
            Value::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    // Each separator's count, in the order the values first hold it.
    let mut counts: Vec<(char, usize)> = Vec::new();
    for c in first.iter().flat_map(|text| text.chars()) {
        if !matches!(c, ';' | ':' | '|' | '\t') {
            continue;
        }
        match counts.iter_mut().find(|(seen, _)| *seen == c) {
            Some((_, n)) => *n += 1,
            None => counts.push((c, 1)),
        }
    }
    // The most frequent, the first met on a tie.
    let Some(separator) = counts
        .iter()
        .fold(None::<(char, usize)>, |best, &(c, n)| match best {
            Some((_, most)) if most >= n => best,
            _ => Some((c, n)),
        })
        .map(|(c, _)| c)
    else {
        return false;
    };
    let values: BTreeSet<&str> = first.iter().copied().collect();
    let parts: BTreeSet<&str> = first
        .iter()
        .flat_map(|text| text.split(separator))
        .collect();
    // `values > 1.5 * parts`, in integers.
    2 * values.len() > 3 * parts.len() && parts.len() <= MOST_CATEGORIES
}

/// What guessing a table's key found.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Guess {
    Key(String),
    None,
    /// Several columns tied for the best score, in table order: the table
    /// gets no key, and the reference warns, naming them.
    Tied(Vec<String>),
}

/// The table's key among its identifiers, or, with none, among the columns
/// named as the table is, by the reference's score (`infer_primary_key`):
/// a name that is the table's (in its singular, plural, snake or camel
/// forms) followed by `id`, `hash`, `key`, `code` or `uuid`, or the table's
/// name itself, and how distinct its values are. A score of 4 or more
/// counts; a tie for the best leaves the table without a key, the tied
/// columns named.
fn guess_key(table: &str, columns: &[InferredColumn], samples: &[Vec<Option<Value>>]) -> Guess {
    let mut candidates: Vec<usize> = (0..columns.len())
        .filter(|&i| columns[i].stype == Stype::Id)
        .collect();
    if candidates.is_empty() {
        let lowered = table.to_lowercase();
        let singular = lowered.strip_suffix('s');
        candidates = (0..columns.len())
            .filter(|&i| {
                let column = columns[i].name.to_lowercase();
                column == lowered || singular == Some(column.as_str())
            })
            .collect();
    }
    if candidates.is_empty() {
        return Guess::None;
    }
    let variants = table_name_variants(table);
    let mut scores: Vec<(usize, f64)> = Vec::new();
    for &i in &candidates {
        let name = &columns[i].name;
        let lowered = name.to_lowercase();
        let mut score: u32 = if lowered == "id" { 4 } else { 0 };
        let named = |variant: &str| -> u32 {
            suffix_score(&lowered, variant) + u32::from(table == "studies" && name == "nct_id")
        };
        if variants.contains(&lowered) {
            // The reference stops at the form the name equals, having
            // added what the forms it met first gave, in an order that
            // varies by run. No form scores a name equal to another form:
            // the forms differ only by their endings (`s`, `ies` for `y`,
            // `ing`, `info`) and underscores, none of them a key's suffix,
            // so what the forms met first add is always nothing and the
            // order never shows (`a_name_equal_to_a_form_scores_four`).
            score += 4;
        } else {
            score += variants.iter().map(|v| named(v)).sum::<u32>();
        }
        let sample = &samples[i];
        let share = values::distinct(sample) as f64 / sample.len() as f64;
        scores.push((i, f64::from(score) + 3.0 * share));
    }
    scores.retain(|(_, score)| *score >= 4.0);
    // Stable, so a tie keeps the columns' order.
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    match scores.as_slice() {
        [] => Guess::None,
        [(i, _)] => Guess::Key(columns[*i].name.clone()),
        [(i, best), (_, next), ..] if best != next => Guess::Key(columns[*i].name.clone()),
        [(_, best), ..] => Guess::Tied(
            scores
                .iter()
                .filter(|(_, score)| score == best)
                .map(|(i, _)| columns[*i].name.clone())
                .collect(),
        ),
    }
}

/// What one form of the table's name, `variant`, gives a column named
/// `lowered` (lowercased) by its key suffix: 5 for the form followed by the
/// suffix, with or without an underscore, and stopping there; else 2 for
/// each way the name ends so.
fn suffix_score(lowered: &str, variant: &str) -> u32 {
    let mut add = 0;
    for suffix in ["id", "hash", "key", "code", "uuid"] {
        if !lowered.ends_with(suffix) {
            continue;
        }
        if lowered == format!("{variant}_{suffix}") || lowered == format!("{variant}{suffix}") {
            add += 5;
            break;
        }
        if lowered.ends_with(&format!("{variant}_{suffix}")) {
            add += 2;
        }
        if lowered.ends_with(&format!("{variant}{suffix}")) {
            add += 2;
        }
    }
    add
}

/// The table's name in the forms a key's name may take it in: as given and
/// without an `info` ending, each lowercased, in snake_case and joined, in
/// the singular and plural, and without an `ing`.
fn table_name_variants(table: &str) -> BTreeSet<String> {
    let chars: Vec<char> = table.chars().collect();
    let lowered = table.to_lowercase();
    let mut bases = vec![table.to_owned()];
    if lowered.ends_with("_info") && chars.len() >= 5 {
        bases.push(chars[..chars.len() - 5].iter().collect());
    } else if lowered.ends_with("info") && chars.len() >= 4 {
        bases.push(chars[..chars.len() - 4].iter().collect());
    }
    let drop = |text: &str, n: usize| -> String {
        let chars: Vec<char> = text.chars().collect();
        chars[..chars.len().saturating_sub(n)].iter().collect()
    };
    let mut variants = BTreeSet::new();
    for base in bases {
        let lower = base.to_lowercase();
        let forms = [
            lower.clone(),
            snake_case(&base).to_lowercase(),
            base.replace('_', "").to_lowercase(),
        ];
        for form in &forms {
            variants.insert(form.clone());
            if lower.ends_with('s') {
                variants.insert(drop(form, 1));
            } else {
                variants.insert(format!("{form}s"));
            }
            if lower.ends_with("ies") {
                variants.insert(format!("{}y", drop(form, 3)));
            } else if lower.ends_with('y') {
                variants.insert(format!("{}ies", drop(form, 1)));
            }
            if lower.ends_with("ing") {
                variants.insert(drop(form, 3));
            }
        }
    }
    variants
}

/// The reference's snake_case: `(.)([A-Z][a-z]+)` to `\1_\2`, then
/// `([a-z0-9])([A-Z])` to `\1_\2`.
fn snake_case(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut first = Vec::with_capacity(chars.len() + 4);
    let mut i = 0;
    while i < chars.len() {
        let word = i + 2 < chars.len()
            && chars[i] != '\n'
            && chars[i + 1].is_ascii_uppercase()
            && chars[i + 2].is_ascii_lowercase();
        if word {
            let mut end = i + 2;
            while end < chars.len() && chars[end].is_ascii_lowercase() {
                end += 1;
            }
            first.push(chars[i]);
            first.push('_');
            first.extend_from_slice(&chars[i + 1..end]);
            i = end;
        } else {
            first.push(chars[i]);
            i += 1;
        }
    }
    let mut second = String::with_capacity(first.len() + 4);
    let mut i = 0;
    while i < first.len() {
        let c = first[i];
        if (c.is_ascii_lowercase() || c.is_ascii_digit())
            && first.get(i + 1).is_some_and(char::is_ascii_uppercase)
        {
            second.push(c);
            second.push('_');
            second.push(first[i + 1]);
            i += 2;
        } else {
            second.push(c);
            i += 1;
        }
    }
    second
}

/// What the preview holds of each column inferred, for the time column's
/// choice: its values as instants (`values::instants`, a time of day's as
/// none, since the engine's reading drops them; text as the engine reads
/// it, which may not decide), what each row holds (text as written), and
/// its first value held.
struct Read<'a> {
    instants: &'a [Result<Vec<Option<i128>>, TextUnread>],
    presence: &'a [Vec<Option<Value>>],
    firsts: &'a [Option<Scalar>],
}

/// The first real day: a value before it, such as 0001-01-01 or
/// 1900-01-01, stands for no date (`_FIRST_REAL_DAY`, 1901-01-01 UTC), in
/// nanoseconds since the epoch.
const FIRST_REAL_DAY: i128 = -2_177_452_800 * 1_000_000_000;

/// The microseconds since the epoch of Python's `datetime.min`,
/// 0001-01-01T00:00, and `datetime.max`, 9999-12-31T23:59:59.999999.
const DATETIME_MIN_US: i64 = -62_135_596_800_000_000;
const DATETIME_MAX_US: i64 = 253_402_300_799_999_999;

/// The offset east of UTC, in seconds, a zone named as an Arrow timestamp
/// names one has at an instant: the host's zone database's, as the run
/// resolves zones (`sdm_engine::time::Zones`), local mean time before the
/// zone's first transition included. Inference has no zones of its own, so
/// the caller hands them in.
pub trait WallClocks {
    /// The offset of `zone` at `utc_micros` microseconds since the epoch,
    /// or `None` where the run's zones do not hold `zone`.
    fn offset_seconds(&self, zone: &str, utc_micros: i64) -> Option<i32>;
}

/// The reference's time column choice reads the first value the preview
/// holds of each timestamp it chooses among, but a stamp of a load or a
/// date of birth (`_refused_for_values`, `present.iloc[0]`), as a Python
/// `datetime`: the UTC instant first (`datetime(1970, 1, 1)` and the
/// microseconds), then, for a value with a zone, its wall clock in that
/// zone at that instant (pyarrow's `as_py`, `astimezone`). Where either
/// reading, the UTC instant or the wall clock, lies outside the years 1 to
/// 9999 its Schema.from_source fails unnamed (an `OverflowError`). The core
/// refuses the table there by name, `VALUE_OUT_OF_RANGE` naming the table
/// and the column, the first such in table order (register P1B-81, T1.36
/// fixes 3, 11 and 13). `clocks` gives a zone's offset, as the run resolves zones; a
/// zone they do not hold is read at its UTC instant. A later row's value,
/// and a nanosecond's (which never leaves those years in any zone), are
/// read as before. The describe run calls this where the reference's
/// choice reads the value: after the whole table's reading is asked for,
/// before the choice ([`finish`]).
pub fn first_values_past_the_years(
    previewed: &Previewed,
    clocks: &dyn WallClocks,
) -> Result<(), CoreError> {
    let table = &previewed.table;
    for i in candidates(&table.columns, &table.primary_key) {
        let name = &table.columns[i].name;
        if names::is_load_stamp(name) || names::is_birth_date(name) {
            continue;
        }
        let Some(Scalar::Timestamp {
            unit: Unit::Us,
            zone,
            value,
        }) = &previewed.firsts[i]
        else {
            continue;
        };
        let offset = zone
            .as_deref()
            .and_then(|zone| clocks.offset_seconds(zone, *value))
            .unwrap_or(0);
        let wall = i128::from(*value) + i128::from(offset) * 1_000_000;
        let years = i128::from(DATETIME_MIN_US)..=i128::from(DATETIME_MAX_US);
        if !years.contains(&i128::from(*value)) || !years.contains(&wall) {
            return Err(CoreError::new(
                Sentence::TimeOutsideYears,
                "the first time a time column's choice reads is outside the years 1 to 9999",
            )
            .with_field("table", Detail::text(table.name.clone()))
            .with_field("column", Detail::text(name.clone())));
        }
    }
    Ok(())
}

/// The timestamps a table's time column is chosen among, in table order:
/// every column of a timestamp's meaning but the key.
fn candidates(columns: &[InferredColumn], key: &[String]) -> Vec<usize> {
    (0..columns.len())
        .filter(|&i| columns[i].stype == Stype::Timestamp && !key.contains(&columns[i].name))
        .collect()
}

/// What one query over every row of a table read for its time column's
/// choice (the reference's `TableRange`): the table's rows, and the range
/// of each column asked, in the order asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WholeTable {
    pub rows: usize,
    pub columns: Vec<(String, ColumnRange)>,
}

impl WholeTable {
    fn range(&self, name: &str) -> Option<&ColumnRange> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, r)| r)
    }
}

/// One column's `MIN`, `MAX` and `COUNT` over every row of its table (the
/// reference's `ColumnRange`), as the choice reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRange {
    /// The rows holding a value.
    pub held: usize,
    /// Whether the smallest value is the largest: text compared as it is
    /// written, any other value as itself.
    pub same: bool,
    /// The smallest value, as the contract's scalar: a date, a timestamp
    /// with its zone, a time of day or text; `None` when no row holds one.
    pub low: Option<Scalar>,
    /// The smallest value as an instant, in nanoseconds since the epoch in
    /// UTC (a timestamp without a zone read as UTC, a date at its
    /// midnight, text as the choice reads text): where the reference reads
    /// the column's start (`_as_instant`); `None` for a time of day, text
    /// that does not read, or no value.
    pub start: Option<i128>,
}

/// The answer to a `column_ranges` request asking `columns`, as admitted:
/// one row, the table's rows, then each column's smallest value, largest
/// value and rows holding one, in the order asked, each value of the
/// column's own type (the caller has held the answer to that shape). Text
/// is read as a time by `times`, to UTC, as the choice reads it.
pub fn whole_table(
    columns: &[String],
    batches: &[RecordBatch],
    times: &dyn TextTimes,
) -> Result<WholeTable, CoreError> {
    let internal = |message: &'static str| CoreError::new(Sentence::Internal, message);
    let batch = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(|| internal("the whole table's reading holds no row"))?;
    if batch.num_rows() != 1
        || batches.iter().map(RecordBatch::num_rows).sum::<usize>() != 1
        || batch.num_columns() != 1 + 3 * columns.len()
    {
        return Err(internal(
            "the whole table's reading is not of the shape asked",
        ));
    }
    let one = std::slice::from_ref(batch);
    let count = |index: usize| -> Result<usize, CoreError> {
        match values::column(one, index, Floats::NanValue)?.first() {
            Some(Some(Value::Int(n))) => usize::try_from(*n)
                .map_err(|_| internal("a count of the whole table's reading is negative")),
            _ => Err(internal("a count of the whole table's reading is not one")),
        }
    };
    let rows = count(0)?;
    let mut read = Vec::with_capacity(columns.len());
    for (at, name) in columns.iter().enumerate() {
        let (low, high, held) = (1 + 3 * at, 2 + 3 * at, 3 + 3 * at);
        let lows = values::column(one, low, Floats::NanValue)?;
        let highs = values::column(one, high, Floats::NanValue)?;
        let data_type = batch.column(low).data_type();
        let (scalar, start) = match (data_type, lows.first()) {
            (_, Some(None) | None) => (None, None),
            (DataType::Utf8, Some(Some(Value::Text(text)))) => (
                Some(Scalar::Utf8(text.clone())),
                times
                    .read(&[Some(text.as_str())], true)
                    .ok()
                    .and_then(|read| read.first().copied().flatten()),
            ),
            (DataType::Time64(_), _) => {
                (values::times_of_day(one, low, 1).into_iter().next(), None)
            }
            _ => (
                values::first_temporal(one, low).map(infinity_as_read),
                values::instants(one, low).first().copied().flatten(),
            ),
        };
        read.push((
            name.clone(),
            ColumnRange {
                held: count(held)?,
                same: lows.first() == highs.first(),
                low: scalar,
                start,
            },
        ));
    }
    Ok(WholeTable {
        rows,
        columns: read,
    })
}

/// A whole table's smallest timestamp as the reference reads it: DuckDB's
/// infinity (Arrow's `i64::MAX`, its negative `-i64::MAX`, of any unit) is
/// the largest or smallest `datetime` its `fetchall` hands over, without a
/// zone, which the reference shows as it is (FX-4); any other value as it
/// is.
fn infinity_as_read(value: Scalar) -> Scalar {
    match value {
        Scalar::Timestamp { value, .. } if value == i64::MAX || value == -i64::MAX => {
            Scalar::Timestamp {
                unit: Unit::Us,
                zone: None,
                value: if value > 0 {
                    DATETIME_MAX_US
                } else {
                    DATETIME_MIN_US
                },
            }
        }
        other => other,
    }
}

/// The time column among the timestamps, as the reference chooses it
/// (`predictive_query.time_columns.choose_time_column`): a stamp of a load
/// or a date of birth is passed over by its name; any other timestamp held
/// in no row, or in fewer than half the rows, or holding one value in every
/// row of a table of more than one, is passed over for its values; a stamp
/// of the last edit is passed over too, and taken only when it is the one
/// timestamp left (settled by the links, [`TimeChoice::for_links`]); of the
/// rest, the one left is taken, else the one creation stamp, else the one
/// whose values start earliest, placeholders left out, the first in table
/// order on a tie. A time of day holds no value the engine's preview
/// reads, so the preview passes it over as holding none. The key is never
/// the time column.
///
/// Where the preview may not hold every row, the values are read over the
/// whole table, `whole` (the reference's `full_range`): its rows, and each
/// column's count of values held and whether its smallest value is its
/// largest, decide whether it is passed over (a time of day included,
/// whose values the source counts), the value named for one held in every
/// row being the preview's first, else the smallest; and where a column
/// starts is its smallest value, unless that reads as no instant or is a
/// placeholder, when it is the preview's earliest real value, as the
/// reference reads it.
fn choose_time_column(
    columns: &[InferredColumn],
    read: &Read<'_>,
    rows: usize,
    key: &[String],
    whole: Option<&WholeTable>,
    failed: bool,
) -> Result<TimeChoice, CoreError> {
    let candidates = candidates(columns, key);
    if candidates.is_empty() {
        return Ok(TimeChoice::default());
    }
    let mut refused = Vec::new();
    let mut unconfirmed = Vec::new();
    let mut plain = Vec::new();
    let mut doubted = Vec::new();
    let mut edited = Vec::new();
    for &i in &candidates {
        let name = &columns[i].name;
        if names::is_load_stamp(name) {
            refused.push((name.clone(), PassedOver::LoadStamp));
            continue;
        }
        if names::is_birth_date(name) {
            refused.push((name.clone(), PassedOver::BirthDate));
            continue;
        }
        let over = whole.and_then(|w| w.range(name).map(|r| (w.rows, r)));
        let why = refused_for_values(&read.presence[i], read.firsts[i].as_ref(), rows, over);
        if let Some(why) = &why {
            if !failed {
                refused.push((name.clone(), why.clone()));
                continue;
            }
            // The whole table's answer failed: what the rows read say is
            // kept and named, not refused.
            unconfirmed.push((name.clone(), why.clone()));
        }
        if names::is_edit_stamp(name) {
            refused.push((name.clone(), PassedOver::EditStamp));
            edited.push(name.clone());
        } else if why.is_some() {
            doubted.push(i);
        } else {
            plain.push(i);
        }
    }
    // A column the rows read cast doubt on is taken only when no other is.
    if plain.is_empty() {
        plain = doubted;
    }
    let chosen = |column: Option<String>, how: Option<Taken>, among: Vec<String>| TimeChoice {
        column,
        instead_of: vec![],
        how,
        among,
        refused: refused.clone(),
        fallback: None,
        unconfirmed: unconfirmed.clone(),
    };
    let names_of = |indices: &[usize]| -> Vec<String> {
        indices.iter().map(|&i| columns[i].name.clone()).collect()
    };
    if plain.is_empty() {
        if let [one] = edited.as_slice() {
            return Ok(TimeChoice {
                fallback: Some(one.clone()),
                ..chosen(None, None, vec![])
            }
            .for_links(false));
        }
        return Ok(chosen(None, None, vec![]));
    }
    let created: Vec<usize> = plain
        .iter()
        .copied()
        .filter(|&i| names::is_creation_stamp(&columns[i].name))
        .collect();
    if let [one] = created.as_slice() {
        let how = if plain.len() > 1 {
            Taken::Created
        } else {
            Taken::Only
        };
        return Ok(chosen(
            Some(columns[*one].name.clone()),
            Some(how),
            names_of(&plain),
        ));
    }
    let among = if created.is_empty() { plain } else { created };
    if let [one] = among.as_slice() {
        return Ok(chosen(
            Some(columns[*one].name.clone()),
            Some(Taken::Only),
            names_of(&among),
        ));
    }
    let earliest = earliest(&among, columns, read, whole)?;
    Ok(chosen(
        Some(columns[earliest].name.clone()),
        Some(Taken::Earliest),
        names_of(&among),
    ))
}

/// The same time policy over a complete constructor snapshot. These are
/// every row, not a SQL preview needing a later aggregate or text reader.
pub(crate) fn frame_time(
    columns: &[InferredColumn],
    presence: &[Vec<Option<Value>>],
    batches: &[RecordBatch],
    rows: usize,
    key: &[String],
) -> Result<TimeChoice, CoreError> {
    let instants: Vec<_> = columns
        .iter()
        .enumerate()
        .map(|(at, column)| {
            Ok(if column.stype == Stype::Timestamp {
                values::instants(batches, at)
            } else {
                vec![]
            })
        })
        .collect();
    let firsts: Vec<_> = columns
        .iter()
        .enumerate()
        .map(|(at, column)| {
            (column.stype == Stype::Timestamp)
                .then(|| values::first_temporal(batches, at))
                .flatten()
        })
        .collect();
    choose_time_column(
        columns,
        &Read {
            instants: &instants,
            presence,
            firsts: &firsts,
        },
        rows,
        key,
        None,
        false,
    )
}

/// Why a column is passed over for its values (`_refused_for_values`):
/// none held; more than half missing; or, in a table of more than one row,
/// one value in every row that holds one, which is named. Read over the
/// preview's `rows` rows, or, where the whole table was read, `whole`'s
/// rows and the column's range there, the value named the preview's first
/// held, else the smallest. Text is compared as it is written, as the
/// reference compares the rows it read.
fn refused_for_values(
    presence: &[Option<Value>],
    first: Option<&Scalar>,
    rows: usize,
    whole: Option<(usize, &ColumnRange)>,
) -> Option<PassedOver> {
    let held: Vec<&Value> = presence.iter().flatten().collect();
    let (rows, held, same, first) = match whole {
        Some((rows, range)) => (
            rows,
            range.held,
            range.held > 0 && range.same,
            first.cloned().or_else(|| range.low.clone()),
        ),
        None => (
            rows,
            held.len(),
            !held.is_empty() && held.iter().all(|v| *v == held[0]),
            first.cloned(),
        ),
    };
    let missing = rows.saturating_sub(held);
    if held == 0 {
        return Some(PassedOver::Constant {
            value: None,
            missing,
            rows,
        });
    }
    if 2 * missing > rows {
        return Some(PassedOver::Sparse { missing, rows });
    }
    if rows > 1 && same {
        return Some(PassedOver::Constant {
            value: first,
            missing,
            rows,
        });
    }
    None
}

/// Whether an instant is a real start: not before 1901 and not the epoch's
/// midnight, which stand for no date.
fn real(instant: i128) -> bool {
    instant >= FIRST_REAL_DAY && instant != 0
}

/// The column of `among` whose values start earliest (`_earliest`), the
/// first in table order on a tie or when none holds a real value: a value
/// before 1901 or the epoch's midnight is a placeholder, no start. Where
/// the whole table was read, a column starts at its smallest value when
/// that is a real instant, and otherwise at the preview's earliest real
/// value, as the reference reads it. Text the preview holds whose reading
/// does not decide is refused there (`text_time`, or `text_time_order` for
/// an order of day and month left open).
fn earliest(
    among: &[usize],
    columns: &[InferredColumn],
    read: &Read<'_>,
    whole: Option<&WholeTable>,
) -> Result<usize, CoreError> {
    let mut best: Option<(i128, usize)> = None;
    for &i in among {
        let over = whole
            .and_then(|w| w.range(&columns[i].name))
            .and_then(|r| r.start)
            .filter(|&start| real(start));
        let start = match over {
            Some(start) => Some(start),
            None => read.instants[i]
                .as_ref()
                .map_err(|unread| match unread {
                    TextUnread::Order => not_yet("text_time_order"),
                    _ => not_yet("text_time"),
                })?
                .iter()
                .flatten()
                .copied()
                .filter(|&v| real(v))
                .min(),
        };
        if let Some(start) = start
            && best.is_none_or(|(low, _)| start < low)
        {
            best = Some((start, i));
        }
    }
    Ok(best.map_or(among[0], |(_, i)| i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names_take_the_reference_forms() {
        let orders = table_name_variants("orders");
        assert!(orders.contains("orders") && orders.contains("order"));
        let users = table_name_variants("UserInfo");
        for form in ["userinfo", "user_info", "user", "users", "userinfos"] {
            assert!(users.contains(form), "{form}: {users:?}");
        }
        let cities = table_name_variants("cities");
        assert!(cities.contains("city"));
        assert_eq!(snake_case("UserInfo"), "User_Info");
        assert_eq!(snake_case("HTTPRequest2Go"), "HTTP_Request2_Go");
        assert_eq!(snake_case("already_snake"), "already_snake");
    }

    /// The reference adds, for a column named as one form of the table's
    /// name, what the forms it meets before that one give, in an order
    /// that varies by run; no other form ever gives such a name anything,
    /// so the core's score of 4 is the reference's in every order. Held
    /// over every table name of up to three characters from an alphabet of
    /// the forms' endings, the suffixes' letters, an underscore and
    /// characters whose lowercase differs in length or kind.
    #[test]
    fn a_name_equal_to_a_form_scores_four() {
        let alphabet: Vec<char> = "aisdgnfoyuehkcIDSKY_\u{130}\u{212a}\u{3a3}"
            .chars()
            .collect();
        let mut names = vec![String::new()];
        let mut checked = 0;
        for _ in 0..3 {
            names = names
                .iter()
                .flat_map(|name| alphabet.iter().map(move |c| format!("{name}{c}")))
                .collect();
            for table in &names {
                let variants = table_name_variants(table);
                for form in &variants {
                    for other in variants.iter().filter(|v| *v != form) {
                        assert_eq!(
                            suffix_score(form, other),
                            0,
                            "{table:?}: {form:?} by {other:?}"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 12_000, "{checked}");
    }
}
