// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What each [`Sentence`] says: one shared template a sentence, data only
//! (invariant 8 of the Phase 3 structure design, T3.17). The core picks the
//! sentence of every error and warning; each host fills its template with
//! the values the error or warning names, as that host writes them, and
//! formats nothing else. [`template`] matches every sentence with no
//! wildcard, so a sentence with no template does not compile.
//!
//! A template is text with placeholders in braces, `{{` and `}}` standing
//! for a brace:
//!
//! - `{name}`: the field `name` as text; `{name[0]}` the first of a list;
//!   `{name:,}` a whole number with its thousands grouped;
//! - `{name!c}`: the field with a conversion, the host's one value
//!   formatter: `r` the value as the host writes it as a literal, `l` a
//!   list of them as an and-list (`'a', 'b' and 'c'`), `a` a list of text
//!   as an and-list as it is, `j` joined by commas, `c` a list of text
//!   joined by commas as it is, `R` joined by semicolons, `d` joined by
//!   dots, `p` a row of values (a list as the host writes a tuple, else as
//!   `r`), `t` a date or time as the host's time type writes it, `T` a list
//!   of times of day or text as the host writes a list of them, and for a
//!   value the caller gave, held as the JSON text the host wrote it in, `G`
//!   as a literal, `S` as text, `N` the name of its type and `I` the whole
//!   number it is (so `{name!I:,}` groups its thousands);
//! - `{name:to(other)}`: a list paired with the list `other`, each pair as
//!   the two literals joined by ` to `, the pairs by commas;
//! - `{name:place}`, `{name:backend}`: a database the field names, by the
//!   names the data layer gives it ([`DatabaseNames`]);
//! - `{host.name}`: a value only the host holds, such as a candidate as the
//!   caller gave it, which the host supplies by name (conversions as
//!   above);
//! - `{advice:id(field, ...)}`: what to write in the host's language, an
//!   [`Advice`] each host spells once, given the fields named.
//!
//! The sentence of the part of a question asked in parts an error happened
//! in (a forecast's step, a ranking's candidate, [`crate::Code::Part`]) is
//! said after the error's own, which ends in a full stop where it does
//! not already end a sentence.
//!
//! A sentence that relays a host library's own message (its date parser's
//! day-first dates, its numeric library's cast overflow, its language's
//! comparison of unlike types) is marked by its [`Owner`]: its template is a neutral shared sentence, and
//! the host whose library it is says it in that library's words. Warnings
//! given one after another are said together where their [`Group`] says
//! so, as data ([`grouping`]).

use serde::Serialize;

use crate::error::{Code, Sentence, WarningCode};

/// Whose words a sentence is: every host's, or, for a message a host's
/// own library gives, that library's, which the host says in its words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    /// The template is every host's.
    Shared,
    /// The host's date parser's own message, of dates read day first.
    HostDates,
    /// The host's numeric library's own message, of a cast that overflows.
    HostCast,
    /// The host language's own message, of a comparison of unlike types.
    HostComparison,
}

/// What a sentence says, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Template {
    pub text: &'static str,
    pub owner: Owner,
    /// The group the warning is said in with the warnings next to it.
    pub group: Option<Group>,
}

const fn said(text: &'static str) -> Template {
    Template {
        text,
        owner: Owner::Shared,
        group: None,
    }
}

const fn owned(text: &'static str, owner: Owner) -> Template {
    Template {
        text,
        owner,
        group: None,
    }
}

const fn grouped(text: &'static str, group: Group) -> Template {
    Template {
        text,
        owner: Owner::Shared,
        group: Some(group),
    }
}

/// Warnings said together: each warning of a group given right after
/// another of it whose [`Grouping::agree`] fields hold the same values is
/// said with it, as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    TimeColumnsGuessed,
    ColumnsLeftOut,
    /// What a semantic model's conversion left out, said as one warning
    /// opened by the model as the host names it
    /// (`CL/materializers/sources.py:285-294`, T3.27).
    ViewReadInPart,
}

impl Group {
    /// Every group.
    pub const ALL: &'static [Group] = &[
        Group::TimeColumnsGuessed,
        Group::ColumnsLeftOut,
        Group::ViewReadInPart,
    ];
}

/// How a group of warnings is said: the code they share, the fields they
/// agree on, the opening (a template over the first warning's fields,
/// naming only those), what goes between two of them, and the closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Grouping {
    pub code: WarningCode,
    pub agree: &'static [&'static str],
    pub opening: &'static str,
    pub between: &'static str,
    pub closing: &'static str,
}

/// How `group` is said, as the reference says it.
pub fn grouping(group: Group) -> Grouping {
    match group {
        Group::TimeColumnsGuessed => Grouping {
            code: WarningCode::TimeColumnsGuessed,
            agree: &[],
            opening: "Some time columns were inferred, or refused, on a guess about what a column holds:\n- ",
            between: "\n- ",
            closing: "",
        },
        Group::ColumnsLeftOut => Grouping {
            code: WarningCode::ColumnsLeftOut,
            agree: &["table"],
            opening: "{table} leaves out ",
            between: ", ",
            closing: ": a request carries one scalar value per cell, and these hold none",
        },
        Group::ViewReadInPart => Grouping {
            code: WarningCode::ViewReadInPart,
            agree: &[],
            opening: "{host.semantic_model!r} has what a graph cannot hold, so it is read in part:\n- ",
            between: "\n- ",
            closing: "",
        },
    }
}

macro_rules! advice {
    ($($variant:ident = $name:literal, [$($argument:literal),*];)*) => {
        /// What to write in the host's language, which each host spells
        /// once: the closed list, each with the fields it is given, in
        /// order.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Advice {
            $($variant,)*
        }

        impl Advice {
            /// Every advice.
            pub const ALL: &'static [Advice] = &[$(Advice::$variant,)*];

            /// The advice's name, as a template writes it.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Advice::$variant => $name,)*
                }
            }

            /// What the advice is given, by name, in order.
            pub const fn arguments(self) -> &'static [&'static str] {
                match self {
                    $(Advice::$variant => &[$($argument),*],)*
                }
            }
        }
    };
}

advice! {
    AppendRelationship = "append_relationship", ["tables"];
    Booleans = "booleans", [];
    BuildRelationship = "build_relationship", [];
    ChooseStatistic = "choose_statistic", [];
    ClearTimeColumn = "clear_time_column", ["table"];
    ContextSamples = "context_samples", ["context_rows_fast"];
    EmptyRequest = "empty_request", [];
    GiveAHop = "give_a_hop", [];
    KeepOneLink = "keep_one_link", ["source_table", "target_table", "others"];
    MaterializeCall = "materialize_call", [];
    MissingTime = "missing_time", [];
    NameCandidates = "name_candidates", ["column"];
    NamePrimaryKey = "name_primary_key", ["table"];
    NameTimeColumn = "name_time_column", ["table", "other"];
    Nothing = "nothing", [];
    OptimizeIndices = "optimize_indices", [];
    PredictCall = "predict_call", [];
    RaiseMostParts = "raise_most_parts", [];
    SemanticModel = "semantic_model", ["metric_view"];
    ResultFrame = "result_frame", [];
    SetPrimaryKey = "set_primary_key", ["target_table", "target_column"];
    SkipKeyCheck = "skip_key_check", [];
    SmallerCalls = "smaller_calls", [];
    TrustExpressions = "trust_expressions", [];
}

/// Sentences whose improvements remain proposed under the recorded owner
/// decisions (T3.27). A producer must not use one until the owner approves
/// it; releasing an approved name here changes neither its wire identity
/// nor its template. The blanket null-anchor refusal remains unapproved.
pub const PROPOSED: &[Sentence] = &[Sentence::NullAnchor];

/// A database a sentence names: by the name the core gives it, as a
/// product (`backend`), and as the place a table is held (`place`). The
/// data layer holds each database's (`sdm_data::DATABASE_NAMES`), since a
/// dialect is its alone; the host's table carries them beside the
/// sentences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DatabaseNames {
    pub name: &'static str,
    pub backend: &'static str,
    pub place: &'static str,
}

/// What `sentence` says.
pub fn template(sentence: Sentence) -> Template {
    use Sentence as S;
    match sentence {
        S::Internal => said(r#"the core failed; this is a defect, not a problem with the request"#),
        S::ProviderFailed => said(r#"the data source failed"#),
        S::TransportFailed => said(
            r#"the transport to the model endpoint failed: the endpoint could not be reached, or its answer could not be read"#,
        ),
        S::UnknownReply => said(r#"the host answered a request the core did not make"#),
        S::ReplyMissing => said(r#"the host went on before answering every request"#),
        S::WrongReply => said(r#"the host answered a request with the wrong kind of reply"#),
        S::RunFinished => said(r#"the run had already finished"#),
        S::RunCancelled => said(r#"the run was cancelled"#),
        S::RunPoisoned => said(r#"the run failed earlier and can no longer be used"#),
        S::RunBusy => said(r#"the run was already in a call; a run takes one call at a time"#),
        S::InvalidMessage => said(r#"the host sent a message the core cannot read"#),
        S::UnknownModel => said(r#"the model named does not exist"#),
        S::MissingExtra => said(r#"an optional dependency this call needs is not installed"#),
        S::SessionsUnsupported => said(r#"the model endpoint does not support sessions"#),
        S::RelationNoPart => said(
            r#"the core asked the data source for a {request} of a relation named by no part, which no source reads"#,
        ),
        S::RelationTooManyParts => said(
            r#"the core asked the data source for a {request} of a relation of {parts:,} name parts, and it reads at most {most:,}"#,
        ),
        S::DerivedRelationUnread => said(
            r#"the core asked the data source for a {request} of a derived relation, which it does not read"#,
        ),
        S::DerivedRelationMalformed => said(
            r#"the core asked the data source for a {request} of a derived relation whose columns do not each read one source column or expression; this is a defect, not a problem with the request"#,
        ),
        S::MetricViewAsTable => said(
            r#"{metric_view!r} is a metric view, not a table: its measures are aggregations with no value per row. Read it as a semantic model instead, with {advice:semantic_model(metric_view)}."#,
        ),
        S::KeySetTypeUnbound => said(
            r#"the core asked the data source to match keys of a type its SQL does not bind as text"#,
        ),
        S::KeySetIntegerRange => said(
            r#"this SQL key binding cannot represent integer keys outside the signed 64-bit range"#,
        ),
        S::DataRequestRefused => said(
            r#"the core asked the data source for a {request} whose {field} it does not take; this is a defect, not a problem with the request"#,
        ),
        S::UnsupportedType => said(
            r#"column {column!r} has the Arrow type {arrow_type}, which the core does not take"#,
        ),
        S::NotYetDescribe => said(
            r#"describing this source needs the {rule!r} rule, which is not one this client's core supports yet"#,
        ),
        S::EmptyTable => said(r#"Table '{source!d}' is empty"#),
        S::TableNameEmpty => said(r#"The name {table!r} must be a non-empty string."#),
        S::TableNameCharacters => {
            said(r#"The name {table!r} contains one or more invalid characters: {characters!r}."#)
        }
        S::TimeOutsideYears => said(
            r#"{table}.{column} holds a time outside the years 1 to 9999, which choosing a time column cannot read"#,
        ),
        S::NoTable => said(r#"no table {table!r} in this schema; it has {tables!r}"#),
        S::KeyNotNames => said(
            r#"a primary key of {table!r} names a column or a sequence of them, and the one given is neither"#,
        ),
        S::KeyNotNamesGiven => {
            said(r#"a primary key names a column or a sequence of them, got {given}"#)
        }
        S::TimeColumnNotAName => said(
            r#"table {table!r} has no column of the name given: a time column is named by its name, as text"#,
        ),
        S::TimeColumnNotANameGiven => {
            said(r#"table {table!r} has no column {given}; it has {columns!r}"#)
        }
        S::ColumnNotAName => said(
            r#"table {table!r} has no column of the name given: a column is named by its name, as text"#,
        ),
        S::ColumnNotANameGiven => {
            said(r#"table {table!r} has no column {given}; it has {columns!r}"#)
        }
        S::EmptyKey => said(r#"table {table!r} was given an empty primary key"#),
        S::NoColumn => said(r#"table {table!r} has no column {column!r}; it has {columns!r}"#),
        S::KeyColumnTwice => said(r#"a primary key of {table!r} names {column!r} more than once"#),
        S::KeyCannotIdentify => said(
            r#"column {column!r} holds {dtype!r}, which cannot identify a row, so it cannot be a primary key"#,
        ),
        S::KeyIsTimeColumn => said(
            r#"column {column!r} is already the time column of table {table!r}, so it cannot also be its primary key"#,
        ),
        S::TimeIsKey => said(
            r#"column {column!r} is already the primary key of table {table!r}, so it cannot also be its time column"#,
        ),
        S::MeaningNotCarried => {
            said(r#"column {column!r} holds {dtype!r}, which cannot mean {stype!r}"#)
        }
        S::NotASemanticType => {
            said(r#"{given_text!r} is not a semantic type; use one of {stypes!r}"#)
        }
        S::NotASemanticTypeGiven => {
            said(r#"{given} is not a semantic type; use one of {stypes!r}"#)
        }
        S::NotASemanticTypeUnread => said(
            r#"the meaning given for {table}.{column} is not a semantic type; use one of {stypes!r}"#,
        ),
        S::LeftOutKey => said(
            r#"{table}.{column} is the table's primary key; set that to something else, or {advice:nothing()}, before leaving the column out"#,
        ),
        S::LeftOutTimeColumn => said(
            r#"{table}.{column} is the table's time column; set that to something else, or {advice:nothing()}, before leaving the column out"#,
        ),
        S::NotARelationship => said(
            r#"a relationship must be a Relationship. Build one with {advice:build_relationship()}"#,
        ),
        S::NotARelationshipGiven => said(
            r#"a relationship must be a Relationship, got {given}. Build one with {advice:build_relationship()}"#,
        ),
        S::LinkUnheld => {
            said(r#"a relationship given names a table or column the schema does not hold"#)
        }
        S::LinkNotTheKey => said(
            r#"{source_table}.{source_column} -> {target_table}.{target_column} does not join to a primary key: table {target_table!r} has {key[0]!r}. A relationship points at the key that identifies a row, so set that key first if it is not the one inference found"#,
        ),
        S::LinkNoKey => said(
            r#"{source_table}.{source_column} -> {target_table}.{target_column} does not join to a primary key: table {target_table!r} has no primary key. A relationship points at the key that identifies a row, so set that key first if it is not the one inference found"#,
        ),
        S::LinkCompositeKey => said(
            r#"{source_table}.{source_column} -> {target_table}.{target_column} points at {target_table!r}, whose primary key is {key!r}. An edge joins one column to one key, so a table with a composite key cannot be the target of a relationship"#,
        ),
        S::LinkTypes => said(
            r#"{source_table}.{source_column} -> {target_table}.{target_column} joins {source_dtype!r} to {target_dtype!r}, which cannot match. Give both columns the same kind of value"#,
        ),
        S::PrimaryKeyAsForeignKey => said(
            r#"Cannot treat the primary key of table '{source_table}' as a foreign key. Remove either the primary key or the link before proceeding."#,
        ),
        S::EdgeKeyTypes => said(
            r#"Edge(src_table={source_table!r}, fkey={source_column!r}, dst_table={target_table!r}) is invalid as foreign key {source_column!r} and primary key {target_column!r} have incompatible data types (got foreign key data type {source_dtype!r} and primary key data type {target_dtype!r})"#,
        ),
        S::ColumnNotInSource => said(
            r#"{table}.{column} is a column of the schema that {source!d} does not hold: the source has no such column. Remove it from the schema, or point the table at a source table that holds it."#,
        ),
        S::ColumnTypeUnmatched => said(
            r#"{table}.{column} is given as {dtype!r}, and the source holds it as {source_dtype!r}, which cannot match. Give the column the type the source holds."#,
        ),
        S::SchemaTableTwice => said(r#"the schema holds table {table!r} twice"#),
        S::SchemaColumnTwice => said(r#"table {table!r} holds column {column!r} twice"#),
        S::IncompatibleSemanticType => said(
            r#"Column {column!r} received an incompatible semantic type (got dtype={dtype!r} and stype={stype!r})"#,
        ),
        S::KeyNotUniqueColumn => said(
            r#"{table}.{column} is not unique enough to identify a row: {repeats:,} of {rows:,} rows repeat a value already used, e.g. {example!p}. {advice:skip_key_check()}"#,
        ),
        S::KeyNotUniqueColumns => said(
            r#"the primary key of {table!r}, {columns!l}, is not unique enough to identify a row: {repeats:,} of {rows:,} rows repeat a value already used, e.g. {example!p}. {advice:skip_key_check()}"#,
        ),
        S::KeyNotUniqueUnenforced => said(
            r#"{table}.{column} is not unique enough to identify a row: {repeats:,} of {rows:,} rows repeat a value already used, e.g. {example!p}. {database:backend} does not enforce declared keys. {advice:skip_key_check()}"#,
        ),
        S::KeyMissingValuesColumn => said(
            r#"{table}.{column} has {missing:,} row(s) with no value, in {column!r}, so those rows cannot be identified; fill or drop them before using it as a primary key. {advice:skip_key_check()}"#,
        ),
        S::KeyMissingValuesColumns => said(
            r#"the primary key of {table!r}, {columns!l}, has {missing:,} row(s) with no value in part of it, so those rows cannot be identified; fill or drop them before using them as a primary key. {advice:skip_key_check()}"#,
        ),
        S::KeyNoValueColumn => said(
            r#"{table}.{column} identifies a row, but {missing:,} row(s) have no value for it. Those rows cannot be scored and the model will not see them; fill or drop them, or name a different primary key."#,
        ),
        S::KeyNoValueColumns => said(
            r#"{table}.{columns!d} identifies a row, but {missing:,} row(s) have no value for it. Those rows cannot be scored and the model will not see them; fill or drop them, or name a different primary key."#,
        ),
        S::KeyRepeatsColumn => said(
            r#"{table}.{column} identifies a row {repeats:,} time(s) it has already been used, e.g. {example!p}. Those rows are indistinguishable to the model. Name a different primary key if this column is a foreign key rather than an identifier."#,
        ),
        S::KeyRepeatsColumns => said(
            r#"{table}.{columns!d} identifies a row {repeats:,} time(s) it has already been used, e.g. {example!p}. Those rows are indistinguishable to the model. Name a different primary key if this column is a foreign key rather than an identifier."#,
        ),
        S::KeyDeclined => said(
            r#"{table}.{column} is named like a key and another table names it too, but {repeats:,} of {rows:,} rows repeat a value already used, e.g. {example!p}, so it does not identify a row and {table!r} gets no primary key. Nothing can link to a table without one: every relationship pointing at {table!r} is dropped, so nothing is gathered through it, and a question about another table reaches it only where {table!r} names that table itself. Keep one row per {column!r} in the source, or name a column that does identify a row with {advice:name_primary_key(table)}."#,
        ),
        S::TimeColumnsTakenEarliest => grouped(
            r#"{table}.{column} was taken as the time column because its values start earliest of {among!l}. If another says when a row became true, name it: {advice:name_time_column(table, other)}. If none does, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsTakenUncheckedEmpty => grouped(
            r#"{table}.{column} was taken as the time column although none of the {rows:,} rows read holds a value, because the query over the whole table failed and the rows past them were not checked. If the whole table is like that, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsTakenUncheckedSparse => grouped(
            r#"{table}.{column} was taken as the time column although {missing:,} of the {rows:,} rows read hold no value, because the query over the whole table failed and the rows past them were not checked. If the whole table is like that, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsTakenUncheckedConstantWhereHeld => grouped(
            r#"{table}.{column} was taken as the time column although every one of the {rows:,} rows read that holds a value holds {value!t}, because the query over the whole table failed and the rows past them were not checked. If the whole table is like that, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsTakenUncheckedConstant => grouped(
            r#"{table}.{column} was taken as the time column although every one of the {rows:,} rows read holds {value!t}, because the query over the whole table failed and the rows past them were not checked. If the whole table is like that, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsTakenEditStamp => grouped(
            r#"{table}.{column} was taken as the time column although it looks like when a row was last changed, because the table has no other timestamp and no table links to it, so it reads as a log of changes. If it does not say when a row became true, clear it: {advice:clear_time_column(table)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedEmpty => grouped(
            r#"{table}.{column} was not taken as the time column because no row holds a value. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedSparse => grouped(
            r#"{table}.{column} was not taken as the time column because {missing:,} of its {rows:,} rows hold no value. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedConstantWhereHeld => grouped(
            r#"{table}.{column} was not taken as the time column because every row that holds a value holds {value!t}. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedConstant => grouped(
            r#"{table}.{column} was not taken as the time column because every row holds {value!t}. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedLoadStamp => grouped(
            r#"{table}.{column} was not taken as the time column because it looks like when a row was loaded, not when it began. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedEditStamp => grouped(
            r#"{table}.{column} was not taken as the time column because it looks like when a row was last changed, not when it began. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::TimeColumnsRefusedBirthDate => grouped(
            r#"{table}.{column} was not taken as the time column because it looks like a date of birth, not when a row began. If it does say when a row became true, name it: {advice:name_time_column(table, column)}."#,
            Group::TimeColumnsGuessed,
        ),
        S::ColumnsLeftOut => grouped(r#"{column} ({dtype})"#, Group::ColumnsLeftOut),
        S::TimeValuesUnread => said(
            r#"Could not parse {unread:,} of {rows:,} value(s) of time column '{column}' (for example {examples!T}); they were dropped as {advice:missing_time()}"#,
        ),
        S::DatesReadDayFirst => owned(
            r#"text in {table}.{column} was read as dates in {format} format, which puts the day first"#,
            Owner::HostDates,
        ),
        S::KeyCandidatesTied => said(
            r#"Found multiple potential primary keys in table '{table}': {columns!T}. Please specify the primary key for this table manually."#,
        ),
        S::LinkCannotJoinNotTheKey => said(
            r#"{source_table}.{source_column} refers to {target_table}.{target_column}, but the primary key of {target_table!r} is {key!l}. A relationship joins to a primary key, so none was made from it. Pass {advice:set_primary_key(target_table, target_column)} if {target_column!r} identifies a row of {target_table!r}."#,
        ),
        S::LinkCannotJoinNoKey => said(
            r#"{source_table}.{source_column} refers to {target_table}.{target_column}, but {target_table!r} has no primary key. A relationship joins to a primary key, so none was made from it. Pass {advice:set_primary_key(target_table, target_column)} if {target_column!r} identifies a row of {target_table!r}."#,
        ),
        S::LinkCannotJoinTypes => said(
            r#"{source_table}.{source_column} names the key of {target_table!r} and holds {source_dtype!r} where that key holds {target_dtype!r}, which cannot match, so no relationship was made from it. Store the two as one type if they are the same identifier."#,
        ),
        S::TablesNameEachOther => said(
            r#"{source_table!r} and {target_table!r} name each other, and a relationship is crossed in both directions, so this is two ways over one pair and an aggregation between them is refused for not having a single one. Declare whichever of the two a question is about and leave the other out."#,
        ),
        S::TwoWaysToATable => said(
            r#"{source_table!r} reaches {target_table!r} through {columns!l}, and an aggregation across the two tables has to be resolved to one of them, so a query that crosses this pair is refused until it is. They are all kept because they are all real, and which one a question means is not something this can decide. To keep {kept!r}, drop the others: {advice:keep_one_link(source_table, target_table, others)}."#,
        ),
        S::NoTables => said(r#"{host.source!r} holds no table to describe"#),
        S::SourceNoPart => said(r#"table {table!r} is read from a source named by no part"#),
        S::SourceNameEmpty => {
            said(r#"table {table!r} is read from a source whose name has an empty part"#)
        }
        S::TableNamedTwice => said(r#"table {table!r} is named twice"#),
        S::NoSuchTable => said(r#"Table '{source!d}' does not exist in the {database:place}"#),
        S::NoSuchTableInSource => said(r#"Table '{source!d}' does not exist in the source"#),
        S::ProviderSchemaMismatch => {
            said(r#"the data source answered with columns other than the ones asked for"#)
        }
        S::ValueNotANumber => said(r#"column {column!r} holds a value that is not a number"#),
        S::Undeclared => {
            said(r#"the data source does not take a {request} request, which this call needs"#)
        }
        S::SeedPastRange => said(
            r#"the data source samples with a seed of at most {max_seed:,}, and this call asks for a larger one"#,
        ),
        S::BigIntegerPastType => said(
            r#"column {column!r} holds a value that {admitted_type}, the type it was described as, cannot hold"#,
        ),
        S::LagPastPayload => said(
            r#"the {lag_timesteps:,} past steps lag_timesteps asks for make the request larger than the {limit:,} bytes the endpoint takes; ask for fewer steps"#,
        ),
        S::DateLiteralPacked => said(
            r#"{query!r} is not a query a relational model can answer: {literal} is not a valid instance of time"#,
        ),
        S::LocalZoneUnread => said(r#"the host's local time zone is not one the core can read"#),
        S::ClockPastYears => {
            said(r#"the host's clock is past the years a date literal can be read in"#)
        }
        S::UnknownZone => {
            said(r#"a time zone is not in the host's zone lookup roots or the bundled database"#)
        }
        S::TestOperationRefused => {
            said(r#"a test-only operation was given an input it does not take"#)
        }
        S::FallbackLimit => said(
            r#"the {request} request is past the {limit} the in-memory fallback of this data source declares, {most:,}"#,
        ),
        S::FallbackUnordered => said(
            r#"the in-memory fallback of this data source does not order {field} of a {request} request, a column of a type it does not order"#,
        ),
        S::FallbackTimePast => said(
            r#"the in-memory fallback of this data source cannot read a time in {field} of a {request} request: it is past what microseconds hold"#,
        ),
        S::FallbackTimeType => said(
            r#"the in-memory fallback of this data source reads a time in {field} of a {request} request only from a timestamp with no zone or a date"#,
        ),
        S::FallbackNoColumn => said(
            r#"the in-memory fallback of this data source cannot count the rows of a table with no column"#,
        ),
        S::FallbackSeedTypes => said(
            r#"the in-memory fallback of this data source joins a key only to seeds of its own type, or integers to integers"#,
        ),
        S::FallbackSqlExpression => said(
            r#"the in-memory fallback does not execute caller-authored SQL expressions in {field} of a {request} request"#,
        ),
        S::HostClockNotUtc => said(r#"the host's clock must be a timestamp in UTC"#),
        S::StateVersion => said(
            r#"the request's state is of version {version}, and this client reads version {reads}: materialize the request again with this client to predict it"#,
        ),
        S::StateUnread => said(
            r#"the request's state is not one this client's core wrote: materialize the request again to predict it"#,
        ),
        S::SeedNegative => said(
            r#"random_seed is {given}, and a seed is not negative; pass a number from 0 upwards, or leave it out to let the sampler choose"#,
        ),
        S::SamplerStateVersion => said(
            r#"the sampler state kept from an earlier materialize is of version {version}, which this client's core does not read; it has been dropped, so the call can be made again"#,
        ),
        S::OptionValuesUnread => said(r#"the call's options are not JSON the core reads"#),
        S::CandidatesNotRanked => said(
            r#"candidates are the rows a ranking is over, and {host.query!r} does not rank anything. Ask it without candidates, or rank with "RANK TOP k", which is the question they are for"#,
        ),
        S::RankingStatistic => said(
            r#"prediction_statistic applies only to a regression, and this question is a ranking, ordered by how likely each candidate is; ask it without prediction_statistic"#,
        ),
        S::RankingColumn => said(
            r#"a ranking is asked one candidate at a time, about what an entity reaches in a window, and this one ranks a column instead. Rank a LIST_DISTINCT over a foreign key."#,
        ),
        S::RankingFilter => said(
            r#"the rows to rank are chosen by a filter, and each candidate is asked about with a filter of its own, so the two would have to be combined into one question this cannot write. Rank without the filter, or narrow the rows before building the schema."#,
        ),
        S::CandidatesMissing => said(
            r#"a ranking is over a set of candidates, and this model has no ranking task to choose them with, so they have to be named: {advice:name_candidates(column)}. One call is made per candidate."#,
        ),
        S::CandidatesEmpty => said(r#"candidates is empty, so there is nothing to rank"#),
        S::TooManyCandidates => said(
            r#"{candidates} candidates means {candidates} calls, one for each, because the model has no task that answers them together. That is past the {limit} this will do without being told to; ask about fewer, or {advice:raise_most_parts()} if the wait is what you want."#,
        ),
        S::TooManySteps => said(
            r#"{steps} forecast steps means {steps} calls, one for each, because the model has no task that answers them together. That is past the {limit} this will do without being told to; ask about fewer, or {advice:raise_most_parts()} if the wait is what you want."#,
        ),
        S::RankingOutputsNone => said(r#"outputs must name at least one field"#),
        S::RankingOutputUnknown => said(
            r#"a ranking gives back prediction, rankings, and {unsupported!j} is not among them. The per-candidate questions it is asked as are classifications; ask one of those directly for the fields a classification gives."#,
        ),
        S::RankingOutputsUnknown => said(
            r#"a ranking gives back prediction, rankings, and {unsupported!j} are not among them. The per-candidate questions it is asked as are classifications; ask one of those directly for the fields a classification gives."#,
        ),
        S::RankingUnscoredOne => said(
            r#"the context gathered for candidate {host.candidates!l} holds one answer only, and a classification needs at least two classes to learn from, so there is nothing to rank and nothing was sent. Gather a wider context with num_context_samples if other entities in the data answer differently, or ask over a table of (entity, candidate) pairs, which scores every candidate from its features."#,
        ),
        S::RankingUnscoredSeveral => said(
            r#"the contexts gathered for candidates {host.candidates!l} each hold one answer only, and a classification needs at least two classes to learn from, so there is nothing to rank and nothing was sent. Gather a wider context with num_context_samples if other entities in the data answer differently, or ask over a table of (entity, candidate) pairs, which scores every candidate from its features."#,
        ),
        S::CandidatesUnscoredOne => said(
            r#"candidate {host.candidates!l} is left out of this ranking, unscored: the context sampled for it holds one answer only, nearly always because no entity in it linked to the candidate in the window, and a classification needs two to learn from. No call was sent for it, so each row ranks only the rest, and the result's unscored_candidates names it. Gather a wider context with num_context_samples if other entities in the data answer differently, or ask over a table of (entity, candidate) pairs, which scores every candidate from its features."#,
        ),
        S::CandidatesUnscoredSeveral => said(
            r#"candidates {host.candidates!l} are left out of this ranking, unscored: the context sampled for each holds one answer only, nearly always because no entity in it linked to the candidate in the window, and a classification needs two to learn from. No call was sent for them, so each row ranks only the rest, and the result's unscored_candidates names them. Gather a wider context with num_context_samples if other entities in the data answer differently, or ask over a table of (entity, candidate) pairs, which scores every candidate from its features."#,
        ),
        S::BatchedPartRanking => said(
            r#"{entities:,} entities are more than the {rows_per_call:,} one call scores, and this question is asked as {parts:,} of them, one per candidate. A part is not batched, and a candidate is reported unscored for the whole ranking, so one cold batch would drop it from every row rather than from its own. Ask about fewer entities at a time"#,
        ),
        S::BatchedPartForecast => said(
            r#"{entities:,} entities are more than the {rows_per_call:,} one call scores, and this question is asked as {parts:,} of them, one per step. A part is not batched, and the calls would be the batches times the steps rather than the steps that were counted. Ask about fewer entities at a time"#,
        ),
        S::CandidateMissing => said(r#"a candidate names a row, and a missing value names none"#),
        S::CandidateBool => said(r#"a candidate names a row, and {advice:booleans()} do not"#),
        S::CandidateUnreadable => said(
            r#"a candidate names a row by its key, and a {host.candidate_type} is not one: a key is looked up by value, so it has to be hashable"#,
        ),
        S::CandidateNotExact => said(
            r#"the candidate {host.candidate} names no row of a column of {dtype}: written as one it becomes {host.written}, which is a different row"#,
        ),
        S::CandidateNotWhole => said(r#"{host.candidate!r} cannot name a row to rank"#),
        S::CandidateUnwritable => said(
            r#"the candidate {host.candidate!r} cannot be written into a query: a key holding a quote, a comma, a bracket or nothing at all would read as part of the question rather than as the row it names"#,
        ),
        S::CandidateType => said(
            r#"candidate {host.candidate!r} is a {type}, which this client cannot write into a question yet"#,
        ),
        S::ProbabilitiesUnread => said(
            r#"the endpoint answered the question about candidate {host.candidate!r} without saying how sure it was, and a ranking is only the order of those. It gave back {classes!r}, and this client knows a positive class spelled "true", "True", "TRUE" or "1"."#,
        ),
        S::ProbabilitiesNone => said(
            r#"the endpoint answered the question about candidate {host.candidate!r} without saying how sure it was, and a ranking is only the order of those. It gave back no probabilities, and this client knows a positive class spelled "true", "True", "TRUE" or "1"."#,
        ),
        S::NotYetPredict => said(
            r#"predicting this request needs the {rule!r} rule, which is not one this client's core supports yet"#,
        ),
        S::ResponseUnread => {
            said(r#"the model endpoint answered with something the core cannot read"#)
        }
        S::ExplanationNotAMapping => said(
            r#"the model endpoint answered an item's explanation with something that is not a mapping, where the contract requires an object"#,
        ),
        S::MetadataNotAMapping => said(
            r#"the model endpoint answered an item's metadata with something that is not a mapping, where the contract requires an object"#,
        ),
        S::ModelLacksShape => said(r#"{model} cannot answer a request needing {unsupported!c}"#),
        S::ModelLacksTask => {
            said(r#"{model} does not serve the task kind {task!r}; it serves {supported!r}"#)
        }
        S::ModelLacksOutputs => said(
            r#"{model} does not produce {unsupported_outputs!r}; it produces {supported_outputs!r}"#,
        ),
        S::ContextPastModel => said(
            r#"this request carries {context_rows:,} context rows and {model} reads at most {context_rows_max:,}. A request built directly is sent as it was given, so choose the rows you want as context, or gather it with {advice:materialize_call()}, which keeps the context under the cap and says how"#,
        ),
        S::NoContextRows => said(
            r#"no context rows are left: every row that could be context is also being scored, and a row carrying the answer being asked for is not sent as context. Score fewer rows, so that the rest are there to learn from"#,
        ),
        S::TooManyClasses => said(
            r#"{model} classifies at most {limit} classes, and the gathered context carries {classes} distinct values of the target; predict a coarser target, or narrow the context to the classes you care about"#,
        ),
        S::OneClassColumn => said(
            r#"every context row has the same {target}, and a classification needs at least two classes to learn from. Gather a wider context with num_context_samples, or ask about entities whose answers differ"#,
        ),
        S::OneClassPredicted => said(
            r#"every context row has the same {predicted}, and a classification needs at least two classes to learn from. Gather a wider context with num_context_samples, or ask about entities whose answers differ"#,
        ),
        S::OneClassQuestion => said(
            r#"every context row has the same answer to {question!r}, and a classification needs at least two classes to learn from. Gather a wider context with num_context_samples, or ask about entities whose answers differ"#,
        ),
        S::OneClassAnswer => said(
            r#"every context row has the same answer, and a classification needs at least two classes to learn from. Gather a wider context with num_context_samples, or ask about entities whose answers differ"#,
        ),
        S::ContextConstantColumn => said(
            r#"every context row has the same {target}, so nothing in the context distinguishes one answer from another. The endpoint will answer, but the answer cannot say much. Gather a wider context with num_context_samples, or ask about entities whose answers differ."#,
        ),
        S::ContextConstantPredicted => said(
            r#"every context row has the same {predicted}, so nothing in the context distinguishes one answer from another. The endpoint will answer, but the answer cannot say much. Gather a wider context with num_context_samples, or ask about entities whose answers differ."#,
        ),
        S::ContextConstantQuestion => said(
            r#"every context row has the same answer to {question!r}, so nothing in the context distinguishes one answer from another. The endpoint will answer, but the answer cannot say much. Gather a wider context with num_context_samples, or ask about entities whose answers differ."#,
        ),
        S::ContextConstantAnswer => said(
            r#"every context row has the same answer, so nothing in the context distinguishes one answer from another. The endpoint will answer, but the answer cannot say much. Gather a wider context with num_context_samples, or ask about entities whose answers differ."#,
        ),
        S::OutputsAbsent => said(
            r#"{model} answered without {absent!r}, which this request asked for, so no column of {advice:result_frame()} will carry it. The answer to the rest of the request is unaffected."#,
        ),
        S::StepsUnexplained => said(
            r#"{model} explained {explained} of {steps} forecast steps, and step {missing!c} carries none, so the explanation this request asked for is not there for every step."#,
        ),
        S::ExplanationOneEntity => said(
            r#"an explanation covers exactly 1 entity, and this request scores {scored:,}; ask for one at a time"#,
        ),
        S::ExplanationFastMode => said(
            r#"an explanation is only produced in the mode the smaller context budget selects, and this request was gathered with num_context_samples above {context_rows_fast:,}. Re-materialize with {advice:context_samples(context_rows_fast)} or fewer"#,
        ),
        S::OptionsUnread => said(
            r#"{model} does not read inference option(s) {unsupported!r}; drop them from inference_config to send the request without them. The keys this client forwards are {forwarded!r}, and the endpoint decides for itself whether it serves the value given for one"#,
        ),
        S::OptionsUnreadStatistic => said(
            r#"{model} does not read inference option(s) {unsupported!r}; drop them from inference_config to send the request without them. The keys this client forwards are {forwarded!r}, and the endpoint decides for itself whether it serves the value given for one. A regression's statistic is chosen with {advice:choose_statistic()} instead: 'median', the default, or 'mean'"#,
        ),
        S::PayloadSmallerCalls => said(
            r#"Context size exceeds the 30MB limit. This call is {bytes:,} bytes, and the endpoint takes at most {limit:,}. What comes down is one of three. For an ordinary question it is the call: {advice:smaller_calls()} below the {rows_per_call:,} this one holds asks for the same entities in more and smaller calls. Where the context is itself over the limit, no call is small enough, and it is num_context_samples, num_neighbors or the columns the graph carries. For a ranking or a forecast, which is one call per candidate or step and is never batched, it is the entities."#,
        ),
        S::PayloadNoSmallerCall => said(
            r#"Context size exceeds the 30MB limit. This call is {bytes:,} bytes, and the endpoint takes at most {limit:,}. A call already scores one row, so there is no smaller one to ask for: what comes down is num_context_samples, num_neighbors or the columns the graph carries."#,
        ),
        S::CallPastEndpoint => said(
            r#"the request is {bytes:,} bytes and the endpoint takes at most {limit:,} bytes; reduce the number of tables in the graph, their columns, the neighbourhood sampled or the context gathered (num_context_samples)"#,
        ),
        S::OutputsNone => said(r#"outputs must name at least one field"#),
        S::NotFinite => said(
            r#"encountered a non-finite value ({value}); NIM requests must contain finite numbers or nulls"#,
        ),
        S::PastInt64 => said(
            r#"Column '{table}.{column}' row {row}: a value is outside the range {dtype} represents (-9223372036854775808 to 9223372036854775807); send the column as strings"#,
        ),
        S::ColumnTimePastNanoseconds => said(
            r#"column {column!r} holds a time past what nanoseconds since the epoch hold, the years 1677 to 2262"#,
        ),
        S::LagNegative => said(
            r#"lag_timesteps is how many past steps of the target are added as features, a whole number of zero or more; it cannot be negative"#,
        ),
        S::LagNotWhole => said(
            r#"lag_timesteps is how many past steps of the target are added as features, a whole number of zero or more; a fraction or a boolean is not one"#,
        ),
        S::LagNotANumber => said(
            r#"lag_timesteps is how many past steps of the target are added as features, a whole number of zero or more; text, {advice:nothing()} or a collection is not one"#,
        ),
        S::LagPastTimes => said(
            r#"lag_timesteps reaches further back than a time can be held (before 1677); ask for fewer steps"#,
        ),
        S::NumContextSamplesNotWhole => said(
            r#"num_context_samples must be a whole number of labelled examples, got {given!N} {given!G}"#,
        ),
        S::NumContextSamplesBelowOne => said(
            r#"num_context_samples must be at least 1; a model predicts from labelled examples, and {given!S} leaves it none"#,
        ),
        S::NumContextSamplesPastMost => said(
            r#"num_context_samples is {given!I:,}, and the endpoint accepts at most {limit:,} context rows"#,
        ),
        S::NumHopsNotWhole => said(r#"num_hops must be a whole number, got {given!N} {given!G}"#),
        S::NumHopsNone => said(r#"num_hops must be a whole number, got {advice:nothing()}"#),
        S::RandomSeedNotWhole => said(r#"random_seed must be a whole number"#),
        S::TopKNotWhole => said(r#"top_k must be a whole number of candidates, got {given!N}"#),
        S::TopKOutOfRange => {
            said(r#"top_k is {given!S}, and this model ranks between 1 and {limit} candidates"#)
        }
        S::RowsPerCall => said(
            r#"rows_per_call is not a whole number of at least one, and a call scores a whole number of rows, at least one"#,
        ),
        S::MaxPqIterationsNotAWholeNumber => owned(
            r#"max_pq_iterations must be a whole number, got {given!N}"#,
            Owner::HostComparison,
        ),
        S::IterationsBelowOne => {
            said(r#"'max_pq_iterations' must be greater than zero (got {given!S})"#)
        }
        S::OptionNotWhole => said(r#"{field} must be a whole number, got {given!N} {given!G}"#),
        S::OptionUnknown => {
            said(r#"the call names an option, {field!r}, that materialize does not take"#)
        }
        S::OptionNotABoolean => said(r#"{field} is true or false, and the value given is neither"#),
        S::OptionValueNotFinite => said(
            r#"{field} carries a value that cannot be sent as JSON (a NaN, an infinity, or a type JSON has no form for)"#,
        ),
        S::NeighborsNotASequence => said(r#"num_neighbors must be a sequence of hop sizes"#),
        S::HopNotWhole => said(
            r#"each hop of num_neighbors is a whole number of neighbors, and one given is not"#,
        ),
        S::HopNegative => {
            said(r#"each hop of num_neighbors is a number of neighbors, and one given is negative"#)
        }
        S::NeighborLookbackRange => {
            said(r#"a neighbor lookback is outside the supported time range"#)
        }
        S::NeighborLookbackNotNumeric => {
            said(r#"a neighbor bound used to size a time window is not numeric"#)
        }
        S::ContextAnchorNotATime => {
            said(r#"context_anchor_time is a time or 'entity', and the value given is neither"#)
        }
        S::ExcludedNotAMapping => said(
            r#"exclude_cols_dict must map a table name to the columns to leave out, got {given!N}"#,
        ),
        S::ExcludedNotASequence => said(
            r#"exclude_cols_dict[{table!r}] must be a sequence of column names, got {given!N}"#,
        ),
        S::StatisticUnknown => said(
            r#"prediction_statistic is {given!G}, and a regression reports 'mean' or 'median' of its predicted distribution"#,
        ),
        S::StatisticNotARegression => said(
            r#"prediction_statistic applies only to a regression, and this question is a {task}; ask it without prediction_statistic"#,
        ),
        S::NoHopNumHops => said(
            r#"num_hops is {hops}, which asks for no hop at all, and the endpoint reads between 1 and {limit}. A request that crosses no relationship is a single-table one: give a hop, such as {advice:give_a_hop()}, or ask it of kumo-tabular"#,
        ),
        S::NoHopNumNeighbors => said(
            r#"num_neighbors is empty, which asks for no hop at all, and the endpoint reads between 1 and {limit}. A request that crosses no relationship is a single-table one: give a hop, such as {advice:give_a_hop()}, or ask it of kumo-tabular"#,
        ),
        S::HopsPastEndpoint => {
            said(r#"this request samples {hops} hops and the endpoint reads at most {limit}"#)
        }
        S::HopsPastEndpointUncounted => {
            said(r#"this request samples more hops than the endpoint reads, at most {limit}"#)
        }
        S::InferenceConfigNotAnObject => {
            said(r#"inference_config is a mapping of option names to values"#)
        }
        S::InferenceOptionUnknown => said(
            r#"inference_config holds {key!r}, which is not an inference option of this task: the options are num_estimators, column_shuffle, category_shuffle and hop_shuffle, with class_shuffle for a classification and target_transforms and output_type for a regression"#,
        ),
        S::InferenceOptionNotABoolean => said(
            r#"inference_config's {key!r} is true or false: a boolean, 0 or 1, or text such as 'true', 'yes', 'on' or 'off'"#,
        ),
        S::InferenceOptionNotOneToFour => {
            said(r#"inference_config's {key!r} is a whole number from 1 to 4"#)
        }
        S::InferenceOptionNotTransforms => said(
            r#"inference_config's {key!r} is a list of transforms, each 'clip', 'power', 'quantile' or {advice:nothing()}"#,
        ),
        S::InferenceOptionNotAnOutputType => {
            said(r#"inference_config's {key!r} is 'mean', 'median' or 'quantiles'"#)
        }
        S::SamplerStateUnread => said(
            r#"the sampler state kept from an earlier materialize is not one this client's core left"#,
        ),
        S::TimeConstantOffset => said(
            r#"the query compares with a time that has a UTC offset, which cannot be compared with times that have none"#,
        ),
        S::TimeConstantPast => said(
            r#"the query compares with a time past what a timestamp of nanoseconds holds, the years 1677 to 2262"#,
        ),
        S::TimeColumnNotATime => said(
            r#"{table}.{column} is the time column of {table!r}, and it does not mean a timestamp, which a time column must"#,
        ),
        S::OptionsWithoutValues => said(
            r#"the call names options without their values, which the core reads only with them"#,
        ),
        S::CompositeKeyFloat => said(
            r#"the primary key of {table!r}, {columns!l}, holds a float, which cannot be part of a key of several columns: its text is not reproducible"#,
        ),
        S::CompositeKeyNameLong => said(
            r#"the primary key of {table!r}, {columns!l}, is folded into a column whose name is longer than a source takes"#,
        ),
        S::CompositeKeyNameTaken => said(
            r#"a column of {table!r} holds the name its primary key of several columns, {columns!l}, is folded into"#,
        ),
        S::ReservedName => said(
            r#"{reserved!c} uses a name reserved by the SQL backends: a column may not start with {prefix!r}, because the generated joins already select columns of that name and the reference would be ambiguous. Rename the column, or project it under another name."#,
        ),
        S::ReservedNames => said(
            r#"{reserved!c} use names reserved by the SQL backends: a column may not start with {prefix!r}, because the generated joins already select columns of that name and the reference would be ambiguous. Rename the column, or project it under another name."#,
        ),
        S::SchemaLinkUnread => said(
            r#"a relationship of the schema cannot be sampled: it does not join a foreign key to its table's primary key"#,
        ),
        S::WireReservedOne => said(
            r#"{reserved[0]} uses a name this client needs for the request it builds, so it cannot also be a column of the data. Rename it, for instance to {renamed[0]!r}."#,
        ),
        S::WireReservedSeveral => said(
            r#"{reserved!a} use names this client needs for the request it builds, so they cannot also be columns of the data. Rename them, for instance {columns:to(renamed)}."#,
        ),
        S::IndicesAbsent => said(
            r#"materialize names the entities to predict for with indices, and none were given"#,
        ),
        S::IndicesNotOneColumn => said(
            r#"indices holds the entities to predict for as one column, and the one given is not"#,
        ),
        S::IndicesEmpty => said(
            r#"indices names no entity, and gathering a relational request means sampling a neighbourhood around each one, so there is nothing here to gather. An empty batch is still answered: build it as {advice:empty_request()} and {advice:predict_call()} returns the empty result without sending"#,
        ),
        S::IndexMissing => {
            said(r#"indices holds a missing value, which names no entity to predict for"#)
        }
        S::AnchorEntityNotStatic => {
            said(r#"Anchor time 'entity' is only valid for static predictive queries"#)
        }
        S::AnchorEntityNoTimeColumn => said(
            r#"Anchor time 'entity' requires the entity table '{table}' to have a time column"#,
        ),
        S::AnchorEntityNotShared => {
            said(r#"Anchor time 'entity' needs to be shared for context and prediction examples"#)
        }
        S::IterationsWholeForWindow => {
            said(r#"max_pq_iterations must be a whole number for a query over a window"#)
        }
        S::EntityTableEmpty => said(r#"the entity table {table!r} holds no row to sample"#),
        S::EntityKeyMissing => said(
            r#"an entity sampled from {table!r} has no value for its primary key, {column!r}, so it cannot be told apart from the others"#,
        ),
        S::TargetNotAFeature => {
            said(r#"the column the query predicts is not one of the features of the entity table"#)
        }
        S::ContextPastRows => said(r#"the context holds more rows than a request takes"#),
        S::HeldBySeveralRowsOne => said(
            r#"Cannot score {keys[0]!r}: more than one row of table '{table}' holds this value of '{key}', so which row is meant would be a guess. The source does not enforce the key; keep one row per value, or key the table on columns that identify a row."#,
        ),
        S::HeldBySeveralRows => said(
            r#"Cannot score {keys!j}: more than one row of table '{table}' holds each of these values of '{key}', so which row is meant would be a guess. The source does not enforce the key; keep one row per value, or key the table on columns that identify a row."#,
        ),
        S::HeldBySeveralRowsMany => said(
            r#"Cannot score {shown!j} and {more:,} more: more than one row of table '{table}' holds each of these values of '{key}', so which row is meant would be a guess. The source does not enforce the key; keep one row per value, or key the table on columns that identify a row."#,
        ),
        S::NotInTable => said(r#""The primary keys {keys!r} do not exist in the '{table}' table""#),
        S::SubgraphPastTables => said(
            r#"the request's subgraph carries {tables:,} tables, and the endpoint reads at most {limit:,}"#,
        ),
        S::Unresolved => {
            said(r#"{query!r} is not a query a relational model can answer: {reasons!R}"#)
        }
        S::AnchorNotATime => said(r#"anchor_time is a time, and the value given is not one"#),
        S::AnchorBeforeData => said(
            r#"Anchor timestamp '{anchor!t}' is before the earliest timestamp '{earliest!t}' in the data."#,
        ),
        S::ContextBeforeDataRefused => said(
            r#"Context anchor timestamp is too early or aggregation time range is too large. To make this prediction, we would need data back to '{context_anchor!t}', however, your data only contains data back to '{earliest!t}'."#,
        ),
        S::ExcludedUnknownTable => said(
            r#"'exclude_cols_dict' names table '{table}', which is not in the graph. Available tables: {tables!r}"#,
        ),
        S::ExcludedNotAFeature => said(
            r#"'exclude_cols_dict' names column '{given!S}' of table '{table}', which is not one of its feature columns (primary and foreign keys are never sent as features). Excludable columns: {columns!r}"#,
        ),
        S::NullComparedNowhere => said(
            r#"the query compares an aggregation with NULL, and no window of a context example holds a value to compare"#,
        ),
        S::NoUniqueForeignKey => said(
            r#"no single foreign key leads from a table the query aggregates to the entity table, so the query cannot be gathered"#,
        ),
        S::ContextTooRestrictive => said(
            r#"Failed to collect any context examples. Is your predictive query too restrictive?"#,
        ),
        S::NoEntityAtAnchor => said(
            r#"Failed to collect any context examples. No row of {entity_table!r} existed at the anchor: it is {anchor!t}, and the earliest {column!r} is {earliest!t}. Context examples only ask about an entity that existed by its anchor. If {column!r} does not record when each entity began, set the time column of {entity_table!r} to {advice:nothing()}."#,
        ),
        S::NoEntityAtAnchorLater => said(
            r#"Failed to collect any context examples. No row of {entity_table!r} existed at the anchor: it is {anchor!t}, and the earliest {column!r} is {earliest!t}. Context examples only ask about an entity that existed by its anchor. If {column!r} does not record when each entity began, set the time column of {entity_table!r} to {advice:nothing()}, or pass an anchor_time after {earliest!t}."#,
        ),
        S::NoEntityPassed => said(
            r#"Failed to collect any context examples. The anchor time is not the reason: it is {anchor!t}, and {tables!j} holds data from {min_time!t} to {max_time!t}, which leaves room after it for the query's whole window. No entity passed {condition!r}, so there was nothing to sample: check that the values it compares against occur in the data, including their case."#,
        ),
        S::AnchorTooLate => said(
            r#"Failed to collect any context examples. The anchor time is most likely the reason: it is {anchor!t}, and {tables!j} holds data from {min_time!t} to {max_time!t}. Context examples come from before the anchor and each needs the query's full window of data after it, so an anchor at or near the end of the range leaves none to draw from. Pass an earlier anchor_time, or shorten the window the query asks about."#,
        ),
        S::NoEntityPassedSeveral => said(
            r#"Failed to collect any context examples. The anchor time is not the reason: it is {anchor!t}, and {tables!j} hold data from {min_time!t} to {max_time!t}, which leaves room after it for the query's whole window. No entity passed {condition!r}, so there was nothing to sample: check that the values it compares against occur in the data, including their case."#,
        ),
        S::AnchorTooLateSeveral => said(
            r#"Failed to collect any context examples. The anchor time is most likely the reason: it is {anchor!t}, and {tables!j} hold data from {min_time!t} to {max_time!t}. Context examples come from before the anchor and each needs the query's full window of data after it, so an anchor at or near the end of the range leaves none to draw from. Pass an earlier anchor_time, or shorten the window the query asks about."#,
        ),
        S::ForecastTopK => said(
            r#"top_k is how many candidates a ranking keeps, and a forecast ranks nothing; it gives back one value per step"#,
        ),
        S::MostEntities => said(
            r#"{entities:,} entities is past the {limit:,} one {advice:materialize_call()} will hold at once, since every batch is built, with its own context, before any is sent; ask about fewer entities at a time"#,
        ),
        S::MostEntitiesParts => said(
            r#"{entities:,} entities gathered {gathered_times:,} times over, {total:,} in all, is past the {limit:,} one {advice:materialize_call()} will hold at once, since every batch is built, with its own context, before any is sent; ask about fewer entities at a time"#,
        ),
        S::MostCalls => said(
            r#"{entities:,} entities are gathered as {calls:,} calls of {rows_per_call:,}, and every one is built, with its own context, before any is sent. That is past the {limit:,} calls this will hold at once; ask about fewer entities at a time"#,
        ),
        S::NotYetMaterialize => said(
            r#"materializing this request needs the {rule!r} rule, which is not one this client's core supports yet"#,
        ),
        S::MissingIndexTable => said(
            r#"Missing {indices} index in {tables} table for optimal database querying. For improving runtime, we strongly suggest to create indices for primary and foreign keys, e.g., automatically by instantiating Kumo Relational via `{advice:optimize_indices()}`."#,
        ),
        S::MissingIndexTables => said(
            r#"Missing {indices} index in {tables} tables for optimal database querying. For improving runtime, we strongly suggest to create indices for primary and foreign keys, e.g., automatically by instantiating Kumo Relational via `{advice:optimize_indices()}`."#,
        ),
        S::MissingIndicesTable => said(
            r#"Missing {indices} indices in {tables} table for optimal database querying. For improving runtime, we strongly suggest to create indices for primary and foreign keys, e.g., automatically by instantiating Kumo Relational via `{advice:optimize_indices()}`."#,
        ),
        S::MissingIndicesTables => said(
            r#"Missing {indices} indices in {tables} tables for optimal database querying. For improving runtime, we strongly suggest to create indices for primary and foreign keys, e.g., automatically by instantiating Kumo Relational via `{advice:optimize_indices()}`."#,
        ),
        S::ContextBeforeData => said(
            r#"Context anchor timestamp is too early or aggregation time range is too large. To form proper input data, we would need data back to '{back_to!t}', however, your data only contains data back to '{earliest!t}'."#,
        ),
        S::ContextBeforeDataUntimed => said(
            r#"Context anchor timestamp is too early or aggregation time range is too large: the context reaches back further than a time can be held."#,
        ),
        S::AnchorAfterData => said(
            r#"Anchor timestamp '{anchor!t}' is after the latest timestamp '{latest!t}' in the data. Please make sure this is intended."#,
        ),
        S::Float32Overflow => owned(
            r#"a value past the range of a 32-bit float was cast to one, as an infinity"#,
            Owner::HostCast,
        ),
        S::ContextAfterAnchor => said(
            r#"Context anchor timestamp (got '{context_anchor!t}') is set to a later date than the prediction anchor timestamp (got '{anchor!t}'). Please make sure this is intended."#,
        ),
        S::ContextLeaks => said(
            r#"Aggregation for context examples at timestamp '{context_anchor!t}' will leak information from the prediction anchor timestamp '{anchor!t}'. Please make sure this is intended."#,
        ),
        S::HopsIgnored => said(
            r#"Received custom 'num_neighbors' option; ignoring custom 'num_hops={num_hops}' option"#,
        ),
        S::ContextCoverage => said(
            r#"Failed to collect {examples:,} context examples within {candidates:,} candidates. To improve coverage, consider increasing the number of PQ iterations using the 'max_pq_iterations' option. This warning will not be shown again in this run."#,
        ),
        S::ContextCapped => said(
            r#"{cap:,} context examples were gathered, which is all this route takes by default; pass num_context_samples to ask for up to {most:,}."#,
        ),
        S::TablesUnlinkedOne => said(
            r#"{tables!j} is in the schema and not in this request: no relationship connects it to {entity_table!r}, so the model answers without it. Say how it joins: {advice:append_relationship(tables)}, or leave it out of the schema to say it is not wanted."#,
        ),
        S::TablesUnreachedOne => said(
            r#"{tables!j} is linked into the schema and not in this request: the sampler reached no row of it from the entities being scored, so the model answers without it. Raise num_hops if it lies further out than the hops sampled, and check the foreign key holds values the parent key actually has if it does not."#,
        ),
        S::ColumnsUnreadOne => said(
            r#"{columns!c} is typed text, which this model does not read, so the request is answered without it and it contributes nothing to the prediction. Type a column categorical if its values are labels rather than prose, or leave it out to say so."#,
        ),
        S::TablesUnlinkedSeveral => said(
            r#"{tables!j} are in the schema and not in this request: no relationship connects them to {entity_table!r}, so the model answers without them. Say how they join: {advice:append_relationship(tables)}, or leave them out of the schema to say they are not wanted."#,
        ),
        S::TablesUnreachedSeveral => said(
            r#"{tables!j} are linked into the schema and not in this request: the sampler reached no row of them from the entities being scored, so the model answers without them. Raise num_hops if they lie further out than the hops sampled, and check the foreign key holds values the parent key actually has if they do not."#,
        ),
        S::ColumnsUnreadSeveral => said(
            r#"{columns!c} are typed text, which this model does not read, so the request is answered without them and they contribute nothing to the prediction. Type a column categorical if its values are labels rather than prose, or leave it out to say so."#,
        ),
        S::FilterChoseExamples => said(
            r#"the entity filter {condition!r} selects the labelled examples the model learns from, and does not narrow the {entities} rows named for scoring. Those are scored as given, whether or not they satisfy it. Pass only indices that satisfy the condition if that is what you meant."#,
        ),
        S::ContextAnswersWindowed => said(
            r#"{examples} of the rows sampled as context are the rows being scored, at the same moment, carrying the value being asked for. Rows that carry the answer to the question being asked are not sent as context, so these do not go into it."#,
        ),
        S::ContextAnswersStatic => said(
            r#"{examples} of the rows sampled as context are the rows being scored, at one moment or another, carrying the value being asked for. Rows that carry the answer to the question being asked are not sent as context, so these do not go into it."#,
        ),
        S::NullKeysLeftOut => said(
            r#"{table}.{column} has no value in {rows:,} of the rows sampled, beside rows that do, so those rows were left out of the request: a row with no key cannot be told apart. Fill or drop them, or name a different primary key."#,
        ),
        S::DateLiteralDayFirst => owned(
            r#"a date literal of the query was read in {format} format, which puts the day first"#,
            Owner::HostDates,
        ),
        S::InvalidStringEscape => said("invalid escape sequence '{escape}'"),
        S::EntityUncast => said(
            r#"the entity at position {position} of indices cannot name a row of a key of {key_type}"#,
        ),
        S::TimePastNanoseconds => said(
            r#"{field} reaches a time past what nanoseconds since the epoch hold, the years 1677 to 2262"#,
        ),
        S::TimePastUnit => said(r#"{field} reaches a time past what its unit holds"#),
        S::InListFraction => said(
            r#"an IN list of the query compares an integer column with a number that has a fraction, which no value of the column equals"#,
        ),
        S::RenderingRefused => said(
            r#"the data source's SQL cannot render the {field} of a {request} request this way"#,
        ),
        S::NoDialect => said(r#"this client has no built-in SQL for the data source asked for"#),
        S::NoDialectNamed => said(r#"this client has no built-in SQL for a {dialect} data source"#),
        S::ValueNotJson => said(r#"column {column!r} holds a value that is not JSON"#),
        S::ValueNotAtScale => {
            said(r#"column {column!r} holds a value that is not at its declared scale"#)
        }
        S::TextNotUtf8 => said(
            r#"column {column!r} holds a JSON string with a lone surrogate, text with no UTF-8 form, which the core cannot read"#,
        ),
        S::UnsupportedTypeUnread => said(
            r#"column {column!r} holds a value of the Arrow type {arrow_type} that the core cannot read"#,
        ),
        S::AdmittedPastType => said(
            r#"column {column!r} holds a value of the Arrow type {arrow_type} that {admitted_type}, the type the core takes it as, cannot hold"#,
        ),
        S::InvalidConfiguration => {
            said(r#"the configuration of this call is not one the core accepts"#)
        }
        S::PartForecastStep => {
            said(r#"This is forecast step {forecast_step} of {forecast_steps}."#)
        }
        S::PartForecastStepOutran => said(
            r#"This is forecast step {forecast_step} of {forecast_steps}. The window named above is that step's rather than the one written; ask for fewer than {forecast_step} steps if the data does not reach that far."#,
        ),
        S::PartRankingCandidate => said(
            r#"This is the question asked for candidate {host.part_candidate!r}, one of {ranking_parts} this ranking is asked as."#,
        ),
        // A semantic model's conversion (T3.27, `old-behaviour/T3.5-views.md`),
        // in the old client's words. A `{reason}`, `{lost}`, `{reasons}` or
        // `{failure}` is a clause the core writes from the old client's
        // words; a quoted value of the definition (`{source}`, `{join}`,
        // `{dimension}`, `{filter}`) is the text the old client quoted it
        // by, Python's, which the core writes.
        S::ViewsUnread => {
            said(r#"the core asked the data source for a semantic model, which it does not read"#)
        }
        S::ViewNameInvalid => said(r#"Invalid metric view name '{name}'"#),
        S::ViewNotFound => said(r#"Metric view '{view}' does not exist or you may not see it"#),
        S::ViewNotAMetricView => said(r#"'{view}' is not a metric view (got type '{got}')"#),
        S::ViewNoDefinition => said(r#"Could not read the definition of metric view '{view}'"#),
        S::ViewNotYaml => said(r#"The {view_kind} definition is not readable YAML: {reason}"#),
        S::ViewNotAMapping => said(r#"The {view_kind} definition is not a YAML mapping"#),
        S::ViewMissesSource => said(r#"The {view_kind} definition misses a 'source'"#),
        S::ViewMissesEntry => said(r#"The {view_kind} definition misses '{entry}'"#),
        S::ViewNotAList => said(r#"The {view_kind} definition's '{section}' is not a list"#),
        S::ViewNestsItself => said(r#"The {view_kind} definition has a join nested in itself"#),
        S::ViewTablesAbsent => {
            said(r#"{absent!r} are not tables of {host.source!r}; it has {offered!r}"#)
        }
        S::ViewEntryUnhashable => {
            said(r#"The {view_kind} definition's '{entry}' is a list or a mapping"#)
        }
        S::ViewEntryNotText => said(r#"The {view_kind} definition's '{entry}' is not text"#),
        S::ViewValueTooLarge => said(
            r#"The {view_kind} definition holds a value too large to write in a message or a statement"#,
        ),
        S::ViewSourceUnsupported => said(
            r#"Unsupported metric view source {source}. Only plain table names are supported as the source of a metric view when converting it into a graph. Create a regular view with this query and point the metric view's source at it"#,
        ),
        S::ViewSourceUnqualified => {
            said(r#"Could not fully qualify table '{table}' referenced by metric view '{view}'"#)
        }
        S::ViewOnMetricView => said(
            r#"'{table}' is a metric view; a metric view built on another metric view is not supported, point its source at the underlying table"#,
        ),
        S::ViewFilterRefused => said(
            r#"Cannot apply the metric view filter {filter} since {reason}. The graph would otherwise read rows the view leaves out. Create a regular view with this WHERE clause and point the metric view's source at it"#,
        ),
        S::ViewFilterRefusedNoDefault => said(
            r#"Cannot apply the metric view filter {filter} since {reason}. The graph would otherwise read rows the view leaves out. Create a regular view with this WHERE clause and point the metric view's source at it, or give the parameter a default of its type"#,
        ),
        S::ViewFilterRefusedUntrusted => said(
            r#"Cannot apply the metric view filter {filter} since {reason}. The graph would otherwise read rows the view leaves out. Create a regular view with this WHERE clause and point the metric view's source at it, or pass {advice:trust_expressions()} to apply it if you trust whoever authored the view"#,
        ),
        S::ViewJoinNameCollides => said(
            r#"Cannot convert metric view '{view}' into a graph since the join name '{table}' collides with the name of the source table"#,
        ),
        S::ViewTableCannotHold => said(
            r#"Cannot convert semantic view '{view}' into a graph since its table '{table}' cannot hold what it declares: {reason}"#,
        ),
        S::ViewInvalid => {
            said(r#"Cannot convert {view_kind} '{view}' into a graph since {reason}"#)
        }
        S::ViewSemanticNotFound => {
            said(r#"Semantic view '{view}' does not exist or you may not read it"#)
        }
        S::ViewSemanticDefinitionUnread => {
            said(r#"Could not read the definition of semantic view '{view}'"#)
        }
        S::ViewTableOmitted => said(
            r#"{host.semantic_model!r} no longer gives the table '{table}' this schema holds: {reason}"#,
        ),
        S::ViewColumnOmitted => said(
            r#"{host.semantic_model!r} no longer gives the column '{column}' of table '{table}' this schema holds: {reason}"#,
        ),
        S::ViewMaterializationIgnored => grouped(
            r#"Ignored the 'materialization' section since it cannot be represented in a graph"#,
            Group::ViewReadInPart,
        ),
        S::ViewFieldsIgnored => grouped(
            r#"Ignored the 'fields' section since the view also has 'dimensions', and {database:backend} reads only those"#,
            Group::ViewReadInPart,
        ),
        S::ViewJoinNotAMapping => grouped(
            r#"Failed to add join {join} since it is not a mapping"#,
            Group::ViewReadInPart,
        ),
        S::ViewJoinNameless => grouped(
            r#"Failed to add join {join} since it misses a name"#,
            Group::ViewReadInPart,
        ),
        S::ViewJoinSkipped => grouped(
            r#"Failed to add join '{join}' since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewJoinSkippedWithIt => grouped(
            r#"Failed to add join '{join}', and with it {lost}, since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewDimensionNoTable => grouped(
            r#"Failed to add dimension '{dimension}' since it references no table of the view"#,
            Group::ViewReadInPart,
        ),
        S::ViewDimensionNotAMapping => grouped(
            r#"Failed to add dimension {dimension} since it is not a mapping"#,
            Group::ViewReadInPart,
        ),
        S::ViewDimensionMisses => grouped(
            r#"Failed to add dimension {dimension} since it misses a name or expression"#,
            Group::ViewReadInPart,
        ),
        S::ViewDimensionRefused => grouped(
            r#"Failed to add dimension '{dimension}' since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewDimensionMultipleTables => grouped(
            r#"Failed to add dimension '{dimension}' since its expression references multiple tables"#,
            Group::ViewReadInPart,
        ),
        S::ViewMeasureRefused => grouped(
            r#"Failed to add measure '{measure}' since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewMeasureColumnShadowed => grouped(
            r#"Did not add the column '{column}', which the measure '{measure}' reads, since the dimension '{dimension}' has its name"#,
            Group::ViewReadInPart,
        ),
        S::ViewColumnRefused => grouped(
            r#"Failed to add column '{column}' to table '{table}': {failure}"#,
            Group::ViewReadInPart,
        ),
        S::ViewJoinUnchecked => grouped(
            r#"Kept join '{join}' with its key '{key}' unchecked, since counting the keys failed: {failure}"#,
            Group::ViewReadInPart,
        ),
        S::ViewDerivedKeyReplaced => grouped(
            r#"Replaced the derived column '{key}' of table '{table}' with its physical source column since join '{join}' references it as a key"#,
            Group::ViewReadInPart,
        ),
        S::ViewEdgeDropped => grouped(
            r#"Failed to add the relationship between '{source_table}' and '{target_table}' since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewCompositeKey => grouped(
            r#"Failed to add primary key for table '{table}' since composite primary keys are not yet supported"#,
            Group::ViewReadInPart,
        ),
        S::ViewColumnCrossTable => grouped(
            r#"Failed to add column '{column}' of table '{table}' since its expression references other tables"#,
            Group::ViewReadInPart,
        ),
        S::ViewColumnsCrossTable => grouped(
            r#"Failed to add columns '{columns!r}' of table '{table}' since their expressions reference other tables"#,
            Group::ViewReadInPart,
        ),
        S::ViewExpressionDropped => grouped(
            r#"Failed to add column '{column}' since {reason}, and expressions taken from the metric view are executed as SQL under your warehouse credentials"#,
            Group::ViewReadInPart,
        ),
        S::ViewExpressionDroppedUntrusted => grouped(
            r#"Failed to add column '{column}' since {reason}, and expressions taken from the metric view are executed as SQL under your warehouse credentials. Pass {advice:trust_expressions()} to keep it if you trust whoever authored the view"#,
            Group::ViewReadInPart,
        ),
        S::ViewTableExpressionDropped => grouped(
            r#"Failed to add column '{column}' of table '{table}' since {reason}, and expressions taken from the semantic view are executed as SQL under your warehouse credentials"#,
            Group::ViewReadInPart,
        ),
        S::ViewTableExpressionDroppedUntrusted => grouped(
            r#"Failed to add column '{column}' of table '{table}' since {reason}, and expressions taken from the semantic view are executed as SQL under your warehouse credentials. Pass {advice:trust_expressions()} to keep it if you trust whoever authored the view"#,
            Group::ViewReadInPart,
        ),
        S::ViewRelationshipRefused => grouped(
            r#"Failed to add relationship '{relationship}' since {reason}"#,
            Group::ViewReadInPart,
        ),
        S::ViewMeasureNote => said(
            r#"'{measure}' is {function}({table}.{column}, 0, 30, days) over the next 30 days"#,
        ),
        S::ViewTimeColumnNote => {
            said(r#"Took '{column}' as the time column of table '{table}', since {reasons}"#)
        }
        S::ViewTimeColumnNoteAdded => said(
            r#"Took '{column}' as the time column of table '{table}', and added it as a column, since {reasons}"#,
        ),
        S::ViewTimeColumnNoteRenamed => said(
            r#"Took '{column}' as the time column of table '{table}', since {reasons}. It is the source column '{source}', added under a name of its own because a dimension has its name"#,
        ),
        S::ViewParameterNote => said(
            r#"Read the parameter '{parameter}' as its default '{default}', as a query of the view without arguments does"#,
        ),
        // Snowflake (T3.27, `old-behaviour/T3.6-snowflake.md`).
        S::NoCurrentSchema => said(
            r#"No current 'schema' set. Please specify the {database:backend} schema manually"#,
        ),
        S::NoCurrentDatabase => said(
            r#"No current 'database' set. Please specify the {database:backend} database manually"#,
        ),
        S::NullAnchor => said(
            r#"an entity's anchor time is missing, and a window of the {database:backend} source cannot end at no time; give every entity an anchor time"#,
        ),
        S::UnspecifiedSchema => {
            said(r#"Unspecified 'schema' for table '{table}' in database '{in_database}'"#)
        }
        S::TableNoColumns => said(r#"Table '{table}' has no columns"#),
        S::SeedIgnored => said(
            r#"The '{database}' backend does not support seeded random sampling, so 'random_seed' is ignored when drawing in-context examples. Repeated calls with the same seed may return different predictions."#,
        ),
        // A Databricks Model Serving endpoint (T3.27, for T3.13,
        // `CL/core/databricks_serving.py`).
        S::ServingValueNotJson => {
            said(r#"this request carries a value that cannot be sent as JSON: {reason}"#)
        }
        S::ServingPastLimit => said(
            r#"this request is {bytes:,} bytes once wrapped for serving endpoint {endpoint!r}, above the {limit:,} byte limit it accepts. Send fewer rows per call: a smaller num_context_samples, or fewer entities to score"#,
        ),
        S::ServingNotJson => {
            said(r#"serving endpoint {endpoint!r} answered with a response that is not JSON"#)
        }
        S::ServingNotAnObject => {
            said(r#"expected a JSON object from serving endpoint {endpoint!r}, got {got}"#)
        }
        S::ServingRefused => {
            said(r#"serving endpoint {endpoint!r} refused the request ({endpoint_code})"#)
        }
        S::ServingCallersFault => {
            said(r#"serving endpoint {endpoint!r} refused the request (INVALID_REQUEST)"#)
        }
        S::ServingPredictionRowCount => {
            said(r#"serving endpoint {endpoint!r} did not return exactly one prediction row"#)
        }
        S::ServingResponseColumnMissing => said(
            r#"serving endpoint {endpoint!r} answered with a row that has no 'response_json' string column"#,
        ),
        S::ServingFailed => said(r#"serving endpoint {endpoint!r} request failed"#),
        S::ServingFailedCoded => {
            said(r#"serving endpoint {endpoint!r} request failed ({provider_code})"#)
        }
    }
}

/// One sentence of the table a host reads.
#[derive(Serialize)]
struct SentenceEntry {
    name: &'static str,
    kind: &'static str,
    code: Option<&'static str>,
    rule: Option<&'static str>,
    template: &'static str,
    owner: Owner,
    group: Option<Group>,
}

#[derive(Serialize)]
struct GroupEntry {
    name: Group,
    #[serde(flatten)]
    grouping: Grouping,
}

#[derive(Serialize)]
struct AdviceEntry {
    name: &'static str,
    arguments: &'static [&'static str],
}

#[derive(Serialize)]
struct Table<'a> {
    sentences: Vec<SentenceEntry>,
    groups: Vec<GroupEntry>,
    advice: Vec<AdviceEntry>,
    databases: &'a [DatabaseNames],
}

/// The whole table, as the JSON a host reads once: every sentence with its
/// code, rule, template, owner and group, the groups and the advice. Data
/// only: nothing is formatted.
pub fn table_json() -> String {
    table_json_naming(&[])
}

/// [`table_json`], naming `databases` as the data layer names them.
pub fn table_json_naming(databases: &[DatabaseNames]) -> String {
    let table = Table {
        sentences: Sentence::ALL
            .iter()
            .map(|&sentence| {
                let template = template(sentence);
                let kind = match sentence.code() {
                    Code::Error(_) => "error",
                    Code::Warning(_) => "warning",
                    Code::Part => "part",
                    Code::Note => "note",
                };
                SentenceEntry {
                    name: sentence.as_str(),
                    kind,
                    code: sentence.code().as_str(),
                    rule: sentence.rule(),
                    template: template.text,
                    owner: template.owner,
                    group: template.group,
                }
            })
            .collect(),
        groups: Group::ALL
            .iter()
            .map(|&name| GroupEntry {
                name,
                grouping: grouping(name),
            })
            .collect(),
        advice: Advice::ALL
            .iter()
            .map(|&advice| AdviceEntry {
                name: advice.as_str(),
                arguments: advice.arguments(),
            })
            .collect(),
        databases,
    };
    serde_json::to_string(&table).expect("the table serializes")
}
