// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data requests, provider capabilities and the data-layer port.
//!
//! The core asks for data through a small, fixed vocabulary of typed
//! requests. Every provider answers `describe` and `scan`; the optional
//! requests make common steps faster, and a provider declares which it
//! supports. A key set never travels as JSON: it is an Arrow attachment the
//! request names by id (decision Q1).
//!
//! The port (decision H14): a [`DataLayer`] plans each request for the
//! run's provider, as SQL statements in a dialect it renders or as the
//! typed request passed to a host provider, with the Arrow schema the plan
//! returns ([`FetchPlan`]), and reads each `describe` answer the provider
//! returned into the catalog's answer, in no dialect's shape
//! ([`crate::catalog`], decision E54). The core holds one data layer, in
//! its composition root, and names no dialect; `sdm-data` implements it.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::catalog::CatalogAnswer;
use crate::error::{CoreError, Detail, ErrorKind, Sentence};
use crate::scalar::Scalar;
use crate::types::{ColumnType, Field, Unit};
use crate::view::{ViewTextAnswer, ViewTextAsk};

/// The column of a seed key set that holds each seed's key (decision H15).
pub const SEED_KEY: &str = "key";
/// The field metadata key (and its value, [`BIG_INTEGER_MARKED`]) the host
/// layer sets on a big-integer column it read from an opaque extension type
/// (a HUGEINT, UHUGEINT or VARINT, register P1B-43), so the core holds the
/// column to one tier and keeps its declared type.
pub const BIG_INTEGER_MARKER: &str = "sdm:big_integer";
pub const BIG_INTEGER_MARKED: &str = "true";
/// The field metadata key (and its value, [`EXACT_NUMBER_MARKED`]) a host
/// provider sets on a column of numbers it hands over as their exact
/// decimal text because their declaration fixes no Arrow decimal (a
/// Postgres `numeric` without a precision): the core reads such a column
/// by its role in the schema, a key (the one column of a declared primary,
/// unique or foreign key, or a key or link of the schema as it stands) as
/// that text and any other as a double (`DescribedColumn::exact_number`),
/// and holds every later fetch of it to the type described.
pub const EXACT_NUMBER_MARKER: &str = "sdm:exact_number";
pub const EXACT_NUMBER_MARKED: &str = "true";
/// The field metadata key a host provider sets on a column of numbers it
/// hands over as their exact decimal text because their declaration is
/// past the digits an Arrow decimal holds (a Postgres `numeric` declared
/// with more than 76): its value is the declared scale, in decimal digits.
/// The data layer reads such a column by its declaration, as the reference
/// reads it (`DataLayer::conform`), and the core never sees the mark.
pub const DECLARED_SCALE_MARKER: &str = "sdm:declared_scale";

/// The magnitude past which an exponent is read as this bound: far past
/// every decimal's 38 places and every float's range, and far within
/// `i64`, so no arithmetic on a scale overflows (T1.32a fix round 1).
const EXPONENT_BOUND: i64 = 1 << 40;

/// A decimal number as written: its sign, its significant digits (no
/// leading or trailing zeros; empty for zero) and the scale they stand at.
pub struct WrittenNumber {
    pub negative: bool,
    pub digits: String,
    pub scale: i64,
}

/// Text written as a decimal number: an optional sign, ASCII digits with
/// an optional point and fraction (at least one digit in all), and an
/// optional exponent (`e` or `E`, an optional sign, digits); `None` for
/// any other text.
pub fn written_number(text: &str) -> Option<WrittenNumber> {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (&text[..at], Some(&text[at + 1..])),
        None => (text, None),
    };
    let (negative, unsigned) = match mantissa.as_bytes().first() {
        Some(b'-') => (true, &mantissa[1..]),
        Some(b'+') => (false, &mantissa[1..]),
        _ => (false, mantissa),
    };
    let (whole, fraction) = match unsigned.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (unsigned, ""),
    };
    let ascii = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
    if whole.len() + fraction.len() == 0 || !ascii(whole) || !ascii(fraction) {
        return None;
    }
    let exponent: i64 = match exponent {
        None => 0,
        Some(written) => {
            let (sign, digits) = match written.as_bytes().first() {
                Some(b'-') => (-1, &written[1..]),
                Some(b'+') => (1, &written[1..]),
                _ => (1, written),
            };
            if digits.is_empty() || !ascii(digits) {
                return None;
            }
            // An exponent past any decimal's or float's reach still reads:
            // it decides only whether the value is out of range or inexact.
            let magnitude = digits.trim_start_matches('0');
            let magnitude = if magnitude.len() > 13 {
                EXPONENT_BOUND
            } else {
                magnitude.parse::<i64>().unwrap_or(0).min(EXPONENT_BOUND)
            };
            sign * magnitude
        }
    };
    // The digits as one run, leading zeros dropped, then trailing zeros,
    // which carry no value, each moving the scale by one. Lengths are of
    // text, far within i64.
    let all = format!("{whole}{fraction}");
    let significant = all.trim_start_matches('0');
    let digits = significant.trim_end_matches('0');
    let length = |part: &str| i64::try_from(part.len()).unwrap_or(EXPONENT_BOUND);
    let dropped = length(significant) - length(digits);
    Some(WrittenNumber {
        negative,
        digits: digits.to_owned(),
        scale: length(fraction) - dropped - exponent,
    })
}

/// The one text of the number `text` writes, the same for every spelling
/// of it (`1.0`, `1.00`, `1E0`): `0`, or its sign, its significant digits
/// and the power of ten they stand at (`1e0`, `-125e-2`); text that writes
/// no number (a `NaN`, an `Infinity`) is itself. A column marked
/// [`EXACT_NUMBER_MARKER`] holds two keys the same exactly when their
/// texts are, as the source compares them, with nothing passed through a
/// float.
pub fn exact_number_text(text: &str) -> String {
    match written_number(text) {
        None => text.to_owned(),
        Some(written) if written.digits.is_empty() => "0".to_owned(),
        Some(written) => format!(
            "{}{}e{}",
            if written.negative { "-" } else { "" },
            written.digits,
            -written.scale
        ),
    }
}
/// The metadata key, on a seed key set's [`SEED_KEY`] column, naming the
/// joined table's key column's declared type as the source spells it (its
/// catalog's `type`, kept in the described schema for a marked big-integer
/// column, `DescribedColumn::declared_type`), set by the core for a
/// `decimal128(38, 0)` seed key: a SQL layer joins the seeds in the key's
/// own type where that is exact (register P1B-43); a host provider may
/// ignore it.
pub const SEED_KEY_TYPE: &str = "sdm:key_declared_type";
/// The column of a seed key set that holds each seed's batch position, and
/// of a `neighbors` or `lookup` result, under [`BATCH_COLUMN`], that names
/// the seed a row answers.
pub const SEED_BATCH: &str = "batch";
/// The column of a `neighbors` or `lookup` result naming the seed each row
/// answers, by its batch position: the reference's own name.
pub const BATCH_COLUMN: &str = "__kumo_batch__";
/// The column of a `lookup` result counting the rows that hold the seed's
/// key: the reference's own name. A dialect may leave it out of an answer
/// whose seeds say the count is not read ([`SEED_COUNT_UNREAD`]) or that
/// the source enforces the key unique ([`SEED_KEY_UNIQUE`]), each key then
/// held by one row.
pub const MATCHES_COLUMN: &str = "__kumo_matches__";
/// The metadata key, on a `lookup`'s seed [`SEED_KEY`] column, saying the
/// source enforces the looked-up key unique (the described column's
/// `unique`), with the value [`MARKED`]: set by the core so a SQL layer
/// whose reference reads such a key by a plain join, counting nothing,
/// renders that join (SQLite's `_by_pkey`); a host provider may ignore it.
pub const SEED_KEY_UNIQUE: &str = "sdm:key_unique";
/// The metadata key, on a `lookup`'s [`SEED_BATCH`] column, saying the
/// core does not read how many rows hold each seed's key, with the value
/// [`MARKED`]: set by the core on the lookups of the walk, which the
/// reference counts on none of its SQL sources but DuckDB; a provider may
/// still count.
pub const SEED_COUNT_UNREAD: &str = "sdm:count_unread";
/// The metadata key, on a window bound's seed column, holding one value
/// of the window's column as the source stores it, as text, which the
/// core read before the table's first windowed read (a `scan` of one
/// non-null value): set only for a provider whose data layer says its
/// source compares a bound with that stored text
/// (`DataLayer::compares_stored_times`, SQLite's, which has no time type),
/// so the layer spells each bound as that value is spelled.
pub const SEED_STORED_TIME: &str = "sdm:stored_time";
/// The value of a metadata flag the core sets ([`SEED_KEY_UNIQUE`],
/// [`SEED_COUNT_UNREAD`]).
pub const MARKED: &str = "true";

/// A sample's seed as an integer, when the scalar it travels as is one:
/// the narrowest exact scalar `materialize` writes, a `u64` or a decimal
/// of scale 0, or any other integer scalar; `None` for any other scalar.
/// The one reading of a seed the core's checks and every dialect's
/// rendering share (carried row U12).
pub fn seed_integer(seed: &Scalar) -> Option<i128> {
    Some(match seed {
        Scalar::Int32(v) => i128::from(*v),
        Scalar::Int64(v) => i128::from(*v),
        Scalar::UInt32(v) => i128::from(*v),
        Scalar::UInt64(v) => i128::from(*v),
        Scalar::Decimal128 {
            scale: 0, value, ..
        } => *value,
        _ => return None,
    })
}

/// Names an attachment that travels beside a control message: Arrow data
/// (a fetch's result, a key set, a predict result) or bytes (a model
/// request or response body).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct AttachmentId(pub u32);

/// A set of keys: an Arrow attachment, and the columns it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeySet {
    pub attachment: AttachmentId,
    pub columns: Vec<Field>,
}

/// What a request reads: a relation of the source, by its name's parts,
/// each quoted alone and never split on a dot, so `["a.b"]` is one table
/// named `a.b` and `["main", "t"]` the table `t` of `main` (design choice
/// 12). Plain data, a list of strings, as easily built by a host in any
/// language as a name is. How many parts a source takes is its dialect's:
/// `sdm-data` renders every dialect's `FROM` from a relation through one
/// function and refuses one of no part, or of more parts than its dialect
/// accepts, by name (rule `relation`); a host provider refuses one it
/// cannot read the same way. Two schema tables over one source are two
/// equal relations.
///
/// A relation may be derived (contract v28, design choice 3 of Phase 3):
/// `columns`, where given, are the relation's columns, each a name with
/// the source column it reads or an SQL text expression over the source's
/// columns, and `filter` an SQL text condition every row read holds, as a
/// view's definition gives them. Both are text of the source's own SQL,
/// opaque to the core, which never reads or writes them: only a provider
/// declaring [`ProviderCapabilities::derived_relations`] is asked for a
/// derived relation, and the core refuses one for any other by name (rule
/// [`DERIVED_RULE`]).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    pub source: Vec<String>,
    /// The relation's columns, in order, where it is derived; empty, the
    /// source's own columns under their own names; or, where `found`, the
    /// columns the source's catalog found.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<RelationColumn>,
    /// The condition, in the source's SQL, every row of the relation holds;
    /// left out, every row of the source is the relation's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// The columns are the ones the source's catalog found, each read by
    /// its own name, with no filter (T3.4, round 2): a provider may read
    /// the table through them, so that one since dropped or renamed fails
    /// the read rather than reading anything else by its name (the
    /// reference's Databricks reads, `_quoted_source_name`), or read the
    /// table whole where no column needs it. Left out, `false`: the
    /// columns, if any, are the relation's own.
    #[serde(default, skip_serializing_if = "is_false")]
    pub found: bool,
}

/// One column of a derived relation: the name the relation gives it, and
/// what it reads, exactly one of a column of the source (`column`) or an
/// expression in the source's SQL over its columns (`expression`).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct RelationColumn {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
}

impl Relation {
    /// The relation of one part, `name`, as a host that names a table by a
    /// string gives it.
    pub fn named(name: impl Into<String>) -> Self {
        Relation::of(vec![name.into()])
    }

    /// The relation of the name parts `source`, read whole.
    pub fn of(source: Vec<String>) -> Self {
        Relation {
            source,
            columns: Vec::new(),
            filter: None,
            found: false,
        }
    }

    /// Whether the relation is derived: it names its columns or a filter.
    pub fn is_derived(&self) -> bool {
        !self.columns.is_empty() || self.filter.is_some()
    }

    /// The relation's name as the source's catalog writes a table's name,
    /// its parts joined by dots, as the reference names a table it reads
    /// (`source_name`), and a declared foreign key names the table it
    /// refers to.
    pub fn dotted(&self) -> String {
        self.source.join(".")
    }

    /// The one part of a relation of one part; `None` for any other.
    pub fn single(&self) -> Option<&str> {
        match self.source.as_slice() {
            [name] => Some(name),
            _ => None,
        }
    }
}

impl From<&str> for Relation {
    fn from(name: &str) -> Self {
        Relation::named(name)
    }
}

impl From<String> for Relation {
    fn from(name: String) -> Self {
        Relation::named(name)
    }
}

/// The rule a refusal of a derived relation names: asked of a provider
/// that does not declare [`ProviderCapabilities::derived_relations`], or
/// one whose columns do not each read exactly one source column or
/// expression, or name a column twice.
pub const DERIVED_RULE: &str = "derived_relation";

/// The refusal of a derived `relation`, of `field` of `request`, by a
/// reader that renders derived relations where `renders` says so: one asked
/// of a reader that does not, `UNSUPPORTED_REQUEST`; one whose columns are
/// not each a name with exactly one of a source column or an expression,
/// or name one column twice, `INVALID_REQUEST`; `None` for a relation read
/// whole,
/// and for a derived one the reader renders. Each names the rule
/// ([`DERIVED_RULE`]), never a part's or an expression's text.
pub fn refuse_derived(
    request: &DataRequest,
    field: &str,
    relation: &Relation,
    renders: bool,
) -> Option<CoreError> {
    if !relation.is_derived() {
        return None;
    }
    let refusal = |sentence, message| {
        Some(
            CoreError::new(sentence, message)
                .with_field("request", Detail::text(request.name()))
                .with_field("field", Detail::text(field)),
        )
    };
    if !renders {
        return refusal(
            Sentence::DerivedRelationUnread,
            "a derived relation is asked of a provider that does not read one",
        );
    }
    // The columns a catalog found are its columns, each by its own name,
    // with no filter: a relation marked so that renames, computes or filters
    // contradicts itself, and is refused rather than read as the table.
    if relation.found
        && (relation.filter.is_some()
            || relation
                .columns
                .iter()
                .any(|c| c.expression.is_some() || c.column.as_ref() != Some(&c.name)))
    {
        return refusal(
            Sentence::DerivedRelationMalformed,
            "a relation of the columns its catalog found renames, computes or filters them",
        );
    }
    let mut names = BTreeSet::new();
    for column in &relation.columns {
        let reads = usize::from(column.column.is_some()) + usize::from(column.expression.is_some());
        if reads != 1 || column.name.is_empty() || !names.insert(column.name.as_str()) {
            return refusal(
                Sentence::DerivedRelationMalformed,
                "a derived relation's column reads no one source column or expression, or is named twice",
            );
        }
    }
    None
}

/// The refusal of `relation`, of `field` of `request`, by a reader that
/// takes from one to `most` parts (rule `relation`): `INVALID_REQUEST`
/// for a relation of no part, which no source reads, and
/// `UNSUPPORTED_REQUEST` for more parts than the reader takes; `None` for
/// a relation it reads. Each names the parts given and the most it takes,
/// never a part's text.
pub fn refuse_relation(
    request: &DataRequest,
    field: &str,
    relation: &Relation,
    most: usize,
) -> Option<CoreError> {
    let parts = relation.source.len();
    let (sentence, message) = match parts {
        0 => (Sentence::RelationNoPart, "a relation names no part"),
        n if n > most => (
            Sentence::RelationTooManyParts,
            "a relation has more name parts than its source reads",
        ),
        _ => return None,
    };
    Some(
        CoreError::new(sentence, message)
            .with_field("request", Detail::text(request.name()))
            .with_field("field", Detail::text(field))
            .with_field("parts", Detail::Scalar(Scalar::UInt64(parts as u64)))
            .with_field("most", Detail::Scalar(Scalar::UInt64(most as u64))),
    )
}

/// The columns of a seed key set holding each seed's key, in the order of
/// the key's columns ([`SEED_KEY`] for a key of one column, `key_1` to
/// `key_<n>` for a key of `n` columns), beside [`SEED_BATCH`] (design
/// choice 1: multi-column keys in `neighbors` and `lookup`).
///
/// Groundwork (carried row P2-MID2-5): contract v26's vocabulary of
/// sampling by a key of several columns (this naming, and a `key` of
/// several columns in `sample_entities`, `neighbors` and `lookup`) is kept
/// for the composite keys a later contract takes. The core refuses a
/// schema keyed by several columns at materialize before it samples, as
/// the reference does (`sdm-core`'s `refuse_composite_keys`; register
/// P2E-5, decision KO2-68), so no sampler request names a key of more than
/// one column yet.
pub fn seed_key_columns(columns: usize) -> Vec<String> {
    match columns {
        1 => vec![SEED_KEY.to_owned()],
        n => (1..=n).map(|i| format!("{SEED_KEY}_{i}")).collect(),
    }
}

/// One bound of a time range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub value: Scalar,
    pub inclusive: bool,
}

/// A condition on the rows a request reads; a request's filters are a
/// conjunction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "filter", rename_all = "snake_case", deny_unknown_fields)]
pub enum Filter {
    /// The row's `columns` equal one of the key set's rows.
    KeyIn {
        columns: Vec<String>,
        keys: KeySet,
    },
    TimeRange {
        column: String,
        lower: Option<Bound>,
        upper: Option<Bound>,
    },
    NotNull {
        column: String,
    },
}

/// Which part of the catalog a `describe` reads.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DescribePart {
    /// The source's table names: one row per table. With tables named,
    /// the relations, tables or views, the source would read under any of
    /// those names written unqualified, by its own resolution of such a
    /// name (on DuckDB, through its search path): none when it would read
    /// none, which is how the core learns a name the caller gave is not
    /// the source's, never by matching names itself (T1.23, decision
    /// E59).
    Tables,
    /// Each table's columns, with their declared types, nullability and
    /// primary-key flags.
    Columns,
    /// Each table's declared primary and unique keys.
    Keys,
    /// Each table's declared foreign keys.
    ForeignKeys,
    /// The tables' indexes, which feed the reference's "Missing N indices"
    /// warning: the source's whole, or one table's where the provider's
    /// index advice reads them so ([`IndexAdvice::LeadingColumn`]).
    Indexes,
    /// The columns of one index, named by a relation as a table is (an
    /// index is its schema's object): one row per column, in the index's
    /// order, its name, null for an expression. Asked for an index whose
    /// columns the catalog's list of keys or indexes does not give
    /// (SQLite's `index_info`, as the reference reads it). A host provider
    /// is never asked it: its `keys` and `indexes` answers give each key's
    /// and index's columns ([`catalog_columns`]), so no key or index it
    /// lists is read by its name.
    IndexColumns,
    /// The namespace a relation of one part is read in, as the source names
    /// a table it holds: one row per part, outermost first, its name
    /// (contract v28, T3.8): a Postgres source's current schema, under
    /// which the reference names a table it cannot find; no row for a
    /// source whose tables are named alone (DuckDB's, SQLite's). Asked only
    /// to name a table the source lacks.
    Namespace,
    /// The definition of the one semantic model named (contract v28, T3.27):
    /// a Databricks metric view's `DESCRIBE TABLE EXTENDED`, a Snowflake
    /// semantic view's `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW`, read into
    /// [`CatalogAnswer::ViewDefinition`]. Only a SQL provider whose dialect
    /// reads views is asked it; every other dialect refuses it by name
    /// ([`Sentence::ViewsUnread`]), and a host provider is never asked it.
    ViewDefinition,
}

/// A column of a table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TableColumn {
    pub table: Relation,
    pub column: String,
}

/// Columns of a table that together make a key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TableKey {
    pub table: Relation,
    pub columns: Vec<String>,
    /// The number a `key_counts` row for this key is labelled with, where
    /// not its position among the keys asked: the reference labels each by
    /// its table's place among the tables with a key, a key it does not
    /// count included (T2.12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<u64>,
    /// The source keeps the key unique (a declared primary or unique key
    /// that may still hold a null): its distinct values are the rows
    /// holding all of it, which a rendering may count as the rows less
    /// those missing part of it, as the reference counts such a key
    /// (`_key_count_sql`; contract v28, carried row P2-T2.2-3). The count
    /// is the same either way.
    #[serde(default, skip_serializing_if = "is_false")]
    pub unique: bool,
}

/// Serde's test for leaving out a flag that is not set.
#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's signature")]
pub(crate) fn is_false(flag: &bool) -> bool {
    !*flag
}

impl TableKey {
    /// The label of the key at `position` among the keys asked.
    pub fn label_at(&self, position: usize) -> u64 {
        self.label.unwrap_or(position as u64)
    }
}

/// One bound of a `neighbors` window: a seed time column, compared with the
/// window column in the source's own `CAST(column AS TIMESTAMP)` wall clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeedBound {
    /// A zone-less `timestamp[ns]` column of the seed key set.
    pub seed_column: String,
    pub inclusive: bool,
}

/// The rows of a `neighbors` request in time: the window column between
/// the bounds a seed gives. A row whose window column is NULL is not
/// returned when a window is set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub column: String,
    pub lower: Option<SeedBound>,
    pub upper: Option<SeedBound>,
}

/// One term of a [`TieBreak`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TieTerm {
    pub column: String,
    /// Order by `CAST(column AS TIMESTAMP)`, the source's wall clock,
    /// rather than the column's own values.
    pub as_timestamp: bool,
    pub descending: bool,
    /// Nulls after every value; the core writes the order the provider
    /// declares (`ProviderCapabilities::null_order`).
    pub nulls_last: bool,
}

/// The total order a `neighbors` request keeps per seed, built by the core
/// from the reference's rule (design, "Data access").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct TieBreak(pub Vec<TieTerm>);

/// A typed request for data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "request", rename_all = "snake_case", deny_unknown_fields)]
pub enum DataRequest {
    /// One part of the catalog for the tables named (all when empty): what
    /// the provider returns is in the shape its plan expects, which the
    /// data layer reads into a [`CatalogAnswer`].
    Describe {
        tables: Vec<Relation>,
        part: DescribePart,
    },
    /// Every matching row of `table`, `columns` in order, paged through to
    /// the end; `limit` asks for any `limit` rows instead.
    Scan {
        table: Relation,
        columns: Vec<String>,
        filters: Vec<Filter>,
        limit: Option<u64>,
    },
    /// The first rows of `table`, for inference.
    Preview {
        table: Relation,
        columns: Vec<String>,
        limit: u64,
    },
    CountRows {
        table: Relation,
        filters: Vec<Filter>,
    },
    /// For each key, in order, one row: its label (its position, where it
    /// gives none), the table's rows, the key's distinct values among rows
    /// holding all of it, and the rows missing any part of it, as the
    /// reference's single `UNION ALL` reads them.
    KeyCounts { keys: Vec<TableKey> },
    /// For each time column, in order, one row: its position and the
    /// smallest and largest `CAST(column AS TIMESTAMP)`, as the reference's
    /// single `UNION ALL` reads them.
    TimeBounds { columns: Vec<TableColumn> },
    /// One row: the rows of `table`, then, for each of `columns` in order,
    /// its smallest and largest value, each of the column's own type and
    /// null when no row holds one, and the rows holding a value, read over
    /// every row in one statement, as the reference's time column choice
    /// reads a table its preview may not hold whole (`_full_range`).
    ColumnRanges {
        table: Relation,
        columns: Vec<String>,
    },
    /// At most one row: a value of `key` that more than one row of `table`
    /// holds, among the rows holding every part of it, as the reference
    /// reads one to show a repeated key (`_repeated_key_sql`).
    RepeatedKey { table: Relation, key: Vec<String> },
    /// A seeded sample of `n` entities of `table` by `key`: `columns` in
    /// order, from the rows meeting every filter. `key_dtype`, where given,
    /// is the data type of a key of one column the draw orders by: the type
    /// the source declares it where its catalog fixes one, else the
    /// schema's; a dialect that hashes only some types (SQLite's, which
    /// hashes an integer key and the `rowid` otherwise) chooses by it once
    /// for the column. Left out, the dialect decides by itself.
    SampleEntities {
        table: Relation,
        key: Vec<String>,
        n: u64,
        seed: Scalar,
        columns: Vec<String>,
        filters: Vec<Filter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key_dtype: Option<crate::ops::Dtype>,
        /// The rows `table` holds, where the provider sizes its draw by
        /// them ([`ProviderCapabilities::sample_rows`]): the core counts
        /// them (`count_rows`) before it samples and names the count here,
        /// as the reference's warehouse samplers read each table's rows
        /// first (contract v28). Left out for any other provider.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rows: Option<u64>,
    },
    /// The rows of `table` whose `key` equals a seed's key, `columns` in
    /// order after the seed's batch position (decision H15). `key` names the
    /// table's key columns in order, one or several (design choice 1; a key
    /// of several is groundwork, [`seed_key_columns`]);
    /// `seeds` holds the key's columns ([`seed_key_columns`]), `batch` and
    /// the seed time columns `window` names. With `per_seed`, at most that many rows per seed, first in
    /// `order`, and the result is ordered by `(batch, order)`; without it,
    /// there is no `order` and the result is unordered.
    Neighbors {
        table: Relation,
        key: Vec<String>,
        columns: Vec<String>,
        seeds: KeySet,
        window: Option<Window>,
        per_seed: Option<u64>,
        order: Option<TieBreak>,
    },
    /// One row per seed whose key `table` holds, `columns` in order after
    /// the seed's batch position and the count of rows holding its key;
    /// which row of a key several rows hold is not defined. `key` and
    /// `seeds` as `neighbors`'.
    Lookup {
        table: Relation,
        key: Vec<String>,
        columns: Vec<String>,
        seeds: KeySet,
    },
}

impl DataRequest {
    /// Each relation the request reads, with the field naming it.
    pub fn relations(&self) -> Vec<(&'static str, &Relation)> {
        match self {
            DataRequest::Describe { tables, .. } => tables.iter().map(|t| ("tables", t)).collect(),
            DataRequest::KeyCounts { keys } => keys.iter().map(|k| ("keys", &k.table)).collect(),
            DataRequest::TimeBounds { columns } => {
                columns.iter().map(|c| ("columns", &c.table)).collect()
            }
            DataRequest::Scan { table, .. }
            | DataRequest::Preview { table, .. }
            | DataRequest::CountRows { table, .. }
            | DataRequest::ColumnRanges { table, .. }
            | DataRequest::RepeatedKey { table, .. }
            | DataRequest::SampleEntities { table, .. }
            | DataRequest::Neighbors { table, .. }
            | DataRequest::Lookup { table, .. } => vec![("table", table)],
        }
    }

    /// Each relation the request reads, to change in place.
    pub fn relations_mut(&mut self) -> Vec<&mut Relation> {
        match self {
            DataRequest::Describe { tables, .. } => tables.iter_mut().collect(),
            DataRequest::KeyCounts { keys } => keys.iter_mut().map(|k| &mut k.table).collect(),
            DataRequest::TimeBounds { columns } => {
                columns.iter_mut().map(|c| &mut c.table).collect()
            }
            DataRequest::Scan { table, .. }
            | DataRequest::Preview { table, .. }
            | DataRequest::CountRows { table, .. }
            | DataRequest::ColumnRanges { table, .. }
            | DataRequest::RepeatedKey { table, .. }
            | DataRequest::SampleEntities { table, .. }
            | DataRequest::Neighbors { table, .. }
            | DataRequest::Lookup { table, .. } => vec![table],
        }
    }

    /// The request's name, as the contract spells it.
    pub fn name(&self) -> &'static str {
        match self {
            DataRequest::Describe { .. } => "describe",
            DataRequest::Scan { .. } => "scan",
            DataRequest::Preview { .. } => "preview",
            DataRequest::CountRows { .. } => "count_rows",
            DataRequest::KeyCounts { .. } => "key_counts",
            DataRequest::TimeBounds { .. } => "time_bounds",
            DataRequest::ColumnRanges { .. } => "column_ranges",
            DataRequest::RepeatedKey { .. } => "repeated_key",
            DataRequest::SampleEntities { .. } => "sample_entities",
            DataRequest::Neighbors { .. } => "neighbors",
            DataRequest::Lookup { .. } => "lookup",
        }
    }

    /// The optional request this is, if it is one; `describe` and `scan`
    /// are what every provider answers.
    pub fn optional(&self) -> Option<OptionalRequest> {
        match self {
            DataRequest::Describe { .. } | DataRequest::Scan { .. } => None,
            DataRequest::Preview { .. } => Some(OptionalRequest::Preview),
            DataRequest::CountRows { .. } => Some(OptionalRequest::CountRows),
            DataRequest::KeyCounts { .. } => Some(OptionalRequest::KeyCounts),
            DataRequest::TimeBounds { .. } => Some(OptionalRequest::TimeBounds),
            DataRequest::ColumnRanges { .. } => Some(OptionalRequest::ColumnRanges),
            DataRequest::RepeatedKey { .. } => Some(OptionalRequest::RepeatedKey),
            DataRequest::SampleEntities { .. } => Some(OptionalRequest::SampleEntities),
            DataRequest::Neighbors { .. } => Some(OptionalRequest::Neighbors),
            DataRequest::Lookup { .. } => Some(OptionalRequest::Lookup),
        }
    }

    /// Refuses a request whose fields contradict each other, so every
    /// field is honored or the request refused, never ignored: a
    /// `neighbors` with `order` and no `per_seed` or the reverse, an empty
    /// tie-break, seeds without the `key` and `batch` columns, or a window
    /// bound naming no zone-less `timestamp[ns]` seed column; a request
    /// asking for no columns or no keys where it must name some.
    pub fn check(&self) -> Result<(), CoreError> {
        match self {
            DataRequest::KeyCounts { keys } => {
                if keys.is_empty() || keys.iter().any(|k| k.columns.is_empty()) {
                    return Err(refused(self, "keys", "a key names no columns"));
                }
            }
            DataRequest::TimeBounds { columns } if columns.is_empty() => {
                return Err(refused(self, "columns", "no time column is named"));
            }
            DataRequest::ColumnRanges { columns, .. } if columns.is_empty() => {
                return Err(refused(self, "columns", "no column is named"));
            }
            DataRequest::RepeatedKey { key, .. } if key.is_empty() => {
                return Err(refused(self, "key", "a key names no columns"));
            }
            DataRequest::Neighbors {
                key,
                seeds,
                window,
                per_seed,
                order,
                ..
            } => {
                check_seeds(self, key, seeds)?;
                match (per_seed, order) {
                    (Some(_), None) => {
                        return Err(refused(
                            self,
                            "order",
                            "a row limit per seed needs an order",
                        ));
                    }
                    (None, Some(_)) => {
                        return Err(refused(
                            self,
                            "order",
                            "an order without a row limit per seed is not kept",
                        ));
                    }
                    (Some(_), Some(TieBreak(terms))) if terms.is_empty() => {
                        return Err(refused(self, "order", "the order has no terms"));
                    }
                    _ => {}
                }
                if let Some(window) = window {
                    if window.lower.is_none() && window.upper.is_none() {
                        return Err(refused(self, "window", "the window has no bound"));
                    }
                    for bound in window.lower.iter().chain(&window.upper) {
                        let seed_time = seeds.columns.iter().any(|f| {
                            f.name == bound.seed_column
                                && f.column_type
                                    == ColumnType::Timestamp {
                                        unit: Unit::Ns,
                                        zone: None,
                                    }
                        });
                        if !seed_time {
                            return Err(refused(
                                self,
                                "window",
                                "a bound names no zone-less timestamp[ns] seed column",
                            ));
                        }
                    }
                }
            }
            DataRequest::Lookup { key, seeds, .. } => check_seeds(self, key, seeds)?,
            _ => {}
        }
        Ok(())
    }
}

/// A key names some columns, and seeds hold the role-named columns of each
/// of its columns ([`seed_key_columns`]) and `batch` (decision H15).
fn check_seeds(request: &DataRequest, key: &[String], seeds: &KeySet) -> Result<(), CoreError> {
    if key.is_empty() {
        return Err(refused(request, "key", "a key names no columns"));
    }
    for role in seed_key_columns(key.len())
        .iter()
        .map(String::as_str)
        .chain([SEED_BATCH])
    {
        if !seeds.columns.iter().any(|f| f.name == role) {
            return Err(refused(
                request,
                "seeds",
                "the seeds lack a key or a batch column",
            ));
        }
    }
    Ok(())
}

/// The refusal of a request whose fields contradict each other, naming the
/// request and the field, never a value.
pub fn refused(request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(Sentence::DataRequestRefused, message)
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text(field))
}

/// The optional requests a provider may declare.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OptionalRequest {
    Preview,
    CountRows,
    KeyCounts,
    TimeBounds,
    ColumnRanges,
    RepeatedKey,
    SampleEntities,
    Neighbors,
    Lookup,
}

impl OptionalRequest {
    /// The request's name, as the contract spells it.
    pub fn name(self) -> &'static str {
        match self {
            OptionalRequest::Preview => "preview",
            OptionalRequest::CountRows => "count_rows",
            OptionalRequest::KeyCounts => "key_counts",
            OptionalRequest::TimeBounds => "time_bounds",
            OptionalRequest::ColumnRanges => "column_ranges",
            OptionalRequest::RepeatedKey => "repeated_key",
            OptionalRequest::SampleEntities => "sample_entities",
            OptionalRequest::Neighbors => "neighbors",
            OptionalRequest::Lookup => "lookup",
        }
    }
}

/// Whether a provider's seeded sampling repeats.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    /// The same seed gives the same rows on this provider.
    SeededStable,
    Unseeded,
}

/// How a provider's source reads a time it stores as text: which of the
/// reference's two readings the core applies to the text it fetches from
/// the provider (register DT-7). The provider declares it; the core picks
/// its reader by it, never by a dialect.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TextTimes {
    /// Each value on its own, as the reference's engine reads a SQL
    /// source's text time column (`to_datetime(values, unit='ns')`).
    ValueByValue,
    /// The column by the client's rules (`date-rules.md`, "The client
    /// rules"), as the reference reads a frame's time column.
    ClientRules,
}

/// Where a provider's source puts a null among the values it orders, in
/// either direction: the order the reference's neighbor reads get from the
/// source, which writes no null order of its own (`_neighbor_order_by`).
/// The provider declares it; the core writes it into the tie-break it asks
/// for ([`TieTerm::nulls_last`]), never by a dialect. A source whose order
/// depends on the direction is not declarable yet: the dialect that needs
/// one adds it under a new contract version.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NullOrder {
    /// Nulls first, ascending or descending.
    First,
    /// Nulls last, ascending or descending.
    Last,
}

/// Which of the source's indexes the reference's sampler counts as serving
/// a read on a key or foreign key, for its warning of the indexes it would
/// use and finds missing. The provider declares it; the core reads the
/// source's indexes, and warns, only when it is declared.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum IndexAdvice {
    /// An index over exactly that one column, its coverage resolved; one
    /// over several columns, or one the reader cannot resolve, serves none.
    SingleColumn,
    /// The reference's SQLite sampler's: each table's indexes read one
    /// table at a time, each index's columns by its name; a read wants an
    /// index on the key, and on each foreign key with its table's time
    /// column, and any index whose first column is the read's first serves
    /// it.
    LeadingColumn,
}

/// What a provider answers: a host provider declares it, and the data
/// layer gives it for each SQL dialect it renders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilities {
    pub optional: BTreeSet<OptionalRequest>,
    /// The most keys one request may carry; `None` is no limit.
    pub max_keys_per_request: Option<u64>,
    pub sampling: Determinism,
    /// The largest seed the provider samples with (register entry RS-7),
    /// which every provider declares: there is no "any seed" default
    /// (design choice 1). The core refuses a sample with a larger seed,
    /// `INVALID_REQUEST` naming the seed, before the sample is planned;
    /// which seeds a provider takes is the provider's to declare, never a
    /// name the core matches.
    pub max_seed: u64,
    /// How the source reads a time it stores as text (register DT-7);
    /// `None` declares no reading, and the core refuses to read text times
    /// from it (`UNSUPPORTED_REQUEST`, rule `text_times`).
    pub text_times: Option<TextTimes>,
    /// Where the source puts a null when it orders (`NullOrder`); `None`
    /// declares none, and the core refuses to ask it for rows in an order
    /// (`UNSUPPORTED_REQUEST`, rule `null_order`).
    pub null_order: Option<NullOrder>,
    /// Which indexes serve the sampler's reads (`IndexAdvice`); `None`
    /// declares no meaningful index advice: the core neither reads the
    /// source's indexes nor warns of missing ones.
    pub index_advice: Option<IndexAdvice>,
    /// The limits within which the core answers an optional request the
    /// provider does not declare itself, in memory, from the rows a `scan`
    /// reads (design choice 1, the in-memory fallback, T2.6); left out,
    /// the provider declares none, and such a request is refused,
    /// `UNSUPPORTED_REQUEST`, as before. A declared request that fails is
    /// the provider's error either way, never answered by the fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<FallbackLimits>,
    /// The provider reads derived relations ([`Relation::columns`],
    /// [`Relation::filter`]; contract v28). Left out, `false`: the core
    /// refuses a derived relation for the provider by name (rule
    /// [`DERIVED_RULE`]).
    #[serde(default, skip_serializing_if = "is_false")]
    pub derived_relations: bool,
    /// The provider sizes an entity sample by the table's rows, as the
    /// reference's warehouse samplers do (a sampled percentage of the
    /// rows): the core counts each sampled table's rows first and names
    /// the count on the sample ([`DataRequest::SampleEntities`]'s `rows`;
    /// contract v28). Left out, `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub sample_rows: bool,
    /// The provider's neighbor reads by a foreign key over a table with a
    /// time column look back no further than the reference's warehouse
    /// samplers do (`_by_fkey`): from each seed's anchor back by five
    /// times the rows asked over the table's rate of rows per referenced
    /// row and second, at most five years, the seconds rounded up. The
    /// core counts the table's and the referenced table's rows, reads the
    /// table's time bounds, and bounds each such read from below by that
    /// look-back, exclusive (contract v28). Left out, `false`: no lower
    /// bound.
    #[serde(default, skip_serializing_if = "is_false")]
    pub neighbor_lookback: bool,
    /// The provider's catalog may fail to read a table's declared keys and
    /// foreign keys where it reads its columns (Unity Catalog's
    /// `table_constraints` on an older runtime or without the privilege):
    /// the core then reads a table's columns and its keys apart, and reads
    /// a failed key or foreign-key read as no declared keys, as the
    /// reference's Databricks reader carries on without them
    /// (`DatabricksTable`'s key reads; contract v28). Left out, `false`: a
    /// failed key read fails the description.
    #[serde(default, skip_serializing_if = "is_false")]
    pub keys_optional: bool,
    /// The vendor whose opaque Arrow extension types (`arrow.opaque`, its
    /// metadata's `vendor_name`) the provider hands over: the host layer
    /// reads an opaque column by that vendor's layout only when the column
    /// names this vendor (register P1B-43). Left out, none, and no opaque
    /// column is read as anything but its storage (contract v28, carried
    /// row P2-MID2-2: before it, a flag of the data layer's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque_vendor: Option<String>,
    /// The provider compares a window's bounds with the stored text of the
    /// window's column, as a source with no time type does (SQLite's
    /// reference spells each bound as one stored value is spelled,
    /// `_time_spelling`): the core then reads one non-null value of a
    /// table's time column (a `scan` of one row) before that table's first
    /// windowed read in a run, and names it on each bound's seed column
    /// ([`SEED_STORED_TIME`]). Left out, `false` (P2-MID2-2).
    #[serde(default, skip_serializing_if = "is_false")]
    pub compares_stored_times: bool,
    /// The provider orders a seeded sample by the key's data type, as
    /// SQLite's does (it hashes an integer key and the `rowid` otherwise):
    /// the core then names the schema's type of a key of one column on each
    /// `sample_entities` (`key_dtype`). Left out, `false` (P2-MID2-2).
    #[serde(default, skip_serializing_if = "is_false")]
    pub samples_by_key_dtype: bool,
    /// The reference reads the provider's rows into a frame of NumPy-backed
    /// floats, where a NaN is missing and `-0.0` is `0.0` (Postgres's rows,
    /// read as Python objects); otherwise as Arrow-backed pandas does, where
    /// a NaN is a value and the two zeros are two values (T2.12). Left out,
    /// `false` (P2-MID2-2).
    #[serde(default, skip_serializing_if = "is_false")]
    pub reads_nan_as_missing: bool,
}

/// The in-memory fallback's limits, as a provider declares them
/// ([`ProviderCapabilities::fallback`]): past either, the fallback fails
/// by name rather than read on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FallbackLimits {
    /// The most rows the fallback reads to answer one request.
    pub max_rows: u64,
    /// The most bytes of Arrow data it holds to answer one request.
    pub max_bytes: u64,
}

/// A built-in SQL dialect, by the name the host gives it. The contract
/// names no dialect and knows none (decision E54): which names a build
/// serves is the data layer's, which refuses one it does not serve before
/// the run exists ([`DataLayer::capabilities`]).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct Dialect(pub String);

impl Dialect {
    /// The dialect named `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Dialect(name.into())
    }

    /// The dialect's name, as the host gave it.
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// The composition root's choice: a built-in SQL provider by dialect, which
/// compiles each request to SQL the host runs, or a host provider that
/// answers the typed requests itself.
///
/// A SQL provider's `namespace` (contract v28) is where the source's tables
/// are, as the source was given it: the name parts before a table's own,
/// outermost first, a Postgres source's schema (`PostgresSource(schema=...)`),
/// each `null` where the source names none, which is then the connection's
/// current one. Every relation of one part a request names is read under
/// it, and the source's tables are listed in it, as the reference
/// qualifies each read by the schema it was given (`PostgresTable`). Left
/// out, empty or all `null`, the connection's current one, as before v28.
/// A dialect takes as many parts as a relation's parts before its table's,
/// and refuses by name more, or a part given inside one left `null`.
///
/// A SQL provider's `view` (contract v28, T3.27) names the semantic model
/// the source is read through, a Databricks metric view or a Snowflake
/// semantic view by its dialect, as the caller named it, with whether the
/// caller trusts its expressions: its logical tables are then the
/// relations of one part a request names, each the logical table's name.
/// Left out, the source's own tables, as before.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "provider", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderChoice {
    Sql {
        dialect: Dialect,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        namespace: Vec<Option<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        view: Option<ViewChoice>,
    },
    Host {
        capabilities: ProviderCapabilities,
    },
}

impl ProviderChoice {
    /// The built-in SQL provider of `dialect`, over the connection's
    /// current namespace.
    pub fn sql(dialect: impl Into<String>) -> Self {
        ProviderChoice::Sql {
            dialect: Dialect::new(dialect),
            namespace: Vec::new(),
            view: None,
        }
    }
}

/// The semantic model a SQL provider's source is read through
/// ([`ProviderChoice::Sql`]'s `view`): its name as the caller gave it, and
/// whether the caller trusts whoever authored it to run its expressions
/// (`semantic_model(name, trust_expressions=...)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ViewChoice {
    pub name: String,
    pub trust_expressions: bool,
}

/// A statement's parameter: a scalar, or a key set the statement's text
/// reads under `relation`. How the keys come to be read by that name is
/// the dialect's, which `sdm-data` renders and its provider binds: the
/// DuckDB provider registers the attachment as a relation of that name; a
/// PostgreSQL dialect may bind the keys as JSON text that a common table
/// expression of that name unpacks, as the reference's PostgreSQL sampler
/// does. `relation` names what the text reads, not an instruction to
/// register an Arrow relation. A dialect that binds its key sets as JSON
/// text names none: [`DataLayer::bind`] turns each into a `text` parameter.
///
/// A `text` parameter (contract v28, T3.27) binds the one text cell of a
/// one-row, one-column Arrow attachment: a key set the dialect writes as
/// JSON text itself, past the control message's size ([`DataLayer::bind`]).
/// In a [`BoundFetch`] the attachment is a local reference the data layer
/// chose; the core installs its batch in the run and names it here by the
/// run's own id before the step is sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "param", rename_all = "snake_case", deny_unknown_fields)]
pub enum SqlParam {
    Value { value: Scalar },
    KeySet { keys: KeySet, relation: String },
    Text { attachment: AttachmentId },
}

/// One statement a SQL provider runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SqlStatement {
    pub dialect: Dialect,
    pub text: String,
    pub params: Vec<SqlParam>,
}

/// How a provider answers one request (decision H14): SQL statements run
/// in order on one session, the last one's rows the answer, or the typed
/// request itself, passed to a host provider that answers it; either way
/// with the Arrow schema the whole plan returns, and answered with one
/// attachment holding every row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "plan", rename_all = "snake_case", deny_unknown_fields)]
pub enum FetchPlan {
    Sql {
        statements: Vec<SqlStatement>,
        expect: ExpectedSchema,
    },
    Host {
        expect: ExpectedSchema,
    },
}

impl FetchPlan {
    /// The schema the plan returns.
    pub fn expect(&self) -> &ExpectedSchema {
        match self {
            FetchPlan::Sql { expect, .. } | FetchPlan::Host { expect } => expect,
        }
    }

    /// Each attachment the plan's statements bind, a key set's or a text's,
    /// that `to` names, named by what `to` maps it to (contract v28,
    /// T3.28): how the core names a bound fetch's batches by the run's ids
    /// ([`DataLayer::bind`]). A host plan binds none.
    pub fn rename_attachments(&mut self, to: &BTreeMap<AttachmentId, AttachmentId>) {
        let FetchPlan::Sql { statements, .. } = self else {
            return;
        };
        for param in statements.iter_mut().flat_map(|s| &mut s.params) {
            let named = match param {
                SqlParam::KeySet { keys, .. } => &mut keys.attachment,
                SqlParam::Text { attachment } => attachment,
                SqlParam::Value { .. } => continue,
            };
            if let Some(renamed) = to.get(named) {
                *named = *renamed;
            }
        }
    }

    /// The attachments the plan's statements bind, key sets and texts, in
    /// order, each as often as it is bound; none for a host plan.
    pub fn key_sets(&self) -> Vec<AttachmentId> {
        match self {
            FetchPlan::Sql { statements, .. } => statements
                .iter()
                .flat_map(|statement| &statement.params)
                .filter_map(|param| match param {
                    SqlParam::KeySet { keys, .. } => Some(keys.attachment),
                    SqlParam::Text { attachment } => Some(*attachment),
                    SqlParam::Value { .. } => None,
                })
                .collect(),
            FetchPlan::Host { .. } => Vec::new(),
        }
    }
}

/// The data-layer port (decision H14): what the core plans every data
/// request through, and reads every catalog answer with. A host provider's
/// choice carries the capabilities it declared; for a SQL dialect the layer
/// fills them itself (decision E33).
pub trait DataLayer: Send + Sync {
    /// What the run's provider answers: the capabilities a host provider
    /// declared, or those this layer fills for its SQL dialect. A dialect
    /// it has no rendering for is refused with `UNSUPPORTED_REQUEST`
    /// naming it.
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError>;

    /// How the run's provider answers `request`. The core has already
    /// refused an optional request the provider does not declare.
    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError>;

    /// The catalog answer the run's provider returned to `request`, a
    /// `describe`, read into the contract's dialect-neutral shape (decision
    /// E54). The core calls it after admission: `batches` already hold the
    /// schema the request's plan expects (decision H16, check (b)). A
    /// request that is not a `describe` is refused with `INTERNAL_ERROR`.
    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError>;

    /// The run's provider's answer to a fetch, admitted, read as the
    /// reference reads its source's values where the provider hands over
    /// the driver's own (T2.14d, MID2 P2-3): a dialect's value rules, such
    /// as Postgres's text of a JSON value and its reading of a `numeric`
    /// past the digits an Arrow decimal holds (each `float_text`, the text
    /// the reference's language gives a double, where it writes one).
    /// `None`, the default and a host provider's: the answer as it is. A
    /// value the rule cannot read is refused by name, never guessed.
    fn conform(
        &self,
        provider: &ProviderChoice,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<(arrow_schema::SchemaRef, Vec<RecordBatch>)>, CoreError> {
        let _ = (provider, schema, batches, float_text);
        Ok(None)
    }

    /// Whether `provider`'s source holds a type per column that a caller's
    /// type for the column must match (plan B's check, T3.4). `true`, the
    /// default and a host provider's; `false` for a source whose columns
    /// hold affinities only (SQLite's), where the old client reads any
    /// column as the type its caller gives (T3.28).
    fn holds_column_types(&self, provider: &ProviderChoice) -> bool {
        let _ = provider;
        true
    }

    /// Whether `provider` reads a source of tables, a database or warehouse
    /// whose tables each `materialize` reads as they stand at that run:
    /// their location where the provider leaves a part to the session,
    /// their columns and their types (the owner's plan B, T3.4). `false`,
    /// the default and a host provider's: the host's tables are its own,
    /// and only a column holding no facts is read from it.
    fn reads_tables_as_they_stand(&self, provider: &ProviderChoice) -> bool {
        let _ = provider;
        false
    }

    /// The database the run's provider reads, by the name its built-in
    /// dialect has (`duckdb`, `sqlite`, `postgres`), which a host words a
    /// refusal naming the source with (T3.8). `None`, the default and a
    /// host provider's: no database is named.
    fn source_database(&self, provider: &ProviderChoice) -> Option<&'static str> {
        let _ = provider;
        None
    }

    /// Whether `provider` leaves a namespace part to the connection that
    /// must be read before any table: one inside a part it gives (a
    /// catalog named and not its schema), or any part of a dialect that
    /// reads every table under its full namespace (Databricks, whose
    /// catalog names a referenced table in full): the core then reads the
    /// source's namespace first
    /// ([`DescribePart::Namespace`]) and runs under the parts it answers
    /// ([`DataLayer::settled`]; T3.4). `false`, the default and a host
    /// provider's.
    fn unsettled(&self, provider: &ProviderChoice) -> bool {
        let _ = provider;
        false
    }

    /// `provider` with its namespace the parts the source answered for it,
    /// each given; as it is, the default and a host provider's.
    fn settled(&self, provider: &ProviderChoice, parts: Vec<String>) -> ProviderChoice {
        let _ = parts;
        provider.clone()
    }

    /// `relation` as the run's provider reads it: under the provider's
    /// namespace where the layer reads one (contract v28), else as it is,
    /// the default and a host provider's (T3.8).
    fn read_as(&self, provider: &ProviderChoice, relation: &Relation) -> Relation {
        let _ = provider;
        relation.clone()
    }

    /// What the run's provider's failure says, as facts each step of the
    /// core reads the one of that fits what it asked (contract v28, T3.27,
    /// design S3): a spelling retry reads `not_found`, a join's table read
    /// `unusable`, an expression sample `refuses_expression`. The dialect
    /// reads them from the driver's SQLSTATE, vendor code and error class,
    /// as the old client's separate predicates do. None set, the default
    /// and a host provider's: every failure stays the provider's error.
    fn failure(&self, provider: &ProviderChoice, error: &ProviderError) -> FailureFacts {
        let _ = (provider, error);
        FailureFacts::default()
    }

    /// The name forms the catalog read of `relation` tries, in order, kept
    /// for that run once one resolves (contract v28, T3.27, design S2):
    /// Snowflake's as given, then upper-cased. As given alone, the default
    /// and a host provider's. How names compare is another question, the
    /// dialect's `name_case`.
    fn spellings(&self, provider: &ProviderChoice, relation: &Relation) -> Vec<Relation> {
        let _ = provider;
        vec![relation.clone()]
    }

    /// `plan`, the run's provider's answer to `request`, with the key sets
    /// it binds, `keys`, by the attachment the plan names each by, bound as
    /// the provider binds them (contract v28, T3.27, design S4): one fetch
    /// or more, each a plan and the batches it binds, named by local
    /// references the core installs in the run and renames to the run's
    /// ids ([`FetchPlan::rename_attachments`]); their answers, in order,
    /// are the request's. A double the binding writes is written as
    /// `float_text` writes it, the text the reference's language gives a
    /// double, which the core passes in (T3.28). The default, and a host
    /// provider's: one fetch, the plan as it is, binding the key sets as
    /// the core made them.
    fn bind(
        &self,
        provider: &ProviderChoice,
        request: &DataRequest,
        plan: FetchPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
        float_text: fn(f64) -> String,
    ) -> Result<BoundRequest, CoreError> {
        let _ = (provider, request, keys, float_text);
        Ok(BoundRequest {
            fetches: vec![BoundFetch {
                plan,
                batches: BTreeMap::new(),
            }],
        })
    }

    /// The answer to one question about a semantic model's text (contract
    /// v28, T3.27, design S5): a name, a condition, an expression checked
    /// against its table's columns, a derived relation rendered, each in
    /// the run's provider's dialect. Refused by name, the default and a
    /// host provider's: no semantic model is read through it.
    fn view_text(
        &self,
        provider: &ProviderChoice,
        ask: ViewTextAsk,
    ) -> Result<ViewTextAnswer, CoreError> {
        let _ = (provider, ask);
        Err(views_unread())
    }
}

/// The refusal of a semantic model by a provider that reads none
/// (`UNSUPPORTED_REQUEST`, rule `view`): the default of
/// [`DataLayer::view_text`], and each dialect's answer to a
/// [`DescribePart::ViewDefinition`] it does not read.
pub fn views_unread() -> CoreError {
    CoreError::new(
        Sentence::ViewsUnread,
        "the data source reads no semantic model",
    )
}

/// What a provider's failure says ([`DataLayer::failure`]): independent
/// facts, each read by the step that asked what it fits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FailureFacts {
    /// The relation asked for does not exist under the name asked (Snowflake
    /// errno 2003): a spelling retry tries the next form.
    pub not_found: bool,
    /// The source refused an expression it was asked to read, for what it
    /// says or reads (Databricks SQLSTATE class `22` or `42`): an expression
    /// sample leaves that column out.
    pub refuses_expression: bool,
    /// Why a table that failed to be read cannot be used, if the failure
    /// says (Databricks `42P01`, `42501` or a permission error class): a
    /// join over it is dropped with that reason.
    pub unusable: Option<Unusable>,
}

/// Why a table cannot be used ([`FailureFacts::unusable`]), as the old
/// client's Databricks reader says it (`unusable_table_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unusable {
    /// It does not exist, or the caller may not see it: Unity Catalog
    /// answers both alike.
    Missing,
    /// The caller may not read it.
    Forbidden,
}

/// A key set's Arrow data, as the core made it or as a dialect binds it.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyData {
    pub schema: arrow_schema::SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// A request bound for the run's provider ([`DataLayer::bind`]; the
/// design's `Bound`, a name the window bound already has): the fetches the
/// core sends in one hop, whose answers, in order, are the request's; any
/// one failing fails the request.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundRequest {
    pub fetches: Vec<BoundFetch>,
}

/// One fetch of a [`BoundRequest`]: its plan, and the batches its
/// statements bind, by the local reference each names them by.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundFetch {
    pub plan: FetchPlan,
    pub batches: BTreeMap<AttachmentId, KeyData>,
}

/// The columns of the answers whose shape the contract fixes (decision
/// E36): every provider answers these requests with exactly these columns,
/// in this order, of these types, and check (b) holds it to them (decision
/// H16). The counts and bounds are aggregates every rendering names by
/// alias, the reference's own aliases, as the result helper columns of
/// `neighbors` and `lookup` are, so each answer is the result of the
/// reference statement it stands for (H9).
///
/// A `describe` answer is a catalog's, whose columns a SQL dialect's own
/// catalog fixes: a SQL plan expects the shape of the statement its dialect
/// runs, which `sdm-data` reads ([`DataLayer::read`]), and a host provider
/// answers in the dialect-neutral shape of [`catalog_columns`] (decision
/// E54).
pub mod results {
    /// `key_counts`: one row per key, its position in the request, the
    /// table's rows, the key's distinct values among rows holding all of
    /// it, and the rows missing part of it.
    pub const KEY_POSITION: &str = "k";
    pub const KEY_ROWS: &str = "n";
    pub const KEY_DISTINCT: &str = "d";
    pub const KEY_MISSING: &str = "z";
    /// `time_bounds`: one row per column, its position in the request and
    /// its smallest and largest `CAST(column AS TIMESTAMP)`, NULL when no
    /// row holds a value.
    pub const BOUNDS_POSITION: &str = "table_index";
    pub const BOUNDS_MIN: &str = "min_date";
    pub const BOUNDS_MAX: &str = "max_date";
}

/// The columns a host provider answers a `describe` with (decision E54):
/// the catalog's answer in no dialect's shape, which the data layer reads
/// into a [`CatalogAnswer`] as it is.
pub mod catalog_columns {
    /// `tables`: one row per table, its name. `indexes`: each index's
    /// table.
    pub const TABLE: &str = "table";
    /// `columns`, for one table: one row per column, in the table's order:
    /// its name, its declared type as the source spells it, whether it is
    /// declared `NOT NULL` and whether it is part of the primary key.
    /// `index_columns`, for one index: one row per column, in the index's
    /// order, its name, null for an expression.
    pub const NAME: &str = "name";
    pub const TYPE: &str = "type";
    pub const NOT_NULL: &str = "not_null";
    pub const PRIMARY_KEY: &str = "primary_key";
    /// `keys`, for one table: one row per key it declares, its kind
    /// ([`KIND_PRIMARY`], [`KIND_UNIQUE`], [`KIND_FOREIGN`] or another the
    /// reader leaves alone) and its columns. `foreign_keys` and `indexes`: the columns of
    /// each.
    pub const KIND: &str = "kind";
    pub const COLUMNS: &str = "columns";
    pub const KIND_PRIMARY: &str = "primary";
    pub const KIND_UNIQUE: &str = "unique";
    pub const KIND_FOREIGN: &str = "foreign";
    /// `foreign_keys`, for one table: one row per column pair of each
    /// foreign key it declares, the key's number naming it, the table it
    /// refers to, and the columns on both sides.
    pub const KEY: &str = "key";
    pub const REFERENCED_TABLE: &str = "referenced_table";
    pub const REFERENCED_COLUMNS: &str = "referenced_columns";
    /// The name of a list's item field in these answers.
    pub const LIST_ITEM: &str = "item";
}

impl DataRequest {
    /// The schema a host provider's answer to this request must have:
    /// fixed for the requests whose answer the contract shapes ([`results`],
    /// and [`catalog_columns`] for a `describe`), a discovery marker for
    /// those whose columns are the source's own (a preview, a scan, a
    /// sample, neighbors, a lookup, the column ranges, whose smallest and
    /// largest values are of each column's type, which the core checks the
    /// shape of itself, and a repeated key, of the key's types) and for a
    /// count no provider renders yet. A SQL dialect's plan expects the same, but for a `describe`,
    /// whose answer is its own catalog's (decision E54).
    pub fn expected(&self) -> ExpectedSchema {
        use ColumnType::{Bool, Int32, Int64, Utf8};
        use catalog_columns as c;
        let timestamp = || ColumnType::Timestamp {
            unit: Unit::Us,
            zone: None,
        };
        let names = || ColumnType::List {
            item: Box::new(field(c::LIST_ITEM, Utf8, true)),
        };
        let fields = match self {
            DataRequest::Describe { part, .. } => match part {
                DescribePart::Tables => vec![field(c::TABLE, Utf8, false)],
                DescribePart::Columns => vec![
                    field(c::NAME, Utf8, false),
                    field(c::TYPE, Utf8, false),
                    field(c::NOT_NULL, Bool, false),
                    field(c::PRIMARY_KEY, Bool, false),
                ],
                DescribePart::Keys => vec![
                    field(c::KIND, Utf8, false),
                    field(c::COLUMNS, names(), false),
                ],
                DescribePart::ForeignKeys => vec![
                    field(c::KEY, Int64, false),
                    field(c::REFERENCED_TABLE, Utf8, false),
                    field(c::COLUMNS, names(), false),
                    field(c::REFERENCED_COLUMNS, names(), false),
                ],
                DescribePart::Indexes => vec![
                    field(c::TABLE, Utf8, false),
                    field(c::COLUMNS, names(), false),
                ],
                DescribePart::IndexColumns => vec![field(c::NAME, Utf8, true)],
                DescribePart::Namespace => vec![field(c::NAME, Utf8, false)],
                // The dialect's own statement's shape: a host provider is
                // never asked for a view's definition.
                DescribePart::ViewDefinition => return ExpectedSchema::Discover {},
            },
            DataRequest::KeyCounts { .. } => vec![
                field(results::KEY_POSITION, Int32, false),
                field(results::KEY_ROWS, Int64, false),
                field(results::KEY_DISTINCT, Int64, false),
                field(results::KEY_MISSING, Int64, false),
            ],
            DataRequest::TimeBounds { .. } => vec![
                field(results::BOUNDS_POSITION, Int32, false),
                field(results::BOUNDS_MIN, timestamp(), true),
                field(results::BOUNDS_MAX, timestamp(), true),
            ],
            DataRequest::Scan { .. }
            | DataRequest::Preview { .. }
            | DataRequest::CountRows { .. }
            | DataRequest::ColumnRanges { .. }
            | DataRequest::RepeatedKey { .. }
            | DataRequest::SampleEntities { .. }
            | DataRequest::Neighbors { .. }
            | DataRequest::Lookup { .. } => return ExpectedSchema::Discover {},
        };
        ExpectedSchema::Schema { fields }
    }
}

/// A field of `name`, of `column_type`, with no metadata.
pub fn field(name: &str, column_type: ColumnType, nullable: bool) -> Field {
    Field {
        name: name.to_owned(),
        column_type,
        nullable,
        metadata: Default::default(),
    }
}

/// The Arrow schema a request expects back, or that the answer discovers
/// it (a preview, a scan).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "expect", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExpectedSchema {
    Discover {},
    Schema { fields: Vec<Field> },
}

/// A provider's failure: its kind, whether trying again may succeed, and
/// the driver's own SQLSTATE, vendor code and error class, and its message
/// as a diagnostic for the data layer alone. What the failure means is the
/// data layer's to say ([`DataLayer::failure`]), from these facts; the
/// message is never echoed into an error the core gives, since drivers
/// quote values there (decision H17).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProviderError {
    pub kind: ErrorKind,
    pub retryable: bool,
    pub sqlstate: Option<String>,
    pub vendor_code: Option<String>,
    /// The driver's error class, a name of its closed list such as
    /// Databricks's `INSUFFICIENT_PERMISSIONS` (contract v28, T3.27), read
    /// as the old client reads it (`error_class`: the driver's own answer,
    /// else the bracketed name its message opens with). Left out, none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<String>,
    /// The driver's message, passed through by the host unread (contract
    /// v28, T3.6): an internal diagnostic only [`DataLayer::failure`] reads,
    /// where the old client reads a failure by its text (Snowflake's
    /// "does not exist", `SNOW/table.py:34-45`), and never echoed into a
    /// public error (decision H17). Left out, none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_message: Option<String>,
}
