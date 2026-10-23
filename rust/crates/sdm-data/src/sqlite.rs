// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQLite dialect (T2.2). Its capabilities are the optional requests it
//! has a rendering for, so a request it cannot render is refused by name
//! before it is planned.
//!
//! Each rendering is the statement the reference's SQLite backend runs for
//! the same step (`SQLiteTable` and `SQLiteSampler` at the reference
//! commit), so each answer is that statement's result (decision H9): the
//! table list from `sqlite_master`, `PRAGMA table_info`, `index_list`,
//! `index_info` and `foreign_key_list`, the first rows in `rowid` order,
//! the key counts' and time bounds' single `UNION ALL` (the time bounds
//! over the stored values, SQLite having no time type), the seeded entity
//! draw of register RS-2, and the neighbor and key reads by `rowid`. Their
//! texts differ from the reference's only where the reference names what
//! the core names otherwise: the key set's relation and columns (the
//! reference's `tmp_<table>_<key>_<address>` and `__kumo_` columns), read
//! here from the key set's own columns, and the source's columns, read
//! from the table's alias.
//!
//! A key set is bound by the provider as a temporary table of the name its
//! parameter gives, one per key set of the run ([`seeds_for`]); the
//! provider creates it, replacing only a table of that name it made itself
//! on the connection, never a relation the caller made. Its columns are
//! those of the key set as ADBC's SQLite driver ingests Arrow
//! (`adbc_ingest(..., temporary=True)`): integers as `INTEGER`, floats as `REAL`,
//! text as `TEXT`, bytes as `BLOB`, a timestamp as ISO 8601 text with a `T`
//! and as many fractional digits as its unit holds, so a nanosecond bound
//! is `YYYY-MM-DDTHH:MM:SS.fffffffff`. The reference ingests its key tables
//! into the source's own database and leaves them there; a temporary table
//! is the connection's own and leaves the database as it was.
//!
//! SQLite stores a time as the text it was written as, and the reference
//! compares a window's bounds with that text, spelled as one stored value
//! is (`_time_spelling`): the layer says so ([`compares_stored_times`]),
//! the core reads the value and names it on each bound's seed column
//! ([`SEED_STORED_TIME`]), and the rendering spells the bound from its
//! ingested text the same way.
//!
//! The statements are built once, by the shared builder ([`crate::sql`]),
//! through [`Sqlite`]'s hooks (T3.18): only what is SQLite's own is here.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use sdm_contract::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, IndexCoverage,
    KeyKind,
};
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, ExpectedSchema, IndexAdvice, KeySet,
    MARKED, MATCHES_COLUMN, NullOrder, ProviderCapabilities, Relation, SEED_BATCH,
    SEED_COUNT_UNREAD, SEED_KEY_UNIQUE, SEED_STORED_TIME, TextTimes, TieTerm, field, seed_integer,
    seed_key_columns,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{ColumnType, CoreError, Detail, Scalar, Sentence};

use crate::mt19937::{HASH_PRIME, coefficients};
use crate::quote::{identifier, text};
use crate::sql::{self, Sample, SeedRead, SqlDialect};
use sdm_contract::data::SqlParam;

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "sqlite";

/// The columns of SQLite's catalog answers, its pragmas' own.
mod catalog {
    /// `sqlite_master`'s table name; `table_info`'s and `index_info`'s
    /// column name; `index_list`'s index name.
    pub const NAME: &str = "name";
    /// `table_info`: each column's declared type, whether it is declared
    /// `NOT NULL`, and its position in the primary key (0 for none).
    pub const TYPE: &str = "type";
    pub const NOTNULL: &str = "notnull";
    pub const PK: &str = "pk";
    /// `index_list`: whether each index is unique, and whether it is
    /// partial.
    pub const UNIQUE: &str = "unique";
    pub const PARTIAL: &str = "partial";
    /// `index_info`: each column's position in the index.
    pub const SEQNO: &str = "seqno";
    /// `foreign_key_list`: each key's number, the table it refers to, and
    /// each pair's column and the column referred to.
    pub const ID: &str = "id";
    pub const TABLE: &str = "table";
    pub const FROM: &str = "from";
    pub const TO: &str = "to";
}

/// The most name parts a SQLite relation takes: a schema (`main`, `temp`
/// or an attached database) and a table (design choice 12).
const MOST_PARTS: usize = 2;

/// The prefix of the name a statement reads its seeds by: the key set the
/// provider binds as a temporary table, read as `temp`'s.
const SEEDS: &str = "__sdm_seeds";

/// The relation a statement answering `request`, which joins `table`, reads
/// its key set by: [`SEEDS`] and the 64-bit FNV-1a digest of the request as
/// the contract writes it, in 16 hex digits (`__sdm_seeds_3f2a9c0d1e4b5a67__`),
/// so each key set of a run has its own name (its attachment is in the
/// request), both hosts and the goldens render one text, and no name a
/// caller would write (`__sdm_seeds__`, `__sdm_seeds_1__`) is the
/// provider's. The reference names its key tables by an object's address,
/// which no caller predicts either. Lengthened by underscores until it is
/// not `table`'s last part in any case (SQLite matches ASCII names without
/// case), which a source table of the same name would hide from an
/// unqualified name.
fn seeds_for(request: &DataRequest, table: &Relation) -> String {
    // FNV-1a: no digest crate shipped for a name no caller writes.
    let written = serde_json::to_vec(request).unwrap_or_default();
    let digest = written
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        });
    let last = table.source.last().map(String::as_str).unwrap_or_default();
    let mut name = format!("{SEEDS}_{digest:016x}__");
    while name.eq_ignore_ascii_case(last) {
        name.push('_');
    }
    name
}

/// The pragma `name` of `relation`: `PRAGMA name("t")`, or for a relation
/// naming its schema, `PRAGMA "s".name("t")`, as SQLite reads a pragma of
/// another schema.
fn pragma(
    request: &DataRequest,
    field: &str,
    name: &str,
    relation: &Relation,
) -> Result<String, CoreError> {
    sql::from(&Sqlite, request, field, relation)?;
    Ok(match relation.source.as_slice() {
        [schema, table] => format!(
            "PRAGMA {}.{name}({})",
            identifier(schema),
            identifier(table)
        ),
        parts => format!(
            "PRAGMA {name}({})",
            identifier(parts.first().map(String::as_str).unwrap_or_default())
        ),
    })
}

/// The SQLite dialect.
pub(crate) struct Sqlite;

impl SqlDialect for Sqlite {
    fn name(&self) -> &'static str {
        NAME
    }

    /// What a SQLite source answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            opaque_vendor: None, reads_nan_as_missing: false,
            derived_relations: false, neighbor_lookback: false, sample_rows: false,
            optional: sql::rendered(),
            max_keys_per_request: None,
            // The seeded draw of RS-2 repeats under a seed.
            sampling: Determinism::SeededStable,
            // SQLite takes a seed of any size (register RS-7); the largest the
            // contract carries.
            max_seed: u64::MAX,
            // A SQL source's text times, read one by one (register DT-7).
            text_times: Some(TextTimes::ValueByValue),
            // SQLite orders a null as the smallest value. The core asks for
            // nulls first; a descending term on the window's column, the one
            // the reference orders descending, holds no null under the window,
            // and any other is written as SQLite orders it.
            null_order: Some(NullOrder::First),
            // The reference's SQLite sampler reads each table's indexes and
            // counts an index as serving a read when its first column is the
            // read's (`index_info[0] == index[0]`).
            index_advice: Some(IndexAdvice::LeadingColumn),
            // Every optional request is rendered: none falls back.
            fallback: None,
            // A time is stored as text, compared as stored (`_time_spelling`).
            compares_stored_times: true,
            // The seeded draw hashes an integer key ([`draw_order`]).
            samples_by_key_dtype: true,
        }
    }

    /// A schema (`main`, `temp` or an attached database) and a table, each
    /// quoted alone, never split on a dot, so `["a.b"]` is `"a.b"` and
    /// `["main", "t"]` is `"main"."t"`.
    fn most_parts(&self) -> usize {
        MOST_PARTS
    }

    fn seeds_for(&self, request: &DataRequest, table: &Relation) -> String {
        seeds_for(request, table)
    }

    /// The temporary table the provider binds, read as `temp`'s.
    fn seeds_relation(&self, name: &str) -> String {
        format!("temp.{}", identifier(name))
    }

    /// The bound spelled as the source's stored value ([`bound`]).
    fn window_bound(&self, seeds: &KeySet, seed_column: &str) -> String {
        bound(seeds, seed_column)
    }

    /// The first rows in `rowid` order, as the reference reads them.
    fn preview_order(&self) -> &'static str {
        " ORDER BY rowid"
    }

    /// Counted in a subquery, since SQLite counts one column distinct at a
    /// time (`_count_distinct`).
    fn distinct_keys(&self, quoted: &[String], table: &str) -> String {
        format!(
            "(SELECT COUNT(*) FROM (SELECT DISTINCT {} FROM {table} WHERE {}))",
            quoted.join(", "),
            quoted
                .iter()
                .map(|c| format!("{c} IS NOT NULL"))
                .collect::<Vec<_>>()
                .join(" AND ")
        )
    }

    fn order_term(&self, term: &TieTerm, alias: &str, windowed: Option<&str>) -> String {
        order_term(term, alias, windowed)
    }

    /// A non-negative integer of any size (register RS-7); the core refuses
    /// none of them first.
    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError> {
        match seed_integer(seed) {
            Some(seed) if u128::try_from(seed).is_ok() => Ok(seed),
            _ => Err(sdm_contract::data::refused(
                request,
                "seed",
                "a sample's seed is a non-negative integer",
            )),
        }
    }

    /// The reference's seeded entity draw on SQLite (`_sample_entity_table`
    /// with a seed, register DF-2 and RS-2): the columns asked for, from the
    /// rows that meet each not-null filter, the `n` rows that hash lowest.
    /// The reference hashes an integer key, its rows sharing a key then in
    /// `rowid` order, and the `rowid` of a table with another kind of key,
    /// which the draw order decides ([`draw_order`]); a key of several
    /// columns hashes the `rowid`.
    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        // Non-negative: the seed is taken only so (`sample_seed`).
        let seed = u128::try_from(sample.seed).unwrap_or_default();
        let order = match (
            sample.key,
            draw_order(sample.relation, sample.key, sample.key_dtype),
        ) {
            ([one], DrawOrder::Key(integer)) => {
                let key = identifier(one);
                format!(
                    "CASE WHEN {integer} THEN {} ELSE {} END, CASE WHEN {integer} THEN {key} END, rowid",
                    seeded_hash(seed, &key),
                    seeded_hash(seed, "rowid"),
                )
            }
            _ => format!("{}, rowid", seeded_hash(seed, "rowid")),
        };
        vec![format!(
            "SELECT {}\nFROM {}{}\nORDER BY {order} LIMIT {}",
            sample.projected, sample.table, sample.filtered, sample.n
        )]
    }

    /// The reference's neighbor reads on SQLite (`_by_time` and `_by_fkey`)
    /// with a limit per seed: the first rows of each seed in `order`, read
    /// by `rowid`, the result in `(batch, order)`.
    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String {
        let windowed = read.window.map(|w| w.column.as_str());
        let ordered = |alias: &str| {
            terms
                .iter()
                .map(|t| order_term(t, alias, windowed))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "SELECT {}, {}\nFROM {} s\nJOIN {table} t\nON t.rowid IN (\n  SELECT rowid\n  FROM {table}\n  WHERE {}\n  ORDER BY {}\n  LIMIT {limit}\n)\nORDER BY {}, {}",
            read.batch(),
            read.projected(),
            read.seeds(),
            read.within("").join("\n  AND "),
            ordered(""),
            identifier(BATCH_COLUMN),
            ordered("t."),
            table = read.table,
        )
    }

    /// The reference's key lookup on SQLite (`_by_pkey`): one row of the
    /// table per seed whose key it holds. A key the source enforces unique
    /// is joined plainly ([`SEED_KEY_UNIQUE`]); any other by the `rowid` of
    /// its first row, with the count of rows holding it unless the seeds say
    /// it is not read ([`SEED_COUNT_UNREAD`]), as the reference counts only
    /// where it scores.
    fn lookup(&self, read: &SeedRead<'_>) -> String {
        let unique = seed_key_columns(read.key.len())
            .iter()
            .all(|seed| marked(read.seeds, seed, SEED_KEY_UNIQUE));
        if unique {
            return read.joined();
        }
        let table = &read.table;
        let matching = read.seed_join("", "==");
        let counted = if marked(read.seeds, SEED_BATCH, SEED_COUNT_UNREAD) {
            String::new()
        } else {
            format!(
                "(SELECT COUNT(*) FROM {table} WHERE {matching}) AS {}, ",
                identifier(MATCHES_COLUMN)
            )
        };
        format!(
            "SELECT {}, {counted}{}\nFROM {} s\nJOIN {table} t\n  ON t.rowid = (\n    SELECT rowid\n    FROM {table}\n    WHERE {matching}\n    LIMIT 1\n)",
            read.batch(),
            read.projected(),
            read.seeds(),
        )
    }

    fn describe(
        &self,
        request: &DataRequest,
        tables: &[Relation],
        part: DescribePart,
        _namespace: &[String],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        Ok((describe(request, tables, part)?, vec![]))
    }

    fn expected(&self, request: &DataRequest) -> ExpectedSchema {
        expected(request)
    }

    fn read(
        &self,
        request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        read(request, part, batches)
    }


}

/// The schema SQLite's answer to `request` has: the key counts' four
/// integers, which no value of the source can retype; discovered for every
/// other answer, whose column types SQLite's driver infers from the values
/// (an empty or all-null column reading as integers), and which the core
/// or this layer's reader checks itself.
fn expected(request: &DataRequest) -> ExpectedSchema {
    use sdm_contract::data::results::{KEY_DISTINCT, KEY_MISSING, KEY_POSITION, KEY_ROWS};
    match request {
        DataRequest::KeyCounts { .. } => ExpectedSchema::Schema {
            fields: [KEY_POSITION, KEY_ROWS, KEY_DISTINCT, KEY_MISSING]
                .into_iter()
                .map(|name| field(name, ColumnType::Int64, false))
                .collect(),
        },
        _ => ExpectedSchema::Discover {},
    }
}

/// One part of the catalog, read as the reference reads it on SQLite: the
/// table list whole, from `sqlite_master`; a table's columns, its indexes
/// (whose unique ones are its keys) and its foreign keys by their pragmas,
/// one table at a time; and an index's columns by `index_info`, one index
/// at a time, the index named by a relation as a table is.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
) -> Result<String, CoreError> {
    let one = || sql::one_table(request, tables);
    Ok(match part {
        DescribePart::Tables if tables.is_empty() => {
            "SELECT name FROM sqlite_master WHERE type='table'".to_owned()
        }
        DescribePart::Tables => held(tables),
        DescribePart::Columns => pragma(request, "tables", "table_info", one()?)?,
        DescribePart::Keys | DescribePart::Indexes => {
            pragma(request, "tables", "index_list", one()?)?
        }
        DescribePart::ForeignKeys => pragma(request, "tables", "foreign_key_list", one()?)?,
        DescribePart::IndexColumns => pragma(request, "tables", "index_info", one()?)?,
        // No row: the reference names a SQLite table it cannot find alone.
        DescribePart::Namespace => "SELECT CAST(NULL AS TEXT) AS name WHERE 0".to_owned(),
    })
}

/// The tables or views SQLite would read under any of `relations`: a name
/// of one part, as SQLite resolves an unqualified name, in any schema
/// (`temp`, `main` or an attached database), its ASCII letters matched
/// without case; a name of two parts, in the schema it names. No row says
/// the source holds no such relation. Asked only when reading a named
/// table's columns failed, which `PRAGMA table_info` never does for a
/// table that is not there (it answers no row, which the core refuses).
fn held(relations: &[Relation]) -> String {
    let selects: Vec<String> = relations
        .iter()
        .map(|relation| {
            let named = match relation.source.as_slice() {
                [schema, table] => format!(
                    "schema = {} COLLATE NOCASE AND name = {} COLLATE NOCASE",
                    text(schema),
                    text(table)
                ),
                parts => format!(
                    "name = {} COLLATE NOCASE",
                    text(parts.first().map(String::as_str).unwrap_or_default())
                ),
            };
            format!(
                "SELECT name FROM pragma_table_list WHERE type IN ('table', 'view') AND {named}"
            )
        })
        .collect();
    selects.join("\nUNION ALL\n")
}

/// The seeded hash of `reference`, an integer expression, under the
/// coefficients of `seed` (register RS-2, the reference's `_seeded_hash`):
/// a cubic polynomial modulo 2**31 - 1, written as the reference writes it.
fn seeded_hash(seed: u128, reference: &str) -> String {
    let p = HASH_PRIME;
    let [c0, c1, c2, c3] = coefficients(seed);
    let x = format!("(({reference} % {p} + {p}) % {p})");
    format!("(((({c3} * {x} + {c2}) % {p}) * {x} + {c1}) % {p} * {x} + {c0}) % {p}")
}

/// Whether the seeded draw hashes a key of one column rather than the
/// `rowid`, as the reference's `_seeded_order_by` decides it: by the type
/// the key's declaration fixes (`_to_dtype`: one holding `INT` an integer,
/// `TEXT`, `CHAR` or `CLOB` text, `REAL`, `FLOA` or `DOUB` a float, case
/// aside), read from the table's `pragma_table_info` in the statement,
/// once for the column; and where it fixes none, by `key_dtype`, the
/// schema's type the core names (an integer type hashes the key). Without
/// `key_dtype` (a request that leaves it out), each row decides by its
/// value's storage class. `None` for a key of several columns, which
/// hashes the `rowid`.
enum DrawOrder {
    Key(String),
    Rowid,
}

fn draw_order(table: &Relation, key: &[String], key_dtype: Option<Dtype>) -> DrawOrder {
    let [one] = key else {
        return DrawOrder::Rowid;
    };
    let column = identifier(one);
    let Some(dtype) = key_dtype else {
        return DrawOrder::Key(format!("typeof({column}) = 'integer'"));
    };
    let integer = matches!(
        dtype,
        Dtype::Int | Dtype::Byte | Dtype::Int16 | Dtype::Int32 | Dtype::Int64
    );
    let arguments = match table.source.as_slice() {
        [schema, name] => format!("{}, {}", text(name), text(schema)),
        parts => text(parts.last().map(String::as_str).unwrap_or_default()),
    };
    DrawOrder::Key(format!(
        "COALESCE((SELECT CASE WHEN instr(t, 'INT') > 0 THEN 1 WHEN instr(t, 'TEXT') + instr(t, 'CHAR') + instr(t, 'CLOB') + instr(t, 'REAL') + instr(t, 'FLOA') + instr(t, 'DOUB') > 0 THEN 0 END FROM (SELECT upper(trim(type)) AS t FROM pragma_table_info({arguments}) WHERE name = {})), {}) = 1",
        text(one),
        u8::from(integer),
    ))
}

/// How the source spells its stored times, as the reference reads it from
/// one stored value (`_TimeSpelling`, `_probe_time_spelling`): the date
/// and time's separator, the number of fractional digits, and a suffix,
/// kept only when it names UTC, the bounds' zone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spelling {
    separator: char,
    digits: usize,
    suffix: String,
}

impl Default for Spelling {
    /// `YYYY-MM-DD HH:MM:SS`, which `datetime()` writes, and what the
    /// reference takes when nothing is read.
    fn default() -> Self {
        Spelling {
            separator: ' ',
            digits: 0,
            suffix: String::new(),
        }
    }
}

impl Spelling {
    /// The spelling `stored`, one stored value, shows: the reference's
    /// `^\d{4}-\d{2}-\d{2}([T ])\d{2}:\d{2}:\d{2}(?:\.(\d+))?(.*)$`, its
    /// suffix kept when it is one of `''`, `Z`, `+00:00` and `+0000`; the
    /// default for anything else.
    pub(crate) fn of(stored: &str) -> Spelling {
        Spelling::read(stored).unwrap_or_default()
    }

    fn read(stored: &str) -> Option<Spelling> {
        let bytes = stored.as_bytes();
        let digits = |range: std::ops::Range<usize>| {
            bytes
                .get(range)
                .is_some_and(|b| b.iter().all(u8::is_ascii_digit))
        };
        let at = |i: usize, c: u8| bytes.get(i) == Some(&c);
        if !(digits(0..4)
            && at(4, b'-')
            && digits(5..7)
            && at(7, b'-')
            && digits(8..10)
            && (at(10, b'T') || at(10, b' '))
            && digits(11..13)
            && at(13, b':')
            && digits(14..16)
            && at(16, b':')
            && digits(17..19))
        {
            return None;
        }
        let separator = char::from(bytes[10]);
        let mut rest = &stored[19..];
        let mut fraction = 0;
        if let Some(after) = rest.strip_prefix('.') {
            let count = after.bytes().take_while(u8::is_ascii_digit).count();
            if count > 0 {
                fraction = count;
                rest = &after[count..];
            }
        }
        // `(.*)$`: a dot matches no line break, and `$` matches before a
        // last one.
        let rest = rest.strip_suffix('\n').unwrap_or(rest);
        if rest.contains('\n') {
            return None;
        }
        let suffix = match rest {
            "" | "Z" | "+00:00" | "+0000" => rest.to_owned(),
            _ => String::new(),
        };
        Some(Spelling {
            separator,
            digits: fraction,
            suffix,
        })
    }

    /// The SQL text of `bound`, a seed column bound as ADBC ingests a
    /// nanosecond timestamp (`YYYY-MM-DDTHH:MM:SS.fffffffff`), spelled
    /// this way: its date, the separator, its time, its fraction cut or
    /// padded with zeros to the digits, and the suffix (`_TimeSpelling.spell`).
    fn spell(&self, bound: &str) -> String {
        let mut parts = vec![
            format!("substr({bound}, 1, 10)"),
            text(&self.separator.to_string()),
            format!("substr({bound}, 12, 8)"),
        ];
        if self.digits > 0 {
            parts.push("'.'".to_owned());
            parts.push(format!("substr({bound}, 21, {})", self.digits.min(9)));
            if self.digits > 9 {
                parts.push(text(&"0".repeat(self.digits - 9)));
            }
        }
        if !self.suffix.is_empty() {
            parts.push(text(&self.suffix));
        }
        parts.join(" || ")
    }
}

/// The SQL text of the seed column `name` as a bound of the window: the
/// bound spelled as the source's stored value the core read and named on
/// it ([`SEED_STORED_TIME`]), or as the reference spells one when it read
/// none.
fn bound(seeds: &KeySet, name: &str) -> String {
    let column = format!("s.{}", identifier(name));
    let stored = seeds
        .columns
        .iter()
        .find(|f| f.name == name)
        .and_then(|f| f.metadata.get(SEED_STORED_TIME));
    stored
        .map_or_else(Spelling::default, |s| Spelling::of(s))
        .spell(&column)
}

/// One term of a neighbor order, as SQLite orders it: a time's stored text
/// orders as its time, so a term asked as a timestamp is the column. SQLite
/// orders a null as the smallest value; a term whose nulls go elsewhere
/// says so, but for the window's own column, which holds no null under the
/// window.
fn order_term(term: &TieTerm, alias: &str, windowed: Option<&str>) -> String {
    let column = format!("{alias}{}", identifier(&term.column));
    let direction = if term.descending { " DESC" } else { "" };
    let natural = term.nulls_last == term.descending;
    let nulls = if natural || windowed == Some(term.column.as_str()) {
        ""
    } else if term.nulls_last {
        " NULLS LAST"
    } else {
        " NULLS FIRST"
    };
    format!("{column}{direction}{nulls}")
}

/// Whether the seeds' column `name` carries the flag `key`.
fn marked(seeds: &KeySet, name: &str, key: &str) -> bool {
    seeds
        .columns
        .iter()
        .find(|f| f.name == name)
        .is_some_and(|f| f.metadata.get(key).map(String::as_str) == Some(MARKED))
}

/// The data type a SQLite column's declared type fixes, as the reference
/// reads it (`SQLiteTable._to_dtype`, SQLite's affinity rules in their
/// order): `INT` an integer, `TEXT`, `CHAR` or `CLOB` text, `REAL`, `FLOA`
/// or `DOUB` a float; none for any other (numeric affinity) and for an
/// `INTERVAL` (KO2-62), whose values say.
fn declared_dtype(declared: &str) -> Option<Dtype> {
    let declared = declared.trim().to_ascii_uppercase();
    if declared.contains("INTERVAL") {
        // The owner's KO2-62 (register P2B-14): an INTERVAL column, whose
        // declared name the reference reads as an integer, is typed by its
        // values.
        None
    } else if declared.contains("INT") {
        Some(Dtype::Int)
    } else if ["TEXT", "CHAR", "CLOB"]
        .iter()
        .any(|t| declared.contains(t))
    {
        Some(Dtype::String)
    } else if ["REAL", "FLOA", "DOUB"]
        .iter()
        .any(|t| declared.contains(t))
    {
        Some(Dtype::Float)
    } else {
        None
    }
}

// --- reading the catalog's answers -------------------------------------------------

fn unread(name: &str, message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message).with_field("column", Detail::text(name))
}

/// The column `name` of `batch`, which SQLite's driver types by its values:
/// `None` for each row where every value is null, whatever its type.
fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a dyn Array>, CoreError> {
    let column = batch.column_by_name(name).ok_or_else(|| {
        unread(
            name,
            "a catalog answer lacks a column SQLite's pragma gives",
        )
    })?;
    if column.null_count() == column.len() {
        return Ok(None);
    }
    Ok(Some(column.as_ref()))
}

/// The text column `name`: each row's text, `None` where it is null.
fn texts(batch: &RecordBatch, name: &str) -> Result<Vec<Option<String>>, CoreError> {
    let Some(column) = column(batch, name)? else {
        return Ok(vec![None; batch.num_rows()]);
    };
    let Some(strings) = column.as_string_opt::<i32>() else {
        return Err(unread(name, "a catalog answer's column is not text"));
    };
    Ok((0..strings.len())
        .map(|i| (!strings.is_null(i)).then(|| strings.value(i).to_owned()))
        .collect())
}

/// The integer column `name`: each row's integer, `None` where it is null.
fn integers(batch: &RecordBatch, name: &str) -> Result<Vec<Option<i64>>, CoreError> {
    let Some(column) = column(batch, name)? else {
        return Ok(vec![None; batch.num_rows()]);
    };
    let Some(values) = column.as_primitive_opt::<Int64Type>() else {
        return Err(unread(name, "a catalog answer's column is not an integer"));
    };
    Ok((0..values.len())
        .map(|i| (!values.is_null(i)).then(|| values.value(i)))
        .collect())
}

/// SQLite's answer to a `describe` of `part`, asked by `request`, read
/// into the contract's catalog answer.
fn read(
    request: &DataRequest,
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    use catalog as c;
    match part {
        DescribePart::Namespace => crate::read::namespace(batches, c::NAME),
        DescribePart::Tables => {
            let mut found = Vec::new();
            for batch in batches {
                found.extend(texts(batch, c::NAME)?.into_iter().flatten());
            }
            Ok(CatalogAnswer::Tables(found))
        }
        DescribePart::Columns => {
            let mut read = Vec::new();
            for batch in batches {
                let names = texts(batch, c::NAME)?;
                let types = texts(batch, c::TYPE)?;
                let not_null = integers(batch, c::NOTNULL)?;
                let pk = integers(batch, c::PK)?;
                for row in 0..batch.num_rows() {
                    if let Some(name) = names[row].clone() {
                        read.push((
                            name,
                            types[row].clone().unwrap_or_default(),
                            not_null[row].is_some_and(|n| n != 0),
                            pk[row].is_some_and(|p| p > 0),
                        ));
                    }
                }
            }
            // The rowid alias as the reference reads it (`SQLiteTable`'s
            // rowid candidates): the one primary key column declared
            // `INTEGER`, kept unique and never null. SQLite makes only a
            // primary key of that one column the alias; the reference takes
            // a composite key's one `INTEGER` column too, and so does this.
            let candidates: Vec<usize> = (0..read.len())
                .filter(|&i| read[i].3 && read[i].1.trim().eq_ignore_ascii_case("INTEGER"))
                .collect();
            let rowid = match candidates.as_slice() {
                [only] => Some(*only),
                _ => None,
            };
            Ok(CatalogAnswer::Columns(
                read.into_iter()
                    .enumerate()
                    .map(
                        |(at, (name, declared_type, not_null, primary_key))| CatalogColumn {
                            canonical: None,
                            name,
                            dtype: declared_dtype(&declared_type),
                            declared_type,
                            not_null: not_null || rowid == Some(at),
                            key_nullable: primary_key,
                            primary_key,
                            unique: rowid == Some(at),
                        },
                    )
                    .collect(),
            ))
        }
        DescribePart::Keys => {
            // Each unique index is a key, its columns read by its name; a
            // partial one too, which keeps a column of its own unique as
            // the reference reads it, but is no key of several columns.
            let mut found = Vec::new();
            for batch in batches {
                let names = texts(batch, c::NAME)?;
                let unique = integers(batch, c::UNIQUE)?;
                let partial = integers(batch, c::PARTIAL)?;
                for row in 0..batch.num_rows() {
                    if let (Some(name), Some(1..)) = (&names[row], unique[row]) {
                        found.push(CatalogKey {
                            kind: if partial[row].is_some_and(|p| p != 0) {
                                KeyKind::PartialUnique
                            } else {
                                KeyKind::Unique
                            },
                            columns: vec![],
                            index: Some(name.clone()),
                        });
                    }
                }
            }
            Ok(CatalogAnswer::Keys(found))
        }
        DescribePart::ForeignKeys => {
            let mut found = Vec::new();
            for batch in batches {
                let ids = integers(batch, c::ID)?;
                let tables = texts(batch, c::TABLE)?;
                let froms = texts(batch, c::FROM)?;
                let tos = texts(batch, c::TO)?;
                for row in 0..batch.num_rows() {
                    found.push(CatalogForeignKey {
                        key: ids[row].unwrap_or_default(),
                        referenced_table: tables[row].clone().unwrap_or_default(),
                        columns: froms[row].clone().into_iter().collect(),
                        referenced_columns: tos[row].clone().into_iter().collect(),
                    });
                }
            }
            Ok(CatalogAnswer::ForeignKeys(found))
        }
        DescribePart::Indexes => {
            let DataRequest::Describe { tables, .. } = request else {
                return Err(unread("name", "an index list answered another request"));
            };
            let table = match tables.as_slice() {
                [table] => table.source.last().cloned().unwrap_or_default(),
                _ => return Err(unread("name", "an index list read no one table")),
            };
            let mut found = Vec::new();
            for batch in batches {
                for name in texts(batch, c::NAME)?.into_iter().flatten() {
                    found.push(CatalogIndex {
                        table: table.clone(),
                        columns: IndexCoverage::Named(name),
                    });
                }
            }
            Ok(CatalogAnswer::Indexes(found))
        }
        DescribePart::IndexColumns => {
            let mut found: Vec<(i64, Option<String>)> = Vec::new();
            for batch in batches {
                let positions = integers(batch, c::SEQNO)?;
                let names = texts(batch, c::NAME)?;
                for row in 0..batch.num_rows() {
                    found.push((positions[row].unwrap_or_default(), names[row].clone()));
                }
            }
            // In the index's order, as the reference sorts it.
            found.sort_by_key(|(position, _)| *position);
            Ok(CatalogAnswer::IndexColumns(
                found.into_iter().map(|(_, name)| name).collect(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_time_s_spelling_is_read_as_the_reference_reads_it() {
        let spelled = |s: &str| {
            let s = Spelling::of(s);
            (s.separator, s.digits, s.suffix)
        };
        assert_eq!(spelled("2024-01-01 00:00:00"), (' ', 0, String::new()));
        assert_eq!(
            spelled("2024-01-01T00:00:00.000000"),
            ('T', 6, String::new())
        );
        assert_eq!(spelled("2024-01-01 00:00:00.5Z"), (' ', 1, "Z".to_owned()));
        assert_eq!(
            spelled("2024-01-01 00:00:00+00:00"),
            (' ', 0, "+00:00".to_owned())
        );
        assert_eq!(
            spelled("2024-01-01 00:00:00+02:00"),
            (' ', 0, String::new())
        );
        assert_eq!(spelled("2024-01-01 00:00:00.Z"), (' ', 0, String::new()));
        assert_eq!(spelled("2024-01-01"), (' ', 0, String::new()));
        assert_eq!(spelled("2024-01-01 00:00:00Z\n"), (' ', 0, "Z".to_owned()));
        assert_eq!(spelled("2024-01-01 00:00:00Z\nx"), (' ', 0, String::new()));
    }

    #[test]
    fn a_bound_is_spelled_from_its_ingested_text() {
        let spelling = Spelling {
            separator: 'T',
            digits: 11,
            suffix: "Z".to_owned(),
        };
        assert_eq!(
            spelling.spell("s.\"end\""),
            "substr(s.\"end\", 1, 10) || 'T' || substr(s.\"end\", 12, 8) || '.' || substr(s.\"end\", 21, 9) || '00' || 'Z'"
        );
        assert_eq!(
            Spelling::default().spell("b"),
            "substr(b, 1, 10) || ' ' || substr(b, 12, 8)"
        );
    }

    #[test]
    fn the_reference_s_hash_of_its_default_seed() {
        assert_eq!(
            seeded_hash(42, "\"user_id\""),
            "((((787846414 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1935803228) % 2147483647) * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1273642419) % 2147483647 * ((\"user_id\" % 2147483647 + 2147483647) % 2147483647) + 1608637542) % 2147483647"
        );
    }
}
