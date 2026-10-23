// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DuckDB dialect. Its capabilities are the optional requests it has a
//! rendering for, so a request it cannot render is refused by name before
//! it is planned.
//!
//! As built at T1.11b it renders the catalog: `describe` in each part,
//! `preview`, `key_counts` and `time_bounds`. Each rendering is the
//! statement the reference runs for the same step on DuckDB, written the
//! same way, so each answer is that statement's result (decision H9) and
//! the driver capture can hold the two texts equal: the table list, `PRAGMA
//! table_info`, the constraints and foreign keys of one table, the
//! indexes, the first 1,000 rows, the key counts' and the time bounds'
//! single `UNION ALL` (the T1.11a row of the design maps each). Names are
//! quoted as the reference quotes them: an identifier in double quotes and
//! a table name compared as text in single quotes, each with its quote
//! doubled, which DuckDB's standard-conforming strings make sufficient.
//!
//! From T1.11c it renders the sampler's requests too: `sample_entities` as
//! the reference's reservoir sample, text for text, and `neighbors` and
//! `lookup` as the reference's joins against a key set, which the provider
//! binds as the relation the plan names ([`SEEDS`], or longer where the
//! table joined has that name). Their texts are not
//! the reference's: its key table's name is drawn from an object's address,
//! and its columns are its own `__kumo_` names; the seeds' role-named
//! columns (`key`, `batch`) could meet a source column's name, so the
//! table's columns are read from its alias. What each returns is the
//! reference statement's result, its helper columns under the reference's
//! names (decision H9 gates on the results, not the text).
//!
//! From T1.32e it renders `column_ranges`, the reference's one query over
//! every row of a table for its time column's choice, and `repeated_key`,
//! its read of one value a key repeats, each in its text.
//!
//! From T1.21c it reads its catalog's answers too ([`read`], decision E54):
//! the columns of `PRAGMA table_info`, `duckdb_tables()`,
//! `duckdb_constraints()` and `duckdb_indexes()`, which its plans expect
//! ([`expected`]), its constraint types and its index expressions, read
//! into the contract's dialect-neutral catalog answer, so the core reads
//! no DuckDB.
//!
//! The statements are built once, by the shared builder ([`crate::sql`]),
//! through [`Duckdb`]'s hooks (T3.18): only what is DuckDB's own is here.

use arrow_array::RecordBatch;
use sdm_contract::catalog::{CatalogAnswer, CatalogIndex, IndexCoverage, KeyKind};
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, ExpectedSchema, IndexAdvice,
    MATCHES_COLUMN, NullOrder, ProviderCapabilities, Relation, SEED_BATCH, SEED_KEY_TYPE,
    TextTimes, TieTerm, field, seed_integer,
};
use sdm_contract::{ColumnType, CoreError, Detail, Scalar, Sentence};

use crate::quote::{identifier, text};
use crate::read::{self, ColumnNames, ForeignKeyNames};
use crate::sql::{self, Sample, SeedRead, SqlDialect};
use sdm_contract::data::SqlParam;

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "duckdb";

/// The DuckDB dialect.
pub(crate) struct Duckdb;

/// The columns of DuckDB's catalog answers: what the statements it renders
/// for `describe` return, the reference's statements' own.
mod catalog {
    /// `duckdb_tables()` and `duckdb_indexes()`: the table's name.
    pub const TABLE_NAME: &str = "table_name";
    /// `PRAGMA table_info`: each column's position, name, declared type,
    /// whether it is declared `NOT NULL`, its default and whether it is
    /// part of the primary key.
    pub const CID: &str = "cid";
    pub const NAME: &str = "name";
    pub const TYPE: &str = "type";
    pub const NOTNULL: &str = "notnull";
    pub const DFLT_VALUE: &str = "dflt_value";
    pub const PK: &str = "pk";
    /// `duckdb_constraints()`: each constraint's type and columns, and for
    /// a foreign key its index, the table referred to and its columns.
    pub const CONSTRAINT_TYPE: &str = "constraint_type";
    pub const CONSTRAINT_COLUMN_NAMES: &str = "constraint_column_names";
    pub const CONSTRAINT_INDEX: &str = "constraint_index";
    pub const REFERENCED_TABLE: &str = "referenced_table";
    pub const REFERENCED_COLUMN_NAMES: &str = "referenced_column_names";
    /// `duckdb_indexes()`: each index's expressions, as one text.
    pub const EXPRESSIONS: &str = "expressions";
    /// The item field of DuckDB's lists.
    pub const LIST_ITEM: &str = "l";
}

/// The name a statement reads its seeds by: the key set the provider binds
/// for it (registered as a relation, on the statement's own cursor), unless
/// the table it joins has that name ([`seeds_for`]).
const SEEDS: &str = "__sdm_seeds__";

/// The relation a statement joining `table` reads its seeds by: [`SEEDS`],
/// lengthened by underscores until it is not the name of `table`'s table
/// (its last part) in any case (DuckDB matches names without case), so a
/// source table of any name is read, never shadowed by the seeds bound on
/// the cursor. The statement names no other table.
fn seeds_for(table: &Relation) -> String {
    let last = table.source.last().map(String::as_str).unwrap_or_default();
    let mut name = SEEDS.to_owned();
    while name.to_lowercase() == last.to_lowercase() {
        name.push('_');
    }
    name
}

/// The most name parts a DuckDB relation takes: a database, a schema and a
/// table (design choice 12).
const MOST_PARTS: usize = 3;

/// The largest seed DuckDB's `REPEATABLE` takes: its parser refuses one
/// from 2**31 (register entry RS-7). The dialect declares it as its
/// capability (`capabilities`), and the core refuses a larger seed by
/// reading it, before the sample is planned.
const MAX_SEED: u64 = (1 << 31) - 1;

/// The catalog's own tables of the database and schema an unquoted name
/// resolves in, which is where each table is read from.
const IN_CURRENT_SCHEMA: &str =
    "database_name = current_database() AND schema_name = current_schema()";

/// The schema DuckDB's answer to `request` has: its catalog's own for a
/// `describe`, what the contract fixes for every other request
/// (`DataRequest::expected`).
fn expected(request: &DataRequest) -> ExpectedSchema {
    use ColumnType::{Bool, Int32, Int64, Utf8};
    use catalog as c;
    let DataRequest::Describe { part, .. } = request else {
        return request.expected();
    };
    let names = || ColumnType::List {
        item: Box::new(field(c::LIST_ITEM, Utf8, true)),
    };
    let fields = match part {
        DescribePart::Tables => vec![field(c::TABLE_NAME, Utf8, false)],
        DescribePart::Columns => vec![
            field(c::CID, Int32, false),
            field(c::NAME, Utf8, false),
            field(c::TYPE, Utf8, false),
            field(c::NOTNULL, Bool, false),
            field(c::DFLT_VALUE, Utf8, true),
            field(c::PK, Bool, false),
        ],
        DescribePart::Keys => vec![
            field(c::CONSTRAINT_TYPE, Utf8, false),
            field(c::CONSTRAINT_COLUMN_NAMES, names(), false),
        ],
        DescribePart::ForeignKeys => vec![
            field(c::CONSTRAINT_INDEX, Int64, false),
            field(c::REFERENCED_TABLE, Utf8, false),
            field(c::CONSTRAINT_COLUMN_NAMES, names(), false),
            field(c::REFERENCED_COLUMN_NAMES, names(), false),
        ],
        DescribePart::Indexes => vec![
            field(c::TABLE_NAME, Utf8, false),
            field(c::EXPRESSIONS, Utf8, true),
        ],
        // Never planned: DuckDB lists an index's columns with the index.
        DescribePart::IndexColumns => return ExpectedSchema::Discover {},
        // In the contract's own shape.
        DescribePart::Namespace => return request.expected(),
    };
    ExpectedSchema::Schema { fields }
}

/// DuckDB's answer to a `describe` of `part`, which holds [`expected`]'s
/// schema, read into the contract's catalog answer.
fn read(part: DescribePart, batches: &[RecordBatch]) -> Result<CatalogAnswer, CoreError> {
    use catalog as c;
    match part {
        // None: the reference names a DuckDB table it cannot find alone.
        DescribePart::Namespace => {
            read::namespace(batches, sdm_contract::data::catalog_columns::NAME)
        }
        DescribePart::Tables => read::tables(batches, c::TABLE_NAME),
        DescribePart::Columns => read::columns(
            batches,
            &ColumnNames {
                name: c::NAME,
                declared_type: c::TYPE,
                not_null: c::NOTNULL,
                primary_key: c::PK,
            },
        ),
        DescribePart::Keys => read::keys(
            batches,
            c::CONSTRAINT_TYPE,
            c::CONSTRAINT_COLUMN_NAMES,
            key_kind,
        ),
        DescribePart::ForeignKeys => read::foreign_keys(
            batches,
            &ForeignKeyNames {
                key: c::CONSTRAINT_INDEX,
                referenced_table: c::REFERENCED_TABLE,
                columns: c::CONSTRAINT_COLUMN_NAMES,
                referenced_columns: c::REFERENCED_COLUMN_NAMES,
            },
        ),
        DescribePart::Indexes => {
            let mut found = Vec::new();
            for batch in batches {
                let tables = read::strings(batch, c::TABLE_NAME)?;
                let expressions = read::strings(batch, c::EXPRESSIONS)?;
                for row in 0..batch.num_rows() {
                    // An index with no expressions covers no column the
                    // reference reads (it skips such a row).
                    if let (Some(table), Some(expressions)) =
                        (read::text(tables, row), read::text(expressions, row))
                    {
                        found.push(CatalogIndex {
                            table,
                            columns: indexed_columns(&expressions),
                        });
                    }
                }
            }
            Ok(CatalogAnswer::Indexes(found))
        }
        DescribePart::IndexColumns => Err(CoreError::new(
            Sentence::Internal,
            "DuckDB answers no index's columns by its name",
        )),
    }
}

/// A constraint type as `duckdb_constraints()` spells it.
fn key_kind(constraint_type: &str) -> KeyKind {
    match constraint_type {
        "PRIMARY KEY" => KeyKind::Primary,
        "UNIQUE" => KeyKind::Unique,
        _ => KeyKind::Other,
    }
}

/// The columns an index's `expressions` name, as the reference reads them
/// (`_indexed_column` of its DuckDB sampler): a list in brackets of one
/// expression, quoted or not, is that one column, its quotes taken off.
/// A list holding a comma is unresolved, never split: DuckDB writes a comma
/// between expressions and inside one alike (`[a, b]`,
/// `['(COALESCE(b, \',\'))']`, `['"a,b"']`), so its pieces would be a
/// guess, and the reference reads no column from it either. Text not in
/// brackets is unresolved too, the reference reading none.
pub(crate) fn indexed_columns(expressions: &str) -> IndexCoverage {
    let expression = expressions.trim();
    let Some(inner) = expression
        .strip_prefix('[')
        .and_then(|e| e.strip_suffix(']'))
    else {
        return IndexCoverage::Unresolved;
    };
    let inner = inner.trim();
    if inner.contains(',') {
        return IndexCoverage::Unresolved;
    }
    IndexCoverage::Resolved(vec![unquoted(inner)])
}

/// One expression with its single quotes, then its double quotes, taken
/// off, as the reference's `strip("'")` and slice take them.
fn unquoted(expression: &str) -> String {
    let expression = expression.trim_matches('\'');
    if expression.starts_with('"') && expression.ends_with('"') {
        // The reference's slice: a lone quote is the empty name.
        let inner = expression.get(1..expression.len().saturating_sub(1).max(1));
        return inner.unwrap_or_default().replace("\"\"", "\"");
    }
    expression.to_owned()
}

impl SqlDialect for Duckdb {
    fn name(&self) -> &'static str {
        NAME
    }

    /// What a DuckDB source answers through this layer.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            compares_stored_times: false,
            reads_nan_as_missing: false,
            samples_by_key_dtype: false,
            derived_relations: false,
            neighbor_lookback: false,
            keys_optional: false,
            sample_rows: false,
            optional: sql::rendered(),
            max_keys_per_request: None,
            // DuckDB's reservoir sampling repeats under a seed (REPEATABLE).
            sampling: Determinism::SeededStable,
            max_seed: MAX_SEED,
            // The reference's engine reads a SQL source's text times one by
            // one (register DT-7).
            text_times: Some(TextTimes::ValueByValue),
            // DuckDB's default null order, which the reference's neighbor
            // order relies on, writing none: last, in either direction.
            null_order: Some(NullOrder::Last),
            // The reference's DuckDB sampler counts an index over exactly one
            // column (an ART index serves a scan only then) and warns of the
            // rest (`_get_missing_index_dict`).
            index_advice: Some(IndexAdvice::SingleColumn),
            // Every optional request is rendered: none falls back.
            fallback: None,
            // DuckDB's driver hands a HUGEINT, UHUGEINT or VARINT over as
            // its opaque extension type (register P1B-43).
            opaque_vendor: Some(OPAQUE_VENDOR.to_owned()),
        }
    }

    /// A database, a schema and a table (design choice 12): each part
    /// quoted alone as an identifier, never split on a dot, so `["a.b"]` is
    /// `"a.b"` and `["main", "t"]` is `"main"."t"`, which DuckDB resolves as
    /// it resolves any name so written.
    fn most_parts(&self) -> usize {
        MOST_PARTS
    }

    fn seeds_for(&self, _request: &DataRequest, table: &Relation) -> String {
        seeds_for(table)
    }

    /// The key compared exactly whatever its type ([`seed_cast`]).
    fn key_equals(
        &self,
        read: &SeedRead<'_>,
        key: &str,
        seed_column: &str,
        seed: &str,
        op: &str,
    ) -> String {
        match seed_cast(read, seed_column) {
            Some(Integer::Bignum) => {
                format!("{key} {op} TRY_CAST(CAST({seed} AS HUGEINT) AS BIGNUM)")
            }
            Some(Integer::Native(name)) => format!("{key} {op} TRY_CAST({seed} AS {name})"),
            None => format!("{key} {op} {seed}"),
        }
    }

    /// The source's own wall clock, `CAST(column AS TIMESTAMP)`.
    fn as_time(&self, value: &str) -> String {
        format!("CAST({value} AS TIMESTAMP)")
    }

    /// A seed past the range the dialect declares never reaches here: the
    /// core refuses it first (register entry RS-7), so one that does is the
    /// core's error, not the caller's.
    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError> {
        match seed_integer(seed) {
            Some(seed) if (0..=i128::from(MAX_SEED)).contains(&seed) => Ok(seed),
            Some(seed) if seed >= 0 => Err(CoreError::new(
                Sentence::Internal,
                "a seed past the range the dialect declares reached its rendering",
            )
            .with_field("request", Detail::text(request.name()))
            .with_field("field", Detail::text("seed"))),
            _ => Err(sdm_contract::data::refused(
                request,
                "seed",
                "a sample's seed is a non-negative integer",
            )),
        }
    }

    /// The reference's reservoir sample of an entity table
    /// (`_random_sample_sql` on DuckDB): the columns asked for, from the
    /// rows that meet each not-null filter, `n` rows under the seed. The key
    /// names the columns the entities are told apart by, which the sample
    /// projects, and which the core keeps one row of each of, as the
    /// reference does.
    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        vec![format!(
            "SELECT {}\nFROM (SELECT * FROM {}{})\nUSING SAMPLE reservoir({} ROWS) REPEATABLE ({})",
            sample.projected, sample.table, sample.filtered, sample.n, sample.seed
        )]
    }

    /// The reference's neighbor reads (`_by_time` and `_by_fkey` on
    /// DuckDB) with a limit per seed: the joined read, then the first rows
    /// of each seed in `order`, the result in `(batch, order)`.
    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String {
        let order = terms
            .iter()
            .map(|t| self.order_term(t, "t.", None))
            .collect::<Vec<_>>()
            .join(", ");
        let batch = identifier(SEED_BATCH);
        format!(
            "{}\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.{batch}\n  ORDER BY {order}\n) <= {limit}\nORDER BY s.{batch}, {order}",
            read.joined()
        )
    }

    /// The reference's key lookup (`_by_pkey` on DuckDB): one row of the
    /// table per seed whose key it holds, with the count of rows holding
    /// that key.
    fn lookup(&self, read: &SeedRead<'_>) -> String {
        let batch = identifier(SEED_BATCH);
        format!(
            "SELECT s.{batch} AS {helper}, COUNT(*) OVER (PARTITION BY s.{batch}) AS {matches}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON {on}\nQUALIFY ROW_NUMBER() OVER (\n  PARTITION BY s.{batch}\n) = 1",
            helper = identifier(BATCH_COLUMN),
            matches = identifier(MATCHES_COLUMN),
            projected = read.projected(),
            seeds = read.seeds(),
            table = read.table,
            on = read.seed_join("t.", "="),
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
        _request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        read(part, batches)
    }
}

/// One part of the catalog. The table list and the indexes are read whole,
/// as the reference reads them, so a request naming tables is refused
/// (tables named arrive with the overrides, T1.24); the columns, keys and
/// foreign keys are read one table at a time, one statement each, as the
/// reference reads them, so a request naming none or several is refused.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
) -> Result<String, CoreError> {
    let one = || sql::one_table(request, tables);
    Ok(match part {
        DescribePart::Tables if tables.is_empty() => format!(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND NOT internal AND {IN_CURRENT_SCHEMA}"
        ),
        DescribePart::Tables => held(tables),
        // No row: the reference names a DuckDB table it cannot find alone.
        DescribePart::Namespace => format!(
            "SELECT CAST(NULL AS VARCHAR) AS {} WHERE FALSE",
            identifier(sdm_contract::data::catalog_columns::NAME)
        ),
        DescribePart::Columns => columns(one()?),
        DescribePart::Keys => format!(
            "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()\nWHERE {}",
            in_catalog(one()?, "table_name", "", "  ")
        ),
        DescribePart::ForeignKeys => format!(
            "SELECT constraint_index, referenced_table,\n       constraint_column_names, referenced_column_names\nFROM duckdb_constraints()\nWHERE {}\n  AND constraint_type = 'FOREIGN KEY'",
            in_catalog(one()?, "table_name", "", "  ")
        ),
        DescribePart::Indexes => {
            sql::whole(request, tables)?;
            "SELECT table_name, expressions FROM duckdb_indexes()".to_owned()
        }
        // DuckDB's index list gives each index's columns
        // (`IndexAdvice::SingleColumn`): none is read by its name.
        DescribePart::IndexColumns => {
            return Err(sql::not_taken(
                request,
                "tables",
                "DuckDB lists an index's columns with its indexes, never by its name",
            ));
        }
    })
}

/// The relations, tables or views, DuckDB would read under any of `names`
/// written unqualified, as it resolves such a name: a temporary relation,
/// or one of the current database in a schema of the search path
/// (`current_schemas(true)`, its implicit schemas included), whose name
/// equals one of `names` once each of their ASCII letters is lowered, as
/// DuckDB folds a name (ASCII letters only, so `"Äpfel"` is not read as
/// `äpfel`). A relation of another schema or another attached database is
/// not one the name reads. Asked only when reading a named table's columns
/// has failed: no row says the source holds no such relation, so the
/// caller named one it lacks; any row, that the failure was another
/// (T1.23 fix 3). A search path naming a schema of another attached
/// database (`SET search_path = 'other.main'`) is not followed: the
/// filter holds the current database. No provider of this build sets
/// such a path on the connection it reads through, so it is unreachable
/// today; recorded here rather than ported (T1.23's review).
///
/// A relation of several parts is read by its parts, as [`in_catalog`]
/// finds it, each a condition of its own beside the names of one part.
fn held(relations: &[Relation]) -> String {
    let names: Vec<&str> = relations.iter().filter_map(Relation::single).collect();
    let qualified: Vec<&Relation> = relations.iter().filter(|r| r.single().is_none()).collect();
    if qualified.is_empty() {
        return held_names(&names);
    }
    let read = |relation: &&Relation, kind: &str| {
        format!(
            "SELECT {kind}_name AS table_name FROM duckdb_{kind}s() WHERE NOT internal AND {}",
            in_catalog(relation, &format!("{kind}_name"), "", "  "),
        )
    };
    let mut selects: Vec<String> = Vec::new();
    if !names.is_empty() {
        selects.push(held_names(&names));
    }
    for relation in &qualified {
        selects.push(read(relation, "table"));
        selects.push(read(relation, "view"));
    }
    selects.join("\nUNION ALL\n")
}

/// SQL `text` with each of its ASCII letters lowered, as DuckDB folds an
/// identifier to look it up in its catalog (ASCII letters only, so
/// `"Äpfel"` is not `äpfel`): two names are the same name to DuckDB when
/// their folds are equal, quoted or not.
fn fold(text: &str) -> String {
    format!("translate({text}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')")
}

/// [`held`] for relations of one part, by their names.
fn held_names(names: &[&str]) -> String {
    let asked = names
        .iter()
        .map(|name| fold(&self::text(name)))
        .collect::<Vec<_>>()
        .join(", ");
    let resolved = "(temporary OR (database_name = current_database() AND list_contains(current_schemas(true), schema_name)))";
    format!(
        "SELECT table_name FROM duckdb_tables() WHERE NOT internal AND {resolved} AND {} IN ({asked})\nUNION ALL\nSELECT view_name AS table_name FROM duckdb_views() WHERE NOT internal AND {resolved} AND {} IN ({asked})",
        fold("table_name"),
        fold("view_name"),
    )
}

/// The columns of `table`, in `PRAGMA table_info`'s columns and order.
///
/// The reference reads them with `PRAGMA table_info`, and so does this
/// rendering, text for text, for every name the pragma reads as itself. The
/// pragma reads its argument as a qualified name, though: a dot splits it,
/// and a double quote opens or closes a quoted part, so a name holding
/// either is read as another name, which DuckDB refuses (a parser error for
/// `double"quote`, no such schema for `a.b`) or, where one exists, reads in
/// its place (`main.x` read as `x`). Quoting cannot help: DuckDB 1.5.6
/// unquotes no doubled quote there. Such a table's columns are read by its
/// own name, as text, from `duckdb_columns()` and its primary key from
/// `duckdb_constraints()`, as the other catalog reads are, the answer's
/// columns, types and rows those the pragma gives for a table of an
/// ordinary name. Where the reference refuses (its `TableNotFoundError`,
/// `INVALID_REQUEST`), the core now describes the table: register entry
/// P1B-5, proposed. A name no table holds (in its exact case) gives no
/// rows where the pragma raises DuckDB's catalog error; no caller asks for
/// one today (`describe_source` reads the names the table list gives), and
/// the core refuses a table with no columns, `INVALID_REQUEST`, the code
/// the reference gives a table its pragma cannot read.
///
/// A relation of several parts is read by the pragma too, its parts quoted
/// alone (`PRAGMA table_info("main"."t")`), which DuckDB resolves as it
/// resolves the same name in a `FROM`, a dot inside a part kept; a part
/// holding a double quote, which the pragma does not unquote, is read from
/// the catalog by its parts ([`in_catalog`]).
fn columns(table: &Relation) -> String {
    let parts = &table.source;
    if !parts.iter().any(|p| p.contains('"')) && (parts.len() > 1 || !parts[0].contains('.')) {
        let quoted: Vec<String> = parts.iter().map(|p| identifier(p)).collect();
        return format!("PRAGMA table_info({})", quoted.join("."));
    }
    format!(
        "SELECT CAST(c.column_index - 1 AS INTEGER) AS cid, c.column_name AS name,\n       c.data_type AS type, NOT c.is_nullable AS notnull,\n       c.column_default AS dflt_value,\n       COALESCE(list_contains(k.columns, c.column_name), false) AS pk\nFROM duckdb_columns() c\nLEFT JOIN (\n  SELECT constraint_column_names AS columns\n  FROM duckdb_constraints()\n  WHERE {}\n    AND constraint_type = 'PRIMARY KEY'\n) k ON true\nWHERE {}\nORDER BY c.column_index",
        in_catalog(table, "table_name", "", "    "),
        in_catalog(table, "c.table_name", "c.", "  "),
    )
}

/// The condition on a catalog function's rows (`duckdb_constraints()`,
/// `duckdb_columns()`, `duckdb_tables()`, `duckdb_views()`) holding
/// `relation`, its table's name compared as text in the column `table`,
/// the database and schema in `prefix`'s `database_name` and
/// `schema_name`, the condition's later lines indented by `indent`: one
/// part, the table of that name in the database and schema an unquoted
/// name resolves in, as Phase 1 read it; three parts, the database, schema
/// and table so named; two parts, `[x, t]`, as DuckDB resolves `x.t`: the
/// table `t` of the schema `x` of the current database, or, where the
/// current database has no schema `x`, of the default schema `main` of the
/// database `x` (where both exist DuckDB refuses the name as ambiguous,
/// and the read of its columns fails first).
///
/// A relation of several parts names its database, schema and table as
/// DuckDB resolves them, each compared by [`fold`], as DuckDB compares
/// identifiers, so `["S", "u"]` is the table `["s", "u"]` is, as
/// `"S"."u"` reads it. One part keeps the text comparison of its table's
/// name that the reference makes (`DuckDBTable`'s constraints, 6298913c),
/// so a one-part request renders as it did.
fn in_catalog(relation: &Relation, table: &str, prefix: &str, indent: &str) -> String {
    let (database, schema) = (
        format!("{prefix}database_name"),
        format!("{prefix}schema_name"),
    );
    // A part against a catalog name, as DuckDB compares identifiers.
    let named = |column: &str, part: &str| {
        format!("{} = {}", fold(column), text(&part.to_ascii_lowercase()))
    };
    match relation.source.as_slice() {
        [database_part, schema_part, name] => format!(
            "{}\n{indent}AND {} AND {}",
            named(table, name),
            named(&database, database_part),
            named(&schema, schema_part)
        ),
        [qualifier, name] => format!(
            "{}\n{indent}AND (({database} = current_database() AND {})\n{indent}  OR ({} AND {schema} = 'main'\n{indent}      AND NOT EXISTS (SELECT 1 FROM duckdb_schemas() WHERE database_name = current_database() AND {})))",
            named(table, name),
            named(&schema, qualifier),
            named(&database, qualifier),
            named("schema_name", qualifier),
        ),
        parts => format!(
            "{table} = {}\n{indent}AND {database} = current_database() AND {schema} = current_schema()",
            text(parts.first().map(String::as_str).unwrap_or_default())
        ),
    }
}

/// The vendor DuckDB names in the metadata of its opaque Arrow extension
/// types (`arrow.opaque`'s `vendor_name`), which the host layer reads by
/// DuckDB's layouts only for a column naming it (register P1B-43).
pub(crate) const OPAQUE_VENDOR: &str = "DuckDB";

/// The integer type a seed key column, `seed_column`, is cast to before it
/// is compared with the table's key: `t.key = s.key`, as before part 3,
/// but for a `decimal128(38, 0)` seed key whose column names the key's
/// declared type ([`SEED_KEY_TYPE`], which the core carries from what
/// `describe_source` read of the catalog, so no statement is added) and
/// that type is an integer one: then the seed is cast to the key's own
/// type, `t.key = TRY_CAST(s.key AS type)`, so the two compare as one type,
/// exactly, by the key's native equality, and a seed the type cannot hold
/// is null and matches nothing (register P1B-43). Without it a VARINT key
/// cannot be compared with a DECIMAL(38, 0) at all (DuckDB has no such
/// cast), and a HUGEINT seed would meet a UHUGEINT key through DOUBLE. A
/// VARINT (BIGNUM) key takes its seed through HUGEINT, which holds every
/// 38-digit value exactly, since DuckDB's DECIMAL to BIGNUM cast is not
/// implemented (its TRY_CAST gives null). Any other key type, a float or a
/// decimal with a scale, or no declared type, keeps `t.key = s.key`.
///
/// A key of several columns joins each column to its seed column by the
/// same rule, the conditions joined by `AND` (design choice 1; composite
/// keys, T2.12).
fn seed_cast(read: &SeedRead<'_>, seed_column: &str) -> Option<Integer> {
    read.seeds
        .columns
        .iter()
        .find(|c| {
            c.name == seed_column
                && c.column_type
                    == ColumnType::Decimal128 {
                        precision: 38,
                        scale: 0,
                    }
        })
        .and_then(|c| c.metadata.get(SEED_KEY_TYPE))
        .and_then(|declared| integer_type(declared))
}

/// An integer key type as [`seed_cast`] casts a seed to it.
enum Integer {
    /// VARINT, which DuckDB's catalog spells BIGNUM.
    Bignum,
    /// Any other, as the SQL spells it: rendered here, never the catalog's
    /// text, so nothing of the catalog reaches the statement.
    Native(String),
}

/// The integer type DuckDB's catalog `declared` spells, if it is one: the
/// signed and unsigned integers, HUGEINT, UHUGEINT, BIGNUM (VARINT) and a
/// DECIMAL of scale 0.
fn integer_type(declared: &str) -> Option<Integer> {
    const NATIVE: [&str; 10] = [
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "HUGEINT",
        "UHUGEINT",
    ];
    let declared = declared.trim().to_ascii_uppercase();
    if declared == "BIGNUM" || declared == "VARINT" {
        return Some(Integer::Bignum);
    }
    if let Some(name) = NATIVE.iter().find(|n| **n == declared) {
        return Some(Integer::Native((*name).to_owned()));
    }
    let digits = declared.strip_prefix("DECIMAL(")?.strip_suffix(",0)")?;
    let precision: u8 = digits.parse().ok()?;
    (1..=38)
        .contains(&precision)
        .then(|| Integer::Native(format!("DECIMAL({precision},0)")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_key_type_is_read_and_anything_else_is_not() {
        let native = |declared: &str| match integer_type(declared) {
            Some(Integer::Native(name)) => Some(name),
            _ => None,
        };
        assert_eq!(native("UHUGEINT").as_deref(), Some("UHUGEINT"));
        assert_eq!(native(" bigint ").as_deref(), Some("BIGINT"));
        assert_eq!(native("DECIMAL(38,0)").as_deref(), Some("DECIMAL(38,0)"));
        assert_eq!(native("DECIMAL(20,0)").as_deref(), Some("DECIMAL(20,0)"));
        assert!(matches!(integer_type("BIGNUM"), Some(Integer::Bignum)));
        assert!(matches!(integer_type("VARINT"), Some(Integer::Bignum)));
        for other in [
            "DOUBLE",
            "FLOAT",
            "DECIMAL(10,2)",
            "DECIMAL(39,0)",
            "DECIMAL(0,0)",
            "VARCHAR",
            "BIGINT); DROP TABLE t; --",
            "",
        ] {
            assert!(integer_type(other).is_none(), "{other}");
        }
    }
}
