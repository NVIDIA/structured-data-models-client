// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Postgres dialect (T2.3). Its capabilities are the optional requests
//! it has a rendering for, every one the core asks, so none falls back.
//!
//! Each rendering is the statement the reference's Postgres engine runs for
//! the same step where one exists (`rfm/backend/postgres`), written the same
//! way but for its relation, which is the core's (design choice 12): the
//! preview, the key counts, the column ranges, the repeated key and the
//! entity sample text for text but for the `FROM`, which the reference
//! qualifies with the schema it read the table from. What each returns is
//! that statement's result (decision H9 gates on results, not text).
//!
//! The catalog is read from `pg_catalog`, each answer in the contract's own
//! shape (`catalog_columns`), so the data layer reads it as it reads a host
//! provider's ([`crate::read::host`]). A relation is found as Postgres
//! finds the same name in a `FROM`: `to_regclass` over its parts, each
//! quoted alone, so every statement of a run reads one relation, a view
//! included, and a name the source does not hold answers no columns, which
//! the core refuses as the reference refuses a table it cannot find. The
//! reference reads a one-part name in `current_schema()` only; the two
//! differ only where the search path finds the name elsewhere (a temporary
//! table, `pg_catalog`, or a later schema), where the core reads the
//! relation every other statement of the run reads.
//!
//! A key set is bound as the reference's Postgres sampler binds one: JSON
//! text in the statement's one parameter, `$1`, unpacked by a common table
//! expression named as the plan's relation ([`SEEDS`]; `P1 d:131`). The
//! provider writes the key set's rows as a JSON array of rows, each row an
//! array of its cells' texts in the key set's column order (null for a
//! null), which Postgres's own input functions read ([`seed_cte`]): a
//! number key as the Postgres type of its Arrow type, so the join compares
//! the two keys as Postgres compares numbers of two types, exactly (an
//! integer and a `numeric` meet when their values are equal, whatever
//! their scales, and two `numeric` values float64 would merge stay
//! apart; design choice 5, lesson 19, KO2-35); a text key in the joined
//! column's own declared type, through `jsonb_populate_record` over the
//! table's row type, so a `uuid`, an enum or a `varchar` key is compared
//! as itself (the reference casts to the type it read from the catalog);
//! the batch as `bigint`; a seed time as `timestamp`.

use std::collections::BTreeSet;

use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    BATCH_COLUMN, DataRequest, DescribePart, Determinism, Dialect, FetchPlan, Filter, KeySet,
    MATCHES_COLUMN, NullOrder, OptionalRequest, ProviderCapabilities, Relation, SEED_BATCH,
    SqlParam, SqlStatement, TableColumn, TableKey, TextTimes, TieBreak, TieTerm, Window,
    catalog_columns as c, seed_key_columns,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

use arrow_array::RecordBatch;

/// The dialect's name on the wire.
pub(crate) const NAME: &str = "postgres";

/// The optional requests this dialect renders.
const RENDERED: &[OptionalRequest] = &[
    OptionalRequest::CountRows,
    OptionalRequest::Preview,
    OptionalRequest::KeyCounts,
    OptionalRequest::TimeBounds,
    OptionalRequest::ColumnRanges,
    OptionalRequest::RepeatedKey,
    OptionalRequest::SampleEntities,
    OptionalRequest::Neighbors,
    OptionalRequest::Lookup,
];

/// The name a statement reads its seeds by: the common table expression
/// the key set's JSON is unpacked into. A CTE's name shadows a table of
/// the same name only where the statement names it unqualified, and the
/// joined table is always named by its relation, so no name is lengthened
/// here; [`seeds_for`] still steps past a one-part relation of that name,
/// which would otherwise read the seeds.
pub(crate) const SEEDS: &str = "__sdm_seeds__";

/// The CTE a statement joining `table` reads its seeds by: [`SEEDS`],
/// lengthened by underscores until it is not `table`'s one part, so a
/// source table of any name is read, never shadowed by the seeds.
fn seeds_for(table: &Relation) -> String {
    let mut name = SEEDS.to_owned();
    if let Some(single) = table.single() {
        while name == single {
            name.push('_');
        }
    }
    name
}

/// The most name parts a Postgres relation takes: a schema and a table
/// (design choice 12).
const MOST_PARTS: usize = 2;

/// The `FROM` target of `relation`, through the data layer's one rendering
/// of a relation ([`crate::relation`]): each part quoted alone, never split
/// on a dot. One of no part, or of more than [`MOST_PARTS`], is refused by
/// name (rule `relation`).
fn from(request: &DataRequest, field: &str, relation: &Relation) -> Result<String, CoreError> {
    crate::relation(request, field, relation, MOST_PARTS, identifier)
}

/// The catalog's object id of `relation`, found as a `FROM` finds the same
/// name (`to_regclass` over the quoted parts), or null where the source
/// holds no relation of that name.
fn regclass(request: &DataRequest, field: &str, relation: &Relation) -> Result<String, CoreError> {
    Ok(format!(
        "to_regclass({})",
        text(&from(request, field, relation)?)
    ))
}

/// What a Postgres source answers through this layer.
pub(crate) fn capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        optional: RENDERED.iter().copied().collect::<BTreeSet<_>>(),
        max_keys_per_request: None,
        // `setseed` before `ORDER BY random()` repeats the draw on one
        // server (register entry RS-6, KO2-24).
        sampling: Determinism::SeededStable,
        // The reference hashes the seed's text into `setseed`'s range, so
        // it takes any integer seed (`_postgres_seed`).
        max_seed: u64::MAX,
        // The reference's engine reads a SQL source's text times one by
        // one (register DT-7).
        text_times: Some(TextTimes::ValueByValue),
        // Postgres orders a null after every value ascending, the order
        // every term the core asks with nulls among its values has: the
        // reference's neighbor order is the time column descending, then
        // the key (or each other column) ascending, and the time column's
        // nulls never reach the order, since the window that comes with a
        // time column leaves them out (register ORD-2). Each term is still
        // rendered with the null order the core writes into it.
        null_order: Some(NullOrder::Last),
        // The reference's Postgres sampler neither reads the indexes nor
        // warns of missing ones.
        index_advice: None,
        // Every optional request is rendered: none falls back.
        fallback: None,
    }
}

/// The statements answering `request` on Postgres, with the schema the
/// contract fixes for its answer (`DataRequest::expected`): the catalog's
/// answers are in the contract's own shape.
pub(crate) fn plan(request: &DataRequest) -> Result<FetchPlan, CoreError> {
    let mut texts = vec![];
    let mut params = vec![];
    let text = match request {
        DataRequest::Describe { tables, part } => describe(request, tables, *part)?,
        DataRequest::Preview {
            table,
            columns,
            limit,
        } => {
            let table = from(request, "table", table)?;
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "a preview names no columns",
                ));
            }
            format!("SELECT {} FROM {table} LIMIT {limit}", identifiers(columns))
        }
        DataRequest::KeyCounts { keys } => key_counts(request, keys)?,
        DataRequest::TimeBounds { columns } => time_bounds(request, columns)?,
        DataRequest::ColumnRanges { table, columns } => {
            let table = from(request, "table", table)?;
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "the column ranges name no columns",
                ));
            }
            column_ranges(&table, columns)
        }
        DataRequest::RepeatedKey { table, key } => {
            let table = from(request, "table", table)?;
            if key.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "key",
                    "a repeated key names no columns",
                ));
            }
            repeated_key(&table, key)
        }
        DataRequest::SampleEntities {
            table,
            key,
            n,
            seed,
            columns,
            filters,
        } => {
            let table = from(request, "table", table)?;
            let (seeded, text) = sample_entities(request, &table, key, *n, seed, columns, filters)?;
            texts.push(seeded);
            text
        }
        DataRequest::Neighbors {
            table,
            key,
            columns,
            seeds,
            window,
            per_seed,
            order,
        } => {
            let seeds_name = seeds_for(table);
            let cte = seed_cte(request, table, &seeds_name, key, seeds)?;
            let table = from(request, "table", table)?;
            params.push(SqlParam::KeySet {
                keys: seeds.clone(),
                relation: seeds_name.clone(),
            });
            let body = neighbors(
                request,
                &table,
                &seeds_name,
                key,
                columns,
                window.as_ref(),
                *per_seed,
                order.as_ref(),
            )?;
            format!("{cte}{body}")
        }
        DataRequest::Lookup {
            table,
            key,
            columns,
            seeds,
        } => {
            let seeds_name = seeds_for(table);
            let cte = seed_cte(request, table, &seeds_name, key, seeds)?;
            let table = from(request, "table", table)?;
            params.push(SqlParam::KeySet {
                keys: seeds.clone(),
                relation: seeds_name.clone(),
            });
            format!(
                "{cte}{}",
                lookup(request, &table, &seeds_name, key, columns)?
            )
        }
        DataRequest::Scan {
            table,
            columns,
            filters,
            limit,
        } => {
            let table = from(request, "table", table)?;
            if columns.is_empty() {
                return Err(sdm_contract::data::refused(
                    request,
                    "columns",
                    "a scan names no columns",
                ));
            }
            let mut text = format!(
                "SELECT {} FROM {table}{}",
                identifiers(columns),
                conditions(request, filters, "")?
            );
            if let Some(limit) = limit {
                text.push_str(&format!(" LIMIT {limit}"));
            }
            text
        }
        DataRequest::CountRows { table, filters } => format!(
            "SELECT COUNT(*) AS {} FROM {}{}",
            identifier("rows"),
            from(request, "table", table)?,
            conditions(request, filters, "")?
        ),
    };
    let statement = |text: String, params: Vec<SqlParam>| SqlStatement {
        dialect: Dialect::new(NAME),
        text,
        params,
    };
    let mut statements: Vec<SqlStatement> = texts
        .into_iter()
        .flatten()
        .map(|text| statement(text, vec![]))
        .collect();
    statements.push(statement(text, params));
    Ok(FetchPlan::Sql {
        statements,
        expect: request.expected(),
    })
}

/// A Postgres answer to a `describe` of `part`, which is in the contract's
/// own shape (`catalog_columns`), read as a host provider's is.
pub(crate) fn read(
    part: DescribePart,
    batches: &[RecordBatch],
) -> Result<CatalogAnswer, CoreError> {
    crate::read::host(part, batches)
}

/// A request field this rendering does not take, refused by name.
fn not_taken(request: &DataRequest, field: &str, message: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        message,
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("field", Detail::text(field))
}

/// One part of the catalog, in the contract's shape. The table list and
/// the indexes are read whole, as the reference reads the table list, and
/// a table list naming tables answers each the source holds (asked when a
/// named table's columns could not be read); the columns, keys and foreign
/// keys are read one table at a time, as the reference reads them.
fn describe(
    request: &DataRequest,
    tables: &[Relation],
    part: DescribePart,
) -> Result<String, CoreError> {
    for table in tables {
        from(request, "tables", table)?;
    }
    let refused = |message: &str| {
        CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            message,
        )
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text("tables"))
    };
    let one = || match tables {
        [table] => regclass(request, "tables", table),
        _ => Err(refused(
            "this part of the catalog is read for one table at a time",
        )),
    };
    Ok(match part {
        // The reference's listing (`information_schema.tables`, the base
        // tables of the current schema, by name in the catalog's own
        // order), the name as text.
        DescribePart::Tables if tables.is_empty() => format!(
            "SELECT table_name::text AS {}\nFROM information_schema.tables\nWHERE table_schema = current_schema()\n  AND table_type = 'BASE TABLE'\nORDER BY table_name",
            identifier(c::TABLE)
        ),
        DescribePart::Tables => {
            let mut selects = Vec::with_capacity(tables.len());
            for table in tables {
                let found = regclass(request, "tables", table)?;
                selects.push(format!(
                    "SELECT relname::text AS {}\nFROM pg_catalog.pg_class\nWHERE oid = {found}",
                    identifier(c::TABLE)
                ));
            }
            selects.join("\nUNION ALL\n")
        }
        DescribePart::Columns => format!(
            "SELECT a.attname::text AS {name},\n       pg_catalog.format_type(a.atttypid, a.atttypmod) AS {kind},\n       a.attnotnull AS {not_null},\n       COALESCE(a.attnum = ANY (p.conkey), false) AS {primary}\nFROM pg_catalog.pg_attribute a\nLEFT JOIN pg_catalog.pg_constraint p\n  ON p.conrelid = a.attrelid AND p.contype = 'p'\nWHERE a.attrelid = {table}\n  AND a.attnum > 0 AND NOT a.attisdropped\nORDER BY a.attnum",
            name = identifier(c::NAME),
            kind = identifier(c::TYPE),
            not_null = identifier(c::NOT_NULL),
            primary = identifier(c::PRIMARY_KEY),
            table = one()?,
        ),
        DescribePart::Keys => format!(
            "SELECT CASE k.contype WHEN 'p' THEN {primary} ELSE {unique} END AS {kind},\n       {columns} AS {names}\nFROM pg_catalog.pg_constraint k\nWHERE k.conrelid = {table}\n  AND k.contype IN ('p', 'u')\nORDER BY k.conname",
            primary = text(c::KIND_PRIMARY),
            unique = text(c::KIND_UNIQUE),
            kind = identifier(c::KIND),
            columns = attribute_names("k.conrelid", "k.conkey"),
            names = identifier(c::COLUMNS),
            table = one()?,
        ),
        DescribePart::ForeignKeys => format!(
            "SELECT f.oid::bigint AS {key},\n       r.relname::text AS {referenced},\n       {columns} AS {names},\n       {referenced_columns} AS {referenced_names}\nFROM pg_catalog.pg_constraint f\nJOIN pg_catalog.pg_class r ON r.oid = f.confrelid\nWHERE f.conrelid = {table}\n  AND f.contype = 'f'\nORDER BY f.conname",
            key = identifier(c::KEY),
            referenced = identifier(c::REFERENCED_TABLE),
            columns = attribute_names("f.conrelid", "f.conkey"),
            names = identifier(c::COLUMNS),
            referenced_columns = attribute_names("f.confrelid", "f.confkey"),
            referenced_names = identifier(c::REFERENCED_COLUMNS),
            table = one()?,
        ),
        DescribePart::Indexes => {
            if !tables.is_empty() {
                return Err(refused("this part of the catalog is read whole"));
            }
            // Each index of a table of the current schema, its columns in
            // order, an expression's position a null name (unresolved).
            format!(
                "SELECT t.relname::text AS {table},\n       {columns} AS {names}\nFROM pg_catalog.pg_index i\nJOIN pg_catalog.pg_class t ON t.oid = i.indrelid\nWHERE t.relnamespace = current_schema()::regnamespace\nORDER BY t.relname, i.indexrelid",
                table = identifier(c::TABLE),
                columns = attribute_names("i.indrelid", "i.indkey::int2[]"),
                names = identifier(c::COLUMNS),
            )
        }
    })
}

/// The names of `relation`'s attributes numbered in `numbers`, in their
/// order, as a `text[]`; a number naming no attribute (an index's
/// expression) a null.
fn attribute_names(relation: &str, numbers: &str) -> String {
    format!(
        "ARRAY(SELECT a.attname::text\n             FROM unnest({numbers}) WITH ORDINALITY AS n(attnum, position)\n             LEFT JOIN pg_catalog.pg_attribute a\n               ON a.attrelid = {relation} AND a.attnum = n.attnum\n             ORDER BY n.position)"
    )
}

/// One `SELECT` per key, in order, labelled by its position: the table's
/// rows, the key's distinct values among the rows holding all of it, and
/// the rows missing part of it, as the reference counts them
/// (`_count_distinct` of its Postgres table: a row of several values is
/// counted as not null when only part of it is, so those rows are left out
/// by hand).
fn key_counts(request: &DataRequest, keys: &[TableKey]) -> Result<String, CoreError> {
    let tables = keys
        .iter()
        .map(|key| from(request, "keys", &key.table))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(keys
        .iter()
        .zip(tables)
        .enumerate()
        .map(|(position, (key, table))| {
            let quoted: Vec<String> = key.columns.iter().map(|c| identifier(c)).collect();
            let missing = quoted
                .iter()
                .map(|c| format!("{c} IS NULL"))
                .collect::<Vec<_>>()
                .join(" OR ");
            let distinct = match quoted.as_slice() {
                [one] => format!("COUNT(DISTINCT {one})"),
                several => format!(
                    "COUNT(DISTINCT ({})) FILTER (WHERE {})",
                    several.join(", "),
                    several
                        .iter()
                        .map(|c| format!("{c} IS NOT NULL"))
                        .collect::<Vec<_>>()
                        .join(" AND ")
                ),
            };
            format!(
                "SELECT {position} AS k, COUNT(*) AS n, {distinct} AS d, COUNT(CASE WHEN {missing} THEN 1 END) AS z\nFROM {table}"
            )
        })
        .collect::<Vec<_>>()
        .join("\nUNION ALL\n"))
}

/// One `SELECT` per time column, in order, labelled by its position: the
/// smallest and largest values in the source's own wall clock, `CAST(column
/// AS timestamp)` (a `timestamptz` in the session's zone, a date at its
/// midnight), the zone-less timestamps the request takes back. The
/// reference reads the column's own type and converts it in the host.
fn time_bounds(request: &DataRequest, columns: &[TableColumn]) -> Result<String, CoreError> {
    let mut selects = Vec::with_capacity(columns.len());
    for (position, at) in columns.iter().enumerate() {
        let table = from(request, "columns", &at.table)?;
        let time = format!("CAST({} AS timestamp)", identifier(&at.column));
        selects.push(format!(
            "SELECT {position} AS table_index, MIN({time}) AS min_date, MAX({time}) AS max_date\nFROM {table}"
        ));
    }
    Ok(selects.join("\nUNION ALL\n"))
}

/// The reference's one query over every row of a table for its time
/// column's choice (`Table._full_range`): `COUNT(*)`, then each column's
/// `MIN`, `MAX` and `COUNT`, in its text; `table` its rendered relation.
fn column_ranges(table: &str, columns: &[String]) -> String {
    let mut selects = vec!["COUNT(*)".to_owned()];
    for column in columns {
        let quoted = identifier(column);
        selects.push(format!("MIN({quoted}), MAX({quoted}), COUNT({quoted})"));
    }
    format!("SELECT {}\nFROM {table}", selects.join(", "))
}

/// The reference's read of one value of a key that several rows hold
/// (`Table._repeated_key_sql`), in its text; `table` its rendered relation.
fn repeated_key(table: &str, key: &[String]) -> String {
    let quoted = identifiers(key);
    let present = key
        .iter()
        .map(|c| format!("{} IS NOT NULL", identifier(c)))
        .collect::<Vec<_>>()
        .join(" AND ");
    format!(
        "SELECT {quoted}\nFROM {table}\nWHERE {present}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1"
    )
}

/// A seed's integer value, if the scalar is an integer.
fn integer(seed: &Scalar) -> Option<i128> {
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

/// The reference's sample of an entity table on Postgres
/// (`_sample_entity_table`), `table` its rendered relation: `setseed` with
/// the seed's place in its range ([`seed_value`]), then the columns asked
/// for, from the rows that meet each not-null filter, in `random()` order,
/// the first `n`, in its text (register entry RS-6, KO2-24). The first item
/// is the statement before the sample, the second the sample.
fn sample_entities(
    request: &DataRequest,
    table: &str,
    key: &[String],
    n: u64,
    seed: &Scalar,
    columns: &[String],
    filters: &[Filter],
) -> Result<(Option<String>, String), CoreError> {
    if columns.is_empty() {
        return Err(sdm_contract::data::refused(
            request,
            "columns",
            "a sample names no columns",
        ));
    }
    if key.is_empty() || key.iter().any(|k| !columns.contains(k)) {
        return Err(sdm_contract::data::refused(
            request,
            "key",
            "a sample's key is not among the columns it projects",
        ));
    }
    let Some(seed) = integer(seed) else {
        return Err(sdm_contract::data::refused(
            request,
            "seed",
            "a sample's seed is an integer",
        ));
    };
    let seeded = format!("SELECT setseed({})", float_text(seed_value(seed)));
    let mut text = format!("SELECT {}\nFROM {table}", identifiers(columns));
    let filtered = conditions(request, filters, "\n")?;
    text.push_str(&filtered);
    text.push_str(&format!("\nORDER BY random()\nLIMIT {n}"));
    Ok((Some(seeded), text))
}

/// The reference's map of an integer seed into `setseed`'s `[-1, 1]`
/// (`_postgres_seed`): the first eight bytes of BLAKE2b-64 of the seed's
/// decimal text, read big-endian, scaled as Python's floats scale it
/// (`2.0 * value / (2**64 - 1) - 1.0`, each step rounded to a double).
pub(crate) fn seed_value(seed: i128) -> f64 {
    let digest = crate::blake2b::digest8(seed.to_string().as_bytes());
    let value = u64::from_be_bytes(digest);
    #[allow(
        clippy::cast_precision_loss,
        reason = "an integer to the nearest double, as the reference computes it"
    )]
    let (value, most) = (value as f64, u64::MAX as f64);
    2.0 * value / most - 1.0
}

/// A finite double as the shortest text that reads back as it, with a
/// fraction or an exponent so Postgres reads a `double precision`.
fn float_text(value: f64) -> String {
    let text = format!("{value:?}");
    if text.contains(['.', 'e', 'E']) {
        text
    } else {
        format!("{text}.0")
    }
}

/// The `WHERE` clause of `filters`, opened by `lead` (a space or a line
/// break), or nothing for none: each a not-null filter, as the reference's
/// sampler writes one, joined by `AND`; another kind of filter is not
/// taken.
fn conditions(request: &DataRequest, filters: &[Filter], lead: &str) -> Result<String, CoreError> {
    let mut conditions = Vec::with_capacity(filters.len());
    for filter in filters {
        let Filter::NotNull { column } = filter else {
            return Err(not_taken(
                request,
                "filters",
                "this rendering takes not-null filters only",
            ));
        };
        conditions.push(format!("{} IS NOT NULL", identifier(column)));
    }
    if conditions.is_empty() {
        return Ok(String::new());
    }
    let lead = if lead.is_empty() { " " } else { lead };
    Ok(format!("{lead}WHERE {}", conditions.join(" AND ")))
}

/// The common table expression `name` that unpacks the key set bound as
/// `$1` (the module's binding): one row per JSON row, each key column read
/// in the type `table` declares for the key column it joins, through
/// `jsonb_populate_record` over the table's row type, and every other
/// column of the key set (the batch and the seed times) in its own
/// type, with its trailing line break.
fn seed_cte(
    request: &DataRequest,
    table: &Relation,
    name: &str,
    key: &[String],
    seeds: &KeySet,
) -> Result<String, CoreError> {
    let row_type = from(request, "table", table)?;
    let seed_keys = seed_key_columns(key.len());
    let mut selects = Vec::with_capacity(seeds.columns.len());
    let mut built = Vec::with_capacity(key.len());
    for (position, column) in seeds.columns.iter().enumerate() {
        let cell = format!("r.value->>{position}");
        let read = if let Some(at) = seed_keys.iter().position(|k| *k == column.name) {
            match key_type(&column.column_type) {
                Some(kind) => format!("({cell})::{kind}"),
                None => {
                    built.push(format!("{}, r.value->{position}", text(&key[at])));
                    format!("k.{}", identifier(&key[at]))
                }
            }
        } else if column.name == SEED_BATCH {
            format!("({cell})::bigint")
        } else {
            match &column.column_type {
                sdm_contract::ColumnType::Timestamp { zone: None, .. } => {
                    format!("({cell})::timestamp")
                }
                _ => {
                    return Err(not_taken(
                        request,
                        "seeds",
                        "a seed column past the key, the batch and zone-less times",
                    ));
                }
            }
        };
        selects.push(format!("{read} AS {}", identifier(&column.name)));
    }
    let carried = seeds
        .columns
        .iter()
        .filter(|c| seed_keys.contains(&c.name))
        .count();
    if carried != key.len() {
        return Err(sdm_contract::data::refused(
            request,
            "seeds",
            "the seeds do not carry every key column",
        ));
    }
    let typed = if built.is_empty() {
        String::new()
    } else {
        format!(
            "\n  CROSS JOIN LATERAL jsonb_populate_record(NULL::{row_type}, jsonb_build_object({})) AS k",
            built.join(", ")
        )
    };
    Ok(format!(
        "WITH {name} AS (\n  SELECT {selects}\n  FROM jsonb_array_elements($1::jsonb) AS r(value){typed}\n)\n",
        name = identifier(name),
        selects = selects.join(", "),
    ))
}

/// The Postgres type a seed key of `column_type` is read as, for a number,
/// a boolean, a date, a time or bytes: the type of its own values, which
/// Postgres compares with the joined key's by value; `None` for text and
/// any other, read in the joined column's declared type ([`seed_cte`]).
fn key_type(column_type: &sdm_contract::ColumnType) -> Option<&'static str> {
    use sdm_contract::ColumnType as T;
    Some(match column_type {
        T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::Uint8 | T::Uint16 | T::Uint32 => "bigint",
        T::Uint64 | T::Decimal128 { .. } | T::Decimal256 { .. } => "numeric",
        T::Float32 => "real",
        T::Float64 => "double precision",
        T::Bool => "boolean",
        T::Date32 => "date",
        T::Timestamp { zone: None, .. } => "timestamp",
        T::Timestamp { zone: Some(_), .. } => "timestamptz",
        T::Binary => "bytea",
        _ => return None,
    })
}

/// The condition joining the seeds to `table`'s `key`, each key column to
/// its seed column, both in the key's declared type ([`seed_cte`]).
fn seed_join(key: &[String]) -> String {
    key.iter()
        .zip(seed_key_columns(key.len()))
        .map(|(column, seed)| format!("t.{} = s.{}", identifier(column), identifier(&seed)))
        .collect::<Vec<_>>()
        .join(" AND ")
}

/// The table's columns asked for, each read from its alias and renamed to a
/// position of its own (`c0`, `c1`, ...), so no source column's name meets
/// a helper's inside the statement; the outer select names them back.
fn positioned(
    request: &DataRequest,
    columns: &[String],
) -> Result<Vec<(String, String)>, CoreError> {
    if columns.is_empty() {
        return Err(sdm_contract::data::refused(
            request,
            "columns",
            "the request names no columns",
        ));
    }
    Ok(columns
        .iter()
        .enumerate()
        .map(|(i, c)| (format!("c{i}"), c.clone()))
        .collect())
}

/// One term of a neighbor order, as Postgres orders it, each term's null
/// order written out.
fn order_term(term: &TieTerm) -> String {
    let column = format!("t.{}", identifier(&term.column));
    let value = if term.as_timestamp {
        format!("CAST({column} AS timestamp)")
    } else {
        column
    };
    format!(
        "{value} {} {}",
        if term.descending { "DESC" } else { "ASC" },
        if term.nulls_last {
            "NULLS LAST"
        } else {
            "NULLS FIRST"
        }
    )
}

/// The reference's neighbor reads (`_by_time` and `_by_fkey` on Postgres):
/// the rows of `table` whose `key` is a seed's key, within the seed's
/// window in the source's own `CAST(column AS timestamp)` wall clock, and,
/// with a limit per seed, the first rows of each seed in `order`, read
/// for each seed (`JOIN LATERAL`, as the reference reads them), the result
/// in `(batch, order)`.
#[allow(clippy::too_many_arguments, reason = "the request's fields, rendered")]
fn neighbors(
    request: &DataRequest,
    table: &str,
    seeds: &str,
    key: &[String],
    columns: &[String],
    window: Option<&Window>,
    per_seed: Option<u64>,
    order: Option<&TieBreak>,
) -> Result<String, CoreError> {
    let positioned = positioned(request, columns)?;
    let mut within = vec![seed_join(key)];
    if let Some(window) = window {
        let time = format!("CAST(t.{} AS timestamp)", identifier(&window.column));
        if let Some(upper) = &window.upper {
            let op = if upper.inclusive { "<=" } else { "<" };
            within.push(format!("{time} {op} s.{}", identifier(&upper.seed_column)));
        }
        if let Some(lower) = &window.lower {
            let op = if lower.inclusive { ">=" } else { ">" };
            within.push(format!("{time} {op} s.{}", identifier(&lower.seed_column)));
        }
    }
    let batch = format!(
        "s.{} AS {}",
        identifier(SEED_BATCH),
        identifier(BATCH_COLUMN)
    );
    let seeds = identifier(seeds);
    match (per_seed, order) {
        (Some(limit), Some(TieBreak(terms))) => {
            let order = terms.iter().map(order_term).collect::<Vec<_>>().join(", ");
            let inner = positioned
                .iter()
                .map(|(alias, column)| format!("t.{} AS {alias}", identifier(column)))
                .collect::<Vec<_>>()
                .join(", ");
            let outer = positioned
                .iter()
                .map(|(alias, column)| format!("n.{alias} AS {}", identifier(column)))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "SELECT {batch}, {outer}\nFROM {seeds} s\nCROSS JOIN LATERAL (\n  SELECT {inner}, ROW_NUMBER() OVER (ORDER BY {order}) AS r\n  FROM {table} t\n  WHERE {within}\n  ORDER BY {order}\n  LIMIT {limit}\n) AS n\nORDER BY s.{batch_column}, n.r",
                within = within.join("\n    AND "),
                batch_column = identifier(SEED_BATCH),
            ))
        }
        (None, None) => {
            let projected = positioned
                .iter()
                .map(|(_, column)| format!("t.{}", identifier(column)))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "SELECT {batch}, {projected}\nFROM {seeds} s\nJOIN {table} t\n  ON {}",
                within.join("\n AND ")
            ))
        }
        _ => Err(sdm_contract::data::refused(
            request,
            "order",
            "a limit per seed comes with an order, and an order with a limit",
        )),
    }
}

/// The reference's key lookup (`_by_pkey` on Postgres): one row of `table`
/// per seed whose key it holds, with the count of rows holding that key,
/// the result in batch order.
fn lookup(
    request: &DataRequest,
    table: &str,
    seeds: &str,
    key: &[String],
    columns: &[String],
) -> Result<String, CoreError> {
    let positioned = positioned(request, columns)?;
    let batch = identifier(SEED_BATCH);
    let inner = positioned
        .iter()
        .map(|(alias, column)| format!("t.{} AS {alias}", identifier(column)))
        .collect::<Vec<_>>()
        .join(", ");
    let outer = positioned
        .iter()
        .map(|(alias, column)| format!("l.{alias} AS {}", identifier(column)))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "SELECT l.b AS {helper}, l.m AS {matches}, {outer}\nFROM (\n  SELECT s.{batch} AS b, COUNT(*) OVER (PARTITION BY s.{batch}) AS m, {inner},\n         ROW_NUMBER() OVER (PARTITION BY s.{batch}) AS r\n  FROM {seeds} s\n  JOIN {table} t\n    ON {on}\n) AS l\nWHERE l.r = 1\nORDER BY l.b",
        helper = identifier(BATCH_COLUMN),
        matches = identifier(MATCHES_COLUMN),
        seeds = identifier(seeds),
        on = seed_join(key),
    ))
}

/// `name` as an identifier: in double quotes, each one in it doubled.
fn identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn identifiers(names: &[String]) -> String {
    names
        .iter()
        .map(|n| identifier(n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `value` as a text literal: in single quotes, each one in it doubled,
/// and, where it holds a backslash, as an escape string with each
/// backslash doubled, so it reads as written whatever
/// `standard_conforming_strings` says.
fn text(value: &str) -> String {
    let quoted = value.replace('\'', "''");
    if value.contains('\\') {
        format!("E'{}'", quoted.replace('\\', "\\\\"))
    } else {
        format!("'{quoted}'")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_maps_as_the_reference_maps_it() {
        // `_postgres_seed(42)` and `_postgres_seed(0)` as the reference's
        // Python computes them.
        assert_eq!(seed_value(42), -0.314_812_071_551_568_76);
        assert_eq!(
            float_text(-0.314_812_071_551_568_76),
            "-0.31481207155156876"
        );
        assert_eq!(float_text(1.0), "1.0");
    }

    #[test]
    fn a_text_with_a_backslash_is_an_escape_string() {
        assert_eq!(text("a'b"), "'a''b'");
        assert_eq!(text("a\\b'"), "E'a\\\\b'''");
    }

    #[test]
    fn a_table_named_as_the_seeds_is_stepped_past() {
        assert_eq!(seeds_for(&Relation::named("t")), SEEDS);
        assert_eq!(seeds_for(&Relation::named(SEEDS)), format!("{SEEDS}_"));
    }
}
