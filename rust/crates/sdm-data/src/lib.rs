// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data layer (Phase 1b): it plans each typed data request as SQL in a
//! dialect, or passes it to a host provider (decision H14). SQL dialects
//! live here and in no other domain crate; the wire names a source's
//! backend only for this crate to interpret.
//!
//! [`Data`] is the data layer `sdm_host::default_root` builds (decision
//! E30). A host provider is answered with the typed request itself, and is
//! asked only for what it declared; a built-in SQL provider is answered
//! with statements this crate renders for its dialect, and its
//! capabilities are the requests this crate renders for that dialect. As
//! built at T1.11b DuckDB renders the catalog (`describe` in each part,
//! `preview`, `key_counts`, `time_bounds`), and from T1.11c the sampler's
//! `sample_entities`, `neighbors` and `lookup`, and from T1.32e
//! `column_ranges` and `repeated_key`; the rest come with the
//! tasks that reach them. A plan expects the schema the contract fixes for
//! the request's answer, or discovers it where the answer's columns are the
//! source's own (`DataRequest::expected`, decision E36), so a host provider
//! is held to the same shape as a rendering; but a `describe`, whose answer
//! is a catalog's: a dialect's plan expects its own catalog's shape, and a
//! host provider answers in the contract's.
//!
//! [`DataLayer::read`] reads each `describe` answer, after the core has
//! admitted it, into the contract's dialect-neutral catalog answer
//! (decision E54): a dialect's reader knows its catalog's columns and
//! spelling (DuckDB's `PRAGMA table_info`, its constraint types, its index
//! expressions), so no other crate does. `l0_scan.py`'s `dialects` check
//! keeps them here.

#![forbid(unsafe_code)]

mod blake2b;
mod duckdb;
mod keys;
mod mt19937;
mod pg_values;
mod postgres;
mod quote;
mod read;
mod sqlite;

pub use keys::canonical_decimal_text;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, Dialect, FetchPlan, ProviderCapabilities, ProviderChoice, Relation,
    refuse_relation,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The SQL dialects this build knows by name. It renders DuckDB's,
/// SQLite's and Postgres's (T2.2, T2.3); the others are refused by name,
/// as a dialect it has no rendering for, and a name it does not know is
/// refused without being repeated.
const KNOWN: &[&str] = &[
    duckdb::NAME,
    sqlite::NAME,
    postgres::NAME,
    "databricks",
    "snowflake",
];

/// The databases this build's data layer provides for, by the names the
/// case format and the contract runners use (T2.1a): the dialects it
/// renders. The private, plain-data marker of the staging rule
/// (`contract/runners/run_python.py`'s `check_legs`): the Python host's
/// native module reports it as `PROVIDERS`, and a database's core legs run
/// exactly when it is listed. A dialect's package adds its name with its
/// rendering.
pub const PROVIDERS: &[&str] = &[duckdb::NAME, sqlite::NAME, postgres::NAME];

/// A dialect this build renders.
enum Rendered {
    Duckdb,
    Sqlite,
    Postgres,
}

/// The dialect `dialect` names, if this build renders it.
fn rendered(dialect: &Dialect) -> Result<Rendered, CoreError> {
    match dialect.name() {
        duckdb::NAME => Ok(Rendered::Duckdb),
        sqlite::NAME => Ok(Rendered::Sqlite),
        postgres::NAME => Ok(Rendered::Postgres),
        _ => Err(no_dialect(dialect)),
    }
}

/// The data layer of a shipped build: DuckDB's SQL, and host providers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Data;

impl Data {
    pub fn new() -> Self {
        Data
    }
}

impl DataLayer for Data {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        match provider {
            ProviderChoice::Host { capabilities } => Ok(capabilities.clone()),
            ProviderChoice::Sql { dialect, .. } => match rendered(dialect)? {
                Rendered::Duckdb => Ok(duckdb::capabilities()),
                Rendered::Sqlite => Ok(sqlite::capabilities()),
                Rendered::Postgres => Ok(postgres::capabilities()),
            },
        }
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        match provider {
            ProviderChoice::Host { .. } => Ok(FetchPlan::Host {
                expect: request.expected(),
            }),
            ProviderChoice::Sql { dialect, namespace } => match rendered(dialect)? {
                Rendered::Duckdb => {
                    no_namespace(request, &qualifying(request, namespace)?)?;
                    duckdb::plan(request)
                }
                Rendered::Sqlite => {
                    no_namespace(request, &qualifying(request, namespace)?)?;
                    sqlite::plan(request)
                }
                Rendered::Postgres => {
                    let namespace = qualifying(request, namespace)?;
                    postgres::plan(&in_namespace(request, &namespace), &namespace)
                }
            },
        }
    }

    fn opaque_vendor(&self, provider: &ProviderChoice) -> Option<&'static str> {
        match provider {
            ProviderChoice::Host { .. } => None,
            ProviderChoice::Sql { dialect, .. } => match rendered(dialect).ok()? {
                Rendered::Duckdb => Some(duckdb::OPAQUE_VENDOR),
                Rendered::Sqlite => None,
                Rendered::Postgres => None,
            },
        }
    }

    fn compares_stored_times(&self, provider: &ProviderChoice) -> bool {
        match provider {
            ProviderChoice::Host { .. } => false,
            ProviderChoice::Sql { dialect, .. } => {
                matches!(rendered(dialect), Ok(Rendered::Sqlite))
            }
        }
    }

    fn samples_by_key_dtype(&self, provider: &ProviderChoice) -> bool {
        match provider {
            ProviderChoice::Host { .. } => false,
            ProviderChoice::Sql { dialect, .. } => {
                matches!(rendered(dialect), Ok(Rendered::Sqlite))
            }
        }
    }

    fn conform(
        &self,
        provider: &ProviderChoice,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<(arrow_schema::SchemaRef, Vec<RecordBatch>)>, CoreError> {
        match provider {
            ProviderChoice::Sql { dialect, .. }
                if matches!(rendered(dialect), Ok(Rendered::Postgres)) =>
            {
                pg_values::conform(schema, batches, float_text)
            }
            _ => Ok(None),
        }
    }

    fn bound_keys(
        &self,
        provider: &ProviderChoice,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<Option<(arrow_schema::SchemaRef, Vec<RecordBatch>)>, CoreError> {
        match provider {
            ProviderChoice::Sql { dialect, .. }
                if matches!(rendered(dialect), Ok(Rendered::Postgres)) =>
            {
                pg_values::bound_keys(schema, batches)
            }
            _ => Ok(None),
        }
    }

    fn reads_nan_as_missing(&self, provider: &ProviderChoice) -> bool {
        match provider {
            ProviderChoice::Host { .. } => false,
            ProviderChoice::Sql { dialect, .. } => {
                matches!(rendered(dialect), Ok(Rendered::Postgres))
            }
        }
    }

    fn source_database(&self, provider: &ProviderChoice) -> Option<&'static str> {
        match provider {
            ProviderChoice::Host { .. } => None,
            ProviderChoice::Sql { dialect, .. } => match rendered(dialect).ok()? {
                Rendered::Duckdb => Some(duckdb::NAME),
                Rendered::Sqlite => Some(sqlite::NAME),
                Rendered::Postgres => Some(postgres::NAME),
            },
        }
    }

    fn read_as(&self, provider: &ProviderChoice, relation: &Relation) -> Relation {
        let parts = match provider {
            ProviderChoice::Sql { namespace, .. } if relation.source.len() == 1 => {
                // The parts given, as `qualifying` reads them; a namespace
                // it refuses was refused before any relation was read.
                namespace
                    .iter()
                    .skip_while(|part| part.is_none())
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        };
        Relation {
            source: parts
                .into_iter()
                .chain(relation.source.iter().cloned())
                .collect(),
        }
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                ErrorKind::Internal,
                ErrorCode::InternalError,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let answer = match provider {
            ProviderChoice::Host { .. } => read::host(*part, batches)?,
            ProviderChoice::Sql { dialect, .. } => match rendered(dialect)? {
                Rendered::Duckdb => duckdb::read(*part, batches)?,
                Rendered::Sqlite => sqlite::read(request, *part, batches)?,
                Rendered::Postgres => postgres::read(*part, batches)?,
            },
        };
        debug_assert_eq!(answer.part(), *part);
        Ok(answer)
    }
}

/// The one rendering of a relation every dialect's `FROM` goes through
/// (design choice 12): each part of `relation` quoted alone by the
/// dialect's `quote`, never split on a dot, joined by dots. A relation of
/// no part, or of more than `most`, the parts the dialect accepts, is
/// refused by name (rule `relation`, [`refuse_relation`]) as `field` of
/// `request`. The design gives each dialect its count: DuckDB 1 to 3
/// (database, schema, table); SQLite 1 or 2, the first naming `main` or an
/// attached database; Postgres 1 or 2, schema and table; each declared
/// with its dialect's rendering.
pub(crate) fn relation(
    request: &DataRequest,
    field: &str,
    relation: &Relation,
    most: usize,
    quote: fn(&str) -> String,
) -> Result<String, CoreError> {
    if let Some(refusal) = refuse_relation(request, field, relation, most) {
        return Err(refusal);
    }
    Ok(relation
        .source
        .iter()
        .map(|part| quote(part))
        .collect::<Vec<_>>()
        .join("."))
}

/// The refusal of a SQL dialect this build renders nothing for, naming it
/// when it is one this build knows by name.
/// The parts a relation of one part is read under, of the provider's
/// `namespace` (contract v28): the parts given, outermost first, those
/// left `null` before them dropped, since a relation of fewer parts is
/// read in the connection's current ones. A part given inside one left
/// `null` cannot be written as a relation, and is refused by name.
fn qualifying(
    request: &DataRequest,
    namespace: &[Option<String>],
) -> Result<Vec<String>, CoreError> {
    let given: Vec<&Option<String>> = namespace.iter().skip_while(|part| part.is_none()).collect();
    if given.iter().any(|part| part.is_none()) {
        return Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "a namespace part left to the connection inside one given cannot qualify a relation",
        )
        .with_field("request", Detail::text(request.name()))
        .with_field("field", Detail::text("namespace")));
    }
    Ok(given.into_iter().flatten().cloned().collect())
}

/// `request` with each relation of one part read under `namespace`, the
/// parts [`qualifying`] gives: the namespace's parts, then the table's, as
/// the reference qualifies each read of a Postgres source by the schema it
/// was given (`PostgresTable._quoted_source_name`). A relation of several
/// parts already names where it is, and is left as it is.
fn in_namespace(request: &DataRequest, namespace: &[String]) -> DataRequest {
    let mut request = request.clone();
    if namespace.is_empty() {
        return request;
    }
    let under = |relation: &mut Relation| {
        if relation.source.len() == 1 {
            relation.source.splice(0..0, namespace.iter().cloned());
        }
    };
    match &mut request {
        DataRequest::Describe { tables, .. } => tables.iter_mut().for_each(under),
        DataRequest::KeyCounts { keys } => keys.iter_mut().for_each(|k| under(&mut k.table)),
        DataRequest::TimeBounds { columns } => {
            columns.iter_mut().for_each(|c| under(&mut c.table));
        }
        DataRequest::Scan { table, .. }
        | DataRequest::Preview { table, .. }
        | DataRequest::CountRows { table, .. }
        | DataRequest::ColumnRanges { table, .. }
        | DataRequest::RepeatedKey { table, .. }
        | DataRequest::SampleEntities { table, .. }
        | DataRequest::Neighbors { table, .. }
        | DataRequest::Lookup { table, .. } => under(table),
    }
    request
}

/// Refuses a namespace given to a dialect whose sources have none to name
/// (DuckDB's and SQLite's connectors take no namespace), by name.
fn no_namespace(request: &DataRequest, namespace: &[String]) -> Result<(), CoreError> {
    if namespace.is_empty() {
        return Ok(());
    }
    Err(CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "this dialect's sources have no namespace to read under",
    )
    .with_field("request", Detail::text(request.name()))
    .with_field("field", Detail::text("namespace")))
}

fn no_dialect(dialect: &Dialect) -> CoreError {
    let error = CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "this build has no built-in SQL provider for the dialect asked for",
    );
    match KNOWN.iter().find(|known| **known == dialect.name()) {
        Some(known) => error.with_field("dialect", Detail::text(*known)),
        None => error,
    }
}
