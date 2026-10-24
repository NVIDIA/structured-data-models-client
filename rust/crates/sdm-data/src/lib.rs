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
//!
//! [`view_expr`] checks the expressions a metric or semantic view holds,
//! as each dialect reads them, before any reaches a statement (T3.5).
//!
//! The statements are written once, in [`sql`]'s shared builder, over its
//! `SqlDialect` trait (T3.18, design fix E): each dialect's module holds
//! only what differs, and [`Data`] finds a dialect by its name in one
//! lookup.

#![forbid(unsafe_code)]

mod blake2b;
mod databricks;
mod duckdb;
mod keys;
mod mt19937;
mod pg_values;
mod postgres;
mod quote;
mod read;
mod sql;
#[cfg(test)]
mod sql_tests;
mod sqlite;
pub mod view_expr;

pub use keys::canonical_decimal_text;
pub use pg_values::json_text;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, FetchPlan, ProviderCapabilities, ProviderChoice,
    Relation, SqlStatement, views_unread,
};
use sdm_contract::wording::DatabaseNames;
use sdm_contract::{CoreError, Detail, Sentence};
use sql::SqlDialect;

/// The SQL dialects this build knows by name. It renders DuckDB's,
/// SQLite's and Postgres's (T2.2, T2.3); the others are refused by name,
/// as a dialect it has no rendering for, and a name it does not know is
/// refused without being repeated.
const KNOWN: &[&str] = &[
    duckdb::NAME,
    sqlite::NAME,
    postgres::NAME,
    databricks::NAME,
    "snowflake",
];

/// The databases this build's data layer provides for, by the names the
/// case format and the contract runners use (T2.1a): the dialects it
/// renders. The private, plain-data marker of the staging rule
/// (`contract/runners/run_python.py`'s `check_legs`): the Python host's
/// native module reports it as `PROVIDERS`, and a database's core legs run
/// exactly when it is listed. A dialect's package adds its name with its
/// rendering.
pub const PROVIDERS: &[&str] = &[
    duckdb::NAME,
    sqlite::NAME,
    postgres::NAME,
    // Databricks's two legs, a SQL warehouse and a Spark session, which
    // both run the one `databricks` dialect (T3.4).
    "databricks-sql",
    "databricks-spark",
];

/// Each database a sentence may name (`{database:backend}`,
/// `{database:place}`, T3.17), as the reference names it: as a product, and
/// as the place a table is held. A warehouse is named as the reference's
/// remote data backend.
pub const DATABASE_NAMES: &[DatabaseNames] = &[
    DatabaseNames {
        name: duckdb::NAME,
        backend: "DuckDB",
        place: "DuckDB database",
    },
    DatabaseNames {
        name: sqlite::NAME,
        backend: "SQLite",
        place: "SQLite database",
    },
    DatabaseNames {
        name: postgres::NAME,
        backend: "PostgreSQL",
        place: "PostgreSQL database",
    },
    DatabaseNames {
        name: "databricks",
        backend: "Databricks",
        place: "remote data backend",
    },
    DatabaseNames {
        name: "snowflake",
        backend: "Snowflake",
        place: "remote data backend",
    },
];

/// The dialect `dialect` names, if this build renders it.
fn rendered(dialect: &Dialect) -> Result<&'static dyn SqlDialect, CoreError> {
    sql::dialect(dialect.name()).ok_or_else(|| no_dialect(dialect))
}

/// The dialect of `provider`, if it is a built-in SQL provider of a
/// dialect this build renders.
fn sql_dialect(provider: &ProviderChoice) -> Option<&'static dyn SqlDialect> {
    match provider {
        ProviderChoice::Sql { dialect, .. } => sql::dialect(dialect.name()),
        ProviderChoice::Host { .. } => None,
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
            ProviderChoice::Sql { dialect, view, .. } => {
                let dialect = rendered(dialect)?;
                // No dialect reads a semantic model yet (T3.5 and T3.6 do):
                // refused by name, never read as the source's own tables.
                if view.is_some() {
                    return Err(views_unread());
                }
                Ok(dialect.capabilities())
            }
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
            ProviderChoice::Sql {
                dialect, namespace, ..
            } => {
                let dialect = rendered(dialect)?;
                // The namespace itself, read where a part is left to the
                // connection inside one given, which the core then reads
                // every relation under (T3.4).
                if let DataRequest::Describe {
                    part: DescribePart::Namespace,
                    ..
                } = request
                    && given_parts(namespace).is_none()
                {
                    let parts: Vec<String> = namespace
                        .iter()
                        .map(|part| part.clone().unwrap_or_default())
                        .collect();
                    deeper(request, &parts, dialect.namespace_depth())?;
                    let (text, params) = dialect.current_namespace(request, namespace)?;
                    return Ok(FetchPlan::Sql {
                        statements: vec![SqlStatement {
                            dialect: Dialect::new(dialect.name()),
                            text,
                            params,
                        }],
                        expect: dialect.expected(request),
                    });
                }
                // A request of a dialect that reads every table under its
                // full namespace, whose every relation names it in full (a
                // table the core reads where it was described), reads none
                // of the parts the provider leaves to the connection (T3.4).
                let under = request.relations();
                if dialect.settles_namespace()
                    && !under.is_empty()
                    && under
                        .iter()
                        .all(|(_, r)| r.source.len() == dialect.namespace_depth() + 1)
                {
                    return sql::plan(dialect, request, &[]);
                }
                let namespace = qualifying(request, namespace)?;
                deeper(request, &namespace, dialect.namespace_depth())?;
                sql::plan(dialect, &in_namespace(request, &namespace), &namespace)
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
        match sql_dialect(provider) {
            Some(dialect) => dialect.conform(schema, batches, float_text),
            None => Ok(None),
        }
    }

    fn bound_keys(
        &self,
        provider: &ProviderChoice,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<Option<(arrow_schema::SchemaRef, Vec<RecordBatch>)>, CoreError> {
        match sql_dialect(provider) {
            Some(dialect) => dialect.bound_keys(schema, batches),
            None => Ok(None),
        }
    }

    fn source_database(&self, provider: &ProviderChoice) -> Option<&'static str> {
        Some(sql_dialect(provider)?.name())
    }

    fn reads_tables_as_they_stand(&self, provider: &ProviderChoice) -> bool {
        sql_dialect(provider).is_some()
    }

    fn unsettled(&self, provider: &ProviderChoice) -> bool {
        match provider {
            ProviderChoice::Sql {
                dialect, namespace, ..
            } => {
                given_parts(namespace).is_none()
                    || sql::dialect(dialect.name()).is_some_and(|d| {
                        d.settles_namespace()
                            && namespace.iter().flatten().count() < d.namespace_depth()
                    })
            }
            ProviderChoice::Host { .. } => false,
        }
    }

    fn settled(&self, provider: &ProviderChoice, parts: Vec<String>) -> ProviderChoice {
        match provider {
            ProviderChoice::Sql { dialect, view, .. } => ProviderChoice::Sql {
                dialect: dialect.clone(),
                namespace: parts.into_iter().map(Some).collect(),
                view: view.clone(),
            },
            ProviderChoice::Host { .. } => provider.clone(),
        }
    }

    fn read_as(&self, provider: &ProviderChoice, relation: &Relation) -> Relation {
        let parts = match provider {
            ProviderChoice::Sql { namespace, .. } if relation.source.len() == 1 => {
                // A namespace no relation can write was refused before any
                // relation was read.
                given_parts(namespace).unwrap_or_default()
            }
            _ => Vec::new(),
        };
        Relation::of(
            parts
                .into_iter()
                .chain(relation.source.iter().cloned())
                .collect(),
        )
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                Sentence::Internal,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let answer = match provider {
            ProviderChoice::Host { .. } => read::host(*part, batches)?,
            ProviderChoice::Sql { dialect, .. } => {
                rendered(dialect)?.read(request, *part, batches)?
            }
        };
        debug_assert_eq!(answer.part(), *part);
        Ok(answer)
    }
}

/// The parts a relation of one part is read under, of the provider's
/// `namespace` (contract v28): the parts given, outermost first, those
/// left `null` before them dropped, since a relation of fewer parts is
/// read in the connection's current ones. A part given inside one left
/// `null` cannot be written as a relation, and is refused by name.
fn qualifying(
    request: &DataRequest,
    namespace: &[Option<String>],
) -> Result<Vec<String>, CoreError> {
    given_parts(namespace).ok_or_else(|| {
        sql::not_taken(
            request,
            "namespace",
            "a namespace part left to the connection inside one given cannot qualify a relation",
        )
    })
}

/// The one rule of a namespace's parts a relation is read under: those
/// given, leading parts left `null` dropped; `None` where a part given
/// follows one left `null`, which no relation can write.
fn given_parts(namespace: &[Option<String>]) -> Option<Vec<String>> {
    let given: Vec<&Option<String>> = namespace.iter().skip_while(|part| part.is_none()).collect();
    if given.iter().any(|part| part.is_none()) {
        return None;
    }
    Some(given.into_iter().flatten().cloned().collect())
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
    for relation in request.relations_mut() {
        if relation.source.len() == 1 {
            relation.source.splice(0..0, namespace.iter().cloned());
        }
    }
    request
}

/// Refuses, by name, a namespace of more parts than the dialect's sources
/// take (design fix C): any for DuckDB's and SQLite's, which have none to
/// name, more than a schema for Postgres's, more than a catalog and a
/// schema for Databricks's.
fn deeper(request: &DataRequest, namespace: &[String], depth: usize) -> Result<(), CoreError> {
    if namespace.len() <= depth {
        return Ok(());
    }
    if depth == 0 {
        return Err(sql::not_taken(
            request,
            "namespace",
            "this dialect's sources have no namespace to read under",
        ));
    }
    Err(sql::not_taken(
        request,
        "namespace",
        "a namespace of more parts than this dialect's sources take",
    )
    .with_field(
        "parts",
        Detail::Scalar(sdm_contract::Scalar::UInt64(namespace.len() as u64)),
    )
    .with_field(
        "most",
        Detail::Scalar(sdm_contract::Scalar::UInt64(depth as u64)),
    ))
}

/// The refusal of a SQL dialect this build renders nothing for, naming it
/// when it is one this build knows by name.
fn no_dialect(dialect: &Dialect) -> CoreError {
    let message = "this build has no built-in SQL provider for the dialect asked for";
    match KNOWN.iter().find(|known| **known == dialect.name()) {
        Some(known) => CoreError::new(Sentence::NoDialectNamed, message)
            .with_field("dialect", Detail::text(*known)),
        None => CoreError::new(Sentence::NoDialect, message),
    }
}
