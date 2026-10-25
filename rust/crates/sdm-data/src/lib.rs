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
mod json;
mod key_text;
mod keys;
mod mt19937;
mod pg_values;
mod postgres;
mod quote;
mod read;
mod snowflake;
mod sql;
#[cfg(test)]
mod sql_tests;
mod sqlite;
pub mod view_expr;

pub use keys::canonical_decimal_text;
pub use pg_values::json_text;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use std::collections::BTreeMap;

use sdm_contract::data::{
    AttachmentId, BoundFetch, BoundRequest, DataLayer, DataRequest, DescribePart, Dialect,
    ExpectedSchema, FailureFacts, FetchPlan, KeyData, ProviderCapabilities, ProviderChoice,
    ProviderError, Relation, SqlResultConversion, SqlStatement, SqlTimestampEncoding,
    SqlTimestampKind, views_unread,
};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
use sdm_contract::wording::DatabaseNames;
use sdm_contract::{CoreError, Detail, Sentence};
use sql::SqlDialect;

/// The SQL dialects this build knows by name, each of which it renders;
/// a name it does not know is refused without being repeated.
const KNOWN: &[&str] = &[
    duckdb::NAME,
    sqlite::NAME,
    postgres::NAME,
    databricks::NAME,
    snowflake::NAME,
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
    // A connector connection or a Snowpark session's (T3.6).
    snowflake::NAME,
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
        name: snowflake::NAME,
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
        ProviderChoice::Host { .. } | ProviderChoice::Frames { .. } => None,
    }
}

/// The data layer of a shipped build: DuckDB's SQL, and host providers.
/// It reads semantic models only where it was given the white space their
/// checks read text by ([`Data::reading_views`]), the rule the core holds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Data {
    spaces: Option<&'static view_expr::Spaces>,
}

impl Data {
    pub fn new() -> Self {
        Data { spaces: None }
    }

    /// A data layer that reads semantic models, each text question of one
    /// read by `spaces`, Python's white space as the core hands it in.
    pub fn reading_views(spaces: &'static view_expr::Spaces) -> Self {
        Data {
            spaces: Some(spaces),
        }
    }

    /// The dialect of `provider` and the white space, where this layer
    /// reads its semantic models.
    fn views(
        &self,
        provider: &ProviderChoice,
    ) -> Option<(view_expr::ExprDialect, &'static view_expr::Spaces)> {
        Some((sql_dialect(provider)?.expr_dialect()?, self.spaces?))
    }
}

impl DataLayer for Data {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        match provider {
            ProviderChoice::Frames { .. } => Err(sdm_contract::frames::not_provider()),
            ProviderChoice::Host { capabilities } => Ok(capabilities.clone()),
            ProviderChoice::Sql { dialect, view, .. } => {
                let rendered_dialect = rendered(dialect)?;
                // A semantic model is read only by a dialect that reads one,
                // with the white space it is read by; refused by name, never
                // read as the source's own tables.
                if view.is_some() && self.views(provider).is_none() {
                    return Err(views_unread());
                }
                Ok(rendered_dialect.capabilities())
            }
        }
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        self.render(request, provider, 0)
    }

    fn result_metadata(
        &self,
        provider: &ProviderChoice,
        request: &DataRequest,
        plan: &FetchPlan,
    ) -> Result<Option<FetchPlan>, CoreError> {
        if !matches!(
            provider,
            ProviderChoice::Sql {
                timestamp_encoding: Some(SqlTimestampEncoding::Microseconds),
                ..
            }
        ) || matches!(
            request,
            DataRequest::Describe { .. }
                | DataRequest::KeyCounts { .. }
                | DataRequest::CountRows { .. }
        ) {
            return Ok(None);
        }
        if sql_dialect(provider).is_none_or(|d| {
            d.timestamp_micros("value", SqlTimestampKind::WallTime)
                .is_none()
        }) {
            return Err(CoreError::new(
                Sentence::Undeclared,
                "the data source does not support the requested timestamp transport",
            )
            .with_field("request", Detail::text("timestamp_encoding")));
        }
        let FetchPlan::Sql { statements, .. } = plan else {
            return Ok(None);
        };
        let mut statements = statements.clone();
        let last = statements.last_mut().ok_or_else(|| {
            CoreError::new(Sentence::Internal, "a SQL result plan has no statement")
        })?;
        last.text = format!("DESCRIBE QUERY {}", last.text);
        Ok(Some(FetchPlan::Sql {
            statements,
            expect: ExpectedSchema::Discover {},
            result_conversions: Vec::new(),
        }))
    }

    fn project_result(
        &self,
        provider: &ProviderChoice,
        request: &DataRequest,
        plan: &FetchPlan,
        metadata: &[RecordBatch],
    ) -> Result<FetchPlan, CoreError> {
        let mut conversions = Vec::new();
        let mut position = 0_u32;
        for batch in metadata {
            let names = read::text_at(batch, 0)?;
            let types = read::text_at(batch, 1)?;
            for row in 0..batch.num_rows() {
                let name = read::text(&names, row).ok_or_else(|| {
                    CoreError::new(Sentence::Internal, "a result metadata column has no name")
                })?;
                let kind = read::text(&types, row).ok_or_else(|| {
                    CoreError::new(Sentence::Internal, "a result metadata column has no type")
                })?;
                let kind = kind.trim().to_ascii_lowercase();
                let source_kind = if kind == "timestamp_ntz" || kind.starts_with("timestamp_ntz(") {
                    Some(SqlTimestampKind::WallTime)
                } else if kind == "timestamp"
                    || kind.starts_with("timestamp(")
                    || kind == "timestamp_ltz"
                    || kind.starts_with("timestamp_ltz(")
                {
                    Some(SqlTimestampKind::Instant)
                } else {
                    None
                };
                if let Some(source_kind) = source_kind {
                    conversions.push(SqlResultConversion::TimestampMicros {
                        column: position,
                        name,
                        source_kind,
                    });
                }
                position = position.checked_add(1).ok_or_else(|| {
                    CoreError::new(Sentence::Internal, "too many result metadata columns")
                })?;
            }
        }
        if position == 0 {
            return Err(CoreError::new(
                Sentence::Internal,
                "the warehouse returned no result metadata columns",
            ));
        }
        if conversions.is_empty() {
            return Ok(plan.clone());
        }
        self.render_projected(request, provider, 0, &conversions)
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

    /// `plan` bound as the run's provider binds it: a built-in dialect's key
    /// set written as the text its statement reads where it binds one so,
    /// split past its budget, one fetch per chunk ([`key_text`], T3.28); a
    /// host provider's plan, or one binding the core's Arrow key set as a
    /// relation, one fetch as it is.
    fn bind(
        &self,
        provider: &ProviderChoice,
        request: &DataRequest,
        plan: FetchPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
        float_text: fn(f64) -> String,
    ) -> Result<BoundRequest, CoreError> {
        let conversions = match &plan {
            FetchPlan::Sql {
                result_conversions, ..
            } => result_conversions.clone(),
            _ => Vec::new(),
        };
        match sql_dialect(provider) {
            Some(dialect) => key_text::bind(dialect, request, plan, keys, float_text, &|offset| {
                self.render_projected(request, provider, offset, &conversions)
            }),
            None => Ok(BoundRequest {
                fetches: vec![BoundFetch {
                    plan,
                    batches: BTreeMap::new(),
                }],
            }),
        }
    }

    fn holds_column_types(&self, provider: &ProviderChoice) -> bool {
        sql_dialect(provider).is_none_or(|dialect| dialect.holds_column_types())
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
            ProviderChoice::Host { .. } | ProviderChoice::Frames { .. } => false,
        }
    }

    /// The full namespace answer, settled by its dialect: an already
    /// resolved answer is retained; raw session parts are combined with
    /// supplied parts according to that dialect's namespace alignment.
    fn settled(&self, provider: &ProviderChoice, parts: Vec<String>) -> ProviderChoice {
        match provider {
            ProviderChoice::Sql {
                dialect,
                namespace,
                view,
                timestamp_encoding,
            } => {
                let parts = match sql_dialect(provider) {
                    Some(dialect) => dialect.settled_namespace(namespace, parts),
                    None => parts,
                };
                ProviderChoice::Sql {
                    timestamp_encoding: *timestamp_encoding,
                    dialect: dialect.clone(),
                    namespace: parts.into_iter().map(Some).collect(),
                    view: view.clone(),
                }
            }
            ProviderChoice::Host { .. } | ProviderChoice::Frames { .. } => provider.clone(),
        }
    }

    fn view_kind(&self, provider: &ProviderChoice) -> Option<sdm_contract::view::ViewKind> {
        sql_dialect(provider)?.view_kind()
    }

    fn failure(&self, provider: &ProviderChoice, error: &ProviderError) -> FailureFacts {
        sql_dialect(provider).map_or_else(FailureFacts::default, |dialect| dialect.failure(error))
    }

    fn view_text(
        &self,
        provider: &ProviderChoice,
        ask: ViewTextAsk,
    ) -> Result<ViewTextAnswer, CoreError> {
        let (Some((dialect, spaces)), Some(sql)) = (self.views(provider), sql_dialect(provider))
        else {
            return Err(views_unread());
        };
        Ok(view_expr::answer(ask, dialect, spaces, &|data_type| {
            sql.declared_type(data_type)
        }))
    }

    fn spellings(&self, provider: &ProviderChoice, relation: &Relation) -> Vec<Relation> {
        sql_dialect(provider).map_or_else(|| vec![relation.clone()], |d| d.spellings(relation))
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
            ProviderChoice::Frames { .. } => return Err(sdm_contract::frames::not_provider()),
            ProviderChoice::Host { .. } => read::host(*part, batches)?,
            ProviderChoice::Sql { dialect, .. } if *part == DescribePart::ViewDefinition => {
                let spaces = self.spaces.ok_or_else(views_unread)?;
                rendered(dialect)?.read_view(request, batches, spaces)?
            }
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

impl Data {
    /// [`DataLayer::plan`], a key set's seeds numbered from `offset`: a
    /// later chunk's statements, rendered as the first's are ([`key_text`]).
    fn render(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        offset: usize,
    ) -> Result<FetchPlan, CoreError> {
        self.render_projected(request, provider, offset, &[])
    }

    fn render_projected(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        offset: usize,
        conversions: &[SqlResultConversion],
    ) -> Result<FetchPlan, CoreError> {
        match provider {
            ProviderChoice::Frames { .. } => Err(sdm_contract::frames::not_provider()),
            ProviderChoice::Host { .. } => Ok(FetchPlan::Host {
                expect: request.expected(),
            }),
            ProviderChoice::Sql {
                dialect, namespace, ..
            } => {
                let dialect = rendered(dialect)?;
                let plan = |request: &DataRequest, namespace: &[String]| {
                    if conversions.is_empty() {
                        sql::plan(dialect, request, namespace, offset)
                    } else {
                        sql::plan_projected(dialect, request, namespace, offset, conversions)
                    }
                };
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
                        result_conversions: Vec::new(),
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
                    return plan(request, &[]);
                }
                let namespace = if matches!(
                    request,
                    DataRequest::Describe {
                        part: DescribePart::ViewDefinition,
                        ..
                    }
                ) && dialect.view_kind()
                    == Some(sdm_contract::view::ViewKind::Semantic)
                {
                    // The semantic lookup binds a name, prefixed only by
                    // the source parts explicitly supplied by its caller.
                    // Missing parts are the warehouse's to resolve, even
                    // where a physical relation would need a namespace read.
                    namespace.iter().flatten().cloned().collect()
                } else {
                    qualifying(request, namespace)?
                };
                deeper(request, &namespace, dialect.namespace_depth())?;
                plan(&in_namespace(request, &namespace), &namespace)
            }
        }
    }
}
