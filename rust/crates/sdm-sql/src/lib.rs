// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQL planning and catalog decoding for the built-in sources.
//! This crate owns dialect-specific SQL; the core sends every source typed
//! requests and never names a dialect.
//!
//! A built-in source answers the core's typed requests through its SQL kit
//! ([`kit::SqlKit`]), over [`Sql`]: the source's statements rendered in its
//! dialect ([`plan::SqlPlan`]), its key sets bound as the dialect binds
//! them, its answers read as the reference reads them, and its catalog's
//! answers read into neutral facts, and written in the contract's own
//! shapes. [`Databases`] is the core's data layer: the semantic models a
//! source is read through, by the database it declares.
//!
//! [`sql`] shares statement construction through its dialect trait, while each
//! connector module supplies its differences.
//! [`view_expr`] validates semantic-view expressions before they enter statements.

#![forbid(unsafe_code)]

mod blake2b;
mod databricks;
mod duckdb;
mod failure;
mod json;
mod key_text;
mod keys;
pub mod kit;
mod mt19937;
mod pg_values;
pub mod plan;
mod postgres;
mod quote;
mod snowflake;
mod sql;
#[cfg(test)]
mod sql_tests;
mod sqlite;
mod standard;
pub mod view_expr;

pub use keys::canonical_decimal_text;
pub use pg_values::json_text;

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use std::collections::BTreeMap;

use plan::{
    Dialect, SqlChoice, SqlPlan, SqlResultConversion, SqlStatement, SqlTimestampEncoding,
    SqlTimestampKind,
};
use sdm_contract::catalog_read;
use sdm_contract::data::{
    AttachmentId, BoundRequest, DataLayer, DataRequest, DescribePart, ExpectedSchema, FailureFacts,
    KeyData, ProviderCapabilities, ProviderError, Relation, views_unread,
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

/// The databases this build's SQL kit provides for, by the names the
/// case format and the contract runners use: the dialects it
/// renders. The private, plain-data marker of the staging rule
/// (the current runner admission check): the Python host's
/// native module reports it as `PROVIDERS`, and a database's core legs run
/// exactly when it is listed. A dialect's package adds its name with its
/// rendering.
pub const PROVIDERS: &[&str] = &[
    duckdb::NAME,
    sqlite::NAME,
    postgres::NAME,
    // Databricks's two legs, a SQL warehouse and a Spark session, which
    // both run the one `databricks` dialect.
    "databricks-sql",
    "databricks-spark",
    // A connector connection or a Snowpark session's.
    snowflake::NAME,
];

/// Each database a sentence may name (`{database:backend}`,
/// `{database:place}`), as the reference names it: as a product, and
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

/// The dialect of `source`, if it is one this build renders.
fn sql_dialect(source: &SqlChoice) -> Option<&'static dyn SqlDialect> {
    sql::dialect(source.dialect.name())
}

/// The dialect whose semantic models a source declaring `capabilities` is
/// read through: the one its declared database names (contract v76).
fn view_dialect(capabilities: &ProviderCapabilities) -> Option<&'static dyn SqlDialect> {
    sql::dialect(capabilities.database.as_deref()?)
}

/// The SQL of the built-in sources: each request rendered in a source's
/// dialect, bound, conformed and read as its dialect does. It reads
/// semantic models only where it was given the white space their checks
/// read text by ([`Sql::reading_views`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sql {
    spaces: Option<&'static view_expr::Spaces>,
}

/// The core's data layer over the built-in databases, by the database a
/// source declares ([`ProviderCapabilities::database`]): the semantic
/// models it is read through, with the white space their checks read text
/// by.
#[derive(Debug, Clone, Copy)]
pub struct Databases {
    spaces: &'static view_expr::Spaces,
}

impl Databases {
    /// The data layer reading semantic models, each text question of one
    /// read by `spaces`, Python's white space as the core hands it in.
    pub fn new(spaces: &'static view_expr::Spaces) -> Self {
        Databases { spaces }
    }

    /// The dialect of the database `capabilities` declares, where it reads
    /// semantic models.
    fn dialect(
        capabilities: &ProviderCapabilities,
    ) -> Option<(&'static dyn SqlDialect, view_expr::ExprDialect)> {
        let dialect = view_dialect(capabilities)?;
        Some((dialect, dialect.expr_dialect()?))
    }
}

impl DataLayer for Databases {
    fn view_kind(
        &self,
        capabilities: &ProviderCapabilities,
    ) -> Option<sdm_contract::view::ViewKind> {
        view_dialect(capabilities)?.view_kind()
    }

    fn view_text(
        &self,
        capabilities: &ProviderCapabilities,
        ask: ViewTextAsk,
    ) -> Result<ViewTextAnswer, CoreError> {
        let (sql, dialect) = Databases::dialect(capabilities).ok_or_else(views_unread)?;
        Ok(view_expr::answer(ask, dialect, self.spaces, &|data_type| {
            sql.declared_type(data_type)
        }))
    }

    fn read_view(
        &self,
        capabilities: &ProviderCapabilities,
        request: &DataRequest,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        view_dialect(capabilities)
            .ok_or_else(views_unread)?
            .read_view(request, batches, self.spaces)
    }
}

impl Sql {
    pub fn new() -> Self {
        Sql { spaces: None }
    }

    /// The SQL that reads semantic models, each text question of one read
    /// by `spaces`, Python's white space as the core hands it in.
    pub fn reading_views(spaces: &'static view_expr::Spaces) -> Self {
        Sql {
            spaces: Some(spaces),
        }
    }

    /// The dialect of `source` and the white space, where this reads its
    /// semantic models.
    fn views(
        &self,
        source: &SqlChoice,
    ) -> Option<(view_expr::ExprDialect, &'static view_expr::Spaces)> {
        Some((sql_dialect(source)?.expr_dialect()?, self.spaces?))
    }

    /// What `source` answers: every request and filter its dialect writes.
    /// A dialect this build has no rendering for is refused with
    /// `UNSUPPORTED_REQUEST` naming it, as is a semantic model its dialect
    /// does not read.
    pub fn capabilities(&self, source: &SqlChoice) -> Result<ProviderCapabilities, CoreError> {
        let rendered_dialect = rendered(&source.dialect)?;
        // A semantic model is read only by a dialect that reads one,
        // with the white space it is read by; refused by name, never
        // read as the source's own tables.
        if source.view.is_some() && self.views(source).is_none() {
            return Err(views_unread());
        }
        Ok(rendered_dialect.capabilities())
    }

    /// The statements `source` answers `request` with.
    pub fn plan(&self, request: &DataRequest, source: &SqlChoice) -> Result<SqlPlan, CoreError> {
        let mut plan = self.render(request, source, 0)?;
        if sql_dialect(source).is_some_and(|dialect| dialect.name() == "postgres") {
            plan.result_conversions
                .push(SqlResultConversion::ZonedTextUtc);
        }
        Ok(plan)
    }

    /// A schema-only preflight for a driver's lossless result transport,
    /// where `source` asks for one: `plan`'s last statement described, not
    /// run. It binds empty key sets; it never executes the sampled data
    /// query. None where the source asks for no transport, or the request
    /// carries no timestamp to transport.
    pub fn result_metadata(
        &self,
        source: &SqlChoice,
        request: &DataRequest,
        plan: &SqlPlan,
    ) -> Result<Option<SqlPlan>, CoreError> {
        if source.timestamp_encoding != Some(SqlTimestampEncoding::Microseconds)
            || matches!(
                request,
                DataRequest::Describe { .. }
                    | DataRequest::KeyCounts { .. }
                    | DataRequest::CountRows { .. }
            )
        {
            return Ok(None);
        }
        if sql_dialect(source).is_none_or(|d| {
            d.timestamp_micros("value", SqlTimestampKind::WallTime)
                .is_none()
        }) {
            return Err(CoreError::new(
                Sentence::Undeclared,
                "the data source does not support the requested timestamp transport",
            )
            .with_field("request", Detail::text("timestamp_encoding")));
        }
        let mut statements = plan.statements.clone();
        let last = statements.last_mut().ok_or_else(|| {
            CoreError::new(Sentence::Internal, "a SQL result plan has no statement")
        })?;
        last.text = format!("DESCRIBE QUERY {}", last.text);
        Ok(Some(SqlPlan {
            statements,
            expect: ExpectedSchema::Discover {},
            result_conversions: Vec::new(),
        }))
    }

    /// `plan` projected using the warehouse's preflight result types,
    /// `metadata` ([`Sql::result_metadata`]): each timestamp column
    /// transported as microseconds. The host only executes the explicit
    /// output conversions.
    pub fn project_result(
        &self,
        source: &SqlChoice,
        request: &DataRequest,
        plan: &SqlPlan,
        metadata: &[RecordBatch],
    ) -> Result<SqlPlan, CoreError> {
        let mut conversions = Vec::new();
        let mut position = 0_u32;
        for batch in metadata {
            let names = catalog_read::text_at(batch, 0)?;
            let types = catalog_read::text_at(batch, 1)?;
            for row in 0..batch.num_rows() {
                let name = catalog_read::text(&names, row).ok_or_else(|| {
                    CoreError::new(Sentence::Internal, "a result metadata column has no name")
                })?;
                let kind = catalog_read::text(&types, row).ok_or_else(|| {
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
        self.render_projected(request, source, 0, &conversions)
    }

    /// An answer of `source`'s driver, read as the reference reads its
    /// source's values where the driver hands over its own (MID2 P2-3): a
    /// dialect's value rules, such as Postgres's text of a JSON value and
    /// its reading of a `numeric` past the digits an Arrow decimal holds
    /// (each `float_text`, the text the reference's language gives a
    /// double, where it writes one). `None`: the answer as it is. A value
    /// the rule cannot read is refused by name, never guessed.
    pub fn conform(
        &self,
        source: &SqlChoice,
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        float_text: fn(f64) -> String,
    ) -> Result<Option<(arrow_schema::SchemaRef, Vec<RecordBatch>)>, CoreError> {
        match sql_dialect(source) {
            Some(dialect) => dialect.conform(schema, batches, float_text),
            None => Ok(None),
        }
    }

    /// `plan` bound as `source` binds it, its key sets `keys` by the
    /// attachment the plan names each by: a key set written as the text
    /// its statement reads where the dialect binds one so, split past its
    /// budget, one fetch per chunk ([`key_text`]); a plan binding the
    /// core's Arrow key set as a relation, one fetch as it is. A double the
    /// binding writes is written as `float_text` writes it.
    pub fn bind(
        &self,
        source: &SqlChoice,
        request: &DataRequest,
        plan: SqlPlan,
        keys: &BTreeMap<AttachmentId, KeyData>,
        float_text: fn(f64) -> String,
    ) -> Result<BoundRequest<SqlPlan>, CoreError> {
        let conversions = plan.result_conversions.clone();
        match sql_dialect(source) {
            Some(dialect) => key_text::bind(dialect, request, plan, keys, float_text, &|offset| {
                self.render_projected(request, source, offset, &conversions)
            }),
            None => Ok(BoundRequest::whole(plan, request)),
        }
    }

    /// Whether `source` holds a type per column that a caller's type for
    /// the column must match: `false` for a source whose columns hold
    /// affinities only (SQLite's), where the old client reads any column as
    /// the type its caller gives.
    pub fn holds_column_types(&self, source: &SqlChoice) -> bool {
        sql_dialect(source).is_none_or(|dialect| dialect.holds_column_types())
    }

    /// Whether `source`'s tables are read as they stand at every run, as a
    /// database's or warehouse's are: their location where the source
    /// leaves a part to the session, their columns and their types.
    pub fn reads_tables_as_they_stand(&self, source: &SqlChoice) -> bool {
        sql_dialect(source).is_some()
    }

    /// The database `source` reads, by the name its dialect has (`duckdb`,
    /// `sqlite`, `postgres`), which a host words a refusal naming the source
    /// with.
    pub fn source_database(&self, source: &SqlChoice) -> Option<String> {
        Some(sql_dialect(source)?.name().to_owned())
    }

    /// Whether `source` leaves a namespace part to the connection that
    /// must be read before any table: one inside a part it gives (a catalog
    /// named and not its schema), or any part of a dialect that reads every
    /// table under its full namespace (Databricks, whose catalog names a
    /// referenced table in full); its namespace is then read first
    /// ([`DescribePart::Namespace`]) and settled by the parts it answers
    /// ([`Sql::settled`]).
    pub fn unsettled(&self, source: &SqlChoice) -> bool {
        given_parts(&source.namespace).is_none()
            || sql_dialect(source).is_some_and(|d| {
                d.settles_namespace()
                    && source.namespace.iter().flatten().count() < d.namespace_depth()
            })
    }

    /// `source` with its namespace the full namespace answer, settled by
    /// its dialect: an already resolved answer is retained; raw session
    /// parts are combined with supplied parts according to that dialect's
    /// namespace alignment.
    pub fn settled(&self, source: &SqlChoice, parts: Vec<String>) -> SqlChoice {
        let parts = match sql_dialect(source) {
            Some(dialect) => dialect.settled_namespace(&source.namespace, parts),
            None => parts,
        };
        SqlChoice {
            namespace: parts.into_iter().map(Some).collect(),
            ..source.clone()
        }
    }

    /// The kind of semantic model `source` reads, known before its
    /// definition is fetched.
    pub fn view_kind(&self, source: &SqlChoice) -> Option<sdm_contract::view::ViewKind> {
        sql_dialect(source)?.view_kind()
    }

    /// What `source`'s driver failure `error` says, as facts each step of
    /// the core selects according to the operation: a spelling retry reads
    /// `not_found`, a join's table read `unusable`, an expression sample
    /// `refuses_expression`. The dialect reads them from the driver's
    /// SQLSTATE, vendor code and error class, as the old client's separate
    /// predicates do.
    pub fn failure(&self, source: &SqlChoice, error: &ProviderError) -> FailureFacts {
        match sql_dialect(source) {
            Some(dialect) => {
                let mut facts = dialect.failure(error);
                facts.reason = failure::reason(error);
                facts.denied = failure::denied(error);
                facts
            }
            None => FailureFacts::default(),
        }
    }

    /// The answer to one question about a semantic model's text in
    /// `source`'s dialect: a name, a condition, an expression checked
    /// against its table's columns, a derived relation rendered. Refused by
    /// name where the dialect reads no semantic model.
    pub fn view_text(
        &self,
        source: &SqlChoice,
        ask: ViewTextAsk,
    ) -> Result<ViewTextAnswer, CoreError> {
        let (Some((dialect, spaces)), Some(sql)) = (self.views(source), sql_dialect(source)) else {
            return Err(views_unread());
        };
        Ok(view_expr::answer(ask, dialect, spaces, &|data_type| {
            sql.declared_type(data_type)
        }))
    }

    /// The name forms the catalog read of `relation` tries, in order, kept
    /// for that run once one resolves: Snowflake's as given, then
    /// upper-cased. How names compare is another question, the dialect's
    /// `name_case`.
    pub fn spellings(&self, source: &SqlChoice, relation: &Relation) -> Vec<Relation> {
        sql_dialect(source).map_or_else(|| vec![relation.clone()], |d| d.spellings(relation))
    }

    /// `relation` as `source` reads it: one of one part under the parts of
    /// `source`'s namespace given ([`given_parts`]).
    pub fn read_as(&self, source: &SqlChoice, relation: &Relation) -> Relation {
        if relation.source.len() != 1 {
            return relation.clone();
        }
        // A namespace no relation can write was refused before any
        // relation was read.
        let parts = given_parts(&source.namespace).unwrap_or_default();
        Relation::of(
            parts
                .into_iter()
                .chain(relation.source.iter().cloned())
                .collect(),
        )
    }

    /// `source`'s answer to `request`, a `describe`, read from its
    /// dialect's own catalog rows into the contract's dialect-neutral shape
    /// (decision E54). A request that is not a `describe` is refused with
    /// `INTERNAL_ERROR`.
    pub fn read(
        &self,
        request: &DataRequest,
        source: &SqlChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                Sentence::Internal,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let dialect = rendered(&source.dialect)?;
        let answer = if *part == DescribePart::ViewDefinition {
            let spaces = self.spaces.ok_or_else(views_unread)?;
            dialect.read_view(request, batches, spaces)?
        } else {
            dialect.read(request, *part, batches)?
        };
        debug_assert_eq!(answer.part(), *part);
        Ok(answer)
    }
}

/// The parts a relation of one part is read under, of the source's
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

impl Sql {
    /// [`Sql::plan`], a key set's seeds numbered from `offset`: a later
    /// chunk's statements, rendered as the first's are ([`key_text`]).
    fn render(
        &self,
        request: &DataRequest,
        source: &SqlChoice,
        offset: usize,
    ) -> Result<SqlPlan, CoreError> {
        self.render_projected(request, source, offset, &[])
    }

    fn render_projected(
        &self,
        request: &DataRequest,
        source: &SqlChoice,
        offset: usize,
        conversions: &[SqlResultConversion],
    ) -> Result<SqlPlan, CoreError> {
        let namespace = &source.namespace;
        let dialect = rendered(&source.dialect)?;
        let plan = |request: &DataRequest, namespace: &[String]| {
            if conversions.is_empty() {
                sql::plan(dialect, request, namespace, offset)
            } else {
                sql::plan_projected(dialect, request, namespace, offset, conversions)
            }
        };
        // The namespace itself, read where a part is left to the
        // connection inside one given, which every relation is then read
        // under.
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
            return Ok(SqlPlan {
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
        // of the parts the source leaves to the connection.
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
        ) && dialect.view_kind() == Some(sdm_contract::view::ViewKind::Semantic)
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
