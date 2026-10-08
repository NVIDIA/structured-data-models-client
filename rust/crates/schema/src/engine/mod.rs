// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine's reading of tables in a source: `rfm/base/table.py`, each
//! backend's `table.py`, the key count and the graph, with every statement
//! written by `sdm_dialects` and run through the host ([`Db`]).
//!
//! What Python reads lazily, once (`cached_property`), is read here at the
//! same point and kept, so the same statements are asked in the same
//! circumstances.

pub mod build;
pub mod gathering;
pub mod infer;
pub mod result;
pub mod state;
pub mod views;

use std::collections::{BTreeMap, BTreeSet};

use arrow_schema::DataType;
use sdm_dialects::{KeyCountPart, Statement};
use sdm_pql::types::{Dtype, Stype};
use sdm_protocol::{Asker, Outcome, Param};

use crate::dt::Host;
use crate::frame;
use crate::py::{self, str_repr};
use crate::pyval::PyVal;
use infer::{SampleColumn, SampleFrame, SampleKind};
use result::ResultTable;

/// The rows a table's types are inferred from.
pub const NUM_SAMPLE_ROWS: i64 = 1000;

/// Which source a table is in, as Phase 6 names its dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Sqlite,
    Duckdb,
    Postgres,
    Snow,
    Databricks,
}

impl Backend {
    pub fn parse(name: &str) -> Option<Backend> {
        Some(match name {
            "sqlite" => Backend::Sqlite,
            "duckdb" => Backend::Duckdb,
            "postgres" => Backend::Postgres,
            "snow" => Backend::Snow,
            "databricks" => Backend::Databricks,
            _ => return None,
        })
    }

    /// `Source.backend`, as the client names it.
    pub fn source_name(self) -> &'static str {
        match self {
            Backend::Sqlite => "sqlite",
            Backend::Duckdb => "duckdb",
            Backend::Postgres => "postgres",
            Backend::Snow => "snowflake",
            Backend::Databricks => "databricks",
        }
    }

    /// The class a driver's error has where the engine lets it through.
    fn driver_error(self) -> &'static str {
        match self {
            Backend::Sqlite => "OperationalError",
            Backend::Duckdb => "CatalogException",
            Backend::Postgres => "DatabaseError",
            Backend::Snow => "ProgrammingError",
            Backend::Databricks => "ServerOperationError",
        }
    }

    fn quote(self, name: &str) -> String {
        match self {
            Backend::Databricks => sdm_dialects::fragments::quote_ident(name, "`"),
            _ => sdm_dialects::fragments::quote_ident(name, "\""),
        }
    }
}

/// `quote_ident(name)`, with double quotes.
pub fn quote(name: &str) -> String {
    sdm_dialects::fragments::quote_ident(name, "\"")
}

/// A Python exception, by class and text, and what it was raised from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyExc {
    pub class: String,
    pub message: String,
    /// The errors it was raised from, outermost first, as
    /// `sdm_protocol::Diagnostic::chain` writes them.
    pub chain: Vec<serde_json::Value>,
    pub details: Box<serde_json::Value>,
}

/// The module-qualified name of an exception class today's code raises, by
/// the name the core knows it by.
pub fn qualified_class(class: &str) -> String {
    crate::diagnostics::public_class(class)
}

/// A link of a chain: an exception by its module-qualified class and its
/// text.
pub fn link(class: &str, message: impl Into<String>) -> serde_json::Value {
    crate::diagnostics::cause_link(class, message)
}

/// `RuntimeError("Table '...' is empty")`, as the engine raises it for a
/// table of no rows in its source.
pub fn empty_table(source_name: &str) -> PyExc {
    PyExc::new("RuntimeError", format!("Table '{source_name}' is empty"))
}

/// The source name an [`empty_table`] error's message names.
pub fn emptied(class: &str, message: &str) -> Option<String> {
    if class != "RuntimeError" {
        return None;
    }
    let name = message
        .strip_prefix("Table '")?
        .strip_suffix("' is empty")?;
    Some(name.to_string())
}

/// The link that stands for the error the host met running the last
/// statement.
pub fn from_query() -> serde_json::Value {
    serde_json::json!({"from": "query"})
}

impl PyExc {
    pub fn new(class: &str, message: impl Into<String>) -> Self {
        PyExc {
            class: class.to_string(),
            message: message.into(),
            chain: Vec::new(),
            details: Box::new(serde_json::json!({})),
        }
    }

    /// The same exception, raised from `links`, outermost first.
    #[must_use]
    pub fn caused_by(mut self, links: impl IntoIterator<Item = serde_json::Value>) -> Self {
        self.chain.extend(links);
        self
    }

    /// This exception as links of another's chain: itself, then what it was
    /// raised from.
    pub fn as_links(&self) -> Vec<serde_json::Value> {
        let mut out = vec![link(&self.class, self.message.clone())];
        if self
            .details
            .as_object()
            .is_some_and(|details| !details.is_empty())
        {
            out[0]["details"] = (*self.details).clone();
        }
        out.extend(self.chain.iter().cloned());
        out
    }

    /// `str(KeyError(message))`, which is the message's `repr`.
    pub fn key_error(message: &str) -> Self {
        PyExc::new("KeyError", str_repr(message))
    }

    /// Whether `except ValueError` catches it.
    pub fn is_value_error(&self) -> bool {
        matches!(
            self.class.as_str(),
            "ValueError"
                | "TableNotFoundError"
                | "MetricViewAsTableError"
                | "UnsupportedJoinError"
                | "UnsupportedFilterError"
                | "NoTablesFoundError"
        )
    }
}

/// What the engine's code gives: a value, or the exception it raised.
pub type ER<T> = Result<T, PyExc>;

fn raise<T>(class: &str, message: impl Into<String>) -> ER<T> {
    Err(PyExc::new(class, message))
}

/// A statement that failed in the database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqlError {
    pub sqlstate: Option<String>,
    pub vendor_code: Option<i64>,
    pub message: String,
}

/// What runs the engine's statements: the host, through the run loop.
pub struct Db(pub Asker);

impl Db {
    /// The statement's result, or the database's error.
    pub async fn run(&self, statement: &Statement) -> Result<ResultTable, SqlError> {
        let (sql, params) = statement.clone();
        match self
            .0
            .ask(sql, params.unwrap_or_default(), Vec::new())
            .await
        {
            Outcome::Data(result) => Ok(ResultTable::from_batches(
                &result.schema(),
                result.batches(),
            )),
            Outcome::Error {
                sqlstate,
                vendor_code,
                message,
            } => Err(SqlError {
                sqlstate,
                vendor_code,
                message,
            }),
            Outcome::Response(_) | Outcome::ModelError(_) => Err(SqlError {
                sqlstate: None,
                vendor_code: None,
                message: "a model's answer was submitted for a query".to_string(),
            }),
        }
    }
}

/// `table._run(statement)`: the result, or the driver's error raised.
pub async fn must(db: &Db, backend: Backend, statement: &Statement) -> ER<ResultTable> {
    match db.run(statement).await {
        Ok(table) => Ok(table),
        Err(e) => raise(backend.driver_error(), e.message),
    }
}

/// `table._rows(statement)`.
pub async fn rows(db: &Db, backend: Backend, statement: &Statement) -> ER<Vec<Vec<PyVal>>> {
    let table = must(db, backend, statement).await?;
    Ok(if backend == Backend::Postgres {
        table.pg_rows()
    } else {
        table.rows()
    })
}

/// What warnings the engine gives, by class and text.
#[derive(Default, Clone, Debug)]
pub struct Warnings(pub Vec<(String, String)>);

impl Warnings {
    pub fn user(&mut self, message: impl Into<String>) {
        self.0
            .push(("builtins.UserWarning".to_string(), message.into()));
    }
}

/// What the engine reads with: the host and its warnings.
pub struct Ctx<'a> {
    pub db: &'a Db,
    pub host: &'a Host,
    pub warnings: Warnings,
}

/// `SourceColumn`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceColumn {
    pub name: String,
    pub dtype: Option<Dtype>,
    pub is_primary_key: bool,
    pub is_unique_key: bool,
    pub is_nullable: bool,
}

/// `SourceForeignKey`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceFk {
    pub name: String,
    pub dst_table: String,
    pub primary_key: String,
}

/// `ColumnSpec`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: String,
    pub expr: Option<String>,
    pub dtype: Option<Dtype>,
    pub stype: Option<Stype>,
}

impl ColumnSpec {
    pub fn plain(name: &str) -> Self {
        ColumnSpec {
            name: name.to_string(),
            expr: None,
            dtype: None,
            stype: None,
        }
    }
}

/// An engine `Column`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EColumn {
    pub name: String,
    pub expr: Option<String>,
    pub dtype: Dtype,
    pub stype: Stype,
    pub is_pk: bool,
    pub is_time: bool,
    pub is_end: bool,
}

impl EColumn {
    pub fn is_source(&self) -> bool {
        self.expr.is_none()
    }

    /// `column.stype = value`, checked as `Column.__setattr__` checks it.
    pub fn set_stype(&mut self, value: Stype) -> ER<()> {
        if !value.supports_dtype(self.dtype) {
            return raise(
                "ValueError",
                format!(
                    "Column '{}' received an incompatible semantic type (got dtype='{}' and stype='{}')",
                    self.name,
                    self.dtype.value(),
                    value.value()
                ),
            );
        }
        if self.is_pk && value != Stype::Id {
            return raise(
                "ValueError",
                format!(
                    "Primary key '{}' must have 'ID' semantic type (got '{}')",
                    self.name,
                    value.value()
                ),
            );
        }
        if self.is_time && value != Stype::Timestamp {
            return raise(
                "ValueError",
                format!(
                    "Time column '{}' must have 'timestamp' semantic type (got '{}')",
                    self.name,
                    value.value()
                ),
            );
        }
        if self.is_end && value != Stype::Timestamp {
            return raise(
                "ValueError",
                format!(
                    "End time column '{}' must have 'timestamp' semantic type (got '{}')",
                    self.name,
                    value.value()
                ),
            );
        }
        self.stype = value;
        Ok(())
    }
}

/// What a table is constructed with.
#[derive(Clone, Debug)]
pub struct TableSpec {
    pub name: String,
    pub source_name: Option<String>,
    pub place: Place,
    pub columns: Option<Vec<ColumnSpec>>,
    pub primary_key: PkArg,
    pub time_column: Option<String>,
}

impl TableSpec {
    /// A table under `name`, read whole, keyed as its source declares.
    pub fn named(name: &str, source_name: Option<&str>, place: Place) -> Self {
        TableSpec {
            name: name.to_string(),
            source_name: source_name.map(str::to_string),
            place,
            columns: None,
            primary_key: PkArg::Missing,
            time_column: None,
        }
    }
}

/// A row of Snowflake's `SHOW TABLES`, by field.
type Listing = Vec<(String, PyVal)>;

/// A constraint's type and name, and its columns by position.
type Constraint = ((String, String), Vec<(i64, String)>);

/// The primary key a table is constructed with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PkArg {
    /// Not given: taken from the source's catalog.
    Missing,
    None,
    Name(String),
}

/// Where a table lives, beside its name.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Place {
    /// Databricks' catalog or Snowflake's database.
    pub catalog: Option<String>,
    pub schema: Option<String>,
    /// A Databricks metric view's filter.
    pub row_filter: Option<String>,
}

/// An engine table over a source.
#[derive(Clone, Debug)]
pub struct EngineTable {
    pub backend: Backend,
    pub name: String,
    /// `_source_name`: the table's own name, unqualified.
    pub raw_source_name: String,
    pub place: Place,
    pub columns: Vec<EColumn>,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
    pub end_time_column: Option<String>,
    pub time_choice: Option<crate::time_columns::Choice>,
    pub time_inferred: bool,
    expr_sample: SampleFrame,
    source_columns: Option<Vec<SourceColumn>>,
    source_fks: Option<Vec<SourceFk>>,
    source_sample: Option<SampleFrame>,
    unique_constraints: Vec<BTreeSet<String>>,
    /// Each source column's type as its catalog spells it.
    spelled: BTreeMap<String, String>,
    sample_source: Option<String>,
    /// Snowflake's `SHOW TABLES LIKE` rows; `Some(None)` where it failed.
    listed: Option<Option<Vec<Listing>>>,
    pub is_hybrid: Option<bool>,
    /// The columns a composite key folds, where the key is one.
    pub composite_key: Option<Vec<String>>,
}

fn int_of(value: &PyVal) -> Option<i64> {
    match value {
        PyVal::Int(n) => i64::try_from(*n).ok(),
        PyVal::Bool(b) => Some(i64::from(*b)),
        PyVal::Float(x) => Some(*x as i64),
        PyVal::Decimal(t) => t.split('.').next().and_then(|s| s.parse().ok()),
        PyVal::Str(s) => s.trim().parse().ok(),
        PyVal::Numpy(_, inner) => int_of(inner),
        _ => None,
    }
}

fn truthy(value: &PyVal) -> bool {
    crate::metric_view::truthy(value)
}

fn text_of(value: &PyVal) -> String {
    match value {
        PyVal::Str(s) => s.clone(),
        other => other.str_(),
    }
}

impl EngineTable {
    /// A table, read as its backend's constructor reads it.
    pub async fn new(ctx: &mut Ctx<'_>, backend: Backend, spec: TableSpec) -> ER<EngineTable> {
        let TableSpec {
            name,
            source_name,
            mut place,
            columns,
            primary_key,
            time_column,
        } = spec;
        let name = name.as_str();
        if matches!(backend, Backend::Snow | Backend::Databricks)
            && (place.catalog.is_none() || place.schema.is_none())
        {
            let statement = if backend == Backend::Snow {
                sdm_dialects::snow::current_namespace()
            } else {
                sdm_dialects::databricks::current_namespace()
            };
            let found = rows(ctx.db, backend, &statement).await?;
            let Some(row) = found.first() else {
                return raise("IndexError", "list index out of range");
            };
            let given = |v: &Option<String>| v.clone().filter(|s| !s.is_empty());
            let cell = |i: usize| match row.get(i) {
                None | Some(PyVal::None) => None,
                Some(v) => Some(text_of(v)),
            };
            place.catalog = given(&place.catalog).or_else(|| cell(0));
            if place.catalog.is_none() {
                return raise("AssertionError", "");
            }
            place.schema = given(&place.schema).or_else(|| cell(1));
        }
        if matches!(backend, Backend::Snow | Backend::Databricks) && place.schema.is_none() {
            let shown = source_name
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| name.to_string());
            let within = if backend == Backend::Snow {
                "database"
            } else {
                "catalog"
            };
            return raise(
                "ValueError",
                format!(
                    "Unspecified 'schema' for table '{shown}' in {within} '{}'",
                    place.catalog.clone().unwrap_or_default()
                ),
            );
        }
        let source_name = source_name.filter(|s| !s.is_empty());
        let source_name = source_name.as_deref();
        let time_column = time_column.as_deref();
        let mut table = EngineTable {
            backend,
            name: name.to_string(),
            raw_source_name: source_name.unwrap_or(name).to_string(),
            place,
            columns: Vec::new(),
            primary_key: None,
            time_column: None,
            end_time_column: None,
            time_choice: None,
            time_inferred: false,
            expr_sample: SampleFrame::empty_index(NUM_SAMPLE_ROWS as usize),
            source_columns: None,
            source_fks: None,
            source_sample: None,
            unique_constraints: Vec::new(),
            spelled: BTreeMap::new(),
            sample_source: None,
            listed: None,
            is_hybrid: None,
            composite_key: None,
        };
        let specs = match columns {
            Some(specs) => specs,
            None => table
                .source_column_dict(ctx)
                .await?
                .iter()
                .map(|c| ColumnSpec::plain(&c.name))
                .collect(),
        };
        table.add_columns(ctx, specs).await?;
        match primary_key {
            PkArg::Missing => {
                if table.columns.iter().any(EColumn::is_source)
                    && let Some(pk) = table.source_primary_key(ctx).await?
                    && table.column(&pk).is_some_and(EColumn::is_source)
                {
                    table.set_primary_key(Some(&pk))?;
                }
            }
            PkArg::None => {}
            PkArg::Name(pk) => {
                if table.column(&pk).is_none() {
                    table.add_columns(ctx, vec![ColumnSpec::plain(&pk)]).await?;
                }
                table.set_primary_key(Some(&pk))?;
            }
        }
        if let Some(time) = time_column {
            if table.column(time).is_none() {
                table
                    .add_columns(ctx, vec![ColumnSpec::plain(time)])
                    .await?;
            }
            table.set_time_column(Some(time))?;
        }
        Ok(table)
    }

    /// `source_name`, qualified as each backend qualifies it.
    pub fn source_name(&self) -> String {
        match self.backend {
            Backend::Postgres => format!(
                "{}.{}",
                self.place.schema.clone().unwrap_or_default(),
                self.raw_source_name
            ),
            Backend::Snow | Backend::Databricks => format!(
                "{}.{}.{}",
                self.place.catalog.clone().unwrap_or_default(),
                self.place.schema.clone().unwrap_or_default(),
                self.raw_source_name
            ),
            _ => self.raw_source_name.clone(),
        }
    }

    /// `_quoted_source_name`.
    pub fn quoted_source_name(&self) -> String {
        match self.backend {
            Backend::Postgres => format!(
                "{}.{}",
                quote(self.place.schema.as_deref().unwrap_or_default()),
                quote(&self.raw_source_name)
            ),
            Backend::Snow => format!(
                "{}.{}.{}",
                quote(self.place.catalog.as_deref().unwrap_or_default()),
                quote(self.place.schema.as_deref().unwrap_or_default()),
                quote(&self.raw_source_name)
            ),
            Backend::Databricks => {
                let name = self.databricks_name();
                let scoped = self.source_columns.as_ref().filter(|columns| {
                    columns.iter().any(|column| {
                        sdm_dialects::view_expr::SESSION_VALUES
                            .contains(&column.name.to_lowercase().as_str())
                    })
                });
                let projection = scoped.map_or_else(
                    || "*".to_string(),
                    |columns| {
                        columns
                            .iter()
                            .map(|column| format!("source.{}", self.backend.quote(&column.name)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                );
                if scoped.is_none() && self.place.row_filter.is_none() {
                    return name;
                }
                let condition = self
                    .place
                    .row_filter
                    .as_ref()
                    .map_or_else(String::new, |filter| format!(" WHERE ({filter})"));
                format!("(SELECT {projection} FROM {name} AS source{condition})")
            }
            _ => quote(&self.raw_source_name),
        }
    }

    fn databricks_name(&self) -> String {
        [
            self.place.catalog.as_deref().unwrap_or_default(),
            self.place.schema.as_deref().unwrap_or_default(),
            &self.raw_source_name,
        ]
        .iter()
        .map(|p| Backend::Databricks.quote(p))
        .collect::<Vec<_>>()
        .join(".")
    }

    pub fn set_row_filter(&mut self, filter: Option<String>) -> ER<()> {
        if self.source_sample.is_some() || !self.expr_sample.columns.is_empty() {
            return raise(
                "ValueError",
                format!(
                    "Cannot change the row filter of table '{}' since its rows have already been read",
                    self.name
                ),
            );
        }
        self.place.row_filter = filter;
        Ok(())
    }

    /// `_quote_key_column`.
    pub fn quote_key_column(&self, name: &str) -> String {
        self.backend.quote(name)
    }

    pub fn column(&self, name: &str) -> Option<&EColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    fn column_mut(&mut self, name: &str) -> Option<&mut EColumn> {
        self.columns.iter_mut().find(|c| c.name == name)
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.column(name).is_some()
    }

    fn not_found(&self, name: &str) -> PyExc {
        PyExc::key_error(&format!(
            "Column '{name}' not found in table '{}'",
            self.name
        ))
    }

    /// `primary_key_columns`: the columns a composite key folds, or the key.
    pub fn primary_key_columns(&self) -> Vec<String> {
        match &self.composite_key {
            Some(names) => names.clone(),
            None => self.primary_key.iter().cloned().collect(),
        }
    }

    // --- The source's catalog -------------------------------------------------------

    /// `_source_column_dict`.
    pub async fn source_column_dict(&mut self, ctx: &mut Ctx<'_>) -> ER<&[SourceColumn]> {
        if self.source_columns.is_none() {
            let columns = self.get_source_columns(ctx).await?;
            if columns.is_empty() {
                return raise(
                    "ValueError",
                    format!("Table '{}' has no columns", self.name),
                );
            }
            self.source_columns = Some(columns);
        }
        Ok(self.source_columns.as_deref().unwrap_or(&[]))
    }

    async fn source_column(&mut self, ctx: &mut Ctx<'_>, name: &str) -> ER<Option<SourceColumn>> {
        Ok(self
            .source_column_dict(ctx)
            .await?
            .iter()
            .find(|c| c.name == name)
            .cloned())
    }

    /// `_source_primary_key`.
    pub async fn source_primary_key(&mut self, ctx: &mut Ctx<'_>) -> ER<Option<String>> {
        let keys: Vec<String> = self
            .source_column_dict(ctx)
            .await?
            .iter()
            .filter(|c| c.is_primary_key)
            .map(|c| c.name.clone())
            .collect();
        Ok(if keys.len() == 1 {
            Some(keys[0].clone())
        } else {
            None
        })
    }

    /// `source_foreign_keys`.
    pub async fn source_foreign_keys(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceFk>> {
        if self.source_fks.is_none() {
            let fks = self.get_source_foreign_keys(ctx).await?;
            // Keyed by name: a later key of the same name replaces the first.
            let mut by_name: Vec<SourceFk> = Vec::new();
            for fk in fks {
                if let Some(slot) = by_name.iter_mut().find(|k| k.name == fk.name) {
                    *slot = fk;
                } else {
                    by_name.push(fk);
                }
            }
            self.source_fks = Some(by_name);
        }
        Ok(self.source_fks.clone().unwrap_or_default())
    }

    async fn get_source_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        match self.backend {
            Backend::Sqlite => self.sqlite_columns(ctx).await,
            Backend::Duckdb => self.duckdb_columns(ctx).await,
            Backend::Postgres => self.postgres_columns(ctx).await,
            Backend::Snow => self.snow_columns(ctx).await,
            Backend::Databricks => self.databricks_columns(ctx).await,
        }
    }

    async fn sqlite_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        let quoted = self.quoted_source_name();
        let columns = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::sqlite::table_info(&quoted),
        )
        .await?;
        if columns.is_empty() {
            return raise(
                "TableNotFoundError",
                format!(
                    "Table '{}' does not exist in the SQLite database",
                    self.source_name()
                ),
            );
        }
        let mut unique_keys: BTreeSet<String> = BTreeSet::new();
        let mut constraints: Vec<BTreeSet<String>> = Vec::new();
        let indices = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::sqlite::index_list(&quoted),
        )
        .await?;
        for index in &indices {
            let index_name = index.get(1).map(text_of).unwrap_or_default();
            let is_unique = index.get(2).is_some_and(truthy);
            let partial = index.len() > 4 && truthy(&index[4]);
            if is_unique {
                let info = rows(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::sqlite::index_info(&index_name),
                )
                .await?;
                if info.len() == 1 {
                    unique_keys.insert(info[0].get(2).map(text_of).unwrap_or_default());
                }
                let names: Vec<Option<String>> = info
                    .iter()
                    .map(|r| match r.get(2) {
                        Some(PyVal::None) | None => None,
                        Some(v) => Some(text_of(v)),
                    })
                    .collect();
                if !partial && names.iter().all(Option::is_some) {
                    constraints.push(names.into_iter().flatten().collect());
                }
            }
        }
        let rowid_candidates: Vec<String> = columns
            .iter()
            .filter(|c| {
                c.get(5).is_some_and(truthy) && text_of(&c[2]).trim().to_uppercase() == "INTEGER"
            })
            .map(|c| text_of(&c[1]))
            .collect();
        let rowid = if rowid_candidates.len() == 1 {
            Some(rowid_candidates[0].clone())
        } else {
            None
        };
        if let Some(r) = &rowid {
            unique_keys.insert(r.clone());
            constraints.push(BTreeSet::from([r.clone()]));
        }
        self.unique_constraints = constraints;
        Ok(columns
            .iter()
            .map(|c| {
                let name = text_of(&c[1]);
                SourceColumn {
                    dtype: sqlite_dtype(&c[2]),
                    is_primary_key: truthy(&c[5]),
                    is_unique_key: unique_keys.contains(&name),
                    is_nullable: Some(&name) != rowid.as_ref() && !truthy(&c[3]),
                    name,
                }
            })
            .collect())
    }

    async fn duckdb_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        let quoted = self.quoted_source_name();
        let not_found = || {
            PyExc::new(
                "TableNotFoundError",
                format!(
                    "Table '{}' does not exist in the DuckDB database",
                    self.source_name()
                ),
            )
        };
        let columns = match ctx.db.run(&sdm_dialects::duckdb::table_info(&quoted)).await {
            Ok(t) => t.rows(),
            Err(_) => return Err(not_found().caused_by([from_query()])),
        };
        if columns.is_empty() {
            return Err(not_found());
        }
        let constraints = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::duckdb::key_constraints(&self.source_name()),
        )
        .await?;
        let mut unique_keys = BTreeSet::new();
        let mut primary_keys = BTreeSet::new();
        self.unique_constraints = constraints
            .iter()
            .filter(|r| matches!(text_of(&r[0]).as_str(), "PRIMARY KEY" | "UNIQUE"))
            .map(|r| list_texts(&r[1]).into_iter().collect())
            .collect();
        for r in &constraints {
            let names = list_texts(&r[1]);
            if names.len() != 1 {
                continue;
            }
            match text_of(&r[0]).as_str() {
                "PRIMARY KEY" => {
                    primary_keys.insert(names[0].clone());
                    unique_keys.insert(names[0].clone());
                }
                "UNIQUE" => {
                    unique_keys.insert(names[0].clone());
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        for c in &columns {
            let name = text_of(&c[1]);
            let spelled = text_of(&c[2]);
            self.spelled
                .insert(name.clone(), spelled.trim().to_uppercase());
            let is_pk = truthy(&c[5]) || primary_keys.contains(&name);
            out.push(SourceColumn {
                dtype: duckdb_dtype(&spelled),
                is_primary_key: is_pk,
                is_unique_key: unique_keys.contains(&name),
                is_nullable: !is_pk && !truthy(&c[3]),
                name,
            });
        }
        Ok(out)
    }

    async fn postgres_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        let schema = self.place.schema.clone().unwrap_or_default();
        let found = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::postgres::columns(&schema, &self.raw_source_name),
        )
        .await?;
        if found.is_empty() {
            return raise(
                "TableNotFoundError",
                format!(
                    "Table '{}' does not exist in the PostgreSQL database",
                    self.source_name()
                ),
            );
        }
        self.place.schema = Some(text_of(&found[0][5]));
        self.raw_source_name = text_of(&found[0][6]);
        let schema = self.place.schema.clone().unwrap_or_default();
        let constraints = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::postgres::key_constraints(&schema, &self.raw_source_name),
        )
        .await?;
        let mut grouped: Vec<Constraint> = Vec::new();
        for r in &constraints {
            let key = (text_of(&r[0]), text_of(&r[1]));
            let entry = (int_of(&r[3]).unwrap_or(0), text_of(&r[2]));
            match grouped.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(entry),
                None => grouped.push((key, vec![entry])),
            }
        }
        for ((ctype, name), positioned) in &grouped {
            if ctype != "PRIMARY KEY" || positioned.len() == 1 {
                continue;
            }
            let mut sorted = positioned.clone();
            sorted.sort();
            let columns: Vec<String> = sorted.into_iter().map(|(_, c)| c).collect();
            return raise(
                "ValueError",
                format!(
                    "Composite primary key constraint '{name}' on PostgreSQL table '{}' uses columns {}. Automatic PostgreSQL graph discovery currently supports only single-column primary keys.",
                    self.source_name(),
                    py::str_list_repr(&columns)
                ),
            );
        }
        self.unique_constraints = grouped
            .iter()
            .filter(|((t, _), _)| t == "PRIMARY KEY" || t == "UNIQUE")
            .map(|(_, cols)| cols.iter().map(|(_, c)| c.clone()).collect())
            .collect();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for r in &constraints {
            *counts.entry(text_of(&r[1])).or_default() += 1;
        }
        let mut primary = BTreeSet::new();
        let mut unique = BTreeSet::new();
        let mut relational = BTreeSet::new();
        for r in &constraints {
            if counts[&text_of(&r[1])] != 1 {
                continue;
            }
            let (ctype, column) = (text_of(&r[0]), text_of(&r[2]));
            if ctype == "PRIMARY KEY" {
                primary.insert(column.clone());
            }
            if ctype == "PRIMARY KEY" || ctype == "UNIQUE" {
                unique.insert(column.clone());
            }
            relational.insert(column);
        }
        self.spelled = found
            .iter()
            .map(|r| (text_of(&r[0]), text_of(r.last().unwrap_or(&PyVal::None))))
            .collect();
        Ok(found
            .iter()
            .map(|r| {
                let name = text_of(&r[0]);
                let data_type = text_of(&r[1]);
                let dtype = if relational.contains(&name)
                    && matches!(
                        data_type.trim().to_lowercase().as_str(),
                        "numeric" | "decimal"
                    ) {
                    Some(Dtype::String)
                } else {
                    postgres_dtype(&data_type, &r[2], &r[3])
                };
                SourceColumn {
                    dtype,
                    is_primary_key: primary.contains(&name),
                    is_unique_key: unique.contains(&name),
                    is_nullable: text_of(&r[4]) != "NO",
                    name,
                }
            })
            .collect())
    }

    async fn snow_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        let names = (
            self.place.catalog.clone().unwrap_or_default(),
            self.place.schema.clone().unwrap_or_default(),
            self.raw_source_name.clone(),
        );
        let folded = (
            names.0.to_uppercase(),
            names.1.to_uppercase(),
            names.2.to_uppercase(),
        );
        let candidates = if names == folded {
            vec![names]
        } else {
            vec![names, folded]
        };
        let mut found: Option<Vec<Vec<PyVal>>> = None;
        for candidate in candidates {
            let quoted = format!(
                "{}.{}.{}",
                quote(&candidate.0),
                quote(&candidate.1),
                quote(&candidate.2)
            );
            match ctx
                .db
                .run(&sdm_dialects::snow::describe_table(&quoted))
                .await
            {
                Ok(table) => {
                    self.place.catalog = Some(candidate.0);
                    self.place.schema = Some(candidate.1);
                    self.raw_source_name = candidate.2;
                    found = Some(table.rows());
                    break;
                }
                Err(e) => {
                    if !(e.vendor_code == Some(2003)
                        || e.message.to_lowercase().contains("does not exist"))
                    {
                        return raise(self.backend.driver_error(), e.message);
                    }
                }
            }
        }
        let Some(rows) = found else {
            return raise(
                "TableNotFoundError",
                format!(
                    "Table '{}' does not exist in the remote data backend",
                    self.source_name()
                ),
            );
        };
        let mut out = Vec::new();
        for r in &rows {
            let name = text_of(&r[0]);
            let dtype = text_of(&r[1]);
            self.spelled
                .insert(name.clone(), dtype.trim().to_uppercase());
            out.push(SourceColumn {
                dtype: snow_dtype(Some(&dtype)),
                is_primary_key: text_of(&r[5]).trim().to_uppercase() == "Y",
                is_unique_key: text_of(&r[6]).trim().to_uppercase() == "Y",
                is_nullable: text_of(&r[3]).trim().to_uppercase() == "Y",
                name,
            });
        }
        Ok(out)
    }

    async fn databricks_columns(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceColumn>> {
        let catalog = Backend::Databricks.quote(self.place.catalog.as_deref().unwrap_or_default());
        let schema = self.place.schema.clone().unwrap_or_default();
        let found = rows(
            ctx.db,
            self.backend,
            &sdm_dialects::databricks::columns(&catalog, &schema, &self.raw_source_name),
        )
        .await?;
        if found.is_empty() {
            return raise(
                "TableNotFoundError",
                format!(
                    "Table '{}' does not exist in the remote data backend",
                    self.source_name()
                ),
            );
        }
        if found[0]
            .get(6)
            .is_some_and(|value| text_of(value) == "METRIC_VIEW")
        {
            let mut error = PyExc::new(
                "MetricViewAsTableError",
                format!(
                    "'{}' is a metric view, not a table. Its measures are aggregations with no value per row, so it cannot be read as a table. Load it with Graph.from_databricks_metric_view('{}', connection, catalog='{}', schema='{}') instead.",
                    self.source_name(),
                    self.raw_source_name,
                    self.place.catalog.as_deref().unwrap_or_default(),
                    self.place.schema.as_deref().unwrap_or_default()
                ),
            );
            error.details = Box::new(serde_json::json!({"name": self.raw_source_name}));
            return Err(error);
        }
        self.place.catalog = Some(text_of(&found[0][3]));
        self.place.schema = Some(text_of(&found[0][4]));
        self.raw_source_name = text_of(&found[0][5]);
        let catalog = Backend::Databricks.quote(self.place.catalog.as_deref().unwrap_or_default());
        let schema = self.place.schema.clone().unwrap_or_default();
        let mut primary = BTreeSet::new();
        let mut unique = BTreeSet::new();
        if let Ok(constraints) = ctx
            .db
            .run(&sdm_dialects::databricks::key_constraints(
                &catalog,
                &schema,
                &self.raw_source_name,
            ))
            .await
        {
            let constraints = constraints.rows();
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for r in &constraints {
                *counts.entry(text_of(&r[1])).or_default() += 1;
            }
            for r in &constraints {
                if counts[&text_of(&r[1])] != 1 {
                    continue;
                }
                if text_of(&r[0]).trim().to_uppercase() == "PRIMARY KEY" {
                    primary.insert(text_of(&r[2]));
                }
                unique.insert(text_of(&r[2]));
            }
        }
        Ok(found
            .iter()
            .map(|r| {
                let name = text_of(&r[0]);
                SourceColumn {
                    dtype: databricks_dtype(&text_of(&r[1])),
                    is_primary_key: primary.contains(&name),
                    is_unique_key: unique.contains(&name),
                    is_nullable: text_of(&r[2]).trim().to_uppercase() != "NO",
                    name,
                }
            })
            .collect())
    }

    async fn get_source_foreign_keys(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<SourceFk>> {
        let quoted = self.quoted_source_name();
        match self.backend {
            Backend::Sqlite => {
                let found = rows(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::sqlite::foreign_key_list(&quoted),
                )
                .await?;
                let counts = count_by(&found, 0);
                Ok(found
                    .iter()
                    .filter(|r| counts[&r[0].eq_key()] == 1)
                    .map(|r| SourceFk {
                        name: text_of(&r[3]),
                        dst_table: text_of(&r[2]),
                        primary_key: text_of(&r[4]),
                    })
                    .collect())
            }
            Backend::Duckdb => {
                let found = rows(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::duckdb::foreign_keys(&self.source_name()),
                )
                .await?;
                let counts = count_by(&found, 0);
                Ok(found
                    .iter()
                    .filter(|r| counts[&r[0].eq_key()] == 1 && list_texts(&r[2]).len() == 1)
                    .map(|r| SourceFk {
                        name: list_texts(&r[2])[0].clone(),
                        dst_table: text_of(&r[1]),
                        primary_key: list_texts(&r[3]).first().cloned().unwrap_or_default(),
                    })
                    .collect())
            }
            Backend::Postgres => {
                let schema = self.place.schema.clone().unwrap_or_default();
                let found = rows(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::postgres::foreign_keys(&schema, &self.raw_source_name),
                )
                .await?;
                let counts = count_by(&found, 0);
                for r in &found {
                    if counts[&r[0].eq_key()] == 1 {
                        continue;
                    }
                    let name = text_of(&r[0]);
                    let mut constraint: Vec<&Vec<PyVal>> =
                        found.iter().filter(|x| text_of(&x[0]) == name).collect();
                    constraint.sort_by_key(|x| int_of(&x[5]).unwrap_or(0));
                    let columns: Vec<String> = constraint.iter().map(|x| text_of(&x[1])).collect();
                    let referenced: Vec<String> =
                        constraint.iter().map(|x| text_of(&x[4])).collect();
                    return raise(
                        "ValueError",
                        format!(
                            "Composite foreign key constraint '{name}' on PostgreSQL table '{}' uses columns {} referencing {}. Automatic PostgreSQL graph discovery currently supports only single-column foreign keys.",
                            self.source_name(),
                            py::str_list_repr(&columns),
                            py::str_list_repr(&referenced)
                        ),
                    );
                }
                Ok(found
                    .iter()
                    .map(|r| SourceFk {
                        name: text_of(&r[1]),
                        dst_table: format!("{}.{}", text_of(&r[2]), text_of(&r[3])),
                        primary_key: text_of(&r[4]),
                    })
                    .collect())
            }
            Backend::Snow => {
                let mut found = rows(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::snow::imported_keys(&quoted),
                )
                .await?;
                found.sort_by_key(|r| r.iter().map(PyVal::str_).collect::<Vec<_>>());
                let counts = count_by(&found, 13);
                Ok(found
                    .iter()
                    .filter(|r| r.len() > 13 && counts[&r[13].eq_key()] == 1)
                    .map(|r| SourceFk {
                        name: text_of(&r[8]),
                        dst_table: format!(
                            "{}.{}.{}",
                            text_of(&r[1]),
                            text_of(&r[2]),
                            text_of(&r[3])
                        ),
                        primary_key: text_of(&r[4]),
                    })
                    .collect())
            }
            Backend::Databricks => {
                let catalog =
                    Backend::Databricks.quote(self.place.catalog.as_deref().unwrap_or_default());
                let schema = self.place.schema.clone().unwrap_or_default();
                let found = match ctx
                    .db
                    .run(&sdm_dialects::databricks::foreign_keys(
                        &catalog,
                        &schema,
                        &self.raw_source_name,
                    ))
                    .await
                {
                    Ok(t) => t.rows(),
                    Err(_) => return Ok(Vec::new()),
                };
                let counts = count_by(&found, 0);
                Ok(found
                    .iter()
                    .filter(|r| counts[&r[0].eq_key()] == 1)
                    .map(|r| SourceFk {
                        name: text_of(&r[1]),
                        dst_table: format!(
                            "{}.{}.{}",
                            text_of(&r[2]),
                            text_of(&r[3]),
                            text_of(&r[4])
                        ),
                        primary_key: text_of(&r[5]),
                    })
                    .collect())
            }
        }
    }

    // --- Samples ---------------------------------------------------------------------

    async fn order_terms(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<String>> {
        let pk = self.source_primary_key(ctx).await?;
        let columns: Vec<sdm_dialects::fragments::SourceColumn> = self
            .source_column_dict(ctx)
            .await?
            .iter()
            .map(|c| sdm_dialects::fragments::SourceColumn {
                name: c.name.clone(),
                dtype: c
                    .dtype
                    .and_then(|d| sdm_dialects::py::Dtype::parse(d.value()).ok()),
            })
            .collect();
        Ok(match self.backend {
            Backend::Postgres => {
                if let Some(key) = pk {
                    return Ok(vec![quote(&key)]);
                }
                let mut known = Vec::new();
                let mut unknown = Vec::new();
                for c in self.source_column_dict(ctx).await? {
                    let r = quote(&c.name);
                    if c.dtype.is_none() || c.dtype == Some(Dtype::Unsupported) {
                        unknown.push(r);
                    } else {
                        known.push(r);
                    }
                }
                let mut terms = Vec::new();
                if !known.is_empty() {
                    let binary = format!("record_send(ROW({}))", known.join(", "));
                    terms.push(format!("md5({binary}) COLLATE \"C\""));
                    terms.push(binary);
                }
                if !unknown.is_empty() {
                    let text = format!("CAST(ROW({}) AS text)", unknown.join(", "));
                    if known.is_empty() {
                        terms.push(format!("md5({text}) COLLATE \"C\""));
                    }
                    terms.push(format!("{text} COLLATE \"C\""));
                }
                terms
            }
            Backend::Snow => {
                sdm_dialects::fragments::sample_order_terms(&columns, pk.as_deref(), quote, "HASH")
            }
            Backend::Databricks => sdm_dialects::fragments::sample_order_terms(
                &columns,
                pk.as_deref(),
                |n| Backend::Databricks.quote(n),
                "xxhash64",
            ),
            _ => Vec::new(),
        })
    }

    /// Snowflake's `_listing`: what `SHOW TABLES` says of this table.
    async fn snow_listing(&mut self, ctx: &mut Ctx<'_>) -> ER<Option<Vec<(String, PyVal)>>> {
        if self.listed.is_none() {
            let stmt = sdm_dialects::snow::tables_like(
                self.place.catalog.as_deref().unwrap_or_default(),
                self.place.schema.as_deref().unwrap_or_default(),
                &self.raw_source_name,
            );
            self.listed = Some(match ctx.db.run(&stmt).await {
                Ok(table) => {
                    let fields: Vec<String> = table
                        .column_names()
                        .iter()
                        .map(|f| f.to_lowercase())
                        .collect();
                    Some(
                        table
                            .rows()
                            .into_iter()
                            .map(|r| fields.iter().cloned().zip(r).collect())
                            .collect(),
                    )
                }
                Err(_) => None,
            });
        }
        let Some(Some(rows)) = &self.listed else {
            return Ok(None);
        };
        let get = |row: &Vec<(String, PyVal)>, f: &str| {
            row.iter().find(|(k, _)| k == f).map(|(_, v)| v.clone())
        };
        let mut named: Vec<&Vec<(String, PyVal)>> = rows
            .iter()
            .filter(|r| matches!(get(r, "name"), Some(PyVal::Str(s)) if s == self.raw_source_name))
            .collect();
        if named.is_empty() {
            let folded = self.raw_source_name.to_uppercase();
            named = rows
                .iter()
                .filter(|r| get(r, "name").unwrap_or(PyVal::None).str_().to_uppercase() == folded)
                .collect();
        }
        if named.len() != 1 {
            return Ok(None);
        }
        let fields = [
            "rows",
            "bytes",
            "kind",
            "is_external",
            "is_hybrid",
            "is_iceberg",
            "is_dynamic",
        ];
        let mut out = Vec::new();
        for f in fields {
            match get(named[0], f) {
                Some(v) => out.push((f.to_string(), v)),
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    async fn snow_prefilter(&mut self, ctx: &mut Ctx<'_>) -> ER<Option<String>> {
        let Some(listed) = self.snow_listing(ctx).await? else {
            return Ok(None);
        };
        let get = |f: &str| {
            listed
                .iter()
                .find(|(k, _)| k == f)
                .map(|(_, v)| v.clone())
                .unwrap_or(PyVal::None)
        };
        if !["TABLE", "TRANSIENT", "TEMPORARY"]
            .contains(&get("kind").str_().trim().to_uppercase().as_str())
        {
            return Ok(None);
        }
        for flag in ["is_external", "is_hybrid", "is_iceberg", "is_dynamic"] {
            if get(flag).str_().trim().to_uppercase() == "Y" {
                return Ok(None);
            }
        }
        let (Some(num_rows), Some(num_bytes)) = (int_of(&get("rows")), int_of(&get("bytes")))
        else {
            return Ok(None);
        };
        if num_rows <= 0 {
            return Ok(None);
        }
        use sdm_dialects::py::Fraction;
        let fraction = |n: i128, d: i128| {
            Fraction::new(n, d).map_err(|e| PyExc::new("ValueError", e.to_string()))
        };
        let one = Fraction::integer(1);
        let partitions = fraction(i128::from(num_bytes), 16 << 20)?;
        let partitions = if one >= partitions { one } else { partitions };
        let by_rows = fraction(100_000, i128::from(num_rows))?;
        let by_partitions = fraction(128 * partitions.denominator, partitions.numerator)?;
        let wanted = if by_rows >= by_partitions {
            by_rows
        } else {
            by_partitions
        };
        let found = sdm_dialects::fragments::prefilter_fraction(wanted)
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?;
        let Some(f) = found else {
            return Ok(None);
        };
        let percent = sdm_dialects::fragments::percent_literal(f)
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?;
        Ok(Some(sdm_dialects::snow::block_sample(
            &percent,
            &sdm_dialects::int::Int::from(0),
        )))
    }

    async fn postgres_prefilter(&mut self, ctx: &mut Ctx<'_>) -> ER<Option<String>> {
        if self.source_primary_key(ctx).await?.is_some() {
            return Ok(None);
        }
        let found = match ctx
            .db
            .run(&sdm_dialects::postgres::relation_size(
                &self.quoted_source_name(),
            ))
            .await
        {
            Ok(t) => t.pg_rows(),
            Err(_) => return Ok(None),
        };
        let Some(row) = found.first() else {
            return Ok(None);
        };
        let kind = text_of(&row[0]);
        if !matches!(kind.as_str(), "r" | "m" | "p") || matches!(row[1], PyVal::None) {
            return Ok(None);
        }
        let blocks = int_of(&row[1]).unwrap_or(0);
        if blocks < 16_384 {
            return Ok(None);
        }
        use sdm_dialects::py::Fraction;
        let f = Fraction::new(2048, i128::from(blocks))
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?;
        let Some(f) = sdm_dialects::fragments::prefilter_fraction(f)
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?
        else {
            return Ok(None);
        };
        let percent = sdm_dialects::fragments::percent_literal(f)
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?;
        Ok(Some(sdm_dialects::postgres::block_sample(
            &self.quoted_source_name(),
            &percent,
        )))
    }

    async fn databricks_sample_source(&mut self, ctx: &mut Ctx<'_>) -> ER<String> {
        if let Some(s) = &self.sample_source {
            return Ok(s.clone());
        }
        let source = self.quoted_source_name();
        let name = self.databricks_name();
        let mut size: Option<i64> = None;
        if let Ok(detail) = ctx
            .db
            .run(&sdm_dialects::databricks::describe_detail(&name))
            .await
            && detail.num_rows > 0
            && let Some(c) = detail.columns.iter().find(|c| c.name() == "sizeInBytes")
        {
            size = c.values().first().and_then(int_of);
        }
        let large = match size {
            Some(s) => s >= 1 << 30,
            None => match ctx
                .db
                .run(&sdm_dialects::databricks::describe_extended(&name))
                .await
            {
                Err(_) => false,
                Ok(t) => {
                    let described: Vec<(String, String)> = t
                        .rows()
                        .into_iter()
                        .filter(|r| r.len() >= 2 && !matches!(r[0], PyVal::None))
                        .map(|r| {
                            (
                                text_of(&r[0]).trim().to_string(),
                                text_of(&r[1]).trim().to_string(),
                            )
                        })
                        .collect();
                    let get = |k: &str| {
                        described
                            .iter()
                            .rev()
                            .find(|(n, _)| n == k)
                            .map(|(_, v)| v.clone())
                            .unwrap_or_default()
                    };
                    if get("Type").to_uppercase() == "VIEW" {
                        true
                    } else {
                        let stats = get("Statistics");
                        let digits: String =
                            stats.chars().take_while(char::is_ascii_digit).collect();
                        !digits.is_empty()
                            && stats[digits.len()..].starts_with(" bytes")
                            && digits.parse::<i64>().is_ok_and(|n| n >= 1 << 30)
                    }
                }
            },
        };
        let settled = if large {
            sdm_dialects::databricks::bounded(&source, 100_000)
        } else {
            source
        };
        self.sample_source = Some(settled.clone());
        Ok(settled)
    }

    /// `_read_sample(projections)` of the warehouse backends: the result,
    /// without its order terms; `None` where it read no rows (Snowflake).
    async fn read_sample(&mut self, ctx: &mut Ctx<'_>, projections: &[String]) -> ER<ResultTable> {
        let order = self.order_terms(ctx).await?;
        let backend = self.backend;
        let sample = |source: &str| match backend {
            Backend::Postgres => {
                sdm_dialects::postgres::sample(source, projections, &order, NUM_SAMPLE_ROWS)
            }
            Backend::Snow => {
                sdm_dialects::snow::sample(source, projections, &order, NUM_SAMPLE_ROWS)
            }
            _ => sdm_dialects::databricks::sample(source, projections, &order, NUM_SAMPLE_ROWS),
        };
        let table = if self.backend == Backend::Databricks {
            let source = self.databricks_sample_source(ctx).await?;
            must(ctx.db, backend, &sample(&source)).await?
        } else if let Some(settled) = self.sample_source.clone() {
            must(ctx.db, backend, &sample(&settled)).await?
        } else {
            let source = self.quoted_source_name();
            let prefiltered = if self.backend == Backend::Postgres {
                self.postgres_prefilter(ctx).await?
            } else {
                self.snow_prefilter(ctx)
                    .await?
                    .map(|clause| format!("{source} {clause}"))
            };
            let mut table = must(
                ctx.db,
                backend,
                &sample(prefiltered.as_deref().unwrap_or(&source)),
            )
            .await?;
            let mut settled = source.clone();
            if let Some(p) = &prefiltered {
                if table.num_rows >= NUM_SAMPLE_ROWS as usize {
                    settled = p.clone();
                } else {
                    table = must(ctx.db, backend, &sample(&source)).await?;
                }
            }
            self.sample_source = Some(settled);
            table
        };
        Ok(table.drop_first(order.len()))
    }

    fn empty_error(&self) -> PyExc {
        empty_table(&self.source_name())
    }

    /// `_get_source_sample_df`.
    async fn get_source_sample(&mut self, ctx: &mut Ctx<'_>) -> ER<SampleFrame> {
        let dtypes: Vec<(String, Option<Dtype>)> = self
            .source_column_dict(ctx)
            .await?
            .iter()
            .map(|c| (c.name.clone(), c.dtype))
            .collect();
        let projections: Vec<String> = dtypes.iter().map(|(n, _)| self.backend.quote(n)).collect();
        let table = match self.backend {
            Backend::Sqlite => {
                let t = must(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::sqlite::first_rows(
                        &self.quoted_source_name(),
                        &projections,
                        NUM_SAMPLE_ROWS,
                    ),
                )
                .await?;
                if t.num_rows == 0 {
                    return Err(self.empty_error());
                }
                t
            }
            Backend::Duckdb => {
                let t = must(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::duckdb::first_rows(
                        &self.quoted_source_name(),
                        &projections,
                        NUM_SAMPLE_ROWS,
                    ),
                )
                .await?;
                if t.num_rows == 0 {
                    return Err(self.empty_error());
                }
                t
            }
            _ => {
                let t = self.read_sample(ctx, &projections).await?;
                if t.num_rows == 0 {
                    return Err(self.empty_error());
                }
                t
            }
        };
        Ok(sanitize(&table, &dtypes, self.backend, ctx))
    }

    /// `_source_sample_df`.
    pub async fn source_sample(&mut self, ctx: &mut Ctx<'_>) -> ER<SampleFrame> {
        if self.source_sample.is_none() {
            let sample = self.get_source_sample(ctx).await?;
            self.source_sample = Some(sample);
        }
        Ok(self.source_sample.clone().unwrap_or_default())
    }

    /// `_get_expr_sample_df(columns)`.
    async fn get_expr_sample(
        &mut self,
        ctx: &mut Ctx<'_>,
        specs: &[(String, String, Option<Dtype>)],
    ) -> ER<SampleFrame> {
        let projections: Vec<String> = specs
            .iter()
            .map(|(n, e, _)| format!("{e} AS {}", self.backend.quote(n)))
            .collect();
        let table = match self.backend {
            Backend::Sqlite => {
                must(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::sqlite::first_rows(
                        &self.quoted_source_name(),
                        &projections,
                        NUM_SAMPLE_ROWS,
                    ),
                )
                .await?
            }
            Backend::Duckdb => {
                must(
                    ctx.db,
                    self.backend,
                    &sdm_dialects::duckdb::first_rows(
                        &self.quoted_source_name(),
                        &projections,
                        NUM_SAMPLE_ROWS,
                    ),
                )
                .await?
            }
            _ => self.read_sample(ctx, &projections).await?,
        };
        if table.num_rows == 0 {
            return Err(self.empty_error());
        }
        let dtypes: Vec<(String, Option<Dtype>)> =
            specs.iter().map(|(n, _, d)| (n.clone(), *d)).collect();
        Ok(sanitize(&table, &dtypes, self.backend, ctx))
    }

    /// `_update_expr_sample_df(columns)`.
    async fn update_expr_sample(
        &mut self,
        ctx: &mut Ctx<'_>,
        specs: &[(String, String, Option<Dtype>)],
    ) -> ER<()> {
        if specs.is_empty() {
            return Ok(());
        }
        let fresh = self.get_expr_sample(ctx, specs).await?;
        self.expr_sample = SampleFrame::concat(&self.expr_sample, &fresh);
        Ok(())
    }

    /// `_get_sample_df()`.
    pub async fn sample_frame(&mut self, ctx: &mut Ctx<'_>) -> ER<SampleFrame> {
        let mut frames: Vec<SampleFrame> = Vec::new();
        if self.columns.iter().any(EColumn::is_source) {
            frames.push(self.source_sample(ctx).await?);
        }
        if self.columns.iter().any(|c| !c.is_source()) {
            let missing: Vec<(String, String, Option<Dtype>)> = self
                .columns
                .iter()
                .filter(|c| !c.is_source() && !self.expr_sample.has(&c.name))
                .map(|c| {
                    (
                        c.name.clone(),
                        c.expr.clone().unwrap_or_default(),
                        Some(c.dtype),
                    )
                })
                .collect();
            self.update_expr_sample(ctx, &missing).await?;
            frames.push(self.expr_sample.clone());
        }
        Ok(match frames.len() {
            0 => SampleFrame::empty_index(NUM_SAMPLE_ROWS as usize),
            1 => frames.remove(0),
            _ => SampleFrame::concat(&frames[0], &frames[1]),
        })
    }

    // --- Columns and keys -------------------------------------------------------------

    /// The sample of the column `spec` adds: from the source's sample, or
    /// from the expressions' for a derived column.
    async fn sample_column(&mut self, ctx: &mut Ctx<'_>, spec: &ColumnSpec) -> ER<SampleColumn> {
        let frame = if spec.expr.is_none() {
            self.source_sample(ctx).await?
        } else {
            self.expr_sample.clone()
        };
        frame
            .column(&spec.name)
            .cloned()
            .ok_or_else(|| PyExc::key_error(&spec.name))
    }

    /// `add_columns(specs)`.
    pub async fn add_columns(&mut self, ctx: &mut Ctx<'_>, specs: Vec<ColumnSpec>) -> ER<()> {
        if specs.is_empty() {
            return Ok(());
        }
        let exprs: Vec<&ColumnSpec> = specs.iter().filter(|s| s.expr.is_some()).collect();
        if !exprs.is_empty() {
            if exprs.iter().any(|s| s.dtype.is_none() || s.stype.is_none()) {
                let wanted: Vec<(String, String, Option<Dtype>)> = exprs
                    .iter()
                    .map(|s| (s.name.clone(), s.expr.clone().unwrap_or_default(), s.dtype))
                    .collect();
                self.update_expr_sample(ctx, &wanted).await?;
            } else {
                let names: Vec<String> = exprs.iter().map(|s| s.name.clone()).collect();
                self.expr_sample.drop(&names);
            }
        }
        for spec in specs {
            if self.has_column(&spec.name) {
                return Err(PyExc::key_error(&format!(
                    "Column '{}' already exists in table '{}'",
                    spec.name, self.name
                )));
            }
            let mut dtype = spec.dtype;
            let mut stype = spec.stype;
            if spec.expr.is_none() {
                let Some(source) = self.source_column(ctx, &spec.name).await? else {
                    return raise(
                        "ValueError",
                        format!(
                            "Column '{}' does not exist in the underlying source table",
                            spec.name
                        ),
                    );
                };
                if dtype.is_none() {
                    dtype = source.dtype;
                }
                if dtype == Some(Dtype::Unsupported) {
                    return raise(
                        "ValueError",
                        format!(
                            "Encountered unsupported data type for column '{}' in table '{}'. Please either change the column's data type or remove the column from this table.",
                            spec.name, self.name
                        ),
                    );
                }
            }
            let dtype = match dtype {
                Some(d) => d,
                None => {
                    let ser = self.sample_column(ctx, &spec).await?;
                    match infer::infer_dtype(&ser) {
                        Ok(d) => d,
                        Err(why) => {
                            // Raised from the `ValueError` inference raised.
                            return Err(PyExc::new(
                                "RuntimeError",
                                format!(
                                    "Encountered unsupported data type '{}' for column '{}' in table '{}'. Please either cast the column to a supported type before building the table (for example df['{}'] = df['{}'].astype('int64')) or remove the column from this table.",
                                    infer::dtype_name(&ser),
                                    spec.name,
                                    self.name,
                                    spec.name,
                                    spec.name
                                ),
                            )
                            .caused_by([link("ValueError", why.0)]));
                        }
                    }
                }
            };
            let stype = match stype.take() {
                Some(s) => s,
                None => {
                    let ser = self.sample_column(ctx, &spec).await?;
                    let (found, warnings) = infer::infer_stype(&ser, dtype, ctx.host);
                    for w in warnings {
                        ctx.warnings.user(w);
                    }
                    match found {
                        Ok(s) => s,
                        Err(_) => {
                            return raise(
                                "RuntimeError",
                                format!(
                                    "Could not determine semantic type for column '{}' with data type '{}' in table '{}'. Please either change the column's data type or remove the column from this table.",
                                    spec.name,
                                    dtype.value(),
                                    self.name
                                ),
                            );
                        }
                    }
                }
            };
            let mut column = EColumn {
                name: spec.name.clone(),
                expr: spec.expr.clone(),
                dtype,
                stype,
                is_pk: false,
                is_time: false,
                is_end: false,
            };
            column.set_stype(stype)?;
            self.columns.push(column);
        }
        Ok(())
    }

    /// `remove_column(name)`.
    pub fn remove_column(&mut self, name: &str) -> ER<()> {
        if !self.has_column(name) {
            return Err(self.not_found(name));
        }
        if self.primary_key.as_deref() == Some(name) {
            self.set_primary_key(None)?;
        }
        if self.time_column.as_deref() == Some(name) {
            self.set_time_column(None)?;
        }
        if self.end_time_column.as_deref() == Some(name) {
            self.set_end_time_column(None)?;
        }
        self.columns.retain(|c| c.name != name);
        Ok(())
    }

    /// The `primary_key` setter, for one name or none.
    pub fn set_primary_key(&mut self, name: Option<&str>) -> ER<()> {
        if let Some(n) = name {
            if Some(n) == self.time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a primary key since it is already defined to be a time column"
                    ),
                );
            }
            if Some(n) == self.end_time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a primary key since it is already defined to be an end time column"
                    ),
                );
            }
            if !self.has_column(n) {
                return Err(self.not_found(n));
            }
        }
        if let Some(old) = self.primary_key.clone()
            && let Some(c) = self.column_mut(&old)
        {
            c.is_pk = false;
        }
        let Some(n) = name else {
            self.primary_key = None;
            return Ok(());
        };
        let column = self.column_mut(n).expect("checked");
        column.set_stype(Stype::Id)?;
        column.is_pk = true;
        self.primary_key = Some(n.to_string());
        Ok(())
    }

    /// The `time_column` setter.
    pub fn set_time_column(&mut self, name: Option<&str>) -> ER<()> {
        if let Some(n) = name {
            if Some(n) == self.primary_key.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a time column since it is already defined to be a primary key"
                    ),
                );
            }
            if Some(n) == self.end_time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a time column since it is already defined to be an end time column"
                    ),
                );
            }
        }
        if let Some(name) = name {
            let Some(column) = self.column(name) else {
                return Err(self.not_found(name));
            };
            let mut checked = column.clone();
            checked.set_stype(Stype::Timestamp)?;
        }
        self.time_inferred = false;
        self.time_choice = None;
        if let Some(old) = self.time_column.clone()
            && let Some(c) = self.column_mut(&old)
        {
            c.is_time = false;
        }
        let Some(n) = name else {
            self.time_column = None;
            return Ok(());
        };
        let Some(column) = self.column_mut(n) else {
            return Err(self.not_found(n));
        };
        column.set_stype(Stype::Timestamp)?;
        column.is_time = true;
        self.time_column = Some(n.to_string());
        Ok(())
    }

    /// The `end_time_column` setter.
    pub fn set_end_time_column(&mut self, name: Option<&str>) -> ER<()> {
        if let Some(n) = name {
            if Some(n) == self.primary_key.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as an end time column since it is already defined to be a primary key"
                    ),
                );
            }
            if Some(n) == self.time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as an end time column since it is already defined to be a time column"
                    ),
                );
            }
        }
        if let Some(old) = self.end_time_column.clone()
            && let Some(c) = self.column_mut(&old)
        {
            c.is_end = false;
        }
        let Some(n) = name else {
            self.end_time_column = None;
            return Ok(());
        };
        let Some(column) = self.column_mut(n) else {
            return Err(self.not_found(n));
        };
        column.set_stype(Stype::Timestamp)?;
        column.is_end = true;
        self.end_time_column = Some(n.to_string());
        Ok(())
    }

    /// `infer_primary_key()`.
    pub async fn infer_primary_key(&mut self, ctx: &mut Ctx<'_>) -> ER<()> {
        if self.primary_key.is_some() {
            return Ok(());
        }
        if self.columns.iter().any(EColumn::is_source) {
            if let Some(pk) = self.source_primary_key(ctx).await?
                && self.column(&pk).is_some_and(EColumn::is_source)
            {
                return self.set_primary_key(Some(&pk));
            }
            let unique: Vec<String> = self
                .source_column_dict(ctx)
                .await?
                .iter()
                .filter(|c| c.is_unique_key)
                .map(|c| c.name.clone())
                .collect();
            if unique.len() == 1 && self.column(&unique[0]).is_some_and(EColumn::is_source) {
                return self.set_primary_key(Some(&unique[0]));
            }
        }
        let mut candidates: Vec<String> = self
            .columns
            .iter()
            .filter(|c| c.stype == Stype::Id)
            .map(|c| c.name.clone())
            .collect();
        if candidates.is_empty() {
            let table = self.name.to_lowercase();
            for c in &self.columns {
                let col = c.name.to_lowercase();
                if table == col || (table.ends_with('s') && table[..table.len() - 1] == col) {
                    candidates.push(c.name.clone());
                }
            }
        }
        let df = self.sample_frame(ctx).await?;
        let (found, warnings) = infer::infer_primary_key(&self.name, &df, &candidates);
        for w in warnings {
            ctx.warnings.user(w);
        }
        if let Some(pk) = found {
            return self.set_primary_key(Some(&pk));
        }
        Ok(())
    }

    /// `infer_time_column()`.
    pub async fn infer_time_column(&mut self, ctx: &mut Ctx<'_>) -> ER<()> {
        if self.time_column.is_some() {
            return Ok(());
        }
        let candidates: Vec<String> = self
            .columns
            .iter()
            .filter(|c| {
                c.stype == Stype::Timestamp && Some(&c.name) != self.end_time_column.as_ref()
            })
            .map(|c| c.name.clone())
            .collect();
        self.infer_time_column_among(ctx, &candidates).await
    }

    pub async fn infer_time_column_among(
        &mut self,
        ctx: &mut Ctx<'_>,
        candidates: &[String],
    ) -> ER<()> {
        let df = self.sample_frame(ctx).await?;
        let attempted = df.len() == NUM_SAMPLE_ROWS as usize;
        let eligible: Vec<String> = candidates
            .iter()
            .filter(|name| {
                !crate::time_columns::load_stamp(name) && !crate::time_columns::birth_date(name)
            })
            .cloned()
            .collect();
        let whole = if attempted && !eligible.is_empty() {
            self.full_time_range(ctx, &eligible).await
        } else {
            None
        };
        let choice =
            crate::time_columns::choose(&df, candidates, whole.as_ref(), attempted, ctx.host);
        self.set_time_column(choice.column.as_deref())?;
        for doubt in &choice.unconfirmed {
            if choice.column.as_ref() == Some(&doubt.column) {
                ctx.warnings.0.push((
                    crate::diagnostics::TIME_COLUMN_WARNING.to_string(),
                    doubt.taken_unchecked(
                        &self.name,
                        &format!(
                            "graph[{}].time_column = None",
                            crate::py::str_repr(&self.name)
                        ),
                    ),
                ));
            }
        }
        self.time_choice = Some(choice);
        self.time_inferred = true;
        Ok(())
    }

    async fn full_time_range(
        &self,
        ctx: &mut Ctx<'_>,
        names: &[String],
    ) -> Option<crate::time_columns::TableRange> {
        let mut projections = vec!["COUNT(*)".to_string()];
        for name in names {
            let column = self.column(name)?;
            let expression = column
                .expr
                .as_ref()
                .map_or_else(|| self.quote_key_column(name), |expr| format!("({expr})"));
            projections.push(format!(
                "MIN({expression}), MAX({expression}), COUNT({expression})"
            ));
        }
        let statement = (
            format!(
                "SELECT {}\nFROM {}",
                projections.join(", "),
                self.quoted_source_name()
            ),
            None,
        );
        let read = rows(ctx.db, self.backend, &statement).await.ok()?;
        let row = read.first()?;
        let count = usize::try_from(int_of(row.first()?)?).ok()?;
        let mut columns = BTreeMap::new();
        for (index, name) in names.iter().enumerate() {
            columns.insert(
                name.clone(),
                crate::time_columns::Range {
                    low: row.get(1 + 3 * index)?.clone(),
                    high: row.get(2 + 3 * index)?.clone(),
                    held: usize::try_from(int_of(row.get(3 + 3 * index)?)?).ok()?,
                },
            );
        }
        Some(crate::time_columns::TableRange {
            rows: count,
            columns,
        })
    }

    pub fn settle_time_column(&mut self, linked: bool) -> ER<()> {
        if !self.time_inferred {
            return Ok(());
        }
        let Some(choice) = &self.time_choice else {
            return Ok(());
        };
        let settled = choice.for_links(linked);
        if let Some(column) = &settled.column
            && (!self.has_column(column)
                || self.primary_key.as_ref() == Some(column)
                || self.end_time_column.as_ref() == Some(column))
        {
            return Ok(());
        }
        if &settled != choice {
            self.set_time_column(settled.column.as_deref())?;
            self.time_choice = Some(settled);
            self.time_inferred = true;
        }
        Ok(())
    }

    /// `infer_metadata()`.
    pub async fn infer_metadata(&mut self, ctx: &mut Ctx<'_>) -> ER<()> {
        if self.primary_key.is_none() {
            self.infer_primary_key(ctx).await?;
        }
        if self.time_column.is_none() {
            self.infer_time_column(ctx).await?;
        }
        Ok(())
    }

    // --- Key counts --------------------------------------------------------------------

    /// `_declares_key(names)`.
    pub async fn declares_key(&mut self, ctx: &mut Ctx<'_>, names: &[String]) -> ER<bool> {
        if names.len() != 1 {
            return Ok(false);
        }
        Ok(self
            .source_column(ctx, &names[0])
            .await?
            .is_some_and(|c| c.is_primary_key || c.is_unique_key))
    }

    /// `_enforces_key(names)`: unique, never null.
    pub async fn enforces_key(&mut self, ctx: &mut Ctx<'_>, names: &[String]) -> ER<(bool, bool)> {
        let columns = self.source_column_dict(ctx).await?.to_vec();
        if self.backend == Backend::Snow {
            let primary: BTreeSet<String> = columns
                .iter()
                .filter(|c| c.is_primary_key)
                .map(|c| c.name.clone())
                .collect();
            let named: BTreeSet<String> = names.iter().cloned().collect();
            if primary.is_empty() || !primary.is_subset(&named) {
                return Ok((false, false));
            }
            if !self.snow_is_hybrid(ctx).await? {
                return Ok((false, false));
            }
            let never_null = names
                .iter()
                .all(|n| columns.iter().any(|c| &c.name == n && !c.is_nullable));
            return Ok((true, never_null));
        }
        if names.iter().any(|n| !columns.iter().any(|c| &c.name == n)) {
            return Ok((false, false));
        }
        let named: BTreeSet<String> = names.iter().cloned().collect();
        let unique = self.unique_constraints.iter().any(|c| c.is_subset(&named));
        let never_null = names
            .iter()
            .all(|n| columns.iter().any(|c| &c.name == n && !c.is_nullable));
        Ok((unique, never_null))
    }

    /// Snowflake's `_hybrid_tables()`: the hybrid tables of the schema.
    async fn snow_hybrid_tables(&self, ctx: &mut Ctx<'_>) -> ER<BTreeSet<String>> {
        let stmt = sdm_dialects::snow::tables_in_schema(
            self.place.catalog.as_deref().unwrap_or_default(),
            self.place.schema.as_deref().unwrap_or_default(),
        );
        let Ok(listed) = ctx.db.run(&stmt).await else {
            return Ok(BTreeSet::new());
        };
        let fields: Vec<String> = listed
            .column_names()
            .iter()
            .map(|f| f.to_lowercase())
            .collect();
        let (Some(name), Some(hybrid)) = (
            fields.iter().position(|f| f == "name"),
            fields.iter().position(|f| f == "is_hybrid"),
        ) else {
            return Ok(BTreeSet::new());
        };
        Ok(listed
            .rows()
            .into_iter()
            .filter(|r| r[hybrid].str_().trim().to_uppercase() == "Y")
            .map(|r| text_of(&r[name]))
            .collect())
    }

    async fn snow_is_hybrid(&mut self, ctx: &mut Ctx<'_>) -> ER<bool> {
        if let Some(h) = self.is_hybrid {
            return Ok(h);
        }
        let hybrid = self
            .snow_hybrid_tables(ctx)
            .await?
            .contains(&self.raw_source_name);
        self.is_hybrid = Some(hybrid);
        Ok(hybrid)
    }

    fn key_count_statement(&self, parts: &[KeyCountPart]) -> Statement {
        match self.backend {
            Backend::Sqlite => sdm_dialects::sqlite::key_counts(parts),
            Backend::Duckdb => sdm_dialects::duckdb::key_counts(parts),
            Backend::Postgres => sdm_dialects::postgres::key_counts(parts),
            Backend::Snow => sdm_dialects::snow::key_counts(parts),
            Backend::Databricks => sdm_dialects::databricks::key_counts(parts),
        }
    }

    fn repeated_key_statement(&self, names: &[String]) -> Statement {
        let source = self.quoted_source_name();
        let columns: Vec<String> = names.iter().map(|n| self.quote_key_column(n)).collect();
        match self.backend {
            Backend::Sqlite => sdm_dialects::sqlite::repeated_key(&source, &columns),
            Backend::Duckdb => sdm_dialects::duckdb::repeated_key(&source, &columns),
            Backend::Postgres => sdm_dialects::postgres::repeated_key(&source, &columns),
            Backend::Snow => sdm_dialects::snow::repeated_key(&source, &columns),
            Backend::Databricks => sdm_dialects::databricks::repeated_key(&source, &columns),
        }
    }
}

fn list_texts(value: &PyVal) -> Vec<String> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => items.iter().map(text_of).collect(),
        _ => Vec::new(),
    }
}

fn count_by(rows: &[Vec<PyVal>], index: usize) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for r in rows {
        if let Some(v) = r.get(index) {
            *counts.entry(v.eq_key()).or_default() += 1;
        }
    }
    counts
}

fn sqlite_dtype(value: &PyVal) -> Option<Dtype> {
    if matches!(value, PyVal::None) {
        return None;
    }
    let d = text_of(value).trim().to_uppercase();
    if d.contains("INT") {
        return Some(Dtype::Int);
    }
    if d.contains("TEXT") || d.contains("CHAR") || d.contains("CLOB") {
        return Some(Dtype::String);
    }
    if d.contains("REAL") || d.contains("FLOA") || d.contains("DOUB") {
        return Some(Dtype::Float);
    }
    None
}

fn scale_of(spelled: &str) -> Option<i64> {
    let last = spelled.split(',').next_back()?;
    let inside = last.split(')').next()?;
    inside.trim().parse().ok()
}

/// `DuckDBTable._to_dtype`.
pub fn duckdb_dtype(spelled: &str) -> Option<Dtype> {
    let d = spelled.trim().to_uppercase();
    if let Some(element) = d.strip_suffix("[]") {
        let e = duckdb_dtype(element)?;
        return Some(if e.is_int() {
            Dtype::Intlist
        } else if e.is_float() {
            Dtype::Floatlist
        } else if e.is_string() {
            Dtype::Stringlist
        } else {
            Dtype::Unsupported
        });
    }
    let has = |words: &[&str]| words.iter().any(|w| d.contains(w));
    if has(&["UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"])
        || has(&["TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT"])
    {
        return Some(Dtype::Int);
    }
    if d.starts_with("DECIMAL") || d.starts_with("NUMERIC") {
        return Some(match scale_of(&d) {
            Some(0) => Dtype::Int,
            _ => Dtype::Float,
        });
    }
    if has(&["REAL", "FLOAT", "DOUBLE"]) {
        return Some(Dtype::Float);
    }
    if has(&["VARCHAR", "CHAR", "STRING", "TEXT"]) {
        return Some(Dtype::String);
    }
    if d == "BOOLEAN" {
        return Some(Dtype::Bool);
    }
    if d.starts_with("DATE") || d.starts_with("TIMESTAMP") {
        return Some(Dtype::Date);
    }
    if d.starts_with("TIME") {
        // A time of day, which no date goes with: text, not a moment the
        // table's rows could be ordered by (today's took it as one, and
        // could read none of its values).
        return Some(Dtype::String);
    }
    if d.starts_with("BLOB") {
        return Some(Dtype::Binary);
    }
    None
}

fn postgres_dtype(data_type: &str, udt: &PyVal, scale: &PyVal) -> Option<Dtype> {
    let d = data_type.trim().to_lowercase();
    let udt = match udt {
        PyVal::None => String::new(),
        v => text_of(v).trim().to_lowercase(),
    };
    Some(match d.as_str() {
        "smallint" | "integer" | "bigint" => Dtype::Int,
        "real" | "double precision" => Dtype::Float,
        "numeric" | "decimal" => {
            if int_of(scale) == Some(0) && !matches!(scale, PyVal::None) {
                Dtype::Int
            } else {
                Dtype::Float
            }
        }
        "boolean" => Dtype::Bool,
        "character" | "character varying" | "text" | "name" | "uuid" | "json" | "jsonb" | "xml"
        | "inet" | "cidr" | "macaddr" | "macaddr8" => Dtype::String,
        "bytea" => Dtype::Binary,
        "date" => Dtype::Date,
        _ if d.starts_with("timestamp") => Dtype::Date,
        // A time of day: text, as for DuckDB.
        _ if d.starts_with("time") => Dtype::String,
        "array" => match udt.as_str() {
            "_int2" | "_int4" | "_int8" => Dtype::Intlist,
            "_float4" | "_float8" | "_numeric" => Dtype::Floatlist,
            "_text" | "_varchar" | "_bpchar" | "_uuid" => Dtype::Stringlist,
            _ => Dtype::Unsupported,
        },
        _ => {
            if [
                "interval",
                "range",
                "multirange",
                "composite",
                "record",
                "point",
                "line",
                "polygon",
            ]
            .iter()
            .any(|w| d.contains(w))
            {
                Dtype::Unsupported
            } else {
                return None;
            }
        }
    })
}

/// `SnowTable._to_dtype`.
pub fn snow_dtype(spelled: Option<&str>) -> Option<Dtype> {
    let d = spelled?.trim().to_uppercase();
    if d.starts_with("NUMBER") {
        return Some(
            match d
                .split(',')
                .next_back()
                .and_then(|s| s.split(')').next())
                .and_then(|s| s.trim().parse::<i64>().ok())
            {
                Some(0) => Dtype::Int,
                Some(_) => Dtype::Float,
                None => Dtype::Float,
            },
        );
    }
    if d == "FLOAT" {
        return Some(Dtype::Float);
    }
    if d.starts_with("VARCHAR") {
        return Some(Dtype::String);
    }
    if d.starts_with("BINARY") {
        return Some(Dtype::Binary);
    }
    if d == "BOOLEAN" {
        return Some(Dtype::Bool);
    }
    if d.starts_with("DATE") || d.starts_with("TIMESTAMP") {
        return Some(Dtype::Date);
    }
    if d.starts_with("TIME") {
        // A time of day, which no date goes with: text, not a moment the
        // table's rows could be ordered by (today's took it as one, and
        // could read none of its values).
        return Some(Dtype::String);
    }
    if d.starts_with("VECTOR") {
        let inner = d
            .split(',')
            .next()
            .and_then(|s| s.split('(').nth(1))
            .map(str::trim);
        return Some(match inner {
            Some("INT") => Dtype::Intlist,
            Some("FLOAT") => Dtype::Floatlist,
            _ => Dtype::Unsupported,
        });
    }
    if d.starts_with("ARRAY") {
        let inner = d
            .split_once('(')
            .and_then(|(_, rest)| rest.rsplit_once(')').map(|(i, _)| i.to_string()));
        if let Some(inner) = inner
            && let Some(e) = snow_dtype(Some(&inner))
        {
            if e.is_int() {
                return Some(Dtype::Intlist);
            }
            if e.is_float() {
                return Some(Dtype::Floatlist);
            }
            if e.is_string() {
                return Some(Dtype::Stringlist);
            }
        }
        return Some(Dtype::Unsupported);
    }
    if [
        "DECFLOAT",
        "VARIANT",
        "OBJECT",
        "MAP",
        "FILE",
        "GEOGRAPHY",
        "GEOMETRY",
    ]
    .iter()
    .any(|w| d.contains(w))
    {
        return Some(Dtype::Unsupported);
    }
    None
}

/// `DatabricksTable._to_dtype`.
pub fn databricks_dtype(spelled: &str) -> Option<Dtype> {
    let d = spelled.trim().to_lowercase();
    if matches!(
        d.as_str(),
        "tinyint" | "byte" | "smallint" | "short" | "int" | "integer" | "bigint" | "long"
    ) {
        return Some(Dtype::Int);
    }
    if matches!(d.as_str(), "float" | "real" | "double") {
        return Some(Dtype::Float);
    }
    if d.starts_with("decimal") || d.starts_with("numeric") || d.starts_with("dec") {
        let inside = d.find('(').zip(d.find(')'));
        return Some(match inside {
            Some((a, b)) if a < b => {
                let parts: Vec<&str> = d[a + 1..b].split(',').collect();
                let scale = if parts.len() > 1 {
                    parts[1].trim().parse::<i64>().ok()
                } else {
                    Some(0)
                };
                match scale {
                    Some(0) => Dtype::Int,
                    _ => Dtype::Float,
                }
            }
            _ => Dtype::Float,
        });
    }
    if matches!(d.as_str(), "boolean" | "bool") {
        return Some(Dtype::Bool);
    }
    if d.starts_with("string") || d.starts_with("varchar") || d.starts_with("char") {
        return Some(Dtype::String);
    }
    if d.starts_with("binary") {
        return Some(Dtype::Binary);
    }
    if d == "date" || d.starts_with("timestamp") {
        return Some(Dtype::Date);
    }
    if d.starts_with("array") {
        if let (Some(a), Some(b)) = (d.find('<'), d.rfind('>'))
            && a < b
            && let Some(e) = databricks_dtype(&d[a + 1..b])
        {
            if e.is_int() {
                return Some(Dtype::Intlist);
            }
            if e.is_float() {
                return Some(Dtype::Floatlist);
            }
            if e.is_string() {
                return Some(Dtype::Stringlist);
            }
        }
        return Some(Dtype::Unsupported);
    }
    if [
        "interval",
        "map",
        "struct",
        "variant",
        "object",
        "geography",
        "geometry",
        "void",
    ]
    .iter()
    .any(|w| d.contains(w))
    {
        return Some(Dtype::Unsupported);
    }
    None
}

fn json_dumps(value: &PyVal) -> String {
    match value {
        PyVal::None => "null".into(),
        PyVal::Bool(b) => if *b { "true" } else { "false" }.into(),
        PyVal::Int(n) => n.to_string(),
        PyVal::Float(x) => sdm_dialects::pyjson::float_text(*x),
        PyVal::Str(s) => sdm_dialects::pyjson::quote(s),
        PyVal::List(items) => format!(
            "[{}]",
            items.iter().map(json_dumps).collect::<Vec<_>>().join(",")
        ),
        PyVal::Dict(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(k, v)| format!(
                    "{}:{}",
                    sdm_dialects::pyjson::quote(&text_of(k)),
                    json_dumps(v)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        other => other.str_(),
    }
}

/// A sample read into pandas and sanitized as each backend does.
fn sanitize(
    table: &ResultTable,
    dtypes: &[(String, Option<Dtype>)],
    backend: Backend,
    ctx: &mut Ctx<'_>,
) -> SampleFrame {
    let dtype_of = |name: &str| dtypes.iter().find(|(n, _)| n == name).and_then(|(_, d)| *d);
    let mut columns = Vec::new();
    if backend == Backend::Postgres {
        let values = table.pg_columns(false);
        for (c, cells) in table.columns.iter().zip(values) {
            let dtype = dtype_of(c.name());
            let (kind, cells) = match dtype {
                Some(d) if d.is_int() => (
                    SampleKind::Nullable("Int64"),
                    cells
                        .into_iter()
                        .map(|v| match v {
                            PyVal::None => PyVal::Na,
                            PyVal::Decimal(t) => PyVal::Int(
                                t.split('.')
                                    .next()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0),
                            ),
                            other => other,
                        })
                        .collect(),
                ),
                Some(d) if d.is_float() => (
                    SampleKind::Nullable("Float64"),
                    cells
                        .into_iter()
                        .map(|v| match v {
                            PyVal::None => PyVal::Na,
                            PyVal::Decimal(t) => PyVal::Float(t.parse().unwrap_or(f64::NAN)),
                            PyVal::Int(n) => PyVal::Float(n as f64),
                            other => other,
                        })
                        .collect(),
                ),
                Some(Dtype::Bool) => (
                    SampleKind::Nullable("boolean"),
                    cells
                        .into_iter()
                        .map(|v| {
                            if matches!(v, PyVal::None) {
                                PyVal::Na
                            } else {
                                v
                            }
                        })
                        .collect(),
                ),
                Some(Dtype::String) => (
                    SampleKind::Nullable("string"),
                    cells
                        .into_iter()
                        .map(|v| match v {
                            PyVal::None | PyVal::Na => PyVal::Na,
                            PyVal::Float(x) if x.is_nan() => PyVal::Na,
                            PyVal::List(_) | PyVal::Dict(_) => PyVal::Str(json_dumps(&v)),
                            PyVal::Str(s) => PyVal::Str(s),
                            // A time of day stays one, so that its text is
                            // not read as a moment today.
                            PyVal::Time { .. } => v,
                            other => PyVal::Str(other.str_()),
                        })
                        .collect(),
                ),
                _ => (SampleKind::Object, cells),
            };
            columns.push(SampleColumn {
                name: c.name().to_string(),
                kind,
                cells,
            });
        }
    } else {
        for c in &table.columns {
            columns.push(SampleColumn {
                name: c.name().to_string(),
                kind: SampleKind::Arrow(c.data_type().clone()),
                cells: c.values(),
            });
        }
    }
    for column in &mut columns {
        let dtype = dtype_of(&column.name);
        if dtype == Some(Dtype::Time) {
            let times = infer::to_times(column, ctx.host);
            let source = column.cells.clone();
            let coerced: Vec<&PyVal> = source
                .iter()
                .zip(&times)
                .filter(|(v, t)| t.is_none() && !frame::missing(v))
                .map(|(v, _)| v)
                .collect();
            if !coerced.is_empty() {
                let mut examples: Vec<PyVal> = Vec::new();
                for v in &coerced {
                    if !examples.iter().any(|e| e.eq_key() == v.eq_key()) {
                        examples.push((*v).clone());
                    }
                }
                let shown: Vec<String> = examples.iter().take(3).map(PyVal::repr).collect();
                ctx.warnings.user(format!(
                    "Could not parse {} of {} value(s) of time column '{}' (for example [{}]); they were dropped as 'NaT'",
                    py::thousands(coerced.len() as i64),
                    py::thousands(source.len() as i64),
                    column.name,
                    shown.join(", ")
                ));
            }
            column.kind = SampleKind::Datetime64;
            column.cells = times
                .into_iter()
                .map(|t| match t {
                    Some(n) => PyVal::Timestamp(crate::dt::civil::Wall::from_epoch_nanos(n)),
                    None => PyVal::NaT,
                })
                .collect();
        } else if dtype.is_some_and(Dtype::is_list) {
            column.cells = column
                .cells
                .iter()
                .map(|v| match v {
                    PyVal::Str(s) => serde_json::from_str::<serde_json::Value>(s)
                        .ok()
                        .and_then(|j| PyVal::decode(&j).ok())
                        .unwrap_or_else(|| v.clone()),
                    other => other.clone(),
                })
                .collect();
            if column.kind != SampleKind::Object
                && !matches!(
                    column.kind,
                    SampleKind::Arrow(DataType::List(_) | DataType::LargeList(_))
                )
            {
                column.kind = SampleKind::Object;
            }
        }
    }
    SampleFrame {
        columns,
        index: table.num_rows,
    }
}

/// `KeyCount`.
#[derive(Clone, Debug)]
pub struct KeyCount {
    pub rows: Option<i64>,
    pub distinct: Option<i64>,
    pub nulls: i64,
    pub enforced: bool,
    /// `repr` of a value more than one row holds, if any.
    pub example: Option<String>,
}

impl KeyCount {
    pub fn repeats(&self) -> i64 {
        match (self.rows, self.distinct) {
            (Some(r), Some(d)) => r - self.nulls - d,
            _ => 0,
        }
    }
}

/// `_before_counting_keys(keys)`: on Snowflake, each schema's hybrid
/// tables listed once for every table whose declared key `keys` names;
/// nothing on the other backends, which read it with the columns.
pub async fn before_counting_keys(
    ctx: &mut Ctx<'_>,
    tables: &mut [EngineTable],
    keys: &[(usize, Vec<String>)],
) -> ER<()> {
    if tables.first().is_some_and(|t| t.backend == Backend::Snow) {
        let mut asked: Vec<((String, String), Vec<usize>)> = Vec::new();
        for (table, names) in keys {
            let t = &mut tables[*table];
            if t.is_hybrid.is_some() {
                continue;
            }
            let primary: BTreeSet<String> = t
                .source_column_dict(ctx)
                .await?
                .iter()
                .filter(|c| c.is_primary_key)
                .map(|c| c.name.clone())
                .collect();
            let named: BTreeSet<String> = names.iter().cloned().collect();
            if !primary.is_empty() && primary.is_subset(&named) {
                let schema = (
                    t.place.catalog.clone().unwrap_or_default(),
                    t.place.schema.clone().unwrap_or_default(),
                );
                match asked.iter_mut().find(|(s, _)| *s == schema) {
                    Some((_, v)) => v.push(*table),
                    None => asked.push((schema, vec![*table])),
                }
            }
        }
        for (_, group) in asked {
            let hybrid = tables[group[0]].snow_hybrid_tables(ctx).await?;
            for i in group {
                tables[i].is_hybrid = Some(hybrid.contains(&tables[i].raw_source_name));
            }
        }
    }
    Ok(())
}

/// `count_keys(keys)`, over tables of one backend.
pub async fn count_keys(
    ctx: &mut Ctx<'_>,
    tables: &mut [EngineTable],
    keys: &[(usize, Vec<String>)],
) -> ER<Vec<KeyCount>> {
    let mut counts: Vec<KeyCount> = keys
        .iter()
        .map(|_| KeyCount {
            rows: None,
            distinct: None,
            nulls: 0,
            enforced: true,
            example: None,
        })
        .collect();
    before_counting_keys(ctx, tables, keys).await?;
    let mut parts: Vec<KeyCountPart> = Vec::new();
    let mut unique_by_label: Vec<(usize, bool)> = Vec::new();
    for (label, (table, names)) in keys.iter().enumerate() {
        let (unique, never_null) = tables[*table].enforces_key(ctx, names).await?;
        if unique && never_null {
            continue;
        }
        unique_by_label.push((label, unique));
        let t = &tables[*table];
        parts.push(KeyCountPart {
            label: label as i64,
            source: t.quoted_source_name(),
            columns: names.iter().map(|n| t.quote_key_column(n)).collect(),
            unique,
        });
    }
    if parts.is_empty() {
        return Ok(counts);
    }
    let first = &tables[keys[unique_by_label[0].0].0];
    let backend = first.backend;
    let stmt = first.key_count_statement(&parts);
    for row in rows(ctx.db, backend, &stmt).await? {
        let label = int_of(&row[0]).unwrap_or(0) as usize;
        let unique = unique_by_label
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, u)| *u)
            .unwrap_or(false);
        if label < counts.len() {
            counts[label] = KeyCount {
                rows: int_of(&row[1]),
                distinct: int_of(&row[2]),
                nulls: int_of(&row[3]).unwrap_or(0),
                enforced: unique,
                example: None,
            };
        }
    }
    for label in 0..counts.len() {
        if counts[label].repeats() <= 0 {
            continue;
        }
        let (table, names) = &keys[label];
        let stmt = tables[*table].repeated_key_statement(names);
        let found = rows(ctx.db, backend, &stmt).await?;
        if let Some(row) = found.first() {
            counts[label].example = Some(if names.len() == 1 {
                row[0].repr()
            } else {
                PyVal::Tuple(row.clone()).repr()
            });
        }
    }
    Ok(counts)
}

/// `Edge`, and its `repr`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub src_table: String,
    pub fkey: String,
    pub dst_table: String,
}

impl Edge {
    pub fn repr(&self) -> String {
        format!(
            "Edge(src_table={}, fkey={}, dst_table={})",
            str_repr(&self.src_table),
            str_repr(&self.fkey),
            str_repr(&self.dst_table)
        )
    }
}

/// The engine's `Graph`, as far as reading a source builds it.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub tables: Vec<EngineTable>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn settle_time_columns(&mut self) -> ER<()> {
        for table in &mut self.tables {
            table.settle_time_column(self.edges.iter().any(|edge| edge.dst_table == table.name))?;
        }
        Ok(())
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| t.name == name)
    }

    /// `graph[name]`.
    pub fn get(&self, name: &str) -> ER<usize> {
        self.index(name)
            .ok_or_else(|| PyExc::key_error(&format!("Table '{name}' not found in graph")))
    }

    /// `add_table(table)`.
    pub fn add_table(&mut self, table: EngineTable) -> ER<()> {
        if self.index(&table.name).is_some() {
            return Err(PyExc::key_error(&format!(
                "Cannot add table with name '{}' to this graph; table names must be globally unique.",
                table.name
            )));
        }
        self.tables.push(table);
        self.settle_time_columns()?;
        Ok(())
    }

    /// `remove_table(name)`.
    pub fn remove_table(&mut self, name: &str) -> ER<()> {
        let i = self.get(name)?;
        self.tables.remove(i);
        self.edges
            .retain(|e| e.src_table != name && e.dst_table != name);
        self.settle_time_columns()?;
        Ok(())
    }

    /// `link(src_table, fkey, dst_table)` with one column.
    pub fn link(&mut self, src: &str, fkey: &str, dst: &str) -> ER<()> {
        // `_resolve_foreign_key`: one column cannot name a row of a table
        // whose identity is spread across several.
        if let Some(i) = self.index(dst) {
            let destination = &self.tables[i];
            if destination.composite_key.is_some()
                && destination.primary_key.as_deref() != Some(fkey)
            {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot link '{src}' to '{dst}' on '{fkey}': '{dst}' identifies a row by \
                         {} together, so one column cannot name one of its rows. Reference \
                         every part of the identity.",
                        crate::py::str_list_repr(&destination.primary_key_columns())
                    ),
                );
            }
        }
        let edge = Edge {
            src_table: src.to_string(),
            fkey: fkey.to_string(),
            dst_table: dst.to_string(),
        };
        if self.edges.contains(&edge) {
            return raise(
                "ValueError",
                format!("{} already exists in the graph", edge.repr()),
            );
        }
        let Some(si) = self.index(src) else {
            return raise(
                "ValueError",
                format!("Source table '{src}' does not exist in the graph"),
            );
        };
        if self.index(dst).is_none() {
            return raise(
                "ValueError",
                format!("Destination table '{dst}' does not exist in the graph"),
            );
        }
        let Some(column) = self.tables[si].column(fkey) else {
            return raise(
                "ValueError",
                format!("Source key '{fkey}' does not exist as a column in source table '{src}'"),
            );
        };
        if !Stype::Id.supports_dtype(column.dtype) {
            return raise(
                "ValueError",
                format!(
                    "Cannot use '{fkey}' in source table '{src}' as a foreign key due to its incompatible data type. Foreign keys must have data type 'int', 'float' or 'string' (got '{}')",
                    column.dtype.value()
                ),
            );
        }
        self.tables[si]
            .column_mut(fkey)
            .expect("checked")
            .set_stype(Stype::Id)?;
        self.edges.push(edge);
        self.settle_time_columns()?;
        Ok(())
    }

    /// `unlink(src, fkey, dst)`.
    pub fn unlink(&mut self, edge: &Edge) -> ER<()> {
        let Some(i) = self.edges.iter().position(|e| e == edge) else {
            return raise(
                "ValueError",
                format!("{} is not present in the graph", edge.repr()),
            );
        };
        self.edges.remove(i);
        self.settle_time_columns()?;
        Ok(())
    }

    /// `_edge_error(edge)`.
    pub fn edge_error(&self, edge: &Edge) -> Option<String> {
        let src = &self.tables[self.index(&edge.src_table)?];
        let Some(src_key) = src.column(&edge.fkey) else {
            return Some(format!(
                "Edge {} is invalid since table '{}' no longer has a column '{}'. Remove the link with `unlink()` before removing the column.",
                edge.repr(),
                edge.src_table,
                edge.fkey
            ));
        };
        let dst = &self.tables[self.index(&edge.dst_table)?];
        let Some(dst_key) = dst.primary_key.as_deref().and_then(|k| dst.column(k)) else {
            return Some(format!(
                "Edge {} is invalid since table '{}' does not have a primary key. Add either a primary key or remove the link before proceeding.",
                edge.repr(),
                edge.dst_table
            ));
        };
        if src.primary_key.as_deref() == Some(edge.fkey.as_str()) {
            return Some(format!(
                "Cannot treat the primary key of table '{}' as a foreign key. Remove either the primary key or the link before proceeding.",
                edge.src_table
            ));
        }
        let number = |d: Dtype| d.is_int() || d.is_float();
        let (src_number, src_string) = (number(src_key.dtype), src_key.dtype.is_string());
        let (dst_number, dst_string) = (number(dst_key.dtype), dst_key.dtype.is_string());
        if !src_number && !src_string {
            return Some(format!(
                "{} is invalid as foreign key must be a number or string (got '{}')",
                edge.repr(),
                src_key.dtype.value()
            ));
        }
        if src_number != dst_number || src_string != dst_string {
            return Some(format!(
                "{} is invalid as foreign key '{}' and primary key '{}' have incompatible data types (got foreign key data type '{}' and primary key data type '{}')",
                edge.repr(),
                edge.fkey,
                dst_key.name,
                src_key.dtype.value(),
                dst_key.dtype.value()
            ));
        }
        None
    }

    /// `_drop_invalid_edges()`.
    pub fn drop_invalid_edges(&mut self) -> ER<Vec<String>> {
        let mut messages = Vec::new();
        for edge in self.edges.clone() {
            if let Some(reason) = self.edge_error(&edge) {
                self.unlink(&edge)?;
                messages.push(format!(
                    "Failed to add the relationship between '{}' and '{}' since {reason}",
                    edge.src_table, edge.dst_table
                ));
            }
        }
        Ok(messages)
    }

    /// `validate()`.
    pub fn validate(&self) -> ER<()> {
        if self.tables.is_empty() {
            return raise(
                "ValueError",
                "At least one table needs to be added to the graph",
            );
        }
        for edge in &self.edges {
            if let Some(reason) = self.edge_error(edge) {
                return raise("ValueError", reason);
            }
        }
        Ok(())
    }
}

/// A statement's parameters, as `Statement` carries them.
pub fn params(values: &[&str]) -> Option<Vec<Param>> {
    Some(
        values
            .iter()
            .map(|v| Param::Text((*v).to_string()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};

    use super::*;

    /// A table already read, with these columns, over SQLite.
    fn bare(name: &str, columns: &[(&str, Dtype, Stype)]) -> EngineTable {
        EngineTable {
            backend: Backend::Sqlite,
            name: name.to_string(),
            raw_source_name: name.to_string(),
            place: Place::default(),
            columns: columns
                .iter()
                .map(|(n, d, s)| EColumn {
                    name: n.to_string(),
                    expr: None,
                    dtype: *d,
                    stype: *s,
                    is_pk: false,
                    is_time: false,
                    is_end: false,
                })
                .collect(),
            primary_key: None,
            time_column: None,
            end_time_column: None,
            expr_sample: SampleFrame::empty_index(0),
            time_choice: None,
            time_inferred: false,
            source_columns: Some(
                columns
                    .iter()
                    .map(|(n, d, _)| SourceColumn {
                        name: n.to_string(),
                        dtype: Some(*d),
                        is_primary_key: false,
                        is_unique_key: false,
                        is_nullable: true,
                    })
                    .collect(),
            ),
            source_fks: Some(Vec::new()),
            source_sample: None,
            unique_constraints: Vec::new(),
            spelled: BTreeMap::new(),
            sample_source: None,
            listed: None,
            is_hybrid: None,
            composite_key: None,
        }
    }

    fn message<T: std::fmt::Debug>(result: ER<T>) -> String {
        match result {
            Err(e) => format!("{}: {}", e.class, e.message),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    fn orders() -> EngineTable {
        bare(
            "orders",
            &[
                ("id", Dtype::Int, Stype::Numerical),
                ("user_id", Dtype::Int, Stype::Numerical),
                ("at", Dtype::Date, Stype::Timestamp),
                ("tags", Dtype::Stringlist, Stype::Multicategorical),
            ],
        )
    }

    #[test]
    fn upstream_session_columns_are_qualified_before_and_after_filtering() {
        let mut table = bare(
            "events",
            &[
                ("USER", Dtype::String, Stype::Text),
                ("id", Dtype::Int, Stype::Id),
            ],
        );
        table.backend = Backend::Databricks;
        table.place.catalog = Some("cat".into());
        table.place.schema = Some("sch".into());
        assert_eq!(
            table.quoted_source_name(),
            "(SELECT source.`USER`, source.`id` FROM `cat`.`sch`.`events` AS source)"
        );
        table.set_row_filter(Some("id > 1".into())).unwrap();
        assert_eq!(
            table.quoted_source_name(),
            "(SELECT source.`USER`, source.`id` FROM `cat`.`sch`.`events` AS source WHERE (id > 1))"
        );
    }

    #[test]
    fn upstream_filter_cannot_change_after_rows_are_cached() {
        let mut table = orders();
        table.set_row_filter(Some("id > 1".into())).unwrap();
        table.source_sample = Some(SampleFrame::empty_index(0));
        assert_eq!(
            message(table.set_row_filter(None)),
            "ValueError: Cannot change the row filter of table 'orders' since its rows have already been read"
        );
        assert_eq!(table.place.row_filter.as_deref(), Some("id > 1"));
    }

    #[test]
    fn upstream_edit_choice_survives_cache_and_refused_setters_then_follows_links() {
        let mut table = orders();
        table.set_primary_key(Some("id")).unwrap();
        table.set_time_column(Some("at")).unwrap();
        table.time_choice = Some(crate::time_columns::Choice {
            column: Some("at".into()),
            how: Some("edit_stamp".into()),
            among: vec!["at".into()],
            fallback: Some("at".into()),
            ..Default::default()
        });
        table.time_inferred = true;
        assert!(table.set_time_column(Some("absent")).is_err());
        assert!(table.time_inferred);
        assert_eq!(table.time_column.as_deref(), Some("at"));
        let recovered = EngineTable::from_json(&table.to_json()).unwrap();
        assert_eq!(recovered.time_choice, table.time_choice);
        assert!(recovered.time_inferred);
        let mut graph = Graph {
            tables: vec![
                recovered,
                bare("children", &[("order_id", Dtype::Int, Stype::Id)]),
            ],
            edges: vec![],
        };
        graph.link("children", "order_id", "orders").unwrap();
        assert_eq!(graph.tables[0].time_column, None);
        let edge = graph.edges[0].clone();
        graph.unlink(&edge).unwrap();
        assert_eq!(graph.tables[0].time_column.as_deref(), Some("at"));
        graph.tables[0].set_time_column(None).unwrap();
        graph.link("children", "order_id", "orders").unwrap();
        graph.unlink(&edge).unwrap();
        assert_eq!(graph.tables[0].time_column, None);
    }

    #[test]
    fn keys_and_times_refuse_each_other_as_the_engine_does() {
        let mut t = orders();
        t.set_time_column(Some("at")).unwrap();
        assert_eq!(
            message(t.set_primary_key(Some("at"))),
            "ValueError: Cannot specify column 'at' as a primary key since it is already defined to be a time column"
        );
        assert_eq!(
            message(t.set_end_time_column(Some("at"))),
            "ValueError: Cannot specify column 'at' as an end time column since it is already defined to be a time column"
        );
        t.set_primary_key(Some("id")).unwrap();
        assert_eq!(t.column("id").map(|c| c.stype), Some(Stype::Id));
        assert_eq!(
            message(t.columns[0].set_stype(Stype::Numerical)),
            "ValueError: Primary key 'id' must have 'ID' semantic type (got 'numerical')"
        );
        assert_eq!(
            message(t.set_primary_key(Some("nope"))),
            "KeyError: \"Column 'nope' not found in table 'orders'\""
        );
        assert_eq!(
            message(t.set_primary_key(Some("tags"))),
            "ValueError: Column 'tags' received an incompatible semantic type (got dtype='stringlist' and stype='ID')"
        );
    }

    #[test]
    fn removing_a_column_lets_go_of_its_roles() {
        let mut t = orders();
        t.set_primary_key(Some("id")).unwrap();
        let (mut t, asked) = answering(Vec::new(), |db| async move {
            let host = Host::assumed();
            let mut ctx = Ctx {
                db: &db,
                host: &host,
                warnings: Warnings::default(),
            };
            t.add_columns(&mut ctx, Vec::new()).await.unwrap();
            t
        });
        assert!(asked.is_empty());
        t.columns.push(EColumn {
            name: "until".into(),
            expr: None,
            dtype: Dtype::Date,
            stype: Stype::Timestamp,
            is_pk: false,
            is_time: false,
            is_end: false,
        });
        t.set_end_time_column(Some("until")).unwrap();
        t.remove_column("until").unwrap();
        t.remove_column("id").unwrap();
        assert_eq!(
            (t.primary_key.clone(), t.end_time_column.clone()),
            (None, None)
        );
        assert_eq!(
            message(t.remove_column("id")),
            "KeyError: \"Column 'id' not found in table 'orders'\""
        );
    }

    /// What `body` gives over a [`Db`] that answers each statement it asks
    /// with the next of `batches`; and the statements it asked.
    fn answering<T, F>(batches: Vec<RecordBatch>, body: impl FnOnce(Db) -> F) -> (T, Vec<String>)
    where
        T: Send + 'static,
        F: std::future::Future<Output = T> + Send + 'static,
    {
        let mut run = sdm_protocol::AsyncRun::new(|asker| {
            let body = body(Db(asker));
            async move { Ok(body.await) }
        });
        let mut batches = batches.into_iter();
        let mut asked = Vec::new();
        loop {
            match sdm_protocol::Run::next(&mut run).unwrap() {
                sdm_protocol::Step::Done(value) => return (value, asked),
                sdm_protocol::Step::CallModel(_) => panic!("a schema run calls no model"),
                sdm_protocol::Step::RunQueries(queries) => {
                    for query in queries {
                        asked.push(query.sql);
                        let batch = batches.next().expect("an answer for every statement");
                        let outcome = Outcome::Data(sdm_protocol::QueryResult::from_batch(batch));
                        sdm_protocol::Run::submit(&mut run, query.id, outcome).unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn a_graph_links_validates_and_drops_as_the_engine_does() {
        let mut users = bare("users", &[("id", Dtype::String, Stype::Text)]);
        users.set_primary_key(Some("id")).unwrap();
        let mut g = Graph::default();
        g.add_table(orders()).unwrap();
        g.add_table(users).unwrap();
        assert_eq!(
            message(g.add_table(orders())),
            "KeyError: \"Cannot add table with name 'orders' to this graph; table names must be globally unique.\""
        );
        assert_eq!(
            message(g.link("nope", "user_id", "users")),
            "ValueError: Source table 'nope' does not exist in the graph"
        );
        assert_eq!(
            message(g.link("orders", "tags", "users")),
            "ValueError: Cannot use 'tags' in source table 'orders' as a foreign key due to its incompatible data type. Foreign keys must have data type 'int', 'float' or 'string' (got 'stringlist')"
        );
        g.link("orders", "user_id", "users").unwrap();
        assert_eq!(
            message(g.link("orders", "user_id", "users")),
            "ValueError: Edge(src_table='orders', fkey='user_id', dst_table='users') already exists in the graph"
        );
        assert_eq!(
            message(g.validate()),
            "ValueError: Edge(src_table='orders', fkey='user_id', dst_table='users') is invalid as foreign key 'user_id' and primary key 'id' have incompatible data types (got foreign key data type 'int' and primary key data type 'string')"
        );
        let dropped = g.drop_invalid_edges().unwrap();
        assert_eq!(dropped.len(), 1);
        assert!(dropped[0].starts_with(
            "Failed to add the relationship between 'orders' and 'users' since Edge("
        ));
        g.validate().unwrap();
        g.remove_table("users").unwrap();
        assert_eq!(g.tables.len(), 1);
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns.into_iter().map(|(_, a)| a).collect(),
        )
        .unwrap()
    }

    #[test]
    fn counts_keys_in_one_statement_and_reads_an_example_of_a_repeat() {
        let counts = batch(vec![
            ("label", Arc::new(Int64Array::from(vec![0, 1]))),
            ("rows", Arc::new(Int64Array::from(vec![10, 4]))),
            ("distinct", Arc::new(Int64Array::from(vec![10, 2]))),
            ("nulls", Arc::new(Int64Array::from(vec![0, 1]))),
        ]);
        let example = batch(vec![("k", Arc::new(StringArray::from(vec!["a"])))]);
        let (found, asked) = answering(vec![counts, example], |db| async move {
            let host = Host::assumed();
            let mut ctx = Ctx {
                db: &db,
                host: &host,
                warnings: Warnings::default(),
            };
            let mut tables = vec![orders(), bare("users", &[("id", Dtype::String, Stype::Id)])];
            count_keys(
                &mut ctx,
                &mut tables,
                &[(0, vec!["id".to_string()]), (1, vec!["id".to_string()])],
            )
            .await
            .unwrap()
        });
        assert_eq!((found[0].repeats(), found[1].repeats()), (0, 1));
        assert_eq!(found[1].example.as_deref(), Some("'a'"));
        assert_eq!(found[1].nulls, 1);
        assert_eq!(asked.len(), 2);
        assert!(asked[0].contains("UNION ALL"));
    }

    #[test]
    fn reads_each_catalogs_spelling_of_a_type() {
        assert_eq!(duckdb_dtype("DECIMAL(18,0)"), Some(Dtype::Int));
        assert_eq!(duckdb_dtype("INTEGER[]"), Some(Dtype::Intlist));
        assert_eq!(snow_dtype(Some("NUMBER(38,2)")), Some(Dtype::Float));
        assert_eq!(snow_dtype(Some("ARRAY(VARCHAR)")), Some(Dtype::Stringlist));
        assert_eq!(databricks_dtype("decimal(10,0)"), Some(Dtype::Int));
        assert_eq!(databricks_dtype("array<bigint>"), Some(Dtype::Intlist));
        assert_eq!(
            databricks_dtype("map<string,int>"),
            Some(Dtype::Unsupported)
        );
        assert_eq!(
            sqlite_dtype(&PyVal::Str("VARCHAR(20)".into())),
            Some(Dtype::String)
        );
        assert_eq!(
            postgres_dtype("numeric", &PyVal::None, &PyVal::Int(0)),
            Some(Dtype::Int)
        );
    }
}
