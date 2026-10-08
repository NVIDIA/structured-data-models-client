// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Graph.from_sqlite`, `from_duckdb`, `from_postgres`, `from_snowflake` and
//! `from_databricks`, as `sdm_client`'s `engine_graph` calls them: over a
//! source, without edges.

use super::{Backend, Ctx, ER, EngineTable, Graph, Place, PyExc, TableSpec, rows, text_of};
use crate::pyval::PyVal;

/// The engine's backend names, as `_reraise_as_graph_error` writes them.
fn engine_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Sqlite => "SQLite",
        Backend::Duckdb => "DuckDB",
        Backend::Postgres => "PostgreSQL",
        Backend::Snow => "Snowflake",
        Backend::Databricks => "Databricks",
    }
}

/// A source's namespace: Databricks' catalog or Snowflake's database, and
/// the schema.
#[derive(Clone, Debug, Default)]
pub struct Namespace {
    pub catalog: Option<String>,
    pub schema: Option<String>,
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty())
}

async fn first_row(
    ctx: &mut Ctx<'_>,
    backend: Backend,
    statement: &sdm_dialects::Statement,
) -> ER<Vec<PyVal>> {
    let found = rows(ctx.db, backend, statement).await?;
    match found.into_iter().next() {
        Some(row) => Ok(row),
        None => Err(PyExc::new("IndexError", "list index out of range")),
    }
}

fn optional_text(value: Option<&PyVal>) -> Option<String> {
    match value {
        None | Some(PyVal::None) => None,
        Some(v) => Some(text_of(v)),
    }
}

/// The tables of a graph read from `backend`: `tables` as (name, name in the
/// source), or every table the namespace lists.
async fn construct(
    ctx: &mut Ctx<'_>,
    backend: Backend,
    namespace: &Namespace,
    tables: Option<&[(String, String)]>,
) -> ER<Graph> {
    let mut catalog = namespace.catalog.clone();
    let mut schema = namespace.schema.clone();
    match backend {
        Backend::Snow | Backend::Databricks if catalog.is_none() || schema.is_none() => {
            let statement = if backend == Backend::Snow {
                sdm_dialects::snow::current_namespace()
            } else {
                sdm_dialects::databricks::current_namespace()
            };
            let row = first_row(ctx, backend, &statement).await?;
            catalog = non_empty(catalog).or_else(|| optional_text(row.first()));
            if catalog.is_none() {
                return Err(PyExc::new("AssertionError", ""));
            }
            schema = non_empty(schema).or_else(|| optional_text(row.get(1)));
        }
        Backend::Postgres => {
            if schema.is_none() {
                let row =
                    first_row(ctx, backend, &sdm_dialects::postgres::current_schema()).await?;
                schema = optional_text(row.first());
            }
            if schema.is_none() {
                return Err(PyExc::new(
                    "ValueError",
                    "No current 'schema' set. Please specify the PostgreSQL schema manually",
                ));
            }
        }
        _ => {}
    }
    let named: Vec<(String, Option<String>)> = match tables {
        Some(given) => given
            .iter()
            .map(|(n, s)| (n.clone(), Some(s.clone())))
            .collect(),
        None => {
            let (statement, place) = match backend {
                Backend::Sqlite => (
                    sdm_dialects::sqlite::list_tables(),
                    "the SQLite database".to_string(),
                ),
                Backend::Duckdb => (
                    sdm_dialects::duckdb::list_tables(),
                    "the DuckDB database".to_string(),
                ),
                Backend::Postgres => {
                    let s = schema.clone().unwrap_or_default();
                    (
                        sdm_dialects::postgres::list_tables(&s),
                        format!("schema '{s}'"),
                    )
                }
                Backend::Snow | Backend::Databricks => {
                    let Some(s) = schema.clone() else {
                        let which = if backend == Backend::Snow {
                            "Snowflake"
                        } else {
                            "Databricks"
                        };
                        return Err(PyExc::new(
                            "ValueError",
                            format!(
                                "No current 'schema' set. Please specify the {which} schema manually"
                            ),
                        ));
                    };
                    let c = catalog.clone().unwrap_or_default();
                    if backend == Backend::Snow {
                        (
                            sdm_dialects::snow::list_tables(&c, &s),
                            format!("schema '{s}' of database '{c}'"),
                        )
                    } else {
                        (
                            sdm_dialects::databricks::list_tables(
                                &Backend::Databricks.quote(&c),
                                &s,
                            ),
                            format!("schema '{s}' of catalog '{c}'"),
                        )
                    }
                }
            };
            let listed = rows(ctx.db, backend, &statement).await?;
            if listed.is_empty() {
                return Err(PyExc::new(
                    "NoTablesFoundError",
                    format!(
                        "No tables found in {place}. Check the name, or pass `tables=[...]` explicitly."
                    ),
                ));
            }
            listed.iter().map(|r| (text_of(&r[0]), None)).collect()
        }
    };
    let place = Place {
        catalog,
        schema,
        row_filter: None,
    };
    let mut graph = Graph::default();
    let mut built = Vec::new();
    for (name, source_name) in &named {
        let table = EngineTable::new(
            ctx,
            backend,
            TableSpec::named(name, source_name.as_deref(), place.clone()),
        )
        .await?;
        built.push(table);
    }
    for table in built {
        graph.add_table(table)?;
    }
    Ok(graph)
}

/// Whether `_reraise_as_graph_error` lets an error through unchanged: the
/// engine's own, and built-ins.
fn is_own_or_builtin(class: &str) -> bool {
    matches!(
        class,
        "ValueError"
            | "KeyError"
            | "RuntimeError"
            | "TypeError"
            | "IndexError"
            | "AssertionError"
            | "TableNotFoundError"
            | "MetricViewAsTableError"
            | "NoTablesFoundError"
            | "UnsupportedJoinError"
            | "UnsupportedFilterError"
            | "GraphConstructionError"
    )
}

/// `engine_graph(source, tables, infer_metadata=False)`'s graph: the tables
/// as their source describes them, nothing inferred.
pub async fn read_graph_as_described(
    ctx: &mut Ctx<'_>,
    backend: Backend,
    namespace: &Namespace,
    tables: Option<&[(String, String)]>,
) -> ER<Graph> {
    match construct(ctx, backend, namespace, tables).await {
        Ok(graph) => Ok(graph),
        Err(e) if !is_own_or_builtin(&e.class) => Err(PyExc::new(
            "GraphConstructionError",
            format!(
                "Reading the {} source failed while building the graph: {}: {}",
                engine_name(backend),
                e.class,
                e.message
            ),
        )),
        Err(other) => Err(other),
    }
}

/// `engine_graph(source, tables, infer_metadata=True)`'s graph, before
/// `sdm_client` translates the engine's refusals.
pub async fn read_graph(
    ctx: &mut Ctx<'_>,
    backend: Backend,
    namespace: &Namespace,
    tables: Option<&[(String, String)]>,
) -> ER<Graph> {
    let mut graph = match construct(ctx, backend, namespace, tables).await {
        Ok(graph) => graph,
        Err(e) if !is_own_or_builtin(&e.class) => {
            return Err(PyExc::new(
                "GraphConstructionError",
                format!(
                    "Reading the {} source failed while building the graph: {}: {}",
                    engine_name(backend),
                    e.class,
                    e.message
                ),
            ));
        }
        Err(other) => return Err(other),
    };
    for i in 0..graph.tables.len() {
        graph.tables[i].infer_metadata(ctx).await?;
    }
    graph.settle_time_columns()?;
    Ok(graph)
}
