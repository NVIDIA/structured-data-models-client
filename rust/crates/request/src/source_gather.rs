// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The graph a relational request is gathered from over tables that stay
//! in their source (`RelationalMaterializer._build_graph` over a source,
//! `EngineTables.graph`), and building the SQL sampler over it
//! (`KumoRelational.__init__`, `SQLSampler.__init__` and each backend's
//! sampler): the keys its source enforces, the row counts, the missing
//! indices, and what its statements depend on.

use sdm_dialects::py::Py;
use sdm_pql::types::Stype;
use sdm_sampling::driver::inputs::{
    Backend as SBackend, ColumnInfo, Inputs, SourceColumnInfo, SqlTable, State, TableInfo,
};
use sdm_sampling::driver::output::Warning;
use sdm_sampling::driver::sampler::Sampler;
use sdm_schema::client::{Key, Schema};
use sdm_schema::dt::Host;
use sdm_schema::engine::build::Namespace;
use sdm_schema::engine::{Backend, Ctx, Graph, PyExc};
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value};

use crate::schemas::SourceSchema;

/// `RESERVED_COLUMN_PREFIX` of the SQL backends.
const RESERVED_COLUMN_PREFIX: &str = "__kumo_";

/// Why the graph over a source was not built.
pub enum Refused {
    /// The engine raised.
    Engine(PyExc),
    /// The client refused, as it says.
    Client(sdm_protocol::Diagnostic),
}

impl From<PyExc> for Refused {
    fn from(error: PyExc) -> Self {
        Refused::Engine(error)
    }
}

fn namespace(handle: &SourceSchema, backend: Backend) -> Namespace {
    let space = handle.source.get("namespace");
    let get = |k: &str| {
        space
            .and_then(|n| n.get(k))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    Namespace {
        catalog: if backend == Backend::Snow {
            get("database")
        } else {
            get("catalog")
        },
        schema: get("schema"),
    }
}

/// `EngineTables.graph(source, tables)`: a fresh graph, without edges, over
/// copies of the tables kept, reading any not kept yet.
async fn engine_tables_graph(
    ctx: &mut Ctx<'_>,
    handle: &SourceSchema,
    schema: &Schema,
) -> Result<Graph, Refused> {
    let backend = handle.backend().map_err(|e| PyExc::new("ValueError", e))?;
    let wanted: Vec<(String, String, Vec<String>)> = schema
        .tables
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                t.source_name.clone().unwrap_or_else(|| t.name.clone()),
                t.columns.iter().map(|c| c.name.clone()).collect(),
            )
        })
        .collect();
    let mut kept = handle.engine.clone();
    let missing: Vec<(String, String)> = wanted
        .iter()
        .filter(|(_, source_name, _)| !kept.iter().any(|(n, _)| n == source_name))
        .map(|(_, source_name, _)| (source_name.clone(), source_name.clone()))
        .collect();
    if !missing.is_empty() {
        let semantic = handle
            .source
            .get("semantic_model")
            .is_some_and(|m| !m.is_null());
        if semantic {
            // `semantic_model_graph(source)`: the model read again, as a
            // schema built by hand has read nothing of it; its tables are
            // kept by the names the model gives them.
            let info = sdm_schema::source::SourceInfo {
                backend,
                namespace: handle
                    .source
                    .get("namespace")
                    .and_then(Value::as_object)
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
                semantic_model: handle
                    .source
                    .get("semantic_model")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                trust_expressions: handle
                    .source
                    .get("trust_expressions")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            };
            let read = sdm_schema::engine::views::semantic_model_graph(ctx, &info).await?;
            for table in read.tables {
                if !kept.iter().any(|(n, _)| *n == table.name) {
                    kept.push((table.name.clone(), table));
                }
            }
        } else {
            let read = sdm_schema::engine::build::read_graph_as_described(
                ctx,
                backend,
                &namespace(handle, backend),
                Some(&missing),
            )
            .await;
            let read = match read {
                Ok(read) => read,
                Err(e) if e.class == "NoTablesFoundError" => {
                    return Err(Refused::Client(crate::error::invalid(format!(
                        "{} holds no table to describe",
                        source_repr(handle)
                    ))));
                }
                Err(e) if e.class == "TableNotFoundError" => {
                    return Err(Refused::Client(crate::error::invalid(e.message)));
                }
                Err(e) => return Err(Refused::Engine(e)),
            };
            for (name, _) in &missing {
                let i = read.get(name)?;
                if !kept.iter().any(|(n, _)| n == name) {
                    kept.push((name.clone(), read.tables[i].clone()));
                }
            }
        }
    }
    let mut graph = Graph::default();
    for (name, source_name, columns) in &wanted {
        let Some((_, table)) = kept.iter().find(|(n, _)| n == source_name) else {
            return Err(Refused::Engine(PyExc::key_error(source_name)));
        };
        graph.add_table(table.copy_for_one_gather(name, columns)?)?;
    }
    Ok(graph)
}

/// `repr(source)`, as the client writes it in a refusal.
fn source_repr(handle: &SourceSchema) -> String {
    let info = sdm_schema::binding::source_args(
        handle
            .source
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or("sqlite"),
        &handle.source,
        &serde_json::json!({}),
    );
    match info {
        Ok((info, _)) => info.repr(),
        Err(_) => "the source".to_string(),
    }
}

/// `_build_graph()` over a source: every key, time column and semantic type
/// written from the schema, and every relationship linked, once.
pub async fn build_graph(
    ctx: &mut Ctx<'_>,
    handle: &SourceSchema,
    schema: &Schema,
) -> Result<Graph, Refused> {
    let mut graph = engine_tables_graph(ctx, handle, schema).await?;
    for table in &schema.tables {
        let i = graph.get(&table.name)?;
        match &table.primary_key {
            Some(Key::Many(names)) if names.len() > 1 => {
                let names = names.clone();
                graph.tables[i]
                    .set_composite_primary_key(ctx, &names)
                    .await?;
            }
            _ => {}
        }
        let engine = &mut graph.tables[i];
        match &table.primary_key {
            None => engine.set_primary_key(None)?,
            Some(Key::One(name)) => engine.set_primary_key(Some(name))?,
            Some(Key::Many(names)) if names.len() == 1 => {
                engine.set_primary_key(Some(&names[0]))?;
            }
            Some(Key::Many(names)) if names.is_empty() => {
                return Err(Refused::Engine(PyExc::new(
                    "ValueError",
                    format!(
                        "Cannot use an empty composite primary key on table '{}'",
                        table.name
                    ),
                )));
            }
            Some(Key::Many(_)) => {}
        }
        engine.set_time_column(table.time_column.as_deref())?;
        for column in &table.columns {
            if engine.has_column(&column.name) {
                let stype = Stype::parse(&column.stype).ok_or_else(|| {
                    PyExc::new(
                        "ValueError",
                        format!("'{}' is not a valid Stype", column.stype),
                    )
                })?;
                engine.set_column_stype(&column.name, stype)?;
            }
        }
    }
    let mut linked: Vec<(String, String, String)> = Vec::new();
    for link in &schema.relationships {
        let edge = (
            link.source_table.clone(),
            link.source_column.clone(),
            link.target_table.clone(),
        );
        if linked.contains(&edge) {
            continue;
        }
        linked.push(edge);
        graph.link(&link.source_table, &link.source_column, &link.target_table)?;
    }
    let misdirected: Vec<String> = schema
        .relationships
        .iter()
        .filter(|link| {
            let key = schema
                .tables
                .iter()
                .find(|t| t.name == link.target_table)
                .map(|t| sdm_schema::client::key_names_of(&t.primary_key))
                .unwrap_or_default();
            !key.contains(&link.target_column)
        })
        .map(|link| {
            format!(
                "{}.{} -> {}.{}",
                link.source_table, link.source_column, link.target_table, link.target_column
            )
        })
        .collect();
    if !misdirected.is_empty() {
        return Err(Refused::Client(crate::error::invalid(format!(
            "{} name a target column that is not the primary key of the table they point at, \
             and an edge can only join to that",
            misdirected.join(", ")
        ))));
    }
    if graph.edges.len() != linked.len() {
        return Err(Refused::Client(crate::error::structured(
            "INTERNAL_ERROR",
            "the graph built for this request does not match the schema",
        )));
    }
    Ok(graph)
}

/// `_reject_reserved_columns(graph)`, over the source's own columns, as the
/// sampler's (`driver/sql.rs`) reads them: a composite key's derived column
/// is the client's.
fn reject_reserved_columns(graph: &Graph) -> Result<(), PyExc> {
    let mut collisions: Vec<String> = graph
        .tables
        .iter()
        .flat_map(|t| {
            t.columns
                .iter()
                .filter(|c| c.expr.is_none() && c.name.starts_with(RESERVED_COLUMN_PREFIX))
                .map(move |c| format!("{}.{}", t.name, c.name))
        })
        .collect();
    if collisions.is_empty() {
        return Ok(());
    }
    collisions.sort();
    Err(PyExc::new(
        "ValueError",
        format!(
            "{} {} reserved by the SQL backends: a column may not start with {}, because the \
             generated joins already select columns of that name and the reference would be \
             ambiguous. Rename the column, or project it under another name.",
            collisions.join(", "),
            if collisions.len() > 1 {
                "use names"
            } else {
                "uses a name"
            },
            sdm_pql::pyrepr::str_repr(RESERVED_COLUMN_PREFIX)
        ),
    ))
}

fn sampler_backend(backend: Backend) -> SBackend {
    match backend {
        Backend::Sqlite => SBackend::Sqlite,
        Backend::Duckdb => SBackend::Duckdb,
        Backend::Postgres => SBackend::Postgres,
        Backend::Snow => SBackend::Snowflake,
        Backend::Databricks => SBackend::Databricks,
    }
}

/// A listed value as the dialects take Python values.
fn py_of(value: &PyVal) -> Py {
    match value {
        PyVal::None => Py::None,
        PyVal::Bool(b) => Py::Bool(*b),
        PyVal::Int(i) => Py::Int(sdm_dialects::int::Int::from_i128(*i)),
        PyVal::BigInt(text) => sdm_dialects::int::Int::parse(text).map_or(Py::None, Py::Int),
        PyVal::Float(f) => Py::Float(*f),
        PyVal::Str(s) => Py::Str(s.clone()),
        PyVal::Decimal(text) => Py::Decimal(text.clone()),
        PyVal::Na => Py::Na,
        PyVal::NaT => Py::NaT,
        other => Py::Str(other.str_()),
    }
}

/// `KumoRelational(graph)` over a source: the graph validated, and the SQL
/// sampler built, with the warnings building it gave.
pub async fn sampler(
    ctx: &mut Ctx<'_>,
    asker: &sdm_protocol::Asker,
    mut graph: Graph,
    host: &Host,
) -> Result<(Sampler, Inputs, Vec<Warning>), PyExc> {
    graph.validate()?;
    let backend = graph.tables.first().map(|t| t.backend).ok_or_else(|| {
        PyExc::new(
            "ValueError",
            "At least one table needs to be added to the graph",
        )
    })?;
    reject_reserved_columns(&graph)?;
    // The keys the source enforces.
    let keyed: Vec<(usize, Vec<String>)> = graph
        .tables
        .iter()
        .enumerate()
        .filter(|(_, t)| t.primary_key.is_some())
        .map(|(i, t)| (i, t.primary_key_columns()))
        .collect();
    sdm_schema::engine::before_counting_keys(ctx, &mut graph.tables, &keyed).await?;
    let mut enforced: Vec<String> = Vec::new();
    for (i, names) in &keyed {
        if graph.tables[*i].enforces_key(ctx, names).await?.0 {
            enforced.push(graph.tables[*i].name.clone());
        }
    }
    enforced.sort();
    // What each table's statements read.
    let mut tables = Vec::new();
    for table in &mut graph.tables {
        let source_columns = table.source_column_dict(ctx).await?.to_vec();
        let mut described = Vec::new();
        for column in &table.columns {
            if !column.is_source() {
                continue;
            }
            let Some(source) = source_columns.iter().find(|c| c.name == column.name) else {
                return Err(PyExc::key_error(&column.name));
            };
            described.push(SourceColumnInfo {
                name: source.name.clone(),
                dtype: source.dtype.map(|d| d.value().to_string()),
                is_primary_key: source.is_primary_key,
                is_unique_key: source.is_unique_key,
                is_nullable: source.is_nullable,
            });
        }
        let mut sql = SqlTable {
            source_name: table.quoted_source_name(),
            unquoted_name: Some(table.source_name()),
            source_columns: described,
            ..SqlTable::default()
        };
        match backend {
            Backend::Duckdb => {
                sql.zoned = table
                    .columns
                    .iter()
                    .filter(|c| table.is_zoned_time(&c.name))
                    .map(|c| c.name.clone())
                    .collect();
            }
            Backend::Postgres => {
                sql.postgres_types = table
                    .columns
                    .iter()
                    .map(|c| {
                        (
                            c.name.clone(),
                            table.postgres_type(&c.name).unwrap_or_else(|| {
                                sdm_dialects::postgres::element_type(
                                    sdm_dialects::py::Dtype::parse(c.dtype.value())
                                        .unwrap_or(sdm_dialects::py::Dtype::String),
                                )
                                .to_string()
                            }),
                        )
                    })
                    .collect();
            }
            Backend::Snow => {
                sql.snowflake_types = table.snowflake_types().into_iter().collect();
                sql.listed = table.listed_rows().map(|rows| {
                    Py::List(
                        rows.iter()
                            .map(|row| {
                                Py::Dict(row.iter().map(|(k, v)| (k.clone(), py_of(v))).collect())
                            })
                            .collect(),
                    )
                    .encode()
                    .unwrap_or(Value::Null)
                });
                sql.identity = Some((
                    table.place.catalog.clone().unwrap_or_default(),
                    table.place.schema.clone().unwrap_or_default(),
                    table.raw_source_name.clone(),
                ));
            }
            _ => {}
        }
        tables.push(TableInfo {
            name: table.name.clone(),
            primary_key: table.primary_key.clone(),
            time_column: table.time_column.clone(),
            end_time_column: table.end_time_column.clone(),
            columns: table
                .columns
                .iter()
                .map(|c| ColumnInfo {
                    name: c.name.clone(),
                    dtype: Some(c.dtype.value().to_string()),
                    stype: Some(c.stype.value().to_string()),
                    expr: c.expr.clone(),
                    is_source: c.is_source(),
                })
                .collect(),
            sql: Some(sql),
        });
    }
    // Each backend's sampler.
    let mut state = State {
        enforced_key_tables: enforced,
        ..State::default()
    };
    if matches!(
        backend,
        Backend::Postgres | Backend::Databricks | Backend::Snow
    ) {
        for table in &mut graph.tables {
            let n = table.num_rows(ctx).await?;
            state.num_rows.insert(table.name.clone(), n);
        }
        if backend == Backend::Snow {
            // Counting read each table's listing where it was not read yet.
            for (info, table) in tables.iter_mut().zip(&graph.tables) {
                if let Some(sql) = info.sql.as_mut()
                    && sql.listed.is_none()
                {
                    sql.listed = table.listed_rows().map(|rows| {
                        Py::List(
                            rows.iter()
                                .map(|row| {
                                    Py::Dict(
                                        row.iter().map(|(k, v)| (k.clone(), py_of(v))).collect(),
                                    )
                                })
                                .collect(),
                        )
                        .encode()
                        .unwrap_or(Value::Null)
                    });
                }
            }
        }
    }
    let inputs = Inputs {
        backend: sampler_backend(backend),
        tables,
        edges: graph
            .edges
            .iter()
            .map(|e| (e.src_table.clone(), e.fkey.clone(), e.dst_table.clone()))
            .collect(),
        state,
        arguments: Map::new(),
        data: Vec::new(),
        argument_tables: Default::default(),
    };
    let sampler = Sampler::new(&inputs, host.clone()).map_err(raised)?;
    let db = sdm_sampling::driver::Db::new(asker.clone());
    let (_, warning) = sdm_sampling::driver::init::missing_indices(&sampler, &inputs, &db, false)
        .await
        .map_err(raised)?;
    Ok((sampler, inputs, warning.into_iter().collect()))
}

/// A sampling error as the exception today's code raises.
fn raised(error: sdm_sampling::driver::Raised) -> PyExc {
    let message = if error.class == "KeyError" {
        sdm_pql::pyrepr::str_repr(&error.message)
    } else {
        error.message
    };
    PyExc::new(&error.class, message).caused_by(error.chain)
}
