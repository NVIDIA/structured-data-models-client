// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Schema.from_source`: tables described from what the engine reads of
//! them in their source, keyed, counted and linked as `sdm_client` does.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::client::{
    self, Col, Key, NamedKeys, Refusal, Rel, SDM_WARNING, Said, Schema, Table,
    a_key_missing_values, apply_keys_and_times, check_name, not_unique_enough,
    tables_naming_each_column, too_many_repeats, warn_a_key_has_no_value, warn_a_key_repeats,
    warn_a_key_was_declined,
};
use crate::engine::build::{Namespace, read_graph};
use crate::engine::{self, Backend, Ctx, EngineTable, Graph, PyExc};
use crate::py::str_repr;

impl From<PyExc> for Refusal {
    fn from(e: PyExc) -> Self {
        Refusal::class(&e.class, e.message).caused_by(e.chain)
    }
}

type H<T> = Result<T, Refusal>;

/// What `schema_from_source` is told about the source.
#[derive(Clone, Debug)]
pub struct SourceInfo {
    pub backend: Backend,
    /// The namespace, in order.
    pub namespace: Vec<(String, Value)>,
    pub semantic_model: Option<String>,
    pub trust_expressions: bool,
}

impl SourceInfo {
    fn get(&self, key: &str) -> Option<String> {
        self.namespace
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_str().map(str::to_string))
    }

    /// The namespace as the engine reads it.
    pub fn engine_namespace(&self) -> Namespace {
        Namespace {
            catalog: match self.backend {
                Backend::Snow => self.get("database"),
                _ => self.get("catalog"),
            },
            schema: self.get("schema"),
        }
    }

    /// `repr(source)`.
    pub fn repr(&self) -> String {
        let class = match self.backend {
            Backend::Databricks => "DatabricksSource",
            Backend::Snow => "SnowflakeSource",
            Backend::Postgres => "PostgresSource",
            Backend::Duckdb => "DuckDBSource",
            Backend::Sqlite => "SQLiteSource",
        };
        let parts: Vec<String> = self
            .namespace
            .iter()
            .map(|(k, v)| format!("{k}={}", client::json_repr(v)))
            .collect();
        let source = format!("{class}({})", parts.join(", "));
        match &self.semantic_model {
            None => source,
            Some(name) => format!(
                "SemanticModel({} in {source}{})",
                str_repr(name),
                if self.trust_expressions {
                    ", trust_expressions=True"
                } else {
                    ""
                }
            ),
        }
    }
}

/// `Schema.from_source`'s keyword arguments.
#[derive(Clone, Debug, Default)]
pub struct SourceArgs {
    pub tables: Option<Vec<(String, String)>>,
    /// The refusal of the first table `tables` names that is not a pair of
    /// non-empty names; `tables` holds the ones before it.
    pub not_a_table: Option<String>,
    pub primary_keys: Option<NamedKeys>,
    pub time_columns: Option<Vec<(String, Value)>>,
    pub relationships: Option<Vec<Rel>>,
    /// The refusal of the first link that is not a `Relationship`;
    /// `relationships` holds the ones before it.
    pub not_a_link: Option<String>,
    pub check_keys: bool,
}

/// `wire_dtype(engine_dtype)`.
pub fn wire_dtype(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "bool" => "bool",
        "int" => "int64",
        "byte" => "int32",
        "int16" => "int32",
        "int32" => "int32",
        "int64" => "int64",
        "float" => "float64",
        "float32" => "float32",
        "float64" => "float64",
        "string" => "string",
        "date" => "timestamp[us]",
        "time" => "timestamp[us]",
        _ => return None,
    })
}

const SEMANTIC_TYPES: [&str; 7] = [
    "ID",
    "categorical",
    "multicategorical",
    "numerical",
    "sequence",
    "text",
    "timestamp",
];

/// `_describe_source_table(engine_table, source_name, name=name)`.
pub fn describe_source_table(
    engine_table: &EngineTable,
    source_name: &str,
    name: Option<&str>,
    said: &mut Said,
) -> H<Table> {
    let mut columns: Vec<Col> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    for c in &engine_table.columns {
        let dtype = wire_dtype(c.dtype.value());
        let stype = c.stype.value();
        match dtype {
            Some(d) if SEMANTIC_TYPES.contains(&stype) => {
                let column = Col::new(&c.name, d, stype)?;
                match columns.iter_mut().find(|x| x.name == c.name) {
                    Some(slot) => *slot = column,
                    None => columns.push(column),
                }
            }
            _ => unreadable.push(format!("{} ({})", c.name, c.dtype.value())),
        }
    }
    let table_name = name.unwrap_or(&engine_table.name).to_string();
    if !unreadable.is_empty() {
        said.warn(
            SDM_WARNING,
            format!(
                "{table_name} leaves out {}: a request carries one scalar value per cell, and these hold none",
                unreadable.join(", ")
            ),
        );
    }
    check_name(&table_name)?;
    if source_name.is_empty() {
        return Err(Refusal::invalid(format!(
            "table {} needs the name it has in its source, got {}",
            str_repr(&table_name),
            str_repr(source_name)
        )));
    }
    for c in &columns {
        check_name(&c.name)?;
    }
    let mut table = Table {
        name: table_name,
        data: None,
        source_name: Some(source_name.to_string()),
        columns,
        primary_key: None,
        time_column: None,
        duplicate_keys: None,
    };
    let keys = engine_table.primary_key_columns();
    if !keys.is_empty() && keys.iter().all(|k| table.col(k).is_some()) {
        let key = if keys.len() == 1 {
            Key::One(keys[0].clone())
        } else {
            Key::Many(keys.clone())
        };
        table.set_primary_key_alone(Some(key), said)?;
    }
    if let Some(time) = &engine_table.time_column
        && table.col(time).is_some()
        && !keys.contains(time)
    {
        table.set_time_column_alone(Some(time))?;
    }
    Ok(table)
}

fn index_of(built: &[Table], name: &str) -> usize {
    built
        .iter()
        .position(|t| t.name == name)
        .expect("a table of the schema")
}

/// `_table_named(by_source_name, dst_table)`.
fn table_named(by_source_name: &[(String, String)], dst_table: &str) -> Option<String> {
    if let Some((_, name)) = by_source_name.iter().find(|(s, _)| s == dst_table) {
        return Some(name.clone());
    }
    let folded: Vec<&String> = by_source_name
        .iter()
        .filter(|(s, _)| py_lower(s) == py_lower(dst_table))
        .map(|(_, n)| n)
        .collect();
    if folded.len() == 1 {
        Some(folded[0].clone())
    } else {
        None
    }
}

fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

/// `_declared_links(graph, tables, built)`.
async fn declared_links(
    ctx: &mut Ctx<'_>,
    graph: &mut Graph,
    chosen: &[(String, String)],
    built: &[Table],
) -> H<Vec<Rel>> {
    let mut by_source_name: Vec<(String, String)> = Vec::new();
    for (name, _) in chosen {
        let i = graph.get(name)?;
        let source_name = graph.tables[i].source_name();
        match by_source_name.iter_mut().find(|(s, _)| *s == source_name) {
            Some(slot) => slot.1 = name.clone(),
            None => by_source_name.push((source_name, name.clone())),
        }
    }
    let mut links = Vec::new();
    for (name, _) in chosen {
        let i = graph.get(name)?;
        for key in graph.tables[i].source_foreign_keys(ctx).await? {
            let Some(target) = table_named(&by_source_name, &key.dst_table) else {
                continue;
            };
            if built[index_of(built, name)].col(&key.name).is_none() {
                continue;
            }
            if built[index_of(built, &target)]
                .col(&key.primary_key)
                .is_none()
            {
                continue;
            }
            links.push(Rel::new(name, &key.name, &target, &key.primary_key));
        }
    }
    Ok(links)
}

/// `_key_by_the_referenced_column(built, declared)`.
fn key_by_the_referenced_column(built: &mut [Table], declared: &[Rel], said: &mut Said) -> H<()> {
    let mut referenced: Vec<(String, BTreeSet<String>)> = Vec::new();
    for link in declared {
        match referenced.iter_mut().find(|(t, _)| *t == link.target_table) {
            Some((_, set)) => {
                set.insert(link.target_column.clone());
            }
            None => referenced.push((
                link.target_table.clone(),
                BTreeSet::from([link.target_column.clone()]),
            )),
        }
    }
    for (name, columns) in referenced {
        let i = index_of(built, &name);
        if built[i].primary_key.is_none() && columns.len() == 1 {
            let column = columns.into_iter().next().expect("one");
            built[i].set_primary_key_alone(Some(Key::One(column)), said)?;
        }
    }
    Ok(())
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Databricks => "Databricks",
        Backend::Snow => "Snowflake",
        Backend::Postgres => "PostgreSQL",
        Backend::Duckdb => "DuckDB",
        Backend::Sqlite => "SQLite",
    }
}

const SKIP_THE_CHECK: &str = "Name another key with primary_keys=, or pass check_keys=False to skip (one full scan per table).";

/// `_check_source_keys(...)`.
#[allow(clippy::too_many_arguments)]
async fn check_source_keys(
    ctx: &mut Ctx<'_>,
    backend: Backend,
    built: &mut [Table],
    graph: &mut Graph,
    found: &[(String, Vec<String>)],
    stated: &BTreeSet<String>,
    infer_links: bool,
    said: &mut Said,
) -> H<()> {
    for t in built.iter_mut() {
        t.duplicate_keys = Some(0);
    }
    let keyed: Vec<(String, Vec<String>)> = built
        .iter()
        .filter_map(|t| t.primary_key.as_ref().map(|k| (t.name.clone(), k.names())))
        .collect();
    if keyed.is_empty() {
        return Ok(());
    }
    let keys: Vec<(usize, Vec<String>)> = keyed
        .iter()
        .map(|(name, names)| graph.get(name).map(|i| (i, names.clone())))
        .collect::<Result<_, _>>()?;
    let counts = engine::count_keys(ctx, &mut graph.tables, &keys).await?;
    let mut declined: Vec<(String, engine::KeyCount)> = Vec::new();
    for ((name, names), count) in keyed.iter().zip(counts) {
        let (Some(rows), Some(distinct)) = (count.rows, count.distinct) else {
            continue;
        };
        let ti = index_of(built, name);
        let found_key = found
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, k)| k.clone())
            .unwrap_or_default();
        let declared = !stated.contains(name) && *names == found_key && {
            let gi = graph.get(name)?;
            graph.tables[gi].declares_key(ctx, names).await?
        };
        let repeats = count.repeats();
        let example = count.example.clone().unwrap_or_else(|| "None".to_string());
        if !stated.contains(name) && !declared {
            if (distinct as f64) < rows as f64 * 0.99 {
                built[ti].set_primary_key_alone(None, said)?;
                declined.push((name.clone(), count));
                continue;
            }
        } else if count.nulls != 0 && stated.contains(name) {
            let empty = if names.len() == 1 {
                Some(names.as_slice())
            } else {
                None
            };
            let missing = a_key_missing_values(name, names, count.nulls, empty);
            return Err(Refusal::invalid(format!("{missing}. {SKIP_THE_CHECK}")));
        } else if too_many_repeats(repeats, rows.max(0) as usize) {
            let found_out = not_unique_enough(name, names, repeats, rows.max(0) as usize, &example);
            let unenforced = if declared && !count.enforced {
                format!("{} does not enforce declared keys. ", backend_name(backend))
            } else {
                String::new()
            };
            return Err(Refusal::invalid(format!(
                "{found_out}. {unenforced}{SKIP_THE_CHECK}"
            )));
        }
        if count.nulls != 0 {
            warn_a_key_has_no_value(said, name, &names.join("."), count.nulls);
        }
        if repeats != 0 {
            warn_a_key_repeats(
                said,
                name,
                names,
                repeats,
                &example,
                " Name a different primary key if this column is a foreign key rather than an identifier.",
            );
        }
        built[ti].duplicate_keys = Some(repeats);
    }
    if !infer_links || declined.is_empty() {
        return Ok(());
    }
    let named_elsewhere = tables_naming_each_column(built);
    let keyed_by: BTreeSet<String> = built
        .iter()
        .filter_map(|t| match &t.primary_key {
            Some(Key::One(k)) => Some(k.clone()),
            _ => None,
        })
        .collect();
    for (name, count) in declined {
        let names = &keyed.iter().find(|(n, _)| *n == name).expect("keyed").1;
        let column = &names[0];
        let (rows, distinct) = (count.rows.unwrap_or(0), count.distinct.unwrap_or(0));
        let elsewhere = named_elsewhere
            .get(column)
            .is_some_and(|s| s.iter().any(|t| *t != name));
        if names.len() > 1
            || keyed_by.contains(column)
            || !elsewhere
            || (distinct as f64) < rows as f64 * 0.9
            || count.repeats() == 0
        {
            continue;
        }
        let example = count.example.clone().unwrap_or_else(|| "None".to_string());
        warn_a_key_was_declined(
            said,
            &name,
            column,
            count.repeats(),
            rows,
            &example,
            &format!(
                "Keep one row per {} in the source, or name a column that does identify a row with primary_keys={{{}: ...}}.",
                str_repr(column),
                str_repr(&name)
            ),
        );
    }
    Ok(())
}

/// The engine's warnings, said as `sdm_client`'s caller sees them.
fn pass_on(ctx: &mut Ctx<'_>, said: &mut Said) {
    for (category, message) in std::mem::take(&mut ctx.warnings.0) {
        if category == crate::diagnostics::TIME_COLUMN_WARNING {
            continue;
        }
        said.warn(crate::diagnostics::warning_category(&category), message);
    }
}

fn key_names(key: &Option<Key>) -> Vec<String> {
    key.as_ref().map(Key::names).unwrap_or_default()
}

/// `Schema.from_source(source, **args)`.
pub async fn from_source(
    ctx: &mut Ctx<'_>,
    info: &SourceInfo,
    args: &SourceArgs,
    said: &mut Said,
) -> H<Schema> {
    let mut kept = Vec::new();
    from_source_keeping(ctx, info, args, said, &mut kept).await
}

/// [`from_source`], and the engine tables it read, kept for gathering
/// (`schema._engine_tables`, an `EngineTables`), each under its name in the
/// source.
pub async fn from_source_keeping(
    ctx: &mut Ctx<'_>,
    info: &SourceInfo,
    args: &SourceArgs,
    said: &mut Said,
    kept: &mut Vec<(String, EngineTable)>,
) -> H<Schema> {
    match read_source(ctx, info, args, said, kept).await {
        // Today's code let the engine's bare `RuntimeError` through.
        Err(refusal) => match engine::emptied(&refusal.code, &refusal.message) {
            Some(name) => Err(Refusal::invalid(format!(
                "Table {} in {} is empty, and a schema reads each table's column types and keys \
                 from its rows. Add rows to it, or leave it out with tables=[...] naming the \
                 others.",
                str_repr(&name),
                info.repr()
            ))
            .caused_by(
                std::iter::once(engine::link(&refusal.code, refusal.message)).chain(refusal.chain),
            )),
            None => Err(refusal),
        },
        read => read,
    }
}

async fn read_source(
    ctx: &mut Ctx<'_>,
    info: &SourceInfo,
    args: &SourceArgs,
    said: &mut Said,
    kept: &mut Vec<(String, EngineTable)>,
) -> H<Schema> {
    if let Some(tables) = &args.tables {
        if tables.is_empty() && args.not_a_table.is_none() {
            return Err(Refusal::invalid(
                "a schema needs at least one table; leave tables out to take every table the source describes",
            ));
        }
        for (name, source_name) in tables {
            if name.is_empty() {
                return Err(Refusal::invalid(format!(
                    "table names must be non-empty strings; got {}",
                    str_repr(name)
                )));
            }
            if source_name.is_empty() {
                return Err(Refusal::invalid(format!(
                    "table {} must map to its name in the source, a non-empty string; got {}",
                    str_repr(name),
                    str_repr(source_name)
                )));
            }
        }
    }
    if let Some(refused) = &args.not_a_table {
        return Err(Refusal::invalid(refused.clone()));
    }
    let semantic = info.semantic_model.is_some();
    let mut built: Vec<Table> = Vec::new();
    let mut choices = BTreeMap::new();
    let declared: Vec<Rel>;
    let mut engine_graph: Option<Graph> = None;
    if semantic {
        let graph = engine::views::semantic_model_graph(ctx, info).await;
        pass_on(ctx, said);
        let graph = graph?;
        for table in &graph.tables {
            keep(kept, &table.name, table);
        }
        let offered: Vec<String> = graph.tables.iter().map(|t| t.name.clone()).collect();
        let chosen: Vec<(String, String)> = match &args.tables {
            Some(t) => t.clone(),
            None => offered.iter().map(|n| (n.clone(), n.clone())).collect(),
        };
        let absent: BTreeSet<String> = chosen
            .iter()
            .map(|(_, m)| m.clone())
            .filter(|m| !offered.contains(m))
            .collect();
        if !absent.is_empty() {
            let mut sorted_offered = offered.clone();
            sorted_offered.sort();
            let absent: Vec<String> = absent.into_iter().collect();
            return Err(Refusal::invalid(format!(
                "{} are not tables of {}; it has {}",
                crate::py::str_list_repr(&absent),
                info.repr(),
                crate::py::str_list_repr(&sorted_offered)
            )));
        }
        let mut as_named: Vec<(String, String)> = Vec::new();
        for (name, model) in &chosen {
            match as_named.iter_mut().find(|(m, _)| m == model) {
                Some(slot) => slot.1 = name.clone(),
                None => as_named.push((model.clone(), name.clone())),
            }
        }
        for (name, model) in &chosen {
            let i = graph.get(model)?;
            if let Some(choice) = &graph.tables[i].time_choice {
                choices.insert(name.clone(), choice.clone());
            }
            let table = describe_source_table(&graph.tables[i], model, Some(name), said)?;
            match built.iter_mut().find(|t| t.name == *name) {
                Some(slot) => *slot = table,
                None => built.push(table),
            }
        }
        declared = model_links(&graph, &built, &as_named);
    } else {
        let read = read_graph(
            ctx,
            info.backend,
            &info.engine_namespace(),
            args.tables.as_deref(),
        )
        .await;
        pass_on(ctx, said);
        let mut graph = match read {
            Ok(g) => g,
            Err(e) if e.class == "NoTablesFoundError" => {
                return Err(Refusal::invalid(format!(
                    "{} holds no table to describe",
                    info.repr()
                ))
                .caused_by(e.as_links()));
            }
            Err(e) if e.class == "TableNotFoundError" => {
                let links = e.as_links();
                return Err(Refusal::invalid(e.message).caused_by(links));
            }
            Err(error) if error.class == "MetricViewAsTableError" => {
                let name = error.details["name"].as_str().unwrap_or_default();
                let mut refusal = Refusal::invalid(format!(
                    "{} is a metric view, not a table: its measures are aggregations with no value per row. Read it as a semantic model instead, with source.semantic_model({}).",
                    str_repr(name), str_repr(name)
                )).caused_by(error.as_links());
                refusal
                    .details
                    .insert("metric_view".into(), Value::String(name.to_string()));
                return Err(refusal);
            }
            Err(other) => return Err(other.into()),
        };
        let chosen: Vec<(String, String)> = match &args.tables {
            Some(t) => t.clone(),
            None => graph
                .tables
                .iter()
                .map(|t| (t.name.clone(), t.name.clone()))
                .collect(),
        };
        for (name, named) in &chosen {
            let i = graph.get(name)?;
            if let Some(choice) = &graph.tables[i].time_choice {
                choices.insert(name.clone(), choice.clone());
            }
            let table = describe_source_table(&graph.tables[i], named, None, said)?;
            built.push(table);
        }
        let links = declared_links(ctx, &mut graph, &chosen, &built).await;
        pass_on(ctx, said);
        declared = links?;
        if args.relationships.is_none() {
            key_by_the_referenced_column(&mut built, &declared, said)?;
        }
        engine_graph = Some(graph);
    }
    // `read.keep({named: graph[name] ...})`: the very tables the keys are
    // checked on below, so what checking reads is kept with them.
    let keep_after: Vec<(String, String)> = match &args.tables {
        Some(t) => t.clone(),
        None => engine_graph
            .as_ref()
            .map(|g| {
                g.tables
                    .iter()
                    .map(|t| (t.name.clone(), t.name.clone()))
                    .collect()
            })
            .unwrap_or_default(),
    };
    let found: Vec<(String, Vec<String>)> = built
        .iter()
        .map(|t| (t.name.clone(), key_names(&t.primary_key)))
        .collect();
    apply_keys_and_times(
        &mut built,
        args.primary_keys.as_deref().unwrap_or(&[]),
        args.time_columns.as_deref().unwrap_or(&[]),
        said,
    )?;
    match (&mut engine_graph, args.check_keys) {
        (Some(graph), true) => {
            let stated: BTreeSet<String> = args
                .primary_keys
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|(n, _)| n.clone())
                .collect();
            let checked = check_source_keys(
                ctx,
                info.backend,
                &mut built,
                graph,
                &found,
                &stated,
                args.relationships.is_none(),
                said,
            )
            .await;
            pass_on(ctx, said);
            checked?;
        }
        _ => {
            for t in &mut built {
                t.duplicate_keys = None;
            }
        }
    }
    if let Some(graph) = &engine_graph {
        for (name, named) in &keep_after {
            if let Ok(i) = graph.get(name) {
                keep(kept, named, &graph.tables[i]);
            }
        }
    }
    let links =
        match &args.relationships {
            None => {
                let covered: BTreeSet<(String, String)> = declared
                    .iter()
                    .map(|l| (l.source_table.clone(), l.source_column.clone()))
                    .collect();
                let mut all = declared.clone();
                all.extend(client::infer_relationships(&built).into_iter().filter(|l| {
                    !covered.contains(&(l.source_table.clone(), l.source_column.clone()))
                }));
                client::joinable_inferred_links(&built, &all, said)
            }
            Some(given) => {
                let checked = client::checked_links(&built, given)?;
                if let Some(refused) = &args.not_a_link {
                    return Err(Refusal::invalid(refused.clone()));
                }
                checked
            }
        };
    client::settle_time_choices(
        &mut built,
        &mut choices,
        args.time_columns.as_deref().unwrap_or(&[]),
        &links,
    )?;
    said.ignoring.push(SDM_WARNING.to_string());
    let made = Schema::new(&built, &links, said);
    said.ignoring.pop();
    let mut schema = made?;
    schema.mark_key_columns(said)?;
    client::warn_time_choices(
        &schema,
        &choices,
        args.time_columns.as_deref().unwrap_or(&[]),
        said,
    );
    let rels = schema.relationships.clone();
    client::warn_two_ways_to_reach_one_table(&rels, said);
    Ok(schema)
}

/// `EngineTables.keep`: a table kept under `name` unless one is already.
fn keep(kept: &mut Vec<(String, EngineTable)>, name: &str, table: &EngineTable) {
    if !kept.iter().any(|(n, _)| n == name) {
        kept.push((name.to_string(), table.clone()));
    }
}

/// `_model_links(graph, built, as_named)`.
fn model_links(graph: &Graph, built: &[Table], as_named: &[(String, String)]) -> Vec<Rel> {
    let named = |model: &str| {
        as_named
            .iter()
            .find(|(m, _)| m == model)
            .map(|(_, n)| n.clone())
    };
    let mut links = Vec::new();
    for edge in &graph.edges {
        let (Some(source_name), Some(target_name)) =
            (named(&edge.src_table), named(&edge.dst_table))
        else {
            continue;
        };
        let target = &built[index_of(built, &target_name)];
        let source = &built[index_of(built, &source_name)];
        let Some(Key::One(key)) = &target.primary_key else {
            continue;
        };
        if source.col(&edge.fkey).is_none() {
            continue;
        }
        links.push(Rel::new(&source_name, &edge.fkey, &target_name, key));
    }
    links
}

/// A refusal the engine raised, as a `PyExc`.
pub fn refusal_of(e: &PyExc) -> Refusal {
    Refusal::class(&e.class, e.message.clone()).caused_by(e.chain.clone())
}

#[cfg(test)]
mod diagnostic_presentation_tests {
    use super::*;
    use sdm_protocol::{AsyncRun, Run, Step};

    #[test]
    fn source_warnings_suppress_time_and_preserve_public_view_category() {
        let mut run = AsyncRun::new(|asker| async move {
            let database = engine::Db(asker);
            let host = crate::dt::Host::assumed();
            let mut context = Ctx {
                db: &database,
                host: &host,
                warnings: engine::Warnings(vec![
                    (
                        crate::diagnostics::TIME_COLUMN_WARNING.into(),
                        "time".into(),
                    ),
                    (
                        crate::diagnostics::VIEW_CONVERSION_WARNING.into(),
                        "view".into(),
                    ),
                    ("builtins.DeprecationWarning".into(), "prefix".into()),
                ]),
            };
            let mut said = Said::default();
            pass_on(&mut context, &mut said);
            assert!(context.warnings.0.is_empty());
            Ok(said
                .warnings
                .into_iter()
                .map(|warning| (warning.category, warning.message))
                .collect::<Vec<_>>())
        });
        assert_eq!(
            run.next().unwrap(),
            Step::Done(vec![
                (
                    "sdm_connectors._warnings.ViewConversionWarning".into(),
                    "view".into()
                ),
                ("builtins.DeprecationWarning".into(), "prefix".into()),
            ])
        );
    }
}
