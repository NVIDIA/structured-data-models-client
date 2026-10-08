// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PredictiveQuery.materialize`: the checks of its arguments, the route a
//! request is gathered on, and the single-table route
//! (`materializers/tabular.py`).

use sdm_protocol::Asker;
use sdm_schema::client::Schema;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, json};
use std::sync::Arc;

use crate::error::{R, Warnings, invalid, structured};
use crate::frames::Table;
use crate::models::{Profile, TABULAR_CONTEXT_ROWS_MAX};
use crate::query::{Resolved, col_name, resolve, table_name};
use crate::request::{FrameArg, FrameArgs, Request, from_frames};
use crate::schemas::{DataSchema, Tables};
use crate::series::is_na;

/// `MAX_TOP_K` of the language's model validator.
pub const MAX_TOP_K: i64 = 20;

pub type SequenceOperation = dyn Fn(&str) -> R<PyVal> + Send + Sync;

#[derive(Clone)]
pub struct SequenceProtocol(pub Arc<SequenceOperation>);

impl std::fmt::Debug for SequenceProtocol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SequenceProtocol")
    }
}

impl SequenceProtocol {
    pub fn items(&self, operation: &str) -> R<Vec<PyVal>> {
        match (self.0)(operation)? {
            PyVal::List(items) => Ok(items),
            _ => Err(crate::error::value_error(
                "sequence operation did not return items",
            )),
        }
    }

    pub fn length(&self) -> R<usize> {
        crate::pyops::integer_value(&(self.0)("len")?)
            .and_then(|value| value.to_i128())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| crate::error::value_error("sequence length is not an integer"))
    }

    pub fn truth(&self) -> R<bool> {
        match (self.0)("bool")? {
            PyVal::Bool(value) => Ok(value),
            _ => Err(crate::error::value_error("sequence truth is not a boolean")),
        }
    }
}

/// What `materialize(...)` is given.
#[derive(Clone, Debug, Default)]
pub struct Args {
    pub indices: Option<PyVal>,
    pub indices_protocol: Option<SequenceProtocol>,
    pub predict: Option<FrameArg>,
    pub num_context_samples: Option<PyVal>,
    pub model: Option<PyVal>,
    pub candidates: Option<PyVal>,
    pub candidates_protocol: Option<SequenceProtocol>,
    pub task: Option<PyVal>,
    /// The other options, in the order given.
    pub options: Vec<(String, PyVal)>,
}

/// `_is_hashable(value)`.
fn is_hashable(value: &PyVal) -> bool {
    match value {
        PyVal::List(_) | PyVal::Dict(_) | PyVal::NdArray(..) => false,
        PyVal::Tuple(items) => items.iter().all(is_hashable),
        PyVal::Opaque { kind, .. } => kind != "mapping" && kind != "container",
        _ => true,
    }
}

/// The items of a sequence argument, or `None` where it is not one.
fn sequence(value: &PyVal) -> Option<Vec<PyVal>> {
    match value {
        PyVal::List(items) | PyVal::Tuple(items) => Some(items.clone()),
        _ => None,
    }
}

/// `_validate_materialize_args(...)`.
#[allow(clippy::too_many_arguments)]
fn validate_args(
    relational: bool,
    model_id: Option<&str>,
    len_schema: usize,
    relationships: usize,
    args: &mut Args,
) -> R<()> {
    let model = model_id.unwrap_or("None");
    if args.indices.is_some() && args.predict.is_some() {
        return Err(invalid(
            "pass either indices=[...] to score rows the tables already contain, or \
             predict=<frame> to score rows they do not, not both",
        ));
    }
    if args.indices.is_none() && args.predict.is_none() {
        return Err(invalid(
            "materialize() needs the rows to score: pass indices=[...] naming the entities. \
             There is no all-rows default, because it would silently score an entire table",
        ));
    }
    if !relational && len_schema != 1 {
        return Err(invalid(format!(
            "{model} answers from a single table, and this schema has {len_schema}. Pass \
             model=KumoRelational() to gather them as a graph"
        )));
    }
    if !relational && relationships > 0 {
        return Err(invalid(format!(
            "{model} reads no relationships, and this schema declares {relationships}. Pass \
             model=KumoRelational() to gather it as a graph, or drop the relationship to ask \
             about the table on its own"
        )));
    }
    if !relational && !args.options.is_empty() {
        let mut names: Vec<String> = args.options.iter().map(|(k, _)| k.clone()).collect();
        names.sort();
        return Err(invalid(format!(
            "gathering options apply only to relational materialization: {}. Pass \
             indices=[...] with model=KumoRelational() to gather this schema as a graph",
            sdm_schema::py::str_list_repr(&names)
        )));
    }
    if args.predict.is_some() {
        if len_schema != 1 {
            return Err(invalid(format!(
                "predict=<frame> requires a single-table schema; got {len_schema} tables. For a \
                 relational schema, use indices=[...] to select entities already in the tables"
            )));
        }
        if relational {
            let instead = if relationships > 0 {
                "drop the relationship to gather this table for the tabular model"
            } else {
                "gather this request for the tabular model"
            };
            return Err(invalid(format!(
                "predict=<frame> is the single-table form, and {} scores rows the tables already \
                 contain. Pass indices=[...] naming them, or {instead}",
                model_id.unwrap_or("a relational gather")
            )));
        }
        if args.num_context_samples.is_some() {
            return Err(invalid(
                "num_context_samples cannot be used with predict=<frame>: the schema table is the \
                 context. To choose the context, build the schema from the rows you want",
            ));
        }
    }
    if let Some(indices) = &args.indices {
        let items = match &args.indices_protocol {
            Some(protocol) => Some(protocol.items("iter")?),
            None => sequence(indices),
        };
        let Some(items) = items else {
            return Err(invalid(format!(
                "indices must be a sequence of entity keys, got {}",
                crate::models::type_name(indices)
            )));
        };
        let unusable: Vec<&PyVal> = items.iter().filter(|v| !is_hashable(v)).collect();
        if !unusable.is_empty() {
            let length = match &args.indices_protocol {
                Some(protocol) => protocol.length()?,
                None => items.len(),
            };
            let mut kinds: Vec<String> = unusable
                .iter()
                .map(|v| crate::models::type_name(v))
                .collect();
            kinds.sort();
            kinds.dedup();
            return Err(invalid(format!(
                "indices must be entity keys, and {} of the {} given cannot be one: {}. A key is \
                 looked up by value, so it has to be hashable. This is what a row of \
                 json.loads(...) or frame.to_dict(\"records\") gives you; pass the key field of \
                 each instead, e.g. [row['<key>'] for row in rows].",
                unusable.len(),
                length,
                sdm_schema::py::str_list_repr(&kinds)
            )));
        }
        if args.indices_protocol.is_some() {
            args.indices = Some(PyVal::List(items));
        }
    }
    if let Some(n) = &args.num_context_samples {
        let whole = matches!(n, PyVal::Int(_) | PyVal::BigInt(_))
            || matches!(n, PyVal::Numpy(t, _) if t.starts_with("int") || t.starts_with("uint"));
        if !whole {
            return Err(invalid(format!(
                "num_context_samples must be a whole number of labelled examples, got {} {}",
                crate::models::type_name(n),
                crate::pyops::py_repr(n)
            )));
        }
        let value = crate::pyops::integer_value(n)
            .and_then(|d| d.to_i128())
            .unwrap_or(0);
        if value < 1 {
            return Err(invalid(format!(
                "num_context_samples must be at least 1; a model predicts from labelled \
                 examples, and {value} leaves it none"
            )));
        }
    }
    Ok(())
}

/// `_candidates_to_rank(query, options)`.
pub fn candidates_to_rank(query: &Resolved, options: &[(String, PyVal)]) -> R<Option<i64>> {
    let Some((_, given)) = options.iter().find(|(k, _)| k == "top_k") else {
        return Ok(query.top_k);
    };
    let whole = matches!(given, PyVal::Int(_) | PyVal::BigInt(_))
        || matches!(given, PyVal::Numpy(t, _) if t.starts_with("int") || t.starts_with("uint"));
    if !whole {
        return Err(invalid(format!(
            "top_k must be a whole number of candidates, got {}",
            crate::models::type_name(given)
        )));
    }
    let value = crate::pyops::integer_value(given).and_then(|d| d.to_i64());
    match value {
        Some(k) if (1..=MAX_TOP_K).contains(&k) => Ok(Some(k)),
        _ => Err(invalid(format!(
            "top_k is {}, and this model ranks between 1 and {MAX_TOP_K} candidates",
            crate::pyops::py_str(given)
        ))),
    }
}

/// What gathering needs: the schema it is over and its frames.
pub struct Gathering<'a> {
    pub handle: &'a DataSchema,
    pub schema: &'a Schema,
    pub tables: &'a mut Tables,
    pub host: &'a Host,
    /// The query's relational model, kept between its materializes.
    pub model: &'a mut Option<crate::relational::Model>,
    /// Whether the sampler has said it found too few examples, which it
    /// says once in a process (`_coverage_warned`); the host keeps it.
    pub coverage_warned: &'a mut bool,
}

/// `PredictiveQuery(text, schema).materialize(**args)`.
pub async fn materialize(
    asker: &Asker,
    text: &str,
    gathering: Gathering<'_>,
    mut args: Args,
    said: &mut Warnings,
) -> R<Request> {
    let Gathering {
        handle,
        schema,
        tables,
        host,
        model,
        coverage_warned,
    } = gathering;
    let spec = match &args.model {
        Some(model) => Some(crate::models::spec_for(model)?),
        None => None,
    };
    let in_place = handle.source.is_some();
    let relational = match spec {
        Some(spec) => spec.profile == Profile::Relational,
        None => in_place || schema.tables.len() > 1 || !schema.relationships.is_empty(),
    };
    if in_place && args.predict.is_some() && relational {
        return Err(invalid(
            "predict= carries rows, and a schema over tables in their source is sampled there \
             instead; name the entities to score with indices=",
        ));
    }
    if in_place && !relational {
        return Err(invalid(
            "a schema over tables in their source is sampled with SQL, which only the relational \
             model reads; the tabular model takes rows in memory. Use model=KumoRelational() with \
             indices=, or build the schema with Schema.from_data from the rows you want",
        ));
    }
    validate_args(
        relational,
        spec.map(|s| s.model_id),
        schema.tables.len(),
        schema.relationships.len(),
        &mut args,
    )?;
    let profile = if relational {
        Profile::Relational
    } else {
        Profile::Tabular
    };
    let asked = resolve(text, schema, profile, host, said)?;
    let candidates = match &args.candidates_protocol {
        Some(protocol) => Some(protocol.items("list")?),
        None => args
            .candidates
            .as_ref()
            .map(|given| sequence(given).unwrap_or_else(|| vec![given.clone()])),
    };
    let rewritten = crate::decompose::parts_of(
        &asked,
        text,
        candidates.as_deref(),
        &args.options,
        Some(schema),
    )?;
    if let Some(task) = &args.task
        && (relational || rewritten.is_some())
    {
        let why = if rewritten.is_some() {
            "a decomposed question"
        } else {
            "a relational query"
        };
        let mut capital = why.to_string();
        capital[..1].make_ascii_uppercase();
        return Err(invalid(format!(
            "task={} settles what a single-table question is asking, which is the only route \
             that has to decide. {capital} resolves to its own kind, so state it in the query \
             rather than beside it",
            crate::pyops::py_repr(task)
        ))
        .with_detail("task", json!(crate::pyops::py_repr(task))));
    }
    if let Some(rewritten) = rewritten {
        let gathering = Gathering {
            handle,
            schema,
            tables,
            host,
            model,
            coverage_warned,
        };
        return in_parts(asker, text, gathering, &args, rewritten, said).await;
    }
    if let Some(predict) = &args.predict {
        return tabular_table(
            &asked,
            handle,
            schema,
            tables,
            predict,
            args.task.as_ref(),
            host,
            said,
        );
    }
    if !relational {
        return tabular_indices(&asked, handle, schema, tables, &args, host, said);
    }
    let indices = args.indices.as_ref().and_then(sequence).unwrap_or_default();
    let num_context_samples = args
        .num_context_samples
        .as_ref()
        .and_then(crate::pyops::integer_value)
        .and_then(|d| d.to_i64());
    crate::relational::materialize(
        asker,
        crate::relational::Relational {
            query: &asked,
            handle,
            schema,
            tables,
            host,
            indices,
            indices_protocol: args.indices_protocol.as_ref(),
            num_context_samples,
            options: &args.options,
            model,
            coverage_warned,
        },
        said,
    )
    .await
}

/// `PredictiveQuery._in_parts(rewritten, **how)`: each question the
/// query is asked as, gathered like any other, and kept together.
async fn in_parts(
    asker: &Asker,
    text: &str,
    gathering: Gathering<'_>,
    args: &Args,
    rewritten: crate::request::Decomposition,
    said: &mut Warnings,
) -> R<Request> {
    let Gathering {
        handle,
        schema,
        tables,
        host,
        model: _,
        coverage_warned,
    } = gathering;
    let given = |v: &Option<PyVal>| v.clone().filter(|v| !matches!(v, PyVal::None));
    let how = Args {
        indices: given(&args.indices),
        indices_protocol: args.indices_protocol.clone(),
        predict: args.predict.clone(),
        num_context_samples: given(&args.num_context_samples),
        model: given(&args.model),
        candidates: None,
        candidates_protocol: None,
        task: None,
        options: args
            .options
            .iter()
            .filter(|(k, v)| k != "top_k" && !matches!(v, PyVal::None))
            .cloned()
            .collect(),
    };
    let mut gathered = Vec::new();
    for (part, about) in &rewritten.parts {
        // Each part is a query of its own, with a model of its own.
        let mut fresh = None;
        let asked = Box::pin(materialize(
            asker,
            part,
            Gathering {
                handle,
                schema,
                tables,
                host,
                model: &mut fresh,
                coverage_warned: &mut *coverage_warned,
            },
            how.clone(),
            said,
        ))
        .await;
        match asked {
            Ok(request) => gathered.push(request),
            Err(error) if crate::parts::is_structured(&error) && !is_missing_extra(&error) => {
                return Err(crate::parts::about_the_part(error, &rewritten, about));
            }
            Err(error) => return Err(error),
        }
    }
    let mut whole = gathered
        .first()
        .cloned()
        .ok_or_else(|| structured("INTERNAL_ERROR", "a question asked in no parts"))?;
    whole.parts = Some(gathered);
    whole.decomposition = Some(rewritten);
    whole.asked = Some(text.to_string());
    Ok(whole)
}

/// Whether an error is `MissingExtraError`, which a part passes on as it is.
fn is_missing_extra(error: &sdm_protocol::Diagnostic) -> bool {
    error.class.as_deref() == Some(crate::error::MISSING_EXTRA)
}

// --- The single-table route --------------------------------------------------

/// `_rows(table)`: the schema table's frame.
fn rows_of(handle: &DataSchema, tables: &Tables, name: &str) -> R<Table> {
    let _ = handle;
    tables
        .get(&DataSchema::frame_key(name))
        .cloned()
        .ok_or_else(|| {
            invalid(format!(
                "{} stays in its source, and the tabular model reads rows in memory; build the \
                 schema with Schema.from_data",
                sdm_pql::pyrepr::str_repr(name)
            ))
        })
}

/// The question's entity and target columns, refused by name where the
/// frame no longer has them: today's code met pandas' bare `KeyError`, or
/// a refusal of the scored rows that did not say why.
fn require_what_the_question_reads(rows: &Table, entity: &str, columns: &[&String]) -> R<()> {
    let gone: Vec<String> = columns
        .iter()
        .filter(|c| !rows.has(c))
        .map(|c| format!("{entity}.{c}"))
        .collect();
    if gone.is_empty() {
        return Ok(());
    }
    let named = sdm_schema::client::and_list(&gone);
    let is = if gone.len() == 1 { "is" } else { "are" };
    Err(invalid(format!(
        "{named} {is} no longer in the frame of {} that Schema.from_data read, and the question \
         reads {}. Put {} back, or build the schema again from the frame as it is now",
        sdm_pql::pyrepr::str_repr(entity),
        if gone.len() == 1 { "it" } else { "them" },
        if gone.len() == 1 { "it" } else { "them" },
    )))
}

/// `_task_kind(query, target, stated)`.
fn task_kind(
    query: &Resolved,
    target: &crate::series::Series,
    stated: Option<&PyVal>,
    said: &mut Warnings,
) -> R<String> {
    let stype = query.target_stype.clone().unwrap_or_default();
    if stype != "categorical" && stype != "numerical" {
        return Err(structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "tabular materialization requires a categorical or numerical target; {} has \
                 semantic type {}",
                sdm_pql::pyrepr::str_repr(&query.target_column),
                sdm_pql::pyrepr::str_repr(&stype)
            ),
        ));
    }
    if let Some(stated) = stated {
        let ok = matches!(stated, PyVal::Str(s) if s == "classification" || s == "regression");
        if !ok {
            return Err(structured(
                "UNSUPPORTED_REQUEST",
                format!(
                    "the tabular model does not serve the task kind {}; it serves \
                     ['classification', 'regression']",
                    crate::pyops::py_repr(stated)
                ),
            )
            .with_detail("task", crate::layout::encode_deep(stated))
            .with_detail("supported", json!(["classification", "regression"])));
        }
        return Ok(crate::pyops::py_str(stated));
    }
    if stype == "categorical" {
        return Ok("classification".to_string());
    }
    if crate::tabular::holds_two_whole_numbers(target)? {
        said.sdm(format!(
            "{} holds two whole numbers, which is how a binary label is usually written, so this \
             is being asked as a classification. If they are quantities rather than labels, pass \
             task=\"regression\" to ask for one; pass task=\"classification\" to say this was \
             meant and silence this.",
            query.target_column
        ));
        return Ok("classification".to_string());
    }
    Ok("regression".to_string())
}

/// `_context_within_limit(candidates, num_context_samples, table, remedy)`.
fn context_within_limit(
    candidates: Table,
    num_context_samples: Option<i64>,
    table: &str,
    remedy: &str,
    said: &mut Warnings,
) -> R<Table> {
    let wanted = match num_context_samples {
        Some(n) => {
            if n > TABULAR_CONTEXT_ROWS_MAX {
                return Err(structured(
                    "UNSUPPORTED_REQUEST",
                    format!(
                        "num_context_samples is {}, and the endpoint accepts at most {} context \
                         rows",
                        sdm_schema::py::thousands(n),
                        sdm_schema::py::thousands(TABULAR_CONTEXT_ROWS_MAX)
                    ),
                ));
            }
            n
        }
        None => TABULAR_CONTEXT_ROWS_MAX,
    };
    if candidates.len() as i64 <= wanted {
        return Ok(candidates);
    }
    if num_context_samples.is_none() {
        said.sdm(format!(
            "{table} has {} rows that could be context and the endpoint accepts {}, so {} were \
             sampled. {remedy}",
            sdm_schema::py::thousands(candidates.len() as i64),
            sdm_schema::py::thousands(TABULAR_CONTEXT_ROWS_MAX),
            sdm_schema::py::thousands(wanted)
        ));
    }
    let mut stream = sdm_random::RandomState::new(42);
    let mut chosen: Vec<usize> = stream
        .choice(candidates.len() as i64, wanted as usize)
        .map_err(|e| crate::error::value_error(e.to_string()))?
        .into_iter()
        .map(|p| p as usize)
        .collect();
    chosen.sort_unstable();
    Ok(candidates.take(&chosen))
}

/// The rows of `frame` whose `column` is answered.
fn answered_rows(frame: &Table, column: &str) -> Vec<usize> {
    frame
        .series(column)
        .map(|s| {
            s.cells
                .iter()
                .enumerate()
                .filter(|(_, c)| !is_na(c))
                .map(|(i, _)| i)
                .collect()
        })
        .unwrap_or_default()
}

/// `TabularMaterializer.table(query, predict, task)`.
#[allow(clippy::too_many_arguments)]
fn tabular_table(
    query: &Resolved,
    handle: &DataSchema,
    schema: &Schema,
    tables: &mut Tables,
    predict: &FrameArg,
    stated: Option<&PyVal>,
    host: &Host,
    said: &mut Warnings,
) -> R<Request> {
    let _ = schema;
    let entity = table_name(&query.entity_column);
    let entity_col = col_name(&query.entity_column);
    let target_col = col_name(&query.target_column);
    let rows = rows_of(handle, tables, &entity)?;
    require_what_the_question_reads(&rows, &entity, &[&entity_col, &target_col])?;
    let without_scored = match predict {
        FrameArg::Frame(p) if rows.has(&entity_col) && p.has(&entity_col) => {
            let scored: Vec<String> = p
                .series(&entity_col)
                .map(|s| s.cells.iter().map(PyVal::eq_key).collect())
                .unwrap_or_default();
            if scored.is_empty() {
                rows.clone()
            } else {
                let keep: Vec<usize> = rows
                    .series(&entity_col)
                    .map(|s| {
                        s.cells
                            .iter()
                            .enumerate()
                            .filter(|(_, c)| !scored.contains(&c.eq_key()))
                            .map(|(i, _)| i)
                            .collect()
                    })
                    .unwrap_or_default();
                rows.take(&keep)
            }
        }
        _ => rows.clone(),
    };
    let carrying = if without_scored.has(&target_col) {
        without_scored.take(&answered_rows(&without_scored, &target_col))
    } else {
        without_scored
    };
    let context = context_within_limit(
        carrying,
        None,
        &entity,
        "This route takes no num_context_samples, so choose the context by building the schema \
         from the rows you want.",
        said,
    )?;
    let target = context
        .series(&target_col)
        .ok_or_else(|| crate::error::builtin("KeyError", sdm_pql::pyrepr::str_repr(&target_col)))?;
    let kind = task_kind(query, &target, stated, said)?;
    let mut request = from_frames(
        FrameArgs {
            context: FrameArg::Frame(context),
            predict: predict.clone(),
            target: PyVal::Str(target_col),
            task: PyVal::Str(kind),
            related: Vec::new(),
            schema: Some(handle.clone()),
            entity_table: entity,
            inference: Map::new(),
            task_options: Map::new(),
        },
        tables,
        host,
        said,
    )?;
    request.query = Some(PyVal::Str(query.text.clone()));
    Ok(request)
}

/// `TabularMaterializer.indices(query, indices, num_context_samples, task)`.
fn tabular_indices(
    query: &Resolved,
    handle: &DataSchema,
    schema: &Schema,
    tables: &mut Tables,
    args: &Args,
    host: &Host,
    said: &mut Warnings,
) -> R<Request> {
    let _ = schema;
    let entity = table_name(&query.entity_column);
    let rows = rows_of(handle, tables, &entity)?;
    let entity_col = col_name(&query.entity_column);
    let target_col = col_name(&query.target_column);
    require_what_the_question_reads(&rows, &entity, &[&entity_col, &target_col])?;
    let indices = match &args.indices_protocol {
        Some(protocol) => protocol.items("list")?,
        None => args.indices.as_ref().and_then(sequence).unwrap_or_default(),
    };
    // `entity_table[entity_col]`, of a frame that may have lost the column
    // since its schema read it.
    let keys = rows
        .series(&entity_col)
        .map(|s| s.cells)
        .ok_or_else(|| crate::error::builtin("KeyError", sdm_pql::pyrepr::str_repr(&entity_col)))?;
    let mut positions: Vec<(String, Vec<usize>)> = Vec::new();
    for (at, value) in keys.iter().enumerate() {
        let key = value.eq_key();
        match positions.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1.push(at),
            None => positions.push((key, vec![at])),
        }
    }
    let lookup = |value: &PyVal| {
        positions
            .iter()
            .find(|(k, _)| *k == value.eq_key())
            .map(|(_, p)| p.clone())
    };
    let missing: Vec<&PyVal> = indices.iter().filter(|v| lookup(v).is_none()).collect();
    if !missing.is_empty() {
        let shown: Vec<String> = missing
            .iter()
            .take(10)
            .map(|v| crate::pyops::py_repr(v))
            .collect();
        let tail = if missing.len() > 10 { ", ..." } else { "" };
        return Err(invalid(format!(
            "{} of the {} indices are not in {}: [{}]{tail}. Pass predict=<frame> to score rows \
             not present in the table",
            missing.len(),
            indices.len(),
            query.entity_column,
            shown.join(", ")
        )));
    }
    let scored: Vec<usize> = indices
        .iter()
        .flat_map(|v| lookup(v).unwrap_or_default())
        .collect();
    // `.drop(columns=[target_col])`, which pandas refuses for a column the
    // frame no longer has.
    let Some(answers) = rows.series(&target_col) else {
        return Err(crate::error::builtin(
            "KeyError",
            sdm_pql::pyrepr::str_repr(&format!(
                "{} not found in axis",
                sdm_schema::py::str_list_repr(std::slice::from_ref(&target_col))
            )),
        ));
    };
    let predict = rows.take(&scored).drop(std::slice::from_ref(&target_col));
    let mut in_context: Vec<bool> = answers.cells.iter().map(|c| !is_na(c)).collect();
    for &at in &scored {
        if at < in_context.len() {
            in_context[at] = false;
        }
    }
    let candidates: Vec<usize> = in_context
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(i, _)| i)
        .collect();
    let n = args
        .num_context_samples
        .as_ref()
        .and_then(crate::pyops::integer_value)
        .and_then(|d| d.to_i64());
    let context = context_within_limit(
        rows.take(&candidates),
        n,
        &entity,
        "Pass num_context_samples to choose how many, or select the rows yourself.",
        said,
    )?;
    let target = context
        .series(&target_col)
        .ok_or_else(|| crate::error::builtin("KeyError", sdm_pql::pyrepr::str_repr(&target_col)))?;
    let kind = task_kind(query, &target, args.task.as_ref(), said)?;
    let mut request = from_frames(
        FrameArgs {
            context: FrameArg::Frame(context),
            predict: FrameArg::Frame(predict),
            target: PyVal::Str(target_col),
            task: PyVal::Str(kind),
            related: Vec::new(),
            schema: Some(handle.clone()),
            entity_table: entity,
            inference: Map::new(),
            task_options: Map::new(),
        },
        tables,
        host,
        said,
    )?;
    request.query = Some(PyVal::Str(query.text.clone()));
    Ok(request)
}
