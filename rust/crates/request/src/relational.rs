// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The relational route (`materializers/relational.py`): the checks and
//! warnings around gathering, and the gathering itself as the engine does
//! it – `KumoRelational._get_task_table`, the `TaskTable` it makes
//! (`rfm/inference/task.py`), `materialize_task` and `_iter_task_requests`
//! – with the sampling done by the Phase 7.5 driver and each envelope built
//! by [`crate::payload::predict_request_to_json`].

use crate::pyops::TaggedValue;
use arrow_array::RecordBatch;
use sdm_protocol::{Asker, Diagnostic};
use sdm_sampling::driver::inputs::Inputs;
use sdm_sampling::driver::sampler::Sampler;
use sdm_sampling::driver::{Db, Raised};
use sdm_sampling::frame::Frame as SFrame;
use sdm_sampling::graph_store::TableReport;
use sdm_schema::client::{Rel, Schema};
use sdm_schema::dt::Host;
use sdm_schema::engine::PyExc;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use crate::error::{R, Warnings, builtin, invalid, structured};
use crate::gather::{Built, EngineGraph};
use crate::models::relational_limits as limits;
use crate::payload::{
    InferenceConfig, PredictRequest, PredictionContext, SampledTable, predict_request_to_json,
    validate_payload_table_rows,
};
use crate::query::{Resolved, col_name, table_name};
use crate::request::{Request, Target, Task};
use crate::schemas::{DataSchema, Tables};
use crate::series::Series;

/// `RelationalMaterializer._MATERIALIZE_OPTIONS`.
const MATERIALIZE_OPTIONS: [&str; 13] = [
    "anchor_time",
    "context_anchor_time",
    "exclude_cols_dict",
    "inference_config",
    "lag_timesteps",
    "max_pq_iterations",
    "num_hops",
    "num_neighbors",
    "prediction_statistic",
    "random_seed",
    "return_embeddings",
    "top_k",
    "use_prediction_time",
];

/// `RelationalMaterializer._RESERVED`.
const RESERVED: [&str; 2] = ["instance_id", "__kumo_anchor_time"];

/// `_RANDOM_SEED` of the engine.
const RANDOM_SEED: i64 = 42;

/// `_MAX_SIZE`: 30 MiB.
pub const MAX_SIZE: usize = 30 * 1024 * 1024;

/// `_SIZE_LIMIT_MSG`.
const SIZE_LIMIT_MSG: &str = "Context size exceeds the 30MB limit. {stats}\nPlease reduce either \
     the number of tables in the graph, their number of columns (e.g., large text columns), \
     neighborhood configuration, or the run mode. If none of this is possible, please create a \
     feature request at 'https://github.com/NVIDIA/structured-data-models-client' if you must go \
     beyond this for your use-case.";

/// `_DEFAULT_NUM_NEIGHBORS[run_mode]`.
fn default_num_neighbors(run_mode: &str) -> Vec<i64> {
    match run_mode {
        "debug" => vec![16, 16, 4, 4, 1, 1],
        "normal" | "best" => vec![64, 64, 8, 8, 4, 4],
        _ => vec![32, 32, 8, 8, 4, 4],
    }
}

/// `_MAX_CONTEXT_SIZE[run_mode]`.
fn max_context_size(run_mode: &str) -> usize {
    match run_mode {
        "debug" => 100,
        "normal" => 5_000,
        "best" => 10_000,
        _ => 1_000,
    }
}

/// `_MAX_PRED_SIZE[task_type]`.
fn max_pred_size(task_type: &str) -> usize {
    if task_type == "temporal_link_prediction" {
        200
    } else {
        1_000
    }
}

fn option<'a>(options: &'a [(String, PyVal)], name: &str) -> Option<&'a PyVal> {
    options.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn is_whole(value: &PyVal) -> bool {
    matches!(value, PyVal::Int(_) | PyVal::BigInt(_))
        || matches!(value, PyVal::Numpy(t, _) if t.starts_with("int") || t.starts_with("uint"))
}

fn whole_of(value: &PyVal) -> Option<i64> {
    crate::pyops::integer_value(value).and_then(|d| d.to_i64())
}

// --- Translating the engine's refusals ---------------------------------------------------

/// Whether `_engine_errors` passes a class on as the caller's mistake.
fn is_caller_error(class: &str) -> bool {
    matches!(
        class,
        "ValueError"
            | "KeyError"
            | "TypeError"
            | "KumoRelationalError"
            | "TaskReferenceError"
            | "NoEligibleEntityError"
            | "TableNotFoundError"
            | "UnknownDatasetError"
            | "NoTablesFoundError"
            | "GraphConstructionError"
            | "InvalidResponseError"
            | "ClientInitializationError"
            | "OutOfBoundsDatetime"
            | "DateParseError"
            | "ParserError"
            | "ValidationError"
            | "UnicodeDecodeError"
    )
}

/// `_sanitization_detail(model)`: the rows the graph dropped, and why.
fn sanitization_detail(report: &[(String, TableReport)]) -> String {
    let mut reasons = Vec::new();
    for (name, table) in report {
        let dropped = table.input_rows as i64 - table.output_rows as i64;
        if dropped <= 0 {
            continue;
        }
        let causes: Vec<String> = [
            ("no primary key", table.null_primary_key_rows),
            ("a repeated primary key", table.duplicate_primary_key_rows),
            ("no time", table.null_time_rows),
        ]
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(label, count)| format!("{count} with {label}"))
        .collect();
        let mut reason = format!("{name}: {dropped} of {} rows dropped", table.input_rows);
        if !causes.is_empty() {
            reason.push_str(&format!(" ({})", causes.join(", ")));
        }
        reasons.push(reason);
    }
    if reasons.is_empty() {
        String::new()
    } else {
        format!(" [{}]", reasons.join("; "))
    }
}

/// `_what_the_engine_refused(said)`.
fn what_the_engine_refused(said: &str) -> Map<String, Value> {
    let lowered = said.to_lowercase();
    let mut details = Map::new();
    if [
        "aggregation time range",
        "we would need data back to",
        "anchor timestamp is too early",
    ]
    .iter()
    .any(|mark| lowered.contains(mark))
    {
        details.insert("window_outran_the_data".into(), json!(true));
    }
    details
}

/// A table read as having rows that has none with a time now, refused by
/// name: today's code ran on into the out-of-range time its empty range
/// stands for (`pd.Timestamp.max` for the earliest time), and said only that.
fn emptied_since(sampler: &Sampler, error: &PyExc) -> Option<Diagnostic> {
    if error.class != "OutOfBoundsDatetime" {
        return None;
    }
    let mut empty: Vec<&String> = sampler
        .min_time
        .iter()
        .filter(|(_, earliest)| **earliest == sdm_sampling::driver::sampler::TIMESTAMP_MAX)
        .map(|(table, _)| table)
        .collect();
    if empty.is_empty() {
        return None;
    }
    empty.sort();
    let named: Vec<String> = empty.iter().map(|t| t.to_string()).collect();
    let (them, holds, it) = if named.len() == 1 {
        (
            format!("Table {}", sdm_pql::pyrepr::str_repr(&named[0])),
            "holds",
            "it",
        )
    } else {
        (
            format!("Tables {}", sdm_schema::client::and_list(&named)),
            "hold",
            "them",
        )
    };
    Some(
        invalid(format!(
            "{them} {holds} no row with a time in the source now, so there is no time to ask \
             this question at. Fill {it} again, or read the schema again and leave {it} out"
        ))
        .caused_by(error.as_links()),
    )
}

/// What the engine raised, as `_engine_errors(model)` re-raises it.
fn engine_error(error: PyExc, report: &[(String, TableReport)]) -> Diagnostic {
    // Raised from what the engine raised, and what that was raised from.
    let links = error.as_links();
    if !is_caller_error(&error.class) {
        return structured(
            "INTERNAL_ERROR",
            format!(
                "the relational engine failed while building this request: {}: {}",
                error.class, error.message
            ),
        )
        .caused_by(links);
    }
    let detail = if error.class == "TypeError" {
        String::new()
    } else {
        sanitization_detail(report)
    };
    invalid(format!("{}{detail}", error.message))
        .with_details(what_the_engine_refused(&error.message))
        .caused_by(links)
}

/// A sampling run's error as the exception today's code raises.
fn exc_of(raised: Raised) -> PyExc {
    let message = if raised.class == "KeyError" {
        sdm_pql::pyrepr::str_repr(&raised.message)
    } else {
        raised.message
    };
    PyExc::new(&raised.class, message).caused_by(raised.chain)
}

/// The warnings a sampling run gave, in order.
fn give(said: &mut Warnings, warnings: Vec<sdm_sampling::driver::output::Warning>) {
    for warning in warnings {
        said.push(warning.diagnostic());
    }
}

// --- The checks before gathering ------------------------------------------------------

/// `_statistic_asked_for(options)`.
fn statistic_asked_for(options: &[(String, PyVal)]) -> R<Option<String>> {
    let Some(given) = option(options, "prediction_statistic") else {
        return Ok(None);
    };
    match given {
        PyVal::Str(s) if crate::models::PREDICTION_STATISTICS.contains(&s.as_str()) => {
            Ok(Some(s.clone()))
        }
        _ => {
            let mut named: Vec<&str> = crate::models::PREDICTION_STATISTICS.to_vec();
            named.sort_unstable();
            Err(invalid(format!(
                "prediction_statistic is {}, and a regression reports {} of its predicted \
                 distribution",
                crate::pyops::py_repr(given),
                named
                    .iter()
                    .map(|n| sdm_pql::pyrepr::str_repr(n))
                    .collect::<Vec<_>>()
                    .join(" or ")
            )))
        }
    }
}

/// `_unreserved(column)`.
fn unreserved(column: &str) -> String {
    let suggestion = format!("{}_from_the_data", column.trim_start_matches('_'));
    if RESERVED.contains(&suggestion.as_str()) {
        format!("{column}_")
    } else {
        suggestion
    }
}

/// Whether the client needs `name` for a column of its own: a wire column,
/// or the column a composite key is folded into (`__kumo_key_...`), which a
/// frame column of that name made today's gather refuse as a key "without
/// 'ID' semantic type" the caller never named.
fn reserved(name: &str) -> bool {
    RESERVED.contains(&name) || name.starts_with(crate::gather::DERIVED_PREFIX)
}

/// `_refuse_a_name_the_wire_reserves()`, over the columns the schema
/// described and the frames as they stand: today's read only the schema, so
/// a reserved name added to a frame after `Schema.from_data` met pandas'
/// "cannot insert" or was sent.
fn refuse_a_name_the_wire_reserves(
    schema: &Schema,
    frames: &[(String, Vec<Series>, usize)],
) -> R<()> {
    let mut taken: Vec<(String, String)> = Vec::new();
    for table in &schema.tables {
        for column in &table.columns {
            if reserved(&column.name) {
                taken.push((table.name.clone(), column.name.clone()));
            }
        }
    }
    for (table, columns, _) in frames {
        for name in columns.iter().filter_map(|s| s.name.as_deref()) {
            if reserved(name) {
                taken.push((table.clone(), name.to_string()));
            }
        }
    }
    if taken.is_empty() {
        return Ok(());
    }
    taken.sort();
    taken.dedup();
    let named = sdm_schema::client::and_list(
        &taken
            .iter()
            .map(|(t, c)| format!("{t}.{c}"))
            .collect::<Vec<_>>(),
    );
    let many = taken.len() > 1;
    let renamed = if many {
        taken
            .iter()
            .map(|(_, c)| {
                format!(
                    "{} to {}",
                    sdm_pql::pyrepr::str_repr(c),
                    sdm_pql::pyrepr::str_repr(&unreserved(c))
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        format!("to {}", sdm_pql::pyrepr::str_repr(&unreserved(&taken[0].1)))
    };
    Err(structured(
        "INVALID_SCHEMA",
        format!(
            "{named} {} this client needs for the request it builds, so {} of the data. Rename \
             {}, for instance {renamed}.",
            if many { "use names" } else { "uses a name" },
            if many {
                "they cannot also be columns"
            } else {
                "it cannot also be a column"
            },
            if many { "them" } else { "it" },
        ),
    ))
}

/// `LIST_DISTINCT` of a key ranks the rows of the one table the key links
/// to. Linked to several, today's ended in the engine's
/// `NotImplementedError` (`_get_task_type`), said as `INTERNAL_ERROR`.
fn refuse_a_list_of_no_one_table(query: &Resolved, schema: &Schema) -> R<()> {
    let Some(aggregation) = &query.final_aggregation else {
        return Ok(());
    };
    if aggregation.aggr != "LIST_DISTINCT" {
        return Ok(());
    }
    let target = &aggregation.target_text;
    let (table, column) = (table_name(target), col_name(target));
    let mut linked: Vec<String> = Vec::new();
    for link in &schema.relationships {
        if link.source_table == table
            && link.source_column == column
            && !linked.contains(&link.target_table)
        {
            linked.push(link.target_table.clone());
        }
    }
    // A key linked to none is the language's to refuse, and it does.
    if linked.len() < 2 {
        return Ok(());
    }
    linked.sort();
    let message = format!(
        "{target} links to {}, so LIST_DISTINCT({target}) cannot say which table's rows it \
         ranks. Keep one of these relationships in the schema for this question",
        sdm_schema::client::and_list(&linked)
    );
    Err(crate::error::unsupported(message, Map::new()))
}

/// `_warn_two_whole_numbers_are_being_asked_as_quantities(query)`.
fn warn_two_whole_numbers(
    query: &Resolved,
    schema: &Schema,
    frames: &[(String, Vec<Series>, usize)],
    said: &mut Warnings,
) -> R<()> {
    let named = query.target_column.clone();
    let (table, column) = match named.split_once('.') {
        Some((t, c)) => (t.to_string(), c.to_string()),
        None => (named.clone(), String::new()),
    };
    let Some(described) = schema.tables.iter().find(|t| t.name == table) else {
        return Ok(());
    };
    let Some(col) = described.columns.iter().find(|c| c.name == column) else {
        return Ok(());
    };
    if col.stype != "numerical" {
        return Ok(());
    }
    let Some(series) = frames
        .iter()
        .find(|(n, _, _)| *n == table)
        .and_then(|(_, data, _)| data.iter().find(|s| s.name.as_deref() == Some(&column)))
    else {
        return Ok(());
    };
    if !crate::tabular::holds_two_whole_numbers(series)? {
        return Ok(());
    }
    said.sdm(format!(
        "{named} holds two whole numbers, which is how a binary label is usually written, and \
         this is being asked as a regression because that is what the column says it holds. If \
         they are labels, say so on the column: schema[{}][{}].stype = 'categorical'.",
        sdm_pql::pyrepr::str_repr(&table),
        sdm_pql::pyrepr::str_repr(&column)
    ));
    Ok(())
}

/// `_warn_the_filter_chose_only_the_examples(query, indices)`.
fn warn_the_filter_chose_only_the_examples(
    query: &Resolved,
    indices: &[PyVal],
    protocol: Option<&crate::materialize::SequenceProtocol>,
    said: &mut Warnings,
) -> R<()> {
    let nonempty = match protocol {
        Some(protocol) => protocol.truth()?,
        None => !indices.is_empty(),
    };
    if !nonempty {
        return Ok(());
    }
    let Some(condition) = &query.entity_filter else {
        return Ok(());
    };
    let length = match protocol {
        Some(protocol) => protocol.length()?,
        None => indices.len(),
    };
    said.sdm(format!(
        "the entity filter {} selects the labelled examples the model learns from, and does \
         not narrow the {} rows named for scoring. Those are scored as given, whether or not \
         they satisfy it. Pass only indices that satisfy the condition if that is what you \
         meant.",
        sdm_pql::pyrepr::str_repr(condition),
        length
    ));
    Ok(())
}

/// `_joins_nothing(schema, link)`.
fn joins_nothing(link: &Rel, frames: &[(String, Vec<Series>, usize)]) -> R<bool> {
    let column = |table: &str, name: &str| -> Option<&Series> {
        frames
            .iter()
            .find(|(n, _, _)| n == table)
            .and_then(|(_, data, _)| data.iter().find(|s| s.name.as_deref() == Some(name)))
    };
    let (Some(values), Some(keys)) = (
        column(&link.source_table, &link.source_column),
        column(&link.target_table, &link.target_column),
    ) else {
        return Ok(false);
    };
    let values = crate::tabular::distinct_present(values)?;
    let keys = crate::tabular::distinct_present(keys)?;
    if values.is_empty() || keys.is_empty() {
        return Ok(false);
    }
    let keyset: std::collections::HashSet<String> = keys.iter().map(isin_key).collect();
    Ok(!values.iter().any(|value| keyset.contains(&isin_key(value))))
}

/// A value as `Series.isin` matches it: numbers by value.
fn isin_key(value: &PyVal) -> String {
    match crate::pyops::item(value) {
        PyVal::Int(n) => format!("n:{}", n as f64),
        PyVal::Float(x) => format!("n:{x}"),
        PyVal::Bool(b) => format!("n:{}", f64::from(u8::from(b))),
        other => other.eq_key(),
    }
}

/// `_warn_of_relationships_that_join_nothing(schema)`.
fn warn_of_relationships_that_join_nothing(
    schema: &Schema,
    frames: &[(String, Vec<Series>, usize)],
    said: &mut Warnings,
) -> R<()> {
    let mut empty = Vec::new();
    for link in &schema.relationships {
        if schema
            .tables
            .iter()
            .any(|table| table.name == link.source_table)
            && schema
                .tables
                .iter()
                .any(|table| table.name == link.target_table)
            && joins_nothing(link, frames)?
        {
            empty.push(link);
        }
    }
    let Some(first) = empty.first() else {
        return Ok(());
    };
    let many = empty.len() > 1;
    said.sdm(format!(
        "{} {} none of the rows given: no value of the foreign key appears in the key it points \
         at, so the tables are related in the schema and in no row of the data, and the \
         question is answered as though {} were not declared. Check that the foreign key is the \
         column naming the parent row; one spelled differently from the key it points at has to \
         be said outright, as relationships=[Relationship({}, <that column>, {}, {})].",
        empty
            .iter()
            .map(|l| l.text())
            .collect::<Vec<_>>()
            .join(", "),
        if many { "join" } else { "joins" },
        if many { "they" } else { "it" },
        sdm_pql::pyrepr::str_repr(&first.source_table),
        sdm_pql::pyrepr::str_repr(&first.target_table),
        sdm_pql::pyrepr::str_repr(&first.target_column),
    ));
    Ok(())
}

/// `_run_mode_for(num_context_samples)`.
fn run_mode_for(num_context_samples: Option<i64>) -> R<&'static str> {
    let Some(n) = num_context_samples else {
        return Ok("fast");
    };
    if n <= limits::CONTEXT_ROWS_FAST {
        return Ok("fast");
    }
    if n <= limits::CONTEXT_ROWS_MAX {
        return Ok("best");
    }
    Err(structured(
        "UNSUPPORTED_REQUEST",
        format!(
            "num_context_samples is {}, and the endpoint accepts at most {} context rows",
            sdm_schema::py::thousands(n),
            sdm_schema::py::thousands(limits::CONTEXT_ROWS_MAX)
        ),
    ))
}

/// `_check_entity_count(indices, task_kind)`.
fn check_entity_count(
    count: usize,
    task_kind: Option<&str>,
    protocol: Option<&crate::materialize::SequenceProtocol>,
) -> R<()> {
    let (ceiling, described) = match task_kind {
        Some(kind) if kind.contains("link_prediction") => (
            limits::PREDICT_ROWS_TEMPORAL_LINK_PREDICTION,
            "a temporal link prediction scores at most",
        ),
        _ => (limits::PREDICT_ROWS_MAX, "the endpoint scores at most"),
    };
    if count as i64 > ceiling {
        let count = match protocol {
            Some(protocol) => protocol.length()?,
            None => count,
        };
        return Err(structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "{} entities were asked for, and {described} {} at a time; split them across \
                 calls",
                sdm_schema::py::thousands(count as i64),
                sdm_schema::py::thousands(ceiling)
            ),
        ));
    }
    Ok(())
}

/// `_check_shape(options, limits)`.
fn check_shape(options: &[(String, PyVal)]) -> R<()> {
    let neighbors = option(options, "num_neighbors").filter(|v| !matches!(v, PyVal::None));
    let neighbor_list = match neighbors {
        None => None,
        Some(PyVal::List(items) | PyVal::Tuple(items)) => Some(items.clone()),
        Some(_) => {
            return Err(invalid("num_neighbors must be a sequence of hop sizes"));
        }
    };
    let num_hops = option(options, "num_hops").filter(|v| !matches!(v, PyVal::None));
    if let Some(hops) = num_hops
        && !is_whole(hops)
    {
        return Err(invalid(format!(
            "num_hops must be a whole number, got {} {}",
            crate::models::type_name(hops),
            crate::pyops::py_repr(hops)
        )));
    }
    let seed = option(options, "random_seed").filter(|v| !matches!(v, PyVal::None));
    if let Some(seed) = seed {
        if !is_whole(seed) {
            return Err(invalid(format!(
                "random_seed must be a whole number, got {} {}",
                crate::models::type_name(seed),
                crate::pyops::py_repr(seed)
            )));
        }
        if crate::pyops::integer_value(seed).is_some_and(|d| d.negative && !d.is_zero()) {
            return Err(invalid(format!(
                "random_seed is {}, and a seed is not negative; pass a number from 0 upwards, or \
                 leave it out to let the sampler choose",
                crate::pyops::py_str(seed)
            )));
        }
    }
    let hops: Option<i64> = match (&neighbor_list, num_hops) {
        (Some(list), _) => Some(list.len() as i64),
        (None, Some(h)) => whole_of(h),
        (None, None) => None,
    };
    if let Some(hops) = hops {
        if hops < 1 {
            let spelled = if neighbor_list.is_some() {
                "num_neighbors is empty".to_string()
            } else {
                format!("num_hops is {hops}")
            };
            return Err(invalid(format!(
                "{spelled}, which asks for no hop at all, and the endpoint reads between 1 and \
                 {}. A request that crosses no relationship is a single-table one: give a hop, \
                 such as num_neighbors=[8, 8], or ask it of kumo-tabular",
                limits::HOPS_MAX
            )));
        }
        if hops > limits::HOPS_MAX {
            return Err(structured(
                "UNSUPPORTED_REQUEST",
                format!(
                    "this request samples {hops} hops and the endpoint reads at most {}",
                    limits::HOPS_MAX
                ),
            ));
        }
    }
    Ok(())
}

/// `_columns_to_leave_out(query, options)`.
fn columns_to_leave_out(
    query: &Resolved,
    options: &[(String, PyVal)],
) -> R<Vec<(String, Vec<String>)>> {
    let mut merged: Vec<(String, Vec<String>)> = query.exclude_cols.clone();
    let given = option(options, "exclude_cols_dict").filter(|v| !matches!(v, PyVal::None));
    let Some(given) = given else {
        return Ok(merged);
    };
    let PyVal::Dict(pairs) = given else {
        return Err(invalid(format!(
            "exclude_cols_dict must map a table name to the columns to leave out, got {}",
            crate::models::type_name(given)
        )));
    };
    for (table, columns) in pairs {
        let table = crate::pyops::py_str(table);
        let names: Vec<String> = match columns {
            PyVal::List(items) | PyVal::Tuple(items) => {
                items.iter().map(crate::pyops::py_str).collect()
            }
            other => {
                return Err(invalid(format!(
                    "exclude_cols_dict[{}] must be a sequence of column names, got {}",
                    sdm_pql::pyrepr::str_repr(&table),
                    crate::models::type_name(other)
                )));
            }
        };
        let at = match merged.iter().position(|(t, _)| *t == table) {
            Some(at) => at,
            None => {
                merged.push((table.clone(), Vec::new()));
                merged.len() - 1
            }
        };
        for name in names {
            if !merged[at].1.contains(&name) {
                merged[at].1.push(name);
            }
        }
    }
    Ok(merged)
}

// --- The engine's task table ------------------------------------------------------------

/// `TaskTable`: a task's context and prediction examples.
#[derive(Clone, Debug)]
pub struct TaskTable {
    pub task_type: String,
    pub context: SFrame,
    pub pred: SFrame,
    pub entity_table_names: Vec<String>,
    /// `ANCHOR_TIMESTAMP`, or `TaskTable.ENTITY_TIME`.
    pub time_column: Option<String>,
    pub step_size: Option<i64>,
    pub num_forecasts: i64,
    /// `_query`.
    pub query: String,
}

const ENTITY: &str = "ENTITY";
const TARGET: &str = "TARGET";
const ENTITY_TIME: &str = "__entity_time__";

impl TaskTable {
    /// `TaskTable(...)` of what `_get_task_table` sampled.
    fn new(done: &sdm_sampling::driver::task::TaskDone) -> Result<Self, PyExc> {
        let (context, pred) = done.frames().map_err(exc_of)?;
        if context.rows == 0 {
            return Err(PyExc::new("ValueError", "No context examples given"));
        }
        if pred.rows == 0 {
            return Err(PyExc::new(
                "ValueError",
                "Provide at least one entity to predict for",
            ));
        }
        Ok(TaskTable {
            task_type: done.task_type.clone(),
            context,
            pred,
            entity_table_names: done.entity_table_names.clone(),
            time_column: Some(done.time_column.clone()),
            step_size: done.step_size,
            num_forecasts: done.num_forecasts,
            query: String::new(),
        })
    }

    /// `num_context_examples`.
    pub fn num_context_examples(&self) -> usize {
        self.context.rows
    }

    /// `num_prediction_examples`.
    pub fn num_prediction_examples(&self) -> usize {
        self.pred.rows
    }

    /// `use_entity_time`.
    pub fn use_entity_time(&self) -> bool {
        self.time_column.as_deref() == Some(ENTITY_TIME)
    }

    /// `narrow_context(start, length)`.
    pub fn narrow_context(&self, start: usize, length: usize) -> Result<Self, PyExc> {
        let end = (start + length).min(self.context.rows);
        let positions: Vec<usize> = (start.min(end)..end).collect();
        let mut out = self.clone();
        out.context = self.context.take(&positions).map_err(exc_of)?;
        Ok(out)
    }

    /// `narrow_prediction(start, length)`.
    pub fn narrow_prediction(&self, start: usize, length: usize) -> Result<Self, PyExc> {
        let end = (start + length).min(self.pred.rows);
        let positions: Vec<usize> = (start.min(end)..end).collect();
        let mut out = self.clone();
        out.pred = self.pred.take(&positions).map_err(exc_of)?;
        Ok(out)
    }

    /// `feature_columns`' names: every column but the entity, the target and
    /// the time column.
    pub fn feature_columns(&self) -> Vec<String> {
        self.context
            .names
            .iter()
            .filter(|n| {
                n.as_str() != ENTITY
                    && n.as_str() != TARGET
                    && Some(n.as_str()) != self.time_column.as_deref()
            })
            .cloned()
            .collect()
    }

    /// `has_prediction_targets`.
    pub fn has_prediction_targets(&self) -> bool {
        self.pred.position(TARGET).is_some()
    }
}

// --- Gathering -----------------------------------------------------------------------

/// The engine, built for one request: the graph and its sampler.
struct Engine {
    sampler: Sampler,
    inputs: Inputs,
    db: Db,
    report: Vec<(String, TableReport)>,
}

impl Engine {
    /// The sampler's inputs with this call's arguments.
    fn call(&self, arguments: Map<String, Value>) -> Inputs {
        let mut inputs = self.inputs.clone();
        inputs.arguments = arguments;
        inputs
    }
}

/// An anchor argument as the sampler reads it: `'entity'`, or a time and
/// its unit, made naive UTC first (`_check_anchor_time`).
fn anchor_argument(value: Option<&PyVal>, name: &str, host: &Host) -> Result<Value, PyExc> {
    let Some(value) = value else {
        return Ok(Value::Null);
    };
    let stamp = |nanos: i128| -> Value {
        let unit = if nanos % 1_000_000_000 == 0 {
            "s"
        } else if nanos % 1_000_000 == 0 {
            "ms"
        } else if nanos % 1_000 == 0 {
            "us"
        } else {
            "ns"
        };
        json!({"ns": nanos as i64, "unit": unit})
    };
    match value {
        PyVal::None => Ok(Value::Null),
        PyVal::Timestamp(wall) => Ok(stamp(wall.epoch_nanos())),
        PyVal::TimestampTz(wall, offset) => Ok(stamp(
            wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000,
        )),
        PyVal::Str(s) if s == "entity" => Ok(json!("entity")),
        other => {
            let hint = match other {
                PyVal::Str(s) => {
                    let pql_host = sdm_pql::dates::Host {
                        year: host.year,
                        pads_year: host.pads_year,
                    };
                    if sdm_pql::dates::to_datetime(s, pql_host).value.is_some() {
                        format!("; try pd.Timestamp({})", sdm_pql::pyrepr::str_repr(s))
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            };
            Err(PyExc::new(
                "TypeError",
                format!(
                    "'{name}' must be a pandas.Timestamp or the literal 'entity' (got {} {}){hint}",
                    crate::models::type_name(other),
                    crate::pyops::py_repr(other)
                ),
            ))
        }
    }
}

/// A seed as the sampler reads it: decimal text.
fn seed_argument(options: &[(String, PyVal)]) -> Value {
    match option(options, "random_seed") {
        None => json!(RANDOM_SEED.to_string()),
        Some(PyVal::None) => Value::Null,
        Some(seed) => {
            crate::pyops::integer_value(seed).map_or(Value::Null, |d| json!(d.format_f()))
        }
    }
}

/// An `indices` value as JSON holds it.
fn index_json(value: &PyVal) -> Result<Value, PyExc> {
    match crate::pyops::item(value) {
        PyVal::None => Ok(Value::Null),
        PyVal::Bool(b) => Ok(json!(b)),
        // An int past int64, exactly: a float of it would name another key.
        PyVal::Int(n) => {
            Ok(i64::try_from(n).map_or_else(|_| json!({"$int": n.to_string()}), |i| json!(i)))
        }
        PyVal::BigInt(digits) => Ok(json!({"$int": digits})),
        PyVal::Float(x) => Ok(crate::wire::float_json(x)),
        PyVal::Str(s) => Ok(json!(s)),
        // A `Decimal`, cast to the key's dtype as pandas casts it.
        PyVal::Decimal(text) => Ok(json!({"$decimal": text})),
        other => Err(PyExc::new(
            "ValueError",
            format!(
                "an index {} that JSON does not hold",
                crate::pyops::py_repr(&other)
            ),
        )),
    }
}

/// `_get_task_table(query, indices, ...)`.
async fn get_task_table(
    engine: &mut Engine,
    query: &Resolved,
    indices: &[PyVal],
    run_mode: &str,
    options: &[(String, PyVal)],
    host: &Host,
    said: &mut Warnings,
) -> Result<TaskTable, PyExc> {
    let anchor_time = anchor_argument(option(options, "anchor_time"), "anchor_time", host)?;
    let context_anchor_time = anchor_argument(
        option(options, "context_anchor_time"),
        "context_anchor_time",
        host,
    )?;
    let whole_option = |name: &str, default: i64| -> Value {
        option(options, name)
            .and_then(whole_of)
            .map_or(json!(default), |n| json!(n))
    };
    let mut arguments = Map::new();
    arguments.insert("query".into(), query.tree.clone());
    arguments.insert(
        "indices".into(),
        Value::Array(indices.iter().map(index_json).collect::<Result<_, _>>()?),
    );
    arguments.insert("anchor_time".into(), anchor_time);
    arguments.insert("context_anchor_time".into(), context_anchor_time);
    arguments.insert("run_mode".into(), json!(run_mode));
    arguments.insert("lag_timesteps".into(), whole_option("lag_timesteps", 0));
    arguments.insert(
        "max_pq_iterations".into(),
        whole_option("max_pq_iterations", 10),
    );
    arguments.insert("random_seed".into(), seed_argument(options));
    let inputs = engine.call(arguments);
    let done =
        sdm_sampling::driver::task::get_task_table_with(&mut engine.sampler, &inputs, &engine.db)
            .await;
    let done = match done {
        Ok(done) => done,
        Err(raised) => {
            give(said, raised.warnings.clone());
            return Err(exc_of(raised));
        }
    };
    give(said, done.warnings.clone());
    TaskTable::new(&done)
}

/// What one batch of a task is gathered with.
struct Batching<'a> {
    run_mode: &'a str,
    num_neighbors: Vec<i64>,
    exclude: &'a [(String, Vec<String>)],
    top_k: Option<i64>,
    options: &'a [(String, PyVal)],
    inference_config: InferenceConfig,
}

/// `_get_context(task, ...)`: the batch's subgraph, as the envelope reads it.
async fn get_context(
    engine: &mut Engine,
    task: &TaskTable,
    how: &Batching<'_>,
    said: &mut Warnings,
) -> Result<PredictionContext, PyExc> {
    let concat = |name: &str| -> Result<sdm_sampling::frame::Series, PyExc> {
        let a = task.context.column(name).map_err(exc_of)?;
        let b = task.pred.column(name).map_err(exc_of)?;
        sdm_sampling::frame::concat_series(&[a, b]).map_err(exc_of)
    };
    let entity = concat(ENTITY)?;
    let mut inputs = engine.call(Map::new());
    let batch = |s: &sdm_sampling::frame::Series| {
        crate::sampled::series_batch(s).map_err(|why| PyExc::new("ArrowInvalid", why))
    };
    inputs
        .data
        .push(("#entity_pkey".to_string(), batch(&entity)?));
    inputs
        .argument_tables
        .insert("entity_pkey".to_string(), "#entity_pkey".to_string());
    let mut anchor_dtype = Value::Null;
    if let Some(time) = &task.time_column
        && !task.use_entity_time()
    {
        let anchor = concat(time)?;
        anchor_dtype = json!(crate::sampled::dtype_name(&anchor));
        inputs
            .data
            .push(("#anchor_time".to_string(), batch(&anchor)?));
        inputs
            .argument_tables
            .insert("anchor_time".to_string(), "#anchor_time".to_string());
    }
    let features = task.feature_columns();
    let mut feature_frame = None;
    if !features.is_empty() {
        let context = task.context.select(&features).map_err(exc_of)?;
        let pred = task.pred.select(&features).map_err(exc_of)?;
        let frame = SFrame::concat(&[&context, &pred]).map_err(exc_of)?;
        inputs.data.push((
            "#features".to_string(),
            crate::sampled::frame_batch(&frame).map_err(|why| PyExc::new("ArrowInvalid", why))?,
        ));
        inputs
            .argument_tables
            .insert("features".to_string(), "#features".to_string());
        feature_frame = Some(frame);
    }
    let mut arguments = Map::new();
    arguments.insert("task_type".into(), json!(task.task_type));
    arguments.insert("entity_table_names".into(), json!(task.entity_table_names));
    arguments.insert(
        "entity_pkey_dtype".into(),
        json!(crate::sampled::dtype_name(&entity)),
    );
    arguments.insert("num_context".into(), json!(task.num_context_examples()));
    arguments.insert("time_column".into(), json!(task.time_column));
    arguments.insert("anchor_time_dtype".into(), anchor_dtype);
    arguments.insert("run_mode".into(), json!(how.run_mode));
    arguments.insert("num_neighbors".into(), json!(how.num_neighbors));
    arguments.insert(
        "exclude_cols_dict".into(),
        Value::Object(
            how.exclude
                .iter()
                .map(|(t, c)| (t.clone(), json!(c)))
                .collect(),
        ),
    );
    arguments.insert("random_seed".into(), seed_argument(how.options));
    arguments.insert("validate_references".into(), json!(false));
    inputs.arguments = arguments;
    let done =
        sdm_sampling::driver::task::get_context_with(&mut engine.sampler, &inputs, &engine.db)
            .await;
    let done = match done {
        Ok(done) => done,
        Err(raised) => {
            give(said, raised.warnings.clone());
            return Err(exc_of(raised));
        }
    };
    give(said, done.subgraph.warnings.clone());
    let subgraph = done.subgraph;
    let tables = subgraph
        .tables
        .iter()
        .map(|t| SampledTable {
            name: t.name.clone(),
            df: crate::sampled::columns_from(&t.df),
            rows: t.df.rows,
            row: t.row.clone(),
            batch: t.batch.clone(),
            num_sampled_nodes: t.num_sampled_nodes.clone(),
            stype_dict: t.stype_dict.clone(),
            primary_key: t.primary_key.clone(),
        })
        .collect();
    let target = |frame: &SFrame| -> Result<Series, PyExc> {
        Ok(crate::sampled::series_from(
            TARGET,
            frame.column(TARGET).map_err(exc_of)?,
        ))
    };
    Ok(PredictionContext {
        task_type: task.task_type.clone(),
        entity_table_names: task.entity_table_names.clone(),
        anchor_time: subgraph.anchor_time.clone(),
        tables,
        links: subgraph.links.clone(),
        y_train: target(&task.context)?,
        y_test: if task.has_prediction_targets() {
            Some(target(&task.pred)?)
        } else {
            None
        },
        task_table: feature_frame.map(|f| crate::sampled::columns_from(&f)),
        top_k: how.top_k,
        step_size: task.step_size,
        num_forecasts: task.num_forecasts,
    })
}

/// `_resolve_task_request_options`' inference configuration: the task's
/// default, or the caller's, checked as the dataclass checks it.
fn inference_config(task_type: &str, given: Option<&PyVal>) -> Result<InferenceConfig, PyExc> {
    let mut config = InferenceConfig::from_task_type(task_type);
    if let Some((class, fields)) = given.and_then(dataclass_of) {
        // A dataclass the caller made is used as it is, whatever the task.
        config = InferenceConfig::from_task_type(match class.as_str() {
            "ClassificationInferenceConfig" => "binary_classification",
            "RegressionInferenceConfig" => "regression",
            _ => "",
        });
        return apply_fields(config, &class, &fields);
    }
    let fields: Vec<(String, PyVal)> = match given {
        None | Some(PyVal::None) => return Ok(config),
        Some(PyVal::Dict(pairs)) => pairs
            .iter()
            .map(|(k, v)| (crate::pyops::py_str(k), v.clone()))
            .collect(),
        Some(other) => {
            return Err(PyExc::new(
                "TypeError",
                format!(
                    "inference_config must be an InferenceConfig or a dict, got {}",
                    crate::models::type_name(other)
                ),
            ));
        }
    };
    let class = match config.kind.as_str() {
        "classification" => "ClassificationInferenceConfig",
        "regression" => "RegressionInferenceConfig",
        _ => "InferenceConfig",
    };
    apply_fields(config, class, &fields)
}

/// `{"$inference_config": {"class", "fields"}}`: a dataclass a caller made,
/// by its class and fields.
fn dataclass_of(given: &PyVal) -> Option<(String, Vec<(String, PyVal)>)> {
    let PyVal::Dict(items) = given else {
        return None;
    };
    let [(PyVal::Str(tag), PyVal::Dict(body))] = items.as_slice() else {
        return None;
    };
    if tag != "$inference_config" {
        return None;
    }
    let get = |k: &str| {
        body.iter()
            .find(|(key, _)| matches!(key, PyVal::Str(s) if s == k))
            .map(|(_, v)| v)
    };
    let class = match get("class") {
        Some(PyVal::Str(s)) => s.clone(),
        _ => "InferenceConfig".to_string(),
    };
    let fields = match get("fields") {
        Some(PyVal::Dict(pairs)) => pairs
            .iter()
            .map(|(k, v)| (crate::pyops::py_str(k), v.clone()))
            .collect(),
        _ => Vec::new(),
    };
    Some((class, fields))
}

/// The configuration `class(**fields)` makes, checked as the dataclass
/// checks it.
fn apply_fields(
    mut config: InferenceConfig,
    class: &str,
    fields: &[(String, PyVal)],
) -> Result<InferenceConfig, PyExc> {
    let mut errors: Vec<String> = Vec::new();
    for (name, value) in fields {
        let flag = |v: &PyVal| match crate::pyops::item(v) {
            PyVal::Bool(b) => Some(b),
            PyVal::Int(0) => Some(false),
            PyVal::Int(1) => Some(true),
            _ => None,
        };
        match (name.as_str(), config.kind.as_str()) {
            ("num_estimators", _) => match whole_of(value) {
                Some(n) if (1..=4).contains(&n) && !matches!(value, PyVal::Bool(_)) => {
                    config.num_estimators = n;
                }
                _ => errors.push(format!(
                    "{name}\n  Input should be a valid integer between 1 and 4"
                )),
            },
            ("column_shuffle", _) => match flag(value) {
                Some(b) => config.column_shuffle = b,
                None => errors.push(format!("{name}\n  Input should be a valid boolean")),
            },
            ("category_shuffle", _) => match flag(value) {
                Some(b) => config.category_shuffle = b,
                None => errors.push(format!("{name}\n  Input should be a valid boolean")),
            },
            ("hop_shuffle", _) => match flag(value) {
                Some(b) => config.hop_shuffle = b,
                None => errors.push(format!("{name}\n  Input should be a valid boolean")),
            },
            ("class_shuffle", "classification") => match flag(value) {
                Some(b) => config.class_shuffle = b,
                None => errors.push(format!("{name}\n  Input should be a valid boolean")),
            },
            ("target_transforms", "regression") => match value {
                PyVal::List(items) | PyVal::Tuple(items) => {
                    config.target_transforms = items
                        .iter()
                        .map(|v| match v {
                            PyVal::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                }
                _ => errors.push(format!("{name}\n  Input should be a valid list")),
            },
            ("output_type", "regression") => match value {
                PyVal::Str(s) if matches!(s.as_str(), "mean" | "median" | "quantiles") => {
                    config.output_type = s.clone();
                }
                _ => errors.push(format!(
                    "{name}\n  Input should be 'mean', 'median' or 'quantiles'"
                )),
            },
            _ => errors.push(format!("{name}\n  Unexpected keyword argument")),
        }
    }
    if errors.is_empty() {
        return Ok(config);
    }
    Err(PyExc::new(
        "ValidationError",
        format!(
            "{} validation error{} for {class}\n{}",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            errors.join("\n")
        ),
    ))
}

/// `context_size_stats(context)`, as far as the core counts a frame's
/// memory: the sampled rows, edges and tables.
fn context_size_stats(context: &PredictionContext) -> String {
    let mut stats: Vec<(String, i64, usize)> = context
        .tables
        .iter()
        .map(|t| {
            let size: usize = t.df.iter().map(crate::sampled::memory_usage).sum::<usize>() + 132;
            (t.name.clone(), t.num_rows(), size)
        })
        .collect();
    stats.sort_by_key(|(_, _, size)| std::cmp::Reverse(*size));
    let nodes: i64 = stats.iter().map(|(_, n, _)| n).sum();
    let edges: i64 = context
        .links
        .iter()
        .map(|l| l.num_sampled_edges.iter().sum::<i64>())
        .sum();
    let top: Vec<String> = stats
        .iter()
        .take(5)
        .map(|(name, _, size)| format!("{name}: {:.2}MB", *size as f64 / (1024.0 * 1024.0)))
        .collect();
    format!(
        "Current context contains {} nodes and {} edges across {} tables. Top-{} tables \
         contributing most to the context size: {}",
        sdm_schema::py::thousands(nodes),
        sdm_schema::py::thousands(edges),
        stats.len(),
        top.len(),
        top.join(", ")
    )
}

/// `materialize_task(task, ...)` and `_iter_task_requests`: the task's one
/// batch, gathered and encoded.
async fn materialize_task(
    engine: &mut Engine,
    task: TaskTable,
    how: Batching<'_>,
    said: &mut Warnings,
) -> Result<Vec<Value>, PyExc> {
    let options = how.options;
    let entity_table = task.entity_table_names.first().cloned().unwrap_or_default();
    let entity = {
        let a = task.context.column(ENTITY).map_err(exc_of)?;
        let b = task.pred.column(ENTITY).map_err(exc_of)?;
        sdm_sampling::frame::concat_series(&[a, b]).map_err(exc_of)?
    };
    engine
        .sampler
        .validate_entity_references(&entity_table, &entity)
        .map_err(exc_of)?;
    let max_ctx = max_context_size(how.run_mode);
    let mut task = task;
    if task.num_context_examples() > max_ctx {
        task = task.narrow_context(0, max_ctx)?;
    }
    if task.task_type == "forecasting" && task.num_forecasts > task.num_context_examples() as i64 {
        return Err(PyExc::new(
            "ValueError",
            format!(
                "The number of forecast steps ({}) exceeds the number of available in-context \
                 examples ({}). Please provide more historical data or reduce the number of \
                 forecast steps.",
                sdm_schema::py::thousands(task.num_forecasts),
                sdm_schema::py::thousands(task.num_context_examples() as i64)
            ),
        ));
    }
    let batch_size = task.num_prediction_examples();
    let max_batch = max_pred_size(&task.task_type);
    if batch_size > max_batch {
        return Err(PyExc::new(
            "ValueError",
            format!(
                "Cannot predict for more than {} entities at once (got {}). Pass `batch_size=` to \
                 `predict(...)` (for example `batch_size={}`, or `batch_size='max'`) to process \
                 entities in batches.",
                sdm_schema::py::thousands(max_batch as i64),
                sdm_schema::py::thousands(batch_size as i64),
                max_batch.min(500)
            ),
        ));
    }
    let mut payloads = Vec::new();
    let mut start = 0;
    let mut batch_index = 0;
    while start < task.num_prediction_examples() {
        let stop = (start + batch_size).min(task.num_prediction_examples());
        let batch_task = task.narrow_prediction(start, batch_size)?;
        let mut context = get_context(engine, &batch_task, &how, said).await?;
        context.y_test = None;
        let request = PredictRequest {
            run_mode: how.run_mode.to_string(),
            query: task.query.clone(),
            use_prediction_time: option(options, "use_prediction_time")
                .is_some_and(crate::pyops::py_bool),
            inference_config: Some(how.inference_config.clone()),
            return_embeddings: option(options, "return_embeddings")
                .is_some_and(crate::pyops::py_bool),
        };
        let payload = predict_request_to_json(&context, &request, false)?;
        let rows = payload
            .py_get("predict")
            .and_then(|p| p.py_get("instance_table"))
            .and_then(|t| t.py_get("rows"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if rows != stop - start {
            return Err(PyExc::new(
                "RuntimeError",
                format!(
                    "Request batch {batch_index} serialized {} prediction rows; expected {} \
                     from its source range.",
                    sdm_schema::py::thousands(rows as i64),
                    sdm_schema::py::thousands((stop - start) as i64)
                ),
            ));
        }
        let size = crate::size::value_bytes(&payload)
            .map_err(|e| PyExc::new("ValueError", e.to_string()))?;
        if size > MAX_SIZE {
            return Err(PyExc::new(
                "ValueError",
                SIZE_LIMIT_MSG.replace("{stats}", &context_size_stats(&context)),
            ));
        }
        validate_payload_table_rows(&payload, batch_index)?;
        payloads.push(payload);
        start += batch_size.max(1);
        batch_index += 1;
    }
    Ok(payloads)
}

/// `_check_limits(part, limits)`.
fn check_limits(payload: &Value) -> R<()> {
    let Some(rows) = payload
        .py_get("predict")
        .and_then(|p| p.py_get("instance_table"))
        .and_then(|t| t.py_get("rows"))
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    let kind = payload
        .py_get("task")
        .and_then(|t| t.py_get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let cap = if kind == "temporal_link_prediction" {
        limits::PREDICT_ROWS_TEMPORAL_LINK_PREDICTION
    } else {
        limits::PREDICT_ROWS_MAX
    };
    if rows.len() as i64 > cap {
        return Err(structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "this batch scores {} rows and the endpoint accepts at most {cap} for a {kind} task",
                rows.len()
            ),
        ));
    }
    if let Some(Value::Object(carried)) = payload
        .py_get("schema")
        .and_then(|s| s.py_get("related_tables"))
        && carried.len() as i64 > limits::TABLES_MAX
    {
        return Err(structured(
            "UNSUPPORTED_REQUEST",
            format!(
                "this request carries {} tables and the endpoint reads at most {}; sample fewer \
                 hops, or keep fewer relationships in the schema",
                carried.len(),
                limits::TABLES_MAX
            ),
        ));
    }
    Ok(())
}

/// `_warn_rows_the_graph_dropped(model)`.
fn warn_rows_the_graph_dropped(report: &[(String, TableReport)], said: &mut Warnings) {
    let detail = sanitization_detail(report);
    if detail.is_empty() {
        return;
    }
    said.sdm(format!(
        "the graph was built without some of the rows given to it {}. Those rows are not in the \
         context the model sees; fill the missing values or drop the rows yourself if that is \
         not what you meant.",
        detail.trim()
    ));
}

/// What the materializer gathers with.
pub struct Relational<'a> {
    pub query: &'a Resolved,
    pub handle: &'a DataSchema,
    pub schema: &'a Schema,
    pub tables: &'a Tables,
    pub host: &'a Host,
    pub indices: Vec<PyVal>,
    pub indices_protocol: Option<&'a crate::materialize::SequenceProtocol>,
    pub num_context_samples: Option<i64>,
    pub options: &'a [(String, PyVal)],
    /// The query's model (`RelationalMaterializer._model`): the sampler of an
    /// earlier materialize of the same query, which this one goes on with,
    /// and where this one leaves it.
    pub model: &'a mut Option<Model>,
    /// The process's `_coverage_warned`, which the host keeps.
    pub coverage_warned: &'a mut bool,
}

/// A query's relational model as it is kept between its materializes
/// (`RelationalMaterializer._model`): the sampler's description and what it
/// learnt, and, over frames, the graph's own tables. Today's model holds the
/// graph it built at the query's first gather, so a frame changed after that
/// does not reach a later gather of the same query; these tables are built
/// from the frames' cells, and share no buffer with them.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub json: Value,
    pub tables: Vec<(String, RecordBatch)>,
}

impl Model {
    /// The name the graph's table `name` is given among a call's tables.
    pub fn table_key(name: &str) -> String {
        format!("model/{name}")
    }
}

/// Frames by table name: each one's columns, and its number of rows.
type NamedFrames = Vec<(String, Vec<Series>, usize)>;

/// The frames of the schema's tables, as `_as_typed` gives them, and what
/// their column labels are beyond their text.
fn typed_frames(
    schema: &Schema,
    handle: &DataSchema,
    tables: &Tables,
    host: &Host,
) -> R<(NamedFrames, crate::gather::Labels)> {
    let frames = handle.frames(tables).map_err(crate::error::value_error)?;
    let mut out = Vec::new();
    let mut labels = crate::gather::Labels::default();
    for described in &schema.tables {
        let Some((_, frame)) = frames.iter().find(|(n, _)| *n == described.name) else {
            continue;
        };
        let columns = crate::gather::as_typed(described, frame, host)?;
        out.push((described.name.clone(), columns, frame.len()));
        if frame.multi_index() {
            labels.multi_index.push(described.name.clone());
        }
        for column in frame.columns() {
            if frame.tag(&column).py_get("name_repr").is_some() {
                labels.not_text.push((described.name.clone(), column));
            }
        }
    }
    Ok((out, labels))
}

/// The frames as given, before any is typed.
fn given_frames(
    schema: &Schema,
    handle: &DataSchema,
    tables: &Tables,
) -> R<Vec<(String, Vec<Series>, usize)>> {
    if handle.source.is_some() {
        // A table in its source has no rows here to look at.
        return Ok(Vec::new());
    }
    let frames = handle.frames(tables).map_err(crate::error::value_error)?;
    Ok(schema
        .tables
        .iter()
        .filter_map(|described| {
            let (_, frame) = frames.iter().find(|(n, _)| *n == described.name)?;
            let columns = frame
                .columns()
                .iter()
                .filter_map(|name| frame.series(name))
                .collect();
            Some((described.name.clone(), columns, frame.len()))
        })
        .collect())
}

/// `_get_model()`: the graph built from the schema, and the sampler over it.
async fn get_model(asker: &Asker, gathering: &Relational<'_>, said: &mut Warnings) -> R<Engine> {
    let mut checked = sdm_schema::client::Said::default();
    let mut schema = gathering.schema.clone();
    schema
        .check_consistent(&mut checked)
        .map_err(crate::request::from_schema)?;
    for warning in checked.warnings {
        said.push(Diagnostic::warning(&warning.category, warning.message));
    }
    if let Some(source) = &gathering.handle.source {
        return over_a_source(asker, source, &schema, gathering.host, said).await;
    }
    let (frames, labels) =
        typed_frames(&schema, gathering.handle, gathering.tables, gathering.host)?;
    let graph: EngineGraph =
        match crate::gather::build_graph(&schema, frames, &labels, gathering.host, said) {
            Ok(graph) => graph,
            Err(Built::Refused(refusal)) => return Err(refusal),
            Err(Built::Engine(error)) => return Err(engine_error(error, &[])),
        };
    graph.validate().map_err(|e| engine_error(e, &[]))?;
    let inputs = graph.inputs().map_err(|e| engine_error(e, &[]))?;
    let sampler = Sampler::new(&inputs, gathering.host.clone())
        .map_err(|raised| engine_error(exc_of(raised), &[]))?;
    give(said, sampler.built_warnings());
    let report = sampler.sanitization_report();
    Ok(Engine {
        sampler,
        inputs,
        db: Db::new(asker.clone()),
        report,
    })
}

/// The model as the query keeps it between materializes: the sampler's
/// description and what it learnt, and the graph's tables.
fn model_of(engine: &Engine) -> Model {
    let mut inputs = engine.inputs.clone();
    inputs.state = engine.sampler.state();
    for (table, rows) in engine.sampler.listed() {
        if let Some(info) = inputs.tables.iter_mut().find(|t| t.name == table)
            && let Some(sql) = info.sql.as_mut()
        {
            sql.listed = sdm_dialects::py::Py::List(
                rows.into_iter().map(sdm_dialects::py::Py::Dict).collect(),
            )
            .encode()
            .ok();
        }
    }
    let tables = std::mem::take(&mut inputs.data);
    Model {
        json: inputs.to_json(),
        tables,
    }
}

/// The query's model again, as [`model_of`] kept it: over frames, the graph
/// it built, with the sampler's state as it was left.
fn engine_from_model(asker: &Asker, gathering: &Relational<'_>, model: &Model) -> R<Engine> {
    let inputs = Inputs::parse(&model.json.to_string(), model.tables.clone())
        .map_err(|raised| engine_error(exc_of(raised), &[]))?;
    let sampler = Sampler::new(&inputs, gathering.host.clone())
        .map_err(|raised| engine_error(exc_of(raised), &[]))?;
    let report = sampler.sanitization_report();
    Ok(Engine {
        sampler,
        inputs,
        db: Db::new(asker.clone()),
        report,
    })
}

/// [`get_model`] over tables that stay in their source.
async fn over_a_source(
    asker: &Asker,
    source: &crate::schemas::SourceSchema,
    schema: &Schema,
    host: &Host,
    said: &mut Warnings,
) -> R<Engine> {
    let db = sdm_schema::engine::Db(asker.clone());
    let mut ctx = sdm_schema::engine::Ctx {
        db: &db,
        host,
        warnings: sdm_schema::engine::Warnings::default(),
    };
    let built = crate::source_gather::build_graph(&mut ctx, source, schema).await;
    for (category, message) in std::mem::take(&mut ctx.warnings.0) {
        said.push(sdm_schema::diagnostics::warning(&category, message));
    }
    let graph = match built {
        Ok(graph) => graph,
        Err(crate::source_gather::Refused::Client(refusal)) => return Err(refusal),
        Err(crate::source_gather::Refused::Engine(error)) => return Err(engine_error(error, &[])),
    };
    let made = crate::source_gather::sampler(&mut ctx, asker, graph, host).await;
    for (category, message) in std::mem::take(&mut ctx.warnings.0) {
        said.push(sdm_schema::diagnostics::warning(&category, message));
    }
    let (sampler, inputs, warnings) = made.map_err(|e| engine_error(e, &[]))?;
    give(said, warnings);
    Ok(Engine {
        sampler,
        inputs,
        db: Db::new(asker.clone()),
        report: Vec::new(),
    })
}

/// `RelationalMaterializer.materialize(query, indices, num_context_samples,
/// options)`.
pub async fn materialize(
    asker: &Asker,
    gathering: Relational<'_>,
    said: &mut Warnings,
) -> R<Request> {
    let options = gathering.options;
    let mut unknown: Vec<String> = options
        .iter()
        .map(|(k, _)| k.clone())
        .filter(|k| !MATERIALIZE_OPTIONS.contains(&k.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        unknown.dedup();
        let accepted: Vec<String> = MATERIALIZE_OPTIONS.iter().map(|s| s.to_string()).collect();
        return Err(invalid(format!(
            "unknown materialize option(s) {}; accepted: {}",
            sdm_schema::py::str_list_repr(&unknown),
            sdm_schema::py::str_list_repr(&accepted)
        )));
    }
    let indices = gathering.indices.clone();
    let length = || match gathering.indices_protocol {
        Some(protocol) => protocol.length(),
        None => Ok(indices.len()),
    };
    if length()? == 0 {
        return Err(invalid(
            "indices names no entity, and gathering a relational request means sampling a \
             neighbourhood around each one, so there is nothing here to gather. An empty batch \
             is still answered: build it as Request(predict=<no rows>) and predict() returns the \
             empty result without sending",
        ));
    }
    let query = gathering.query;
    let schema = gathering.schema;
    let statistic = statistic_asked_for(options)?;
    let given = given_frames(schema, gathering.handle, gathering.tables)?;
    refuse_a_name_the_wire_reserves(schema, &given)?;
    refuse_a_list_of_no_one_table(query, schema)?;
    warn_two_whole_numbers(query, schema, &given, said)?;
    warn_the_filter_chose_only_the_examples(query, &indices, gathering.indices_protocol, said)?;
    warn_of_relationships_that_join_nothing(schema, &given, said)?;
    drop(given);

    // `_gather(query, indices, num_context_samples, options)`.
    let num_context_samples = gathering.num_context_samples;
    let run_mode = run_mode_for(num_context_samples)?;
    check_entity_count(length()?, None, gathering.indices_protocol)?;
    check_shape(options)?;
    let mut engine = match gathering.model.clone() {
        Some(model) => engine_from_model(asker, &gathering, &model)?,
        None => get_model(asker, &gathering, said).await?,
    };
    // The sampler says it found too few examples once in a process.
    engine.sampler.coverage_warned = *gathering.coverage_warned;
    // Today's query keeps the model its gather built even when the gather
    // is then refused, so a later gather goes on with that graph.
    let gathered: R<Request> = async {
        warn_rows_the_graph_dropped(&engine.report, said);
        let report = engine.report.clone();
        let indices = match gathering.indices_protocol {
            Some(protocol) => protocol.items("list")?,
            None => indices.clone(),
        };
        let mut task = get_task_table(
            &mut engine,
            query,
            &indices,
            run_mode,
            options,
            gathering.host,
            said,
        )
        .await
        .map_err(|e| {
            emptied_since(&engine.sampler, &e).unwrap_or_else(|| engine_error(e, &report))
        })?;
        task.query = query.text.clone();
        check_entity_count(length()?, Some(&task.task_type), gathering.indices_protocol)?;
        if num_context_samples.is_none()
            && task.num_context_examples() as i64 >= limits::CONTEXT_ROWS_FAST
        {
            said.sdm(format!(
                "{} context examples were gathered, which is all this route takes by default; pass \
                 num_context_samples to ask for up to {}.",
                sdm_schema::py::thousands(limits::CONTEXT_ROWS_FAST),
                sdm_schema::py::thousands(limits::CONTEXT_ROWS_MAX)
            ));
        }
        if let Some(n) = num_context_samples
            && task.num_context_examples() > n as usize
        {
            task = task
                .narrow_context(0, n as usize)
                .map_err(|e| engine_error(e, &report))?;
        }
        let top_k = crate::materialize::candidates_to_rank(query, options)?;
        let exclude = columns_to_leave_out(query, options)?;
        let payloads = {
            let how = resolve_options(&task, run_mode, options, top_k, &exclude, said)
                .map_err(|e| engine_error(e, &report))?;
            materialize_task(&mut engine, task, how, said)
                .await
                .map_err(|e| engine_error(e, &report))?
        };
        if payloads.is_empty() {
            return Err(invalid("the engine gathered nothing for this query"));
        }
        for payload in &payloads {
            check_limits(payload)?;
        }
        let mut payloads = payloads;
        if let Some(statistic) = statistic {
            record_the_statistic(&mut payloads, &statistic)?;
        }
        let entity_table = table_name(&query.entity_column);
        let sent = tables_sent(&payloads, schema);
        warn_of_tables_left_out(schema, &sent, &entity_table, said);
        warn_of_columns_the_model_does_not_read(schema, &exclude, &sent, said);
        let windowed = query.windowed;
        drop_context_that_answers_the_question(&mut payloads, windowed, said);
        materialized(
            payloads,
            gathering.handle.clone(),
            query.text.clone(),
            indices,
            Some(entity_table),
            windowed,
            option(options, "inference_config"),
        )
    }
    .await;
    *gathering.coverage_warned = engine.sampler.coverage_warned;
    *gathering.model = Some(model_of(&engine));
    gathered
}

/// `_resolve_task_request_options(task, ...)`: the run mode, the hops and
/// the inference configuration.
fn resolve_options<'a>(
    task: &TaskTable,
    run_mode: &'a str,
    options: &'a [(String, PyVal)],
    top_k: Option<i64>,
    exclude: &'a [(String, Vec<String>)],
    said: &mut Warnings,
) -> Result<Batching<'a>, PyExc> {
    let num_hops = option(options, "num_hops").and_then(whole_of).unwrap_or(2);
    let given = option(options, "num_neighbors").filter(|v| !matches!(v, PyVal::None));
    if num_hops != 2 && given.is_some() {
        said.user(format!(
            "Received custom 'num_neighbors' option; ignoring custom 'num_hops={num_hops}' option"
        ));
    }
    let num_neighbors = match given {
        Some(PyVal::List(items) | PyVal::Tuple(items)) => items
            .iter()
            .map(|v| {
                whole_of(v).ok_or_else(|| {
                    PyExc::new(
                        "TypeError",
                        format!(
                            "'{}' object cannot be interpreted as an integer",
                            crate::models::type_name(v)
                        ),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            let key = if crate::cells::is_link_pred(&task.task_type) {
                "fast"
            } else {
                run_mode
            };
            let defaults = default_num_neighbors(key);
            if !(1..=defaults.len() as i64).contains(&num_hops) {
                return Err(PyExc::new(
                    "ValueError",
                    format!(
                        "'num_hops' must be between 1 and {} (got {num_hops})",
                        defaults.len()
                    ),
                ));
            }
            defaults.into_iter().take(num_hops as usize).collect()
        }
    };
    let inference_config = inference_config(&task.task_type, option(options, "inference_config"))?;
    Ok(Batching {
        run_mode,
        num_neighbors,
        exclude,
        top_k,
        options,
        inference_config,
    })
}

/// `_record_the_statistic(payloads, statistic)`.
fn record_the_statistic(payloads: &mut [Value], statistic: &str) -> R<()> {
    for payload in payloads.iter_mut() {
        let kind = payload
            .py_get("task")
            .and_then(|t| t.py_get("kind"))
            .map(|k| crate::pyops::py_str(&crate::keys::json_cell(k)))
            .unwrap_or_default();
        if !crate::models::SUMMARISED_TASK_KINDS.contains(&kind.as_str()) {
            return Err(invalid(format!(
                "prediction_statistic applies only to a regression, and this question is a \
                 {kind}; ask it without prediction_statistic"
            )));
        }
        if let Some(object) = payload.as_object_mut() {
            let output = object.entry("output").or_insert_with(|| json!({}));
            if let Some(output) = output.as_object_mut() {
                output.insert("prediction_statistic".into(), json!(statistic));
            }
        }
    }
    Ok(())
}

/// `_tables_sent(payloads, schema)`.
fn tables_sent(payloads: &[Value], schema: &Schema) -> Vec<String> {
    let mut carried: Vec<String> = Vec::new();
    let mut described = false;
    for payload in payloads {
        if let Some(Value::Object(related)) = payload
            .py_get("schema")
            .and_then(|s| s.py_get("related_tables"))
        {
            described = true;
            for name in related.keys() {
                if !carried.contains(name) {
                    carried.push(name.clone());
                }
            }
        }
    }
    if described {
        carried
    } else {
        schema.tables.iter().map(|t| t.name.clone()).collect()
    }
}

/// `_tables_linked_to(schema, entity_table)`.
fn tables_linked_to(schema: &Schema, entity_table: &str) -> Vec<String> {
    let mut seen = vec![entity_table.to_string()];
    let mut queue = vec![entity_table.to_string()];
    while let Some(table) = queue.pop() {
        for link in &schema.relationships {
            let nearby = if link.source_table == table {
                &link.target_table
            } else if link.target_table == table {
                &link.source_table
            } else {
                continue;
            };
            if !seen.contains(nearby) {
                seen.push(nearby.clone());
                queue.push(nearby.clone());
            }
        }
    }
    seen
}

/// `_warn_of_tables_left_out(schema, sent, entity_table)`.
fn warn_of_tables_left_out(
    schema: &Schema,
    sent: &[String],
    entity_table: &str,
    said: &mut Warnings,
) {
    let mut missing: Vec<String> = schema
        .tables
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !sent.contains(n))
        .collect();
    if missing.is_empty() {
        return;
    }
    missing.sort();
    let linked = tables_linked_to(schema, entity_table);
    let unlinked: Vec<&String> = missing.iter().filter(|n| !linked.contains(n)).collect();
    let unreached: Vec<&String> = missing.iter().filter(|n| linked.contains(n)).collect();
    let names = |list: &[&String]| {
        list.iter()
            .map(|n| sdm_pql::pyrepr::str_repr(n))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !unlinked.is_empty() {
        let many = unlinked.len() > 1;
        let (them, are, it_is, joins) = if many {
            ("them", "are", "they are", "they join")
        } else {
            ("it", "is", "it is", "it joins")
        };
        said.sdm(format!(
            "{} {are} in the schema and not in this request: no relationship connects {them} to \
             {}, so the model answers without {them}. Say how {joins}: \
             schema.relationships.append(Relationship({}, <foreign key>, <parent table>, <its \
             primary key>)), or leave {them} out of the schema to say {it_is} not wanted.",
            names(&unlinked),
            sdm_pql::pyrepr::str_repr(entity_table),
            sdm_pql::pyrepr::str_repr(unlinked[0]),
        ));
    }
    if !unreached.is_empty() {
        let many = unreached.len() > 1;
        let (them, are, lies, does) = if many {
            ("them", "are", "they lie", "they do")
        } else {
            ("it", "is", "it lies", "it does")
        };
        said.sdm(format!(
            "{} {are} linked into the schema and not in this request: the sampler reached no row \
             of {them} from the entities being scored, so the model answers without {them}. \
             Raise num_hops if {lies} further out than the hops sampled, and check the foreign \
             key holds values the parent key actually has if {does} not.",
            names(&unreached),
        ));
    }
}

/// `_warn_of_columns_the_model_does_not_read(schema, left_out, sent)`.
fn warn_of_columns_the_model_does_not_read(
    schema: &Schema,
    left_out: &[(String, Vec<String>)],
    sent: &[String],
    said: &mut Warnings,
) {
    let unread = ["text"];
    let mut ignored: Vec<String> = Vec::new();
    for table in &schema.tables {
        if !sent.contains(&table.name) {
            continue;
        }
        let excluded: &[String] = left_out
            .iter()
            .find(|(t, _)| *t == table.name)
            .map_or(&[], |(_, c)| c.as_slice());
        for column in &table.columns {
            if unread.contains(&column.stype.as_str()) && !excluded.contains(&column.name) {
                ignored.push(format!("{}.{}", table.name, column.name));
            }
        }
    }
    if ignored.is_empty() {
        return;
    }
    ignored.sort();
    let many = ignored.len() > 1;
    said.sdm(format!(
        "{} {} typed {}, which this model does not read, so the request is answered without {} \
         and {} nothing to the prediction. Type a column categorical if its values are labels \
         rather than prose, or leave it out to say so.",
        ignored.join(", "),
        if many { "are" } else { "is" },
        unread.join(" or "),
        if many { "them" } else { "it" },
        if many {
            "they contribute"
        } else {
            "it contributes"
        },
    ));
}

/// A wire cell as a key a Python set compares it by.
fn cell_key(value: &Value) -> String {
    match value {
        Value::Number(n) => format!("n:{}", n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => format!("n:{}", u8::from(*b)),
        Value::String(s) => format!("s:{s}"),
        Value::Null => "none".to_string(),
        other => format!("o:{other}"),
    }
}

/// `_positions(table, anchor_column)`.
fn positions(table: &Value, anchor_column: &str) -> Option<(usize, Option<usize>, Vec<Value>)> {
    let columns = table.py_get("columns")?.as_array()?;
    let rows = table.py_get("rows")?.as_array()?;
    let reference = columns.iter().position(|c| {
        c.as_str()
            .is_some_and(|s| s.starts_with(crate::payload::ENTITY_REFERENCE_PREFIX))
    })?;
    let anchor = columns
        .iter()
        .position(|c| c.as_str() == Some(anchor_column));
    Some((reference, anchor, rows.clone()))
}

/// `_drop_context_that_answers_the_question(payloads, windowed=windowed)`.
fn drop_context_that_answers_the_question(
    payloads: &mut [Value],
    windowed: bool,
    said: &mut Warnings,
) {
    let mut found: Vec<(usize, Vec<String>)> = Vec::new();
    for (at, payload) in payloads.iter().enumerate() {
        let (Some(context), Some(predict)) = (payload.py_get("context"), payload.py_get("predict"))
        else {
            continue;
        };
        if !context.is_object() || !predict.is_object() {
            continue;
        }
        let anchor_column = payload
            .py_get("task")
            .and_then(|t| t.py_get("anchor_time_column"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(crate::payload::ANCHOR_TIME_PREFIX)
            .to_string();
        // `_entities_at_anchors(predict.instance_table, ...)`.
        let mut asked: Vec<(String, String)> = Vec::new();
        if let Some(table) = predict.py_get("instance_table")
            && let Some((reference, anchor, rows)) = positions(table, &anchor_column)
            && (!windowed || anchor.is_some())
        {
            let needed = if windowed {
                reference.max(anchor.unwrap_or(0))
            } else {
                reference
            };
            for row in &rows {
                let Some(cells) = row.as_array() else {
                    continue;
                };
                if cells.len() <= needed {
                    continue;
                }
                let when = if windowed {
                    cell_key(&cells[anchor.unwrap_or(0)])
                } else {
                    "none".to_string()
                };
                let pair = (cell_key(&cells[reference]), when);
                if !asked.contains(&pair) {
                    asked.push(pair);
                }
            }
        }
        if asked.is_empty() {
            continue;
        }
        // `_instances_matching(context.instance_table, asked, ...)`.
        let mut answered: Vec<String> = Vec::new();
        if let Some(table) = context.py_get("instance_table")
            && let Some((reference, anchor, rows)) = positions(table, &anchor_column)
            && let Some(at_id) = table
                .py_get("columns")
                .and_then(Value::as_array)
                .and_then(|c| c.iter().position(|n| n.as_str() == Some("instance_id")))
            && (!windowed || anchor.is_some())
        {
            let needed = if windowed {
                reference.max(at_id).max(anchor.unwrap_or(0))
            } else {
                reference.max(at_id)
            };
            for row in &rows {
                let Some(cells) = row.as_array() else {
                    continue;
                };
                if cells.len() <= needed {
                    continue;
                }
                let when = if windowed {
                    cell_key(&cells[anchor.unwrap_or(0)])
                } else {
                    "none".to_string()
                };
                if asked.contains(&(cell_key(&cells[reference]), when)) {
                    let id = cell_key(&cells[at_id]);
                    if !answered.contains(&id) {
                        answered.push(id);
                    }
                }
            }
        }
        if answered.is_empty() {
            continue;
        }
        found.push((at, answered));
    }
    if found.is_empty() {
        return;
    }
    let total: usize = found.iter().map(|(_, a)| a.len()).sum();
    let when = if windowed {
        "at the same moment"
    } else {
        "at one moment or another"
    };
    said.sdm(format!(
        "{total} of the rows sampled as context are the rows being scored, {when}, carrying the \
         value being asked for. Rows that carry the answer to the question being asked are not \
         sent as context, so these do not go into it."
    ));
    for (at, answered) in found {
        let Some(context) = payloads[at].get_mut("context") else {
            continue;
        };
        if let Some(table) = context.get_mut("instance_table") {
            remove_instances(table, &answered);
        }
        if let Some(Value::Object(related)) = context.get_mut("related_tables") {
            for table in related.values_mut() {
                remove_instances(table, &answered);
            }
        }
    }
}

/// `_remove_instances(table, instances)`.
fn remove_instances(table: &mut Value, instances: &[String]) {
    let Some(at) = table
        .py_get("columns")
        .and_then(Value::as_array)
        .and_then(|c| c.iter().position(|n| n.as_str() == Some("instance_id")))
    else {
        return;
    };
    if let Some(Value::Array(rows)) = table.get_mut("rows") {
        rows.retain(|row| {
            !row.as_array()
                .is_some_and(|cells| cells.len() > at && instances.contains(&cell_key(&cells[at])))
        });
    }
}

/// `_entity_table_in(payloads)`: the table the query asks about, as the
/// envelopes record it (`task.entity_table_names`), or `''`.
pub fn entity_table_in(payloads: &[Value]) -> String {
    for payload in payloads {
        if let Ok(payload) = envelope_mapping(payload)
            && let Some(task) = payload.py_get("task")
            && let Ok(task) = envelope_mapping(task)
            && let Some(named) = task.py_get("entity_table_names")
            && let Ok(PyVal::List(names) | PyVal::Tuple(names)) = PyVal::decode(named)
            && let Some(first) = names.first()
        {
            return first.str_();
        }
    }
    String::new()
}

/// The request a gather made of its envelopes: `Request.materialized` as
/// `materialize` calls it, with what the caller asked of the inference
/// config.
fn materialized(
    payloads: Vec<Value>,
    schema: DataSchema,
    query: String,
    indices: Vec<PyVal>,
    entity_table: Option<String>,
    windowed: bool,
    inference_config: Option<&PyVal>,
) -> R<Request> {
    let config = match inference_config {
        None | Some(PyVal::None) => json!({}),
        Some(given) => match dataclass_of(given) {
            Some((class, fields)) => json!({"$inference_config": {
                "class": class,
                "fields": fields.iter().map(|(k, v)| (k.clone(), crate::layout::encode_deep(v))).collect::<Map<String, Value>>(),
            }}),
            None => crate::layout::encode_deep(given),
        },
    };
    let config = match config {
        Value::Object(_) => config,
        _ => json!({}),
    };
    let mut requested = Map::new();
    requested.insert("inference_config".into(), config);
    from_envelopes(
        payloads,
        schema,
        Some(query),
        indices,
        entity_table,
        windowed,
        requested,
    )
}

/// `Request.materialized(payloads=..., schema=..., query=..., indices=...,
/// entity_table=..., windowed=..., requested_inference=...)`: a request of
/// envelopes, its task and inference read off the first one, and
/// `entity_table`, where not given, off the first that names one.
/// `requested_inference` is what the caller asked, each value as tagged
/// JSON, kept as given.
pub fn from_envelopes(
    payloads: Vec<Value>,
    schema: DataSchema,
    query: Option<String>,
    indices: Vec<PyVal>,
    entity_table: Option<String>,
    windowed: bool,
    requested_inference: impl Into<Value>,
) -> R<Request> {
    from_envelope_values(
        payloads,
        schema,
        query.map(PyVal::Str),
        PyVal::List(indices),
        entity_table.map(PyVal::Str),
        PyVal::Bool(windowed),
        (requested_inference.into(), None),
    )
}

pub fn from_envelope_values(
    payloads: Vec<Value>,
    schema: DataSchema,
    query: Option<PyVal>,
    indices: PyVal,
    entity_table: Option<PyVal>,
    windowed: PyVal,
    inference_protocols: (Value, Option<crate::pyops::HostProtocols>),
) -> R<Request> {
    let (requested_inference, protocols) = inference_protocols;
    let entity_table = if let Some(entity_table) = entity_table {
        entity_table
    } else {
        let mut entity_table = String::new();
        for payload in &payloads {
            let task = if crate::pyops::is_instance(
                payload,
                crate::pyops::InstanceFamily::Mapping,
                protocols.as_ref(),
            )? {
                crate::pyops::mapping_get(payload, "task", &Value::Null, protocols.as_ref())?
            } else {
                Value::Null
            };
            let task = if crate::pyops::truth_with(
                &PyVal::decode(&task).map_err(crate::error::value_error)?,
                protocols.as_ref(),
            )? {
                task
            } else {
                json!({})
            };
            let named = crate::pyops::mapping_get(
                &task,
                "entity_table_names",
                &Value::Null,
                protocols.as_ref(),
            )?;
            if crate::pyops::is_instance(
                &named,
                crate::pyops::InstanceFamily::ListTuple,
                protocols.as_ref(),
            )? && crate::pyops::truth_with(
                &PyVal::decode(&named).map_err(crate::error::value_error)?,
                protocols.as_ref(),
            )? {
                let first = crate::pyops::getitem(&named, &PyVal::Int(0), protocols.as_ref())?;
                entity_table = PyVal::decode(&first)
                    .map_err(crate::error::value_error)?
                    .str_();
                break;
            }
        }
        PyVal::Str(entity_table)
    };
    let first = payloads
        .first()
        .ok_or_else(|| builtin("IndexError", "tuple index out of range"))?;
    let empty = json!({});
    let get = |value: &Value, key: &str, default: &Value| {
        crate::pyops::mapping_get(value, key, default, protocols.as_ref())
    };
    let target = get(&get(first, "task", &empty)?, "target", &empty)?;
    let kind = get(
        &get(first, "task", &empty)?,
        "kind",
        &json!("classification"),
    )?;
    let column_name = get(&target, "column_name", &json!(""))?;
    let dtype = get(&target, "dtype", &json!(""))?;
    let mut held_options = None;
    let mut held_inference = None;
    let options = envelope_dict_with(
        Some(&get(&get(first, "task", &empty)?, "options", &empty)?),
        protocols.as_ref(),
        &mut held_options,
    )?;
    let inference = envelope_dict_with(
        Some(&get(first, "inference", &empty)?),
        protocols.as_ref(),
        &mut held_inference,
    )?;
    Ok(Request {
        protocols: None,
        coerced_mappings: [("options", held_options), ("inference", held_inference)]
            .into_iter()
            .filter_map(|(name, held)| held.map(|value| (name.into(), value)))
            .collect(),
        task: Task {
            kind: PyVal::decode(&kind).map_err(crate::error::value_error)?,
            target: Target {
                column_name: PyVal::decode(&column_name).map_err(crate::error::value_error)?,
                dtype: PyVal::decode(&dtype).map_err(crate::error::value_error)?,
                table_name: None,
            },
            entity_table: Some(entity_table.clone()),
            options,
        },
        schema,
        entity_table: Some(entity_table),
        query,
        inference,
        requested_inference: {
            let given = requested_inference;
            let given = if crate::pyops::truth_with(
                &PyVal::decode(&given).map_err(crate::error::value_error)?,
                protocols.as_ref(),
            )? {
                given
            } else {
                json!({})
            };
            let mut held = None;
            let pairs = envelope_dict_with(Some(&given), protocols.as_ref(), &mut held)?;
            held.unwrap_or_else(|| crate::layout::encode_mapping(&pairs))
        },
        halves: None,
        payloads: Some(payloads),
        indices,
        windowed,
        parts: None,
        decomposition: None,
        asked: None,
        context_fingerprint: None,
    })
}

fn envelope_mapping(value: &Value) -> R<Map<String, Value>> {
    match PyVal::decode(value)
        .map_err(crate::error::value_error)?
        .semantic()
    {
        PyVal::Dict(mapping) => Ok(mapping
            .iter()
            .filter_map(|(key, value)| match key.semantic() {
                PyVal::Str(key) => Some((key.clone(), crate::layout::encode_deep(value))),
                _ => None,
            })
            .collect()),
        other => Err(builtin(
            "AttributeError",
            format!(
                "'{}' object has no attribute 'get'",
                crate::models::type_name(other)
            ),
        )),
    }
}

#[cfg(test)]
fn envelope_dict(value: Option<&Value>) -> R<Vec<(PyVal, PyVal)>> {
    envelope_dict_with(value, None, &mut None)
}

fn envelope_dict_with(
    value: Option<&Value>,
    protocols: Option<&crate::pyops::HostProtocols>,
    held: &mut Option<Value>,
) -> R<Vec<(PyVal, PyVal)>> {
    let decoded = value
        .map(PyVal::decode)
        .transpose()
        .map_err(crate::error::value_error)?;
    if let Some(given @ PyVal::Held { .. }) = &decoded {
        let copied = crate::pyops::mapping_copy(&crate::layout::encode_deep(given), protocols)?;
        let view = PyVal::decode(&copied).map_err(crate::error::value_error)?;
        let PyVal::Dict(pairs) = view.semantic() else {
            return Err(crate::error::value_error(
                "a mapping copy fact must carry a mapping view",
            ));
        };
        if !matches!(view, PyVal::Held { .. }) {
            return Err(crate::error::value_error(
                "a mapping copy fact must retain its owner",
            ));
        }
        *held = Some(copied);
        return Ok(pairs.clone());
    }
    let items = match decoded {
        None => return Ok(Vec::new()),
        Some(PyVal::Dict(mapping)) => return Ok(mapping),
        Some(PyVal::List(items) | PyVal::Tuple(items)) => items,
        Some(PyVal::Str(text)) => text
            .chars()
            .map(|character| PyVal::Str(character.to_string()))
            .collect(),
        Some(other) => {
            return Err(builtin(
                "TypeError",
                format!(
                    "'{}' object is not iterable",
                    crate::models::type_name(&other)
                ),
            ));
        }
    };
    let host_keys = items.iter().any(|item| match item {
        PyVal::List(pair) | PyVal::Tuple(pair) => {
            pair.first().is_some_and(crate::pyops::has_host_value)
        }
        PyVal::Dict(pair) => pair
            .first()
            .is_some_and(|(key, _)| crate::pyops::has_host_value(key)),
        _ => false,
    });
    let mut dictionary = Value::Null;
    let mut mapping: Vec<(PyVal, PyVal)> = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        let pair = match item {
            PyVal::List(pair) | PyVal::Tuple(pair) => pair,
            PyVal::Str(text) => text
                .chars()
                .map(|character| PyVal::Str(character.to_string()))
                .collect(),
            PyVal::Dict(pair) => pair.into_iter().map(|(key, _)| key).collect(),
            _ => {
                return Err(builtin(
                    "TypeError",
                    format!(
                        "cannot convert dictionary update sequence element #{index} to a sequence"
                    ),
                ));
            }
        };
        if pair.len() != 2 {
            return Err(builtin(
                "ValueError",
                format!(
                    "dictionary update sequence element #{index} has length {}; 2 is required",
                    pair.len()
                ),
            ));
        }
        let key = &pair[0];
        if host_keys {
            let protocols = protocols.ok_or_else(|| {
                crate::error::value_error("mapping assignment requires host protocol facts")
            })?;
            let response = protocols.call(
                "mapping_set",
                &json!([
                    dictionary,
                    crate::layout::encode_deep(key),
                    crate::layout::encode_deep(&pair[1])
                ]),
            )?;
            let decoded = PyVal::decode(&response).map_err(crate::error::value_error)?;
            let PyVal::Dict(pairs) = decoded.semantic() else {
                return Err(crate::error::value_error(
                    "a mapping assignment fact must carry a mapping view",
                ));
            };
            if !matches!(decoded, PyVal::Held { .. }) || pairs.len() > mapping.len() + 1 {
                return Err(crate::error::value_error(
                    "a mapping assignment fact must retain its owner and insertion bounds",
                ));
            }
            mapping = pairs.clone();
            dictionary = response.clone();
            *held = Some(response);
            continue;
        }
        crate::pyops::require_hashable(key)?;
        let key = key.clone();
        let value = pair[1].clone();
        if let Some((_, previous)) = mapping
            .iter_mut()
            .find(|(existing, _)| crate::result::py_eq(existing, &key))
        {
            *previous = value;
        } else {
            mapping.push((key, value));
        }
    }
    Ok(mapping)
}

/// `KumoRelational(graph).materialize_task(task)` with its defaults: a task
/// table made by the caller, gathered over a graph made by the caller.
pub async fn materialize_task_over(
    asker: &Asker,
    graph: &EngineGraph,
    task: TaskTable,
    host: &Host,
) -> Result<Vec<Value>, PyExc> {
    graph.validate()?;
    let inputs = graph.inputs()?;
    let sampler = Sampler::new(&inputs, host.clone()).map_err(exc_of)?;
    let report = sampler.sanitization_report();
    let mut engine = Engine {
        sampler,
        inputs,
        db: Db::new(asker.clone()),
        report,
    };
    let mut quiet = Warnings::default();
    let how = resolve_options(&task, "fast", &[], None, &[], &mut quiet)?;
    materialize_task(&mut engine, task, how, &mut quiet).await
}

impl TaskTable {
    /// `TaskTable(task_type, context_df, pred_df, entity_table_name,
    /// 'ENTITY', 'TARGET', time_column)` of frames made by the caller.
    pub fn of_frames(
        task_type: &str,
        context: SFrame,
        pred: SFrame,
        entity_table: &str,
        time_column: Option<&str>,
    ) -> Result<Self, PyExc> {
        if context.rows == 0 {
            return Err(PyExc::new("ValueError", "No context examples given"));
        }
        if pred.rows == 0 {
            return Err(PyExc::new(
                "ValueError",
                "Provide at least one entity to predict for",
            ));
        }
        Ok(TaskTable {
            task_type: task_type.to_string(),
            context,
            pred,
            entity_table_names: vec![entity_table.to_string()],
            time_column: time_column.map(str::to_string),
            step_size: None,
            num_forecasts: 1,
            query: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn join_frame(table: &str, dtype: &str, cells: Vec<PyVal>) -> (String, Vec<Series>, usize) {
        let rows = cells.len();
        (
            table.into(),
            vec![Series::new(
                Some("id".into()),
                crate::series::Dtype::parse(dtype),
                cells,
            )],
            rows,
        )
    }

    #[test]
    fn relationship_unique_refuses_unsupported_arrow_kernel_before_empty_values() {
        let link = Rel::new("child", "id", "parent", "id");
        for dtype in ["decimal32(8, 2)", "decimal64(16, 2)"] {
            for cells in [
                vec![],
                vec![PyVal::None],
                vec![PyVal::Decimal("1.00".into())],
            ] {
                for parent_decimal in [false, true] {
                    let source = if parent_decimal {
                        "int64".into()
                    } else {
                        format!("{dtype}[pyarrow]")
                    };
                    let target = if parent_decimal {
                        format!("{dtype}[pyarrow]")
                    } else {
                        "int64".into()
                    };
                    let frames = vec![
                        join_frame("child", &source, cells.clone()),
                        join_frame("parent", &target, cells.clone()),
                    ];
                    let error = joins_nothing(&link, &frames).unwrap_err();
                    assert_eq!(
                        error.class.as_deref(),
                        Some("pyarrow.lib.ArrowNotImplementedError")
                    );
                    assert_eq!(error.code, None);
                    assert_eq!(
                        error.message,
                        format!("Function 'unique' has no kernel matching input types ({dtype})")
                    );
                    assert!(error.chain.is_empty());
                    assert!(error.details.is_empty());
                }
            }
        }
    }

    #[test]
    fn relationship_unique_preserves_missing_input_and_child_first_order() {
        let link = Rel::new("child", "id", "parent", "id");
        let child = join_frame(
            "child",
            "decimal32(8, 2)[pyarrow]",
            vec![PyVal::Decimal("1.00".into())],
        );
        assert!(!joins_nothing(&link, std::slice::from_ref(&child)).unwrap());
        let mut parent = join_frame(
            "parent",
            "decimal64(16, 2)[pyarrow]",
            vec![PyVal::Decimal("1.00".into())],
        );
        assert!(!joins_nothing(&link, std::slice::from_ref(&parent)).unwrap());
        parent.1[0].name = Some("other".into());
        assert!(!joins_nothing(&link, &[child.clone(), parent.clone()]).unwrap());
        parent.1[0].name = Some("id".into());
        let mut unnamed_child = child.clone();
        unnamed_child.1[0].name = Some("other".into());
        assert!(!joins_nothing(&link, &[unnamed_child, parent.clone()]).unwrap());
        assert!(
            joins_nothing(&link, &[child, parent])
                .unwrap_err()
                .message
                .ends_with("(decimal32(8, 2))")
        );
    }

    #[test]
    fn relationship_unique_keeps_supported_numeric_membership() {
        let link = Rel::new("child", "id", "parent", "id");
        for dtype in [
            "int64",
            "decimal128(20, 2)[pyarrow]",
            "decimal256(40, 2)[pyarrow]",
        ] {
            for (cells, expected) in [
                (vec![], false),
                (vec![PyVal::None], false),
                (vec![PyVal::Int(1)], false),
                (vec![PyVal::Int(2)], true),
            ] {
                let frames = vec![
                    join_frame("child", dtype, cells),
                    join_frame("parent", dtype, vec![PyVal::Int(1)]),
                ];
                assert_eq!(joins_nothing(&link, &frames).unwrap(), expected);
                let empty_parent = vec![
                    join_frame("child", dtype, vec![PyVal::Int(1)]),
                    join_frame("parent", dtype, vec![]),
                ];
                assert!(!joins_nothing(&link, &empty_parent).unwrap());
            }
        }
    }

    #[test]
    fn envelope_dictionary_keys_keep_python_identity_and_equality() {
        let value = json!([
            [1, "one"],
            ["1", "text"],
            [true, "bool"],
            [null, "none"],
            [1.0, "float"]
        ]);
        let pairs = envelope_dict(Some(&value)).expect("valid pairs");
        assert_eq!(
            pairs,
            vec![
                (PyVal::Int(1), PyVal::Str("float".into())),
                (PyVal::Str("1".into()), PyVal::Str("text".into())),
                (PyVal::None, PyVal::Str("none".into())),
            ]
        );
        let encoded = crate::layout::encode_mapping(&pairs);
        assert_eq!(PyVal::decode(&encoded).unwrap(), PyVal::Dict(pairs));
    }

    #[test]
    fn envelope_metadata_and_typed_mappings_survive_request_round_trip() {
        let schema =
            DataSchema::from_json(&json!({"kind": "data", "tables": [], "options": {}})).unwrap();
        for value in [
            Value::Null,
            json!(false),
            json!(0),
            json!([]),
            json!({}),
            json!(["x"]),
            json!({"$tuple": [false]}),
            json!({"$map": [[true, {"$tuple": [1]}]]}),
            json!({"$int": "18446744073709551617"}),
            json!({"$dict": {"$tuple": [false]}}),
        ] {
            let payload = json!({
                "task": {
                    "kind": value,
                    "target": {"column_name": value, "dtype": value},
                    "options": [[true, "first"], [1, "last"], ["1", "text"]],
                },
                "inference": [[null, "none"]],
            });
            let request = from_envelopes(
                vec![payload],
                schema.clone(),
                None,
                Vec::new(),
                Some("users".into()),
                false,
                Map::new(),
            )
            .unwrap();
            let mut tables = crate::schemas::Tables::default();
            let restored = Request::from_json(&request.to_json(&mut tables), &tables).unwrap();
            assert_eq!(restored.task, request.task);
            assert_eq!(restored.inference, request.inference);
            assert_eq!(restored.task.kind, PyVal::decode(&value).unwrap());
            assert_eq!(
                restored.task.target.column_name,
                PyVal::decode(&value).unwrap()
            );
            assert_eq!(restored.task.target.dtype, PyVal::decode(&value).unwrap());
        }
    }

    /// What `_entity_table_in` gives, each line as today's Python gives it.
    #[test]
    fn the_entity_table_is_read_off_the_first_envelope_that_names_one() {
        let named = |names: Value| json!({"task": {"entity_table_names": names}});
        assert_eq!(
            entity_table_in(&[named(json!(["users", "orders"]))]),
            "users"
        );
        assert_eq!(
            entity_table_in(&[
                json!({"task": {}}),
                named(json!([])),
                named(json!(["shops"]))
            ]),
            "shops"
        );
        assert_eq!(entity_table_in(&[named(json!([7]))]), "7");
        assert_eq!(entity_table_in(&[named(json!("users"))]), "");
        assert_eq!(entity_table_in(&[json!("not an envelope")]), "");
        assert_eq!(entity_table_in(&[]), "");
    }

    /// `Request.materialized` reads its task and inference off the first
    /// envelope, its entity table off the first that names one where the
    /// caller names none, and keeps what was asked as it was given.
    #[test]
    fn a_request_of_envelopes_reads_what_it_was_not_given_off_them() {
        let schema = DataSchema::from_json(&json!({
            "kind": "data", "tables": [], "options": {}
        }))
        .expect("a schema");
        let envelope = |names: Value| {
            json!({
                "task": {
                    "kind": "regression",
                    "target": {"column_name": "TARGET", "dtype": "float32"},
                    "entity_table_names": names,
                },
                "inference": {"run_mode": "fast"},
            })
        };
        let asked: Map<String, Value> = [("inference_config".to_string(), Value::Null)]
            .into_iter()
            .collect();
        let request = from_envelopes(
            vec![envelope(json!([])), envelope(json!(["users"]))],
            schema.clone(),
            None,
            Vec::new(),
            None,
            false,
            asked.clone(),
        )
        .expect("valid envelopes");
        assert_eq!(
            request.entity_table.as_ref().and_then(PyVal::as_str),
            Some("users")
        );
        assert_eq!(
            request.task.entity_table.as_ref().and_then(PyVal::as_str),
            Some("users")
        );
        assert_eq!(request.task.kind, PyVal::Str("regression".into()));
        assert_eq!(request.task.target.column_name, PyVal::Str("TARGET".into()));
        assert_eq!(
            request.inference,
            vec![(PyVal::Str("run_mode".into()), PyVal::Str("fast".into()))]
        );
        assert_eq!(request.requested_inference, Value::Object(asked));
        assert_eq!(request.query, None);
        let named = from_envelopes(
            vec![envelope(json!([]))],
            schema,
            Some("q".to_string()),
            Vec::new(),
            Some("shops".to_string()),
            true,
            Map::new(),
        )
        .expect("valid envelopes");
        assert_eq!(
            named.entity_table.as_ref().and_then(PyVal::as_str),
            Some("shops")
        );
        assert_eq!(named.windowed, PyVal::Bool(true));
    }

    /// `_refuse_a_name_the_wire_reserves` reads the columns the schema
    /// described and those its frames hold now.
    #[test]
    fn a_name_the_wire_reserves_is_refused_where_the_schema_or_a_frame_has_it() {
        let table = |name: &str, columns: &[&str]| sdm_schema::client::Table {
            name: name.to_string(),
            data: None,
            source_name: None,
            columns: columns
                .iter()
                .map(|c| sdm_schema::client::Col {
                    name: c.to_string(),
                    dtype: "int64".to_string(),
                    stype: "numerical".to_string(),
                    untyped: false,
                    stype_declared: false,
                    before_key: None,
                    name_repr: None,
                })
                .collect(),
            primary_key: None,
            time_column: None,
            duplicate_keys: None,
        };
        let plain = Schema {
            tables: vec![table("users", &["user_id", "age"])],
            relationships: Vec::new(),
        };
        assert!(refuse_a_name_the_wire_reserves(&plain, &[]).is_ok());
        let column = |name: &str| {
            Series::new(
                Some(name.to_string()),
                crate::series::Dtype::Int("int64".into()),
                vec![],
            )
        };
        let added = refuse_a_name_the_wire_reserves(
            &plain,
            &[(
                "users".into(),
                vec![column("user_id"), column("instance_id")],
                0,
            )],
        )
        .unwrap_err();
        assert!(
            added.message.starts_with("'users.instance_id' uses a name"),
            "{}",
            added.message
        );
        let taken = Schema {
            tables: vec![
                table("users", &["user_id", "instance_id"]),
                table("orders", &["__kumo_anchor_time"]),
            ],
            relationships: Vec::new(),
        };
        let refused = refuse_a_name_the_wire_reserves(&taken, &[]).unwrap_err();
        assert_eq!(refused.code.as_deref(), Some("INVALID_SCHEMA"));
        assert_eq!(
            refused.message,
            "'orders.__kumo_anchor_time' and 'users.instance_id' use names this client needs \
             for the request it builds, so they cannot also be columns of the data. Rename \
             them, for instance '__kumo_anchor_time' to 'kumo_anchor_time_from_the_data', \
             'instance_id' to 'instance_id_from_the_data'."
        );
    }
}
