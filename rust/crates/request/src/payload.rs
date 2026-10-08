// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `rfm/inference/payload.py`: the relational envelope.

use std::collections::BTreeSet;

use serde_json::Value;

/// `_payload_tables_with_names(payload)`.
fn tables_with_names(payload: &Value) -> Vec<(String, &Value)> {
    let instance_name = payload
        .get("task")
        .and_then(|t| t.get("entity_table_names"))
        .and_then(Value::as_array)
        .and_then(|n| n.first())
        .and_then(Value::as_str)
        .unwrap_or("instance_table")
        .to_string();
    let mut out = Vec::new();
    for section in ["context", "predict"] {
        let Some(Value::Object(half)) = payload.get(section) else {
            continue;
        };
        if let Some(instance @ Value::Object(_)) = half.get("instance_table") {
            out.push((instance_name.clone(), instance));
        }
        if let Some(Value::Object(related)) = half.get("related_tables") {
            for (name, table) in related {
                if table.is_object() {
                    out.push((name.clone(), table));
                }
            }
        }
    }
    out
}

/// `high_cardinality_columns(payload, limit)`: text columns of more than
/// `limit` distinct values, most first.
pub fn high_cardinality_columns(payload: &Value, limit: usize) -> Vec<(String, String, usize)> {
    let mut counts: Vec<((String, String), usize)> = Vec::new();
    for (table_name, table) in tables_with_names(payload) {
        let (Some(columns), Some(rows)) = (
            table.get("columns").and_then(Value::as_array),
            table.get("rows").and_then(Value::as_array),
        ) else {
            continue;
        };
        for (index, column) in columns.iter().enumerate() {
            let mut distinct: BTreeSet<String> = BTreeSet::new();
            for row in rows {
                let Some(cells) = row.as_array() else {
                    continue;
                };
                let Some(cell) = cells.get(index) else {
                    continue;
                };
                match cell {
                    Value::String(s) => {
                        distinct.insert(s.clone());
                    }
                    Value::Array(items) => {
                        for item in items {
                            if let Value::String(s) = item {
                                distinct.insert(s.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
            if distinct.len() > limit {
                let name = column
                    .as_str()
                    .map_or_else(|| column.to_string(), str::to_string);
                let key = (table_name.clone(), name);
                match counts.iter_mut().find(|(k, _)| *k == key) {
                    Some(entry) => entry.1 = entry.1.max(distinct.len()),
                    None => counts.push((key, distinct.len())),
                }
            }
        }
    }
    let mut out: Vec<(String, String, usize)> =
        counts.into_iter().map(|((t, c), n)| (t, c, n)).collect();
    out.sort_by_key(|entry| std::cmp::Reverse(entry.2));
    out
}

// --- The envelope (`predict_request_to_json`) ------------------------------------------

use sdm_schema::engine::PyExc;
use sdm_schema::pyval::PyVal;
use serde_json::{Map, json};

use crate::cells::{
    cell_json_value, class_label, dtype_name, is_link_pred, target_json_value, to_json,
};
use crate::series::{Dtype, Series};

pub const INSTANCE_ID: &str = "instance_id";
pub const SYNTHETIC_NODE_ID: &str = "__node_id";
pub const ENTITY_REFERENCE_PREFIX: &str = "__kumo_entity_ref";
pub const ANCHOR_TIME_PREFIX: &str = "__kumo_anchor_time";
/// `REVERSE_RELATION_PREFIX`.
pub const REVERSE_RELATION_PREFIX: &str = "__###REV###__";
/// `MAX_TABLE_ROWS`.
pub const MAX_TABLE_ROWS: usize = 10_000;
/// `pd.Timestamp.min.value`.
const TIMESTAMP_MIN: i64 = i64::MIN + 1;

/// What the engine raises.
type ER<T> = Result<T, PyExc>;

fn value_error<T>(message: impl Into<String>) -> ER<T> {
    Err(PyExc::new("ValueError", message))
}

/// A diagnostic of the cell rules as the `ValueError` they raise.
fn raised(diagnostic: sdm_protocol::Diagnostic) -> PyExc {
    PyExc::new(&crate::error::short_class(&diagnostic), diagnostic.message)
}

/// A `SampledTable`, its frame read as request columns.
#[derive(Clone, Debug)]
pub struct SampledTable {
    pub name: String,
    pub df: Vec<Series>,
    pub rows: usize,
    pub row: Option<Vec<i64>>,
    pub batch: Vec<i64>,
    pub num_sampled_nodes: Vec<i64>,
    pub stype_dict: Vec<(String, String)>,
    pub primary_key: Option<String>,
}

impl SampledTable {
    /// `num_rows`.
    pub fn num_rows(&self) -> i64 {
        self.num_sampled_nodes.iter().sum()
    }

    fn column(&self, name: &str) -> Option<&Series> {
        self.df.iter().find(|s| s.name.as_deref() == Some(name))
    }
}

/// `PredictionContext`.
#[derive(Clone, Debug)]
pub struct PredictionContext {
    pub task_type: String,
    pub entity_table_names: Vec<String>,
    /// `sampled_subgraph.anchor_time`.
    pub anchor_time: Vec<i64>,
    pub tables: Vec<SampledTable>,
    pub links: Vec<sdm_sampling::driver::output::SampledLink>,
    pub y_train: Series,
    pub y_test: Option<Series>,
    /// The task's feature columns, context rows first.
    pub task_table: Option<Vec<Series>>,
    pub top_k: Option<i64>,
    pub step_size: Option<i64>,
    pub num_forecasts: i64,
}

impl PredictionContext {
    /// `num_train`.
    pub fn num_train(&self) -> usize {
        self.y_train.len()
    }

    /// `sampled_subgraph.batch_size`.
    pub fn batch_size(&self) -> usize {
        self.anchor_time.len()
    }

    /// `num_test`.
    pub fn num_test(&self) -> usize {
        self.batch_size().saturating_sub(self.num_train())
    }

    fn table(&self, name: &str) -> Option<&SampledTable> {
        self.tables.iter().find(|t| t.name == name)
    }

    fn target_name(&self) -> String {
        self.y_train
            .name
            .clone()
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "TARGET".to_string())
    }
}

/// An inference configuration: `InferenceConfig` and its subclasses.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceConfig {
    /// `base`, `classification` or `regression`.
    pub kind: String,
    pub num_estimators: i64,
    pub column_shuffle: bool,
    pub category_shuffle: bool,
    pub hop_shuffle: bool,
    pub class_shuffle: bool,
    pub target_transforms: Vec<Option<String>>,
    pub output_type: String,
}

impl InferenceConfig {
    fn base(kind: &str) -> Self {
        Self {
            kind: kind.to_string(),
            num_estimators: 1,
            column_shuffle: false,
            category_shuffle: false,
            hop_shuffle: false,
            class_shuffle: false,
            target_transforms: vec![Some("quantile".to_string())],
            output_type: "median".to_string(),
        }
    }

    /// `InferenceConfig.from_task_type(task_type)`.
    pub fn from_task_type(task_type: &str) -> Self {
        if is_classification(task_type) {
            Self::base("classification")
        } else if matches!(task_type, "regression" | "forecasting") {
            Self::base("regression")
        } else {
            Self::base("base")
        }
    }

    /// `_inference_config_to_json(config)`.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("kind".into(), json!("base"));
        out.insert("num_estimators".into(), json!(self.num_estimators));
        out.insert("column_shuffle".into(), json!(self.column_shuffle));
        out.insert("category_shuffle".into(), json!(self.category_shuffle));
        out.insert("hop_shuffle".into(), json!(self.hop_shuffle));
        match self.kind.as_str() {
            "classification" => {
                out.insert("kind".into(), json!("classification"));
                out.insert("class_shuffle".into(), json!(self.class_shuffle));
            }
            "regression" => {
                out.insert("kind".into(), json!("regression"));
                out.insert("target_transforms".into(), json!(self.target_transforms));
                out.insert("output_type".into(), json!(self.output_type));
            }
            _ => {}
        }
        Value::Object(out)
    }
}

/// `TaskType(...).is_classification`.
pub fn is_classification(task_type: &str) -> bool {
    matches!(
        task_type,
        "binary_classification" | "multiclass_classification" | "multilabel_classification"
    )
}

/// `RFMPredictRequest`, as far as the envelope reads it.
#[derive(Clone, Debug)]
pub struct PredictRequest {
    pub run_mode: String,
    pub query: String,
    pub use_prediction_time: bool,
    pub inference_config: Option<InferenceConfig>,
    pub return_embeddings: bool,
}

/// `_output_fields(task_type, ...)`.
pub fn output_fields(
    task_type: &str,
    return_embeddings: bool,
    explain: bool,
    quantile_output: bool,
) -> Vec<String> {
    let mut fields = vec![if is_link_pred(task_type) {
        "rankings"
    } else if quantile_output {
        "quantiles"
    } else {
        "prediction"
    }];
    if is_classification(task_type) {
        fields.push("probabilities");
    }
    if return_embeddings {
        fields.push("embeddings");
    }
    if explain {
        fields.push("explanation");
    }
    fields.into_iter().map(str::to_string).collect()
}

/// A frame the envelope is built from: its columns in order, each with
/// its cells.
#[derive(Clone, Debug, Default)]
struct Frame {
    columns: Vec<Series>,
    rows: usize,
}

impl Frame {
    fn position(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|s| s.name.as_deref() == Some(name))
    }

    fn names(&self) -> Vec<String> {
        self.columns
            .iter()
            .map(|s| s.name.clone().unwrap_or_default())
            .collect()
    }

    /// `df[name] = series`: replaced where it is, else added last.
    fn set(&mut self, series: Series) {
        match self.position(series.name.as_deref().unwrap_or("")) {
            Some(at) => self.columns[at] = series,
            None => self.columns.push(series),
        }
    }

    /// `df.insert(at, name, series)`, which pandas refuses for a name the
    /// frame already has.
    fn insert(&mut self, at: usize, series: Series) -> ER<()> {
        let name = series.name.clone().unwrap_or_default();
        if self.position(&name).is_some() {
            return value_error(format!("cannot insert {name}, already exists"));
        }
        self.columns.insert(at, series);
        Ok(())
    }

    /// `df.iloc[positions]`.
    fn take(&self, positions: &[usize]) -> Frame {
        Frame {
            columns: self
                .columns
                .iter()
                .map(|s| {
                    Series::new(
                        s.name.clone(),
                        s.dtype.clone(),
                        positions.iter().map(|&i| s.cells[i].clone()).collect(),
                    )
                })
                .collect(),
            rows: positions.len(),
        }
    }

    fn drop(&self, name: &str) -> Frame {
        Frame {
            columns: self
                .columns
                .iter()
                .filter(|s| s.name.as_deref() != Some(name))
                .cloned()
                .collect(),
            rows: self.rows,
        }
    }
}

fn int64_series(name: &str, values: impl IntoIterator<Item = i64>) -> Series {
    Series::new(
        Some(name.to_string()),
        Dtype::Int("int64".to_string()),
        values
            .into_iter()
            .map(|v| PyVal::Int(i128::from(v)))
            .collect(),
    )
}

/// `_unique_internal_column(occupied, prefix)`.
fn unique_internal_column(occupied: &[String], prefix: &str) -> String {
    if !occupied.iter().any(|o| o == prefix) {
        return prefix.to_string();
    }
    let mut suffix = 1;
    while occupied.iter().any(|o| *o == format!("{prefix}_{suffix}")) {
        suffix += 1;
    }
    format!("{prefix}_{suffix}")
}

/// `_entity_values(context, table_name)`.
fn entity_values(context: &PredictionContext, table_name: &str) -> Option<(Vec<PyVal>, Dtype)> {
    let table = context.table(table_name)?;
    let key = table.primary_key.as_deref()?;
    let key_values = table.column(key)?;
    let mut first: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (position, instance) in table.batch.iter().enumerate() {
        first.entry(*instance).or_insert(position);
    }
    let mut values = Vec::new();
    for instance in 0..context.batch_size() as i64 {
        let Some(&position) = first.get(&instance) else {
            values.push(PyVal::None);
            continue;
        };
        let row_index = match &table.row {
            None => position as i64,
            Some(row) => row[position],
        };
        if row_index < 0 || row_index as usize >= key_values.len() {
            values.push(PyVal::None);
            continue;
        }
        values.push(key_values.cells[row_index as usize].clone());
    }
    Some((values, key_values.dtype.clone()))
}

fn entity_reference(name: &str, values: Vec<PyVal>, source_dtype: &Dtype) -> Series {
    let width = match source_dtype {
        Dtype::Float(bits @ (16 | 32)) => Some(*bits),
        Dtype::Nullable(dtype) if dtype == "Float32" => Some(32),
        _ => None,
    };
    let numeric = !values.is_empty() && values.iter().all(|value| matches!(value, PyVal::Float(_)));
    let mut series = crate::sampled::series_of_values(name, values);
    if let Some(bits) = width
        && numeric
    {
        series.dtype = Dtype::Float(bits);
    }
    series
}

/// The instance frame, each entity table's reference column, and the
/// anchor column where there is one.
type Instances = (Frame, Vec<(String, String)>, Option<String>);

/// `_instance_dataframe(context)`: the instances, their entity references,
/// anchor times and targets.
fn instance_dataframe(context: &PredictionContext) -> ER<Instances> {
    let batch_size = context.batch_size();
    let mut df = Frame {
        columns: vec![int64_series(INSTANCE_ID, 0..batch_size as i64)],
        rows: batch_size,
    };
    let target_name = context.target_name();
    let mut occupied = vec![INSTANCE_ID.to_string(), target_name.clone()];
    if let Some(features) = &context.task_table {
        occupied.extend(features.iter().map(|s| s.name.clone().unwrap_or_default()));
    }
    let mut references = Vec::new();
    for (i, table_name) in context.entity_table_names.iter().enumerate() {
        let Some((values, source_dtype)) = entity_values(context, table_name) else {
            return value_error(format!(
                "Entity table {} does not expose a primary key for Universal payload \
                 materialization.",
                sdm_pql::pyrepr::str_repr(table_name)
            ));
        };
        let column = unique_internal_column(&occupied, &format!("{ENTITY_REFERENCE_PREFIX}_{i}"));
        occupied.push(column.clone());
        references.push((table_name.clone(), column.clone()));
        df.set(entity_reference(&column, values, &source_dtype));
    }
    let mut anchor_column = None;
    if !context.anchor_time.is_empty() && !context.anchor_time.iter().all(|&t| t == TIMESTAMP_MIN) {
        let column = unique_internal_column(&occupied, ANCHOR_TIME_PREFIX);
        occupied.push(column.clone());
        df.set(Series::new(
            Some(column.clone()),
            Dtype::Datetime("ns".to_string(), None),
            context
                .anchor_time
                .iter()
                .map(|&ns| {
                    if ns == i64::MIN {
                        PyVal::NaT
                    } else {
                        PyVal::Timestamp(sdm_schema::dt::civil::Wall::from_epoch_nanos(i128::from(
                            ns,
                        )))
                    }
                })
                .collect(),
        ));
        anchor_column = Some(column);
    }
    let mut targets = vec![PyVal::None; batch_size];
    let num_train = context.num_train();
    for (i, value) in context.y_train.cells.iter().enumerate() {
        if i < batch_size {
            targets[i] = target_json_value(&context.task_type, value).map_err(raised)?;
        }
    }
    if let Some(test) = &context.y_test {
        for (i, value) in test.cells.iter().enumerate() {
            if num_train + i < batch_size {
                targets[num_train + i] =
                    target_json_value(&context.task_type, value).map_err(raised)?;
            }
        }
    }
    df.set(Series::new(Some(target_name), Dtype::Object, targets));
    if let Some(features) = &context.task_table {
        for feature in features {
            df.set(crate::sampled::series_of_values(
                feature.name.as_deref().unwrap_or(""),
                feature.cells.clone(),
            ));
        }
    }
    Ok((df, references, anchor_column))
}

/// `_occurrence_dataframe(table)`: one row per occurrence of a row in an
/// instance's neighbourhood, and the column that keys it.
fn occurrence_dataframe(table: &SampledTable) -> ER<(Frame, String)> {
    let mut df = Frame {
        columns: table.df.clone(),
        rows: table.rows,
    };
    let batch = &table.batch;
    match &table.row {
        Some(row) => {
            if row.len() == batch.len() && !row.is_empty() {
                let max = *row.iter().max().unwrap_or(&0);
                let min = *row.iter().min().unwrap_or(&0);
                if max as usize >= df.rows || min < 0 {
                    return value_error(
                        "Sampled RFM context references rows outside the table frame.",
                    );
                }
                let positions: Vec<usize> = row.iter().map(|&r| r as usize).collect();
                df = df.take(&positions);
            } else if row.is_empty() {
                df = df.take(&[]);
            } else if df.rows != batch.len() {
                return value_error("Sampled RFM context has incompatible row and batch lengths.");
            }
        }
        None => {
            if df.rows != batch.len() {
                return value_error(
                    "Sampled RFM context cannot be converted to table-oriented JSON because \
                     rows are not aligned with batch instances.",
                );
            }
        }
    }
    if df.rows != batch.len() {
        return value_error("Sampled RFM context has incompatible table and batch lengths.");
    }
    df.insert(0, int64_series(INSTANCE_ID, batch.iter().copied()))?;
    let key_column = match &table.primary_key {
        Some(key) if df.position(key).is_some() => key.clone(),
        _ => {
            let key = unique_internal_column(&df.names(), SYNTHETIC_NODE_ID);
            df.insert(1, int64_series(&key, 0..df.rows as i64))?;
            key
        }
    };
    Ok((df, key_column))
}

/// `SampledSubgraph.rev_edge_type(edge_type)`.
fn rev_edge_type(edge: &(String, String, String)) -> (String, String, String) {
    let (src, rel, dst) = edge;
    match rel.strip_prefix(REVERSE_RELATION_PREFIX) {
        Some(forward) => (dst.clone(), forward.to_string(), src.clone()),
        None => (
            dst.clone(),
            format!("{REVERSE_RELATION_PREFIX}{rel}"),
            src.clone(),
        ),
    }
}

/// `_logical_forward_edge_type(edge_type)`.
fn logical_forward_edge_type(edge: &(String, String, String)) -> (String, String, String) {
    match edge.1.strip_prefix(REVERSE_RELATION_PREFIX) {
        Some(forward) => (edge.2.clone(), forward.to_string(), edge.0.clone()),
        None => edge.clone(),
    }
}

/// `_edge_pairs(context, edge_type, link)`.
fn edge_pairs(
    context: &PredictionContext,
    edge: &(String, String, String),
    link: &sdm_sampling::driver::output::SampledLink,
) -> Vec<(i64, i64)> {
    if link.layout == "REV" {
        let rev = rev_edge_type(edge);
        let Some(rev_link) = context.links.iter().find(|l| l.edge_type == rev) else {
            return Vec::new();
        };
        return edge_pairs(context, &rev, rev_link)
            .into_iter()
            .map(|(src, dst)| (dst, src))
            .collect();
    }
    let num_edges: i64 = link.num_sampled_edges.iter().sum();
    if num_edges == 0 {
        return Vec::new();
    }
    let row: Vec<i64> = link.row.clone().unwrap_or_else(|| (0..num_edges).collect());
    if link.layout == "CSC" {
        let Some(colptr) = &link.col else {
            return Vec::new();
        };
        let target_count = context.table(&edge.2).map_or(0, SampledTable::num_rows);
        if colptr.len() as i64 != target_count + 1 {
            return Vec::new();
        }
        let mut pairs = Vec::new();
        for dst in 0..target_count as usize {
            let (start, end) = (colptr[dst] as usize, colptr[dst + 1] as usize);
            for &src in row.get(start..end.min(row.len())).unwrap_or(&[]) {
                pairs.push((src, dst as i64));
            }
        }
        return pairs;
    }
    let col: Vec<i64> = link.col.clone().unwrap_or_else(|| (0..num_edges).collect());
    row.into_iter().zip(col).collect()
}

/// `_logical_forward_edge_pairs(context, edge_type, link)`.
fn logical_forward_edge_pairs(
    context: &PredictionContext,
    edge: &(String, String, String),
    link: &sdm_sampling::driver::output::SampledLink,
) -> Vec<(i64, i64)> {
    let pairs = edge_pairs(context, edge, link);
    if edge.1.starts_with(REVERSE_RELATION_PREFIX) {
        return pairs.into_iter().map(|(s, d)| (d, s)).collect();
    }
    pairs
}

/// One related table of the envelope, as `_payload_tables` builds it.
struct Related {
    name: String,
    df: Frame,
    key_column: String,
    stype_overrides: Vec<(String, String)>,
}

fn set_override(overrides: &mut Vec<(String, String)>, column: &str, stype: &str) {
    match overrides.iter_mut().find(|(c, _)| c == column) {
        Some(entry) => entry.1 = stype.to_string(),
        None => overrides.push((column.to_string(), stype.to_string())),
    }
}

/// `_populate_relationship_columns(...)`: each foreign key sampled, as a
/// column of its table, and the relationships the envelope declares.
fn populate_relationship_columns(
    context: &PredictionContext,
    related: &mut [Related],
    references: &[(String, String)],
) -> ER<Vec<Value>> {
    let mut relationships = Vec::new();
    for table_name in &context.entity_table_names {
        let source_column = references
            .iter()
            .find(|(t, _)| t == table_name)
            .map(|(_, c)| c.clone())
            .unwrap_or_default();
        let Some(target) = related.iter().find(|r| r.name == *table_name) else {
            return value_error(format!(
                "Entity table {} is missing from the sampled subgraph.",
                sdm_pql::pyrepr::str_repr(table_name)
            ));
        };
        let table = context.table(table_name);
        if table.and_then(|t| t.primary_key.as_deref()) != Some(target.key_column.as_str()) {
            return value_error(format!(
                "Entity table {} requires a declared scalar primary key.",
                sdm_pql::pyrepr::str_repr(table_name)
            ));
        }
        relationships.push(json!({
            "source_columns": [source_column],
            "target_table": table_name,
            "target_columns": [target.key_column],
        }));
    }
    let mut processed: Vec<(String, String, String)> = Vec::new();
    for link in &context.links {
        let logical = logical_forward_edge_type(&link.edge_type);
        if processed.contains(&logical) {
            continue;
        }
        let (src_table, fkey, dst_table) = logical.clone();
        let Some(src_at) = related.iter().position(|r| r.name == src_table) else {
            continue;
        };
        let Some(dst_at) = related.iter().position(|r| r.name == dst_table) else {
            continue;
        };
        let pairs = logical_forward_edge_pairs(context, &link.edge_type, link);
        if pairs.is_empty() {
            continue;
        }
        let src_rows = related[src_at].df.rows;
        let dst_key = related[dst_at].key_column.clone();
        let dst_values = related[dst_at]
            .df
            .position(&dst_key)
            .map(|at| related[dst_at].df.columns[at].cells.clone())
            .unwrap_or_default();
        let destination_dtype = related[dst_at]
            .df
            .position(&dst_key)
            .map(|at| related[dst_at].df.columns[at].dtype.clone())
            .unwrap_or(Dtype::Object);
        let dst_rows = related[dst_at].df.rows;
        let mut fkey_values = vec![PyVal::None; src_rows];
        let mut any = false;
        for (src, dst) in pairs {
            if src >= 0 && (src as usize) < src_rows && dst >= 0 && (dst as usize) < dst_rows {
                fkey_values[src as usize] = dst_values[dst as usize].clone();
                any = true;
            }
        }
        if !any {
            continue;
        }
        if fkey_values.iter().all(|v| matches!(v, PyVal::None)) {
            continue;
        }
        let source = &mut related[src_at];
        let reference = if matches!(&destination_dtype, Dtype::Nullable(name) if name == "Float32")
        {
            entity_reference(&fkey, fkey_values, &destination_dtype)
        } else {
            crate::sampled::series_of_values(&fkey, fkey_values)
        };
        source.df.set(reference);
        set_override(&mut source.stype_overrides, &fkey, "ID");
        relationships.push(json!({
            "source_table": src_table,
            "source_columns": [fkey],
            "target_table": dst_table,
            "target_columns": [dst_key],
        }));
        processed.push(logical);
    }
    Ok(relationships)
}

/// `_column_dtypes(df)`.
fn column_dtypes(df: &Frame) -> ER<Vec<(String, String)>> {
    df.columns
        .iter()
        .map(|s| {
            Ok((
                s.name.clone().unwrap_or_default(),
                dtype_name(s).map_err(raised)?,
            ))
        })
        .collect()
}

/// `_target_dtype(task_type, target)`.
fn target_dtype(task_type: &str, target: &Series) -> ER<String> {
    if is_link_pred(task_type) {
        return Ok("stringlist".to_string());
    }
    if task_type == "binary_classification" {
        return Ok("bool".to_string());
    }
    let dtype = dtype_name(target).map_err(raised)?;
    if task_type == "multiclass_classification" && dtype == "bool" {
        return Ok("string".to_string());
    }
    Ok(dtype)
}

fn validate_decimal_classes(classes: &[PyVal], dtype: &str) -> ER<()> {
    if !classes
        .iter()
        .any(|value| matches!(value.semantic(), PyVal::Decimal(_)))
    {
        return Ok(());
    }
    let mut encoded = std::collections::HashMap::new();
    for value in classes {
        let cell = cell_json_value(value, dtype).map_err(raised)?.eq_key();
        let exact = value.eq_key();
        if let Some(previous) = encoded.insert(cell, exact.clone())
            && previous != exact
        {
            return value_error(
                "Distinct decimal target classes become identical in the numeric wire format. Convert the target to strings to preserve its classes.",
            );
        }
    }
    Ok(())
}

/// `_target_stype(task_type)`.
fn target_stype(task_type: &str) -> &'static str {
    if is_link_pred(task_type) {
        "multicategorical"
    } else if is_classification(task_type) {
        "categorical"
    } else {
        "numerical"
    }
}

/// `_dataframe_table(df, dtypes, table_name)`.
fn dataframe_table(df: &Frame, dtypes: &[(String, String)], table_name: &str) -> ER<Value> {
    let columns = df.names();
    let cell_dtypes: Vec<String> = columns
        .iter()
        .map(|c| {
            dtypes
                .iter()
                .find(|(n, _)| n == c)
                .map(|(_, d)| d.clone())
                .ok_or_else(|| PyExc::key_error(c))
        })
        .collect::<ER<_>>()?;
    let mut rows = Vec::with_capacity(df.rows);
    for row_index in 0..df.rows {
        let mut cells = Vec::with_capacity(columns.len());
        for (column_index, series) in df.columns.iter().enumerate() {
            let value = &series.cells[row_index];
            match cell_json_value(value, &cell_dtypes[column_index]) {
                Ok(cell) => cells.push(to_json(&cell)),
                Err(error) => {
                    return value_error(format!(
                        "Column '{table_name}.{}' row {row_index}: {}",
                        columns[column_index], error.message
                    ));
                }
            }
        }
        rows.push(Value::Array(cells));
    }
    Ok(json!({"format": "arrays", "columns": columns, "rows": rows}))
}

/// `_schema_for_dataframe(dtypes, stype_overrides, primary_key)`.
fn schema_for_dataframe(
    dtypes: &[(String, String)],
    stype_overrides: &[(String, String)],
    primary_key: &Value,
) -> Value {
    let keys: Vec<&str> = match primary_key {
        Value::String(s) => vec![s.as_str()],
        Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    let mut columns = Map::new();
    for (name, dtype) in dtypes {
        let mut column = Map::new();
        column.insert("dtype".into(), json!(dtype));
        if let Some((_, stype)) = stype_overrides.iter().find(|(c, _)| c == name) {
            column.insert("stype".into(), json!(stype));
        }
        if keys.contains(&name.as_str()) {
            column.insert("nullable".into(), json!(false));
        }
        columns.insert(name.clone(), Value::Object(column));
    }
    json!({"columns": columns, "primary_key": primary_key})
}

/// `predict_request_to_json(request, explain=explain)`: the envelope.
pub fn predict_request_to_json(
    context: &PredictionContext,
    request: &PredictRequest,
    explain: bool,
) -> ER<Value> {
    let quantiles = request
        .inference_config
        .as_ref()
        .is_some_and(|c| c.kind == "regression" && c.output_type == "quantiles");
    let fields = output_fields(
        &context.task_type,
        request.return_embeddings,
        explain,
        quantiles,
    );
    let mut metadata = Map::new();
    if !request.query.is_empty() {
        metadata.insert("source_query".into(), json!(request.query));
    }
    base_payload(context, request, json!({"fields": fields}), metadata)
}

/// `_base_payload(...)`.
fn base_payload(
    context: &PredictionContext,
    request: &PredictRequest,
    output: Value,
    mut metadata: Map<String, Value>,
) -> ER<Value> {
    // `_payload_tables(context)`.
    let (instance_df, references, anchor_column) = instance_dataframe(context)?;
    let mut related: Vec<Related> = Vec::new();
    for table in &context.tables {
        let (df, key_column) = occurrence_dataframe(table)?;
        let mut stype_overrides = vec![(INSTANCE_ID.to_string(), "ID".to_string())];
        for (column, stype) in &table.stype_dict {
            set_override(&mut stype_overrides, column, stype);
        }
        if table.primary_key.as_deref() != Some(key_column.as_str()) {
            set_override(&mut stype_overrides, &key_column, "ID");
        }
        related.push(Related {
            name: table.name.clone(),
            df,
            key_column,
            stype_overrides,
        });
    }
    let relationships = populate_relationship_columns(context, &mut related, &references)?;
    let num_train = context.num_train();
    let target_name = context.target_name();
    let mut instance_dtypes = column_dtypes(&instance_df)?;
    let target_type = target_dtype(&context.task_type, &context.y_train)?;
    match instance_dtypes.iter_mut().find(|(n, _)| *n == target_name) {
        Some(entry) => entry.1 = target_type.clone(),
        None => instance_dtypes.push((target_name.clone(), target_type.clone())),
    }
    let mut related_dtypes = Vec::new();
    for table in &related {
        related_dtypes.push((table.name.clone(), column_dtypes(&table.df)?));
    }
    let context_rows: Vec<usize> = (0..num_train.min(instance_df.rows)).collect();
    let predict_rows: Vec<usize> = (num_train.min(instance_df.rows)..instance_df.rows).collect();
    let context_instance = instance_df.take(&context_rows);
    let predict_instance = instance_df.take(&predict_rows).drop(&target_name);

    metadata.insert("num_context_examples".into(), json!(num_train));
    metadata.insert("num_prediction_examples".into(), json!(context.num_test()));

    // `_task_spec(context, tables)`.
    let mut target = Map::new();
    target.insert("column_name".into(), json!(target_name));
    target.insert("dtype".into(), json!(target_type));
    match context.task_type.as_str() {
        "binary_classification" => {
            target.insert("classes".into(), json!(["false", "true"]));
            target.insert("positive_class".into(), json!("true"));
        }
        "multiclass_classification" => {
            let mut classes: Vec<PyVal> = Vec::new();
            for value in &context.y_train.cells {
                if !classes.iter().any(|c| c.eq_key() == value.eq_key()) {
                    classes.push(value.clone());
                }
            }
            validate_decimal_classes(&classes, &target_type)?;
            let labels = classes
                .iter()
                .map(|v| class_label(v, &target_type))
                .collect::<Result<Vec<_>, _>>()
                .map_err(raised)?;
            target.insert("classes".into(), json!(labels));
        }
        _ => {}
    }
    let mut task = Map::new();
    task.insert("kind".into(), json!(context.task_type));
    task.insert("target".into(), Value::Object(target));
    task.insert(
        "entity_table_names".into(),
        json!(context.entity_table_names),
    );
    if let Some(anchor) = &anchor_column {
        task.insert("anchor_time_column".into(), json!(anchor));
    }
    if let Some(k) = context.top_k {
        task.insert("top_k".into(), json!(k));
    }
    if let Some(step) = context.step_size {
        task.insert("step_size".into(), json!(step));
    }
    if context.num_forecasts != 1 {
        task.insert("num_forecasts".into(), json!(context.num_forecasts));
    }

    // `_schema_spec(context, tables)`.
    let mut instance_overrides = vec![(INSTANCE_ID.to_string(), "ID".to_string())];
    for (_, column) in &references {
        set_override(&mut instance_overrides, column, "ID");
    }
    if let Some(anchor) = &anchor_column {
        set_override(&mut instance_overrides, anchor, "timestamp");
    }
    set_override(
        &mut instance_overrides,
        &target_name,
        target_stype(&context.task_type),
    );
    let mut instance_schema =
        schema_for_dataframe(&instance_dtypes, &instance_overrides, &json!(INSTANCE_ID));
    if let Some(anchor) = &anchor_column
        && let Some(column) = instance_schema
            .get_mut("columns")
            .and_then(|c| c.get_mut(anchor))
            .and_then(Value::as_object_mut)
    {
        column.insert("nullable".into(), json!(false));
    }
    let mut related_schemas = Map::new();
    for (table, (_, dtypes)) in related.iter().zip(&related_dtypes) {
        related_schemas.insert(
            table.name.clone(),
            schema_for_dataframe(
                dtypes,
                &table.stype_overrides,
                &json!([INSTANCE_ID, table.key_column]),
            ),
        );
    }
    let schema = json!({
        "instance_table": instance_schema,
        "related_tables": related_schemas,
        "relationships": relationships,
    });

    let mut context_related = Map::new();
    let mut predict_related = Map::new();
    for (table, (_, dtypes)) in related.iter().zip(&related_dtypes) {
        let instance_ids: Vec<i64> = table
            .df
            .position(INSTANCE_ID)
            .map(|at| {
                table.df.columns[at]
                    .cells
                    .iter()
                    .map(|v| match v {
                        PyVal::Int(n) => *n as i64,
                        _ => 0,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let (in_context, in_predict): (Vec<usize>, Vec<usize>) =
            (0..table.df.rows).partition(|&i| instance_ids[i] < num_train as i64);
        context_related.insert(
            table.name.clone(),
            dataframe_table(&table.df.take(&in_context), dtypes, &table.name)?,
        );
        predict_related.insert(
            table.name.clone(),
            dataframe_table(&table.df.take(&in_predict), dtypes, &table.name)?,
        );
    }
    let context_instance_json =
        dataframe_table(&context_instance, &instance_dtypes, "instance_table")?;
    let predict_instance_json =
        dataframe_table(&predict_instance, &instance_dtypes, "instance_table")?;
    let mut inference = Map::new();
    inference.insert("run_mode".into(), json!(request.run_mode));
    inference.insert(
        "use_prediction_time".into(),
        json!(request.use_prediction_time),
    );
    inference.insert(
        "inference_config".into(),
        request
            .inference_config
            .as_ref()
            .map_or(Value::Null, InferenceConfig::to_json),
    );
    Ok(json!({
        "model": "kumo-relational",
        "task": Value::Object(task),
        "schema": schema,
        "context": {
            "instance_table": context_instance_json,
            "related_tables": context_related,
        },
        "predict": {
            "instance_table": predict_instance_json,
            "related_tables": predict_related,
        },
        "output": output,
        "inference": Value::Object(inference),
        "metadata": Value::Object(metadata),
    }))
}

/// `validate_payload_table_rows(payload, batch_index=...)`.
pub fn validate_payload_table_rows(payload: &Value, batch_index: usize) -> ER<()> {
    for section in ["context", "predict"] {
        let rows = payload
            .get(section)
            .and_then(|s| s.get("instance_table"))
            .and_then(|t| t.get("rows"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if rows > MAX_TABLE_ROWS {
            return value_error(format!(
                "Request batch {batch_index} table '{section}.instance_table' contains {} rows, \
                 exceeding the {}-row limit",
                sdm_schema::py::thousands(rows as i64),
                sdm_schema::py::thousands(MAX_TABLE_ROWS as i64)
            ));
        }
    }
    Ok(())
}

/// `session_create_payload(payload)`.
pub fn session_create_payload(payload: &Value) -> Value {
    sections(payload, &["model", "task", "schema", "context", "metadata"])
}

/// `session_predict_payload(payload)`.
pub fn session_predict_payload(payload: &Value) -> Value {
    sections(payload, &["predict", "output", "inference", "metadata"])
}

fn sections(payload: &Value, keep: &[&str]) -> Value {
    let mut out = Map::new();
    for key in keep {
        if let Some(value) = payload.get(*key) {
            out.insert((*key).to_string(), value.clone());
        }
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_references_keep_numpy_scalar_widths_without_narrowing_arrow() {
        for (source, expected) in [
            (Dtype::Float(16), Dtype::Float(16)),
            (Dtype::Float(32), Dtype::Float(32)),
            (Dtype::Nullable("Float32".into()), Dtype::Float(32)),
            (Dtype::Float(64), Dtype::Float(64)),
            (Dtype::Arrow("float".into()), Dtype::Float(64)),
        ] {
            let values = vec![PyVal::Float(1.25), PyVal::Float(-3.75)];
            let reference = entity_reference("reference", values.clone(), &source);
            assert_eq!(reference.dtype, expected);
            assert_eq!(reference.cells, values);
            assert_eq!(
                crate::wire::infer_tfm_dtype(&reference),
                if expected.float_itemsize() <= 4 {
                    "float32"
                } else {
                    "float64"
                }
            );
        }
    }

    #[test]
    fn missing_or_mixed_entity_references_keep_value_inference() {
        for source in [
            Dtype::Float(16),
            Dtype::Float(32),
            Dtype::Nullable("Float32".into()),
        ] {
            for values in [
                vec![PyVal::Float(1.25), PyVal::None],
                vec![PyVal::Float(1.25), PyVal::Int(2)],
                Vec::new(),
            ] {
                let expected = crate::sampled::series_of_values("reference", values.clone());
                let reference = entity_reference("reference", values, &source);
                assert_eq!(reference.dtype, expected.dtype);
                assert_eq!(reference.cells.len(), expected.cells.len());
                for (actual, expected) in reference.cells.iter().zip(&expected.cells) {
                    if matches!(expected, PyVal::Float(value) if value.is_nan()) {
                        assert!(matches!(actual, PyVal::Float(value) if value.is_nan()));
                    } else {
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        let reference =
            entity_reference("reference", vec![PyVal::Float(f64::NAN)], &Dtype::Float(32));
        assert_eq!(reference.dtype, Dtype::Float(32));
        assert!(matches!(reference.cells[0], PyVal::Float(value) if value.is_nan()));
    }

    #[test]
    fn decimal_class_encoding_refuses_distinct_values_that_collapse() {
        let classes = [
            PyVal::Decimal("0.100000000000000001".into()),
            PyVal::Decimal("0.100000000000000002".into()),
        ];
        assert_eq!(
            cell_json_value(&classes[0], "float64").unwrap(),
            cell_json_value(&classes[1], "float64").unwrap()
        );
        assert_eq!(
            validate_decimal_classes(&classes, "float64")
                .unwrap_err()
                .class,
            "ValueError"
        );
        assert!(validate_decimal_classes(&classes, "string").is_ok());
    }

    #[test]
    fn decimal_class_encoding_preserves_equal_and_exact_whole_values() {
        for (values, dtype) in [
            (vec!["1.0", "1.00", "0.0", "-0.00"], "float64"),
            (vec!["9007199254740992", "9007199254740993"], "int64"),
            (vec!["9223372036854775808", "9223372036854775809"], "string"),
            (vec!["1.20", "1.21"], "float64"),
        ] {
            let classes: Vec<_> = values
                .into_iter()
                .map(|value| PyVal::Decimal(value.into()))
                .collect();
            assert!(validate_decimal_classes(&classes, dtype).is_ok());
        }
    }

    /// `df.insert` refuses a name the frame has, as pandas does, which is
    /// what a frame given an `instance_id` after its schema read it meets.
    #[test]
    fn a_column_the_frame_has_is_not_inserted_again() {
        let mut df = Frame {
            columns: vec![int64_series("instance_id", 0..2)],
            rows: 2,
        };
        let refused = df.insert(0, int64_series(INSTANCE_ID, 0..2)).unwrap_err();
        assert_eq!(refused.class, "ValueError");
        assert_eq!(refused.message, "cannot insert instance_id, already exists");
        df.insert(0, int64_series("__synthetic", 0..2)).unwrap();
        assert_eq!(df.names(), ["__synthetic", "instance_id"]);
    }
}
