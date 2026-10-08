// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Sampler.__init__` (`rfm/base/sampler.py`): the maps every sampler keeps
//! of its graph, its time ranges, and the backend it samples through.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{Array, ArrayRef, Float64Array};
use sdm_labels::column::{Column, TimeUnit};
use sdm_labels::executor::{Batch, Frame as LabelFrame};
use sdm_schema::dt::Host;

use super::inputs::{Backend, Inputs, TableInfo};
use super::local::LocalBackend;
use super::query::SamplerQuery;
use super::sql::SqlBackend;
use super::{Db, Raised};
use crate::frame::{Dtype, Frame, Series};
use crate::graph_store::rev_edge_type;
use crate::neighbor::EdgeType;

/// `pd.Timestamp.max` and `pd.Timestamp.min`, in nanoseconds.
pub const TIMESTAMP_MAX: i64 = i64::MAX;
pub const TIMESTAMP_MIN: i64 = i64::MIN + 1;

/// The backend a sampler reads through.
pub enum Through {
    Local(Box<LocalBackend>),
    Sql(Box<SqlBackend>),
}

/// A sampler: its graph's maps, its state, and its backend.
pub struct Sampler {
    pub backend: Backend,
    pub tables: Vec<TableInfo>,
    pub edges: Vec<EdgeType>,
    pub edge_types: Vec<EdgeType>,
    /// Source table to `(foreign key, destination table)`, in table order.
    pub foreign_key_dict: Vec<(String, Vec<(String, String)>)>,
    /// Destination table to `(source table, foreign key)`.
    pub rev_foreign_key_dict: Vec<(String, Vec<(String, String)>)>,
    pub primary_key_dict: Vec<(String, String)>,
    pub time_column_dict: Vec<(String, String)>,
    pub end_time_column_dict: Vec<(String, String)>,
    /// Each table's feature columns and their stypes: every column but its
    /// primary key and its foreign keys.
    pub table_stype_dict: Vec<(String, Vec<(String, String)>)>,
    pub min_time: HashMap<String, i64>,
    pub max_time: HashMap<String, i64>,
    /// The unit of each table's earliest and latest `Timestamp`.
    pub min_time_unit: HashMap<String, String>,
    pub max_time_unit: HashMap<String, String>,
    pub coverage_warned: bool,
    pub through: Through,
    pub host: Host,
    pub warned: Mutex<Vec<super::output::Warning>>,
}

fn lookup<'a, T>(dict: &'a [(String, T)], key: &str) -> Option<&'a T> {
    dict.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

impl Sampler {
    pub fn new(inputs: &Inputs, host: Host) -> Result<Self, Raised> {
        let edges: Vec<EdgeType> = inputs.edges.clone();
        let mut edge_types = Vec::new();
        for edge in &edges {
            edge_types.push(edge.clone());
            edge_types.push(rev_edge_type(edge));
        }
        let mut foreign_key_dict: Vec<(String, Vec<(String, String)>)> = inputs
            .tables
            .iter()
            .map(|t| (t.name.clone(), Vec::new()))
            .collect();
        let mut rev_foreign_key_dict = foreign_key_dict.clone();
        for (src, fkey, dst) in &edges {
            match foreign_key_dict.iter_mut().find(|(t, _)| t == src) {
                Some((_, list)) => list.push((fkey.clone(), dst.clone())),
                None => return Err(Raised::key_error(src)),
            }
            match rev_foreign_key_dict.iter_mut().find(|(t, _)| t == dst) {
                Some((_, list)) => list.push((src.clone(), fkey.clone())),
                None => return Err(Raised::key_error(dst)),
            }
        }
        let primary_key_dict = inputs
            .tables
            .iter()
            .filter_map(|t| t.primary_key.clone().map(|k| (t.name.clone(), k)))
            .collect();
        let time_column_dict = inputs
            .tables
            .iter()
            .filter_map(|t| t.time_column.clone().map(|k| (t.name.clone(), k)))
            .collect();
        let end_time_column_dict = inputs
            .tables
            .iter()
            .filter_map(|t| t.end_time_column.clone().map(|k| (t.name.clone(), k)))
            .collect();
        let foreign_keys: Vec<(&String, &String)> =
            edges.iter().map(|(src, fkey, _)| (src, fkey)).collect();
        let table_stype_dict = inputs
            .tables
            .iter()
            .map(|t| {
                let stypes = t
                    .columns
                    .iter()
                    .filter(|c| t.primary_key.as_deref() != Some(c.name.as_str()))
                    .filter(|c| !foreign_keys.contains(&(&t.name, &c.name)))
                    .map(|c| (c.name.clone(), c.stype.clone().unwrap_or_default()))
                    .collect();
                (t.name.clone(), stypes)
            })
            .collect();
        let through = match inputs.backend {
            Backend::Local => Through::Local(Box::new(LocalBackend::new(inputs, &host)?)),
            _ => Through::Sql(Box::new(SqlBackend::new(inputs)?)),
        };
        Ok(Self {
            backend: inputs.backend,
            tables: inputs.tables.clone(),
            edges,
            edge_types,
            foreign_key_dict,
            rev_foreign_key_dict,
            primary_key_dict,
            time_column_dict,
            end_time_column_dict,
            table_stype_dict,
            min_time: inputs.state.min_time.clone(),
            max_time: inputs.state.max_time.clone(),
            min_time_unit: inputs.state.min_time_unit.clone(),
            max_time_unit: inputs.state.max_time_unit.clone(),
            coverage_warned: inputs.state.coverage_warned,
            through,
            host,
            warned: Mutex::new(Vec::new()),
        })
    }

    /// The warnings reading results gave since they were last taken, in the
    /// order Python gives them; none in memory, where the tables were read
    /// when the graph was built.
    pub fn take_warnings(&self) -> Vec<super::output::Warning> {
        let mut warnings =
            std::mem::take(&mut *self.warned.lock().unwrap_or_else(PoisonError::into_inner));
        if let Through::Sql(sql) = &self.through {
            warnings.extend(sql.take_warnings());
        }
        warnings
    }

    /// `sanitization_report.tables`: in memory each table's counts, and
    /// none through SQL, where the report is not available.
    pub fn sanitization_report(&self) -> Vec<(String, crate::graph_store::TableReport)> {
        match &self.through {
            Through::Local(local) => local.sanitization_report().to_vec(),
            Through::Sql(_) => Vec::new(),
        }
    }

    /// The warnings building the sampler gave, which today's code gives as
    /// the graph is built rather than as it is sampled.
    pub fn built_warnings(&self) -> Vec<super::output::Warning> {
        match &self.through {
            Through::Local(local) => local.built_warnings().to_vec(),
            Through::Sql(_) => Vec::new(),
        }
    }

    /// What the sampler holds from the calls it served, to start the next
    /// from: its times, and what its backend read.
    pub fn state(&self) -> super::inputs::State {
        let mut state = super::inputs::State {
            min_time: self.min_time.clone(),
            max_time: self.max_time.clone(),
            min_time_unit: self.min_time_unit.clone(),
            max_time_unit: self.max_time_unit.clone(),
            coverage_warned: self.coverage_warned,
            ..Default::default()
        };
        if let Through::Local(local) = &self.through {
            state.neighbor_generator = Some(local.generator());
        }
        if let Through::Sql(sql) = &self.through {
            state.enforced_key_tables = sql.enforced_key_tables.clone();
            state.num_rows = sql.num_rows.clone();
            state.time_spellings = sql
                .time_spellings
                .lock()
                .map(|m| m.clone())
                .unwrap_or_default();
        }
        state
    }

    /// Snowflake's `_listed` of each table, where it was read.
    pub fn listed(&self) -> Vec<(String, Vec<sdm_dialects::helpers::Listing>)> {
        match &self.through {
            Through::Sql(sql) => sql
                .listed
                .lock()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
            Through::Local(_) => Vec::new(),
        }
    }

    pub fn table(&self, name: &str) -> Result<&TableInfo, Raised> {
        self.tables
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Raised::key_error(name))
    }

    /// `primary_key_dict[table]`.
    pub fn primary_key(&self, table: &str) -> Result<&String, Raised> {
        lookup(&self.primary_key_dict, table).ok_or_else(|| Raised::key_error(table))
    }

    pub fn has_primary_key(&self, table: &str) -> bool {
        lookup(&self.primary_key_dict, table).is_some()
    }

    pub fn time_column(&self, table: &str) -> Option<&String> {
        lookup(&self.time_column_dict, table)
    }

    pub fn end_time_column(&self, table: &str) -> Option<&String> {
        lookup(&self.end_time_column_dict, table)
    }

    pub fn foreign_keys(&self, table: &str) -> Result<&Vec<(String, String)>, Raised> {
        lookup(&self.foreign_key_dict, table).ok_or_else(|| Raised::key_error(table))
    }

    pub fn rev_foreign_keys(&self, table: &str) -> Result<&Vec<(String, String)>, Raised> {
        lookup(&self.rev_foreign_key_dict, table).ok_or_else(|| Raised::key_error(table))
    }

    pub fn stypes(&self, table: &str) -> Result<&Vec<(String, String)>, Raised> {
        lookup(&self.table_stype_dict, table).ok_or_else(|| Raised::key_error(table))
    }

    fn time_tables(&self, table_names: Option<&[String]>) -> Vec<String> {
        let given: Vec<String> = match table_names {
            Some(names) if !names.is_empty() => names.to_vec(),
            _ => self
                .time_column_dict
                .iter()
                .map(|(t, _)| t.clone())
                .collect(),
        };
        let mut out: Vec<String> = Vec::new();
        for name in given {
            if !out.contains(&name) {
                out.push(name);
            }
        }
        out
    }

    /// `get_min_time(table_names)`.
    pub async fn get_min_time(
        &mut self,
        db: &Db,
        table_names: Option<&[String]>,
    ) -> Result<i64, Raised> {
        Ok(self.min_time_stamp(db, table_names).await?.0)
    }

    /// `get_max_time(table_names)`.
    pub async fn get_max_time(
        &mut self,
        db: &Db,
        table_names: Option<&[String]>,
    ) -> Result<i64, Raised> {
        Ok(self.max_time_stamp(db, table_names).await?.0)
    }

    /// `get_min_time(table_names)` and the unit of the `Timestamp` it gives.
    pub async fn min_time_stamp(
        &mut self,
        db: &Db,
        table_names: Option<&[String]>,
    ) -> Result<(i64, String), Raised> {
        let tables = self.time_tables(table_names);
        let unknown: Vec<String> = tables
            .iter()
            .filter(|t| !self.min_time.contains_key(*t))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            self.read_time_ranges_of(db, &unknown).await?;
        }
        // `min([...] + [pd.Timestamp.max])`: the first of the least.
        let mut out = (TIMESTAMP_MAX, "ns".to_string());
        let mut found = false;
        for table in &tables {
            let value = *self
                .min_time
                .get(table)
                .ok_or_else(|| Raised::key_error(table))?;
            if !found || value < out.0 {
                let unit = self
                    .min_time_unit
                    .get(table)
                    .cloned()
                    .unwrap_or_else(|| "ns".into());
                out = (value, unit);
                found = true;
            }
        }
        Ok(out)
    }

    /// `get_max_time(table_names)` and the unit of the `Timestamp` it gives.
    pub async fn max_time_stamp(
        &mut self,
        db: &Db,
        table_names: Option<&[String]>,
    ) -> Result<(i64, String), Raised> {
        let tables = self.time_tables(table_names);
        let unknown: Vec<String> = tables
            .iter()
            .filter(|t| !self.max_time.contains_key(*t))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            self.read_time_ranges_of(db, &unknown).await?;
        }
        // `max([...] + [pd.Timestamp.min])`: the first of the greatest.
        let mut out = (TIMESTAMP_MIN, "ns".to_string());
        let mut found = false;
        for table in &tables {
            let value = *self
                .max_time
                .get(table)
                .ok_or_else(|| Raised::key_error(table))?;
            if !found || value > out.0 {
                let unit = self
                    .max_time_unit
                    .get(table)
                    .cloned()
                    .unwrap_or_else(|| "ns".into());
                out = (value, unit);
                found = true;
            }
        }
        Ok(out)
    }

    async fn read_time_ranges_of(&mut self, db: &Db, unknown: &[String]) -> Result<(), Raised> {
        let ranges = match &self.through {
            Through::Local(local) => local
                .min_max_time_dict(unknown)
                .into_iter()
                .map(|(t, (min, max))| (t, ((min, "ns".to_string()), (max, "ns".to_string()))))
                .collect(),
            Through::Sql(sql) => sql.min_max_time_dict(self, db, unknown).await?,
        };
        for (table, ((min, min_unit), (max, max_unit))) in ranges {
            self.min_time.insert(table.clone(), min);
            self.max_time.insert(table.clone(), max);
            self.min_time_unit.insert(table.clone(), min_unit);
            self.max_time_unit.insert(table, max_unit);
        }
        Ok(())
    }

    /// The label executor's answer over one batch of examples: the labels,
    /// with their pandas dtype, and the mask of the examples they are for.
    pub fn execute_query(
        &self,
        query: &SamplerQuery,
        feat: &[(String, Frame)],
        time: &[(String, Series)],
        batch: &[(String, Vec<i64>)],
        anchor: &[Option<i64>],
        filter_na: bool,
    ) -> Result<(Series, Vec<bool>), Raised> {
        let mut input = Batch {
            host: self.host.clone(),
            ..Batch::default()
        };
        for (name, frame) in feat {
            input.feat.push((name.clone(), label_frame(frame)?));
            for (column, series) in frame.names.iter().zip(&frame.columns) {
                if matches!(series.dtype, Dtype::Arrow)
                    && let arrow_schema::DataType::Decimal128(_, scale) = series.array.data_type()
                {
                    input
                        .arrow_decimal_scales
                        .push((format!("{name}.{column}"), *scale));
                }
            }
        }
        for (name, series) in time {
            input.time.push((name.clone(), label_column(series)?));
        }
        for (name, rows) in batch {
            input.batch.push((name.clone(), rows.clone()));
        }
        input.anchor = Column::Time(TimeUnit::Ns, anchor.to_vec());
        let result = sdm_labels::executor::execute_query_with(&query.labels, &input, filter_na);
        let mut warnings = self.take_warnings();
        warnings.extend(input.warnings.into_inner().into_iter().map(|warning| {
            super::output::Warning {
                category: warning.category().into(),
                message: warning.message().into(),
                masked_float32_cast: warning.masked_float32_cast(),
            }
        }));
        self.warned
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(warnings);
        let (labels, mask) = result?;
        let source = label_source_dtype(query, feat);
        Ok((label_series(&labels, source)?, mask))
    }

    /// `validate_entity_references(table, entity_pkey)`: in memory, every
    /// key must name a row the graph store kept; a SQL backend checks none.
    pub fn validate_entity_references(
        &self,
        table: &str,
        entity_pkey: &Series,
    ) -> Result<(), Raised> {
        match &self.through {
            Through::Local(local) => local.validate_entity_references(table, entity_pkey),
            Through::Sql(_) => Ok(()),
        }
    }

    /// `_sample_entity_table`.
    pub async fn sample_entity_table(
        &self,
        db: &Db,
        table: &str,
        columns: &[String],
        num_rows: i64,
        random_seed: Option<&sdm_dialects::int::Int>,
        entity_ids: Option<&[super::query::EntityId]>,
    ) -> Result<Frame, Raised> {
        match &self.through {
            Through::Local(local) => {
                local.sample_entity_table(self, table, columns, num_rows, random_seed, entity_ids)
            }
            Through::Sql(sql) => {
                sql.sample_entity_table(self, db, table, columns, num_rows, random_seed, entity_ids)
                    .await
            }
        }
    }
}

/// A frame as the label executor reads it.
fn label_frame(frame: &Frame) -> Result<LabelFrame, Raised> {
    let columns = frame
        .names
        .iter()
        .zip(&frame.columns)
        .map(|(name, series)| Ok((name.clone(), label_column(series)?)))
        .collect::<Result<Vec<_>, Raised>>()?;
    Ok(LabelFrame { columns })
}

/// A column as the label executor reads one of its dtype.
pub fn label_column(series: &Series) -> Result<Column, Raised> {
    let numpy_type = series.dtype.numpy_type(series.array.as_ref());
    let array: ArrayRef = match (&series.dtype, series.array.data_type()) {
        // The executor casts a numeric ArrowDtype column to numpy before it
        // computes with it, so it is read as the numpy column it becomes.
        (Dtype::Arrow, t)
            if t.is_integer() || t.is_floating() || *t == arrow_schema::DataType::Boolean =>
        {
            return sdm_labels::arrow::column(series.array.as_ref(), None).map_err(Raised::from);
        }
        // A `Categorical` is read as the column of its categories' values.
        (Dtype::Category, _) => {
            let values = crate::frame::decoded(series.array.as_ref())
                .ok_or_else(|| Raised::internal("a category column that is no dictionary"))?;
            return sdm_labels::arrow::column(values.as_ref(), None).map_err(Raised::from);
        }
        _ => series.array.clone(),
    };
    sdm_labels::arrow::column(array.as_ref(), Some(&numpy_type)).map_err(Raised::from)
}

/// The dtype of the column a label is the value of, where it is a column's
/// value: what an untouched text or object label keeps.
fn label_source_dtype(query: &SamplerQuery, feat: &[(String, Frame)]) -> Option<Dtype> {
    let fqn = query.final_target_column.as_ref()?;
    let (table, column) = fqn.split_once('.')?;
    let frame = lookup(feat, table)?;
    frame.column(column).ok().map(|s| s.dtype.clone())
}

/// The labels as a column of their pandas dtype.
pub fn label_series(labels: &Column, source: Option<Dtype>) -> Result<Series, Raised> {
    let array = sdm_labels::arrow::to_arrow(labels)?;
    let dtype = match labels {
        Column::Bool(_) => Dtype::Numpy("bool".into()),
        Column::MBool(_) => Dtype::Extension("boolean".into()),
        Column::I64(_) => Dtype::Numpy("int64".into()),
        Column::MI64(_) => Dtype::Extension("Int64".into()),
        Column::F32(_) => Dtype::Numpy("float32".into()),
        Column::F64(_) => Dtype::Numpy("float64".into()),
        Column::MF64(_) => Dtype::Extension("Float64".into()),
        Column::Time(unit, _) => Dtype::Numpy(format!("datetime64[{}]", unit.name())),
        Column::Str(_) => source.unwrap_or_else(|| Dtype::Extension("str".into())),
        Column::Obj(_)
            if matches!(
                array.data_type(),
                arrow_schema::DataType::Duration(_) | arrow_schema::DataType::Binary
            ) =>
        {
            source.unwrap_or(Dtype::Object)
        }
        Column::Obj(_) => Dtype::Object,
    };
    Ok(Series::new(dtype, array))
}

/// `pd.Series([], dtype=float)`.
pub fn empty_float_series() -> Series {
    Series::new(
        Dtype::Numpy("float64".into()),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
    )
}

/// `pd.concat(ys, ignore_index=True)` of label columns.
pub fn concat_labels(parts: &[Series]) -> Result<Series, Raised> {
    if parts.is_empty() {
        return Ok(empty_float_series());
    }
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let refs: Vec<&Series> = parts.iter().collect();
    crate::frame::concat_series(&refs)
}

/// The labels' first `n`, as `series.iloc[:n]`.
pub fn head(series: &Series, n: usize) -> Series {
    let n = n.min(series.len());
    series.slice(0, n)
}

/// The labels from `start`, as `series.iloc[start:]`.
pub fn tail(series: &Series, start: usize) -> Series {
    let start = start.min(series.len());
    series.slice(start, series.len() - start)
}
