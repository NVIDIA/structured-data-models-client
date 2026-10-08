// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The graph a relational request is gathered from, over frames:
//! `RelationalMaterializer._build_graph` and `_as_typed`, and the parts of
//! the engine it runs – `Graph.from_data`, `LocalTable`, the key, time
//! column and semantic type setters, composite keys
//! (`rfm/base/composite_key.py`), `Graph.link` and `Graph.validate` – as
//! far as building it from a schema reaches them; and what the sampler
//! starts from, read off the graph ([`EngineGraph::inputs`]).

use sdm_pql::types::{Dtype as EDtype, Stype};
use sdm_sampling::driver::inputs::{Backend, ColumnInfo, Inputs, State, TableInfo};
use sdm_schema::client::{Key, Schema};
use sdm_schema::dt::Host;
use sdm_schema::engine::infer::{SampleColumn, SampleKind, infer_dtype, infer_stype};
use sdm_schema::engine::{EColumn, PyExc};
use sdm_schema::pyval::PyVal;
use serde_json::Map;

use crate::error::{R, Warnings, structured};
use crate::frames::Table;
use crate::series::{Dtype, Series, infer_dtype as content_of};

/// `composite_key.DERIVED_PREFIX`.
pub const DERIVED_PREFIX: &str = "__kumo_key_";

/// What the engine raises: its class and `str(error)`.
pub type ER<T> = Result<T, PyExc>;

fn raise<T>(class: &str, message: impl Into<String>) -> ER<T> {
    Err(PyExc::new(class, message))
}

// --- `_as_typed` ---------------------------------------------------------------------

/// `_holds_text(values)`.
fn holds_text(series: &Series) -> bool {
    let text_dtype = series.dtype.is_object() || series.dtype.is_string_dtype();
    text_dtype && content_of(&series.cells) == "string"
}

/// `_holds_decimal(values)`.
fn holds_decimal(series: &Series) -> bool {
    if series.dtype.arrow_decimal().is_some() {
        return true;
    }
    series.dtype.is_object() && content_of(&series.cells) == "decimal"
}

/// `_holds_timestamps_loosely(values, declared)`.
fn holds_timestamps_loosely(series: &Series, declared: &str) -> bool {
    if !declared.starts_with("timestamp") || !series.dtype.is_object() {
        return false;
    }
    matches!(
        content_of(&series.cells),
        "datetime" | "datetime64" | "date"
    )
}

/// `_to_time(values, where)`: a text time column read as dates.
fn to_time(series: &Series, place: &str, host: &Host) -> R<Series> {
    let values: Vec<Option<String>> = series
        .cells
        .iter()
        .map(|v| match v {
            PyVal::Str(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    match sdm_schema::dates::time_column(&values, place, host) {
        Ok(sdm_schema::dates::TimeColumn::Untouched(_)) => Ok(series.clone()),
        Ok(sdm_schema::dates::TimeColumn::Read(read)) => {
            let dtype = Dtype::parse(&read.dtype());
            let cells = read
                .values
                .iter()
                .map(|v| match v {
                    None => PyVal::NaT,
                    Some(ns) => {
                        let wall = sdm_schema::dt::civil::Wall::from_epoch_nanos(*ns);
                        match read.tz {
                            sdm_schema::dt::pandas::ColumnTz::Naive => PyVal::Timestamp(wall),
                            sdm_schema::dt::pandas::ColumnTz::Fixed(offset) => PyVal::TimestampTz(
                                sdm_schema::dt::civil::Wall::from_epoch_nanos(
                                    *ns + i128::from(offset) * 1_000_000_000,
                                ),
                                offset,
                            ),
                        }
                    }
                })
                .collect();
            Ok(Series::new(series.name.clone(), dtype, cells))
        }
        Err(refusal) => Err(crate::schemas::schema_refusal(
            &sdm_schema::client::Refusal {
                code: refusal.code,
                message: refusal.message,
                chain: Vec::new(),
                details: Box::default(),
            },
        )),
    }
}

/// `_to_declared(values, dtype, where)`: decimals as the schema typed them.
fn to_declared(series: &Series, dtype: &str, place: &str) -> R<Series> {
    let decimal = |v: &PyVal| crate::pyops::decimal_value(v).filter(crate::num::Dec::is_finite);
    if dtype.starts_with("float") {
        let cells = series
            .cells
            .iter()
            .map(|v| match crate::pyops::decimal_value(v) {
                Some(d) if !d.is_nan() => PyVal::Float(d.to_f64()),
                _ => PyVal::Float(f64::NAN),
            })
            .collect();
        return Ok(Series::new(series.name.clone(), Dtype::Float(64), cells));
    }
    if !dtype.starts_with("int") {
        let cells = series
            .cells
            .iter()
            .map(|v| {
                if crate::series::is_na(v)
                    || crate::pyops::decimal_value(v).is_some_and(|d| d.is_nan())
                {
                    PyVal::None
                } else {
                    PyVal::Str(crate::pyops::py_str(v))
                }
            })
            .collect();
        return Ok(Series::new(series.name.clone(), Dtype::Object, cells));
    }
    for v in &series.cells {
        if let Some(d) = decimal(v)
            && !d.is_integral()
        {
            return Err(crate::error::invalid(format!(
                "column {place} is declared {} and holds {}, which is not a whole number. \
                 Sending it would round the value without saying so; declare the column \
                 float64, or round it yourself if that is what you meant",
                sdm_pql::pyrepr::str_repr(dtype),
                crate::pyops::py_repr(v)
            )));
        }
    }
    let whole: Vec<Option<i128>> = series
        .cells
        .iter()
        .map(|v| decimal(v).and_then(|d| d.to_i128()))
        .collect();
    let any_missing = whole.iter().any(Option::is_none);
    let cells = whole
        .iter()
        .map(|w| match w {
            Some(n) => PyVal::Int(*n),
            None => PyVal::Na,
        })
        .collect();
    let dtype = if any_missing {
        Dtype::Nullable("Int64".to_string())
    } else {
        Dtype::Int("int64".to_string())
    };
    Ok(Series::new(series.name.clone(), dtype, cells))
}

/// `pd.to_datetime(values, utc=True)` of an `object` column of times.
fn to_utc(series: &Series) -> Series {
    let cells = series
        .cells
        .iter()
        .map(|v| match v {
            PyVal::TimestampTz(wall, offset) => PyVal::TimestampTz(
                sdm_schema::dt::civil::Wall::from_epoch_nanos(
                    wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000,
                ),
                0,
            ),
            PyVal::Timestamp(wall) => PyVal::TimestampTz(*wall, 0),
            PyVal::DateTime(wall, None) => PyVal::TimestampTz(*wall, 0),
            PyVal::DateTime(wall, Some(offset)) => PyVal::TimestampTz(
                sdm_schema::dt::civil::Wall::from_epoch_nanos(
                    wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000,
                ),
                0,
            ),
            PyVal::Date(y, m, d) => {
                PyVal::TimestampTz(sdm_schema::dt::civil::Wall::date(*y, *m, *d), 0)
            }
            _ => PyVal::NaT,
        })
        .collect();
    Series::new(
        series.name.clone(),
        Dtype::Datetime("us".to_string(), Some("UTC".to_string())),
        cells,
    )
}

/// `_as_typed(table)`: the table's rows, decimals cast to what the schema
/// typed them, and a text time column read as dates.
pub fn as_typed(table: &sdm_schema::client::Table, frame: &Table, host: &Host) -> R<Vec<Series>> {
    let mut columns: Vec<Series> = frame
        .columns()
        .iter()
        .filter_map(|name| frame.series(name))
        .collect();
    if frame.is_empty() {
        return Ok(columns);
    }
    // Under a key that is not text, today's `frame.assign(**cast)` is
    // refused by Python before pandas is called.
    let mut cast_under_a_key_not_text = false;
    for column in &table.columns {
        // `name not in frame.columns`: a label that is not text is found
        // only by a key of the same value.
        let Some(at) = columns.iter().position(|s| {
            s.name.as_deref() == Some(column.name.as_str())
                && label_repr(frame, &column.name) == column.name_repr
        }) else {
            continue;
        };
        let place = format!("{}.{}", table.name, column.name);
        let series = &columns[at];
        let cast =
            if table.time_column.as_deref() == Some(column.name.as_str()) && holds_text(series) {
                Some(to_time(series, &place, host)?)
            } else if holds_decimal(series) {
                Some(to_declared(series, &column.dtype, &place)?)
            } else if holds_timestamps_loosely(series, &column.dtype) {
                Some(to_utc(series))
            } else if column.dtype == "string" && series.dtype.is_integer() {
                // Integers the schema calls text, such as a `uint64` key of
                // 2**63 or more, which no wire integer holds: sent as the
                // text the schema says, where today's sent the numbers
                // and failed on them.
                Some(to_declared(series, "string", &place)?)
            } else {
                None
            };
        if let Some(cast) = cast {
            columns[at] = cast;
            cast_under_a_key_not_text |= column.name_repr.is_some();
        }
    }
    if cast_under_a_key_not_text {
        // `_engine_errors` reads a `TypeError` raised in its own module as a
        // call it made wrongly.
        return Err(structured(
            "INTERNAL_ERROR",
            "the relational engine was called incorrectly by this client: TypeError: keywords \
             must be strings",
        ));
    }
    Ok(columns)
}

/// The `repr` of a frame's column label that is not text, as the host's
/// describer tags it (`name_repr`).
fn label_repr(frame: &Table, name: &str) -> Option<String> {
    frame
        .tag(name)
        .get("name_repr")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

// --- The engine's tables ---------------------------------------------------------------

/// How the engine's inference reads a column of this pandas dtype.
fn sample_kind(dtype: &Dtype) -> SampleKind {
    use arrow_schema::DataType;
    match dtype {
        Dtype::Bool => SampleKind::Arrow(DataType::Boolean),
        Dtype::Int(name) => SampleKind::Arrow(match name.as_str() {
            "int8" => DataType::Int8,
            "int16" => DataType::Int16,
            "int32" => DataType::Int32,
            "uint8" => DataType::UInt8,
            "uint16" => DataType::UInt16,
            "uint32" => DataType::UInt32,
            "uint64" => DataType::UInt64,
            _ => DataType::Int64,
        }),
        Dtype::Float(64) => SampleKind::Arrow(DataType::Float64),
        Dtype::Float(32) => SampleKind::Arrow(DataType::Float32),
        Dtype::Float(_) => SampleKind::Nullable("float16"),
        Dtype::Datetime(..) => SampleKind::Datetime64,
        Dtype::Timedelta(_) => {
            SampleKind::Arrow(DataType::Duration(arrow_schema::TimeUnit::Nanosecond))
        }
        Dtype::Object => SampleKind::Object,
        Dtype::Str => SampleKind::Nullable("str"),
        Dtype::String(_) | Dtype::Category => SampleKind::Nullable("string"),
        Dtype::Nullable(name) => SampleKind::Nullable(match name.as_str() {
            "Int8" => "Int8",
            "Int16" => "Int16",
            "Int32" => "Int32",
            "UInt8" => "UInt8",
            "UInt16" => "UInt16",
            "UInt32" => "UInt32",
            "UInt64" => "UInt64",
            "Float32" => "Float32",
            "Float64" => "Float64",
            "boolean" => "boolean",
            _ => "Int64",
        }),
        Dtype::Arrow(name) => {
            if let Some((p, s)) = dtype.arrow_decimal() {
                SampleKind::Arrow(DataType::Decimal128(p as u8, s as i8))
            } else {
                SampleKind::Arrow(match name.as_str() {
                    "bool" => DataType::Boolean,
                    "int8" => DataType::Int8,
                    "int16" => DataType::Int16,
                    "int32" => DataType::Int32,
                    "int64" => DataType::Int64,
                    "uint8" => DataType::UInt8,
                    "uint16" => DataType::UInt16,
                    "uint32" => DataType::UInt32,
                    "uint64" => DataType::UInt64,
                    "float" => DataType::Float32,
                    "double" => DataType::Float64,
                    "string" => DataType::Utf8,
                    "large_string" => DataType::LargeUtf8,
                    "binary" => DataType::Binary,
                    "date32[day]" => DataType::Date32,
                    "date64[ms]" => DataType::Date64,
                    n if n.starts_with("timestamp") => {
                        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
                    }
                    // An Arrow duration is no `timedelta64` to the engine,
                    // which reads none (`is_timedelta64_dtype` is false), so
                    // it is refused by name as other such types are.
                    _ => DataType::Null,
                })
            }
        }
        // No type the inference reads, so it refuses the column.
        Dtype::Other(_) => SampleKind::Arrow(DataType::Null),
    }
}

/// A column as the engine's inference reads it.
fn sample_column(series: &Series) -> SampleColumn {
    let cells = match series.dtype {
        Dtype::Category => series
            .cells
            .iter()
            .map(|v| match v {
                PyVal::Float(x) if x.is_nan() => PyVal::None,
                other => other.clone(),
            })
            .collect(),
        // An array is list-like, as a list is, to the inference.
        Dtype::Object => series
            .cells
            .iter()
            .map(|v| match v {
                PyVal::NdArray(_, items) => PyVal::List(items.clone()),
                other => other.clone(),
            })
            .collect(),
        _ => series.cells.clone(),
    };
    SampleColumn {
        name: series.name.clone().unwrap_or_default(),
        kind: sample_kind(&series.dtype),
        cells,
    }
}

/// `LocalTable`: a table backed by a frame.
#[derive(Clone, Debug)]
pub struct LocalTable {
    pub name: String,
    /// `_data`: the frame's columns, in order.
    pub data: Vec<Series>,
    pub rows: usize,
    pub columns: Vec<EColumn>,
    pub primary_key: Option<String>,
    /// The columns a composite key folds, where the key is one.
    pub primary_key_columns: Option<Vec<String>>,
    pub time_column: Option<String>,
    pub end_time_column: Option<String>,
}

impl LocalTable {
    /// `LocalTable(df, name)`: every column of the frame, each typed as the
    /// engine infers it, with no key and no time column.
    pub fn new(
        name: &str,
        data: Vec<Series>,
        rows: usize,
        host: &Host,
        said: &mut Warnings,
    ) -> Result<Self, Built> {
        Self::labelled(name, data, rows, &[], host, said)
    }

    /// [`LocalTable::new`] of a frame whose columns named in `not_text` are
    /// labelled by a value that is not text, which `ColumnSpec.coerce`
    /// refuses before any column is added.
    pub fn labelled(
        name: &str,
        data: Vec<Series>,
        rows: usize,
        not_text: &[String],
        host: &Host,
        said: &mut Warnings,
    ) -> Result<Self, Built> {
        if rows == 0 {
            return raise("ValueError", "Data frame is empty").map_err(Built::Engine);
        }
        let mut seen = std::collections::HashSet::new();
        if !data
            .iter()
            .all(|s| seen.insert(s.name.clone().unwrap_or_default()))
        {
            return raise("ValueError", "Data frame must have unique column names")
                .map_err(Built::Engine);
        }
        if data.iter().any(|s| s.name.as_deref() == Some("")) {
            return raise("ValueError", "Data frame must have non-empty column names")
                .map_err(Built::Engine);
        }
        if let Some(label) = data
            .iter()
            .filter_map(|s| s.name.as_deref())
            .find(|n| not_text.iter().any(|t| t == n))
        {
            return raise(
                "TypeError",
                format!("Unable to coerce 'ColumnSpec' from '{label}'"),
            )
            .map_err(Built::Engine);
        }
        let mut table = LocalTable {
            name: name.to_string(),
            data,
            rows,
            columns: Vec::new(),
            primary_key: None,
            primary_key_columns: None,
            time_column: None,
            end_time_column: None,
        };
        let names: Vec<String> = table
            .data
            .iter()
            .map(|s| s.name.clone().unwrap_or_default())
            .collect();
        for column in names {
            if let Err(error) = table.add_column(&column, host, said) {
                return Err(
                    match table
                        .data
                        .iter()
                        .find(|s| s.name.as_deref() == Some(&column))
                    {
                        // The engine has no reading of the column's type.
                        Some(series) if error.class == "RuntimeError" => Built::Refused(
                            no_reading_of(&table.name, &column, &series.dtype.name(), error),
                        ),
                        _ => Built::Engine(error),
                    },
                );
            }
        }
        Ok(table)
    }

    /// `add_column(name)` of a column of the frame.
    fn add_column(&mut self, name: &str, host: &Host, said: &mut Warnings) -> ER<()> {
        if self.has_column(name) {
            return raise(
                "KeyError",
                sdm_pql::pyrepr::str_repr(&format!(
                    "Column '{name}' already exists in table '{}'",
                    self.name
                )),
            );
        }
        let Some(series) = self.data.iter().find(|s| s.name.as_deref() == Some(name)) else {
            return raise(
                "ValueError",
                format!("Column '{name}' does not exist in the underlying source table"),
            );
        };
        let sample = sample_column(series);
        let dtype = infer_dtype(&sample).map_err(|why| {
            // Raised from the `ValueError` inference raised.
            PyExc::new(
                "RuntimeError",
                format!(
                    "Encountered unsupported data type '{}' for column '{name}' in table '{}'. \
                     Please either cast the column to a supported type before building the table \
                     (for example df['{name}'] = df['{name}'].astype('int64')) or remove the \
                     column from this table.",
                    series.dtype.name(),
                    self.name
                ),
            )
            .caused_by([sdm_schema::engine::link("ValueError", why.0)])
        })?;
        let (stype, warnings) = infer_stype(&sample, dtype, host);
        for warning in warnings {
            said.user(warning);
        }
        let stype = stype.map_err(|_| {
            PyExc::new(
                "RuntimeError",
                format!(
                    "Could not determine semantic type for column '{name}' with data type '{}' \
                     in table '{}'. Please either change the column's data type or remove the \
                     column from this table.",
                    dtype.value(),
                    self.name
                ),
            )
        })?;
        self.columns.push(EColumn {
            name: name.to_string(),
            expr: None,
            dtype,
            stype,
            is_pk: false,
            is_time: false,
            is_end: false,
        });
        Ok(())
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

    /// `primary_key_columns`.
    pub fn primary_key_columns(&self) -> Vec<String> {
        match &self.primary_key_columns {
            Some(names) => names.clone(),
            None => self.primary_key.iter().cloned().collect(),
        }
    }

    /// The `primary_key` setter.
    pub fn set_primary_key(
        &mut self,
        key: Option<&Key>,
        host: &Host,
        said: &mut Warnings,
    ) -> ER<()> {
        let name = match key {
            None => None,
            Some(Key::One(name)) => Some(name.clone()),
            Some(Key::Many(names)) if names.is_empty() => {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot use an empty composite primary key on table '{}'",
                        self.name
                    ),
                );
            }
            Some(Key::Many(names)) if names.len() > 1 => {
                return self.set_composite_primary_key(names, host, said);
            }
            Some(Key::Many(names)) => Some(names[0].clone()),
        };
        self.set_single_primary_key(name.as_deref())
    }

    fn set_single_primary_key(&mut self, name: Option<&str>) -> ER<()> {
        if let Some(n) = name {
            if Some(n) == self.time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a primary key since it is already \
                         defined to be a time column"
                    ),
                );
            }
            if Some(n) == self.end_time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a primary key since it is already \
                         defined to be an end time column"
                    ),
                );
            }
            if !self.has_column(n) {
                return Err(self.not_found(n));
            }
        }
        let mut stale = None;
        if self.primary_key_columns.is_some() && name != self.primary_key.as_deref() {
            stale = self.primary_key.take();
        }
        self.primary_key_columns = None;
        if let Some(old) = self.primary_key.clone()
            && let Some(c) = self.column_mut(&old)
        {
            c.is_pk = false;
        }
        let Some(n) = name else {
            self.primary_key = None;
            self.drop_derived_key(stale.as_deref());
            return Ok(());
        };
        let column = self.column_mut(n).expect("checked above");
        column.set_stype(Stype::Id)?;
        column.is_pk = true;
        self.primary_key = Some(n.to_string());
        self.drop_derived_key(stale.as_deref());
        Ok(())
    }

    /// `_drop_derived_key(derived)` and `LocalTable._forget_derived_key`.
    fn drop_derived_key(&mut self, derived: Option<&str>) {
        let Some(derived) = derived.filter(|d| d.starts_with(DERIVED_PREFIX)) else {
            return;
        };
        self.columns.retain(|c| c.name != derived);
        self.data.retain(|s| s.name.as_deref() != Some(derived));
    }

    /// `_derived_key_column_name(names)`.
    fn derived_key_column_name(&self, names: &[String]) -> ER<String> {
        let encoded = sdm_schema::composite::encode_identity(names)
            .map_err(|message| PyExc::new("ValueError", message))?;
        let derived = format!("{DERIVED_PREFIX}{encoded}");
        if self.has_column(&derived) && !derived.starts_with(DERIVED_PREFIX) {
            return raise(
                "ValueError",
                format!(
                    "Cannot fold {} into an identity for table '{}': it already holds a column \
                     named '{derived}'. Rename that column.",
                    sdm_schema::py::str_list_repr(names),
                    self.name
                ),
            );
        }
        Ok(derived)
    }

    /// `LocalTable._materialize_derived_key(derived, names)`: the folded
    /// identity as a new column.
    fn materialize_derived_key(
        &mut self,
        derived: &str,
        names: &[String],
        host: &Host,
        said: &mut Warnings,
    ) -> ER<()> {
        let frame: Vec<(String, Vec<PyVal>)> = self
            .data
            .iter()
            .map(|s| (s.name.clone().unwrap_or_default(), s.cells.clone()))
            .collect();
        let folded = sdm_schema::composite::encode_frame(&frame, names)
            .map_err(|message| PyExc::new("ValueError", message))?;
        let series = Series::new(
            Some(derived.to_string()),
            Dtype::Object,
            folded.into_iter().map(PyVal::Str).collect(),
        );
        match self
            .data
            .iter()
            .position(|s| s.name.as_deref() == Some(derived))
        {
            Some(at) => self.data[at] = series,
            None => self.data.push(series),
        }
        if !self.has_column(derived) {
            self.add_column(derived, host, said)?;
        }
        Ok(())
    }

    /// `_set_composite_primary_key(names)`.
    fn set_composite_primary_key(
        &mut self,
        names: &[String],
        host: &Host,
        said: &mut Warnings,
    ) -> ER<()> {
        let missing: Vec<String> = names
            .iter()
            .filter(|n| !self.has_column(n))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return raise(
                "ValueError",
                format!(
                    "Cannot use {} as a composite primary key of table '{}': column(s) {} are \
                     not present",
                    sdm_schema::py::str_list_repr(names),
                    self.name,
                    sdm_schema::py::str_list_repr(&missing)
                ),
            );
        }
        for name in names {
            if Some(name.as_str()) == self.time_column.as_deref()
                || Some(name.as_str()) == self.end_time_column.as_deref()
            {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot use column '{name}' in a composite primary key since it is \
                         already defined to be a time column"
                    ),
                );
            }
        }
        refuse_unfoldable_dtypes(
            &names
                .iter()
                .map(|n| (n.clone(), self.column(n).map(|c| c.dtype)))
                .collect::<Vec<_>>(),
        )?;
        let derived = self.derived_key_column_name(names)?;
        self.materialize_derived_key(&derived, names, host, said)?;
        self.set_single_primary_key(Some(&derived))?;
        self.primary_key_columns = Some(names.to_vec());
        Ok(())
    }

    /// The `time_column` setter.
    pub fn set_time_column(&mut self, name: Option<&str>) -> ER<()> {
        if let Some(n) = name {
            if Some(n) == self.primary_key.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a time column since it is already \
                         defined to be a primary key"
                    ),
                );
            }
            if Some(n) == self.end_time_column.as_deref() {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot specify column '{n}' as a time column since it is already \
                         defined to be an end time column"
                    ),
                );
            }
        }
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

    /// `table[name].stype = stype`.
    pub fn set_stype(&mut self, name: &str, stype: Stype) -> ER<()> {
        let Some(column) = self.column_mut(name) else {
            return Err(self.not_found(name));
        };
        column.set_stype(stype)
    }

    /// The frame `pa.Table.from_pandas(table._data)` gives the sampler.
    fn batch(&self) -> ER<arrow_array::RecordBatch> {
        crate::sampled::pandas_batch(&self.data, self.rows).map_err(|why| {
            PyExc::new(
                "ArrowInvalid",
                format!("table '{}' cannot be read by the core: {why}", self.name),
            )
        })
    }
}

/// `composite_key.refuse_unfoldable_dtypes(dtypes)`.
pub fn refuse_unfoldable_dtypes(dtypes: &[(String, Option<EDtype>)]) -> ER<()> {
    for (name, dtype) in dtypes {
        if let Some(dtype) = dtype
            && matches!(dtype, EDtype::Float)
        {
            return raise(
                "ValueError",
                format!(
                    "composite key column '{name}' is {}; a floating-point column cannot be \
                     part of a row's identity because its text is not reproducible. Use an \
                     exact column, or add a column holding the identity you intend.",
                    dtype.value()
                ),
            );
        }
    }
    Ok(())
}

/// The engine's `Graph`, over frames.
#[derive(Clone, Debug, Default)]
pub struct EngineGraph {
    pub tables: Vec<LocalTable>,
    /// `(src_table, fkey, dst_table)`, in the order linked.
    pub edges: Vec<(String, String, String)>,
}

impl EngineGraph {
    fn index(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| t.name == name)
    }

    /// `graph[name]`.
    pub fn table(&self, name: &str) -> ER<&LocalTable> {
        self.index(name)
            .map(|i| &self.tables[i])
            .ok_or_else(|| PyExc::key_error(&format!("Table '{name}' not found in graph")))
    }

    fn table_mut(&mut self, name: &str) -> ER<&mut LocalTable> {
        match self.index(name) {
            Some(i) => Ok(&mut self.tables[i]),
            None => Err(PyExc::key_error(&format!(
                "Table '{name}' not found in graph"
            ))),
        }
    }

    fn edge_repr(edge: &(String, String, String)) -> String {
        format!(
            "Edge(src_table={}, fkey={}, dst_table={})",
            sdm_pql::pyrepr::str_repr(&edge.0),
            sdm_pql::pyrepr::str_repr(&edge.1),
            sdm_pql::pyrepr::str_repr(&edge.2)
        )
    }

    /// `link(src_table, fkey, dst_table)`, one column named.
    pub fn link(&mut self, src: &str, fkey: &str, dst: &str) -> ER<()> {
        if let Some(i) = self.index(dst) {
            let destination = &self.tables[i];
            if destination.primary_key_columns.is_some()
                && destination.primary_key.as_deref() != Some(fkey)
            {
                return raise(
                    "ValueError",
                    format!(
                        "Cannot link '{src}' to '{dst}' on '{fkey}': '{dst}' identifies a row by \
                         {} together, so one column cannot name one of its rows. Reference \
                         every part of the identity.",
                        sdm_schema::py::str_list_repr(&destination.primary_key_columns())
                    ),
                );
            }
        }
        let edge = (src.to_string(), fkey.to_string(), dst.to_string());
        if self.edges.contains(&edge) {
            return raise(
                "ValueError",
                format!("{} already exists in the graph", Self::edge_repr(&edge)),
            );
        }
        if self.index(src).is_none() {
            return raise(
                "ValueError",
                format!("Source table '{src}' does not exist in the graph"),
            );
        }
        if self.index(dst).is_none() {
            return raise(
                "ValueError",
                format!("Destination table '{dst}' does not exist in the graph"),
            );
        }
        let source = self.table_mut(src)?;
        let Some(column) = source.column(fkey) else {
            return raise(
                "ValueError",
                format!("Source key '{fkey}' does not exist as a column in source table '{src}'"),
            );
        };
        if !Stype::Id.supports_dtype(column.dtype) {
            return raise(
                "ValueError",
                format!(
                    "Cannot use '{fkey}' in source table '{src}' as a foreign key due to its \
                     incompatible data type. Foreign keys must have data type 'int', 'float' or \
                     'string' (got '{}')",
                    column.dtype.value()
                ),
            );
        }
        source.set_stype(fkey, Stype::Id)?;
        self.edges.push(edge);
        Ok(())
    }

    /// `_edge_error(edge)`.
    fn edge_error(&self, edge: &(String, String, String)) -> ER<Option<String>> {
        let (src_table, fkey, dst_table) = edge;
        let shown = Self::edge_repr(edge);
        let src = self.table(src_table)?;
        let Some(src_key) = src.column(fkey) else {
            return Ok(Some(format!(
                "Edge {shown} is invalid since table '{src_table}' no longer has a column \
                 '{fkey}'. Remove the link with `unlink()` before removing the column."
            )));
        };
        let dst = self.table(dst_table)?;
        let Some(dst_key) = dst.primary_key.as_deref().and_then(|k| dst.column(k)) else {
            return Ok(Some(format!(
                "Edge {shown} is invalid since table '{dst_table}' does not have a primary key. \
                 Add either a primary key or remove the link before proceeding."
            )));
        };
        if src.primary_key.as_deref() == Some(fkey.as_str()) {
            return Ok(Some(format!(
                "Cannot treat the primary key of table '{src_table}' as a foreign key. Remove \
                 either the primary key or the link before proceeding."
            )));
        }
        let number = |d: EDtype| d.is_int() || d.is_float();
        let (src_number, src_string) = (number(src_key.dtype), src_key.dtype.is_string());
        let (dst_number, dst_string) = (number(dst_key.dtype), dst_key.dtype.is_string());
        if !src_number && !src_string {
            return Ok(Some(format!(
                "{shown} is invalid as foreign key must be a number or string (got '{}')",
                src_key.dtype.value()
            )));
        }
        if src_number != dst_number || src_string != dst_string {
            return Ok(Some(format!(
                "{shown} is invalid as foreign key '{fkey}' and primary key '{}' have \
                 incompatible data types (got foreign key data type '{}' and primary key data \
                 type '{}')",
                dst_key.name,
                src_key.dtype.value(),
                dst_key.dtype.value()
            )));
        }
        Ok(None)
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
            if let Some(reason) = self.edge_error(edge)? {
                return raise("ValueError", reason);
            }
        }
        Ok(())
    }

    /// What the sampler starts from: the graph's tables and edges, its
    /// frames, no state learned yet, and no arguments.
    pub fn inputs(&self) -> ER<Inputs> {
        let mut tables = Vec::new();
        let mut data = Vec::new();
        for table in &self.tables {
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
                        is_source: c.expr.is_none(),
                    })
                    .collect(),
                sql: None,
            });
            data.push((table.name.clone(), table.batch()?));
        }
        Ok(Inputs {
            backend: Backend::Local,
            tables,
            edges: self.edges.clone(),
            state: State::default(),
            arguments: Map::new(),
            data,
            argument_tables: Default::default(),
        })
    }
}

/// What a frame's column labels are, beyond their text, as the host's
/// describer tags them: the tables whose columns are a pandas `MultiIndex`,
/// and each column labelled by a value that is not text (`name_repr`).
#[derive(Clone, Debug, Default)]
pub struct Labels {
    pub multi_index: Vec<String>,
    /// `(table, column)`.
    pub not_text: Vec<(String, String)>,
}

impl Labels {
    fn not_text(&self, table: &str) -> Vec<String> {
        self.not_text
            .iter()
            .filter(|(t, _)| t == table)
            .map(|(_, c)| c.clone())
            .collect()
    }
}

/// `Graph.from_data(frames, edges=[], infer_metadata=False)`; the frames
/// named in `multi_index` have a pandas `MultiIndex` for columns, which
/// `LocalTable` refuses after an empty frame and before anything else.
pub fn from_data(
    frames: Vec<(String, Vec<Series>, usize)>,
    labels: &Labels,
    host: &Host,
    said: &mut Warnings,
) -> Result<EngineGraph, Built> {
    let mut graph = EngineGraph::default();
    for (name, columns, rows) in frames {
        if rows > 0 && labels.multi_index.contains(&name) {
            return raise("ValueError", "Data frame must not have a multi-index")
                .map_err(Built::Engine);
        }
        let not_text = labels.not_text(&name);
        let table = LocalTable::labelled(&name, columns, rows, &not_text, host, said)?;
        if graph.index(&name).is_some() {
            return Err(Built::Engine(PyExc::key_error(&format!(
                "Cannot add table with name '{name}' to this graph; table names must be \
                 globally unique."
            ))));
        }
        graph.tables.push(table);
    }
    Ok(graph)
}

/// The graph `_build_graph` builds of an in-memory schema: every key, time
/// column and semantic type written from the schema, and every relationship
/// linked, once.
pub fn build_graph(
    schema: &Schema,
    frames: Vec<(String, Vec<Series>, usize)>,
    labels: &Labels,
    host: &Host,
    said: &mut Warnings,
) -> Result<EngineGraph, Built> {
    let mut graph = from_data(frames, labels, host, said)?;
    for table in &schema.tables {
        let engine = graph.table_mut(&table.name).map_err(Built::Engine)?;
        engine
            .set_primary_key(table.primary_key.as_ref(), host, said)
            .map_err(Built::Engine)?;
        engine
            .set_time_column(table.time_column.as_deref())
            .map_err(Built::Engine)?;
        for column in &table.columns {
            if engine.has_column(&column.name) {
                let stype = Stype::parse(&column.stype).ok_or_else(|| {
                    Built::Engine(PyExc::new(
                        "ValueError",
                        format!("'{}' is not a valid Stype", column.stype),
                    ))
                })?;
                engine
                    .set_stype(&column.name, stype)
                    .map_err(Built::Engine)?;
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
        graph
            .link(&link.source_table, &link.source_column, &link.target_table)
            .map_err(Built::Engine)?;
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
        return Err(Built::Refused(crate::error::invalid(format!(
            "{} name a target column that is not the primary key of the table they point at, \
             and an edge can only join to that",
            misdirected.join(", ")
        ))));
    }
    if graph.edges.len() != linked.len() {
        return Err(Built::Refused(structured(
            "INTERNAL_ERROR",
            "the graph built for this request does not match the schema",
        )));
    }
    Ok(graph)
}

/// A column of a type the engine has no reading of, refused by name where
/// today's code ended in the engine's `RuntimeError`, which is kept as the
/// cause.
fn no_reading_of(table: &str, column: &str, dtype: &str, error: PyExc) -> sdm_protocol::Diagnostic {
    structured(
        "UNSUPPORTED_REQUEST",
        format!(
            "{table}.{column} is {dtype}, a type no model reads. Cast it to numbers, text or \
             dates before building the schema, for example df[{quoted}] = \
             df[{quoted}].astype('str'), or leave it out of the frame",
            quoted = sdm_pql::pyrepr::str_repr(column),
        ),
    )
    .caused_by(error.as_links())
}

/// Why a graph was not built: the engine raised, or the client refused.
#[derive(Debug)]
pub enum Built {
    Engine(PyExc),
    Refused(sdm_protocol::Diagnostic),
}

impl From<Built> for PyExc {
    /// The engine's error, or the refusal as the `ValueError` a caller of
    /// the engine alone would meet.
    fn from(built: Built) -> Self {
        match built {
            Built::Engine(error) => error,
            Built::Refused(refusal) => PyExc::new("ValueError", refusal.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(name: &str, values: &[i64]) -> Series {
        Series::new(
            Some(name.to_string()),
            Dtype::Int("int64".to_string()),
            values.iter().map(|&v| PyVal::Int(i128::from(v))).collect(),
        )
    }

    fn refusal(frames: Vec<(String, Vec<Series>, usize)>, multi_index: &[String]) -> String {
        let mut said = Warnings::default();
        let labels = Labels {
            multi_index: multi_index.to_vec(),
            not_text: Vec::new(),
        };
        match from_data(frames, &labels, &Host::assumed(), &mut said) {
            Ok(_) => String::new(),
            Err(error) => PyExc::from(error).message,
        }
    }

    /// `LocalTable.__init__` refuses repeated names before
    /// `ColumnSpec.coerce` refuses a label that is not text, the first of
    /// them in the frame's order.
    #[test]
    fn a_label_not_text_is_refused_after_the_frames_names() {
        let refused = |columns: Vec<Series>, not_text: &[&str]| {
            let labels = Labels {
                multi_index: Vec::new(),
                not_text: not_text
                    .iter()
                    .map(|c| ("users".to_string(), c.to_string()))
                    .collect(),
            };
            let mut said = Warnings::default();
            match from_data(
                vec![("users".into(), columns, 1)],
                &labels,
                &Host::assumed(),
                &mut said,
            ) {
                Ok(_) => String::new(),
                Err(error) => {
                    let error = PyExc::from(error);
                    format!("{}: {}", error.class, error.message)
                }
            }
        };
        assert_eq!(
            refused(
                vec![ids("a", &[1]), ids("5", &[2]), ids("7", &[3])],
                &["7", "5"]
            ),
            "TypeError: Unable to coerce 'ColumnSpec' from '5'"
        );
        assert_eq!(
            refused(vec![ids("5", &[1]), ids("5", &[2])], &["5"]),
            "ValueError: Data frame must have unique column names"
        );
        assert_eq!(refused(vec![ids("a", &[1]), ids("5", &[2])], &[]), "");
    }

    /// `_as_typed` finds a schema's column in its frame only under a key of
    /// the label's own type, and a decimal column cast under a key that is
    /// not text is refused as `frame.assign(**cast)` refuses it.
    #[test]
    fn a_decimal_column_is_cast_only_under_a_key_of_its_labels_type() {
        let decimals = Series::new(
            Some("5".to_string()),
            Dtype::Object,
            vec![PyVal::Decimal("1.5".to_string()); 2],
        );
        let mut frame = Table::from_series(&[ids("user_id", &[1, 2]), decimals], 2);
        let labelled = |frame: &mut Table, repr: Option<&str>| {
            let mut tag = frame.tag("5");
            if let (Some(repr), Some(map)) = (repr, tag.as_object_mut()) {
                map.insert("name_repr".into(), serde_json::json!(repr));
            }
            frame.tags.insert("5".into(), tag);
        };
        let table = |repr: Option<&str>| sdm_schema::client::Table {
            name: "users".to_string(),
            data: None,
            source_name: None,
            columns: vec![sdm_schema::client::Col {
                name: "5".to_string(),
                dtype: "float64".to_string(),
                stype: "numerical".to_string(),
                untyped: false,
                stype_declared: false,
                before_key: None,
                name_repr: repr.map(str::to_string),
            }],
            primary_key: None,
            time_column: None,
            duplicate_keys: None,
        };
        let host = Host::assumed();
        let cast = |frame: &Table, repr: Option<&str>| {
            as_typed(&table(repr), frame, &host).map(|columns| columns[1].dtype.clone())
        };
        // A text label, and a text key: cast.
        assert_eq!(cast(&frame, None).ok(), Some(Dtype::Float(64)));
        labelled(&mut frame, Some("5"));
        // An int label and a text key, '5' in frame.columns being False.
        assert_eq!(cast(&frame, None).ok(), Some(Dtype::Object));
        // An int label and an int key: refused.
        let refused = cast(&frame, Some("5")).expect_err("refused");
        assert_eq!(refused.code.as_deref(), Some("INTERNAL_ERROR"));
        assert_eq!(
            refused.message,
            "the relational engine was called incorrectly by this client: TypeError: keywords \
             must be strings"
        );
    }

    /// `LocalTable.__init__` refuses an empty frame first, then columns that
    /// are a `MultiIndex`, then repeated or empty names.
    #[test]
    fn a_multi_index_is_refused_after_an_empty_frame_and_before_its_names() {
        let multi = vec!["users".to_string()];
        let repeated = vec![ids("a", &[1]), ids("a", &[2])];
        assert_eq!(
            refusal(vec![("users".into(), repeated.clone(), 1)], &multi),
            "Data frame must not have a multi-index"
        );
        assert_eq!(
            refusal(vec![("users".into(), repeated.clone(), 0)], &multi),
            "Data frame is empty"
        );
        assert_eq!(
            refusal(vec![("users".into(), repeated, 1)], &[]),
            "Data frame must have unique column names"
        );
        assert_eq!(
            refusal(vec![("users".into(), vec![ids("a", &[1])], 1)], &[]),
            ""
        );
    }
}
