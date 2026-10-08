// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a run starts from: the description the host gives of the call
//! (`capture_sampling.py`, `describe`), read into the graph, the sampler's
//! state and the arguments, and the tables beside it.

use std::collections::HashMap;

use arrow_array::RecordBatch;
use serde_json::{Map, Value};

use super::Raised;
use crate::frame::{Dtype, Frame, Series};

fn bad(message: impl Into<String>) -> Raised {
    Raised::value_error(format!("the sampling inputs: {}", message.into()))
}

fn object<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>, Raised> {
    value
        .as_object()
        .ok_or_else(|| bad(format!("{what} is not an object")))
}

fn field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value, Raised> {
    object.get(name).ok_or_else(|| bad(format!("no {name}")))
}

pub(crate) fn text(value: &Value, what: &str) -> Result<String, Raised> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| bad(format!("{what} is not text")))
}

pub(crate) fn optional_text(value: Option<&Value>, what: &str) -> Result<Option<String>, Raised> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(v) => text(v, what).map(Some),
    }
}

pub(crate) fn int(value: &Value, what: &str) -> Result<i64, Raised> {
    value
        .as_i64()
        .ok_or_else(|| bad(format!("{what} is not an integer")))
}

fn texts(value: Option<&Value>, what: &str) -> Result<Vec<String>, Raised> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items.iter().map(|v| text(v, what)).collect(),
        Some(_) => Err(bad(format!("{what} is not a list"))),
    }
}

fn text_map(value: Option<&Value>, what: &str) -> Result<Vec<(String, String)>, Raised> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Object(items)) => items
            .iter()
            .map(|(k, v)| Ok((k.clone(), text(v, what)?)))
            .collect(),
        Some(_) => Err(bad(format!("{what} is not an object"))),
    }
}

/// The backend a sampler reads through (`DataBackend`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Local,
    Sqlite,
    Duckdb,
    Postgres,
    Snowflake,
    Databricks,
}

impl Backend {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "local" => Backend::Local,
            "sqlite" => Backend::Sqlite,
            "duckdb" => Backend::Duckdb,
            "postgres" => Backend::Postgres,
            "snowflake" => Backend::Snowflake,
            "databricks" => Backend::Databricks,
            _ => return None,
        })
    }

    /// The character the backend quotes identifiers with
    /// (`SQLSampler._QUOTE_CHAR`).
    pub fn quote_char(self) -> &'static str {
        match self {
            Backend::Databricks => "`",
            _ => "\"",
        }
    }
}

/// A column of a table, as the graph declares it.
#[derive(Clone, Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: Option<String>,
    pub stype: Option<String>,
    /// A `LocalExpression`'s text, for a column computed by one.
    pub expr: Option<String>,
    pub is_source: bool,
}

/// `SourceColumn`: what the database says of a column.
#[derive(Clone, Debug)]
pub struct SourceColumnInfo {
    pub name: String,
    pub dtype: Option<String>,
    pub is_primary_key: bool,
    pub is_unique_key: bool,
    pub is_nullable: bool,
}

/// What a SQL sampler reads of one table.
#[derive(Clone, Debug, Default)]
pub struct SqlTable {
    /// The quoted name statements read the table by.
    pub source_name: String,
    /// `Table.source_name`: the name as the database lists it.
    pub unquoted_name: Option<String>,
    pub source_columns: Vec<SourceColumnInfo>,
    /// DuckDB: the columns holding `TIMESTAMP WITH TIME ZONE` values.
    pub zoned: Vec<String>,
    /// PostgreSQL: each column's type, as casts spell it.
    pub postgres_types: Vec<(String, String)>,
    /// Snowflake: each source column's type.
    pub snowflake_types: Vec<(String, String)>,
    /// Snowflake: the expression columns read as zoned.
    pub zoned_expr_columns: Vec<String>,
    /// Snowflake: what `SHOW TABLES` listed of tables named like this one,
    /// where it was read already.
    pub listed: Option<Value>,
    /// Snowflake: the database, schema and source name `SHOW TABLES` asks
    /// about.
    pub identity: Option<(String, String, String)>,
}

impl SqlTable {
    pub fn source_column(&self, name: &str) -> Option<&SourceColumnInfo> {
        self.source_columns.iter().find(|c| c.name == name)
    }
}

/// A table of the graph.
#[derive(Clone, Debug)]
pub struct TableInfo {
    pub name: String,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
    pub end_time_column: Option<String>,
    pub columns: Vec<ColumnInfo>,
    pub sql: Option<SqlTable>,
}

impl TableInfo {
    pub fn column(&self, name: &str) -> Option<&ColumnInfo> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn sql(&self) -> Result<&SqlTable, Raised> {
        self.sql
            .as_ref()
            .ok_or_else(|| bad(format!("table {} has no SQL description", self.name)))
    }
}

/// How one SQLite table writes its times (`_TimeSpelling`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeSpelling {
    pub separator: String,
    pub digits: i64,
    pub suffix: String,
}

impl Default for TimeSpelling {
    fn default() -> Self {
        Self {
            separator: " ".to_string(),
            digits: 0,
            suffix: String::new(),
        }
    }
}

/// What the sampler holds from earlier calls.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// `_min_time_dict` and `_max_time_dict`, in nanoseconds.
    pub min_time: HashMap<String, i64>,
    pub max_time: HashMap<String, i64>,
    /// The unit of each of those `Timestamp`s, such as `us`.
    pub min_time_unit: HashMap<String, String>,
    pub max_time_unit: HashMap<String, String>,
    /// The module's `_coverage_warned`.
    pub coverage_warned: bool,
    pub enforced_key_tables: Vec<String>,
    /// `num_rows_dict`: `None` where the backend could not count.
    pub num_rows: HashMap<String, Option<i64>>,
    /// SQLite's `_time_spelling_dict`.
    pub time_spellings: HashMap<String, TimeSpelling>,
    /// The in-memory neighbor sampler's generator as the last run left it
    /// (its key and position), which today's model keeps between calls.
    pub neighbor_generator: Option<(Vec<u32>, usize)>,
}

/// Everything a run starts from.
#[derive(Clone, Debug)]
pub struct Inputs {
    pub backend: Backend,
    pub tables: Vec<TableInfo>,
    pub edges: Vec<(String, String, String)>,
    pub state: State,
    pub arguments: Map<String, Value>,
    /// The tables the host gave, by name: in memory the graph's own, and
    /// the arguments that are a `Series`.
    pub data: Vec<(String, RecordBatch)>,
    /// Which of [`Inputs::data`] holds each argument that is a `Series`.
    pub argument_tables: HashMap<String, String>,
}

fn column(value: &Value) -> Result<ColumnInfo, Raised> {
    let c = object(value, "a column")?;
    Ok(ColumnInfo {
        name: text(field(c, "name")?, "a column's name")?,
        dtype: optional_text(c.get("dtype"), "a dtype")?,
        stype: optional_text(c.get("stype"), "an stype")?,
        expr: optional_text(c.get("expr"), "an expression")?,
        is_source: c.get("is_source").and_then(Value::as_bool).unwrap_or(true),
    })
}

fn source_column(value: &Value) -> Result<SourceColumnInfo, Raised> {
    let c = object(value, "a source column")?;
    let flag = |name: &str| c.get(name).and_then(Value::as_bool).unwrap_or(false);
    Ok(SourceColumnInfo {
        name: text(field(c, "name")?, "a source column's name")?,
        dtype: optional_text(c.get("dtype"), "a dtype")?,
        is_primary_key: flag("is_primary_key"),
        is_unique_key: flag("is_unique_key"),
        is_nullable: flag("is_nullable"),
    })
}

fn sql_table(value: &Value) -> Result<SqlTable, Raised> {
    let s = object(value, "a table's SQL")?;
    let identity = match s.get("identity") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let i = object(v, "an identity")?;
            Some((
                text(field(i, "database")?, "a database")?,
                text(field(i, "schema")?, "a schema")?,
                text(field(i, "source_name")?, "a source name")?,
            ))
        }
    };
    Ok(SqlTable {
        source_name: text(field(s, "source_name")?, "a source name")?,
        unquoted_name: optional_text(s.get("unquoted_name"), "a source name")?,
        source_columns: match s.get("source_columns") {
            Some(Value::Array(items)) => {
                items.iter().map(source_column).collect::<Result<_, _>>()?
            }
            _ => Vec::new(),
        },
        zoned: texts(s.get("zoned"), "a zoned column")?,
        postgres_types: text_map(s.get("postgres_types"), "a PostgreSQL type")?,
        snowflake_types: text_map(s.get("snowflake_types"), "a Snowflake type")?,
        zoned_expr_columns: texts(s.get("zoned_expr_columns"), "a zoned column")?,
        listed: s.get("listed").filter(|v| !v.is_null()).cloned(),
        identity,
    })
}

fn table(value: &Value) -> Result<TableInfo, Raised> {
    let t = object(value, "a table")?;
    Ok(TableInfo {
        name: text(field(t, "name")?, "a table's name")?,
        primary_key: optional_text(t.get("primary_key"), "a primary key")?,
        time_column: optional_text(t.get("time_column"), "a time column")?,
        end_time_column: optional_text(t.get("end_time_column"), "an end time column")?,
        columns: match t.get("columns") {
            Some(Value::Array(items)) => items.iter().map(column).collect::<Result<_, _>>()?,
            _ => return Err(bad("a table has no columns")),
        },
        sql: match t.get("sql") {
            None | Some(Value::Null) => None,
            Some(v) => Some(sql_table(v)?),
        },
    })
}

fn times(value: Option<&Value>) -> Result<HashMap<String, i64>, Raised> {
    match value {
        None | Some(Value::Null) => Ok(HashMap::new()),
        Some(Value::Object(items)) => items
            .iter()
            .map(|(k, v)| Ok((k.clone(), int(v, "a time")?)))
            .collect(),
        Some(_) => Err(bad("times are not an object")),
    }
}

fn state(value: &Value) -> Result<State, Raised> {
    let s = object(value, "the state")?;
    let num_rows = match s.get("num_rows") {
        None | Some(Value::Null) => HashMap::new(),
        Some(Value::Object(items)) => items
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    if v.is_null() {
                        None
                    } else {
                        Some(int(v, "a row count")?)
                    },
                ))
            })
            .collect::<Result<_, Raised>>()?,
        Some(_) => return Err(bad("row counts are not an object")),
    };
    let time_spellings = match s.get("time_spellings") {
        None | Some(Value::Null) => HashMap::new(),
        Some(Value::Object(items)) => items
            .iter()
            .map(|(k, v)| {
                let parts = v.as_array().ok_or_else(|| bad("a time spelling"))?;
                if parts.len() != 3 {
                    return Err(bad("a time spelling"));
                }
                Ok((
                    k.clone(),
                    TimeSpelling {
                        separator: text(&parts[0], "a separator")?,
                        digits: int(&parts[1], "digits")?,
                        suffix: text(&parts[2], "a suffix")?,
                    },
                ))
            })
            .collect::<Result<_, Raised>>()?,
        Some(_) => return Err(bad("time spellings are not an object")),
    };
    Ok(State {
        min_time: times(s.get("min_time"))?,
        max_time: times(s.get("max_time"))?,
        min_time_unit: text_map(s.get("min_time_unit"), "a unit")?
            .into_iter()
            .collect(),
        max_time_unit: text_map(s.get("max_time_unit"), "a unit")?
            .into_iter()
            .collect(),
        coverage_warned: s
            .get("coverage_warned")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        enforced_key_tables: texts(s.get("enforced_key_tables"), "a table")?,
        num_rows,
        time_spellings,
        neighbor_generator: match s.get("neighbor_generator") {
            None | Some(Value::Null) => None,
            Some(v) => {
                let g = object(v, "a generator")?;
                let key = match field(g, "key")? {
                    Value::Array(words) => words
                        .iter()
                        .map(|w| {
                            w.as_u64()
                                .and_then(|w| u32::try_from(w).ok())
                                .ok_or_else(|| bad("a generator's word"))
                        })
                        .collect::<Result<Vec<u32>, Raised>>()?,
                    _ => return Err(bad("a generator's key is not a list")),
                };
                let pos = field(g, "pos")?
                    .as_u64()
                    .and_then(|p| usize::try_from(p).ok())
                    .ok_or_else(|| bad("a generator's position"))?;
                Some((key, pos))
            }
        },
    })
}

impl Inputs {
    pub fn parse(text_json: &str, data: Vec<(String, RecordBatch)>) -> Result<Self, Raised> {
        let value: Value =
            serde_json::from_str(text_json).map_err(|e| bad(format!("not JSON: {e}")))?;
        let root = object(&value, "the inputs")?;
        let backend_name = text(field(root, "backend")?, "the backend")?;
        let backend = Backend::parse(&backend_name)
            .ok_or_else(|| bad(format!("no backend {backend_name:?}")))?;
        let tables = match field(root, "tables")? {
            Value::Array(items) => items.iter().map(table).collect::<Result<_, _>>()?,
            _ => return Err(bad("tables is not a list")),
        };
        let edges = match field(root, "edges")? {
            Value::Array(items) => items
                .iter()
                .map(|edge| {
                    let parts = edge.as_array().ok_or_else(|| bad("an edge"))?;
                    if parts.len() != 3 {
                        return Err(bad("an edge is not three names"));
                    }
                    Ok((
                        text(&parts[0], "an edge's source")?,
                        text(&parts[1], "an edge's key")?,
                        text(&parts[2], "an edge's destination")?,
                    ))
                })
                .collect::<Result<_, _>>()?,
            _ => return Err(bad("edges is not a list")),
        };
        let arguments = object(field(root, "arguments")?, "the arguments")?.clone();
        let argument_tables = text_map(root.get("argument_tables"), "an argument table")?
            .into_iter()
            .collect();
        Ok(Self {
            backend,
            tables,
            edges,
            state: state(field(root, "state")?)?,
            arguments,
            data,
            argument_tables,
        })
    }

    pub fn table(&self, name: &str) -> Result<&TableInfo, Raised> {
        self.tables
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Raised::key_error(name))
    }

    /// The host's table `name`.
    pub fn data(&self, name: &str) -> Result<&RecordBatch, Raised> {
        self.data
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
            .ok_or_else(|| bad(format!("no table {name:?} was given")))
    }

    /// The argument that is a `Series`, as the host gave it: its one column
    /// with the dtype the arguments name.
    pub fn series_argument(&self, name: &str, dtype_key: &str) -> Result<Series, Raised> {
        let table = self
            .argument_tables
            .get(name)
            .ok_or_else(|| bad(format!("no table holds {name}")))?;
        let batch = self.data(table)?;
        if batch.num_columns() != 1 {
            return Err(bad(format!("{name} is not one column")));
        }
        let dtype = optional_text(self.arguments.get(dtype_key), "a dtype")?;
        let array = batch.column(0).clone();
        let dtype = Dtype::of_column(dtype.as_deref(), array.as_ref());
        Ok(Series::new(dtype, array))
    }

    pub fn argument(&self, name: &str) -> Result<&Value, Raised> {
        field(&self.arguments, name)
    }
}

/// A table the host gave as `pa.Table.from_pandas` writes a frame, read
/// back as that frame: each column's dtype from the pandas metadata.
pub fn frame_of(batch: &RecordBatch) -> Frame {
    let schema = batch.schema();
    let metadata: HashMap<String, String> = schema
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let dtypes = sdm_labels::arrow::pandas_dtypes(&metadata);
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let columns = names
        .iter()
        .zip(batch.columns())
        .map(|(name, array)| {
            let dtype = Dtype::of_column(dtypes.get(name).map(String::as_str), array.as_ref());
            Series::new(dtype, array.clone())
        })
        .collect();
    Frame::new(names, columns, batch.num_rows())
}

impl Inputs {
    /// The description [`Inputs::parse`] reads, of everything but the
    /// tables the host gives beside it.
    pub fn to_json(&self) -> Value {
        use serde_json::json;
        let backend = match self.backend {
            Backend::Local => "local",
            Backend::Sqlite => "sqlite",
            Backend::Duckdb => "duckdb",
            Backend::Postgres => "postgres",
            Backend::Snowflake => "snowflake",
            Backend::Databricks => "databricks",
        };
        let pairs = |items: &[(String, String)]| -> Value {
            Value::Object(items.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
        };
        let tables: Vec<Value> = self
            .tables
            .iter()
            .map(|t| {
                let mut entry = json!({
                    "name": t.name,
                    "primary_key": t.primary_key,
                    "time_column": t.time_column,
                    "end_time_column": t.end_time_column,
                    "columns": t.columns.iter().map(|c| json!({
                        "name": c.name,
                        "dtype": c.dtype,
                        "stype": c.stype,
                        "expr": c.expr,
                        "is_source": c.is_source,
                    })).collect::<Vec<_>>(),
                });
                if let Some(sql) = &t.sql {
                    entry["sql"] = json!({
                        "source_name": sql.source_name,
                        "unquoted_name": sql.unquoted_name,
                        "source_columns": sql.source_columns.iter().map(|c| json!({
                            "name": c.name,
                            "dtype": c.dtype,
                            "is_primary_key": c.is_primary_key,
                            "is_unique_key": c.is_unique_key,
                            "is_nullable": c.is_nullable,
                        })).collect::<Vec<_>>(),
                        "zoned": sql.zoned,
                        "postgres_types": pairs(&sql.postgres_types),
                        "snowflake_types": pairs(&sql.snowflake_types),
                        "zoned_expr_columns": sql.zoned_expr_columns,
                        "listed": sql.listed,
                        "identity": sql.identity.as_ref().map(|(d, s, n)| json!({
                            "database": d, "schema": s, "source_name": n,
                        })),
                    });
                }
                entry
            })
            .collect();
        let sorted = |map: &HashMap<String, i64>| -> Value {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), json!(map[k])))
                    .collect(),
            )
        };
        let sorted_text = |map: &HashMap<String, String>| -> Value {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), json!(map[k])))
                    .collect(),
            )
        };
        let mut num_rows: Vec<(&String, &Option<i64>)> = self.state.num_rows.iter().collect();
        num_rows.sort();
        let mut spellings: Vec<(&String, &TimeSpelling)> =
            self.state.time_spellings.iter().collect();
        spellings.sort_by(|a, b| a.0.cmp(b.0));
        let mut described = json!({
            "backend": backend,
            "tables": tables,
            "edges": self.edges.iter().map(|(s, f, d)| json!([s, f, d])).collect::<Vec<_>>(),
            "state": {
                "min_time": sorted(&self.state.min_time),
                "max_time": sorted(&self.state.max_time),
                "min_time_unit": sorted_text(&self.state.min_time_unit),
                "max_time_unit": sorted_text(&self.state.max_time_unit),
                "coverage_warned": self.state.coverage_warned,
                "enforced_key_tables": self.state.enforced_key_tables,
                "num_rows": Value::Object(num_rows.into_iter().map(|(k, v)| (k.clone(), json!(v))).collect()),
                "time_spellings": Value::Object(spellings.into_iter().map(|(k, s)| (k.clone(), json!([s.separator, s.digits, s.suffix]))).collect()),
            },
            "arguments": {},
            "argument_tables": {},
        });
        if let Some((key, pos)) = &self.state.neighbor_generator {
            described["state"]["neighbor_generator"] = json!({"key": key, "pos": pos});
        }
        described
    }
}
