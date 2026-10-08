// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`render_json`]: one call of the dialect-call capture, answered.
//!
//! A call names a backend, a function as the capture names it, and its
//! arguments as the capture encodes them, every parameter by name. The
//! arguments are read into the typed inputs the function takes, strictly: a
//! missing, extra or mistyped argument is an error, never a guess.

use serde_json::{Value, json};

use crate::arrow::{ArrowArray, ArrowType};
use crate::fragments::{self, SourceColumn};
use crate::helpers::{self, Described, Listing, TIME_SPELLING_CLASS, TimeSpelling};
use crate::int::Int;
use crate::py::{Dtype, Fraction, Py, Series, float_json};
use crate::{Error, KeyCountPart, Param, Statement};
use crate::{databricks, duckdb, postgres, snow, sqlite, view_expr};

const BACKENDS: [&str; 5] = ["databricks", "duckdb", "postgres", "snow", "sqlite"];

/// What a call gives.
enum Output {
    Statement(Statement),
    Value(Py),
}

impl From<Statement> for Output {
    fn from(statement: Statement) -> Self {
        Self::Statement(statement)
    }
}

impl From<Py> for Output {
    fn from(value: Py) -> Self {
        Self::Value(value)
    }
}

fn text(value: impl Into<String>) -> Output {
    Output::Value(Py::Str(value.into()))
}

fn optional_text(value: Option<impl Into<String>>) -> Output {
    Output::Value(value.map_or(Py::None, |v| Py::Str(v.into())))
}

fn param_json(param: &Param) -> Value {
    match param {
        Param::Null => Value::Null,
        Param::Bool(b) => json!(b),
        Param::Int(i) => json!(i),
        Param::Float(x) => float_json(*x),
        Param::Text(s) => json!(s),
    }
}

/// A call's output, its parts as JSON text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rendered {
    /// A statement function's SQL, and its parameters as the capture
    /// encodes them, `null` where it takes none.
    Statement { sql: String, params_json: String },
    /// Any other function's result, as the capture encodes it.
    Value { result_json: String },
}

/// One call: `function` of `backend`, as the capture names them, with
/// `args_json`, the call's `args`.
pub fn render(backend: &str, function: &str, args_json: &str) -> Result<Rendered, Error> {
    if !BACKENDS.contains(&backend) {
        return Err(Error::new(format!("no backend {backend:?}")));
    }
    let value: Value = serde_json::from_str(args_json)
        .map_err(|e| Error::new(format!("the arguments are not JSON: {e}")))?;
    let Value::Object(map) = value else {
        return Err(Error::new("the arguments are not a JSON object"));
    };
    let args = map
        .iter()
        .map(|(k, v)| Ok((k.clone(), Py::decode(v)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    let to_text =
        |value: &Value| serde_json::to_string(value).map_err(|e| Error::new(e.to_string()));
    Ok(match call(backend, function, &Args(args))? {
        Output::Statement((sql, params)) => {
            let params = params.map_or(Value::Null, |p| {
                Value::Array(p.iter().map(param_json).collect())
            });
            Rendered::Statement {
                sql,
                params_json: to_text(&params)?,
            }
        }
        Output::Value(value) => Rendered::Value {
            result_json: to_text(&value.encode()?)?,
        },
    })
}

/// The JSON text `{"sql": ..., "params": ...}` of a statement function's
/// call, or `{"result": ...}` of any other, encoded as the capture encodes
/// them. `args_json` is the call's `args`.
pub fn render_json(backend: &str, function: &str, args_json: &str) -> Result<String, Error> {
    let parse =
        |text: &str| serde_json::from_str::<Value>(text).map_err(|e| Error::new(e.to_string()));
    let rendered = match render(backend, function, args_json)? {
        Rendered::Statement { sql, params_json } => {
            json!({"sql": sql, "params": parse(&params_json)?})
        }
        Rendered::Value { result_json } => json!({"result": parse(&result_json)?}),
    };
    serde_json::to_string(&rendered).map_err(|e| Error::new(e.to_string()))
}

/// A call's arguments, by name.
struct Args(Vec<(String, Py)>);

impl Args {
    /// Requires the arguments to be exactly `names`.
    fn expect(&self, names: &[&str]) -> Result<(), Error> {
        let mut given: Vec<&str> = self.0.iter().map(|(k, _)| k.as_str()).collect();
        let mut wanted = names.to_vec();
        given.sort_unstable();
        wanted.sort_unstable();
        if given != wanted {
            return Err(Error::new(format!(
                "the arguments are {given:?}, not {wanted:?}"
            )));
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Result<&Py, Error> {
        self.0
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| Error::new(format!("no argument {name:?}")))
    }

    fn wrong(name: &str, what: &str, value: &Py) -> Error {
        Error::new(format!("{name} is {value:?}, not {what}"))
    }

    fn str(&self, name: &str) -> Result<&str, Error> {
        match self.get(name)? {
            Py::Str(s) => Ok(s),
            other => Err(Self::wrong(name, "text", other)),
        }
    }

    fn opt_str(&self, name: &str) -> Result<Option<&str>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.str(name).map(Some),
        }
    }

    fn strs(&self, name: &str) -> Result<Vec<String>, Error> {
        self.items(name)?
            .iter()
            .map(|item| match item {
                Py::Str(s) => Ok(s.clone()),
                other => Err(Self::wrong(name, "a list of text", other)),
            })
            .collect()
    }

    fn items(&self, name: &str) -> Result<&[Py], Error> {
        match self.get(name)? {
            Py::List(items) | Py::Tuple(items) => Ok(items),
            other => Err(Self::wrong(name, "a list", other)),
        }
    }

    fn opt_items(&self, name: &str) -> Result<Option<&[Py]>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.items(name).map(Some),
        }
    }

    fn big(&self, name: &str) -> Result<&Int, Error> {
        match self.get(name)? {
            Py::Int(int) => Ok(int),
            other => Err(Self::wrong(name, "an integer", other)),
        }
    }

    fn opt_big(&self, name: &str) -> Result<Option<&Int>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.big(name).map(Some),
        }
    }

    fn int(&self, name: &str) -> Result<i64, Error> {
        self.big(name)?
            .to_i64()
            .ok_or_else(|| Error::new(format!("{name} is out of range")))
    }

    fn opt_int(&self, name: &str) -> Result<Option<i64>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.int(name).map(Some),
        }
    }

    fn bool(&self, name: &str) -> Result<bool, Error> {
        match self.get(name)? {
            Py::Bool(b) => Ok(*b),
            other => Err(Self::wrong(name, "a bool", other)),
        }
    }

    fn opt_float(&self, name: &str) -> Result<Option<f64>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            Py::Float(x) => Ok(Some(*x)),
            other => Err(Self::wrong(name, "a float", other)),
        }
    }

    fn dtype(&self, name: &str) -> Result<Dtype, Error> {
        match self.get(name)? {
            Py::Dtype(dtype) => Ok(*dtype),
            other => Err(Self::wrong(name, "a Dtype", other)),
        }
    }

    fn opt_dtype(&self, name: &str) -> Result<Option<Dtype>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.dtype(name).map(Some),
        }
    }

    fn arrow(&self, name: &str) -> Result<&ArrowArray, Error> {
        match self.get(name)? {
            Py::Arrow(array) => Ok(array),
            other => Err(Self::wrong(name, "an Arrow array", other)),
        }
    }

    fn opt_arrow(&self, name: &str) -> Result<Option<&ArrowArray>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            _ => self.arrow(name).map(Some),
        }
    }

    fn arrow_type(&self, name: &str) -> Result<&ArrowType, Error> {
        match self.get(name)? {
            Py::ArrowType(t) => Ok(t),
            other => Err(Self::wrong(name, "an Arrow type", other)),
        }
    }

    fn fraction(&self, name: &str) -> Result<Fraction, Error> {
        match self.get(name)? {
            Py::Fraction(f) => Ok(*f),
            other => Err(Self::wrong(name, "a Fraction", other)),
        }
    }

    fn series(&self, name: &str) -> Result<&Series, Error> {
        match self.get(name)? {
            Py::Series(series) => Ok(series),
            other => Err(Self::wrong(name, "a Series", other)),
        }
    }

    /// A list of `(text, text)` pairs.
    fn pairs(&self, name: &str) -> Result<Vec<(String, String)>, Error> {
        self.items(name)?
            .iter()
            .map(|item| match item {
                Py::Tuple(pair) | Py::List(pair) => match pair.as_slice() {
                    [Py::Str(a), Py::Str(b)] => Ok((a.clone(), b.clone())),
                    _ => Err(Self::wrong(name, "a list of pairs of text", item)),
                },
                other => Err(Self::wrong(name, "a list of pairs of text", other)),
            })
            .collect()
    }

    /// A list of `(name, Arrow array)` pairs.
    fn key_columns(&self, name: &str) -> Result<Vec<(String, ArrowArray)>, Error> {
        self.items(name)?
            .iter()
            .map(|item| match item {
                Py::Tuple(pair) | Py::List(pair) => match pair.as_slice() {
                    [Py::Str(a), Py::Arrow(b)] => Ok((a.clone(), b.clone())),
                    _ => Err(Self::wrong(name, "a list of (name, array) pairs", item)),
                },
                other => Err(Self::wrong(name, "a list of (name, array) pairs", other)),
            })
            .collect()
    }

    fn parts(&self, name: &str) -> Result<Vec<KeyCountPart>, Error> {
        self.items(name)?
            .iter()
            .map(|item| {
                let fields = record(item, "sdm_relational_engine.rfm.base.source.KeyCountPart")?;
                let fields = Args(fields.to_vec());
                fields.expect(&["label", "source", "columns", "unique"])?;
                Ok(KeyCountPart {
                    label: fields.int("label")?,
                    source: fields.str("source")?.to_string(),
                    columns: fields.strs("columns")?,
                    unique: fields.bool("unique")?,
                })
            })
            .collect()
    }

    fn source_columns(&self, name: &str) -> Result<Vec<SourceColumn>, Error> {
        self.items(name)?
            .iter()
            .map(|item| {
                let fields = record(item, "sdm_relational_engine.rfm.base.source.SourceColumn")?;
                let fields = Args(fields.to_vec());
                Ok(SourceColumn {
                    name: fields.str("name")?.to_string(),
                    dtype: fields.opt_dtype("dtype")?,
                })
            })
            .collect()
    }

    fn quote(&self, name: &str) -> Result<char, Error> {
        match self.get(name)? {
            Py::Quote(c) => Ok(*c),
            other => Err(Self::wrong(name, "a quoting function", other)),
        }
    }

    /// Ids bound as parameters, one each.
    fn params(&self, name: &str) -> Result<Option<Vec<Param>>, Error> {
        self.opt_items(name)?
            .map(|items| items.iter().map(param).collect())
            .transpose()
    }

    /// A dict of text keys.
    fn dict(&self, name: &str) -> Result<Option<Vec<(String, Py)>>, Error> {
        match self.get(name)? {
            Py::None => Ok(None),
            Py::Dict(fields) => Ok(Some(fields.clone())),
            other => Err(Self::wrong(name, "a dict", other)),
        }
    }
}

fn record<'a>(item: &'a Py, class: &str) -> Result<&'a [(String, Py)], Error> {
    match item {
        Py::Record(c, fields) if c == class => Ok(fields),
        other => Err(Error::new(format!("{other:?} is not a {class}"))),
    }
}

/// A Python value as a driver binds it.
fn param(value: &Py) -> Result<Param, Error> {
    Ok(match value {
        Py::None => Param::Null,
        Py::Bool(b) => Param::Bool(*b),
        Py::Int(int) => Param::Int(
            int.to_i64()
                .ok_or_else(|| Error::new("an integer parameter out of range"))?,
        ),
        Py::Float(x) => Param::Float(*x),
        Py::Str(s) => Param::Text(s.clone()),
        Py::Np(_, value) => param(value)?,
        other => return Err(Error::new(format!("no parameter of {other:?}"))),
    })
}

fn texts(values: Vec<String>) -> Py {
    Py::List(values.into_iter().map(Py::Str).collect())
}

fn keys_record(backend: &str, sql: String, param: String) -> Py {
    Py::Record(
        format!("sdm_relational_engine.rfm.backend.{backend}.dialect._Keys"),
        vec![
            ("sql".into(), Py::Str(sql)),
            ("param".into(), Py::Str(param)),
        ],
    )
}

fn chunks(chunks: Vec<Vec<Py>>) -> Py {
    Py::List(chunks.into_iter().map(Py::List).collect())
}

fn call(backend: &str, function: &str, a: &Args) -> Result<Output, Error> {
    if let Some(output) = shared(function, a)? {
        return Ok(output);
    }
    match backend {
        "sqlite" => sqlite_call(function, a),
        "duckdb" => duckdb_call(function, a),
        "postgres" => postgres_call(function, a),
        "databricks" => databricks_call(function, a),
        _ => snow_call(function, a),
    }
}

/// The functions every backend answers alike.
fn shared(function: &str, a: &Args) -> Result<Option<Output>, Error> {
    Ok(Some(match function {
        "sql.quote_ident" => {
            a.expect(&["ident", "char"])?;
            text(fragments::quote_ident(a.str("ident")?, a.str("char")?))
        }
        "utils.ordered_sample_sql" => {
            a.expect(&["source", "projections", "order_terms", "num_rows"])?;
            text(fragments::ordered_sample_sql(
                a.str("source")?,
                &a.strs("projections")?,
                &a.strs("order_terms")?,
                a.int("num_rows")?,
            ))
        }
        "utils.key_count_select" => {
            a.expect(&["label", "source", "columns", "unique", "count_distinct"])?;
            text(fragments::key_count_select(
                a.int("label")?,
                a.str("source")?,
                &a.strs("columns")?,
                a.bool("unique")?,
                a.str("count_distinct")?,
            ))
        }
        "utils.count_distinct_values" => {
            a.expect(&["columns"])?;
            text(fragments::count_distinct_values(&a.strs("columns")?))
        }
        "utils.repeated_key_select" => {
            a.expect(&["source", "columns"])?;
            text(fragments::repeated_key_select(
                a.str("source")?,
                &a.strs("columns")?,
            ))
        }
        "utils.sample_order_terms" => {
            a.expect(&["columns", "primary_key", "quote", "hash_function"])?;
            let char = a.quote("quote")?.to_string();
            Output::Value(texts(fragments::sample_order_terms(
                &a.source_columns("columns")?,
                a.opt_str("primary_key")?,
                |name| fragments::quote_ident(name, &char),
                a.str("hash_function")?,
            )))
        }
        "utils.percent_literal" => {
            a.expect(&["fraction"])?;
            text(fragments::percent_literal(a.fraction("fraction")?)?)
        }
        "utils.prefilter_fraction" => {
            a.expect(&["wanted"])?;
            Output::Value(
                fragments::prefilter_fraction(a.fraction("wanted")?)?
                    .map_or(Py::None, Py::Fraction),
            )
        }
        "utils.random_state.randint" => {
            a.expect(&["random_seed", "low", "high", "size"])?;
            let seed = a.big("random_seed")?;
            if seed.is_negative() {
                return Err(Error::new("Seed must be between 0 and 2**32 - 1"));
            }
            let size =
                usize::try_from(a.int("size")?).map_err(|_| Error::new("a negative size"))?;
            let mut state = sdm_random::random_state(&seed.magnitude_words())
                .map_err(|e| Error::new(e.to_string()))?;
            let drawn = state
                .randint(a.int("low")?, a.int("high")?, size)
                .map_err(|e| Error::new(e.to_string()))?;
            Output::Value(Py::List(drawn.into_iter().map(Py::int).collect()))
        }
        "view_expr._unsafe_expr_reason" => {
            a.expect(&["expr", "dialect"])?;
            optional_text(view_expr::unsafe_expr_reason(
                a.str("expr")?,
                a.opt_str("dialect")?,
            ))
        }
        "view_expr._untrusted_call_reason" => {
            a.expect(&["expr", "dialect", "columns"])?;
            optional_text(view_expr::untrusted_call_reason(
                a.str("expr")?,
                a.str("dialect")?,
                &a.strs("columns")?,
            )?)
        }
        _ => return Ok(None),
    }))
}

fn unknown(backend: &str, function: &str) -> Error {
    Error::new(format!("{backend} has no function {function:?}"))
}

fn holds_any(a: &Args, write: fn(&str, &str, &[String]) -> Statement) -> Result<Output, Error> {
    a.expect(&["table", "identity", "values"])?;
    Ok(write(a.str("table")?, a.str("identity")?, &a.strs("values")?).into())
}

fn sqlite_call(function: &str, a: &Args) -> Result<Output, Error> {
    Ok(match function {
        "list_tables" => {
            a.expect(&[])?;
            sqlite::list_tables().into()
        }
        "table_info" | "index_list" | "foreign_key_list" => {
            a.expect(&["table"])?;
            let table = a.str("table")?;
            match function {
                "table_info" => sqlite::table_info(table),
                "index_list" => sqlite::index_list(table),
                _ => sqlite::foreign_key_list(table),
            }
            .into()
        }
        "index_info" => {
            a.expect(&["index"])?;
            sqlite::index_info(a.str("index")?).into()
        }
        "first_rows" => {
            a.expect(&["table", "projections", "num_rows"])?;
            sqlite::first_rows(a.str("table")?, &a.strs("projections")?, a.int("num_rows")?).into()
        }
        "holds_any" => holds_any(a, sqlite::holds_any)?,
        "count_distinct" => {
            a.expect(&["table", "columns"])?;
            text(sqlite::count_distinct(a.str("table")?, &a.strs("columns")?))
        }
        "key_counts" => {
            a.expect(&["parts"])?;
            sqlite::key_counts(&a.parts("parts")?).into()
        }
        "repeated_key" => {
            a.expect(&["table", "columns"])?;
            sqlite::repeated_key(a.str("table")?, &a.strs("columns")?).into()
        }
        "seeded_hash" => {
            a.expect(&["random_seed", "ref"])?;
            text(sqlite::seeded_hash(a.big("random_seed")?, a.str("ref")?)?)
        }
        "first_time" => {
            a.expect(&["table", "column"])?;
            sqlite::first_time(a.str("table")?, a.str("column")?).into()
        }
        "time_ranges" => {
            a.expect(&["tables"])?;
            sqlite::time_ranges(&a.pairs("tables")?).into()
        }
        "entity_rows" => {
            a.expect(&[
                "table",
                "projections",
                "filters",
                "order_by",
                "num_rows",
                "key",
                "ids",
            ])?;
            let ids = a.params("ids")?;
            sqlite::entity_rows(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.opt_str("order_by")?,
                a.opt_int("num_rows")?,
                a.opt_str("key")?,
                ids.as_deref(),
            )
            .into()
        }
        "_sqlite_type" => {
            a.expect(&["arrow_type"])?;
            optional_text(sqlite::sqlite_type(a.arrow_type("arrow_type")?))
        }
        "_json_values" => {
            a.expect(&["values"])?;
            Output::Value(Py::List(sqlite::json_values(a.arrow("values")?)?))
        }
        "_keys" => {
            a.expect(&["columns"])?;
            let keys = sqlite::keys(&a.key_columns("columns")?)?;
            Output::Value(keys_record("sqlite", keys.sql, keys.param))
        }
        "rows_by_key" => {
            a.expect(&[
                "table",
                "key",
                "ids",
                "projections",
                "unique",
                "tie_break",
                "count_matches",
            ])?;
            sqlite::rows_by_key(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.bool("unique")?,
                &a.strs("tie_break")?,
                a.bool("count_matches")?,
            )?
            .into()
        }
        "neighbors" => {
            a.expect(&[
                "table",
                "key",
                "ids",
                "projections",
                "num_neighbors",
                "order_by",
                "time",
                "end_times",
            ])?;
            sqlite::neighbors(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.int("num_neighbors")?,
                a.str("order_by")?,
                a.opt_str("time")?,
                a.opt_arrow("end_times")?,
            )?
            .into()
        }
        "window" => {
            a.expect(&[
                "table",
                "key",
                "ids",
                "projections",
                "time",
                "end_times",
                "start_times",
                "order_by",
            ])?;
            sqlite::window(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.str("time")?,
                a.arrow("end_times")?,
                a.opt_arrow("start_times")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "SQLiteSampler._probe_time_spelling" => {
            a.expect(&["table_name", "time_column", "found"])?;
            let found = a.opt_items("found")?;
            let spelling = helpers::sqlite_probe_time_spelling(a.opt_str("time_column")?, found);
            Output::Value(time_spelling_record(&spelling))
        }
        "_TimeSpelling.spell" => {
            a.expect(&["separator", "digits", "suffix", "times"])?;
            let spelling = TimeSpelling {
                separator: a.str("separator")?.to_string(),
                digits: a.int("digits")?,
                suffix: a.str("suffix")?.to_string(),
            };
            Output::Value(Py::Series(helpers::sqlite_spell(
                &spelling,
                a.series("times")?,
            )?))
        }
        _ => return Err(unknown("sqlite", function)),
    })
}

fn time_spelling_record(spelling: &TimeSpelling) -> Py {
    Py::Record(
        TIME_SPELLING_CLASS.into(),
        vec![
            ("separator".into(), Py::Str(spelling.separator.clone())),
            ("digits".into(), Py::int(spelling.digits)),
            ("suffix".into(), Py::Str(spelling.suffix.clone())),
        ],
    )
}

fn duckdb_call(function: &str, a: &Args) -> Result<Output, Error> {
    Ok(match function {
        "list_tables" | "indices" => {
            a.expect(&[])?;
            if function == "indices" {
                duckdb::indices()
            } else {
                duckdb::list_tables()
            }
            .into()
        }
        "table_info" | "key_constraints" | "foreign_keys" | "count_rows" => {
            a.expect(&["table"])?;
            let table = a.str("table")?;
            match function {
                "table_info" => duckdb::table_info(table),
                "key_constraints" => duckdb::key_constraints(table),
                "foreign_keys" => duckdb::foreign_keys(table),
                _ => duckdb::count_rows(table),
            }
            .into()
        }
        "first_rows" => {
            a.expect(&["table", "projections", "num_rows"])?;
            duckdb::first_rows(a.str("table")?, &a.strs("projections")?, a.int("num_rows")?).into()
        }
        "holds_any" => holds_any(a, duckdb::holds_any)?,
        "count_distinct" => {
            a.expect(&["columns"])?;
            text(duckdb::count_distinct(&a.strs("columns")?))
        }
        "key_counts" => {
            a.expect(&["parts"])?;
            duckdb::key_counts(&a.parts("parts")?).into()
        }
        "repeated_key" => {
            a.expect(&["table", "columns"])?;
            duckdb::repeated_key(a.str("table")?, &a.strs("columns")?).into()
        }
        "time_column" => {
            a.expect(&["column", "zoned"])?;
            text(duckdb::time_column(a.str("column")?, a.bool("zoned")?))
        }
        "time_ranges" => {
            a.expect(&["tables"])?;
            duckdb::time_ranges(&a.pairs("tables")?).into()
        }
        "random_rows" => {
            a.expect(&[
                "table",
                "projections",
                "filters",
                "num_rows",
                "random_seed",
                "order_by",
            ])?;
            duckdb::random_rows(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.int("num_rows")?,
                a.opt_big("random_seed")?,
                a.str("order_by")?,
            )
            .into()
        }
        "rows_by_ids" => {
            a.expect(&["table", "projections", "filters", "key", "ids", "order_by"])?;
            let ids = a.params("ids")?.ok_or_else(|| Error::new("no ids"))?;
            duckdb::rows_by_ids(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.str("key")?,
                &ids,
                a.str("order_by")?,
            )
            .into()
        }
        "_duckdb_type" => {
            a.expect(&["arrow_type"])?;
            text(duckdb::duckdb_type(a.arrow_type("arrow_type")?))
        }
        "_json_values" => {
            a.expect(&["values"])?;
            Output::Value(Py::List(duckdb::json_values(a.arrow("values")?)?))
        }
        "_keys" => {
            a.expect(&["columns"])?;
            let keys = duckdb::keys(&a.key_columns("columns")?)?;
            Output::Value(keys_record("duckdb", keys.sql, keys.param))
        }
        "rows_by_key" => {
            a.expect(&["table", "key", "ids", "projections", "order_by"])?;
            duckdb::rows_by_key(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "neighbors" => {
            a.expect(&[
                "table",
                "key",
                "ids",
                "projections",
                "num_neighbors",
                "order_by",
                "time",
                "end_times",
            ])?;
            duckdb::neighbors(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.int("num_neighbors")?,
                a.str("order_by")?,
                a.opt_str("time")?,
                a.opt_arrow("end_times")?,
            )?
            .into()
        }
        "window" => {
            a.expect(&[
                "table",
                "key",
                "ids",
                "projections",
                "time",
                "end_times",
                "start_times",
                "order_by",
            ])?;
            duckdb::window(
                a.str("table")?,
                a.str("key")?,
                a.arrow("ids")?,
                &a.strs("projections")?,
                a.str("time")?,
                a.arrow("end_times")?,
                a.opt_arrow("start_times")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "DuckDBTable._is_zoned_time" => {
            a.expect(&[
                "name",
                "source_column",
                "is_source",
                "source_type",
                "zoned_expr",
            ])?;
            let is_source = match a.get("is_source")? {
                Py::None => false,
                _ => a.bool("is_source")?,
            };
            Output::Value(Py::Bool(helpers::duckdb_is_zoned_time(
                a.bool("source_column")?,
                is_source,
                a.opt_str("source_type")?,
                a.bool("zoned_expr")?,
            )))
        }
        _ => return Err(unknown("duckdb", function)),
    })
}

fn postgres_call(function: &str, a: &Args) -> Result<Output, Error> {
    Ok(match function {
        "current_schema" => {
            a.expect(&[])?;
            postgres::current_schema().into()
        }
        "list_tables" => {
            a.expect(&["schema"])?;
            postgres::list_tables(a.str("schema")?).into()
        }
        "columns" | "key_constraints" | "foreign_keys" => {
            a.expect(&["schema", "table"])?;
            let (schema, table) = (a.str("schema")?, a.str("table")?);
            match function {
                "columns" => postgres::columns(schema, table),
                "key_constraints" => postgres::key_constraints(schema, table),
                _ => postgres::foreign_keys(schema, table),
            }
            .into()
        }
        "sample" => {
            a.expect(&["table", "projections", "order_terms", "num_rows"])?;
            postgres::sample(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("order_terms")?,
                a.int("num_rows")?,
            )
            .into()
        }
        "relation_size" | "count_rows" => {
            a.expect(&["table"])?;
            if function == "count_rows" {
                postgres::count_rows(a.str("table")?)
            } else {
                postgres::relation_size(a.str("table")?)
            }
            .into()
        }
        "block_sample" => {
            a.expect(&["table", "percent"])?;
            text(postgres::block_sample(a.str("table")?, a.str("percent")?))
        }
        "holds_any" => holds_any(a, postgres::holds_any)?,
        "count_distinct" => {
            a.expect(&["columns"])?;
            text(postgres::count_distinct(&a.strs("columns")?))
        }
        "key_counts" => {
            a.expect(&["parts"])?;
            postgres::key_counts(&a.parts("parts")?).into()
        }
        "repeated_key" => {
            a.expect(&["table", "columns"])?;
            postgres::repeated_key(a.str("table")?, &a.strs("columns")?).into()
        }
        "_json_default" => {
            a.expect(&["obj"])?;
            Output::Value(databricks::json_default(a.get("obj")?)?)
        }
        "dumps" => {
            a.expect(&["obj"])?;
            text(postgres::dumps(a.get("obj")?)?)
        }
        "chunk_rows" => {
            a.expect(&["rows", "max_bytes"])?;
            Output::Value(chunks(postgres::chunk_rows(
                a.items("rows")?,
                a.int("max_bytes")?,
            )?))
        }
        "element_type" => {
            a.expect(&["dtype"])?;
            text(postgres::element_type(a.dtype("dtype")?))
        }
        "coerce_id" => {
            a.expect(&["value", "dtype", "postgres_type"])?;
            Output::Value(postgres::coerce_id(
                a.get("value")?,
                a.dtype("dtype")?,
                a.str("postgres_type")?,
            )?)
        }
        "serialize_time" => {
            a.expect(&["value", "utc"])?;
            optional_text(postgres::serialize_time(a.get("value")?, a.bool("utc")?)?)
        }
        "postgres_seed" => {
            a.expect(&["random_seed"])?;
            Output::Value(Py::Float(postgres::postgres_seed(a.big("random_seed")?)))
        }
        "tie_break_term" => {
            a.expect(&["ref", "is_string"])?;
            text(postgres::tie_break_term(
                a.str("ref")?,
                a.bool("is_string")?,
            ))
        }
        "json_cte" => {
            a.expect(&["postgres_type", "fields", "time_type"])?;
            text(postgres::json_cte(
                a.str("postgres_type")?,
                &a.strs("fields")?,
                a.str("time_type")?,
            ))
        }
        "time_ranges" => {
            a.expect(&["tables"])?;
            postgres::time_ranges(&a.pairs("tables")?).into()
        }
        "set_seed" => {
            a.expect(&["random_seed"])?;
            postgres::set_seed(a.big("random_seed")?).into()
        }
        "random_rows" => {
            a.expect(&["table", "projections", "filters", "num_rows"])?;
            postgres::random_rows(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.int("num_rows")?,
            )
            .into()
        }
        "rows_by_ids" => {
            a.expect(&[
                "table",
                "projections",
                "filters",
                "key",
                "postgres_type",
                "ids",
                "order_by",
            ])?;
            postgres::rows_by_ids(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.str("key")?,
                a.str("postgres_type")?,
                a.get("ids")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "rows_by_key" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "postgres_type",
                "ids",
                "order_by",
            ])?;
            postgres::rows_by_key(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.str("postgres_type")?,
                a.get("ids")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "neighbors" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "postgres_type",
                "rows",
                "fields",
                "time",
                "time_type",
                "order_by",
                "num_neighbors",
            ])?;
            postgres::neighbors(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.str("postgres_type")?,
                a.get("rows")?,
                &a.strs("fields")?,
                a.opt_str("time")?,
                a.str("time_type")?,
                a.str("order_by")?,
                a.int("num_neighbors")?,
            )?
            .into()
        }
        "window" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "postgres_type",
                "rows",
                "fields",
                "time",
                "time_type",
                "order_by",
            ])?;
            postgres::window(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.str("postgres_type")?,
                a.get("rows")?,
                &a.strs("fields")?,
                a.str("time")?,
                a.str("time_type")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "PostgresSampler._utc_bounds" => {
            a.expect(&["table_name", "column", "dtype"])?;
            Output::Value(Py::Bool(helpers::postgres_utc_bounds(
                a.opt_dtype("dtype")?,
            )))
        }
        "PostgresTable._prefiltered_sample_source" => {
            a.expect(&[
                "schema",
                "source_name",
                "primary_key",
                "relation",
                "min_blocks",
                "target_blocks",
            ])?;
            optional_text(helpers::postgres_prefiltered_sample_source(
                a.str("schema")?,
                a.str("source_name")?,
                a.opt_str("primary_key")?,
                a.opt_items("relation")?,
                a.big("min_blocks")?,
                a.big("target_blocks")?,
            )?)
        }
        _ => return Err(unknown("postgres", function)),
    })
}

fn databricks_call(function: &str, a: &Args) -> Result<Output, Error> {
    Ok(match function {
        "current_namespace" => {
            a.expect(&[])?;
            databricks::current_namespace().into()
        }
        "list_tables" => {
            a.expect(&["catalog", "schema"])?;
            databricks::list_tables(a.str("catalog")?, a.str("schema")?).into()
        }
        "columns" | "key_constraints" | "foreign_keys" => {
            a.expect(&["catalog", "schema", "table"])?;
            let (catalog, schema, table) = (a.str("catalog")?, a.str("schema")?, a.str("table")?);
            match function {
                "columns" => databricks::columns(catalog, schema, table),
                "key_constraints" => databricks::key_constraints(catalog, schema, table),
                _ => databricks::foreign_keys(catalog, schema, table),
            }
            .into()
        }
        "describe_extended" => {
            a.expect(&["view"])?;
            databricks::describe_extended(a.str("view")?).into()
        }
        "describe_detail" | "count_rows" => {
            a.expect(&["table"])?;
            if function == "count_rows" {
                databricks::count_rows(a.str("table")?)
            } else {
                databricks::describe_detail(a.str("table")?)
            }
            .into()
        }
        "bounded" => {
            a.expect(&["table", "num_rows"])?;
            text(databricks::bounded(a.str("table")?, a.int("num_rows")?))
        }
        "sample" => {
            a.expect(&["table", "projections", "order_terms", "num_rows"])?;
            databricks::sample(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("order_terms")?,
                a.int("num_rows")?,
            )
            .into()
        }
        "holds_any" => holds_any(a, databricks::holds_any)?,
        "key_counts" => {
            a.expect(&["parts"])?;
            databricks::key_counts(&a.parts("parts")?).into()
        }
        "repeated_key" => {
            a.expect(&["table", "columns"])?;
            databricks::repeated_key(a.str("table")?, &a.strs("columns")?).into()
        }
        "_json_default" => {
            a.expect(&["obj"])?;
            Output::Value(databricks::json_default(a.get("obj")?)?)
        }
        "dumps" => {
            a.expect(&["obj"])?;
            text(databricks::dumps(a.get("obj")?)?)
        }
        "chunk_rows" => {
            a.expect(&["rows", "max_bytes"])?;
            Output::Value(chunks(databricks::chunk_rows(
                a.items("rows")?,
                a.int("max_bytes")?,
            )?))
        }
        "element_type" => {
            a.expect(&["dtype"])?;
            text(databricks::element_type(a.dtype("dtype")?))
        }
        "coerce_id" => {
            a.expect(&["value", "dtype"])?;
            Output::Value(databricks::coerce_id(a.get("value")?, a.dtype("dtype")?)?)
        }
        "time_ranges" => {
            a.expect(&["tables"])?;
            databricks::time_ranges(&a.pairs("tables")?).into()
        }
        "entity_rows" => {
            a.expect(&[
                "table",
                "projections",
                "filters",
                "key",
                "order_by",
                "percent",
                "random_seed",
                "num_rows",
                "ids",
                "dtype",
            ])?;
            databricks::entity_rows(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.str("key")?,
                a.str("order_by")?,
                a.opt_float("percent")?,
                a.opt_big("random_seed")?,
                a.opt_int("num_rows")?,
                a.opt_items("ids")?,
                a.opt_dtype("dtype")?,
            )?
            .into()
        }
        "rows_by_key" => {
            a.expect(&["table", "projections", "key", "dtype", "ids", "order_by"])?;
            databricks::rows_by_key(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("ids")?,
                &a.strs("order_by")?,
            )?
            .into()
        }
        "neighbors" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "dtype",
                "rows",
                "windowed",
                "time",
                "bounds",
                "order_by",
                "num_neighbors",
            ])?;
            databricks::neighbors(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("rows")?,
                a.bool("windowed")?,
                a.opt_str("time")?,
                &a.strs("bounds")?,
                a.str("order_by")?,
                a.int("num_neighbors")?,
            )?
            .into()
        }
        "window" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "dtype",
                "rows",
                "with_start",
                "time",
                "bounds",
                "order_by",
            ])?;
            databricks::window(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("rows")?,
                a.bool("with_start")?,
                a.str("time")?,
                &a.strs("bounds")?,
                a.str("order_by")?,
            )?
            .into()
        }
        "DatabricksSampler._bound_format" => {
            a.expect(&["table_name", "column", "dtype"])?;
            text(helpers::databricks_bound_format(a.opt_dtype("dtype")?))
        }
        "DatabricksTable._is_large" => {
            a.expect(&["detail_size", "described", "min_bytes"])?;
            let described: Option<Described> = a
                .dict("described")?
                .map(|fields| {
                    fields
                        .into_iter()
                        .map(|(k, v)| match v {
                            Py::Str(s) => Ok((k, s)),
                            other => Err(Error::new(format!("described {k} is {other:?}"))),
                        })
                        .collect::<Result<_, Error>>()
                })
                .transpose()?;
            Output::Value(Py::Bool(helpers::databricks_is_large(
                a.get("detail_size")?,
                described.as_ref(),
                a.big("min_bytes")?,
            )?))
        }
        _ => return Err(unknown("databricks", function)),
    })
}

fn snow_call(function: &str, a: &Args) -> Result<Output, Error> {
    let zone = |a: &Args| -> Result<(bool, bool, Option<String>), Error> {
        a.expect(&[
            "table_name",
            "column",
            "source_column",
            "zoned_expr",
            "source_type",
        ])?;
        Ok((
            a.bool("source_column")?,
            a.bool("zoned_expr")?,
            a.opt_str("source_type")?.map(str::to_string),
        ))
    };
    Ok(match function {
        "current_namespace" => {
            a.expect(&[])?;
            snow::current_namespace().into()
        }
        "list_tables" | "tables_in_schema" => {
            a.expect(&["database", "schema"])?;
            let (database, schema) = (a.str("database")?, a.str("schema")?);
            if function == "list_tables" {
                snow::list_tables(database, schema)
            } else {
                snow::tables_in_schema(database, schema)
            }
            .into()
        }
        "describe_table" | "imported_keys" => {
            a.expect(&["table"])?;
            if function == "describe_table" {
                snow::describe_table(a.str("table")?)
            } else {
                snow::imported_keys(a.str("table")?)
            }
            .into()
        }
        "tables_like" => {
            a.expect(&["database", "schema", "table"])?;
            snow::tables_like(a.str("database")?, a.str("schema")?, a.str("table")?).into()
        }
        "semantic_view" => {
            a.expect(&["name"])?;
            snow::semantic_view(a.str("name")?).into()
        }
        "sample" => {
            a.expect(&["table", "projections", "order_terms", "num_rows"])?;
            snow::sample(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("order_terms")?,
                a.int("num_rows")?,
            )
            .into()
        }
        "block_sample" => {
            a.expect(&["percent", "seed"])?;
            text(snow::block_sample(a.str("percent")?, a.big("seed")?))
        }
        "holds_any" => holds_any(a, snow::holds_any)?,
        "key_counts" => {
            a.expect(&["parts"])?;
            snow::key_counts(&a.parts("parts")?).into()
        }
        "repeated_key" => {
            a.expect(&["table", "columns"])?;
            snow::repeated_key(a.str("table")?, &a.strs("columns")?).into()
        }
        "time_ranges" => {
            a.expect(&["tables"])?;
            snow::time_ranges(&a.pairs("tables")?).into()
        }
        "seeded_key_hash" => {
            a.expect(&["key", "dtype", "is_date", "random_seed"])?;
            text(snow::seeded_key_hash(
                a.str("key")?,
                a.opt_dtype("dtype")?,
                a.bool("is_date")?,
                a.big("random_seed")?,
            ))
        }
        "entity_rows" => {
            a.expect(&[
                "table",
                "projections",
                "filters",
                "order_by",
                "num_rows",
                "seeded",
                "key",
                "ids",
                "prefilter",
            ])?;
            let ids = a.params("ids")?;
            snow::entity_rows(
                a.str("table")?,
                &a.strs("projections")?,
                &a.strs("filters")?,
                a.str("order_by")?,
                a.int("num_rows")?,
                a.opt_str("seeded")?,
                a.opt_str("key")?,
                ids.as_deref(),
                a.opt_str("prefilter")?,
            )
            .into()
        }
        "zone_of" => {
            a.expect(&["snowflake_type"])?;
            optional_text(snow::zone_of(a.opt_str("snowflake_type")?))
        }
        "_bound" => {
            a.expect(&["value", "zone"])?;
            text(snow::bound(a.str("value")?, a.opt_str("zone")?))
        }
        "_outer_bound" => {
            a.expect(&["zone"])?;
            text(snow::outer_bound(a.opt_str("zone")?))
        }
        "_key_list_time" => {
            a.expect(&["index", "zone"])?;
            text(snow::key_list_time(a.int("index")?, a.opt_str("zone")?))
        }
        "_cast" => {
            a.expect(&["dtype"])?;
            text(snow::cast(a.dtype("dtype")?))
        }
        "rows_by_key" => {
            a.expect(&["table", "projections", "key", "dtype", "ids", "order_by"])?;
            snow::rows_by_key(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("ids")?,
                &a.strs("order_by")?,
            )?
            .into()
        }
        "neighbors" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "dtype",
                "rows",
                "windowed",
                "time",
                "bounds",
                "order_by",
                "num_neighbors",
                "zone",
            ])?;
            snow::neighbors(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("rows")?,
                a.bool("windowed")?,
                a.opt_str("time")?,
                &a.strs("bounds")?,
                a.str("order_by")?,
                a.int("num_neighbors")?,
                a.opt_str("zone")?,
            )?
            .into()
        }
        "window" => {
            a.expect(&[
                "table",
                "projections",
                "key",
                "dtype",
                "rows",
                "with_start",
                "time",
                "bounds",
                "order_by",
                "zone",
            ])?;
            snow::window(
                a.str("table")?,
                &a.strs("projections")?,
                a.str("key")?,
                a.dtype("dtype")?,
                a.get("rows")?,
                a.bool("with_start")?,
                a.str("time")?,
                &a.strs("bounds")?,
                a.str("order_by")?,
                a.opt_str("zone")?,
            )?
            .into()
        }
        "SnowSampler._zone" => {
            let (source_column, zoned_expr, source_type) = zone(a)?;
            optional_text(helpers::snow_zone(
                source_column,
                zoned_expr,
                source_type.as_deref(),
            ))
        }
        "SnowSampler._bound_format" => {
            let (source_column, zoned_expr, source_type) = zone(a)?;
            text(helpers::snow_bound_format(
                source_column,
                zoned_expr,
                source_type.as_deref(),
            ))
        }
        "SnowTable._prefilter" => {
            a.expect(&[
                "rows",
                "seed",
                "listing",
                "sampled_kinds",
                "prefilter_partitions",
                "micro_partition_bytes",
            ])?;
            optional_text(helpers::snow_prefilter(
                a.big("rows")?,
                a.big("seed")?,
                a.dict("listing")?.as_ref(),
                &a.strs("sampled_kinds")?,
                a.big("prefilter_partitions")?,
                a.big("micro_partition_bytes")?,
            )?)
        }
        "SnowTable._listing" => {
            a.expect(&["source_name", "listed", "fields"])?;
            let listed: Option<Vec<Listing>> = a
                .opt_items("listed")?
                .map(|rows| {
                    rows.iter()
                        .map(|row| match row {
                            Py::Dict(fields) => Ok(fields.clone()),
                            other => Err(Error::new(format!("a listed row is {other:?}"))),
                        })
                        .collect::<Result<_, Error>>()
                })
                .transpose()?;
            Output::Value(
                helpers::snow_listing(
                    a.str("source_name")?,
                    listed.as_deref(),
                    &a.strs("fields")?,
                )?
                .map_or(Py::None, Py::Dict),
            )
        }
        _ => return Err(unknown("snow", function)),
    })
}
