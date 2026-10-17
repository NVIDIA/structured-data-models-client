// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust host's SQLite (T2.2): a case's tables loaded by the one
//! contract loader, and the dev-only `rusqlite` provider, which answers a
//! fetch step as the Python host's SQLite provider does through ADBC.
//!
//! [`load`] builds a case's data by the contract loader
//! (`contract/runners/databases.py`, through `contract/tools/load_case.py`,
//! run by the Python `SDM_PYTHON` names, `python3` by default), so this
//! host reads the same loaded database the reference and python-core read
//! (design choice 2).
//!
//! [`Sqlite`] runs the statements a plan carries, in order, and hands the
//! last one's rows back as Arrow typed as ADBC's SQLite driver types them,
//! so the two core hosts' answers are equal (`cross-host`): each column's
//! type inferred from the values of its first 1,024 rows, an integer as
//! `int64`, a float as `double` (an integer beside one made a float), text
//! as `string` (a number beside it made SQLite's text of it), bytes as
//! `binary` (anything beside them made its bytes), a column holding no
//! value as `int64`; a later row of another type than its column's made
//! that type where ADBC makes it, and failing the read where ADBC fails it
//! (a float or text in an integer column, text in a float column). Every
//! field is nullable, as ADBC's are. A key set is bound as ADBC's
//! `adbc_ingest(..., mode='replace', temporary=True)` binds it: a
//! temporary table of the name its parameter gives, dropped first if
//! there, each column declared and each value written as ADBC writes
//! Arrow (an integer or a Boolean `INTEGER`, a float `REAL`, text and a
//! date `TEXT`, bytes `BLOB`, a timestamp with no declared type, as ISO
//! 8601 text with a `T` and its unit's fractional digits); a decimal, or
//! an unsigned integer past `int64`, is refused as ADBC refuses it.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use duckdb::arrow::array::{
    Array, ArrayRef, BinaryBuilder, BooleanArray, Date32Array, Float32Array, Float64Array,
    Float64Builder, Int8Array, Int16Array, Int32Array, Int64Array, Int64Builder, LargeBinaryArray,
    LargeStringArray, StringArray, StringBuilder, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use duckdb::arrow::record_batch::RecordBatch;
use rusqlite::types::{Value as Stored, ValueRef};
use sdm_contract::ErrorKind;
use sdm_contract::data::{AttachmentId, FetchPlan, ProviderError, SqlParam};
use sdm_contract::run::FetchStep;

use crate::driver::DriverLog;
use crate::duck::Fetched;

/// The rows ADBC's SQLite driver reads a batch of, and infers a column's
/// type from the first of (`adbc.sqlite.query.batch_rows`'s default).
const BATCH_ROWS: usize = 1024;

/// Loads `case_file`'s data into a new SQLite file at `out` through the
/// contract loader, run from `repository`; a case the loader refuses by
/// name is an error naming it.
pub fn load(repository: &Path, case_file: &Path, out: &Path) -> Result<()> {
    let python = std::env::var_os("SDM_PYTHON").unwrap_or_else(|| "python3".into());
    let done = Command::new(&python)
        .arg("-I")
        .arg(repository.join("contract/tools/load_case.py"))
        .arg("sqlite")
        .arg(case_file)
        .arg(out)
        .current_dir(repository)
        .stdin(Stdio::null())
        .output()
        .with_context(|| {
            format!(
                "running {} for the contract loader",
                python.to_string_lossy()
            )
        })?;
    if !done.status.success() {
        bail!(
            "the contract loader did not load {}: {}",
            case_file.display(),
            String::from_utf8_lossy(&done.stderr).trim()
        );
    }
    Ok(())
}

/// A SQLite source read through `rusqlite`, as the Python host reads one
/// through ADBC, and the texts it ran, for the run's record
/// ([`DriverLog::record`]): every statement, and each key set's binding.
pub struct Sqlite {
    connection: rusqlite::Connection,
    ran: RefCell<Vec<String>>,
    /// The key-set tables this provider made on the connection: only these
    /// are ever replaced, as the Python host's provider replaces only its own.
    made: RefCell<BTreeSet<String>>,
}

impl Sqlite {
    /// The database file at `path`, opened as a user opens one.
    pub fn open(path: &Path) -> Result<Sqlite> {
        Ok(Sqlite {
            connection: rusqlite::Connection::open(path)
                .with_context(|| format!("opening {}", path.display()))?,
            ran: RefCell::new(Vec::new()),
            made: RefCell::new(BTreeSet::new()),
        })
    }

    /// An in-memory database, for tests.
    pub fn in_memory() -> Result<Sqlite> {
        Ok(Sqlite {
            connection: rusqlite::Connection::open_in_memory()?,
            ran: RefCell::new(Vec::new()),
            made: RefCell::new(BTreeSet::new()),
        })
    }

    /// The connection, to set a source up.
    pub fn connection(&self) -> &rusqlite::Connection {
        &self.connection
    }

    /// Every text run, statements and bindings, in order: this provider's
    /// own log, as the Python runner's statement log is ADBC's.
    pub fn logged(&self) -> Vec<String> {
        self.ran.borrow().clone()
    }

    /// The answer to one fetch step, its key sets bound from `keys` by
    /// attachment id, recording in `log` the cursor, each binding and each
    /// statement run with its result; the provider's error as the
    /// contract carries it, never the driver's text.
    pub fn answer_logged(
        &self,
        step: &FetchStep,
        keys: &BTreeMap<AttachmentId, Fetched>,
        log: &mut DriverLog,
    ) -> Result<Fetched, ProviderError> {
        let failed = || ProviderError {
            kind: ErrorKind::DataSource,
            retryable: false,
            sqlstate: None,
            vendor_code: None,
        };
        let FetchPlan::Sql { statements, .. } = &step.plan else {
            return Err(failed());
        };
        let Some((last, before)) = statements.split_last() else {
            return Err(failed());
        };
        let handle = log.cursor();
        let mut bound = BTreeMap::new();
        for statement in statements {
            for param in &statement.params {
                let SqlParam::KeySet {
                    keys: set,
                    relation,
                } = param
                else {
                    return Err(failed());
                };
                match bound.insert(relation.clone(), set.attachment) {
                    Some(earlier) if earlier != set.attachment => return Err(failed()),
                    Some(_) => continue,
                    None => {}
                }
                let fetched = keys.get(&set.attachment).ok_or_else(failed)?;
                let mine = self.made.borrow().contains(relation);
                let sql =
                    ingest(&self.connection, relation, fetched, mine).map_err(|_| failed())?;
                self.made.borrow_mut().insert(relation.clone());
                self.ran.borrow_mut().push(sql.clone());
                log.bound(handle, relation, &sql, &fetched.schema, &fetched.batches);
            }
        }
        for statement in before {
            log.statement(handle, &statement.text);
            self.ran.borrow_mut().push(statement.text.clone());
            if self.connection.execute_batch(&statement.text).is_err() {
                log.failed();
                return Err(failed());
            }
        }
        log.statement(handle, &last.text);
        self.ran.borrow_mut().push(last.text.clone());
        match read(&self.connection, &last.text) {
            Ok(fetched) => {
                log.answered(&fetched.schema, &fetched.batches);
                Ok(fetched)
            }
            Err(_) => {
                log.failed();
                Err(failed())
            }
        }
    }
}

// --- binding a key set, as ADBC ingests Arrow ---------------------------------------

fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The type ADBC's ingest declares a column of `data_type` with (none for
/// a timestamp), or the refusal of one it does not ingest.
fn declared(data_type: &DataType) -> Result<&'static str> {
    Ok(match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => "INTEGER",
        DataType::Float32 | DataType::Float64 => "REAL",
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Date32 => "TEXT",
        DataType::Binary | DataType::LargeBinary => "BLOB",
        DataType::Timestamp(_, _) => "",
        other => bail!("ADBC's SQLite ingest takes no column of {other}"),
    })
}

/// `keys` bound as the temporary table `name`, created, or replaced where
/// it is `mine` (one this provider made): the statement that created it,
/// as the record names a binding by. A table of that name it did not make
/// fails the creation and is left as it was.
fn ingest(
    connection: &rusqlite::Connection,
    name: &str,
    keys: &Fetched,
    mine: bool,
) -> Result<String> {
    let table = format!("temp.{}", quoted(name));
    let columns = keys
        .schema
        .fields()
        .iter()
        .map(|f| {
            let kind = declared(f.data_type())?;
            Ok(if kind.is_empty() {
                quoted(f.name())
            } else {
                format!("{} {kind}", quoted(f.name()))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let create = format!(
        "CREATE TEMPORARY TABLE {} ({})",
        quoted(name),
        columns.join(", ")
    );
    if mine {
        connection.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))?;
    }
    connection.execute_batch(&create)?;
    let batch = concat_batches(&keys.schema, &keys.batches)?;
    let marks = vec!["?"; batch.num_columns()].join(", ");
    let mut insert = connection.prepare(&format!("INSERT INTO {table} VALUES ({marks})"))?;
    let columns: Vec<Vec<Stored>> = batch
        .columns()
        .iter()
        .map(|c| stored(c.as_ref()))
        .collect::<Result<_>>()?;
    for row in 0..batch.num_rows() {
        let values: Vec<&Stored> = columns.iter().map(|c| &c[row]).collect();
        insert.execute(rusqlite::params_from_iter(values))?;
    }
    Ok(create)
}

/// An instant as ADBC writes one: `YYYY-MM-DDTHH:MM:SS`, then a point and
/// the fraction to the unit's digits for a unit finer than a second.
fn instant_text(count: i64, unit: TimeUnit) -> Result<String> {
    let (per, digits) = match unit {
        TimeUnit::Second => (1i64, 0usize),
        TimeUnit::Millisecond => (1_000, 3),
        TimeUnit::Microsecond => (1_000_000, 6),
        TimeUnit::Nanosecond => (1_000_000_000, 9),
    };
    let seconds = count.div_euclid(per);
    let fraction = count.rem_euclid(per);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let date = date_text(days)?;
    let mut text = format!(
        "{date}T{:02}:{:02}:{:02}",
        of_day / 3600,
        of_day / 60 % 60,
        of_day % 60
    );
    if digits > 0 {
        text.push_str(&format!(".{fraction:0digits$}"));
    }
    Ok(text)
}

/// The civil date `days` after 1970-01-01, as `YYYY-MM-DD` (Howard
/// Hinnant's `civil_from_days`).
fn date_text(days: i64) -> Result<String> {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    if !(0..=9999).contains(&year) {
        bail!("a date ADBC does not write");
    }
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

/// Each value of `column` as ADBC's ingest writes it.
fn stored(column: &dyn Array) -> Result<Vec<Stored>> {
    macro_rules! each {
        ($array:ty, $value:expr) => {{
            let array = column
                .as_any()
                .downcast_ref::<$array>()
                .context("a key set's column is not of its type")?;
            (0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        Ok(Stored::Null)
                    } else {
                        #[allow(clippy::redundant_closure_call)]
                        ($value)(array.value(i))
                    }
                })
                .collect::<Result<Vec<_>>>()?
        }};
    }
    Ok(match column.data_type() {
        DataType::Boolean => each!(BooleanArray, |v: bool| Ok(Stored::Integer(i64::from(v)))),
        DataType::Int8 => each!(Int8Array, |v: i8| Ok(Stored::Integer(i64::from(v)))),
        DataType::Int16 => each!(Int16Array, |v: i16| Ok(Stored::Integer(i64::from(v)))),
        DataType::Int32 => each!(Int32Array, |v: i32| Ok(Stored::Integer(i64::from(v)))),
        DataType::Int64 => each!(Int64Array, |v: i64| Ok(Stored::Integer(v))),
        DataType::UInt8 => each!(UInt8Array, |v: u8| Ok(Stored::Integer(i64::from(v)))),
        DataType::UInt16 => each!(UInt16Array, |v: u16| Ok(Stored::Integer(i64::from(v)))),
        DataType::UInt32 => each!(UInt32Array, |v: u32| Ok(Stored::Integer(i64::from(v)))),
        DataType::UInt64 => each!(UInt64Array, |v: u64| Ok(Stored::Integer(
            i64::try_from(v).context("an unsigned integer past int64, which ADBC refuses")?
        ))),
        DataType::Float32 => each!(Float32Array, |v: f32| Ok(Stored::Real(f64::from(v)))),
        DataType::Float64 => each!(Float64Array, |v: f64| Ok(Stored::Real(v))),
        DataType::Utf8 => each!(StringArray, |v: &str| Ok(Stored::Text(v.to_owned()))),
        DataType::LargeUtf8 => each!(LargeStringArray, |v: &str| Ok(Stored::Text(v.to_owned()))),
        DataType::Binary => each!(duckdb::arrow::array::BinaryArray, |v: &[u8]| Ok(
            Stored::Blob(v.to_vec())
        )),
        DataType::LargeBinary => each!(LargeBinaryArray, |v: &[u8]| Ok(Stored::Blob(v.to_vec()))),
        DataType::Date32 => each!(Date32Array, |v: i32| Ok(Stored::Text(date_text(
            i64::from(v)
        )?))),
        DataType::Timestamp(TimeUnit::Second, _) => each!(TimestampSecondArray, |v: i64| Ok(
            Stored::Text(instant_text(v, TimeUnit::Second)?)
        )),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            each!(TimestampMillisecondArray, |v: i64| Ok(Stored::Text(
                instant_text(v, TimeUnit::Millisecond)?
            )))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            each!(TimestampMicrosecondArray, |v: i64| Ok(Stored::Text(
                instant_text(v, TimeUnit::Microsecond)?
            )))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            each!(TimestampNanosecondArray, |v: i64| Ok(Stored::Text(
                instant_text(v, TimeUnit::Nanosecond)?
            )))
        }
        other => bail!("ADBC's SQLite ingest takes no column of {other}"),
    })
}

// --- reading a result, as ADBC types one ------------------------------------------

/// A column's type as ADBC's SQLite driver infers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inferred {
    /// No value read yet; `int64` if none comes.
    None,
    Int64,
    Double,
    String,
    Binary,
}

impl Inferred {
    /// The type after a value of `value`'s storage class, while inferring.
    fn with(self, value: &ValueRef<'_>) -> Inferred {
        match (self, value) {
            (_, ValueRef::Null) => self,
            (Inferred::Binary, _) | (_, ValueRef::Blob(_)) => Inferred::Binary,
            (Inferred::String, _) | (_, ValueRef::Text(_)) => Inferred::String,
            (Inferred::Double, _) | (_, ValueRef::Real(_)) => Inferred::Double,
            (_, ValueRef::Integer(_)) => Inferred::Int64,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Inferred::None | Inferred::Int64 => DataType::Int64,
            Inferred::Double => DataType::Float64,
            Inferred::String => DataType::Utf8,
            Inferred::Binary => DataType::Binary,
        }
    }
}

/// SQLite's text of a number, as `sqlite3_column_text` gives it: an
/// integer's digits; a float as SQLite writes one, which SQLite itself
/// renders.
fn number_text(connection: &rusqlite::Connection, value: &Stored) -> Result<String> {
    Ok(match value {
        Stored::Integer(v) => v.to_string(),
        other => connection.query_row("SELECT CAST(?1 AS TEXT)", [other], |row| row.get(0))?,
    })
}

/// Every row of `sql`'s result, typed as ADBC types it.
fn read(connection: &rusqlite::Connection, sql: &str) -> Result<Fetched> {
    let mut statement = connection.prepare(sql)?;
    let names: Vec<String> = statement
        .column_names()
        .iter()
        .map(|n| (*n).to_owned())
        .collect();
    let width = names.len();
    let mut rows: Vec<Vec<Stored>> = Vec::new();
    let mut queried = statement.query([])?;
    while let Some(row) = queried.next()? {
        rows.push(
            (0..width)
                .map(|i| row.get::<_, Stored>(i))
                .collect::<rusqlite::Result<_>>()?,
        );
    }
    let mut types = vec![Inferred::None; width];
    for row in rows.iter().take(BATCH_ROWS) {
        for (kind, value) in types.iter_mut().zip(row) {
            *kind = kind.with(&ValueRef::from(value));
        }
    }
    let mut fields = Vec::with_capacity(width);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(width);
    for (at, (name, kind)) in names.iter().zip(&types).enumerate() {
        fields.push(Field::new(name, kind.data_type(), true));
        let values = rows.iter().map(|r| &r[at]);
        let mismatch = || anyhow::anyhow!("type mismatch in column {at}, as ADBC refuses it");
        columns.push(match kind {
            Inferred::None | Inferred::Int64 => {
                let mut built = Int64Builder::new();
                for value in values {
                    match value {
                        Stored::Null => built.append_null(),
                        Stored::Integer(v) => built.append_value(*v),
                        _ => return Err(mismatch()),
                    }
                }
                Arc::new(built.finish())
            }
            Inferred::Double => {
                let mut built = Float64Builder::new();
                for value in values {
                    match value {
                        Stored::Null => built.append_null(),
                        Stored::Integer(v) => built.append_value(*v as f64),
                        Stored::Real(v) => built.append_value(*v),
                        _ => return Err(mismatch()),
                    }
                }
                Arc::new(built.finish())
            }
            Inferred::String => {
                let mut built = StringBuilder::new();
                for value in values {
                    match value {
                        Stored::Null => built.append_null(),
                        Stored::Text(t) => built.append_value(t),
                        Stored::Integer(_) | Stored::Real(_) => {
                            built.append_value(number_text(connection, value)?);
                        }
                        Stored::Blob(_) => return Err(mismatch()),
                    }
                }
                Arc::new(built.finish())
            }
            Inferred::Binary => {
                let mut built = BinaryBuilder::new();
                for value in values {
                    match value {
                        Stored::Null => built.append_null(),
                        Stored::Blob(b) => built.append_value(b),
                        Stored::Text(t) => built.append_value(t.as_bytes()),
                        Stored::Integer(_) | Stored::Real(_) => {
                            built.append_value(number_text(connection, value)?.as_bytes());
                        }
                    }
                }
                Arc::new(built.finish())
            }
        });
    }
    let schema = Arc::new(Schema::new(fields));
    let batches = if rows.is_empty() {
        vec![]
    } else {
        vec![RecordBatch::try_new(Arc::clone(&schema), columns)?]
    };
    Ok(Fetched { schema, batches })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key set's table never replaces a relation the provider did not
    /// make: a caller's own temporary table of the key set's name fails the
    /// fetch and holds its row after; the provider's own table is made, and
    /// replaced on the next fetch.
    #[test]
    fn a_key_set_never_replaces_a_relation_the_provider_did_not_make() {
        use sdm_contract::data::{DataRequest, Dialect, ProviderChoice};
        use sdm_contract::run::ReplyId;

        let source = Sqlite::in_memory().unwrap();
        let connection = source.connection();
        connection
            .execute_batch(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, age REAL);
                 INSERT INTO users VALUES (101, 10), (102, 20);",
            )
            .unwrap();
        let request: DataRequest = serde_json::from_str(
            r#"{"request": "lookup", "table": {"source": ["users"]}, "key": ["id"],
                "columns": ["age"], "seeds": {"attachment": 0, "columns": [
                  {"name": "key", "nullable": true, "type": {"type": "int64"}},
                  {"name": "batch", "nullable": false, "type": {"type": "int64"}}]}}"#,
        )
        .unwrap();
        let sqlite = ProviderChoice::Sql {
            dialect: Dialect::new("sqlite"),
        };
        let plan = sdm_host::default_root()
            .data_access(&sqlite)
            .unwrap()
            .plan(&request)
            .unwrap();
        let FetchPlan::Sql { statements, .. } = &plan else {
            panic!("not SQL");
        };
        let [SqlParam::KeySet { relation, .. }] = &statements[0].params[..] else {
            panic!("no key set");
        };
        let relation = relation.clone();
        let step = FetchStep {
            id: ReplyId(1),
            request,
            plan,
            continuation: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, true),
            Field::new("batch", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![101])),
                Arc::new(Int64Array::from(vec![0])),
            ],
        )
        .unwrap();
        let keys = BTreeMap::from([(
            AttachmentId(0),
            Fetched {
                schema,
                batches: vec![batch],
            },
        )]);
        let theirs = quoted(&relation);
        connection
            .execute_batch(&format!(
                "CREATE TEMP TABLE {theirs} (important TEXT);
                 INSERT INTO temp.{theirs} VALUES ('keep me');"
            ))
            .unwrap();
        let answer =
            |source: &Sqlite| source.answer_logged(&step, &keys, &mut DriverLog::default());
        assert!(answer(&source).is_err());
        let kept: String = connection
            .query_row(&format!("SELECT * FROM temp.{theirs}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(kept, "keep me");
        connection
            .execute_batch(&format!("DROP TABLE temp.{theirs}"))
            .unwrap();
        for _ in 0..2 {
            let fetched = answer(&source).unwrap();
            assert_eq!(
                fetched.batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                1
            );
        }
    }

    #[test]
    fn instants_and_dates_are_written_as_adbc_writes_them() {
        assert_eq!(
            instant_text(1_700_000_000_123_456_789, TimeUnit::Nanosecond).unwrap(),
            "2023-11-14T22:13:20.123456789"
        );
        assert_eq!(
            instant_text(-1, TimeUnit::Nanosecond).unwrap(),
            "1969-12-31T23:59:59.999999999"
        );
        assert_eq!(
            instant_text(1_700_000_000, TimeUnit::Second).unwrap(),
            "2023-11-14T22:13:20"
        );
        assert_eq!(date_text(19_000).unwrap(), "2022-01-08");
    }

    #[test]
    fn a_result_is_typed_as_adbc_types_it() {
        let source = Sqlite::in_memory().unwrap();
        let read = |sql: &str| read(source.connection(), sql);
        let types = |sql: &str| -> Vec<DataType> {
            read(sql)
                .unwrap()
                .schema
                .fields()
                .iter()
                .map(|f| f.data_type().clone())
                .collect()
        };
        assert_eq!(
            types("SELECT 1 AS a, NULL AS b, 1.5 AS c, 'x' AS d, x'00' AS e"),
            [
                DataType::Int64,
                DataType::Int64,
                DataType::Float64,
                DataType::Utf8,
                DataType::Binary
            ]
        );
        assert_eq!(
            types("WITH t(v) AS (VALUES (1),(2.5)) SELECT v FROM t"),
            [DataType::Float64]
        );
        assert_eq!(
            types("WITH t(v) AS (VALUES (1),('a')) SELECT v FROM t"),
            [DataType::Utf8]
        );
        assert_eq!(
            types("WITH t(v) AS (VALUES (x'00'),(1)) SELECT v FROM t"),
            [DataType::Binary]
        );
        assert_eq!(types("SELECT 1 AS a WHERE 0"), [DataType::Int64]);
        // A later row of another type: as ADBC, an integer column fails.
        assert!(
            read(
                "WITH RECURSIVE t(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM t WHERE i<2000) \
                  SELECT CASE WHEN i<1500 THEN i ELSE 'z' END AS v FROM t"
            )
            .is_err()
        );
        let text = read(
            "WITH RECURSIVE t(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM t WHERE i<2000) \
             SELECT CASE WHEN i<1500 THEN 'a' ELSE 7 END AS v FROM t",
        )
        .unwrap();
        let last = text.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(1999)
            .to_owned();
        assert_eq!(last, "7");
    }
}
