// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The toy run, "count the rows of table T": the smallest run that asks for a
//! query and reads its result, so the loop can be driven end to end before
//! any real logic is ported.

use std::mem;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrowPrimitiveType};
use arrow_schema::DataType;

use crate::{Diagnostic, Outcome, Query, QueryId, QueryResult, Run, Step, codes};

/// The column the count is read from.
const COLUMN: &str = "n";

/// A run that counts the rows of one table.
#[derive(Debug)]
pub struct CountRows {
    table: String,
    state: State,
}

#[derive(Debug)]
enum State {
    Start,
    Waiting(QueryId),
    Answered(Outcome),
    Finished,
}

/// A run that counts the rows of `table`.
pub fn count_rows(table: impl Into<String>) -> CountRows {
    CountRows {
        table: table.into(),
        state: State::Start,
    }
}

impl CountRows {
    fn query(&self) -> Query {
        Query {
            id: QueryId(1),
            sql: format!(
                "SELECT COUNT(*) AS {COLUMN} FROM {}",
                quote_ident(&self.table)
            ),
            params: Vec::new(),
            columns: vec![COLUMN.to_owned()],
        }
    }
}

impl Run<u64> for CountRows {
    fn next(&mut self) -> Result<Step<u64>, Diagnostic> {
        match mem::replace(&mut self.state, State::Finished) {
            State::Start => {
                let query = self.query();
                self.state = State::Waiting(query.id);
                Ok(Step::RunQueries(vec![query]))
            }
            State::Waiting(id) => {
                self.state = State::Waiting(id);
                Err(Diagnostic::new(
                    codes::OUTCOME_MISSING,
                    format!("query {id} has no outcome yet; submit it before going on"),
                ))
            }
            State::Answered(Outcome::Data(result)) => read_count(&result).map(Step::Done),
            State::Answered(Outcome::Error {
                sqlstate,
                vendor_code,
                message,
            }) => Err(query_failed(&self.table, sqlstate, vendor_code, &message)),
            State::Answered(Outcome::Response(_) | Outcome::ModelError(_)) => Err(Diagnostic::new(
                codes::WRONG_OUTCOME,
                "a model's answer was submitted for a query",
            )),
            State::Finished => Err(finished()),
        }
    }

    fn submit(&mut self, id: QueryId, outcome: Outcome) -> Result<(), Diagnostic> {
        match &self.state {
            State::Waiting(waiting) if *waiting == id => {
                self.state = State::Answered(outcome);
                Ok(())
            }
            State::Finished => Err(finished()),
            _ => Err(Diagnostic::new(
                codes::UNKNOWN_QUERY,
                format!("the run is not waiting on a query {id}"),
            )),
        }
    }
}

fn finished() -> Diagnostic {
    Diagnostic::new(codes::RUN_FINISHED, "the run is already finished")
}

fn query_failed(
    table: &str,
    sqlstate: Option<String>,
    vendor_code: Option<i64>,
    message: &str,
) -> Diagnostic {
    let mut detail = Vec::new();
    if let Some(sqlstate) = sqlstate {
        detail.push(format!("SQLSTATE {sqlstate}"));
    }
    if let Some(code) = vendor_code {
        detail.push(format!("code {code}"));
    }
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({})", detail.join(", "))
    };
    Diagnostic::new(
        codes::QUERY_FAILED,
        format!("counting the rows of {table:?} failed{detail}: {message}"),
    )
}

fn unreadable(message: String) -> Diagnostic {
    Diagnostic::new(codes::UNREADABLE_RESULT, message)
}

/// The count in the result's one row, from its column [`COLUMN`].
fn read_count(result: &QueryResult) -> Result<u64, Diagnostic> {
    let schema = result.schema();
    let Ok(index) = schema.index_of(COLUMN) else {
        let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        return Err(unreadable(format!(
            "the result has no column {COLUMN:?}; it has {names:?}"
        )));
    };
    let rows = result.num_rows();
    if rows != 1 {
        return Err(unreadable(format!(
            "a count is one row, and the result has {rows}"
        )));
    }
    let Some(column) = result
        .batches()
        .iter()
        .find(|b| b.num_rows() == 1)
        .map(|b| b.column(index))
    else {
        return Err(unreadable("the result's one row is in no batch".to_owned()));
    };
    if column.is_null(0) {
        return Err(unreadable("the count is null".to_owned()));
    }
    let value = match column.data_type() {
        DataType::Int8 => signed::<Int8Type>(column),
        DataType::Int16 => signed::<Int16Type>(column),
        DataType::Int32 => signed::<Int32Type>(column),
        DataType::Int64 => signed::<Int64Type>(column),
        DataType::UInt8 => unsigned::<UInt8Type>(column),
        DataType::UInt16 => unsigned::<UInt16Type>(column),
        DataType::UInt32 => unsigned::<UInt32Type>(column),
        DataType::UInt64 => unsigned::<UInt64Type>(column),
        other => {
            return Err(unreadable(format!(
                "a count is an integer, and the result's is {other}"
            )));
        }
    };
    value.ok_or_else(|| unreadable("the count is negative".to_owned()))
}

fn signed<T>(column: &dyn Array) -> Option<u64>
where
    T: ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    u64::try_from(column.as_primitive::<T>().value(0).into()).ok()
}

fn unsigned<T>(column: &dyn Array) -> Option<u64>
where
    T: ArrowPrimitiveType,
    T::Native: Into<u64>,
{
    Some(column.as_primitive::<T>().value(0).into())
}

/// `name` as an ANSI SQL identifier, in double quotes, as SQLite and
/// PostgreSQL read it; the toy run is only driven against SQLite.
///
/// Not every warehouse reads it so: Databricks takes `"t"` for a string and
/// quotes identifiers in backticks. Quoting for each dialect belongs to the
/// `dialects` crate, which a run that reaches a warehouse will go through.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

    use super::*;

    fn result(name: &str, values: ArrayRef) -> Outcome {
        let batch = RecordBatch::try_from_iter([(name, values)]).expect("a batch");
        Outcome::Data(QueryResult::from_batch(batch))
    }

    fn asked(run: &mut CountRows) -> Query {
        match run.next() {
            Ok(Step::RunQueries(mut queries)) if queries.len() == 1 => queries.remove(0),
            other => panic!("expected one query, got {other:?}"),
        }
    }

    #[test]
    fn asks_for_one_count_with_the_table_quoted() {
        let mut run = count_rows("we\"ird");
        let query = asked(&mut run);
        assert_eq!(query.sql, "SELECT COUNT(*) AS n FROM \"we\"\"ird\"");
        assert!(query.params.is_empty());
        assert_eq!(query.columns, ["n"]);
    }

    #[test]
    fn reads_the_count_from_the_submitted_result() {
        let mut run = count_rows("t");
        let query = asked(&mut run);
        run.submit(query.id, result("n", Arc::new(Int64Array::from(vec![7]))))
            .expect("submitted");
        assert_eq!(run.next(), Ok(Step::Done(7)));
        assert_eq!(run.next().unwrap_err().code_str(), codes::RUN_FINISHED);
    }

    #[test]
    fn refuses_an_outcome_it_did_not_ask_for() {
        let mut run = count_rows("t");
        let outcome = result("n", Arc::new(Int64Array::from(vec![7])));
        let early = run.submit(QueryId(1), outcome.clone()).unwrap_err();
        assert_eq!(early.code_str(), codes::UNKNOWN_QUERY);
        let query = asked(&mut run);
        let wrong = run.submit(QueryId(query.id.0 + 1), outcome).unwrap_err();
        assert_eq!(wrong.code_str(), codes::UNKNOWN_QUERY);
        assert_eq!(run.next().unwrap_err().code_str(), codes::OUTCOME_MISSING);
    }

    #[test]
    fn a_failed_query_is_a_diagnostic_with_its_message() {
        let mut run = count_rows("gone");
        let query = asked(&mut run);
        let outcome = Outcome::Error {
            sqlstate: Some("42P01".to_owned()),
            vendor_code: Some(7),
            message: "no such table: gone".to_owned(),
        };
        run.submit(query.id, outcome).expect("submitted");
        let error = run.next().unwrap_err();
        assert_eq!(error.code_str(), codes::QUERY_FAILED);
        assert!(
            error
                .message
                .ends_with("(SQLSTATE 42P01, code 7): no such table: gone")
        );
    }

    #[test]
    fn an_unreadable_result_is_a_diagnostic() {
        let cases = [
            result("n", Arc::new(Int64Array::from(Vec::<i64>::new()))),
            result("n", Arc::new(Int64Array::from(vec![1, 2]))),
            result("other", Arc::new(Int64Array::from(vec![1]))),
            result("n", Arc::new(Int64Array::from(vec![None]))),
            result("n", Arc::new(Int64Array::from(vec![-1]))),
            result("n", Arc::new(StringArray::from(vec!["1"]))),
        ];
        for outcome in cases {
            let mut run = count_rows("t");
            let query = asked(&mut run);
            run.submit(query.id, outcome).expect("submitted");
            assert_eq!(run.next().unwrap_err().code_str(), codes::UNREADABLE_RESULT);
        }
    }
}
