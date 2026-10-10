// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What this host's DuckDB provider ran for one case, recorded as the
//! Python runners' driver capture records it (the observation's `driver`
//! section, from T1.11e), so the comparator holds this host's statements
//! and their results to the reference's as H9 says: each statement in the
//! order run, with the cursor it ran on (each fetch's own, numbered from 1
//! as the Python capture numbers derived cursors), the public call it ran
//! in, and its result in canonical form ([`crate::values`]); each key set
//! bound, with the statement that created its temporary table; and DuckDB's
//! own query log, turned on after the case's setup and read after the
//! case, whose counts and texts the capture check holds this record to.

use anyhow::{Result, anyhow};
use duckdb::Connection;
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

use crate::values::{arrow_type, values};

/// The record of one case's statements, bindings and cursors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriverLog {
    /// The public call the statements now run belong to: `from_source`
    /// for `describe_source`, `materialize`, `predict`.
    pub stage: &'static str,
    statements: Vec<Value>,
    bindings: Vec<Value>,
    binding_texts: Vec<String>,
    cursors: usize,
}

fn schema_of(schema: &SchemaRef) -> Value {
    Value::Array(
        schema
            .fields()
            .iter()
            .map(|f| json!({"name": f.name(), "type": arrow_type(f.data_type())}))
            .collect(),
    )
}

impl DriverLog {
    /// A new cursor's handle.
    pub fn cursor(&mut self) -> usize {
        self.cursors += 1;
        self.cursors
    }

    /// A statement run on `handle`, before it is run; its result or error
    /// is added by [`DriverLog::answered`] or [`DriverLog::failed`].
    pub fn statement(&mut self, handle: usize, sql: &str) {
        self.statements.push(json!({
            "handle": handle,
            "stage": self.stage,
            "sql": sql,
            "parameters": null,
            "results": [],
            "error": null,
        }));
    }

    /// The last statement's result, read whole as Arrow.
    pub fn answered(&mut self, schema: &SchemaRef, batches: &[RecordBatch]) {
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let fields: Vec<_> = schema.fields().iter().map(|f| f.as_ref()).collect();
        if let Some(Value::Array(results)) = self
            .statements
            .last_mut()
            .and_then(|statement| statement.get_mut("results"))
        {
            results.push(json!({
                "method": "query_arrow",
                "rows": rows,
                "schema": schema_of(schema),
                "values": values(&fields, batches),
            }));
        }
    }

    /// The last statement failed.
    pub fn failed(&mut self) {
        if let Some(statement) = self.statements.last_mut() {
            statement["error"] = json!("duckdb::Error");
        }
    }

    /// A key set bound on `handle` as the relation `name`, by `sql`.
    pub fn bound(
        &mut self,
        handle: usize,
        name: &str,
        sql: &str,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) {
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let fields: Vec<_> = schema.fields().iter().map(|f| f.as_ref()).collect();
        self.binding_texts.push(sql.to_string());
        self.bindings.push(json!({
            "handle": handle,
            "stage": self.stage,
            "name": name,
            "type": "arrow",
            "sql": sql,
            "rows": rows,
            "schema": schema_of(schema),
            "values": values(&fields, batches),
        }));
    }

    /// The observation's `driver` section, against the texts DuckDB
    /// logged (`logged`, from [`query_log`]): a logged statement that is
    /// one of the bindings' is counted as a binding; each other one is
    /// compared, in order, with the statement recorded, but a `PRAGMA`,
    /// which DuckDB logs as its table function, as the Python runner
    /// compares them.
    pub fn record(&self, logged: &[String]) -> Value {
        let (bound, ran): (Vec<&String>, Vec<&String>) = logged
            .iter()
            .partition(|text| self.binding_texts.contains(text));
        let mut compared = 0;
        let mut differing = Vec::new();
        if ran.len() == self.statements.len() {
            for (position, (statement, text)) in self.statements.iter().zip(&ran).enumerate() {
                let sql = statement["sql"].as_str().unwrap_or_default();
                if sql.trim_start().to_uppercase().starts_with("PRAGMA") {
                    continue;
                }
                compared += 1;
                if sql != text.as_str() {
                    differing.push(position);
                }
            }
        }
        json!({
            "capture": {
                "statements": self.statements,
                "bindings": self.bindings,
                "events": [],
                "incomplete": [],
                "passed_through": [],
                "cursors": self.cursors,
            },
            "query_log": {
                "statements": ran.len(),
                "bindings": bound.len(),
                "compared": compared,
                "differing": differing,
            },
        })
    }
}

/// Turns DuckDB's own query log on, for every connection to the database.
pub fn enable_query_log(connection: &Connection) -> Result<()> {
    connection
        .execute_batch("CALL enable_logging('QueryLog')")
        .map_err(|e| anyhow!("turning DuckDB's query log on: {e}"))
}

/// Each query DuckDB logged since [`enable_query_log`], in order.
pub fn query_log(connection: &Connection) -> Result<Vec<String>> {
    let mut statement = connection
        .prepare("SELECT message FROM duckdb_logs WHERE type = 'QueryLog' ORDER BY context_id")
        .map_err(|e| anyhow!("{e}"))?;
    let rows = statement
        .query_map([], |row| row.get::<_, Option<String>>(0))
        .map_err(|e| anyhow!("{e}"))?;
    let messages = rows
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("{e}"))?;
    Ok(messages
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect())
}
