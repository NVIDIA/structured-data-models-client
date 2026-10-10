// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust host's DuckDB: a connection as the Python host's driver
//! has it, and the DuckDB provider, which answers a fetch step by running
//! the statements its plan carries, in order, on a cursor of its own (a
//! connection to the same database, as the Python provider's cursor is),
//! and handing every batch of the last one's result back as one attachment.
//!
//! The Python wheel of DuckDB links ICU, so a session there takes `SET
//! TimeZone` and casts a `TIMESTAMPTZ` in that zone. The `icu` feature of
//! `duckdb` 1.10506.0 needs DuckDB's sources beside the crate, which the
//! published crate does not carry, so it cannot be built (decision E37).
//! [`open`] loads ICU explicitly instead, the extension DuckDB signs for
//! this exact version, installed from DuckDB's repository the first time;
//! autoloading and autoinstalling stay off, so nothing else loads, and the
//! loaded ICU's version is checked.

use anyhow::{Context, Result, anyhow, bail};
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::{Config, Connection};
use sdm_contract::ErrorKind;
use sdm_contract::data::{FetchPlan, ProviderError};
use sdm_contract::run::FetchStep;

/// The DuckDB version the reference runs, which this host must link.
pub const DUCKDB_VERSION: &str = "v1.5.6";

/// A fresh in-memory database, with ICU loaded and extension autoloading
/// off.
pub fn open() -> Result<Connection> {
    let config = Config::default()
        .enable_autoload_extension(false)
        .context("turning extension autoloading off")?;
    let connection =
        Connection::open_in_memory_with_flags(config).context("opening an in-memory DuckDB")?;
    let version: String = connection.query_row("SELECT version()", [], |row| row.get(0))?;
    if version != DUCKDB_VERSION {
        bail!("DuckDB {version} is linked, and the reference runs {DUCKDB_VERSION}");
    }
    if connection.execute_batch("LOAD icu").is_err() {
        // Tests in parallel may install it at once; whichever install
        // lands, the load is what must succeed.
        let _ = connection.execute_batch("INSTALL icu");
        connection
            .execute_batch("LOAD icu")
            .context("installing and loading DuckDB's ICU extension")?;
    }
    let (loaded, icu): (bool, String) = connection.query_row(
        "SELECT loaded, extension_version FROM duckdb_extensions() WHERE extension_name = 'icu'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if !loaded || icu != DUCKDB_VERSION {
        bail!("ICU {icu} is not the reference DuckDB's ({DUCKDB_VERSION}), or is not loaded");
    }
    Ok(connection)
}

/// A query's result: its schema and every batch the driver handed over.
pub struct Fetched {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// The answer to one fetch step, or the provider's error as the contract
/// carries it: its kind, never the driver's text.
pub fn answer(connection: &Connection, step: &FetchStep) -> Result<Fetched, ProviderError> {
    let failed = || ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: None,
        vendor_code: None,
    };
    let FetchPlan::Sql { statements, .. } = &step.plan else {
        // A host plan is a host provider's to answer; this one runs SQL.
        return Err(failed());
    };
    let Some((last, before)) = statements.split_last() else {
        return Err(failed());
    };
    // Parameters, and key sets bound as relations, arrive with the requests
    // that carry them (T1.11c); a statement with any is not run.
    if statements.iter().any(|s| !s.params.is_empty()) {
        return Err(failed());
    }
    // A cursor of its own, as the Python provider and the reference read:
    // a connection to the same database, which takes the database's
    // settings and none of the session's.
    let cursor = connection.try_clone().map_err(|_| failed())?;
    for statement in before {
        cursor
            .execute_batch(&statement.text)
            .map_err(|_| failed())?;
    }
    run(&cursor, &last.text).map_err(|_| failed())
}

/// Every batch of `sql`'s result, with its schema, empty or not.
pub fn run(connection: &Connection, sql: &str) -> Result<Fetched> {
    let mut statement = connection.prepare(sql).map_err(|e| anyhow!("{e}"))?;
    let arrow = statement.query_arrow([]).map_err(|e| anyhow!("{e}"))?;
    let schema = arrow.get_schema();
    let batches: Vec<RecordBatch> = arrow.collect();
    Ok(Fetched { schema, batches })
}
