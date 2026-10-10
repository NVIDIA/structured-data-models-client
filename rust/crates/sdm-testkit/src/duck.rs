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
//! this exact version, installed from DuckDB's repository the first time
//! (a signature-checked network bootstrap, not a hermetic one; CI caches
//! the extension directory), its bytes checked against the digest recorded
//! for the platform before it is loaded; autoloading and autoinstalling
//! stay off, so nothing else loads, and the loaded ICU's version is
//! checked.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use duckdb::arrow::compute::concat_batches;
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::vtab::arrow::{ArrowVTab, arrow_recordbatch_to_query_params};
use duckdb::{Config, Connection};
use sdm_contract::ErrorKind;
use sdm_contract::data::{AttachmentId, FetchPlan, ProviderError, SqlParam};
use sdm_contract::run::FetchStep;

/// The DuckDB version the reference runs, which this host must link.
pub const DUCKDB_VERSION: &str = "v1.5.6";

/// The SHA-256 of DuckDB's signed ICU extension for v1.5.6, by platform,
/// as `extensions.duckdb.org` serves it (decompressed), which is the file
/// `INSTALL icu` writes.
const ICU_SHA256: [(&str, &str); 4] = [
    (
        "linux_amd64",
        "f7bdb7b30da48406cc58429b8b5d901a1aea5f8b9c0bd1928642f0ab262d5d85",
    ),
    (
        "linux_arm64",
        "414bd5f36d5efd15cf92d008a7e9369ada3031d584e8a992bff73f9d3b444749",
    ),
    (
        "osx_amd64",
        "9cc1f9c2c261de0af335d530749d23e5f472bb85c76f0552ab34adfb5cf1dfd3",
    ),
    (
        "osx_arm64",
        "3578a29bb3e05024eaf75c849d502089d25ed2b629823ed97508a86e62acbba7",
    ),
];

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
    // Installed once per machine from DuckDB's repository, signature
    // checked by DuckDB; its bytes are checked here against the digest
    // recorded for the platform before it is loaded (decision E37).
    let installed = |connection: &Connection| -> Result<Option<String>> {
        Ok(connection.query_row(
            "SELECT install_path FROM duckdb_extensions() WHERE extension_name = 'icu' AND installed",
            [],
            |row| row.get(0),
        ).ok())
    };
    let path = match installed(&connection)? {
        Some(path) => path,
        None => {
            // Tests in parallel may install it at once; whichever install
            // lands, the file must then be there.
            let _ = connection.execute_batch("INSTALL icu");
            installed(&connection)?.context("installing DuckDB's ICU extension")?
        }
    };
    let platform: String =
        connection.query_row("SELECT platform FROM pragma_platform()", [], |row| {
            row.get(0)
        })?;
    let Some((_, digest)) = ICU_SHA256.iter().find(|(name, _)| *name == platform) else {
        bail!("no recorded ICU digest for the platform {platform}");
    };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    if crate::case::sha256(&bytes) != *digest {
        bail!("the ICU extension at {path} is not the one recorded for {platform}");
    }
    connection
        .execute_batch("LOAD icu")
        .context("loading DuckDB's ICU extension")?;
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

/// The answer to one fetch step whose plan binds no key set, or the
/// provider's error as the contract carries it: its kind, never the
/// driver's text.
pub fn answer(connection: &Connection, step: &FetchStep) -> Result<Fetched, ProviderError> {
    answer_with(connection, step, &BTreeMap::new())
}

/// The answer to one fetch step, its key sets bound from `keys`, the
/// attachments the step named, by id: each as the relation its plan names
/// it by, on the step's own cursor, as the Python provider registers one;
/// a key set's rows travel as Arrow, never as text (decision Q1). A plan
/// with a scalar parameter, which no request carries yet, is not run.
pub fn answer_with(
    connection: &Connection,
    step: &FetchStep,
    keys: &BTreeMap<AttachmentId, Fetched>,
) -> Result<Fetched, ProviderError> {
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
    // A cursor of its own, as the Python provider and the reference read:
    // a connection to the same database, which takes the database's
    // settings and none of the session's.
    let cursor = connection.try_clone().map_err(|_| failed())?;
    // Each name binds one key set; two under one name are refused, as the
    // Python provider refuses them, never one kept over the other.
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
            bind(&cursor, relation, fetched).map_err(|_| failed())?;
        }
    }
    for statement in before {
        cursor
            .execute_batch(&statement.text)
            .map_err(|_| failed())?;
    }
    run(&cursor, &last.text).map_err(|_| failed())
}

/// Binds `keys` as the relation `name` on `cursor`: a temporary table of
/// the key set's rows, read from Arrow by `duckdb-rs`'s Arrow table
/// function, so every value keeps its type (an unsigned 64-bit integer, a
/// decimal of any scale) and nothing passes through text or a float.
fn bind(cursor: &Connection, name: &str, keys: &Fetched) -> Result<()> {
    cursor
        .register_table_function::<ArrowVTab>("arrow")
        .map_err(|e| anyhow!("{e}"))?;
    let batch = concat_batches(&keys.schema, &keys.batches).map_err(|e| anyhow!("{e}"))?;
    let params = arrow_recordbatch_to_query_params(batch);
    let quoted = format!("\"{}\"", name.replace('"', "\"\""));
    cursor
        .execute(
            &format!("CREATE TEMPORARY TABLE {quoted} AS SELECT * FROM arrow(?, ?)"),
            params,
        )
        .map_err(|e| anyhow!("{e}"))?;
    Ok(())
}

/// Every batch of `sql`'s result, with its schema, empty or not.
pub fn run(connection: &Connection, sql: &str) -> Result<Fetched> {
    let mut statement = connection.prepare(sql).map_err(|e| anyhow!("{e}"))?;
    let arrow = statement.query_arrow([]).map_err(|e| anyhow!("{e}"))?;
    let schema = arrow.get_schema();
    let batches: Vec<RecordBatch> = arrow.collect();
    Ok(Fetched { schema, batches })
}
