// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm-contract-run`: the pure-Rust host over contract cases.
//!
//! ```text
//! sdm-contract-run [--pre-given-specs] --out FILE CASE_OR_DIRECTORY...
//! sdm-contract-run --providers
//! sdm-contract-run --replay RECORDING --out FILE CASE_OR_DIRECTORY...
//! sdm-contract-run --replay RECORDING --replay-leg LEG --out FILE CASE_OR_DIRECTORY...
//! sdm-contract-run --database sqlite --out FILE CASE_OR_DIRECTORY...
//! sdm-contract-run --database memory [--layer default] --out FILE CASE_OR_DIRECTORY...
//! sdm-contract-run --kit DATABASE [--fault truncate|silent_capability] --out REPORT
//! ```
//!
//! `--kit` runs the provider conformance kit on this host's provider for
//! DATABASE, `duckdb`, `sqlite` or `memory` ([`sdm_testkit::kit`], T2.5;
//! `memory` the minimal non-SQL test provider, T2.7): it writes
//! REPORT and exits 0 when every case passes, 1 when any fails, and 2
//! when it cannot run.
//!
//! `--database sqlite` (T2.2) runs each case that runs on SQLite over its
//! tables loaded into a new SQLite file by the contract loader, read
//! through the dev-only `rusqlite` provider ([`sdm_testkit::sqlite`]),
//! every call the case makes through its `until` (T2.1d), each observation
//! keyed `<case>@sqlite`.
//!
//! `--database memory` (T2.7) runs each case that reads data over its
//! DuckDB base copied into the minimal non-SQL test provider
//! ([`sdm_testkit::memory`]), through the second data layer swapped in at
//! the composition root ([`sdm_testkit::layer`]), each observation keyed
//! `<case>@memory`; a case that reads no data is not run there. With
//! `--layer default` the same provider runs through
//! `sdm_host::default_root()`'s layer (`sdm-data`'s) instead, keyed the
//! same, so the two runs differ in their data layer alone.
//!
//! `--replay` runs each case that runs on Postgres there, its fetches
//! answered from a recording of a Postgres server's answers
//! (`run_python.py --record-fetches`; [`sdm_testkit::replay`], KO2-20),
//! each observation keyed `<case>@postgres`.
//! With `--replay-leg databricks-sql|databricks-spark|snowflake|lakebase`,
//! it replays that leg's actual fetch recording and its recorded starting
//! arguments, including the source's namespace and semantic view. It
//! renders and checks every request normally; no warehouse is emulated.
//!
//! `--providers` prints the databases this host has a provider for
//! ([`sdm_testkit::PROVIDERS`]) as a JSON list, the staging rule's marker
//! for rust-host (T2.1a).
//!
//! Runs every case through `HostRun`, building a case's data from
//! `contract/bases.yaml` when its operation reads data, and writes one
//! observation per case,
//! one JSON object per line, for `scripts/rust_core/compare.py`. A runner
//! failure (a case it cannot load or run, a step S0 never takes) exits
//! non-zero and leaves no output file, an earlier run's included;
//! observations are written only when every case was recorded. The
//! identity's commit is the checkout the command runs in.
//!
//! `--pre-given-specs` reads every case's `core.describe_source.source` as
//! contract 22's, written before T1.32h, and runs it as the current shape
//! ([`sdm_testkit::pre_given`]); a bundle frozen before then cannot be
//! rewritten. Without it, a spec in that shape is a runner failure.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Result, bail};
use sdm_testkit::{bases, case, host, observation, pre_given, replay};

/// Whether `case`'s operations read data: it names `describe_source`.
fn reads_data(case: &case::Case) -> bool {
    case.core
        .as_ref()
        .is_some_and(|core| core.describe_source.is_some())
}

fn run(arguments: &[String]) -> Result<(PathBuf, usize)> {
    let (pre_given_specs, arguments) = match arguments {
        [flag, rest @ ..] if flag == "--pre-given-specs" => (true, rest),
        _ => (false, arguments),
    };
    let (database, arguments) = match arguments {
        [flag, database, rest @ ..]
            if flag == "--database" && (database == "sqlite" || database == "memory") =>
        {
            (Some(database.as_str()), rest)
        }
        [flag, database, ..] if flag == "--database" => {
            bail!(
                "--database {database}: this host runs DuckDB, SQLite, the non-SQL memory provider or a Postgres replay"
            )
        }
        _ => (None, arguments),
    };
    let sqlite = database == Some("sqlite");
    // The data layer a `--database memory` run composes: the second one
    // (T2.7), or, with `--layer default`, the default root's.
    let (root, arguments) = match arguments {
        [flag, layer, rest @ ..] if flag == "--layer" && layer == "default" => {
            if database != Some("memory") {
                bail!("--layer default: only with --database memory");
            }
            (sdm_host::default_root(), rest)
        }
        [flag, layer, ..] if flag == "--layer" => {
            bail!("--layer {layer}: the one layer to name is `default`")
        }
        _ => (sdm_testkit::layer::root(), arguments),
    };
    let (recording, arguments) = match arguments {
        [flag, path, rest @ ..] if flag == "--replay" => (
            Some(replay::Recording::load(std::path::Path::new(path))?),
            rest,
        ),
        _ => (None, arguments),
    };
    if recording.is_some() && database.is_some() {
        bail!("--replay cannot select another --database");
    }
    let (replay_leg, arguments) = match arguments {
        [flag, leg, rest @ ..] if flag == "--replay-leg" => {
            if recording.is_none() || database.is_some() {
                bail!("--replay-leg needs --replay and cannot select another --database");
            }
            let warehouse = match leg.as_str() {
                "postgres" => None,
                "databricks-sql" => Some(case::Warehouse::DatabricksSql),
                "databricks-spark" => Some(case::Warehouse::DatabricksSpark),
                "snowflake" => Some(case::Warehouse::Snowflake),
                "lakebase" => Some(case::Warehouse::Lakebase),
                _ => bail!("an unsupported replay leg"),
            };
            ((leg.as_str(), warehouse), rest)
        }
        _ => (("postgres", None), arguments),
    };
    let (out, cases) = match arguments {
        [flag, out, cases @ ..] if flag == "--out" && !cases.is_empty() => (
            PathBuf::from(out),
            cases.iter().map(PathBuf::from).collect::<Vec<_>>(),
        ),
        _ => bail!(
            "usage: sdm-contract-run [--pre-given-specs] [--database sqlite | --database memory [--layer default] | --replay RECORDING [--replay-leg LEG]] --out FILE CASE_OR_DIRECTORY..."
        ),
    };
    let mut partial = out.clone().into_os_string();
    partial.push(".partial");
    // A failed run leaves no output, not an earlier run's.
    for stale in [out.as_os_str(), partial.as_os_str()] {
        match std::fs::remove_file(stale) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(error.into());
            }
            _ => {}
        }
    }
    // The cases that run on the database this run answers from: a
    // warehouse-only case (S2's `warehouses`) runs on none of them, and is
    // selected out, never run on DuckDB (`case::load_for`, T3.2).
    let on = if sqlite {
        case::Database::Sqlite
    } else if recording.is_some() {
        case::Database::Postgres
    } else {
        case::Database::Duckdb
    };
    let mut loaded = case::load(&cases)?;
    if loaded.is_empty() {
        bail!("no cases");
    }
    loaded.retain(|each| match replay_leg.1 {
        Some(warehouse) => each.case.warehouses().contains(&warehouse),
        None => each.case.databases().contains(&on),
    });
    if loaded.is_empty() {
        bail!("no case runs on the database asked for");
    }
    if pre_given_specs {
        for each in &mut loaded {
            pre_given::upgrade(&mut each.case)?;
        }
    }
    let here = std::env::current_dir()?;
    let identity = observation::Identity::new(&here)?;
    let bases_path = observation::repository(&here)?.join("contract/bases.yaml");
    let bases = bases::load(&bases_path)?;
    let bases_digest = case::sha256(&std::fs::read(&bases_path)?);
    let repository = observation::repository(&here)?;
    let mut lines = String::new();
    let mut count = 0;
    for each in &loaded {
        let (observed, keyed) = match &recording {
            None if sqlite => {
                if !each.case.databases().contains(&case::Database::Sqlite) {
                    continue;
                }
                let mut keyed = each.clone();
                keyed.case.id = format!("{}@sqlite", each.case.id);
                let observed = host::run_sqlite(&each.case, &bases, &repository)?;
                (observed, keyed)
            }
            None if database == Some("memory") => {
                if !reads_data(&each.case) {
                    continue;
                }
                let mut keyed = each.clone();
                keyed.case.id = format!("{}@memory", each.case.id);
                (host::run_memory(&each.case, &bases, &root)?, keyed)
            }
            None => (host::run(&each.case, &bases)?, each.clone()),
            Some(recording) => {
                let mut keyed = each.clone();
                keyed.case.id = format!("{}@{}", each.case.id, replay_leg.0);
                // A diagnostic case whose core asks past what python-core
                // recorded is observed refused (`replay::PAST_THE_RECORDING`);
                // an acceptance case's fails the run (T2.8 fixes 3 and 4).
                (
                    host::run_replayed_on(&each.case, recording, replay_leg.0)?,
                    keyed,
                )
            }
        };
        let each = &keyed;
        count += 1;
        // On SQLite every case reads data, its calls made through its
        // until (T2.1d).
        let read_bases = sqlite || replay_leg.1.is_some() || reads_data(&each.case);
        let digest = read_bases.then_some(bases_digest.as_str());
        lines.push_str(&observation::write(&identity, each, &observed, digest).to_string());
        lines.push('\n');
    }
    if count == 0 {
        bail!("no case runs on the database asked for");
    }
    std::fs::write(&partial, lines)?;
    std::fs::rename(&partial, &out)?;
    Ok((out, count))
}

/// `--kit DATABASE [--fault F] --out REPORT`: whether every case passed.
fn kit(arguments: &[String]) -> Result<bool> {
    let (database, fault, out) = match arguments {
        [kit, database, out_flag, out] if kit == "--kit" && out_flag == "--out" => {
            (database, None, out)
        }
        [kit, database, fault_flag, fault, out_flag, out]
            if kit == "--kit" && fault_flag == "--fault" && out_flag == "--out" =>
        {
            (database, Some(fault.as_str()), out)
        }
        _ => bail!(
            "usage: sdm-contract-run --kit DATABASE [--fault truncate|silent_capability] --out REPORT"
        ),
    };
    let repository = observation::repository(&std::env::current_dir()?)?;
    let (report, passed) = sdm_testkit::kit::run(&repository, database, fault)?;
    std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(passed)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments == ["--providers"] {
        println!("{}", serde_json::json!(sdm_testkit::PROVIDERS));
        return ExitCode::SUCCESS;
    }
    if arguments.first().is_some_and(|a| a == "--kit") {
        return match kit(&arguments) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::from(1),
            Err(failure) => {
                eprintln!("provider kit: {failure:#}");
                ExitCode::from(2)
            }
        };
    }
    match run(&arguments) {
        Ok((_, count)) => {
            println!("rust-host: {count} cases recorded");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("runner failure: {failure:#}");
            ExitCode::FAILURE
        }
    }
}
