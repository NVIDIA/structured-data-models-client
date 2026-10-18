// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4's two checks, shared by each database's goldens (T2.4; design,
//! locked check 3): `duckdb_goldens.rs`, `sqlite_goldens.rs` and
//! `postgres_goldens.rs` each pin, verbatim, every statement its dialect
//! renders for the contract cases, beside the request it answers.
//!
//! - [`render`]: each golden request, planned here, renders exactly its
//!   statements, so a change of the SQL without a golden update fails.
//! - [`run_is_pinned`]: every fetch a case run made on the database (the
//!   observations `SDM_L4_OBSERVATIONS` names, python-core's run in
//!   `contract-python (ref-pandas3, <database>)`, which executed each
//!   statement on that database and whose results the comparator held to
//!   the reference's) is a golden, its statements the golden's, so the
//!   goldens are the SQL the run executed and a request the dialect newly
//!   answers is never left unreviewed. CI's `l4-goldens (<database>)` names
//!   the run; with no run named, as in `cargo test --workspace`, there is
//!   nothing to hold and the check says so. A run named but missing,
//!   unreadable, empty, or holding no fetch on the database fails.
//! - [`seeded_faults`]: both checks shown to fail on their seeded faults.
//!
//! A failure prints each missing golden ready to paste, as the run made it.

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice};
use sdm_data::Data;
use serde_json::Value;

/// Each request, as the contract writes it, and its statements in order.
pub type Goldens = &'static [(&'static str, &'static [&'static str])];

type Owned = Vec<(String, Vec<String>)>;

fn owned(goldens: Goldens) -> Owned {
    goldens
        .iter()
        .map(|(request, statements)| {
            (
                (*request).to_owned(),
                statements.iter().map(|s| (*s).to_owned()).collect(),
            )
        })
        .collect()
}

/// A golden as the file writes it.
fn literal(request: &str, statements: &[String]) -> String {
    let texts: Vec<String> = statements.iter().map(|s| format!("{s:?}")).collect();
    format!("(r##\"{request}\"##, &[{}]),", texts.join(", "))
}

/// The goldens whose request does not render exactly its statements.
fn misrendered(dialect: &str, goldens: &Owned) -> Vec<String> {
    let choice = ProviderChoice::Sql {
        dialect: Dialect::new(dialect),
    };
    let mut problems = Vec::new();
    for (request, statements) in goldens {
        let parsed: DataRequest = match serde_json::from_str(request) {
            Ok(parsed) => parsed,
            Err(error) => {
                problems.push(format!("{request}: not a request: {error}"));
                continue;
            }
        };
        match Data::new().plan(&parsed, &choice) {
            Ok(FetchPlan::Sql {
                statements: planned,
                ..
            }) => {
                let texts: Vec<String> = planned.iter().map(|s| s.text.clone()).collect();
                if &texts != statements {
                    problems.push(format!("renders\n  {}", literal(request, &texts)));
                }
            }
            Ok(_) => problems.push(format!("{request}: not planned as SQL")),
            Err(error) => problems.push(format!("{request}: refused: {error:?}")),
        }
    }
    problems
}

/// Every golden request renders exactly its statements on `dialect`.
pub fn render(dialect: &str, goldens: Goldens) {
    assert!(!goldens.is_empty(), "no goldens on {dialect}");
    let problems = misrendered(dialect, &owned(goldens));
    assert!(
        problems.is_empty(),
        "{dialect}: {} golden(s) not what the dialect renders:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// Whether an observation's case key is on `database`: DuckDB's keep the
/// bare case id, the others are `<case>@<database>`.
fn on(database: &str, case: &str) -> bool {
    if database == "duckdb" {
        !case.contains('@')
    } else {
        case.strip_suffix(database)
            .is_some_and(|rest| rest.ends_with('@'))
    }
}

/// The fetches of `observations` (one JSON object per line) on `database`
/// that no golden holds with their statements, each ready to paste; an
/// error where the run is unreadable or holds no fetch on the database.
fn unpinned(database: &str, observations: &str, goldens: &Owned) -> Result<Vec<String>, String> {
    let mut pinned: Vec<(Value, &Vec<String>)> = Vec::new();
    for (request, statements) in goldens {
        let value = serde_json::from_str(request)
            .map_err(|error| format!("a golden is not JSON: {error}: {request}"))?;
        pinned.push((value, statements));
    }
    let (mut cases, mut fetches) = (0, 0);
    let mut problems: Vec<String> = Vec::new();
    for (number, line) in observations.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let observation: Value = serde_json::from_str(line)
            .map_err(|error| format!("line {}: not JSON: {error}", number + 1))?;
        let Some(case) = observation["case"].as_str() else {
            return Err(format!("line {}: names no case", number + 1));
        };
        if !on(database, case) {
            continue;
        }
        cases += 1;
        let Some(made) = observation["core"]["fetches"].as_array() else {
            continue;
        };
        for fetch in made {
            fetches += 1;
            let request = &fetch["request"];
            let Some(planned) = fetch["plan"]["statements"].as_array() else {
                problems.push(format!("{case}: a fetch planned no SQL: {request}"));
                continue;
            };
            let mut texts = Vec::new();
            for statement in planned {
                if statement["dialect"].as_str() != Some(database) {
                    problems.push(format!(
                        "{case}: a statement of dialect {}, not {database}",
                        statement["dialect"]
                    ));
                }
                texts.push(statement["text"].as_str().unwrap_or_default().to_owned());
            }
            if !pinned.iter().any(|(r, s)| r == request && **s == texts) {
                let entry = literal(&request.to_string(), &texts);
                if !problems.contains(&entry) {
                    problems.push(entry);
                }
            }
        }
    }
    if cases == 0 {
        return Err(format!("the run holds no observation on {database}"));
    }
    if fetches == 0 {
        return Err(format!(
            "the run's {cases} observation(s) on {database} hold no fetch"
        ));
    }
    Ok(problems)
}

/// The run `SDM_L4_OBSERVATIONS` names, held to the goldens.
#[allow(
    clippy::disallowed_methods,
    reason = "a test reading the run CI names, not the core: L4's evidence"
)]
pub fn run_is_pinned(database: &str, goldens: Goldens) {
    let Some(path) = std::env::var_os("SDM_L4_OBSERVATIONS").filter(|p| !p.is_empty()) else {
        eprintln!("{database}: SDM_L4_OBSERVATIONS names no run: nothing to hold the goldens to");
        return;
    };
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: unreadable: {error}", path.to_string_lossy()));
    match unpinned(database, &text, &owned(goldens)) {
        Ok(problems) => assert!(
            problems.is_empty(),
            "{database}: {} fetch(es) the run made are no golden; each as the run made it:\n{}",
            problems.len(),
            problems.join("\n")
        ),
        Err(error) => panic!("{database}: {error}"),
    }
}

/// A run of one case on `database` making each golden's fetch.
fn run_of(database: &str, goldens: &Owned) -> String {
    let case = if database == "duckdb" {
        "a.case".to_owned()
    } else {
        format!("a.case@{database}")
    };
    let fetches: Vec<Value> = goldens
        .iter()
        .map(|(request, statements)| {
            serde_json::json!({
                "request": serde_json::from_str::<Value>(request).unwrap(),
                "plan": {
                    "plan": "sql",
                    "statements": statements
                        .iter()
                        .map(|text| serde_json::json!({"dialect": database, "params": [], "text": text}))
                        .collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    serde_json::json!({"case": case, "core": {"fetches": fetches}}).to_string()
}

/// Both checks fail on their seeded faults, on `goldens`.
pub fn seeded_faults(database: &str, goldens: Goldens) {
    let goldens = owned(goldens);
    // Controls: the goldens render, and a run of every golden is pinned.
    assert!(misrendered(database, &goldens).is_empty());
    let run = run_of(database, &goldens);
    assert_eq!(unpinned(database, &run, &goldens), Ok(Vec::new()));

    // A seeded SQL change without a golden update: the golden's text no
    // longer what the dialect renders, and a run of the changed SQL no
    // golden.
    let mut changed = goldens.clone();
    changed[0].1[0].push(' ');
    assert_eq!(misrendered(database, &changed).len(), 1);
    let ran = run_of(database, &changed);
    assert_eq!(unpinned(database, &ran, &goldens).map(|p| p.len()), Ok(1));

    // A request no golden holds.
    let fewer = goldens[1..].to_vec();
    assert_eq!(unpinned(database, &run, &fewer).map(|p| p.len()), Ok(1));

    // A run that holds nothing to judge fails, never passes: empty, of
    // another database, with no fetch, or not JSON.
    assert!(unpinned(database, "", &goldens).is_err());
    let other = if database == "duckdb" {
        "sqlite"
    } else {
        "duckdb"
    };
    assert!(unpinned(database, &run_of(other, &goldens), &goldens).is_err());
    assert!(unpinned(database, &run_of(database, &Vec::new()), &goldens).is_err());
    assert!(unpinned(database, "{", &goldens).is_err());
}
