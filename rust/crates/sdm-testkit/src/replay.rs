// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pure-Rust host's Postgres provider: it replays recorded fetch
//! results instead of reading Postgres through a Rust driver (KO2-20).
//!
//! A recording is what a Postgres server answered each fetch the core asked
//! for one case, in order: the fetch's request and plan as the core wrote
//! them, and the answer, its rows as an Arrow IPC stream, or the provider's
//! error as the contract carries it. The Python runner writes one
//! (`run_python.py --record-fetches`) as python-core runs the case on a
//! loaded Postgres database, through the Python host's Postgres provider, so
//! the rows are the server's answers to the core's own statements. Replayed,
//! each fetch must be the one recorded at its position, its request and
//! plan equal, as JSON, to the recorded ones; any other fetch, or one past
//! the recording's end, fails the run by name (a runner failure, never a
//! provider error the core could take for the source's), so a change to
//! what the core asks or how the dialect renders it shows as a recording
//! to make again, never as stale rows answering another question. A fetch
//! past the recording's end is its own error, [`PastTheRecording`]; for a
//! diagnostic case it is instead answered as the provider's error,
//! [`PAST_THE_RECORDING`] its vendor code ([`Replay::past_the_end`], T2.8
//! fix 4), so the case is observed, refused, and its difference reported:
//! python-core records only the fetches it made, and on a case the Python
//! host fails before the core reads (`s0.boundary.depth-999`, whose query
//! Python's parser cannot read), the core asks more.
//!
//! Each recorded fetch names the text of every statement the Postgres
//! driver ran for it, in order, as python-core's driver capture recorded
//! them; the replay hands them back with the answer, and the test host
//! records them as run (its query log and driver record), so rust-host's
//! record of what ran is python-core's, each result its own read of the
//! answer replayed.
//!
//! The key sets a plan binds are not compared: they are the core's own,
//! made from the answers replayed before them, and the plan names each by
//! its schema, which is compared.
//!
//! The file is JSON Lines, one line per case, `{"format": "sdm-replay/1",
//! "case": <key>, "fetches": [{"request", "plan", "statements", "data" |
//! "error"}]}`,
//! `data` the IPC stream in standard base64.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use duckdb::arrow::record_batch::RecordBatch;
use sdm_contract::ErrorKind;
use sdm_contract::data::ProviderError;
use sdm_contract::run::FetchStep;
use serde::Deserialize;
use serde_json::Value;

use crate::duck::Fetched;

/// The recording's format, named on every line.
pub const FORMAT: &str = "sdm-replay/1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    format: String,
    case: String,
    fetches: Vec<Fetch>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fetch {
    request: Value,
    plan: Value,
    statements: Vec<String>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    error: Option<ProviderError>,
}

/// One recorded fetch: what the core asked, the statements the driver ran
/// for it, and its answer.
struct Recorded {
    request: Value,
    plan: Value,
    statements: Vec<String>,
    answer: Answer,
}

/// What the replay answers a fetch with: the statements the driver ran for
/// it, in order, and its rows or the provider's error.
pub struct Replayed<'a> {
    pub statements: &'a [String],
    pub answer: std::result::Result<Fetched, ProviderError>,
}

/// One recorded answer.
enum Answer {
    Data(Vec<u8>),
    Error(ProviderError),
}

/// The recorded fetches of every case in a recording, by the case's key
/// (`<case>@postgres`).
pub struct Recording {
    cases: BTreeMap<String, Vec<Recorded>>,
}

impl Recording {
    /// The recording at `path`: every line valid, no case twice.
    pub fn load(path: &Path) -> Result<Recording> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("the recording {} cannot be read", path.display()))?;
        let mut cases = BTreeMap::new();
        for (number, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
            let line: Line = serde_json::from_str(line)
                .map_err(|_| anyhow!("the recording's line {} is not a case's", number + 1))?;
            if line.format != FORMAT {
                bail!("the recording's line {} is not {FORMAT}", number + 1);
            }
            let mut fetches = Vec::with_capacity(line.fetches.len());
            for fetch in line.fetches {
                let answer = match (fetch.data, fetch.error) {
                    (Some(data), None) => Answer::Data(
                        base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .map_err(|_| anyhow!("{}: a fetch's data is not base64", line.case))?,
                    ),
                    (None, Some(error)) => Answer::Error(error),
                    _ => bail!("{}: a fetch has not one of data and error", line.case),
                };
                fetches.push(Recorded {
                    request: fetch.request,
                    plan: fetch.plan,
                    statements: fetch.statements,
                    answer,
                });
            }
            if cases.insert(line.case.clone(), fetches).is_some() {
                bail!("the recording holds {} twice", line.case);
            }
        }
        Ok(Recording { cases })
    }

    /// The replay of the case keyed `case`, from its first fetch.
    pub fn replay(&self, case: &str) -> Result<Replay<'_>> {
        let fetches = self
            .cases
            .get(case)
            .ok_or_else(|| anyhow!("{case}: the recording holds no fetches for it"))?;
        Ok(Replay {
            case: case.to_owned(),
            fetches,
            next: Cell::new(0),
            past_the_end: false,
        })
    }
}

/// A fetch the core asked past the end of the case's recording: its
/// position, and how many were recorded. The runner fails the run on it,
/// but for a diagnostic case, which it leaves unobserved.
#[derive(Debug)]
pub struct PastTheRecording {
    pub case: String,
    pub position: usize,
    pub recorded: usize,
}

impl std::fmt::Display for PastTheRecording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: the core asked fetch {}, past the {} recorded",
            self.case, self.position, self.recorded
        )
    }
}

impl std::error::Error for PastTheRecording {}

/// The vendor code of the provider error a diagnostic case's fetch past
/// its recording is answered with.
pub const PAST_THE_RECORDING: &str = "past the recording";

/// One case's recorded fetches, answered in order.
pub struct Replay<'a> {
    case: String,
    fetches: &'a [Recorded],
    next: Cell<usize>,
    past_the_end: bool,
}

impl Replay<'_> {
    /// The replay, a fetch past the recording answered as the provider's
    /// error ([`PAST_THE_RECORDING`]) rather than failing the run: for a
    /// diagnostic case.
    #[must_use]
    pub fn past_the_end(self) -> Self {
        Replay {
            past_the_end: true,
            ..self
        }
    }

    /// The recorded answer to `step`, which must be the next fetch
    /// recorded: the statements run for it, and its rows or the provider's
    /// error recorded; a fetch other than the one recorded is this host's
    /// failure.
    pub fn answer(&self, step: &FetchStep) -> Result<Replayed<'_>> {
        let position = self.next.get();
        let Some(Recorded {
            request,
            plan,
            statements,
            answer,
        }) = self.fetches.get(position)
        else {
            if self.past_the_end {
                eprintln!(
                    "rust-host: {}: the core asked fetch {position}, past the {} recorded: a \
                     diagnostic case, answered as the provider's error",
                    self.case,
                    self.fetches.len()
                );
                return Ok(Replayed {
                    statements: &[],
                    answer: Err(ProviderError {
                        driver_message: None,
                        error_class: None,
                        driver_message: None,
                        kind: ErrorKind::DataSource,
                        retryable: false,
                        sqlstate: None,
                        vendor_code: Some(PAST_THE_RECORDING.to_owned()),
                    }),
                });
            }
            return Err(PastTheRecording {
                case: self.case.clone(),
                position,
                recorded: self.fetches.len(),
            }
            .into());
        };
        if serde_json::to_value(&step.request)? != *request
            || serde_json::to_value(&step.plan)? != *plan
        {
            bail!(
                "{}: the core's fetch {position} is not the one recorded (its request or plan differs): record the case again",
                self.case
            );
        }
        self.next.set(position + 1);
        Ok(Replayed {
            statements,
            answer: match answer {
                Answer::Error(error) => Err(error.clone()),
                Answer::Data(bytes) => Ok(read_stream(bytes)
                    .with_context(|| format!("{}: fetch {position}'s data", self.case))?),
            },
        })
    }

    /// The statements the driver ran for every fetch answered so far, in
    /// order: the case's query log.
    pub fn ran(&self) -> Vec<String> {
        self.fetches[..self.next.get()]
            .iter()
            .flat_map(|fetch| fetch.statements.iter().cloned())
            .collect()
    }

    /// Whether every recorded fetch was asked.
    pub fn finished(&self) -> bool {
        self.next.get() == self.fetches.len()
    }
}

/// An Arrow IPC stream's schema and batches.
fn read_stream(bytes: &[u8]) -> Result<Fetched> {
    let reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| anyhow!("not an Arrow IPC stream: {e}"))?;
    let schema = reader.schema();
    let batches = reader
        .collect::<std::result::Result<Vec<RecordBatch>, _>>()
        .map_err(|e| anyhow!("an unreadable batch: {e}"))?;
    Ok(Fetched { schema, batches })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use duckdb::arrow::array::Int64Array;
    use duckdb::arrow::datatypes::{DataType, Field, Schema};
    use serde_json::json;

    use super::*;

    fn step(column: &str) -> FetchStep {
        serde_json::from_value(json!({
            "id": 1,
            "request": {"request": "scan", "table": {"source": ["t"]},
                        "columns": [column], "filters": [], "limit": null},
            "plan": {"plan": "host", "expect": {"expect": "discover"}},
        }))
        .unwrap()
    }

    fn recording(dir: &Path) -> Recording {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1, 2]))])
                .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut bytes, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let asked = step("v");
        let line = json!({
            "format": FORMAT,
            "case": "c@postgres",
            "fetches": [
                {"request": asked.request, "plan": asked.plan, "statements": ["SELECT 1"],
                 "data": base64::engine::general_purpose::STANDARD.encode(&bytes)},
                {"request": asked.request, "plan": asked.plan, "statements": [],
                 "error": {"kind": "data_source", "retryable": false,
                           "sqlstate": "42P01", "vendor_code": null}},
            ],
        });
        let path = dir.join("recording.jsonl");
        std::fs::write(&path, format!("{line}\n")).unwrap();
        Recording::load(&path).unwrap()
    }

    #[test]
    fn each_fetch_is_answered_in_order_and_another_fetch_fails_by_name() {
        let dir = std::env::temp_dir().join(format!("sdm-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let recording = recording(&dir);
        let replay = recording.replay("c@postgres").unwrap();
        let first = replay.answer(&step("v")).unwrap();
        assert_eq!(first.statements, ["SELECT 1"]);
        let Ok(fetched) = first.answer else {
            panic!("the recorded rows");
        };
        assert_eq!(fetched.batches[0].num_rows(), 2);
        assert!(!replay.finished());
        let Err(failed) = replay.answer(&step("v")).unwrap().answer else {
            panic!("the recorded error");
        };
        assert_eq!(failed.sqlstate.as_deref(), Some("42P01"));
        assert!(replay.finished());
        // Past the recording, and a fetch other than the one recorded.
        assert!(replay.answer(&step("v")).is_err());
        let other = recording.replay("c@postgres").unwrap();
        let Err(refused) = other.answer(&step("w")) else {
            panic!("a fetch other than the one recorded is answered");
        };
        let refused = refused.to_string();
        assert!(refused.contains("not the one recorded"), "{refused}");
        assert!(recording.replay("d@postgres").is_err());
        std::fs::remove_file(dir.join("recording.jsonl")).unwrap();
    }
}
