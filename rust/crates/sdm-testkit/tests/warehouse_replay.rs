// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The existing strict replay can answer warehouse fetches without an emulator.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use sdm_contract::data::{AttachmentId, FetchPlan};
use sdm_contract::ops::{DescribeSource, Operation};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{Attachment, HostRun, default_root, encode, read};
use sdm_testkit::host;
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "sdm-warehouse-replay-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}

fn provider(leg: &str, view: bool) -> Value {
    let (dialect, namespace) = match leg {
        "databricks-sql" | "databricks-spark" => {
            ("databricks", vec!["test_catalog", "test_schema"])
        }
        "snowflake" => ("snowflake", vec!["test_database", "test_schema"]),
        "lakebase" => ("postgres", vec!["test_schema"]),
        _ => unreachable!(),
    };
    let mut provider = json!({"provider": "sql", "dialect": dialect, "namespace": namespace});
    if view {
        provider["view"] = json!({"name": "metric_view", "trust_expressions": false});
    }
    provider
}

fn core(provider: &Value) -> Value {
    json!({"describe_source": {
        "provider": provider,
        "source": {"tables": [{"name": "people", "source": {"source": ["users"]}}],
                   "primary_keys": {}, "time_columns": {}},
        "check_keys": false,
    }})
}

/// Ask the actual core for its first fetch batch, and record driver failures.
/// Tests do not implement a warehouse or manufacture a successful table answer.
fn recording(leg: &str, core: &Value) -> Value {
    let spec = &core["describe_source"];
    let message = encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: host::facts().unwrap(),
        operation: Operation::DescribeSource(DescribeSource {
            provider: serde_json::from_value(spec["provider"].clone()).unwrap(),
            source: AttachmentId(1),
            check_keys: false,
        }),
    })))
    .unwrap();
    let run = HostRun::start(
        &default_root(),
        &message,
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec["source"]).unwrap()),
        )]),
    )
    .unwrap();
    let out = run.next().unwrap();
    let ControlMessage::Step(Step::Fetch { requests }) = read(&out).unwrap() else {
        panic!("describe did not ask a fetch");
    };
    assert!(!requests.is_empty());
    let fetches: Vec<Value> = requests
        .iter()
        .map(|step| {
            let FetchPlan::Sql { statements, .. } = &step.plan else {
                panic!("a warehouse fetch is not SQL");
            };
            json!({"request": step.request, "plan": step.plan,
               "statements": statements.iter().map(|s| s.text.clone()).collect::<Vec<_>>(),
               "error": {"kind": "data_source", "retryable": false}})
        })
        .collect();
    json!({"format": "sdm-replay/1", "case": format!("t310.replay@{leg}"),
        "core": core, "fetches": fetches})
}

fn case(path: &Path, leg: &str) -> PathBuf {
    let mut case: Value = serde_json::from_str(include_str!("contract-run-legs-s2.json")).unwrap();
    case["id"] = json!("t310.replay");
    case["warehouses"] = json!([leg]);
    case["until"] = json!("from_source");
    case["expect"] = json!({"outcome": "success", "stage": "from_source"});
    case["input"]["from_source"] = json!({"tables": {"people": "users"}});
    let file = path.join("case.json");
    std::fs::write(&file, serde_json::to_vec(&case).unwrap()).unwrap();
    file
}

fn replay(dir: &Path, leg: &str, line: &Value, case_leg: &str) -> (Output, PathBuf) {
    let recorded = dir.join("recording.jsonl");
    std::fs::write(&recorded, format!("{line}\n")).unwrap();
    let case = case(dir, case_leg);
    let out = dir.join("out.jsonl");
    let done = Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .args([
            "--replay",
            recorded.to_str().unwrap(),
            "--replay-leg",
            leg,
            "--out",
        ])
        .arg(&out)
        .arg(case)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    (done, out)
}

#[test]
fn every_warehouse_replays_the_actual_provider_and_source_arguments() {
    for leg in [
        "databricks-sql",
        "databricks-spark",
        "snowflake",
        "lakebase",
    ] {
        let dir = Scratch::new();
        let core = core(&provider(leg, false));
        let (done, out) = replay(&dir.0, leg, &recording(leg, &core), leg);
        assert!(
            done.status.success(),
            "{}",
            String::from_utf8_lossy(&done.stderr)
        );
        let observed: Value =
            serde_json::from_str(std::fs::read_to_string(out).unwrap().trim()).unwrap();
        assert_eq!(observed["case"], format!("t310.replay@{leg}"));
        assert_eq!(
            observed["core"]["inputs"][0]["provider"],
            core["describe_source"]["provider"]
        );
        assert_eq!(
            observed["core"]["inputs"][0]["source"],
            core["describe_source"]["source"]
        );
        assert_eq!(observed["outcome"]["result"], "refusal");
    }
}

#[test]
fn namespace_and_metric_view_are_preserved_and_cannot_answer_another_fetch() {
    let leg = "databricks-sql";
    for change in ["none", "namespace", "view"] {
        let dir = Scratch::new();
        let original = core(&provider(leg, true));
        let mut line = recording(leg, &original);
        if change == "namespace" {
            line["core"]["describe_source"]["provider"]["namespace"] = json!([]);
        } else if change == "view" {
            line["core"]["describe_source"]["provider"]
                .as_object_mut()
                .unwrap()
                .remove("view");
        }
        let (done, out) = replay(&dir.0, leg, &line, leg);
        if change == "none" {
            assert!(
                done.status.success(),
                "{}",
                String::from_utf8_lossy(&done.stderr)
            );
        } else {
            assert!(!done.status.success());
            assert!(!out.exists());
        }
    }
}

#[test]
fn missing_start_wrong_provider_wrong_leg_and_skipped_case_are_not_passes() {
    let leg = "databricks-sql";
    for fault in ["start", "provider", "key", "case", "unasked", "schema"] {
        let dir = Scratch::new();
        let mut line = recording(leg, &core(&provider(leg, false)));
        let mut case_leg = leg;
        match fault {
            "start" => {
                line.as_object_mut().unwrap().remove("core");
            }
            "provider" => {
                line["core"]["describe_source"]["provider"] = provider("lakebase", false);
            }
            "key" => {
                line["case"] = json!("t310.replay@databricks-spark");
            }
            "case" => {
                case_leg = "databricks-spark";
            }
            "unasked" => {
                let fetch = line["fetches"][0].clone();
                line["fetches"].as_array_mut().unwrap().push(fetch);
            }
            "schema" => {
                line["core"]["materialize"] = json!({"provider": provider(leg, false), "indices": [], "anchor_time": null, "seed": null, "schema": {"injected": true}});
            }
            _ => unreachable!(),
        }
        let (done, out) = replay(&dir.0, leg, &line, case_leg);
        assert!(!done.status.success(), "fault {fault} passed");
        assert!(!out.exists(), "fault {fault} wrote an observation");
    }
}

#[test]
fn a_started_refusal_before_any_fetch_is_observed() {
    let leg = "databricks-sql";
    let dir = Scratch::new();
    let mut core = core(&provider(leg, false));
    // The dialect cannot interpret a namespace with three parts, so the
    // actual core refuses before requesting any SQL from the provider.
    core["describe_source"]["provider"]["namespace"] = json!(["one", "two", "three"]);
    let line = json!({"format": "sdm-replay/1", "case": "t310.replay@databricks-sql",
                      "core": core, "fetches": []});
    let (done, out) = replay(&dir.0, leg, &line, leg);
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let observed: Value =
        serde_json::from_str(std::fs::read_to_string(out).unwrap().trim()).unwrap();
    assert_eq!(observed["outcome"]["result"], "refusal");
    assert_eq!(observed["core"]["fetches"], json!([]));
}
