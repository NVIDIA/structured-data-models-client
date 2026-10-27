// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real DuckDB fetches exercise the private replay boundary without a warehouse emulator.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads a public case and writes a temporary real-fetch recording"
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine;

use super::*;

const KEY: &str = "public-query-validation-replay";

fn public_case() -> Case {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contract/cases/s1/refuse.unknown-column.json");
    let mut case = crate::case::load(&[path]).unwrap().remove(0).case;
    case.until = Until::Materialize;
    let core = case.core.as_mut().unwrap();
    core.validate_query = None;
    core.materialize = None;
    core.predict = None;
    case
}

/// Record the real native description and real DuckDB answers, not a
/// description supplied as replay input or a manufactured refusal.
fn description_recording(case: &Case, extra_fetch: bool) -> Recording {
    let contract = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract");
    let bases = crate::bases::load(&contract.join("bases.yaml")).unwrap();
    let connection = database(case, &bases).unwrap();
    let spec = case
        .core
        .as_ref()
        .unwrap()
        .describe_source
        .as_ref()
        .unwrap();
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: facts().unwrap(),
        operation: Operation::DescribeSource(DescribeSource {
            provider: serde_json::from_value(spec.provider.clone()).unwrap(),
            source: AttachmentId(1),
            check_keys: spec.check_keys,
        }),
    }));
    let handle = HostRun::start(
        &default_root(),
        &encode(&start).unwrap(),
        BTreeMap::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec.source).unwrap()),
        )]),
    )
    .unwrap();
    let mut fetches = Vec::new();
    let mut log = DriverLog::default();
    loop {
        let message = handle.next().expect("the real description must succeed");
        match read(&message).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::DescribeSource(_),
            }) => break,
            ControlMessage::Step(Step::Fetch { requests }) => {
                for step in requests {
                    assert!(fetches.len() < 100, "description did not finish");
                    let fetched =
                        duck::answer_logged(&connection, &step, &BTreeMap::new(), &mut log)
                            .expect("the existing DuckDB provider must answer");
                    let mut bytes = Vec::new();
                    {
                        let mut writer =
                            arrow_ipc::writer::StreamWriter::try_new(&mut bytes, &fetched.schema)
                                .unwrap();
                        for batch in &fetched.batches {
                            writer.write(batch).unwrap();
                        }
                        writer.finish().unwrap();
                    }
                    let FetchPlan::Sql { statements, .. } = &step.plan else {
                        panic!("the real DuckDB provider must receive SQL");
                    };
                    fetches.push(json!({
                        "request": step.request,
                        "plan": step.plan,
                        "statements": statements.iter().map(|s| &s.text).collect::<Vec<_>>(),
                        "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                    }));
                    handle
                        .reply(
                            &encode(&ControlMessage::FetchReply(FetchReply {
                                id: step.id,
                                outcome: FetchOutcome::Data {
                                    data: AttachmentId(1),
                                    continuation: None,
                                },
                            }))
                            .unwrap(),
                            BTreeMap::from([(
                                AttachmentId(1),
                                Attachment::Arrow(ArrowData::new(fetched.schema, fetched.batches)),
                            )]),
                        )
                        .unwrap();
                }
            }
            _ => panic!("unexpected native description step"),
        }
    }
    assert!(!fetches.is_empty());
    if extra_fetch {
        fetches.push(fetches[0].clone());
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "sdm-native-validation-replay-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("recording.jsonl");
    let line = json!({"format": "sdm-replay/1", "case": KEY,
                      "core": case.core, "fetches": fetches});
    std::fs::write(&path, format!("{line}\n")).unwrap();
    let recording = Recording::load(&path).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    recording
}

fn replayed(case: &Case, edits: &[SchemaEdit], extra_fetch: bool) -> (Observed, Result<()>) {
    let recording = description_recording(case, extra_fetch);
    let replay = recording.replay(KEY).unwrap();
    let observed = describe_on(
        case,
        case.core
            .as_ref()
            .unwrap()
            .describe_source
            .as_ref()
            .unwrap(),
        Source::Replay {
            answers: &replay,
            preserve_provider: true,
        },
        &mut DriverLog::default(),
        edits,
    )
    .unwrap();
    let complete = require_replayed_completion(case, &observed, KEY, true, &replay);
    (observed, complete)
}

#[test]
fn a_partial_recording_runs_real_native_validation_and_refuses_the_missing_column() {
    let case = public_case();
    let (observed, complete) = replayed(&case, &[], false);
    complete.unwrap();
    assert_eq!(observed.operations, ["describe_source", "validate_query"]);
    assert_eq!(observed.answer, Answer::Validated(false));
    assert_eq!(
        observed.answer.outcome(),
        ("refusal", Some("INVALID_REQUEST".into()))
    );
    assert_eq!(
        observed.inputs[1]["profile"],
        json!({"name": "RELATIONAL", "model": null})
    );
    assert!(observed.materialized.is_none());
}

#[test]
fn accepted_validation_does_not_invent_a_missing_materialize_start() {
    let mut case = public_case();
    case.input.pql = case.input.pql.replace("orders.cost", "orders.price");
    let (observed, complete) = replayed(&case, &[], false);
    assert_eq!(observed.answer, Answer::Validated(true));
    assert_eq!(observed.operations, ["describe_source", "validate_query"]);
    assert!(
        complete
            .unwrap_err()
            .to_string()
            .contains("no materialize start")
    );
    assert!(observed.materialized.is_none());
}

#[test]
fn validation_uses_the_schema_rebuilt_and_edited_by_this_host() {
    let mut case = public_case();
    case.input.pql = case.input.pql.replace("orders.cost", "orders.price");
    let edit = SchemaEdit::DeleteColumn {
        table: "orders".into(),
        column: "price".into(),
    };
    let (observed, complete) = replayed(&case, &[edit], false);
    complete.unwrap();
    assert_eq!(
        observed.operations,
        ["describe_source", "edit_schema", "validate_query"]
    );
    assert_eq!(observed.answer, Answer::Validated(false));
    let orders = observed.inputs[1]["schema"]["tables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|table| table["name"] == "orders")
        .unwrap();
    assert!(
        !orders["columns"]
            .as_array()
            .unwrap()
            .iter()
            .any(|column| column["name"] == "price")
    );
    let original = observed.described.as_ref().unwrap()["tables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|table| table["name"] == "orders")
        .unwrap();
    assert!(
        original["columns"]
            .as_array()
            .unwrap()
            .iter()
            .any(|column| column["name"] == "price")
    );
}

#[test]
fn an_earlier_edit_refusal_is_not_masked_by_unconvertible_later_options() {
    let mut case = public_case();
    case.input.materialize.indices_as = Some(crate::case::IndicesAs::Decimal);
    let edit = SchemaEdit::SetPrimaryKey {
        table: "orders".into(),
        column: "missing".into(),
    };
    let (observed, complete) = replayed(&case, &[edit], false);
    complete.unwrap();
    assert_eq!(observed.operations, ["describe_source", "edit_schema"]);
    assert_eq!(observed.answer.outcome().0, "refusal");
}

#[test]
fn a_real_validation_refusal_does_not_excuse_unconsumed_fetches() {
    let case = public_case();
    let (observed, complete) = replayed(&case, &[], true);
    assert_eq!(observed.answer, Answer::Validated(false));
    assert!(complete.unwrap_err().to_string().contains("fewer fetches"));
}

#[test]
fn an_explicit_recorded_validator_profile_is_preserved() {
    let mut case = public_case();
    let original = crate::case::load(&[PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contract/cases/s1/refuse.unknown-column.json")])
    .unwrap()
    .remove(0)
    .case;
    let mut validation = original.core.unwrap().validate_query.unwrap();
    validation.profile["model"] = json!("recorded-model");
    let expected = validation.profile.clone();
    case.core.as_mut().unwrap().validate_query = Some(validation);
    let (observed, complete) = replayed(&case, &[], false);
    complete.unwrap();
    assert_eq!(observed.inputs[1]["profile"], expected);
    assert_eq!(observed.answer, Answer::Validated(false));
}
