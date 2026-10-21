// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Rust scripted endpoint against the shared vector file
//! (`contract/scripted-endpoint.json`), which the Python endpoint answers
//! too (`contract/tests/test_scripted_endpoint.py`): every vector's status
//! and answer, and a vector for every reason the rule refuses with.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::endpoint::{Endpoint, FAILED, REASONS, Told, answer, answer_call, answer_told};
use serde_json::Value;

fn vectors() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/scripted-endpoint.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_rule_answers_each_vector() {
    let document = vectors();
    assert_eq!(document["format"], "sdm-scripted-endpoint/1");
    let vectors = document["vectors"].as_array().unwrap();
    // Not vacuous: the file holds the vectors T1.11a recorded.
    assert!(vectors.len() >= 20, "{}", vectors.len());
    let mut names = BTreeSet::new();
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        assert!(names.insert(name), "{name} twice");
        let (status, answered) = answer(
            vector["method"].as_str().unwrap(),
            vector["path"].as_str().unwrap(),
            vector["body"].as_str().unwrap().as_bytes(),
        );
        assert_eq!(
            u64::from(status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, vector["answer"], "{name}");
        // The answer's bytes read back as the same JSON, a float as a float.
        let bytes = serde_json::to_vec(&answered).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            vector["answer"],
            "{name}"
        );
    }
}

#[test]
fn every_reason_has_a_vector() {
    let document = vectors();
    let refused: BTreeSet<&str> = document["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["status"] == 400)
        .map(|v| v["answer"]["error"].as_str().unwrap())
        .collect();
    assert_eq!(refused, REASONS.into_iter().collect());
}

/// Four different requests, so an answer given to the wrong call shows
/// (the calls of `tests/test_t21a_scripted_calls.py`).
const CALLS: [&str; 4] = [
    "regression",
    "regression, instance_id not the first column",
    "binary classification answers the first class",
    "regression",
];

fn vector(name: &str) -> Value {
    vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|vector| vector["name"] == name)
        .unwrap()
        .clone()
}

fn asked(endpoint: &mut Endpoint, name: &str) -> (u64, Value) {
    let vector = vector(name);
    let (status, answered) = endpoint.answer(
        vector["method"].as_str().unwrap(),
        vector["path"].as_str().unwrap(),
        vector["body"].as_str().unwrap().as_bytes(),
    );
    (u64::from(status), answered)
}

fn expected(fail_call: Option<u64>) -> Vec<(u64, Value)> {
    CALLS
        .iter()
        .zip(0_u64..)
        .map(|(name, call)| {
            if fail_call == Some(call) {
                (
                    u64::from(FAILED),
                    serde_json::json!({"error": "scripted_failure", "call": call}),
                )
            } else {
                let vector = vector(name);
                (vector["status"].as_u64().unwrap(), vector["answer"].clone())
            }
        })
        .collect()
}

/// T2.1a: calls are numbered from zero, the chosen one (the first and the
/// last included) answered 500 naming its number and every other by the
/// rule; a call past the last, or none, fails nothing.
#[test]
fn the_endpoint_numbers_its_calls_and_fails_the_chosen_one() {
    let distinct: BTreeSet<String> = expected(None)
        .iter()
        .map(|(_, answer)| answer.to_string())
        .collect();
    assert_eq!(distinct.len(), 3);
    for fail_call in [None, Some(0), Some(2), Some(3), Some(4)] {
        let mut endpoint = Endpoint::new(fail_call);
        let answers: Vec<_> = CALLS
            .iter()
            .map(|name| asked(&mut endpoint, name))
            .collect();
        let failed = fail_call.filter(|&call| call < 4);
        assert_eq!(answers, expected(failed), "{fail_call:?}");
        assert_eq!(endpoint.calls(), 4);
    }
}

#[test]
fn the_rule_numbers_the_call_it_is_given() {
    let vector = vector("regression");
    let asked = (
        vector["method"].as_str().unwrap(),
        vector["path"].as_str().unwrap(),
        vector["body"].as_str().unwrap().as_bytes(),
    );
    assert_eq!(
        answer_call(asked.0, asked.1, asked.2, 1, Some(1)),
        (
            FAILED,
            serde_json::json!({"error": "scripted_failure", "call": 1})
        )
    );
    assert_eq!(
        answer_call(asked.0, asked.1, asked.2, 0, Some(1)),
        answer(asked.0, asked.1, asked.2)
    );
    // A failure is the chosen call's whatever it asks, a 404 included.
    assert_eq!(answer_call("GET", "/", b"", 0, Some(0)).0, FAILED);
}

/// T2.8: an endpoint told to answer probabilities answers each of the
/// file's `probabilities` vectors, as call number `call`, as the Python
/// endpoint does: the status, the answer, and each item's classes in the
/// order given.
#[test]
fn the_rule_answers_each_probabilities_vector() {
    let document = vectors();
    let vectors = document["probabilities"].as_array().unwrap();
    // Not vacuous: probabilities given on calls past the first, of a binary
    // and a multiclass, and a call asking for none answered by the rule.
    assert!(vectors.len() >= 5, "{}", vectors.len());
    let given = |v: &Value| v["answer"]["predictions"][0].get("probabilities").is_some();
    assert!(vectors.iter().any(|v| given(v) && v["call"] != 0));
    assert!(vectors.iter().any(|v| !given(v)));
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        let (status, answered) = sdm_testkit::endpoint::answer_with_probabilities(
            vector["method"].as_str().unwrap(),
            vector["path"].as_str().unwrap(),
            vector["body"].as_str().unwrap().as_bytes(),
            vector["call"].as_u64().unwrap(),
        );
        assert_eq!(
            u64::from(status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, vector["answer"], "{name}");
        let classes = |answer: &Value| -> Vec<Vec<String>> {
            answer["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    item.get("probabilities")
                        .and_then(Value::as_object)
                        .map(|p| p.keys().cloned().collect())
                        .unwrap_or_default()
                })
                .collect()
        };
        assert_eq!(classes(&answered), classes(&vector["answer"]), "{name}");
    }
}

/// T2.9: an endpoint told to answer quantiles answers each of the file's
/// `quantiles` vectors, as call number `call`, as the Python endpoint
/// does: the status, the answer, and each item's quantiles in the order
/// given; told to answer probabilities only, or nothing, it answers none.
#[test]
fn the_rule_answers_each_quantiles_vector() {
    let document = vectors();
    let vectors = document["quantiles"].as_array().unwrap();
    // Not vacuous: quantiles given on calls past the first, and a call
    // asking for none, and a classification, answered by the rule.
    assert!(vectors.len() >= 5, "{}", vectors.len());
    let given = |v: &Value| v["answer"]["predictions"][0].get("quantiles").is_some();
    assert!(vectors.iter().any(|v| given(v) && v["call"] != 0));
    assert!(vectors.iter().any(|v| !given(v)));
    let spread = |answer: &Value| -> Vec<Vec<String>> {
        answer["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                item.get("quantiles")
                    .and_then(Value::as_object)
                    .map(|q| q.keys().cloned().collect())
                    .unwrap_or_default()
            })
            .collect()
    };
    for vector in vectors {
        let name = vector["name"].as_str().unwrap();
        let asked = (
            vector["method"].as_str().unwrap(),
            vector["path"].as_str().unwrap(),
            vector["body"].as_str().unwrap().as_bytes(),
            vector["call"].as_u64().unwrap(),
        );
        let (status, answered) =
            sdm_testkit::endpoint::answer_as_asked(asked.0, asked.1, asked.2, asked.3, false, true);
        assert_eq!(
            u64::from(status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, vector["answer"], "{name}");
        assert_eq!(spread(&answered), spread(&vector["answer"]), "{name}");
        for (probabilities, quantiles) in [(false, false), (true, false)] {
            let (_, answered) = sdm_testkit::endpoint::answer_as_asked(
                asked.0,
                asked.1,
                asked.2,
                asked.3,
                probabilities,
                quantiles,
            );
            assert!(
                spread(&answered).iter().all(Vec::is_empty),
                "{name}: quantiles where none were to be given"
            );
        }
    }
}

/// The answer of each of the file's `section` vectors by an endpoint told
/// `told`, as call number `call`, with the vector's name.
fn answered_told(section: &str, told: Told) -> Vec<(String, Value, u16, Value)> {
    let document = vectors();
    document[section]
        .as_array()
        .unwrap()
        .iter()
        .map(|vector| {
            let (status, answered) = answer_told(
                vector["method"].as_str().unwrap(),
                vector["path"].as_str().unwrap(),
                vector["body"].as_str().unwrap().as_bytes(),
                vector["call"].as_u64().unwrap(),
                told,
            );
            (
                vector["name"].as_str().unwrap().to_owned(),
                vector.clone(),
                status,
                answered,
            )
        })
        .collect()
}

/// Whether any item of `answer` carries `field`.
fn carries(answer: &Value, field: &str) -> bool {
    answer["predictions"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item.get(field).is_some()))
}

/// T2.13: an endpoint told to answer explanations answers each of the
/// file's `explanation` vectors, as call number `call`, as the Python
/// endpoint does; untold, or told everything else, it answers none.
#[test]
fn the_rule_answers_each_explanation_vector() {
    let told = Told {
        explanation: true,
        ..Told::default()
    };
    let answers = answered_told("explanation", told);
    // Not vacuous: an explanation given on a call past the first and over
    // several rows, and a call asking for none answered by the rule.
    assert!(answers.len() >= 5, "{}", answers.len());
    let given = |v: &Value| carries(&v["answer"], "explanation");
    assert!(answers.iter().any(|(_, v, ..)| given(v) && v["call"] != 0));
    assert!(answers.iter().any(|(_, v, ..)| {
        given(v) && v["answer"]["predictions"].as_array().unwrap().len() > 1
    }));
    assert!(answers.iter().any(|(_, v, ..)| !given(v)));
    for (name, vector, status, answered) in &answers {
        assert_eq!(
            u64::from(*status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, &vector["answer"], "{name}");
    }
    let everything_else = Told {
        probabilities: true,
        quantiles: true,
        item_metadata: true,
        ..Told::default()
    };
    for told in [Told::default(), everything_else] {
        for (name, _, _, answered) in answered_told("explanation", told) {
            assert!(
                !carries(&answered, "explanation"),
                "{name}: an explanation where none was to be given"
            );
        }
    }
}

/// T2.13: an endpoint told to answer item metadata answers each of the
/// file's `item_metadata` vectors, as call number `call`, as the Python
/// endpoint does; untold, or told everything else, no item carries any.
#[test]
fn the_rule_answers_each_item_metadata_vector() {
    let told = Told {
        item_metadata: true,
        ..Told::default()
    };
    let answers = answered_told("item_metadata", told);
    assert!(answers.len() >= 5, "{}", answers.len());
    let given = |v: &Value| carries(&v["answer"], "metadata");
    assert!(answers.iter().any(|(_, v, ..)| given(v) && v["call"] != 0));
    assert!(answers.iter().any(|(_, v, ..)| v["status"] == 400));
    for (name, vector, status, answered) in &answers {
        assert_eq!(
            u64::from(*status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, &vector["answer"], "{name}");
    }
    let everything_else = Told {
        probabilities: true,
        quantiles: true,
        explanation: true,
        ..Told::default()
    };
    for told in [Told::default(), everything_else] {
        for (name, _, _, answered) in answered_told("item_metadata", told) {
            assert!(
                !carries(&answered, "metadata"),
                "{name}: item metadata where none was to be given"
            );
        }
    }
}

/// T2.13: the rule as `answer_as_asked` gave it before `Told` is
/// `answer_told` told only probabilities and quantiles, every earlier
/// section's vector answered the same either way.
#[test]
fn answer_as_asked_is_answer_told_without_the_new_fields() {
    let document = vectors();
    for section in ["vectors", "probabilities", "quantiles"] {
        for vector in document[section].as_array().unwrap() {
            let asked = (
                vector["method"].as_str().unwrap(),
                vector["path"].as_str().unwrap(),
                vector["body"].as_str().unwrap().as_bytes(),
                vector["call"].as_u64().unwrap_or(0),
            );
            for (probabilities, quantiles) in [(false, false), (true, false), (false, true)] {
                assert_eq!(
                    sdm_testkit::endpoint::answer_as_asked(
                        asked.0,
                        asked.1,
                        asked.2,
                        asked.3,
                        probabilities,
                        quantiles
                    ),
                    answer_told(
                        asked.0,
                        asked.1,
                        asked.2,
                        asked.3,
                        Told {
                            probabilities,
                            quantiles,
                            ..Told::default()
                        }
                    ),
                    "{section}: {}",
                    vector["name"]
                );
            }
        }
    }
}

/// T2.11: an endpoint told to answer embeddings answers each of the file's
/// `embeddings` vectors, as call number `call`, as the Python endpoint
/// does; untold, or told everything else, it answers none.
#[test]
fn the_rule_answers_each_embeddings_vector() {
    let told = Told {
        embeddings: true,
        ..Told::default()
    };
    let answers = answered_told("embeddings", told);
    // Not vacuous: embeddings given on a call past the first and over
    // several rows, and a call asking for none answered by the rule.
    assert!(answers.len() >= 5, "{}", answers.len());
    let given = |v: &Value| carries(&v["answer"], "embeddings");
    assert!(answers.iter().any(|(_, v, ..)| given(v) && v["call"] != 0));
    assert!(answers.iter().any(|(_, v, ..)| {
        given(v) && v["answer"]["predictions"].as_array().unwrap().len() > 1
    }));
    assert!(answers.iter().any(|(_, v, ..)| !given(v)));
    for (name, vector, status, answered) in &answers {
        assert_eq!(
            u64::from(*status),
            vector["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(answered, &vector["answer"], "{name}");
    }
    let everything_else = Told {
        probabilities: true,
        quantiles: true,
        explanation: true,
        item_metadata: true,
        ..Told::default()
    };
    for told in [Told::default(), everything_else] {
        for (name, _, _, answered) in answered_told("embeddings", told) {
            assert!(
                !carries(&answered, "embeddings"),
                "{name}: embeddings where none were to be given"
            );
        }
    }
}
