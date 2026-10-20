// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The S0 cases of `contract/cases` through the pure-Rust host: each loads,
//! runs through `HostRun` and meets its expected outcome and code. CI's
//! `rust-test` job also runs `sdm-contract-run` over them and compares its
//! observations with the reference's (`scripts/rust_core/compare.py`).
//!
//! From T1.6c a query with a date literal with a run of five digits or more
//! (D8) ends `validate_query` with the core's refusal by name,
//! `INVALID_REQUEST` naming `operation` `validate_query` and `rule`
//! `date_literal_packed`, where the reference reads the literal.

use std::path::PathBuf;

use sdm_testkit::case::{self, Status, Until};
use sdm_testkit::host::{self, Answer};

/// Whether `query` holds a date literal with a run of five digits or more
/// (D8): a `DATE` token, three runs of digits joined by one of `/`, `-` or
/// `.`, with such a run, or followed by a `TIME` token, three runs joined
/// by `:`, with one.
fn holds_packed_date(query: &str) -> bool {
    let words: Vec<&str> = query
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ','))
        .collect();
    let runs = |word: &str, separators: &[char]| -> Option<Vec<usize>> {
        let sep = word.chars().find(|c| separators.contains(c))?;
        let parts: Vec<&str> = word.split(sep).collect();
        let digits = |p: &&str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
        (parts.len() == 3 && parts.iter().all(digits))
            .then(|| parts.iter().map(|p| p.len()).collect())
    };
    words.iter().enumerate().any(|(i, word)| {
        let Some(date) = runs(word, &['/', '-', '.']) else {
            return false;
        };
        let time = words.get(i + 1).and_then(|next| runs(next, &[':']));
        date.iter().chain(time.iter().flatten()).any(|n| *n >= 5)
    })
}

/// Whether `answer` is the core's refusal by name of a D8 literal (T1.6c).
fn is_packed_refusal(answer: &Answer) -> bool {
    matches!(
        answer,
        Answer::Error { kind, code, fields }
            if kind == "invalid_request"
                && code == "INVALID_REQUEST"
                && fields.get("operation").map(String::as_str) == Some("validate_query")
                && fields.get("rule").map(String::as_str) == Some("date_literal_packed")
    )
}

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/cases")
}

fn bases() -> sdm_testkit::bases::Bases {
    sdm_testkit::bases::load(&cases_dir().join("../bases.yaml")).unwrap()
}

#[test]
fn the_s0_cases_meet_their_expectation_on_the_pure_rust_host() {
    let loaded = case::load(&[cases_dir()]).unwrap();
    let s0: Vec<_> = loaded
        .iter()
        .filter(|each| each.case.slice == "S0")
        .collect();
    assert!(s0.len() >= 2, "the accepted and the refused S0 case");
    let mut answers = Vec::new();
    let mut packed = 0;
    for each in s0 {
        let case = &each.case;
        assert_eq!(case.until, Until::Materialize);
        let observed = host::run(case, &bases()).unwrap();
        assert_eq!(observed.operations, ["validate_query"]);
        // T1.6c: a case whose query holds a D8 literal is refused by name,
        // whatever its status or register entry; an acceptance case's own
        // expectation must then be that refusal too.
        if holds_packed_date(&case.input.pql) {
            packed += 1;
            assert!(
                is_packed_refusal(&observed.answer),
                "{}: {:?}",
                case.id,
                observed.answer
            );
            if case.status == Status::Diagnostic {
                continue;
            }
        } else if case.status == Status::Diagnostic {
            // Gating nothing against the reference, but held to the core's
            // answer its register entry records, so a drift fails here:
            // escaped surrogates (PQL-1) and the core's deepest query
            // (PQL-2), which the core accepts where the reference refuses or
            // raises.
            let today = match case.register.as_deref() {
                Some("PQL-1" | "PQL-2") => ("success", None),
                other => panic!("{}: a diagnostic S0 case under {other:?}", case.id),
            };
            assert_eq!(observed.answer.outcome(), today, "{}", case.id);
            continue;
        }
        let (result, code) = observed.answer.outcome();
        assert_eq!(result, case.expect.outcome, "{}", case.id);
        assert_eq!(code, case.expect.code, "{}", case.id);
        answers.push(observed.answer);
    }
    assert!(answers.contains(&Answer::Validated(true)));
    assert!(answers.contains(&Answer::Validated(false)));
    // The D8 cases of T1.12's corpus (`s0.boundary.date-packed` and
    // `s0.boundary.date-long-field`) are still there to be refused.
    assert!(packed >= 2, "{packed} S0 cases with a packed date");
}

/// The S0 case `s0.refuse.date-impossible`, over the shop base, with `pql`
/// in place of its query.
fn with_query(pql: &str) -> case::Case {
    let mut loaded = case::load(&[cases_dir()]).unwrap();
    let at = loaded
        .iter()
        .position(|each| each.case.id == "s0.refuse.date-impossible")
        .unwrap();
    let mut case = loaded.remove(at).case;
    case.input.pql = pql.to_owned();
    case
}

/// T1.6c brief, Tests: "a D8 literal is refused by name on ... rust-host,
/// while D1 to D7 and D9 literals still read as before". `validate_query`
/// through `HostRun` ends with the core's error for each D8 shape (packed
/// first fields of six, eight, twelve and fourteen digits, a long field in
/// the time, a long zero-padded field, ones the reference refuses, one in
/// an array); a plain date, a time of day, a two-digit year, the year 0
/// and a day-first literal are accepted, and D9's refused with the
/// diagnostic (`Validated(false)`), not the named rule.
#[test]
fn a_packed_date_literal_is_refused_by_name_on_the_pure_rust_host() {
    let filtered = |condition: &str| {
        format!("PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) FOR EACH users.user_id")
    };
    for literal in [
        "230203-06-09",
        "20240101-01-01",
        "202401011230-1-1",
        "20240101123456-1-1",
        "20240101/00/123456",
        "2024-01-01 00000:00:00",
        "00001/1/2",
        "10000-01-01",
        "12345-1-1",
    ] {
        let query = filtered(&format!("orders.ts > {literal}"));
        assert!(holds_packed_date(&query), "{literal}");
        let observed = host::run(&with_query(&query), &bases()).unwrap();
        assert!(
            is_packed_refusal(&observed.answer),
            "{literal}: {:?}",
            observed.answer
        );
        assert_eq!(
            observed.answer.outcome(),
            ("refusal", Some("INVALID_REQUEST".to_owned()))
        );
    }
    let in_array = filtered("orders.ts IN (2024-01-01, 20240101-01-01)");
    let observed = host::run(&with_query(&in_array), &bases()).unwrap();
    assert!(is_packed_refusal(&observed.answer), "{:?}", observed.answer);
    for (literal, accepted) in [
        ("2024-01-31", true),
        ("2024-01-31 23:59:59", true),
        ("1/2/24", true),
        ("0000-01-01", true),
        ("13/01/2024", true),
        ("2024-01-01 0000:00:00", true),
        ("2023-02-29", false),
        ("123-4-567", false),
    ] {
        let query = filtered(&format!("orders.ts > {literal}"));
        assert!(!holds_packed_date(&query), "{literal}");
        let observed = host::run(&with_query(&query), &bases()).unwrap();
        assert_eq!(observed.answer, Answer::Validated(accepted), "{literal}");
    }
}

/// The accepted S0 case as JSON, for a test to change.
fn accepted() -> serde_json::Value {
    let path = cases_dir().join("s0/s0.accept.sum-window.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// A scratch directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("sdm-testkit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// Loads `case` written to a file: the loader's error, if any.
fn load_error(name: &str, case: &serde_json::Value) -> Option<String> {
    load_text_error(name, &serde_json::to_string(case).unwrap())
}

/// [`load_error`] of a case given as its file's text.
fn load_text_error(name: &str, case: &str) -> Option<String> {
    let directory = scratch(name);
    let path = directory.join("case.json");
    std::fs::write(&path, case).unwrap();
    let error = case::load(&[path]).err().map(|error| error.to_string());
    std::fs::remove_dir_all(&directory).unwrap();
    error
}

#[test]
fn the_accepted_case_loads_unchanged() {
    assert_eq!(load_error("unchanged", &accepted()), None);
}

#[test]
fn an_unknown_key_names_the_file_and_position_and_never_the_key() {
    let mut case = accepted();
    case["secret-canary"] = serde_json::json!(1);
    let error = load_error("top", &case).expect("refused");
    assert!(
        error.contains("case.json") && error.contains("line 1"),
        "{error}"
    );
    assert!(!error.contains("secret-canary"), "{error}");
}

/// The same key `contract/tests` shows the Python loader refusing.
#[test]
fn an_unknown_key_inside_the_call_arguments_is_refused() {
    let mut case = accepted();
    case["input"]["materialize"]["ignored_typo"] = serde_json::json!(true);
    assert!(load_error("nested-materialize", &case).is_some());
    let mut case = accepted();
    case["input"]["from_source"] = serde_json::json!({"ignored_typo": {}});
    assert!(load_error("nested-from-source", &case).is_some());
}

/// `contract/tests/malformed-cases.json`'s changes to the AdventureWorks
/// case: the ones `tests/test_contract_cases.py` shows the Python loader
/// refusing are refused here too, and the ones it takes are taken (T1.21b).
#[test]
fn the_malformed_edits_and_relationships_the_python_loader_refuses_are_refused() {
    let listed: serde_json::Value = serde_json::from_slice(
        &std::fs::read(cases_dir().join("../tests/malformed-cases.json")).unwrap(),
    )
    .unwrap();
    let case: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            cases_dir()
                .join("..")
                .join("..")
                .join(listed["case"].as_str().unwrap()),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(load_error("adventureworks", &case), None);
    // Each change's value is set as the file writes it: a placeholder goes
    // in, and the value's own text replaces it, since a JSON value read
    // into `serde_json::Value` loses an integer past 2**64 (T2.1a's
    // candidates).
    #[derive(serde::Deserialize)]
    struct Value<'a> {
        #[serde(borrow)]
        value: &'a serde_json::value::RawValue,
    }
    let text = std::fs::read_to_string(cases_dir().join("../tests/malformed-cases.json")).unwrap();
    #[derive(serde::Deserialize)]
    struct Lists<'a> {
        #[serde(borrow)]
        refused: Vec<Value<'a>>,
        #[serde(borrow)]
        accepted: Vec<Value<'a>>,
    }
    let raw: Lists = serde_json::from_str(&text).unwrap();
    let raw_value = |change: &serde_json::Value| {
        let lists = [&listed["refused"], &listed["accepted"]];
        let raws = [&raw.refused, &raw.accepted];
        for (list, raws) in lists.into_iter().zip(raws) {
            if let Some(at) = list
                .as_array()
                .unwrap()
                .iter()
                .position(|item| std::ptr::eq(item, change))
            {
                return raws[at].value.get().to_owned();
            }
        }
        panic!("{} is not a listed change", change["why"]);
    };
    let changed = |change: &serde_json::Value| {
        const PLACEHOLDER: &str = "\u{1}t21a-the-value\u{1}";
        let mut changed = case.clone();
        let path = change["path"].as_array().unwrap();
        let mut at = &mut changed;
        for key in path {
            at = match key {
                serde_json::Value::String(name) => &mut at[name.as_str()],
                _ => &mut at[key.as_u64().unwrap() as usize],
            };
        }
        *at = serde_json::Value::String(PLACEHOLDER.to_owned());
        let placeholder = serde_json::to_string(PLACEHOLDER).unwrap();
        let text = serde_json::to_string(&changed).unwrap();
        assert_eq!(text.matches(&placeholder).count(), 1);
        text.replace(&placeholder, &raw_value(change))
    };
    let refused = listed["refused"].as_array().unwrap();
    assert!(refused.len() >= 54);
    for change in refused {
        assert!(
            load_text_error("malformed", &changed(change)).is_some(),
            "{}",
            change["why"]
        );
    }
    let omitted = listed["omitted"].as_array().unwrap();
    assert!(!omitted.is_empty());
    for change in omitted {
        let mut without = case.clone();
        let path = change["path"].as_array().unwrap();
        let mut at = &mut without;
        for key in &path[..path.len() - 1] {
            at = &mut at[key.as_str().unwrap()];
        }
        at.as_object_mut()
            .unwrap()
            .remove(path[path.len() - 1].as_str().unwrap())
            .unwrap();
        assert_eq!(load_error("omitted", &without), None, "{}", change["why"]);
    }
    for change in listed["accepted"].as_array().unwrap() {
        assert_eq!(
            load_text_error("accepted", &changed(change)),
            None,
            "{}",
            change["why"]
        );
    }
}

/// `databases` and `exclusion` come together (`dependentRequired`), which a
/// one-path change of `malformed-cases.json` cannot set; the Python
/// loader's twin is `tests/test_t21a_case_format.py` (T2.1a).
#[test]
fn databases_come_only_with_their_exclusion() {
    let with = |pairs: &[(&str, serde_json::Value)]| {
        let mut case = accepted();
        for (key, value) in pairs {
            case[*key] = value.clone();
        }
        case
    };
    let narrowed = serde_json::json!(["duckdb", "sqlite"]);
    let row = serde_json::json!("KO2-19");
    let both = with(&[("databases", narrowed.clone()), ("exclusion", row.clone())]);
    assert_eq!(load_error("both", &both), None);
    let directory = scratch("narrowed");
    let path = directory.join("case.json");
    std::fs::write(&path, serde_json::to_vec(&both).unwrap()).unwrap();
    let [loaded] = <[_; 1]>::try_from(case::load(&[path]).unwrap()).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert_eq!(
        loaded.case.databases(),
        [case::Database::Duckdb, case::Database::Sqlite]
    );
    assert_eq!(
        case::load(&[cases_dir()]).unwrap()[0].case.databases(),
        case::Database::ALL
    );
    for alone in [
        with(&[("databases", narrowed.clone())]),
        with(&[("exclusion", row.clone())]),
    ] {
        assert!(load_error("alone", &alone).is_some());
    }
    for (databases, exclusion) in [
        (serde_json::json!([]), row.clone()),
        (serde_json::json!(["duckdb", "duckdb"]), row.clone()),
        (serde_json::json!(["duckdb", "oracle"]), row.clone()),
        (serde_json::json!(["DuckDB"]), row.clone()),
        (serde_json::json!("duckdb"), row.clone()),
        (narrowed.clone(), serde_json::json!("the owner said so")),
        (narrowed.clone(), serde_json::json!("KO2-")),
        (
            serde_json::json!(["duckdb", "sqlite", "postgres"]),
            row.clone(),
        ),
    ] {
        let case = with(&[("databases", databases), ("exclusion", exclusion)]);
        assert!(load_error("refused", &case).is_some(), "{case}");
    }
}

/// rust-host's marker for the staging rule (T2.1a): `--providers` prints
/// the test kit's provider list as JSON, DuckDB among them.
#[test]
fn the_host_reports_its_providers_as_plain_data() {
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .arg("--providers")
        .output()
        .unwrap();
    assert!(run.status.success());
    let listed: Vec<String> = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(listed, sdm_testkit::PROVIDERS);
    assert!(listed.iter().any(|name| name == "duckdb"));
}

#[test]
fn a_failed_run_leaves_no_output_an_earlier_run_wrote() {
    let directory = scratch("stale");
    let mut case = accepted();
    case.as_object_mut().unwrap().remove("core");
    let cases = directory.join("case.json");
    std::fs::write(&cases, serde_json::to_vec(&case).unwrap()).unwrap();
    let out = directory.join("rust-host.jsonl");
    std::fs::write(&out, "an earlier run's observations\n").unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .arg("--out")
        .arg(&out)
        .arg(&cases)
        .output()
        .unwrap();
    let left = out.exists();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("runner failure"));
    assert!(!left, "the earlier run's output survived");
}

#[test]
fn the_host_records_the_inputs_it_encoded() {
    let mut loaded = case::load(&[cases_dir()]).unwrap().remove(0).case;
    // A profile spelled with an explicit empty capabilities map: the host
    // encodes it without one, and records what it encoded.
    let spec = loaded
        .core
        .as_mut()
        .unwrap()
        .validate_query
        .as_mut()
        .unwrap();
    spec.profile["capabilities"] = serde_json::json!({});
    let observed = host::run(&loaded, &bases()).unwrap();
    assert!(observed.inputs[0]["profile"].get("capabilities").is_none());
}

#[test]
fn a_case_without_a_core_operation_is_this_hosts_failure() {
    let mut case = case::load(&[cases_dir()]).unwrap().remove(0).case;
    case.core = None;
    assert!(host::run(&case, &bases()).is_err());
}

/// `contract/tests/index_literals.json`, which the Python loader's tests
/// read too.
fn index_literals() -> serde_json::Value {
    let path = cases_dir().join("../tests/index_literals.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The accepted case with `literal` as its one index, written as is.
fn with_index(name: &str, literal: &str) -> Result<case::Index, String> {
    let mut case = accepted();
    case["input"]["materialize"]["indices"] = serde_json::json!(["@index@"]);
    let text = serde_json::to_string(&case)
        .unwrap()
        .replace("\"@index@\"", literal);
    let directory = scratch(name);
    let path = directory.join("case.json");
    std::fs::write(&path, text).unwrap();
    let loaded = case::load(&[path]);
    std::fs::remove_dir_all(&directory).unwrap();
    loaded
        .map(|mut loaded| loaded.remove(0).case.input.materialize.indices.remove(0))
        .map_err(|error| error.to_string())
}

#[test]
fn an_index_is_an_integer_literal_of_any_size_or_text() {
    let literals = index_literals();
    for literal in literals["accepted"].as_array().unwrap() {
        let literal = literal.as_str().unwrap();
        let index = with_index("index-accepted", literal).unwrap();
        let expected = match serde_json::from_str::<serde_json::Value>(literal) {
            Ok(serde_json::Value::String(text)) => case::Index::Text(text),
            _ => case::Index::Integer(literal.to_string()),
        };
        assert_eq!(index, expected, "{literal}");
    }
    for literal in literals["refused"].as_array().unwrap() {
        let literal = literal.as_str().unwrap();
        assert!(with_index("index-refused", literal).is_err(), "{literal}");
    }
}

/// T2.1t: the sampling options are kept as the text the case wrote, any
/// JSON value: an integer past 2**64, a 401-digit number, nested in a list
/// or an object too, `null`, a boolean and text, so no value the reference
/// takes or refuses passes through a double or fails to load.
#[test]
fn the_sampling_options_are_kept_exactly_as_written() {
    let wide = format!("1{}", "0".repeat(400));
    let given = [
        ("random_seed", "18446744073709551617".to_owned()),
        ("num_hops", wide.clone()),
        ("num_neighbors", format!("[{wide}, -1, 2.50]")),
        ("max_pq_iterations", "1e400".to_owned()),
        ("use_prediction_time", "\"yes\"".to_owned()),
        ("return_embeddings", "true".to_owned()),
        (
            "exclude_cols_dict",
            "{\"users\": [\"age\"], \"n\": 18446744073709551617}".to_owned(),
        ),
    ];
    let mut case = accepted();
    for (name, _) in &given {
        case["input"]["materialize"][*name] = serde_json::json!(format!("<{name}>"));
    }
    let mut text = serde_json::to_string(&case).unwrap();
    for (name, written) in &given {
        text = text.replace(&format!("\"<{name}>\""), written);
    }
    let directory = scratch("sampling-options");
    let path = directory.join("case.json");
    std::fs::write(&path, &text).unwrap();
    let loaded = case::load(&[path]).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    let call = &loaded[0].case.input.materialize;
    let read = [
        &call.random_seed,
        &call.num_hops,
        &call.num_neighbors,
        &call.max_pq_iterations,
        &call.use_prediction_time,
        &call.return_embeddings,
        &call.exclude_cols_dict,
    ];
    for ((name, written), held) in given.iter().zip(read) {
        assert_eq!(
            held.as_ref().map(|w| w.0.as_str()),
            Some(written.as_str()),
            "{name}"
        );
    }
    // null is kept as itself, the public call's None; absent is none.
    let mut case = accepted();
    case["input"]["materialize"]["random_seed"] = serde_json::Value::Null;
    let directory = scratch("sampling-options-null");
    let path = directory.join("case.json");
    std::fs::write(&path, serde_json::to_string(&case).unwrap()).unwrap();
    let loaded = case::load(&[path]).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    let call = &loaded[0].case.input.materialize;
    assert_eq!(
        call.random_seed.as_ref().map(|w| w.0.as_str()),
        Some("null")
    );
    assert_eq!(call.num_hops, None);
}
