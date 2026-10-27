// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.17: the core picks every refusal's and warning's sentence from one
//! closed list, and each sentence has one shared template (invariant 8).
//! `wording::template` matches every `Sentence` with no wildcard, so a
//! sentence with no template does not compile; these tests render each
//! template once, as a skeleton with every placeholder named, and hold the
//! table's data together: its advice, its groups, its host-owned
//! sentences, and the codes and rules each sentence gives.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::wording::{self, Advice, Group, Owner};
use sdm_contract::{Code, CoreError, Detail, ErrorCode, Sentence, WarningCode};

/// One placeholder of a template: the name before any `!` or `:`, its
/// conversion and its spec.
#[derive(Debug)]
struct Placeholder {
    name: String,
    conversion: Option<char>,
    spec: Option<String>,
}

/// The conversions the placeholder grammar names (`wording` module docs).
const CONVERSIONS: &[char] = &[
    'r', 'l', 'a', 'j', 'c', 'R', 't', 'T', 'G', 'S', 'N', 'I', 'p', 'd',
];
/// The specs a field may carry: a count with its thousands grouped, a
/// database by the names the table gives it, and a list paired with
/// another field's (`to(<field>)`).
const SPECS: &[&str] = &[",", "place", "backend"];

/// Whether `spec` is one a field may carry.
fn known_spec(spec: &str) -> bool {
    SPECS.contains(&spec)
        || spec
            .strip_prefix("to(")
            .and_then(|rest| rest.strip_suffix(')'))
            .is_some_and(|field| !field.is_empty())
}

/// The placeholders of `text`, `{{` and `}}` read as braces, or why it
/// does not parse.
fn placeholders(text: &str) -> Result<Vec<Placeholder>, String> {
    let mut found = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
            }
            '}' => return Err(format!("a lone }} in {text:?}")),
            '{' => {
                let mut inside = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some('{') => return Err(format!("a nested {{ in {text:?}")),
                        Some(c) => inside.push(c),
                        None => return Err(format!("an unclosed {{ in {text:?}")),
                    }
                }
                let (head, spec) = match inside.split_once(':') {
                    Some((head, spec)) => (head.to_owned(), Some(spec.to_owned())),
                    None => (inside.clone(), None),
                };
                let (name, conversion) = match head.split_once('!') {
                    Some((name, conversion)) => {
                        let mut it = conversion.chars();
                        let (Some(c), None) = (it.next(), it.next()) else {
                            return Err(format!("conversion {conversion:?} in {text:?}"));
                        };
                        (name.to_owned(), Some(c))
                    }
                    None => (head, None),
                };
                if name.is_empty() {
                    return Err(format!("an empty placeholder in {text:?}"));
                }
                found.push(Placeholder {
                    name,
                    conversion,
                    spec,
                });
            }
            _ => {}
        }
    }
    Ok(found)
}

/// An advice placeholder's spec, `id(argument, ...)`, as its id and its
/// arguments.
fn advice_call(spec: &str) -> Option<(&str, Vec<&str>)> {
    let (id, rest) = spec.split_once('(')?;
    let arguments = rest.strip_suffix(')')?;
    let arguments = if arguments.trim().is_empty() {
        Vec::new()
    } else {
        arguments.split(',').map(str::trim).collect()
    };
    Some((id, arguments))
}

/// `text` with every placeholder replaced by its name in angle brackets and
/// every doubled brace by one: what a host shows, its values unknown.
fn skeleton(text: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut inside = String::new();
                for c in chars.by_ref() {
                    if c == '}' {
                        break;
                    }
                    inside.push(c);
                }
                out.push('<');
                out.push_str(&inside);
                out.push('>');
            }
            c => out.push(c),
        }
    }
    if out.trim().is_empty() {
        return Err("an empty sentence".to_owned());
    }
    Ok(out)
}

#[test]
fn every_sentence_has_a_template_that_renders() {
    let advice: BTreeMap<&str, Advice> = Advice::ALL.iter().map(|a| (a.as_str(), *a)).collect();
    assert!(Sentence::ALL.len() >= 150, "{}", Sentence::ALL.len());
    for &sentence in Sentence::ALL {
        let template = wording::template(sentence);
        let found = placeholders(template.text)
            .unwrap_or_else(|why| panic!("{}: {why}", sentence.as_str()));
        for placeholder in &found {
            let at = sentence.as_str();
            if placeholder.name == "advice" {
                // `{advice:<id>(<field>, ...)}`: an advice the closed list
                // holds, given as many fields as it takes.
                assert!(placeholder.conversion.is_none(), "{at}");
                let spec = placeholder.spec.as_deref().unwrap_or("");
                let (id, arguments) =
                    advice_call(spec).unwrap_or_else(|| panic!("{at}: advice {spec:?}"));
                let named = advice
                    .get(id)
                    .unwrap_or_else(|| panic!("{at}: no advice {id:?}"));
                assert_eq!(arguments.len(), named.arguments().len(), "{at}: {id}");
                assert!(arguments.iter().all(|a| !a.is_empty()), "{at}: {id}");
                continue;
            }
            if let Some(conversion) = placeholder.conversion {
                assert!(CONVERSIONS.contains(&conversion), "{at}: !{conversion}");
            }
            if let Some(spec) = &placeholder.spec {
                assert!(known_spec(spec), "{at}: :{spec}");
            }
            let base = placeholder
                .name
                .split(['.', '['])
                .next()
                .unwrap_or_default();
            assert!(
                base.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "{at}: {}",
                placeholder.name
            );
            // A value the host holds is named under `host.`.
            if base == "host" {
                assert!(placeholder.name.starts_with("host."), "{at}");
            }
        }
        let shown =
            skeleton(template.text).unwrap_or_else(|why| panic!("{}: {why}", sentence.as_str()));
        assert!(!shown.contains("<>"), "{}: {shown}", sentence.as_str());
    }
}

#[test]
fn every_sentence_is_named_once_and_reads_back() {
    let names: BTreeSet<&str> = Sentence::ALL.iter().map(|s| s.as_str()).collect();
    assert_eq!(names.len(), Sentence::ALL.len());
    for &sentence in Sentence::ALL {
        let text = serde_json::to_string(&sentence).unwrap();
        assert_eq!(text, format!("\"{}\"", sentence.as_str()));
        let read: Sentence = serde_json::from_str(&text).unwrap();
        assert_eq!(read, sentence);
    }
}

#[test]
fn every_code_has_its_sentences_and_every_sentence_one_code() {
    // Each error code the core gives and each warning code is said by at
    // least one sentence; the public codes are the ones they were.
    for code in ErrorCode::ALL {
        assert!(
            Sentence::ALL.iter().any(|s| s.code() == Code::Error(code)),
            "{code}"
        );
    }
    for &code in WarningCode::ALL {
        assert!(
            Sentence::ALL
                .iter()
                .any(|s| s.code() == Code::Warning(code)),
            "{}",
            code.as_str()
        );
        let text = serde_json::to_string(&code).unwrap();
        assert_eq!(text, format!("\"{}\"", code.as_str()));
    }
    let warnings: BTreeSet<&str> = WarningCode::ALL.iter().map(|c| c.as_str()).collect();
    for code in [
        "TIME_COLUMNS_GUESSED",
        "COLUMNS_LEFT_OUT",
        "KEY_REPEATS",
        "CANDIDATES_UNSCORED",
        "FLOAT32_OVERFLOW",
        "TWO_WAYS_TO_A_TABLE",
    ] {
        assert!(warnings.contains(code), "{code}");
    }
}

#[test]
fn a_refusal_carries_its_sentence_and_the_sentence_s_rule() {
    let error = CoreError::new(Sentence::PrimaryKeyAsForeignKey, "m");
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert_eq!(error.sentence, Sentence::PrimaryKeyAsForeignKey);
    assert!(error.holds_to_table());
    // The rule a sentence names is the public detail callers branch on.
    let keyed = CoreError::new(Sentence::KeyNotNamesGiven, "m");
    assert_eq!(keyed.fields["rule"], Detail::text("key_not_names"));
    assert_eq!(Sentence::KeyNotNames.rule(), Some("key_not_names"));
    // A sentence of another code than the error's does not cross.
    let mut crossed = CoreError::new(Sentence::PrimaryKeyAsForeignKey, "m");
    crossed.sentence = Sentence::RunBusy;
    assert!(!crossed.holds_to_table());
    // An error travels with its sentence; one without it is not the
    // contract's.
    let text = serde_json::to_value(&error).unwrap();
    assert_eq!(text["sentence"], "primary_key_as_foreign_key");
    let mut bare = text.clone();
    bare.as_object_mut().unwrap().remove("sentence");
    assert!(serde_json::from_value::<CoreError>(bare).is_err());
}

#[test]
fn the_primary_key_named_as_a_foreign_key_is_said_as_the_old_client_says_it() {
    // `Graph._edge_error` in the old client's engine (T3.16 review r4,
    // P2-2): the table named, as the core's refusal names it.
    assert_eq!(
        wording::template(Sentence::PrimaryKeyAsForeignKey).text,
        "Cannot treat the primary key of table '{source_table}' as a foreign key. Remove \
         either the primary key or the link before proceeding."
    );
}

#[test]
fn every_advice_is_used_and_named_once() {
    let mut used = BTreeSet::new();
    for &sentence in Sentence::ALL {
        for placeholder in placeholders(wording::template(sentence).text).unwrap() {
            if placeholder.name == "advice" {
                let (id, _) = advice_call(placeholder.spec.as_deref().unwrap()).unwrap();
                used.insert(id.to_owned());
            }
        }
    }
    let named: BTreeSet<String> = Advice::ALL.iter().map(|a| a.as_str().to_owned()).collect();
    assert_eq!(named.len(), Advice::ALL.len());
    assert_eq!(used, named);
    for advice in ["skip_key_check", "clear_time_column", "name_time_column"] {
        assert!(named.contains(advice), "{advice}");
    }
}

#[test]
fn groups_are_data_over_one_code_each() {
    let mut seen = BTreeSet::new();
    for &sentence in Sentence::ALL {
        let Some(group) = wording::template(sentence).group else {
            continue;
        };
        seen.insert(group);
        // Only warnings group, and each group is one code's.
        let Code::Warning(code) = sentence.code() else {
            panic!("{} groups and is no warning", sentence.as_str());
        };
        assert_eq!(code, wording::grouping(group).code, "{}", sentence.as_str());
    }
    let all: BTreeSet<Group> = Group::ALL.iter().copied().collect();
    assert_eq!(seen, all);
    let time = wording::grouping(Group::TimeColumnsGuessed);
    assert!(time.agree.is_empty());
    assert_eq!(
        time.opening,
        "Some time columns were inferred, or refused, on a guess about what a column holds:\n- "
    );
    assert_eq!(time.between, "\n- ");
    let left_out = wording::grouping(Group::ColumnsLeftOut);
    assert_eq!(left_out.agree, &["table"]);
    for group in Group::ALL {
        let grouping = wording::grouping(*group);
        for text in [grouping.opening, grouping.between, grouping.closing] {
            for placeholder in placeholders(text).unwrap() {
                // Or a value only the host holds, the same for every
                // warning of the call (T3.27: the semantic model read in
                // part, as the host names it).
                assert!(
                    grouping.agree.contains(&placeholder.name.as_str())
                        || placeholder.name.starts_with("host."),
                    "a group's words name only the fields its warnings agree on"
                );
            }
        }
    }
    // Both kinds of time-column notice group together.
    assert_eq!(
        wording::template(Sentence::TimeColumnsTakenEarliest).group,
        wording::template(Sentence::TimeColumnsRefusedEmpty).group
    );
}

#[test]
fn a_sentence_that_relays_a_host_library_is_marked_host_owned() {
    let owned: BTreeMap<&str, Owner> = Sentence::ALL
        .iter()
        .map(|s| (s.as_str(), wording::template(*s).owner))
        .filter(|(_, owner)| *owner != Owner::Shared)
        .collect();
    assert_eq!(
        owned,
        BTreeMap::from([
            ("dates_read_day_first", Owner::HostDates),
            ("date_literal_day_first", Owner::HostDates),
            ("float32_overflow", Owner::HostCast),
            ("inference_option_not_finite", Owner::HostCast),
            (
                "max_pq_iterations_not_a_whole_number",
                Owner::HostComparison
            ),
        ])
    );
    // A host-owned sentence still has a neutral shared one.
    for name in owned.keys() {
        let sentence = Sentence::ALL.iter().find(|s| s.as_str() == *name).unwrap();
        assert!(!wording::template(*sentence).text.is_empty());
    }
}

#[test]
fn the_sentences_the_hosts_choose_between_today_are_the_core_s() {
    // Each pair a host told apart from fields is two sentences of the core.
    for (one, other) in [
        (
            Sentence::TimeColumnsTakenEarliest,
            Sentence::TimeColumnsRefusedEmpty,
        ),
        (Sentence::KeyNotNames, Sentence::KeyNotNamesGiven),
        (Sentence::MostEntities, Sentence::MostEntitiesParts),
        (Sentence::BatchedPartForecast, Sentence::BatchedPartRanking),
        (
            Sentence::CandidatesUnscoredOne,
            Sentence::CandidatesUnscoredSeveral,
        ),
    ] {
        assert_ne!(
            wording::template(one).text,
            wording::template(other).text,
            "{} and {}",
            one.as_str(),
            other.as_str()
        );
        assert_eq!(one.code(), other.code());
    }
    // The table a host reads is the whole of it, as JSON.
    let table: serde_json::Value = serde_json::from_str(&wording::table_json()).unwrap();
    assert_eq!(
        table["sentences"].as_array().unwrap().len(),
        Sentence::ALL.len()
    );
    assert_eq!(table["advice"].as_array().unwrap().len(), Advice::ALL.len());
    assert_eq!(table["groups"].as_array().unwrap().len(), Group::ALL.len());
}

#[test]
fn the_part_a_refusal_happened_in_is_a_sentence_of_the_core_s() {
    // A forecast's step, with the remedy where the step's window outran
    // the data, and a ranking's candidate: sentences said after an error's
    // own, of any code (T3.17 review r2).
    let parts: BTreeSet<&str> = Sentence::ALL
        .iter()
        .filter(|s| s.code() == Code::Part)
        .map(|s| s.as_str())
        .collect();
    assert_eq!(
        parts,
        BTreeSet::from([
            "part_forecast_step",
            "part_forecast_step_outran",
            "part_ranking_candidate",
        ])
    );
    let error = CoreError::new(Sentence::AnchorBeforeData, "m").in_part(Sentence::PartForecastStep);
    assert_eq!(error.part, Some(Sentence::PartForecastStep));
    assert!(error.holds_to_table());
    let mut crossed = error.clone();
    crossed.part = Some(Sentence::Internal);
    assert!(!crossed.holds_to_table());
}

#[test]
fn frame_validity_requirements_keep_one_shared_sentence_and_static_words() {
    let template = wording::template(Sentence::FrameNoValidRows).text;
    assert_eq!(
        template,
        "Found no valid rows in table '{table}'. Please make sure that there exists valid non-N/A {requirements} in this table."
    );
    assert_eq!(
        template
            .replace("{table}", "events")
            .replace("{requirements}", "primary keys"),
        "Found no valid rows in table 'events'. Please make sure that there exists valid non-N/A primary keys in this table."
    );
    assert_eq!(
        template
            .replace("{table}", "events")
            .replace("{requirements}", "primary key and time column pairs"),
        "Found no valid rows in table 'events'. Please make sure that there exists valid non-N/A primary key and time column pairs in this table."
    );
}

#[test]
fn frame_task_reference_error_keeps_optional_shared_sanitization_detail() {
    let template = wording::template(Sentence::FrameTaskReferences).text;
    assert_eq!(
        template,
        "Task references {rows} unresolved row(s) in table '{table}' after graph sanitization{detail}"
    );
    let base = template.replace("{rows}", "1").replace("{table}", "users");
    assert_eq!(
        base.replace("{detail}", ""),
        "Task references 1 unresolved row(s) in table 'users' after graph sanitization"
    );
    assert_eq!(
        base.replace(
            "{detail}",
            " [orders: 1 of 64 rows dropped (1 with a repeated primary key)]"
        ),
        "Task references 1 unresolved row(s) in table 'users' after graph sanitization [orders: 1 of 64 rows dropped (1 with a repeated primary key)]"
    );
}
