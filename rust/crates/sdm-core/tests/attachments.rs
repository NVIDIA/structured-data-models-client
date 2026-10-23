// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON attachments dispatch decodes, held to the contract's locked
//! classification of them (`$defs/Attachments`, T1.22; subagent P3-3 of
//! its review): each `take_json` in `src/dispatch.rs` decodes the type the
//! lock names for that attachment, and every attachment the lock names on
//! the way in is decoded there, so a change to either fails here. Seeded:
//! a retyped decode and a dropped one.
//!
//! A decode is read as a statement, however rustfmt breaks it over lines,
//! and from whichever value names the attachment: the operation's request,
//! the request state, a forecast's later step or the sampler state (M1,
//! T2.1b: the two-line decode of `SamplerState.memo` and the decode of
//! `ForecastStep.query` were not read before).

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads dispatch's source, whose decodes it checks"
)]

use serde_json::Value;

/// The attachments the core reads from a `start`, by the lock's names.
const READ: [&str; 12] = [
    "ValidateQuery.query",
    "ValidateQuery.schema",
    "DescribeSource.source",
    "EditSchema.schema",
    "EditSchema.edit",
    "Materialize.schema",
    "Materialize.query",
    "Materialize.candidates",
    "RequestState.query",
    "ForecastStep.query",
    "RankingPart.query",
    "SamplerState.memo",
];

/// The values dispatch reads an attachment's id from, other than an
/// operation's `request`, and the contract type each is.
const HOLDERS: [(&str, &str); 4] = [
    ("request.state.", "RequestState"),
    ("step.", "ForecastStep"),
    ("part.", "RankingPart"),
    ("state.", "SamplerState"),
];

/// Each `take_json` of `source` (dispatch's text) as the lock's name of the
/// attachment it reads and the Rust type it decodes: the operation is the
/// arm it is in (`Operation::<name>(request)`), the field what it takes
/// from `request`, and `request.state` is a `RequestState`.
fn decodes(source: &str) -> Result<Vec<(String, String)>, String> {
    let mut found = Vec::new();
    let mut arm: Option<&str> = None;
    // Each statement on one line: a line ending in `=` continues on the
    // next, as rustfmt breaks a long `let`.
    let joined = source
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("=\n", "= ");
    for line in joined.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Operation::")
            && let Some((name, _)) = rest.split_once("(request)")
        {
            arm = Some(name);
        }
        // An attachment a message may leave out is read by `take_json_if`
        // into an `Option` of the type the lock names.
        let (left, right, optional) =
            if let Some((l, r)) = line.split_once(" = take_json(&mut attachments, ") {
                (l, r, false)
            } else if let Some((l, r)) = line.split_once(" = take_json_if(&mut attachments, ") {
                (l, r, true)
            } else {
                continue;
            };
        let decoded = left
            .rsplit_once(": ")
            .map(|(_, ty)| ty.to_owned())
            .ok_or_else(|| format!("a take_json with no type: {line}"))?;
        let decoded = if optional {
            decoded
                .strip_prefix("Option<")
                .and_then(|ty| ty.strip_suffix('>'))
                .map(str::to_owned)
                .ok_or_else(|| format!("a take_json_if not into an Option: {line}"))?
        } else {
            decoded
        };
        let field = right
            .split_once(')')
            .map(|(field, _)| field)
            .ok_or_else(|| format!("a take_json not read: {line}"))?;
        let held = HOLDERS
            .iter()
            .find_map(|(prefix, ty)| field.strip_prefix(prefix).map(|part| (*ty, part)));
        let name = match (held, field.strip_prefix("request.")) {
            (Some((ty, part)), _) => format!("{ty}.{part}"),
            (None, Some(part)) => format!(
                "{}.{part}",
                arm.ok_or_else(|| format!("a take_json outside an arm: {line}"))?
            ),
            (None, None) => return Err(format!("a take_json of no known value: {line}")),
        };
        found.push((name, decoded));
    }
    Ok(found)
}

/// The Rust type the lock gives the attachment `name`.
fn locked(attachments: &Value, name: &str) -> Option<String> {
    let property = attachments.get(name)?;
    match (property.get("$ref"), property.get("type")) {
        (Some(Value::String(target)), _) => target.rsplit('/').next().map(str::to_owned),
        (None, Some(Value::String(kind))) if kind == "string" => Some("String".into()),
        // A list of one type, as `Materialize.candidates` (T2.8).
        (None, Some(Value::String(kind))) if kind == "array" => {
            let item = property.get("items")?.get("$ref")?.as_str()?;
            item.rsplit('/').next().map(|ty| format!("Vec<{ty}>"))
        }
        _ => None,
    }
}

/// What differs between dispatch's decodes (`source`) and the lock.
fn mismatches(source: &str) -> Vec<String> {
    let schema: Value = serde_json::from_str(&sdm_contract::schema_json()).unwrap();
    let attachments = &schema["$defs"]["Attachments"]["properties"];
    let found = match decodes(source) {
        Ok(found) => found,
        Err(problem) => return vec![problem],
    };
    let mut problems = Vec::new();
    for (name, decoded) in &found {
        match locked(attachments, name) {
            Some(ty) if ty == *decoded => {}
            Some(ty) => problems.push(format!("{name}: dispatch decodes {decoded}, the lock {ty}")),
            None => problems.push(format!("{name}: decoded by dispatch, not in the lock")),
        }
    }
    for name in READ {
        if !found.iter().any(|(n, _)| n == name) {
            problems.push(format!("{name}: in the lock, not decoded by dispatch"));
        }
    }
    problems
}

fn dispatch() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dispatch.rs")).unwrap()
}

#[test]
fn dispatch_decodes_each_attachment_as_the_lock_types_it() {
    let source = dispatch();
    assert_eq!(decodes(&source).unwrap().len(), READ.len());
    assert_eq!(mismatches(&source), Vec::<String>::new());
}

#[test]
fn a_retyped_or_dropped_decode_fails() {
    let source = dispatch();
    let retyped = source.replace(
        "let source: SourceSpec =",
        "let source: SourceDescription =",
    );
    assert_ne!(retyped, source);
    assert_eq!(
        mismatches(&retyped),
        ["DescribeSource.source: dispatch decodes SourceDescription, the lock SourceSpec"]
    );
    let dropped = source.replace(
        "let query: String = take_json(&mut attachments, request.state.query)?;",
        "",
    );
    assert_ne!(dropped, source);
    assert_eq!(
        mismatches(&dropped),
        ["RequestState.query: in the lock, not decoded by dispatch"]
    );
    // M1: the decodes read before T2.1b missed, the two-line one and the
    // one from a forecast's step, are each held too.
    let retyped = source.replace("let memo: SamplerMemo =", "let memo: SamplerState =");
    assert_ne!(retyped, source);
    assert_eq!(
        mismatches(&retyped),
        ["SamplerState.memo: dispatch decodes SamplerState, the lock SamplerMemo"]
    );
    let dropped = source.replace(
        "let query: String = take_json(&mut attachments, step.query)?;",
        "",
    );
    assert_ne!(dropped, source);
    assert_eq!(
        mismatches(&dropped),
        ["ForecastStep.query: in the lock, not decoded by dispatch"]
    );
}
