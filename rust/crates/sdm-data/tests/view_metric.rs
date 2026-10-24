// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's expression helpers (T3.5, `DB/metric_view.py` and
//! `RE/graph.py:424-430`) against `contract/view-step-vectors.json`: every
//! call the old client's own tests made of them, with its answer, and its
//! answers to the recorder's probes. Where the old client answered with a
//! sentence quoting a value, the decision is held here and the sentence in
//! the core (`sdm-core/tests/view_words.rs`).

use sdm_data::view_expr::metric::{
    self, DefaultKind, DimensionRef, JoinRef, KeyRef, MeasureInput, MeasureReason, Parameter,
    ParameterDefault, ParameterRead, Unread,
};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-step-vectors.json");

fn document() -> Json {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(document["format"], "sdm-view-step-vectors/1");
    document
}

fn texts(value: &Json) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

fn optional_text(value: &Json) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// Runs `check` over the vectors of `kind`, and fails listing each that
/// differs.
fn each(kind: &str, least: usize, check: impl Fn(&Json) -> Option<String>) {
    let document = document();
    let vectors = document[kind].as_array().unwrap();
    assert!(vectors.len() >= least, "{} {kind} vectors", vectors.len());
    let wrong: Vec<String> = vectors.iter().filter_map(check).collect();
    assert!(
        wrong.is_empty(),
        "{} of {} {kind} vectors differ:\n{}",
        wrong.len(),
        vectors.len(),
        wrong.join("\n")
    );
}

/// The text of a recorded name, or `None` for a value that is no text,
/// which no name parse reads.
fn name_text(value: &Json) -> Option<&str> {
    (value["t"] == "str").then(|| value["v"].as_str().unwrap())
}

#[test]
fn table_and_view_names_are_split_as_the_old_client_split_them() {
    for (kind, read) in [
        (
            "table_references",
            metric::table_reference as fn(&str) -> Option<Vec<String>>,
        ),
        ("view_names", metric::view_name),
    ] {
        each(kind, 10, |vector| {
            let parts = name_text(&vector["name"]).and_then(read);
            let expected = (!vector["parts"].is_null()).then(|| texts(&vector["parts"]));
            (parts != expected).then(|| format!("{}: {parts:?} where {expected:?}", vector["name"]))
        });
    }
}

fn key_ref(value: &Json) -> KeyRef {
    KeyRef {
        qualifier: optional_text(&value[0]),
        column: value[1].as_str().unwrap().to_owned(),
    }
}

#[test]
fn join_conditions_are_read_as_the_old_client_read_them() {
    each("join_conditions", 20, |vector| {
        let keys = name_text(&vector["condition"]).and_then(metric::join_condition);
        let expected = vector["keys"]
            .as_array()
            .map(|sides| [key_ref(&sides[0]), key_ref(&sides[1])]);
        (keys != expected).then(|| format!("{}: {keys:?} where {expected:?}", vector["condition"]))
    });
}

#[test]
fn names_and_qualifiers_are_read_as_the_old_client_read_them() {
    each("unquote", 50, |vector| {
        let out = metric::unquote(vector["name"].as_str().unwrap());
        (out != vector["out"].as_str().unwrap()).then(|| format!("{vector}: {out}"))
    });
    each("references_alias", 100, |vector| {
        let found =
            metric::references_alias(vector["expr"].as_str().unwrap(), &texts(&vector["path"]));
        (found != vector["found"].as_bool().unwrap()).then(|| format!("{vector}: {found}"))
    });
    each("strip_alias", 50, |vector| {
        let out = metric::strip_alias(vector["expr"].as_str().unwrap(), &texts(&vector["path"]));
        (out != vector["out"].as_str().unwrap()).then(|| format!("{vector}: {out}"))
    });
    each("derived_column", 20, |vector| {
        let column = metric::derived_column(vector["expr"].as_str().unwrap());
        (column != optional_text(&vector["column"])).then(|| format!("{vector}: {column:?}"))
    });
    each("renamed_column", 20, |vector| {
        let column = metric::renamed_column(vector["expr"].as_str().unwrap());
        (column != optional_text(&vector["column"])).then(|| format!("{vector}: {column:?}"))
    });
}

/// A recorded default as the data layer takes it: its kind and the text
/// Python's `str` gives it (the core writes that text, `view_words`).
fn default_of(value: &Json, text: &str) -> ParameterDefault {
    let kind = match value["t"].as_str().unwrap() {
        "bool" => DefaultKind::Bool(value["v"].as_bool().unwrap()),
        "int" => DefaultKind::Int(value["v"].as_str().unwrap().parse().unwrap()),
        "date" => DefaultKind::Date,
        "datetime" => DefaultKind::DateTime {
            iso: value["v"].as_str().unwrap().to_owned(),
        },
        _ => DefaultKind::Other,
    };
    ParameterDefault {
        kind,
        text: text.to_owned(),
    }
}

#[test]
fn a_parameter_s_default_reads_as_the_old_client_read_it() {
    each("default_text", 500, |vector| {
        let default = default_of(&vector["default"], vector["default_str"].as_str().unwrap());
        let read = metric::default_text(vector["data_type"].as_str().unwrap(), &default);
        match (&read, vector["text"].as_str()) {
            (Ok(text), Some(expected)) if text == expected => None,
            (Err(_), None) => None,
            _ => Some(format!("{vector}: {read:?}")),
        }
    });
}

/// The recorded parameters, as the data layer holds them once read.
fn parameters(value: &Json) -> Vec<(String, Parameter)> {
    value
        .as_object()
        .unwrap()
        .iter()
        .map(|(folded, entry)| {
            let read = match entry[1].as_str() {
                Some(literal) => Ok(ParameterRead {
                    literal: literal.to_owned(),
                    text: entry[2].as_str().unwrap().to_owned(),
                }),
                None => Err(Unread::NoDefault),
            };
            (
                folded.clone(),
                Parameter {
                    name: entry[0].as_str().unwrap().to_owned(),
                    read,
                },
            )
        })
        .collect()
}

#[test]
fn parameters_and_fields_are_substituted_as_the_old_client_substituted_them() {
    each("substitute_parameters", 100, |vector| {
        let read = metric::substitute_parameters(
            vector["expr"].as_str().unwrap(),
            &parameters(&vector["parameters"]),
        );
        let got = match &read {
            Ok((out, used)) => serde_json::json!({"out": out, "used": used}),
            Err(error) => serde_json::json!({"error": error.to_string()}),
        };
        let expected = match vector.get("error") {
            Some(error) => serde_json::json!({"error": error["message"]}),
            None => serde_json::json!({"out": vector["out"], "used": vector["used"]}),
        };
        (got != expected).then(|| format!("{}: {got} where {expected}", vector["expr"]))
    });
    each("substitute_fields", 100, |vector| {
        let fields: Vec<(String, Option<String>)> = vector["fields"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, expr)| (name.clone(), optional_text(expr)))
            .collect();
        let qualifiers: Vec<Vec<String>> = vector["qualifiers"]
            .as_array()
            .unwrap()
            .iter()
            .map(texts)
            .collect();
        let read =
            metric::substitute_fields(vector["expr"].as_str().unwrap(), &fields, &qualifiers);
        let got = match &read {
            Ok((out, refused)) => serde_json::json!({"out": out, "refused": refused}),
            Err(error) => serde_json::json!({"error": error.to_string()}),
        };
        let expected = match vector.get("error") {
            Some(error) => serde_json::json!({"error": error["message"]}),
            None => serde_json::json!({"out": vector["out"], "refused": vector["refused"]}),
        };
        (got != expected).then(|| format!("{}: {got} where {expected}", vector["expr"]))
    });
    each("star", 100, |vector| {
        let read = metric::reads_star(vector["expr"].as_str().unwrap());
        let got = match &read {
            Ok(star) => serde_json::json!({"star": star}),
            Err(error) => serde_json::json!({"error": error.to_string()}),
        };
        let expected = match vector.get("error") {
            Some(error) => serde_json::json!({"error": error["message"]}),
            None => serde_json::json!({"star": vector["star"]}),
        };
        (got != expected).then(|| format!("{}: {got} where {expected}", vector["expr"]))
    });
}

fn candidates(value: &Json) -> Vec<(Vec<String>, String)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| (texts(&pair[0]), pair[1].as_str().unwrap().to_owned()))
        .collect()
}

#[test]
fn a_filter_is_kept_or_refused_as_the_old_client_kept_or_refused_it() {
    each("check_filter", 50, |vector| {
        let filter = &vector["filter"];
        if filter["t"] == "null" {
            return None; // No filter, read by the caller.
        }
        let read = metric::check_filter(name_text(filter), &candidates(&vector["candidates"]));
        match (&read, vector["sql"].as_str()) {
            (Ok(sql), Some(expected)) if sql == expected => None,
            (Err(_), None) if vector.get("error").is_some() => None,
            _ => Some(format!("{vector}: {read:?}")),
        }
    });
}

#[test]
fn measures_are_read_as_the_old_client_read_them() {
    each("read_measures", 50, |vector| {
        let section = &vector["measures"];
        let measures: Vec<MeasureInput> = (section["t"] == "list")
            .then(|| section["v"].as_array().unwrap())
            .map(|items| {
                items
                    .iter()
                    .filter(|cfg| cfg["t"] == "map")
                    .filter_map(|cfg| {
                        let entry = |key: &str| {
                            cfg["v"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|pair| pair[0]["t"] == "str" && pair[0]["v"] == key)
                                .map(|pair| pair[1].clone())
                        };
                        let name = entry("name").filter(|n| n["t"] == "str")?;
                        Some(MeasureInput {
                            name: name["v"].as_str().unwrap().to_owned(),
                            expr: entry("expr")
                                .filter(|e| e["t"] == "str")
                                .map(|e| e["v"].as_str().unwrap().to_owned()),
                            window: entry("window").is_some(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let dimensions: Vec<(String, DimensionRef)> = vector["dimensions"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(folded, d)| {
                (
                    folded.clone(),
                    DimensionRef {
                        name: d[0].as_str().unwrap().to_owned(),
                        alias: d[1].as_str().unwrap().to_owned(),
                        expr: d[2].as_str().unwrap().to_owned(),
                    },
                )
            })
            .collect();
        let (read, refused) = metric::read_measures(
            &measures,
            &candidates(&vector["candidates"]),
            &texts(&vector["kept"]),
            &dimensions,
            &texts(&vector["refused"]),
            &texts(&vector["parameters"]),
        );
        let got: Vec<Json> = read
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "inputs": m.inputs.iter().map(|(a, c)| vec![a, c]).collect::<Vec<_>>(),
                    "aggregation": m.aggregation.as_ref().map(|(f, c)| serde_json::json!([f, c])),
                })
            })
            .collect();
        let names: Vec<String> = refused.iter().map(|r| r.name.clone()).collect();
        let expected_names: Vec<String> = texts(&vector["messages"])
            .iter()
            .map(|m| m.split('\'').nth(1).unwrap().to_owned())
            .collect();
        let named_refused = refused.iter().all(|r| match &r.reason {
            MeasureReason::NamesRefused(_) | MeasureReason::Unreadable(_) => true,
        });
        (Json::Array(got.clone()) != vector["read"] || names != expected_names || !named_refused)
            .then(|| format!("{}: {got:?} {names:?}", vector["measures"]))
    });
}

#[test]
fn join_keys_resolve_as_the_old_client_resolved_them() {
    each("resolve_join_keys", 10, |vector| {
        let join = &vector["join"];
        let condition = &join["condition"];
        let join = JoinRef {
            alias: join["alias"].as_str().unwrap().to_owned(),
            parent_alias: join["parent_alias"].as_str().unwrap().to_owned(),
            path: texts(&join["path"]),
            condition: [key_ref(&condition[0]), key_ref(&condition[1])],
        };
        let keys = metric::resolve_join_keys(
            &join,
            &texts(&vector["child_columns"]),
            &texts(&vector["parent_columns"]),
        );
        let expected = vector.get("keys").map(|keys| {
            let keys = texts(keys);
            (keys[0].clone(), keys[1].clone(), keys[2].clone())
        });
        (keys != expected).then(|| format!("{vector}: {keys:?}"))
    });
}
