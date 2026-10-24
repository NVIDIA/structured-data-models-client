// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The words of a view's conversion (T3.5): the data and schema layers
//! answer with typed facts, and the core words them, quoting a value as
//! Python's `repr` and `str` write it through `sdm-pql`, the one crate that
//! writes Python's text. Each text is held to the old client's, from
//! `contract/view-expr-vectors.json`, `view-yaml-vectors.json` and
//! `view-step-vectors.json`. T3.27 turns each into a `Sentence`.

use sdm_core::view_words;
use sdm_data::view_expr::metric::{self as data, FilterRefusal, ParameterInput};
use sdm_data::view_expr::{self, ExprDialect, ViewKind};
use sdm_schema::view::metric;
use sdm_schema::view::model::{self, Edge, EdgeFacts, MeasureNote, TimeColumnNote};
use sdm_schema::view::yaml::{self, Document, NodeId, Value};
use serde_json::Value as Json;

const EXPR: &str = include_str!("../../../../contract/view-expr-vectors.json");
const YAML: &str = include_str!("../../../../contract/view-yaml-vectors.json");
const STEPS: &str = include_str!("../../../../contract/view-step-vectors.json");

fn texts(value: &Json) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

fn dialect(value: &Json) -> Option<ExprDialect> {
    match value.as_str() {
        None => None,
        Some("databricks") => Some(ExprDialect::Databricks),
        Some(_) => Some(ExprDialect::Snowflake),
    }
}

/// Runs `check` over the vectors of `kind` in `document`, and fails listing
/// each that differs.
fn each(document: &str, kind: &str, least: usize, check: impl Fn(&Json) -> Option<String>) {
    let document: Json = serde_json::from_str(document).unwrap();
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

/// A recorded value, the typed form the recorder writes, as YAML text
/// that reads back to it; `None` for one holding itself.
fn yaml_of(json: &Json) -> Option<String> {
    let inner = &json["v"];
    Some(match json["t"].as_str().unwrap() {
        "null" => "~".into(),
        "bool" => inner.as_bool().unwrap().to_string(),
        "int" => inner.as_str().unwrap().to_owned(),
        "float" => {
            let x: f64 = inner.as_str().unwrap().parse().unwrap();
            if x.is_nan() {
                ".nan".into()
            } else if x.is_infinite() {
                if x > 0.0 {
                    ".inf".into()
                } else {
                    "-.inf".into()
                }
            } else {
                let shortest = format!("{x:e}");
                let (mantissa, exponent) = shortest.split_once('e').unwrap();
                let mantissa = if mantissa.contains('.') {
                    mantissa.to_owned()
                } else {
                    format!("{mantissa}.0")
                };
                let exponent: i32 = exponent.parse().unwrap();
                format!("{mantissa}e{exponent:+}")
            }
        }
        "str" => serde_json::to_string(inner.as_str().unwrap()).unwrap(),
        "bytes" => format!("!!binary \"{}\"", inner.as_str().unwrap()),
        "date" | "datetime" => inner.as_str().unwrap().to_owned(),
        "list" => {
            let items = inner.as_array().unwrap();
            if !items.is_empty() && items.iter().all(|i| i["t"] == "tuple") {
                let pairs: Option<Vec<String>> = items
                    .iter()
                    .map(|i| {
                        Some(format!(
                            "{{{}: {}}}",
                            yaml_of(&i["v"][0])?,
                            yaml_of(&i["v"][1])?
                        ))
                    })
                    .collect();
                format!("!!pairs [{}]", pairs?.join(", "))
            } else {
                let items: Option<Vec<String>> = items.iter().map(yaml_of).collect();
                format!("[{}]", items?.join(", "))
            }
        }
        "set" => {
            let members: Option<Vec<String>> = inner
                .as_array()
                .unwrap()
                .iter()
                .map(|m| Some(format!("{}: ~", yaml_of(m)?)))
                .collect();
            format!("!!set {{{}}}", members?.join(", "))
        }
        "map" => {
            let pairs: Option<Vec<String>> = inner
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| Some(format!("{}: {}", yaml_of(&pair[0])?, yaml_of(&pair[1])?)))
                .collect();
            format!("{{{}}}", pairs?.join(", "))
        }
        _ => return None,
    })
}

/// A recorded value, loaded: its document and node.
fn loaded(json: &Json) -> Option<(Document, NodeId)> {
    let document = yaml::load(&format!("v: {}", yaml_of(json)?)).unwrap();
    let node = document.get(document.root(), "v").unwrap();
    Some((document, node))
}

#[test]
fn a_yaml_value_is_quoted_as_the_old_client_quoted_it() {
    let document: Json = serde_json::from_str(YAML).unwrap();
    let mut wrong = Vec::new();
    for vector in document["vectors"].as_array().unwrap() {
        let Some(repr) = vector["repr"].as_str() else {
            continue;
        };
        let Ok(document) = yaml::load(vector["yaml"].as_str().unwrap()) else {
            continue; // A registered scanner difference (P3J-6).
        };
        let root = document.root();
        if view_words::value_repr(&document, root) != repr {
            wrong.push(format!(
                "{} where {repr}",
                view_words::value_repr(&document, root)
            ));
        }
        if let Some(text) = vector["str"].as_str()
            && view_words::value_text(&document, root) != text
        {
            wrong.push(format!(
                "{} where {text}",
                view_words::value_text(&document, root)
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_expression_s_refusal_is_worded_as_the_old_client_worded_it() {
    each(EXPR, "unsafe", 400, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let reason = view_expr::unsafe_reason(expr, dialect(&vector["dialect"]))
            .map(|r| view_words::expr_reason(&r));
        (reason.as_deref() != vector["reason"].as_str())
            .then(|| format!("{expr:?}: {reason:?} where {}", vector["reason"]))
    });
    each(EXPR, "calls", 10_000, |vector| {
        let expr = vector["expr"].as_str().unwrap();
        let columns = texts(&vector["columns"]);
        let reason =
            view_expr::untrusted_call_reason(expr, dialect(&vector["dialect"]).unwrap(), &columns)
                .map(|r| view_words::expr_reason(&r));
        (reason.as_deref() != vector["reason"].as_str())
            .then(|| format!("{expr:?}: {reason:?} where {}", vector["reason"]))
    });
    each(EXPR, "expr_message", 30, |vector| {
        let view = match vector["view"].as_str().unwrap() {
            "metric view" => ViewKind::MetricView,
            _ => ViewKind::SemanticView,
        };
        let message = view_expr::expression_message(
            vector["column"].as_str().unwrap(),
            vector["expr"].as_str().unwrap(),
            view,
            dialect(&vector["dialect"]).unwrap(),
            vector["trust_expressions"].as_bool().unwrap(),
            &texts(&vector["columns"]),
            &view_words::quote,
        );
        (message.as_deref() != vector["message"].as_str()).then(|| format!("{vector}: {message:?}"))
    });
}

/// Rows 64 to 66 by name.
#[test]
fn unsafe_text_and_untrusted_calls_are_worded_by_name() {
    let reason = |expr: &str| {
        view_expr::unsafe_reason(expr, Some(ExprDialect::Databricks))
            .map(|r| view_words::expr_reason(&r))
    };
    assert_eq!(
        reason("x; DROP TABLE t").as_deref(),
        Some("it contains ';'")
    );
    assert_eq!(
        reason("trim(FROM users.country)").as_deref(),
        Some("it contains a sub-query")
    );
    assert_eq!(
        reason("é + 1").as_deref(),
        Some("it has 'é' outside quotes, which is not ASCII")
    );
    assert_eq!(
        reason("'a").as_deref(),
        Some("it leaves \"'\" unpaired, so which part of it is quoted text cannot be established")
    );
    assert_eq!(
        view_expr::expression_message(
            "column 'x'",
            "trim(FROM users.country)",
            ViewKind::MetricView,
            ExprDialect::Databricks,
            true,
            &["country"],
            &view_words::quote,
        )
        .as_deref(),
        Some(
            "Failed to add column 'x' since it contains a sub-query, and expressions taken \
             from the metric view are executed as SQL under your warehouse credentials"
        )
    );
    let call =
        view_expr::untrusted_call_reason("secret('s', 'k')", ExprDialect::Databricks, &["x"])
            .map(|r| view_words::expr_reason(&r));
    assert_eq!(
        call.as_deref(),
        Some("it calls 'secret', which is not a pure built-in SQL function")
    );
}

#[test]
fn a_parameter_is_read_and_worded_as_the_old_client_read_it() {
    each(STEPS, "default_text", 500, |vector| {
        let (document, default) = loaded(&vector["default"])?;
        let data_type = vector["data_type"].as_str().unwrap();
        let read = data::default_text(
            data_type,
            &view_words::parameter_default(&document, default),
        );
        let got = match &read {
            Ok(text) => text.clone(),
            Err(refusal) => {
                view_words::default_refusal_text(refusal, data_type, &document, default)
            }
        };
        let expected = vector["text"]
            .as_str()
            .or(vector["error"]["message"].as_str())
            .unwrap();
        (got != expected).then(|| {
            format!(
                "{data_type} {}: {got} where {expected}",
                vector["default_repr"]
            )
        })
    });
    each(STEPS, "read_parameters", 10, |vector| {
        let section = yaml_of(&vector["parameters"])?;
        let spec = metric::parse(&format!("source: c.s.t\nparameters: {section}\n")).unwrap();
        let described: Vec<(String, String)> = vector["types"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        let inputs: Vec<ParameterInput> = view_words::parameter_inputs(&spec);
        let read = data::read_parameters(&inputs, &described);
        let got: serde_json::Map<String, Json> = read
            .iter()
            .map(|(folded, parameter)| {
                let default = spec
                    .parameters
                    .iter()
                    .rev()
                    .find(|p| p.name.to_lowercase() == *folded)
                    .and_then(|p| p.default);
                let literal = parameter.read.as_ref().ok().map(|r| r.literal.clone());
                (
                    folded.clone(),
                    serde_json::json!([
                        parameter.name,
                        literal,
                        view_words::parameter_text(parameter, &spec.document, default)
                    ]),
                )
            })
            .collect();
        (Json::Object(got.clone()) != vector["read"])
            .then(|| format!("{}: {got:?}", vector["read"]))
    });
    each(STEPS, "parameter_note", 1, |vector| {
        let text = view_words::parameter_note_text(
            vector["parameter"].as_str().unwrap(),
            vector["default"].as_str().unwrap(),
        );
        (text != vector["text"].as_str().unwrap()).then_some(text)
    });
}

#[test]
fn a_filter_and_a_measure_are_refused_in_the_old_client_s_words() {
    each(STEPS, "check_filter", 50, |vector| {
        let error = vector.get("error")?;
        let (document, filter) = loaded(&vector["filter"])?;
        let candidates: Vec<(Vec<String>, String)> = vector["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (texts(&pair[0]), pair[1].as_str().unwrap().to_owned()))
            .collect();
        let refusal: FilterRefusal =
            data::check_filter(document.text(filter), &candidates).unwrap_err();
        let text = view_words::filter_refusal_text(&document, filter, &refusal);
        (text != error["message"].as_str().unwrap())
            .then(|| format!("{text} where {}", error["message"]))
    });
    each(STEPS, "read_measures", 50, |vector| {
        let messages = texts(&vector["messages"]);
        if messages.is_empty() {
            return None;
        }
        let (document, section) = loaded(&vector["measures"])?;
        let measures: Vec<data::MeasureInput> = match document.value(section) {
            Value::Seq(items) => items
                .iter()
                .filter_map(|&cfg| {
                    Some(data::MeasureInput {
                        name: document.text(document.get(cfg, "name")?)?.to_owned(),
                        expr: document
                            .get(cfg, "expr")
                            .and_then(|e| document.text(e))
                            .map(str::to_owned),
                        window: document.has(cfg, "window"),
                    })
                })
                .collect(),
            _ => vec![],
        };
        let dimensions: Vec<(String, data::DimensionRef)> = vector["dimensions"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(folded, d)| {
                (
                    folded.clone(),
                    data::DimensionRef {
                        name: d[0].as_str().unwrap().to_owned(),
                        alias: d[1].as_str().unwrap().to_owned(),
                        expr: d[2].as_str().unwrap().to_owned(),
                    },
                )
            })
            .collect();
        let candidates: Vec<(Vec<String>, String)> = vector["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (texts(&pair[0]), pair[1].as_str().unwrap().to_owned()))
            .collect();
        let (_, refused) = data::read_measures(
            &measures,
            &candidates,
            &texts(&vector["kept"]),
            &dimensions,
            &texts(&vector["refused"]),
            &texts(&vector["parameters"]),
        );
        let got: Vec<String> = refused
            .iter()
            .map(view_words::measure_refusal_text)
            .collect();
        (got != messages).then(|| format!("{got:?} where {messages:?}"))
    });
}

#[test]
fn the_graph_s_steps_are_worded_as_the_old_client_worded_them() {
    each(STEPS, "join_key_reason", 10, |vector| {
        let count = &vector["count"];
        let reason = model::join_key_reason(
            count["rows"].as_u64(),
            count["distinct"].as_u64(),
            count["repeats"].as_u64().unwrap(),
        )?;
        let text = view_words::key_reason_text(
            &reason,
            vector["table"].as_str().unwrap(),
            vector["key"].as_str().unwrap(),
            count["example_repr"].as_str(),
        );
        (Some(text.as_str()) != vector["reason"].as_str()).then_some(text)
    });
    each(STEPS, "skipped_join", 20, |vector| {
        let text = view_words::skipped_join_text(
            vector["alias"].as_str().unwrap(),
            vector["reason"].as_str().unwrap(),
            &texts(&vector["nested"]),
            &texts(&vector["dimensions"]),
        );
        (text != vector["text"].as_str().unwrap()).then_some(text)
    });
    each(STEPS, "measure_notes", 20, |vector| {
        let got: Vec<String> = vector["notes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                view_words::measure_note_text(&MeasureNote {
                    measure: n["measure"].as_str().unwrap().to_owned(),
                    table: n["table"].as_str().unwrap().to_owned(),
                    function: n["function"].as_str().unwrap().to_owned(),
                    column: n["column"].as_str().map(str::to_owned),
                })
            })
            .collect();
        let expected: Vec<String> = vector["notes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["text"].as_str().unwrap().to_owned())
            .collect();
        (got != expected).then(|| format!("{got:?}"))
    });
    each(STEPS, "time_column", 30, |vector| {
        let note = vector["note"].as_object()?;
        let text = view_words::time_note_text(&TimeColumnNote {
            table: note["table"].as_str().unwrap().to_owned(),
            column: note["column"].as_str().unwrap().to_owned(),
            instead_of: texts(&note["instead_of"]),
            source_column: note["source_column"].as_str().map(str::to_owned),
            added: note["added"].as_bool().unwrap(),
            window_order: note["window_order"].as_bool().unwrap(),
        });
        (text != note["text"].as_str().unwrap()).then_some(text)
    });
    each(STEPS, "edge_error", 5, |vector| {
        let expected = vector["reason"].as_str()?;
        let dtype = |name: &str| serde_json::from_value(Json::String(name.to_owned())).unwrap();
        let facts = EdgeFacts {
            fkey: vector["edge"][1].as_str().unwrap().to_owned(),
            fkey_dtype: vector["fkey_dtype"].as_str().map(dtype),
            target_key: vector["target_key"].as_array().map(|k| {
                (
                    k[0].as_str().unwrap().to_owned(),
                    dtype(k[1].as_str().unwrap()),
                )
            }),
            source_key: vector["source_key"].as_str().map(str::to_owned),
        };
        let edge = Edge {
            source: vector["edge"][0].as_str().unwrap().to_owned(),
            key: vector["edge"][1].as_str().unwrap().to_owned(),
            target: vector["edge"][2].as_str().unwrap().to_owned(),
        };
        let text = view_words::edge_reason_text(&edge, &model::edge_reason(&facts).unwrap());
        (text != expected).then_some(text)
    });
}

/// `_edge_error`'s and `_join_key_reason`'s texts by name, the vectors
/// holding few.
#[test]
fn each_edge_reason_and_key_reason_is_worded_by_name() {
    let edge = Edge {
        source: "orders".into(),
        key: "placed".into(),
        target: "users".into(),
    };
    let text = |reason| view_words::edge_reason_text(&edge, &reason);
    assert_eq!(
        text(model::EdgeReason::NoPrimaryKey),
        "Edge Edge(src_table='orders', fkey='placed', dst_table='users') is invalid since table \
         'users' does not have a primary key. Add either a primary key or remove the link before \
         proceeding."
    );
    assert_eq!(
        text(model::EdgeReason::NotNumberOrString {
            dtype: sdm_contract::ops::Dtype::Date
        }),
        "Edge(src_table='orders', fkey='placed', dst_table='users') is invalid as foreign key must \
         be a number or string (got 'date')"
    );
    assert_eq!(
        view_words::key_reason_text(
            &model::KeyReason::NotUnique {
                repeats: 1200,
                rows: 34567
            },
            "users",
            "id",
            Some("'a'"),
        ),
        "its key 'id' is not unique in table 'users': 1,200 of 34,567 rows repeat a value, e.g. \
         'a'. A graph table needs one row per key, so point the join at a table or view with one \
         row per 'id'"
    );
}

/// The long tokens' refusals (T3.5 fix 1), worded as the old client
/// worded them.
#[test]
fn a_long_token_s_refusal_is_worded_as_the_old_client_worded_it() {
    each(EXPR, "long_tokens", 15, |vector| {
        let r = &vector["recipe"];
        let part = |i: usize| r[i].as_str().unwrap();
        let called = format!(
            "{}{}{}",
            part(0),
            part(1).repeat(r[4].as_u64().unwrap() as usize),
            part(3)
        );
        let reason =
            view_expr::untrusted_call_reason(&called, ExprDialect::Databricks, &["x", "y", "z"])
                .map(|r| view_words::expr_reason(&r));
        (reason.as_deref() != vector["calls"]["databricks"].as_str())
            .then(|| format!("{r}: {reason:?}"))
    });
}
