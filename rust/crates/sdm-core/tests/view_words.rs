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
use sdm_schema::view::metric::Parameter as ViewParameter;
use sdm_schema::view::model::{self, Edge, EdgeFacts, MeasureNote, TimeColumnNote};
use sdm_schema::view::yaml::{self, Date, Value};
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

/// A recorded value, the typed form the recorder writes, as YAML gives it.
fn value_of(json: &Json) -> Value {
    let inner = &json["v"];
    match json["t"].as_str().unwrap() {
        "null" => Value::Null,
        "bool" => Value::Bool(inner.as_bool().unwrap()),
        "int" => Value::Int(inner.as_str().unwrap().parse().unwrap()),
        "float" => Value::Float(inner.as_str().unwrap().parse().unwrap()),
        "str" => Value::Str(inner.as_str().unwrap().to_owned()),
        "date" => {
            let parts: Vec<&str> = inner.as_str().unwrap().split('-').collect();
            Value::Date(Date {
                year: parts[0].parse().unwrap(),
                month: parts[1].parse().unwrap(),
                day: parts[2].parse().unwrap(),
            })
        }
        "datetime" => {
            // Only the recorder's probes give one: written with `isoformat`.
            let text = inner.as_str().unwrap();
            let loaded = yaml::load(&format!("v: {}", text.replacen('T', " ", 1))).unwrap();
            match loaded {
                Value::Map(mut pairs) => match pairs.remove(0).1 {
                    moment @ Value::Timestamp(_) => moment,
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            }
        }
        "list" => Value::Seq(inner.as_array().unwrap().iter().map(value_of).collect()),
        "map" => Value::Map(
            inner
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (value_of(&pair[0]), value_of(&pair[1])))
                .collect(),
        ),
        other => panic!("{other}"),
    }
}

#[test]
fn a_yaml_value_is_quoted_as_the_old_client_quoted_it() {
    let document: Json = serde_json::from_str(YAML).unwrap();
    let mut wrong = Vec::new();
    for vector in document["vectors"].as_array().unwrap() {
        let Some(repr) = vector["repr"].as_str() else {
            continue;
        };
        let value = yaml::load(vector["yaml"].as_str().unwrap()).unwrap();
        if view_words::value_repr(&value) != repr {
            wrong.push(format!("{} where {repr}", view_words::value_repr(&value)));
        }
        if let Some(text) = vector["str"].as_str()
            && view_words::value_text(&value) != text
        {
            wrong.push(format!("{} where {text}", view_words::value_text(&value)));
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
        let default = value_of(&vector["default"]);
        let data_type = vector["data_type"].as_str().unwrap();
        let read = data::default_text(data_type, &view_words::parameter_default(&default));
        let got = match &read {
            Ok(text) => text.clone(),
            Err(refusal) => view_words::default_refusal_text(refusal, data_type, &default),
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
        let section = &vector["parameters"];
        let parameters: Vec<ViewParameter> = if section["t"] == "list" {
            section["v"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|cfg| match value_of(cfg) {
                    entry @ Value::Map(_) => {
                        let name = entry.get("name")?.as_str()?.to_owned();
                        Some(ViewParameter {
                            name,
                            data_type: entry.get("data_type").cloned().unwrap_or(Value::Null),
                            default: entry.get("default").cloned().unwrap_or(Value::Null),
                        })
                    }
                    _ => None,
                })
                .collect()
        } else {
            vec![]
        };
        let described: Vec<(String, String)> = vector["types"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        let inputs: Vec<ParameterInput> = view_words::parameter_inputs(&parameters);
        let read = data::read_parameters(&inputs, &described);
        let got: serde_json::Map<String, Json> = read
            .iter()
            .map(|(folded, parameter)| {
                let default = parameters
                    .iter()
                    .rev()
                    .find(|p| p.name.to_lowercase() == *folded)
                    .map(|p| &p.default)
                    .unwrap();
                let literal = parameter.read.as_ref().ok().map(|r| r.literal.clone());
                (
                    folded.clone(),
                    serde_json::json!([
                        parameter.name,
                        literal,
                        view_words::parameter_text(parameter, default)
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
        let filter = value_of(&vector["filter"]);
        let candidates: Vec<(Vec<String>, String)> = vector["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (texts(&pair[0]), pair[1].as_str().unwrap().to_owned()))
            .collect();
        let refusal: FilterRefusal = data::check_filter(filter.as_str(), &candidates).unwrap_err();
        let text = view_words::filter_refusal_text(&filter, &refusal);
        (text != error["message"].as_str().unwrap())
            .then(|| format!("{text} where {}", error["message"]))
    });
    each(STEPS, "read_measures", 50, |vector| {
        let messages = texts(&vector["messages"]);
        if messages.is_empty() {
            return None;
        }
        let section = value_of(&vector["measures"]);
        let measures: Vec<data::MeasureInput> = match &section {
            Value::Seq(items) => items
                .iter()
                .filter_map(|cfg| {
                    Some(data::MeasureInput {
                        name: cfg.get("name")?.as_str()?.to_owned(),
                        expr: cfg.get("expr").and_then(Value::as_str).map(str::to_owned),
                        window: cfg.get("window").is_some(),
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
        let example = (!count["example"].is_null()).then(|| value_of(&count["example"]));
        let text = view_words::key_reason_text(
            &reason,
            vector["table"].as_str().unwrap(),
            vector["key"].as_str().unwrap(),
            example.as_ref(),
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
            Some(&Value::Str("a".into())),
        ),
        "its key 'id' is not unique in table 'users': 1,200 of 34,567 rows repeat a value, e.g. \
         'a'. A graph table needs one row per key, so point the join at a table or view with one \
         row per 'id'"
    );
}
