// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A validated query read back from the `tree` [`ParsedQuery::to_json`]
//! writes (`json_serde.to_json_dict` of a `ValidatedPredictiveQuery`), so a
//! host that holds a resolved query can hand it to the core as it is.
//!
//! A node's kind is read from its fields, as the tree carries no type tags.
//! Every field the tree holds is kept as written: the windows, the types and
//! the locations. The one thing it leaves out, which column a `COUNT`
//! counts, is marked again as validation marks it.

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::{Map, Value};

use crate::ast::{Base, ConstValue, Kind, Location, Node, NodeDtype, NodeRef};
use crate::dor::{DateOffsetRange, DorRef};
use crate::pyint::PyInt;
use crate::query::{ParsedQuery, ValidatedQuery};
use crate::response::ValidationResponse;
use crate::types::{AggregationType, BoolOp, Dtype, Op, ProblemType, Stype, TimeUnit};

fn bad(message: impl Into<String>) -> String {
    format!("not a query tree: {}", message.into())
}

fn object<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| bad(format!("{what} is not an object")))
}

fn text<'a>(node: &'a Map<String, Value>, field: &str) -> Result<&'a str, String> {
    node.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("{field} is not text")))
}

fn optional_text(node: &Map<String, Value>, field: &str) -> Result<Option<String>, String> {
    match node.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(bad(format!("{field} is not text"))),
    }
}

fn int(value: &Value, what: &str) -> Result<PyInt, String> {
    match value {
        Value::Number(n) if n.is_i64() || n.is_u64() => PyInt::parse(&n.to_string()),
        _ => Err(bad(format!("{what} is not an integer"))),
    }
}

fn window(value: Option<&Value>) -> Result<Option<DorRef>, String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let range = object(value, "a window")?;
    let start = match range.get("start") {
        None | Some(Value::Null) => None,
        Some(v) => Some(int(v, "a window's start")?),
    };
    let end = int(range.get("end").unwrap_or(&Value::Null), "a window's end")?;
    let unit = TimeUnit::parse(text(range, "unit")?).ok_or_else(|| bad("a window's unit"))?;
    // The tree holds a range validation accepted, which `new` accepts again.
    let range = DateOffsetRange::new(start.as_ref(), &end, unit).map_err(bad)?;
    Ok(Some(range))
}

fn location(value: Option<&Value>) -> Result<Option<Location>, String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let at = object(value, "a location")?;
    let number = |field: &str| -> Result<usize, String> {
        at.get(field)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| bad(format!("a location's {field}")))
    };
    Ok(Some(Location {
        start_row: number("start_row")?,
        start_col: number("start_col")?,
        end_row: number("end_row")?,
        end_col: number("end_col")?,
        data_available: at
            .get("data_available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    }))
}

fn dtype(value: Option<&Value>) -> Result<Option<NodeDtype>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) => Dtype::parse(name)
            .map(|d| Some(NodeDtype::Plain(d)))
            .ok_or_else(|| bad(format!("the dtype {name:?}"))),
        Some(Value::Object(array)) => {
            let nested = text(array, "nested_dtype")?;
            Dtype::parse(nested)
                .map(|d| Some(NodeDtype::Array(d)))
                .ok_or_else(|| bad(format!("the dtype {nested:?}")))
        }
        Some(_) => Err(bad("a dtype is neither text nor an array")),
    }
}

fn child(node: &Map<String, Value>, field: &str) -> Result<NodeRef, String> {
    read_node(
        node.get(field)
            .ok_or_else(|| bad(format!("a node has no {field}")))?,
    )
}

/// One node of the tree, and its children.
pub fn read_node(value: &Value) -> Result<NodeRef, String> {
    let node = object(value, "a node")?;
    let base = Base {
        date_offset_range: window(node.get("date_offset_range"))?,
        dtype: dtype(node.get("dtype_maybe"))?,
        stype: match optional_text(node, "stype_maybe")? {
            None => None,
            Some(name) => {
                Some(Stype::parse(&name).ok_or_else(|| bad(format!("the stype {name:?}")))?)
            }
        },
        location: location(node.get("location"))?,
    };
    let kind = if let Some(fqn) = node.get("fqn") {
        Kind::Column {
            fqn: fqn
                .as_str()
                .ok_or_else(|| bad("a column's fqn"))?
                .to_string(),
            count_col: false,
        }
    } else if node.contains_key("aggr") {
        Kind::Aggregation {
            target: child(node, "target")?,
            aggr: AggregationType::parse(text(node, "aggr")?)
                .ok_or_else(|| bad("an aggregation"))?,
            aggr_time_range: window(node.get("aggr_time_range"))?,
            group_by: optional_text(node, "group_by")?,
            time_col: optional_text(node, "time_col")?,
        }
    } else if node.contains_key("rhs_target") {
        Kind::Join {
            rhs_target: child(node, "rhs_target")?,
            lhs_key: text(node, "lhs_key")?.to_string(),
            rhs_key: text(node, "rhs_key")?.to_string(),
        }
    } else if node.contains_key("bool_op") {
        Kind::Logical {
            left: child(node, "left")?,
            op: BoolOp::parse(text(node, "bool_op")?).ok_or_else(|| bad("a boolean operator"))?,
            right: match node.get("right") {
                None | Some(Value::Null) => None,
                Some(right) => Some(read_node(right)?),
            },
        }
    } else if node.contains_key("condition") {
        Kind::Filter {
            target: child(node, "target")?,
            condition: child(node, "condition")?,
        }
    } else if node.contains_key("op") {
        Kind::Condition {
            target: child(node, "target")?,
            op: Op::parse(text(node, "op")?).ok_or_else(|| bad("an operator"))?,
            value: child(node, "value")?,
            input_op: optional_text(node, "input_op")?,
        }
    } else if let Some(value) = node.get("value") {
        Kind::Constant {
            value: match value {
                Value::String(s) => ConstValue::Text(s.clone()),
                Value::Array(items) => {
                    ConstValue::List(items.iter().map(read_node).collect::<Result<_, _>>()?)
                }
                _ => return Err(bad("a constant's value")),
            },
        }
    } else {
        return Err(bad("a node of no known kind"));
    };
    Ok(Rc::new(RefCell::new(Node { base, kind })))
}

/// The validated query a tree holds. Its validation response is not read
/// back: a tree is only written for a query that validated.
pub fn read_query(tree: &Value) -> Result<ValidatedQuery, String> {
    let query = object(tree, "the tree")?;
    let optional_node = |field: &str| -> Result<Option<NodeRef>, String> {
        match query.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(node) => read_node(node).map(Some),
        }
    };
    let entity_ast = child(query, "entity_ast")?;
    let target_ast = child(query, "target_ast")?;
    let whatif_ast = optional_node("whatif_ast")?;
    let rfm_entity_ids = optional_node("rfm_entity_ids")?;
    for node in [Some(&target_ast), whatif_ast.as_ref()]
        .into_iter()
        .flatten()
    {
        crate::validator::mark_count_cols(node);
    }
    let flag = |field: &str| query.get(field).and_then(Value::as_bool).unwrap_or(false);
    let parsed = ParsedQuery {
        entity_ast,
        target_ast,
        whatif_ast,
        top_k: match query.get("top_k") {
            None | Some(Value::Null) => None,
            Some(v) => Some(int(v, "top_k")?),
        },
        problem_type: match optional_text(query, "problem_type")? {
            None => None,
            Some(name) => Some(
                ProblemType::parse(&name)
                    .ok_or_else(|| bad(format!("the problem type {name:?}")))?,
            ),
        },
        rfm_query: flag("rfm_query"),
        for_each: optional_text(query, "for_each")?.unwrap_or_else(|| "FOR EACH".to_string()),
        rfm_entity_ids,
        num_forecasts: match query.get("num_forecasts") {
            None | Some(Value::Null) => PyInt::from_i64(1),
            Some(v) => int(v, "num_forecasts")?,
        },
        evaluate: flag("evaluate"),
        explain: flag("explain"),
    };
    Ok(ValidatedQuery {
        query: parsed,
        validation_response: ValidationResponse::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every resolved query of the PQL corpus reads back to its tree.
    #[test]
    fn every_corpus_tree_reads_back_to_itself() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/parity/pql/valid.json");
        let corpus: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut read = 0;
        for case in corpus["cases"].as_array().unwrap() {
            let tree = &case["expected_tree"];
            if tree.is_null() {
                continue;
            }
            let query = read_query(tree).unwrap_or_else(|e| panic!("{}: {e}", case["id"]));
            let again: Value =
                serde_json::from_str(&query.query.to_json(&ValidationResponse::default()).render())
                    .unwrap();
            let mut expected = tree.clone();
            expected["validation_response"] = again["validation_response"].clone();
            assert_eq!(again, expected, "{}", case["id"]);
            read += 1;
        }
        assert!(read >= 150, "only {read} trees read");
    }
}
