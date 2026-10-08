// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON a host hands the schema core and gets back
//! (`tests/rust_core/schema/CONTRACT.md`).

use std::sync::Arc;

use serde_json::Value;

use crate::client::{self, DataArgs, Key, Rel, Said};
use crate::dt::Host;
use crate::frame::Frame;

/// An edit applied after inference: `setattr(schema[table][column], set,
/// value)`; with no column, `setattr(schema[table], set, value)`; or, as
/// a link, `schema.relationships.append(Relationship(*value))`.
#[derive(Clone, Debug)]
pub struct Edit {
    pub table: String,
    pub column: String,
    pub set: String,
    pub value: Value,
    /// `column`, `table` or `link`.
    pub kind: String,
}

impl Edit {
    /// The edit, made on `schema`.
    pub fn apply(&self, schema: &mut client::Schema, said: &mut Said) -> client::R<()> {
        match self.kind.as_str() {
            "table" => schema.edit_table(&self.table, &self.set, &self.value, said),
            "link" => {
                let names: Vec<String> = self
                    .value
                    .as_array()
                    .map(|p| {
                        p.iter()
                            .map(|v| v.as_str().unwrap_or_default().to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                if names.len() != 4 {
                    return Err(client::Refusal::invalid("a relationship is four names"));
                }
                schema.append_link(Rel::new(&names[0], &names[1], &names[2], &names[3]), said)
            }
            _ => schema.edit(&self.table, &self.column, &self.set, &self.value, said),
        }
    }
}

/// `Schema.from_data`'s keyword arguments, and the edits after it.
pub struct FromData {
    pub args: DataArgs,
    pub edits: Vec<Edit>,
}

fn text(value: &Value, what: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{what} is text"))
}

/// `primary_keys`: `null` or `{table: column or [column, ...]}`.
pub fn keys(value: Option<&Value>) -> Result<Option<client::NamedKeys>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) => {
            let mut out = Vec::new();
            for (table, key) in map {
                let key = match key {
                    Value::String(s) => Ok(Some(Key::One(s.clone()))),
                    Value::Array(items) if items.iter().all(Value::is_string) => {
                        Ok(Some(Key::Many(
                            items
                                .iter()
                                .map(|i| text(i, "a key column"))
                                .collect::<Result<_, _>>()?,
                        )))
                    }
                    Value::Null => Ok(None),
                    other => Err(client::json_repr(other)),
                };
                out.push((table.clone(), key));
            }
            Ok(Some(out))
        }
        Some(_) => Err("primary_keys is null or an object".to_string()),
    }
}

/// `time_columns`: `null` or `{table: column}`.
pub fn time_columns(value: Option<&Value>) -> Result<Option<Vec<(String, Value)>>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) => Ok(Some(
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        )),
        Some(_) => Err("time_columns is null or an object".to_string()),
    }
}

/// `relationships`: `null` or a list of four names each.
pub fn relationships(value: Option<&Value>) -> Result<Option<Vec<Rel>>, String> {
    let (links, refused) = relationships_given(value)?;
    match refused {
        Some(message) => Err(message),
        None => Ok(links),
    }
}

/// `relationships` as a caller gave them: the links up to the first value
/// that is not a `Relationship` (`{"$object": {"type", "repr"}}`), and the
/// refusal of that one, which is made where `_checked_links` reaches it.
pub fn relationships_given(
    value: Option<&Value>,
) -> Result<(Option<Vec<Rel>>, Option<String>), String> {
    match value {
        None | Some(Value::Null) => Ok((None, None)),
        Some(Value::Array(items)) => {
            let mut out = Vec::new();
            for item in items {
                let Some(parts) = item.as_array().filter(|p| p.len() == 4) else {
                    let (type_name, shown) = match crate::pyval::PyVal::decode(item) {
                        Ok(crate::pyval::PyVal::Opaque {
                            type_name, repr, ..
                        }) => (type_name, repr),
                        _ => ("object".to_string(), item.to_string()),
                    };
                    return Ok((
                        Some(out),
                        Some(format!(
                            "a relationship must be a Relationship, got {type_name}: {shown}. Build one with Relationship(source_table, source_column, target_table, target_column)"
                        )),
                    ));
                };
                let names: Vec<String> = parts
                    .iter()
                    .map(|p| {
                        p.as_str()
                            .map_or_else(|| client::json_repr(p), str::to_string)
                    })
                    .collect();
                out.push(Rel::new(&names[0], &names[1], &names[2], &names[3]));
            }
            Ok((Some(out), None))
        }
        Some(_) => Err("relationships is null or a list".to_string()),
    }
}

/// `tables` of `Schema.from_source`: `null`, `{name: source_name}`, or a list
/// of `[name, source_name]` pairs, either of which may be other than text.
/// The names up to the first pair `_require_table_names` refuses, and that
/// refusal.
/// Source tables as named, and the refusal of the first that is not a pair
/// of names.
pub type SourceTables = (Option<Vec<(String, String)>>, Option<String>);

pub fn source_tables(value: Option<&Value>) -> Result<SourceTables, String> {
    let pairs: Vec<(Value, Value)> = match value {
        None | Some(Value::Null) => return Ok((None, None)),
        Some(Value::Object(map)) => map
            .iter()
            .map(|(k, v)| (Value::String(k.clone()), v.clone()))
            .collect(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|pair| {
                let pair = pair.as_array().filter(|p| p.len() == 2);
                pair.map(|p| (p[0].clone(), p[1].clone()))
                    .ok_or_else(|| "a table is a pair of names".to_string())
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("tables is null, an object or a list of pairs".to_string()),
    };
    let mut out = Vec::new();
    for (name, source_name) in pairs {
        let Some(name) = name.as_str().filter(|n| !n.is_empty()) else {
            return Ok((
                Some(out),
                Some(format!(
                    "table names must be non-empty strings; got {}",
                    client::json_repr(&name)
                )),
            ));
        };
        let Some(source) = source_name.as_str().filter(|n| !n.is_empty()) else {
            return Ok((
                Some(out),
                Some(format!(
                    "table {} must map to its name in the source, a non-empty string; got {}",
                    crate::py::str_repr(name),
                    client::json_repr(&source_name)
                )),
            ));
        };
        out.push((name.to_string(), source.to_string()));
    }
    Ok((Some(out), None))
}

/// `options_json` of `schema_from_data`.
pub fn data_args(options: &Value) -> Result<FromData, String> {
    let map = options.as_object().ok_or("the options are a JSON object")?;
    let edits = match map.get("edits") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => {
            let mut out = Vec::new();
            for item in items {
                let get = |k: &str| item.get(k).ok_or_else(|| format!("an edit has {k}"));
                if let Some(link) = item.get("link") {
                    out.push(Edit {
                        table: String::new(),
                        column: String::new(),
                        set: String::new(),
                        value: link.clone(),
                        kind: "link".to_string(),
                    });
                    continue;
                }
                if let Some(attribute) = item.get("set_table") {
                    out.push(Edit {
                        table: text(get("table")?, "an edit's table")?,
                        column: String::new(),
                        set: text(attribute, "an edit's attribute")?,
                        value: get("value")?.clone(),
                        kind: "table".to_string(),
                    });
                    continue;
                }
                out.push(Edit {
                    table: text(get("table")?, "an edit's table")?,
                    column: text(get("column")?, "an edit's column")?,
                    set: text(get("set")?, "an edit's attribute")?,
                    value: get("value")?.clone(),
                    kind: "column".to_string(),
                });
            }
            out
        }
        Some(_) => return Err("edits is a list".to_string()),
    };
    let (links, not_a_link) = relationships_given(map.get("relationships"))?;
    Ok(FromData {
        args: DataArgs {
            primary_keys: keys(map.get("primary_keys"))?,
            time_columns: time_columns(map.get("time_columns"))?,
            relationships: links,
            not_a_link,
        },
        edits,
    })
}

/// `schema_from_data`'s answer: the description or refusal, and the
/// warnings, as JSON text.
pub fn schema_from_data(
    frames: Vec<(String, Arc<Frame>)>,
    given: &FromData,
    host: &Host,
) -> (String, String) {
    let mut said = Said::default();
    let result = client::from_data(frames, &given.args, host, &mut said).and_then(|mut schema| {
        for edit in &given.edits {
            edit.apply(&mut schema, &mut said)?;
        }
        Ok(schema)
    });
    client::outcome_json(&result, &said)
}

/// `schema_from_source`'s `backend`, `source_json` and `args_json`, read.
pub fn source_args(
    backend: &str,
    source: &Value,
    args: &Value,
) -> Result<(crate::source::SourceInfo, crate::source::SourceArgs), String> {
    let backend = crate::engine::Backend::parse(backend).ok_or_else(|| {
        format!("no backend {backend:?}; use databricks, duckdb, postgres, snow or sqlite")
    })?;
    let source = source.as_object().ok_or("the source is a JSON object")?;
    let namespace = match source.get("namespace") {
        Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None | Some(Value::Null) => Vec::new(),
        Some(_) => return Err("the namespace is an object".to_string()),
    };
    let semantic_model = match source.get("semantic_model") {
        None | Some(Value::Null) => None,
        Some(v) => Some(text(v, "a semantic model's name")?),
    };
    let trust_expressions = match source.get("trust_expressions") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("trust_expressions is a boolean".to_string()),
    };
    let args = args.as_object().ok_or("the arguments are a JSON object")?;
    let (tables, not_a_table) = source_tables(args.get("tables"))?;
    let (links, not_a_link) = relationships_given(args.get("relationships"))?;
    let check_keys = match args.get("check_keys") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("check_keys is a boolean".to_string()),
    };
    Ok((
        crate::source::SourceInfo {
            backend,
            namespace,
            semantic_model,
            trust_expressions,
        },
        crate::source::SourceArgs {
            tables,
            not_a_table,
            primary_keys: keys(args.get("primary_keys"))?,
            time_columns: time_columns(args.get("time_columns"))?,
            relationships: links,
            not_a_link,
            check_keys,
        },
    ))
}
