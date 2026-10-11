// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only (the `test-ops` feature): a JSON value checked against a
//! definition of the contract's own schema ([`crate::schema_json`]), so a
//! producer's tests can hold what it writes to the definition the lock
//! holds (T1.22: the request state's payload).
//!
//! It reads the keywords the contract's definitions use and fails closed
//! on any other, so a definition cannot pass by naming a keyword it does
//! not check: `type`, `properties`, `required`, `additionalProperties`,
//! `items`, `prefixItems`, `anyOf`, `oneOf`, `enum`, `const`, `minimum`,
//! `$ref` within the schema, and the annotations (`description`, `title`,
//! `examples`, `format` and any `x-` keyword), which it does not read.

use serde_json::Value;

/// What is wrong with `value` as the definition `name` of the contract's
/// schema, the first problem found, by its path; `Ok` when it holds.
pub fn check(name: &str, value: &Value) -> Result<(), String> {
    let root: Value =
        serde_json::from_str(&crate::schema_json()).map_err(|e| format!("the schema: {e}"))?;
    let definition = root["$defs"]
        .get(name)
        .ok_or_else(|| format!("the schema defines no {name}"))?;
    node(&root, definition, value, "$")
}

const ANNOTATIONS: [&str; 4] = ["description", "title", "examples", "format"];

fn type_holds(kind: &str, value: &Value) -> Result<bool, String> {
    Ok(match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        other => return Err(format!("a type this check does not read: {other}")),
    })
}

fn node(root: &Value, schema: &Value, value: &Value, at: &str) -> Result<(), String> {
    let map = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{at}: nothing is allowed here")),
        Value::Object(map) => map,
        _ => return Err(format!("{at}: the schema here is not a schema")),
    };
    for (keyword, rule) in map {
        if ANNOTATIONS.contains(&keyword.as_str()) || keyword.starts_with("x-") {
            continue;
        }
        match keyword.as_str() {
            "$ref" => {
                let name = rule
                    .as_str()
                    .and_then(|r| r.strip_prefix("#/$defs/"))
                    .ok_or_else(|| format!("{at}: a $ref this check does not follow"))?;
                let target = root["$defs"]
                    .get(name)
                    .ok_or_else(|| format!("{at}: $ref to no definition {name}"))?;
                node(root, target, value, at)?;
            }
            "type" => {
                let kinds: Vec<&str> = match rule {
                    Value::String(one) => vec![one.as_str()],
                    Value::Array(several) => several.iter().filter_map(Value::as_str).collect(),
                    _ => return Err(format!("{at}: type is not a type")),
                };
                let mut holds = false;
                for kind in kinds {
                    holds |= type_holds(kind, value)?;
                }
                if !holds {
                    return Err(format!("{at}: not of type {rule}"));
                }
            }
            "const" if value != rule => return Err(format!("{at}: not {rule}")),
            "const" => {}
            "enum" => {
                let options = rule
                    .as_array()
                    .ok_or_else(|| format!("{at}: enum is not a list"))?;
                if !options.contains(value) {
                    return Err(format!("{at}: not one of {rule}"));
                }
            }
            "minimum" => {
                let floor = rule
                    .as_f64()
                    .ok_or_else(|| format!("{at}: minimum is not a number"))?;
                if value.as_f64().is_some_and(|v| v < floor) {
                    return Err(format!("{at}: below {floor}"));
                }
            }
            "required" => {
                let Value::Object(object) = value else {
                    continue;
                };
                for key in rule.as_array().into_iter().flatten() {
                    let key = key
                        .as_str()
                        .ok_or_else(|| format!("{at}: required is not text"))?;
                    if !object.contains_key(key) {
                        return Err(format!("{at}: {key} is missing"));
                    }
                }
            }
            "properties" => {
                let Value::Object(object) = value else {
                    continue;
                };
                let properties = rule
                    .as_object()
                    .ok_or_else(|| format!("{at}: properties is not a map"))?;
                for (key, item) in object {
                    if let Some(sub) = properties.get(key) {
                        node(root, sub, item, &format!("{at}.{key}"))?;
                    }
                }
            }
            "additionalProperties" => {
                let Value::Object(object) = value else {
                    continue;
                };
                let named = map.get("properties").and_then(Value::as_object);
                for (key, item) in object {
                    if !named.is_some_and(|n| n.contains_key(key)) {
                        node(root, rule, item, &format!("{at}.{key}"))
                            .map_err(|e| format!("{e} (a key the definition does not name)"))?;
                    }
                }
            }
            "items" => {
                let Value::Array(items) = value else { continue };
                let skip = map
                    .get("prefixItems")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                for (n, item) in items.iter().enumerate().skip(skip) {
                    node(root, rule, item, &format!("{at}[{n}]"))?;
                }
            }
            "prefixItems" => {
                let Value::Array(items) = value else { continue };
                let prefix = rule
                    .as_array()
                    .ok_or_else(|| format!("{at}: prefixItems is not a list"))?;
                for (n, (sub, item)) in prefix.iter().zip(items).enumerate() {
                    node(root, sub, item, &format!("{at}[{n}]"))?;
                }
            }
            "anyOf" | "oneOf" => {
                let options = rule
                    .as_array()
                    .ok_or_else(|| format!("{at}: {keyword} is not a list"))?;
                let held = options
                    .iter()
                    .filter(|o| node(root, o, value, at).is_ok())
                    .count();
                if held == 0 || (keyword == "oneOf" && held > 1) {
                    return Err(format!("{at}: {held} of {keyword}'s options hold"));
                }
            }
            other => return Err(format!("{at}: a keyword this check does not read: {other}")),
        }
    }
    Ok(())
}
