// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only (the `schema-check` feature, which `test-ops` brings): a JSON value checked against a
//! definition of the contract's own schema ([`crate::schema_json`]), so a
//! producer's tests can hold what it writes to the definition the lock
//! holds (T1.22: the request state's payload, whose definition `sdm-wire`'s
//! typed payload derives from T1.28w).
//!
//! It is not a JSON Schema implementation: it reads only the keywords the
//! definition it is used on holds (the request state's payload: `type` of
//! `object`, `array`, `string`, `boolean`, `integer` or `null`, or a list
//! of them (from T2.13, a target transform's `["string", "null"]`), `properties`,
//! `required`, `additionalProperties`, `items`, `anyOf`, `enum`, `minimum`,
//! the schemas `true` and `false`, and the annotations `description`,
//! `title`, `format` and any `x-` keyword, which it does not read). It
//! walks the whole definition before it reads the value, and fails on any
//! other keyword anywhere in it: under a property the value does not
//! carry, under a type the value is not, or in an `anyOf` option not tried
//! or not holding as much as on the value's own path (M3, T2.1b), so no
//! value passes a definition holding a rule this check would not read. A
//! definition that comes to need another keyword needs it added here, with
//! a test (T1.28w).

use serde_json::Value;

/// What is wrong with `value` as the definition `name` of the contract's
/// schema, the first problem found, by its path; `Ok` when it holds.
pub fn check(name: &str, value: &Value) -> Result<(), String> {
    let root: Value =
        serde_json::from_str(&crate::schema_json()).map_err(|e| format!("the schema: {e}"))?;
    let definition = root["$defs"]
        .get(name)
        .ok_or_else(|| format!("the schema defines no {name}"))?;
    readable(definition, "$")?;
    node(definition, value, "$")
}

/// Fails on the first keyword of `schema`, or of any subschema in it, that
/// [`node`] does not read, whatever value it would be given (M3): the
/// whole definition, walked through every subschema keyword it reads.
fn readable(schema: &Value, at: &str) -> Result<(), String> {
    let map = match schema {
        Value::Bool(_) => return Ok(()),
        Value::Object(map) => map,
        _ => return Err(format!("{at}: a schema this check does not read")),
    };
    for (keyword, rule) in map {
        if ANNOTATIONS.contains(&keyword.as_str()) || keyword.starts_with("x-") {
            continue;
        }
        match keyword.as_str() {
            "type" => {
                for kind in kinds(rule, at)? {
                    type_holds(kind, &Value::Null)?;
                }
            }
            "enum" | "required" if rule.is_array() => {}
            "minimum" if rule.is_number() => {}
            "properties" => {
                let properties = rule
                    .as_object()
                    .ok_or_else(|| format!("{at}: properties is not a map"))?;
                for (key, sub) in properties {
                    readable(sub, &format!("{at}.{key}"))?;
                }
            }
            "additionalProperties" => readable(rule, &format!("{at}.*"))?,
            "items" => readable(rule, &format!("{at}[]"))?,
            "anyOf" => {
                let options = rule
                    .as_array()
                    .ok_or_else(|| format!("{at}: anyOf is not a list"))?;
                for (n, option) in options.iter().enumerate() {
                    readable(option, &format!("{at}|{n}"))?;
                }
            }
            "enum" | "required" | "minimum" => {
                return Err(format!(
                    "{at}: {keyword} is not of the form this check reads"
                ));
            }
            other => return Err(format!("{at}: a keyword this check does not read: {other}")),
        }
    }
    Ok(())
}

const ANNOTATIONS: [&str; 3] = ["description", "title", "format"];

/// The types a `type` keyword names: one, or a list of one or more.
fn kinds<'a>(rule: &'a Value, at: &str) -> Result<Vec<&'a str>, String> {
    let unread = || format!("{at}: a type this check does not read: {rule}");
    match rule {
        Value::String(kind) => Ok(vec![kind.as_str()]),
        Value::Array(list) if !list.is_empty() => list
            .iter()
            .map(|kind| kind.as_str().ok_or_else(unread))
            .collect(),
        _ => Err(unread()),
    }
}

fn type_holds(kind: &str, value: &Value) -> Result<bool, String> {
    Ok(match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "integer" => value.is_i64() || value.is_u64(),
        "null" => value.is_null(),
        other => return Err(format!("a type this check does not read: {other}")),
    })
}

fn node(schema: &Value, value: &Value, at: &str) -> Result<(), String> {
    let map = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{at}: nothing is allowed here")),
        Value::Object(map) => map,
        _ => return Err(format!("{at}: a schema this check does not read")),
    };
    for (keyword, rule) in map {
        if ANNOTATIONS.contains(&keyword.as_str()) || keyword.starts_with("x-") {
            continue;
        }
        match keyword.as_str() {
            "type" => {
                let mut holds = false;
                for kind in kinds(rule, at)? {
                    holds |= type_holds(kind, value)?;
                }
                if !holds {
                    return Err(format!("{at}: not of type {rule}"));
                }
            }
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
                        node(sub, item, &format!("{at}.{key}"))?;
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
                        node(rule, item, &format!("{at}.{key}"))
                            .map_err(|e| format!("{e} (a key the definition does not name)"))?;
                    }
                }
            }
            "items" => {
                let Value::Array(items) = value else { continue };
                for (n, item) in items.iter().enumerate() {
                    node(rule, item, &format!("{at}[{n}]"))?;
                }
            }
            "anyOf" => {
                let options = rule
                    .as_array()
                    .ok_or_else(|| format!("{at}: anyOf is not a list"))?;
                if !options.iter().any(|o| node(o, value, at).is_ok()) {
                    return Err(format!("{at}: none of anyOf's options hold"));
                }
            }
            other => return Err(format!("{at}: a keyword this check does not read: {other}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{node, readable};

    #[test]
    fn a_keyword_it_does_not_read_fails_wherever_it_is_in_the_definition() {
        // M3: each where the value never reaches it, a value that holds.
        let value = json!({"n": 1});
        for schema in [
            // Under a property the value does not carry.
            json!({"type": "object", "properties": {"n": {"type": "integer"}, "m": {"pattern": "a"}}}),
            // Under items, the value no list.
            json!({"type": "object", "properties": {"n": {"type": "integer"}}, "additionalProperties": {"items": {"const": 1}}}),
            // In an anyOf option not tried, the first holding.
            json!({"anyOf": [{"type": "object"}, {"oneOf": [true]}]}),
            // A type the value is not, of a kind the check does not read.
            json!({"anyOf": [{"type": "object"}, {"type": "number"}]}),
        ] {
            let err = readable(&schema, "$").expect_err("an unread keyword passed");
            assert!(err.contains("does not read"), "{schema}: {err}");
        }
        // The same definitions without it read.
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}, "m": {"enum": ["a"]}}, "required": ["n"], "additionalProperties": false});
        assert_eq!(readable(&schema, "$"), Ok(()));
        assert_eq!(node(&schema, &value, "$"), Ok(()));
    }

    #[test]
    fn the_contract_s_state_payload_is_read_whole() {
        // The whole definition reads: a value that is not an object fails
        // on its type, not on a keyword.
        let err = super::check("StatePayload", &json!(null)).unwrap_err();
        assert!(err.contains("not of type"), "{err}");
    }

    #[test]
    fn a_keyword_or_type_the_payload_does_not_use_fails_closed() {
        for schema in [
            json!({"const": 1}),
            json!({"oneOf": [true]}),
            json!({"$ref": "#/$defs/StatePayload"}),
            json!({"prefixItems": [true]}),
            json!({"pattern": "a"}),
            json!({"maximum": 1}),
            json!({"type": "number"}),
            json!({"type": ["string", "number"]}),
            json!({"type": []}),
            json!({"type": [1]}),
            json!(1),
        ] {
            let err = node(&schema, &json!(1), "$").expect_err("an unread keyword passed");
            assert!(err.contains("does not read"), "{schema}: {err}");
        }
    }

    #[test]
    fn the_definition_holds_to_the_keywords_it_reads() {
        let schema = json!({
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0}, "k": {"enum": ["a"]}},
            "required": ["n"],
            "additionalProperties": false
        });
        assert_eq!(node(&schema, &json!({"n": 1, "k": "a"}), "$"), Ok(()));
        for (value, problem) in [
            (json!({"n": -1}), "below"),
            (json!({"n": "1"}), "not of type"),
            (json!({"k": "a"}), "n is missing"),
            (json!({"n": 1, "k": "b"}), "not one of"),
            (json!({"n": 1, "x": 1}), "nothing is allowed here"),
        ] {
            let err = node(&schema, &value, "$").expect_err("a fault held");
            assert!(err.contains(problem), "{value}: {err}");
        }
        let either = json!({"anyOf": [{"type": "string"}, {"type": "array", "items": true}]});
        assert_eq!(node(&either, &json!([1]), "$"), Ok(()));
        assert!(node(&either, &json!(1), "$").is_err());
        // A list of types holds where any of them does (T2.13).
        let nullable = json!({"type": "array", "items": {"type": ["string", "null"]}});
        assert_eq!(readable(&nullable, "$"), Ok(()));
        assert_eq!(node(&nullable, &json!(["quantile", null]), "$"), Ok(()));
        let err = node(&nullable, &json!(["quantile", 1]), "$").unwrap_err();
        assert!(err.contains("not of type"), "{err}");
        assert!(node(&json!({"type": "null"}), &json!(0), "$").is_err());
    }
}
