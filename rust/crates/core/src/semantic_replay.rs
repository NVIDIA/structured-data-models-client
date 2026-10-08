// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use sdm_protocol::Outcome;
use sdm_schema::dt::Host;
use serde_json::Value;

fn fields(value: &Value, expected: &[&str]) -> Result<(), String> {
    let actual = value
        .as_object()
        .ok_or("semantic input must be an object")?;
    if actual.len() != expected.len() || expected.iter().any(|name| !actual.contains_key(*name)) {
        return Err("unexpected semantic input fields".into());
    }
    Ok(())
}

pub fn replay(
    input: &Value,
    host: Host,
    blobs: &BTreeMap<String, &[u8]>,
    directory: &str,
) -> Result<(Value, Value), String> {
    fields(input, &["initial_schema", "queries"])?;
    let initial = &input["initial_schema"];
    fields(initial, &["function", "args"])?;
    if initial["function"].as_str() != Some("schema_source_run") {
        return Err("semantic replay requires original schema-source arguments".into());
    }
    let arguments = initial["args"]
        .as_array()
        .ok_or("invalid semantic source arguments")?
        .iter()
        .map(|value| crate::public_replay::decode(value, blobs, directory))
        .collect::<Result<Vec<_>, _>>()?;
    if arguments.len() != 3 || arguments.iter().any(|value| value.as_str().is_none()) {
        return Err("semantic source requires three text arguments".into());
    }
    let mut queries = Vec::new();
    let mut outcomes = Vec::new();
    for query in input["queries"]
        .as_array()
        .ok_or("invalid semantic query tape")?
    {
        fields(query, &["sql", "params", "outcome"])?;
        let sql = query["sql"].as_str().ok_or("invalid semantic statement")?;
        if !query["params"].is_null() && !query["params"].is_array() {
            return Err("invalid semantic query parameters".into());
        }
        let outcome = crate::public_replay::outcome(&query["outcome"], blobs, directory)?;
        if !matches!(outcome, Outcome::Data(_) | Outcome::Error { .. }) {
            return Err("semantic query cannot consume a model outcome".into());
        }
        queries.push((sql.to_owned(), query["params"].clone()));
        outcomes.push((false, outcome));
    }
    crate::flow::semantic_source(arguments, host, queries, outcomes).map_err(|error| error.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use serde_json::json;

    fn rejected(input: Value) -> String {
        replay(&input, Host::assumed(), &BTreeMap::new(), "semantic")
            .expect_err("invalid replay input")
    }

    #[test]
    fn semantic_input_does_not_accept_computed_schema_or_later_arguments() {
        let input =
            json!({"initial_schema":{"function":"schema_source_run","args":[]},"queries":[]});
        let mut extra = input.clone();
        extra["schema"] = json!({});
        assert_eq!(rejected(extra), "unexpected semantic input fields");
        let mut wrong = input.clone();
        wrong["initial_schema"]["function"] = json!("materialize_run");
        assert_eq!(
            rejected(wrong),
            "semantic replay requires original schema-source arguments"
        );
        let mut later = input;
        later["initial_schema"]["later_args"] = json!([]);
        assert_eq!(rejected(later), "unexpected semantic input fields");
    }

    #[test]
    fn semantic_queries_require_exact_external_inputs() {
        let input = json!({"initial_schema":{"function":"schema_source_run","args":["snowflake","{}","{}"]},"queries":[{"sql":"SELECT 1","params":[],"outcome":{"response":"{}"}}]});
        assert_eq!(
            rejected(input.clone()),
            "semantic query cannot consume a model outcome"
        );
        let mut missing = input.clone();
        missing["queries"][0].as_object_mut().unwrap().remove("sql");
        assert_eq!(rejected(missing), "unexpected semantic input fields");
        let mut parameters = input;
        parameters["queries"][0]["params"] = json!("invalid");
        assert_eq!(rejected(parameters), "invalid semantic query parameters");
    }

    fn source_input(backend: &str, sql: &str, params: Value, outcome: Value) -> Value {
        json!({
            "initial_schema": {
                "function": "schema_source_run",
                "args": [backend, "{\"semantic_model\":\"example\"}", "{}"],
            },
            "queries": [{"sql": sql, "params": params, "outcome": outcome}],
        })
    }

    #[test]
    fn semantic_replay_preserves_null_parameters_and_computes_refusals() {
        let sql = "DESCRIBE TABLE EXTENDED `example`";
        let input = source_input(
            "databricks",
            sql,
            Value::Null,
            json!({"error": {"message": "source unavailable"}}),
        );
        let (observation, queries) =
            replay(&input, Host::assumed(), &BTreeMap::new(), "semantic").unwrap();
        assert!(
            observation["error"]["message"]
                .as_str()
                .unwrap()
                .contains("source unavailable")
        );
        assert_eq!(observation["warnings"], json!([]));
        assert_eq!(queries, json!([[sql, null]]));

        for mutation in ["sql", "params", "missing", "extra"] {
            let mut changed = input.clone();
            match mutation {
                "sql" => changed["queries"][0]["sql"] = json!("SELECT changed"),
                "params" => changed["queries"][0]["params"] = json!([]),
                "missing" => changed["queries"] = json!([]),
                "extra" => {
                    let query = changed["queries"][0].clone();
                    changed["queries"].as_array_mut().unwrap().push(query);
                }
                _ => unreachable!(),
            }
            assert!(
                replay(&changed, Host::assumed(), &BTreeMap::new(), "semantic").is_err(),
                "accepted {mutation} mutation"
            );
        }
    }

    #[test]
    fn semantic_replay_decodes_arrow_before_computing_a_definition_refusal() {
        let definition: ArrayRef = Arc::new(StringArray::from(vec!["{}"]));
        let batch = RecordBatch::try_from_iter([("definition", definition)]).unwrap();
        let mut bytes = Vec::new();
        let mut writer =
            arrow_ipc::writer::FileWriter::try_new(&mut bytes, &batch.schema()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let digest = "a".repeat(64);
        let blobs = BTreeMap::from([(format!("semantic/tables/{digest}.arrow"), bytes.as_slice())]);
        let sql = "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)";
        let input = source_input(
            "snow",
            sql,
            json!(["example"]),
            json!({"data": {"$table": digest}}),
        );
        let (observation, queries) = replay(&input, Host::assumed(), &blobs, "semantic").unwrap();
        assert_eq!(observation["error"]["code"], json!("KeyError"));
        assert_eq!(observation["error"]["message"], json!("'tables'"));
        assert_eq!(queries, json!([[sql, ["example"]]]));
        assert_eq!(rejected(input), "missing Arrow input dependency");
    }
}
