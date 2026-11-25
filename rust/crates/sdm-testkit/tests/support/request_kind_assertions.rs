// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Small source queries for the 45 authored provider inputs. These deliberately
// bypass SDM planning, provider dispatch and result admission.
use duckdb::Connection;
use sdm_host::ArrowData;
use serde_json::Value;

fn quoted(value: &Value) -> String {
    format!("\"{}\"", value.as_str().unwrap().replace('"', "\"\""))
}
fn table(relation: &Value) -> String {
    let parts = relation["source"].as_array().unwrap();
    assert_eq!(parts.len(), 1);
    quoted(&parts[0])
}
fn columns(values: &Value) -> String {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(quoted)
        .collect::<Vec<_>>()
        .join(", ")
}
fn literal(value: &Value) -> String {
    format!("'{}'", value.as_str().unwrap().replace('\'', "''"))
}

pub fn source_answer(connection: &Connection, fetch: &Value) -> ArrowData {
    let r = &fetch["request"];
    let kind = r["request"].as_str().unwrap();
    let sql = match kind {
        "describe" => {
            let part = r["part"].as_str().unwrap();
            match part {
                "tables" => "SELECT table_name FROM information_schema.tables WHERE table_schema='main'".into(),
                "indexes" => "SELECT table_name, expressions FROM duckdb_indexes()".into(),
                _ => {
                    let name = literal(&r["tables"][0]["source"][0]);
                    match part {
                        "columns" => format!("SELECT * FROM pragma_table_info({name})"),
                        "keys" => format!("SELECT constraint_type, constraint_column_names FROM duckdb_constraints() WHERE table_name={name}"),
                        "foreign_keys" => format!("SELECT constraint_index, referenced_table, constraint_column_names, referenced_column_names FROM duckdb_constraints() WHERE table_name={name} AND constraint_type='FOREIGN KEY'"),
                        _ => panic!("unhandled catalog part {part}"),
                    }
                }
            }
        }
        "key_counts" => r["keys"].as_array().unwrap().iter().enumerate().map(|(i, key)| {
            assert_eq!(key["columns"].as_array().unwrap().len(), 1);
            let c = quoted(&key["columns"][0]);
            format!("SELECT {i}::INTEGER AS k, COUNT(*) AS n, COUNT(DISTINCT {c}) AS d, COUNT(*)-COUNT({c}) AS z FROM {}", table(&key["table"]))
        }).collect::<Vec<_>>().join(" UNION ALL "),
        "time_bounds" => r["columns"].as_array().unwrap().iter().enumerate().map(|(i, col)| {
            let c = quoted(&col["column"]);
            format!("SELECT {i}::INTEGER AS table_index, MIN({c}) AS min_date, MAX({c}) AS max_date FROM {}", table(&col["table"]))
        }).collect::<Vec<_>>().join(" UNION ALL "),
        _ => {
            let source = table(&r["table"]);
            let filters = r["filters"].as_array().map(|filters| filters.iter().map(|f| {
                assert_eq!(f["filter"], "not_null");
                format!("{} IS NOT NULL", quoted(&f["column"]))
            }).collect::<Vec<_>>()).unwrap_or_default();
            let filter = if filters.is_empty() { String::new() } else { format!(" WHERE {}", filters.join(" AND ")) };
            if kind == "count_rows" {
                format!("SELECT COUNT(*) AS rows FROM {source}{filter}")
            } else {
                let projected = columns(&r["columns"]);
                match kind {
                    "scan" | "preview" => format!("SELECT {projected} FROM {source}{filter}"),
                    // The database owns reservoir sampling. Compare its direct
                    // seeded draw, not another call through SDM's planner.
                    "sample_entities" => {
                        let n = r["n"].as_u64().unwrap();
                        let seed = r["seed"]["v"].as_str().unwrap().parse::<u64>().unwrap();
                        format!("SELECT {projected} FROM (SELECT * FROM {source}{filter}) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})")
                    }
                    "neighbors" | "lookup" => {
                        assert_eq!(r["key"].as_array().unwrap().len(), 1);
                        assert!(r["window"].is_null() && r["per_seed"].is_null() && r["order"].is_null());
                        let key = quoted(&r["key"][0]);
                        fetch["keys"].as_array().unwrap().iter().enumerate().map(|(i, seed)| {
                            let seed = seed.as_i64().unwrap();
                            let matches = if kind == "lookup" {
                                format!(", (SELECT COUNT(*) FROM {source} WHERE {key}={seed}) AS __kumo_matches__")
                            } else { String::new() };
                            format!("SELECT {i}::BIGINT AS __kumo_batch__{matches}, {projected} FROM {source} WHERE {key}={seed}")
                        }).collect::<Vec<_>>().join(" UNION ALL ")
                    }
                    _ => panic!("unhandled input kind {kind}"),
                }
            }
        }
    };
    let mut statement = connection.prepare(&sql).unwrap();
    let arrow = statement.query_arrow([]).unwrap();
    let schema = arrow.get_schema();
    ArrowData::new(schema, arrow.collect())
}
