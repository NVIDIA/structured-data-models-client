// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode shared literal fixture fields. No product code computes expectations.
use serde_json::{Map, Value};

pub fn read(text: &str) -> Value {
    let file: Value = serde_json::from_str(text).unwrap();
    if file["format"] != "sdm-shared-fixture/1" {
        return file;
    }
    let tables = file["tables"].as_array().unwrap();
    expand(&file["document"], tables, tables.len())
}

fn expand(node: &Value, tables: &[Value], before: usize) -> Value {
    match node {
        Value::Object(object) if object.contains_key("$fixture_table") => {
            assert_eq!(object.len(), 1);
            let index = object["$fixture_table"].as_u64().unwrap() as usize;
            assert!(index < before, "invalid or cyclic fixture table reference");
            let table = &tables[index];
            let fields = table["fields"].as_array().unwrap();
            let columns = table["values"].as_array().unwrap();
            assert_eq!(fields.len(), columns.len());
            Value::Array(
                table["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        let cells = row.as_array().unwrap();
                        assert_eq!(cells.len(), fields.len());
                        let mut record = Map::new();
                        for ((field, column), cell) in fields.iter().zip(columns).zip(cells) {
                            let cell = cell.as_i64().unwrap();
                            assert!(cell >= -1);
                            if cell >= 0 {
                                let value = &column.as_array().unwrap()[cell as usize];
                                assert!(
                                    record
                                        .insert(
                                            field.as_str().unwrap().to_owned(),
                                            expand(value, tables, index)
                                        )
                                        .is_none()
                                );
                            }
                        }
                        Value::Object(record)
                    })
                    .collect(),
            )
        }
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), expand(value, tables, before)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|value| expand(value, tables, before))
                .collect(),
        ),
        value => value.clone(),
    }
}
