// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The schema a query is resolved against (`predictive_query.schema`):
//! tables, their columns and keys, and the groups of columns that link them.

use serde_json::Value;

use crate::json::Json;
use crate::types::{Dtype, Stype, TimestampUnit};

/// A column of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub stype: Stype,
    pub dtype: Dtype,
    /// How a timestamp column is written: a [`TimestampUnit`]'s value or a
    /// format string.
    pub timestamp_format: Option<String>,
}

/// The columns of a table, and which of them carry its identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableDefinition {
    pub cols: Vec<Column>,
    pub pkey: Option<String>,
    pub time_col: Option<String>,
    pub end_time_col: Option<String>,
}

/// A column within a schema.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnKey {
    pub table_name: String,
    pub col_name: String,
}

impl ColumnKey {
    pub fn new(table_name: &str, col_name: &str) -> Self {
        ColumnKey {
            table_name: table_name.to_string(),
            col_name: col_name.to_string(),
        }
    }
}

/// Columns linked together, such as a primary key and its foreign keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnKeyGroup {
    pub columns: Vec<ColumnKey>,
}

/// An edge between two tables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Relationship {
    pub src_table: String,
    pub fkey: String,
    pub dst_table: String,
}

/// A schema: its tables, in the order given, and the groups linking them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaDefinition {
    pub tables: Vec<(String, TableDefinition)>,
    pub col_groups: Vec<ColumnKeyGroup>,
}

impl SchemaDefinition {
    pub fn table(&self, name: &str) -> Option<&TableDefinition> {
        self.tables.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    pub fn has_table(&self, name: &str) -> bool {
        self.table(name).is_some()
    }

    /// The table, which the caller has already found in the schema.
    pub fn table_or_empty(&self, name: &str) -> TableDefinition {
        self.table(name).cloned().unwrap_or(TableDefinition {
            cols: Vec::new(),
            pkey: None,
            time_col: None,
            end_time_col: None,
        })
    }

    /// Every column of every group, in order.
    pub fn all_keys(&self) -> Vec<&ColumnKey> {
        self.col_groups
            .iter()
            .flat_map(|g| g.columns.iter())
            .collect()
    }

    /// The schema as `json_serde` reads it, refusing what pydantic would.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("a schema is an object with tables and col_groups")?;
        let tables = object
            .get("tables")
            .and_then(Value::as_object)
            .ok_or("a schema's tables are an object of table definitions")?;
        let mut out = Vec::new();
        for (name, table) in tables {
            out.push((
                name.clone(),
                table_from(table).map_err(|e| format!("table {name}: {e}"))?,
            ));
        }
        let groups = object
            .get("col_groups")
            .and_then(Value::as_array)
            .ok_or("a schema's col_groups are a list")?;
        let col_groups = groups
            .iter()
            .map(group_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SchemaDefinition {
            tables: out,
            col_groups,
        })
    }

    /// The schema as `json_serde.to_json_dict` writes it.
    pub fn to_json(&self) -> Json {
        let opt = |v: &Option<String>| Json::opt_str(v.clone());
        Json::object(vec![
            (
                "tables",
                Json::Object(
                    self.tables
                        .iter()
                        .map(|(name, t)| {
                            (
                                name.clone(),
                                Json::object(vec![
                                    (
                                        "cols",
                                        Json::Array(
                                            t.cols
                                                .iter()
                                                .map(|c| {
                                                    Json::object(vec![
                                                        ("name", Json::str(&c.name)),
                                                        ("stype", Json::str(c.stype.value())),
                                                        ("dtype", Json::str(c.dtype.value())),
                                                        (
                                                            "timestamp_format",
                                                            opt(&c.timestamp_format),
                                                        ),
                                                    ])
                                                })
                                                .collect(),
                                        ),
                                    ),
                                    ("pkey", opt(&t.pkey)),
                                    ("time_col", opt(&t.time_col)),
                                    ("end_time_col", opt(&t.end_time_col)),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "col_groups",
                Json::Array(
                    self.col_groups
                        .iter()
                        .map(|g| {
                            Json::object(vec![(
                                "columns",
                                Json::Array(
                                    g.columns
                                        .iter()
                                        .map(|k| {
                                            Json::object(vec![
                                                ("table_name", Json::str(&k.table_name)),
                                                ("col_name", Json::str(&k.col_name)),
                                            ])
                                        })
                                        .collect(),
                                ),
                            )])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

fn text(value: &Value, what: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{what} is a string"))
}

/// An optional name, where `""` means none (`empty_str_to_none`).
fn optional_name(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<String>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("{key} is a string or null")),
    }
}

fn table_from(value: &Value) -> Result<TableDefinition, String> {
    let object = value.as_object().ok_or("a table is an object")?;
    let cols = object
        .get("cols")
        .and_then(Value::as_array)
        .ok_or("a table's cols are a list")?
        .iter()
        .map(column_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TableDefinition {
        cols,
        pkey: optional_name(object, "pkey")?,
        time_col: optional_name(object, "time_col")?,
        end_time_col: optional_name(object, "end_time_col")?,
    })
}

fn column_from(value: &Value) -> Result<Column, String> {
    let object = value.as_object().ok_or("a column is an object")?;
    let name = text(
        object.get("name").unwrap_or(&Value::Null),
        "a column's name",
    )?;
    let stype_text = text(object.get("stype").unwrap_or(&Value::Null), "stype")?;
    let dtype_text = text(object.get("dtype").unwrap_or(&Value::Null), "dtype")?;
    let stype = Stype::parse(&stype_text).ok_or_else(|| format!("unknown stype {stype_text:?}"))?;
    let dtype = Dtype::parse(&dtype_text).ok_or_else(|| format!("unknown dtype {dtype_text:?}"))?;
    let timestamp_format = match object.get("timestamp_format") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => {
            Some(TimestampUnit::parse(s).map_or_else(|| s.clone(), |u| u.value().to_string()))
        }
        Some(_) => return Err("timestamp_format is a string or null".to_string()),
    };
    Ok(Column {
        name,
        stype,
        dtype,
        timestamp_format,
    })
}

fn group_from(value: &Value) -> Result<ColumnKeyGroup, String> {
    let columns = value
        .get("columns")
        .and_then(Value::as_array)
        .ok_or("a column group's columns are a list")?;
    let columns = columns
        .iter()
        .map(|key| {
            Ok(ColumnKey {
                table_name: text(key.get("table_name").unwrap_or(&Value::Null), "table_name")?,
                col_name: text(key.get("col_name").unwrap_or(&Value::Null), "col_name")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ColumnKeyGroup { columns })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_writes_a_schema() {
        let text = r#"{"tables": {"t": {"cols": [{"name": "id", "stype": "ID", "dtype": "int",
            "timestamp_format": null}, {"name": "at", "stype": "timestamp",
            "dtype": "time", "timestamp_format": "ms"}], "pkey": "id", "time_col": "", "end_time_col": null}},
            "col_groups": [{"columns": [{"table_name": "t", "col_name": "id"}]}]}"#;
        let schema = SchemaDefinition::from_json(text).unwrap();
        assert_eq!(schema.table("t").unwrap().time_col, None);
        assert_eq!(schema.all_keys().len(), 1);
        let at = &schema.table("t").unwrap().cols[1];
        assert_eq!(
            at.timestamp_format.as_deref(),
            Some(TimestampUnit::Millisecond.value())
        );
        let back = schema.to_json().render();
        assert!(back.contains("\"time_col\": null"));
        for bad in ["{not json", "[]", "{\"tables\": 3}", "{\"tables\": {}}"] {
            assert!(SchemaDefinition::from_json(bad).is_err(), "{bad}");
        }
    }
}
