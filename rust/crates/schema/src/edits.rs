// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The edits a host makes on its own objects of a schema: a column, a table
//! held on its own, or a whole schema (`sdm_client/schema.py`'s setters,
//! constructors and checks).
//!
//! The host keeps the objects a caller holds and edits; the core decides
//! what an edit does to them. A host hands the core what its objects say
//! (the *scope*: one column, one table, or a schema) and the edit, and is
//! given back what they say afterwards, and the warnings, or the refusal.
//!
//! An edit, as JSON:
//!
//! | `op` | Is |
//! |---|---|
//! | `column_init` | `Column(name, dtype, stype)`: `name`, `dtype` and `stype` as given |
//! | `set_column` | `setattr(column, attribute, value)`, for `stype` or `untyped`; `table` and `column` name it |
//! | `set_table` | `setattr(table, attribute, value)`, for `primary_key` or `time_column` |
//! | `set_links` | `schema.relationships = links`: each link four names, or a value that is not a `Relationship` (`{"$object": ...}`) |
//! | `mark_keys` | which columns the schema joins on, derived again |
//! | `check_consistent` | `check_consistent()` of the scope |
//! | `table_init` | a table's constructor: `name`, `source_name` (a table in its source), `columns` (the names), and its frame's column names (`frame_columns`) |
//! | `delete_column` | `del table[column]` of a table in its source |

use serde_json::Value;

use crate::client::{
    self, Key, R, Refusal, Rel, SEMANTIC_TYPES, Said, Schema, Table, check_name, check_stype,
    json_repr, key_names, key_of_json, supports_dtype,
};
use crate::frame::Frame;
use crate::py::{self, str_repr};

/// What an edit is made on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// A column held on its own: the schema's one table, named `""`, holds it.
    Column,
    /// A table held on its own: the schema's one table, with no links.
    Table,
    /// A schema.
    Schema,
}

impl Scope {
    pub fn parse(name: &str) -> Option<Scope> {
        match name {
            "column" => Some(Scope::Column),
            "table" => Some(Scope::Table),
            "schema" => Some(Scope::Schema),
            _ => None,
        }
    }
}

/// `repr` of a name a caller gave, which need not be text.
fn name_repr(value: &Value) -> String {
    match value {
        Value::String(s) => str_repr(s),
        other => json_repr(other),
    }
}

fn sorted_semantic_types() -> String {
    let mut sorted: Vec<&str> = SEMANTIC_TYPES.to_vec();
    sorted.sort_unstable();
    py::str_list_repr(&sorted)
}

/// `Column._check_stype(value)`, of a value that need not be text.
fn check_stype_value(value: &Value) -> R<String> {
    match value {
        Value::String(text) => {
            check_stype(text)?;
            Ok(text.clone())
        }
        other => Err(Refusal::invalid(format!(
            "{} is not a semantic type; use one of {}",
            json_repr(other),
            sorted_semantic_types()
        ))),
    }
}

/// `Column._check_supports_dtype(stype, dtype)`.
fn check_supports(name: &Value, stype: &str, dtype: &Value) -> R<()> {
    let Value::String(dtype) = dtype else {
        // No dtype, or one that is not a wire dtype's name: nothing to hold
        // the meaning against.
        return Ok(());
    };
    if !supports_dtype(stype, dtype) {
        return Err(Refusal::invalid(format!(
            "column {} holds {}, which cannot mean {}",
            name_repr(name),
            str_repr(dtype),
            str_repr(stype)
        )));
    }
    Ok(())
}

fn text_of<'a>(op: &'a Value, key: &str) -> &'a str {
    op.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn field<'a>(op: &'a Value, key: &str) -> &'a Value {
    op.get(key).unwrap_or(&Value::Null)
}

/// The edit `op` made on `schema`, the scope a host handed over.
pub fn apply(schema: &mut Schema, scope: Scope, op: &Value, said: &mut Said) -> R<()> {
    match text_of(op, "op") {
        "column_init" => column_init(op),
        "set_column" => set_column(schema, scope, op, said),
        "set_table" => set_table(schema, scope, op, said),
        "set_links" => set_links(schema, op, said),
        "mark_keys" => schema.mark_key_columns(said),
        "check_consistent" => check_consistent(schema, scope, op, said),
        "table_init" => table_init(schema, op),
        "delete_column" => delete_column(schema, op),
        other => Err(Refusal::class(
            "ValueError",
            format!("no schema edit {}", str_repr(other)),
        )),
    }
}

/// `Column(name, dtype, stype)`: the `stype` assignment of its `__init__`,
/// checked against the `dtype` assigned before it, then `__post_init__`.
fn column_init(op: &Value) -> R<()> {
    let stype = check_stype_value(field(op, "stype"))?;
    check_supports(field(op, "name"), &stype, field(op, "dtype"))?;
    Ok(())
}

fn table_at(schema: &Schema, op: &Value) -> R<usize> {
    let name = text_of(op, "table");
    client::table_index(&schema.tables, name).ok_or_else(|| {
        Refusal::invalid(format!(
            "no table {} in this schema; it has {}",
            str_repr(name),
            py::str_list_repr(&client::sorted_names(schema.tables.iter().map(|t| &t.name)))
        ))
    })
}

fn column_at(table: &Table, op: &Value) -> R<usize> {
    let name = field(op, "column");
    name.as_str()
        .and_then(|n| table.col_index(n))
        .ok_or_else(|| {
            Refusal::invalid(format!(
                "table {} has no column {}; it has {}",
                str_repr(&table.name),
                name_repr(name),
                table.sorted_columns()
            ))
        })
}

/// `setattr(column, attribute, value)`.
fn set_column(schema: &mut Schema, scope: Scope, op: &Value, said: &mut Said) -> R<()> {
    let attribute = text_of(op, "attribute");
    let value = field(op, "value");
    if attribute == "untyped" {
        return Err(Refusal::invalid(
            "'untyped' is inferred from the values and cannot be set; assign the dtype you want instead",
        ));
    }
    if attribute != "stype" {
        return Ok(());
    }
    if scope == Scope::Schema {
        let table = text_of(op, "table").to_string();
        let column = text_of(op, "column").to_string();
        let ti = table_at(schema, op)?;
        column_at(&schema.tables[ti], op)?;
        // A value that is not text is refused as the check refuses it.
        let stype = check_stype_value(value)?;
        return schema.edit(&table, &column, "stype", &Value::String(stype), said);
    }
    let ti = table_at(schema, op)?;
    let ci = column_at(&schema.tables[ti], op)?;
    let stype = check_stype_value(value)?;
    if scope == Scope::Table {
        return schema.tables[ti].set_stype_alone(ci, &stype);
    }
    // A column on its own: the checks, the record that its meaning was
    // chosen, and the meaning; there is no table to order and no key to
    // stay.
    let column = &mut schema.tables[ti].columns[ci];
    check_supports(
        &Value::String(column.name.clone()),
        &stype,
        &Value::String(column.dtype.clone()),
    )?;
    column.stype_declared = true;
    column.stype.clone_from(&stype);
    if stype == "ID" {
        column.before_key = None;
    }
    Ok(())
}

/// `setattr(table, attribute, value)`, for the two attributes a table
/// derives the rest of itself from.
fn set_table(schema: &mut Schema, scope: Scope, op: &Value, said: &mut Said) -> R<()> {
    let attribute = text_of(op, "attribute");
    let value = field(op, "value");
    let ti = table_at(schema, op)?;
    match attribute {
        "primary_key" => {
            let key = key_of_json(value)?;
            if scope == Scope::Schema {
                let name = schema.tables[ti].name.clone();
                return schema.edit_table(&name, "primary_key", &key_value(&key), said);
            }
            schema.tables[ti].set_primary_key_alone(key, said)
        }
        "time_column" => {
            let name = match value {
                Value::Null => {
                    schema.tables[ti].time_column = None;
                    return Ok(());
                }
                Value::String(name) => name.clone(),
                other => {
                    let table = &schema.tables[ti];
                    return Err(Refusal::invalid(format!(
                        "table {} has no column {}; it has {}",
                        str_repr(&table.name),
                        json_repr(other),
                        table.sorted_columns()
                    )));
                }
            };
            if scope != Scope::Schema {
                return schema.tables[ti].set_time_column_alone(Some(&name));
            }
            let table = &schema.tables[ti];
            table.check_column(&name)?;
            if key_names(&table.primary_key).contains(&name) {
                return Err(Refusal::invalid(format!(
                    "column {} is already the primary key of table {}, so it cannot also be its time column",
                    str_repr(&name),
                    str_repr(&table.name)
                )));
            }
            let table_name = table.name.clone();
            schema.edit(
                &table_name,
                &name,
                "stype",
                &Value::String("timestamp".into()),
                said,
            )?;
            schema.tables[ti].time_column = Some(name);
            Ok(())
        }
        other => Err(Refusal::class(
            "ValueError",
            format!("a table's {} is not one the core sets", str_repr(other)),
        )),
    }
}

fn key_value(key: &Option<Key>) -> Value {
    client::key_json(key)
}

/// The links a host handed over: each four names, or what was not a
/// `Relationship`, refused in order.
fn links_of(given: &Value) -> R<Vec<Rel>> {
    let mut out = Vec::new();
    for link in given.as_array().map(Vec::as_slice).unwrap_or_default() {
        let names: Option<Vec<String>> = link.as_array().filter(|p| p.len() == 4).map(|p| {
            p.iter()
                .map(|n| n.as_str().map_or_else(|| json_repr(n), str::to_string))
                .collect()
        });
        match names {
            Some(n) => out.push(Rel::new(&n[0], &n[1], &n[2], &n[3])),
            None => {
                let (type_name, repr) = match crate::pyval::PyVal::decode(link) {
                    Ok(crate::pyval::PyVal::Opaque {
                        type_name, repr, ..
                    }) => (type_name, repr),
                    Ok(other) => (py_type_name(&other), other.repr()),
                    Err(_) => ("object".to_string(), link.to_string()),
                };
                return Err(Refusal::invalid(format!(
                    "a relationship must be a Relationship, got {type_name}: {repr}. Build one with Relationship(source_table, source_column, target_table, target_column)"
                )));
            }
        }
    }
    Ok(out)
}

fn py_type_name(value: &crate::pyval::PyVal) -> String {
    use crate::pyval::PyVal;
    match value {
        PyVal::None => "NoneType",
        PyVal::Bool(_) => "bool",
        PyVal::Int(_) | PyVal::BigInt(_) => "int",
        PyVal::Float(_) => "float",
        PyVal::Str(_) => "str",
        PyVal::List(_) => "list",
        PyVal::Tuple(_) => "tuple",
        PyVal::Dict(_) => "dict",
        _ => "object",
    }
    .to_string()
}

/// `schema.relationships = links`: checked in order, each kept once, the
/// two ways to one table said, and the key columns derived again.
fn set_links(schema: &mut Schema, op: &Value, said: &mut Said) -> R<()> {
    let given = field(op, "links");
    // Checked one at a time, in order, so a link that is not a
    // `Relationship` is refused where it stands among the others.
    let mut checked = Vec::new();
    for link in given.as_array().map(Vec::as_slice).unwrap_or_default() {
        let one = links_of(&Value::Array(vec![link.clone()]))?;
        client::check_relationships(&schema.tables, &one)?;
        checked.extend(one);
    }
    schema.set_relationships(&checked, said)
}

/// `check_consistent()`: of a schema, or of a column on its own.
fn check_consistent(schema: &mut Schema, scope: Scope, op: &Value, said: &mut Said) -> R<()> {
    if scope == Scope::Schema {
        return schema.check_consistent(said);
    }
    let ti = table_at(schema, op)?;
    let ci = column_at(&schema.tables[ti], op)?;
    let column = &schema.tables[ti].columns[ci];
    if op.get("dtype_none").and_then(Value::as_bool) == Some(true)
        || op.get("stype_none").and_then(Value::as_bool) == Some(true)
    {
        return Ok(());
    }
    check_supports(
        &Value::String(column.name.clone()),
        &column.stype,
        &Value::String(column.dtype.clone()),
    )
}

/// `_check_name(name)`, of a name that need not be text.
fn check_name_value(name: &Value) -> R<()> {
    match name {
        Value::String(text) => check_name(text),
        other => Err(Refusal::invalid(format!(
            "The name {} must be a non-empty string.",
            json_repr(other)
        ))),
    }
}

/// A table's constructor: its names, and the cells of its frame.
fn table_init(schema: &Schema, op: &Value) -> R<()> {
    let name = field(op, "name");
    check_name_value(name)?;
    let shown = name
        .as_str()
        .map_or_else(|| json_repr(name), str::to_string);
    if let Some(source_name) = op.get("source_name") {
        let usable = source_name.as_str().is_some_and(|s| !s.is_empty());
        if !usable {
            return Err(Refusal::invalid(format!(
                "table {} needs the name it has in its source, got {}",
                name_repr(name),
                json_repr(source_name)
            )));
        }
        for column in op
            .get("columns")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            check_name_value(column)?;
        }
        return Ok(());
    }
    let frame: Option<&Frame> = schema.tables.first().and_then(|t| t.data.as_deref());
    for column in op
        .get("frame_columns")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        check_name_value(column)?;
        let Some(text) = column.as_str() else {
            continue;
        };
        if let Some(cells) = frame.and_then(|f| f.column(text)) {
            client::require_scalar_cells(cells, &format!("{shown}.{text}"))?;
        }
    }
    Ok(())
}

/// `del table[column]`, of a table in its source.
fn delete_column(schema: &mut Schema, op: &Value) -> R<()> {
    let ti = table_at(schema, op)?;
    let table = &schema.tables[ti];
    let ci = column_at(table, op)?;
    let column = table.columns[ci].name.clone();
    let role = if key_names(&table.primary_key).contains(&column) {
        Some("primary key")
    } else if table.time_column.as_deref() == Some(column.as_str()) {
        Some("time column")
    } else {
        None
    };
    if let Some(role) = role {
        return Err(Refusal::invalid(format!(
            "{}.{column} is the table's {role}; set that to something else, or None, before leaving the column out",
            table.name
        )));
    }
    schema.tables[ti].columns.remove(ci);
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::client::Col;

    fn col(name: &str, dtype: &str, stype: &str) -> Col {
        Col {
            name: name.into(),
            name_repr: None,
            dtype: dtype.into(),
            stype: stype.into(),
            untyped: false,
            stype_declared: false,
            before_key: None,
        }
    }

    fn one_table(columns: Vec<Col>) -> Schema {
        Schema {
            tables: vec![Table {
                name: "users".into(),
                data: None,
                source_name: None,
                columns,
                primary_key: None,
                time_column: None,
                duplicate_keys: Some(0),
            }],
            relationships: Vec::new(),
        }
    }

    #[test]
    fn a_column_built_with_a_meaning_its_values_cannot_carry_is_refused() {
        let error = column_init(&json!({"name": "age", "dtype": "int64", "stype": "timestamp"}))
            .expect_err("refused");
        assert_eq!(
            error.message,
            "column 'age' holds 'int64', which cannot mean 'timestamp'"
        );
        let error = column_init(&json!({"name": "age", "dtype": "int64", "stype": 5}))
            .expect_err("refused");
        assert!(error.message.starts_with("5 is not a semantic type"));
        column_init(&json!({"name": "age", "dtype": null, "stype": "timestamp"}))
            .expect("no dtype");
    }

    #[test]
    fn a_timestamp_named_on_a_table_of_a_schema_orders_it() {
        let mut schema = one_table(vec![
            col("user_id", "int64", "ID"),
            col("at", "timestamp[us]", "timestamp"),
        ]);
        let mut said = Said::default();
        apply(
            &mut schema,
            Scope::Schema,
            &json!({"op": "set_table", "table": "users", "attribute": "time_column", "value": "at"}),
            &mut said,
        )
        .expect("set");
        assert_eq!(schema.tables[0].time_column.as_deref(), Some("at"));
        assert!(schema.tables[0].columns[1].stype_declared);
    }

    #[test]
    fn a_link_that_is_not_a_relationship_is_refused_in_order() {
        let mut schema = one_table(vec![col("user_id", "int64", "ID")]);
        let mut said = Said::default();
        let error = apply(
            &mut schema,
            Scope::Schema,
            &json!({"op": "set_links", "links": [
                ["users", "nope", "users", "user_id"],
                {"$object": {"type": "tuple", "repr": "('a',)"}},
            ]}),
            &mut said,
        )
        .expect_err("refused");
        assert!(error.message.contains("has no column 'nope'"), "{error:?}");
        let error = apply(
            &mut schema,
            Scope::Schema,
            &json!({"op": "set_links", "links": [{"$object": {"type": "tuple", "repr": "('a',)"}}]}),
            &mut said,
        )
        .expect_err("refused");
        assert!(
            error
                .message
                .starts_with("a relationship must be a Relationship, got tuple: ('a',)"),
            "{error:?}"
        );
    }

    #[test]
    fn a_key_column_of_a_table_in_its_source_is_not_left_out() {
        let mut schema = one_table(vec![
            col("user_id", "int64", "ID"),
            col("age", "int64", "numerical"),
        ]);
        schema.tables[0].primary_key = Some(Key::One("user_id".into()));
        let error = delete_column(&mut schema, &json!({"table": "users", "column": "user_id"}))
            .expect_err("refused");
        assert!(
            error
                .message
                .starts_with("users.user_id is the table's primary key")
        );
        delete_column(&mut schema, &json!({"table": "users", "column": "age"})).expect("left out");
        assert_eq!(schema.tables[0].columns.len(), 1);
    }
}
