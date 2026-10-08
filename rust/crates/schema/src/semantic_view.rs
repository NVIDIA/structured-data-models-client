// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Snowflake semantic view's YAML, read on its own: its logical tables,
//! the tables behind them, their keys and columns, and its relationships,
//! every name as it spells (a quoted name without its quotes).
//!
//! The engine reads a view through the warehouse
//! ([`crate::engine::views::from_snowflake_semantic_view`]); this is the
//! same reading of the document alone, for a host that wants to show or
//! check a view before it is read.

use serde_json::{Value, json};

use crate::metric_view::{get, type_name};
use crate::pyval::PyVal;
use crate::yaml;

/// Why a document is not a semantic view this can read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub class: String,
    pub message: String,
}

fn refused<T>(class: &str, message: impl Into<String>) -> Result<T, Refused> {
    Err(Refused {
        class: class.to_string(),
        message: message.into(),
    })
}

/// A name as the view spells it: a quoted name is what is inside its
/// quotes, inner quotes undoubled.
pub fn spelled(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() >= 2 && chars[0] == '"' && chars[chars.len() - 1] == '"' {
        let inner: String = chars[1..chars.len() - 1].iter().collect();
        return inner.replace("\"\"", "\"");
    }
    name.to_string()
}

fn field<'a>(value: &'a PyVal, key: &str, at: &str) -> Result<Option<&'a PyVal>, Refused> {
    match value {
        PyVal::Dict(_) => Ok(get(value, &PyVal::Str(key.to_string()))),
        other => refused(
            "ValueError",
            format!("{at} is a mapping, not {}", type_name(other)),
        ),
    }
}

fn required<'a>(value: &'a PyVal, key: &str, at: &str) -> Result<&'a PyVal, Refused> {
    field(value, key, at)?.ok_or_else(|| Refused {
        class: "ValueError".to_string(),
        message: format!("{at} has no {key:?}"),
    })
}

fn name(value: &PyVal, at: &str) -> Result<String, Refused> {
    match value {
        PyVal::Str(s) => Ok(spelled(s)),
        other => refused(
            "ValueError",
            format!("{at} is a name, not {}", type_name(other)),
        ),
    }
}

fn list<'a>(value: Option<&'a PyVal>, at: &str) -> Result<&'a [PyVal], Refused> {
    match value {
        None | Some(PyVal::None) => Ok(&[]),
        Some(PyVal::List(items)) => Ok(items),
        Some(other) => refused(
            "ValueError",
            format!("{at} is a list, not {}", type_name(other)),
        ),
    }
}

fn optional_text(value: Option<&PyVal>, at: &str) -> Result<Option<String>, Refused> {
    match value {
        None | Some(PyVal::None) => Ok(None),
        Some(PyVal::Str(s)) => Ok(Some(s.clone())),
        Some(other) => refused(
            "ValueError",
            format!("{at} is text, not {}", type_name(other)),
        ),
    }
}

fn table(cfg: &PyVal, index: usize) -> Result<Value, Refused> {
    let at = format!("tables[{index}]");
    let table_name = name(required(cfg, "name", &at)?, &format!("{at}.name"))?;
    let base = required(cfg, "base_table", &at)?;
    let base_at = format!("{at}.base_table");
    let part = |key: &str| -> Result<Option<String>, Refused> {
        match field(base, key, &base_at)? {
            None | Some(PyVal::None) => Ok(None),
            Some(v) => name(v, &format!("{base_at}.{key}")).map(Some),
        }
    };
    let (database, schema) = (part("database")?, part("schema")?);
    let Some(source) = part("table")? else {
        return refused("ValueError", format!("{base_at} has no \"table\""));
    };
    let key = match field(cfg, "primary_key", &at)? {
        None | Some(PyVal::None) => Vec::new(),
        Some(key) => {
            let key_at = format!("{at}.primary_key");
            list(
                field(key, "columns", &key_at)?,
                &format!("{key_at}.columns"),
            )?
            .iter()
            .map(|c| name(c, &format!("{key_at}.columns")))
            .collect::<Result<_, _>>()?
        }
    };
    let mut columns = Vec::new();
    for section in ["dimensions", "time_dimensions", "facts"] {
        let section_at = format!("{at}.{section}");
        for (i, column) in list(field(cfg, section, &at)?, &section_at)?
            .iter()
            .enumerate()
        {
            let column_at = format!("{section_at}[{i}]");
            let column_name = name(
                required(column, "name", &column_at)?,
                &format!("{column_at}.name"),
            )?;
            let expr = optional_text(
                field(column, "expr", &column_at)?,
                &format!("{column_at}.expr"),
            )?;
            let data_type = optional_text(
                field(column, "data_type", &column_at)?,
                &format!("{column_at}.data_type"),
            )?;
            columns.push(json!({
                "name": column_name,
                "section": section,
                "expr": expr,
                "data_type": data_type,
            }));
        }
    }
    let metrics: Vec<String> = list(field(cfg, "metrics", &at)?, &format!("{at}.metrics"))?
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let metric_at = format!("{at}.metrics[{i}]");
            name(
                required(m, "name", &metric_at)?,
                &format!("{metric_at}.name"),
            )
        })
        .collect::<Result<_, _>>()?;
    Ok(json!({
        "name": table_name,
        "base_table": {"database": database, "schema": schema, "table": source},
        "primary_key": key,
        "columns": columns,
        "metrics": metrics,
    }))
}

fn relationship(cfg: &PyVal, index: usize) -> Result<Value, Refused> {
    let at = format!("relationships[{index}]");
    let text = |key: &str| -> Result<String, Refused> {
        name(required(cfg, key, &at)?, &format!("{at}.{key}"))
    };
    let pairs_at = format!("{at}.relationship_columns");
    let pairs: Vec<Value> = list(field(cfg, "relationship_columns", &at)?, &pairs_at)?
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let pair_at = format!("{pairs_at}[{i}]");
            Ok(json!([
                name(
                    required(pair, "left_column", &pair_at)?,
                    &format!("{pair_at}.left_column")
                )?,
                name(
                    required(pair, "right_column", &pair_at)?,
                    &format!("{pair_at}.right_column")
                )?,
            ]))
        })
        .collect::<Result<_, Refused>>()?;
    Ok(json!({
        "name": text("name")?,
        "left_table": text("left_table")?,
        "right_table": text("right_table")?,
        "columns": pairs,
    }))
}

/// The view `text` defines, read under the name `view`, as JSON:
/// `{"name", "tables", "relationships"}`.
pub fn parse(text: &str, view: &str) -> Result<Value, Refused> {
    let cfg = yaml::safe_load(text).map_err(|e| Refused {
        class: e.class.to_string(),
        message: e.message,
    })?;
    if !matches!(cfg, PyVal::Dict(_)) {
        return refused(
            "ValueError",
            format!(
                "the semantic view {view:?} is a mapping, not {}",
                type_name(&cfg)
            ),
        );
    }
    let tables: Vec<Value> = list(field(&cfg, "tables", "the view")?, "tables")?
        .iter()
        .enumerate()
        .map(|(i, t)| table(t, i))
        .collect::<Result<_, _>>()?;
    let relationships: Vec<Value> =
        list(field(&cfg, "relationships", "the view")?, "relationships")?
            .iter()
            .enumerate()
            .map(|(i, r)| relationship(r, i))
            .collect::<Result<_, _>>()?;
    let declared = match field(&cfg, "name", "the view")? {
        None | Some(PyVal::None) => None,
        Some(v) => Some(name(v, "name")?),
    };
    Ok(json!({
        "name": declared.unwrap_or_else(|| spelled(view)),
        "tables": tables,
        "relationships": relationships,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_quoted_names_as_they_spell() {
        let text = "name: SV\ntables:\n  - name: '\"orders\"'\n    base_table:\n      database: TEST\n      schema: SHOP\n      table: '\"orders\"'\n    primary_key:\n      columns: ['\"id\"']\n    dimensions:\n      - name: '\"amount\"'\n        expr: '\"orders\".\"amount\"'\n";
        let read = parse(text, "SV").expect("a view");
        assert_eq!(read["tables"][0]["name"], "orders");
        assert_eq!(read["tables"][0]["base_table"]["table"], "orders");
        assert_eq!(read["tables"][0]["primary_key"][0], "id");
        assert_eq!(read["tables"][0]["columns"][0]["name"], "amount");
    }

    #[test]
    fn refuses_what_is_not_a_view() {
        assert_eq!(parse("- 1\n", "SV").unwrap_err().class, "ValueError");
        assert_eq!(parse("tables: 3\n", "SV").unwrap_err().class, "ValueError");
        assert!(parse("tables: [\n", "SV").is_err());
    }
}
