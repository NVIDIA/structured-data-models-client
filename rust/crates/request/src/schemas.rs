// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A schema as the request code holds one between calls.
//!
//! A schema over frames is held as today's `Schema` holds one: what it says
//! (every table's columns, keys and time column, and the relationships) as it
//! was described when it was made and edited, and the caller's frames, read
//! as they stand at each call. A frame changed after the schema was made
//! is read changed, and the schema's description is not made again from it.
//!
//! JSON form: `{"kind": "data", "tables": [NAME, ...], "options": {...},
//! "schema": SCHEMA}`, `options` the arguments and edits of
//! `Schema.from_data` and `SCHEMA` what [`schema_json`] writes; the frames
//! are named `schema/NAME` in the tables a call is given.

use std::sync::Arc;

use arrow_array::RecordBatch;
use sdm_protocol::Diagnostic;
use sdm_schema::client::{self, Said, Schema};
use sdm_schema::dt::Host;
use sdm_schema::frame::Frame;
use serde_json::{Value, json};

use crate::error::{R, STRUCTURED};
use crate::frames::Table;

/// The tables a call is given, by name, with their tags.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    pub tables: Vec<(String, Table)>,
}

impl Tables {
    /// The tables `batches` with their tags from `tags`, an object of names
    /// to each table's tags.
    pub fn new(batches: Vec<(String, RecordBatch)>, tags: &Value) -> Self {
        Self {
            tables: batches
                .into_iter()
                .map(|(name, batch)| {
                    let table = Table::new(batch, tags.get(&name));
                    (name, table)
                })
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Table> {
        self.tables.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    pub fn insert(&mut self, name: &str, table: Table) {
        match self.tables.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = table,
            None => self.tables.push((name.to_string(), table)),
        }
    }

    /// Every table's tags, by name.
    pub fn tags(&self) -> Value {
        Value::Object(
            self.tables
                .iter()
                .map(|(n, t)| (n.clone(), t.tags_json()))
                .collect(),
        )
    }
}

/// A refusal of the schema core as the class today's code raises it.
pub fn schema_refusal(refusal: &client::Refusal) -> Diagnostic {
    let code = refusal.code.as_str();
    let is_code = code
        .chars()
        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit());
    let made = if is_code {
        Diagnostic::refusal(STRUCTURED, Some(code), refusal.message.clone())
    } else {
        Diagnostic::refusal(&refusal.raised_as(), None, refusal.message.clone())
    };
    made.with_details((*refusal.details).clone())
        .caused_by(refusal.chain.iter().cloned())
}

/// What describing a schema over frames gave: the schema and what it said,
/// or what it was refused with and what it said before.
pub type Described = Result<(Schema, Said), (client::Refusal, Said)>;

/// A schema as held between calls: over frames, or over tables in a source.
#[derive(Clone, Debug)]
pub struct DataSchema {
    /// The tables, in the order `Schema.from_data` was given them.
    pub names: Vec<String>,
    /// `Schema.from_data`'s arguments and the edits after it.
    pub options: Value,
    /// What the schema over frames says, as it was described from them and
    /// edited; `None` until [`DataSchema::described`] has described it.
    pub described: Option<Schema>,
    /// A schema over a source: what `Schema.from_source` described, and the
    /// engine tables it read.
    pub source: Option<Box<SourceSchema>>,
}

/// A schema over tables in a source, as held between calls.
#[derive(Clone, Debug)]
pub struct SourceSchema {
    /// The source: its backend, namespace and semantic model.
    pub source: Value,
    pub schema: Schema,
    /// `schema._engine_tables`: each table read, by its name in the source.
    pub engine: Vec<(String, sdm_schema::engine::EngineTable)>,
}

impl SourceSchema {
    /// The source's backend, as the engine names it.
    pub fn backend(&self) -> Result<sdm_schema::engine::Backend, String> {
        let name = self
            .source
            .get("backend")
            .and_then(Value::as_str)
            .ok_or("a source names its backend")?;
        sdm_schema::engine::Backend::parse(name).ok_or_else(|| format!("no backend {name:?}"))
    }

    fn to_json(&self) -> Value {
        json!({
            "kind": "source",
            "source": self.source,
            "schema": schema_json(&self.schema),
            "engine": self.engine.iter().map(|(n, t)| {
                // A composite key's fold is checked against the rows its
                // types were read from, so those are kept for it.
                let composite = self.schema.tables.iter().any(|s| {
                    s.source_name.as_deref().unwrap_or(&s.name) == n
                        && matches!(&s.primary_key, Some(sdm_schema::client::Key::Many(k)) if k.len() > 1)
                });
                json!([n, if composite { t.to_json() } else { t.to_json_without_sample() }])
            }).collect::<Vec<_>>(),
        })
    }

    fn from_json(value: &Value) -> Result<Self, String> {
        Ok(SourceSchema {
            source: value.get("source").cloned().unwrap_or(Value::Null),
            schema: schema_of(value.get("schema").unwrap_or(&Value::Null))?,
            engine: value
                .get("engine")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .map(|pair| {
                            Ok((
                                pair.get(0)
                                    .and_then(Value::as_str)
                                    .ok_or("an engine table's name")?
                                    .to_string(),
                                sdm_schema::engine::EngineTable::from_json(
                                    pair.get(1).unwrap_or(&Value::Null),
                                )?,
                            ))
                        })
                        .collect::<Result<_, String>>()
                })
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

fn key_json(key: &Option<sdm_schema::client::Key>) -> Value {
    match key {
        None => Value::Null,
        Some(sdm_schema::client::Key::One(name)) => json!(name),
        Some(sdm_schema::client::Key::Many(names)) => json!(names),
    }
}

fn key_of(value: Option<&Value>) -> Option<sdm_schema::client::Key> {
    match value {
        Some(Value::String(name)) => Some(sdm_schema::client::Key::One(name.clone())),
        Some(Value::Array(names)) => Some(sdm_schema::client::Key::Many(
            names
                .iter()
                .filter_map(|n| n.as_str().map(str::to_string))
                .collect(),
        )),
        _ => None,
    }
}

/// A schema over a source as plain JSON, every column's description kept.
pub fn schema_json(schema: &Schema) -> Value {
    json!({
        "tables": schema.tables.iter().map(|t| json!({
            "name": t.name,
            "source_name": t.source_name,
            "columns": t.columns.iter().map(|c| {
                let mut column = json!({
                    "name": c.name,
                    "dtype": c.dtype,
                    "stype": c.stype,
                    "untyped": c.untyped,
                    "stype_declared": c.stype_declared,
                    "before_key": c.before_key,
                });
                if let Some(repr) = &c.name_repr {
                    column["name_repr"] = json!(repr);
                }
                column
            }).collect::<Vec<_>>(),
            "primary_key": key_json(&t.primary_key),
            "time_column": t.time_column,
            "duplicate_keys": t.duplicate_keys,
        })).collect::<Vec<_>>(),
        "relationships": schema.relationships.iter().map(|r| json!([
            r.source_table, r.source_column, r.target_table, r.target_column
        ])).collect::<Vec<_>>(),
    })
}

/// [`schema_json`] read back.
pub fn schema_of(value: &Value) -> Result<Schema, String> {
    let text = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_string);
    let tables = value
        .get("tables")
        .and_then(Value::as_array)
        .ok_or("a schema has tables")?
        .iter()
        .map(|t| {
            Ok(sdm_schema::client::Table {
                name: text(t.get("name")).ok_or("a table has a name")?,
                data: None,
                source_name: text(t.get("source_name")),
                columns: t
                    .get("columns")
                    .and_then(Value::as_array)
                    .ok_or("a table has columns")?
                    .iter()
                    .map(|c| {
                        let flag = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(false);
                        Ok(sdm_schema::client::Col {
                            name: text(c.get("name")).ok_or("a column has a name")?,
                            dtype: text(c.get("dtype")).unwrap_or_default(),
                            stype: text(c.get("stype")).unwrap_or_default(),
                            untyped: flag("untyped"),
                            stype_declared: flag("stype_declared"),
                            before_key: text(c.get("before_key")),
                            name_repr: text(c.get("name_repr")),
                        })
                    })
                    .collect::<Result<_, String>>()?,
                primary_key: key_of(t.get("primary_key")),
                time_column: text(t.get("time_column")),
                duplicate_keys: t.get("duplicate_keys").and_then(Value::as_i64),
            })
        })
        .collect::<Result<_, String>>()?;
    let relationships = value
        .get("relationships")
        .and_then(Value::as_array)
        .map(|rels| {
            rels.iter()
                .filter_map(|r| {
                    let part = |i: usize| r.get(i).and_then(Value::as_str);
                    Some(sdm_schema::client::Rel::new(
                        part(0)?,
                        part(1)?,
                        part(2)?,
                        part(3)?,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Schema {
        tables,
        relationships,
    })
}

impl DataSchema {
    pub fn to_json(&self) -> Value {
        if let Some(source) = &self.source {
            return source.to_json();
        }
        self.data_json()
    }

    fn data_json(&self) -> Value {
        let mut held = json!({"kind": "data", "tables": self.names, "options": self.options});
        if let Some(described) = &self.described {
            held["schema"] = schema_json(described);
        }
        held
    }

    /// The JSON form a request carries: a schema over a source without the
    /// engine tables, which only gathering reads.
    pub fn to_request_json(&self) -> Value {
        if let Some(source) = &self.source {
            let mut slim = (**source).clone();
            slim.engine = Vec::new();
            return slim.to_json();
        }
        self.data_json()
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        if value.get("kind").and_then(Value::as_str) == Some("source") {
            let source = SourceSchema::from_json(value)?;
            return Ok(DataSchema {
                names: source
                    .schema
                    .tables
                    .iter()
                    .map(|t| t.name.clone())
                    .collect(),
                options: json!({}),
                described: None,
                source: Some(Box::new(source)),
            });
        }
        if value.get("kind").and_then(Value::as_str) != Some("data") {
            return Err("a schema over frames is of kind data".to_string());
        }
        let names = value
            .get("tables")
            .and_then(Value::as_array)
            .ok_or("a schema names its tables")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("a table name is text"))
            .collect::<Result<_, _>>()?;
        let described = match value.get("schema") {
            None | Some(Value::Null) => None,
            Some(said) => Some(schema_of(said)?),
        };
        Ok(Self {
            names,
            options: value.get("options").cloned().unwrap_or_else(|| json!({})),
            described,
            source: None,
        })
    }

    /// A schema over the frames `names`, to be described from `options`,
    /// `Schema.from_data`'s arguments and the edits after it.
    pub fn over_frames(names: Vec<String>, options: Value) -> Self {
        Self {
            names,
            options,
            described: None,
            source: None,
        }
    }

    /// The name the frame of table `name` is given among a call's tables.
    pub fn frame_key(name: &str) -> String {
        format!("schema/{name}")
    }

    /// The frames, by table name.
    pub fn frames<'a>(&self, tables: &'a Tables) -> Result<Vec<(String, &'a Table)>, String> {
        self.names
            .iter()
            .map(|name| {
                tables
                    .get(&Self::frame_key(name))
                    .map(|t| (name.clone(), t))
                    .ok_or_else(|| format!("no frame was given for table {name}"))
            })
            .collect()
    }

    /// Every frame, by table name, as the schema core reads one.
    fn schema_frames(&self, tables: &Tables) -> Result<Vec<(String, Arc<Frame>)>, String> {
        self.frames(tables)?
            .into_iter()
            .map(|(name, table)| Ok((name, Arc::new(table.schema_frame()?))))
            .collect()
    }

    /// `Schema.from_data(frames, **options)` and the edits after it, made
    /// from the frames in `tables`.
    pub fn describe(&self, tables: &Tables, host: &Host) -> Result<Described, String> {
        let frames = self.schema_frames(tables)?;
        let given = sdm_schema::binding::data_args(&self.options)?;
        let mut said = Said::default();
        let made =
            client::from_data(frames, &given.args, host, &mut said).and_then(|mut schema| {
                for edit in &given.edits {
                    edit.apply(&mut schema, &mut said)?;
                }
                Ok(schema)
            });
        Ok(match made {
            Ok(schema) => Ok((schema, said)),
            Err(refusal) => Err((refusal, said)),
        })
    }

    /// This schema with what it says described from the frames in `tables`,
    /// as `Schema.from_data` describes it, and the warnings that gave.
    pub fn described(self, tables: &Tables, host: &Host) -> R<(Self, Said)> {
        if self.source.is_some() {
            return Ok((self, Said::default()));
        }
        let (schema, said) = self
            .describe(tables, host)
            .map_err(crate::error::value_error)?
            .map_err(|(refusal, _)| schema_refusal(&refusal))?;
        Ok((self.saying(schema), said))
    }

    /// This schema, saying what `schema` says; its rows are not kept.
    pub fn saying(mut self, mut schema: Schema) -> Self {
        for table in &mut schema.tables {
            table.data = None;
        }
        self.described = Some(schema);
        self
    }

    /// `edit` made on this schema over the frames in `tables`, as an
    /// assignment, `del` or `append` on today's makes it, and the warnings
    /// it gave.
    pub fn edited(&self, edit: Value, tables: &Tables, host: &Host) -> R<(Self, Said)> {
        let mut schema = self.build(tables, host)?;
        let made = sdm_schema::binding::data_args(&json!({"edits": [edit.clone()]}))
            .map_err(crate::error::value_error)?;
        let mut said = Said::default();
        for change in &made.edits {
            change
                .apply(&mut schema, &mut said)
                .map_err(|r| schema_refusal(&r))?;
        }
        let mut edited = self.clone();
        if let Some(Value::Array(edits)) = edited.options.get_mut("edits") {
            edits.push(edit);
        } else if let Some(options) = edited.options.as_object_mut() {
            options.insert("edits".into(), json!([edit]));
        }
        Ok((edited.saying(schema), said))
    }

    /// The schema over the frames in `tables`, as they stand: what it says
    /// as it was described, each table's rows those of its frame now; a
    /// schema over a source as it was described.
    ///
    /// A table whose frame is not given has no rows here: it is one a
    /// caller replaced with a `SourceTable` in `schema.tables`, which holds
    /// none, and what reads its rows refuses it as today's code does.
    pub fn build(&self, tables: &Tables, _host: &Host) -> R<Schema> {
        if let Some(source) = &self.source {
            return Ok(source.schema.clone());
        }
        let Some(described) = &self.described else {
            return Err(crate::error::value_error(
                "a schema over frames is described when it is made",
            ));
        };
        let mut schema = described.clone();
        for table in &mut schema.tables {
            if let Some(frame) = tables.get(&Self::frame_key(&table.name)) {
                let frame = frame.schema_frame().map_err(crate::error::value_error)?;
                table.data = Some(Arc::new(frame));
            }
        }
        Ok(schema)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};

    use super::*;

    fn users(columns: &[(&str, ArrayRef)]) -> Tables {
        let batch =
            RecordBatch::try_from_iter(columns.iter().map(|(n, a)| (n.to_string(), Arc::clone(a))))
                .expect("a batch");
        Tables::new(vec![(DataSchema::frame_key("users"), batch)], &json!({}))
    }

    fn ids() -> ArrayRef {
        Arc::new(Int64Array::from(vec![1, 2, 3]))
    }

    /// What the schema says is what it said when it was made; the rows are
    /// the frame's as it stands at each call.
    #[test]
    fn a_held_schema_says_what_it_said_and_reads_the_frame_as_it_stands() {
        let host = Host::assumed();
        let made = users(&[
            ("user_id", ids()),
            ("age", Arc::new(Float64Array::from(vec![20.0, 30.0, 40.0]))),
        ]);
        let options = json!({"primary_keys": null, "time_columns": null,
            "relationships": null, "edits": []});
        let (held, _) = DataSchema::over_frames(vec!["users".into()], options)
            .described(&made, &host)
            .expect("described");
        let held = DataSchema::from_json(&held.to_json()).expect("read back");

        let now = users(&[("user_id", ids())]);
        let schema = held.build(&now, &host).expect("built");
        let table = &schema.tables[0];
        let said: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(said, ["user_id", "age"]);
        assert_eq!(table.columns[1].stype, "numerical");
        let rows = table.data.as_ref().expect("the frame's rows");
        assert!(rows.columns.iter().all(|c| c.name != "age"));
    }

    /// A table whose frame is not given, one a caller replaced with a
    /// `SourceTable`, is built without rows.
    #[test]
    fn a_table_without_its_frame_is_built_without_rows() {
        let host = Host::assumed();
        let made = users(&[("user_id", ids())]);
        let options = json!({"primary_keys": null, "time_columns": null,
            "relationships": null, "edits": []});
        let (held, _) = DataSchema::over_frames(vec!["users".into()], options)
            .described(&made, &host)
            .expect("described");
        let schema = held.build(&Tables::default(), &host).expect("built");
        assert!(schema.tables[0].data.is_none());
        assert_eq!(schema.tables[0].columns.len(), 1);
    }

    #[test]
    fn a_schema_over_frames_is_described_before_it_is_built() {
        let held = DataSchema::over_frames(vec!["users".into()], json!({}));
        let error = held
            .build(&users(&[("user_id", ids())]), &Host::assumed())
            .expect_err("not described");
        assert!(error.message.contains("described when it is made"));
    }
}
