// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use sdm_request::layout::encode_deep;

#[derive(Debug, Default)]
pub(crate) struct Inputs {
    sources: VecDeque<Vec<Plain>>,
    replacements: VecDeque<(String, Plain, Value)>,
    cleanup: VecDeque<String>,
}

impl Inputs {
    pub(crate) fn new(
        sources: Vec<Vec<Plain>>,
        replacements: Vec<(String, Plain, Value)>,
        cleanup: Vec<String>,
    ) -> Self {
        Self {
            sources: sources.into(),
            replacements: replacements.into(),
            cleanup: cleanup.into(),
        }
    }

    pub(super) fn finish(&self) -> Result<(), InputError> {
        if !self.sources.is_empty() || !self.replacements.is_empty() || !self.cleanup.is_empty() {
            return Err(bad("unused carry caller inputs"));
        }
        Ok(())
    }

    pub(super) fn close(&mut self, session: &str) -> bool {
        self.cleanup
            .pop_front()
            .is_some_and(|path| path == format!("/v1/sessions/{session}"))
    }
}

impl Flow {
    fn carry_source(&mut self) -> Result<SchemaH, Result<Diagnostic, InputError>> {
        let arguments = self
            .carry
            .sources
            .pop_front()
            .ok_or_else(|| Err(bad("missing carry source inputs")))?;
        let done = self.call("schema_source_run", arguments).map_err(Ok)?;
        let Plain::Tuple(parts) = done else {
            return Err(Err(bad("source returned no pair")));
        };
        if parts.len() != 2 {
            return Err(Err(bad("source returned no pair")));
        }
        let state: Value = serde_json::from_str(
            parts[0]
                .as_str()
                .ok_or_else(|| Err(bad("source state is not text")))?,
        )
        .map_err(|_| Err(bad("invalid source state")))?;
        let warnings: Value = serde_json::from_str(
            parts[1]
                .as_str()
                .ok_or_else(|| Err(bad("source warnings are not text")))?,
        )
        .map_err(|_| Err(bad("invalid source warnings")))?;
        self.give(Some(&warnings));
        if let Some(error) = Self::schema_refused(&state) {
            return Err(Ok(error));
        }
        Ok(SchemaH {
            state,
            frames: Frames::default(),
        })
    }

    fn carry_replace(&mut self, operation: &str, table: &str) -> Result<(), InputError> {
        let (name, raw, tags) = self
            .carry
            .replacements
            .pop_front()
            .ok_or_else(|| bad("missing carry caller frame"))?;
        if name != operation {
            return Err(bad("carry caller frame order differs"));
        }
        let described = Frames::of(&raw, &tags);
        if described.tables.len() != 1
            || described.tables[0].0 != table
            || tags
                .as_object()
                .is_none_or(|tags| tags.len() != 1 || !tags.contains_key(table))
        {
            return Err(bad("carry caller frame domain differs"));
        }
        let replacement = Table::new(described.tables[0].1.clone(), tags.get(table));
        self.change_frame(table, |_| Ok(replacement.clone()))
    }

    fn carry_edit(
        &mut self,
        scope: &str,
        schema: Value,
        op: Value,
        frames: &Frames,
    ) -> Result<Value, Result<Diagnostic, InputError>> {
        let state = json!({"scope": scope, "schema": schema, "op": op, "tags": frames.tags});
        let done = self
            .call(
                "schema_edit",
                vec![Plain::Str(state.to_string()), frames.plain()],
            )
            .map_err(Ok)?;
        let Plain::Tuple(parts) = done else {
            return Err(Err(bad("schema edit returned no pair")));
        };
        if parts.len() != 2 {
            return Err(Err(bad("schema edit returned no pair")));
        }
        let state = serde_json::from_str(
            parts[0]
                .as_str()
                .ok_or_else(|| Err(bad("schema edit state is not text")))?,
        )
        .map_err(|_| Err(bad("invalid schema edit state")))?;
        let warnings: Value = serde_json::from_str(
            parts[1]
                .as_str()
                .ok_or_else(|| Err(bad("schema edit warnings are not text")))?,
        )
        .map_err(|_| Err(bad("invalid schema edit warnings")))?;
        self.give(Some(&warnings));
        Ok(state)
    }

    pub(super) fn carry_column(
        &mut self,
        table: &str,
        key: Value,
        name: &str,
        dtype: &str,
        stype: &str,
    ) -> Result<(), Result<Diagnostic, InputError>> {
        self.carry_edit(
            "column",
            json!({"tables": [], "relationships": []}),
            json!({"op": "column_init", "name": name, "dtype": dtype, "stype": stype}),
            &Frames::default(),
        )?;
        let schema = self
            .schema
            .as_mut()
            .ok_or_else(|| Err(bad("carry column has no schema")))?;
        let tables = schema
            .state
            .pointer_mut("/schema/tables")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| Err(bad("carry schema has no tables")))?;
        let found = tables
            .iter_mut()
            .find(|value| value["name"] == table)
            .ok_or_else(|| Err(bad("carry schema lacks table")))?;
        let columns = found["columns"]
            .as_array_mut()
            .ok_or_else(|| Err(bad("carry table lacks columns")))?;
        let column = json!({"name": key, "dtype": dtype, "stype": stype, "untyped": false, "stype_declared": true, "before_key": null});
        if let Some(existing) = columns.iter_mut().find(|value| value["name"] == key) {
            *existing = column;
        } else {
            columns.push(column);
        }
        Ok(())
    }

    pub(super) fn carry_call(
        &mut self,
        name: &str,
        args: &Map<String, Value>,
    ) -> Result<(), Result<Diagnostic, InputError>> {
        match name {
            "Carry.source_constructor" => {
                let mut schema = self.carry_source()?;
                let links = schema
                    .state
                    .pointer("/schema/relationships")
                    .cloned()
                    .ok_or_else(|| Err(bad("source schema has no links")))?;
                let said = self.carry_edit(
                    "schema",
                    schema.state["schema"].clone(),
                    json!({"op": "set_links", "links": links}),
                    &Frames::default(),
                )?;
                schema.state["schema"] = said;
                schema.state["engine"] = json!([]);
                self.schema = Some(schema);
            }
            "Carry.source_reused_table" => {
                self.schema = Some(self.carry_source()?);
                let other = self.carry_source()?;
                let products = other
                    .state
                    .pointer("/schema/tables")
                    .and_then(Value::as_array)
                    .and_then(|tables| tables.iter().find(|table| table["name"] == "products"))
                    .cloned()
                    .ok_or_else(|| Err(bad("source has no reused table")))?;
                self.schema.as_mut().unwrap().state["schema"]["tables"]
                    .as_array_mut()
                    .ok_or_else(|| Err(bad("source has no tables")))?
                    .push(products);
            }
            "Carry.source_only" => {
                self.schema = Some(self.carry_source()?);
            }
            "Carry.declared_missing_frame" => {
                self.carry_column("users", json!("ghost"), "ghost", "float64", "numerical")?;
                let mut schema = self
                    .schema
                    .clone()
                    .ok_or_else(|| Err(bad("missing declared schema")))?;
                for (name, table) in &self.tables.clone() {
                    let columns = table
                        .columns()
                        .iter()
                        .map(|column| json!(column))
                        .collect::<Vec<_>>();
                    self.carry_edit(
                        "table",
                        json!({"tables": [], "relationships": []}),
                        json!({"op": "table_init", "name": name, "frame_columns": columns}),
                        &Frames::default(),
                    )?;
                }
                let links = schema.state["schema"]["relationships"].clone();
                schema.state["schema"] = self.carry_edit(
                    "schema",
                    schema.state["schema"].clone(),
                    json!({"op": "set_links", "links": links}),
                    &Frames::default(),
                )?;
                self.schema = Some(schema);
            }
            "Carry.decimal_key" => {
                self.carry_replace(name, "users").map_err(Err)?;
                self.carry_column("users", json!(5), "5", "float64", "numerical")?;
            }
            "Carry.integer_key" => {
                self.carry_replace(name, "people").map_err(Err)?;
            }
            "Carry.list_feature" => {
                match args.get("backend").and_then(Value::as_str) {
                    Some("memory") => {
                        let names = self
                            .tables
                            .iter()
                            .map(|(name, _)| name.clone())
                            .collect::<Vec<_>>();
                        self.schema = Some(self.schema_from_data(&names, &Map::new()).map_err(Ok)?);
                        self.carry_replace(name, "users").map_err(Err)?;
                    }
                    Some("duckdb") => {
                        self.schema = Some(self.carry_source()?);
                    }
                    _ => return Err(Err(bad("unapproved carry list backend"))),
                }
                self.carry_column("users", json!("tags"), "tags", "string", "multicategorical")?;
            }
            "Carry.request_mutation" => self.carry_request(args)?,
            _ => return Err(Err(bad("unsupported carry call"))),
        }
        Ok(())
    }

    fn carry_request(
        &mut self,
        args: &Map<String, Value>,
    ) -> Result<(), Result<Diagnostic, InputError>> {
        let gathered = self
            .request
            .clone()
            .ok_or_else(|| Err(bad("carry mutation has no request")))?;
        let tables = sdm_request::schemas::Tables::new(
            gathered.frames.tables.clone(),
            &Value::Object(gathered.frames.tags.clone()),
        );
        let request = sdm_request::request::Request::from_json(&gathered.state, &tables)
            .map_err(|error| Err(bad(error)))?;
        let mut payloads = request
            .payloads()
            .ok_or_else(|| Err(bad("carry mutation has no payloads")))?;
        let payload = payloads
            .first_mut()
            .ok_or_else(|| Err(bad("carry mutation has no payload")))?;
        match args.get("mutation").and_then(Value::as_str) {
            Some("missing_predict_reference") => {
                let columns = payload
                    .pointer_mut("/predict/instance_table/columns")
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| Err(bad("carry predict has no columns")))?;
                let reference = columns
                    .iter_mut()
                    .find(|name| {
                        name.as_str()
                            .is_some_and(|name| name.starts_with("__kumo_entity_ref"))
                    })
                    .ok_or_else(|| Err(bad("carry predict has no entity reference")))?;
                *reference = json!("ref0");
            }
            Some("context_null") => {
                payload["context"] = Value::Null;
            }
            Some("tuple_context") => {
                let rows = payload
                    .pointer_mut("/context/instance_table/rows")
                    .ok_or_else(|| Err(bad("carry context has no rows")))?;
                let decoded = PyVal::decode(rows).map_err(|error| Err(bad(error)))?;
                let PyVal::List(rows_in) = decoded else {
                    return Err(Err(bad("carry rows are not a list")));
                };
                let rows_out = rows_in
                    .into_iter()
                    .map(|row| match row {
                        PyVal::List(values) => Ok(PyVal::Tuple(values)),
                        _ => Err(Err(bad("carry row is not a list"))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                *rows = encode_deep(&PyVal::Tuple(rows_out));
            }
            _ => return Err(Err(bad("unsupported carry request mutation"))),
        }
        let state = json!({"payloads": payloads, "schema": request.schema.to_json(), "query": request.query.as_ref().map(encode_deep), "indices": encode_deep(&request.indices), "entity_table": request.entity_table.as_ref().map(encode_deep), "windowed": encode_deep(&request.windowed), "requested_inference": request.requested_inference, "tags": gathered.frames.tags});
        let done = self
            .call(
                "request_materialized",
                vec![Plain::Str(state.to_string()), gathered.frames.plain()],
            )
            .map_err(Ok)?;
        let Plain::Tuple(parts) = done else {
            return Err(Err(bad("materialized request returned no pair")));
        };
        if parts.len() != 2 {
            return Err(Err(bad("materialized request returned no pair")));
        }
        let state: Value = serde_json::from_str(
            parts[0]
                .as_str()
                .ok_or_else(|| Err(bad("request state is not text")))?,
        )
        .map_err(|_| Err(bad("invalid request state")))?;
        self.give(state.get("warnings"));
        self.request = Some(RequestH {
            state: state["request"].clone(),
            frames: Frames::of(&parts[1], &state["tags"]),
        });
        Ok(())
    }
}
