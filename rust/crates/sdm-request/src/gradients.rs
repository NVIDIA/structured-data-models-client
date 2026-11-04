// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated model gradients interpreted against the exact request that produced them.
//! Hosts only turn these semantic records into their own containers and table views.
use crate::predict::{self, Entity, Row};
use sdm_contract::{
    CoreError,
    result::{GradientCollection, GradientRecord, GradientKeyKind},
};
use sdm_wire::json::{self, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const FORMAT: &str = "sdm_gradient_v1";
fn bad(reason: &'static str) -> CoreError {
    predict::bad_response("prediction.explanation", reason)
}
fn array<'a>(value: Option<&'a Value>) -> Result<&'a [Value], CoreError> {
    match value {
        Some(Value::Array(v)) => Ok(v),
        _ => Err(bad("invalid_shape")),
    }
}
fn members(value: &Value) -> Result<&[(String, Value)], CoreError> {
    match value {
        Value::Object(v) => Ok(v),
        _ => Err(bad("invalid_shape")),
    }
}
fn text(value: Option<&Value>) -> Result<&str, CoreError> {
    value
        .and_then(Value::as_text)
        .ok_or_else(|| bad("invalid_shape"))
}
fn index(value: &Value) -> Result<usize, CoreError> {
    match value {
        Value::Number(n) => n.parse().map_err(|_| bad("invalid_row_index")),
        _ => Err(bad("invalid_row_index")),
    }
}
fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
pub fn entity_json(id: &Entity) -> Value {
    match id {
        Entity::None => Value::Null,
        Entity::Integer(v) => Value::Number(v.to_string()),
        Entity::Float(v) => Value::Number(v.to_string()),
        Entity::Text(v) => Value::Text(v.clone()),
        Entity::Written(v) => v.clone(),
    }
}
fn encoded(value: &Value) -> String {
    String::from_utf8_lossy(&value.to_bytes()).into_owned()
}

/// A known format is checked strictly; unknown formats remain on the raw accessor.
/// The returned entry includes hop arrays as semantic facts, not presentation policy.
pub struct Decoder<'a> {
    payload: &'a Value,
    context: Option<Context<'a>>,
}
struct Context<'a> {
    instance: Table<'a>,
    tables: BTreeMap<&'a str, Table<'a>>,
    hops: BTreeMap<&'a str, Vec<Option<usize>>>,
}
impl<'a> Decoder<'a> {
    pub fn new(payload: &'a Value) -> Self {
        Self {
            payload,
            context: None,
        }
    }
    pub fn decode(
        &mut self,
        explanation: Option<&Value>,
        row_index: i64,
    ) -> Result<Option<Value>, CoreError> {
        let Some(explanation) = explanation else {
            return Ok(None);
        };
        if explanation.get("format").and_then(Value::as_text) != Some(FORMAT) {
            return Ok(None);
        }
        let payload = self.payload;
        if self.context.is_none() {
            let predict = payload
                .get("predict")
                .ok_or_else(|| bad("missing_predict"))?;
            let instance = Table::read(
                predict
                    .get("instance_table")
                    .ok_or_else(|| bad("missing_predict"))?,
            )?;
            let mut tables = BTreeMap::new();
            if let Some(related) = predict.get("related_tables") {
                for (name, table) in members(related)? {
                    tables.insert(name.as_str(), Table::read(table)?);
                }
            }
            let distances = if tables.is_empty() {
                BTreeMap::new()
            } else {
                hops(
                    payload.get("schema").ok_or_else(|| bad("missing_schema"))?,
                    &instance,
                    &tables,
                )?
            };
            self.context = Some(Context {
                instance,
                tables,
                hops: distances,
            });
        }
        let context = self.context.as_ref().unwrap();
        let details = explanation
            .get("details")
            .ok_or_else(|| bad("missing_details"))?;
        if text(details.get("input_space"))? != "preprocessed_numerical" {
            return Err(bad("unsupported_input_space"));
        }
        let outputs = array(details.get("outputs"))?;
        if outputs.is_empty() {
            return Err(bad("missing_outputs"));
        }
        for output in outputs {
            match text(output.get("kind"))? {
                "probability" => {
                    if output.get("class").is_none_or(|v| {
                        matches!(v, Value::Null | Value::Array(_) | Value::Object(_))
                    }) {
                        return Err(bad("invalid_output"));
                    }
                }
                "prediction" => {
                    if !matches!(
                        output.get("statistic").and_then(Value::as_text),
                        Some("mean" | "median")
                    ) {
                        return Err(bad("invalid_output"));
                    }
                }
                "quantile" => {
                    let Some(Value::Number(n)) = output.get("level") else {
                        return Err(bad("invalid_output"));
                    };
                    if !n
                        .parse::<f64>()
                        .is_ok_and(|v| v.is_finite() && (0.0..=1.0).contains(&v))
                    {
                        return Err(bad("invalid_output"));
                    }
                }
                _ => return Err(bad("invalid_output")),
            }
        }
        let mut entry = object([("outputs", Value::Array(outputs.to_vec()))]);
        let instance = &context.instance;
        let row = usize::try_from(row_index).map_err(|_| bad("invalid_row_index"))?;
        if row >= instance.rows.len() {
            return Err(bad("invalid_row_index"));
        }
        let block = details
            .get("instance_table")
            .ok_or_else(|| bad("missing_instance"))?;
        let (positions, columns, gradients) =
            block_data(block, outputs.len(), instance.rows.len())?;
        if columns.iter().any(|c|instance.at(c.as_text().unwrap()).is_none()) { return Err(bad("unknown_column")); }
        if positions != [row] {
            return Err(bad("wrong_instance_row"));
        }
        if payload.get("model").and_then(Value::as_text) == Some("kumo-tabular") {
            if let Some(related) = details.get("related_tables") {
                if !members(related)?.is_empty() {
                    return Err(bad("unexpected_related_table"));
                }
            }
            let flat = gradients
                .iter()
                .map(|v| Ok(array(Some(v))?[0].clone()))
                .collect::<Result<Vec<_>, CoreError>>()?;
            entry.set(
                "instance",
                object([
                    ("columns", Value::Array(columns)),
                    ("gradients", Value::Array(flat)),
                ]),
            );
        } else {
            let schema = payload.get("schema").ok_or_else(|| bad("missing_schema"))?;
            let table_map = &context.tables;
            let hops = &context.hops;
            let blocks = members(
                details
                    .get("related_tables")
                    .ok_or_else(|| bad("missing_related_tables"))?,
            )?;
            let mut related = Vec::new();
            let mut distances = Vec::new();
            for (name, block) in blocks {
                let table = table_map
                    .get(name.as_str())
                    .ok_or_else(|| bad("unknown_table"))?;
                let described = schema
                    .get("related_tables")
                    .and_then(|v| v.get(name))
                    .ok_or_else(|| bad("unknown_table"))?;
                let (positions, columns, gradients) =
                    block_data(block, outputs.len(), table.rows.len())?;
            if columns.iter().any(|c|table.at(c.as_text().unwrap()).is_none()) { return Err(bad("unknown_column")); }
                if let (Some(scope), Some(instance_scope)) =
                    (table.at("instance_id"), instance.at("instance_id"))
                {
                    let owner = &instance.rows[row][instance_scope];
                    if positions.iter().any(|&at| table.rows[at][scope] != *owner) {
                        return Err(bad("wrong_related_owner"));
                    }
                }
                let keys = key_columns(described)?;
                let synthetic = BTreeSet::<&str>::new();
                let mut item = object([
                    (
                        "rows",
                        Value::Array(
                            positions
                                .iter()
                                .map(|&at| {
                                    Value::Object(
                                        table
                                            .names
                                            .iter()
                                            .zip(table.rows[at])
                                            .filter(|(column, _)| {
                                                column.as_str() != "instance_id"
                                                    && !synthetic.contains(column.as_str())
                                            })
                                            .map(|(column, value)| (column.clone(), value.clone()))
                                            .collect(),
                                    )
                                })
                                .collect(),
                        ),
                    ),
                    ("columns", Value::Array(columns)),
                    ("gradients", Value::Array(gradients)),
                ]);
                if keys.len() == 1 && synthetic.is_empty() {
                    if let Some(at) = table.at(keys[0]) {
                        let dtype = described
                            .get("columns")
                            .and_then(|v| v.get(keys[0]))
                            .and_then(|v| v.get("dtype"))
                            .and_then(Value::as_text);
                        item.set(
                            "keys",
                            Value::Array(
                                positions
                                    .iter()
                                    .map(|&row| {
                                        entity_json(&predict::entity(&table.rows[row][at], dtype))
                                    })
                                    .collect(),
                            ),
                        );
                    }
                }
                distances.push((
                    name.clone(),
                    Value::Array(
                        positions
                            .iter()
                            .map(|&at| {
                                hops[name.as_str()][at]
                                    .map_or(Value::Null, |v| Value::Number(v.to_string()))
                            })
                            .collect(),
                    ),
                ));
                related.push((name.clone(), item));
            }
            entry.set("related", Value::Object(related));
            entry.set("hops", Value::Object(distances));
        }
        Ok(Some(entry))
    }
}
fn key_columns(described: &Value) -> Result<Vec<&str>, CoreError> {
    match described.get("primary_key") {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Text(v)) => Ok((v != "instance_id")
            .then_some(v.as_str())
            .into_iter()
            .collect()),
        Some(Value::Array(v)) => v
            .iter()
            .map(|v| text(Some(v)))
            .filter(|v| !matches!(v, Ok("instance_id")))
            .collect(),
        _ => Err(bad("invalid_key")),
    }
}
type Block = (Vec<usize>, Vec<Value>, Vec<Value>);
fn block_data(block: &Value, outputs: usize, rows: usize) -> Result<Block, CoreError> {
    let positions = array(block.get("row_indices"))?
        .iter()
        .map(index)
        .collect::<Result<Vec<_>, _>>()?;
    if positions.iter().any(|&v| v >= rows)
        || positions.iter().collect::<BTreeSet<_>>().len() != positions.len()
    {
        return Err(bad("invalid_row_index"));
    }
    let columns = array(block.get("columns"))?;
    let names = columns
        .iter()
        .map(|v| text(Some(v)))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if names.len() != columns.len() {
        return Err(bad("duplicate_column"));
    }
    let gradients = array(block.get("gradients"))?;
    if gradients.len() != outputs {
        return Err(bad("gradient_shape"));
    }
    for output in gradients {
        let values = array(Some(output))?;
        if values.len() != positions.len() {
            return Err(bad("gradient_shape"));
        }
        for row in values {
            let cells = array(Some(row))?;
            if cells.len() != columns.len() {
                return Err(bad("gradient_shape"));
            }
            for cell in cells {
                if !matches!(cell,Value::Number(n) if n.parse::<f64>().is_ok_and(f64::is_finite)) {
                    return Err(bad("nonfinite_gradient"));
                }
            }
        }
    }
    Ok((positions, columns.to_vec(), gradients.to_vec()))
}
struct Table<'a> {
    names: Vec<String>,
    rows: Vec<&'a [Value]>,
}
impl<'a> Table<'a> {
    fn read(value: &'a Value) -> Result<Self, CoreError> {
        let names = array(value.get("columns"))?
            .iter()
            .map(|v| Ok(text(Some(v))?.to_owned()))
            .collect::<Result<Vec<_>, CoreError>>()?;
        let rows = array(value.get("rows"))?
            .iter()
            .map(|v| array(Some(v)))
            .collect::<Result<Vec<_>, _>>()?;
        if rows.iter().any(|v| v.len() != names.len()) {
            return Err(bad("request_shape"));
        }
        Ok(Self { names, rows })
    }
    fn at(&self, column: &str) -> Option<usize> {
        self.names.iter().position(|v| v == column)
    }
    fn joins(&self, columns: &[Value]) -> Result<Vec<Option<String>>, CoreError> {
        let at = columns
            .iter()
            .map(|v| {
                self.at(text(Some(v))?)
                    .ok_or_else(|| bad("relationship_column"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if at.is_empty() {
            return Err(bad("relationship_column"));
        }
        let scope = self.at("instance_id");
        Ok(self
            .rows
            .iter()
            .map(|row| {
                let mut values = vec![scope.map_or(Value::Null, |s| row[s].clone())];
                for &position in &at {
                    if matches!(row[position], Value::Null) {
                        return None;
                    }
                    values.push(row[position].clone())
                }
                crate::result::candidate_key(&crate::result::candidate(Value::Array(values)))
                    .ok()
                    .flatten()
            })
            .collect())
    }
}
type Node<'a> = (&'a str, usize);
fn hops<'a>(
    schema: &Value,
    instance: &Table<'_>,
    tables: &BTreeMap<&'a str, Table<'_>>,
) -> Result<BTreeMap<&'a str, Vec<Option<usize>>>, CoreError> {
    let mut hops = tables
        .iter()
        .map(|(&name, t)| (name, vec![None; t.rows.len()]))
        .collect::<BTreeMap<_, _>>();
    let mut edges: BTreeMap<Node<'a>, Vec<Node<'a>>> = BTreeMap::new();
    let mut reached = VecDeque::new();
    for link in array(schema.get("relationships"))? {
        let target = text(link.get("target_table"))?;
        let Some((&target, table)) = tables.get_key_value(target) else {
            continue;
        };
        let mut lookup: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (at, value) in table
            .joins(array(link.get("target_columns"))?)?
            .into_iter()
            .enumerate()
        {
            if let Some(value) = value {
                lookup.entry(value).or_default().push(at)
            }
        }
        let source = link.get("source_table").and_then(Value::as_text);
        let (source, table) = match source {
            None => (None, instance),
            Some(source) => {
                let Some((&name, table)) = tables.get_key_value(source) else {
                    continue;
                };
                (Some(name), table)
            }
        };
        for (position, value) in table
            .joins(array(link.get("source_columns"))?)?
            .into_iter()
            .enumerate()
        {
            let Some(value) = value else { continue };
            for &at in lookup.get(&value).into_iter().flatten() {
                if let Some(source) = source {
                    edges
                        .entry((source, position))
                        .or_default()
                        .push((target, at));
                    edges
                        .entry((target, at))
                        .or_default()
                        .push((source, position));
                } else if hops[target][at].is_none() {
                    hops.get_mut(target).unwrap()[at] = Some(0);
                    reached.push_back((target, at));
                }
            }
        }
    }
    while let Some(node) = reached.pop_front() {
        let distance = hops[node.0][node.1].unwrap() + 1;
        for &(name, at) in edges.get(&node).into_iter().flatten() {
            if hops[name][at].is_none() {
                hops.get_mut(name).unwrap()[at] = Some(distance);
                reached.push_back((name, at));
            }
        }
    }
    Ok(hops)
}

/// Every predicted row participates in key selection, including unexplained rows.
pub fn records(rows: &[Row]) -> Vec<GradientRecord> {
    rows.iter()
        .map(|row| GradientRecord {
            row_index: row.row_index,
            forecast_step: row.forecast_step,
            id: encoded(&entity_json(&row.id)),
            explanation: row.gradient.as_ref().map(encoded),
        })
        .collect()
}
/// A single canonical identity policy for endpoint, session and manually merged results.
pub fn collection(rows: Vec<GradientRecord>) -> Result<Option<GradientCollection>, CoreError> {
    if rows.iter().all(|row| row.explanation.is_none()) {
        return Ok(None);
    }
    let mut seen = BTreeSet::new();
    let mut by_id = true;
    for row in &rows {
        let id = json::read(row.id.as_bytes()).map_err(|_| bad("invalid_identity"))?;
        let key = crate::result::candidate_key(&crate::result::candidate(id))?;
        if row.id == "null" || key.is_none_or(|key| !seen.insert((key, row.forecast_step))) {
            by_id = false
        }
    }
    let mut rows = rows;
    rows.sort_by_key(|row| (row.row_index, row.forecast_step));
    Ok(Some(GradientCollection {
        keyed_by: if by_id { GradientKeyKind::Id } else { GradientKeyKind::RowIndex },
        rows,
    }))
}

/// Independently produced results receive consecutive positions while forecasts retain their step.
pub fn merge_records<'a>(
    parts: impl IntoIterator<Item = &'a [GradientRecord]>,
) -> Result<Option<GradientCollection>, CoreError> {
    let mut rows = Vec::new();
    let mut offset = 0i64;
    for part in parts {
        let positions = part.iter().map(|r| r.row_index).collect::<BTreeSet<_>>();
        let mapping = positions
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, i as i64 + offset))
            .collect::<BTreeMap<_, _>>();
        for row in part {
            let mut row = row.clone();
            row.row_index = mapping[&row.row_index];
            rows.push(row);
        }
        offset = offset
            .checked_add(mapping.len() as i64)
            .ok_or_else(|| bad("too_many_rows"))?;
    }
    collection(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn member_mut<'a>(v:&'a mut Value,name:&str)-> &'a mut Value { let Value::Object(fields)=v else {panic!("object")}; &mut fields.iter_mut().find(|(key,_)|key==name).unwrap().1 }
    fn value(input:serde_json::Value)->Value { json::read(&serde_json::to_vec(&input).unwrap()).unwrap() }
    fn tabular()->Value { value(serde_json::json!({"model":"kumo-tabular","predict":{"instance_table":{"columns":["age"],"rows":[[20],[40]]}}})) }
    fn explanation(row:i64)->Value { value(serde_json::json!({"format":FORMAT,"details":{"input_space":"preprocessed_numerical","outputs":[{"kind":"prediction","statistic":"mean"}],"instance_table":{"row_indices":[row],"columns":["age"],"gradients":[[[0.25]]]},"related_tables":{}}})) }
    #[test]
    fn each_tabular_row_and_unknown_legacy_format() {
      let payload=tabular();let mut decoder=Decoder::new(&payload);
      for row in 0..2 { let decoded=decoder.decode(Some(&explanation(row)),row).unwrap().unwrap(); assert_eq!(encoded(decoded.get("instance").unwrap()),r#"{"columns":["age"],"gradients":[[0.25]]}"#); }
      assert!(decoder.decode(Some(&value(serde_json::json!({"format":"legacy","details":null}))),0).unwrap().is_none());
    }
    #[test]
    fn malformed_indices_shapes_and_columns_fail() {
      let payload=tabular(); let mut decoder=Decoder::new(&payload);
      for row in [-1,2] { assert!(decoder.decode(Some(&explanation(row)),0).is_err()); }
      let mut malformed=explanation(0);let details=member_mut(&mut malformed,"details");
      member_mut(details,"instance_table").set("gradients",value(serde_json::json!([[[0.1,0.2]]])));
      assert!(decoder.decode(Some(&malformed),0).is_err());
      let mut malformed=explanation(0);member_mut(member_mut(&mut malformed,"details"),"instance_table").set("columns",Value::texts(["missing"]));
      assert!(decoder.decode(Some(&malformed),0).is_err());
    }
    #[test]
    fn hops_scope_numeric_join_and_wide_keys() {
      let payload=value(serde_json::json!({"model":"kumo-relational","schema":{"related_tables":{"users":{"primary_key":["instance_id","id"],"columns":{"id":{"dtype":"int64"}}},"events":{"primary_key":["instance_id","id"],"columns":{"id":{"dtype":"int64"}}}},"relationships":[{"source_columns":["user"],"target_table":"users","target_columns":["id"]},{"source_table":"events","source_columns":["user"],"target_table":"users","target_columns":["id"]}]},"predict":{"instance_table":{"columns":["instance_id","user"],"rows":[[0,9007199254740993i64],[1,7]]},"related_tables":{"users":{"columns":["instance_id","id","age"],"rows":[[0,9007199254740993i64,42],[1,7,30]]},"events":{"columns":["instance_id","id","user","amount"],"rows":[[0,5,9007199254740993i64,10],[1,6,7.0,20],[0,7,7,30]]}}}}));
      let mut wire=explanation(0);let details=member_mut(&mut wire,"details");details.set("instance_table",value(serde_json::json!({"row_indices":[0],"columns":[],"gradients":[[[]]]})));details.set("related_tables",value(serde_json::json!({"users":{"row_indices":[0],"columns":["age"],"gradients":[[[0.2]]]},"events":{"row_indices":[0,2],"columns":["amount"],"gradients":[[[0.1],[0.0]]]}})));
      let decoded=Decoder::new(&payload).decode(Some(&wire),0).unwrap().unwrap();
      assert_eq!(encoded(decoded.get("hops").unwrap()),r#"{"users":[0],"events":[1,null]}"#);
      assert_eq!(encoded(decoded.get("related").unwrap().get("users").unwrap().get("keys").unwrap()),"[9007199254740993]");
      let context=Decoder::new(&payload);assert!(context.context.is_none());
    }
    fn record(id:&str,row_index:i64,step:Option<i64>,explained:bool)->GradientRecord { GradientRecord {id:id.into(),row_index,forecast_step:step,explanation:explained.then(||"{}".into())} }
    #[test]
    fn duplicate_ids_use_global_positions_including_unexplained_rows() {
      let first=[record("7",0,None,false),record("9",1,None,true)];let second=[record("7.0",0,None,true)];
      let merged=merge_records([first.as_slice(),second.as_slice()]).unwrap().unwrap();
      assert_eq!(merged.keyed_by,GradientKeyKind::RowIndex);assert_eq!(merged.rows.iter().map(|r|r.row_index).collect::<Vec<_>>(),[0,1,2]);
      let forecast=collection(vec![record("9007199254740993",0,Some(1),true),record("9007199254740993",0,Some(2),true)]).unwrap().unwrap();assert_eq!(forecast.keyed_by,GradientKeyKind::Id);
    }
}
