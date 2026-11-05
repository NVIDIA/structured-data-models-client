// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated model gradients interpreted against the exact request that produced them.
//! Hosts only turn these semantic records into their own containers and table views.
use crate::predict::{self, Entity, Row};
use sdm_contract::{
    CoreError,
    result::{
        GradientCollection, GradientKeyKind, GradientRecord, MergeCandidate, MergeDescriptor,
    },
};
use sdm_wire::json::{self, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const FORMAT: &str = "sdm_gradient_v1";
fn bad(reason: &'static str) -> CoreError {
    predict::bad_response("prediction.explanation", reason)
}
fn array(value: Option<&Value>) -> Result<&[Value], CoreError> {
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
type LabelRules = (predict::AnswerType, fn(f64) -> String);
pub struct Decoder<'a> {
    payload: &'a Value,
    context: Option<Context<'a>>,
    labels: Option<LabelRules>,
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
            labels: None,
        }
    }
    pub fn with_answers(
        mut self,
        answers: predict::AnswerType,
        float_text: fn(f64) -> String,
    ) -> Self {
        self.labels = Some((answers, float_text));
        self
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
        let task = payload
            .get("task")
            .and_then(|v| v.get("kind"))
            .and_then(Value::as_text);
        for output in outputs {
            let kind = text(output.get("kind"))?;
            if task == Some("regression") && kind == "probability"
                || matches!(
                    task,
                    Some("classification" | "binary_classification" | "multiclass_classification")
                ) && kind != "probability"
            {
                return Err(bad("output_task_mismatch"));
            }
            match kind {
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
        let mut outputs = outputs.to_vec();
        if let Some((answers, float_text)) = self.labels {
            let classes = outputs
                .iter()
                .filter(|v| v.get("kind").and_then(Value::as_text) == Some("probability"))
                .map(|v| {
                    (
                        v.get("class")
                            .map(|v| v.as_text().map_or_else(|| encoded(v), str::to_owned))
                            .unwrap(),
                        0.0,
                    )
                })
                .collect::<Vec<_>>();
            let mut normalized = predict::spelled(&classes, answers, float_text).into_iter();
            for output in &mut outputs {
                if output.get("kind").and_then(Value::as_text) == Some("probability") {
                    output.set("class", Value::Text(normalized.next().unwrap().0));
                }
            }
        }
        let mut entry = object([("outputs", Value::Array(outputs.clone()))]);
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
        if positions != [row] {
            return Err(bad("wrong_instance_row"));
        }
        if payload.get("model").and_then(Value::as_text) == Some("kumo-tabular") {
            if let Some(related) = details.get("related_tables")
                && !members(related)?.is_empty()
            {
                return Err(bad("unexpected_related_table"));
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
                if let (Some(scope), Some(instance_scope)) =
                    (table.at("instance_id"), instance.at("instance_id"))
                {
                    let owner = &instance.rows[row][instance_scope];
                    if positions.iter().any(|&at| table.rows[at][scope] != *owner) {
                        return Err(bad("wrong_related_owner"));
                    }
                }
                let generated = payload
                    .get("client_generated_keys")
                    .and_then(|keys| keys.get(name))
                    .and_then(Value::as_text);
                let keys = key_columns(described)?
                    .into_iter()
                    .filter(|key| Some(*key) != generated)
                    .collect::<Vec<_>>();
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
                                                    && Some(column.as_str()) != generated
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
                if keys.len() == 1
                    && let Some(at) = table.at(keys[0])
                {
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
    if names.iter().any(|name| name.is_empty()) || names.len() != columns.len() {
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
/// Interpret relationship distances identically for gradients and context statistics.
pub(crate) fn context_hops<'a>(
    schema: &Value,
    context: &'a Value,
) -> Result<BTreeMap<&'a str, Vec<Option<usize>>>, CoreError> {
    let instance = Table::read(
        context
            .get("instance_table")
            .ok_or_else(|| bad("request_shape"))?,
    )?;
    let tables = members(
        context
            .get("related_tables")
            .ok_or_else(|| bad("request_shape"))?,
    )?
    .iter()
    .map(|(name, value)| Ok((name.as_str(), Table::read(value)?)))
    .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
    hops(schema, &instance, &tables)
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
            native_id: None,
            entry_owner: None,
            row_index: row.gradient_position.unwrap_or(row.row_index),
            answer_index: row.answer_index,
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
        let identity = identity(row)?;
        if matches!(&identity, MergeCandidate::Json { value } if matches!(value.as_str(), "null" | "NaN" | "Infinity" | "-Infinity"))
            || matches!(
                &identity,
                MergeCandidate::List { .. }
                    | MergeCandidate::Mapping { .. }
                    | MergeCandidate::Unsupported { .. }
            )
        {
            by_id = false;
        }
        let key = crate::result::candidate_key(&identity)?;
        if key.is_none_or(|key| !seen.insert((key, row.forecast_step))) {
            by_id = false
        }
    }
    let mut rows = rows;
    rows.sort_by_key(|row| row.answer_index);
    Ok(Some(GradientCollection {
        keyed_by: if by_id {
            GradientKeyKind::Id
        } else {
            GradientKeyKind::RowIndex
        },
        rows,
    }))
}

fn identity(row: &GradientRecord) -> Result<MergeCandidate, CoreError> {
    row.native_id.clone().map_or_else(
        || {
            json::read_answer(row.id.as_bytes())
                .map(crate::result::candidate)
                .map_err(|_| bad("invalid_identity"))
        },
        Ok,
    )
}

/// Resolve host-owned manual entries against the same row identity policy as model results.
pub fn manual_records(part: &MergeDescriptor) -> Result<Vec<GradientRecord>, CoreError> {
    if part.manual_gradients.is_empty() {
        return Ok(part.gradient_rows.clone());
    }
    let mut probe = part.gradient_rows.clone();
    for row in &mut probe {
        row.explanation = Some("{}".into());
    }
    let keyed = collection(probe)?.ok_or_else(manual_error)?.keyed_by;
    let mut rows = part.gradient_rows.clone();
    let mut positions = BTreeMap::new();
    for (at, row) in rows.iter().enumerate() {
        let key = if keyed == GradientKeyKind::Id {
            identity(row)?
        } else {
            MergeCandidate::Json {
                value: row.row_index.to_string(),
            }
        };
        let key = match row.forecast_step {
            Some(step) => MergeCandidate::Tuple {
                values: vec![
                    key,
                    MergeCandidate::Json {
                        value: step.to_string(),
                    },
                ],
            },
            None => key,
        };
        let key = crate::result::candidate_key(&key)?.ok_or_else(manual_error)?;
        if positions.insert(key, at).is_some() {
            return Err(manual_error());
        }
    }
    for entry in &part.manual_gradients {
        let wanted = crate::result::candidate_key(&entry.key)?.ok_or_else(manual_error)?;
        let row = &mut rows[*positions.get(&wanted).ok_or_else(manual_error)?];
        if row.entry_owner.replace(entry.owner).is_some() {
            return Err(manual_error());
        }
        row.explanation = Some("{}".into());
    }
    Ok(rows)
}

fn manual_error() -> CoreError {
    CoreError::new(
        sdm_contract::Sentence::ResultMergeManualIdentity,
        "manual explanation keys must identify result rows by id or row_index and forecast step; use predict() for decoded results",
    )
}

/// Preserve unique positions; resolve collisions between results while retaining forecast steps.
pub fn merge_records<'a>(
    parts: impl IntoIterator<Item = &'a [GradientRecord]>,
) -> Result<Option<GradientCollection>, CoreError> {
    let parts = parts.into_iter().collect::<Vec<_>>();
    if parts
        .iter()
        .flat_map(|part| part.iter())
        .all(|row| row.explanation.is_none())
    {
        return Ok(None);
    }
    for part in &parts {
        let mut unique = BTreeSet::new();
        if part
            .iter()
            .any(|row| !unique.insert((row.row_index, row.forecast_step)))
        {
            return Err(manual_error());
        }
    }
    let mut seen = BTreeSet::new();
    let collisions = parts
        .iter()
        .flat_map(|p| p.iter())
        .any(|r| !seen.insert((r.row_index, r.forecast_step)));
    let mut rows = Vec::new();
    let mut positions = BTreeMap::new();
    for (part_index, part) in parts.into_iter().enumerate() {
        let mut ordered = part.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|r| r.answer_index);
        for row in ordered {
            let mut row = row.clone();
            if collisions {
                let next = i64::try_from(positions.len()).map_err(|_| bad("too_many_rows"))?;
                row.row_index = *positions.entry((part_index, row.row_index)).or_insert(next);
            }
            row.answer_index = i64::try_from(rows.len()).map_err(|_| bad("too_many_rows"))?;
            rows.push(row);
        }
    }
    collection(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn member_mut<'a>(v: &'a mut Value, name: &str) -> &'a mut Value {
        let Value::Object(fields) = v else {
            panic!("object")
        };
        &mut fields.iter_mut().find(|(key, _)| key == name).unwrap().1
    }
    fn value(input: serde_json::Value) -> Value {
        json::read(&serde_json::to_vec(&input).unwrap()).unwrap()
    }
    fn tabular() -> Value {
        value(
            serde_json::json!({"model":"kumo-tabular","predict":{"instance_table":{"columns":["age"],"rows":[[20],[40]]}}}),
        )
    }
    fn explanation(row: i64) -> Value {
        value(
            serde_json::json!({"format":FORMAT,"details":{"input_space":"preprocessed_numerical","outputs":[{"kind":"prediction","statistic":"mean"}],"instance_table":{"row_indices":[row],"columns":["age"],"gradients":[[[0.25]]]},"related_tables":{}}}),
        )
    }
    #[test]
    fn each_tabular_row_and_unknown_legacy_format() {
        let payload = tabular();
        let mut decoder = Decoder::new(&payload);
        for row in 0..2 {
            let decoded = decoder
                .decode(Some(&explanation(row)), row)
                .unwrap()
                .unwrap();
            assert_eq!(
                encoded(decoded.get("instance").unwrap()),
                r#"{"columns":["age"],"gradients":[[0.25]]}"#
            );
        }
        assert!(
            decoder
                .decode(
                    Some(&value(
                        serde_json::json!({"format":"legacy","details":null})
                    )),
                    0
                )
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn malformed_indices_shapes_and_columns_fail() {
        let payload = tabular();
        let mut decoder = Decoder::new(&payload);
        for row in [-1, 2] {
            assert!(decoder.decode(Some(&explanation(row)), 0).is_err());
        }
        let mut malformed = explanation(0);
        let details = member_mut(&mut malformed, "details");
        member_mut(details, "instance_table")
            .set("gradients", value(serde_json::json!([[[0.1, 0.2]]])));
        assert!(decoder.decode(Some(&malformed), 0).is_err());
        let mut malformed = explanation(0);
        member_mut(member_mut(&mut malformed, "details"), "instance_table")
            .set("columns", Value::texts(["age", "age"]));
        assert!(decoder.decode(Some(&malformed), 0).is_err());
    }
    #[test]
    fn hops_scope_numeric_join_and_wide_keys() {
        let payload = value(
            serde_json::json!({"model":"kumo-relational","schema":{"related_tables":{"users":{"primary_key":["instance_id","id"],"columns":{"id":{"dtype":"int64"}}},"events":{"primary_key":["instance_id","id"],"columns":{"id":{"dtype":"int64"}}}},"relationships":[{"source_columns":["user"],"target_table":"users","target_columns":["id"]},{"source_table":"events","source_columns":["user"],"target_table":"users","target_columns":["id"]}]},"predict":{"instance_table":{"columns":["instance_id","user"],"rows":[[0,9007199254740993i64],[1,7]]},"related_tables":{"users":{"columns":["instance_id","id","age"],"rows":[[0,9007199254740993i64,42],[1,7,30]]},"events":{"columns":["instance_id","id","user","amount"],"rows":[[0,5,9007199254740993i64,10],[1,6,7.0,20],[0,7,7,30]]}}}}),
        );
        let mut wire = explanation(0);
        let details = member_mut(&mut wire, "details");
        details.set(
            "instance_table",
            value(serde_json::json!({"row_indices":[0],"columns":[],"gradients":[[[]]]})),
        );
        details.set("related_tables",value(serde_json::json!({"users":{"row_indices":[0],"columns":["age"],"gradients":[[[0.2]]]},"events":{"row_indices":[0,2],"columns":["amount"],"gradients":[[[0.1],[0.0]]]}})));
        let decoded = Decoder::new(&payload)
            .decode(Some(&wire), 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            encoded(decoded.get("hops").unwrap()),
            r#"{"users":[0],"events":[1,null]}"#
        );
        assert_eq!(
            encoded(
                decoded
                    .get("related")
                    .unwrap()
                    .get("users")
                    .unwrap()
                    .get("keys")
                    .unwrap()
            ),
            "[9007199254740993]"
        );
        let context = Decoder::new(&payload);
        assert!(context.context.is_none());
    }
    fn record(id: &str, row_index: i64, step: Option<i64>, explained: bool) -> GradientRecord {
        GradientRecord {
            native_id: None,
            entry_owner: None,
            id: id.into(),
            row_index,
            answer_index: row_index,
            forecast_step: step,
            explanation: explained.then(|| "{}".into()),
        }
    }
    #[test]
    fn duplicate_ids_use_global_positions_including_unexplained_rows() {
        let first = [record("7", 0, None, false), record("9", 1, None, true)];
        let second = [record("7.0", 0, None, true)];
        let merged = merge_records([first.as_slice(), second.as_slice()])
            .unwrap()
            .unwrap();
        assert_eq!(merged.keyed_by, GradientKeyKind::RowIndex);
        assert_eq!(
            merged.rows.iter().map(|r| r.row_index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let forecast = collection(vec![
            record("9007199254740993", 0, Some(1), true),
            record("9007199254740993", 0, Some(2), true),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(forecast.keyed_by, GradientKeyKind::Id);
    }
    #[test]
    fn output_labels_share_prediction_column_rules() {
        let payload = tabular();
        for (answers, label, expected) in [
            (predict::AnswerType::Int, "1", "1"),
            (predict::AnswerType::Bool, "true", "True"),
            (predict::AnswerType::FloatClass, "1", "1.0"),
        ] {
            let mut wire = explanation(0);
            member_mut(&mut wire, "details").set(
                "outputs",
                value(serde_json::json!([{"kind":"probability","class":label}])),
            );
            let output = Decoder::new(&payload)
                .with_answers(answers, |v| format!("{v:.1}"))
                .decode(Some(&wire), 0)
                .unwrap()
                .unwrap();
            assert_eq!(
                output.get("outputs").unwrap(),
                &value(serde_json::json!([{"kind":"probability","class":expected}]))
            );
        }
    }
    #[test]
    fn nested_merges_and_nonmonotonic_positions_keep_every_entry() {
        let mut a = record("7", 5, None, true);
        a.answer_index = 0;
        let mut b = record("7", 2, None, true);
        b.answer_index = 1;
        let first = collection(vec![a, b]).unwrap().unwrap();
        assert_eq!(
            first.rows.iter().map(|r| r.row_index).collect::<Vec<_>>(),
            [5, 2]
        );
        let last = [record("7", 0, None, true)];
        let nested = merge_records([first.rows.as_slice(), last.as_slice()])
            .unwrap()
            .unwrap();
        assert_eq!(
            nested.rows.iter().map(|r| r.row_index).collect::<Vec<_>>(),
            [5, 2, 0]
        );
    }
    #[test]
    fn ordinary_nested_merges_do_not_apply_gradient_identity_rules() {
        let rows = [record("1", 0, None, false), record("1", 0, None, false)];
        assert!(merge_records([rows.as_slice()]).unwrap().is_none());
    }

    #[test]
    fn duplicate_positions_within_one_part_are_rejected() {
        let rows = [record("1", 0, None, true), record("2", 0, None, true)];
        assert!(merge_records([rows.as_slice()]).is_err());
    }

    #[test]
    fn complex_nonfinite_and_python_equal_ids_use_positions() {
        for id in ["[1,2]", "{\"k\":1}", "NaN", "Infinity"] {
            let result = collection(vec![record(id, 0, None, true)])
                .unwrap()
                .unwrap();
            assert_eq!(result.keyed_by, GradientKeyKind::RowIndex);
        }
        let result = collection(vec![
            record("1", 0, None, true),
            record("true", 1, None, true),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(result.keyed_by, GradientKeyKind::RowIndex);
    }
}
