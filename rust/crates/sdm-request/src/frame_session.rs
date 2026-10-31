// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure materialized-frame session policy; no transport or native owners.
use crate::tabular::session as keys;
use sdm_contract::{CoreError, Detail, Sentence, session::SessionKeys};
use sdm_wire::json::Value;

pub fn invalid(reason: &str) -> CoreError {
    CoreError::new(Sentence::FrameSessionInput, reason).with_field("reason", Detail::text(reason))
}
pub fn subset(value: &Value, names: &[&str]) -> Value {
    Value::Object(
        names
            .iter()
            .filter_map(|name| value.get(name).map(|v| ((*name).into(), v.clone())))
            .collect(),
    )
}
fn table<'a>(payload: &'a Value, half: &str) -> Result<&'a Value, CoreError> {
    payload
        .get(half)
        .and_then(|v| v.get("instance_table"))
        .ok_or_else(|| invalid("session request has no instance table"))
}
fn columns(value: &Value) -> Result<Vec<String>, CoreError> {
    let Some(Value::Array(columns)) = value.get("columns") else {
        return Err(invalid("session table has no columns"));
    };
    columns
        .iter()
        .map(|c| {
            c.as_text()
                .map(str::to_owned)
                .ok_or_else(|| invalid("session column is not a name"))
        })
        .collect()
}
pub fn row_keys(payload: &Value, half: &str, windowed: bool) -> Result<SessionKeys, CoreError> {
    let table = table(payload, half)?;
    let columns = columns(table)?;
    let mut primary = columns
        .iter()
        .find(|c| c.starts_with("__kumo_entity_ref"))
        .cloned()
        .into_iter()
        .collect::<Vec<_>>();
    if primary.is_empty() {
        match payload
            .get("schema")
            .and_then(|s| s.get("instance_table"))
            .and_then(|t| t.get("primary_key"))
        {
            Some(Value::Text(name)) => primary.push(name.clone()),
            Some(Value::Array(names)) => {
                primary.extend(names.iter().filter_map(Value::as_text).map(str::to_owned))
            }
            _ => {}
        }
    }
    if windowed {
        let anchor = payload
            .get("task")
            .and_then(|t| t.get("anchor_time_column"))
            .and_then(Value::as_text)
            .unwrap_or("__kumo_anchor_time");
        if columns.iter().any(|c| c == anchor) {
            primary.push(anchor.into());
        }
    }
    keys::encoded_keys(table, &primary)
}
/// Remove held-out instance rows and only their associated related rows.
/// The caller's original materialized request is never modified.
pub fn holdout(payload: &Value, holdout: &SessionKeys) -> Result<(Value, usize, usize), CoreError> {
    let named = row_keys(payload, "context", false)?;
    let selection = keys::filter_holdout(&named.rows, &holdout.rows)?;
    let original = table(payload, "context")?;
    let names = columns(original)?;
    let mut context = payload
        .get("context")
        .cloned()
        .ok_or_else(|| invalid("context absent"))?;
    let kept = keys::select_encoded_rows(original.clone(), &selection.retained_rows)?;
    if let Some(at) = names.iter().position(|c| c == "instance_id") {
        let Some(Value::Array(rows)) = original.get("rows") else {
            return Err(invalid("context rows absent"));
        };
        let retained = selection
            .retained_rows
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let removed = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| !retained.contains(i))
            .filter_map(|(_, r)| match r {
                Value::Array(v) => v.get(at).map(Value::to_bytes),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(Value::Object(tables)) = context.get("related_tables") {
            let mut related = Vec::new();
            for (name, table) in tables {
                let names = columns(table)?;
                let Some(at) = names.iter().position(|c| c == "instance_id") else {
                    related.push((name.clone(), table.clone()));
                    continue;
                };
                let Some(Value::Array(rows)) = table.get("rows") else {
                    return Err(invalid("related rows absent"));
                };
                let keep = rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| match r {
                        Value::Array(v)
                            if v.get(at).is_none_or(|v| !removed.contains(&v.to_bytes())) =>
                        {
                            Some(i)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                related.push((
                    name.clone(),
                    keys::select_encoded_rows(table.clone(), &keep)?,
                ));
            }
            context.set("related_tables", Value::Object(related));
        }
    }
    context.set("instance_table", kept);
    let mut payload = payload.clone();
    payload.set("context", context);
    Ok((payload, selection.removed_rows, selection.named_keys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::session::SessionKeyPart;

    fn payload() -> Value {
        sdm_wire::json::read(br#"{
          "task":{"anchor_time_column":"when"},
          "context":{"instance_table":{"format":"arrays",
            "columns":["instance_id","__kumo_entity_ref_user","when"],
            "rows":[[1,"9007199254740993","2024-01-01"],[2,"4","2024-01-02"]]},
            "related_tables":{"events":{"format":"arrays","columns":["instance_id","value"],"rows":[[1,7],[2,8]]}}},
          "predict":{"instance_table":{"format":"arrays",
            "columns":["instance_id","__kumo_entity_ref_user","when"],
            "rows":[[3,"9007199254740993","2024-01-02"]]}}
        }"#).unwrap()
    }
    #[test]
    fn exact_numeric_holdout_removes_only_its_related_instances() {
        let original = payload();
        let hold = SessionKeys {
            rows: vec![vec![SessionKeyPart::Numeric {
                decimal: "9007199254740993".into(),
            }]],
        };
        let (filtered, removed, named) = holdout(&original, &hold).unwrap();
        assert_eq!((removed, named), (1, 1));
        assert_eq!(row_keys(&original, "context", false).unwrap().rows.len(), 2);
        assert_eq!(row_keys(&filtered, "context", false).unwrap().rows.len(), 1);
        let related = filtered
            .get("context")
            .unwrap()
            .get("related_tables")
            .unwrap()
            .get("events")
            .unwrap()
            .get("rows")
            .unwrap();
        assert_eq!(related.to_bytes(), b"[[2,8]]");
    }
    #[test]
    fn temporal_overlap_includes_the_actual_anchor_column() {
        let data = payload();
        assert_eq!(
            keys::overlap_count(
                &row_keys(&data, "context", false).unwrap().rows,
                &row_keys(&data, "predict", false).unwrap().rows
            )
            .unwrap(),
            1
        );
        assert_eq!(
            keys::overlap_count(
                &row_keys(&data, "context", true).unwrap().rows,
                &row_keys(&data, "predict", true).unwrap().rows
            )
            .unwrap(),
            0
        );
    }
}
