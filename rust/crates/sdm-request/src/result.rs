// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared result metadata and stable candidate selection, independent of row storage.
use sdm_contract::result::{MergeCandidate as C, MergeDescriptor, MergeTimeUnit, ResultMergePlan};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::{
    json::{self, Value},
    key,
};
use std::collections::BTreeSet;

/// Truth of a normalized explanation; hosts only supply this observation for opaque owners.
pub fn truth(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Text(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Number(v) => v.parse::<f64>().is_ok_and(|v| v != 0.0),
    }
}
fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}
fn quoted(s: &str) -> String {
    format!(
        "'{}'",
        s.replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}
fn framed(tag: &str, values: impl IntoIterator<Item = String>) -> String {
    let mut out = tag.to_owned();
    for v in values {
        out.push_str(&format!("{}:{v}", v.len()));
    }
    out
}
/// JSON candidates arrive from Arrow/Java; Python uses the same scalar observations.
pub fn candidate(value: Value) -> C {
    match value {
        Value::Array(values) => C::List {
            values: values.into_iter().map(candidate).collect(),
        },
        Value::Object(entries) => C::Mapping {
            entries: entries
                .into_iter()
                .map(|(k, v)| (candidate(Value::Text(k)), candidate(v)))
                .collect(),
        },
        value => C::Json {
            value: String::from_utf8(value.to_bytes()).expect("JSON is UTF-8"),
        },
    }
}
pub(crate) fn candidate_key(value: &C) -> Result<Option<String>, CoreError> {
    Ok(Some(match value {
        C::Json { value } => {
            match json::read_answer(value.as_bytes())
                .map_err(|_| invalid("invalid result candidate JSON"))?
            {
                Value::Null => key::MISSING.to_owned(),
                Value::Bool(v) => key::boolean(v),
                Value::Text(v) => key::text(&v),
                Value::Number(n) if json::is_integer_token(&n) => key::exact(&n),
                Value::Number(n) => {
                    let value = n
                        .parse::<f64>()
                        .map_err(|_| invalid("invalid result numeric candidate"))?;
                    let Some(key) = key::float(value) else {
                        return Ok(None);
                    };
                    key
                }
                value => return candidate_key(&candidate(value)),
            }
        }
        C::Bytes { value } => framed("b", value.iter().map(|v| v.to_string())),
        C::Decimal { value } => {
            let digits = value.strip_prefix('-').unwrap_or(value);
            let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
            if whole.is_empty()
                || !whole
                    .bytes()
                    .chain(fraction.bytes())
                    .all(|v| v.is_ascii_digit())
                || (digits.contains('.') && fraction.is_empty())
            {
                return Err(invalid("invalid decimal candidate"));
            }
            key::exact(value)
        }
        C::Date { ordinal } => format!("d{ordinal}"),
        C::DateTime { ticks, unit, aware } => {
            // Decimal placement is exact and never casts through nanoseconds.
            let digits = ticks.strip_prefix('-').unwrap_or(ticks);
            if digits.is_empty() || !digits.bytes().all(|v| v.is_ascii_digit()) {
                return Err(invalid("invalid temporal candidate ticks"));
            }
            let scale = match unit {
                MergeTimeUnit::S => 0,
                MergeTimeUnit::Ms => 3,
                MergeTimeUnit::Us => 6,
                MergeTimeUnit::Ns => 9,
                MergeTimeUnit::Ps => 12,
                MergeTimeUnit::Fs => 15,
                MergeTimeUnit::As => 18,
            };
            let padded = format!("{:0>width$}", digits, width = scale + 1);
            let (whole, fraction) = padded.split_at(padded.len() - scale);
            let number = format!(
                "{}{whole}.{fraction}",
                if ticks.starts_with('-') { "-" } else { "" }
            );
            format!("t{aware}{}", key::exact(&number))
        }
        C::Tuple { values } | C::List { values } => {
            let mut keys = Vec::new();
            for value in values {
                let Some(key) = candidate_key(value)? else {
                    return Ok(None);
                };
                keys.push(key);
            }
            framed(
                if matches!(value, C::Tuple { .. }) {
                    "tuple"
                } else {
                    "list"
                },
                keys,
            )
        }
        C::Mapping { entries } => {
            let mut keys = Vec::new();
            for (k, v) in entries {
                let (Some(k), Some(v)) = (candidate_key(k)?, candidate_key(v)?) else {
                    return Ok(None);
                };
                keys.push(framed("pair", [k, v]));
            }
            keys.sort();
            framed("map", keys)
        }
        C::Unsupported { type_name } => {
            return Err(CoreError::new(
                Sentence::ResultMergeCandidateType,
                "unsupported result candidate type",
            )
            .with_field("type_name", Detail::text(type_name)));
        }
    }))
}

/// Empty/single/model/output ordering matches the public client; row owners stay host-side.
fn merge_columns(parts: &[MergeDescriptor]) -> Result<Option<String>, CoreError> {
    let mut selected = None;
    for part in parts {
        if let Some(raw) = &part.columns {
            let normalized = serde_json::to_string(&crate::columns::validate(raw)?)
                .map_err(|_| invalid("invalid analysis encoding"))?;
            if selected.as_ref().is_some_and(|prior| prior != &normalized) {
                return Err(CoreError::new(
                    Sentence::ResultMergeColumns,
                    "cannot merge different column analyses; supply request-level columns explicitly",
                ));
            }
            selected = Some(normalized);
        }
    }
    Ok(selected)
}

pub fn plan(parts: &[MergeDescriptor]) -> Result<ResultMergePlan, CoreError> {
    let Some(first) = parts.first() else {
        return Err(CoreError::new(
            Sentence::ResultMergeEmpty,
            "no parts to merge",
        ));
    };
    let gradient_parts = parts
        .iter()
        .map(crate::gradients::manual_records)
        .collect::<Result<Vec<_>, _>>()?;
    let mut plan = ResultMergePlan {
        columns: merge_columns(parts)?,
        single: parts.len() == 1,
        batches: parts.len(),
        batch_request_ids: parts.iter().map(|p| p.request_id.clone()).collect(),
        explanation_part: None,
        candidate_positions: vec![],
        gradients: crate::gradients::merge_records(gradient_parts.iter().map(Vec::as_slice))?,
        names_rows: true,
        names_step: false,
    };
    if plan.single {
        plan.explanation_part = first.has_explanation.then_some(0);
        plan.names_rows = first.names_rows;
        plan.names_step = first.names_step;
        plan.candidate_positions = (0..first.candidates.len())
            .map(|i| u32::try_from(i).map_err(|_| invalid("too many result candidates")))
            .collect::<Result<_, _>>()?;
        return Ok(plan);
    }
    let models: BTreeSet<_> = parts.iter().map(|p| p.model.as_str()).collect();
    if models.len() > 1 {
        return Err(CoreError::new(
            Sentence::ResultMergeModels,
            "cannot merge results from different models",
        )
        .with_field(
            "models",
            Detail::List(
                models
                    .into_iter()
                    .map(|v| Scalar::Utf8(v.to_owned()))
                    .collect(),
            ),
        ));
    }
    let shapes: BTreeSet<_> = parts.iter().map(|p| &p.outputs).collect();
    if shapes.len() > 1 {
        let shapes = shapes
            .into_iter()
            .map(|v| {
                format!(
                    "({}{})",
                    v.iter().map(|s| quoted(s)).collect::<Vec<_>>().join(", "),
                    if v.len() == 1 { "," } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(CoreError::new(
            Sentence::ResultMergeOutputs,
            "parts were asked for different outputs",
        )
        .with_field("outputs", Detail::text(format!("[{shapes}]"))));
    }
    let mut seen = BTreeSet::new();
    let mut offset = 0u32;
    let mut keys = Vec::new();
    for (part_index, part) in parts.iter().enumerate() {
        if plan.explanation_part.is_none() && part.has_explanation {
            plan.explanation_part = Some(part_index);
        }
        plan.names_rows &= part.names_rows;
        plan.names_step |= part.names_step;
        for candidate in &part.candidates {
            let key = candidate_key(candidate)?;
            if key.as_ref().is_none_or(|key| !seen.contains(key)) {
                plan.candidate_positions.push(offset);
            }
            keys.extend(key);
            offset = offset
                .checked_add(1)
                .ok_or_else(|| invalid("too many result candidates"))?;
        }
        seen.extend(keys.drain(..));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn descriptor(candidates: Vec<C>) -> MergeDescriptor {
        MergeDescriptor {
            columns: None,
            manual_gradients: vec![],
            model: "m".into(),
            request_id: "one".into(),
            outputs: vec!["prediction".into()],
            has_explanation: false,
            names_rows: true,
            names_step: false,
            candidates,
            gradient_rows: vec![],
        }
    }
    fn json(s: &str) -> C {
        C::Json { value: s.into() }
    }
    #[test]
    fn exact_values_and_stable_union_are_shared_across_hosts() {
        let plan = plan(&[
            descriptor(vec![
                json("1"),
                json("1.0"),
                json("true"),
                json("9007199254740993"),
                json("9007199254740992"),
            ]),
            descriptor(vec![json("1"), json("NaN"), json("NaN")]),
        ])
        .unwrap();
        assert_eq!(plan.candidate_positions, [0, 1, 2, 3, 4, 6, 7]);
        let a = json("{\"a\":[1,null],\"b\":2}");
        let b = json("{\"b\":2.0,\"a\":[true,null]}");
        assert_eq!(candidate_key(&a).unwrap(), candidate_key(&b).unwrap());
    }
    #[test]
    fn decimals_key_by_exact_value() {
        let decimal = |v: &str| C::Decimal { value: v.into() };
        assert_eq!(
            candidate_key(&decimal("-1.50")).unwrap(),
            candidate_key(&json("-1.5")).unwrap()
        );
        assert_eq!(
            candidate_key(&decimal("1.000")).unwrap(),
            candidate_key(&json("true")).unwrap()
        );
        assert_ne!(
            candidate_key(&decimal("0.1")).unwrap(),
            candidate_key(&json("0.1")).unwrap()
        );
        for invalid in ["", "-", "1.", ".5", "1e3", "1.2.3"] {
            assert!(candidate_key(&decimal(invalid)).is_err(), "{invalid}");
        }
    }
    #[test]
    fn mixed_resolution_times_do_not_overflow_or_conflate_dates() {
        let a = C::DateTime {
            ticks: "253402300799999999".into(),
            unit: MergeTimeUnit::Us,
            aware: false,
        };
        let b = C::DateTime {
            ticks: "253402300799999999000".into(),
            unit: MergeTimeUnit::Ns,
            aware: false,
        };
        assert_eq!(candidate_key(&a).unwrap(), candidate_key(&b).unwrap());
        let b = C::DateTime {
            ticks: "253402300799999999".into(),
            unit: MergeTimeUnit::Us,
            aware: true,
        };
        assert_ne!(candidate_key(&a).unwrap(), candidate_key(&b).unwrap());
        assert_ne!(
            candidate_key(&C::Date { ordinal: 719163 }).unwrap(),
            candidate_key(&C::DateTime {
                ticks: "0".into(),
                unit: MergeTimeUnit::S,
                aware: false
            })
            .unwrap()
        );
    }
    #[test]
    fn headers_precede_candidate_validation_and_single_preserves_identity() {
        let one = descriptor(vec![C::Unsupported {
            type_name: "custom".into(),
        }]);
        assert!(plan(std::slice::from_ref(&one)).unwrap().single);
        let mut second = descriptor(vec![]);
        second.model = "other".into();
        second.outputs = vec![];
        let model = plan(&[one.clone(), second.clone()]).unwrap_err();
        assert_eq!(model.sentence, Sentence::ResultMergeModels);
        assert_eq!(
            model.fields.get("models"),
            Some(&Detail::List(vec![
                Scalar::Utf8("m".into()),
                Scalar::Utf8("other".into())
            ]))
        );
        second.model = "m".into();
        let outputs = plan(&[one.clone(), second]).unwrap_err();
        assert_eq!(outputs.sentence, Sentence::ResultMergeOutputs);
        assert!(outputs.fields.contains_key("outputs"));
        let candidate = plan(&[one, descriptor(vec![])]).unwrap_err();
        assert_eq!(candidate.sentence, Sentence::ResultMergeCandidateType);
        assert_eq!(
            candidate.fields.get("type_name"),
            Some(&Detail::text("custom"))
        );
        assert_eq!(plan(&[]).unwrap_err().sentence, Sentence::ResultMergeEmpty);
    }
}

/// Shared signed attribution policy, invoked only by explicit report requests.
pub fn feature_importance(
    input: &sdm_contract::result::FeatureImportanceInput,
) -> Result<sdm_contract::result::FeatureImportanceOutput, CoreError> {
    use sdm_contract::result::{FeatureImportanceOutput, FeatureImportanceRow};
    let invalid = |path: &str| {
        CoreError::new(
            Sentence::FeatureImportanceScore,
            "explanation score must be finite numeric attribution",
        )
        .with_field("path", sdm_contract::Detail::text(path))
    };
    let mut scores = std::collections::BTreeMap::<(&str, &str), f64>::new();
    for cell in &input.cells {
        let value: serde_json::Value =
            serde_json::from_str(&cell.score).map_err(|_| invalid(&cell.path))?;
        let value = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid(&cell.path))?;
        let score = scores.entry((&cell.table, &cell.column)).or_default();
        *score += value;
        if !score.is_finite() {
            return Err(invalid(&cell.path));
        }
    }
    // Scale before summing to avoid overflow of the absolute normalization mass.
    let max = scores.values().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let mass: f64 = if input.normalize && max > 0.0 {
        scores.values().map(|v| (v / max).abs()).sum()
    } else {
        0.0
    };
    Ok(FeatureImportanceOutput {
        rows: scores
            .into_iter()
            .map(|((table, column), score)| FeatureImportanceRow {
                table: table.into(),
                column: column.into(),
                score: if input.normalize && mass > 0.0 {
                    (score / max) / mass
                } else {
                    score
                },
            })
            .collect(),
    })
}

#[cfg(test)]
mod feature_importance_tests {
    use super::*;
    use sdm_contract::result::{AttributionCell, FeatureImportanceInput};

    fn cell(column: &str, score: &str) -> AttributionCell {
        AttributionCell {
            table: "t".into(),
            column: column.into(),
            score: score.into(),
            path: column.into(),
        }
    }

    #[test]
    fn signed_aggregate_normalization_and_empty() {
        let input = FeatureImportanceInput {
            cells: vec![cell("b", "4"), cell("a", "-2"), cell("b", "2")],
            normalize: true,
        };
        let out = feature_importance(&input).unwrap();
        assert_eq!(out.rows[0].column, "a");
        assert_eq!(out.rows[0].score, -0.25);
        assert_eq!(out.rows[1].score, 0.75);
        assert!(
            feature_importance(&FeatureImportanceInput {
                cells: vec![],
                normalize: true
            })
            .unwrap()
            .rows
            .is_empty()
        );
        let zeros = feature_importance(&FeatureImportanceInput {
            cells: vec![cell("a", "-0.0")],
            normalize: true,
        })
        .unwrap();
        assert_eq!(zeros.rows[0].score, 0.0);
    }

    #[test]
    fn rejects_invalid_scores_and_overflow() {
        for value in [
            "true",
            "\"2\"",
            "NaN",
            "Infinity",
            "-Infinity",
            "1e400",
            "{}",
        ] {
            assert!(
                feature_importance(&FeatureImportanceInput {
                    cells: vec![cell("bad", value)],
                    normalize: false
                })
                .is_err(),
                "{value}"
            );
        }
        assert!(
            feature_importance(&FeatureImportanceInput {
                cells: vec![cell("a", "1e308"), cell("a", "1e308")],
                normalize: false
            })
            .is_err()
        );
        let huge = feature_importance(&FeatureImportanceInput {
            cells: vec![cell("a", "123456789012345678901234567890")],
            normalize: false,
        })
        .unwrap();
        assert!(huge.rows[0].score.is_finite());
    }
}
