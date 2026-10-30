// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-neutral direct shape guards and shared model preflight facts.

use sdm_contract::direct::{BoundDirectCall, ColumnFlags, DirectCallFacts, DirectShape, FrameKind};
use sdm_contract::ops::{DescribedTable, Dtype, Stype};
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use std::collections::BTreeSet;

const TASKS: [&str; 7] = [
    "classification",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "forecasting",
    "temporal_link_prediction",
    "ranking",
];

pub fn invalid(sentence: Sentence) -> CoreError {
    CoreError::new(sentence, "a direct request does not satisfy its contract")
}
pub fn texts(values: impl IntoIterator<Item = String>) -> Detail {
    Detail::List(values.into_iter().map(Scalar::Utf8).collect())
}

fn frame(kind: &FrameKind, parameter: &str) -> Result<(), CoreError> {
    let (native_type, hint) = match kind {
        FrameKind::Frame => return Ok(()),
        FrameKind::Dict => ("dict", "; wrap a single row with pd.DataFrame([row])"),
        FrameKind::Series => ("Series", "; wrap a single row with pd.DataFrame([row])"),
        FrameKind::Other { native_type } => {
            (native_type.as_str(), "; build one with pd.DataFrame(...)")
        }
    };
    Err(invalid(Sentence::DirectFrame)
        .with_field("parameter", Detail::text(parameter))
        .with_field("native_type", Detail::text(native_type))
        .with_field("hint", Detail::text(hint)))
}

/// Runs before any column data or schema is converted.
pub fn arguments(input: &DirectShape) -> Result<(), CoreError> {
    frame(&input.context_kind, "context")?;
    frame(&input.predict_kind, "predict")?;
    for (name, kind) in &input.related {
        frame(kind, &format!("related_tables[{name:?}]"))?;
    }
    if !input.related.is_empty() {
        let mut names: Vec<_> = input.related.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        return Err(invalid(Sentence::DirectRelated).with_field("tables", texts(names)));
    }
    if !input
        .task
        .as_deref()
        .is_some_and(|task| TASKS.contains(&task))
    {
        return Err(invalid(Sentence::DirectTask));
    }
    Ok(())
}

pub fn shape(input: &DirectShape) -> Result<(), CoreError> {
    arguments(input)?;
    if input.target_containers {
        return Err(
            invalid(Sentence::DirectMultilabel).with_field("target", Detail::text(&input.target))
        );
    }
    if input.context.rows == 0 {
        return Err(invalid(Sentence::DirectEmpty));
    }
    if !input.context.columns.contains(&input.target) {
        let mut columns = input.context.columns.clone();
        columns.sort();
        return Err(invalid(Sentence::DirectTargetMissing)
            .with_field("target", Detail::text(&input.target))
            .with_field("columns", texts(columns)));
    }
    if input.predict.columns.contains(&input.target) {
        return Err(
            invalid(Sentence::DirectTargetScored).with_field("target", Detail::text(&input.target))
        );
    }
    Ok(())
}

/// The schema inference boundary's ordinary string labels, checked in the core.
pub fn column_names(columns: &[String], table: &str) -> Result<(), CoreError> {
    let mut names = BTreeSet::new();
    let mut duplicate = BTreeSet::new();
    for name in columns {
        if !names.insert(name) {
            duplicate.insert(name.clone());
        }
    }
    if !duplicate.is_empty() {
        return Err(invalid(Sentence::DirectDuplicateColumns)
            .with_field("table", Detail::text(table))
            .with_field("columns", texts(duplicate)));
    }
    Ok(())
}

/// Resolve model/output/capability/task/capacity before inspecting scoring values.
pub fn bind(input: DirectCallFacts) -> Result<BoundDirectCall, CoreError> {
    crate::predict::bind_direct(input)
}

/// Validate the producer's typed decision against its original consumed facts.
pub fn bound(value: &BoundDirectCall) -> Result<(), CoreError> {
    if bind(value.facts.clone())? != *value {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "bound direct call differs from its consumed facts",
        ));
    }
    Ok(())
}

/// Ordered pre-schema constructor checks over already-observed data.
pub fn population(shape: &DirectShape, known_targets: usize) -> Result<(), CoreError> {
    if known_targets == 0 {
        return Err(invalid(Sentence::DirectKnownTarget));
    }
    let extra: Vec<_> = shape
        .predict
        .columns
        .iter()
        .filter(|name| !shape.context.columns.contains(name))
        .cloned()
        .collect();
    if !extra.is_empty() {
        return Err(invalid(Sentence::DirectExtra).with_field("columns", texts(extra)));
    }
    Ok(())
}

/// Scoring compatibility and required features, after schema inference.
pub fn scored(
    shape: &DirectShape,
    table: &DescribedTable,
    flags: &[ColumnFlags],
    arriving: &[Dtype],
    supports: fn(Stype, Dtype) -> bool,
) -> Result<(), CoreError> {
    for (name, dtype) in shape.predict.columns.iter().zip(arriving) {
        scored_column(table, flags, name, Some(*dtype), supports)?;
    }
    scored_complete(shape, table)
}

pub fn scored_column(
    table: &DescribedTable,
    flags: &[ColumnFlags],
    name: &str,
    arriving: Option<Dtype>,
    supports: fn(Stype, Dtype) -> bool,
) -> Result<(), CoreError> {
    if let Some(dtype) = arriving {
        if let Some(column) = table.columns.iter().find(|column| &column.name == name) {
            let flags = flags.iter().find(|flags| &flags.name == name);
            if flags.is_some_and(|flags| flags.untyped && !flags.stype_declared) {
                return Ok(());
            }
            if !supports(column.stype, dtype) {
                return Err(invalid(Sentence::DirectScored)
                    .with_field("arriving", Detail::text(dtype.wire_name()))
                    .with_field("column", Detail::text(name))
                    .with_field("dtype", Detail::text(column.dtype.wire_name()))
                    .with_field("stype", Detail::text(stype_name(column.stype)))
                    .with_field("table", Detail::text(&table.name)));
            }
        }
    }
    Ok(())
}

pub fn scored_complete(shape: &DirectShape, table: &DescribedTable) -> Result<(), CoreError> {
    let missing: Vec<_> = shape
        .context
        .columns
        .iter()
        .filter(|name| {
            *name != &shape.target
                && !table.primary_key.contains(name)
                && !shape.predict.columns.contains(name)
        })
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(invalid(Sentence::DirectMissing).with_field("columns", texts(missing)));
    }
    Ok(())
}

pub fn stype_name(value: Stype) -> &'static str {
    match value {
        Stype::Numerical => "numerical",
        Stype::Categorical => "categorical",
        Stype::Multicategorical => "multicategorical",
        Stype::Id => "ID",
        Stype::Text => "text",
        Stype::Timestamp => "timestamp",
        Stype::Sequence => "sequence",
        Stype::Image => "image",
        Stype::Unsupported => "unsupported",
    }
}
