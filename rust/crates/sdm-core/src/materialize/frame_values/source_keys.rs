// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native key policy over useful immutable constructor and identity facts.

use super::*;

pub(super) fn constructed(values: &FrameConstructorValues) -> Result<Dtype, CoreError> {
    sdm_schema::frames::native_column::dtype(values)
}

pub(super) fn native(reason: &str, table: &str, column: &str) -> CoreError {
    sdm_engine::frame_identity::refusal(reason, table, column)
}

pub(crate) fn definition(value: &FrameCompositeDefinition) -> Result<(), CoreError> {
    let names = value
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    if names.len() < 2
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
        || value.derived != sdm_engine::graph::folded_name(&names)
        || value.derived.len() > sdm_engine::graph::FOLDED_NAME_MOST
        || value.columns.iter().any(|column| {
            !matches!(
                column.dtype,
                Dtype::Bool
                    | Dtype::Int
                    | Dtype::String
                    | Dtype::Binary
                    | Dtype::Date
                    | Dtype::Timedelta
            )
        })
    {
        return Err(protocol(
            "native identity definition contradicts its ordered component meaning",
        ));
    }
    Ok(())
}

pub(super) fn key(
    table: FrameTableMeaning,
    primary_key: Vec<String>,
) -> Result<FrameKeyMeaning, CoreError> {
    if table
        .constructed
        .iter()
        .map(|column| &column.name)
        .collect::<BTreeSet<_>>()
        .len()
        != table.constructed.len()
        || table.table.columns.iter().any(|column| {
            !table
                .constructed
                .iter()
                .any(|fixed| column.name == fixed.name)
        })
    {
        return Err(protocol(
            "constructed table columns differ from their fixed source identities",
        ));
    }
    let composite = if primary_key.len() > 1 {
        let mut columns = Vec::with_capacity(primary_key.len());
        let missing = primary_key
            .iter()
            .filter(|name| !table.constructed.iter().any(|column| &column.name == *name))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(native("components_missing", &table.table.name, "")
                .with_field(
                    "columns",
                    Detail::List(primary_key.iter().cloned().map(Scalar::Utf8).collect()),
                )
                .with_field(
                    "missing",
                    Detail::List(missing.into_iter().map(Scalar::Utf8).collect()),
                ));
        }
        for name in &primary_key {
            let column = table
                .constructed
                .iter()
                .find(|column| &column.name == name)
                .expect("all components were checked");
            if matches!(
                column.dtype,
                Dtype::Float | Dtype::Float32 | Dtype::Float64 | Dtype::Floatlist
            ) {
                return Err(
                    native("approximate_component", &table.table.name, name).with_field(
                        "dtype",
                        Detail::text(if column.dtype == Dtype::Floatlist {
                            "floatlist"
                        } else {
                            "float"
                        }),
                    ),
                );
            }
            columns.push(FrameIdentityComponent {
                name: column.name.clone(),
                dtype: column.dtype,
            });
        }
        let derived = sdm_engine::graph::folded_name(&primary_key);
        if derived.len() > sdm_engine::graph::FOLDED_NAME_MOST {
            return Err(native("name_long", &table.table.name, "")
                .with_field(
                    "limit",
                    Detail::Scalar(Scalar::UInt64(sdm_engine::graph::FOLDED_NAME_MOST as u64)),
                )
                .with_field(
                    "columns",
                    Detail::List(primary_key.iter().cloned().map(Scalar::Utf8).collect()),
                ));
        }
        let selected = FrameCompositeDefinition {
            table: table.table.name.clone(),
            source: table.table.source.clone(),
            columns,
            derived,
            codec: FrameIdentityCodec::EscapedTextV1,
        };
        definition(&selected)?;
        Some(selected)
    } else {
        None
    };
    Ok(FrameKeyMeaning {
        table,
        primary_key,
        composite,
    })
}

pub(super) fn nulls(input: FrameKeyNulls) -> Result<FrameKeyNulls, CoreError> {
    definition(&input.definition)?;
    if !input
        .definition
        .columns
        .iter()
        .any(|column| column.name == input.column)
    {
        return Err(protocol(
            "native missingness names an undeclared identity component",
        ));
    }
    if input.missing {
        return Err(native(
            "component_null",
            &input.definition.table,
            &input.column,
        ));
    }
    Ok(input)
}

pub(super) fn metadata(input: FrameTableMetadata) -> Result<FrameTableMeaning, CoreError> {
    let expected = key(input.key.table.clone(), input.key.primary_key.clone())?;
    if expected != input.key
        || input.table.name != expected.table.table.name
        || input.table.source != expected.table.table.source
    {
        return Err(protocol(
            "native metadata no longer names its selected constructed table key",
        ));
    }
    let mut table = input.table;
    if let Some(definition) = &expected.composite {
        if let Some(fixed) = expected
            .table
            .constructed
            .iter()
            .find(|column| column.name == definition.derived)
        {
            if !crate::edit::supports(Stype::Id, fixed.dtype) {
                return Err(native("key_dtype", &table.name, &fixed.name)
                    .with_field(
                        "dtype",
                        Detail::text(super::super::name_of(serde_json::to_value(fixed.dtype))),
                    )
                    .with_field("stype", Detail::text("ID")));
            }
            if let Some(column) = table
                .columns
                .iter()
                .find(|column| column.name == definition.derived)
            {
                if !crate::edit::supports(column.stype, fixed.dtype) {
                    return Err(native("key_dtype", &table.name, &column.name)
                        .with_field(
                            "dtype",
                            Detail::text(super::super::name_of(serde_json::to_value(fixed.dtype))),
                        )
                        .with_field(
                            "stype",
                            Detail::text(super::super::name_of(serde_json::to_value(column.stype))),
                        ));
                }
                if column.stype != Stype::Id {
                    return Err(
                        native("primary_stype", &table.name, &column.name).with_field(
                            "stype",
                            Detail::text(super::super::name_of(serde_json::to_value(column.stype))),
                        ),
                    );
                }
            }
        }
    }
    // Later caller dtype edits cannot recast already constructed rows.
    for column in &mut table.columns {
        if let Some(frozen) = expected
            .table
            .table
            .columns
            .iter()
            .find(|frozen| frozen.name == column.name)
        {
            column.dtype.clone_from(&frozen.dtype);
        }
    }
    table.primary_key = expected.primary_key;
    table.frame_key_form = expected.table.table.frame_key_form;
    Ok(FrameTableMeaning {
        table,
        constructed: expected.table.constructed,
    })
}

/// Native task classification reads the final registered graph before labels.
pub(super) fn task(graph: &Graph, target: &Shape) -> Result<(), CoreError> {
    if let Shape::Aggregated(window) = target
        && window.aggregation == labels::Aggregation::ListDistinct
        && super::super::link_target_role(graph, target).is_none()
    {
        let mut error = CoreError::new(
            Sentence::FrameNativeInternal,
            "native LIST_DISTINCT task has no unique registered target foreign key",
        )
        .with_field(
            "native_key_reason",
            Detail::text("unregistered_list_target"),
        )
        .with_field(
            "column",
            Detail::text(
                window
                    .column
                    .as_deref()
                    .ok_or_else(|| protocol("LIST_DISTINCT task lacks its target column"))?,
            ),
        )
        .with_field("table", Detail::text(&window.table));
        error.public_details = Some(Box::default());
        return Err(error);
    }
    Ok(())
}

/// Preserve the native ValueError cause at this reached frame task boundary.
/// The original semantic sentence and validation ordering remain unchanged.
pub(super) fn task_error(mut error: CoreError) -> CoreError {
    if matches!(
        error.sentence,
        Sentence::IterationsBelowOne | Sentence::AnchorEntityNotShared
    ) {
        error
            .fields
            .insert("native_key_reason".into(), Detail::text("task_validation"));
        error.public_details = Some(Box::default());
    }
    error
}
