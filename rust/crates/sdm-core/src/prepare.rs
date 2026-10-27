// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The fixed model-neutral constructor and per-call native observation boundary.

use crate::run::{ArrowData, Asker, Attachment, Attachments};
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use sdm_contract::data::{AttachmentId, Relation};
use sdm_contract::ops::{OperationResult, SourceDescription, Stype};
use sdm_contract::prepare::*;
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_request::predict as policy;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const TASKS: [&str; 7] = [
    "classification",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "forecasting",
    "temporal_link_prediction",
    "ranking",
];
fn error(sentence: Sentence) -> CoreError {
    CoreError::new(sentence, "direct request preparation refused")
}
pub(crate) fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}
fn text_list(values: impl IntoIterator<Item = String>) -> Detail {
    Detail::List(values.into_iter().map(Scalar::Utf8).collect())
}
fn positions(values: impl IntoIterator<Item = usize>) -> Detail {
    Detail::List(
        values
            .into_iter()
            .map(|n| Scalar::UInt64(n as u64))
            .collect(),
    )
}
fn version(actual: u32, expected: u32) -> Result<(), CoreError> {
    if actual != expected {
        return Err(invalid(
            "direct preparation value has an unsupported version",
        ));
    }
    Ok(())
}
fn frame(form: FrameForm, name: &str) -> Result<(), CoreError> {
    let (kind, hint) = match form {
        FrameForm::Frame => return Ok(()),
        FrameForm::Dict => (
            "dict".to_owned(),
            "; wrap a single row with pd.DataFrame([row])",
        ),
        FrameForm::Series => (
            "Series".to_owned(),
            "; wrap a single row with pd.DataFrame([row])",
        ),
        FrameForm::Other { native_type } => (native_type, "; build one with pd.DataFrame(...)"),
    };
    Err(error(Sentence::DirectFrame)
        .with_field("name", Detail::text(name))
        .with_field("native_type", Detail::text(kind))
        .with_field("hint", Detail::text(hint)))
}
fn lookup_bounds(read: &NativeLookup, count: usize) -> Result<(), CoreError> {
    if read.positions.iter().any(|&n| n >= count)
        || read.positions.iter().collect::<BTreeSet<_>>().len() != read.positions.len()
    {
        return Err(invalid(
            "a native target lookup names an invalid column ordinal",
        ));
    }
    Ok(())
}
fn lookup(
    read: &NativeLookup,
    count: usize,
    source: &'static str,
    ordinal: Option<usize>,
) -> Result<(), CoreError> {
    lookup_bounds(read, count)?;
    if let Some(kind) = &read.unhashable {
        let mut fault = error(Sentence::DirectLookup)
            .with_field("native_type", Detail::text(kind))
            .with_field("lookup", Detail::text(source));
        if let Some(ordinal) = ordinal {
            fault = fault.with_field("ordinal", Detail::Scalar(Scalar::UInt64(ordinal as u64)));
        }
        return Err(fault);
    }
    Ok(())
}
fn header_shape(header: &FrameHeader) -> Result<(), CoreError> {
    if header
        .columns
        .iter()
        .enumerate()
        .any(|(n, c)| c.ordinal != n)
        || header.duplicates.iter().any(|&n| n >= header.columns.len())
        || header.duplicates.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid("a frame header has invalid ordinal bindings"));
    }
    Ok(())
}
pub(crate) fn names(header: &FrameHeader) -> Result<Vec<String>, CoreError> {
    header
        .columns
        .iter()
        .map(|c| match &c.name {
            HeaderName::Text { value } => Ok(value.clone()),
            _ => Err(error(Sentence::DirectUnfinished).with_field(
                "family",
                Detail::text(
                    "encodable string labels (unencodable or non-string labels remain unfinished)",
                ),
            )),
        })
        .collect()
}
fn check_names(header: &FrameHeader, table: &str, half: &str) -> Result<(), CoreError> {
    if !header.duplicates.is_empty() {
        return Err(error(Sentence::DirectDuplicateColumns)
            .with_field("table", Detail::text(table))
            .with_field("half", Detail::text(half))
            .with_field("positions", positions(header.duplicates.iter().copied())));
    }
    let nontext: Vec<usize> = header
        .columns
        .iter()
        .enumerate()
        .filter_map(|(n, c)| matches!(c.name, HeaderName::NonText).then_some(n))
        .collect();
    if !nontext.is_empty() {
        return Err(error(Sentence::DirectColumnNames)
            .with_field("table", Detail::text(table))
            .with_field("half", Detail::text(half))
            .with_field("positions", positions(nontext)));
    }
    names(header)?;
    Ok(())
}

/// Verify the detached positional Arrow snapshot before any value semantics.
pub(crate) fn coherent(header: &FrameHeader, rows: &ArrowData) -> Result<(), CoreError> {
    header_shape(header)?;
    if header.rows
        != rows
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>()
        || header.columns.len() != rows.schema.fields().len()
    {
        return Err(invalid(
            "direct frame header row/field counts disagree with Arrow",
        ));
    }
    for (n, column) in header.columns.iter().enumerate() {
        let field = rows.schema.field(n);
        let floating = matches!(
            field.data_type(),
            arrow_schema::DataType::Float32 | arrow_schema::DataType::Float64
        );
        if floating != column.float_storage.is_some() {
            return Err(invalid(
                "a direct floating column lacks coherent native storage facts",
            ));
        }
        let physical = sdm_contract::ColumnType::from_arrow(field.name(), field.data_type())?;
        if field.name() != &n.to_string() || Some(physical) != column.physical {
            return Err(invalid(
                "direct frame header physical type or positional name disagrees with Arrow",
            ));
        }
    }
    Ok(())
}
pub(crate) fn take_rows(
    attached: &mut Attachments,
    id: AttachmentId,
) -> Result<ArrowData, CoreError> {
    match attached.remove(&id) {
        Some(Attachment::Arrow(rows)) => Ok(rows),
        _ => Err(invalid("direct frame input does not name Arrow data")),
    }
}
pub(crate) fn renamed(header: &FrameHeader, rows: &ArrowData) -> Result<ArrowData, CoreError> {
    let names = names(header)?;
    let schema = Arc::new(Schema::new(
        rows.schema
            .fields()
            .iter()
            .zip(&names)
            .map(|(field, name)| Field::new(name, field.data_type().clone(), field.is_nullable()))
            .collect::<Vec<_>>(),
    ));
    let batches = rows
        .batches
        .iter()
        .map(|batch| {
            RecordBatch::try_new_with_options(
                schema.clone(),
                batch.columns().to_vec(),
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| invalid("direct frame names do not bind to their Arrow columns"))
        })
        .collect::<Result<_, _>>()?;
    Ok(ArrowData::new(schema, batches))
}
/// Held schemas use language kinds for edits, while a direct tabular body
/// must retain the caller's actual wire width from the same column facts.
fn read_schema(schema: sdm_contract::ops::HeldSchema) -> Result<sdm_schema::held::Read, CoreError> {
    let mut read = sdm_schema::held::read(schema.clone(), &BTreeMap::new(), true)?;
    for table in &mut read.description.tables {
        let Some(native) = schema.tables.iter().find(|t| t.name == table.name) else {
            continue;
        };
        for column in &mut table.columns {
            if let Some(native) = native.columns.iter().find(|c| c.name == column.name) {
                if let Ok(dtype) =
                    serde_json::from_value(serde_json::Value::String(native.dtype.clone()))
                {
                    column.dtype = dtype;
                }
            }
        }
    }
    Ok(read)
}

fn known(rows: &ArrowData, at: usize) -> usize {
    rows.batches
        .iter()
        .map(|b| b.column(at).len() - b.column(at).null_count())
        .sum()
}

fn constructor_context(
    headers: &ConstructorHeaders,
    context: &ArrowData,
    predict: &ArrowData,
    entity_table: &str,
    infer_schema: bool,
) -> Result<ConstructorContext, CoreError> {
    version(headers.version, DIRECT_PREPARED_VERSION)?;
    coherent(&headers.context, context)?;
    coherent(&headers.predict, predict)?;
    if headers.target >= headers.context.columns.len() {
        return Err(invalid("direct target ordinal is outside the header"));
    }
    if known(context, headers.target) == 0 {
        return Err(error(Sentence::DirectKnownTarget));
    }
    let extra: Vec<usize> = headers
        .predict_context
        .iter()
        .enumerate()
        .filter_map(|(n, v)| v.is_empty().then_some(n))
        .collect();
    if !extra.is_empty() {
        return Err(error(Sentence::DirectExtra)
            .with_field("half", Detail::text("predict"))
            .with_field("positions", positions(extra)));
    }
    if infer_schema {
        check_names(&headers.context, entity_table, "context")?;
    }
    let target = names(&headers.context)?[headers.target].clone();
    Ok(ConstructorContext {
        version: DIRECT_PREPARED_VERSION,
        task: headers.task.clone(),
        target,
        entity_table: entity_table.into(),
    })
}

pub(crate) async fn run(
    asker: Asker,
    input: Preparation,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let value = match input {
        Preparation::EncodeBody { input } => return encode_body(&asker, input, attached),
        Preparation::CheckResponseRows {
            scope,
            rows,
        } => serde_json::to_value(sdm_request::tabular::response::check_rows(
            scope,
            rows,
            crate::pql::float_text,
        )?),
        Preparation::InterpretResponseJson { checked, json } => serde_json::to_value(
            sdm_request::tabular::response::interpret_json(checked, json)?,
        ),
        Preparation::SelectResponseMember { interpreted, literal } => serde_json::to_value(
            sdm_request::tabular::response::select_member(interpreted, literal)?,
        ),
        Preparation::AttributeResponseRow { selected, string } => serde_json::to_value(
            sdm_request::tabular::response::attribute(selected, string, crate::pql::float_text)?,
        ),
        Preparation::LayoutResponse { data, attributed, decimal_limit } => {
            return crate::prepare_response::layout(&asker, data, attributed, decimal_limit, attached);
        }
        Preparation::Overlap { input } => {
            serde_json::to_value(overlap(&asker, input, &mut attached).await?)
        }
        Preparation::Envelope { input } => {
            serde_json::to_value(envelope(&asker, input, &mut attached).await?)
        }
        Preparation::Arguments {
            context,
            predict,
            related,
            task,
        } => {
            frame(context, "context")?;
            frame(predict, "predict")?;
            for (ordinal, form) in related.iter().enumerate() {
                frame(form.clone(), "related_tables").map_err(|error| {
                    error.with_field(
                        "related_ordinal",
                        Detail::Scalar(Scalar::UInt64(ordinal as u64)),
                    )
                })?;
            }
            if !related.is_empty() {
                return Err(error(Sentence::DirectRelated));
            }
            let Some(task) = task.value().filter(|task| TASKS.contains(&task.as_str())) else {
                return Err(error(Sentence::DirectTask));
            };
            serde_json::to_value(ConstructorArguments {
                version: DIRECT_PREPARED_VERSION,
                task: task.clone(),
            })
        }
        Preparation::Headers {
            arguments,
            context,
            predict,
            target_context,
            target_predict,
            predict_context,
        } => {
            version(arguments.version, DIRECT_PREPARED_VERSION)?;
            header_shape(&context)?;
            header_shape(&predict)?;
            lookup(&target_context, context.columns.len(), "target_context", None)?;
            if context.rows == 0 {
                return Err(error(Sentence::DirectEmpty));
            }
            if target_context.positions.is_empty() {
                return Err(error(Sentence::DirectTargetMissing));
            }
            lookup(&target_predict, predict.columns.len(), "target_predict", None)?;
            if !target_predict.positions.is_empty() {
                return Err(error(Sentence::DirectTargetScored));
            }
            if target_context.positions.len() > 1 {
                return Err(error(Sentence::DirectAmbiguous).with_field("source", Detail::text("known_target")));
            }
            if predict_context.len() != predict.columns.len()
                || predict_context
                    .iter()
                    .flatten()
                    .any(|&n| n >= context.columns.len())
            {
                return Err(invalid(
                    "scored-column native lookup facts have invalid ordinals",
                ));
            }
            serde_json::to_value(ConstructorHeaders {
                version: DIRECT_PREPARED_VERSION,
                task: arguments.task,
                context,
                predict,
                target: target_context.positions[0],
                predict_context,
            })
        }
        Preparation::Context { headers, context, predict, entity_table } => {
            let context = take_rows(&mut attached, context)?;
            let predict = take_rows(&mut attached, predict)?;
            serde_json::to_value(constructor_context(&headers, &context, &predict, &entity_table, true)?)
        }
        Preparation::InferredSchema { context, schema } => {
            version(context.version, DIRECT_PREPARED_VERSION)?;
            let description = read_schema(schema.clone())?.description;
            serde_json::to_value(ConstructorSchema { version: DIRECT_PREPARED_VERSION, task: context.task, target: context.target, entity_table: context.entity_table, description, schema })
        }
        Preparation::AdoptSchema { schema } => {
            let mut read = read_schema(schema.clone())?;
            sdm_schema::assemble::mark_keys(&mut read.description);
            let mut held = schema;
            for (table, current) in held.tables.iter_mut().zip(&read.description.tables) {
                table.time_column = current.time_column.clone();
                for (column, canonical) in table.columns.iter_mut().zip(&current.columns) {
                    column.stype = canonical.stype;
                    column.stype_before_key = canonical.stype_before_key;
                }
            }
            serde_json::to_value(held)
        }
        Preparation::ConsistentSchema { schema } => {
            let read = read_schema(schema.clone())?;
            check_consistent(&read)?;
            serde_json::to_value(schema)
        }
        Preparation::Schema {
            headers,
            context,
            predict,
            entity_table,
            schema,
        } => {
            let context = take_rows(&mut attached, context)?;
            let predict = take_rows(&mut attached, predict)?;
            let early = constructor_context(&headers, &context, &predict, &entity_table, schema.is_none())?;
            let target = early.target;
            let context = renamed(&headers.context, &context)?;
            let original_schema = schema.clone();
            let description = if let Some(schema) = schema {
                read_schema(schema)?.description
            } else {
                let relation = Relation::named(&entity_table);
                // Compatibility callers own this fixed Arrow snapshot. The
                // Python host uses the separate shared frame boundaries.
                let (nomination, first) = sdm_schema::frames::nominate(&entity_table, &relation, &context.schema, &context.batches)?;
                for notice in first { asker.warn(notice); }
                let (adopted, repeated) = sdm_schema::frames::adopt(nomination, &context.schema, &context.batches)?;
                for notice in repeated { asker.warn(notice); }
                let completed = sdm_schema::frames::choose_time(adopted, &context.schema, &context.batches)?;
                let read = sdm_schema::frames::observe(&completed.table, completed.time, &context.schema, &context.batches)?;
                let description = SourceDescription { tables: vec![completed.table], links: vec![] };
                for notice in sdm_schema::frames::final_notices(&[read], &description, false)? { asker.warn(notice); }
                description
            };
            let schema = original_schema.unwrap_or_else(|| {
                sdm_schema::held::held(
                    &description,
                    &description
                        .tables
                        .iter()
                        .flat_map(|t| t.columns.iter().map(|c| (t.name.clone(), c.name.clone())))
                        .collect(),
                    &BTreeMap::new(),
                )
            });
            serde_json::to_value(ConstructorSchema {
                version: DIRECT_PREPARED_VERSION,
                task: headers.task,
                target,
                entity_table,
                description,
                schema,
            })
        }
        Preparation::FinishConstructor {
            schema,
            context,
            predict,
            context_header,
            predict_header,
            target_context,
            context_predict,
        } => {
            version(schema.version, DIRECT_PREPARED_VERSION)?;
            let context = take_rows(&mut attached, context)?;
            let predict = take_rows(&mut attached, predict)?;
            coherent(&context_header, &context)?;
            coherent(&predict_header, &predict)?;
            if context_predict.len() != context_header.columns.len() {
                return Err(invalid("fresh constructor membership facts have another width"));
            }
            let ConstructorSchema { version: _, task, target, entity_table, description, schema } = schema;
            lookup_bounds(&target_context, context_header.columns.len())?;
            let target_positions: Vec<_> = context_header.columns.iter().enumerate().filter_map(|(at,column)|matches!(&column.name,HeaderName::Text { value } if value==&target).then_some(at)).collect();
            if target_positions != target_context.positions {
                return Err(invalid("fresh target lookup disagrees with its named header"));
            }
            for (column, found) in context_header.columns.iter().zip(&context_predict) {
                lookup_bounds(found, predict_header.columns.len())?;
                if let HeaderName::Text { value } = &column.name {
                    let actual: Vec<_> = predict_header.columns.iter().enumerate().filter_map(|(at,column)|matches!(&column.name,HeaderName::Text { value: candidate } if candidate==value).then_some(at)).collect();
                    if found.positions != actual {
                        return Err(invalid("fresh membership lookup disagrees with named headers"));
                    }
                }
            }
            let table = description.tables.iter().find(|t| t.name == entity_table);
            if let Some(table) = table {
                let arriving =
                    sdm_schema::frames::storage_dtypes(&predict.schema, &predict.batches)?;
                for (at, dtype) in arriving.into_iter().enumerate() {
                    let HeaderName::Text { value: name } = &predict_header.columns[at].name else {
                        continue;
                    };
                    if let Some(stated) = table.columns.iter().find(|c| c.name == *name) {
                        if schema.tables.iter().find(|t| t.name == entity_table)
                            .and_then(|t| t.columns.iter().find(|c| c.name == *name))
                            .is_some_and(|c| c.untyped && !c.stype_declared)
                        {
                            continue;
                        }
                        if predict_header
                            .columns
                            .iter()
                            .filter(|column| matches!(&column.name, HeaderName::Text { value } if value == name))
                            .count()
                            > 1
                        {
                            return Err(error(Sentence::DirectPredictDtype).with_field("ordinal", Detail::Scalar(Scalar::UInt64(at as u64))));
                        }
                        if !crate::edit::supports(stated.stype, dtype) {
                            return Err(error(Sentence::DirectScored)
                                .with_field("column", Detail::text(name))
                                .with_field("arriving", Detail::text(dtype.wire_name()))
                                .with_field("dtype", Detail::text(stated.dtype.wire_name()))
                                .with_field("stype", Detail::text(stype(stated.stype)))
                                .with_field("table", Detail::text(&entity_table)));
                        }
                    }
                }
            }
            let table = table.ok_or_else(|| {
                let mut tables: Vec<_> = description.tables.iter().map(|table|table.name.clone()).collect();
                tables.sort();
                error(Sentence::NoTable).with_field("table", Detail::text(&entity_table)).with_field("tables",text_list(tables))
            })?;
            let mut missing = Vec::new();
            for (at, (column, found)) in context_header.columns.iter().zip(&context_predict).enumerate() {
                lookup(found, predict_header.columns.len(), "context_predict", Some(at))?;
                let identity = matches!(&column.name, HeaderName::Text { value } if table.primary_key.contains(value));
                if !target_context.positions.contains(&at) && !identity && found.positions.is_empty() {
                    missing.push(at);
                }
            }
            if !missing.is_empty() {
                return Err(error(Sentence::DirectMissing)
                    .with_field("half", Detail::text("context"))
                    .with_field("positions", positions(missing)));
            }
            lookup(&target_context, context_header.columns.len(), "target_context", None)?;
            if target_context.positions.is_empty() {
                return Err(error(Sentence::DirectSchemaMissing).with_field("name", Detail::text(&target)));
            }
            let constant = if target_context.positions.len() > 1 {
                // The fresh target is a DataFrame: dropna removes rows with
                // any missing target. A remaining row reaches Series truth;
                // an empty result returns before that comparison.
                if context.batches.iter().any(|batch|(0..batch.num_rows()).any(|row|target_context.positions.iter().all(|&at|batch.column(at).is_valid(row)))) {
                    return Err(error(Sentence::DirectAmbiguous).with_field("source", Detail::text("target_variation")));
                }
                false
            } else {
                sdm_schema::frames::present_distinct(&context.batches, target_context.positions[0])? == 1
            };
            if constant
                && (task.contains("classification") || task == "regression")
            {
                asker.warn(
                    CoreWarning::new(
                        if task == "regression" { Sentence::DirectConstantRegression } else { Sentence::DirectConstantClass },
                        "every context row has the same target",
                    ).with_field("target", Detail::text(&target)),
                );
            }
            serde_json::to_value(PreparedDirectRequest {
                version: DIRECT_PREPARED_VERSION,
                task,
                target,
                description,
                schema,
            })
        }
        Preparation::ConstructorTask { prepared, entity_table, tables, table } => {
            version(prepared.version, DIRECT_PREPARED_VERSION)?;
            let present = tables.iter().any(|name| matches!(name,HeaderName::Text { value } if value==&entity_table));
            if present != table.is_some() {
                return Err(invalid("native task table lookup disagrees with its names"));
            }
            let sorted_names = |names: Vec<HeaderName>| {
                let mut names: Vec<_> = names.into_iter().filter_map(|name|match name {HeaderName::Text {value}=>Some(value),_=>None}).collect();
                names.sort();
                text_list(names)
            };
            let Some(table) = table else {
                return Err(error(Sentence::NoTable).with_field("table",Detail::text(entity_table)).with_field("tables",sorted_names(tables)));
            };
            let present = table.columns.iter().any(|name|matches!(name,HeaderName::Text { value } if value==&prepared.target));
            if present != table.target_dtype.is_some() {
                return Err(invalid("native task target lookup disagrees with its names"));
            }
            let Some(dtype) = table.target_dtype else {
                return Err(error(Sentence::NoColumn).with_field("table",Detail::text(entity_table)).with_field("column",Detail::text(prepared.target)).with_field("columns",sorted_names(table.columns)));
            };
            serde_json::to_value(PreparedDirectTask {kind:prepared.task,target:prepared.target,table:entity_table,dtype})
        }
        Preparation::Model { model } => {
            let model = policy::resolve_model(&model)
                .map_err(|_| {
                    let mut names: Vec<String> = policy::capabilities()
                        .iter()
                        .map(|spec| spec.model.to_owned())
                        .collect();
                    names.sort();
                    error(Sentence::DirectUnknownModel)
                        .with_field("model", Detail::text(&model))
                        .with_field("known_models", text_list(names))
                })?
                .to_owned();
            serde_json::to_value(CallPreparation::ModelResolved {
                version: CALL_PREPARATION_VERSION,
                model,
            })
        }
        Preparation::Outputs {
            call,
            outputs,
            kind,
        } => {
            let CallPreparation::ModelResolved { version: v, model } = call else {
                return Err(invalid("outputs require a model-resolved call"));
            };
            version(v, CALL_PREPARATION_VERSION)?;
            let named = outputs.is_some();
            let outputs =
                policy::resolve_outputs(&model, kind.as_deref().unwrap_or(""), outputs.as_deref())?;
            serde_json::to_value(CallPreparation::OutputsResolved {
                version: v,
                model,
                outputs,
                named,
            })
        }
        Preparation::Capabilities {
            call,
            kind,
            anchored,
            relationships,
            tables,
        } => {
            let CallPreparation::OutputsResolved {
                version: v,
                model,
                outputs,
                named,
            } = call
            else {
                return Err(invalid("capabilities require an outputs-resolved call"));
            };
            version(v, CALL_PREPARATION_VERSION)?;
            policy::validate_shape(&model, &kind, anchored, relationships, tables)?;
            serde_json::to_value(CallPreparation::ShapeAccepted {
                version: v,
                model,
                outputs,
                named,
                shape: CallShape {
                    kind,
                    anchored,
                    relationships,
                    tables,
                },
            })
        }
        Preparation::Task { call, kind } => {
            let CallPreparation::ShapeAccepted {
                version: v,
                model,
                outputs,
                named,
                shape,
            } = call
            else {
                return Err(invalid("task validation requires a shape-accepted call"));
            };
            version(v, CALL_PREPARATION_VERSION)?;
            policy::validate_task_outputs(&model, &kind, &outputs)?;
            serde_json::to_value(CallPreparation::TaskAccepted {
                version: v,
                model,
                outputs,
                named,
                shape,
                kind,
            })
        }
        Preparation::Capacity { call, rows } => {
            let CallPreparation::TaskAccepted {
                version: v,
                model,
                outputs,
                named,
                shape,
                kind,
            } = call
            else {
                return Err(invalid("capacity requires a task-accepted call"));
            };
            version(v, CALL_PREPARATION_VERSION)?;
            policy::validate_capacity(&model, rows)?;
            serde_json::to_value(CallPreparation::Bound {
                version: v,
                model,
                outputs,
                named,
                shape,
                kind,
                context_rows: rows,
            })
        }
    }
    .map_err(|_| invalid("a prepared request value cannot serialize"))?;
    let value = asker.attach(Attachment::Bytes(
        serde_json::to_vec(&value).map_err(|_| invalid("a prepared value cannot serialize"))?,
    ))?;
    Ok(OperationResult::PrepareRequest(PreparedRequest { value }))
}

pub(crate) fn stype(value: Stype) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Shared policy checks the earlier accepted observations, not later warning effects.
struct Bound<'a> {
    model: &'a str,
    outputs: &'a [String],
    named: bool,
    kind: &'a str,
    shape: &'a CallShape,
    context_rows: usize,
}
fn checked_bound(call: &CallPreparation) -> Result<Bound<'_>, CoreError> {
    let CallPreparation::Bound {
        version: v,
        model,
        outputs,
        named,
        kind,
        shape,
        context_rows,
    } = call
    else {
        return Err(invalid("direct data requires a bound call value"));
    };
    version(*v, CALL_PREPARATION_VERSION)?;
    policy::resolve_model(model)?;
    policy::validate_shape(
        model,
        &shape.kind,
        shape.anchored,
        shape.relationships,
        shape.tables,
    )?;
    policy::validate_task_outputs(model, kind, outputs)?;
    policy::validate_capacity(model, *context_rows)?;
    Ok(Bound {
        model,
        outputs,
        named: *named,
        kind,
        shape,
        context_rows: *context_rows,
    })
}
fn arriving(rows: &ArrowData) -> Result<Vec<(String, sdm_contract::ops::Dtype)>, CoreError> {
    Ok(rows
        .schema
        .fields()
        .iter()
        .zip(sdm_schema::frames::storage_dtypes(
            &rows.schema,
            &rows.batches,
        )?)
        .map(|(f, d)| (f.name().clone(), d))
        .collect())
}
fn float_storage(header: &FrameHeader) -> Vec<Option<FloatStorage>> {
    header.columns.iter().map(|c| c.float_storage).collect()
}
async fn overlap(
    asker: &Asker,
    input: DirectPrelude,
    attached: &mut Attachments,
) -> Result<DirectOverlap, CoreError> {
    let bound = checked_bound(&input.bound)?;
    // These are the observations consumed before any one-shot advice can
    // mutate native values or replace the references the adapter later takes.
    if input.task != bound.kind
        || input.context_header.rows != bound.context_rows
        || input.schema.tables.len() != bound.shape.tables
        || (!input.schema.links.is_empty()) != bound.shape.relationships
    {
        return Err(invalid(
            "direct prelude task, capacity or schema shape disagrees with its bound call",
        ));
    }
    if bound.model != policy::TABULAR {
        return Err(error(Sentence::DirectUnfinished).with_field(
            "family",
            Detail::text("the completed KumoTabular direct prediction family"),
        ));
    }
    let context = take_rows(attached, input.context)?;
    let scoring = take_rows(attached, input.predict)?;
    coherent(&input.context_header, &context)?;
    coherent(&input.predict_header, &scoring)?;
    let context = renamed(&input.context_header, &context)?;
    let scoring = renamed(&input.predict_header, &scoring)?;
    let read = read_schema(input.schema.clone())?;
    let (shown, notice) = sdm_request::tabular::leakage(
        &read.description,
        &input.schema,
        &input.entity_table,
        &context.batches,
        &scoring.batches,
        &arriving(&scoring)?,
        &float_storage(&input.predict_header),
        &input.target,
        crate::pql::float_text,
    )?;
    if let Some(notice) = notice {
        asker.warn(notice);
        asker.flush_warnings().await;
    }
    Ok(DirectOverlap {
        version: DIRECT_OVERLAP_VERSION,
        bound: input.bound,
        shown,
        predict_rows: input.predict_header.rows,
    })
}
fn check_consistent(read: &sdm_schema::held::Read) -> Result<(), CoreError> {
    for table in &read.description.tables {
        for column in &table.columns {
            if !crate::edit::supports(column.stype, column.dtype) {
                return Err(CoreError::new(
                    Sentence::MeaningNotCarried,
                    "a schema meaning contradicts its storage",
                )
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name))
                .with_field("dtype", Detail::text(column.dtype.wire_name()))
                .with_field("stype", Detail::text(stype(column.stype))));
            }
        }
    }
    sdm_schema::assemble::check_links(&read.description, &read.types)?;
    Ok(())
}

async fn envelope(
    asker: &Asker,
    input: DirectEnvelope,
    attached: &mut Attachments,
) -> Result<TabularEnvelope, CoreError> {
    version(input.overlap.version, DIRECT_OVERLAP_VERSION)?;
    let bound = checked_bound(&input.overlap.bound)?;
    if input.overlap.shown > input.overlap.predict_rows {
        return Err(invalid("overlap count exceeds its captured scoring rows"));
    }
    if bound.model != policy::TABULAR {
        return Err(invalid(
            "a tabular envelope requires a checked tabular call",
        ));
    }
    // Fresh adapter captures are after overlap advice. They are not a second
    // model-policy boundary: source warning effects can change these facts.
    let context = take_rows(attached, input.context)?;
    let scoring = take_rows(attached, input.predict)?;
    coherent(&input.context_header, &context)?;
    coherent(&input.predict_header, &scoring)?;
    let scoring = renamed(&input.predict_header, &scoring)?;
    let read = read_schema(input.schema.clone())?;
    check_consistent(&read)?;
    let (columns, notices) = sdm_request::tabular::envelope(
        &read.description,
        &input.schema,
        &input.entity_table,
        &names(&input.context_header)?,
        &scoring.batches,
        &arriving(&scoring)?,
        &float_storage(&input.predict_header),
        &input.target,
        &input.task,
        crate::edit::supports,
    )?;
    if !notices.is_empty() {
        for notice in notices {
            asker.warn(notice);
        }
        asker.flush_warnings().await;
    }
    Ok(TabularEnvelope {
        version: TABULAR_ENVELOPE_VERSION,
        bound: input.overlap.bound,
        columns,
    })
}

/// Consume source-position body observations with the earlier resolved columns.
fn encode_body(
    asker: &Asker,
    input: DirectBody,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    version(input.envelope.version, TABULAR_ENVELOPE_VERSION)?;
    let bound = checked_bound(&input.envelope.bound)?;
    if bound.model != policy::TABULAR {
        return Err(invalid("encoded direct body belongs to another model"));
    }
    let outputs = bound.outputs.to_vec();
    let context = take_rows(&mut attached, input.context)?;
    let predict = take_rows(&mut attached, input.predict)?;
    coherent(&input.context_header, &context)?;
    coherent(&input.predict_header, &predict)?;
    let context = renamed(&input.context_header, &context)?;
    let predict = renamed(&input.predict_header, &predict)?;
    let payload = sdm_request::tabular::body(
        &input.envelope.columns,
        &context.batches,
        &predict.batches,
        &input.primary_key,
        &input.task,
        &outputs,
        crate::pql::float_text,
    )?;
    let body = EncodedDirectBody {
        version: ENCODED_DIRECT_BODY_VERSION,
        bound: input.envelope.bound,
        columns: input.envelope.columns,
        kind: input.task,
        primary_key: input.primary_key,
        context_shape: EncodedTableShape {
            columns: names(&input.context_header)?,
            rows: input.context_header.rows,
        },
        predict_shape: EncodedTableShape {
            columns: names(&input.predict_header)?,
            rows: input.predict_header.rows,
        },
        payload: asker.attach(Attachment::Bytes(payload.to_bytes()))?,
    };
    let value = asker.attach(Attachment::Bytes(
        serde_json::to_vec(&body).map_err(|_| invalid("encoded direct body cannot serialize"))?,
    ))?;
    Ok(OperationResult::PrepareRequest(PreparedRequest { value }))
}

pub(crate) async fn predict(
    asker: Asker,
    input: DirectPredict,
    request: sdm_contract::ops::Predict,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let call = match &input {
        DirectPredict::Empty { bound, .. } => bound,
        DirectPredict::Encoded { body, .. } => {
            version(body.version, ENCODED_DIRECT_BODY_VERSION)?;
            &body.bound
        }
    };
    let bound = checked_bound(call)?;
    if bound.model != request.model
        || bound.outputs != request.outputs
        || bound.named == request.default_outputs
    {
        return Err(invalid(
            "direct predict envelope disagrees with its bound call",
        ));
    }
    if bound.model != policy::TABULAR {
        return Err(error(Sentence::DirectUnfinished).with_field(
            "family",
            Detail::text("the completed KumoTabular direct prediction family"),
        ));
    }
    let outputs = bound.outputs.to_vec();
    let named = bound.named;
    let model = bound.model.to_owned();
    let kind = bound.kind.to_owned();
    let payload = match input {
        DirectPredict::Empty {
            schema,
            entity_table,
            predict_header,
            ..
        } => {
            if predict_header.rows != 0 {
                return Err(invalid("empty direct predict has scoring rows"));
            }
            let read = read_schema(schema)?;
            let table = read
                .description
                .tables
                .iter()
                .find(|t| t.name == entity_table);
            let scoring_names = names(&predict_header)?;
            let names_rows = table.is_some_and(|t| {
                !t.primary_key.is_empty()
                    && t.primary_key.iter().all(|key| scoring_names.contains(key))
            });
            let payload = sdm_wire::json::Value::Object(vec![(
                "task".into(),
                sdm_wire::json::Value::Object(vec![(
                    "kind".into(),
                    sdm_wire::json::Value::text(&kind),
                )]),
            )]);
            return crate::predict::finish_direct(
                &asker,
                policy::empty(&model, &payload),
                outputs,
                Some(names_rows),
            );
        }
        DirectPredict::Encoded { body, inference } => {
            let Some(Attachment::Bytes(bytes)) = attached.remove(&body.payload) else {
                return Err(invalid("encoded direct body bytes are missing"));
            };
            let payload = sdm_wire::json::read(&bytes)
                .map_err(|_| invalid("encoded direct body is unreadable"))?;
            sdm_request::tabular::check_body(&payload, &body, &outputs)?;
            sdm_request::tabular::complete_body(payload, &request.request_id, inference)?
        }
    };
    let bytes = policy::call_bytes(&payload)?;
    let (response_value, response) =
        crate::predict::call_response(&asker, bytes, request.model_endpoint.as_ref(), true).await?;
    crate::prepare_response::decode(&asker, &payload, response_value, &response, outputs, named)
}
