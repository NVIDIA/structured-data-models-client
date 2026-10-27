// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The fixed model-neutral constructor and per-call native observation boundary.

use crate::attached::take_json;
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
fn lookup(read: &NativeLookup, count: usize) -> Result<(), CoreError> {
    if read.positions.iter().any(|&n| n >= count)
        || read.positions.iter().collect::<BTreeSet<_>>().len() != read.positions.len()
    {
        return Err(invalid(
            "a native target lookup names an invalid column ordinal",
        ));
    }
    if let Some(kind) = &read.unhashable {
        return Err(error(Sentence::DirectLookup).with_field("native_type", Detail::text(kind)));
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

pub(crate) async fn run(
    asker: Asker,
    input: Preparation,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let value = match input {
        Preparation::Arguments {
            context,
            predict,
            related,
            task,
        } => {
            frame(context, "context")?;
            frame(predict, "predict")?;
            for (ordinal, (_, form)) in related.iter().enumerate() {
                frame(form.clone(), "related_tables").map_err(|error| {
                    error.with_field(
                        "related_ordinal",
                        Detail::Scalar(Scalar::UInt64(ordinal as u64)),
                    )
                })?;
            }
            if !related.is_empty() {
                let mut named: Vec<String> = related.iter().map(|(name, _)| name.clone()).collect();
                named.sort();
                return Err(
                    error(Sentence::DirectRelated).with_field("related_tables", text_list(named))
                );
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
            lookup(&target_context, context.columns.len())?;
            if context.rows == 0 {
                return Err(error(Sentence::DirectEmpty));
            }
            if target_context.positions.is_empty() {
                return Err(error(Sentence::DirectTargetMissing));
            }
            lookup(&target_predict, predict.columns.len())?;
            if !target_predict.positions.is_empty() {
                return Err(error(Sentence::DirectTargetScored));
            }
            if target_context.positions.len() > 1 {
                return Err(error(Sentence::DirectAmbiguous));
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
        Preparation::Schema {
            headers,
            context,
            predict,
            entity_table,
            schema,
        } => {
            version(headers.version, DIRECT_PREPARED_VERSION)?;
            let context = take_rows(&mut attached, context)?;
            let predict = take_rows(&mut attached, predict)?;
            coherent(&headers.context, &context)?;
            coherent(&headers.predict, &predict)?;
            if headers.target >= headers.context.columns.len() {
                return Err(invalid("direct target ordinal is outside the header"));
            }
            if known(&context, headers.target) == 0 {
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
            if schema.is_none() {
                check_names(&headers.context, &entity_table, "context")?;
            }
            let context = renamed(&headers.context, &context)?;
            let predict = renamed(&headers.predict, &predict)?;
            let target = names(&headers.context)?[headers.target].clone();
            let original_schema = schema.clone();
            let description = if let Some(schema) = schema {
                read_schema(schema)?.description
            } else {
                let relation = Relation::named(&entity_table);
                let mut read = sdm_schema::frames::read(
                    &entity_table,
                    &relation,
                    &context.schema,
                    &context.batches,
                )?;
                for notice in read.hold_inferred()? {
                    asker.warn(notice);
                }
                let duplicates = if read.table.primary_key.is_empty() {
                    0
                } else {
                    read.count(&read.table.primary_key).repeats()
                };
                let mut table = sdm_schema::assemble::described_table(
                    read.table.clone(),
                    Some(Scalar::UInt64(duplicates)),
                );
                if !table.primary_key.is_empty() {
                    table.frame_key_form = Some(sdm_contract::frames::FrameKeyForm::Scalar);
                }
                let mut description = SourceDescription {
                    tables: vec![table],
                    links: vec![],
                };
                sdm_schema::assemble::mark_keys(&mut description);
                for notice in sdm_schema::frames::final_notices(&[read], &description, false)? {
                    asker.warn(notice);
                }
                description
            };
            let table = description.tables.iter().find(|t| t.name == entity_table);
            if let Some(table) = table {
                let arriving =
                    sdm_schema::frames::storage_dtypes(&predict.schema, &predict.batches)?;
                for (at, dtype) in arriving.into_iter().enumerate() {
                    let name = predict.schema.field(at).name().clone();
                    if let Some(stated) = table.columns.iter().find(|c| c.name == name) {
                        if original_schema
                            .as_ref()
                            .and_then(|s| s.tables.iter().find(|t| t.name == entity_table))
                            .and_then(|t| t.columns.iter().find(|c| c.name == name))
                            .is_some_and(|c| c.untyped && !c.stype_declared)
                        {
                            continue;
                        }
                        if predict
                            .schema
                            .fields()
                            .iter()
                            .filter(|f| f.name() == &name)
                            .count()
                            > 1
                        {
                            return Err(error(Sentence::DirectPredictDtype));
                        }
                        if !crate::edit::supports(stated.stype, dtype) {
                            return Err(error(Sentence::DirectScored)
                                .with_field("column", Detail::text(&name))
                                .with_field("arriving", Detail::text(dtype.wire_name()))
                                .with_field("dtype", Detail::text(stated.dtype.wire_name()))
                                .with_field("stype", Detail::text(stype(stated.stype)))
                                .with_field("table", Detail::text(&entity_table)));
                        }
                    }
                }
            }
            let table = table.ok_or_else(|| {
                error(Sentence::DirectSchemaMissing).with_field("name", Detail::text(&entity_table))
            })?;
            let missing: Vec<String> = names(&headers.context)?
                .into_iter()
                .filter(|name| {
                    name != &target
                        && !table.primary_key.contains(name)
                        && !predict.schema.fields().iter().any(|f| f.name() == name)
                })
                .collect();
            if !missing.is_empty() {
                return Err(
                    error(Sentence::DirectMissing).with_field("columns", text_list(missing))
                );
            }
            if sdm_schema::frames::present_distinct(&context.batches, headers.target)? == 1
                && (headers.task.contains("classification") || headers.task == "regression")
            {
                asker.warn(
                    CoreWarning::new(
                        if headers.task == "regression" {
                            Sentence::DirectConstantRegression
                        } else {
                            Sentence::DirectConstantClass
                        },
                        "every context row has the same target",
                    )
                    .with_field("target", Detail::text(&target)),
                );
            }
            if !table.columns.iter().any(|c| c.name == target) {
                return Err(
                    error(Sentence::DirectSchemaMissing).with_field("name", Detail::text(&target))
                );
            }
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
            serde_json::to_value(PreparedDirectRequest {
                version: DIRECT_PREPARED_VERSION,
                task: headers.task,
                target,
                description,
                schema,
            })
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
            })
        }
        Preparation::Task { call, kind } => {
            let CallPreparation::ShapeAccepted {
                version: v,
                model,
                outputs,
                named,
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
                kind,
            })
        }
        Preparation::Capacity { call, rows } => {
            let CallPreparation::TaskAccepted {
                version: v,
                model,
                outputs,
                named,
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

/// Predict only after the host has reached and completed each native bind read.
pub(crate) async fn predict(
    asker: Asker,
    input: DirectPredict,
    request: sdm_contract::ops::Predict,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let CallPreparation::Bound {
        version: v,
        model,
        outputs,
        named,
        kind,
        ..
    } = input.bound
    else {
        return Err(invalid("direct predict requires a bound call value"));
    };
    version(v, CALL_PREPARATION_VERSION)?;
    if model != request.model || outputs != request.outputs || named == request.default_outputs {
        return Err(invalid(
            "direct predict envelope disagrees with its bound call",
        ));
    }
    if model != policy::TABULAR {
        return Err(error(Sentence::DirectUnfinished).with_field(
            "family",
            Detail::text("the completed KumoTabular direct prediction family"),
        ));
    }
    let native_schema = input.schema.clone();
    let read = read_schema(input.schema)?;
    if input.empty {
        if input.context.is_some() || input.predict.is_some() || input.predict_header.rows != 0 {
            return Err(invalid(
                "empty direct predict must name no Arrow halves and no scoring rows",
            ));
        }
        let table = read
            .description
            .tables
            .iter()
            .find(|t| t.name == input.entity_table);
        let scoring_names = names(&input.predict_header)?;
        let names_rows = table.is_some_and(|table| {
            !table.primary_key.is_empty()
                && table
                    .primary_key
                    .iter()
                    .all(|key| scoring_names.contains(key))
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
    let context_header = input
        .context_header
        .ok_or_else(|| invalid("nonempty direct prediction lacks a context header"))?;
    let context = take_rows(
        &mut attached,
        input
            .context
            .ok_or_else(|| invalid("nonempty direct prediction lacks context rows"))?,
    )?;
    let scoring = take_rows(
        &mut attached,
        input
            .predict
            .ok_or_else(|| invalid("nonempty direct prediction lacks scoring rows"))?,
    )?;
    coherent(&context_header, &context)?;
    coherent(&input.predict_header, &scoring)?;
    let context = renamed(&context_header, &context)?;
    let scoring = renamed(&input.predict_header, &scoring)?;
    let arriving = sdm_schema::frames::storage_dtypes(&scoring.schema, &scoring.batches)?;
    let arriving: Vec<_> = scoring
        .schema
        .fields()
        .iter()
        .zip(arriving)
        .map(|(field, dtype)| (field.name().clone(), dtype))
        .collect();
    if let Some(warning) = sdm_request::tabular::leakage(
        &read.description,
        &native_schema,
        &input.entity_table,
        &context.batches,
        &scoring.batches,
        &arriving,
        &input.target,
        crate::pql::float_text,
    )? {
        asker.warn(warning);
    }
    // Consistency is consumed at the adapter's body boundary, rather than
    // rerunning constructor target/feature checks on the current frames.
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
    let (payload, notices) = sdm_request::tabular::body(
        &read.description,
        &native_schema,
        &input.entity_table,
        &context.batches,
        &scoring.batches,
        &arriving,
        &input.target,
        &input.task,
        &outputs,
        &request.request_id,
        input.inference,
        crate::edit::supports,
        crate::pql::float_text,
    )?;
    for warning in notices {
        asker.warn(warning);
    }
    let bytes = policy::call_bytes(&payload)?;
    let answer = crate::predict::call(
        &asker,
        &payload,
        bytes,
        request.model_endpoint.as_ref(),
        true,
    )
    .await?;
    if named {
        let absent = policy::absent(&answer, &outputs);
        if !absent.is_empty() {
            let notice = policy::absent_notice(&model, &absent);
            asker
                .warn(CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields));
        }
    }
    crate::predict::finish_direct(&asker, answer, outputs, None)
}
