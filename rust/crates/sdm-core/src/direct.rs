// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composition of direct constructor data, tabular calls and attributed replies.
//! Native conversion surrounds these useful data values; no operation asks for
//! a language callback or carries an opaque continuation.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions};
use sdm_contract::direct::*;
use sdm_contract::ops::{HeldSchema, OperationResult, Predict};
use sdm_contract::{CoreError, CoreWarning, Detail, Sentence};
use sdm_request::direct::{invalid, texts};
use sdm_request::{predict, tabular};
use sdm_wire::json::Value;
use sdm_wire::response::PredictionResponse;

use crate::attached::take_json;
use crate::run::{ArrowData, Asker, Attachment, Attachments, internal};

pub(crate) fn value<T: serde::Serialize>(
    asker: &Asker,
    value: &T,
) -> Result<OperationResult, CoreError> {
    let bytes =
        serde_json::to_vec(value).map_err(|_| internal("a direct value cannot serialize"))?;
    Ok(OperationResult::DirectValue(DirectValue {
        value: asker.attach(Attachment::Bytes(bytes))?,
    }))
}

pub(crate) fn encode_column(
    asker: &Asker,
    request: DirectColumnInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectColumnEncoding = take_json(&mut attached, request.input)?;
    let column = rows(&mut attached, request.values, &input.column)?;
    let values = tabular::encode_column(&column.batches, input.dtype, crate::pql::float_text)?;
    value(asker, &EncodedDirectColumn { values })
}

pub(crate) fn infer_column(
    asker: &Asker,
    request: DirectColumnInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectHeader = take_json(&mut attached, request.input)?;
    if input.columns.len() != 1 {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct dtype observation requires exactly one column",
        ));
    }
    let column = rows(&mut attached, request.values, &input)?;
    let dtype = sdm_schema::direct::storage_dtypes(&column.schema, &column.batches)?[0];
    value(asker, &sdm_schema::direct::dtype::declared(dtype, false))
}

pub(crate) fn infer_time_start(
    asker: &Asker,
    request: DirectColumnInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectHeader = take_json(&mut attached, request.input)?;
    if input.columns.len() != 1 {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct time observation requires exactly one column",
        ));
    }
    let column = rows(&mut attached, request.values, &input)?;
    value(
        asker,
        &sdm_schema::direct::time_start(&column.schema, &column.batches)?,
    )
}

fn rows(
    attached: &mut Attachments,
    id: sdm_contract::data::AttachmentId,
    header: &DirectHeader,
) -> Result<ArrowData, CoreError> {
    let Some(Attachment::Arrow(mut data)) = attached.remove(&id) else {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "a direct frame does not name Arrow",
        ));
    };
    if data
        .schema
        .fields()
        .iter()
        .map(|field| field.name())
        .ne(header.columns.iter())
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct headers do not match admitted Arrow columns",
        ));
    }
    for field in data.schema.fields() {
        storage::read(field)?;
    }
    if data
        .batches
        .iter()
        .any(|batch| batch.schema() != data.schema)
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct batch schemas disagree with their attachment storage facts",
        ));
    }
    if data.schema.fields().is_empty() {
        if data
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>()
            != 0
            && data
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
                != header.rows
        {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "zero-column Arrow rows disagree with the explicit count",
            ));
        }
        data.batches = vec![
            RecordBatch::try_new_with_options(
                Arc::clone(&data.schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(header.rows)),
            )
            .map_err(|_| internal("zero-column direct rows cannot be represented"))?,
        ];
    }
    if data
        .batches
        .iter()
        .map(RecordBatch::num_rows)
        .sum::<usize>()
        != header.rows
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct header row count disagrees with Arrow",
        ));
    }
    Ok(data)
}

fn schema(held: &HeldSchema) -> Result<sdm_schema::held::Read, CoreError> {
    let mut read = sdm_schema::held::read(held.clone(), &BTreeMap::new(), true)?;
    // Preserve the concrete wire width beside the generic language family.
    for table in &mut read.description.tables {
        if let Some(native) = held.tables.iter().find(|native| native.name == table.name) {
            for column in &mut table.columns {
                if let Some(native) = native
                    .columns
                    .iter()
                    .find(|native| native.name == column.name)
                    && let Ok(dtype) =
                        serde_json::from_value(serde_json::Value::String(native.dtype.clone()))
                {
                    column.dtype = dtype;
                }
            }
        }
    }
    Ok(read)
}

pub(crate) fn prepare(
    asker: &Asker,
    request: DirectFrames,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectPreparation = take_json(&mut attached, request.input)?;
    sdm_request::direct::shape(&input.shape)?;
    let context = rows(&mut attached, request.context, &input.shape.context)?;
    let scoring = rows(&mut attached, request.predict, &input.shape.predict)?;
    let target = &input.shape.target;
    let target_at = context
        .schema
        .index_of(target)
        .map_err(|_| internal("checked target is absent"))?;
    let known: usize = context
        .batches
        .iter()
        .map(|batch| batch.column(target_at).len() - batch.column(target_at).null_count())
        .sum();
    sdm_request::direct::population(&input.shape, known)?;
    let held = if let Some(schema) = input.schema {
        schema
    } else {
        sdm_request::direct::column_names(&input.shape.context.columns, &input.shape.entity_table)?;
        let (held, warnings) = sdm_schema::direct::infer(
            &input.shape.entity_table,
            &context.schema,
            &context.batches,
        )?;
        for warning in warnings {
            asker.warn(warning);
        }
        held
    };
    let read = schema(&held)?;
    let table = read
        .description
        .tables
        .iter()
        .find(|table| table.name == input.shape.entity_table)
        .ok_or_else(|| {
            invalid(Sentence::NoTable)
                .with_field("table", Detail::text(&input.shape.entity_table))
                .with_field(
                    "tables",
                    texts(
                        read.description
                            .tables
                            .iter()
                            .map(|table| table.name.clone()),
                    ),
                )
        })?;
    let arriving = sdm_schema::direct::storage_dtypes(&scoring.schema, &scoring.batches)?;
    sdm_request::direct::scored(
        &input.shape,
        table,
        &input.flags,
        &arriving,
        crate::edit::supports,
    )?;
    let task = input
        .shape
        .task
        .clone()
        .ok_or_else(|| internal("a checked direct shape has no task"))?;
    if sdm_schema::direct::distinct(&context.batches, target_at)? == 1
        && (task.contains("classification") || task == "regression")
    {
        asker.warn(
            CoreWarning::new(
                if task == "regression" {
                    Sentence::DirectConstantRegression
                } else {
                    Sentence::DirectConstantClass
                },
                "every context row has the same target",
            )
            .with_field("target", Detail::text(target)),
        );
    }
    let native_table = held
        .tables
        .iter()
        .find(|table| table.name == input.shape.entity_table)
        .ok_or_else(|| internal("held table vanished"))?;
    let column = native_table
        .columns
        .iter()
        .find(|column| &column.name == target)
        .ok_or_else(|| {
            invalid(Sentence::NoColumn)
                .with_field("table", Detail::text(&input.shape.entity_table))
                .with_field("column", Detail::text(target))
                .with_field(
                    "columns",
                    texts(
                        native_table
                            .columns
                            .iter()
                            .map(|column| column.name.clone()),
                    ),
                )
        })?;
    let description = DirectDescription {
        schema: held.clone(),
        task,
        target: target.clone(),
        entity_table: input.shape.entity_table,
        target_dtype: column.dtype.clone(),
    };
    value(asker, &description)
}

fn stype(value: sdm_contract::ops::Stype) -> String {
    sdm_request::direct::stype_name(value).to_owned()
}

pub(crate) fn check_preparation(
    asker: &Asker,
    input: sdm_contract::direct::schema::PreparationObservation,
) -> Result<OperationResult, CoreError> {
    use sdm_contract::direct::schema::PreparationObservation as P;
    match input {
        P::TabularMeaning {
            inferred,
            different,
            classification,
        } => {
            return value(
                asker,
                &sdm_contract::direct::schema::MeaningResult {
                    stype: Some(tabular::observed_meaning(
                        inferred,
                        different,
                        classification,
                    )?),
                },
            );
        }
        P::CoherentMeaning {
            inferred,
            dtype,
            declared,
        } => {
            return value(
                asker,
                &sdm_contract::direct::schema::MeaningResult {
                    stype: tabular::coherent_meaning(
                        inferred,
                        dtype,
                        declared,
                        crate::edit::supports,
                    ),
                },
            );
        }
        P::ConstancyPopulation { empty, diverse } => {
            let reached = match (empty, diverse) {
                (Some(empty), None) => !empty,
                (None, Some(diverse)) => !diverse,
                _ => {
                    return Err(CoreError::protocol(
                        Sentence::InvalidMessage,
                        "one reached constancy observation is required",
                    ));
                }
            };
            return value(asker, &reached);
        }
        P::ConstancyKind {
            classification,
            regression,
        } => {
            if classification == regression.is_some() {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "regression comparison must follow rejected classification membership",
                ));
            }
            return value(asker, &(classification || regression.unwrap_or(false)));
        }
        P::ConstancyNotice { classification } => {
            asker.warn(CoreWarning::new(
                if classification {
                    Sentence::DirectNativeConstantClass
                } else {
                    Sentence::DirectNativeConstantRegression
                },
                "every context row has the same target",
            ));
        }
        P::Admission { observation } => sdm_request::direct::admission(observation)?,
        P::Arguments { shape } => {
            sdm_request::direct::arguments(&shape)?;
        }
        P::Consistent { schema } => {
            consistent(&schema)?;
        }
        P::Declarations { columns } => {
            let dtypes = columns
                .into_iter()
                .map(|(name, dtype, untyped)| {
                    let parsed = serde_json::from_value(serde_json::Value::String(dtype.clone()))
                        .unwrap_or_else(|_| sdm_schema::held::wire_dtype(&dtype));
                    (name, (!untyped).then_some(parsed))
                })
                .collect();
            return value(asker, &DirectDtypeDeclarations { dtypes });
        }
        P::Population { shape, known } => sdm_request::direct::population(&shape, known)?,
        P::MissingKey {
            table,
            column,
            count,
        } => {
            if count > 0 {
                asker.warn(
                    CoreWarning::new(
                        Sentence::DirectNativeMissingKey,
                        "an inferred key has missing values",
                    )
                    .with_field("table", Detail::text(table))
                    .with_field("column", Detail::text(column))
                    .with_field(
                        "count",
                        Detail::Scalar(sdm_contract::Scalar::UInt64(count as u64)),
                    ),
                );
            }
        }
        P::Names { table, columns } => sdm_request::direct::column_names(&columns, &table)?,
        P::ScoredColumn {
            table,
            schema: held,
            flags,
            name,
            finding,
        } => {
            let read = schema(&held)?;
            if let Some(table) = read.description.tables.iter().find(|t| t.name == table) {
                sdm_request::direct::scored_column(
                    table,
                    &flags,
                    &name,
                    (!finding.untyped).then_some(finding.dtype),
                    crate::edit::supports,
                )?;
            }
        }
        P::ScoredComplete {
            shape,
            schema: held,
        } => {
            let read = schema(&held)?;
            let table = read
                .description
                .tables
                .iter()
                .find(|t| t.name == shape.entity_table)
                .ok_or_else(|| {
                    invalid(Sentence::NoTable)
                        .with_field("table", Detail::text(&shape.entity_table))
                })?;
            sdm_request::direct::scored_complete(&shape, table)?;
        }
        P::Constancy {
            target,
            task,
            known,
            distinct,
        } => {
            if known > 0
                && distinct == 1
                && (task.contains("classification") || task == "regression")
            {
                asker.warn(
                    CoreWarning::new(
                        if task == "regression" {
                            Sentence::DirectConstantRegression
                        } else {
                            Sentence::DirectConstantClass
                        },
                        "every context row has the same target",
                    )
                    .with_field("target", Detail::text(target)),
                );
            }
        }
    }
    value(asker, &true)
}

pub(crate) fn inspect_pair(
    asker: &Asker,
    request: DirectFrames,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: TabularInput = take_json(&mut attached, request.input)?;
    let context = rows(&mut attached, request.context, &input.context)?;
    let scoring = rows(&mut attached, request.predict, &input.predict)?;
    if let (Ok(read), Ok(arriving)) = (
        schema(&input.schema),
        sdm_schema::direct::storage_dtypes(&scoring.schema, &scoring.batches),
    ) && let Some(table) = read
        .description
        .tables
        .iter()
        .find(|table| table.name == input.entity_table)
    {
        let arriving: Vec<_> = input
            .predict
            .columns
            .iter()
            .cloned()
            .zip(arriving)
            .collect();
        if let Some(warning) = tabular::overlap(
            table,
            &input.flags,
            &input.target,
            &context.batches,
            &scoring.batches,
            &arriving,
            crate::pql::float_text,
        ) {
            asker.warn(warning);
        }
    }
    value(asker, &true)
}

fn consistent(held: &HeldSchema) -> Result<sdm_schema::held::Read, CoreError> {
    let read = schema(held)?;
    for table in &read.description.tables {
        for column in &table.columns {
            if !crate::edit::supports(column.stype, column.dtype) {
                return Err(CoreError::new(
                    Sentence::MeaningNotCarried,
                    "the schema's meaning contradicts its storage",
                )
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name))
                .with_field("dtype", Detail::text(column.dtype.wire_name()))
                .with_field("stype", Detail::text(stype(column.stype))));
            }
        }
    }
    sdm_schema::assemble::check_links(&read.description, &read.types)?;
    Ok(read)
}

pub(crate) fn resolve_observed(
    asker: &Asker,
    observed: ObservedTabularInput,
) -> Result<OperationResult, CoreError> {
    let input = observed;
    sdm_request::direct::bound(&input.bound)?;
    if input.bound.model != predict::TABULAR {
        return Err(invalid(Sentence::DirectRelationalFrames)
            .with_field("model", Detail::text(&input.bound.model)));
    }
    let columns = TabularColumns {
        dtypes: input.dtypes,
        columns: input.columns,
        target: input.target,
    };
    if !input.text_columns.is_empty() {
        asker.warn(
            CoreWarning::new(
                Sentence::DirectTextCategorical,
                "tabular text is categorical",
            )
            .with_field("columns", texts(input.text_columns)),
        );
    }
    value(
        asker,
        &ResolvedTabular {
            bound: input.bound,
            columns,
        },
    )
}

pub(crate) fn inspect_observed_keys(
    asker: &Asker,
    request: DirectFrames,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: ObservedKeyPair = take_json(&mut attached, request.input)?;
    let context = rows(&mut attached, request.context, &input.context)?;
    let scoring = rows(&mut attached, request.predict, &input.predict)?;
    if input.context.columns != input.predict.columns {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "observed key components differ between the two frames",
        ));
    }
    if let Some(warning) = tabular::observed_overlap(
        &context.batches,
        &scoring.batches,
        &input.target,
        input.scoring_rows,
    )? {
        asker.warn(warning);
    }
    value(asker, &true)
}

pub(crate) fn resolve(
    asker: &Asker,
    request: DirectFrames,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: TabularInput = take_json(&mut attached, request.input)?;
    sdm_request::direct::bound(&input.bound)?;
    if input.bound.model != predict::TABULAR {
        return Err(invalid(Sentence::DirectRelationalFrames)
            .with_field("model", Detail::text(&input.bound.model)));
    }
    let _context = rows(&mut attached, request.context, &input.context)?;
    let scoring = rows(&mut attached, request.predict, &input.predict)?;
    let read = consistent(&input.schema)?;
    let arriving: Vec<_> = input
        .predict
        .columns
        .iter()
        .cloned()
        .zip(sdm_schema::direct::storage_dtypes(
            &scoring.schema,
            &scoring.batches,
        )?)
        .collect();
    let (columns, warnings) = tabular::envelope(
        &read.description,
        &input.flags,
        &input.entity_table,
        &input.context.columns,
        &scoring.batches,
        &arriving,
        &scoring
            .schema
            .fields()
            .iter()
            .map(|field| storage::floats(field))
            .collect::<Result<Vec<_>, _>>()?,
        &input.target,
        &input.kind,
        crate::edit::supports,
    )?;
    for warning in warnings {
        asker.warn(warning);
    }
    value(
        asker,
        &ResolvedTabular {
            bound: input.bound,
            columns,
        },
    )
}

pub(crate) fn encode(
    asker: &Asker,
    request: DirectFrames,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: TabularEncoding = take_json(&mut attached, request.input)?;
    let ResolvedTabular { bound, columns } = input.resolved;
    sdm_request::direct::bound(&bound)?;
    if bound.model != predict::TABULAR {
        return Err(invalid(Sentence::DirectRelationalFrames)
            .with_field("model", Detail::text(&bound.model)));
    }
    let context = rows(&mut attached, request.context, &input.context)?;
    let scoring = rows(&mut attached, request.predict, &input.predict)?;
    let payload = tabular::body(
        &columns,
        &context.batches,
        &scoring.batches,
        &input.primary_key,
        &input.kind,
        &bound.outputs,
        crate::pql::float_text,
    )?;
    let payload = asker.attach(Attachment::Bytes(payload.to_bytes()))?;
    value(
        asker,
        &EncodedDirectBody {
            bound,
            columns,
            kind: input.kind,
            primary_key: input.primary_key,
            context_shape: EncodedTableShape {
                columns: input.context.columns,
                rows: input.context.rows,
            },
            predict_shape: EncodedTableShape {
                columns: input.predict.columns,
                rows: input.predict.rows,
            },
            payload,
        },
    )
}

/// Decode all model fields before the first row's native ID conversion.
fn decoded(
    payload: &Value,
    answer: &Value,
) -> Result<(PredictionResponse, ResponseScope, Vec<DecodedDirectRow>), CoreError> {
    let response = PredictionResponse::read(answer).map_err(|refusal| {
        if refusal.field == "prediction.row_index" && refusal.reason == "missing" {
            CoreError::new(
                Sentence::TabularMissingRowIndex,
                "the response is missing a required row index",
            )
        } else {
            predict::refused(refusal)
        }
    })?;
    let (scope, rows) = tabular::response::decode(payload, &response, crate::pql::float_text)?;
    Ok((response, scope, rows))
}

pub(crate) async fn predict(
    asker: Asker,
    request: Predict,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    if request.state.is_some() || request.held.is_some() {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "a direct prediction also names gathered state",
        ));
    }
    let direct = request
        .direct
        .ok_or_else(|| internal("direct input is absent"))?;
    let input: DirectPrediction = take_json(&mut attached, direct.input)?;
    let (bound, payload, deferred_options) = match input {
        DirectPrediction::Empty {
            bound,
            target_dtype,
        } => {
            sdm_request::direct::bound(&bound)?;
            if request.model != bound.model
                || request.outputs != bound.outputs
                || request.default_outputs == bound.named
            {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "empty direct call selection differs from its bound outputs",
                ));
            }
            if direct.payload.is_some() {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "an empty direct call names a body",
                ));
            }
            let empty = Value::Object(vec![(
                "task".into(),
                Value::Object(vec![
                    ("kind".into(), Value::Text(bound.kind.clone())),
                    (
                        "target".into(),
                        Value::Object(vec![("dtype".into(), Value::Text(target_dtype))]),
                    ),
                ]),
            )]);
            return crate::predict::finish(
                &asker,
                sdm_request::tabular::response::empty(&bound.model, &empty)?,
                bound.outputs,
            );
        }
        DirectPrediction::DeferredOptions {
            body,
            inference_present,
        } => {
            sdm_request::direct::bound(&body.bound)?;
            if direct.payload != Some(body.payload) {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "deferred body differs from its attachment",
                ));
            }
            let Some(Attachment::Bytes(bytes)) = attached.remove(&body.payload) else {
                return Err(internal("a deferred direct body is absent"));
            };
            let payload = crate::predict::payload(&bytes)?;
            tabular::check_body(&payload, &body, &body.bound.outputs)?;
            let mut payload =
                tabular::complete_body(payload, &request.request_id, DirectInference::default())?;
            if inference_present {
                payload.set("inference", Value::Null);
            }
            (body.bound, payload, true)
        }
        DirectPrediction::Encoded { body, inference } => {
            sdm_request::direct::bound(&body.bound)?;
            if direct.payload != Some(body.payload) {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "direct body attachment differs from encoded data",
                ));
            }
            let Some(Attachment::Bytes(bytes)) = attached.remove(&body.payload) else {
                return Err(internal("a direct body is absent"));
            };
            let payload = crate::predict::payload(&bytes)?;
            tabular::check_body(&payload, &body, &body.bound.outputs)?;
            let payload = tabular::complete_body(payload, &request.request_id, inference)?;
            (body.bound, payload, false)
        }
    };
    if request.model != bound.model
        || request.outputs != bound.outputs
        || request.default_outputs == bound.named
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct call selection differs from its bound outputs",
        ));
    }
    if deferred_options {
        use sdm_contract::direct::native_response::{NativeDirectCall, PreparedDirectCall};
        let payload = asker.attach(Attachment::Bytes(payload.to_bytes()))?;
        let call = NativeDirectCall { bound, payload };
        let value = asker.attach(Attachment::Bytes(
            serde_json::to_vec(&call).map_err(|_| internal("prepared call does not serialize"))?,
        ))?;
        return Ok(OperationResult::PreparedDirectCall(PreparedDirectCall {
            value,
        }));
    }
    let answer =
        crate::predict::call_answer(&asker, payload.to_bytes(), request.model_endpoint.as_ref())
            .await?;
    decoded_result(&asker, bound, payload, answer)
}

fn decoded_result(
    asker: &Asker,
    bound: BoundDirectCall,
    payload: Value,
    answer: Value,
) -> Result<OperationResult, CoreError> {
    let (_response, scope, rows) = decoded(&payload, &answer)?;
    let payload = asker.attach(Attachment::Bytes(payload.to_bytes()))?;
    let response = asker.attach(Attachment::Bytes(answer.to_bytes()))?;
    let rows = asker.attach(Attachment::Bytes(
        serde_json::to_vec(&rows).map_err(|_| internal("decoded rows do not serialize"))?,
    ))?;
    let data = DirectResponseData {
        scope,
        outputs: bound.outputs.clone(),
        named: bound.named,
        bound,
        payload,
        response,
        rows,
    };
    let value = asker.attach(Attachment::Bytes(
        serde_json::to_vec(&data).map_err(|_| internal("decoded response does not serialize"))?,
    ))?;
    Ok(OperationResult::DecodedDirectResponse(
        DecodedDirectResponse { value },
    ))
}

pub(crate) fn decode_response(
    asker: &Asker,
    request: DecodeDirectResponseInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectDecodeInput = take_json(&mut attached, request.input)?;
    sdm_request::direct::bound(&input.bound)?;
    if input.payload != request.payload || input.response != request.response {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "reply decoder attachment bindings differ",
        ));
    }
    let bytes = |attached: &mut Attachments, id| match attached.remove(&id) {
        Some(Attachment::Bytes(bytes)) => Ok(bytes),
        _ => Err(internal("response bytes are absent")),
    };
    let payload = crate::predict::payload(&bytes(&mut attached, request.payload)?)?;
    let answer = sdm_wire::json::read_answer(&bytes(&mut attached, request.response)?)
        .map_err(|_| internal("response bytes are not readable"))?;
    let (response, _, _) = decoded(&payload, &answer)?;
    if !tabular::response::native_reply_representable(&payload, &response) {
        return Ok(OperationResult::NativeDirectResponse);
    }
    decoded_result(asker, input.bound, payload, answer)
}

pub(crate) fn layout(
    asker: &Asker,
    request: LayoutDirect,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: DirectLayoutInput = take_json(&mut attached, request.input)?;
    sdm_request::direct::bound(&input.data.bound)?;
    if input.data.outputs != input.data.bound.outputs || input.data.named != input.data.bound.named
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "response outputs differ from the original bound selection",
        ));
    }
    if input.data.payload != request.payload || input.data.response != request.response {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "response layout attachment bindings differ",
        ));
    }
    let bytes = |attached: &mut Attachments, id| match attached.remove(&id) {
        Some(Attachment::Bytes(bytes)) => Ok(bytes),
        _ => Err(internal("response bytes are absent")),
    };
    let payload = crate::predict::payload(&bytes(&mut attached, request.payload)?)?;
    let answer = sdm_wire::json::read_answer(&bytes(&mut attached, request.response)?)
        .map_err(|_| internal("response bytes are no longer readable"))?;
    let (response, scope, rows) = decoded(&payload, &answer)?;
    if scope != input.data.scope || input.attributed.iter().map(|row| &row.row).ne(rows.iter()) {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "attribution does not belong to this response",
        ));
    }
    let part = tabular::response::layout(
        &payload,
        &response,
        &scope,
        input.attributed,
        0,
        crate::pql::float_text,
    )?;
    if input.data.named {
        let missing = predict::absent(&part, &input.data.outputs);
        if !missing.is_empty() {
            let notice = predict::absent_notice(&input.data.bound.model, &missing);
            asker
                .warn(CoreWarning::new(notice.sentence, notice.message).with_fields(notice.fields));
        }
    }
    crate::predict::finish(asker, part, input.data.outputs)
}

pub(crate) fn native_outputs(
    asker: &Asker,
    input: sdm_contract::direct::native_response::NativeOutputQuery,
) -> Result<OperationResult, CoreError> {
    sdm_request::direct::bound(&input.bound)?;
    let names: Vec<_> = input
        .bound
        .outputs
        .iter()
        .filter(|name| {
            input.rows > 0 && input.bound.named && predict::LOOKED_FOR.contains(&name.as_str())
        })
        .cloned()
        .collect();
    if let Some(present) = input.present {
        if present.iter().map(|(name, _)| name).ne(names.iter()) {
            return Err(CoreError::protocol(
                Sentence::InvalidMessage,
                "native output observations differ from requested fields",
            ));
        }
        let missing: Vec<_> = present
            .into_iter()
            .filter_map(|(name, present)| (!present).then_some(name))
            .collect();
        if !missing.is_empty() {
            let notice = predict::absent_notice(&input.bound.model, &missing);
            asker.warn(
                CoreWarning::new(Sentence::DirectOutputsAbsent, notice.message)
                    .with_fields(notice.fields),
            );
        }
    }
    value(asker, &names)
}

/// Encode only the supplied table, reusing the one-shot column/cell encoder.
pub(crate) fn encode_table(
    asker: &Asker,
    request: DirectColumnInput,
    mut attached: Attachments,
) -> Result<OperationResult, CoreError> {
    let input: TabularTableEncoding = take_json(&mut attached, request.input)?;
    let table = rows(&mut attached, request.values, &input.header)?;
    let encoded = tabular::encode_table(&input.columns, &table.batches, crate::pql::float_text)?;
    let table = asker.attach(Attachment::Bytes(encoded.to_bytes()))?;
    value(
        asker,
        &EncodedTabularTable {
            table,
            shape: EncodedTableShape {
                columns: input.header.columns,
                rows: input.header.rows,
            },
        },
    )
}

/// Assemble source-less schemas using the existing schema policies.
pub(crate) fn assemble_frame_schema(
    asker: &Asker,
    input: sdm_contract::frames::AssembleFrameSchema,
) -> Result<OperationResult, CoreError> {
    let read = sdm_schema::held::read(input.schema, &BTreeMap::new(), false)?;
    let choices = input
        .times
        .into_iter()
        .map(|facts| {
            let (choice, pending) = sdm_schema::direct::observed::time_choice(facts)?;
            if !pending.is_empty() {
                return Err(internal(
                    "frame time observation requires completed earliest-time facts",
                ));
            }
            Ok(choice)
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    let mut assembled = sdm_schema::assemble::assemble_frames(
        read.description,
        choices,
        &input.primary_keys,
        &input.time_columns,
        input.relationships.as_deref(),
    )?;
    for name in input.primary_keys.keys() {
        let table = assembled
            .description
            .tables
            .iter_mut()
            .find(|t| &t.name == name)
            .ok_or_else(|| internal("frame key override names an absent table"))?;
        let count = input
            .key_counts
            .get(name)
            .ok_or_else(|| internal("frame key override lacks whole-table counts"))?;
        if count.missing > count.rows || count.distinct > count.rows - count.missing {
            return Err(internal("frame key counts exceed their population"));
        }
        if count.missing > 0 && table.primary_key.len() > 1 {
            if count.missing_columns.is_empty()
                || count
                    .missing_columns
                    .iter()
                    .any(|c| !table.primary_key.contains(c))
            {
                return Err(internal(
                    "frame key missing-column observations disagree with the key",
                ));
            }
            return Err(CoreError::new(
                Sentence::FrameKeyMissingParts,
                "frame key contains missing parts",
            )
            .with_field("table", Detail::text(name))
            .with_field(
                "columns",
                Detail::List(
                    table
                        .primary_key
                        .iter()
                        .cloned()
                        .map(sdm_contract::Scalar::Utf8)
                        .collect(),
                ),
            )
            .with_field(
                "parts",
                Detail::List(
                    count
                        .missing_columns
                        .iter()
                        .cloned()
                        .map(sdm_contract::Scalar::Utf8)
                        .collect(),
                ),
            )
            .with_field(
                "missing",
                Detail::Scalar(sdm_contract::Scalar::UInt64(count.missing)),
            ));
        }
        let inferred = sdm_schema::infer::InferredTable {
            name: name.clone(),
            source: table.source.clone(),
            columns: vec![],
            left_out: vec![],
            unread_times: vec![],
            day_first: vec![],
            tied_keys: vec![],
            primary_key: table.primary_key.clone(),
            time: Default::default(),
            time_column: table.time_column.clone(),
            unique: vec![],
            declared: Default::default(),
        };
        let checked = sdm_schema::keys::hold(
            &mut [inferred],
            &[(
                0,
                sdm_schema::keys::Counted {
                    rows: count.rows,
                    distinct: count.distinct,
                    missing: count.missing,
                    example: count.example.clone(),
                },
            )],
            &std::collections::BTreeSet::from([name.clone()]),
            false,
        )?;
        table.duplicate_keys = Some(sdm_contract::Scalar::UInt64(
            count.rows - count.missing - count.distinct,
        ));
        for warning in checked.warnings {
            asker.warn(warning);
        }
    }
    for warning in assembled.warnings {
        asker.warn(warning);
    }
    let held = sdm_schema::held::held(&assembled.description, &read.unfacted, &read.retyped);
    value(asker, &held)
}
