// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow-native host composition. No pandas observations or language callbacks.

use crate::attached::take_json;
use crate::run::{ArrowData, Asker, Attachment, Attachments, internal};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use sdm_contract::direct::storage::{
    DIRECT_STORAGE_KEY, DirectStorage, StorageOwner, TimestampUnit,
};
use sdm_contract::direct::*;
use sdm_contract::ops::{HeldSchema, OperationResult};
use sdm_contract::tabular_arrow::*;
use sdm_contract::{CoreError, Detail, HostFacts, Sentence};
use sdm_engine::time::Zones;
use sdm_request::{direct, tabular};
use sdm_wire::json::Value;
use std::sync::Arc;

fn invalid(message: &str) -> CoreError {
    CoreError::new(Sentence::ArrowTabularInput, message).with_field("reason", Detail::text(message))
}

/// Normalize physical Arrow types into the shared typed-column representation.
/// All metadata is derived here from Arrow itself, never a claimed Python origin.
fn normalize(data: ArrowData) -> Result<ArrowData, CoreError> {
    let mut fields = Vec::new();
    let mut types = Vec::new();
    for field in data.schema.fields() {
        let mut dtype = field.data_type().clone();
        if let DataType::Dictionary(_, value) = dtype {
            dtype = *value;
        }
        let storage = match &dtype {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Null => {
                dtype = DataType::Utf8;
                DirectStorage::String
            }
            DataType::Timestamp(unit, zone) => {
                let unit = match unit {
                    TimeUnit::Second => TimestampUnit::Seconds,
                    TimeUnit::Millisecond => TimestampUnit::Milliseconds,
                    TimeUnit::Microsecond => TimestampUnit::Microseconds,
                    TimeUnit::Nanosecond => TimestampUnit::Nanoseconds,
                };
                let zone = zone.as_ref().map(ToString::to_string);
                dtype = DataType::Int64;
                DirectStorage::Timestamp { unit, zone }
            }
            DataType::Decimal128(precision, scale) | DataType::Decimal256(precision, scale) => {
                DirectStorage::Decimal {
                    precision: *precision,
                    scale: *scale,
                }
            }
            DataType::Float32 | DataType::Float64 => DirectStorage::Primitive {
                owner: StorageOwner::Arrow,
                floats: Some(FloatStorage::ValidityMasked),
            },
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => DirectStorage::Primitive {
                owner: StorageOwner::Arrow,
                floats: None,
            },
            _ => return Err(invalid("unsupported Arrow column type")),
        };
        fields.push(
            Field::new(field.name(), dtype.clone(), true).with_metadata(
                [(
                    DIRECT_STORAGE_KEY.into(),
                    serde_json::to_string(&storage)
                        .map_err(|_| internal("storage serialization"))?,
                )]
                .into(),
            ),
        );
        types.push(dtype);
    }
    let schema = Arc::new(Schema::new(fields));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(&types)
                .map(|(column, dtype)| {
                    if column.data_type() == dtype {
                        Ok(Arc::clone(column))
                    } else {
                        arrow_cast::cast(column, dtype)
                            .map_err(|_| invalid("Arrow column cannot be normalized"))
                    }
                })
                .collect::<Result<Vec<ArrayRef>, CoreError>>()?;
            RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| invalid("invalid Arrow batch"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowData::new(schema, batches))
}
fn take_table(
    attached: &mut Attachments,
    id: sdm_contract::data::AttachmentId,
) -> Result<ArrowData, CoreError> {
    match attached.remove(&id) {
        Some(Attachment::Arrow(data)) => normalize(data),
        _ => Err(internal("missing Arrow input")),
    }
}
fn header(data: &ArrowData) -> DirectHeader {
    DirectHeader {
        rows: data.batches.iter().map(RecordBatch::num_rows).sum(),
        columns: data
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect(),
    }
}
fn flags(schema: &HeldSchema, name: &str) -> Vec<ColumnFlags> {
    schema
        .tables
        .iter()
        .find(|t| t.name == name)
        .into_iter()
        .flat_map(|t| &t.columns)
        .map(|c| ColumnFlags {
            name: c.name.clone(),
            untyped: false,
            stype_declared: true,
        })
        .collect()
}
fn describe(asker: &Asker, table: &ArrowData, name: &str) -> Result<HeldSchema, CoreError> {
    let (schema, warnings) = sdm_schema::direct::infer(name, &table.schema, &table.batches)?;
    for warning in warnings {
        asker.warn(warning);
    }
    Ok(schema)
}
fn prepare(
    asker: &Asker,
    request: &ArrowRequest,
    context: &ArrowData,
    scoring: &ArrowData,
) -> Result<DirectDescription, CoreError> {
    let shape = DirectShape {
        target_containers: false,
        context_kind: FrameKind::Frame,
        predict_kind: FrameKind::Frame,
        related: vec![],
        task: Some(request.task.clone()),
        target: request.target.clone(),
        entity_table: request.entity_table.clone(),
        context: header(context),
        predict: header(scoring),
    };
    direct::shape(&shape)?;
    let at = context
        .schema
        .index_of(&request.target)
        .map_err(|_| internal("target absent"))?;
    let known = context
        .batches
        .iter()
        .map(|b| b.num_rows() - b.column(at).null_count())
        .sum();
    direct::population(&shape, known)?;
    let held = match &request.schema {
        Some(s) => s.clone(),
        None => describe(asker, context, &request.entity_table)?,
    };
    let read = crate::direct::schema(&held)?;
    let table = read
        .description
        .tables
        .iter()
        .find(|t| t.name == request.entity_table)
        .ok_or_else(|| invalid("entity table absent"))?;
    let arriving = sdm_schema::direct::storage_dtypes(&scoring.schema, &scoring.batches)?;
    direct::scored(
        &shape,
        table,
        &flags(&held, &request.entity_table),
        &arriving,
        crate::edit::supports,
    )?;
    let target_dtype = held
        .tables
        .iter()
        .find(|t| t.name == request.entity_table)
        .and_then(|t| t.columns.iter().find(|c| c.name == request.target))
        .ok_or_else(|| invalid("target absent from schema"))?
        .dtype
        .clone();
    Ok(DirectDescription {
        schema: held,
        task: request.task.clone(),
        target: request.target.clone(),
        entity_table: request.entity_table.clone(),
        target_dtype,
    })
}
struct Prepared {
    bound: BoundDirectCall,
    columns: TabularColumns,
    keys: Vec<String>,
}
fn bound(
    asker: &Asker,
    input: &ArrowPrediction,
    context: &ArrowData,
    scoring: &ArrowData,
) -> Result<Prepared, CoreError> {
    let description = prepare(asker, &input.request, context, scoring)?;
    let bound = direct::bind(DirectCallFacts {
        model: input.model.clone(),
        outputs: input.outputs.clone(),
        kind: input.request.task.clone(),
        anchored: false,
        relationships: !description.schema.links.is_empty(),
        tables: description.schema.tables.len(),
        context_rows: header(context).rows,
    })?;
    if bound.model != sdm_request::predict::TABULAR {
        return Err(direct::invalid(Sentence::DirectRelationalFrames)
            .with_field("model", Detail::text(&bound.model)));
    }
    let read = crate::direct::schema(&description.schema)?;
    let arriving = header(scoring)
        .columns
        .into_iter()
        .zip(sdm_schema::direct::storage_dtypes(
            &scoring.schema,
            &scoring.batches,
        )?)
        .collect::<Vec<_>>();
    let floats = scoring
        .schema
        .fields()
        .iter()
        .map(|f| storage::floats(f))
        .collect::<Result<Vec<_>, _>>()?;
    let (columns, warnings) = tabular::envelope(
        &read.description,
        &flags(&description.schema, &input.request.entity_table),
        &input.request.entity_table,
        &header(context).columns,
        &scoring.batches,
        &arriving,
        &floats,
        &input.request.target,
        &input.request.task,
        crate::edit::supports,
    )?;
    for warning in warnings {
        asker.warn(warning);
    }
    let keys = description
        .schema
        .tables
        .iter()
        .find(|t| t.name == input.request.entity_table)
        .ok_or_else(|| invalid("entity table absent"))?
        .primary_key
        .clone();
    Ok(Prepared {
        bound,
        columns,
        keys,
    })
}
fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
async fn predict(
    asker: &Asker,
    input: ArrowPrediction,
    context: ArrowData,
    scoring: ArrowData,
) -> Result<OperationResult, CoreError> {
    let prepared = bound(asker, &input, &context, &scoring)?;
    let payload = tabular::body(
        &prepared.columns,
        &context.batches,
        &scoring.batches,
        &prepared.keys,
        &input.request.task,
        &prepared.bound.outputs,
        crate::pql::float_text,
    )?;
    let payload = tabular::complete_body(payload, &input.request_id, DirectInference::default())?;
    if header(&scoring).rows == 0 {
        return crate::predict::finish(
            asker,
            tabular::response::empty(&input.model, &payload)?,
            prepared.bound.outputs,
        );
    }
    let answer =
        crate::predict::call_answer(asker, payload.to_bytes(), input.endpoint.as_ref()).await?;
    finish(asker, payload, answer, prepared.bound.outputs)
}
fn finish(
    asker: &Asker,
    payload: Value,
    answer: Value,
    outputs: Vec<String>,
) -> Result<OperationResult, CoreError> {
    let response = sdm_wire::response::PredictionResponse::read(&answer)
        .map_err(sdm_request::predict::refused)?;
    let part = tabular::response::portable_layout(&payload, &response, crate::pql::float_text)?;
    crate::predict::finish(asker, part, outputs)
}

pub(crate) async fn run(
    asker: Asker,
    operation: ArrowTabular,
    mut attached: Attachments,
    facts: HostFacts,
    zones: Arc<dyn Zones>,
) -> Result<OperationResult, CoreError> {
    match operation {
        ArrowTabular::Describe { input, table } => {
            let input: ArrowDescription = take_json(&mut attached, input)?;
            let data = take_table(&mut attached, table)?;
            crate::direct::value(&asker, &describe(&asker, &data, &input.entity_table)?)
        }
        ArrowTabular::Prepare {
            input,
            context,
            predict,
        } => {
            let input: ArrowRequest = take_json(&mut attached, input)?;
            let context = take_table(&mut attached, context)?;
            let predict = take_table(&mut attached, predict)?;
            crate::direct::value(&asker, &prepare(&asker, &input, &context, &predict)?)
        }
        ArrowTabular::Predict {
            input,
            context,
            predict: scoring,
        } => {
            let input = take_json(&mut attached, input)?;
            let context = take_table(&mut attached, context)?;
            let scoring = take_table(&mut attached, scoring)?;
            predict(&asker, input, context, scoring).await
        }
        ArrowTabular::Fit {
            input,
            context,
            predict,
        } => {
            let input = take_json(&mut attached, input)?;
            let context = take_table(&mut attached, context)?;
            let scoring = take_table(&mut attached, predict)?;
            fit(&asker, input, context, scoring).await
        }
        ArrowTabular::Score { input, predict } => {
            let input = take_json(&mut attached, input)?;
            let scoring = take_table(&mut attached, predict)?;
            score(&asker, input, scoring).await
        }
        ArrowTabular::Close { input } => close(&asker, take_json(&mut attached, input)?).await,
        ArrowTabular::MaterializePredict {
            input,
            table,
            predict,
        } => {
            let input = take_json(&mut attached, input)?;
            let table = take_table(&mut attached, table)?;
            let predict = take_table(&mut attached, predict)?;
            materialize(&asker, input, table, Some(predict), &facts, zones.as_ref())
        }
        ArrowTabular::Materialize { input, table } => {
            let input = take_json(&mut attached, input)?;
            let table = take_table(&mut attached, table)?;
            materialize(&asker, input, table, None, &facts, zones.as_ref())
        }
    }
}

async fn session_call(
    asker: &Asker,
    endpoint: sdm_contract::run::Endpoint,
    payload: Value,
) -> Result<Value, CoreError> {
    use crate::run::{ModelResult, Reply};
    match asker
        .call_model(endpoint, "application/json".into(), payload.to_bytes())
        .await
    {
        Reply::Model(ModelResult::Response { status, body, .. })
            if (200..300).contains(&status) =>
        {
            if body.is_empty() {
                Ok(object([]))
            } else {
                crate::predict::payload(&body)
            }
        }
        Reply::Model(ModelResult::TransportError { kind, retryable }) => {
            Err(CoreError::transport(kind, retryable))
        }
        Reply::Model(ModelResult::Response { status, .. } | ModelResult::Failed { status, .. }) => {
            Err(
                sdm_request::predict::bad_response("status", "not_a_success").with_field(
                    "status",
                    Detail::Scalar(sdm_contract::Scalar::UInt16(status)),
                ),
            )
        }
        _ => Err(internal("session call received non-model reply")),
    }
}
async fn fit(
    asker: &Asker,
    input: ArrowFit,
    context: ArrowData,
    scoring: ArrowData,
) -> Result<OperationResult, CoreError> {
    use sdm_contract::{run::Endpoint, session::*};
    use tabular::session as policy;
    if input.prediction.endpoint.is_some() {
        return Err(invalid("this transport does not support sessions"));
    }
    let prepared = bound(asker, &input.prediction, &context, &scoring)?;
    let encoded =
        tabular::encode_table(&prepared.columns, &context.batches, crate::pql::float_text)?;
    let keys = policy::encoded_keys(&encoded, &prepared.keys)?;
    let selection = policy::filter_holdout(&keys.rows, &input.holdout.rows)?;
    let keys = SessionKeys {
        rows: selection
            .retained_rows
            .iter()
            .map(|&at| keys.rows[at].clone())
            .collect(),
    };
    let encoded = policy::select_encoded_rows(encoded, &selection.retained_rows)?;
    let mut payload = policy::pinned_metadata(&prepared.bound, &prepared.columns, &prepared.keys)?;
    payload.set("context", object([("instance_table", encoded)]));
    payload.set(
        "metadata",
        object([("request_id", Value::Text(input.prediction.request_id))]),
    );
    let answer = session_call(asker, Endpoint::CreateSession, payload).await?;
    let session_id = answer
        .get("session_id")
        .or_else(|| answer.get("id"))
        .and_then(Value::as_text)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| sdm_request::predict::bad_response("session_id", "missing"))?
        .to_owned();
    let state = TabularSessionState {
        version: TABULAR_SESSION_VERSION,
        session_id,
        closed: false,
        pinned_model: prepared.bound.model.clone(),
        pinned_kind: prepared.bound.kind.clone(),
        bound: prepared.bound,
        columns: prepared.columns,
        primary_key: prepared.keys,
        question: question(&input.prediction.request),
        pinned_keys: sdm_contract::data::AttachmentId(0),
        held_back: selection.removed_rows,
        pinned_over_a_window: false,
    };
    crate::direct::value(asker, &ArrowSession { state, keys })
}
async fn score(
    asker: &Asker,
    input: ArrowScore,
    scoring: ArrowData,
) -> Result<OperationResult, CoreError> {
    use tabular::session as policy;
    let state = &input.session.state;
    policy::validate_state(state)?;
    if state.closed {
        return Err(invalid("session is closed"));
    }
    if input.request.target != state.columns.target || input.request.task != state.pinned_kind {
        return Err(invalid("request does not match fitted target and task"));
    }
    let asked = question(&input.request);
    if policy::check_question(state.question.as_ref(), asked.as_ref()).is_some() {
        return Err(invalid(
            "request question or context does not match fitted session",
        ));
    }
    let expected = state
        .columns
        .columns
        .iter()
        .filter(|c| c.name != state.columns.target)
        .map(|c| c.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let actual = header(&scoring).columns;
    if actual
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>()
        != expected
    {
        return Err(invalid("scoring columns do not match fitted schema"));
    }
    let dtypes = sdm_schema::direct::storage_dtypes(&scoring.schema, &scoring.batches)?;
    for (name, arriving) in actual.iter().zip(dtypes) {
        let pinned = state
            .columns
            .dtypes
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| internal("pinned column absent"))?
            .1;
        if !crate::edit::supports(
            state
                .columns
                .columns
                .iter()
                .find(|c| &c.name == name)
                .ok_or_else(|| internal("pinned column absent"))?
                .stype,
            arriving,
        ) || (pinned != arriving
            && !(matches!(
                pinned,
                sdm_contract::ops::Dtype::Float32 | sdm_contract::ops::Dtype::Float64
            ) && matches!(
                arriving,
                sdm_contract::ops::Dtype::Int64
                    | sdm_contract::ops::Dtype::Float32
                    | sdm_contract::ops::Dtype::Float64
            )))
        {
            return Err(invalid("scoring dtype does not match fitted schema"));
        }
    }
    let encoded = tabular::encode_table(&state.columns, &scoring.batches, crate::pql::float_text)?;
    let keys = policy::encoded_keys(&encoded, &state.primary_key)?;
    let overlap = policy::overlap_count(&input.session.keys.rows, &keys.rows)?;
    if overlap != 0 {
        asker.warn(
            sdm_contract::CoreWarning::new(
                Sentence::DirectOverlap,
                "scoring rows carry their answers in context",
            )
            .with_field(
                "shown",
                Detail::Scalar(sdm_contract::Scalar::UInt64(overlap as u64)),
            )
            .with_field(
                "rows",
                Detail::Scalar(sdm_contract::Scalar::UInt64(keys.rows.len() as u64)),
            )
            .with_field("target", Detail::text(&state.columns.target)),
        );
    }
    let payload = object([
        ("predict", object([("instance_table", encoded)])),
        (
            "output",
            object([(
                "fields",
                Value::Array(
                    state
                        .bound
                        .outputs
                        .iter()
                        .cloned()
                        .map(Value::Text)
                        .collect(),
                ),
            )]),
        ),
        (
            "metadata",
            object([("request_id", Value::Text(input.request_id))]),
        ),
    ]);
    let mut layout = policy::state_metadata(state)?;
    if let Value::Object(fields) = &payload {
        for (name, value) in fields {
            layout.set(name, value.clone());
        }
    }
    if header(&scoring).rows == 0 {
        return crate::predict::finish(
            asker,
            tabular::response::empty(&state.pinned_model, &layout)?,
            state.bound.outputs.clone(),
        );
    }
    let answer = session_call(
        asker,
        sdm_contract::run::Endpoint::SessionPredict {
            session_id: state.session_id.clone(),
        },
        payload,
    )
    .await?;
    finish(asker, layout, answer, state.bound.outputs.clone())
}
async fn close(asker: &Asker, mut session: ArrowSession) -> Result<OperationResult, CoreError> {
    tabular::session::validate_state(&session.state)?;
    if !session.state.closed {
        session_call(
            asker,
            sdm_contract::run::Endpoint::DeleteSession {
                session_id: session.state.session_id.clone(),
            },
            object([]),
        )
        .await?;
        session.state = tabular::session::close_after_success(&session.state, true)?;
        session.keys.rows.clear();
    }
    crate::direct::value(asker, &session)
}

fn select(data: &ArrowData, rows: &[usize], omit: Option<&str>) -> Result<ArrowData, CoreError> {
    let batch = arrow_select::concat::concat_batches(&data.schema, &data.batches)
        .map_err(|_| internal("Arrow batches cannot concatenate"))?;
    let indices =
        arrow_array::UInt64Array::from(rows.iter().map(|&n| n as u64).collect::<Vec<_>>());
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (field, column) in data.schema.fields().iter().zip(batch.columns()) {
        if omit == Some(field.name().as_str()) {
            continue;
        }
        fields.push(field.clone());
        columns.push(
            arrow_select::take::take(column.as_ref(), &indices, None)
                .map_err(|_| internal("row selection failed"))?,
        );
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&schema),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .map_err(|_| internal("selected batch invalid"))?;
    Ok(ArrowData::new(schema, vec![batch]))
}
fn materialize(
    asker: &Asker,
    input: ArrowMaterialize,
    data: ArrowData,
    supplied: Option<ArrowData>,
    facts: &HostFacts,
    zones: &dyn Zones,
) -> Result<OperationResult, CoreError> {
    use sdm_contract::ops::{ProfileName, QueryColumn, QuerySchema, QueryTable};
    use sdm_contract::pql::{NodeKind, ProblemType};
    use sdm_contract::tabular::{DecideTabularTask, TabularTaskDecision};
    if input.schema.tables.len() != 1 || !input.schema.links.is_empty() {
        return Err(invalid(
            "Tabular PQL requires one table without relationships",
        ));
    }
    let read = crate::direct::schema(&input.schema)?;
    let schema = QuerySchema {
        tables: read
            .description
            .tables
            .iter()
            .map(|t| QueryTable {
                name: t.name.clone(),
                columns: t
                    .columns
                    .iter()
                    .map(|c| QueryColumn {
                        name: c.name.clone(),
                        dtype: c.dtype,
                        stype: c.stype,
                    })
                    .collect(),
                primary_key: t.primary_key.first().cloned(),
                time_column: t.time_column.clone(),
                end_time_column: None,
            })
            .collect(),
        links: vec![],
    };
    let validation = crate::pql::resolve(
        &input.query,
        &schema,
        &crate::profiles::Profile::named(ProfileName::Tabular),
        facts,
        zones,
    )?;
    for warning in validation.warnings {
        asker.warn(warning);
    }
    let query = validation
        .query
        .ok_or_else(|| invalid("PQL is not valid for this table and Tabular model"))?;
    let (NodeKind::Column { column: entity }, NodeKind::Column { column: target }) =
        (&query.entity.kind, &query.target.kind)
    else {
        return Err(invalid(
            "Tabular PQL needs an entity column and a target column",
        ));
    };
    let table = &input.schema.tables[0];
    let keys = entity_keys(&data, &entity.column)?;
    let source = keys
        .rows
        .iter()
        .map(tabular::session::canonical_key)
        .collect::<Result<Vec<_>, _>>()?;
    let mut requested = input
        .indices
        .iter()
        .map(|k| tabular::session::canonical_key(&vec![k.clone()]))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(scoring) = &supplied {
        requested = entity_keys(scoring, &entity.column)?
            .rows
            .iter()
            .map(tabular::session::canonical_key)
            .collect::<Result<Vec<_>, _>>()?;
    }
    if supplied.is_none() && requested.iter().any(Option::is_none) {
        return Err(invalid("missing entity keys cannot be requested"));
    }
    let expanded = sdm_engine::tabular::expand_keys(&source, &requested);
    if supplied.is_none() && !expanded.missing.is_empty() {
        return Err(invalid("requested entity keys are not present"));
    }
    let target_at = data
        .schema
        .index_of(&target.column)
        .map_err(|_| invalid("target column absent"))?;
    let known = data
        .batches
        .iter()
        .flat_map(|b| (0..b.num_rows()).map(move |r| b.column(target_at).is_valid(r)))
        .collect::<Vec<_>>();
    let candidates = sdm_engine::tabular::candidates(&known, &expanded.rows)
        .ok_or_else(|| internal("invalid row positions"))?;
    let decision = sdm_engine::tabular::context_decision(
        candidates.len(),
        input.num_context_samples,
        sdm_request::predict::CONTEXT_ROWS_MAX,
    )
    .map_err(|_| invalid("context sample count is invalid"))?;
    let context_rows = if decision.reduced {
        let mut rows = sdm_engine::tabular::sample_offsets(candidates.len(), decision.wanted)
            .map_err(|_| invalid("sample count exceeds population"))?
            .into_iter()
            .map(|at| candidates[at])
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    } else {
        candidates
    };
    let context = select(&data, &context_rows, None)?;
    let scoring = match supplied {
        Some(scoring) => scoring,
        None => select(&data, &expanded.rows, Some(&target.column))?,
    };
    let selected = describe(asker, &context, &table.name)?;
    let target_column = selected.tables[0]
        .columns
        .iter()
        .find(|c| c.name == target.column)
        .ok_or_else(|| internal("selected target missing"))?;
    let task = match query.problem_type {
        Some(ProblemType::Classify) => "classification".to_owned(),
        _ => {
            let encoded = encoded_column(&context, &target.column)?;
            let distinct = match encoded.get("rows") {
                Some(Value::Array(rows)) => rows
                    .iter()
                    .filter_map(|r| match r {
                        Value::Array(c) => c.first(),
                        _ => None,
                    })
                    .map(Value::to_bytes)
                    .collect::<std::collections::BTreeSet<_>>(),
                _ => Default::default(),
            };
            let binary=distinct.len()==2 && distinct.iter().all(|v| matches!(sdm_wire::json::read(v), Ok(Value::Number(n)) if n.parse::<f64>().is_ok_and(|n| n.is_finite() && n.fract()==0.0)));
            match tabular::decide_materialized_task(DecideTabularTask::Kind {
                target_stype: direct::stype_name(target_column.stype).into(),
                stated: None,
                binary_whole: Some(binary),
            }) {
                TabularTaskDecision::Task { kind, .. } => kind,
                _ => return Err(invalid("target is not supported by Tabular")),
            }
        }
    };
    let request = ArrowRequest {
        target: target.column.clone(),
        task,
        entity_table: table.name.clone(),
        schema: Some(input.schema.clone()),
        query: Some(input.query.clone()),
        context_identity: None,
    };
    let description = prepare(asker, &request, &context, &scoring)?;
    let context = asker.attach(Attachment::Arrow(native(context)?))?;
    let predict = asker.attach(Attachment::Arrow(native(scoring)?))?;
    crate::direct::value(
        asker,
        &serde_json::json!({"description":description,"context":context,"predict":predict}),
    )
}
fn question(request: &ArrowRequest) -> Option<sdm_contract::session::SessionQuestion> {
    request
        .query
        .as_ref()
        .map(|query| sdm_contract::session::SessionQuestion {
            query: Some(query.clone()),
            kind: request.task.clone(),
            target: request.target.clone(),
            dtype: request
                .schema
                .as_ref()
                .and_then(|s| s.tables.iter().find(|t| t.name == request.entity_table))
                .and_then(|t| t.columns.iter().find(|c| c.name == request.target))
                .map(|c| c.dtype.clone())
                .unwrap_or_default(),
            entity_table: Some(request.entity_table.clone()),
            context: request.context_identity.clone(),
        })
}

/// Export logical Arrow types, not the encoder's internal physical representation.
fn native(data: ArrowData) -> Result<ArrowData, CoreError> {
    let fields = data
        .schema
        .fields()
        .iter()
        .map(|field| {
            let storage = field
                .metadata()
                .get(DIRECT_STORAGE_KEY)
                .and_then(|s| serde_json::from_str::<DirectStorage>(s).ok());
            let dtype = if let Some(DirectStorage::Timestamp { unit, zone }) = storage {
                let unit = match unit {
                    TimestampUnit::Seconds => TimeUnit::Second,
                    TimestampUnit::Milliseconds => TimeUnit::Millisecond,
                    TimestampUnit::Microseconds => TimeUnit::Microsecond,
                    TimestampUnit::Nanoseconds => TimeUnit::Nanosecond,
                };
                DataType::Timestamp(unit, zone.map(Into::into))
            } else {
                field.data_type().clone()
            };
            Field::new(field.name(), dtype, field.is_nullable())
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    arrow_cast::cast(column, field.data_type())
                        .map_err(|_| internal("native Arrow export failed"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|_| internal("native Arrow batch invalid"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowData::new(schema, batches))
}

fn encoded_column(data: &ArrowData, entity: &str) -> Result<Value, CoreError> {
    let at = data
        .schema
        .index_of(entity)
        .map_err(|_| invalid("entity column absent"))?;
    let schema = Arc::new(
        data.schema
            .project(&[at])
            .map_err(|_| internal("key projection failed"))?,
    );
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            batch
                .project(&[at])
                .map_err(|_| internal("key projection failed"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtypes = sdm_schema::direct::storage_dtypes(&schema, &batches)?;
    let columns = TabularColumns {
        dtypes: vec![(entity.to_owned(), dtypes[0])],
        columns: vec![],
        target: String::new(),
    };
    tabular::encode_table(&columns, &batches, crate::pql::float_text)
}

fn entity_keys(
    data: &ArrowData,
    entity: &str,
) -> Result<sdm_contract::session::SessionKeys, CoreError> {
    tabular::session::encoded_keys(&encoded_column(data, entity)?, &[entity.to_owned()])
}
