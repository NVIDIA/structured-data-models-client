// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Caller-owned labels enter the existing graph sampler without a PQL target.
use super::*;
use crate::run::{Attachments, Run};
use sdm_contract::examples::{ExampleCall, ExampleKind, ExampleSource, MaterializeExamples};

fn refuse(reason: &str) -> CoreError {
    CoreError::new(Sentence::SchemaRefinement, reason).with_field("reason", Detail::text(reason))
}

pub(crate) fn start(
    root: &crate::compose::CompositionRoot,
    request: MaterializeExamples,
    attachments: Attachments,
    facts: HostFacts,
) -> Result<Run, CoreError> {
    match &request.source {
        ExampleSource::Sql { provider, .. } => {
            let data = root.data_access(provider)?;
            Ok(Run::with_data(data, move |asker| async move {
                run(asker, request, attachments, facts).await
            }))
        }
        ExampleSource::Frames { .. } => Ok(Run::new(move |asker| async move {
            run(asker, request, attachments, facts).await
        })),
    }
}

fn arrow(attachments: &mut Attachments, id: AttachmentId) -> Result<RecordBatch, CoreError> {
    let Some(Attachment::Arrow(data)) = attachments.remove(&id) else {
        return Err(internal("example table attachment is absent"));
    };
    one_batch(&crate::arrow_logical::logical(data)?)
}

fn anchors(rows: &RecordBatch, temporal: bool, default: i64) -> Result<Vec<i64>, CoreError> {
    let Some(column) = rows.column_by_name("ANCHOR_TIMESTAMP") else {
        return if temporal {
            Err(refuse(
                "temporal and forecast examples require ANCHOR_TIMESTAMP",
            ))
        } else {
            Ok(vec![default; rows.num_rows()])
        };
    };
    if column.null_count() == rows.num_rows() && !temporal {
        return Ok(vec![default; rows.num_rows()]);
    }
    if !matches!(
        column.data_type(),
        DataType::Timestamp(..) | DataType::Date32 | DataType::Date64
    ) {
        return Err(refuse(
            "ANCHOR_TIMESTAMP must contain Arrow timestamps or dates",
        ));
    }
    times_ns(column, "ANCHOR_TIMESTAMP")?
        .into_iter()
        .map(|v| {
            v.filter(|v| *v != i64::MIN)
                .ok_or_else(|| refuse("example anchors must not be null"))
        })
        .collect()
}

fn check_columns(rows: &RecordBatch, context: bool) -> Result<(), CoreError> {
    if rows.num_rows() == 0 {
        return Err(refuse("context and prediction examples must be nonempty"));
    }
    let unique: BTreeSet<_> = rows
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    if unique.len() != rows.num_columns() {
        return Err(refuse("example column names must be unique"));
    }
    let allowed = if context {
        vec!["ENTITY", "ANCHOR_TIMESTAMP", "TARGET"]
    } else {
        vec!["ENTITY", "ANCHOR_TIMESTAMP"]
    };
    if rows
        .schema()
        .fields()
        .iter()
        .any(|f| !allowed.contains(&f.name().as_str()))
        || rows.column_by_name("ENTITY").is_none()
        || (context && rows.column_by_name("TARGET").is_none())
    {
        return Err(refuse(
            "example columns are ENTITY, optional ANCHOR_TIMESTAMP, and context-only TARGET",
        ));
    }
    if rows
        .column_by_name("ENTITY")
        .is_some_and(|a| a.null_count() != 0)
    {
        return Err(refuse("example ENTITY values must not be null"));
    }
    Ok(())
}

async fn run(
    asker: Asker,
    request: MaterializeExamples,
    mut attachments: Attachments,
    facts: HostFacts,
) -> Result<OperationResult, CoreError> {
    let call: ExampleCall = crate::attached::take_json(&mut attachments, request.input)?;
    if call.rows_per_call == Some(0) {
        return Err(refuse("rows_per_call must be positive"));
    }
    let context = arrow(&mut attachments, request.context)?;
    let predict = arrow(&mut attachments, request.predict)?;
    check_columns(&context, true)?;
    check_columns(&predict, false)?;
    let temporal = matches!(
        call.task.kind,
        ExampleKind::Forecasting | ExampleKind::TemporalLinkPrediction
    );
    let forecasting = call.task.kind == ExampleKind::Forecasting;
    let linking = call.task.kind == ExampleKind::TemporalLinkPrediction;
    if linking != call.task.link_target_table.is_some() {
        return Err(refuse("only temporal link tasks require link_target_table"));
    }
    if !forecasting && (call.task.step_size_ns.is_some() || call.task.num_forecasts.is_some()) {
        return Err(refuse("forecast settings require a forecasting task"));
    }
    let forecast = if forecasting {
        let step = call
            .task
            .step_size_ns
            .filter(|v| *v > 0)
            .ok_or_else(|| refuse("forecast step_size_ns must be positive"))?;
        let count = call
            .task
            .num_forecasts
            .filter(|v| *v > 0 && *v as usize <= context.num_rows())
            .ok_or_else(|| {
                refuse("num_forecasts must be positive and no larger than the context")
            })?;
        step.checked_mul(i64::from(count))
            .ok_or_else(|| refuse("forecast horizon exceeds the timestamp range"))?;
        Some((step, count))
    } else {
        None
    };
    let seed = call
        .random_seed
        .to_u128()
        .ok_or_else(|| refuse("random_seed exceeds the supported range"))?;
    let inference = request
        .inference
        .map(|id| {
            let Some(Attachment::Bytes(bytes)) = attachments.remove(&id) else {
                return Err(internal("example inference attachment absent"));
            };
            sdm_wire::json::read_options(&bytes)
                .map_err(|_| refuse("inference must be a JSON object"))
        })
        .transpose()?;
    let task = match call.task.kind {
        ExampleKind::BinaryClassification => Task::Binary,
        ExampleKind::MulticlassClassification => Task::Multiclass,
        _ => Task::Regression,
    };
    let options = if linking {
        options::frame_link_options(inference.as_ref())?
    } else {
        options::options(task, inference.as_ref())?
    };
    let labels = named(&context, "TARGET")?.clone();
    if labels.null_count() != 0 {
        return Err(refuse("supplied TARGET labels must not be null"));
    }
    if !linking {
        payload::validate_supplied_target(task, &labels, crate::pql::float_text)?;
    }
    let frames;
    let description = match request.source {
        ExampleSource::Frames { graph } => {
            let Some(Attachment::FrameGraph(owner)) = attachments.remove(&graph) else {
                return Err(internal("example graph owner absent"));
            };
            let description = owner.description();
            frames = Some(owner);
            description
        }
        ExampleSource::Sql { schema, .. } => {
            frames = None;
            let held = crate::attached::take_json(&mut attachments, schema)?;
            crate::describe::initialize(&asker, Some(&held), &facts)
                .await?
                .ok_or_else(|| internal("example source not initialized"))?
        }
    };
    let _lease = frames.as_ref().map(frames::FrameGraph::lease).transpose()?;
    let graph = frames
        .as_ref()
        .map_or_else(|| Graph::new(&description), |f| Ok(f.graph()))?;
    let entity = graph
        .table(&call.task.entity_table)
        .ok_or_else(|| refuse("entity_table is absent from the schema"))?;
    let key = entity
        .primary_key
        .as_ref()
        .ok_or_else(|| refuse("entity_table requires a primary key"))?;
    let destination = call
        .task
        .link_target_table
        .as_ref()
        .map(|name| {
            graph
                .table(name)
                .ok_or_else(|| refuse("link_target_table is absent from the schema"))
        })
        .transpose()?;
    if destination.is_some_and(|table| table.primary_key.is_none()) {
        return Err(refuse("link_target_table requires a primary key"));
    }
    let nulls = if frames.is_some() {
        NullOrder::Last
    } else {
        sampling_capabilities(&asker)?
    };
    let mut memo = Memo {
        asker: &asker,
        memo: SamplerMemo::default(),
    };
    if frames.is_none() {
        table_rows(&mut memo, &graph).await?;
    }
    let mut times = Times::new(
        crate::describe::text_reader(&facts)?,
        if frames.is_some() {
            Some(TextTimes::ValueByValue)
        } else {
            asker.capabilities()?.text_times
        },
        &memo.memo,
    );
    times.frames = frames.clone();
    times.text_utc |= frames.is_some();
    let bounded = graph
        .tables
        .iter()
        .filter(|t| t.time_column.is_some())
        .collect::<Vec<_>>();
    let default = time_bounds(&mut memo, &times, &bounded).await?.max;
    let train_anchors = anchors(&context, temporal, default)?;
    let predict_anchors = anchors(&predict, temporal, default)?;
    if let Some((step, count)) = forecast {
        for anchor in &predict_anchors {
            anchor
                .checked_add(
                    step.checked_mul(i64::from(count))
                        .ok_or_else(|| refuse("forecast horizon overflow"))?,
                )
                .ok_or_else(|| refuse("forecast anchor plus horizon overflow"))?;
        }
    }
    let key_type = if let Some(f) = &frames {
        f.column_type(&entity.name, key)?
    } else {
        entity
            .column(key)
            .ok_or_else(|| refuse("entity key column absent"))?
            .received
            .to_arrow()
    };
    let train_keys = keys::cast(named(&context, "ENTITY")?, &key_type)?;
    let test_keys = keys::cast(named(&predict, "ENTITY")?, &key_type)?;
    if keys::read(&train_keys)?
        .iter()
        .chain(keys::read(&test_keys)?.iter())
        .any(Option::is_none)
    {
        return Err(refuse("example ENTITY values must be valid identifiers"));
    }
    let link = if let Some(table) = destination {
        Some(link_labels(table, &labels, &frames)?)
    } else {
        None
    };
    if let Some(table) = destination {
        validate_link_references(&asker, table, &labels, frames.as_ref(), &times).await?;
    }
    let mode = options::run_mode(
        Some(options::Integer {
            negative: false,
            magnitude: Some(context.num_rows() as u128),
        }),
        None,
    )?;
    options::hops(
        None,
        call.num_neighbors.as_ref().map(|v| options::Integer {
            negative: false,
            magnitude: Some(v.len() as u128),
        }),
    )?;
    let mut asked = options::values(None)?;
    asked.neighbors = call.num_neighbors.as_ref().map(|sizes| {
        sizes
            .iter()
            .map(|v| sdm_wire::json::Value::Number(v.to_string()))
            .collect()
    });
    let hops = options::neighbors(&asked, mode)?;
    let lookback = if frames.is_some() {
        vec![]
    } else {
        lookbacks(&mut memo, &times, &graph).await?
    };
    let names = crate::subgraph::sample_columns(&graph, entity, &[]);
    let per_call = if forecasting {
        1
    } else {
        let task = if linking {
            options::BatchTask::TemporalLink
        } else {
            options::BatchTask::Ordinary
        };
        usize::try_from(options::rows_one_task_call_scores(call.rows_per_call, task))
            .map_err(|_| internal("batch size overflow"))?
    };
    let mut calls = Vec::new();
    for start in (0..predict.num_rows()).step_by(per_call) {
        let count = per_call.min(predict.num_rows() - start);
        let all_keys = arrow_select::concat::concat(&[
            train_keys.as_ref(),
            test_keys.slice(start, count).as_ref(),
        ])
        .map_err(|_| refuse("context and prediction ENTITY types differ"))?;
        let anchors = train_anchors
            .iter()
            .copied()
            .chain(predict_anchors[start..start + count].iter().copied())
            .collect::<Vec<_>>();
        let rows = if let Some(f) = &frames {
            f.lookup(entity, &all_keys, &names)?
        } else {
            times.read(
                &asker,
                entity,
                one_batch(
                    &ask(
                        &asker,
                        sample::by_pkey(
                            entity,
                            names.clone(),
                            key_set(&asker, sample::seeds(all_keys.clone(), vec![])?)?,
                            true,
                        )?,
                    )
                    .await?,
                )?,
            )?
        };
        let positions = batches(&rows, all_keys.len())?;
        let mut ordered = vec![None; all_keys.len()];
        for (row, position) in positions.into_iter().enumerate() {
            if position >= ordered.len() || ordered[position].replace(row).is_some() {
                return Err(refuse("an example entity has multiple source rows"));
            }
        }
        let order = ordered
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| refuse("an example entity is absent from its source table"))?;
        if let Some(matches) = rows.column_by_name(MATCHES_COLUMN)
            && matches
                .as_primitive::<Int64Type>()
                .iter()
                .any(|n| n != Some(1))
        {
            return Err(refuse("an example entity has multiple source rows"));
        }
        let rows = take_batch(&rows, &order)?;
        let rows = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(
                names
                    .iter()
                    .map(|n| rows.schema().field_with_name(n).cloned())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| internal("entity feature absent"))?,
            )),
            names
                .iter()
                .map(|n| named(&rows, n).cloned())
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|_| internal("entity features do not align"))?;
        let walked = walk_examples(
            crate::subgraph::Walk {
                link_entity: call.task.link_target_table.as_deref(),
                asker: &asker,
                graph: &graph,
                times: &times,
                anchors: &anchors,
                nulls,
                entity,
                neighbors: &hops,
                excluded: &[],
                lookback: &lookback,
            },
            frames.as_ref(),
            &all_keys,
            rows,
            Some(seed),
        )
        .await?;
        if let Some(destination) = destination
            && !walked.tables.iter().any(|t| t.name == destination.name)
        {
            return Err(refuse(
                "link destination must be reachable in the sampled graph",
            ));
        }
        let sampled = Sampled {
            num_train: context.num_rows(),
            num_test: count,
            anchors,
            kind: task,
            labels: labels::Labels::Float(vec![]),
            windowed: temporal,
            tables: walked.tables,
            links: walked.links,
            entity_table: entity.name.clone(),
            source_query: String::new(),
            run_mode: mode,
            use_prediction_time: false,
            return_embeddings: false,
            options: options.clone(),
            statistic: None,
            top_k: None,
            lags: vec![],
            float_text: crate::pql::float_text,
        };
        let built = payload::build_supplied(
            &sampled,
            link.as_ref(),
            &payload::Supplied {
                labels: &labels,
                forecast,
            },
        )?;
        calls.push((built.bytes, count as u64));
    }
    let mut calls = calls.into_iter();
    let (first, _) = calls.next().ok_or_else(|| internal("no example payload"))?;
    let (payload, query) = attach_request(&asker, first, "")?;
    let batches = calls
        .map(|(bytes, rows)| {
            Ok(RequestBatch {
                payload: asker.attach(Attachment::Bytes(bytes))?,
                rows,
                request_id: None,
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(OperationResult::Materialize(Materialized {
        state: RequestState {
            version: REQUEST_STATE_VERSION,
            payload,
            query,
            forecast: None,
            batches: (!batches.is_empty()).then_some(batches),
            ranking: None,
        },
        query: None,
        examples: Some(sdm_contract::examples::ExampleFacts {
            entity_table: entity.name.clone(),
            kind: call.task.kind,
            windowed: temporal,
            num_forecasts: forecast.map_or(1, |(_, count)| count),
        }),
    }))
}

fn link_labels(
    table: &Table,
    labels: &ArrayRef,
    frames: &Option<frames::FrameGraph>,
) -> Result<payload::LinkTarget, CoreError> {
    let key = table
        .primary_key
        .as_ref()
        .ok_or_else(|| refuse("link destination requires a key"))?;
    let dtype = if let Some(f) = frames {
        f.column_type(&table.name, key)?
    } else {
        table
            .column(key)
            .ok_or_else(|| refuse("link key absent"))?
            .received
            .to_arrow()
    };
    let mut groups = Vec::with_capacity(labels.len());
    for row in 0..labels.len() {
        let values = match labels.data_type() {
            DataType::List(_) => labels.as_list::<i32>().value(row),
            DataType::LargeList(_) => labels.as_list::<i64>().value(row),
            _ => {
                return Err(refuse(
                    "link TARGET labels must be lists of destination keys",
                ));
            }
        };
        let values = keys::cast(&values, &dtype)?;
        let values = sdm_engine::frame_labels::identifier_values(&values)?
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| refuse("link labels cannot contain null destination keys"))?;
        groups.push(values);
    }
    Ok(payload::LinkTarget {
        table: table.name.clone(),
        labels: groups,
    })
}

async fn validate_link_references(
    asker: &Asker,
    table: &Table,
    labels: &ArrayRef,
    frames: Option<&frames::FrameGraph>,
    times: &Times,
) -> Result<(), CoreError> {
    let key = table
        .primary_key
        .as_ref()
        .ok_or_else(|| refuse("link destination requires a key"))?;
    let values = match labels.data_type() {
        DataType::List(_) => {
            let a = labels.as_list::<i32>();
            let offsets = a.value_offsets();
            a.values().slice(
                offsets[0] as usize,
                (offsets[a.len()] - offsets[0]) as usize,
            )
        }
        DataType::LargeList(_) => {
            let a = labels.as_list::<i64>();
            let offsets = a.value_offsets();
            a.values().slice(
                offsets[0] as usize,
                (offsets[a.len()] - offsets[0]) as usize,
            )
        }
        _ => return Err(refuse("link TARGET must contain lists")),
    };
    if values.is_empty() {
        return Ok(());
    }
    let dtype = frames.map_or_else(
        || {
            Ok(table
                .column(key)
                .ok_or_else(|| refuse("link key absent"))?
                .received
                .to_arrow())
        },
        |f| f.column_type(&table.name, key),
    )?;
    let values = keys::cast(&values, &dtype)?;
    let columns = vec![key.clone()];
    let rows = if let Some(f) = frames {
        f.lookup(table, &values, &columns)?
    } else {
        times.read(
            asker,
            table,
            one_batch(
                &ask(
                    asker,
                    sample::by_pkey(
                        table,
                        columns,
                        key_set(asker, sample::seeds(values.clone(), vec![])?)?,
                        true,
                    )?,
                )
                .await?,
            )?,
        )?
    };
    let positions = batches(&rows, values.len())?;
    let found = positions.into_iter().collect::<BTreeSet<_>>();
    if found.len() != values.len() {
        return Err(refuse("a link label names a missing destination entity"));
    }
    Ok(())
}
