// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production ground-truth extraction. No request payload or neighbor walk.
use super::*;
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int64Array, ListArray, StringArray, UInt64Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::Field as ArrowField;

/// Only label inputs: no model options, request state, neighbors or scoring rows.
pub(crate) struct Inputs {
    pub frames: Option<frames::FrameGraph>,
    pub description: SourceDescription,
    pub query: String,
    pub pending: Option<sdm_contract::ops::HeldSchema>,
    pub types: sdm_schema::assemble::HostTypes,
    pub facts: HostFacts,
    pub zones: Arc<dyn sdm_engine::time::Zones>,
    pub anchor_time: Option<Scalar>,
    pub anchor_entity: bool,
    pub seed: Option<Seed>,
    pub unseeded: bool,
    pub source: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Options {
    pub size: u64,
    pub max_iterations: u64,
}

pub(crate) fn source(
    root: &crate::compose::CompositionRoot,
    request: sdm_contract::labels::LabelsSource,
    mut attachments: crate::run::Attachments,
    facts: HostFacts,
    zones: Arc<dyn sdm_engine::time::Zones>,
) -> Result<crate::run::Run, CoreError> {
    let held: sdm_contract::ops::HeldSchema =
        crate::attached::take_json(&mut attachments, request.schema)?;
    let call: sdm_contract::labels::LabelsCall =
        crate::attached::take_json(&mut attachments, request.input)?;
    let access = root.data_access(&request.provider)?;
    Ok(crate::run::Run::with_data(
        access,
        move |asker| async move {
            let source_tables = asker
                .data()
                .is_some_and(crate::compose::DataAccess::reads_a_source);
            let lacking = !sdm_schema::held::lacking_facts(&held).is_empty();
            let read = sdm_schema::held::read(held.clone(), &BTreeMap::new(), lacking)?;
            meanings_carried(&read.description, Some(&read.types))?;
            sdm_schema::assemble::check_links(&read.description, &read.types)?;
            let description = read.by_the_source();
            for table in &description.tables {
                asker.guard_held(
                    &table.source,
                    table
                        .columns
                        .iter()
                        .map(|column| column.name.clone())
                        .collect(),
                );
            }
            let inputs = Inputs {
                frames: None,
                description,
                query: call.query,
                pending: (source_tables || lacking).then_some(held),
                types: read.types,
                facts,
                zones,
                anchor_time: call.anchor_time,
                anchor_entity: call.anchor_entity,
                seed: call.seed,
                unseeded: call.unseeded,
                source: "labels".into(),
            };
            let rows = extract(
                &asker,
                inputs,
                Options {
                    size: call.size,
                    max_iterations: call.max_iterations,
                },
            )
            .await?;
            finish(&asker, rows)
        },
    ))
}

pub(crate) fn finish(asker: &Asker, rows: ArrowData) -> Result<OperationResult, CoreError> {
    let data = asker.attach(crate::run::Attachment::Arrow(rows))?;
    Ok(OperationResult::Labels(sdm_contract::labels::LabelRows {
        data,
    }))
}

fn invalid_option(field: &str, message: &str) -> CoreError {
    CoreError::new(Sentence::LabelsInvalid, message)
        .with_field("field", Detail::text(field))
        .with_field("reason", Detail::text(message))
}

pub(crate) async fn extract(
    asker: &Asker,
    mut inputs: Inputs,
    options: Options,
) -> Result<ArrowData, CoreError> {
    if options.max_iterations == 0 {
        return Err(invalid_option(
            "max_iterations",
            "max_iterations must be positive",
        ));
    }
    let size = usize::try_from(options.size)
        .map_err(|_| invalid_option("size", "size exceeds this platform"))?;
    let trials = options
        .size
        .checked_mul(options.max_iterations)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid_option("size", "size times max_iterations exceeds this platform"))?;
    let frames = inputs.frames.clone();
    let mut memo = Memo {
        asker,
        memo: SamplerMemo {
            source: inputs.source.clone(),
            ..SamplerMemo::default()
        },
    };
    consistent(&mut inputs.description, &inputs.types)?;
    if let Some(located) =
        crate::describe::initialize(asker, inputs.pending.as_ref(), &inputs.facts).await?
    {
        inputs.description = located;
        sdm_schema::assemble::mark_keys(&mut inputs.description);
    }
    kept_checks(&inputs.description)?;
    if frames.is_none() {
        refuse_composite_keys(&inputs.description)?;
    }
    let graph = match &frames {
        Some(frame) => frame.graph(),
        None => Graph::new(&inputs.description)?,
    };
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    let schema = query_schema(&inputs.description);
    let validation = crate::pql::resolve(
        &inputs.query,
        &schema,
        &profile,
        &inputs.facts,
        &*inputs.zones,
    )?;
    for warning in &validation.warnings {
        asker.warn(warning.clone());
    }
    let validated = validation
        .query
        .as_ref()
        .ok_or_else(|| unresolved(&inputs.query, &validation.diagnostics))?;
    if validation.ranking.is_some() {
        return Err(not_yet("ranking_labels"));
    }
    let forecast_steps = validation.steps.as_ref().map_or(1, |v| v.count);
    let selected_ids = validated.entity_ids.clone();
    let single;
    let validated = if let Some(plan) = &validation.steps {
        let text = plan.step(1);
        single = crate::pql::resolve(&text, &schema, &profile, &inputs.facts, &*inputs.zones)?;
        single
            .query
            .as_ref()
            .ok_or_else(|| unresolved(&text, &single.diagnostics))?
    } else {
        validated
    };
    let asked = asked(validated)?;
    let entity = graph
        .table(&asked.entity_table)
        .ok_or_else(|| internal("labels entity absent"))?;
    let key = entity
        .primary_key
        .as_deref()
        .filter(|key| *key == asked.entity_column)
        .ok_or_else(|| not_yet("entity_key"))?;
    let key_type = entity
        .column(key)
        .ok_or_else(|| internal("labels key absent"))?
        .received
        .to_arrow();
    let output_type = target_type(&asked, &graph)?;
    let seed = match inputs.seed {
        Some(seed) => seed,
        None if inputs.unseeded => {
            Seed::from(u128::from(fresh_seed(&inputs.facts.now, u32::MAX as u64)))
        }
        None => Seed::from(DEFAULT_SEED),
    };
    if seed.is_negative() {
        return Err(invalid_option("random_seed", "seed must be nonnegative"));
    }
    let entropy = Entropy::from_hex(seed.magnitude_hex())
        .ok_or_else(|| invalid_option("random_seed", "invalid seed"))?;
    let seed_value = seed_scalar(&seed)?;
    if inputs.anchor_entity && asked.range.is_some() {
        return Err(invalid_option(
            "anchor_time",
            "entity anchors require a static target",
        ));
    }
    if inputs.anchor_entity && entity.time_column.is_none() {
        return Err(invalid_option(
            "anchor_time",
            "entity anchors require a time column",
        ));
    }
    if size == 0 {
        return result(
            arrow_array::new_empty_array(&key_type),
            vec![],
            arrow_array::new_empty_array(&output_type),
        );
    }
    if frames.is_none() {
        sampling_capabilities(asker)?;
        if asker.capabilities()?.sample_rows {
            table_rows(
                &mut memo,
                &Graph {
                    tables: vec![entity.clone()],
                    edges: vec![],
                },
            )
            .await?;
        }
    }
    let mut times = Times::new(
        crate::describe::text_reader(&inputs.facts)?,
        if frames.is_some() {
            Some(TextTimes::ValueByValue)
        } else {
            asker.capabilities()?.text_times
        },
        &memo.memo,
    );
    times.frames = frames.clone();
    times.text_utc |= frames.is_some();
    let mut windows = Vec::new();
    asked.target.windowed(&mut windows);
    let target_fetches = fetches(&graph, &asked.entity_table, &windows)?;
    let all_fetches = fetches(&graph, &asked.entity_table, &asked.every_windowed())?;
    let bounded: Vec<&Table> = if target_fetches.is_empty() {
        graph
            .tables
            .iter()
            .filter(|t| t.time_column.is_some())
            .collect()
    } else {
        target_fetches.iter().map(|f| f.table).collect()
    };
    let span = asked.range.map(|range| {
        (
            Offset::of(range.start.unwrap_or(0), range.unit),
            Offset::of(range.end, range.unit),
        )
    });
    let bounds = if inputs.anchor_entity || bounded.is_empty() {
        None
    } else {
        Some(time_bounds(&mut memo, &times, &bounded).await?)
    };
    let anchor = match inputs.anchor_time {
        Some(Scalar::Timestamp {
            unit: Unit::Ns,
            value,
            ..
        }) => value,
        Some(Scalar::Timestamp {
            unit: Unit::Us,
            value,
            ..
        }) => value
            .checked_mul(1000)
            .ok_or_else(|| sample::out_of_range("anchor_time"))?,
        Some(_) => {
            return Err(invalid_option(
                "anchor_time",
                "anchor_time must be a timestamp",
            ));
        }
        None => {
            let latest = bounds.as_ref().map_or(TIME_MIN, |b| b.max);
            match span {
                Some((_, end)) => Offset {
                    count: end
                        .count
                        .checked_mul(
                            i64::try_from(forecast_steps)
                                .map_err(|_| sample::out_of_range("anchor_time"))?,
                        )
                        .ok_or_else(|| sample::out_of_range("anchor_time"))?,
                    unit: end.unit,
                }
                .negated()
                .add_to(latest, "anchor_time")?,
                None => latest,
            }
        }
    };
    if bounds.as_ref().is_some_and(|b| anchor < b.min) {
        return Err(invalid_option(
            "anchor_time",
            "anchor precedes available data",
        ));
    }
    let mut columns = BTreeSet::from([key.to_owned()]);
    if let Some(time) = &entity.time_column {
        columns.insert(time.clone());
    }
    asked.target.entity_columns(&mut columns);
    for shape in &asked.choosing {
        shape.entity_columns(&mut columns);
    }
    let columns = entity.ordered(&columns);
    let explicit = if validation.steps.is_some() {
        selected_ids
            .as_ref()
            .map(|node| first_id(node, &key_type))
            .transpose()?
    } else {
        None
    };
    let sampled = if let Some(frame) = &frames {
        match &explicit {
            Some(keys) => frame.lookup(entity, keys, &columns)?,
            None => frame.entities(entity, &columns, trials, &seed)?,
        }
    } else {
        let mut filters = vec![Filter::NotNull { column: key.into() }];
        if let Some(time) = &entity.time_column {
            filters.push(Filter::NotNull {
                column: time.clone(),
            });
        }
        if let Some(keys) = explicit {
            let batch = RecordBatch::try_from_iter([(key, keys)])
                .map_err(|_| internal("label key schema"))?;
            filters.push(Filter::KeyIn {
                columns: vec![key.into()],
                keys: key_set(asker, batch)?,
            });
        }
        times.read(
            asker,
            entity,
            one_batch(
                &ask(
                    asker,
                    DataRequest::SampleEntities {
                        table: entity.source.clone(),
                        key: vec![key.into()],
                        n: trials as u64,
                        seed: seed_value,
                        columns,
                        filters,
                        key_dtype: asker
                            .samples_by_key_dtype()
                            .then(|| entity.column(key).map(|c| c.dtype))
                            .flatten(),
                        rows: if asker.capabilities()?.sample_rows {
                            rows_of(&memo.memo, &entity.name)
                        } else {
                            None
                        },
                    },
                )
                .await?,
            )?,
        )?
    };
    if sampled.num_rows() == 0 {
        return Err(invalid_option("query", "no eligible entity exists"));
    }
    let mut seen = BTreeSet::new();
    let keys = crate::subgraph::key_values(entity, key, named(&sampled, key)?)?;
    let kept: Vec<usize> = (0..keys.len())
        .filter(|i| seen.insert(keys[*i].clone()))
        .collect();
    let entity_key = take(named(&sampled, key)?, &kept)?;
    let entity_time = entity
        .time_column
        .as_ref()
        .map(|time| {
            let values = times_ns(named(&sampled, time)?, time)?;
            Ok::<_, CoreError>(kept.iter().map(|i| values[*i]).collect::<Vec<_>>())
        })
        .transpose()?;
    let mut rng = default_rng(&entropy);
    let mut examples = if inputs.anchor_entity {
        sample::entity_examples(
            &mut rng,
            kept.len(),
            entity_time.as_deref().unwrap_or_default(),
            sample::MISSING_ANCHOR,
        )
    } else {
        sample::static_examples(&mut rng, kept.len(), entity_time.as_deref(), anchor)
    };
    if examples.index.is_empty() {
        return Err(invalid_option(
            "anchor_time",
            "no entity is eligible at the anchor",
        ));
    }
    if validation.steps.is_some() {
        let selected = examples.index[0];
        let (start, end) = span.ok_or_else(|| internal("forecast has no window"))?;
        let step = Offset {
            count: end.count - start.count,
            unit: end.unit,
        };
        examples.index.clear();
        examples.anchors.clear();
        let mut at = anchor;
        let produced_steps = forecast_steps.min(trials as u64);
        for step_index in 0..produced_steps {
            examples.index.push(selected);
            examples.anchors.push(at);
            if step_index + 1 < produced_steps {
                at = step.add_to(at, "anchor_time")?;
            }
        }
    }
    examples.index.truncate(trials);
    examples.anchors.truncate(trials);
    let entity_rows: Vec<usize> = examples.index.iter().map(|i| kept[*i]).collect();
    let labelling = Labelling {
        asker,
        times: &times,
        sampled: &sampled,
        entity_rows: &entity_rows,
        entity_key: &entity_key,
        examples: &examples,
    };
    let fetched = labelling.read(&all_fetches).await?;
    let labelled = label_examples(
        &asked,
        &labelling,
        fetched,
        &examples,
        &nan_rows(&entity_key),
    )?;
    let positions: Vec<usize> = (0..labelled.mask.len())
        .filter(|i| labelled.mask[*i])
        .take(size)
        .collect();
    let key_positions: Vec<usize> = positions.iter().map(|i| examples.index[*i]).collect();
    let target = target_array(labelled.labels.truncated(positions.len()), &output_type)?;
    result(
        take(&entity_key, &key_positions)?,
        positions.iter().map(|i| examples.anchors[*i]).collect(),
        target,
    )
}

fn first_id(node: &PqlNode, dtype: &DataType) -> Result<ArrayRef, CoreError> {
    let value = match &node.kind {
        NodeKind::Condition { value, .. } => value.as_ref(),
        _ => node,
    };
    let value = match &value.kind {
        NodeKind::Constant {
            literal: PqlLiteral::Array { items },
        } => items
            .first()
            .ok_or_else(|| invalid_option("query", "forecast entity list is empty"))?,
        _ => value,
    };
    let values: ArrayRef = match &value.kind {
        NodeKind::Constant {
            literal: PqlLiteral::Int { value },
        } => Arc::new(StringArray::from(vec![value.as_str()])),
        NodeKind::Constant {
            literal: PqlLiteral::Float { text },
        } => Arc::new(StringArray::from(vec![text.as_str()])),
        NodeKind::Constant {
            literal: PqlLiteral::String { value },
        } => Arc::new(StringArray::from(vec![value.as_str()])),
        NodeKind::Constant {
            literal: PqlLiteral::Bool { value },
        } => Arc::new(BooleanArray::from(vec![*value])),
        _ => {
            return Err(invalid_option(
                "query",
                "unsupported forecast entity literal",
            ));
        }
    };
    keys::cast(&values, dtype)
}

fn target_type(asked: &Asked, graph: &Graph) -> Result<DataType, CoreError> {
    Ok(match &asked.target {
        Shape::Links(window) => {
            let dtype = graph
                .table(&window.table)
                .and_then(|t| window.column.as_ref().and_then(|c| t.column(c)))
                .ok_or_else(|| internal("link target column absent"))?
                .received
                .to_arrow();
            DataType::List(Arc::new(ArrowField::new("item", dtype, false)))
        }
        Shape::Condition { .. } | Shape::Both { .. } | Shape::Not(_) => DataType::Boolean,
        Shape::Column(column) => {
            let dtype = graph
                .table(&asked.entity_table)
                .and_then(|table| table.column(column))
                .ok_or_else(|| internal("target column absent"))?
                .received
                .to_arrow();
            // Ask the actual target conversion about an empty column. This also
            // refuses unsupported target storage without sampling any rows.
            let target =
                labels::Target::Column(static_column(&arrow_array::new_empty_array(&dtype), &[])?);
            let labelled = if asked.task == Task::Multiclass {
                labels::label_classes(&target)?
            } else {
                labels::label(&target)?
            };
            target_array(PreparedLabels::Scalar(labelled.labels), &DataType::Null)?
                .data_type()
                .clone()
        }
        _ => DataType::Float32,
    })
}

fn target_array(labels: PreparedLabels, dtype: &DataType) -> Result<ArrayRef, CoreError> {
    let array: ArrayRef = match labels {
        PreparedLabels::Scalar(labels::Labels::Float(v)) => Arc::new(Float32Array::from(v)),
        PreparedLabels::Scalar(labels::Labels::Int(v)) => Arc::new(Int64Array::from(v)),
        PreparedLabels::Scalar(labels::Labels::Bool(v)) => Arc::new(BooleanArray::from(v)),
        PreparedLabels::Scalar(labels::Labels::Text(v)) => Arc::new(StringArray::from(v)),
        PreparedLabels::Links(groups) => {
            use sdm_engine::frame_labels::LinkValue;
            let DataType::List(field) = dtype else {
                return Err(internal("link target type"));
            };
            let mut offsets = vec![0_i32];
            let mut values = Vec::new();
            for group in groups {
                values.extend(group);
                offsets.push(
                    i32::try_from(values.len())
                        .map_err(|_| invalid_option("size", "too many link values"))?,
                );
            }
            let flat: ArrayRef = match values.first() {
                None => arrow_array::new_empty_array(field.data_type()),
                Some(LinkValue::Int(_)) => Arc::new(Int64Array::from(
                    values
                        .iter()
                        .map(|v| {
                            if let LinkValue::Int(v) = v {
                                *v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Some(LinkValue::UInt(_)) => Arc::new(UInt64Array::from(
                    values
                        .iter()
                        .map(|v| {
                            if let LinkValue::UInt(v) = v {
                                *v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Some(LinkValue::Float(_)) => Arc::new(Float64Array::from(
                    values
                        .iter()
                        .map(|v| {
                            if let LinkValue::Float(v) = v {
                                *v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Some(LinkValue::Bool(_)) => Arc::new(BooleanArray::from(
                    values
                        .iter()
                        .map(|v| {
                            if let LinkValue::Bool(v) = v {
                                *v
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Some(LinkValue::Text(_)) => Arc::new(StringArray::from(
                    values
                        .iter()
                        .map(|v| {
                            if let LinkValue::Text(v) = v {
                                v.as_str()
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Some(LinkValue::Decimal(_, scale)) => {
                    let DataType::Decimal128(precision, _) = field.data_type() else {
                        return Err(internal("decimal link storage"));
                    };
                    Arc::new(
                        arrow_array::Decimal128Array::from(
                            values
                                .iter()
                                .map(|v| {
                                    if let LinkValue::Decimal(v, _) = v {
                                        *v
                                    } else {
                                        unreachable!()
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                        .with_precision_and_scale(*precision, *scale)
                        .map_err(|_| internal("decimal link precision"))?,
                    )
                }
            };
            let flat = arrow_cast::cast(&flat, field.data_type())
                .map_err(|_| internal("link label type changed"))?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(field),
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    flat,
                    None,
                )
                .map_err(|_| internal("link label offsets"))?,
            )
        }
    };
    Ok(array)
}

fn result(keys: ArrayRef, anchors: Vec<i64>, target: ArrayRef) -> Result<ArrowData, CoreError> {
    let batch = RecordBatch::try_from_iter([
        ("ENTITY", keys),
        (
            "ANCHOR_TIMESTAMP",
            Arc::new(TimestampNanosecondArray::from(anchors)) as ArrayRef,
        ),
        ("TARGET", target),
    ])
    .map_err(|_| internal("label columns do not align"))?;
    Ok(ArrowData::new(batch.schema(), vec![batch]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_target_storage_is_refused_for_empty_and_populated_columns() {
        let dtype = DataType::Timestamp(ArrowUnit::Nanosecond, None);
        assert!(static_column(&arrow_array::new_empty_array(&dtype), &[]).is_err());
        let populated: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![0]));
        assert!(static_column(&populated, &[0]).is_err());
    }

    #[test]
    fn integer_ground_truth_does_not_round_through_float() {
        let exact = 16_777_217;
        let output = target_array(
            PreparedLabels::Scalar(labels::Labels::Int(vec![exact])),
            &DataType::Int64,
        )
        .unwrap();
        assert_eq!(output.as_primitive::<Int64Type>().value(0), exact);
    }

    #[test]
    fn link_labels_keep_identifier_values_and_empty_lists() {
        use sdm_engine::frame_labels::LinkValue;
        let dtype = DataType::List(Arc::new(ArrowField::new("item", DataType::Int64, false)));
        let output = target_array(
            PreparedLabels::Links(vec![vec![LinkValue::Int(7), LinkValue::Int(12)], vec![]]),
            &dtype,
        )
        .unwrap();
        let lists = output.as_list::<i32>();
        assert_eq!(lists.value_offsets(), &[0, 2, 2]);
        assert_eq!(
            lists.values().as_primitive::<Int64Type>().values().as_ref(),
            &[7, 12]
        );
        assert_eq!(lists.null_count(), 0);
    }

    #[test]
    fn empty_output_keeps_three_typed_columns() {
        let output = result(
            arrow_array::new_empty_array(&DataType::Utf8),
            vec![],
            arrow_array::new_empty_array(&DataType::Boolean),
        )
        .unwrap();
        assert_eq!(
            output
                .schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            ["ENTITY", "ANCHOR_TIMESTAMP", "TARGET"]
        );
        assert_eq!(output.schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(
            output.schema.field(1).data_type(),
            &DataType::Timestamp(ArrowUnit::Nanosecond, None)
        );
        assert_eq!(output.schema.field(2).data_type(), &DataType::Boolean);
    }
}
