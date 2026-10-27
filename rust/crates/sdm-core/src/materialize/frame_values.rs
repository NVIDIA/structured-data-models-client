// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Domain consumers of fixed, independently useful frame materialization values.
//! Shared source wiring is staged here for the complete frame composition.
//! This source candidate has no execution or validation evidence.
//! Graph/task/request policy stays here; native owners and I/O stay in the host.

mod columns;
mod lags;
mod native_guard;
mod owners;
pub(crate) mod questions;
pub use owners::{RetainedFramePayload, RetainedFrameReferences, RetainedFrameTask};
pub use questions::RetainedFrameQuestion;
#[cfg(test)]
mod native_guard_tests;

use arrow_array::UInt64Array;
use sdm_contract::frames::materialize::*;

use super::*;
use crate::attached::{take_json, take_json_if};
use crate::frame_materialize::identity::{Identity, ordered_coverage};
use crate::frame_materialize::values::{
    BoundGraph, attach_array, attach_batch, attach_json, positions, take_array, take_arrow,
    take_strings,
};
use crate::run::Attachments;

/// Native failure facts retain the actual reached operation and ancestry.
/// The final frame guard consumes this inside the core error boundary; this
/// is never a request for a host action or a host-selected error policy.
pub(crate) enum FrameStageError {
    Core(CoreError),
    Native {
        site: NativeSite,
        failure: FrameNativeFailure,
        diagnostic: Option<NativeConstructionDiagnostic>,
        dropped_summary: Option<String>,
    },
}

pub(crate) enum NativeSite {
    Prediction {
        table: String,
    },
    WholeReferences,
    BatchReferences,
    BatchReferenceCast {
        expected_dtype: String,
        input_repr: String,
    },
    Text(ColumnRef),
}

impl From<CoreError> for FrameStageError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl FrameStageError {
    fn into_core(self) -> CoreError {
        match self {
            Self::Core(error) => error,
            Self::Native {
                site,
                failure,
                diagnostic,
                dropped_summary,
            } => {
                if matches!(site, NativeSite::Prediction { .. }) {
                    let eligible = failure.builtin.ancestry.iter().any(|kind| {
                        matches!(
                            kind,
                            sdm_contract::prepare::BuiltinFailureClass::TypeError
                                | sdm_contract::prepare::BuiltinFailureClass::ValueError
                        )
                    });
                    if eligible
                        == matches!(
                            diagnostic,
                            Some(NativeConstructionDiagnostic::NotReached) | None
                        )
                    {
                        return protocol(
                            "native constructor failure and reached diagnostic do not agree",
                        );
                    }
                }
                let constructor_message = match (&site, &diagnostic) {
                    (
                        NativeSite::Prediction { table },
                        Some(NativeConstructionDiagnostic::Completed {
                            first,
                            dtype_display,
                        }),
                    ) => {
                        let eligible = failure.builtin.ancestry.iter().any(|kind| {
                            matches!(
                                kind,
                                sdm_contract::prepare::BuiltinFailureClass::TypeError
                                    | sdm_contract::prepare::BuiltinFailureClass::ValueError
                            )
                        });
                        if !eligible {
                            return protocol(
                                "a native constructor diagnostic followed an ineligible failure",
                            );
                        }
                        let where_ = first
                            .as_ref()
                            .map(|first| {
                                format!("; indices[{}] is {}", first.ordinal, first.native_repr)
                            })
                            .unwrap_or_default();
                        Some(format!(
                            "'indices' must hold values matching the primary key of '{table}', which is {dtype_display}{where_}"
                        ))
                    }
                    (
                        NativeSite::BatchReferenceCast {
                            expected_dtype,
                            input_repr,
                        },
                        _,
                    ) => {
                        let eligible = failure.builtin.ancestry.iter().any(|kind| {
                            matches!(
                                kind,
                                sdm_contract::prepare::BuiltinFailureClass::TypeError
                                    | sdm_contract::prepare::BuiltinFailureClass::ValueError
                            )
                        });
                        if !eligible {
                            return protocol(
                                "a native reference retry diagnostic followed an ineligible failure",
                            );
                        }
                        Some(format!(
                            "Could not cast primary keys {input_repr} to the expected data type '{expected_dtype}'"
                        ))
                    }
                    _ => None,
                };
                let (stage, column) = match site {
                    NativeSite::Prediction { .. } => ("prediction", None),
                    NativeSite::WholeReferences => ("whole_references", None),
                    NativeSite::BatchReferences | NativeSite::BatchReferenceCast { .. } => {
                        ("batch_references", None)
                    }
                    NativeSite::Text(column) => ("text", Some(column)),
                };
                let mut error = CoreError::new(
                    Sentence::FrameNativeFailure,
                    "reached native frame conversion failed",
                )
                .with_field("stage", Detail::text(stage))
                .with_field(
                    "failure",
                    Detail::text(
                        serde_json::to_string(&failure).expect("native failure facts serialize"),
                    ),
                );
                if let Some(message) = constructor_message {
                    error = error
                        .with_field("native_constructor_kind", Detail::text("value_error"))
                        .with_field("native_constructor_message", Detail::text(message));
                }
                if let Some(diagnostic) = diagnostic {
                    error = error.with_field(
                        "diagnostic",
                        Detail::text(
                            serde_json::to_string(&diagnostic)
                                .expect("native diagnostic facts serialize"),
                        ),
                    );
                }
                if let Some(summary) = dropped_summary {
                    error = error.with_field("dropped_summary", Detail::text(summary));
                }
                if let Some(column) = column {
                    error = error
                        .with_field("table", Detail::text(column.table))
                        .with_field("column", Detail::text(column.column));
                }
                error
            }
        }
    }
}

/// Preserve an already constructed graph even when a HostRun is cancelled
/// before its first poll. This captures a resource, not a program position.
pub(crate) fn carry(asker: &Asker, input: &FramePreparation, attachments: &Attachments) {
    if let Some(owner) = input.retained_owner().and_then(|id| attachments.get(&id)) {
        if let Attachment::FrameGraph(owner) = owner {
            asker.keep_retained_frame(owner.clone());
        }
    }
}

pub(crate) async fn run(
    asker: Asker,
    input: FramePreparation,
    mut attachments: Attachments,
    facts: HostFacts,
    zones: Arc<dyn sdm_engine::time::Zones>,
    part_scope: Option<FrameQuestionScope>,
) -> Result<OperationResult, CoreError> {
    let value = match input {
        FramePreparation::ColumnMeaning { input } => {
            PreparedFrameValue::ColumnMeaning(columns::meaning(input))
        }
        FramePreparation::IntegerColumn { input } => {
            PreparedFrameValue::IntegerColumn(columns::integer(input)?)
        }
        FramePreparation::TimeColumn { meaning, values } => PreparedFrameValue::SourceColumn(
            columns::time(&asker, meaning, values, &facts, &mut attachments)?,
        ),
        FramePreparation::Question {
            schema,
            query,
            candidates_given,
            statistic_given,
            top_k_given,
        } => PreparedFrameValue::Question(questions::resolve(
            &asker,
            schema,
            query,
            candidates_given,
            statistic_given,
            top_k_given,
            &facts,
            &*zones,
            &mut attachments,
        )?),
        FramePreparation::Questions { input } => {
            PreparedFrameValue::Questions(questions::prepare(&asker, input, &mut attachments)?)
        }
        input @ FramePreparation::Graph { .. } => {
            PreparedFrameValue::Graph(graph(&asker, input, &mut attachments)?)
        }
        input @ FramePreparation::Labels { .. } => PreparedFrameValue::Labels(
            labels(&asker, input, &facts, &*zones, &mut attachments).await?,
        ),
        FramePreparation::Task {
            labelled,
            prediction,
            values,
        } => PreparedFrameValue::Task(
            task(&asker, labelled, prediction, values, &mut attachments)
                .map_err(FrameStageError::into_core)?,
        ),
        FramePreparation::Plan { task, values } => {
            PreparedFrameValue::Plan(plan(&asker, task, values, &mut attachments)?)
        }
        FramePreparation::References { plan, converted } => PreparedFrameValue::References(
            references(&asker, plan, converted, &mut attachments)
                .map_err(FrameStageError::into_core)?,
        ),
        FramePreparation::Context {
            references,
            range,
            converted,
        } => PreparedFrameValue::Context(
            context(&asker, references, range, converted, &mut attachments)
                .map_err(FrameStageError::into_core)?,
        ),
        FramePreparation::Payload {
            context,
            text,
            values,
        } => {
            let next = payload_part(&asker, context, text, values, &mut attachments)
                .map_err(FrameStageError::into_core)?;
            PreparedFrameValue::Part(publish_part(&asker, next)?)
        }
        FramePreparation::Request { plan, parts } => {
            finish_parts(&asker, plan, parts, &mut attachments)?
        }
        FramePreparation::AssessFailure {
            origin,
            failure,
            signature,
            window_outran,
        } => PreparedFrameValue::FailureAssessment(native_guard::assess_in_part(
            origin,
            failure,
            signature,
            part_scope.as_ref(),
            window_outran,
        )?),
        FramePreparation::ReportFailure { report } => return Err(native_guard::report(report)?),
    };
    Ok(OperationResult::PrepareFrames(PreparedFrames {
        value: attach_json(&asker, &value)?,
    }))
}

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn mode(value: FrameRunMode) -> options::RunMode {
    match value {
        FrameRunMode::Fast => options::RunMode::Fast,
        FrameRunMode::Best => options::RunMode::Best,
    }
}

fn frame_mode(value: options::RunMode) -> FrameRunMode {
    match value {
        options::RunMode::Fast => FrameRunMode::Fast,
        options::RunMode::Best => FrameRunMode::Best,
    }
}

/// Local reads feed the existing Labelling builder and aggregation kernels.
/// Broad native seconds select the same ordered candidate rows as the
/// original query hop; exact calendar/nanosecond predicates stay in the
/// existing aggregate_target implementation. No neighbor reset/draw occurs.
pub(super) fn label_rows(
    labelling: &Labelling<'_>,
    frames: &crate::frame_materialize::Context,
    fetches: &[Fetch<'_>],
) -> Result<BTreeMap<String, Fetched>, CoreError> {
    let keys = take(labelling.entity_key, &labelling.examples.index)?;
    let mut fetched = BTreeMap::new();
    for fetch in fetches {
        let windows = label_windows(&labelling.examples.anchors, fetch.start, fetch.end);
        let columns = fetch.table.ordered(&fetch.columns);
        let selected = crate::frame_materialize::lock(frames).temporal_rows(
            &fetch.edge,
            &keys,
            &windows,
            &columns,
        )?;
        let time = fetch
            .table
            .time_column
            .as_deref()
            .ok_or_else(|| internal("the frame label table has no time column"))?;
        fetched.insert(
            fetch.table.name.clone(),
            Fetched {
                strings: crate::frame_materialize::lock(frames)
                    .string_encodings(&fetch.table.name)?,
                times: times_ns(named(&selected.rows, time)?, time)?,
                rows: selected.rows,
                batch: selected.batch,
            },
        );
    }
    Ok(fetched)
}

fn label_windows(
    anchors: &[i64],
    start: Offset,
    end: Offset,
) -> Vec<sdm_engine::frames::TemporalWindow> {
    let lower = offset_seconds(start, false);
    let upper = offset_seconds(end, true);
    anchors
        .iter()
        .map(|anchor| {
            let second = i128::from(anchor.div_euclid(1_000_000_000));
            sdm_engine::frames::TemporalWindow {
                lower: Some(second + lower),
                upper: second + upper,
            }
        })
        .collect()
}

fn one_column(data: &ArrowData) -> Result<ArrayRef, CoreError> {
    let rows = one_batch(data)?;
    if rows.num_columns() != 1 {
        return Err(protocol("frame task input is not one column"));
    }
    Ok(Arc::clone(rows.column(0)))
}

fn data(batch: RecordBatch) -> ArrowData {
    ArrowData::new(batch.schema(), vec![batch])
}

fn labelled_rows(
    keys: &ArrayRef,
    anchors: &[i64],
    labels: &labels::Labels,
) -> Result<RecordBatch, CoreError> {
    use arrow_array::{
        BooleanArray, Decimal128Array, Float32Array, Int64Array, ListArray, StringArray,
    };
    use sdm_engine::labels::LinkValue;
    let labels::Labels::Lists(lists) = labels else {
        return task::labelled_rows(keys, anchors, labels);
    };
    let mut offsets = vec![0_i32];
    let mut flat = Vec::new();
    for list in lists {
        flat.extend(list.iter());
        offsets.push(i32::try_from(flat.len()).map_err(|_| {
            internal("link label offsets exceed the declared Arrow representation")
        })?);
    }
    let mixed = || protocol("typed link labels mix incompatible native value families");
    let values: ArrayRef = match flat.first() {
        None | Some(LinkValue::Int(_)) => Arc::new(Int64Array::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Int(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Float(_)) => Arc::new(Float32Array::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Float(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Bool(_)) => Arc::new(BooleanArray::from(
            flat.iter()
                .map(|value| match value {
                    LinkValue::Bool(value) => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Some(LinkValue::Text(_)) => {
            sdm_engine::strings::encode(
                &flat
                    .iter()
                    .map(|value| match value {
                        LinkValue::Text(value) => Ok(value.clone()),
                        _ => Err(mixed()),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .0
        }
        Some(LinkValue::Decimal(_, scale)) => {
            let scale = *scale;
            let values = flat
                .iter()
                .map(|value| match value {
                    LinkValue::Decimal(value, observed) if *observed == scale => Ok(*value),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(38, scale)
                    .map_err(|_| mixed())?,
            )
        }
    };
    let target: ArrayRef = Arc::new(
        ListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                values.data_type().clone(),
                false,
            )),
            arrow_buffer::OffsetBuffer::new(offsets.into()),
            values,
            None,
        )
        .map_err(|_| mixed())?,
    );
    RecordBatch::try_from_iter([
        ("entity", Arc::clone(keys)),
        (
            "anchor",
            Arc::new(TimestampNanosecondArray::from(anchors.to_vec())) as ArrayRef,
        ),
        ("target", target),
    ])
    .map_err(|_| protocol("typed link labels do not align with their entity and anchor rows"))
}

/// Text representation belongs to the same typed labels, not to physical
/// Binary inference or a separately replayed host conversion.
fn label_encoding(labels: &labels::Labels) -> Option<sdm_contract::data::strings::StringEncoding> {
    use sdm_contract::data::strings::StringEncoding;
    let exceptional = match labels {
        labels::Labels::Text(values) => values.iter().any(|value| value.utf8().is_none()),
        labels::Labels::Lists(values)
            if values
                .iter()
                .flatten()
                .any(|value| matches!(value, labels::LinkValue::Text(_))) =>
        {
            values.iter().flatten().any(
                |value| matches!(value, labels::LinkValue::Text(text) if text.utf8().is_none()),
            )
        }
        _ => return None,
    };
    Some(if exceptional {
        StringEncoding::Utf8Surrogatepass
    } else {
        StringEncoding::Utf8
    })
}

fn read_link_labels(
    target: &ArrayRef,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
) -> Result<labels::Labels, CoreError> {
    use sdm_engine::labels::LinkValue;
    let target = target
        .as_list_opt::<i32>()
        .ok_or_else(|| protocol("typed frame link labels are not a declared Arrow list"))?;
    let lists = (0..target.len())
        .map(|row| {
            let values = target.value(row);
            if values.null_count() != 0 {
                return Err(protocol(
                    "typed frame link labels contain missing child values",
                ));
            }
            Ok(match values.data_type() {
                _ if encoding.is_some() => sdm_engine::strings::owned(&values, encoding.unwrap())
                    .map_err(|_| protocol("frame link text does not match its explicit encoding"))?
                    .into_iter()
                    .map(|value| {
                        value
                            .map(LinkValue::Text)
                            .ok_or_else(|| protocol("frame link text contains missing values"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                DataType::Int64 => values
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied()
                    .map(LinkValue::Int)
                    .collect(),
                DataType::Float32 => values
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .copied()
                    .map(LinkValue::Float)
                    .collect(),
                DataType::Boolean => values
                    .as_boolean()
                    .values()
                    .iter()
                    .map(LinkValue::Bool)
                    .collect(),
                DataType::Utf8 => values
                    .as_string::<i32>()
                    .iter()
                    .map(|value| LinkValue::Text(value.unwrap().into()))
                    .collect(),
                DataType::LargeUtf8 => values
                    .as_string::<i64>()
                    .iter()
                    .map(|value| LinkValue::Text(value.unwrap().into()))
                    .collect(),
                DataType::Decimal128(_, scale) => values
                    .as_primitive::<Decimal128Type>()
                    .values()
                    .iter()
                    .map(|&value| LinkValue::Decimal(value, *scale))
                    .collect(),
                _ => {
                    return Err(protocol(
                        "typed frame link labels contain an incompatible value family",
                    ));
                }
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(labels::Labels::Lists(lists))
}

/// Existing early call guards shared by metadata-only preflight and graph
/// construction. They run before native frame export on the public facade.
fn check_call(
    values: Option<&Value>,
    option_names: &[String],
    prediction_rows: u64,
    limits: &FrameLimits,
) -> Result<(), CoreError> {
    let asked = options::values(values)?;
    options::run_mode(
        asked.num_context_samples,
        asked.num_context_samples_given.as_deref(),
    )?;
    options::hops(asked.num_hops, asked.num_neighbors)?;
    if !option_names.is_empty() {
        return Err(invalid(
            Sentence::OptionsWithoutValues,
            "an option named is not one of the sampling options",
            "options",
        ));
    }
    if prediction_rows == 0 {
        return Err(invalid(
            Sentence::IndicesEmpty,
            "the entities to predict for are none",
            "indices",
        )
        .with_field("reason", Detail::text("empty")));
    }
    refuse_entities(prediction_rows, 1, limits.entities)?;
    options::statistic(asked.prediction_statistic.as_deref())?;
    Ok(())
}

/// The initial graph call owns the actual early checks and immutable snapshot.
/// Its caller presents the separate drop warning after this value is retained.
pub(crate) fn graph(
    asker: &Asker,
    input: FramePreparation,
    attachments: &mut Attachments,
) -> Result<PreparedFrameGraph, CoreError> {
    let FramePreparation::Graph {
        schema,
        frames,
        checkpoint,
        retained,
        native_tables: native,
        values,
        options: option_names,
        prediction_rows,
        limits,
    } = input
    else {
        return Err(protocol("graph preparation requires graph source values"));
    };
    let values: Option<Value> = take_json_if(attachments, values)?;
    check_call(values.as_ref(), &option_names, prediction_rows, &limits)?;
    let held: sdm_contract::ops::HeldSchema = take_json(attachments, schema)?;
    if let Some(owner) = retained {
        if checkpoint.is_some() {
            return Err(protocol(
                "a retained frame graph cannot also reread a raw checkpoint",
            ));
        }
        let manifest: sdm_contract::frames::MaterializeFrames = take_json(attachments, frames)?;
        if !manifest.tables.is_empty() {
            return Err(protocol(
                "a retained frame graph cannot reread caller frames",
            ));
        }
        let graph = BoundGraph::take_retained(attachments, owner)?;
        if graph.native.as_ref() != &native {
            return Err(protocol(
                "retained frame graph no longer matches its native source facts",
            ));
        }
        let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
        let mut description = graph.prepared.description.clone();
        consistent(&mut description, &read.types)?;
        reserved(&description)?;
        kept_checks(&description)?;
        if description != graph.prepared.description {
            return Err(protocol(
                "retained frame graph no longer matches its immutable schema meaning",
            ));
        }
        let value = graph.publish(asker)?;
        crate::frame_materialize::lock(&graph.frames).warn_dropped(asker);
        return Ok(value);
    }
    let frames = crate::frame_materialize::take(frames, checkpoint, attachments)?;
    let (mut description, types) = crate::frame_materialize::description(&frames, held)?;
    consistent(&mut description, &types)?;
    reserved(&description)?;
    kept_checks(&description)?;
    let graph = BoundGraph::prepare(asker, frames, description, native)?;
    let value = graph.publish(asker)?;
    crate::frame_materialize::lock(&graph.frames).warn_dropped(asker);
    Ok(value)
}

fn prediction_type(
    graph: &BoundGraph,
    entity: &ColumnRef,
    keys: &ArrayRef,
) -> Result<FramePredictionType, CoreError> {
    let origin = graph.native_column(entity)?;
    if origin.strings.is_some() {
        Ok(FramePredictionType::NativeString {
            column: entity.clone(),
            native: graph.native_binding(entity)?,
        })
    } else {
        Ok(FramePredictionType::CoreKey {
            physical: ColumnType::from_arrow(&entity.column, keys.data_type())?,
        })
    }
}

fn task_identity(
    meaning: &FrameTaskMeaning,
    prediction: &FramePredictionType,
    target_encoding: Option<sdm_contract::data::strings::StringEncoding>,
    rows: &ArrowData,
) -> Result<String, CoreError> {
    let mut meaning = meaning.clone();
    meaning.identity.clear();
    let mut identity = Identity::new("labelled-task");
    identity.value(&meaning)?;
    identity.value(&(prediction, target_encoding))?;
    identity.arrow(rows)?;
    Ok(identity.finish())
}

/// The existing typed draws produce real labeled entity/anchor/target rows.
/// No prediction value constructor runs before this function finishes.
pub(crate) async fn labels(
    asker: &Asker,
    input: FramePreparation,
    facts: &HostFacts,
    zones: &dyn sdm_engine::time::Zones,
    attachments: &mut Attachments,
) -> Result<LabelledFrameTask, CoreError> {
    let FramePreparation::Labels {
        graph: graph_value,
        query,
        values,
        seed,
        entropy,
        anchor_time,
        anchor_entity,
        num_context_samples,
        limits,
    } = input
    else {
        return Err(protocol("label preparation requires labelled task inputs"));
    };
    let mut graph = BoundGraph::bind(asker, graph_value, attachments)?;
    // Snapshot/index ownership is immutable. Labels use call-local entity
    // entropy, not the shared neighbor stream, and shared temporal kernels
    // can suspend. Actual FrameRun accesses remain individually locked.
    graph.release_access();
    let query: String = take_json(attachments, query)?;
    let values: Option<Value> = take_json_if(attachments, values)?;
    let mut deferred = options::values(values.as_ref())?;
    if deferred.num_context_samples.is_none() {
        deferred.num_context_samples = num_context_samples.map(|count| options::Integer {
            negative: false,
            magnitude: Some(u128::from(count)),
        });
    }
    let run_mode = options::run_mode(
        deferred.num_context_samples,
        deferred.num_context_samples_given.as_deref(),
    )?;
    let requested_context = deferred
        .num_context_samples
        .and_then(|count| count.magnitude)
        .and_then(|count| u64::try_from(count).ok());
    let (seed, unseeded) = match deferred.seed {
        AskedSeed::Absent => (seed, false),
        AskedSeed::Unseeded => (None, true),
        AskedSeed::Given(ref given) => (
            Some(
                Seed::try_from(given.clone())
                    .map_err(|_| protocol("frame seed facts are invalid"))?,
            ),
            false,
        ),
    };
    let seed = match seed {
        Some(seed) => seed,
        None if unseeded => {
            Seed::from(u128::from(entropy.ok_or_else(|| {
                protocol("unseeded frame labels lack their OS entropy fact")
            })?))
        }
        None => Seed::from(DEFAULT_SEED),
    };
    if seed.is_negative() {
        return Err(
            CoreError::new(Sentence::SeedNegative, "a sampling seed cannot be negative")
                .with_field("field", Detail::text("seed"))
                .with_field("reason", Detail::text("negative"))
                .with_field("given", Detail::text(seed_text(Some(&seed)))),
        );
    }
    let seed_value = match seed_scalar(&seed) {
        Ok(value) => value,
        Err(_) if !seed.is_negative() => {
            return Err(crate::compose::past_seed_range(
                OptionalRequest::SampleEntities.name(),
                u32::MAX as u64,
            ));
        }
        Err(error) => return Err(error),
    };
    let entropy = Entropy::from_hex(seed.magnitude_hex())
        .ok_or_else(|| protocol("frame seed magnitude is invalid"))?;
    let resolved = resolve_target(&graph.prepared.description, &query, facts, zones)?;
    let asked = asked(&resolved.validated)?;
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    let query_schema = query_schema(&graph.prepared.description);
    let query_facts = crate::pql::resolve(&query, &query_schema, &profile, facts, zones)?
        .facts(&query_schema)
        .ok_or_else(|| protocol("resolved frame query lacks its facts"))?;
    let entity = graph
        .prepared
        .graph
        .table(&asked.entity_table)
        .ok_or_else(|| protocol("frame task names an absent entity table"))?;
    let key = entity
        .primary_key
        .as_ref()
        .filter(|key| **key == asked.entity_column)
        .ok_or_else(|| not_yet("entity_key"))?;
    let entity_ref = ColumnRef {
        table: entity.name.clone(),
        column: key.clone(),
    };
    let statistic = options::statistic(deferred.prediction_statistic.as_deref())?;
    options::statistic_for(asked.task, statistic.as_ref())?;
    let mut memo = Memo {
        asker,
        frames: Some(graph.frames.clone()),
        memo: SamplerMemo {
            source: graph.prepared.source.clone(),
            ..SamplerMemo::default()
        },
    };
    let times = Times::new(crate::describe::text_reader(facts)?, None, &memo.memo);
    let labelled_fetches = fetches(
        &graph.prepared.graph,
        &asked.entity_table,
        &asked.every_windowed(),
    )?;
    let mut windows = Vec::new();
    asked.target.windowed(&mut windows);
    let target_fetches = fetches(&graph.prepared.graph, &asked.entity_table, &windows)?;
    let bounded = if target_fetches.is_empty() {
        graph
            .prepared
            .graph
            .tables
            .iter()
            .filter(|table| table.time_column.is_some())
            .collect::<Vec<_>>()
    } else {
        target_fetches.iter().map(|fetch| fetch.table).collect()
    };
    let span = asked
        .range
        .as_ref()
        .map(|range| {
            Ok((
                Offset::of(
                    range
                        .start
                        .ok_or_else(|| unvalidated("a window with no start"))?,
                    range.unit,
                ),
                Offset::of(range.end, range.unit),
            ))
        })
        .transpose()?;
    let (context_given, max_iterations) = options::task_table(&deferred)?;
    let (prediction_anchor, context_anchor, earliest) = if anchor_entity {
        if span.is_some() {
            return Err(invalid(
                Sentence::AnchorEntityNotStatic,
                "an anchor at each entity's own time is for a static query only",
                "anchor_time",
            )
            .with_field("reason", Detail::text("entity_not_static")));
        }
        if entity.time_column.is_none() {
            return Err(invalid(
                Sentence::AnchorEntityNoTimeColumn,
                "an anchor at each entity's own time needs the entity table to have a time column",
                "anchor_time",
            )
            .with_field("reason", Detail::text("entity_without_time_column"))
            .with_field("table", Detail::text(&entity.name)));
        }
        if matches!(context_given, AskedAnchor::At(_)) {
            return Err(shared_anchor());
        }
        (TIME_MIN, TIME_MIN, None)
    } else {
        anchors_checked(
            asker,
            &mut memo,
            &times,
            &bounded,
            span.map(|(_, end)| end),
            anchor_time,
            &context_given,
        )
        .await?
    };
    let context_examples = run_mode.context_examples();
    let trials = max_iterations
        .unwrap_or(options::Iterations::Whole(ITERATIONS as u64))
        .trials(context_examples, span.is_some())
        .ok_or_else(|| {
            invalid(
                Sentence::IterationsWholeForWindow,
                "max_pq_iterations is a whole number for a query over a window",
                "max_pq_iterations",
            )
        })?;
    let sampled = entity_sample(
        asker,
        EntityDraw {
            entity,
            key,
            asked: &asked,
            read_times: &times,
            memo: &memo.memo,
            frames: Some(&graph.frames),
            trials,
            seed: &seed,
            seed_value,
            unseeded,
        },
    )
    .await?;
    let prediction_type = prediction_type(&graph, &entity_ref, &sampled.entity_key)?;
    let labelled = label_context(
        asker,
        &mut memo,
        LabelDraw {
            entity,
            key,
            asked: &asked,
            read_times: &times,
            labelled_fetches: &labelled_fetches,
            fetches: &target_fetches,
            windows: &windows,
            entropy: &entropy,
            context_anchor,
            anchor_entity,
            span,
            earliest,
            context_examples,
            trials,
            entity_filter: resolved.entity_filter.as_deref(),
        },
        sampled,
    )
    .await?;
    keep_stored_times(&mut memo, &times);
    let rows = data(labelled_rows(
        &labelled.keys,
        &labelled.anchors,
        &labelled.labels,
    )?);
    let mut meaning = FrameTaskMeaning {
        version: FRAME_MATERIALIZATION_VERSION,
        source: graph.source.clone(),
        identity: String::new(),
        query: resolved.validated,
        source_query: resolved.source_query,
        facts: query_facts,
        entity: entity_ref,
        mode: frame_mode(run_mode),
        seed,
        unseeded,
        prediction_anchor,
        entity_anchor: anchor_entity,
        no_time_marker: bounded.is_empty(),
        requested_context,
        statistic: deferred.prediction_statistic,
        limits,
    };
    let target_encoding = label_encoding(&labelled.labels);
    meaning.identity = task_identity(&meaning, &prediction_type, target_encoding, &rows)?;
    let context = asker.attach(Attachment::Arrow(rows))?;
    Ok(LabelledFrameTask {
        graph: graph.publish(asker)?,
        meaning,
        context: Some(context),
        target_encoding,
        prediction_type,
    })
}

struct BoundLabels {
    graph: BoundGraph,
    value: LabelledFrameTask,
    labelled: Arc<task::LabelledContext>,
    asked: Arc<Asked>,
    rows: ArrowData,
}

fn read_labels(
    rows: &RecordBatch,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
) -> Result<task::LabelledContext, CoreError> {
    if rows
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect::<Vec<_>>()
        != ["entity", "anchor", "target"]
    {
        return Err(protocol(
            "frame labeled context has an unexpected column layout",
        ));
    }
    let keys = Arc::clone(rows.column(0));
    let anchor = rows
        .column(1)
        .as_primitive_opt::<TimestampNanosecondType>()
        .filter(|array| array.null_count() == 0)
        .ok_or_else(|| protocol("frame labeled anchors are not non-null nanoseconds"))?;
    let target = rows.column(2);
    if target.null_count() != 0 {
        return Err(protocol("frame labeled targets contain a missing value"));
    }
    let labels = match target.data_type() {
        _ if encoding.is_some() && !matches!(target.data_type(), DataType::List(_)) => {
            labels::Labels::Text(
                sdm_engine::strings::owned(target, encoding.unwrap())
                    .map_err(|_| {
                        protocol("frame target text does not match its explicit encoding")
                    })?
                    .into_iter()
                    .map(|value| {
                        value.ok_or_else(|| protocol("frame target text contains a missing value"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        DataType::Float32 => {
            labels::Labels::Float(target.as_primitive::<Float32Type>().values().to_vec())
        }
        DataType::Int64 => {
            labels::Labels::Int(target.as_primitive::<Int64Type>().values().to_vec())
        }
        DataType::Boolean => labels::Labels::Bool(target.as_boolean().values().iter().collect()),
        DataType::Utf8 => labels::Labels::Text(
            target
                .as_string::<i32>()
                .iter()
                .map(|value| value.unwrap().into())
                .collect(),
        ),
        DataType::LargeUtf8 => labels::Labels::Text(
            target
                .as_string::<i64>()
                .iter()
                .map(|value| value.unwrap().into())
                .collect(),
        ),
        DataType::List(_) => read_link_labels(target, encoding)?,
        _ => {
            return Err(protocol(
                "frame labeled targets do not have a supported task representation",
            ));
        }
    };
    Ok(task::LabelledContext {
        keys,
        anchors: anchor.values().to_vec(),
        labels,
    })
}

fn bind_labels(
    asker: &Asker,
    value: LabelledFrameTask,
    attachments: &mut Attachments,
) -> Result<BoundLabels, CoreError> {
    let graph = BoundGraph::bind(asker, value.graph.clone(), attachments)?;
    let rows = take_arrow(
        attachments,
        value
            .context
            .ok_or_else(|| protocol("standalone labelled frame data lacks its actual rows"))?,
    )?;
    if value.meaning.version != FRAME_MATERIALIZATION_VERSION
        || value.meaning.source != graph.source
        || task_identity(
            &value.meaning,
            &value.prediction_type,
            value.target_encoding,
            &rows,
        )? != value.meaning.identity
    {
        return Err(protocol(
            "frame labeled task no longer matches its query, rows or anchors",
        ));
    }
    let labelled = read_labels(&one_batch(&rows)?, value.target_encoding)?;
    if prediction_type(&graph, &value.meaning.entity, &labelled.keys)? != value.prediction_type {
        return Err(protocol(
            "frame prediction type no longer matches its original entity dtype",
        ));
    }
    let asked = asked(&value.meaning.query)?;
    if asked.entity_table != value.meaning.entity.table
        || asked.entity_column != value.meaning.entity.column
    {
        return Err(protocol(
            "frame task entity no longer matches its resolved question",
        ));
    }
    Ok(BoundLabels {
        graph,
        value,
        labelled: Arc::new(labelled),
        asked: Arc::new(asked),
        rows,
    })
}

fn prediction_values(
    kind: &FramePredictionType,
    values: FramePredictionValues,
    attachments: &mut Attachments,
    target: &ArrayRef,
    graph: &BoundGraph,
) -> Result<
    (
        ArrayRef,
        Option<sdm_contract::data::strings::StringEncoding>,
    ),
    FrameStageError,
> {
    match (kind, values) {
        (FramePredictionType::CoreKey { .. }, FramePredictionValues::CoreKeys { values }) => Ok((
            keys::cast(&take_array(attachments, values)?, target.data_type())?,
            None,
        )),
        (
            FramePredictionType::NativeString { .. },
            FramePredictionValues::NativeString {
                constructed: NativeFrameConstruction::Value { values },
            },
        ) => Ok((take_strings(attachments, &values)?, Some(values.encoding))),
        (
            FramePredictionType::NativeString { column, .. },
            FramePredictionValues::NativeString {
                constructed:
                    NativeFrameConstruction::Failed {
                        failure,
                        diagnostic,
                    },
            },
        ) => Err(FrameStageError::Native {
            site: NativeSite::Prediction {
                table: column.table.clone(),
            },
            failure,
            diagnostic: Some(diagnostic),
            dropped_summary: graph.dropped_summary(),
        }),
        _ => Err(
            protocol("frame prediction data does not match its resolved constructor type").into(),
        ),
    }
}

/// Narrow only after actual native whole-constructor success. The returned
/// context and prediction arrays are the task data consumed by later stages.
pub(crate) fn task(
    asker: &Asker,
    labelled: LabelledFrameTask,
    prediction: FramePredictionValues,
    values: Option<AttachmentId>,
    attachments: &mut Attachments,
) -> Result<CompletedFrameTask, FrameStageError> {
    let bound = bind_labels(asker, labelled, attachments)?;
    let (predicted, prediction_encoding) = prediction_values(
        &bound.value.prediction_type,
        prediction,
        attachments,
        &bound.labelled.keys,
        &bound.graph,
    )?;
    if prediction_encoding.is_none() && keys::missing(&predicted) {
        return Err(invalid(
            Sentence::IndexMissing,
            "an entity to predict for is missing",
            "indices",
        )
        .into());
    }
    if predicted.is_empty() {
        return Err(invalid(
            Sentence::IndicesEmpty,
            "the entities to predict for are none",
            "indices",
        )
        .with_field("reason", Detail::text("empty"))
        .into());
    }
    let values: Option<Value> = take_json_if(attachments, values)?;
    let deferred = options::values(values.as_ref())?;
    let lag_steps = deferred
        .lag_timesteps
        .map(|lag| {
            lag.refusal(
                bound.asked.range.is_some() && bound.asked.task != options::Kind::TemporalLink,
            )
        })
        .transpose()?
        .unwrap_or(0);
    let lags = lags::FrameLags::prepare(&bound, &predicted, prediction_encoding, lag_steps)?;
    if bound.value.meaning.requested_context.is_none()
        && bound.labelled.keys.len() >= options::RunMode::Fast.context_examples()
    {
        asker.warn(CoreWarning {
            fields: [
                (
                    "cap".to_owned(),
                    count(options::RunMode::Fast.context_examples()),
                ),
                (
                    "most".to_owned(),
                    count(options::RunMode::Best.context_examples()),
                ),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::ContextCapped,
                "the default run mode gathered as many context examples as it takes",
            )
        });
    }
    let narrowed = bound.labelled.as_ref().clone().narrow(
        bound
            .value
            .meaning
            .requested_context
            .and_then(|rows| usize::try_from(rows).ok()),
    );
    let layout = BatchLayout::new(
        bound.asked.task,
        deferred.rows_per_call,
        predicted.len(),
        bound.value.meaning.limits.calls,
    )?;
    let rows_per_call = layout.ranges().next().map(|range| range.len()).unwrap_or(0) as u64;
    let calls = u32::try_from(layout.calls)
        .map_err(|_| protocol("frame task call count exceeds UInt32"))?;
    let mut labelled = bound.value;
    let rows = data(labelled_rows(
        &narrowed.keys,
        &narrowed.anchors,
        &narrowed.labels,
    )?);
    labelled.target_encoding = label_encoding(&narrowed.labels);
    labelled.meaning.identity = task_identity(
        &labelled.meaning,
        &labelled.prediction_type,
        labelled.target_encoding,
        &rows,
    )?;
    labelled.context = None;
    let prediction_rows = predicted.len() as u64;
    let predicted_data = data(
        RecordBatch::try_from_iter([("entity", Arc::clone(&predicted))])
            .map_err(|_| internal("frame prediction values do not form one column"))?,
    );
    if let Some(hops) = options::hops_ignored(&deferred) {
        asker.warn(CoreWarning {
            fields: [("num_hops".to_owned(), Detail::text(hops))].into(),
            ..CoreWarning::new(
                Sentence::HopsIgnored,
                "num_hops is ignored where num_neighbors is given",
            )
        });
    }
    let mut completed = CompletedFrameTask {
        identity: String::new(),
        labelled,
        prediction: None,
        owner: None,
        prediction_encoding,
        prediction_rows,
        rows_per_call,
        calls,
        lag_steps,
    };
    completed.identity = completed_identity(&completed, &predicted_data)?;
    let labels = BoundLabels {
        graph: bound.graph,
        value: completed.labelled.clone(),
        rows,
        labelled: Arc::new(narrowed),
        asked: bound.asked,
    };
    let rows = Arc::new(task_rows(
        &labels,
        &completed,
        one_column(&predicted_data)?,
    )?);
    BoundTask {
        labels,
        value: completed,
        predicted: predicted_data,
        rows,
        owner: None,
        lags,
    }
    .publish(asker)
    .map_err(FrameStageError::from)
}

fn completed_identity(
    value: &CompletedFrameTask,
    predicted: &ArrowData,
) -> Result<String, CoreError> {
    let mut identity = Identity::new("completed-task");
    identity.value(&(
        &value.labelled.meaning.identity,
        value.prediction_encoding,
        value.prediction_rows,
        value.rows_per_call,
        value.calls,
        value.lag_steps,
    ))?;
    identity.arrow(predicted)?;
    Ok(identity.finish())
}

fn align_task_keys(
    context: ArrayRef,
    context_encoding: sdm_contract::data::strings::StringEncoding,
    prediction: ArrayRef,
    prediction_encoding: sdm_contract::data::strings::StringEncoding,
) -> Result<(ArrayRef, ArrayRef), CoreError> {
    use sdm_contract::data::strings::StringEncoding;
    for (values, encoding) in [
        (&context, context_encoding),
        (&prediction, prediction_encoding),
    ] {
        sdm_engine::strings::StringColumn::new(values.as_ref(), encoding).map_err(|_| {
            protocol("native task keys do not match their explicit string encoding")
        })?;
    }
    if context.data_type() == prediction.data_type() {
        return Ok((context, prediction));
    }
    // This aligns physical Arrow offsets/byte views after the real native
    // constructor. It performs no language cast or logical key conversion.
    let physical = if context_encoding == StringEncoding::Utf8
        && prediction_encoding == StringEncoding::Utf8
    {
        DataType::LargeUtf8
    } else {
        DataType::LargeBinary
    };
    let context = arrow_cast::cast(&context, &physical).map_err(|_| {
        protocol("native context key byte buffers cannot share a physical Arrow representation")
    })?;
    let prediction = arrow_cast::cast(&prediction, &physical).map_err(|_| {
        protocol("native prediction key byte buffers cannot share a physical Arrow representation")
    })?;
    Ok((context, prediction))
}

struct BoundTask {
    labels: BoundLabels,
    value: CompletedFrameTask,
    predicted: ArrowData,
    rows: Arc<TaskRows>,
    owner: Option<RetainedFrameTask>,
    lags: Option<Arc<lags::FrameLags>>,
}

impl BoundTask {
    fn retained(&self) -> RetainedFrameTask {
        self.owner
            .clone()
            .unwrap_or_else(|| RetainedFrameTask::new(self))
    }
    fn publish(&self, asker: &Asker) -> Result<CompletedFrameTask, CoreError> {
        let mut value = self.value.clone();
        value.labelled.graph = self.labels.graph.publish(asker)?;
        value.labelled.context = None;
        value.prediction = None;
        value.owner = Some(asker.attach(Attachment::FrameTask(self.retained()))?);
        Ok(value)
    }

    fn entity(&self) -> Result<&Table, CoreError> {
        self.labels
            .graph
            .prepared
            .graph
            .table(&self.value.labelled.meaning.entity.table)
            .ok_or_else(|| protocol("completed frame task lacks its entity table"))
    }
}

fn bind_task(
    asker: &Asker,
    value: CompletedFrameTask,
    attachments: &mut Attachments,
) -> Result<BoundTask, CoreError> {
    if let Some(id) = value.owner {
        if value.prediction.is_some() || value.labelled.context.is_some() {
            return Err(protocol(
                "retained frame task cannot also declare raw replacement rows",
            ));
        }
        let Some(Attachment::FrameTask(owner)) = attachments.remove(&id) else {
            return Err(protocol(
                "frame task lacks its core-created immutable owner",
            ));
        };
        let graph = BoundGraph::bind(asker, value.labelled.graph.clone(), attachments)?;
        return owner.bind(graph, value);
    }
    if value.lag_steps != 0 {
        return Err(protocol(
            "raw completed lag data requires its actual core-created task owner",
        ));
    }
    let labels = bind_labels(asker, value.labelled.clone(), attachments)?;
    let predicted = take_arrow(
        attachments,
        value
            .prediction
            .ok_or_else(|| protocol("raw frame task lacks its prediction rows"))?,
    )?;
    let predicted_keys = one_column(&predicted)?;
    if predicted_keys.is_empty() || value.rows_per_call == 0 {
        return Err(protocol(
            "completed frame task lacks prediction rows or a positive call capacity",
        ));
    }
    if value.identity != completed_identity(&value, &predicted)?
        || value.prediction_rows != predicted_keys.len() as u64
    {
        return Err(protocol(
            "completed frame task no longer matches its prediction values",
        ));
    }
    let native = matches!(
        value.labelled.prediction_type,
        FramePredictionType::NativeString { .. }
    );
    if native != value.prediction_encoding.is_some() {
        return Err(protocol(
            "completed frame task lost its explicit native string meaning",
        ));
    }
    if let Some(encoding) = value.prediction_encoding {
        sdm_engine::strings::StringColumn::new(predicted_keys.as_ref(), encoding).map_err(
            |_| protocol("completed frame prediction encoding does not match its values"),
        )?;
    } else if predicted_keys.data_type() != labels.labelled.keys.data_type() {
        return Err(protocol(
            "completed numeric frame keys changed their core cast type",
        ));
    }
    let layout = BatchLayout::new(
        labels.asked.task,
        Some(value.rows_per_call),
        predicted_keys.len(),
        value.labelled.meaning.limits.calls,
    )?;
    if layout.calls as u64 != u64::from(value.calls) {
        return Err(protocol(
            "completed frame task no longer matches its core-selected batch layout",
        ));
    }
    let rows = Arc::new(task_rows(&labels, &value, predicted_keys)?);
    Ok(BoundTask {
        labels,
        value,
        predicted,
        rows,
        owner: None,
        lags: None,
    })
}

fn task_rows(
    labels: &BoundLabels,
    value: &CompletedFrameTask,
    predicted_keys: ArrayRef,
) -> Result<TaskRows, CoreError> {
    let (context_keys, predicted_keys) = if let Some(prediction_encoding) =
        value.prediction_encoding
    {
        let context_encoding = labels
            .graph
            .native_column(&value.labelled.meaning.entity)?
            .strings
            .ok_or_else(|| protocol("native task source key lost its explicit string encoding"))?;
        align_task_keys(
            Arc::clone(&labels.labelled.keys),
            context_encoding,
            predicted_keys,
            prediction_encoding,
        )?
    } else {
        (Arc::clone(&labels.labelled.keys), predicted_keys)
    };
    TaskRows::new(
        context_keys,
        labels.labelled.anchors.clone(),
        predicted_keys,
        value.labelled.meaning.prediction_anchor,
    )
}

fn static_exclusions(task: &BoundTask) -> Result<Vec<(String, String)>, CoreError> {
    let mut columns = BTreeSet::new();
    task.labels.asked.target.entity_columns(&mut columns);
    let entity = task.entity()?;
    let mut excluded = Vec::new();
    for column in columns {
        if !task
            .labels
            .graph
            .prepared
            .graph
            .features(entity)
            .iter()
            .any(|(name, _)| name == &column)
        {
            return Err(invalid(
                Sentence::TargetNotAFeature,
                "the column a static target reads is not one of its table's features",
                "query",
            ));
        }
        excluded.push((entity.name.clone(), column));
    }
    Ok(excluded)
}

fn selected_exclusions(
    task: &BoundTask,
    deferred: &options::Asked,
    check: bool,
) -> Result<Vec<ColumnRef>, CoreError> {
    let mut excluded = static_exclusions(task)?;
    let caller = options::excluded(deferred)?;
    if check {
        excluded_by_caller(&task.labels.graph.prepared.graph, &caller, &mut excluded)?;
    } else {
        // Preserve the reference's later validity check after whole-task node
        // resolution. The raw late values retain every malformed slot too.
        for (table, columns) in caller {
            for column in columns {
                if let Value::Text(column) = column {
                    if let Ok(column) = column.into_utf8() {
                        if !excluded.contains(&(table.clone(), column.clone())) {
                            excluded.push((table.clone(), column));
                        }
                    }
                }
            }
        }
    }
    Ok(excluded
        .into_iter()
        .map(|(table, column)| ColumnRef { table, column })
        .collect())
}

fn plan_identity(value: &FrameRequestPlan, sampling: &Value) -> Result<String, CoreError> {
    let mut identity = Identity::new("request-plan");
    identity.value(&(
        value.version,
        &value.task.identity,
        &value.inference,
        &value.excluded,
        value.top_k,
    ))?;
    identity.value(sampling)?;
    Ok(identity.finish())
}

pub(crate) fn plan(
    asker: &Asker,
    task: CompletedFrameTask,
    values: Option<AttachmentId>,
    attachments: &mut Attachments,
) -> Result<FrameRequestPlan, CoreError> {
    let task = bind_task(asker, task, attachments)?;
    let sampling: Value = take_json_if(attachments, values)?.unwrap_or(Value::Object(Vec::new()));
    let deferred = options::values(Some(&sampling))?;
    let top_k = deferred.top_k.as_ref().map(options::top_k).transpose()?;
    let inference = options::options(task.labels.asked.task, deferred.inference_config.as_ref())?;
    let excluded = selected_exclusions(&task, &deferred, false)?;
    if deferred.neighbors.is_none() {
        options::neighbors(&deferred, mode(task.value.labelled.meaning.mode))?;
    }
    let mut value = FrameRequestPlan {
        version: FRAME_MATERIALIZATION_VERSION,
        identity: String::new(),
        task: task.publish(asker)?,
        inference,
        excluded,
        top_k,
        sampling_values: attach_json(asker, &sampling)?,
    };
    value.identity = plan_identity(&value, &sampling)?;
    Ok(value)
}

struct BoundPlan {
    task: BoundTask,
    value: FrameRequestPlan,
    sampling: Value,
    deferred: options::Asked,
}

impl BoundPlan {
    fn publish(&self, asker: &Asker) -> Result<FrameRequestPlan, CoreError> {
        let mut value = self.value.clone();
        value.task = self.task.publish(asker)?;
        value.sampling_values = attach_json(asker, &self.sampling)?;
        Ok(value)
    }
}

fn bind_plan(
    asker: &Asker,
    value: FrameRequestPlan,
    attachments: &mut Attachments,
) -> Result<BoundPlan, CoreError> {
    let task = bind_task(asker, value.task.clone(), attachments)?;
    let sampling: Value = take_json(attachments, value.sampling_values)?;
    if value.version != FRAME_MATERIALIZATION_VERSION
        || plan_identity(&value, &sampling)? != value.identity
    {
        return Err(protocol(
            "frame request plan no longer matches its complete task and options",
        ));
    }
    let deferred = options::values(Some(&sampling))?;
    if selected_exclusions(&task, &deferred, false)? != value.excluded
        || options::options(task.labels.asked.task, deferred.inference_config.as_ref())?
            != value.inference
        || deferred.top_k.as_ref().map(options::top_k).transpose()? != value.top_k
    {
        return Err(protocol(
            "frame request plan no longer matches resolved request policy",
        ));
    }
    Ok(BoundPlan {
        task,
        value,
        sampling,
        deferred,
    })
}

fn converted_references(
    task: &BoundTask,
    values: FrameReferenceValues,
    site: NativeSite,
    attachments: &mut Attachments,
    expected: &ArrayRef,
) -> Result<
    (
        ArrayRef,
        Option<sdm_contract::data::strings::StringEncoding>,
    ),
    FrameStageError,
> {
    let (converted, encoding) = match (&task.value.labelled.prediction_type, values) {
        (FramePredictionType::CoreKey { .. }, FrameReferenceValues::CoreKeys { values }) => {
            let array = match values {
                None => Arc::clone(expected),
                Some(id) => {
                    let array = keys::cast(&take_array(attachments, id)?, expected.data_type())?;
                    if array.to_data() != expected.to_data() {
                        return Err(protocol(
                            "core frame reference values changed their ordered task keys",
                        )
                        .into());
                    }
                    array
                }
            };
            (array, None)
        }
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted: NativeFrameStrings::Value { values },
            },
        ) => (take_strings(attachments, &values)?, Some(values.encoding)),
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted: NativeFrameStrings::Failed { failure },
            },
        ) => {
            return Err(FrameStageError::Native {
                site,
                failure,
                diagnostic: None,
                dropped_summary: task.labels.graph.dropped_summary(),
            });
        }
        (
            FramePredictionType::NativeString { .. },
            FrameReferenceValues::NativeString {
                converted:
                    NativeFrameStrings::CastFailed {
                        failure,
                        expected_dtype,
                        input_repr,
                    },
            },
        ) => {
            if !matches!(&site, NativeSite::BatchReferences) {
                return Err(protocol(
                    "native index retry diagnostics cannot replace whole-task validation",
                )
                .into());
            }
            return Err(FrameStageError::Native {
                site: NativeSite::BatchReferenceCast {
                    expected_dtype,
                    input_repr,
                },
                failure,
                diagnostic: None,
                dropped_summary: task.labels.graph.dropped_summary(),
            });
        }
        _ => {
            return Err(protocol(
                "frame reference data does not match its resolved conversion family",
            )
            .into());
        }
    };
    if converted.len() != expected.len() {
        return Err(protocol(
            "converted frame references do not cover the actual ordered task rows",
        )
        .into());
    }
    Ok((converted, encoding))
}

fn reference_identity(plan: &FrameRequestPlan, nodes: &ArrayRef) -> Result<String, CoreError> {
    let mut identity = Identity::new("complete-references");
    identity.value(&plan.identity)?;
    identity.arrow(&data(
        RecordBatch::try_from_iter([("node", Arc::clone(nodes))])
            .map_err(|_| internal("reference rows do not make Arrow data"))?,
    ))?;
    Ok(identity.finish())
}

pub(crate) fn references(
    asker: &Asker,
    value: FrameRequestPlan,
    converted: FrameReferenceValues,
    attachments: &mut Attachments,
) -> Result<ResolvedFrameReferences, FrameStageError> {
    let bound = bind_plan(asker, value, attachments)?;
    let batch = bound
        .task
        .rows
        .batch(0..bound.task.value.prediction_rows as usize)?;
    let converted = converted_references(
        &bound.task,
        converted,
        NativeSite::WholeReferences,
        attachments,
        &batch.keys,
    )?;
    let nodes = crate::frame_materialize::lock(&bound.task.labels.graph.frames)
        .resolve_encoded_references(bound.task.entity()?, &[(&converted.0, converted.1)])?;
    selected_exclusions(&bound.task, &bound.deferred, true)?;
    if bound.task.labels.labelled.keys.len() > TABLE_ROWS {
        return Err(invalid(
            Sentence::ContextPastRows,
            "the context holds more rows than a request takes",
            "query",
        )
        .into());
    }
    let nodes: ArrayRef = Arc::new(UInt64Array::from(
        nodes.into_iter().map(|row| row as u64).collect::<Vec<_>>(),
    ));
    let identity = reference_identity(&bound.value, &nodes)?;
    let owner = RetainedFrameReferences::new(&bound, nodes, identity.clone());
    Ok(ResolvedFrameReferences {
        identity,
        plan: bound.publish(asker)?,
        owner: asker.attach(Attachment::FrameReferences(owner))?,
    })
}

fn check_range(range: FrameBatchRange, rows: u64, per_call: u64) -> Result<(), CoreError> {
    if per_call == 0
        || range.start >= rows
        || range.start % per_call != 0
        || range.stop != range.start + per_call.min(rows - range.start)
    {
        return Err(protocol(
            "sampled context names a range not selected by its complete task",
        ));
    }
    Ok(())
}

fn complete_reference_rows(
    plan: &BoundPlan,
    identity: &str,
    owner: AttachmentId,
    attachments: &mut Attachments,
) -> Result<usize, CoreError> {
    let Some(Attachment::FrameReferences(owner)) = attachments.remove(&owner) else {
        return Err(protocol(
            "complete frame references lack their immutable core owner",
        ));
    };
    owner.validate(plan, identity)
}

pub(crate) fn context(
    asker: &Asker,
    references: ResolvedFrameReferences,
    range: FrameBatchRange,
    converted: FrameReferenceValues,
    attachments: &mut Attachments,
) -> Result<SampledFrameContext, FrameStageError> {
    let bound = bind_plan(asker, references.plan, attachments)?;
    let node_count =
        complete_reference_rows(&bound, &references.identity, references.owner, attachments)?;
    let train = bound.task.labels.labelled.keys.len();
    if node_count != train + bound.task.value.prediction_rows as usize {
        return Err(protocol("whole-task node references do not cover every task row").into());
    }
    check_range(
        range,
        bound.task.value.prediction_rows,
        bound.task.value.rows_per_call,
    )?;
    let batch = bound
        .task
        .rows
        .batch(range.start as usize..range.stop as usize)?;
    let converted = converted_references(
        &bound.task,
        converted,
        NativeSite::BatchReferences,
        attachments,
        &batch.keys,
    )?;
    let entity = bound.task.entity()?;
    // Match LocalSampler::_sample_subgraph: actual batch get_node_id,
    // then anchor lookup/cast, then seed, then count admission and draws.
    // Neither the earlier complete references nor task construction resets
    // the continuing neighbor owner on behalf of this later native call.
    let resolved = crate::frame_materialize::lock(&bound.task.labels.graph.frames)
        .resolve_encoded_references(entity, &[(&converted.0, converted.1)])?;
    let mut anchors = batch.anchors;
    if bound.task.value.labelled.meaning.entity_anchor {
        let time = entity
            .time_column
            .as_ref()
            .ok_or_else(|| protocol("entity-anchored frame task lost its time column"))?;
        let looked_up =
            bound
                .task
                .labels
                .graph
                .selected_rows(&entity.name, &resolved, &[time.clone()])?;
        let times = times_ns(named(&looked_up, time)?, time)?;
        for (row, time) in times.into_iter().enumerate() {
            anchors[row] = time.unwrap_or(sample::MISSING_ANCHOR);
        }
    }
    let excluded = selected_exclusions(&bound.task, &bound.deferred, true)?
        .into_iter()
        .map(|column| (column.table, column.column))
        .collect::<Vec<_>>();
    let hops = if bound.deferred.neighbors.is_some() {
        Vec::new()
    } else {
        options::neighbors(
            &bound.deferred,
            mode(bound.task.value.labelled.meaning.mode),
        )?
    };
    let sampled = {
        let mut frames = crate::frame_materialize::lock(&bound.task.labels.graph.frames);
        frames.walk_rows(
            asker,
            &bound.task.labels.graph.prepared.graph,
            entity,
            &resolved,
            &anchors,
            crate::frame_materialize::FrameWalkOptions {
                hops: &hops,
                neighbor_values: bound.deferred.neighbors.as_deref(),
                excluded: &excluded,
                seed: (!bound.task.value.labelled.meaning.unseeded)
                    .then(|| bound.task.value.labelled.meaning.seed.to_u128())
                    .flatten(),
            },
        )?
    };
    let plan = bound.publish(asker)?;
    let anchors: ArrayRef = Arc::new(TimestampNanosecondArray::from(anchors));
    Ok(sampled.publish(asker, &bound.task.labels.graph, plan, range, anchors)?)
}

struct BoundContext {
    plan: BoundPlan,
    value: SampledFrameContext,
    walked: crate::subgraph::Walked,
    anchors: Vec<i64>,
}

fn bind_context(
    asker: &Asker,
    value: SampledFrameContext,
    attachments: &mut Attachments,
) -> Result<BoundContext, CoreError> {
    let plan = bind_plan(asker, value.plan.clone(), attachments)?;
    check_range(
        value.range,
        plan.task.value.prediction_rows,
        plan.task.value.rows_per_call,
    )?;
    if value.version != FRAME_MATERIALIZATION_VERSION {
        return Err(protocol("unknown sampled frame context version"));
    }
    let mut identity = Identity::new("sampled-context");
    identity.value(&(&plan.value.identity, value.range))?;
    let anchors = take_arrow(attachments, value.anchors)?;
    identity.arrow(&anchors)?;
    let anchors = one_column(&anchors)?;
    let anchors = anchors
        .as_primitive_opt::<TimestampNanosecondType>()
        .filter(|values| values.null_count() == 0)
        .ok_or_else(|| protocol("sampled frame anchors are not non-null nanoseconds"))?
        .values()
        .to_vec();
    let batch = plan
        .task
        .rows
        .batch(value.range.start as usize..value.range.stop as usize)?;
    if anchors.len() != batch.anchors.len()
        || (!plan.task.value.labelled.meaning.entity_anchor && anchors != batch.anchors)
    {
        return Err(protocol(
            "sampled frame context anchors do not align with its exact task batch",
        ));
    }
    let graph = &plan.task.labels.graph;
    let excluded = selected_exclusions(&plan.task, &plan.deferred, true)?
        .into_iter()
        .map(|column| (column.table, column.column))
        .collect::<Vec<_>>();
    let mut tables = Vec::new();
    let mut selected = BTreeMap::new();
    let mut text = Vec::new();
    for table in &value.tables {
        if selected.contains_key(&table.name) {
            return Err(protocol("sampled frame context repeats a table"));
        }
        let domain = graph
            .prepared
            .graph
            .table(&table.name)
            .ok_or_else(|| protocol("sampled frame context names an absent table"))?;
        let stypes = crate::subgraph::features(&graph.prepared.graph, domain, &excluded);
        let primary = if table.name == plan.task.value.labelled.meaning.entity.table {
            domain.primary_key.clone()
        } else {
            None
        };
        if table.source != domain.source || table.stypes != stypes || table.primary_key != primary {
            return Err(protocol(
                "sampled frame columns no longer match their graph and exclusions",
            ));
        }
        let rows = take_arrow(attachments, table.rows)?;
        let source_rows = take_arrow(attachments, table.source_rows)?;
        identity.value(&(
            &table.name,
            &table.source,
            &table.primary_key,
            &table.stypes,
        ))?;
        identity.arrow(&rows)?;
        identity.arrow(&source_rows)?;
        let rows = one_batch(&rows)?;
        let source_rows = positions(&one_column(&source_rows)?)?;
        let batch = positions(named(&rows, sdm_contract::data::BATCH_COLUMN)?)?;
        if source_rows.len() != rows.num_rows() || batch.iter().any(|&row| row >= anchors.len()) {
            return Err(protocol(
                "sampled frame rows do not align with source positions and task ownership",
            ));
        }
        let columns = rows
            .schema()
            .fields()
            .iter()
            .filter(|field| field.name() != sdm_contract::data::BATCH_COLUMN)
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        let expected_columns = domain
            .columns
            .iter()
            .filter(|column| {
                stypes.iter().any(|(name, _)| name == &column.name)
                    || primary.as_ref() == Some(&column.name)
            })
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if columns != expected_columns {
            return Err(protocol(
                "sampled frame raw columns changed their core selection order",
            ));
        }
        let expected = graph.selected_rows(&table.name, &source_rows, &columns)?;
        for (name, column) in columns.iter().zip(expected.columns()) {
            if named(&rows, name)?.to_data() != column.to_data() {
                return Err(protocol(
                    "sampled frame values no longer match selected immutable source positions",
                ));
            }
        }
        let columns = columns
            .into_iter()
            .map(|name| {
                Ok((
                    name.clone(),
                    match graph
                        .native_column(&ColumnRef {
                            table: table.name.clone(),
                            column: name.clone(),
                        })?
                        .strings
                    {
                        Some(encoding) => payload::SampledColumn::Strings(
                            Arc::clone(named(&rows, &name)?),
                            encoding,
                        ),
                        None => payload::SampledColumn::Arrow(Arc::clone(named(&rows, &name)?)),
                    },
                ))
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        for (column, stype) in &stypes {
            if *stype == Stype::Text {
                let logical = ColumnRef {
                    table: table.name.clone(),
                    column: column.clone(),
                };
                text.push(FrameTextColumn {
                    native: graph.native_binding(&logical)?,
                    column: logical,
                });
            }
        }
        selected.insert(table.name.clone(), (source_rows, batch.clone()));
        tables.push(payload::SampledTable {
            name: table.name.clone(),
            primary_key: primary,
            batch,
            columns,
            stypes,
        });
    }
    if !tables
        .iter()
        .any(|table| table.name == plan.task.value.labelled.meaning.entity.table)
    {
        return Err(protocol("sampled frame context lacks its entity table"));
    }
    let mut links = Vec::new();
    for link in &value.links {
        let edge = graph
            .prepared
            .graph
            .edges
            .iter()
            .find(|edge| {
                edge.src == link.source.table
                    && edge.fkey == link.source.column
                    && edge.dst == link.target.table
            })
            .ok_or_else(|| {
                protocol("sampled frame relationship is absent from its immutable graph")
            })?;
        if graph
            .prepared
            .graph
            .table(&edge.dst)
            .and_then(|table| table.primary_key.as_ref())
            != Some(&link.target.column)
        {
            return Err(protocol(
                "sampled frame relationship changed its target key",
            ));
        }
        let pairs = take_arrow(attachments, link.pairs)?;
        identity.value(&(&link.source, &link.target))?;
        identity.arrow(&pairs)?;
        let pairs = one_batch(&pairs)?;
        if pairs.num_columns() != 2 {
            return Err(protocol("sampled relationship pairs are not two columns"));
        }
        let source = positions(pairs.column(0))?;
        let target = positions(pairs.column(1))?;
        let pairs = source.into_iter().zip(target).collect::<Vec<_>>();
        let (source_rows, source_batch) = selected
            .get(&edge.src)
            .ok_or_else(|| protocol("sampled link source table is absent"))?;
        let (target_rows, target_batch) = selected
            .get(&edge.dst)
            .ok_or_else(|| protocol("sampled link target table is absent"))?;
        let expected = graph.selected_link_pairs(
            edge,
            source_rows,
            source_batch,
            target_rows,
            target_batch,
        )?;
        let expected = expected.into_iter().collect::<BTreeSet<_>>();
        if pairs.iter().any(|pair| !expected.contains(pair)) {
            return Err(protocol(
                "sampled relationship pairs no longer match source rows and example ownership",
            ));
        }
        links.push(payload::SampledLink {
            src: edge.src.clone(),
            fkey: edge.fkey.clone(),
            dst: edge.dst.clone(),
            pairs,
        });
    }
    identity.value(&(&value.text, &value.text_identity_input))?;
    if value.text != text
        || value.text_identity_input != sdm_engine::words::simple_identity_input()
        || identity.finish() != value.identity
    {
        return Err(protocol(
            "sampled frame context no longer matches its plan, range, rows, links or text order",
        ));
    }
    Ok(BoundContext {
        plan,
        value,
        walked: crate::subgraph::Walked { tables, links },
        anchors,
    })
}

fn casing_profile(facts: &NativeTextFacts) -> Result<sdm_engine::words::CasingProfile, CoreError> {
    use sdm_engine::words::{CasingProfile, UnicodeDataVersion};
    if facts.target == NativeStringTarget::PandasStringPyarrow {
        return facts.arrow_simple_identity.as_deref().and_then(CasingProfile::from_simple_identity)
            .ok_or_else(|| protocol("native Arrow lowercase facts do not identify a supported complete simple table"));
    }
    let version = match facts.python_unicode_database.as_str() {
        "13.0.0" => UnicodeDataVersion::V13_0,
        "14.0.0" => UnicodeDataVersion::V14_0,
        "15.0.0" => UnicodeDataVersion::V15_0,
        "15.1.0" => UnicodeDataVersion::V15_1,
        "16.0.0" => UnicodeDataVersion::V16_0,
        "17.0.0" => UnicodeDataVersion::V17_0,
        _ => {
            return Err(protocol(
                "native builtin lowercase facts name an unsupported Unicode database",
            ));
        }
    };
    Ok(CasingProfile::FullRoot(version))
}

fn check_text_order(
    expected: &[ColumnRef],
    converted: &[ColumnRef],
    failed: Option<&ColumnRef>,
) -> Result<(), CoreError> {
    if converted.len() > expected.len()
        || converted != &expected[..converted.len()]
        || match failed {
            Some(column) => expected.get(converted.len()) != Some(column),
            None => converted.len() != expected.len(),
        }
    {
        return Err(protocol(
            "native text outcomes do not follow core-selected whole-column conversion order",
        ));
    }
    Ok(())
}

fn normalize_text(
    context: &mut BoundContext,
    text: NativeFrameText,
    attachments: &mut Attachments,
) -> Result<(), FrameStageError> {
    let (columns, failed) = match text {
        NativeFrameText::Converted { columns } => (columns, None),
        NativeFrameText::Failed {
            completed,
            column,
            failure,
        } => (completed, Some((column, failure))),
    };
    check_text_order(
        &context
            .value
            .text
            .iter()
            .map(|column| column.column.clone())
            .collect::<Vec<_>>(),
        &columns
            .iter()
            .map(|column| column.column.clone())
            .collect::<Vec<_>>(),
        failed.as_ref().map(|(column, _)| column),
    )?;
    let mut admitted = Vec::with_capacity(columns.len());
    for converted in columns {
        let array = take_strings(attachments, &converted.values)?;
        let table = context
            .walked
            .tables
            .iter()
            .find(|table| table.name == converted.column.table)
            .ok_or_else(|| protocol("converted native text names an absent sampled table"))?;
        if array.len() != table.batch.len() || array.null_count() != 0 {
            return Err(protocol(
                "converted native text must cover every selected row after native fillna",
            )
            .into());
        }
        admitted.push((converted, array));
    }
    if let Some((column, failure)) = failed {
        return Err(FrameStageError::Native {
            site: NativeSite::Text(column),
            failure,
            diagnostic: None,
            dropped_summary: context.plan.task.labels.graph.dropped_summary(),
        });
    }
    for (converted, array) in admitted {
        let values =
            sdm_engine::strings::StringColumn::new(array.as_ref(), converted.values.encoding)
                .map_err(|_| {
                    protocol("converted native text has an inconsistent physical encoding")
                })?;
        let table = context
            .walked
            .tables
            .iter_mut()
            .find(|table| table.name == converted.column.table)
            .ok_or_else(|| protocol("converted native text names an absent sampled table"))?;
        let column = table
            .columns
            .iter_mut()
            .find(|(name, _)| name == &converted.column.column)
            .ok_or_else(|| protocol("converted native text names an absent sampled column"))?;
        let profile = casing_profile(&converted.facts)?;
        let words = (0..values.len())
            .map(|row| {
                values
                    .get(row)
                    .map(|value| sdm_engine::words::normalize_native(value, profile))
                    .map_err(|_| {
                        protocol("converted native text contains invalid surrogatepass bytes")
                    })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        column.1 = payload::SampledColumn::Words(words);
    }
    Ok(())
}

fn encoded_identity(
    plan: &str,
    context: &str,
    range: FrameBatchRange,
    answering: u64,
    tables: &[String],
    bytes: &[u8],
    logical: &Value,
) -> Result<String, CoreError> {
    let mut identity = Identity::new("encoded-request-part");
    identity.value(&(plan, context, range, answering, tables))?;
    identity.bytes(bytes);
    identity.logical(logical);
    Ok(identity.finish())
}

/// The owned result of the existing encoder before its call-local ids exist.
pub(crate) struct EncodedCall {
    plan: BoundPlan,
    context_identity: String,
    range: FrameBatchRange,
    built: payload::Built,
    tables: Vec<String>,
}

/// Complete one selected context through the existing model request encoder.
/// Lag columns are actual core-computed columns from the existing lag kernel,
/// never native instructions, host-selected rows or a second payload schema.
pub(crate) fn payload_part(
    asker: &Asker,
    value: SampledFrameContext,
    text: NativeFrameText,
    values: Option<AttachmentId>,
    attachments: &mut Attachments,
) -> Result<EncodedCall, FrameStageError> {
    let mut context = bind_context(asker, value, attachments)?;
    normalize_text(&mut context, text, attachments)?;
    if context.walked.tables.len() > SUBGRAPH_TABLES {
        return Err(invalid(
            Sentence::SubgraphPastTables,
            "the request's subgraph carries more tables than the endpoint reads",
            "tables",
        )
        .with_field("tables", count(context.walked.tables.len()))
        .with_field("limit", count(SUBGRAPH_TABLES))
        .into());
    }
    let asked = &context.plan.task.labels.asked;
    let graph = &context.plan.task.labels.graph.prepared.graph;
    let link_target = if let Shape::Aggregated(windowed) = &asked.target {
        if windowed.aggregation == labels::Aggregation::ListDistinct {
            let column = windowed
                .column
                .as_deref()
                .ok_or_else(|| unvalidated("a list target with no column"))?;
            let edges = graph
                .edges
                .iter()
                .filter(|edge| edge.src == windowed.table && edge.fkey == column)
                .collect::<Vec<_>>();
            match edges.as_slice() {
                [edge] => Some(edge.dst.clone()),
                _ => {
                    return Err(CoreError::new(
                        Sentence::LinkTargetNotForeignKey,
                        "the list target is not one foreign key",
                    )
                    .with_field(
                        "column",
                        Detail::text(format!("{}.{}", windowed.table, column)),
                    )
                    .into());
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Some(target) = &link_target {
        if !context
            .walked
            .tables
            .iter()
            .any(|table| &table.name == target)
        {
            return Err(CoreError::new(
                Sentence::LinkHistoryMissing,
                "the sampled subgraph has no historical target entities",
            )
            .into());
        }
    }
    let lags = context
        .plan
        .task
        .lags
        .as_ref()
        .map(|lags| {
            let trained = context.plan.task.labels.labelled.keys.len();
            let largest = usize::try_from(
                context
                    .plan
                    .task
                    .value
                    .rows_per_call
                    .min(context.plan.task.value.prediction_rows),
            )
            .map_err(|_| protocol("native lag batch exceeds host address space"))?;
            lags.columns(trained, largest, context.value.range)
        })
        .transpose()?
        .unwrap_or_default();
    let values: Option<Value> = take_json_if(attachments, values)?;
    let flags = options::values(values.as_ref())?;
    let (use_prediction_time, return_embeddings) = options::request_flags(&flags)?;
    let meaning = &context.plan.task.value.labelled.meaning;
    let anchors = context
        .anchors
        .iter()
        .map(|&anchor| {
            if meaning.no_time_marker && anchor == TIME_MIN {
                TIME_MIN
            } else {
                anchor.div_euclid(1_000_000_000).wrapping_mul(1_000_000_000)
            }
        })
        .collect();
    let tables = context
        .walked
        .tables
        .iter()
        .map(|table| table.name.clone())
        .collect::<Vec<_>>();
    let built = payload::build(&Sampled {
        num_train: context.plan.task.labels.labelled.keys.len(),
        num_test: (context.value.range.stop - context.value.range.start) as usize,
        anchors,
        kind: asked.task,
        labels: context.plan.task.labels.labelled.labels.clone(),
        windowed: asked.range.is_some(),
        tables: context.walked.tables,
        links: context.walked.links,
        entity_table: meaning.entity.table.clone(),
        link_target,
        source_query: meaning.source_query.clone(),
        run_mode: mode(meaning.mode),
        use_prediction_time,
        return_embeddings,
        options: context.plan.value.inference.clone(),
        statistic: meaning.statistic.clone(),
        top_k: context.plan.value.top_k,
        rows_per_call: context.plan.task.value.rows_per_call as usize,
        lags,
        float_text: crate::pql::float_text,
    })?;
    Ok(EncodedCall {
        plan: context.plan,
        context_identity: context.value.identity,
        range: context.value.range,
        built,
        tables,
    })
}

/// Publish one complete immutable call, without previous or future bodies.
/// The host retains these independent outputs in its ordinary batch loop.
fn publish_part(asker: &Asker, next: EncodedCall) -> Result<EncodedFramePart, CoreError> {
    let identity = encoded_identity(
        &next.plan.value.identity,
        &next.context_identity,
        next.range,
        next.built.answering as u64,
        &next.tables,
        &next.built.bytes,
        &next.built.logical,
    )?;
    let mut part = EncodedFramePart {
        identity,
        context_identity: next.context_identity,
        range: next.range,
        payload: AttachmentId(0),
        answering: next.built.answering as u64,
        tables: next.tables,
    };
    let owner = RetainedFramePayload::new(&next.plan, &part, next.built);
    part.payload = asker.attach(Attachment::FramePayload(owner))?;
    Ok(part)
}

/// Assemble complete independent calls exactly once after the batch loop.
/// Ordered ranges and whole-request policy remain core decisions. Native
/// conversion, graph sampling and encoding have already reached their own
/// original positions; no earlier body is replayed through later batches.
pub(crate) fn finish_parts(
    asker: &Asker,
    plan: FrameRequestPlan,
    parts: FrameRequestParts,
    attachments: &mut Attachments,
) -> Result<PreparedFrameValue, CoreError> {
    let bound = bind_plan(asker, plan, attachments)?;
    if parts.version != FRAME_MATERIALIZATION_VERSION
        || parts.plan_identity != bound.value.identity
        || parts.task_identity != bound.task.value.identity
    {
        return Err(protocol(
            "encoded calls belong to another complete task or request plan",
        ));
    }
    let mut parts = parts.parts;
    ordered_coverage(
        &parts.iter().map(|part| part.range).collect::<Vec<_>>(),
        bound.task.value.prediction_rows,
        bound.task.value.rows_per_call,
        false,
    )?;
    let mut sent = BTreeSet::new();
    let mut answering = 0_u64;
    let mut native_payloads = Vec::with_capacity(parts.len());
    for part in &mut parts {
        let Some(Attachment::FramePayload(owner)) = attachments.remove(&part.payload) else {
            return Err(protocol(
                "encoded frame part lacks its core-created complete payload owner",
            ));
        };
        owner.validate(&bound, part)?;
        answering = answering
            .checked_add(part.answering)
            .ok_or_else(|| protocol("frame request answering count exceeds UInt64"))?;
        sent.extend(part.tables.iter().cloned());
        // One final ordinary RequestState body plus its genuine logical
        // owner. Both originate from the same existing typed encoder.
        part.payload = asker.attach(Attachment::Bytes(owner.wire().to_vec()))?;
        native_payloads.push(sdm_contract::scalar::native_value::NativePayload {
            version: sdm_contract::scalar::native_value::NATIVE_VALUE_VERSION,
            identity: part.identity.clone(),
            value: asker.attach(Attachment::FramePayload(owner))?,
            wire: part.payload,
        });
    }
    let excluded = selected_exclusions(&bound.task, &bound.deferred, true)?
        .into_iter()
        .map(|column| (column.table, column.column))
        .collect::<Vec<_>>();
    notices(
        asker,
        &bound.task.labels.graph.prepared.description,
        &bound.task.labels.asked,
        &sent,
        &excluded,
        usize::try_from(answering)
            .map_err(|_| protocol("frame answering count exceeds this host's address space"))?,
        bound.task.labels.asked.range.is_some(),
    );
    let query = attach_json(asker, &bound.task.value.labelled.meaning.source_query)?;
    let first = parts.remove(0);
    let batches = if parts.is_empty() {
        None
    } else {
        Some(
            parts
                .into_iter()
                .map(|part| RequestBatch {
                    payload: part.payload,
                    rows: part.range.stop - part.range.start,
                    request_id: None,
                })
                .collect(),
        )
    };
    Ok(PreparedFrameValue::Request(MaterializedFrames {
        request: Materialized {
            state: RequestState {
                version: REQUEST_STATE_VERSION,
                payload: first.payload,
                query,
                forecast: None,
                batches,
                ranking: None,
            },
            query: Some(bound.task.value.labelled.meaning.facts.clone()),
        },
        payloads: native_payloads,
    }))
}

#[cfg(test)]
mod tests;
