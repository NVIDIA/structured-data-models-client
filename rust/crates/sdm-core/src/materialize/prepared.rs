// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual prepared task populations shared by direct and native completions.
//! Values own sampled rows, parsed meanings and reached time/cache facts.
//! No owner contains an asker, future, callback or execution cursor.

use super::*;

pub(crate) struct PreparedSource {
    pub(super) source: String,
    pub(super) description: SourceDescription,
    pub(super) graph: Graph,
    pub(super) asked: Asked,
    pub(super) source_query: String,
    pub(super) indices: ArrayRef,
    pub(super) seed: Seed,
    pub(super) seed_value: Scalar,
    pub(super) unseeded: bool,
    pub(super) anchor_entity: bool,
    pub(super) anchor: i64,
    pub(super) context_anchor: i64,
    pub(super) span: Option<(Offset, Offset)>,
    pub(super) earliest: Option<i64>,
    pub(super) context_examples: usize,
    pub(super) trials: usize,
    pub(super) entity_filter: Option<String>,
    pub(super) mode: options::RunMode,
    pub(super) deferred: options::Asked,
    pub(super) asked_examples: Option<usize>,
    pub(super) statistic: Option<String>,
    pub(super) most_calls: Option<u32>,
    pub(super) nulls: NullOrder,
    pub(super) no_time_marker: bool,
    pub(super) read_times: Times,
    pub(super) memo: SamplerMemo,
    pub(super) frames: Option<crate::frame_materialize::Context>,
    pub(super) options_identity: String,
    #[cfg(feature = "test-ops")]
    pub(super) labels_only: bool,
}

pub(crate) struct PreparedTask {
    pub(super) source: Arc<PreparedSource>,
    pub(super) population: task::ExamplePopulation,
    pub(super) indices: ArrayRef,
    pub(super) memo: SamplerMemo,
}

/// Borrowed label kernels reconstructed from the same owned task meaning.
/// The owner contains neither these borrows nor a suspended computation.
pub(super) struct Draw<'a> {
    source: &'a PreparedSource,
    entity: &'a Table,
    windows: Vec<&'a Windowed>,
    labelled: Vec<Fetch<'a>>,
    target: Vec<Fetch<'a>>,
    entropy: Entropy,
}

impl<'a> Draw<'a> {
    pub(super) fn new(source: &'a PreparedSource) -> Result<Self, CoreError> {
        let entity = source
            .graph
            .table(&source.asked.entity_table)
            .ok_or_else(|| internal("the captured source lost its entity table"))?;
        let mut windows = Vec::new();
        source.asked.target.windowed(&mut windows);
        Ok(Self {
            source,
            entity,
            labelled: fetches(
                &source.graph,
                &source.asked.entity_table,
                &source.asked.every_windowed(),
            )?,
            target: fetches(&source.graph, &source.asked.entity_table, &windows)?,
            windows,
            entropy: Entropy::from_hex(source.seed.magnitude_hex())
                .ok_or_else(|| internal("the captured source lost its sampling entropy"))?,
        })
    }

    pub(super) fn draw<'b>(&'b self, times: &'b Times) -> LabelDraw<'b> {
        let source = self.source;
        LabelDraw {
            entity: self.entity,
            key: &source.asked.entity_column,
            asked: &source.asked,
            read_times: times,
            labelled_fetches: &self.labelled,
            fetches: &self.target,
            windows: &self.windows,
            entropy: &self.entropy,
            context_anchor: source.context_anchor,
            anchor_entity: source.anchor_entity,
            span: source.span,
            earliest: source.earliest,
            context_examples: source.context_examples,
            trials: source.trials,
            entity_filter: source.entity_filter.as_deref(),
        }
    }
}

pub(super) async fn population(
    asker: &Asker,
    source: Arc<PreparedSource>,
) -> Result<PreparedTask, CoreError> {
    let entity = source
        .graph
        .table(&source.asked.entity_table)
        .ok_or_else(|| internal("the captured source lost its entity table"))?;
    let mut memo = Memo {
        asker,
        frames: source.frames.clone(),
        memo: source.memo.clone(),
    };
    let sampled = entity_sample(
        asker,
        EntityDraw {
            entity,
            key: &source.asked.entity_column,
            asked: &source.asked,
            read_times: &source.read_times,
            memo: &memo.memo,
            frames: source.frames.as_ref(),
            trials: source.trials,
            seed: &source.seed,
            seed_value: source.seed_value.clone(),
            unseeded: source.unseeded,
        },
    )
    .await?;
    let indices = keys::cast(&source.indices, sampled.entity_key.data_type())?;
    let entropy = Entropy::from_hex(source.seed.magnitude_hex())
        .ok_or_else(|| internal("the captured source lost its sampling entropy"))?;
    let labelled_fetches = fetches(
        &source.graph,
        &source.asked.entity_table,
        &source.asked.every_windowed(),
    )?;
    let mut windows = Vec::new();
    source.asked.target.windowed(&mut windows);
    let target_fetches = fetches(&source.graph, &source.asked.entity_table, &windows)?;
    let population = task::example_population(
        &LabelDraw {
            entity,
            key: &source.asked.entity_column,
            asked: &source.asked,
            read_times: &source.read_times,
            labelled_fetches: &labelled_fetches,
            fetches: &target_fetches,
            windows: &windows,
            entropy: &entropy,
            context_anchor: source.context_anchor,
            anchor_entity: source.anchor_entity,
            span: source.span,
            earliest: source.earliest,
            context_examples: source.context_examples,
            trials: source.trials,
            entity_filter: source.entity_filter.as_deref(),
        },
        sampled,
    )?;
    keep_stored_times(&mut memo, &source.read_times);
    Ok(PreparedTask {
        source,
        population,
        indices,
        memo: memo.memo,
    })
}

impl PreparedTask {
    pub(super) async fn context(&self, asker: &Asker) -> Result<task::LabelledContext, CoreError> {
        let source = &self.source;
        let plan = Draw::new(source)?;
        let mut memo = Memo {
            asker,
            frames: source.frames.clone(),
            memo: self.memo.clone(),
        };
        let result = task::label_population(
            asker,
            &mut memo,
            plan.draw(&source.read_times),
            self.population.clone(),
            None,
        )
        .await;
        keep_stored_times(&mut memo, &source.read_times);
        result
    }
}

pub(super) async fn finish(
    asker: &Asker,
    task: &PreparedTask,
    labelled: task::LabelledContext,
) -> Result<Gathered, CoreError> {
    finish_observed(asker, task, labelled, None).await
}

pub(super) async fn finish_native(
    asker: &Asker,
    task: &PreparedTask,
    labelled: task::LabelledContext,
    current: Option<&sdm_wire::json::Value>,
) -> Result<Gathered, CoreError> {
    finish_observed(
        asker,
        task,
        labelled,
        Some(current.and_then(|values| values.get("inference_config"))),
    )
    .await
}

async fn finish_observed(
    asker: &Asker,
    task: &PreparedTask,
    labelled: task::LabelledContext,
    current_inference: Option<Option<&sdm_wire::json::Value>>,
) -> Result<Gathered, CoreError> {
    let PreparedSource {
        description,
        graph,
        asked,
        source_query,
        seed,
        unseeded,
        anchor_entity,
        anchor,
        mode,
        deferred,
        asked_examples,
        statistic,
        most_calls,
        nulls,
        no_time_marker,
        read_times,
        frames,
        ..
    } = task.source.as_ref();
    let (unseeded, anchor_entity, anchor, mode, asked_examples, most_calls, nulls, no_time_marker) = (
        *unseeded,
        *anchor_entity,
        *anchor,
        *mode,
        *asked_examples,
        *most_calls,
        *nulls,
        *no_time_marker,
    );
    let indices = Arc::clone(&task.indices);
    let options::Asked {
        num_context_samples,
        inference_config,
        top_k,
        rows_per_call,
        lag_timesteps,
        ..
    } = deferred;
    let (num_context_samples, rows_per_call, lag_timesteps) =
        (*num_context_samples, *rows_per_call, *lag_timesteps);
    let span = task.source.span;
    let mut memo = Memo {
        asker,
        frames: frames.clone(),
        memo: task.memo.clone(),
    };
    let entity = graph
        .table(&asked.entity_table)
        .ok_or_else(|| internal("the prepared task lost its entity table"))?;
    let key = asked.entity_column.clone();
    let link_target = link_target(graph, &asked.target)?;
    let mut windows = Vec::new();
    asked.target.windowed(&mut windows);
    let fetches = fetches(graph, &asked.entity_table, &windows)?;
    #[cfg(feature = "test-ops")]
    let labels_only = task.source.labels_only;
    let gathered = labelled.keys.len();
    // The target's own past as features (`lag_timesteps`, T2.9), where the
    // reference's engine adds them: its value refused, as the engine
    // refuses it, once the context is labelled; for a temporal query, each
    // aggregated table's rows the lags read, in the labels' order, for the
    // context's examples and the entities predicted for, in that order,
    // before the context is narrowed ([`crate::lag`]).
    // The stored values the labels' reads spelled their bounds by, kept.
    keep_stored_times(&mut memo, read_times);
    let steps = match lag_timesteps {
        None => 0,
        Some(asked) => asked.refusal(!fetches.is_empty() && link_target.is_none())?,
    };
    let lag = match steps {
        0 => None,
        steps => {
            // Each step moves by the target's window's end
            // (`target_timeframe.end_date_offset`).
            let (_, end) = span.ok_or_else(|| internal("a lag without its target's window"))?;
            let keys = arrow_select::concat::concat(&[labelled.keys.as_ref(), indices.as_ref()])
                .map_err(|_| internal("the examples' keys do not concatenate"))?;
            let mut lag_anchors = labelled.anchors.clone();
            lag_anchors.extend(std::iter::repeat_n(anchor, indices.len()));
            let back = crate::lag::shifted(steps, end, &lag_anchors)?;
            // Each table read once from the moved anchors, over its pooled
            // window stretched by the steps (`sample_lagged_target`'s
            // `time_offset_dict`).
            let mut fetched = BTreeMap::new();
            for fetch in &fetches {
                let (starts, ends) = crate::lag::reach(steps, fetch.start, fetch.end, &back)?;
                let rows =
                    fetch_rows(asker, read_times, fetch, Arc::clone(&keys), starts, ends).await?;
                fetched.insert(fetch.table.name.clone(), rows);
            }
            keep_stored_times(&mut memo, read_times);
            let shape = autoregressive(&asked.target);
            let lagged = lag_anchors.len();
            Some(crate::lag::Lag {
                steps,
                end,
                shifted: back,
                target: Box::new(move |anchors: &[i64]| {
                    temporal_target(&shape, &fetched, anchors, lagged, "lag_timesteps")
                }),
            })
        }
    };
    // The default mode capped the context at as many examples as it takes:
    // said, as the client says it, the caller having chosen no size.
    if num_context_samples.is_none() && gathered >= options::RunMode::Fast.context_examples() {
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
    // Then narrowed to the size asked for (`_narrow_context`).
    let labelled = labelled.narrow(asked_examples);
    let num_train = labelled.keys.len();
    let train_keys = labelled.keys;
    let anchors = labelled.anchors;
    let labels = labelled.labels;
    #[cfg(feature = "test-ops")]
    if labels_only {
        return labels_result(asker, &train_keys, &anchors, &labels).map(Gathered::Done);
    }
    // How many entities one call scores: the endpoint's 10,000, or fewer
    // where the caller asked (`rows_per_call`); a request of more is
    // gathered as several calls, each with a context of its own, at most as
    // many as the host's limit, 100 where it names none, refused once the
    // context is gathered, as the client refuses it after its engine's task
    // table (`_refuse_more_entities_than_fit_in_memory`, `main`'s 1fc5a686).
    let num_test = indices.len();
    let layout = BatchLayout::new(asked.task, rows_per_call, num_test, most_calls)?;
    // How many candidates the call asked be kept (`_candidates_to_rank`,
    // T2.8), which the engine writes into the task; then the inference
    // options, read for the task's config class, as the engine reads them
    // before it lays out any request.
    let top_k = top_k.as_ref().map(options::top_k).transpose()?;
    let options = options::options(
        asked.task,
        current_inference.unwrap_or(inference_config.as_ref()),
    )?;
    // A static target's columns, kept out of what the request reads and
    // sends (`get_exclude_cols_dict`); each must be one of its table's
    // features, as the reference's sampler requires.
    let mut excluded_columns = BTreeSet::new();
    asked.target.entity_columns(&mut excluded_columns);
    let mut excluded: Vec<(String, String)> = excluded_columns
        .into_iter()
        .map(|column| (entity.name.clone(), column))
        .collect();
    for (_, column) in &excluded {
        if !graph.features(entity).iter().any(|(n, _)| n == column) {
            return Err(invalid(
                Sentence::TargetNotAFeature,
                "the column a static target reads is not one of its table's features",
                "query",
            ));
        }
    }
    // Then the columns the caller keeps out (`exclude_cols_dict`, T2.11),
    // merged after the query's (`_columns_to_leave_out`), each a feature of
    // a table of the graph, as the reference's sampler requires.
    // (`_columns_to_leave_out`, then the hops `_resolve_options` resolves,
    // then the names `sample_subgraph` checks.)
    let exclude_cols = options::excluded(deferred)?;
    let walk_hops = if frames.is_some() && deferred.neighbors.is_some() {
        // The native frame sampler validates its signed array after
        // seed reset; SQL's numeric-comparison coercions do not apply.
        vec![]
    } else {
        options::neighbors(deferred, mode)?
    };
    // Each fact table's look-back rate, where the provider's neighbor
    // reads look back no further than the reference's warehouse samplers.
    let lookback = if frames.is_some() {
        vec![]
    } else {
        lookbacks(&mut memo, read_times, graph).await?
    };
    let hops_ignored = options::hops_ignored(deferred);
    // The engine says it ignores `num_hops` where `num_neighbors` is given
    // (`_resolve_options`), once the task is labelled.
    if let Some(num_hops) = hops_ignored {
        asker.warn(CoreWarning {
            fields: [("num_hops".to_owned(), Detail::text(&num_hops))].into(),
            ..CoreWarning::new(
                Sentence::HopsIgnored,
                "num_hops is ignored where num_neighbors is given",
            )
        });
    }
    if let Some(frames) = frames {
        // The native iterator validates the WHOLE task before generating
        // its first batch. A later unresolved entity must not let an earlier
        // batch reset or advance the continuing neighbor stream. The same
        // resolver supplies concrete node-reference rows to fixed stages.
        let _ = crate::frame_materialize::lock(frames)
            .resolve_references(entity, &[&train_keys, &indices])?;
    }
    excluded_by_caller(graph, &exclude_cols, &mut excluded)?;
    if num_train > TABLE_ROWS {
        return Err(invalid(
            Sentence::ContextPastRows,
            "the context holds more rows than a request takes",
            "query",
        ));
    }
    // The rows the largest call scores (`_the_largest_batch`), which a
    // payload past the endpoint's body names as the size to come down from.
    let largest = layout.largest();
    let train_anchors = anchors.len();
    let entity_names = crate::subgraph::sample_columns(graph, entity, &excluded);
    // The labelled context's examples before narrowing, then the entities
    // predicted for: the examples the lags were read for (T2.9).
    let labelled_examples = gathered;
    // The request's lagged columns, made once (T2.9 fix 3), and how many
    // entities the whole request scores.
    let mut lag_columns: Option<Vec<sdm_request::payload::LagColumn>> = None;
    let all_scored = num_test;
    // The columns each call reads of its examples' entity rows: with an
    // anchor at each entity's own time, its time column too, which anchors
    // each row (`_sample_subgraph`), sent only where it is a feature.
    let mut entity_read = entity_names.clone();
    if anchor_entity
        && let Some(time) = &entity.time_column
        && !entity_read.contains(time)
    {
        entity_read.push(time.clone());
    }
    let mut gathered = Vec::with_capacity(layout.calls);
    let mut sent = BTreeSet::new();
    let mut answering = 0;
    let task_rows = TaskRows::new(Arc::clone(&train_keys), anchors, indices, anchor)?;
    // Each call's payload, in order: the context's examples and labels, the
    // same for every call, with the call's own entities, and the subgraph
    // walked from both, as the reference's engine builds each batch
    // (`_iter_task_requests`: `narrow_prediction`, then `_get_context`),
    // every one before any is sent.
    for range in layout.ranges() {
        let batch = task_rows.batch(range)?;
        let start = batch.range.start;
        let scored = batch.predicted;
        let num_test = scored.len();
        let examples_total = num_train + num_test;
        let all_keys = batch.keys;
        let mut anchors = batch.anchors;

        // The subgraph: each example's entity row.
        let looked_up = if let Some(frames) = frames {
            crate::frame_materialize::lock(frames).lookup(entity, &all_keys, &entity_read)?
        } else {
            let request = sample::by_pkey(
                entity,
                entity_read.clone(),
                key_set(asker, sample::seeds(Arc::clone(&all_keys), vec![])?)?,
                true,
            )?;
            let name = request.name();
            let pending =
                asker
                    .fetch(request)?
                    .with_lookup(Arc::new(super::native_lookup::LookupContext {
                        entity: entity.clone(),
                        key: key.clone(),
                        scored: Arc::clone(&scored),
                        num_train,
                        range: sdm_contract::frames::materialize::FrameBatchRange {
                            start: batch.range.start as u64,
                            stop: batch.range.end as u64,
                        },
                    }));
            read_times
                .read(
                    asker,
                    entity,
                    one_batch(&crate::describe::data(pending.await, name)?)?,
                )
                .await?
        };
        let positions = super::lookup_rows::validate(entity, &key, &scored, num_train, &looked_up)?;
        if positions.len() != examples_total {
            // The table and the keys not found, in the order the caller gave
            // them: values of the caller's own call, which a refusal may name
            // (KO2-86, narrowing decision H17), as the reference names them.
            let found: BTreeSet<usize> = positions.iter().copied().collect();
            let keys = (num_train..examples_total)
                .filter(|at| !found.contains(at))
                .map(|at| key_scalar(scored.as_ref(), at - num_train))
                .collect();
            return Err(invalid(
                Sentence::NotInTable,
                "an entity to predict for is not in the entity table",
                "indices",
            )
            .with_field("reason", Detail::text("not_in_table"))
            .with_field("table", Detail::text(&entity.name))
            .with_field("keys", Detail::List(keys)));
        }
        let mut order: Vec<usize> = (0..positions.len()).collect();
        order.sort_by_key(|&row| positions[row]);
        let entity_rows = take_batch(&looked_up, &order)?;
        // Each predicted entity anchored at its own time, read from its row.
        if anchor_entity && let Some(time) = &entity.time_column {
            let times = times_ns(named(&entity_rows, time)?, time)?;
            for (at, time) in anchors[train_anchors..].iter_mut().zip(&times[num_train..]) {
                // A null time is a missing anchor (`NaT`), never the
                // no-anchor sentinel.
                *at = time.unwrap_or(sample::MISSING_ANCHOR);
            }
        }
        let entity_rows = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(
                entity_names
                    .iter()
                    .map(|name| {
                        Ok(arrow_schema::Field::new(
                            name,
                            named(&entity_rows, name)?.data_type().clone(),
                            true,
                        ))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?,
            )),
            entity_names
                .iter()
                .map(|name| named(&entity_rows, name).map(Arc::clone))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|_| internal("the entity rows do not make one batch"))?;

        // The neighbors of each example's entity, hop by hop.
        let walked = if let Some(frames) = frames {
            crate::frame_materialize::lock(frames)
                .walk(
                    asker,
                    graph,
                    entity,
                    &all_keys,
                    &anchors,
                    crate::frame_materialize::FrameWalkOptions {
                        hops: &walk_hops,
                        neighbor_values: deferred.neighbors.as_deref(),
                        excluded: &excluded,
                        seed: (!unseeded).then(|| seed.to_u128()).flatten(),
                        link_target: link_target.as_deref(),
                    },
                )?
                .into_walked(&memo.memo.source)?
        } else {
            crate::subgraph::Walk {
                asker,
                graph,
                times: read_times,
                anchors: &anchors,
                nulls,
                entity,
                link_target: link_target.as_deref(),
                neighbors: &walk_hops,
                excluded: &excluded,
                lookback: &lookback,
            }
            .run(entity_rows)
            .await?
        };
        keep_stored_times(&mut memo, read_times);
        // The subgraph carries at most as many tables as the endpoint
        // reads, refused as the reference's engine refuses it
        // (`_get_context`).
        if walked.tables.len() > SUBGRAPH_TABLES {
            return Err(invalid(
                Sentence::SubgraphPastTables,
                "the request's subgraph carries more tables than the endpoint reads",
                "tables",
            )
            .with_field("tables", count(walked.tables.len()))
            .with_field("limit", count(SUBGRAPH_TABLES)));
        }
        if let Some(rhs) = &link_target
            && !walked.tables.iter().any(|table| &table.name == rhs)
        {
            return Err(CoreError::new(
                Sentence::LinkHistoryMissing,
                "the sampled subgraph has no historical target entities",
            ));
        }
        sent.extend(walked.tables.iter().map(|t| t.name.clone()));
        // The lagged columns of this call's examples (T2.9): the narrowed
        // context's, then this call's own entities, each its own row's,
        // sliced from the whole request's columns, so each column's type
        // is the request's whatever this call's rows hold (fix 3), typed
        // over the gathered context before narrowing (fix 4); the
        // columns are computed at the first call's payload, where the
        // reference's size refusal falls, held to the largest call's rows.
        if lag_columns.is_none()
            && let Some(lag) = &lag
        {
            lag.fits(num_train + largest)?;
            lag_columns = Some(
                lag.columns(
                    &(0..num_train)
                        .chain(labelled_examples..labelled_examples + all_scored)
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        let rows: Vec<usize> = (0..num_train)
            .chain(num_train + start..num_train + start + num_test)
            .collect();
        let lags = lag_columns
            .iter()
            .flatten()
            .map(|column| column.rows(&rows))
            .collect();
        // The options the engine's request model holds to a boolean, read
        // as the request is built.
        let (use_prediction_time, return_embeddings) = options::request_flags(deferred)?;

        // Every local graph anchor makes the sampler's seconds round trip.
        // DT-3 keeps the no-time marker explicit instead of wrapping it into
        // a real timestamp; the existing payload layer omits that column.
        // Original feature instants and nanosecond validation stay separate.
        let payload_anchors = if frames.is_some() {
            anchors
                .iter()
                .map(|anchor| {
                    if no_time_marker && *anchor == TIME_MIN {
                        TIME_MIN
                    } else {
                        anchor.div_euclid(1_000_000_000).wrapping_mul(1_000_000_000)
                    }
                })
                .collect()
        } else {
            anchors.clone()
        };
        let built = payload::build(&Sampled {
            num_train,
            num_test,
            anchors: payload_anchors,
            kind: asked.task,
            labels: labels.clone(),
            windowed: asked.range.is_some(),
            tables: walked.tables,
            links: walked.links,
            entity_table: entity.name.clone(),
            link_target: link_target.clone(),
            source_query: source_query.clone(),
            run_mode: mode,
            use_prediction_time,
            return_embeddings,
            options: options.clone(),
            statistic: statistic.clone(),
            rows_per_call: largest,
            top_k,
            lags,
            float_text: crate::pql::float_text,
        })?;
        answering += built.answering;
        gathered.push((
            built.bytes,
            u64::try_from(num_test).map_err(|_| internal("a call's rows past u64"))?,
        ));
    }
    // What the request leaves out, said once as the client says it of all
    // its calls, in its order: the tables no call carries, the columns the
    // model does not read, and the context that answers the question,
    // counted across the calls.
    notices(
        asker,
        description,
        asked,
        &sent,
        &excluded,
        answering,
        asked.range.is_some(),
    );
    Ok(Gathered::Request(gathered, source_query.clone()))
}
