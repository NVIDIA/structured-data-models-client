// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compose pure row selection with the model capability; native stages stay
//! separate so an earlier refusal or warning prevents later observations.

use sdm_contract::tabular::*;
use sdm_contract::{CoreError, Sentence};
use sdm_engine::tabular;

pub(crate) fn expand(input: ExpandTabularKeys) -> ExpandedTabularKeys {
    let rows = tabular::expand_keys(&input.source, &input.requested);
    ExpandedTabularKeys {
        rows: rows.rows,
        missing: rows.missing,
    }
}

pub(crate) fn filter(input: FilterTabularRows) -> Result<FilteredTabularRows, CoreError> {
    let positions = tabular::candidates(&input.known, &input.excluded).ok_or_else(|| {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "tabular excluded position exceeds its population",
        )
    })?;
    Ok(FilteredTabularRows { positions })
}

fn count_error(error: tabular::CountError) -> TabularCountError {
    match error {
        tabular::CountError::Zero => TabularCountError::Zero,
        tabular::CountError::AboveLimit => TabularCountError::AboveLimit,
        tabular::CountError::AbovePopulation => TabularCountError::AbovePopulation,
    }
}

pub(crate) fn decide(input: DecideTabularContext) -> TabularContextDecision {
    match tabular::context_decision(
        input.population,
        input.requested,
        sdm_request::predict::CONTEXT_ROWS_MAX,
    ) {
        Ok(value) => TabularContextDecision::Selected {
            wanted: value.wanted,
            reduced: value.reduced,
            advise_default_reduction: value.advise_default_reduction,
        },
        Err(error) => TabularContextDecision::Refused {
            reason: count_error(error),
        },
    }
}

pub(crate) fn sample(input: SampleTabularRows) -> SampledTabularRows {
    match tabular::sample_offsets(input.population, input.wanted) {
        Ok(offsets) => SampledTabularRows::Sampled { offsets },
        Err(error) => SampledTabularRows::Refused {
            reason: count_error(error),
        },
    }
}

pub(crate) fn membership(input: ObserveTabularMembership) -> ObservedTabularMembership {
    ObservedTabularMembership {
        missing: input
            .present
            .into_iter()
            .enumerate()
            .filter_map(|(at, present)| (!present).then_some(at))
            .collect(),
    }
}

pub(crate) fn order(
    input: OrderSampledTabularRows,
) -> Result<OrderedSampledTabularRows, CoreError> {
    sdm_engine::index_sort::order_sampled(&input.index)
}

/// Fuse ordinary expansion, exclusion, target filtering and capacity policy.
/// Sampling remains after native warning delivery and uses its reached count.
pub(crate) fn select(input: SelectTabularRows) -> Result<SelectedTabularRows, CoreError> {
    let (scored, excluded, population) = match input.mode {
        TabularSelectionMode::Indices { source, requested } => {
            let expanded = tabular::expand_keys(&source, &requested);
            if !expanded.missing.is_empty() {
                return Ok(SelectedTabularRows::Missing {
                    ordinals: expanded.missing,
                });
            }
            let excluded = expanded.rows.clone();
            (expanded.rows, excluded, source.len())
        }
        TabularSelectionMode::Supplied { source, scoring } => {
            let excluded = tabular::matching_rows(&source, &scoring);
            (Vec::new(), excluded, source.len())
        }
    };
    if input.known.len() != population {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "tabular target validity differs from the source population",
        ));
    }
    let candidates = filter(FilterTabularRows {
        known: input.known,
        excluded,
    })?
    .positions;
    let context = decide(DecideTabularContext {
        population: candidates.len(),
        requested: input.requested_context,
    });
    Ok(SelectedTabularRows::Selected {
        scored,
        candidates,
        context,
    })
}

struct NumericObserver(crate::run::Asker);
impl sdm_engine::index_sort::numeric_owner::Observer for NumericObserver {
    fn observe(
        &mut self,
        request: sdm_contract::numeric_owner::NumericOwnerRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<sdm_engine::index_sort::numeric_owner::Observation, CoreError>,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            use sdm_contract::numeric_owner::NumericOwnerOutcome;
            use sdm_engine::index_sort::numeric_owner::Observation;
            let crate::run::Reply::Numeric {
                outcome,
                mut attachments,
            } = self.0.numeric_owner(request).await
            else {
                return Err(CoreError::protocol(
                    Sentence::WrongReply,
                    "numeric observation requires its own reply",
                ));
            };
            match outcome {
                NumericOwnerOutcome::SortOwners {
                    owners_and_facts,
                    source_positions,
                    missing_positions,
                } => Ok(Observation::Captured {
                    owners: crate::attached::take_json(&mut attachments, owners_and_facts)?,
                    positions: crate::attached::take_json(&mut attachments, source_positions)?,
                    missing: crate::attached::take_json(&mut attachments, missing_positions)?,
                }),
                other => Ok(Observation::Control(other)),
            }
        })
    }
}
pub(crate) async fn order_numeric(
    input: sdm_contract::numeric_owner::OrderNumericOwners,
    asker: crate::run::Asker,
) -> Result<sdm_contract::numeric_owner::OrderedNumericOwners, CoreError> {
    sdm_engine::index_sort::numeric_owner::order(input, NumericObserver(asker)).await
}
