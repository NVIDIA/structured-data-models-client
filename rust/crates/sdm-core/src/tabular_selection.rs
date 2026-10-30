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
