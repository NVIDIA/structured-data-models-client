// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use crate::run::{ArrowData, Asker, Attachment, internal};
use sdm_contract::{
    CoreError, CoreWarning, Sentence,
    frames::{TimeSeriesOptions, TimeSeriesPrepared},
    ops::OperationResult,
};
pub(crate) fn prepare(
    asker: &Asker,
    options: TimeSeriesOptions,
    data: ArrowData,
) -> Result<OperationResult, CoreError> {
    let batch = arrow_select::concat::concat_batches(&data.schema, &data.batches)
        .map_err(|_| internal("time-series Arrow batches disagree"))?;
    let made = sdm_data::timeseries::prepare(&batch, &options)?;
    let counts = sdm_schema::keys::count_frame(
        made.entity.schema().as_ref(),
        std::slice::from_ref(&made.entity),
        std::slice::from_ref(&made.key),
    )?;
    if counts.missing != 0 || counts.distinct != counts.rows {
        return Err(
            CoreError::new(Sentence::TimeSeriesInput, "invalid entity keys").with_field(
                "reason",
                sdm_contract::Detail::text("entity keys must be nonnull and unique"),
            ),
        );
    }
    if made.irregular {
        asker.warn(CoreWarning::new(
            Sentence::TimeSeriesIrregular,
            "irregular history interval",
        ));
    }
    let entity = asker.attach(Attachment::Arrow(ArrowData::new(
        made.entity.schema(),
        vec![made.entity],
    )))?;
    let target = asker.attach(Attachment::Arrow(ArrowData::new(
        made.target.schema(),
        vec![made.target],
    )))?;
    crate::direct::value(
        asker,
        &TimeSeriesPrepared {
            entity,
            target,
            key: made.key,
            step_ns: made.step_ns,
            anchor_ns: options.anchor_ns,
            forecasts: options.forecasts,
        },
    )
}
