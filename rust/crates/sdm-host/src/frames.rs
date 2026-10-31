// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frame inputs may alias mutable native storage. Detach before the run
//! borrows them; foreign C-stream ownership alone does not prevent writes.

use arrow_array::{RecordBatch, RecordBatchOptions, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use sdm_contract::{CoreError, Sentence};
use sdm_core::ArrowData;

fn rebuilt(error: arrow_schema::ArrowError) -> CoreError {
    let _ = error;
    CoreError::protocol(
        Sentence::InvalidMessage,
        "detached frame data has an invalid Arrow layout",
    )
}

fn owned(data: ArrayData) -> Result<ArrayData, CoreError> {
    let buffers = data
        .buffers()
        .iter()
        .map(|b| Buffer::from(b.as_slice()))
        .collect();
    let children = data
        .child_data()
        .iter()
        .cloned()
        .map(owned)
        .collect::<Result<Vec<_>, _>>()?;
    let nulls = data.nulls().map(|nulls| {
        NullBuffer::new(BooleanBuffer::new(
            Buffer::from(nulls.inner().values()),
            nulls.offset(),
            nulls.len(),
        ))
    });
    data.into_builder()
        .buffers(buffers)
        .child_data(children)
        .nulls(nulls)
        .build()
        .map_err(rebuilt)
}

/// Own every data/offset/validity/dictionary/child buffer, preserving slices
/// and received type facts. Generic provider admission remains zero-copy.
pub fn detach(data: ArrowData) -> Result<ArrowData, CoreError> {
    let batches = data
        .batches
        .into_iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .map(|array| owned(array.to_data()).map(make_array))
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new_with_options(
                batch.schema(),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(rebuilt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ArrowData {
        schema: data.schema,
        received: data.received,
        batches,
    })
}
