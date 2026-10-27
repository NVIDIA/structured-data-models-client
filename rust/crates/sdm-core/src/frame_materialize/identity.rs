// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identities of useful immutable frame values, never attachment identities.

use arrow_row::{RowConverter, SortField};
use sdm_contract::frames::materialize::FrameBatchRange;
use sdm_contract::{CoreError, Sentence};
use serde::Serialize;

use crate::run::{ArrowData, internal};

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The existing frame fingerprint's value algorithm with domain separation
/// and explicit component boundaries. This is coherence, not authentication.
pub(crate) struct Identity(u64);

impl Identity {
    pub fn new(domain: &str) -> Self {
        let mut identity = Self(0xcbf2_9ce4_8422_2325);
        identity.bytes(b"frame-materialization-value-v1");
        identity.bytes(domain.as_bytes());
        identity
    }

    fn feed(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        self.feed(&(bytes.len() as u64).to_le_bytes());
        self.feed(bytes);
    }

    pub fn value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CoreError> {
        self.bytes(
            &serde_json::to_vec(value)
                .map_err(|_| internal("frame value facts do not serialize"))?,
        );
        Ok(())
    }

    /// Hash the same Arrow RowConverter value bytes used for graph snapshots.
    /// Feeding logical rows across batch boundaries avoids joining or copying
    /// complete source frames and makes empty/repartitioned chunks irrelevant.
    pub fn arrow(&mut self, data: &ArrowData) -> Result<(), CoreError> {
        for field in data.schema.fields() {
            self.value(&(
                field.name(),
                format!("{:?}", field.data_type()),
                field.is_nullable(),
            ))?;
        }
        self.bytes(b"rows");
        self.feed(
            &(data
                .batches
                .iter()
                .map(|batch| batch.num_rows() as u64)
                .sum::<u64>())
            .to_le_bytes(),
        );
        let converter = RowConverter::new(
            data.schema
                .fields()
                .iter()
                .map(|field| SortField::new(field.data_type().clone()))
                .collect(),
        )
        .map_err(|_| invalid("frame value contains an unsupported physical column"))?;
        for batch in &data.batches {
            if batch.schema() != data.schema {
                return Err(invalid(
                    "frame value batches do not match their declared schema",
                ));
            }
            let rows = converter
                .convert_columns(batch.columns())
                .map_err(|_| invalid("frame value row bytes are unreadable"))?;
            for row in rows.iter() {
                self.bytes(row.as_ref());
            }
        }
        Ok(())
    }

    pub fn finish(self) -> String {
        format!("{:016x}", self.0)
    }
}

/// A prefix may be retained between batches. A completed request must cover
/// every prediction exactly once using the task's core-selected call ranges.
pub(crate) fn ordered_coverage(
    ranges: &[FrameBatchRange],
    prediction_rows: u64,
    rows_per_call: u64,
    prefix: bool,
) -> Result<u64, CoreError> {
    if rows_per_call == 0 {
        return Err(invalid("frame task has no positive call capacity"));
    }
    let mut stop = 0;
    for range in ranges {
        let expected = stop + rows_per_call.min(prediction_rows.saturating_sub(stop));
        if range.start != stop || range.stop != expected || range.stop <= range.start {
            return Err(invalid(
                "frame request parts do not cover ordered task ranges",
            ));
        }
        stop = range.stop;
    }
    if stop > prediction_rows || (!prefix && stop != prediction_rows) {
        return Err(invalid(
            "frame request parts do not completely cover prediction rows",
        ));
    }
    Ok(stop)
}
