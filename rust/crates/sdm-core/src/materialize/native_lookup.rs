// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual counted task lookup data before native business-column conversion.

use super::*;
use sdm_contract::data::Relation;
use sdm_contract::frames::materialize::FrameBatchRange;
use sdm_contract::materialize::PreparedLookupRows;
use sdm_contract::run::ReplyId;

#[cfg(test)]
mod tests;

pub(crate) struct LookupContext {
    pub(crate) entity: Table,
    pub(crate) key: String,
    pub(crate) scored: ArrayRef,
    pub(crate) num_train: usize,
    pub(crate) range: FrameBatchRange,
}

#[derive(Clone)]
pub struct RetainedLookupRows {
    context: Arc<LookupContext>,
    source: Relation,
    read: ReplyId,
}

impl std::fmt::Debug for RetainedLookupRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedLookupRows")
            .field("read", &self.read)
            .field("range", &self.context.range)
            .finish_non_exhaustive()
    }
}

impl RetainedLookupRows {
    pub(crate) fn new(context: Arc<LookupContext>, source: Relation, read: ReplyId) -> Self {
        Self {
            context,
            source,
            read,
        }
    }

    pub(crate) fn meaning(&self, owner: AttachmentId) -> PreparedLookupRows {
        PreparedLookupRows {
            owner,
            read: self.read,
            source: self.source.clone(),
            key: self.context.key.clone(),
            range: self.context.range,
            context_rows: self.context.num_train as u64,
            scored_rows: self.context.scored.len() as u64,
        }
    }

    pub(crate) fn matches(&self, meaning: &PreparedLookupRows) -> bool {
        self.meaning(meaning.owner) == *meaning
    }

    pub(crate) fn validate(&self, rows: &RecordBatch) -> Result<Vec<usize>, CoreError> {
        lookup_rows::validate(
            &self.context.entity,
            &self.context.key,
            &self.context.scored,
            self.context.num_train,
            rows,
        )
    }
}
