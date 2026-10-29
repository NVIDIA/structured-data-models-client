// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tabular materialization composes row selection with the existing model cap.

use sdm_contract::CoreError;

pub use sdm_engine::tabular::task::{
    Number as TabularTaskNumber, TaskMeaning as TabularTaskMeaning,
    TaskNumbers as TabularTaskNumbers, Vocabulary as TabularTaskVocabulary,
    vocabulary as tabular_task_vocabulary,
};
pub use sdm_engine::tabular::{
    IndexSelection as TabularIndexSelection, labelled_rows as tabular_labelled_rows,
};

#[derive(Debug, Clone)]
pub struct TabularContext {
    selection: sdm_engine::tabular::ContextSelection,
}

impl TabularContext {
    pub fn new(candidates: Vec<usize>, requested: Option<usize>) -> Result<Self, CoreError> {
        Ok(Self {
            selection: sdm_engine::tabular::ContextSelection::new(
                candidates,
                requested,
                sdm_request::predict::CONTEXT_ROWS_MAX,
            )?,
        })
    }

    pub fn notice_rows(&self) -> Option<usize> {
        self.selection.notice_rows()
    }
    pub fn wanted(&self) -> usize {
        self.selection.wanted()
    }
    pub fn reduced(&self) -> bool {
        self.selection.reduced()
    }
    pub fn select(&self, labels: Option<&[i64]>) -> Result<Vec<usize>, CoreError> {
        self.selection.select(labels)
    }
    pub fn positions(&self, labels: Option<&[i64]>) -> Result<Option<Vec<usize>>, CoreError> {
        self.selection.positions(labels)
    }
}

/// Scoring identities precede source values, including the empty-set shortcut.
#[derive(Debug, Clone)]
pub struct TabularScoringKeys {
    keys: sdm_engine::tabular::ScoringKeys,
}

impl TabularScoringKeys {
    pub fn new(keys: Vec<i64>) -> Self {
        Self {
            keys: sdm_engine::tabular::ScoringKeys::new(keys),
        }
    }
    pub fn needs_source(&self) -> bool {
        self.keys.needs_source()
    }
    pub fn source(&self, keys: &[i64]) -> Result<TabularPopulation, CoreError> {
        if !self.needs_source() {
            return Err(CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "empty scoring identities do not consume source membership",
            ));
        }
        Ok(TabularPopulation {
            rows: self.keys.retain(keys),
        })
    }
}

/// Original source positions retained by an actual supplied-frame holdout.
/// Its next native target mask is defined over exactly this surviving extent.
#[derive(Debug, Clone)]
pub struct TabularPopulation {
    rows: Vec<usize>,
}

impl TabularPopulation {
    pub fn positions(&self) -> Vec<usize> {
        self.rows.clone()
    }
    pub fn labelled_rows(&self, known: &[bool]) -> Result<Vec<usize>, CoreError> {
        if known.len() != self.rows.len() {
            return Err(CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "tabular target mask differs from its held-out native population",
            ));
        }
        tabular_labelled_rows(known, &[])
    }
    pub fn context(
        &self,
        candidates: &[usize],
        requested: Option<usize>,
    ) -> Result<TabularContext, CoreError> {
        let original: Vec<usize> = candidates
            .iter()
            .map(|&position| {
                self.rows.get(position).copied().ok_or_else(|| {
                    CoreError::protocol(
                        sdm_contract::Sentence::InvalidMessage,
                        "tabular candidate lies outside its held-out native population",
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(TabularContext {
            selection: sdm_engine::tabular::ContextSelection::new(
                original,
                requested,
                sdm_request::predict::CONTEXT_ROWS_MAX,
            )?,
        })
    }
}
