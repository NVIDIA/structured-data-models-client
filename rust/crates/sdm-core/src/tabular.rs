// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tabular materialization composes row selection with the existing model cap.

use sdm_contract::CoreError;

pub use sdm_engine::tabular::{
    IndexSelection as TabularIndexSelection, labelled_rows as tabular_labelled_rows,
    matched_rows as tabular_matched_rows,
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
    pub fn select(&self) -> Vec<usize> {
        self.selection.select()
    }
    pub fn positions(&self) -> Option<Vec<usize>> {
        self.selection.reduced().then(|| self.selection.positions())
    }
}
