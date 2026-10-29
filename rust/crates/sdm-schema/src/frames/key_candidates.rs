// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered eligible key populations after all native column meanings.
//! Native cardinality is observed only for the current population; the
//! uniqueness rule and stop-after-first selection remain in the schema core.

use sdm_contract::frames::FrameNomination;
use sdm_contract::ops::DescribedTable;
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};

use crate::keys::KEY_UNIQUENESS;

pub struct Candidates {
    table: DescribedTable,
    rows: u64,
    candidates: Vec<String>,
    current: usize,
    selected: Option<String>,
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

impl Candidates {
    pub fn new(table: DescribedTable, rows: u64) -> Self {
        let candidates = if rows == 0 {
            vec![]
        } else {
            table
                .columns
                .iter()
                .filter(|column| {
                    let lowered = column.name.to_lowercase();
                    ["_id", "_key", "id", "key"]
                        .iter()
                        .any(|suffix| lowered.ends_with(suffix))
                        && crate::assemble::can_identify(column.dtype)
                })
                .map(|column| column.name.clone())
                .collect()
        };
        Self {
            table,
            rows,
            candidates,
            current: 0,
            selected: None,
        }
    }

    pub fn column(&self) -> Option<&str> {
        if self.selected.is_some() {
            None
        } else {
            self.candidates.get(self.current).map(String::as_str)
        }
    }

    pub fn observe(&mut self, distinct: u64, rows: u64) -> Result<(), CoreError> {
        self.rows = rows;
        let candidate = self
            .column()
            .ok_or_else(|| invalid("no native key population was selected"))?
            .to_owned();
        if distinct as f64 >= self.rows as f64 * KEY_UNIQUENESS {
            self.selected = Some(candidate);
        } else {
            self.current += 1;
        }
        Ok(())
    }

    pub fn selected(&self) -> Result<Option<&str>, CoreError> {
        if self.column().is_some() {
            return Err(invalid("native key populations have not been completed"));
        }
        Ok(self.selected.as_deref())
    }

    pub fn finish(
        self,
        missing: Option<(bool, u64)>,
    ) -> Result<(FrameNomination, Vec<CoreWarning>), CoreError> {
        let selected = self.selected()?.map(str::to_owned);
        if selected.is_some() != missing.is_some() {
            return Err(invalid(
                "native nominee missingness does not match its population",
            ));
        }
        let mut notices = vec![];
        if let (Some(name), Some((true, count))) = (&selected, missing) {
            notices.push(
                CoreWarning::new(
                    Sentence::KeyNoValueColumn,
                    "an inferred key has missing values",
                )
                .with_fields(vec![
                    ("table", Detail::text(&self.table.name)),
                    ("column", Detail::text(name)),
                    ("missing", Detail::Scalar(Scalar::UInt64(count))),
                ]),
            );
        }
        Ok((
            FrameNomination {
                table: self.table,
                nominated_key: selected.into_iter().collect(),
                missing: missing.is_some_and(|(any, _)| any),
            },
            notices,
        ))
    }
}
