// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Final keyless-advice populations after native date presentation.
//! Native reads occur only for the populations selected here, preserving
//! their order even when a count or warning changes later source data.

use std::collections::{BTreeMap, BTreeSet};

use sdm_contract::frames::{FrameKeyAdvicePlan, FrameSample, NativeKeyQuality};
use sdm_contract::ops::{DescribedTable, Dtype, Stype};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};

use super::key_adoption;

pub struct Advice {
    plan: FrameKeyAdvicePlan,
    dates: BTreeMap<String, String>,
    named: BTreeMap<String, BTreeSet<String>>,
    keyed: BTreeMap<String, Vec<String>>,
    spoken: BTreeSet<String>,
    declined_table: Option<String>,
    declined_columns: Vec<String>,
    declined_at: usize,
    pending_quality: bool,
    pending_notice: Option<NativeKeyQuality>,
    keyless: BTreeMap<String, Vec<String>>,
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

fn texts(values: &[String]) -> Detail {
    Detail::List(values.iter().cloned().map(Scalar::Utf8).collect())
}

impl Advice {
    pub fn new(plan: FrameKeyAdvicePlan, dates: BTreeMap<String, String>) -> Self {
        let mut named = BTreeMap::<String, BTreeSet<String>>::new();
        let mut keyed = BTreeMap::<String, Vec<String>>::new();
        for table in &plan.description.tables {
            for column in &table.columns {
                named
                    .entry(column.name.clone())
                    .or_default()
                    .insert(table.name.clone());
            }
            if table.frame_key_form != Some(sdm_contract::frames::FrameKeyForm::Sequence)
                && let [key] = table.primary_key.as_slice()
            {
                keyed
                    .entry(key.clone())
                    .or_default()
                    .push(table.name.clone());
            }
        }
        Self {
            plan,
            dates,
            named,
            keyed,
            spoken: BTreeSet::new(),
            declined_table: None,
            declined_columns: vec![],
            declined_at: 0,
            pending_quality: false,
            pending_notice: None,
            keyless: BTreeMap::new(),
        }
    }

    fn table(&self, name: &str) -> Result<&DescribedTable, CoreError> {
        self.plan
            .description
            .tables
            .iter()
            .find(|table| table.name == name)
            .ok_or_else(|| invalid("native advice names an unknown table"))
    }

    pub fn declined_tables(&self) -> Vec<String> {
        if !self.plan.infer_links {
            return vec![];
        }
        self.plan
            .description
            .tables
            .iter()
            .filter(|table| table.primary_key.is_empty())
            .map(|table| table.name.clone())
            .collect()
    }

    pub fn begin_declined(&mut self, name: String, empty: bool) -> Result<(), CoreError> {
        let table = self.table(&name)?;
        if !self.plan.infer_links || !table.primary_key.is_empty() {
            return Err(invalid(
                "native declined-key advice names an ineligible table",
            ));
        }
        self.declined_columns = if empty {
            vec![]
        } else {
            table
                .columns
                .iter()
                .filter(|column| {
                    ["_id", "_key", "id", "key"]
                        .iter()
                        .any(|suffix| column.name.to_lowercase().ends_with(suffix))
                        && crate::assemble::can_identify(column.dtype)
                        && !self.keyed.contains_key(&column.name)
                        && self
                            .named
                            .get(&column.name)
                            .is_some_and(|tables| tables.iter().any(|other| other != &name))
                })
                .map(|column| column.name.clone())
                .collect()
        };
        self.declined_table = Some(name);
        self.declined_at = 0;
        self.pending_quality = false;
        self.pending_notice = None;
        Ok(())
    }

    pub fn declined_column(&self) -> Option<String> {
        self.declined_columns.get(self.declined_at).cloned()
    }

    pub fn declined_distinct(&mut self, distinct: u64, rows: u64) -> Result<bool, CoreError> {
        if self.declined_column().is_none() || self.pending_quality || self.pending_notice.is_some()
        {
            return Err(invalid(
                "native declined-key cardinality is outside its population",
            ));
        }
        self.pending_quality = distinct as f64 >= rows as f64 * crate::keys::NEARLY_A_KEY;
        if !self.pending_quality {
            self.declined_at += 1;
        }
        Ok(self.pending_quality)
    }

    pub fn declined_quality(&mut self, quality: NativeKeyQuality) -> Result<bool, CoreError> {
        if !self.pending_quality {
            return Err(invalid("native declined-key quality was not selected"));
        }
        self.pending_quality = false;
        if quality.duplicates > 0 {
            self.pending_notice = Some(quality);
            Ok(true)
        } else {
            self.declined_at += 1;
            Ok(false)
        }
    }

    pub fn declined_notice(&mut self, rows: u64) -> Result<Vec<CoreWarning>, CoreError> {
        let quality = self
            .pending_notice
            .take()
            .ok_or_else(|| invalid("native declined-key advice has no observed repeats"))?;
        let table_name = self
            .declined_table
            .as_ref()
            .ok_or_else(|| invalid("native declined-key advice lost its table"))?
            .clone();
        let column = self
            .declined_column()
            .ok_or_else(|| invalid("native declined-key advice lost its column"))?;
        let mut fields = key_adoption::fields(
            self.table(&table_name)?,
            std::slice::from_ref(&column),
            &quality,
            rows,
        );
        fields.push(("column", Detail::text(&column)));
        self.spoken.insert(table_name);
        self.declined_at = self.declined_columns.len();
        Ok(vec![
            CoreWarning::new(
                Sentence::FrameKeyDeclined,
                "a native key-shaped column repeats",
            )
            .with_fields(fields),
        ])
    }

    pub fn keyless_tables(&self) -> Vec<String> {
        self.declined_tables()
            .into_iter()
            .filter(|name| !self.spoken.contains(name))
            .collect()
    }

    pub fn keyless_columns(&mut self, name: String, empty: bool) -> Result<Vec<String>, CoreError> {
        let table = self.table(&name)?;
        if !self.plan.infer_links || !table.primary_key.is_empty() || self.spoken.contains(&name) {
            return Err(invalid("native keyless advice names an ineligible table"));
        }
        let columns = if empty {
            vec![]
        } else {
            table
                .columns
                .iter()
                .filter(|column| {
                    Some(&column.name) != table.time_column.as_ref()
                        && matches!(
                            column.dtype,
                            Dtype::String
                                | Dtype::Int
                                | Dtype::Byte
                                | Dtype::Int16
                                | Dtype::Int32
                                | Dtype::Int64
                        )
                })
                .map(|column| column.name.clone())
                .collect()
        };
        self.keyless.insert(name, vec![]);
        Ok(columns)
    }

    pub fn keyless_missing(&self, missing: bool) -> bool {
        !missing
    }

    pub fn keyless_quality(
        &mut self,
        table: String,
        column: String,
        quality: NativeKeyQuality,
    ) -> Result<(), CoreError> {
        self.table(&table)?;
        if quality.duplicates == 0
            && self
                .named
                .get(&column)
                .is_some_and(|tables| tables.len() > 1)
            && !self.keyed.contains_key(&column)
            && self.dates.get(&table) != Some(&column)
        {
            self.keyless
                .get_mut(&table)
                .ok_or_else(|| invalid("native keyless advice has no current population"))?
                .push(column);
        }
        Ok(())
    }

    pub fn keyless_notice(&mut self, name: String) -> Result<Vec<CoreWarning>, CoreError> {
        let candidates = self
            .keyless
            .remove(&name)
            .ok_or_else(|| invalid("native keyless advice has no completed population"))?;
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let elsewhere: BTreeSet<_> = candidates
            .iter()
            .flat_map(|column| self.named[column].iter())
            .filter(|other| *other != &name)
            .collect();
        let many = candidates.len() > 1;
        let shared = elsewhere.len() > 1;
        let fields = vec![
            ("table", Detail::text(name)),
            ("candidates", texts(&candidates)),
            (
                "identify",
                Detail::text(if many { "identify" } else { "identifies" }),
            ),
            ("verb", Detail::text(if many { "are" } else { "is" })),
            (
                "elsewhere",
                Detail::text(if shared {
                    "other tables"
                } else {
                    "another table"
                }),
            ),
            (
                "each",
                Detail::text(if many { "each of them" } else { "it" }),
            ),
            (
                "identifies",
                Detail::text(if many {
                    "one of them identifies"
                } else {
                    "it identifies"
                }),
            ),
            ("subject", Detail::text(if many { "they" } else { "it" })),
            (
                "name_verb",
                Detail::text(if many { "name" } else { "names" }),
            ),
            (
                "tables_word",
                Detail::text(if shared { "tables" } else { "table" }),
            ),
            (
                "need_key",
                Detail::text(if shared {
                    "those tables are the ones that need"
                } else {
                    "that table is the one that needs"
                }),
            ),
        ];
        Ok(vec![
            CoreWarning::new(
                Sentence::FrameKeylessShared,
                "a native keyless table is named elsewhere",
            )
            .with_fields(fields),
        ])
    }

    pub fn ambiguous_notices(&self) -> Vec<CoreWarning> {
        if !self.plan.infer_links {
            return vec![];
        }
        self.keyed
            .iter()
            .filter_map(|(key, names)| {
                if names.len() < 2 {
                    return None;
                }
                let mut names = names.clone();
                names.sort();
                Some(
                    CoreWarning::new(
                        Sentence::FrameKeyAmbiguous,
                        "a native key name is ambiguous",
                    )
                    .with_fields(vec![
                        ("key", Detail::text(key)),
                        ("tables", texts(&names)),
                        ("first", Detail::text(&names[0])),
                    ]),
                )
            })
            .collect()
    }
}

pub fn sample(sample: FrameSample) -> Result<Vec<CoreWarning>, CoreError> {
    if sample.column.column.dtype != Dtype::String
        || sample.column.column.stype != Stype::Categorical
    {
        return Err(invalid(
            "native sample facts require an original categorical string column",
        ));
    }
    Ok(if sample.exists {
        super::sample_notice(
            &sample.column.table,
            &sample.column.column.name,
            sample.distinct,
            sample.present,
        )
        .into_iter()
        .collect()
    } else {
        vec![]
    })
}
