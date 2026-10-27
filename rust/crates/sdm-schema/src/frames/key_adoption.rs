// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Key validation/adoption products over actual selected native populations.
//! The native library owns equality/missingness and example selection.
//! Schema eligibility, thresholds, discriminator choice and notices stay here.

use sdm_contract::frames::{FrameAdoption, FrameInference, NativeKeyQuality};
use sdm_contract::ops::{DescribedTable, SourceDescription, Stype};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};

use crate::keys::KEY_UNIQUENESS;

pub struct Adoption {
    prior: FrameInference,
    table: DescribedTable,
    key: Vec<String>,
    validated: bool,
    validation: Option<NativeKeyQuality>,
    quality: Option<NativeKeyQuality>,
    adopted: bool,
    discriminators: Vec<String>,
    observed: usize,
    distinguishing: Vec<String>,
    eligible: Vec<String>,
    usable_at: usize,
    usable: Vec<String>,
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

pub fn fields(
    table: &DescribedTable,
    key: &[String],
    quality: &NativeKeyQuality,
    rows: u64,
) -> Vec<(&'static str, Detail)> {
    let mut fields = vec![
        ("table", Detail::text(&table.name)),
        ("key", Detail::text(key.join("."))),
        (
            "columns",
            Detail::List(key.iter().cloned().map(Scalar::Utf8).collect()),
        ),
        (
            "repeats",
            Detail::Scalar(Scalar::UInt64(quality.duplicates)),
        ),
        ("rows", Detail::Scalar(Scalar::UInt64(rows))),
        ("example", Detail::List(vec![Scalar::Null; key.len()])),
        ("native_quality", Detail::Scalar(Scalar::Bool(true))),
    ];
    if let Some(row) = quality.example_row {
        fields.push(("example_row", Detail::Scalar(Scalar::UInt64(row))));
    }
    fields
}

impl Adoption {
    pub fn new(prior: FrameInference) -> Result<Self, CoreError> {
        let (table, key, validated) = match &prior {
            FrameInference::Nomination(nomination) => (
                nomination.table.clone(),
                nomination.nominated_key.clone(),
                nomination.missing,
            ),
            FrameInference::Inferred(tables) if tables.len() == 1 => (
                tables[0].table.clone(),
                tables[0].table.primary_key.clone(),
                false,
            ),
            _ => {
                return Err(invalid(
                    "native key adoption requires a nominee or one edited table",
                ));
            }
        };
        Ok(Self {
            prior,
            table,
            key,
            validated,
            validation: None,
            quality: None,
            adopted: false,
            discriminators: vec![],
            observed: 0,
            distinguishing: vec![],
            eligible: vec![],
            usable_at: 0,
            usable: vec![],
        })
    }

    pub fn validation_columns(&self) -> Option<Vec<String>> {
        (!self.key.is_empty() && !self.validated).then(|| self.key.clone())
    }

    pub fn validation_empty(&mut self, empty: bool) {
        if empty {
            self.validated = true;
        }
    }

    pub fn validation_quality(&mut self, quality: NativeKeyQuality) -> Result<(), CoreError> {
        if self.validated || self.key.is_empty() || self.validation.is_some() {
            return Err(invalid(
                "native validation quality arrived outside its population",
            ));
        }
        self.validation = Some(quality);
        Ok(())
    }

    pub fn missing_columns(&self) -> Vec<String> {
        self.validation
            .as_ref()
            .filter(|quality| quality.missing > 0)
            .map(|_| self.key.clone())
            .unwrap_or_default()
    }

    pub fn validation_needs_rows(&self) -> bool {
        self.validation
            .as_ref()
            .is_some_and(|quality| quality.missing == 0)
    }

    pub fn validate(&mut self, missing: Vec<String>, rows: Option<u64>) -> Result<(), CoreError> {
        let quality = self
            .validation
            .as_ref()
            .ok_or_else(|| invalid("native validation has no completed quality"))?;
        if quality.missing > 0 {
            if rows.is_some() || missing.iter().any(|name| !self.key.contains(name)) {
                return Err(invalid(
                    "native missing-key observations do not match their population",
                ));
            }
            let mut fields = fields(&self.table, &self.key, quality, 0);
            fields.push(("missing", Detail::Scalar(Scalar::UInt64(quality.missing))));
            fields.push((
                "missing_columns",
                Detail::List(missing.iter().cloned().map(Scalar::Utf8).collect()),
            ));
            fields.push((
                "pronoun",
                Detail::text(if missing.len() > 1 { "them" } else { "it" }),
            ));
            return Err(CoreError::new(
                Sentence::FrameKeyMissing,
                "a native key holds missing values",
            )
            .with_fields(fields));
        }
        if !missing.is_empty() {
            return Err(invalid(
                "a complete native key has unexpected missing-column observations",
            ));
        }
        let rows =
            rows.ok_or_else(|| invalid("native key validation is missing its current frame size"))?;
        if quality.duplicates as f64 > rows as f64 * (1.0 - KEY_UNIQUENESS) {
            return Err(CoreError::new(
                Sentence::FrameKeyNotUnique,
                "a native key repeats too often",
            )
            .with_fields(fields(&self.table, &self.key, quality, rows)));
        }
        self.validated = true;
        Ok(())
    }

    pub fn adoption_columns(&self) -> Result<Option<Vec<String>>, CoreError> {
        if !self.key.is_empty() && !self.validated {
            return Err(invalid("native adoption preceded key validation"));
        }
        Ok((!self.key.is_empty()).then(|| self.key.clone()))
    }

    pub fn adopt(&mut self, quality: Option<NativeKeyQuality>) -> Result<(), CoreError> {
        if self.adopted || (!self.key.is_empty() && !self.validated) {
            return Err(invalid(
                "native key adoption is outside its completed validation",
            ));
        }
        if self.key.is_empty() && quality.is_some() {
            return Err(invalid("a keyless table has native adoption quality"));
        }
        self.adopted = true;
        self.discriminators = if quality
            .as_ref()
            .is_some_and(|quality| quality.duplicates > 0)
        {
            self.table
                .columns
                .iter()
                .filter(|column| !self.key.contains(&column.name))
                .map(|column| column.name.clone())
                .collect()
        } else {
            vec![]
        };
        self.quality = quality;
        Ok(())
    }

    pub fn discriminators(&self) -> Result<Vec<String>, CoreError> {
        if !self.adopted {
            return Err(invalid("native discriminators preceded adoption quality"));
        }
        Ok(self.discriminators.clone())
    }

    pub fn discriminator(
        &mut self,
        column: String,
        quality: NativeKeyQuality,
    ) -> Result<(), CoreError> {
        if self.discriminators.get(self.observed) != Some(&column) {
            return Err(invalid(
                "native discriminator observations are out of source order",
            ));
        }
        self.observed += 1;
        if quality.duplicates == 0 {
            self.distinguishing.push(column);
        }
        Ok(())
    }

    pub fn usable_columns(&mut self) -> Result<Vec<String>, CoreError> {
        if self.observed != self.discriminators.len() {
            return Err(invalid("native discriminator populations are incomplete"));
        }
        self.eligible = self
            .distinguishing
            .iter()
            .filter(|name| {
                self.table
                    .columns
                    .iter()
                    .find(|column| &column.name == *name)
                    .is_some_and(|column| {
                        crate::assemble::can_identify(column.dtype)
                            && column.stype != Stype::Timestamp
                    })
            })
            .cloned()
            .collect();
        Ok(self.eligible.clone())
    }

    pub fn usable(&mut self, column: String, missing: Option<bool>) -> Result<(), CoreError> {
        if self.eligible.get(self.usable_at) != Some(&column) {
            return Err(invalid("native usable-key observations are out of order"));
        }
        self.usable_at += 1;
        if missing != Some(true) {
            self.usable.push(column);
        }
        Ok(())
    }

    pub fn needs_rows(&self) -> bool {
        self.quality
            .as_ref()
            .is_some_and(|quality| quality.duplicates > 0)
    }

    pub fn finish(
        mut self,
        rows: Option<u64>,
    ) -> Result<(FrameInference, Vec<CoreWarning>), CoreError> {
        if !self.adopted
            || self.observed != self.discriminators.len()
            || self.usable_at != self.eligible.len()
        {
            return Err(invalid("native key adoption populations are incomplete"));
        }
        let mut notices = vec![];
        if let Some(quality) = &self.quality
            && quality.duplicates > 0
        {
            let rows =
                rows.ok_or_else(|| invalid("native repeat advice has no reached frame size"))?;
            let (sentence, others) = if !self.usable.is_empty() {
                (Sentence::FrameKeyRepeatsComposite, &self.usable)
            } else if !self.distinguishing.is_empty() {
                (Sentence::FrameKeyRepeatsUnusable, &self.distinguishing)
            } else {
                (Sentence::FrameKeyRepeats, &self.distinguishing)
            };
            let mut fields = fields(&self.table, &self.key, quality, rows);
            fields.push((
                "others",
                Detail::List(others.iter().cloned().map(Scalar::Utf8).collect()),
            ));
            notices.push(CoreWarning::new(sentence, "a native key repeats").with_fields(fields));
        }
        self.table.primary_key = self.key;
        if let Some(quality) = &self.quality {
            self.table.duplicate_keys = Some(Scalar::UInt64(quality.duplicates));
        } else if self.table.primary_key.is_empty() {
            self.table.duplicate_keys = Some(Scalar::UInt64(0));
        }
        if matches!(&self.prior, FrameInference::Nomination(_))
            && !self.table.primary_key.is_empty()
        {
            self.table.frame_key_form = Some(sdm_contract::frames::FrameKeyForm::Scalar);
        }
        let mut description = SourceDescription {
            tables: vec![self.table],
            links: vec![],
        };
        crate::assemble::mark_keys(&mut description);
        let table = description.tables.remove(0);
        let product = match self.prior {
            FrameInference::Nomination(_) => {
                FrameInference::Adoption(Box::new(FrameAdoption { table }))
            }
            FrameInference::Inferred(mut values) => {
                values[0].table = table;
                FrameInference::Inferred(values)
            }
            _ => return Err(invalid("native key adoption lost its input product")),
        };
        Ok((product, notices))
    }
}
