// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed native constructor meaning from reached whole-sample statistics.
//! No native owner, callback or program counter crosses this boundary.

use sdm_contract::frames::materialize::{
    FrameConstructedColumn, FrameConstructorValues, FrameElementContent,
};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::{ColumnType, CoreError, Sentence};

use crate::{infer, names};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pending {
    Timestamp,
    Separated,
    Elements,
    Split,
    Distinct,
    Population(u64),
    Complete(Stype),
}

/// One useful fixed column meaning, with only its finite native facts pending.
#[derive(Debug)]
pub struct Constructor {
    name: String,
    dtype: Dtype,
    pending: Pending,
}

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "native constructed column observations contradict their typed meaning",
    )
}

/// The one canonical spelling emitted by admitted native type inference.
pub fn canonical(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Bool
            | Dtype::Int
            | Dtype::Float
            | Dtype::String
            | Dtype::Binary
            | Dtype::Date
            | Dtype::Timedelta
    )
}

pub fn dtype(values: &FrameConstructorValues) -> Result<Dtype, CoreError> {
    use ColumnType as T;
    Ok(match values {
        FrameConstructorValues::Enumerated | FrameConstructorValues::Text => Dtype::String,
        FrameConstructorValues::Temporal => Dtype::Date,
        FrameConstructorValues::Duration => Dtype::Timedelta,
        FrameConstructorValues::Elements { content } => match content {
            FrameElementContent::Integer => Dtype::Int,
            FrameElementContent::Real => Dtype::Float,
            FrameElementContent::Boolean => Dtype::Bool,
            FrameElementContent::Duration => Dtype::Timedelta,
            FrameElementContent::Other => Dtype::String,
        },
        FrameConstructorValues::Typed { physical } => match physical {
            T::Bool => Dtype::Bool,
            T::Int8
            | T::Int16
            | T::Int32
            | T::Int64
            | T::Uint8
            | T::Uint16
            | T::Uint32
            | T::Uint64 => Dtype::Int,
            T::Float32 | T::Float64 => Dtype::Float,
            T::Utf8 | T::Null => Dtype::String,
            T::Binary => Dtype::Binary,
            T::Date32 | T::Timestamp { .. } | T::Time64 { .. } => Dtype::Date,
            T::Duration { .. } => Dtype::Timedelta,
            _ => return Err(invalid()),
        },
    })
}

impl Constructor {
    pub fn new(name: String, dtype: Dtype) -> Self {
        let pending = if matches!(
            dtype,
            Dtype::Int | Dtype::Float | Dtype::String | Dtype::Binary
        ) && names::says_id(&name)
        {
            Pending::Complete(Stype::Id)
        } else {
            match dtype {
                Dtype::Date | Dtype::Time => Pending::Complete(Stype::Timestamp),
                Dtype::String => Pending::Timestamp,
                Dtype::Bool => Pending::Complete(Stype::Categorical),
                Dtype::Int | Dtype::Float | Dtype::Timedelta if names::says_measure(&name) => {
                    Pending::Complete(Stype::Numerical)
                }
                Dtype::Int | Dtype::Float | Dtype::Timedelta => Pending::Distinct,
                Dtype::Binary => Pending::Elements,
                Dtype::Stringlist => Pending::Complete(Stype::Multicategorical),
                Dtype::Intlist | Dtype::Floatlist => Pending::Complete(Stype::Sequence),
                _ => Pending::Complete(Stype::Unsupported),
            }
        };
        Self {
            name,
            dtype,
            pending,
        }
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn timestamp_sample(&self) -> bool {
        self.pending == Pending::Timestamp
    }
    pub fn separated_sample(&self) -> bool {
        self.pending == Pending::Separated
    }
    pub fn element_sample(&self) -> bool {
        self.pending == Pending::Elements
    }

    pub fn element_count(&mut self, _parts: u64) -> Result<(), CoreError> {
        if !self.element_sample() {
            return Err(invalid());
        }
        self.pending = Pending::Distinct;
        Ok(())
    }

    pub fn distinct_sample(&self) -> bool {
        self.pending == Pending::Distinct
    }

    pub fn timestamp_counts(&mut self, parsed: u64, present: u64) -> Result<(), CoreError> {
        if !self.timestamp_sample() || parsed > present {
            return Err(invalid());
        }
        self.pending = if infer::native_timestamp_share(&self.name, parsed, present) {
            Pending::Complete(Stype::Timestamp)
        } else {
            Pending::Separated
        };
        Ok(())
    }

    /// Select from actual native joined Unicode points, including tie order.
    pub fn separator(&mut self, points: &[u32]) -> Result<Option<char>, CoreError> {
        if !self.separated_sample() || points.iter().any(|point| *point > 0x10ffff) {
            return Err(invalid());
        }
        let mut counts: Vec<(char, usize)> = Vec::new();
        for point in points {
            let Some(c) = char::from_u32(*point).filter(|c| matches!(c, ';' | ':' | '|' | '\t'))
            else {
                continue;
            };
            match counts.iter_mut().find(|(seen, _)| *seen == c) {
                Some((_, count)) => *count += 1,
                None => counts.push((c, 1)),
            }
        }
        let selected = counts
            .into_iter()
            .fold(None, |best: Option<(char, usize)>, item| match best {
                Some((_, count)) if count >= item.1 => best,
                _ => Some(item),
            })
            .map(|(c, _)| c);
        self.pending = if selected.is_some() {
            Pending::Split
        } else {
            Pending::Distinct
        };
        Ok(selected)
    }

    pub fn separated_counts(&mut self, whole: u64, parts: u64) -> Result<(), CoreError> {
        if self.pending != Pending::Split {
            return Err(invalid());
        }
        self.pending = if infer::native_multicategory_share(whole, parts) {
            Pending::Complete(Stype::Multicategorical)
        } else {
            Pending::Distinct
        };
        Ok(())
    }

    /// True only when the original threshold reaches its later length read.
    pub fn distinct_count(&mut self, distinct: u64) -> Result<bool, CoreError> {
        if !self.distinct_sample() {
            return Err(invalid());
        }
        self.pending = if distinct < 20 {
            Pending::Complete(Stype::Categorical)
        } else {
            Pending::Population(distinct)
        };
        Ok(distinct >= 20)
    }

    pub fn population_count(&mut self, present: u64) -> Result<(), CoreError> {
        let Pending::Population(distinct) = self.pending else {
            return Err(invalid());
        };
        if distinct > present {
            return Err(invalid());
        }
        self.pending = Pending::Complete(
            if infer::native_categorical_share(self.dtype, distinct, present) {
                Stype::Categorical
            } else {
                match self.dtype {
                    Dtype::String => Stype::Text,
                    Dtype::Binary => Stype::Categorical,
                    _ => Stype::Numerical,
                }
            },
        );
        Ok(())
    }

    pub fn meaning(&self) -> Option<Stype> {
        match self.pending {
            Pending::Complete(stype) => Some(stype),
            _ => None,
        }
    }

    pub fn complete(&self) -> Result<FrameConstructedColumn, CoreError> {
        Ok(FrameConstructedColumn {
            name: self.name.clone(),
            dtype: self.dtype,
            stype: self.meaning().ok_or_else(invalid)?,
        })
    }
}
