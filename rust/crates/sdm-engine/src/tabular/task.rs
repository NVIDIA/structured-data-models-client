// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PQL tabular task meaning from actual scoped native observations.

use std::collections::BTreeMap;

use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};

fn invalid(message: &'static str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// Vocabulary consumed by the native set/hash/equality observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vocabulary {
    pub stypes: Vec<String>,
    pub tasks: Vec<String>,
    pub categorical: String,
}

pub fn vocabulary() -> Vocabulary {
    Vocabulary {
        stypes: vec!["categorical".into(), "numerical".into()],
        tasks: vec!["classification".into(), "regression".into()],
        categorical: "categorical".into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Meaning {
    Target,
    Stated,
    Absent,
    Categorical,
    Numerical,
}

/// Admitted target and actual optional stated-task membership meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskMeaning {
    meaning: Meaning,
}

impl TaskMeaning {
    pub fn new(target_supported: bool) -> Result<Self, CoreError> {
        if !target_supported {
            let mut error = CoreError::new(
                Sentence::TabularTargetStype,
                "the tabular target semantic type is not supported",
            )
            .with_field("column", Detail::Scalar(Scalar::Null))
            .with_field("stype", Detail::Scalar(Scalar::Null));
            error.public_details = Some(Box::new(BTreeMap::new()));
            return Err(error);
        }
        Ok(Self {
            meaning: Meaning::Target,
        })
    }

    /// Facts are absent when their native operation was not reached.
    pub fn stated(
        &self,
        present: bool,
        string: Option<bool>,
        member: Option<bool>,
    ) -> Result<Self, CoreError> {
        if self.meaning != Meaning::Target {
            return Err(invalid(
                "tabular stated-task facts require an admitted target",
            ));
        }
        match (present, string, member) {
            (false, None, None) => {
                return Ok(Self {
                    meaning: Meaning::Absent,
                });
            }
            (true, Some(false), None) | (true, Some(true), Some(_)) => {}
            _ => {
                return Err(invalid(
                    "tabular stated-task facts do not describe reached native membership",
                ));
            }
        }
        if string != Some(true) || member != Some(true) {
            let supported =
                Detail::List(vocabulary().tasks.into_iter().map(Scalar::Utf8).collect());
            let mut error = CoreError::new(
                Sentence::ModelLacksTask,
                "the tabular model does not serve the stated task kind",
            )
            .with_field("model", Detail::text("the tabular model"))
            .with_field("task", Detail::Scalar(Scalar::Null))
            .with_field("supported", supported.clone())
            .with_field("native_model_task", Detail::Scalar(Scalar::Bool(true)))
            .with_field("native_tabular_task", Detail::Scalar(Scalar::Bool(true)));
            error.public_details = Some(Box::new(BTreeMap::from([
                ("task".into(), Detail::Scalar(Scalar::Null)),
                ("supported".into(), supported),
            ])));
            return Err(error);
        }
        Ok(Self {
            meaning: Meaning::Stated,
        })
    }

    pub fn inferred(&self, categorical: bool) -> Result<Self, CoreError> {
        if self.meaning != Meaning::Absent {
            return Err(invalid(
                "only an absent tabular task observes target inference",
            ));
        }
        Ok(Self {
            meaning: if categorical {
                Meaning::Categorical
            } else {
                Meaning::Numerical
            },
        })
    }

    pub fn original(&self) -> bool {
        self.meaning == Meaning::Stated
    }

    pub fn kind(&self) -> Option<&'static str> {
        (self.meaning == Meaning::Categorical).then_some("classification")
    }

    pub fn numbers(&self, cardinality: usize) -> Result<TaskNumbers, CoreError> {
        if self.meaning != Meaning::Numerical {
            return Err(invalid(
                "this tabular task does not observe native distinct numbers",
            ));
        }
        Ok(TaskNumbers {
            cardinality,
            observed: 0,
            end: if cardinality == 2 {
                End::Open
            } else {
                End::Nonwhole
            },
        })
    }
}

/// Completed data from one reached native number observation, without its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    Boolean,
    Integral,
    Decimal {
        finite: bool,
        equal: Option<bool>,
        returned: Option<bool>,
    },
    RealValue {
        bits: [u8; 8],
    },
    RealFailure {
        overflow_error: bool,
        type_error: bool,
        value_error: bool,
    },
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    Open,
    Nonwhole,
    Exhausted,
    NativeFailure,
}

/// Native distinct cardinality and its actually consumed ordered numeric prefix.
/// A reported length of two does not authenticate how many an iterator yields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskNumbers {
    cardinality: usize,
    observed: usize,
    end: End,
}

impl TaskNumbers {
    fn open(&self, ordinal: usize) -> Result<(), CoreError> {
        if self.cardinality != 2 || self.end != End::Open || ordinal != self.observed {
            return Err(invalid(
                "tabular numeric observation does not extend its complete native prefix",
            ));
        }
        Ok(())
    }

    pub fn number(&self, ordinal: usize, value: Number) -> Result<Self, CoreError> {
        self.open(ordinal)?;
        let end = match &value {
            Number::Boolean | Number::Integral => End::Open,
            Number::Decimal {
                finite: true,
                equal: Some(equal),
                returned: None,
            } => {
                if *equal {
                    End::Open
                } else {
                    End::Nonwhole
                }
            }
            Number::Decimal {
                finite: false,
                equal: None,
                returned: Some(truth),
            } => {
                if *truth {
                    End::Open
                } else {
                    End::Nonwhole
                }
            }
            Number::Decimal { .. } => {
                return Err(invalid(
                    "tabular Decimal facts include missing or unreached native observations",
                ));
            }
            Number::RealValue { bits } => {
                let value = f64::from_le_bytes(*bits);
                if value.is_finite() && value.fract() == 0.0 {
                    End::Open
                } else {
                    End::Nonwhole
                }
            }
            Number::RealFailure {
                overflow_error,
                type_error,
                value_error,
            } => {
                if *overflow_error || *type_error || *value_error {
                    End::Nonwhole
                } else {
                    End::NativeFailure
                }
            }
            Number::Other => End::Nonwhole,
        };
        let mut next = self.clone();
        next.observed = self
            .observed
            .checked_add(1)
            .ok_or_else(|| invalid("tabular native numeric prefix exceeds the host extent"))?;
        next.end = end;
        Ok(next)
    }

    pub fn exhausted(&self, ordinal: usize) -> Result<Self, CoreError> {
        self.open(ordinal)?;
        let mut next = self.clone();
        next.end = End::Exhausted;
        Ok(next)
    }

    pub fn propagates(&self) -> bool {
        self.end == End::NativeFailure
    }

    pub fn kind(&self) -> Option<&'static str> {
        match self.end {
            End::Nonwhole => Some("regression"),
            End::Exhausted => Some("classification"),
            End::Open | End::NativeFailure => None,
        }
    }

    pub fn notice(&self) -> Option<CoreWarning> {
        (self.end == End::Exhausted).then(|| {
            CoreWarning::new(
                Sentence::TabularBinaryLabel,
                "two whole target values select tabular classification",
            )
        })
    }
}
