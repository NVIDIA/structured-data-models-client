// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core-owned meaning inference from actual native scalar observations.

use sdm_contract::frames::NativeObjectMeaning;
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Sentence};

pub fn check_cell(described: &str, kind: &str, native_type: &str) -> Result<(), CoreError> {
    use sdm_contract::Detail;
    let (sentence, shape) = match kind {
        "scalar" => return Ok(()),
        "bytes" => (Sentence::FrameCellBytes, ""),
        "mapping" => (Sentence::FrameCellContainer, "a mapping"),
        "container" => (Sentence::FrameCellContainer, "a container"),
        _ => return Err(invalid("native cell classification has an unknown kind")),
    };
    Err(
        CoreError::new(sentence, "a native source cell is not scalar").with_fields(vec![
            ("described", Detail::text(described)),
            ("native_type", Detail::text(native_type)),
            ("shape", Detail::text(shape)),
        ]),
    )
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// The next native source observation selected by constructor inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Content,
    Missing(usize),
    IsDecimal(usize),
    WithinIntegerRange(usize),
    Finite(usize),
    NegativeExponent(usize),
    Fractional(usize),
    Wide(usize),
    AtLeastMinimum(usize),
    AtMostMaximum(usize),
    Distinct,
    Complete,
}

/// Content classification is native pandas information, not a wire dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Content {
    Decimal,
    Empty,
    Integer,
    Floating,
    Boolean,
    Date,
    Other,
}

/// The inferred object-Decimal meaning, fixed by the core's traversal.
pub type Meaning = NativeObjectMeaning;

/// A single consumed column. A caller must retain that column's native
/// iterator while answering selected observations; each next column is
/// resolved only after this traversal and its native effects finish.
pub struct Constructor {
    rows: usize,
    next: Read,
    wide: bool,
    meaning: Option<Meaning>,
    integer: bool,
}

impl Constructor {
    pub fn new(rows: usize) -> Self {
        Self {
            rows,
            next: Read::Content,
            wide: false,
            meaning: None,
            integer: false,
        }
    }

    pub fn next(&self) -> Read {
        self.next
    }

    pub fn meaning(&self) -> Option<Meaning> {
        if self.next == Read::Complete {
            self.meaning
        } else {
            None
        }
    }

    pub fn content(&mut self, content: Content) -> Result<(), CoreError> {
        if self.next != Read::Content {
            return Err(invalid(
                "native object content arrived outside its selected read",
            ));
        }
        match content {
            Content::Other => {
                self.finish(Dtype::String, false);
            }
            Content::Floating => {
                self.finish(Dtype::Float64, false);
            }
            Content::Boolean => {
                self.finish(Dtype::Bool, false);
            }
            Content::Date => {
                self.finish(Dtype::Date, false);
            }
            Content::Integer => {
                self.integer = true;
                if self.rows == 0 {
                    self.finish(Dtype::Int64, false);
                } else {
                    self.next = Read::Missing(0);
                }
            }
            Content::Empty => {
                self.finish(Dtype::String, true);
            }
            Content::Decimal => {
                if self.rows == 0 {
                    self.finish(Dtype::Int64, false);
                } else {
                    self.next = Read::Missing(0);
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, dtype: Dtype, untyped: bool) {
        self.meaning = Some(Meaning {
            dtype,
            untyped,
            distinct: None,
        });
        self.next = if dtype == Dtype::String {
            Read::Distinct
        } else {
            Read::Complete
        };
    }

    pub fn distinct(&mut self, distinct: u64) -> Result<(), CoreError> {
        if self.next != Read::Distinct {
            return Err(invalid(
                "native Decimal cardinality arrived outside its selected read",
            ));
        }
        self.meaning
            .as_mut()
            .ok_or_else(|| invalid("native Decimal cardinality has no meaning"))?
            .distinct = Some(distinct);
        self.next = Read::Complete;
        Ok(())
    }

    fn advance(&mut self, row: usize) {
        if row + 1 < self.rows {
            self.next = Read::Missing(row + 1);
        } else {
            self.finish(
                if self.wide {
                    Dtype::String
                } else {
                    Dtype::Int64
                },
                false,
            );
        }
    }

    /// Accept the exact boolean result of the selected native operation.
    /// Native truth conversion belongs to that same operation in the host.
    pub fn observe(&mut self, value: bool) -> Result<(), CoreError> {
        match self.next {
            Read::Missing(row) => {
                if value {
                    self.advance(row);
                } else {
                    self.next = if self.integer {
                        Read::WithinIntegerRange(row)
                    } else {
                        Read::IsDecimal(row)
                    };
                }
            }
            Read::IsDecimal(row) => {
                if value {
                    self.next = Read::Finite(row);
                } else {
                    self.finish(Dtype::Float64, false);
                }
            }
            Read::WithinIntegerRange(row) => {
                if value {
                    self.advance(row);
                } else {
                    self.finish(Dtype::String, false);
                }
            }
            Read::Finite(row) => {
                if value {
                    self.next = Read::NegativeExponent(row);
                } else {
                    self.finish(Dtype::Float64, false);
                }
            }
            Read::NegativeExponent(row) => {
                self.next = if value {
                    Read::Fractional(row)
                } else {
                    Read::Wide(row)
                };
            }
            Read::Fractional(row) => {
                if value {
                    self.finish(Dtype::Float64, false);
                } else {
                    self.next = Read::Wide(row);
                }
            }
            Read::Wide(row) => {
                if value {
                    self.next = Read::AtLeastMinimum(row);
                } else {
                    self.advance(row);
                }
            }
            Read::AtLeastMinimum(row) => {
                if value {
                    self.next = Read::AtMostMaximum(row);
                } else {
                    self.wide = true;
                    self.advance(row);
                }
            }
            Read::AtMostMaximum(row) => {
                self.wide |= !value;
                self.advance(row);
            }
            Read::Content | Read::Distinct | Read::Complete => {
                return Err(invalid(
                    "native Decimal boolean arrived outside a selected read",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fraction_settles_without_observing_the_second_value() {
        let mut inference = Constructor::new(2);
        inference.content(Content::Decimal).unwrap();
        for (read, value) in [
            (Read::Missing(0), false),
            (Read::IsDecimal(0), true),
            (Read::Finite(0), true),
            (Read::NegativeExponent(0), true),
            (Read::Fractional(0), true),
        ] {
            assert_eq!(inference.next(), read);
            inference.observe(value).unwrap();
        }
        assert_eq!(inference.next(), Read::Complete);
        assert_eq!(inference.meaning().unwrap().dtype, Dtype::Float64);
        assert!(inference.observe(true).is_err());
    }

    #[test]
    fn one_wide_whole_does_not_short_circuit_later_fraction() {
        let mut inference = Constructor::new(2);
        inference.content(Content::Decimal).unwrap();
        for answer in [false, true, true, false, true, true, false] {
            inference.observe(answer).unwrap();
        }
        assert_eq!(inference.next(), Read::Missing(1));
        for answer in [false, true, true, true, true] {
            inference.observe(answer).unwrap();
        }
        assert_eq!(inference.meaning().unwrap().dtype, Dtype::Float64);
    }

    #[test]
    fn lower_bound_comparison_short_circuits_the_upper_comparison() {
        let mut inference = Constructor::new(1);
        inference.content(Content::Decimal).unwrap();
        for answer in [false, true, true, false, true, false] {
            inference.observe(answer).unwrap();
        }
        assert_eq!(inference.next(), Read::Distinct);
        inference.distinct(1).unwrap();
        assert_eq!(inference.next(), Read::Complete);
        assert_eq!(inference.meaning().unwrap().dtype, Dtype::String);
    }
}
