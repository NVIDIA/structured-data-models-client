// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The closed native-object Decimal constructor inference traversal.
//!
//! Each observation is the result of one reached native operation, never a
//! callback, expression, or object handle. The core controls short-circuiting
//! so an override on a later Decimal is not called after an earlier fraction
//! has already settled the meaning. Native exceptions abort at their read.

use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Sentence};

/// The next native source observation selected by constructor inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Content,
    Missing(usize),
    IsDecimal(usize),
    Finite(usize),
    NegativeExponent(usize),
    Fractional(usize),
    Wide(usize),
    AtLeastMinimum(usize),
    AtMostMaximum(usize),
    Complete,
}

/// Content classification is native pandas information, not a wire dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Content {
    Decimal,
    Empty,
    Other,
}

/// The inferred object-Decimal meaning, fixed by the core's traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Meaning {
    pub dtype: Dtype,
    pub untyped: bool,
}

/// A single consumed column. A caller must retain that column's native
/// iterator while answering selected observations; each next column is
/// resolved only after this traversal and its native effects finish.
pub struct Constructor {
    rows: usize,
    next: Read,
    wide: bool,
    meaning: Option<Meaning>,
    decimal: bool,
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

impl Constructor {
    pub fn new(rows: usize) -> Self {
        Self {
            rows,
            next: Read::Content,
            wide: false,
            meaning: None,
            decimal: false,
        }
    }

    pub fn next(&self) -> Read {
        self.next
    }

    pub fn meaning(&self) -> Option<Meaning> {
        self.meaning
    }

    /// A non-Decimal object family is returned to its constructor reader,
    /// never recategorized as a Decimal or routed to the original client.
    pub fn is_decimal(&self) -> bool {
        self.decimal
    }

    pub fn content(&mut self, content: Content) -> Result<(), CoreError> {
        if self.next != Read::Content {
            return Err(invalid(
                "native object content arrived outside its selected read",
            ));
        }
        match content {
            Content::Other => self.next = Read::Complete,
            Content::Empty => {
                self.decimal = true;
                self.finish(Dtype::String, true);
            }
            Content::Decimal => {
                self.decimal = true;
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
        self.meaning = Some(Meaning { dtype, untyped });
        self.next = Read::Complete;
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
                    self.next = Read::IsDecimal(row);
                }
            }
            Read::IsDecimal(row) => {
                if value {
                    self.next = Read::Finite(row);
                } else {
                    self.finish(Dtype::Float64, false);
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
            Read::Content | Read::Complete => {
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
        assert_eq!(inference.next(), Read::Complete);
        assert_eq!(inference.meaning().unwrap().dtype, Dtype::String);
    }
}
