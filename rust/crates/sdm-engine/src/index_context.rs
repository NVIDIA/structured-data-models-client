// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sort the core's selected original row ordinals using admitted index facts.
//!
//! This first connected consumer covers native NumPy primitives, categorical
//! codes, RangeIndex values and temporal backing values. It is not routed from
//! a public operation yet. Arrow-backed index semantics, structural objects,
//! MultiIndex and the remaining producer bindings must be composed before that
//! activation; an Arrow physical type alone never selects native semantics.
//!
//! Source: pandas 3.0.6 core/sorting.py get_indexer_indexer/nargsort;
//! indexes/category.py's signed-code engine; arrays/base.py argsort; and the
//! separately source-locked scalar/SIMD kernels. Missing labels are partitioned
//! only after the selected index's monotonic shortcut has been considered.

use crate::index_codes::{CodeWidth, IndexCodes, InvalidCodes};
use crate::index_sort::{SimdArgsortProfile, SimdKey, intel5_argsort, simd_argsort};
use crate::sample::scalar_argsort;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow_schema::{DataType, TimeUnit};

/// Established native algorithm bindings, not user-selectable ordering policy.
/// The Intel5 entry is source-proven; actual old GCC10 binary fallback binding
/// is still a separate obligation and this consumer remains publicly unrouted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumpyProducer {
    Scalar,
    Intel5Avx2SharedGnu,
    Intel7Avx2Gnu14,
    Intel7Avx512Gnu14,
}

/// Native storage semantics, independent of the Arrow value buffer's width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingValues {
    /// Masked arrays consume only their validity mask as native missingness.
    Validity,
    /// Ordinary floating indexes also classify NaNs as missing labels.
    NanOrValidity,
}

/// Internal fact/proof failures; no public refusal or routing mapping exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexOrderError {
    RowCount {
        expected: usize,
        actual: usize,
    },
    RowBounds {
        row: usize,
        source_rows: usize,
    },
    PhysicalType(DataType),
    Codes(InvalidCodes),
    /// A present masked NaN needs the native NaN-fallback kernel, not omission.
    /// This unfinished producer obligation blocks activation of this consumer.
    PresentNanKernel,
}

#[derive(Clone, Debug)]
enum Layout {
    Numpy(MissingValues),
    Range,
    Temporal,
    Categorical {
        codes: IndexCodes,
        dictionary: ArrayRef,
        ordered: bool,
    },
}

/// Original-row-aligned values owned with one immutable admitted frame snapshot.
#[derive(Clone, Debug)]
pub struct OriginalIndexFacts {
    values: ArrayRef,
    source_rows: usize,
    layout: Layout,
    producer: NumpyProducer,
}

fn numeric_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

impl OriginalIndexFacts {
    fn admit(
        values: ArrayRef,
        source_rows: usize,
        layout: Layout,
        producer: NumpyProducer,
    ) -> Result<Self, IndexOrderError> {
        if values.len() != source_rows {
            return Err(IndexOrderError::RowCount {
                expected: source_rows,
                actual: values.len(),
            });
        }
        Ok(Self {
            values,
            source_rows,
            layout,
            producer,
        })
    }

    /// Native NumPy/masked semantics must be supplied explicitly by conversion.
    /// This constructor is not appropriate for an ArrowExtensionArray index.
    pub fn numpy(
        values: ArrayRef,
        source_rows: usize,
        missing: MissingValues,
        producer: NumpyProducer,
    ) -> Result<Self, IndexOrderError> {
        if !numeric_type(values.data_type()) {
            return Err(IndexOrderError::PhysicalType(values.data_type().clone()));
        }
        Self::admit(values, source_rows, Layout::Numpy(missing), producer)
    }

    /// Selection can turn a native RangeIndex into an ordinary integer index.
    pub fn range(
        values: ArrayRef,
        source_rows: usize,
        producer: NumpyProducer,
    ) -> Result<Self, IndexOrderError> {
        if values.data_type() != &DataType::Int64 {
            return Err(IndexOrderError::PhysicalType(values.data_type().clone()));
        }
        Self::admit(values, source_rows, Layout::Range, producer)
    }

    /// Native temporal arrays use scalar argsort even when their backing is i64.
    /// Int64 explicitly denotes native period ordinals, never numeric guessing.
    pub fn temporal(values: ArrayRef, source_rows: usize) -> Result<Self, IndexOrderError> {
        if !matches!(
            values.data_type(),
            DataType::Int64 | DataType::Timestamp(_, _) | DataType::Duration(_)
        ) {
            return Err(IndexOrderError::PhysicalType(values.data_type().clone()));
        }
        Self::admit(values, source_rows, Layout::Temporal, NumpyProducer::Scalar)
    }

    pub fn categorical(
        values: ArrayRef,
        dictionary: ArrayRef,
        ordered: bool,
        source_rows: usize,
        producer: NumpyProducer,
    ) -> Result<Self, IndexOrderError> {
        let codes = IndexCodes::admit(values.clone(), dictionary.len(), source_rows)
            .map_err(IndexOrderError::Codes)?;
        Self::admit(
            values,
            source_rows,
            Layout::Categorical {
                codes,
                dictionary,
                ordered,
            },
            producer,
        )
    }

    pub fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub fn dictionary(&self) -> Option<&ArrayRef> {
        match &self.layout {
            Layout::Categorical { dictionary, .. } => Some(dictionary),
            _ => None,
        }
    }

    pub fn ordered(&self) -> Option<bool> {
        match &self.layout {
            Layout::Categorical { ordered, .. } => Some(*ordered),
            _ => None,
        }
    }

    pub fn code_width(&self) -> Option<CodeWidth> {
        match &self.layout {
            Layout::Categorical { codes, .. } => Some(codes.width()),
            _ => None,
        }
    }

    /// The caller has already filtered and sampled in core original-row space.
    /// Return the same row identities in source-faithful selected-index order.
    pub fn order_selected(&self, selected: &[usize]) -> Result<Vec<usize>, IndexOrderError> {
        for &row in selected {
            if row >= self.source_rows {
                return Err(IndexOrderError::RowBounds {
                    row,
                    source_rows: self.source_rows,
                });
            }
        }
        let categorical = matches!(self.layout, Layout::Categorical { .. });
        let temporal = matches!(self.layout, Layout::Temporal);
        let nan_missing = matches!(self.layout, Layout::Numpy(MissingValues::NanOrValidity));
        macro_rules! narrow {
            ($kind:ty) => {
                self.primitive::<$kind>(selected, categorical, temporal, |_| false, scalar_argsort)
            };
        }
        macro_rules! wide {
            ($kind:ty) => {
                self.primitive::<$kind>(
                    selected,
                    categorical,
                    temporal,
                    |_| false,
                    |keys| wide_order(keys, self.producer),
                )
            };
        }
        macro_rules! time {
            ($kind:ty) => {
                self.primitive::<$kind>(
                    selected,
                    false,
                    true,
                    |value| value == i64::MIN,
                    scalar_argsort,
                )
            };
        }
        if temporal {
            return match self.values.data_type() {
                DataType::Int64 => time!(Int64Type),
                DataType::Timestamp(TimeUnit::Second, _) => time!(TimestampSecondType),
                DataType::Timestamp(TimeUnit::Millisecond, _) => time!(TimestampMillisecondType),
                DataType::Timestamp(TimeUnit::Microsecond, _) => time!(TimestampMicrosecondType),
                DataType::Timestamp(TimeUnit::Nanosecond, _) => time!(TimestampNanosecondType),
                DataType::Duration(TimeUnit::Second) => time!(DurationSecondType),
                DataType::Duration(TimeUnit::Millisecond) => time!(DurationMillisecondType),
                DataType::Duration(TimeUnit::Microsecond) => time!(DurationMicrosecondType),
                DataType::Duration(TimeUnit::Nanosecond) => time!(DurationNanosecondType),
                other => Err(IndexOrderError::PhysicalType(other.clone())),
            };
        }
        match self.values.data_type() {
            DataType::Int8 => narrow!(Int8Type),
            DataType::Int16 => narrow!(Int16Type),
            DataType::Int32 => wide!(Int32Type),
            DataType::Int64 => wide!(Int64Type),
            DataType::UInt8 => narrow!(UInt8Type),
            DataType::UInt16 => narrow!(UInt16Type),
            DataType::UInt32 => wide!(UInt32Type),
            DataType::UInt64 => wide!(UInt64Type),
            DataType::Float32 => self.floating::<Float32Type>(
                selected,
                nan_missing,
                |value| value.is_nan(),
                |keys| wide_order(keys, self.producer),
            ),
            DataType::Float64 => self.floating::<Float64Type>(
                selected,
                nan_missing,
                |value| value.is_nan(),
                |keys| wide_order(keys, self.producer),
            ),
            DataType::Boolean => {
                let values = self
                    .values
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("admitted boolean array");
                order_values(
                    selected,
                    |row| values.value(row),
                    |row| values.is_null(row),
                    false,
                    |_| false,
                    scalar_argsort,
                )
            }
            other => Err(IndexOrderError::PhysicalType(other.clone())),
        }
    }

    fn primitive<A>(
        &self,
        selected: &[usize],
        categorical: bool,
        temporal: bool,
        sentinel: impl Fn(A::Native) -> bool,
        sort: impl Fn(&[A::Native]) -> Vec<usize>,
    ) -> Result<Vec<usize>, IndexOrderError>
    where
        A: ArrowPrimitiveType,
        A::Native: Copy + PartialOrd,
    {
        let values = self
            .values
            .as_any()
            .downcast_ref::<PrimitiveArray<A>>()
            .expect("admitted primitive array");
        order_values(
            selected,
            |row| values.value(row),
            |row| {
                values.is_null(row)
                    || categorical
                        && match &self.layout {
                            Layout::Categorical { codes, .. } => codes
                                .code(row)
                                .expect("admitted original code row")
                                .is_none(),
                            _ => false,
                        }
                    || temporal && sentinel(values.value(row))
            },
            categorical,
            |_| false,
            sort,
        )
    }

    fn floating<A>(
        &self,
        selected: &[usize],
        nan_missing: bool,
        isnan: impl Fn(A::Native) -> bool,
        sort: impl Fn(&[A::Native]) -> Vec<usize>,
    ) -> Result<Vec<usize>, IndexOrderError>
    where
        A: ArrowPrimitiveType,
        A::Native: Copy + PartialOrd,
    {
        let values = self
            .values
            .as_any()
            .downcast_ref::<PrimitiveArray<A>>()
            .expect("admitted floating array");
        order_values(
            selected,
            |row| values.value(row),
            |row| values.is_null(row) || nan_missing && isnan(values.value(row)),
            false,
            |value| isnan(value),
            sort,
        )
    }
}

fn wide_order<T: SimdKey>(keys: &[T], producer: NumpyProducer) -> Vec<usize> {
    match producer {
        NumpyProducer::Scalar => scalar_argsort(keys),
        NumpyProducer::Intel5Avx2SharedGnu => intel5_argsort(keys, SimdArgsortProfile::Avx2Gnu14),
        NumpyProducer::Intel7Avx2Gnu14 => simd_argsort(keys, SimdArgsortProfile::Avx2Gnu14),
        NumpyProducer::Intel7Avx512Gnu14 => simd_argsort(keys, SimdArgsortProfile::Avx512Gnu14),
    }
}

fn order_values<T: Copy + PartialOrd>(
    selected: &[usize],
    value: impl Fn(usize) -> T,
    missing: impl Fn(usize) -> bool,
    codes_monotonic: bool,
    isnan: impl Fn(T) -> bool,
    sort: impl Fn(&[T]) -> Vec<usize>,
) -> Result<Vec<usize>, IndexOrderError> {
    let monotonic = (codes_monotonic || selected.iter().all(|&row| !missing(row)))
        && selected
            .windows(2)
            .all(|pair| value(pair[0]) <= value(pair[1]));
    if monotonic {
        return Ok(selected.to_vec());
    }
    let mut present_rows = Vec::new();
    let mut missing_rows = Vec::new();
    let mut keys = Vec::new();
    for &row in selected {
        if missing(row) {
            missing_rows.push(row);
        } else {
            let key = value(row);
            if isnan(key) {
                return Err(IndexOrderError::PresentNanKernel);
            }
            present_rows.push(row);
            keys.push(key);
        }
    }
    let mut ordered: Vec<usize> = sort(&keys).into_iter().map(|at| present_rows[at]).collect();
    ordered.extend(missing_rows);
    Ok(ordered)
}
