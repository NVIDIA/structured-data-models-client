// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A pandas `Series` as the executor sees one: its dtype and values, with
//! each dtype's own idea of a missing value (`NaN` for floats, `pd.NA` for
//! the nullable dtypes, `NaT` for datetimes, any null for objects).

use std::cmp::Ordering;
use std::fmt;

use sdm_schema::decimal;

use crate::LabelError;

/// The resolution of a `datetime64` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeUnit {
    S,
    Ms,
    Us,
    Ns,
}

impl TimeUnit {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "s" => TimeUnit::S,
            "ms" => TimeUnit::Ms,
            "us" => TimeUnit::Us,
            "ns" => TimeUnit::Ns,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            TimeUnit::S => "s",
            TimeUnit::Ms => "ms",
            TimeUnit::Us => "us",
            TimeUnit::Ns => "ns",
        }
    }

    /// Nanoseconds in one tick.
    pub fn nanos(self) -> i64 {
        match self {
            TimeUnit::S => 1_000_000_000,
            TimeUnit::Ms => 1_000_000,
            TimeUnit::Us => 1_000,
            TimeUnit::Ns => 1,
        }
    }

    /// Ticks in one second.
    pub fn per_second(self) -> i64 {
        1_000_000_000 / self.nanos()
    }
}

/// A Python object in an `object` column, or a scalar a column yields.
#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    None,
    /// `pd.NA`.
    Na,
    /// `pd.NaT`.
    NaT,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Duration(TimeUnit, i64),
    /// A `pd.Timestamp`: its unit and ticks.
    Time(TimeUnit, i64),
    /// A `decimal.Decimal` an Arrow `decimal128` column yields: its unscaled
    /// digits and its scale.
    Decimal(i128, i8),
    List(Vec<Obj>),
    /// numpy scalars, which some columns yield instead of Python's.
    NpInt64(i64),
    NpFloat64(f64),
    NpFloat32(f32),
    NpBool(bool),
}

impl Obj {
    /// `pd.isna(obj)` for a scalar.
    pub fn is_na(&self) -> bool {
        match self {
            Obj::None | Obj::Na | Obj::NaT => true,
            Obj::Float(v) | Obj::NpFloat64(v) => v.is_nan(),
            Obj::NpFloat32(v) => v.is_nan(),
            _ => false,
        }
    }

    /// The value as a number, for comparisons across numeric types that
    /// Python does not make exactly (`exact_order`), and for casts.
    fn number(&self) -> Option<f64> {
        match self {
            Obj::Bool(b) | Obj::NpBool(b) => Some(f64::from(u8::from(*b))),
            Obj::Int(v) | Obj::NpInt64(v) => Some(*v as f64),
            Obj::Float(v) | Obj::NpFloat64(v) => Some(*v),
            Obj::NpFloat32(v) => Some(f64::from(*v)),
            Obj::Decimal(digits, scale) => sdm_schema::frame::decimal_text(*digits, *scale)
                .parse()
                .ok(),
            _ => None,
        }
    }

    fn exact_int(&self) -> Option<i128> {
        match self {
            Obj::Bool(b) | Obj::NpBool(b) => Some(i128::from(u8::from(*b))),
            Obj::Int(v) | Obj::NpInt64(v) => Some(i128::from(*v)),
            _ => None,
        }
    }

    /// An integer or a decimal as unscaled digits and a scale, which
    /// Python compares exactly.
    fn exact(&self) -> Option<(i128, i8)> {
        match self {
            Obj::Decimal(digits, scale) => Some((*digits, *scale)),
            other => other.exact_int().map(|v| (v, 0)),
        }
    }

    /// The value of a float, Python's or numpy's.
    fn float(&self) -> Option<f64> {
        match self {
            Obj::Float(v) | Obj::NpFloat64(v) => Some(*v),
            Obj::NpFloat32(v) => Some(f64::from(*v)),
            _ => None,
        }
    }

    /// The order of two numbers Python compares exactly: integers and
    /// decimals against each other, a decimal against any float, and a
    /// Python int against a Python float (numpy's scalars compare as
    /// floats instead). `Some(None)` against a `NaN`; `None` where the two
    /// are not compared exactly.
    fn exact_order(&self, other: &Obj) -> Option<Option<Ordering>> {
        if let (Some(a), Some(b)) = (self.exact(), other.exact()) {
            return Some(Some(decimal::compare(a, b)));
        }
        let against_float = |exact: &Obj, float: &Obj| {
            let f = float.float()?;
            match exact {
                Obj::Decimal(..) => {}
                Obj::Int(_) | Obj::Bool(_) if matches!(float, Obj::Float(_)) => {}
                _ => return None,
            }
            Some(decimal::compare_float(exact.exact()?, f))
        };
        against_float(self, other)
            .or_else(|| against_float(other, self).map(|order| order.map(Ordering::reverse)))
    }

    /// Python's `==` between two objects.
    pub fn py_eq(&self, other: &Obj) -> bool {
        if let Some(order) = self.exact_order(other) {
            return order == Some(Ordering::Equal);
        }
        if let (Some(a), Some(b)) = (self.number(), other.number()) {
            return a == b;
        }
        match (self, other) {
            (Obj::Str(a), Obj::Str(b)) => a == b,
            (Obj::Bytes(left), Obj::Bytes(right)) => left == right,
            (Obj::Duration(left_unit, left), Obj::Duration(right_unit, right)) => {
                i128::from(*left) * i128::from(left_unit.nanos())
                    == i128::from(*right) * i128::from(right_unit.nanos())
            }
            (Obj::Time(ua, a), Obj::Time(ub, b)) => {
                i128::from(*a) * i128::from(ua.nanos()) == i128::from(*b) * i128::from(ub.nanos())
            }
            (Obj::List(a), Obj::List(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y))
            }
            (Obj::None, Obj::None) => true,
            _ => false,
        }
    }

    /// Python's ordering of two objects, or what it raises: the
    /// `TypeError` of two types it does not order, or the
    /// `decimal.InvalidOperation` of a decimal against a `NaN`.
    pub fn py_cmp(&self, other: &Obj, op: &str) -> Result<Option<Ordering>, LabelError> {
        if let Some(order) = self.exact_order(other) {
            if order.is_none()
                && (matches!(self, Obj::Decimal(..)) || matches!(other, Obj::Decimal(..)))
            {
                return Err(LabelError::new(
                    "InvalidOperation",
                    "[<class 'decimal.InvalidOperation'>]",
                ));
            }
            return Ok(order);
        }
        if let (Some(a), Some(b)) = (self.number(), other.number()) {
            return Ok(a.partial_cmp(&b));
        }
        match (self, other) {
            (Obj::Str(a), Obj::Str(b)) => Ok(Some(a.cmp(b))),
            (Obj::Bytes(left), Obj::Bytes(right)) => Ok(Some(left.cmp(right))),
            (Obj::Duration(left_unit, left), Obj::Duration(right_unit, right)) => Ok(Some(
                (i128::from(*left) * i128::from(left_unit.nanos()))
                    .cmp(&(i128::from(*right) * i128::from(right_unit.nanos()))),
            )),
            (Obj::Time(ua, a), Obj::Time(ub, b)) => Ok(Some(
                (i128::from(*a) * i128::from(ua.nanos()))
                    .cmp(&(i128::from(*b) * i128::from(ub.nanos()))),
            )),
            _ => Err(LabelError::new(
                "TypeError",
                format!(
                    "'{op}' not supported between instances of '{}' and '{}'",
                    self.type_name(),
                    other.type_name()
                ),
            )),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Obj::None => "NoneType",
            Obj::Na => "NAType",
            Obj::NaT => "NaTType",
            Obj::Bool(_) => "bool",
            Obj::Int(_) => "int",
            Obj::Float(_) => "float",
            Obj::Str(_) => "str",
            Obj::Bytes(_) => "bytes",
            Obj::Duration(..) => "Timedelta",
            Obj::Time(..) => "Timestamp",
            Obj::Decimal(..) => "decimal.Decimal",
            Obj::List(_) => "list",
            Obj::NpInt64(_) => "int64",
            Obj::NpFloat64(_) => "float64",
            Obj::NpFloat32(_) => "float32",
            Obj::NpBool(_) => "bool",
        }
    }
}

/// A column: one of the dtypes the executor meets.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// `bool`.
    Bool(Vec<bool>),
    /// `int64`.
    I64(Vec<i64>),
    /// `float32`; `NaN` is missing.
    F32(Vec<f32>),
    /// `float64`; `NaN` is missing.
    F64(Vec<f64>),
    /// `Int64`.
    MI64(Vec<Option<i64>>),
    /// `Float64`.
    MF64(Vec<Option<f64>>),
    /// `boolean`.
    MBool(Vec<Option<bool>>),
    /// `datetime64[unit]`; `None` is `NaT`.
    Time(TimeUnit, Vec<Option<i64>>),
    /// `str`, pandas 3's string dtype; `None` is its `NaN`.
    Str(Vec<Option<String>>),
    /// `object`.
    Obj(Vec<Obj>),
}

impl Default for Column {
    /// An empty `datetime64[ns]` column, what an anchor is before it is set.
    fn default() -> Self {
        Column::Time(TimeUnit::Ns, Vec::new())
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.dtype())
    }
}

/// Picks the rows of `values` where `mask` holds.
fn pick<T: Clone>(values: &[T], mask: &[bool]) -> Vec<T> {
    values
        .iter()
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|(v, _)| v.clone())
        .collect()
}

impl Column {
    /// `str(series.dtype)`.
    pub fn dtype(&self) -> String {
        match self {
            Column::Bool(_) => "bool".into(),
            Column::I64(_) => "int64".into(),
            Column::F32(_) => "float32".into(),
            Column::F64(_) => "float64".into(),
            Column::MI64(_) => "Int64".into(),
            Column::MF64(_) => "Float64".into(),
            Column::MBool(_) => "boolean".into(),
            Column::Time(unit, _) => format!("datetime64[{}]", unit.name()),
            Column::Str(_) => "str".into(),
            Column::Obj(_) => "object".into(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Bool(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::F64(v) => v.len(),
            Column::MI64(v) => v.len(),
            Column::MF64(v) => v.len(),
            Column::MBool(v) => v.len(),
            Column::Time(_, v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Obj(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An empty column of the same dtype.
    pub fn empty_like(&self) -> Column {
        self.filter(&vec![false; self.len()])
    }

    /// `series.notna()`.
    pub fn notna(&self) -> Vec<bool> {
        match self {
            Column::Bool(v) => vec![true; v.len()],
            Column::I64(v) => vec![true; v.len()],
            Column::F32(v) => v.iter().map(|x| !x.is_nan()).collect(),
            Column::F64(v) => v.iter().map(|x| !x.is_nan()).collect(),
            Column::MI64(v) => v.iter().map(Option::is_some).collect(),
            Column::MF64(v) => v.iter().map(Option::is_some).collect(),
            Column::MBool(v) => v.iter().map(Option::is_some).collect(),
            Column::Time(_, v) => v.iter().map(Option::is_some).collect(),
            Column::Str(v) => v.iter().map(Option::is_some).collect(),
            Column::Obj(v) => v.iter().map(|o| !o.is_na()).collect(),
        }
    }

    /// `series[mask].reset_index(drop=True)`.
    pub fn filter(&self, mask: &[bool]) -> Column {
        match self {
            Column::Bool(v) => Column::Bool(pick(v, mask)),
            Column::I64(v) => Column::I64(pick(v, mask)),
            Column::F32(v) => Column::F32(pick(v, mask)),
            Column::F64(v) => Column::F64(pick(v, mask)),
            Column::MI64(v) => Column::MI64(pick(v, mask)),
            Column::MF64(v) => Column::MF64(pick(v, mask)),
            Column::MBool(v) => Column::MBool(pick(v, mask)),
            Column::Time(u, v) => Column::Time(*u, pick(v, mask)),
            Column::Str(v) => Column::Str(pick(v, mask)),
            Column::Obj(v) => Column::Obj(pick(v, mask)),
        }
    }

    pub fn is_float_dtype(&self) -> bool {
        matches!(self, Column::F32(_) | Column::F64(_) | Column::MF64(_))
    }

    pub fn is_integer_dtype(&self) -> bool {
        matches!(self, Column::I64(_) | Column::MI64(_))
    }

    pub fn is_bool_dtype(&self) -> bool {
        matches!(self, Column::Bool(_) | Column::MBool(_))
    }

    pub fn is_datetime(&self) -> bool {
        matches!(self, Column::Time(..))
    }

    /// `astype('float32')`.
    pub fn to_f32(&self) -> Result<Column, String> {
        Ok(Column::F32(match self {
            Column::F32(v) => v.clone(),
            Column::F64(v) => v.iter().map(|x| *x as f32).collect(),
            Column::MF64(v) => v.iter().map(|x| x.map_or(f32::NAN, |x| x as f32)).collect(),
            Column::I64(v) => v.iter().map(|x| *x as f32).collect(),
            Column::MI64(v) => v.iter().map(|x| x.map_or(f32::NAN, |x| x as f32)).collect(),
            Column::Bool(v) => v.iter().map(|x| f32::from(u8::from(*x))).collect(),
            Column::MBool(v) => v
                .iter()
                .map(|x| x.map_or(f32::NAN, |x| f32::from(u8::from(x))))
                .collect(),
            Column::Obj(v) => v
                .iter()
                .map(|o| match o {
                    o if o.is_na() => Ok(f32::NAN),
                    Obj::Str(s) => Err(format!("could not convert string to float: '{s}'")),
                    o => o.number().map(|x| x as f32).ok_or_else(|| {
                        format!(
                            "float() argument must be a string or a real number, not '{}'",
                            o.type_name()
                        )
                    }),
                })
                .collect::<Result<_, _>>()?,
            Column::Str(v) => v
                .iter()
                .map(|s| match s {
                    None => Ok(f32::NAN),
                    Some(s) => s
                        .trim()
                        .parse::<f32>()
                        .map_err(|_| format!("could not convert string to float: '{s}'")),
                })
                .collect::<Result<_, _>>()?,
            Column::Time(..) => {
                return Err("Cannot cast DatetimeArray to dtype float32".to_string());
            }
        }))
    }

    pub fn to_f32_with_warnings(
        &self,
        warnings: &mut Vec<crate::Warning>,
    ) -> Result<Column, String> {
        let output = self.to_f32()?;
        let overflow = match self {
            Self::F64(values) => values
                .iter()
                .copied()
                .find(|value| value.is_finite() && (*value as f32).is_infinite()),
            Self::MF64(values) => values
                .iter()
                .flatten()
                .copied()
                .find(|value| value.is_finite() && (*value as f32).is_infinite()),
            _ => None,
        };
        if let Some(value) = overflow {
            warnings.push(if matches!(self, Self::MF64(_)) {
                crate::Warning::MaskedFloat32Overflow(value)
            } else {
                crate::Warning::Float32Overflow(value)
            });
        }
        Ok(output)
    }

    /// `astype('float64')` of an integer column, as `execute_rel_op`
    /// promotes one.
    pub fn int_to_f64(&self) -> Column {
        match self {
            Column::I64(v) => Column::F64(v.iter().map(|x| *x as f64).collect()),
            Column::MI64(v) => {
                Column::F64(v.iter().map(|x| x.map_or(f64::NAN, |x| x as f64)).collect())
            }
            other => other.clone(),
        }
    }

    /// The executor's cast to a primitive dtype after nulls are dropped:
    /// integers to `int64`, booleans to `bool`.
    pub fn to_primitive(&self) -> Column {
        match self {
            Column::MI64(v) => Column::I64(v.iter().map(|x| x.unwrap_or(0)).collect()),
            Column::MBool(v) => Column::Bool(v.iter().map(|x| x.unwrap_or(false)).collect()),
            other => other.clone(),
        }
    }

    /// The bools of a boolean column, as `to_numpy()` gives them; `None`
    /// where it holds a missing value.
    pub fn bools(&self) -> Option<Vec<bool>> {
        match self {
            Column::Bool(v) => Some(v.clone()),
            Column::MBool(v) if v.iter().all(Option::is_some) => {
                Some(v.iter().map(|x| x.unwrap_or(false)).collect())
            }
            _ => None,
        }
    }

    /// Each value as the object `series.tolist()` gives.
    pub fn objects(&self) -> Vec<Obj> {
        match self {
            Column::Bool(v) => v.iter().map(|x| Obj::Bool(*x)).collect(),
            Column::I64(v) => v.iter().map(|x| Obj::Int(*x)).collect(),
            Column::F32(v) => v.iter().map(|x| Obj::Float(f64::from(*x))).collect(),
            Column::F64(v) => v.iter().map(|x| Obj::Float(*x)).collect(),
            Column::MI64(v) => v.iter().map(|x| x.map_or(Obj::Na, Obj::Int)).collect(),
            Column::MF64(v) => v.iter().map(|x| x.map_or(Obj::Na, Obj::Float)).collect(),
            Column::MBool(v) => v.iter().map(|x| x.map_or(Obj::Na, Obj::Bool)).collect(),
            Column::Time(u, v) => v
                .iter()
                .map(|x| x.map_or(Obj::NaT, |t| Obj::Time(*u, t)))
                .collect(),
            Column::Str(v) => v
                .iter()
                .map(|x| x.clone().map_or(Obj::Float(f64::NAN), Obj::Str))
                .collect(),
            Column::Obj(v) => v.clone(),
        }
    }

    /// Each value as iterating the series yields it: numpy-backed columns
    /// give Python scalars, the nullable ones numpy scalars.
    pub fn iter_objects(&self) -> Vec<Obj> {
        match self {
            Column::MI64(v) => v.iter().map(|x| x.map_or(Obj::Na, Obj::NpInt64)).collect(),
            Column::MF64(v) => v
                .iter()
                .map(|x| x.map_or(Obj::Na, Obj::NpFloat64))
                .collect(),
            Column::MBool(v) => v.iter().map(|x| x.map_or(Obj::Na, Obj::NpBool)).collect(),
            other => other.objects(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_compare_with_floats_exactly() {
        // From Python: Decimal('0.1') == 0.1 is False and orders below it;
        // Decimal('0.5') == 0.5, also as numpy's float32.
        let tenth = Obj::Decimal(1, 1);
        assert!(!tenth.py_eq(&Obj::Float(0.1)));
        assert!(!Obj::Float(0.1).py_eq(&tenth));
        assert_eq!(
            tenth.py_cmp(&Obj::Float(0.1), "<"),
            Ok(Some(Ordering::Less))
        );
        assert_eq!(
            Obj::NpFloat64(0.1).py_cmp(&tenth, "<"),
            Ok(Some(Ordering::Greater))
        );
        assert_eq!(
            tenth.py_cmp(&Obj::NpFloat32(0.1), "<"),
            Ok(Some(Ordering::Less))
        );
        assert!(Obj::Decimal(50, 2).py_eq(&Obj::Float(0.5)));
        assert!(Obj::NpFloat32(0.5).py_eq(&Obj::Decimal(5, 1)));
        assert!(Obj::Decimal(5, 0).py_eq(&Obj::Int(5)));
    }

    #[test]
    fn a_decimal_is_not_ordered_against_nan() {
        // Decimal('1') < float('nan') raises decimal.InvalidOperation; ==
        // is False.
        let one = Obj::Decimal(1, 0);
        assert!(!one.py_eq(&Obj::Float(f64::NAN)));
        let error = one.py_cmp(&Obj::Float(f64::NAN), "<").unwrap_err();
        assert_eq!(error.code, "InvalidOperation");
        assert_eq!(error.message, "[<class 'decimal.InvalidOperation'>]");
        assert!(Obj::NpFloat32(f32::NAN).py_cmp(&one, ">").is_err());
        // An int against NaN is merely unordered.
        assert_eq!(Obj::Int(1).py_cmp(&Obj::Float(f64::NAN), "<"), Ok(None));
    }

    #[test]
    fn python_ints_and_floats_compare_exactly_and_numpys_do_not() {
        // From Python: 2**53 + 1 == 2.0**53 is False and orders above it;
        // np.int64(2**53 + 1) == 2.0**53 is True.
        let big = (1i64 << 53) + 1;
        let float = Obj::Float(2f64.powi(53));
        assert!(!Obj::Int(big).py_eq(&float));
        assert_eq!(
            Obj::Int(big).py_cmp(&float, ">"),
            Ok(Some(Ordering::Greater))
        );
        assert!(Obj::NpInt64(big).py_eq(&float));
        assert!(Obj::Int(big).py_eq(&Obj::NpFloat64(2f64.powi(53))));
        assert!(Obj::Bool(true).py_eq(&Obj::Float(1.0)));
    }

    #[test]
    fn unordered_types_raise_type_errors() {
        let error = Obj::Decimal(1, 0)
            .py_cmp(&Obj::Str("a".into()), "<")
            .unwrap_err();
        assert_eq!(error.code, "TypeError");
        assert_eq!(
            error.message,
            "'<' not supported between instances of 'decimal.Decimal' and 'str'"
        );
    }

    #[test]
    fn float32_cast_warns_once_only_for_finite_overflow() {
        let mut warnings = Vec::new();
        let output = Column::F64(vec![f64::INFINITY, f64::NAN, 1.0, 1e100, -1e100])
            .to_f32_with_warnings(&mut warnings)
            .unwrap();
        assert!(
            matches!(output, Column::F32(values) if values[2] == 1.0 && values[3] == f32::INFINITY && values[4] == f32::NEG_INFINITY)
        );
        assert_eq!(warnings, vec![crate::Warning::Float32Overflow(1e100)]);
        warnings.clear();
        Column::F64(vec![f64::INFINITY, f64::NAN, 1.0])
            .to_f32_with_warnings(&mut warnings)
            .unwrap();
        assert!(warnings.is_empty());
        Column::MF64(vec![None, Some(-1e100)])
            .to_f32_with_warnings(&mut warnings)
            .unwrap();
        assert_eq!(
            warnings,
            vec![crate::Warning::MaskedFloat32Overflow(-1e100)]
        );
    }
}
