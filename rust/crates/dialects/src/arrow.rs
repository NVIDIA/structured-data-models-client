// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arrow-typed columns, as the dialects take key and time lists: each value
//! with the Arrow type it came in, so it is written as pyarrow gives it.

use std::fmt;

use crate::Error;
use crate::py::Py;
use crate::time::{Civil, civil_from_days, offset_text, time_of_day_text};
use crate::tzfile::zone_offset;

/// A time unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Second,
    Milli,
    Micro,
    Nano,
}

impl Unit {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "s" => Self::Second,
            "ms" => Self::Milli,
            "us" => Self::Micro,
            "ns" => Self::Nano,
            _ => return None,
        })
    }

    /// The unit's ticks per second.
    pub fn per_second(self) -> i128 {
        match self {
            Self::Second => 1,
            Self::Milli => 1_000,
            Self::Micro => 1_000_000,
            Self::Nano => 1_000_000_000,
        }
    }

    /// The digits after the point a second of this unit is written with.
    fn digits(self) -> usize {
        match self {
            Self::Second => 0,
            Self::Milli => 3,
            Self::Micro => 6,
            Self::Nano => 9,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Second => "s",
            Self::Milli => "ms",
            Self::Micro => "us",
            Self::Nano => "ns",
        }
    }
}

/// The Arrow types a key or time list comes in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrowType {
    Null,
    Bool,
    Int {
        bits: u32,
        signed: bool,
    },
    Float16,
    Float32,
    Float64,
    String,
    LargeString,
    Binary,
    LargeBinary,
    Date32,
    Date64,
    Time32(Unit),
    Time64(Unit),
    Timestamp(Unit, Option<String>),
    Duration(Unit),
    Decimal {
        wide: bool,
        precision: u32,
        scale: i32,
    },
}

impl ArrowType {
    /// The type `str(type)` spells as `spelled`.
    pub fn parse(spelled: &str) -> Result<Self, Error> {
        let unknown = || Error::new(format!("the Arrow type {spelled:?} is not in the format"));
        let inner = |prefix: &str| {
            spelled
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_suffix(']'))
        };
        if let Some(body) = inner("timestamp[") {
            let (unit, zone) = match body.split_once(", tz=") {
                Some((unit, zone)) if !zone.is_empty() => (unit, Some(zone.to_string())),
                Some(_) => return Err(unknown()),
                None => (body, None),
            };
            return Ok(Self::Timestamp(
                Unit::parse(unit).ok_or_else(unknown)?,
                zone,
            ));
        }
        if let Some(body) = inner("time32[") {
            let unit = Unit::parse(body).filter(|u| matches!(u, Unit::Second | Unit::Milli));
            return Ok(Self::Time32(unit.ok_or_else(unknown)?));
        }
        if let Some(body) = inner("time64[") {
            let unit = Unit::parse(body).filter(|u| matches!(u, Unit::Micro | Unit::Nano));
            return Ok(Self::Time64(unit.ok_or_else(unknown)?));
        }
        if let Some(body) = inner("duration[") {
            return Ok(Self::Duration(Unit::parse(body).ok_or_else(unknown)?));
        }
        for (prefix, wide) in [("decimal128(", false), ("decimal256(", true)] {
            if let Some(body) = spelled
                .strip_prefix(prefix)
                .and_then(|r| r.strip_suffix(')'))
            {
                let (precision, scale) = body.split_once(", ").ok_or_else(unknown)?;
                return Ok(Self::Decimal {
                    wide,
                    precision: precision.parse().map_err(|_| unknown())?,
                    scale: scale.parse().map_err(|_| unknown())?,
                });
            }
        }
        Ok(match spelled {
            "null" => Self::Null,
            "bool" => Self::Bool,
            "int8" => Self::Int {
                bits: 8,
                signed: true,
            },
            "int16" => Self::Int {
                bits: 16,
                signed: true,
            },
            "int32" => Self::Int {
                bits: 32,
                signed: true,
            },
            "int64" => Self::Int {
                bits: 64,
                signed: true,
            },
            "uint8" => Self::Int {
                bits: 8,
                signed: false,
            },
            "uint16" => Self::Int {
                bits: 16,
                signed: false,
            },
            "uint32" => Self::Int {
                bits: 32,
                signed: false,
            },
            "uint64" => Self::Int {
                bits: 64,
                signed: false,
            },
            "halffloat" => Self::Float16,
            "float" => Self::Float32,
            "double" => Self::Float64,
            "string" => Self::String,
            "large_string" => Self::LargeString,
            "binary" => Self::Binary,
            "large_binary" => Self::LargeBinary,
            "date32[day]" => Self::Date32,
            "date64[ms]" => Self::Date64,
            _ => return Err(unknown()),
        })
    }

    pub fn is_floating(&self) -> bool {
        matches!(self, Self::Float16 | Self::Float32 | Self::Float64)
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Self::String | Self::LargeString)
    }

    pub fn is_date(&self) -> bool {
        matches!(self, Self::Date32 | Self::Date64)
    }

    /// Whether values are kept as their integer storage in the format.
    fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::Date32
                | Self::Date64
                | Self::Time32(_)
                | Self::Time64(_)
                | Self::Timestamp(..)
                | Self::Duration(_)
        )
    }
}

impl fmt::Display for ArrowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Bool => f.write_str("bool"),
            Self::Int { bits, signed } => {
                write!(f, "{}int{bits}", if *signed { "" } else { "u" })
            }
            Self::Float16 => f.write_str("halffloat"),
            Self::Float32 => f.write_str("float"),
            Self::Float64 => f.write_str("double"),
            Self::String => f.write_str("string"),
            Self::LargeString => f.write_str("large_string"),
            Self::Binary => f.write_str("binary"),
            Self::LargeBinary => f.write_str("large_binary"),
            Self::Date32 => f.write_str("date32[day]"),
            Self::Date64 => f.write_str("date64[ms]"),
            Self::Time32(unit) => write!(f, "time32[{}]", unit.name()),
            Self::Time64(unit) => write!(f, "time64[{}]", unit.name()),
            Self::Timestamp(unit, None) => write!(f, "timestamp[{}]", unit.name()),
            Self::Timestamp(unit, Some(zone)) => {
                write!(f, "timestamp[{}, tz={zone}]", unit.name())
            }
            Self::Duration(unit) => write!(f, "duration[{}]", unit.name()),
            Self::Decimal {
                wide,
                precision,
                scale,
            } => write!(
                f,
                "decimal{}({precision}, {scale})",
                if *wide { 256 } else { 128 }
            ),
        }
    }
}

/// An Arrow array, or a chunked array of the chunk lengths `chunks`.
///
/// Each value is the Python value `to_pylist()` gives, except a temporal
/// one, which is its integer storage, and a decimal, which is its text.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrowArray {
    pub data_type: ArrowType,
    pub values: Vec<Py>,
    pub chunks: Option<Vec<usize>>,
}

impl ArrowArray {
    /// The array the capture's `{"$arrow": ...}` body holds; `values` are
    /// decoded already.
    pub fn new(
        data_type: ArrowType,
        values: Vec<Py>,
        chunks: Option<Vec<usize>>,
    ) -> Result<Self, Error> {
        if let Some(lengths) = &chunks
            && lengths.iter().sum::<usize>() != values.len()
        {
            return Err(Error::new("the chunk lengths do not add up"));
        }
        for value in &values {
            let fits = match (&data_type, value) {
                (_, Py::None) => true,
                (t, Py::Int(_)) if t.is_temporal() => true,
                (ArrowType::Int { .. }, Py::Int(_)) => true,
                (ArrowType::Bool, Py::Bool(_)) => true,
                (t, Py::Float(_)) if t.is_floating() => true,
                (t, Py::Str(_)) if t.is_string() => true,
                (ArrowType::Decimal { .. }, Py::Str(_)) => true,
                (ArrowType::Binary | ArrowType::LargeBinary, Py::Bytes(_)) => true,
                _ => false,
            };
            if !fits {
                return Err(Error::new(format!(
                    "a {data_type} array cannot hold {value:?}"
                )));
            }
        }
        Ok(Self {
            data_type,
            values,
            chunks,
        })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The integer storage of a temporal value.
    fn storage(value: &Py) -> Result<i128, Error> {
        match value {
            Py::Int(int) => int
                .to_i128()
                .ok_or_else(|| Error::new("a temporal value out of range")),
            _ => Err(Error::new("a temporal value is not an integer")),
        }
    }

    /// `pc.strftime(values, format=...)` of a timestamp array, with `%S`
    /// carrying the unit's fraction and, with `zone_suffix`, `%z` after it.
    pub fn strftime(&self, zone_suffix: bool) -> Result<Vec<Option<String>>, Error> {
        let ArrowType::Timestamp(unit, zone) = &self.data_type else {
            return Err(Error::new("strftime of an array that holds no timestamps"));
        };
        self.values
            .iter()
            .map(|value| {
                if matches!(value, Py::None) {
                    return Ok(None);
                }
                let ticks = Self::storage(value)?;
                let per = unit.per_second();
                let seconds = ticks.div_euclid(per);
                let fraction = ticks.rem_euclid(per);
                let offset = match zone {
                    Some(zone) => zone_offset(
                        zone,
                        i64::try_from(seconds).map_err(|_| Error::new("a time out of range"))?,
                    )?,
                    None => 0,
                };
                let local = Civil::from_epoch_nanos((seconds + i128::from(offset)) * 1_000_000_000);
                let mut text = format!("{}T{}", local.date_text(), local.time_text());
                if unit.digits() > 0 {
                    text.push_str(&format!(".{:0width$}", fraction, width = unit.digits()));
                }
                if zone_suffix {
                    text.push_str(&offset_text(offset, false));
                }
                Ok(Some(text))
            })
            .collect()
    }

    /// `to_pylist()`: what each value is in Python, a date, time or
    /// duration as its own Python object.
    pub fn to_pylist(&self) -> Result<Vec<Py>, Error> {
        self.values
            .iter()
            .map(|value| {
                if matches!(value, Py::None) {
                    return Ok(Py::None);
                }
                Ok(match &self.data_type {
                    ArrowType::Date32 => date_value(Self::storage(value)?)?,
                    ArrowType::Date64 => date_value(Self::storage(value)?.div_euclid(86_400_000))?,
                    ArrowType::Time32(unit) | ArrowType::Time64(unit) => {
                        let micros = Self::storage(value)? * 1_000_000 / unit.per_second();
                        Py::Time(time_of_day_text(micros as i64))
                    }
                    ArrowType::Duration(unit) => {
                        let micros = Self::storage(value)? * 1_000_000 / unit.per_second();
                        Py::Timedelta(micros as i64)
                    }
                    ArrowType::Timestamp(..) => {
                        return Err(Error::new("a timestamp array read as Python values"));
                    }
                    ArrowType::Decimal { .. } => match value {
                        Py::Str(text) => Py::Decimal(text.clone()),
                        _ => return Err(Error::new("a decimal is not text")),
                    },
                    _ => value.clone(),
                })
            })
            .collect()
    }
}

/// The `datetime.date` `days` after the epoch.
fn date_value(days: i128) -> Result<Py, Error> {
    let days = i64::try_from(days).map_err(|_| Error::new("a date out of range"))?;
    let (year, month, day) = civil_from_days(days);
    if !(1..=9999).contains(&year) {
        return Err(Error::new(format!("year {year} is out of range")));
    }
    Ok(Py::Date(format!("{year:04}-{month:02}-{day:02}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int::Int;

    fn storage(value: i64) -> Py {
        Py::Int(Int::from(value))
    }

    #[test]
    fn types_spell_as_pyarrow() {
        for spelled in [
            "timestamp[ns, tz=America/New_York]",
            "timestamp[s]",
            "decimal128(38, 0)",
            "date32[day]",
            "time64[us]",
            "uint64",
            "large_string",
        ] {
            assert_eq!(ArrowType::parse(spelled).unwrap().to_string(), spelled);
        }
        assert!(ArrowType::parse("no_such_type").is_err());
    }

    #[test]
    fn strftime_as_arrow() {
        let array = ArrowArray::new(
            ArrowType::parse("timestamp[ms, tz=+05:30]").unwrap(),
            vec![storage(1_234_567), storage(-1), Py::None],
            None,
        )
        .unwrap();
        assert_eq!(
            array.strftime(true).unwrap(),
            [
                Some("1970-01-01T05:50:34.567+0530".to_string()),
                Some("1970-01-01T05:29:59.999+0530".to_string()),
                None
            ]
        );
    }
}
