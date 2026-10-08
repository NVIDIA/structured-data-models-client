// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A query's result as the engine reads it: `rows(table)`, each value as
//! `to_pylist` gives it, or as psycopg reads it back on PostgreSQL; and the
//! columns of an inference sample.

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, TimeUnit};

use crate::dt::civil::Wall;
use crate::frame::decimal_text;
use crate::pyval::PyVal;

/// The field metadata key holding a PostgreSQL column's type OID.
pub const TYPE_OID_KEY: &str = "postgres.type_oid";

/// One column of a result: its field, and its values across the batches.
#[derive(Clone, Debug)]
pub struct ResultColumn {
    pub field: Field,
    pub arrays: Vec<ArrayRef>,
}

impl ResultColumn {
    pub fn name(&self) -> &str {
        self.field.name()
    }

    pub fn data_type(&self) -> &DataType {
        self.field.data_type()
    }

    /// The PostgreSQL type OID the field carries, if any.
    pub fn oid(&self) -> Option<&str> {
        self.field.metadata().get(TYPE_OID_KEY).map(String::as_str)
    }

    /// `column.to_pylist()`.
    pub fn values(&self) -> Vec<PyVal> {
        let mut out = Vec::new();
        for array in &self.arrays {
            for i in 0..array.len() {
                out.push(value_at(array, i));
            }
        }
        out
    }
}

/// A result, as Arrow gives it.
#[derive(Clone, Debug, Default)]
pub struct ResultTable {
    pub columns: Vec<ResultColumn>,
    pub num_rows: usize,
}

impl ResultTable {
    pub fn from_batches(schema: &arrow_schema::Schema, batches: &[RecordBatch]) -> Self {
        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| ResultColumn {
                field: field.as_ref().clone(),
                arrays: batches.iter().map(|b| b.column(i).clone()).collect(),
            })
            .collect();
        ResultTable {
            columns,
            num_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        }
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name().to_string()).collect()
    }

    /// `rows(table)`: each row as a tuple of Python values.
    pub fn rows(&self) -> Vec<Vec<PyVal>> {
        let columns: Vec<Vec<PyVal>> = self.columns.iter().map(ResultColumn::values).collect();
        (0..self.num_rows)
            .map(|r| columns.iter().map(|c| c[r].clone()).collect())
            .collect()
    }

    /// The PostgreSQL backend's `_rows`: as psycopg reads each value back,
    /// with `shown` the types read only where a row is shown (`uuid`,
    /// `inet`, `cidr`).
    pub fn pg_columns(&self, shown: bool) -> Vec<Vec<PyVal>> {
        self.columns
            .iter()
            .map(|c| {
                let values = c.values();
                match c.oid() {
                    Some("1700") => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::Str(s) => PyVal::Decimal(s),
                            other => other,
                        })
                        .collect(),
                    Some("1231") => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::List(items) => PyVal::List(
                                items
                                    .into_iter()
                                    .map(|i| match i {
                                        PyVal::Str(s) => PyVal::Decimal(s),
                                        other => other,
                                    })
                                    .collect(),
                            ),
                            other => other,
                        })
                        .collect(),
                    Some("114") | Some("3802") => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::Str(s) => serde_json::from_str::<serde_json::Value>(&s)
                                .ok()
                                .and_then(|j| PyVal::decode(&j).ok())
                                .unwrap_or(PyVal::Str(s)),
                            other => other,
                        })
                        .collect(),
                    Some("1266") => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::Str(s) => timetz(&s),
                            other => other,
                        })
                        .collect(),
                    Some(oid @ ("869" | "650")) if shown => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::Str(s) => address(&s, oid),
                            other => other,
                        })
                        .collect(),
                    Some("2950") if shown => values
                        .into_iter()
                        .map(|v| match v {
                            PyVal::Str(s) => PyVal::Uuid(s),
                            other => other,
                        })
                        .collect(),
                    _ => values,
                }
            })
            .collect()
    }

    pub fn pg_rows(&self) -> Vec<Vec<PyVal>> {
        let columns = self.pg_columns(true);
        (0..self.num_rows)
            .map(|r| columns.iter().map(|c| c[r].clone()).collect())
            .collect()
    }

    /// The table without its first `n` columns: `table.select(range(n, ...))`.
    pub fn drop_first(&self, n: usize) -> ResultTable {
        ResultTable {
            columns: self.columns.iter().skip(n).cloned().collect(),
            num_rows: self.num_rows,
        }
    }
}

fn wall(value: i64, unit: &TimeUnit) -> Wall {
    let nanos: i128 = match unit {
        TimeUnit::Second => i128::from(value) * 1_000_000_000,
        TimeUnit::Millisecond => i128::from(value) * 1_000_000,
        TimeUnit::Microsecond => i128::from(value) * 1000,
        TimeUnit::Nanosecond => i128::from(value),
    };
    Wall::from_epoch_nanos(nanos)
}

fn zone_offset(tz: Option<&str>) -> Option<i64> {
    let tz = tz?;
    if tz == "UTC" || tz == "+00:00" || tz == "Z" {
        return Some(0);
    }
    if let Some(rest) = tz.strip_prefix(['+', '-']) {
        let sign = if tz.starts_with('-') { -1 } else { 1 };
        let mut parts = rest.split(':');
        let h: i64 = parts.next()?.parse().ok()?;
        let m: i64 = parts.next().unwrap_or("0").parse().ok()?;
        return Some(sign * (h * 3600 + m * 60));
    }
    Some(0)
}

/// A `datetime.time` from microseconds since midnight.
pub fn time_of_day(micros: i64, offset: Option<i64>) -> PyVal {
    let micros = micros.rem_euclid(86_400_000_000);
    let seconds = micros / 1_000_000;
    PyVal::Time {
        hour: (seconds / 3600) as u32,
        minute: (seconds % 3600 / 60) as u32,
        second: (seconds % 60) as u32,
        micro: (micros % 1_000_000) as u32,
        offset,
    }
}

/// A `timetz` as psycopg reads its text: a time with its zone, or the text
/// where Python has no such time (`24:00:00+00`).
fn timetz(text: &str) -> PyVal {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"\A(\d+):(\d+):(\d+)(?:\.(\d+))?([-+])(\d+)(?::(\d+))?(?::(\d+))?\z")
            .expect("pattern compiles")
    });
    let keep = || PyVal::Str(text.to_string());
    let Some(m) = re.captures(text) else {
        return keep();
    };
    let number = |i: usize| -> Option<i64> {
        match m.get(i) {
            None => Some(0),
            Some(g) => g.as_str().parse().ok(),
        }
    };
    let (Some(hour), Some(minute), Some(second)) = (number(1), number(2), number(3)) else {
        return keep();
    };
    let fraction = m.get(4).map_or(String::new(), |g| g.as_str().to_string());
    if fraction.chars().count() > 6 {
        return keep();
    }
    let micro: i64 = format!("{fraction:0<6}").parse().unwrap_or(0);
    let (Some(h), Some(mi), Some(s)) = (number(6), number(7), number(8)) else {
        return keep();
    };
    let shift = h * 3600 + mi * 60 + s;
    let offset = if &m[5] == "-" { -shift } else { shift };
    if !(0..24).contains(&hour)
        || !(0..60).contains(&minute)
        || !(0..60).contains(&second)
        || offset.abs() >= 86_400
    {
        return keep();
    }
    PyVal::Time {
        hour: hour as u32,
        minute: minute as u32,
        second: second as u32,
        micro: micro as u32,
        offset: Some(offset),
    }
}

/// An `inet`, `cidr` as psycopg reads one where it is shown: its
/// `ipaddress` object.
fn address(text: &str, oid: &str) -> PyVal {
    let v6 = text.contains(':');
    let kind = match (oid, text.contains('/')) {
        ("650", _) => "Network",
        (_, true) => "Interface",
        _ => "Address",
    };
    let class = format!("IPv{}{kind}", if v6 { 6 } else { 4 });
    PyVal::Opaque {
        kind: "object".to_string(),
        repr: format!("{class}('{text}')"),
        type_name: class,
    }
}

/// `array[i].as_py()`.
pub fn value_at(array: &ArrayRef, i: usize) -> PyVal {
    if array.is_null(i) {
        return PyVal::None;
    }
    macro_rules! prim {
        ($t:ty) => {
            PyVal::Int(i128::from(array.as_primitive::<$t>().value(i)))
        };
    }
    match array.data_type() {
        DataType::Null => PyVal::None,
        DataType::Boolean => PyVal::Bool(array.as_boolean().value(i)),
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::UInt8 => prim!(UInt8Type),
        DataType::UInt16 => prim!(UInt16Type),
        DataType::UInt32 => prim!(UInt32Type),
        DataType::UInt64 => prim!(UInt64Type),
        DataType::Float16 => PyVal::Float(f64::from(
            array.as_primitive::<Float16Type>().value(i).to_f32(),
        )),
        DataType::Float32 => PyVal::Float(f64::from(array.as_primitive::<Float32Type>().value(i))),
        DataType::Float64 => PyVal::Float(array.as_primitive::<Float64Type>().value(i)),
        DataType::Utf8 => PyVal::Str(array.as_string::<i32>().value(i).to_string()),
        DataType::LargeUtf8 => PyVal::Str(array.as_string::<i64>().value(i).to_string()),
        DataType::Utf8View => PyVal::Str(array.as_string_view().value(i).to_string()),
        DataType::Binary => PyVal::Bytes(array.as_binary::<i32>().value(i).to_vec()),
        DataType::LargeBinary => PyVal::Bytes(array.as_binary::<i64>().value(i).to_vec()),
        DataType::Date32 => {
            let days = array.as_primitive::<Date32Type>().value(i);
            let (y, m, d) = crate::dt::civil::civil_from_days(i64::from(days));
            PyVal::Date(y, m, d)
        }
        DataType::Date64 => {
            let ms = array.as_primitive::<Date64Type>().value(i);
            let (y, m, d) = crate::dt::civil::civil_from_days(ms.div_euclid(86_400_000));
            PyVal::Date(y, m, d)
        }
        DataType::Timestamp(unit, tz) => {
            let v = match unit {
                TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(i),
                TimeUnit::Millisecond => array.as_primitive::<TimestampMillisecondType>().value(i),
                TimeUnit::Microsecond => array.as_primitive::<TimestampMicrosecondType>().value(i),
                TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(i),
            };
            let w = wall(v, unit);
            let offset = zone_offset(tz.as_deref());
            match offset {
                None => {
                    if matches!(unit, TimeUnit::Nanosecond) {
                        PyVal::Timestamp(w)
                    } else {
                        PyVal::DateTime(w, None)
                    }
                }
                Some(o) => {
                    let local =
                        Wall::from_epoch_nanos(w.epoch_nanos() + i128::from(o) * 1_000_000_000);
                    PyVal::DateTime(local, Some(o))
                }
            }
        }
        DataType::Decimal128(_, scale) => PyVal::Decimal(decimal_text(
            array.as_primitive::<Decimal128Type>().value(i),
            *scale,
        )),
        DataType::Decimal256(_, scale) => PyVal::Decimal(crate::frame::decimal_digits_text(
            &array.as_primitive::<Decimal256Type>().value(i).to_string(),
            *scale,
        )),
        // A time of day, as the `datetime.time` the drivers read.
        DataType::Time32(unit) | DataType::Time64(unit) => {
            let micros: i64 = match unit {
                TimeUnit::Second => {
                    i64::from(array.as_primitive::<Time32SecondType>().value(i)) * 1_000_000
                }
                TimeUnit::Millisecond => {
                    i64::from(array.as_primitive::<Time32MillisecondType>().value(i)) * 1_000
                }
                TimeUnit::Microsecond => array.as_primitive::<Time64MicrosecondType>().value(i),
                TimeUnit::Nanosecond => {
                    array.as_primitive::<Time64NanosecondType>().value(i) / 1_000
                }
            };
            let seconds = micros.div_euclid(1_000_000);
            PyVal::Time {
                hour: (seconds / 3600) as u32,
                minute: (seconds % 3600 / 60) as u32,
                second: (seconds % 60) as u32,
                micro: micros.rem_euclid(1_000_000) as u32,
                offset: None,
            }
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let inner = list.value(i);
            PyVal::List((0..inner.len()).map(|k| value_at(&inner, k)).collect())
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let inner = list.value(i);
            PyVal::List((0..inner.len()).map(|k| value_at(&inner, k)).collect())
        }
        other => PyVal::Opaque {
            kind: "object".to_string(),
            type_name: other.to_string(),
            repr: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_timetz_as_psycopg_does() {
        assert_eq!(
            timetz("01:02:03+05:30").repr(),
            "datetime.time(1, 2, 3, tzinfo=datetime.timezone(datetime.timedelta(seconds=19800)))"
        );
        assert_eq!(
            timetz("10:00:00.5+00").repr(),
            "datetime.time(10, 0, 0, 500000, tzinfo=datetime.timezone.utc)"
        );
        assert_eq!(
            timetz("23:59:59-08").repr(),
            "datetime.time(23, 59, 59, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=57600)))"
        );
        assert_eq!(timetz("24:00:00+00"), PyVal::Str("24:00:00+00".into()));
        assert_eq!(timetz("12:00:00+05:53:28").str_(), "12:00:00+05:53:28");
    }

    #[test]
    fn writes_a_time_of_day_as_python_does() {
        assert_eq!(
            time_of_day(36_000_000_000, None).repr(),
            "datetime.time(10, 0)"
        );
        assert_eq!(
            time_of_day(36_030_000_005, None).repr(),
            "datetime.time(10, 0, 30, 5)"
        );
        assert_eq!(time_of_day(36_030_000_005, None).str_(), "10:00:30.000005");
    }

    #[test]
    fn shows_an_address_as_ipaddress_does() {
        assert_eq!(address("10.0.0.1", "869").repr(), "IPv4Address('10.0.0.1')");
        assert_eq!(
            address("10.0.0.1/24", "869").repr(),
            "IPv4Interface('10.0.0.1/24')"
        );
        assert_eq!(address("::1", "869").repr(), "IPv6Address('::1')");
        assert_eq!(
            address("10.0.0.0/8", "650").repr(),
            "IPv4Network('10.0.0.0/8')"
        );
    }
}
