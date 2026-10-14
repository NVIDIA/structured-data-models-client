// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DuckDB's extension types read as what they mean (register P1B-43, the
//! owner's decision of 2026-10-14, item 26: read, not left out), before
//! [`crate::admission`] takes the data into the closed type set.
//!
//! | Received (a top-level column) | Read as |
//! |---|---|
//! | `fixed_size_binary(16)` carrying `arrow.uuid` | `utf8`, the canonical lower-case 8-4-4-4-12 text |
//! | `arrow.opaque` `bit` (binary storage) | `utf8`, its bits as `0` and `1` |
//! | `arrow.opaque` `time_tz` (`fixed_size_binary(8)`) | `time64[us]`, the time of day as written, its offset dropped |
//! | `arrow.opaque` `hugeint`, `uhugeint` (`fixed_size_binary(16)`), `bignum` (any binary storage) | by the big-integer rule |
//!
//! The big-integer rule (the orchestrator's, 2026-10-14) reads the whole
//! column of the data handed over: `int64` where every value fits it, else
//! `decimal128(38, 0)` where every value fits 38 digits, else `utf8`, each
//! value's exact decimal text (DuckDB's own `CAST(x AS VARCHAR)`). The
//! column read keeps its name, nullability and metadata but the extension's
//! two keys, which would name a storage it no longer has. A value whose
//! bytes the type's layout does not read is refused with `UNSUPPORTED_TYPE`
//! naming the column and the type received, never guessed. Any other
//! extension, or one of these over another storage, is left to admission.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray, Time64MicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

const EXTENSION_NAME: &str = "ARROW:extension:name";
const EXTENSION_METADATA: &str = "ARROW:extension:metadata";
const UUID: &str = "arrow.uuid";
const OPAQUE: &str = "arrow.opaque";

/// The largest magnitude 38 decimal digits hold.
const MOST_38_DIGITS: u128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

/// What a column's extension type means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Meaning {
    Uuid,
    Bit,
    TimeTz,
    HugeInt,
    UHugeInt,
    BigNum,
}

fn is_binary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    )
}

/// What `field`'s extension type means, if it is one read here over the
/// storage DuckDB gives it. An opaque type is known by its `type_name` and
/// storage; its vendor is not read, since a vendor's name is a dialect's,
/// which `l0_scan.py`'s `dialects` check keeps in `sdm-data` (a value whose
/// bytes do not read in the layout is refused, never guessed).
fn meaning(field: &Field) -> Option<Meaning> {
    let metadata = field.metadata();
    let storage = field.data_type();
    match metadata.get(EXTENSION_NAME).map(String::as_str)? {
        UUID if storage == &DataType::FixedSizeBinary(16) => Some(Meaning::Uuid),
        OPAQUE => {
            let opaque: serde_json::Value =
                serde_json::from_str(metadata.get(EXTENSION_METADATA)?).ok()?;
            match (opaque.get("type_name")?.as_str()?, storage) {
                ("hugeint", DataType::FixedSizeBinary(16)) => Some(Meaning::HugeInt),
                ("uhugeint", DataType::FixedSizeBinary(16)) => Some(Meaning::UHugeInt),
                ("time_tz", DataType::FixedSizeBinary(8)) => Some(Meaning::TimeTz),
                ("bignum" | "varint", s) if is_binary(s) => Some(Meaning::BigNum),
                ("bit", s) if is_binary(s) => Some(Meaning::Bit),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The refusal of a value whose bytes the type's layout does not read.
fn unreadable(column: &str, received: &DataType) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedType,
        format!("column {column:?} holds a value of Arrow type {received} the core cannot read"),
    )
    .with_field("column", Detail::text(column))
    .with_field("arrow_type", Detail::text(received.to_string()))
}

/// Each slot of a binary-family array: its bytes, or `None` for a null.
fn slots(array: &ArrayRef) -> Vec<Option<&[u8]>> {
    match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().iter().collect(),
        DataType::LargeBinary => array.as_binary::<i64>().iter().collect(),
        DataType::BinaryView => array.as_binary_view().iter().collect(),
        DataType::FixedSizeBinary(_) => array.as_fixed_size_binary().iter().collect(),
        _ => vec![None; array.len()],
    }
}

/// A big integer: its sign and its magnitude.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Big {
    /// Fits 128 bits of magnitude.
    Small { negative: bool, magnitude: u128 },
    /// Wider: its magnitude, big-endian, no leading zero bytes.
    Wide { negative: bool, magnitude: Vec<u8> },
}

impl Big {
    fn of(negative: bool, magnitude: &[u8]) -> Big {
        let first = magnitude
            .iter()
            .position(|b| *b != 0)
            .unwrap_or(magnitude.len());
        let magnitude = &magnitude[first..];
        if magnitude.len() <= 16 {
            let small = magnitude
                .iter()
                .fold(0u128, |n, b| (n << 8) | u128::from(*b));
            Big::Small {
                negative: negative && small != 0,
                magnitude: small,
            }
        } else {
            Big::Wide {
                negative,
                magnitude: magnitude.to_vec(),
            }
        }
    }

    /// As an i128 where it fits.
    fn as_i128(&self) -> Option<i128> {
        let Big::Small {
            negative,
            magnitude,
        } = self
        else {
            return None;
        };
        if *negative {
            0i128.checked_sub_unsigned(*magnitude)
        } else {
            i128::try_from(*magnitude).ok()
        }
    }

    fn fits_38_digits(&self) -> bool {
        matches!(self, Big::Small { magnitude, .. } if *magnitude <= MOST_38_DIGITS)
    }

    /// Its exact decimal text, as DuckDB writes it.
    fn text(&self) -> String {
        match self {
            Big::Small {
                negative,
                magnitude,
            } => format!("{}{magnitude}", if *negative { "-" } else { "" }),
            Big::Wide {
                negative,
                magnitude,
            } => {
                // Repeated division by 10^9 of the big-endian magnitude.
                let mut digits = magnitude.clone();
                let mut groups = Vec::new();
                while digits.iter().any(|b| *b != 0) {
                    let mut rest = 0u64;
                    for byte in &mut digits {
                        let value = (rest << 8) | u64::from(*byte);
                        *byte = u8::try_from(value / 1_000_000_000).unwrap_or(u8::MAX);
                        rest = value % 1_000_000_000;
                    }
                    groups.push(rest);
                }
                let mut text = String::from(if *negative { "-" } else { "" });
                for (i, group) in groups.iter().rev().enumerate() {
                    if i == 0 {
                        text.push_str(&group.to_string());
                    } else {
                        text.push_str(&format!("{group:09}"));
                    }
                }
                text
            }
        }
    }
}

/// A HUGEINT's 16 little-endian bytes.
fn hugeint(bytes: &[u8]) -> Option<Big> {
    let value = i128::from_le_bytes(bytes.try_into().ok()?);
    Some(Big::Small {
        negative: value < 0,
        magnitude: value.unsigned_abs(),
    })
}

/// A UHUGEINT's 16 little-endian bytes.
fn uhugeint(bytes: &[u8]) -> Option<Big> {
    Some(Big::Small {
        negative: false,
        magnitude: u128::from_le_bytes(bytes.try_into().ok()?),
    })
}

/// A BIGNUM (VARINT): a three-byte big-endian header, the size of the
/// magnitude with the sign bit 0x800000 set for a value not negative and
/// every bit inverted for a negative one, then the big-endian magnitude,
/// its bytes inverted for a negative value.
fn bignum(bytes: &[u8]) -> Option<Big> {
    let (header, data) = (bytes.get(..3)?, bytes.get(3..)?);
    let header = u32::from_be_bytes([0, header[0], header[1], header[2]]);
    let negative = header & 0x80_0000 == 0;
    let size = if negative {
        !header & 0x7f_ffff
    } else {
        header & 0x7f_ffff
    };
    if usize::try_from(size).ok()? != data.len() || data.is_empty() {
        return None;
    }
    let magnitude: Vec<u8> = if negative {
        data.iter().map(|b| !b).collect()
    } else {
        data.to_vec()
    };
    Some(Big::of(negative, &magnitude))
}

/// A BIT: a byte counting the padding bits, then the bits, the first byte's
/// first bits being the padding.
fn bits(bytes: &[u8]) -> Option<String> {
    let (&padding, data) = bytes.split_first()?;
    if padding >= 8 || data.is_empty() {
        return None;
    }
    let all = data.iter().flat_map(|byte| {
        (0..8)
            .rev()
            .map(move |i| if byte >> i & 1 == 1 { '1' } else { '0' })
    });
    Some(all.skip(usize::from(padding)).collect())
}

/// A UUID's 16 big-endian bytes as its canonical lower-case text.
fn uuid(bytes: &[u8]) -> Option<String> {
    if bytes.len() != 16 {
        return None;
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    Some(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}

/// A TIMETZ's 8 little-endian bytes: the microseconds of the day in the top
/// 40 bits, the offset below.
fn time_tz(bytes: &[u8]) -> Option<i64> {
    let bits = u64::from_le_bytes(bytes.try_into().ok()?);
    i64::try_from(bits >> 24).ok()
}

/// Reads each slot of `array` by `read`, a null as a null; `None` if a
/// value's bytes do not read.
fn each<T>(array: &ArrayRef, read: impl Fn(&[u8]) -> Option<T>) -> Option<Vec<Option<T>>> {
    slots(array)
        .into_iter()
        .map(|slot| match slot {
            None => Some(None),
            Some(bytes) => read(bytes).map(Some),
        })
        .collect()
}

/// The type the big-integer rule reads `values` as.
fn big_type(values: &[Option<Big>]) -> DataType {
    let held = || values.iter().flatten();
    if held().all(|v| v.as_i128().is_some_and(|n| i64::try_from(n).is_ok())) {
        DataType::Int64
    } else if held().all(Big::fits_38_digits) {
        DataType::Decimal128(38, 0)
    } else {
        DataType::Utf8
    }
}

fn big_array(values: &[Option<Big>], read_as: &DataType) -> Result<ArrayRef, CoreError> {
    let small = |v: &Big| v.as_i128().unwrap_or_default();
    Ok(match read_as {
        DataType::Int64 => Arc::new(
            values
                .iter()
                .map(|v| {
                    v.as_ref()
                        .map(|v| i64::try_from(small(v)).unwrap_or_default())
                })
                .collect::<Int64Array>(),
        ),
        DataType::Decimal128(precision, scale) => Arc::new(
            values
                .iter()
                .map(|v| v.as_ref().map(small))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| {
                    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, e.to_string())
                })?,
        ),
        _ => Arc::new(
            values
                .iter()
                .map(|v| v.as_ref().map(Big::text))
                .collect::<StringArray>(),
        ),
    })
}

/// `field` as read, its type `read_as`, the extension's keys dropped.
fn read_field(field: &Field, read_as: DataType) -> Field {
    let mut metadata = field.metadata().clone();
    metadata.remove(EXTENSION_NAME);
    metadata.remove(EXTENSION_METADATA);
    Field::new(field.name(), read_as, field.is_nullable()).with_metadata(metadata)
}

/// `schema` and `batches` with each top-level column of an extension type
/// read here (see the module's table) as what it means; unchanged where
/// there is none. The batches are those checked against `schema`.
pub(crate) fn read(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(SchemaRef, Vec<RecordBatch>), CoreError> {
    let meanings: Vec<Option<Meaning>> = schema.fields().iter().map(|f| meaning(f)).collect();
    if meanings.iter().all(Option::is_none) {
        return Ok((Arc::clone(schema), batches.to_vec()));
    }
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut columns: Vec<Vec<ArrayRef>> = batches.iter().map(|b| b.columns().to_vec()).collect();
    for (index, meaning) in meanings.iter().enumerate() {
        let Some(meaning) = meaning else { continue };
        let field = schema.field(index);
        let refused = || unreadable(field.name(), field.data_type());
        let arrays: Vec<ArrayRef> = batches
            .iter()
            .map(|b| Arc::clone(b.column(index)))
            .collect();
        let (read_as, read): (DataType, Vec<ArrayRef>) = match meaning {
            Meaning::Uuid | Meaning::Bit => {
                let text = |bytes: &[u8]| {
                    if *meaning == Meaning::Uuid {
                        uuid(bytes)
                    } else {
                        bits(bytes)
                    }
                };
                let read = arrays
                    .iter()
                    .map(|a| {
                        each(a, text)
                            .map(|v| Arc::new(v.into_iter().collect::<StringArray>()) as ArrayRef)
                            .ok_or_else(refused)
                    })
                    .collect::<Result<_, _>>()?;
                (DataType::Utf8, read)
            }
            Meaning::TimeTz => {
                let read = arrays
                    .iter()
                    .map(|a| {
                        each(a, time_tz)
                            .map(|v| {
                                Arc::new(v.into_iter().collect::<Time64MicrosecondArray>())
                                    as ArrayRef
                            })
                            .ok_or_else(refused)
                    })
                    .collect::<Result<_, _>>()?;
                (DataType::Time64(arrow_schema::TimeUnit::Microsecond), read)
            }
            Meaning::HugeInt | Meaning::UHugeInt | Meaning::BigNum => {
                let decode = match meaning {
                    Meaning::HugeInt => hugeint,
                    Meaning::UHugeInt => uhugeint,
                    _ => bignum,
                };
                let values = arrays
                    .iter()
                    .map(|a| each(a, decode).ok_or_else(refused))
                    .collect::<Result<Vec<_>, _>>()?;
                let all: Vec<Option<Big>> = values.iter().flatten().cloned().collect();
                let read_as = big_type(&all);
                let read = values
                    .iter()
                    .map(|v| big_array(v, &read_as))
                    .collect::<Result<_, _>>()?;
                (read_as, read)
            }
        };
        fields[index] = read_field(field, read_as);
        for (batch, array) in columns.iter_mut().zip(read) {
            batch[index] = array;
        }
    }
    let schema: SchemaRef = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let batches = columns
        .into_iter()
        .zip(batches)
        .map(|(columns, batch)| {
            RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|e| {
                CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, e.to_string())
            })
        })
        .collect::<Result<_, _>>()?;
    Ok((schema, batches))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bignum_reads_its_sign_and_magnitude() {
        let text = |bytes: &[u8]| bignum(bytes).map(|b| b.text());
        assert_eq!(text(&[0x80, 0, 1, 0]).as_deref(), Some("0"));
        assert_eq!(text(&[0x80, 0, 1, 7]).as_deref(), Some("7"));
        assert_eq!(text(&[0x80, 0, 2, 1, 0]).as_deref(), Some("256"));
        assert_eq!(text(&[0x7f, 0xff, 0xfe, 0xfe]).as_deref(), Some("-1"));
        assert_eq!(
            text(&[0x7f, 0xff, 0xfd, 0xfe, 0xff]).as_deref(),
            Some("-256")
        );
        // 10^40, as DuckDB 1.5.6 hands it over.
        let wide = [
            0x80, 0, 0x11, 0x1d, 0x63, 0x29, 0xf1, 0xc3, 0x5c, 0xa4, 0xbf, 0xab, 0xb9, 0xf5, 0x61,
            0, 0, 0, 0, 0,
        ];
        assert_eq!(text(&wide), Some(format!("1{}", "0".repeat(40))));
        assert_eq!(bignum(&[0x80, 0, 2, 7]), None);
        assert_eq!(bignum(&[0x80, 0]), None);
    }

    #[test]
    fn the_big_integer_rule_takes_the_narrowest_type_every_value_fits() {
        let small = |n: i128| {
            Some(Big::Small {
                negative: n < 0,
                magnitude: n.unsigned_abs(),
            })
        };
        assert_eq!(
            big_type(&[small(-7), None, small(i128::from(i64::MAX))]),
            DataType::Int64
        );
        assert_eq!(
            big_type(&[small(i128::from(i64::MIN) - 1)]),
            DataType::Decimal128(38, 0)
        );
        assert_eq!(
            big_type(&[small(-(MOST_38_DIGITS as i128))]),
            DataType::Decimal128(38, 0)
        );
        assert_eq!(
            big_type(&[small(MOST_38_DIGITS as i128 + 1)]),
            DataType::Utf8
        );
        let unsigned = Big::Small {
            negative: false,
            magnitude: u128::MAX,
        };
        assert_eq!(big_type(&[Some(unsigned.clone())]), DataType::Utf8);
        assert_eq!(unsigned.text(), u128::MAX.to_string());
        assert_eq!(big_type(&[]), DataType::Int64);
    }

    #[test]
    fn bits_uuids_and_times_of_day_read_as_duckdb_writes_them() {
        assert_eq!(bits(&[5, 0xfd]).as_deref(), Some("101"));
        assert_eq!(bits(&[6, 0xfc, 0x03]).as_deref(), Some("0000000011"));
        assert_eq!(bits(&[8, 0xff]), None);
        assert_eq!(bits(&[0]), None);
        let id = [
            0xa0, 0xee, 0xbc, 0x99, 0x9c, 0x0b, 0x4e, 0xf8, 0xbb, 0x6d, 0x6b, 0xb9, 0xbd, 0x38,
            0x0a, 0x11,
        ];
        assert_eq!(
            uuid(&id).as_deref(),
            Some("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")
        );
        assert_eq!(uuid(&id[..15]), None);
        // 12:34:56.789+01:30.
        let tz = [0xe7, 0xcb, 0x00, 0x08, 0x26, 0xe6, 0x8b, 0x0a];
        assert_eq!(time_tz(&tz), Some(45_296_789_000));
        assert_eq!(
            hugeint(&(-7i128).to_le_bytes())
                .map(|b| b.text())
                .as_deref(),
            Some("-7")
        );
        assert_eq!(
            uhugeint(&u128::MAX.to_le_bytes()).map(|b| b.text()),
            Some(u128::MAX.to_string())
        );
    }
}
