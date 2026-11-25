// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small native storage observations attached to direct input columns.
//! These describe storage, never the model's chosen dtype or meaning.

use arrow_schema::{DataType, Field};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::FloatStorage;
use crate::{CoreError, Sentence};

pub const DIRECT_STORAGE_KEY: &str = "sdm.direct.storage.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StorageOwner {
    Array,
    Nullable,
    Arrow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TimestampUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimestampUnit {
    pub fn microseconds(self, count: i64) -> i128 {
        let count = i128::from(count);
        match self {
            Self::Seconds => count * 1_000_000,
            Self::Milliseconds => count * 1_000,
            Self::Microseconds => count,
            Self::Nanoseconds => count.div_euclid(1_000),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
pub enum DirectStorage {
    Primitive {
        owner: StorageOwner,
        floats: Option<FloatStorage>,
    },
    String,
    Categorical,
    Timestamp {
        unit: TimestampUnit,
        zone: Option<String>,
    },
    Decimal {
        precision: u8,
        scale: i8,
    },
    /// Values already encoded by the direct cell policy at their native
    /// consumption points. Each UTF-8 cell carries one contract Scalar.
    Encoded {
        dtype: crate::ops::Dtype,
    },
}

/// Decode and validate once at the attachment boundary. Absence is the legacy
/// primitive contract; present but malformed observations never use that default.
pub fn read(field: &Field) -> Result<Option<DirectStorage>, CoreError> {
    if field
        .metadata()
        .keys()
        .any(|key| key.starts_with("sdm.direct.storage.") && key != DIRECT_STORAGE_KEY)
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "direct native storage metadata has an unsupported version",
        ));
    }
    let Some(encoded) = field.metadata().get(DIRECT_STORAGE_KEY) else {
        return Ok(None);
    };
    let invalid = || {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "direct native storage metadata is malformed or disagrees with its column",
        )
    };
    let storage: DirectStorage = serde_json::from_str(encoded).map_err(|_| invalid())?;
    let primitive = matches!(
        field.data_type(),
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
    );
    let floating = matches!(field.data_type(), DataType::Float32 | DataType::Float64);
    let valid = match &storage {
        DirectStorage::Primitive { owner, floats } => {
            primitive
                && floating == floats.is_some()
                && (!floating
                    || *floats
                        == Some(match owner {
                            StorageOwner::Array => FloatStorage::NanMissing,
                            StorageOwner::Nullable | StorageOwner::Arrow => {
                                FloatStorage::ValidityMasked
                            }
                        }))
        }
        DirectStorage::String => matches!(field.data_type(), DataType::Utf8),
        DirectStorage::Categorical => primitive || matches!(field.data_type(), DataType::Utf8),
        DirectStorage::Timestamp { .. } => matches!(field.data_type(), DataType::Int64),
        DirectStorage::Decimal { precision, scale } => matches!(field.data_type(),
            DataType::Decimal128(p, s) | DataType::Decimal256(p, s) if p == precision && s == scale),
        DirectStorage::Encoded { .. } => matches!(field.data_type(), DataType::Utf8),
    };
    if !valid {
        return Err(invalid());
    }
    Ok(Some(storage))
}

pub fn floats(field: &Field) -> Result<Option<FloatStorage>, CoreError> {
    Ok(match read(field)? {
        Some(DirectStorage::Primitive { floats, .. }) => floats,
        None if matches!(field.data_type(), DataType::Float32 | DataType::Float64) => {
            Some(FloatStorage::NanMissing)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(dtype: DataType, metadata: &str) -> Field {
        Field::new("value", dtype, true).with_metadata(
            [(DIRECT_STORAGE_KEY.to_owned(), metadata.to_owned())]
                .into_iter()
                .collect(),
        )
    }

    #[test]
    fn malformed_or_incompatible_facts_never_use_legacy_defaults() {
        for metadata in [
            "{}",
            r#"{"family":"unknown"}"#,
            r#"{"family":"primitive","owner":"arrow","floats":"nan_missing"}"#,
            r#"{"family":"primitive","owner":"array","floats":null}"#,
            r#"{"family":"string"}"#,
        ] {
            assert!(read(&field(DataType::Float64, metadata)).is_err());
        }
    }

    #[test]
    fn masked_storage_survives_the_metadata_round_trip() {
        let observed = DirectStorage::Primitive {
            owner: StorageOwner::Arrow,
            floats: Some(FloatStorage::ValidityMasked),
        };
        let encoded = serde_json::to_string(&observed).expect("storage serializes");
        let field = field(DataType::Float64, &encoded);
        assert_eq!(read(&field).expect("valid storage"), Some(observed));
        assert_eq!(
            floats(&field).expect("valid storage"),
            Some(FloatStorage::ValidityMasked)
        );
    }

    #[test]
    fn categorical_numbers_keep_their_native_family() {
        let field = field(DataType::Int64, r#"{"family":"categorical"}"#);
        assert_eq!(
            read(&field).expect("valid storage"),
            Some(DirectStorage::Categorical)
        );
        assert_eq!(floats(&field).expect("valid storage"), None);
    }

    #[test]
    fn unsupported_metadata_versions_are_not_legacy_inputs() {
        let field = Field::new("value", DataType::Int64, true).with_metadata(
            [("sdm.direct.storage.v2".to_owned(), "{}".to_owned())]
                .into_iter()
                .collect(),
        );
        assert!(read(&field).is_err());
    }

    #[test]
    fn timestamp_units_keep_pre_epoch_fractions_and_wide_dates() {
        assert_eq!(TimestampUnit::Nanoseconds.microseconds(-1), -1);
        assert_eq!(TimestampUnit::Milliseconds.microseconds(-1), -1_000);
        assert_eq!(
            TimestampUnit::Seconds.microseconds(i64::MAX),
            i128::from(i64::MAX) * 1_000_000
        );
    }

    #[test]
    fn decimal_metadata_must_agree_with_the_physical_scale() {
        let encoded = r#"{"family":"decimal","precision":40,"scale":2}"#;
        assert!(read(&field(DataType::Decimal256(40, 2), encoded)).is_ok());
        assert!(read(&field(DataType::Decimal256(40, 0), encoded)).is_err());
    }
}
