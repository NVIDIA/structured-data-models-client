// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The closed type set (plan section 3).
//!
//! [`ColumnType::from_arrow`] is the only way in: an Arrow type outside the
//! set is refused with `kind: unsupported, code: UNSUPPORTED_TYPE`, naming the
//! column and the Arrow type. Converting the forms a host may hand over
//! (`large_*`, `timestamp[s]` and `[ms]`, `large_list`, `fixed_size_list`)
//! into the set is admission, which lives in `sdm-host`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Fields, TimeUnit};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Detail, ErrorKind, codes};

/// The unit of a timestamp, duration or time of day.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum Unit {
    Ns,
    Us,
}

/// The index type of a dictionary.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum IndexType {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

/// A named, typed column or child of a nested type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type", deserialize_with = "strict_column_type")]
    pub column_type: ColumnType,
    pub nullable: bool,
    /// Arrow field metadata, kept as it came.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// A type of the closed set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColumnType {
    Null,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    Decimal128 {
        precision: u8,
        scale: i8,
    },
    Decimal256 {
        precision: u8,
        scale: i8,
    },
    Utf8,
    Binary,
    Date32,
    Timestamp {
        unit: Unit,
        zone: Option<String>,
    },
    Duration {
        unit: Unit,
    },
    Time64 {
        unit: Unit,
    },
    List {
        item: Box<Field>,
    },
    Struct {
        fields: Vec<Field>,
    },
    /// `entries` is the name of Arrow's entries struct, kept so the type
    /// leaves exactly as it came.
    Map {
        entries: String,
        key: Box<Field>,
        value: Box<Field>,
        sorted: bool,
    },
    DictionaryUtf8 {
        index: IndexType,
    },
}

/// The cell vocabulary of the scope manifests: one name per type, with
/// timestamps split by unit and zone (`phase-0/scope-format.md`). Generated
/// from [`ColumnType::cell_type`], so the manifests and the code cannot
/// disagree.
pub fn cell_types() -> Vec<&'static str> {
    let item = || {
        Box::new(Field {
            name: "item".into(),
            column_type: ColumnType::Null,
            nullable: true,
            metadata: BTreeMap::new(),
        })
    };
    let zone = || Some("UTC".to_owned());
    [
        ColumnType::Int8,
        ColumnType::Int16,
        ColumnType::Int32,
        ColumnType::Int64,
        ColumnType::Uint8,
        ColumnType::Uint16,
        ColumnType::Uint32,
        ColumnType::Uint64,
        ColumnType::Float32,
        ColumnType::Float64,
        ColumnType::Decimal128 {
            precision: 38,
            scale: 0,
        },
        ColumnType::Decimal256 {
            precision: 76,
            scale: 0,
        },
        ColumnType::Utf8,
        ColumnType::Binary,
        ColumnType::Date32,
        ColumnType::Timestamp {
            unit: Unit::Ns,
            zone: None,
        },
        ColumnType::Timestamp {
            unit: Unit::Us,
            zone: None,
        },
        ColumnType::Timestamp {
            unit: Unit::Ns,
            zone: zone(),
        },
        ColumnType::Timestamp {
            unit: Unit::Us,
            zone: zone(),
        },
        ColumnType::Duration { unit: Unit::Ns },
        ColumnType::Duration { unit: Unit::Us },
        ColumnType::Time64 { unit: Unit::Us },
        ColumnType::List { item: item() },
        ColumnType::Struct { fields: vec![] },
        ColumnType::Map {
            entries: "entries".into(),
            key: item(),
            value: item(),
            sorted: false,
        },
        ColumnType::DictionaryUtf8 {
            index: IndexType::Int32,
        },
        ColumnType::Bool,
        ColumnType::Null,
    ]
    .iter()
    .map(ColumnType::cell_type)
    .collect()
}

impl ColumnType {
    /// The type's name in the scope manifests' cell vocabulary.
    pub fn cell_type(&self) -> &'static str {
        use ColumnType as C;
        match self {
            C::Null => "null",
            C::Bool => "bool",
            C::Int8 => "int8",
            C::Int16 => "int16",
            C::Int32 => "int32",
            C::Int64 => "int64",
            C::Uint8 => "uint8",
            C::Uint16 => "uint16",
            C::Uint32 => "uint32",
            C::Uint64 => "uint64",
            C::Float32 => "float32",
            C::Float64 => "float64",
            C::Decimal128 { .. } => "decimal128",
            C::Decimal256 { .. } => "decimal256",
            C::Utf8 => "utf8",
            C::Binary => "binary",
            C::Date32 => "date32",
            C::Timestamp {
                unit: Unit::Ns,
                zone: None,
            } => "timestamp-ns",
            C::Timestamp {
                unit: Unit::Us,
                zone: None,
            } => "timestamp-us",
            C::Timestamp {
                unit: Unit::Ns,
                zone: Some(_),
            } => "timestamp-ns-tz",
            C::Timestamp {
                unit: Unit::Us,
                zone: Some(_),
            } => "timestamp-us-tz",
            C::Duration { unit: Unit::Ns } => "duration-ns",
            C::Duration { unit: Unit::Us } => "duration-us",
            C::Time64 { .. } => "time64",
            C::List { .. } => "list",
            C::Struct { .. } => "struct",
            C::Map { .. } => "map",
            C::DictionaryUtf8 { .. } => "dictionary-utf8",
        }
    }

    /// The closed-set type of the Arrow type `data_type` of `column`, or the
    /// named refusal. `column` names the column, with a path through nested
    /// types (`tags.item`).
    pub fn from_arrow(column: &str, data_type: &DataType) -> Result<Self, CoreError> {
        use ColumnType as C;
        let unit = |u: &TimeUnit| match u {
            TimeUnit::Nanosecond => Some(Unit::Ns),
            TimeUnit::Microsecond => Some(Unit::Us),
            TimeUnit::Second | TimeUnit::Millisecond => None,
        };
        let refuse = || Err(unsupported_type(column, data_type));
        Ok(match data_type {
            DataType::Null => C::Null,
            DataType::Boolean => C::Bool,
            DataType::Int8 => C::Int8,
            DataType::Int16 => C::Int16,
            DataType::Int32 => C::Int32,
            DataType::Int64 => C::Int64,
            DataType::UInt8 => C::Uint8,
            DataType::UInt16 => C::Uint16,
            DataType::UInt32 => C::Uint32,
            DataType::UInt64 => C::Uint64,
            DataType::Float32 => C::Float32,
            DataType::Float64 => C::Float64,
            DataType::Decimal128(precision, scale) => C::Decimal128 {
                precision: *precision,
                scale: *scale,
            },
            DataType::Decimal256(precision, scale) => C::Decimal256 {
                precision: *precision,
                scale: *scale,
            },
            DataType::Utf8 => C::Utf8,
            DataType::Binary => C::Binary,
            DataType::Date32 => C::Date32,
            DataType::Timestamp(u, zone) => match unit(u) {
                Some(unit) => C::Timestamp {
                    unit,
                    zone: zone.as_deref().map(str::to_owned),
                },
                None => return refuse(),
            },
            DataType::Duration(u) => match unit(u) {
                Some(unit) => C::Duration { unit },
                None => return refuse(),
            },
            DataType::Time64(u) => match unit(u) {
                Some(unit) => C::Time64 { unit },
                None => return refuse(),
            },
            DataType::List(item) => C::List {
                item: Box::new(Field::from_arrow_in(column, item)?),
            },
            DataType::Struct(fields) => C::Struct {
                fields: fields
                    .iter()
                    .map(|f| Field::from_arrow_in(column, f))
                    .collect::<Result<_, _>>()?,
            },
            DataType::Map(entries, sorted) => {
                let DataType::Struct(children) = entries.data_type() else {
                    return refuse();
                };
                let [key, value] = children.iter().collect::<Vec<_>>()[..] else {
                    return refuse();
                };
                let path = format!("{column}.{}", entries.name());
                C::Map {
                    entries: entries.name().clone(),
                    key: Box::new(Field::from_arrow_in(&path, key)?),
                    value: Box::new(Field::from_arrow_in(&path, value)?),
                    sorted: *sorted,
                }
            }
            DataType::Dictionary(index, values) if **values == DataType::Utf8 => {
                let index = match **index {
                    DataType::Int8 => IndexType::Int8,
                    DataType::Int16 => IndexType::Int16,
                    DataType::Int32 => IndexType::Int32,
                    DataType::Int64 => IndexType::Int64,
                    DataType::UInt8 => IndexType::Uint8,
                    DataType::UInt16 => IndexType::Uint16,
                    DataType::UInt32 => IndexType::Uint32,
                    DataType::UInt64 => IndexType::Uint64,
                    _ => return refuse(),
                };
                C::DictionaryUtf8 { index }
            }
            _ => return refuse(),
        })
    }

    /// The Arrow type this is.
    pub fn to_arrow(&self) -> DataType {
        use ColumnType as C;
        let unit = |u: &Unit| match u {
            Unit::Ns => TimeUnit::Nanosecond,
            Unit::Us => TimeUnit::Microsecond,
        };
        match self {
            C::Null => DataType::Null,
            C::Bool => DataType::Boolean,
            C::Int8 => DataType::Int8,
            C::Int16 => DataType::Int16,
            C::Int32 => DataType::Int32,
            C::Int64 => DataType::Int64,
            C::Uint8 => DataType::UInt8,
            C::Uint16 => DataType::UInt16,
            C::Uint32 => DataType::UInt32,
            C::Uint64 => DataType::UInt64,
            C::Float32 => DataType::Float32,
            C::Float64 => DataType::Float64,
            C::Decimal128 { precision, scale } => DataType::Decimal128(*precision, *scale),
            C::Decimal256 { precision, scale } => DataType::Decimal256(*precision, *scale),
            C::Utf8 => DataType::Utf8,
            C::Binary => DataType::Binary,
            C::Date32 => DataType::Date32,
            C::Timestamp { unit: u, zone } => {
                DataType::Timestamp(unit(u), zone.as_deref().map(Arc::from))
            }
            C::Duration { unit: u } => DataType::Duration(unit(u)),
            C::Time64 { unit: u } => DataType::Time64(unit(u)),
            C::List { item } => DataType::List(Arc::new(item.to_arrow())),
            C::Struct { fields } => {
                DataType::Struct(fields.iter().map(Field::to_arrow).collect::<Fields>())
            }
            C::Map {
                entries,
                key,
                value,
                sorted,
            } => {
                let children = Fields::from(vec![key.to_arrow(), value.to_arrow()]);
                DataType::Map(
                    Arc::new(ArrowField::new(entries, DataType::Struct(children), false)),
                    *sorted,
                )
            }
            C::DictionaryUtf8 { index } => {
                let index = match index {
                    IndexType::Int8 => DataType::Int8,
                    IndexType::Int16 => DataType::Int16,
                    IndexType::Int32 => DataType::Int32,
                    IndexType::Int64 => DataType::Int64,
                    IndexType::Uint8 => DataType::UInt8,
                    IndexType::Uint16 => DataType::UInt16,
                    IndexType::Uint32 => DataType::UInt32,
                    IndexType::Uint64 => DataType::UInt64,
                };
                DataType::Dictionary(Box::new(index), Box::new(DataType::Utf8))
            }
        }
    }
}

impl Field {
    /// The closed-set field of the Arrow field `field`, a top-level column.
    pub fn from_arrow(field: &ArrowField) -> Result<Self, CoreError> {
        Self::from_arrow_at(field.name(), field)
    }

    /// A child of the column at `parent`.
    fn from_arrow_in(parent: &str, field: &ArrowField) -> Result<Self, CoreError> {
        Self::from_arrow_at(&format!("{parent}.{}", field.name()), field)
    }

    fn from_arrow_at(path: &str, field: &ArrowField) -> Result<Self, CoreError> {
        Ok(Self {
            name: field.name().clone(),
            column_type: ColumnType::from_arrow(path, field.data_type())?,
            nullable: field.is_nullable(),
            metadata: field
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        })
    }

    /// The Arrow field this is.
    pub fn to_arrow(&self) -> ArrowField {
        ArrowField::new(&self.name, self.column_type.to_arrow(), self.nullable).with_metadata(
            self.metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }
}

/// Reads a [`ColumnType`], refusing fields a unit variant does not have,
/// which serde's `deny_unknown_fields` lets through for internally tagged
/// unit variants.
pub(crate) fn strict_column_type<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ColumnType, D::Error> {
    use serde::de::Error as _;
    use serde_json::{Map, Value};
    let map = Map::<String, Value>::deserialize(deserializer)?;
    if map.len() > 1
        && let Some(tag) = map.get("type")
    {
        let bare = Value::Object(Map::from_iter([("type".to_owned(), tag.clone())]));
        if serde_json::from_value::<ColumnType>(bare).is_ok() {
            return Err(D::Error::custom(format!(
                "column type {tag} takes no fields besides its type"
            )));
        }
    }
    serde_json::from_value(Value::Object(map)).map_err(D::Error::custom)
}

/// The refusal of an Arrow type outside the set, naming the column and the
/// type as Arrow writes it.
pub fn unsupported_type(column: &str, data_type: &DataType) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        codes::UNSUPPORTED_TYPE,
        format!("column {column:?} has the Arrow type {data_type}, which the core does not take"),
    )
    .with_field("column", Detail::text(column))
    .with_field("arrow_type", Detail::text(data_type.to_string()))
}
