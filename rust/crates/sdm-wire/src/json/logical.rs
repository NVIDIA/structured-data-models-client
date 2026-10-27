// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capture the existing typed payload's ordered values before JSON encoding.
//! This is serde data conversion, with no model schema, normalization,
//! callback, row selection or JSON-body reparsing.

use std::fmt;

use serde::Serialize;
use serde::ser::{self, SerializeMap, SerializeSeq, SerializeStruct};

use super::Value;

pub(super) const NUMBER: &str = "sdm.logical.number";

/// A typed value that cannot be represented by the existing JSON-like data.
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

pub(super) fn capture<T: Serialize + ?Sized>(value: &T) -> Result<Value, Error> {
    value.serialize(Capture)
}

struct Capture;

impl ser::Serializer for Capture {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Sequence;
    type SerializeTuple = Sequence;
    type SerializeTupleStruct = Sequence;
    type SerializeTupleVariant = VariantSequence;
    type SerializeMap = Object;
    type SerializeStruct = Object;
    type SerializeStructVariant = VariantObject;

    fn is_human_readable(&self) -> bool {
        false
    }
    fn serialize_bool(self, value: bool) -> Result<Value, Error> {
        Ok(Value::Bool(value))
    }
    fn serialize_i8(self, value: i8) -> Result<Value, Error> {
        self.serialize_i64(value.into())
    }
    fn serialize_i16(self, value: i16) -> Result<Value, Error> {
        self.serialize_i64(value.into())
    }
    fn serialize_i32(self, value: i32) -> Result<Value, Error> {
        self.serialize_i64(value.into())
    }
    fn serialize_i64(self, value: i64) -> Result<Value, Error> {
        Ok(Value::Number(value.to_string()))
    }
    fn serialize_i128(self, value: i128) -> Result<Value, Error> {
        Ok(Value::Number(value.to_string()))
    }
    fn serialize_u8(self, value: u8) -> Result<Value, Error> {
        self.serialize_u64(value.into())
    }
    fn serialize_u16(self, value: u16) -> Result<Value, Error> {
        self.serialize_u64(value.into())
    }
    fn serialize_u32(self, value: u32) -> Result<Value, Error> {
        self.serialize_u64(value.into())
    }
    fn serialize_u64(self, value: u64) -> Result<Value, Error> {
        Ok(Value::Number(value.to_string()))
    }
    fn serialize_u128(self, value: u128) -> Result<Value, Error> {
        Ok(Value::Number(value.to_string()))
    }
    fn serialize_f32(self, value: f32) -> Result<Value, Error> {
        Ok(if value.is_finite() {
            Value::Number(serde_json::to_string(&value).map_err(<Error as ser::Error>::custom)?)
        } else {
            Value::Null
        })
    }
    fn serialize_f64(self, value: f64) -> Result<Value, Error> {
        Ok(if value.is_finite() {
            Value::Number(serde_json::to_string(&value).map_err(<Error as ser::Error>::custom)?)
        } else {
            Value::Null
        })
    }
    fn serialize_char(self, value: char) -> Result<Value, Error> {
        self.serialize_str(&value.to_string())
    }
    fn serialize_str(self, value: &str) -> Result<Value, Error> {
        Ok(Value::text(value))
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<Value, Error> {
        Ok(Value::Array(
            value
                .iter()
                .map(|value| Value::Number(value.to_string()))
                .collect(),
        ))
    }
    fn serialize_none(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Value, Error> {
        capture(value)
    }
    fn serialize_unit(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value, Error> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        let value = capture(value)?;
        if name == NUMBER {
            let Value::Text(token) = value else {
                return Err(ser::Error::custom(
                    "a logical number must retain its written token",
                ));
            };
            serde_json::value::RawValue::from_string(token.clone())
                .map_err(<Error as ser::Error>::custom)?;
            Ok(Value::Number(token))
        } else {
            Ok(value)
        }
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        Ok(Value::Object(vec![(variant.to_owned(), capture(value)?)]))
    }
    fn serialize_seq(self, length: Option<usize>) -> Result<Sequence, Error> {
        Ok(Sequence(Vec::with_capacity(length.unwrap_or(0))))
    }
    fn serialize_tuple(self, length: usize) -> Result<Sequence, Error> {
        self.serialize_seq(Some(length))
    }
    fn serialize_tuple_struct(self, _: &'static str, length: usize) -> Result<Sequence, Error> {
        self.serialize_seq(Some(length))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
        length: usize,
    ) -> Result<VariantSequence, Error> {
        Ok(VariantSequence {
            name,
            values: Sequence(Vec::with_capacity(length)),
        })
    }
    fn serialize_map(self, length: Option<usize>) -> Result<Object, Error> {
        Ok(Object {
            members: Vec::with_capacity(length.unwrap_or(0)),
            key: None,
        })
    }
    fn serialize_struct(self, _: &'static str, length: usize) -> Result<Object, Error> {
        self.serialize_map(Some(length))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
        length: usize,
    ) -> Result<VariantObject, Error> {
        Ok(VariantObject {
            name,
            values: self.serialize_struct(name, length)?,
        })
    }
}

struct Sequence(Vec<Value>);

impl SerializeSeq for Sequence {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.0.push(capture(value)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        Ok(Value::Array(self.0))
    }
}

impl ser::SerializeTuple for Sequence {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<Value, Error> {
        SerializeSeq::end(self)
    }
}

impl ser::SerializeTupleStruct for Sequence {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<Value, Error> {
        SerializeSeq::end(self)
    }
}

struct VariantSequence {
    name: &'static str,
    values: Sequence,
}

impl ser::SerializeTupleVariant for VariantSequence {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.values.serialize_element(value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(Value::Object(vec![(
            self.name.to_owned(),
            SerializeSeq::end(self.values)?,
        )]))
    }
}

struct Object {
    members: Vec<(String, Value)>,
    key: Option<String>,
}

impl SerializeMap for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        if self.key.is_some() {
            return Err(ser::Error::custom(
                "a logical object key lacks its previous value",
            ));
        }
        let Value::Text(key) = capture(key)? else {
            return Err(ser::Error::custom(
                "a typed payload object key must be text",
            ));
        };
        self.key = Some(key);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .key
            .take()
            .ok_or_else(|| <Error as ser::Error>::custom("a logical object value lacks its key"))?;
        self.members.push((key, capture(value)?));
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        if self.key.is_some() {
            return Err(ser::Error::custom(
                "a logical object's final key lacks its value",
            ));
        }
        Ok(Value::Object(self.members))
    }
}

impl SerializeStruct for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.members.push((key.to_owned(), capture(value)?));
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        SerializeMap::end(self)
    }
}

struct VariantObject {
    name: &'static str,
    values: Object,
}

impl ser::SerializeStructVariant for VariantObject {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.values.serialize_field(key, value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(Value::Object(vec![(
            self.name.to_owned(),
            SerializeStruct::end(self.values)?,
        )]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Payload {
        model: &'static str,
        #[serde(flatten)]
        values: Values,
        #[serde(skip_serializing_if = "Option::is_none")]
        absent: Option<bool>,
    }

    #[derive(Serialize)]
    struct Values {
        columns: Vec<&'static str>,
        rows: Vec<Vec<Value>>,
    }

    #[test]
    fn typed_values_preserve_field_order_omission_and_exact_numeric_tokens() {
        let payload = Payload {
            model: "kumo-relational",
            values: Values {
                columns: vec!["id", "note"],
                rows: vec![vec![
                    Value::Number("18446744073709551616000001".into()),
                    Value::text("ΟΣ 😺\0"),
                    Value::Number("-0".into()),
                    Value::Number("1.00000000000000001e-40".into()),
                ]],
            },
            absent: None,
        };
        let logical = capture(&payload).unwrap();
        assert_eq!(logical.to_bytes(), serde_json::to_vec(&payload).unwrap());
        assert_eq!(
            logical.get("rows").unwrap().to_bytes(),
            r#"[[18446744073709551616000001,"ΟΣ 😺\u0000",-0,1.00000000000000001e-40]]"#.as_bytes()
        );
        assert!(logical.get("absent").is_none());
        let Value::Object(members) = logical else {
            panic!("typed payload object")
        };
        assert_eq!(
            members
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["model", "columns", "rows"]
        );
    }

    #[test]
    fn typed_enum_shapes_use_the_existing_serde_layout() {
        #[derive(Serialize)]
        enum Shape {
            Unit,
            Tuple(i64, bool),
            Named { value: String },
        }
        for shape in [
            Shape::Unit,
            Shape::Tuple(7, true),
            Shape::Named { value: "é".into() },
        ] {
            assert_eq!(
                capture(&shape).unwrap().to_bytes(),
                serde_json::to_vec(&shape).unwrap()
            );
        }
    }
}
