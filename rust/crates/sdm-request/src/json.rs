// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON whose objects keep the order their members were written in, as the
//! reference's payload dictionaries do, and whose numbers are exact: an
//! integer as its digits, a float as the shortest text that reads back as
//! the same double.

use sdm_contract::{CoreError, ErrorCode, ErrorKind};

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    /// A finite double.
    Float(f64),
    Text(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An object of `members`, in order.
    pub fn object<const N: usize>(members: [(&str, Json); N]) -> Json {
        Json::Object(
            members
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        )
    }

    pub fn text(value: &str) -> Json {
        Json::Text(value.to_owned())
    }

    pub fn texts<'a>(values: impl IntoIterator<Item = &'a str>) -> Json {
        Json::Array(values.into_iter().map(Json::text).collect())
    }

    /// A member of an object, by name.
    pub fn get(&self, name: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(k, _)| k == name).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The member `name` of an object, set: in place if it is there,
    /// appended if not, as a dictionary's assignment does.
    pub fn set(&mut self, name: &str, value: Json) {
        if let Json::Object(members) = self {
            match members.iter_mut().find(|(k, _)| k == name) {
                Some(slot) => slot.1 = value,
                None => members.push((name.to_owned(), value)),
            }
        }
    }

    /// The value's UTF-8 JSON text.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CoreError> {
        let mut out = Vec::new();
        self.write(&mut out)?;
        Ok(out)
    }

    fn write(&self, out: &mut Vec<u8>) -> Result<(), CoreError> {
        match self {
            Json::Null => out.extend_from_slice(b"null"),
            Json::Bool(true) => out.extend_from_slice(b"true"),
            Json::Bool(false) => out.extend_from_slice(b"false"),
            Json::Int(value) => out.extend_from_slice(value.to_string().as_bytes()),
            Json::Float(value) => {
                if !value.is_finite() {
                    return Err(CoreError::new(
                        ErrorKind::Internal,
                        ErrorCode::InternalError,
                        "a payload number is not finite",
                    ));
                }
                let text = serde_json::to_string(value).map_err(|_| {
                    CoreError::new(
                        ErrorKind::Internal,
                        ErrorCode::InternalError,
                        "a payload number does not serialize",
                    )
                })?;
                out.extend_from_slice(text.as_bytes());
            }
            Json::Text(value) => {
                let text = serde_json::to_string(value).map_err(|_| {
                    CoreError::new(
                        ErrorKind::Internal,
                        ErrorCode::InternalError,
                        "a payload text does not serialize",
                    )
                })?;
                out.extend_from_slice(text.as_bytes());
            }
            Json::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    item.write(out)?;
                }
                out.push(b']');
            }
            Json::Object(members) => {
                out.push(b'{');
                for (i, (name, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    Json::Text(name.clone()).write(out)?;
                    out.push(b':');
                    value.write(out)?;
                }
                out.push(b'}');
            }
        }
        Ok(())
    }
}
