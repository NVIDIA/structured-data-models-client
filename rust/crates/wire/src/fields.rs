// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the generated types deserialize their fields through.
//!
//! An optional field is an `Option` that serde fills with `None` when the key
//! is absent, and skips when serializing. A key that is present goes through
//! [`present`], so a null is refused unless the field's own type takes it.

use serde::de::{DeserializeOwned, Error};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// A key that is present: its value, which `T` must accept.
pub(crate) fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// A key that must be present and may be null.
#[allow(
    dead_code,
    reason = "the generator uses it for a required nullable field, which the contract has none of yet"
)]
pub(crate) fn nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// An object of a discriminated union, and the value of its `tag` key.
pub(crate) fn tagged<'de, D>(
    deserializer: D,
    tag: &'static str,
) -> Result<(String, Value), D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    let found = match value.get(tag) {
        Some(Value::String(found)) => found.clone(),
        Some(other) => {
            return Err(D::Error::custom(format!(
                "`{tag}` must be a string, not {other}"
            )));
        }
        None if value.is_object() => return Err(D::Error::missing_field(tag)),
        None => return Err(D::Error::custom(format!("expected an object, not {value}"))),
    };
    Ok((found, value))
}

/// The variant of a discriminated union `value` is.
pub(crate) fn variant<T, E>(value: Value) -> Result<T, E>
where
    T: DeserializeOwned,
    E: Error,
{
    T::deserialize(value).map_err(E::custom)
}

/// A discriminated union's tag that selects none of its variants.
pub(crate) fn unknown_tag<E: Error>(
    tag: &str,
    found: &str,
    expected: &'static [&'static str],
) -> E {
    E::custom(format!(
        "unknown `{tag}` {found:?}, expected one of {expected:?}"
    ))
}
