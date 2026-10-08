// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python's conversions of a scalar: `int(value)`, `float(value)`,
//! `bool(value)`, numpy's `.item()`, and the abstract number types the
//! wire's rules ask about (`numbers.Integral`, `numbers.Real`).

use sdm_protocol::Diagnostic;
use sdm_schema::pyval::PyVal;

type ProtocolCall =
    dyn Fn(&str, &serde_json::Value) -> crate::error::R<serde_json::Value> + Send + Sync;

/// Call-local language protocol facts, requested only when an operation reaches them.
#[derive(Clone)]
pub struct HostProtocols(std::sync::Arc<ProtocolCall>);

impl std::fmt::Debug for HostProtocols {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HostProtocols")
    }
}

impl HostProtocols {
    pub fn new(
        call: impl Fn(&str, &serde_json::Value) -> crate::error::R<serde_json::Value>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self(std::sync::Arc::new(call))
    }

    pub fn call(
        &self,
        operation: &str,
        operands: &serde_json::Value,
    ) -> crate::error::R<serde_json::Value> {
        (self.0)(operation, operands)
    }
}

pub fn has_host_value(value: &PyVal) -> bool {
    match value {
        PyVal::Held { .. } => true,
        PyVal::Opaque { kind, .. } => kind == "host_ref",
        PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
            items.iter().any(has_host_value)
        }
        PyVal::Dict(pairs) => pairs
            .iter()
            .any(|(key, value)| has_host_value(key) || has_host_value(value)),
        _ => false,
    }
}

fn host_fact(
    protocols: Option<&HostProtocols>,
    operation: &str,
    value: &PyVal,
) -> crate::error::R<serde_json::Value> {
    protocols
        .ok_or_else(|| {
            crate::error::value_error(format!("{operation} requires host protocol facts"))
        })?
        .call(operation, &crate::layout::encode_deep(value))
}

pub fn truth_with(value: &PyVal, protocols: Option<&HostProtocols>) -> crate::error::R<bool> {
    if has_host_value(value) {
        return host_fact(protocols, "truth", value)?
            .as_bool()
            .ok_or_else(|| crate::error::value_error("a truth host fact must be boolean"));
    }
    Ok(py_bool(value))
}

pub fn hashable_with(value: &PyVal, protocols: Option<&HostProtocols>) -> crate::error::R<()> {
    if has_host_value(value) {
        host_fact(protocols, "hashable", value)?;
        return Ok(());
    }
    if let PyVal::Tuple(items) = value {
        for item in items {
            hashable_with(item, protocols)?;
        }
        return Ok(());
    }
    require_hashable(value)
}

pub fn repr_with(value: &PyVal, protocols: Option<&HostProtocols>) -> crate::error::R<String> {
    if has_host_value(value) {
        return host_fact(protocols, "repr", value)?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| crate::error::value_error("a repr host fact must be text"));
    }
    Ok(value.repr())
}

pub fn string(value: &serde_json::Value) -> Option<&str> {
    if let Some(held) = value
        .as_object()
        .filter(|mapping| mapping.len() == 1)
        .and_then(|mapping| mapping.get("$held"))
    {
        return string(&held["value"]);
    }
    value.as_str()
}

pub fn require_hashable(value: &PyVal) -> crate::error::R<()> {
    match value.semantic() {
        PyVal::Tuple(items) => {
            for item in items {
                require_hashable(item)?;
            }
            Ok(())
        }
        PyVal::List(_) | PyVal::Dict(_) | PyVal::NdArray(..) => Err(crate::error::builtin(
            "TypeError",
            format!("unhashable type: '{}'", crate::models::type_name(value)),
        )),
        _ => Ok(()),
    }
}

use crate::error::builtin;
use crate::num::{Dec, Special};

pub trait TaggedValue {
    fn py_get(&self, key: &str) -> Option<&serde_json::Value>;
}

pub fn mapping_get(
    value: &serde_json::Value,
    key: &str,
    default: &serde_json::Value,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<serde_json::Value> {
    let decoded = PyVal::decode(value).map_err(crate::error::value_error)?;
    if let PyVal::Held { owner, .. } = decoded {
        let operands = if default.is_null() {
            serde_json::json!([value, key])
        } else {
            serde_json::json!([value, key, default])
        };
        let result = protocols
            .ok_or_else(|| crate::error::value_error("mapping get requires host protocol facts"))?
            .call("mapping_get", &operands)?;
        validate_result_owner(
            &PyVal::decode(&result).map_err(crate::error::value_error)?,
            &owner,
        )?;
        return Ok(result);
    }
    if !matches!(decoded, PyVal::Dict(_)) {
        return Err(builtin(
            "AttributeError",
            format!(
                "'{}' object has no attribute 'get'",
                crate::models::type_name(&decoded)
            ),
        ));
    }
    Ok(value.py_get(key).unwrap_or(default).clone())
}

pub fn mapping_copy(
    value: &serde_json::Value,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<serde_json::Value> {
    let source = PyVal::decode(value).map_err(crate::error::value_error)?;
    let PyVal::Held { owner, .. } = source else {
        return Ok(value.clone());
    };
    let copied = protocols
        .ok_or_else(|| crate::error::value_error("mapping copy requires host protocol facts"))?
        .call("mapping_copy", value)?;
    let decoded = PyVal::decode(&copied).map_err(crate::error::value_error)?;
    if !matches!(decoded, PyVal::Held { owner: ref copied_owner, type_name: Some(ref name), ref value, .. }
        if copied_owner == &owner && name == "dict" && matches!(value.as_ref(), PyVal::Dict(_)))
    {
        return Err(crate::error::value_error(
            "a mapping copy fact must retain its scope and an exact dictionary view",
        ));
    }
    validate_result_owner(&decoded, &owner)?;
    Ok(copied)
}

pub fn getitem(
    value: &serde_json::Value,
    key: &PyVal,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<serde_json::Value> {
    let decoded = PyVal::decode(value).map_err(crate::error::value_error)?;
    if let PyVal::Held { owner, .. } = decoded {
        let result = protocols
            .ok_or_else(|| crate::error::value_error("item lookup requires host protocol facts"))?
            .call(
                "getitem",
                &serde_json::json!([value, crate::layout::encode_deep(key)]),
            )?;
        validate_result_owner(
            &PyVal::decode(&result).map_err(crate::error::value_error)?,
            &owner,
        )?;
        return Ok(result);
    }
    let item = match (&decoded, key) {
        (PyVal::Dict(pairs), _) => pairs
            .iter()
            .find(|(candidate, _)| crate::result::py_eq(candidate, key))
            .map(|(_, value)| value),
        (PyVal::List(items) | PyVal::Tuple(items), PyVal::Int(index)) => usize::try_from(*index)
            .ok()
            .and_then(|index| items.get(index)),
        _ => None,
    };
    item.map(crate::layout::encode_deep)
        .ok_or_else(|| crate::error::value_error("item lookup requires an existing key or index"))
}

#[derive(Clone, Copy)]
pub enum InstanceFamily {
    Dict,
    Mapping,
    ListTuple,
}

fn validate_result_owner(value: &PyVal, owner: &Option<String>) -> crate::error::R<()> {
    match value {
        PyVal::Held {
            owner: result_owner,
            value,
            ..
        } => {
            if result_owner != owner {
                return Err(crate::error::value_error(
                    "a host lookup result must retain its operation scope",
                ));
            }
            validate_result_owner(value, owner)?;
        }
        PyVal::Dict(pairs) => {
            for (key, value) in pairs {
                validate_result_owner(key, owner)?;
                validate_result_owner(value, owner)?;
            }
        }
        PyVal::List(items) | PyVal::Tuple(items) | PyVal::NdArray(_, items) => {
            for value in items {
                validate_result_owner(value, owner)?;
            }
        }
        PyVal::Numpy(_, value) => validate_result_owner(value, owner)?,
        _ => {}
    }
    Ok(())
}

pub fn is_instance(
    value: &serde_json::Value,
    family: InstanceFamily,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<bool> {
    let decoded = PyVal::decode(value).map_err(crate::error::value_error)?;
    if matches!(decoded, PyVal::Held { .. }) {
        let name = match family {
            InstanceFamily::Dict => "dict",
            InstanceFamily::Mapping => "mapping",
            InstanceFamily::ListTuple => "list_tuple",
        };
        return protocols
            .ok_or_else(|| {
                crate::error::value_error("instance check requires host protocol facts")
            })?
            .call("is_instance", &serde_json::json!([value, name]))?
            .as_bool()
            .ok_or_else(|| crate::error::value_error("an instance check fact must be boolean"));
    }
    Ok(match family {
        InstanceFamily::Dict | InstanceFamily::Mapping => matches!(decoded, PyVal::Dict(_)),
        InstanceFamily::ListTuple => matches!(decoded, PyVal::List(_) | PyVal::Tuple(_)),
    })
}

pub fn mapping_update(
    mapping: &mut serde_json::Value,
    key: &str,
    value: Option<serde_json::Value>,
    protocols: Option<&HostProtocols>,
) -> crate::error::R<()> {
    let decoded = PyVal::decode(mapping).map_err(crate::error::value_error)?;
    if let PyVal::Held { owner, .. } = decoded {
        let operation = if value.is_some() {
            "mapping_set"
        } else {
            "mapping_pop"
        };
        let updated = protocols
            .ok_or_else(|| {
                crate::error::value_error("mapping mutation requires host protocol facts")
            })?
            .call(operation, &serde_json::json!([mapping, key, value]))?;
        let decoded = PyVal::decode(&updated).map_err(crate::error::value_error)?;
        if !matches!(decoded, PyVal::Held { owner: ref updated_owner, type_name: Some(ref name), ref value, .. }
            if updated_owner == &owner && name == "dict" && matches!(value.as_ref(), PyVal::Dict(_)))
        {
            return Err(crate::error::value_error(
                "a mapping mutation fact must retain its scope and an exact dictionary view",
            ));
        }
        validate_result_owner(&decoded, &owner)?;
        *mapping = updated;
    } else {
        set_field(mapping, key, value);
    }
    Ok(())
}

pub fn sequence(value: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    value.as_array().or_else(|| {
        value
            .as_object()
            .filter(|mapping| mapping.len() == 1)?
            .get("$tuple")?
            .as_array()
    })
}

impl TaggedValue for serde_json::Value {
    fn py_get(&self, key: &str) -> Option<&serde_json::Value> {
        let mapping = self.as_object()?;
        if mapping.len() == 1 {
            if let Some(held) = mapping.get("$held") {
                return held.get("value")?.py_get(key);
            }
            if let Some(pairs) = mapping.get("$map").and_then(serde_json::Value::as_array) {
                return pairs.iter().find_map(|pair| {
                    (pair.get(0).and_then(string) == Some(key))
                        .then(|| pair.get(1))
                        .flatten()
                });
            }
            if let Some(escaped) = mapping.get("$dict") {
                return escaped.get(key);
            }
        }
        mapping.get(key)
    }
}

impl TaggedValue for serde_json::Map<String, serde_json::Value> {
    fn py_get(&self, key: &str) -> Option<&serde_json::Value> {
        self.get(key)
    }
}

pub fn set_field(value: &mut serde_json::Value, key: &str, field: Option<serde_json::Value>) {
    use serde_json::{Value, json};
    if let Some(held) = value
        .as_object()
        .filter(|mapping| mapping.len() == 1)
        .and_then(|mapping| mapping.get("$held"))
    {
        *value = held["value"].clone();
    }
    if value.as_object().is_some_and(|mapping| mapping.len() == 1) {
        if let Some(pairs) = value.get_mut("$map").and_then(Value::as_array_mut) {
            let position = pairs
                .iter()
                .position(|pair| pair.get(0).and_then(string) == Some(key));
            match (position, field) {
                (Some(position), Some(field)) => pairs[position][1] = field,
                (Some(position), None) => {
                    pairs.remove(position);
                }
                (None, Some(field)) => pairs.push(json!([key, field])),
                (None, None) => {}
            }
            return;
        }
        if let Some(escaped) = value.get_mut("$dict") {
            if let Some(mapping) = escaped.as_object_mut() {
                if let Some(field) = field {
                    mapping.insert(key.into(), field);
                } else {
                    mapping.remove(key);
                }
            }
            return;
        }
    }
    if let Some(mapping) = value.as_object_mut() {
        if let Some(field) = field {
            mapping.insert(key.into(), field);
        } else {
            mapping.remove(key);
        }
    }
}

pub fn field_mut<'value>(
    value: &'value mut serde_json::Value,
    key: &str,
) -> Option<&'value mut serde_json::Value> {
    use serde_json::Value;
    if let Some(held) = value
        .as_object()
        .filter(|mapping| mapping.len() == 1)
        .and_then(|mapping| mapping.get("$held"))
    {
        *value = held["value"].clone();
    }
    if value.as_object().is_some_and(|mapping| mapping.len() == 1) {
        if value.get("$map").and_then(Value::as_array).is_some() {
            return value
                .get_mut("$map")?
                .as_array_mut()?
                .iter_mut()
                .find_map(|pair| {
                    if pair.get(0).and_then(string) == Some(key) {
                        pair.get_mut(1)
                    } else {
                        None
                    }
                });
        }
        if value.get("$dict").is_some() {
            return value.get_mut("$dict")?.get_mut(key);
        }
    }
    value.get_mut(key)
}

pub fn mapping_items(value: &serde_json::Value) -> Option<Vec<(PyVal, &serde_json::Value)>> {
    use serde_json::Value;
    let mapping = value.as_object()?;
    if mapping.len() == 1 {
        if let Some(held) = mapping.get("$held") {
            return mapping_items(held.get("value")?);
        }
        if let Some(pairs) = mapping.get("$map").and_then(Value::as_array) {
            return pairs
                .iter()
                .map(|pair| Some((PyVal::decode(pair.get(0)?).ok()?, pair.get(1)?)))
                .collect();
        }
        if let Some(escaped) = mapping.get("$dict") {
            return Some(
                escaped
                    .as_object()?
                    .iter()
                    .map(|(key, value)| (PyVal::Str(key.clone()), value))
                    .collect(),
            );
        }
    }
    Some(
        mapping
            .iter()
            .map(|(key, value)| (PyVal::Str(key.clone()), value))
            .collect(),
    )
}

pub fn mapping_values_mut(value: &mut serde_json::Value) -> Vec<&mut serde_json::Value> {
    use serde_json::Value;
    if let Some(held) = value
        .as_object()
        .filter(|mapping| mapping.len() == 1)
        .and_then(|mapping| mapping.get("$held"))
    {
        *value = held["value"].clone();
    }
    if value.as_object().is_some_and(|mapping| mapping.len() == 1) {
        if value.get("$map").and_then(Value::as_array).is_some() {
            return value
                .get_mut("$map")
                .and_then(Value::as_array_mut)
                .into_iter()
                .flatten()
                .filter_map(|pair| pair.get_mut(1))
                .collect();
        }
        if value.get("$dict").is_some() {
            return value
                .get_mut("$dict")
                .and_then(Value::as_object_mut)
                .into_iter()
                .flat_map(|mapping| mapping.values_mut())
                .collect();
        }
    }
    value
        .as_object_mut()
        .into_iter()
        .flat_map(|mapping| mapping.values_mut())
        .collect()
}

/// `value.item()` of a numpy scalar; anything else as it is. A
/// `datetime64` gives a `datetime` at a unit of microseconds or coarser,
/// and its integer nanoseconds at `ns`.
pub fn item(value: &PyVal) -> PyVal {
    match value {
        PyVal::Numpy(dtype, inner) if dtype.starts_with("datetime64") => {
            let stamp = match inner.as_ref() {
                PyVal::Str(text) => {
                    PyVal::decode(&serde_json::json!({ "$timestamp": text })).unwrap_or(PyVal::NaT)
                }
                other => other.clone(),
            };
            match (dtype.as_str(), stamp) {
                (_, PyVal::NaT) => PyVal::None,
                ("datetime64[ns]", PyVal::Timestamp(wall)) => PyVal::Int(wall.epoch_nanos()),
                (_, PyVal::Timestamp(wall)) => PyVal::DateTime(wall, None),
                (_, other) => other,
            }
        }
        PyVal::Numpy(_, inner) => (**inner).clone(),
        other => other.clone(),
    }
}

/// `isinstance(value, numbers.Integral)`: `bool`, `int`, numpy integers.
pub fn is_integral_type(value: &PyVal) -> bool {
    match value {
        PyVal::Bool(_) | PyVal::Int(_) | PyVal::BigInt(_) => true,
        PyVal::Numpy(name, _) => {
            name.starts_with("int") || name.starts_with("uint") || name == "bool"
        }
        _ => false,
    }
}

/// `isinstance(value, numbers.Real)`: integers, floats and numpy numbers,
/// but not `Decimal`.
pub fn is_real_type(value: &PyVal) -> bool {
    match value {
        PyVal::Float(_) => true,
        PyVal::Numpy(name, _) => {
            name.starts_with("float") || name == "longdouble" || is_integral_type(value)
        }
        other => is_integral_type(other),
    }
}

/// The exact value of an integer, a `bool` as 0 or 1.
pub fn integer_value(value: &PyVal) -> Option<Dec> {
    match value.semantic() {
        PyVal::Bool(b) => Some(Dec::from_i128(i128::from(*b))),
        PyVal::Int(n) => Some(Dec::from_i128(*n)),
        PyVal::BigInt(digits) => Dec::integer(digits),
        PyVal::Numpy(_, inner) => integer_value(inner),
        _ => None,
    }
}

/// A `Decimal`'s value.
pub fn decimal_value(value: &PyVal) -> Option<Dec> {
    match value.semantic() {
        PyVal::Decimal(text) => Dec::parse(text),
        _ => None,
    }
}

/// Python's `int(text)` of a string, or `None` where it raises.
pub fn parse_int(text: &str) -> Option<Dec> {
    let t = text.trim();
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => ("-", &t[1..]),
        Some(b'+') => ("", &t[1..]),
        _ => ("", t),
    };
    if body.is_empty()
        || body.starts_with('_')
        || body.ends_with('_')
        || body.contains("__")
        || !body.bytes().all(|b| b.is_ascii_digit() || b == b'_')
    {
        return None;
    }
    Dec::integer(&format!("{sign}{}", body.replace('_', "")))
}

/// Python's `float(text)` of a string, or `None` where it raises.
pub fn parse_float(text: &str) -> Option<f64> {
    let t = text.trim();
    let lower = t.to_ascii_lowercase();
    let (sign, body) = match lower.as_bytes().first() {
        Some(b'-') => (-1.0, &lower[1..]),
        Some(b'+') => (1.0, &lower[1..]),
        _ => (1.0, lower.as_str()),
    };
    match body {
        "inf" | "infinity" => return Some(sign * f64::INFINITY),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    if body.is_empty() || body.contains("__") || body.starts_with('_') || body.ends_with('_') {
        return None;
    }
    let cleaned = body.replace('_', "");
    if !cleaned
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-'))
    {
        return None;
    }
    let dec = Dec::parse(&cleaned)?;
    if !dec.is_finite() {
        return None;
    }
    Some(sign * dec.to_f64())
}

fn type_name(value: &PyVal) -> String {
    match value {
        PyVal::None => "NoneType".to_string(),
        PyVal::Bool(_) => "bool".to_string(),
        PyVal::Int(_) | PyVal::BigInt(_) => "int".to_string(),
        PyVal::Float(_) => "float".to_string(),
        PyVal::Str(_) => "str".to_string(),
        PyVal::Decimal(_) => "decimal.Decimal".to_string(),
        PyVal::Bytes(_) => "bytes".to_string(),
        PyVal::List(_) => "list".to_string(),
        PyVal::Tuple(_) => "tuple".to_string(),
        PyVal::Dict(_) => "dict".to_string(),
        PyVal::Timestamp(_) | PyVal::TimestampTz(..) => "Timestamp".to_string(),
        PyVal::NaT => "NaTType".to_string(),
        PyVal::Na => "NAType".to_string(),
        PyVal::Numpy(name, _) => format!("numpy.{name}"),
        PyVal::Opaque { type_name, .. } => type_name.clone(),
        _ => "object".to_string(),
    }
}

/// `int(value)`: the value truncated toward zero, or what Python raises.
pub fn py_int(value: &PyVal) -> Result<Dec, Diagnostic> {
    match value {
        PyVal::Numpy(_, inner) => py_int(inner),
        PyVal::Bool(_) | PyVal::Int(_) | PyVal::BigInt(_) => {
            integer_value(value).ok_or_else(|| builtin("ValueError", "an integer"))
        }
        PyVal::Float(x) => {
            if x.is_nan() {
                return Err(builtin("ValueError", "cannot convert float NaN to integer"));
            }
            if x.is_infinite() {
                return Err(builtin(
                    "OverflowError",
                    "cannot convert float infinity to integer",
                ));
            }
            Dec::of_whole_f64(x.trunc()).ok_or_else(|| builtin("ValueError", "a float"))
        }
        PyVal::Decimal(text) => {
            let dec = Dec::parse(text).ok_or_else(|| builtin("ValueError", "a decimal"))?;
            match dec.special {
                Special::Finite => Ok(dec.trunc()),
                Special::Infinity => Err(builtin(
                    "OverflowError",
                    "cannot convert Infinity to integer",
                )),
                _ => Err(builtin("ValueError", "cannot convert NaN to integer")),
            }
        }
        PyVal::Str(text) => parse_int(text).ok_or_else(|| {
            builtin(
                "ValueError",
                format!(
                    "invalid literal for int() with base 10: {}",
                    sdm_pql::pyrepr::str_repr(text)
                ),
            )
        }),
        PyVal::Timedelta(_) | PyVal::Timestamp(_) | PyVal::TimestampTz(..) => Err(builtin(
            "TypeError",
            format!(
                "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                type_name(value)
            ),
        )),
        other => Err(builtin(
            "TypeError",
            format!(
                "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                type_name(other)
            ),
        )),
    }
}

/// `float(value)`, or what Python raises: `OverflowError` for an integer
/// past the largest double, `ValueError` for text that is not a number.
pub fn py_float(value: &PyVal) -> Result<f64, Diagnostic> {
    match value {
        PyVal::Numpy(_, inner) => py_float(inner),
        PyVal::Float(x) => Ok(*x),
        PyVal::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        PyVal::Int(_) | PyVal::BigInt(_) => {
            let exact = integer_value(value).ok_or_else(|| builtin("ValueError", "an integer"))?;
            let x = exact.to_f64();
            if x.is_infinite() {
                return Err(builtin(
                    "OverflowError",
                    "int too large to convert to float",
                ));
            }
            Ok(x)
        }
        PyVal::Decimal(text) => Dec::parse(text)
            .map(|d| d.to_f64())
            .ok_or_else(|| builtin("ValueError", "a decimal")),
        PyVal::Str(text) => parse_float(text).ok_or_else(|| {
            builtin(
                "ValueError",
                format!(
                    "could not convert string to float: {}",
                    sdm_pql::pyrepr::str_repr(text)
                ),
            )
        }),
        other => Err(builtin(
            "TypeError",
            format!(
                "float() argument must be a string or a real number, not '{}'",
                type_name(other)
            ),
        )),
    }
}

/// `bool(value)`.
pub fn py_bool(value: &PyVal) -> bool {
    match value.semantic() {
        PyVal::None | PyVal::NaT => false,
        PyVal::Bool(b) => *b,
        PyVal::Int(n) => *n != 0,
        PyVal::BigInt(d) => d != "0",
        PyVal::Float(x) => *x != 0.0,
        PyVal::Str(s) => !s.is_empty(),
        PyVal::Bytes(b) => !b.is_empty(),
        PyVal::List(v) | PyVal::Tuple(v) | PyVal::NdArray(_, v) => !v.is_empty(),
        PyVal::Dict(v) => !v.is_empty(),
        PyVal::Decimal(text) => !Dec::parse(text).is_some_and(|d| d.is_zero()),
        PyVal::Numpy(_, inner) => py_bool(inner),
        PyVal::Timedelta(n) => *n != 0,
        _ => true,
    }
}

/// `str(value)`.
pub fn py_str(value: &PyVal) -> String {
    match value {
        PyVal::Decimal(text) => Dec::parse(text).map_or_else(|| text.clone(), |d| d.to_sci()),
        other => other.str_(),
    }
}

/// `repr(value)`.
pub fn py_repr(value: &PyVal) -> String {
    match value {
        PyVal::Decimal(text) => Dec::parse(text).map_or_else(|| value.repr(), |d| d.repr()),
        PyVal::Numpy(name, inner) => match inner.as_ref() {
            PyVal::Decimal(_) => value.repr(),
            other => match other {
                PyVal::Float(x) => format!("np.{name}({})", float_text(*x)),
                _ => value.repr(),
            },
        },
        PyVal::Float(x) => float_text(*x),
        other => other.repr(),
    }
}

/// `repr` of a Python float.
pub fn float_text(x: f64) -> String {
    sdm_pql::pyrepr::float_repr(x)
}

#[cfg(test)]
mod consumed_protocol_tests {
    use super::*;
    use serde_json::{Value, json};

    fn held(view: Value) -> Value {
        json!({"$held": {"reference": {"$host_ref": "0"}, "owner": "scope", "type": "dict", "value": view}})
    }

    #[test]
    fn ordinary_container_operations_are_callback_free() {
        let protocols = HostProtocols::new(|_, _| panic!("unexpected host fact"));
        let mut mapping = json!({"field": 7});
        assert_eq!(
            mapping_get(&mapping, "field", &Value::Null, Some(&protocols)).unwrap(),
            json!(7)
        );
        assert!(is_instance(&mapping, InstanceFamily::Dict, Some(&protocols)).unwrap());
        assert_eq!(mapping_copy(&mapping, Some(&protocols)).unwrap(), mapping);
        mapping_update(&mut mapping, "field", None, Some(&protocols)).unwrap();
        assert_eq!(mapping, json!({}));
        assert_eq!(
            getitem(&json!([3]), &PyVal::Int(0), Some(&protocols)).unwrap(),
            json!(3)
        );
    }

    #[test]
    fn held_container_operations_require_the_provider() {
        let mut mapping = held(json!({}));
        assert!(mapping_get(&mapping, "field", &Value::Null, None).is_err());
        assert!(is_instance(&mapping, InstanceFamily::Dict, None).is_err());
        assert!(mapping_copy(&mapping, None).is_err());
        assert!(mapping_update(&mut mapping, "field", None, None).is_err());
        assert!(getitem(&mapping, &PyVal::Str("field".into()), None).is_err());
    }

    #[test]
    fn mapping_results_require_same_scope_and_exact_dictionary_view() {
        let mut foreign = held(json!({}));
        foreign["$held"]["owner"] = json!("foreign");
        let mut non_dictionary = held(json!([]));
        non_dictionary["$held"]["type"] = json!("list");
        for response in [json!({}), foreign, non_dictionary] {
            let protocols = HostProtocols::new(move |_, _| Ok(response.clone()));
            let mut mapping = held(json!({}));
            assert!(mapping_copy(&mapping, Some(&protocols)).is_err());
            assert!(
                mapping_update(&mut mapping, "field", Some(json!(1)), Some(&protocols)).is_err()
            );
        }
    }

    #[test]
    fn type_facts_are_boolean_and_get_results_are_decodable() {
        let protocols = HostProtocols::new(|_, _| Ok(json!({"$int": []})));
        let mapping = held(json!({}));
        assert!(is_instance(&mapping, InstanceFamily::Dict, Some(&protocols)).is_err());
        assert!(mapping_get(&mapping, "field", &Value::Null, Some(&protocols)).is_err());
        assert!(getitem(&mapping, &PyVal::Str("field".into()), Some(&protocols)).is_err());
    }

    #[test]
    fn nested_lookup_results_cannot_cross_operation_scopes() {
        let mut foreign = held(json!({}));
        foreign["$held"]["owner"] = json!("foreign");
        let protocols = HostProtocols::new(move |_, _| Ok(json!({"nested": [foreign]})));
        let mapping = held(json!({}));
        assert!(mapping_get(&mapping, "field", &Value::Null, Some(&protocols)).is_err());
        assert!(getitem(&mapping, &PyVal::Str("field".into()), Some(&protocols)).is_err());
    }
}
