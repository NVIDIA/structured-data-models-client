// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The model API's prediction response, read as the client's generated
//! bindings read it (`PredictionResponse.from_dict` and
//! `PredictionItem.from_dict` of `sdm_api.generated.tfm_api`), for the
//! fields it reads: the response's `id`, `model`, `predictions` and
//! `metadata`, and each item's `id`, `row_index` and `forecast_step`, each
//! required and typed as the bindings require it, checked in their order.
//! For those fields a response the bindings refuse is refused here, and
//! one they read is read (`contract/response-vectors.json` holds the two
//! together). A refusal names the field and why, never a value.
//!
//! From T1.28b an item's `probabilities` is read as the bindings'
//! `_float_dict` reads it: absent or `null` is none, any other value not an
//! object is refused (`not_an_object`), and each member's value is a
//! number, or text a float reads, not a boolean (`not_a_number`) and
//! finite (`not_finite`), kept in the order written.
//!
//! As built at T1.11d the item's `prediction` is taken as written, as the
//! bindings take it, and its other fields (`scores`,
//! `rankings`, `embeddings`, `quantiles`, `explanation`, `metadata`) are
//! not read or type-checked: when one is there and not null, it is named in
//! [`PredictionItem::carried`], for the caller to refuse by name as a field
//! the result does not carry yet (`UNSUPPORTED_REQUEST`, register P1B-4),
//! where the bindings would refuse a malformed one as a response not the
//! contract's (`INVALID_RESPONSE`, decision E45(f)); T1.28 reads them.
//! Members the bindings do not read are ignored, as there.

use crate::json::{Value, whole};

/// Why a response is not one the contract describes: the field, and the
/// fault, never the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refused {
    /// The field, as `prediction.row_index` or `metadata`.
    pub field: &'static str,
    /// `missing`, `null`, `not_text`, `not_whole`, `not_an_object` or
    /// `not_a_sequence`.
    pub reason: &'static str,
}

fn refused(field: &'static str, reason: &'static str) -> Refused {
    Refused { field, reason }
}

/// The fields of an item this build does not read, in the bindings' order.
pub const UNREAD_ITEM_FIELDS: [&str; 6] = [
    "scores",
    "rankings",
    "embeddings",
    "quantiles",
    "explanation",
    "metadata",
];

/// One answer.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionItem {
    /// Which row of the request this answers; a whole number, of any size
    /// a response may write.
    pub row_index: i128,
    pub id: Option<String>,
    pub forecast_step: Option<i128>,
    /// The answer as the response wrote it; absent and `null` alike are
    /// `None`, as the bindings read both.
    pub prediction: Option<Value>,
    /// Each class's probability, in the order written; absent and `null`
    /// alike are `None`.
    pub probabilities: Option<Vec<(String, f64)>>,
    /// The fields of [`UNREAD_ITEM_FIELDS`] this item carries, not null.
    pub carried: Vec<&'static str>,
}

/// The response to one prediction request.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResponse {
    pub id: String,
    pub model: String,
    pub predictions: Vec<PredictionItem>,
    /// The endpoint's `metadata` object, as written.
    pub metadata: Value,
}

impl PredictionResponse {
    /// Reads `value` as the bindings read a response.
    pub fn read(value: &Value) -> Result<PredictionResponse, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("response", "not_an_object"));
        };
        let id = required_text(value, "id", "id")?;
        let model = required_text(value, "model", "model")?;
        let predictions = match value.get("predictions") {
            None => return Err(refused("predictions", "missing")),
            Some(Value::Array(items)) => items
                .iter()
                .map(PredictionItem::read)
                .collect::<Result<_, _>>()?,
            // An object iterates over its names, each of which is text and
            // so not an item; with none, there are no items.
            Some(Value::Object(members)) if members.is_empty() => vec![],
            Some(Value::Object(_)) => return Err(refused("prediction", "not_an_object")),
            Some(_) => return Err(refused("predictions", "not_a_sequence")),
        };
        let metadata = match value.get("metadata") {
            None => return Err(refused("metadata", "missing")),
            Some(object @ Value::Object(_)) => object.clone(),
            Some(_) => return Err(refused("metadata", "not_an_object")),
        };
        Ok(PredictionResponse {
            id,
            model,
            predictions,
            metadata,
        })
    }
}

impl PredictionItem {
    fn read(value: &Value) -> Result<PredictionItem, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("prediction", "not_an_object"));
        };
        let id = match value.get("id") {
            None | Some(Value::Null) => None,
            Some(Value::Text(text)) => Some(text.clone()),
            Some(_) => return Err(refused("prediction.id", "not_text")),
        };
        let row_index = match value.get("row_index") {
            None => return Err(refused("prediction.row_index", "missing")),
            Some(Value::Null) => return Err(refused("prediction.row_index", "null")),
            Some(number) => whole_of(number, "prediction.row_index")?,
        };
        let forecast_step = match value.get("forecast_step") {
            None | Some(Value::Null) => None,
            Some(number) => Some(whole_of(number, "prediction.forecast_step")?),
        };
        let prediction = match value.get("prediction") {
            None | Some(Value::Null) => None,
            Some(answer) => Some(answer.clone()),
        };
        let probabilities = match value.get("probabilities") {
            None | Some(Value::Null) => None,
            Some(Value::Object(members)) => Some(
                members
                    .iter()
                    .map(|(class, p)| Ok((class.clone(), probability(p)?)))
                    .collect::<Result<_, Refused>>()?,
            ),
            Some(_) => return Err(refused("prediction.probabilities", "not_an_object")),
        };
        let carried = UNREAD_ITEM_FIELDS
            .into_iter()
            .filter(|name| !matches!(value.get(name), None | Some(Value::Null)))
            .collect();
        Ok(PredictionItem {
            row_index,
            id,
            forecast_step,
            prediction,
            probabilities,
            carried,
        })
    }
}

fn required_text(value: &Value, name: &str, field: &'static str) -> Result<String, Refused> {
    match value.get(name) {
        None => Err(refused(field, "missing")),
        Some(Value::Text(text)) => Ok(text.clone()),
        Some(_) => Err(refused(field, "not_text")),
    }
}

/// A whole number, as the bindings' `_whole` takes one: an integer, or a
/// float with no fraction; never a boolean or text.
fn whole_of(value: &Value, field: &'static str) -> Result<i128, Refused> {
    match value {
        Value::Number(token) => whole(token).ok_or(refused(field, "not_whole")),
        _ => Err(refused(field, "not_whole")),
    }
}

/// A probability as the bindings' `_finite` reads one: a number, or text
/// Python's `float` reads ([`numeric_text`]), never a boolean, and finite.
fn probability(value: &Value) -> Result<f64, Refused> {
    const FIELD: &str = "prediction.probabilities";
    let read = match value {
        Value::Number(token) => token.parse::<f64>().ok(),
        Value::Text(text) => numeric_text(text),
        _ => None,
    };
    match read {
        None => Err(refused(FIELD, "not_a_number")),
        Some(p) if !p.is_finite() => Err(refused(FIELD, "not_finite")),
        Some(p) => Ok(p),
    }
}

/// The first code point of each run of ten Unicode decimal digits (0 to 9)
/// that every supported Python's `float` takes as the digits they stand for:
/// Unicode 13.0's (`unicodedata.decimal` of Python 3.10, the floor), a
/// portable policy (T1.28b fix round 2): a digit later Unicode versions add,
/// which the reference's Python 3.12 and later take, is refused (register
/// P1B-25).
const DECIMAL_ZEROS: [u32; 65] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x16A60, 0x16B50, 0x1D7CE,
    0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E950, 0x1FBF0,
];

/// The white space every supported Python's `float` strips around a
/// number: what `str.isspace` takes as white space but the four separators
/// 0x1C to 0x1F, which `float` refuses (measured on Python 3.10 to 3.14).
const WHITE_SPACE: [u32; 25] = [
    0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
    0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
];

/// Text as every supported Python's `float` and `int` see it before they
/// read it: each white space a space and each Unicode decimal digit the
/// ASCII digit it stands for, the spaces around it trimmed, and a single
/// underscore between two digits dropped; `None` for text holding any other
/// character outside ASCII, or an underscore anywhere else.
fn ascii_digits(text: &str) -> Option<String> {
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        let code = u32::from(c);
        if WHITE_SPACE.contains(&code) {
            ascii.push(' ');
        } else if let Some(zero) = DECIMAL_ZEROS
            .iter()
            .find(|zero| (**zero..**zero + 10).contains(&code))
        {
            ascii.push(char::from(b'0' + u8::try_from(code - zero).ok()?));
        } else if c.is_ascii() {
            ascii.push(c);
        } else {
            return None;
        }
    }
    let trimmed = ascii.trim_matches(' ');
    let bytes = trimmed.as_bytes();
    // A single underscore between two digits, and nowhere else.
    for (at, byte) in bytes.iter().enumerate() {
        if *byte == b'_'
            && !(at > 0
                && bytes[at - 1].is_ascii_digit()
                && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            return None;
        }
    }
    Some(trimmed.chars().filter(|c| *c != '_').collect())
}

/// Text as Python's `int` reads it (the grammar its documentation gives for
/// a string argument in base 10), as `ascii_digits` sees it: a sign, then
/// decimal digits; the integer's digits, with no leading zero and a minus
/// sign only for a negative one, or `None` for any other text.
pub fn integer_text(text: &str) -> Option<String> {
    let plain = ascii_digits(text)?;
    let negative = plain.starts_with('-');
    let digits = plain.strip_prefix(['+', '-']).unwrap_or(&plain);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    Some(match (digits.is_empty(), negative) {
        (true, _) => "0".to_owned(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.to_owned(),
    })
}

/// Text as Python's `float` reads it (the grammar its documentation gives
/// for a string argument): white space around it, a sign, then `inf`,
/// `infinity` or `nan` in any case, or a decimal number (digits with an
/// optional fraction, or a fraction alone, then an optional exponent), any
/// Unicode decimal digit read as the digit it stands for and a single
/// underscore allowed between two digits; `None` for any other text.
pub fn numeric_text(text: &str) -> Option<f64> {
    let plain = ascii_digits(text)?;
    let unsigned = plain.strip_prefix(['+', '-']).unwrap_or(&plain);
    let lower = unsigned.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity" | "nan") {
        let negative = plain.starts_with('-');
        return Some(match (lower.as_str(), negative) {
            ("nan", _) => f64::NAN,
            (_, false) => f64::INFINITY,
            (_, true) => f64::NEG_INFINITY,
        });
    }
    // digits [. [digits]] | . digits, then [e [sign] digits]
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(at) => (&unsigned[..at], Some(&unsigned[at + 1..])),
        None => (unsigned, None),
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let (whole, fraction) = match mantissa.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (mantissa, None),
    };
    let mantissa_ok = match fraction {
        None => digits(whole),
        Some(fraction) => {
            (digits(whole) && (fraction.is_empty() || digits(fraction)))
                || (whole.is_empty() && digits(fraction))
        }
    };
    let exponent_ok = exponent.is_none_or(|e| digits(e.strip_prefix(['+', '-']).unwrap_or(e)));
    if !(mantissa_ok && exponent_ok) {
        return None;
    }
    plain.parse::<f64>().ok()
}
