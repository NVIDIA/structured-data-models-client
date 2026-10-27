// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON text embedded inside a service ID, projected to its root and members.
//! The caller applies the character bound. This is text JSON, not model bytes:
//! no byte-order decoding, finite-number restriction or outer-message depth cap.

use serde_json::value::RawValue;

#[derive(Debug, Clone, PartialEq)]
pub enum Member {
    Null,
    Boolean(bool),
    Integer(String),
    Float(f64),
    Unicode(Vec<u32>),
    Array,
    Object,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Root {
    pub value: Member,
    pub members: Option<Vec<Member>>,
}

pub fn read(text: &str, decimal_limit: u32) -> Option<Root> {
    let tokens = crate::json::non_finite_tokens(text.as_bytes());
    let marker = (0xE000_u32..=0xF8FF)
        .map(|point| format!("\\u{point:04X}"))
        .find(|marker| !text.contains(marker))?;
    let mut normalized = Vec::with_capacity(text.len() + tokens.len() * 16);
    let mut from = 0;
    for &(at, token) in &tokens {
        normalized.extend_from_slice(&text.as_bytes()[from..at]);
        normalized.push(b'"');
        normalized.extend_from_slice(marker.as_bytes());
        normalized.extend_from_slice(token.as_bytes());
        normalized.push(b'"');
        from = at + token.len();
    }
    normalized.extend_from_slice(&text.as_bytes()[from..]);
    // IgnoredAny/RawValue traverse nested syntax iteratively, without building
    // a recursive value. Every integer is converted by the native JSON reader,
    // even in an object or member the singleton policy later ignores.
    serde_json::from_slice::<serde::de::IgnoredAny>(&normalized).ok()?;
    if !integers_fit(&normalized, decimal_limit) {
        return None;
    }
    let raw: &RawValue = serde_json::from_slice(&normalized).ok()?;
    let value = member(raw, &marker)?;
    let members = if value == Member::Array {
        let items: Vec<&RawValue> = serde_json::from_str(raw.get()).ok()?;
        Some(
            items
                .into_iter()
                .map(|item| member(item, &marker))
                .collect::<Option<_>>()?,
        )
    } else {
        None
    };
    Some(Root { value, members })
}

fn integers_fit(bytes: &[u8], limit: u32) -> bool {
    if limit == 0 {
        return true;
    }
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'"' {
            at += 1;
            while at < bytes.len() {
                match bytes[at] {
                    b'\\' => at += 2,
                    b'"' => {
                        at += 1;
                        break;
                    }
                    _ => at += 1,
                }
            }
        } else if bytes[at] == b'-' || bytes[at].is_ascii_digit() {
            let start = at;
            at += 1;
            while at < bytes.len()
                && matches!(bytes[at], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
            {
                at += 1;
            }
            let number = &bytes[start..at];
            if !number.iter().any(|c| matches!(c, b'.' | b'e' | b'E')) {
                let digits = number.len() - usize::from(number[0] == b'-');
                if digits > limit as usize {
                    return false;
                }
            }
        } else {
            at += 1;
        }
    }
    true
}

fn member(raw: &RawValue, marker: &str) -> Option<Member> {
    let text = raw.get();
    if let Some(token) = text
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .and_then(|s| s.strip_prefix(marker))
    {
        return match token {
            "NaN" => Some(Member::Float(f64::NAN)),
            "Infinity" => Some(Member::Float(f64::INFINITY)),
            "-Infinity" => Some(Member::Float(f64::NEG_INFINITY)),
            _ => None,
        };
    }
    match text.as_bytes().first()? {
        b'[' => Some(Member::Array),
        b'{' => Some(Member::Object),
        b'n' => Some(Member::Null),
        b't' => Some(Member::Boolean(true)),
        b'f' => Some(Member::Boolean(false)),
        b'"' => unicode(text).map(Member::Unicode),
        _ if crate::json::is_integer_token(text) => {
            crate::response::integer_text(text).map(Member::Integer)
        }
        _ => text.parse().ok().map(Member::Float),
    }
}

fn unicode(text: &str) -> Option<Vec<u32>> {
    let mut chars = text.strip_prefix('"')?.strip_suffix('"')?.chars();
    let mut points = Vec::new();
    while let Some(c) = chars.next() {
        if c != '\\' {
            points.push(u32::from(c));
            continue;
        }
        let point = match chars.next()? {
            '"' => u32::from('"'),
            '\\' => u32::from('\\'),
            '/' => u32::from('/'),
            'b' => 8,
            'f' => 12,
            'n' => 10,
            'r' => 13,
            't' => 9,
            'u' => {
                let mut hex = || -> Option<u32> {
                    let mut point = 0;
                    for _ in 0..4 {
                        point = point * 16 + chars.next()?.to_digit(16)?;
                    }
                    Some(point)
                };
                let high = hex()?;
                if (0xD800..=0xDBFF).contains(&high) {
                    let mut next = chars.clone();
                    if next.next() == Some('\\') && next.next() == Some('u') {
                        let mut low = 0;
                        for _ in 0..4 {
                            low = low * 16 + next.next()?.to_digit(16)?;
                        }
                        if (0xDC00..=0xDFFF).contains(&low) {
                            chars = next;
                            0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            high
                        }
                    } else {
                        high
                    }
                } else {
                    high
                }
            }
            _ => return None,
        };
        points.push(point);
    }
    Some(points)
}
