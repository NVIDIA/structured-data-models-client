// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A small JSON writer, so integers of any size are written as their digits.

/// A JSON value. Numbers are kept as the text to write.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn str(text: impl Into<String>) -> Self {
        Json::String(text.into())
    }

    pub fn opt_str(text: Option<impl Into<String>>) -> Self {
        text.map_or(Json::Null, |t| Json::String(t.into()))
    }

    pub fn int(value: impl ToString) -> Self {
        Json::Number(value.to_string())
    }

    pub fn object(fields: Vec<(&str, Json)>) -> Self {
        Json::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    /// The value as compact JSON text.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => out.push_str(n),
            Json::String(s) => out.push_str(&quote(s)),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Object(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&quote(key));
                    out.push_str(": ");
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// `text` as a JSON string literal.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_json() {
        let value = Json::object(vec![
            ("a", Json::int("99999999999999999999999")),
            (
                "b",
                Json::Array(vec![Json::Null, Json::Bool(true), Json::str("q\"\u{1}")]),
            ),
        ]);
        assert_eq!(
            value.render(),
            "{\"a\": 99999999999999999999999, \"b\": [null, true, \"q\\\"\\u0001\"]}"
        );
    }
}
