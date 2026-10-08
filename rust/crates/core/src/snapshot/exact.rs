// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value as Json;

use super::{Error, Result, require};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let mut parser = Parser { bytes, position: 0 };
        let value = parser.value(0)?;
        parser.space();
        require(parser.position == bytes.len(), "trailing exact JSON input")?;
        Ok(value)
    }

    pub(super) fn from_json(value: Json) -> Self {
        match value {
            Json::Null => Self::Null,
            Json::Bool(value) => Self::Bool(value),
            Json::Number(value) => Self::Number(value.to_string()),
            Json::String(value) => Self::String(value),
            Json::Array(values) => Self::Array(values.into_iter().map(Self::from_json).collect()),
            Json::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(name, value)| (name, Self::from_json(value)))
                    .collect(),
            ),
        }
    }

    pub(super) fn field(&self, name: &str) -> Result<&Self> {
        let Self::Object(fields) = self else {
            return Err(Error("expected exact object"));
        };
        fields
            .iter()
            .find_map(|(key, value)| (key == name).then_some(value))
            .ok_or(Error("missing exact field"))
    }

    pub(super) fn field_mut(&mut self, name: &str) -> Result<&mut Self> {
        let Self::Object(fields) = self else {
            return Err(Error("expected exact object"));
        };
        fields
            .iter_mut()
            .find_map(|(key, value)| (key == name).then_some(value))
            .ok_or(Error("missing exact field"))
    }

    pub(super) fn typed(self) -> Result<Self> {
        Ok(match self {
            Self::Object(values) => Self::Object(vec![(
                "mapping".into(),
                Self::Array(
                    values
                        .into_iter()
                        .map(|(name, value)| {
                            Ok(Self::Array(vec![
                                Self::Object(vec![("scalar".into(), Self::String(name))]),
                                value.typed()?,
                            ]))
                        })
                        .collect::<Result<_>>()?,
                ),
            )]),
            Self::Array(values) => Self::Object(vec![(
                "list".into(),
                Self::Array(values.into_iter().map(Self::typed).collect::<Result<_>>()?),
            )]),
            Self::Number(ref number) if number.contains(['.', 'e', 'E']) || number == "-0" => {
                Self::from_json(super::encode_typed(self.narrow()?)?)
            }
            scalar => Self::Object(vec![("scalar".into(), scalar)]),
        })
    }

    pub(super) fn array(&self) -> Result<&[Self]> {
        if let Self::Array(values) = self {
            Ok(values)
        } else {
            Err(Error("expected exact array"))
        }
    }

    pub(super) fn text(&self) -> Result<&str> {
        if let Self::String(value) = self {
            Ok(value)
        } else {
            Err(Error("expected exact text"))
        }
    }

    pub(super) fn integer(&self) -> Result<&str> {
        let Self::Number(value) = self else {
            return Err(Error("expected exact integer"));
        };
        require(!value.contains(['.', 'e', 'E']), "expected exact integer")?;
        Ok(value)
    }

    pub(super) fn narrow(&self) -> Result<Json> {
        Ok(match self {
            Self::Null => Json::Null,
            Self::Bool(value) => Json::Bool(*value),
            Self::String(value) => Json::String(value.clone()),
            Self::Number(value) => {
                let decoded: Json = serde_json::from_str(value)
                    .map_err(|_| Error("number exceeds narrow JSON domain"))?;
                if !value.contains(['.', 'e', 'E']) && value != "-0" {
                    require(
                        decoded.as_i64().is_some() || decoded.as_u64().is_some(),
                        "integer exceeds narrow JSON domain",
                    )?;
                }
                decoded
            }
            Self::Array(values) => {
                Json::Array(values.iter().map(Self::narrow).collect::<Result<_>>()?)
            }
            Self::Object(values) => Json::Object(
                values
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), value.narrow()?)))
                    .collect::<Result<_>>()?,
            ),
        })
    }

    pub(super) fn structure(&self) -> Json {
        match self {
            Self::Array(values) => Json::Array(values.iter().map(Self::structure).collect()),
            Self::Object(values) => Json::Object(
                values
                    .iter()
                    .map(|(name, value)| (name.clone(), value.structure()))
                    .collect(),
            ),
            _ => self.narrow().unwrap_or(Json::Null),
        }
    }

    pub(super) fn to_json(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => value.clone(),
            Self::String(value) => serde_json::to_string(value).unwrap(),
            Self::Array(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(Self::to_json)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Object(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(name, value)| format!(
                        "{}:{}",
                        serde_json::to_string(name).unwrap(),
                        value.to_json()
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| matches!(byte, b' ' | b'\r' | b'\n' | b'\t'))
        {
            self.position += 1;
        }
    }

    fn take(&mut self, expected: u8) -> Result<()> {
        self.space();
        require(
            self.bytes.get(self.position) == Some(&expected),
            "invalid exact JSON delimiter",
        )?;
        self.position += 1;
        Ok(())
    }

    fn string(&mut self) -> Result<String> {
        self.space();
        let start = self.position;
        self.take(b'"')?;
        loop {
            match self.bytes.get(self.position) {
                Some(b'"') => {
                    self.position += 1;
                    return serde_json::from_slice(&self.bytes[start..self.position])
                        .map_err(|_| Error("invalid exact JSON string"));
                }
                Some(b'\\') => self.position += 2,
                Some(_) => self.position += 1,
                None => return Err(Error("unterminated exact JSON string")),
            }
        }
    }

    fn digits(&mut self) -> Result<()> {
        let start = self.position;
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_digit)
        {
            self.position += 1;
        }
        require(self.position > start, "missing exact JSON digits")
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.position;
        if self.bytes.get(self.position) == Some(&b'-') {
            self.position += 1;
        }
        if self.bytes.get(self.position) == Some(&b'0') {
            self.position += 1;
        } else {
            self.digits()?;
        }
        if self.bytes.get(self.position) == Some(&b'.') {
            self.position += 1;
            self.digits()?;
        }
        if self
            .bytes
            .get(self.position)
            .is_some_and(|byte| matches!(byte, b'e' | b'E'))
        {
            self.position += 1;
            if self
                .bytes
                .get(self.position)
                .is_some_and(|byte| matches!(byte, b'+' | b'-'))
            {
                self.position += 1;
            }
            self.digits()?;
        }
        let number = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| Error("invalid exact JSON number"))?;
        Ok(Value::Number(number.to_owned()))
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        require(depth <= 128, "exact JSON nesting limit")?;
        self.space();
        match self.bytes.get(self.position).copied() {
            Some(b'"') => self.string().map(Value::String),
            Some(b'0'..=b'9' | b'-') => self.number(),
            Some(b'n' | b't' | b'f') => {
                let (word, value): (&[u8], Value) = match self.bytes[self.position] {
                    b'n' => (b"null", Value::Null),
                    b't' => (b"true", Value::Bool(true)),
                    _ => (b"false", Value::Bool(false)),
                };
                require(
                    self.bytes[self.position..].starts_with(word),
                    "invalid exact JSON literal",
                )?;
                self.position += word.len();
                Ok(value)
            }
            Some(b'[') => {
                self.position += 1;
                self.space();
                let mut values = Vec::new();
                if self.bytes.get(self.position) != Some(&b']') {
                    loop {
                        values.push(self.value(depth + 1)?);
                        self.space();
                        if self.bytes.get(self.position) == Some(&b']') {
                            break;
                        }
                        self.take(b',')?;
                    }
                }
                self.take(b']')?;
                Ok(Value::Array(values))
            }
            Some(b'{') => {
                self.position += 1;
                self.space();
                let mut values = Vec::new();
                if self.bytes.get(self.position) != Some(&b'}') {
                    loop {
                        let name = self.string()?;
                        require(
                            !values.iter().any(|(key, _)| key == &name),
                            "duplicate exact JSON key",
                        )?;
                        self.take(b':')?;
                        values.push((name, self.value(depth + 1)?));
                        self.space();
                        if self.bytes.get(self.position) == Some(&b'}') {
                            break;
                        }
                        self.take(b',')?;
                    }
                }
                self.take(b'}')?;
                Ok(Value::Object(values))
            }
            _ => Err(Error("invalid exact JSON value")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_integer_input_and_output_are_lossless_without_global_features() {
        for number in [
            "340282366920938463463374607431768211457",
            "-9999999999999999999999999999999999999999999999999999999999999999999999999999",
        ] {
            let encoded = format!("{{\"scalar\":{number}}}");
            let value = Value::parse(encoded.as_bytes()).unwrap();
            assert_eq!(value.to_json(), encoded);
            assert_eq!(value.field("scalar").unwrap().integer().unwrap(), number);
            assert!(value.narrow().is_err());
        }
    }

    #[test]
    fn normal_json_and_private_tag_objects_keep_their_meaning() {
        for source in [
            r#"{"$serde_json::private::Number":"123","tail":true}"#,
            r#"{"$serde_json::private::RawValue":"{}"}"#,
            r#"[null,true,false,"a\n\"",-0,1.25,1e3,18446744073709551615,-9223372036854775808]"#,
        ] {
            let parsed = Value::parse(source.as_bytes()).unwrap();
            let expected: Json = serde_json::from_str(source).unwrap();
            assert_eq!(parsed.narrow().unwrap(), expected);
            assert_eq!(
                serde_json::from_str::<Json>(&parsed.to_json()).unwrap(),
                expected
            );
        }
        for invalid in [
            "[1,]",
            "{\"x\":1,\"x\":2}",
            "01",
            "1.",
            "1e",
            "--1",
            "{} true",
            "\"\\uD800\"",
        ] {
            assert!(Value::parse(invalid.as_bytes()).is_err(), "{invalid}");
        }
    }
}
