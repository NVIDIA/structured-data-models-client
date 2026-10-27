// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bounded singleton-list data notation used by tabular response IDs.
//! This reads literal data only, with no names, calls or expression evaluator.

use super::{literal_number, primitive_text};
use regex::Regex;
use sdm_wire::json::Value;
use std::sync::OnceLock;

#[path = "unicode_aliases.rs"]
mod unicode_aliases;

#[derive(Debug)]
pub(super) enum Name {
    Primitive(String),
    Different,
}
#[derive(Debug)]
pub(super) struct EscapeNotice {
    pub escape: String,
    pub octal: bool,
}
#[derive(Debug)]
pub(super) struct Read {
    pub name: Option<Name>,
    pub notices: Vec<EscapeNotice>,
}

const PROFILES: [&str; 5] = ["13.0.0", "14.0.0", "15.0.0", "15.1.0", "16.0.0"];

pub(super) fn supported(profile: &str) -> bool {
    PROFILES.contains(&profile)
}

fn named_character(name: &str, profile: &str) -> Option<char> {
    let bit = PROFILES.iter().position(|p| *p == profile)?;
    if !name.is_ascii() {
        return None;
    }
    let upper = name.to_ascii_uppercase();
    let cjk = upper.starts_with("CJK UNIFIED IDEOGRAPH-");
    let hangul = upper.starts_with("HANGUL SYLLABLE ");
    // CPython's algorithmic branches are case-sensitive, unlike its hashed
    // canonical-name/alias lookup. CJK permits four or five uppercase hex digits,
    // including a leading zero in the five-digit spelling of a BMP character.
    if cjk {
        let digits = name.strip_prefix("CJK UNIFIED IDEOGRAPH-")?;
        if !matches!(digits.len(), 4 | 5)
            || !digits
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(&c))
        {
            return None;
        }
    }
    if hangul && name != upper {
        return None;
    }
    let character = unicode_names2::character(name)?;
    // The maintained age property's semantics include every earlier assignment.
    static AGES: OnceLock<Vec<Regex>> = OnceLock::new();
    let ages = AGES.get_or_init(|| {
        PROFILES
            .iter()
            .map(|version| {
                let age = version.strip_suffix(".0").unwrap_or(version);
                Regex::new(&format!(r"\A\p{{Age:{age}}}\z")).expect("fixed supported Unicode age")
            })
            .collect()
    });
    if !ages[bit].is_match(&character.to_string()) {
        return None;
    }
    if cjk {
        return Some(character);
    }
    if unicode_names2::name(character)
        .is_some_and(|canonical| canonical.to_string().eq_ignore_ascii_case(name))
    {
        return Some(character);
    }
    unicode_aliases::ALIASES
        .iter()
        .find_map(|(alias, point, mask)| {
            (mask & (1 << bit) != 0 && alias.eq_ignore_ascii_case(name))
                .then(|| char::from_u32(*point))
                .flatten()
        })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Other,
    Real,
    Complex,
    Text,
    Bytes,
}
struct Datum {
    text: Option<String>,
    kind: Kind,
    hashable: bool,
    valid: bool,
    signed: bool,
}
impl Datum {
    fn other(hashable: bool, valid: bool) -> Self {
        Self {
            text: None,
            kind: Kind::Other,
            hashable,
            valid,
            signed: false,
        }
    }
    fn text(text: String, kind: Kind) -> Self {
        Self {
            text: Some(text),
            kind,
            hashable: true,
            valid: true,
            signed: false,
        }
    }
}

pub(super) fn read(value: &str, profile: &str, float_text: fn(f64) -> String) -> Read {
    // The reference bounds characters, rather than UTF-8 bytes.
    if value.chars().count() > 1024 {
        return Read {
            name: None,
            notices: vec![],
        };
    }
    if let Ok(Value::Array(parts)) = sdm_wire::json::read_answer(value.as_bytes()) {
        return Read {
            name: match parts.as_slice() {
                [part] => {
                    Some(primitive_text(part, float_text).map_or(Name::Different, Name::Primitive))
                }
                _ => None,
            },
            notices: vec![],
        };
    }
    let mut reader = Reader {
        chars: value.chars().collect(),
        at: 0,
        profile,
        float_text,
        notices: vec![],
        depth: 1,
    };
    let name = (|| {
        reader.skip();
        reader.take('[')?;
        let values = reader.sequence(']')?;
        reader.skip();
        if reader.at != reader.chars.len() {
            return None;
        }
        let mut values = values;
        match values.as_mut_slice() {
            [one] if one.valid => Some(one.text.take().map_or(Name::Different, Name::Primitive)),
            _ => None,
        }
    })();
    Read {
        name,
        notices: reader.notices,
    }
}

struct Reader<'a> {
    chars: Vec<char>,
    at: usize,
    profile: &'a str,
    float_text: fn(f64) -> String,
    notices: Vec<EscapeNotice>,
    depth: usize,
}
impl Reader<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }
    fn skip(&mut self) {
        loop {
            while self
                .peek()
                .is_some_and(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c'))
            {
                self.at += 1;
            }
            if self.peek() == Some('#') {
                while self.peek().is_some_and(|c| c != '\n') {
                    self.at += 1;
                }
            } else if self.peek() == Some('\\') && self.chars.get(self.at + 1) == Some(&'\n') {
                self.at += 2;
            } else {
                break;
            }
        }
    }
    fn take(&mut self, c: char) -> Option<()> {
        self.skip();
        if self.peek() != Some(c) {
            return None;
        }
        self.at += 1;
        Some(())
    }
    fn sequence(&mut self, end: char) -> Option<Vec<Datum>> {
        self.skip();
        let mut values = Vec::new();
        if self.peek() == Some(end) {
            self.at += 1;
            return Some(values);
        }
        loop {
            values.push(self.value()?);
            self.skip();
            if self.peek() == Some(end) {
                self.at += 1;
                return Some(values);
            }
            self.take(',')?;
            self.skip();
            if self.peek() == Some(end) {
                self.at += 1;
                return Some(values);
            }
        }
    }
    fn value(&mut self) -> Option<Datum> {
        let first = self.atom()?;
        self.skip();
        if matches!(self.peek(), Some('+' | '-')) {
            self.at += 1;
            let right = self.atom()?;
            if first.kind != Kind::Real || right.kind != Kind::Complex || right.signed {
                return None;
            }
            return Some(Datum::other(true, first.valid && right.valid));
        }
        Some(first)
    }
    fn atom(&mut self) -> Option<Datum> {
        self.skip();
        let grouped = matches!(self.peek(), Some('[' | '(' | '{'));
        if grouped {
            self.depth += 1;
            if self.depth > 200 {
                return None;
            }
        }
        let result = self.atom_inner();
        if grouped {
            self.depth -= 1;
        }
        result
    }
    fn atom_inner(&mut self) -> Option<Datum> {
        self.skip();
        let c = self.peek()?;
        match c {
            '[' => {
                self.at += 1;
                let values = self.sequence(']')?;
                Some(Datum::other(false, values.iter().all(|v| v.valid)))
            }
            '(' => {
                self.at += 1;
                self.skip();
                if self.peek() == Some(')') {
                    self.at += 1;
                    return Some(Datum::other(true, true));
                }
                let first = self.value()?;
                self.skip();
                if self.peek() == Some(')') {
                    self.at += 1;
                    return Some(first);
                }
                self.take(',')?;
                let rest = self.sequence(')')?;
                Some(Datum::other(
                    first.hashable && rest.iter().all(|v| v.hashable),
                    first.valid && rest.iter().all(|v| v.valid),
                ))
            }
            '{' => self.mapping(),
            '\'' | '"' => self.strings(),
            '+' | '-' => {
                self.at += 1;
                self.skip();
                let mut number = self.atom()?;
                if !matches!(number.kind, Kind::Real | Kind::Complex) || number.signed {
                    return None;
                }
                if c == '-' && number.kind == Kind::Real {
                    let old = number.text.take()?;
                    number.text = Some(if old.starts_with('-') {
                        old[1..].into()
                    } else if old == "0" {
                        old
                    } else {
                        format!("-{old}")
                    });
                }
                number.signed = true;
                Some(number)
            }
            c if c.is_ascii_digit() || c == '.' => self.number(),
            _ => {
                for (word, text) in [("True", "True"), ("False", "False"), ("None", "None")] {
                    if self.word(word) {
                        return Some(Datum::text(text.into(), Kind::Other));
                    }
                }
                if self.word("set") {
                    self.take('(')?;
                    self.take(')')?;
                    return Some(Datum::other(false, true));
                }
                self.strings()
            }
        }
    }
    fn word(&mut self, word: &str) -> bool {
        let letters = word.chars().collect::<Vec<_>>();
        if self.chars.get(self.at..self.at + letters.len()) != Some(letters.as_slice()) {
            return false;
        }
        if self
            .chars
            .get(self.at + letters.len())
            .is_some_and(|c| c.is_alphanumeric() || *c == '_')
        {
            return false;
        }
        self.at += letters.len();
        true
    }
    fn number(&mut self) -> Option<Datum> {
        let start = self.at;
        while let Some(c) = self.peek() {
            let exponent_sign = matches!(c, '+' | '-')
                && self.at > start
                && matches!(self.chars[self.at - 1], 'e' | 'E');
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.') || exponent_sign {
                self.at += 1;
            } else {
                break;
            }
        }
        let token: String = self.chars[start..self.at].iter().collect();
        if token == "..." {
            return Some(Datum::other(true, true));
        }
        if let Some(imaginary) = token.strip_suffix(['j', 'J']) {
            sdm_wire::response::numeric_text(imaginary)?;
            return Some(Datum {
                text: None,
                kind: Kind::Complex,
                hashable: true,
                valid: true,
                signed: false,
            });
        }
        let text = literal_number(&token, self.float_text)?;
        Some(Datum::text(text, Kind::Real))
    }
    fn mapping(&mut self) -> Option<Datum> {
        self.at += 1;
        self.skip();
        if self.peek() == Some('}') {
            self.at += 1;
            return Some(Datum::other(false, true));
        }
        let first = self.value()?;
        self.skip();
        let dict = self.peek() == Some(':');
        let mut valid = first.valid && first.hashable;
        if dict {
            self.at += 1;
            valid &= self.value()?.valid;
        }
        loop {
            self.skip();
            if self.peek() == Some('}') {
                self.at += 1;
                break;
            }
            self.take(',')?;
            self.skip();
            if self.peek() == Some('}') {
                self.at += 1;
                break;
            }
            let key = self.value()?;
            valid &= key.valid && key.hashable;
            if dict {
                self.take(':')?;
                valid &= self.value()?.valid;
            }
        }
        Some(Datum::other(false, valid))
    }
    fn strings(&mut self) -> Option<Datum> {
        let mut result = self.string()?;
        loop {
            self.skip();
            let start = self.at;
            let Some(next) = self.string() else {
                self.at = start;
                break;
            };
            if next.kind != result.kind {
                return None;
            }
            result.valid &= next.valid;
            match (&mut result.text, next.text) {
                (Some(before), Some(after)) => before.push_str(&after),
                _ => result.text = None,
            }
        }
        Some(result)
    }
    fn string(&mut self) -> Option<Datum> {
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.at += 1;
        }
        let prefix: String = self.chars[start..self.at]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase();
        if !matches!(prefix.as_str(), "" | "r" | "u" | "b" | "br" | "rb") {
            return None;
        }
        let quote = self.peek()?;
        if !matches!(quote, '\'' | '"') {
            return None;
        }
        self.at += 1;
        let triple =
            self.chars.get(self.at) == Some(&quote) && self.chars.get(self.at + 1) == Some(&quote);
        if triple {
            self.at += 2;
        }
        let raw = prefix.contains('r');
        let bytes = prefix.contains('b');
        let mut text = String::new();
        let mut scalar = true;
        let mut notice = None;
        loop {
            let c = self.peek()?;
            self.at += 1;
            if c == quote {
                if !triple {
                    break;
                }
                if self.chars.get(self.at) == Some(&quote)
                    && self.chars.get(self.at + 1) == Some(&quote)
                {
                    self.at += 2;
                    break;
                }
            }
            if !triple && matches!(c, '\n' | '\r') {
                return None;
            }
            if bytes && !c.is_ascii() {
                return None;
            }
            if c != '\\' {
                text.push(c);
                continue;
            }
            let escaped = self.peek()?;
            self.at += 1;
            if raw {
                text.push('\\');
                text.push(escaped);
                continue;
            }
            match escaped {
                '\\' | '\'' | '"' => text.push(escaped),
                'a' => text.push('\x07'),
                'b' => text.push('\x08'),
                'f' => text.push('\x0c'),
                'n' => text.push('\n'),
                'r' => text.push('\r'),
                't' => text.push('\t'),
                'v' => text.push('\x0b'),
                '\n' => {}
                '\r' => {
                    if self.peek() == Some('\n') {
                        self.at += 1;
                    }
                }
                'x' | 'u' | 'U' if escaped == 'x' || !bytes => {
                    let count = match escaped {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let mut point = 0_u32;
                    for _ in 0..count {
                        let c = self.peek()?;
                        self.at += 1;
                        point = point.checked_mul(16)?.checked_add(c.to_digit(16)?)?;
                    }
                    if point > 0x10FFFF {
                        return None;
                    }
                    if let Some(c) = char::from_u32(point) {
                        text.push(c);
                    } else {
                        scalar = false;
                    }
                }
                'N' if !bytes => {
                    if self.peek() != Some('{') {
                        return None;
                    }
                    self.at += 1;
                    let start = self.at;
                    while self.peek().is_some_and(|c| c != '}') {
                        self.at += 1;
                    }
                    if self.peek() != Some('}') {
                        return None;
                    }
                    let name: String = self.chars[start..self.at].iter().collect();
                    self.at += 1;
                    text.push(named_character(&name, self.profile)?);
                }
                '0'..='7' => {
                    let mut point = escaped.to_digit(8)?;
                    let mut digits = escaped.to_string();
                    for _ in 0..2 {
                        let Some(digit) = self.peek().and_then(|c| c.to_digit(8)) else {
                            break;
                        };
                        digits.push(self.peek()?);
                        self.at += 1;
                        point = point * 8 + digit;
                    }
                    if point > 255 && notice.is_none() {
                        notice = Some(EscapeNotice {
                            escape: format!("\\{digits}"),
                            octal: true,
                        });
                    }
                    text.push(char::from_u32(if bytes { point % 256 } else { point })?);
                }
                other => {
                    if notice.is_none() {
                        notice = Some(EscapeNotice {
                            escape: format!("\\{other}"),
                            octal: false,
                        });
                    }
                    text.push('\\');
                    text.push(other);
                }
            }
        }
        if let Some(notice) = notice {
            self.notices.push(notice);
        }
        Some(Datum {
            text: if scalar && !bytes { Some(text) } else { None },
            kind: if bytes { Kind::Bytes } else { Kind::Text },
            hashable: true,
            valid: true,
            signed: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(value: f64) -> String {
        format!("{value:?}")
    }

    #[test]
    fn unicode_names_and_aliases_use_the_actual_native_database_profile() {
        for profile in PROFILES {
            assert_eq!(named_character("digit four", profile), Some('4'));
            assert_eq!(named_character("BACKSPACE", profile), Some('\x08'));
            assert_eq!(named_character("KEYCAP DIGIT FOUR", profile), None);
            assert_eq!(
                named_character("CJK UNIFIED IDEOGRAPH-4E00", profile),
                Some('\u{4E00}')
            );
            assert_eq!(
                named_character("CJK UNIFIED IDEOGRAPH-04E00", profile),
                Some('\u{4E00}')
            );
            assert_eq!(named_character("cjk unified ideograph-4E00", profile), None);
            assert_eq!(named_character("CJK UNIFIED IDEOGRAPH-4e00", profile), None);
            assert_eq!(
                named_character("HANGUL SYLLABLE GA", profile),
                Some('\u{AC00}')
            );
            assert_eq!(named_character("hangul syllable ga", profile), None);
            assert_eq!(
                named_character("EM", profile),
                if matches!(profile, "13.0.0" | "14.0.0") {
                    None
                } else {
                    Some('\x19')
                }
            );
            assert_eq!(
                named_character("KAKTOVIK NUMERAL FOUR", profile),
                if matches!(profile, "13.0.0" | "14.0.0") {
                    None
                } else {
                    Some('\u{1D2C4}')
                }
            );
        }
    }

    #[test]
    fn bounded_reader_preserves_primitive_other_and_invalid_data_distinctions() {
        for literal in [
            "[[]]", "[{}]", "[{1}]", "[(1,)]", "[b'4']", "[4+0j]", "[set()]", "[...]",
        ] {
            assert!(
                matches!(read(literal, "15.0.0", shown).name, Some(Name::Different)),
                "{literal}"
            );
        }
        for literal in [
            "[{[]:1}]", "[{[]}]", "[4+1]", "[04]", "[4,,]", "[+True]", "[--4]",
        ] {
            assert!(read(literal, "15.0.0", shown).name.is_none(), "{literal}");
        }
        for literal in [
            "[4,]",
            "[(4),]",
            "[0x4,]",
            r"['\x34',]",
            r"['\N{DIGIT FOUR}']",
            "[r'4']",
            "['' '4']",
        ] {
            assert!(
                matches!(read(literal,"15.0.0",shown).name,Some(Name::Primitive(v)) if v=="4"),
                "{literal}"
            );
        }
        let unicode = format!("['{}']", "α".repeat(1020));
        assert_eq!(unicode.chars().count(), 1024);
        assert!(matches!(
            read(&unicode, "15.0.0", shown).name,
            Some(Name::Primitive(_))
        ));
        assert!(read(&(unicode + " "), "15.0.0", shown).name.is_none());
    }
}
