// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The escapes of a string literal, read as the reference reads them, by
//! Python's `unicode-escape` codec over the literal's text. Which escapes
//! read, and which make a literal unreadable, decides validity, so this is
//! one of the modules allowed to emulate Python
//! (`scripts/rust_core/emulation-allowlist.txt`, Phase 1 decisions, E9).

/// The text a string literal stands for, its escapes read. Only the
/// grammar's escapes reach it (`\\ \" \' \/ \b \f \n \r \t \uXXXX`): the
/// visitor refuses every other before a literal is read, so the codec's
/// octal, `\x`, `\U`, `\N`, `\a`, `\v` and line-continuation escapes are not
/// ported. An escape it cannot finish, such as `\u00`, makes the literal
/// unreadable; `\/`, which the codec does not know, stands for itself.
///
/// A `\u` escape of a surrogate stands for half a character: a pair is
/// joined into the character it encodes, and a lone one becomes U+FFFD.
pub fn unescape(text: &str) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut units: Vec<u32> = Vec::with_capacity(chars.len());
    let mut i = 0;
    let hex = |slice: &[char], n: usize, what: &str| -> Result<u32, String> {
        if slice.len() < n || !slice[..n].iter().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("truncated {what} escape"));
        }
        let digits: String = slice[..n].iter().collect();
        u32::from_str_radix(&digits, 16).map_err(|e| e.to_string())
    };
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' {
            units.push(c as u32);
            i += 1;
            continue;
        }
        let Some(&next) = chars.get(i + 1) else {
            return Err("\\ at end of string".to_string());
        };
        i += 2;
        match next {
            '\\' | '\'' | '"' => units.push(next as u32),
            'b' => units.push(0x08),
            'f' => units.push(0x0c),
            't' => units.push(0x09),
            'n' => units.push(0x0a),
            'r' => units.push(0x0d),
            'u' => {
                units.push(hex(&chars[i..], 4, "\\uXXXX")?);
                i += 4;
            }
            // `\/`, which the codec does not know, stands for itself.
            other => {
                units.push('\\' as u32);
                units.push(other as u32);
            }
        }
    }
    let mut out = String::with_capacity(units.len());
    let mut k = 0;
    while k < units.len() {
        let u = units[k];
        if (0xD800..0xDC00).contains(&u)
            && let Some(&low) = units.get(k + 1)
            && (0xDC00..0xE000).contains(&low)
        {
            let joined = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00);
            out.push(char::from_u32(joined).unwrap_or('\u{fffd}'));
            k += 2;
            continue;
        }
        out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
        k += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_read_as_the_reference_reads_them() {
        assert_eq!(unescape("a\\\\b").unwrap(), "a\\b");
        assert_eq!(unescape("x\\u0041").unwrap(), "xA");
        assert_eq!(unescape("\\/").unwrap(), "\\/");
        assert_eq!(unescape("é\\u00e9").unwrap(), "éé");
        assert_eq!(unescape("\\ud83d\\ude00\\ud83d").unwrap(), "😀\u{fffd}");
        assert_eq!(
            unescape("\\\"\\'\\b\\f\\n\\r\\t").unwrap(),
            "\"'\u{8}\u{c}\n\r\t"
        );
        assert!(unescape("a\\").is_err());
        assert!(unescape("\\u00").is_err());
    }

    #[test]
    fn no_other_escape_reaches_the_reader() {
        // The codec's other escapes are refused by the visitor before a
        // literal is read, which is why they are not ported.
        for escape in [
            "\\x41",
            "\\101",
            "\\a",
            "\\v",
            "\\U0001F600",
            "\\N{BULLET}",
            "\\€",
            "\\\n",
        ] {
            let query =
                format!("PREDICT COUNT(t.* WHERE t.s = 'a{escape}', 0, 3, days) FOR EACH u.id");
            let parsed = crate::parse(
                &query,
                crate::Mode::Inference,
                crate::dates::Reader {
                    year: 2026,
                    pads_year: false,
                },
            )
            .outcome;
            let Err(refused) = parsed else {
                panic!("{escape:?} reached the reader");
            };
            assert_eq!(
                refused.errors[0].code,
                crate::codes::PARSE_SYNTAX,
                "{escape:?}"
            );
        }
    }
}
