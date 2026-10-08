// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The escapes of a string literal, read as the reference reads them, by
//! Python's `unicode-escape` codec over the literal's text. Which escapes
//! read, and which make a literal unreadable, decides validity, so this is
//! one of the modules allowed to emulate Python
//! (`scripts/rust_core/emulation-allowlist.txt`, Phase 1 decisions, E9).

/// The text a string literal stands for, its escapes read: the grammar's
/// (`\\ \" \' \/ \b \f \n \r \t \uXXXX`) and the octal, `\xXX`,
/// `\UXXXXXXXX`, `\a`, `\v` and line-continuation escapes the reference
/// also reads. An escape it cannot finish, such as `\u00`, makes the literal
/// unreadable; an unknown one, such as `\/`, stands for itself.
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
            '\n' => {}
            '\\' | '\'' | '"' => units.push(next as u32),
            'b' => units.push(0x08),
            'f' => units.push(0x0c),
            't' => units.push(0x09),
            'n' => units.push(0x0a),
            'r' => units.push(0x0d),
            'v' => units.push(0x0b),
            'a' => units.push(0x07),
            '0'..='7' => {
                let mut value = next as u32 - '0' as u32;
                for _ in 0..2 {
                    match chars.get(i) {
                        Some(&d @ '0'..='7') => {
                            value = value * 8 + (d as u32 - '0' as u32);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                units.push(value);
            }
            'x' => {
                units.push(hex(&chars[i..], 2, "\\xXX")?);
                i += 2;
            }
            'u' => {
                units.push(hex(&chars[i..], 4, "\\uXXXX")?);
                i += 4;
            }
            'U' => {
                let value = hex(&chars[i..], 8, "\\UXXXXXXXX")?;
                if value > 0x10FFFF {
                    return Err("illegal Unicode character".to_string());
                }
                units.push(value);
                i += 8;
            }
            'N' => return Err("malformed \\N character escape".to_string()),
            other if (other as u32) > 0xFF => {
                // The character after the backslash was itself written as an
                // escape by `backslashreplace`, whose backslash this one
                // escapes.
                units.push('\\' as u32);
                let code = other as u32;
                let written = if code > 0xFFFF {
                    format!("U{code:08x}")
                } else {
                    format!("u{code:04x}")
                };
                units.extend(written.chars().map(|c| c as u32));
            }
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
        assert_eq!(unescape("\\€").unwrap(), "\\u20ac");
        assert!(unescape("a\\").is_err());
        assert!(unescape("\\u00").is_err());
    }
}
