// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `datetime.strftime` for the directives pandas' format guess writes, in
//! the C locale.

use super::civil::{self, Wall};

/// A parsed moment and its zone, as `datetime` holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aware {
    pub wall: Wall,
    /// The zone's offset in seconds and its name (`tzname()`), if aware.
    pub tz: Option<(i64, Option<String>)>,
}

fn offset_text(seconds: i64) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let total = seconds.unsigned_abs();
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    if s != 0 {
        format!("{sign}{h:02}{m:02}{s:02}")
    } else {
        format!("{sign}{h:02}{m:02}")
    }
}

/// `value.strftime(format)`; `pads_year` is whether this platform writes
/// `%Y` with four digits below 1000.
pub fn strftime(value: &Aware, format: &str, pads_year: bool) -> String {
    let w = &value.wall;
    let mut out = String::new();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(k) = chars.next() else {
            out.push('%');
            break;
        };
        match k {
            'Y' => {
                if pads_year {
                    out.push_str(&format!("{:04}", w.year));
                } else {
                    out.push_str(&w.year.to_string());
                }
            }
            'y' => out.push_str(&format!("{:02}", w.year.rem_euclid(100))),
            'm' => out.push_str(&format!("{:02}", w.month)),
            'd' => out.push_str(&format!("{:02}", w.day)),
            'H' => out.push_str(&format!("{:02}", w.hour)),
            'I' => {
                let h = if w.hour.is_multiple_of(12) {
                    12
                } else {
                    w.hour % 12
                };
                out.push_str(&format!("{h:02}"));
            }
            'M' => out.push_str(&format!("{:02}", w.minute)),
            'S' => out.push_str(&format!("{:02}", w.second)),
            'f' => out.push_str(&format!("{:06}", w.microsecond())),
            'p' => out.push_str(if w.hour < 12 { "AM" } else { "PM" }),
            'b' => out.push_str(civil::MONTHS_SHORT[(w.month as usize).saturating_sub(1) % 12]),
            'B' => out.push_str(civil::MONTHS_LONG[(w.month as usize).saturating_sub(1) % 12]),
            'a' => out.push_str(civil::DAYS_SHORT[civil::weekday(w.year, w.month, w.day) as usize]),
            'A' => out.push_str(civil::DAYS_LONG[civil::weekday(w.year, w.month, w.day) as usize]),
            'z' => {
                if let Some((offset, _)) = &value.tz {
                    out.push_str(&offset_text(*offset));
                }
            }
            'Z' => {
                if let Some((_, Some(name))) = &value.tz {
                    out.push_str(name);
                }
            }
            '%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other);
            }
        }
    }
    out
}
