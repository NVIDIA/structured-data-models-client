// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The rules the core hands the YAML reader from `sdm-pql`, as the tests
//! of this crate, which cannot name `sdm-pql`, give them: Python's 4,300
//! digits and the proleptic Gregorian calendar of `datetime.date`, and the
//! ISO text the recorder writes a date and a time in. `sdm-core`'s tests
//! read with the core's own (`sdm_core::pql::yaml_rules`).

use sdm_schema::view::yaml::{Date, Rules, Timestamp};

pub const RULES: Rules = Rules {
    most_digits: 4300,
    day_ordinal,
};

fn leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn day_ordinal(year: i32, month: u8, day: u8) -> Option<i64> {
    let lengths = [
        31,
        if leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let month = usize::from(month);
    if day == 0 || day > lengths[month - 1] {
        return None;
    }
    let y = i64::from(year) - 1;
    let before: i64 = lengths[..month - 1].iter().map(|&d| i64::from(d)).sum();
    Some(y * 365 + y / 4 - y / 100 + y / 400 + before + i64::from(day))
}

/// `date.isoformat()`.
#[allow(dead_code, reason = "the rules serve several test files")]
pub fn date_iso(date: &Date) -> String {
    format!("{:04}-{:02}-{:02}", date.year, date.month, date.day)
}

/// `datetime.isoformat()`, as the recorder writes a time.
#[allow(dead_code, reason = "the rules serve several test files")]
pub fn datetime_iso(moment: &Timestamp) -> String {
    let mut text = format!(
        "{}T{:02}:{:02}:{:02}",
        date_iso(&moment.date),
        moment.hour,
        moment.minute,
        moment.second
    );
    if moment.microsecond > 0 {
        text.push_str(&format!(".{:06}", moment.microsecond));
    }
    if let Some(offset) = moment.offset {
        let sign = if offset < 0 { '-' } else { '+' };
        let seconds = offset.unsigned_abs();
        text.push_str(&format!(
            "{sign}{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60
        ));
        if seconds % 60 > 0 {
            text.push_str(&format!(":{:02}", seconds % 60));
        }
    }
    text
}
