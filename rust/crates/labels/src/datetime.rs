// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::date_offset::days_from_civil;

/// `pd.Timestamp(text)` of an ISO 8601 date and time, in nanoseconds.
pub fn parse_timestamp(text: &str) -> Option<i64> {
    let text = text.trim();
    let (date, time) = match text.find(['T', ' ']) {
        Some(at) => (&text[..at], Some(&text[at + 1..])),
        None => (text, None),
    };
    let mut parts = date.splitn(3, '-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month)
        || day == 0
        || day > crate::date_offset::days_in_month(year, month)
    {
        return None;
    }
    let mut nanos: i128 = 0;
    if let Some(time) = time {
        let (clock, fraction) = match time.split_once('.') {
            Some((c, f)) => (c, Some(f)),
            None => (time, None),
        };
        let fields: Vec<&str> = clock.split(':').collect();
        if fields.is_empty() || fields.len() > 3 {
            return None;
        }
        let mut seconds: i128 = 0;
        for (i, scale) in [3_600i128, 60, 1].iter().enumerate() {
            if let Some(field) = fields.get(i) {
                let v: i128 = field.parse().ok()?;
                seconds += v * scale;
            }
        }
        nanos = seconds * 1_000_000_000;
        if let Some(fraction) = fraction {
            if fraction.is_empty()
                || fraction.len() > 9
                || !fraction.chars().all(|c| c.is_ascii_digit())
            {
                return None;
            }
            let padded = format!("{fraction:0<9}");
            nanos += padded.parse::<i128>().ok()?;
        }
    }
    let total = i128::from(days_from_civil(year, month, day)) * 86_400_000_000_000 + nanos;
    i64::try_from(total).ok()
}
