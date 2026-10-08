// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time zones as Python's `zoneinfo` reads them, for the `%Z` directive
//! pandas' `strptime` takes a zone name with: the system database's
//! transitions, the rule its footer gives for later years, and the offset
//! before the first transition.
//!
//! The database is the first directory of `zoneinfo.TZPATH` that exists,
//! as `zoneinfo` searches it; a name is opened as written, so a file system
//! that ignores case finds `est` as Python does there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const TZPATH: [&str; 4] = [
    "/usr/share/zoneinfo",
    "/usr/lib/zoneinfo",
    "/usr/share/lib/zoneinfo",
    "/etc/zoneinfo",
];

/// A day a rule switches on: `Mm.w.d`, `Jn` or `n`.
#[derive(Clone, Copy, Debug)]
enum RuleDay {
    /// Month, week (1-5, 5 the last) and weekday (0 Sunday).
    MonthWeekDay(u32, u32, u32),
    /// Day of year 1-365, never counting February 29.
    Julian(u32),
    /// Day of year 0-365, counting February 29.
    Zero(u32),
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    std: i64,
    dst: i64,
    start: (RuleDay, i64),
    end: (RuleDay, i64),
}

/// One zone.
#[derive(Clone, Debug)]
pub struct Zone {
    /// UTC instants of transitions, in seconds, and the offset from each on.
    transitions: Vec<(i64, i64)>,
    before: i64,
    /// The offset after the last transition, when no rule follows.
    after: i64,
    rule: Option<Rule>,
}

fn zones() -> &'static Mutex<HashMap<String, Option<Zone>>> {
    static ZONES: OnceLock<Mutex<HashMap<String, Option<Zone>>>> = OnceLock::new();
    ZONES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The zone `name`, opened as `zoneinfo.ZoneInfo(name)` opens it; `None`
/// where none is found.
pub fn zone(name: &str) -> Option<Zone> {
    let mut cache = zones().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(found) = cache.get(name) {
        return found.clone();
    }
    let found = load(name);
    cache.insert(name.to_string(), found.clone());
    found
}

fn load(name: &str) -> Option<Zone> {
    if name.is_empty()
        || name.starts_with('/')
        || name.split('/').any(|p| p == ".." || p.is_empty())
    {
        return None;
    }
    for root in TZPATH {
        let path = Path::new(root).join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            return parse_tzif(&bytes);
        }
    }
    None
}

/// Every zone `zoneinfo.available_timezones()` lists.
pub fn available() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names = std::collections::BTreeSet::new();
        for root in TZPATH {
            let root = PathBuf::from(root);
            if !root.exists() {
                continue;
            }
            walk(&root, &root, &mut names);
        }
        names.remove("posixrules");
        names.into_iter().collect()
    })
}

fn walk(root: &Path, dir: &Path, names: &mut std::collections::BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let rel = match path.strip_prefix(root) {
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        if dir == root && (rel == "right" || rel == "posix") {
            continue;
        }
        let is_dir = kind.is_dir() || (kind.is_symlink() && path.is_dir());
        if is_dir {
            walk(root, &path, names);
            continue;
        }
        if names.contains(&rel) {
            continue;
        }
        if let Ok(mut file) = std::fs::File::open(&path) {
            use std::io::Read;
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() && &magic == b"TZif" {
                names.insert(rel);
            }
        }
    }
}

fn be_i32(b: &[u8]) -> i64 {
    i64::from(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn be_i64(b: &[u8]) -> i64 {
    i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn parse_tzif(bytes: &[u8]) -> Option<Zone> {
    if bytes.len() < 44 || &bytes[..4] != b"TZif" {
        return None;
    }
    let version = bytes[4];
    let header = |at: usize| -> Option<[usize; 6]> {
        let h = bytes.get(at + 20..at + 44)?;
        let mut out = [0usize; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot =
                u32::from_be_bytes([h[i * 4], h[i * 4 + 1], h[i * 4 + 2], h[i * 4 + 3]]) as usize;
        }
        Some(out)
    };
    // isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt
    let [isut, isstd, leap, timecnt, typecnt, charcnt] = header(0)?;
    let v1_len = timecnt * 5 + typecnt * 6 + charcnt + leap * 8 + isstd + isut;
    let (start, time_size, counts) = if version >= b'2' {
        let at = 44 + v1_len;
        (at + 44, 8, header(at)?)
    } else {
        (44, 4, [isut, isstd, leap, timecnt, typecnt, charcnt])
    };
    let [isut, isstd, leap, timecnt, typecnt, charcnt] = counts;
    let mut p = start;
    let mut times = Vec::with_capacity(timecnt);
    for _ in 0..timecnt {
        let b = bytes.get(p..p + time_size)?;
        times.push(if time_size == 8 { be_i64(b) } else { be_i32(b) });
        p += time_size;
    }
    let idx = bytes.get(p..p + timecnt)?.to_vec();
    p += timecnt;
    let mut types = Vec::with_capacity(typecnt);
    for _ in 0..typecnt {
        let b = bytes.get(p..p + 6)?;
        types.push((be_i32(b), b[4] != 0));
        p += 6;
    }
    p += charcnt + leap * (time_size + 4) + isstd + isut;
    let footer = if version >= b'2' {
        let rest = bytes.get(p..)?;
        let text = std::str::from_utf8(rest).ok()?;
        let text = text.trim_start_matches('\n');
        text.split('\n').next().map(str::to_string)
    } else {
        None
    };
    if types.is_empty() {
        return None;
    }
    let transitions: Vec<(i64, i64)> = times
        .iter()
        .zip(&idx)
        .map(|(&t, &i)| (t, types.get(i as usize).map_or(0, |ty| ty.0)))
        .collect();
    let before = types
        .iter()
        .find(|(_, isdst)| !isdst)
        .map(|ty| ty.0)
        .or_else(|| {
            idx.first()
                .and_then(|&i| types.get(i as usize))
                .map(|ty| ty.0)
        })
        .unwrap_or(0);
    let after_fixed = idx
        .last()
        .and_then(|&i| types.get(i as usize))
        .map_or(types[types.len() - 1].0, |ty| ty.0);
    let rule = footer.as_deref().and_then(parse_posix);
    let after = match &footer {
        Some(text) if !text.is_empty() => match parse_posix_std(text) {
            Some(std) => std,
            None => after_fixed,
        },
        _ => after_fixed,
    };
    Some(Zone {
        transitions,
        before,
        after,
        rule,
    })
}

/// `<+05>-5` or `EST5`: the name and the offset, sign reversed.
fn posix_name_offset(text: &str) -> Option<(i64, &str)> {
    let rest = if let Some(stripped) = text.strip_prefix('<') {
        let end = stripped.find('>')?;
        &stripped[end + 1..]
    } else {
        let end = text
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(text.len());
        if end < 3 {
            return None;
        }
        &text[end..]
    };
    let (seconds, used) = posix_time(rest)?;
    Some((-seconds, &rest[used..]))
}

/// `[+-]hh[:mm[:ss]]`, as seconds, and how much of `text` it took.
fn posix_time(text: &str) -> Option<(i64, usize)> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut sign = 1;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let mut parts = [0i64; 3];
    for (k, part) in parts.iter_mut().enumerate() {
        if k > 0 {
            if i < bytes.len() && bytes[i] == b':' {
                i += 1;
            } else {
                break;
            }
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
        *part = text[start..i].parse().ok()?;
    }
    Some((sign * (parts[0] * 3600 + parts[1] * 60 + parts[2]), i))
}

fn parse_posix_std(text: &str) -> Option<i64> {
    posix_name_offset(text).map(|(offset, _)| offset)
}

fn parse_posix(text: &str) -> Option<Rule> {
    let (std, rest) = posix_name_offset(text)?;
    if rest.is_empty() {
        return None;
    }
    // The DST name, then an optional offset (default std + 1 hour).
    let (name_end, after_name) = if let Some(stripped) = rest.strip_prefix('<') {
        let end = stripped.find('>')?;
        (end + 2, &rest[end + 2..])
    } else {
        let end = rest
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(rest.len());
        (end, &rest[end..])
    };
    if name_end < 3 {
        return None;
    }
    let (dst, rules) = if after_name.starts_with(',') || after_name.is_empty() {
        (std + 3600, after_name)
    } else {
        let (seconds, used) = posix_time(after_name)?;
        (-seconds, &after_name[used..])
    };
    let rules = rules.strip_prefix(',')?;
    let (start, end) = rules.split_once(',')?;
    Some(Rule {
        std,
        dst,
        start: posix_rule_day(start)?,
        end: posix_rule_day(end)?,
    })
}

fn posix_rule_day(text: &str) -> Option<(RuleDay, i64)> {
    let (day, time) = match text.split_once('/') {
        Some((day, time)) => (day, posix_time(time)?.0),
        None => (text, 7200),
    };
    let rule = if let Some(rest) = day.strip_prefix('M') {
        let mut parts = rest.split('.');
        let m = parts.next()?.parse().ok()?;
        let w = parts.next()?.parse().ok()?;
        let d = parts.next()?.parse().ok()?;
        RuleDay::MonthWeekDay(m, w, d)
    } else if let Some(rest) = day.strip_prefix('J') {
        RuleDay::Julian(rest.parse().ok()?)
    } else {
        RuleDay::Zero(day.parse().ok()?)
    };
    Some((rule, time))
}

/// Seconds from the start of `year` (local) to the rule's day at midnight.
fn rule_day_seconds(day: RuleDay, year: i64) -> i64 {
    use super::civil::{days_from_civil, days_in_month, is_leap};
    let jan1 = days_from_civil(year, 1, 1);
    let days = match day {
        RuleDay::MonthWeekDay(m, w, d) => {
            let first = days_from_civil(year, m, 1);
            // 1970-01-01 was a Thursday (weekday 4 counting Sunday as 0).
            let first_wd = (first + 4).rem_euclid(7) as u32;
            let mut dom = 1 + (d + 7 - first_wd) % 7 + (w - 1) * 7;
            let last = days_in_month(year, m);
            while dom > last {
                dom -= 7;
            }
            first + i64::from(dom) - 1 - jan1
        }
        RuleDay::Julian(n) => {
            let n = i64::from(n);
            if is_leap(year) && n >= 60 { n } else { n - 1 }
        }
        RuleDay::Zero(n) => i64::from(n),
    };
    days * 86_400
}

impl Rule {
    /// The offset in force at the UTC instant `t`.
    fn offset_at(&self, t: i64) -> i64 {
        use super::civil::{civil_from_days, days_from_civil};
        let (year, _, _) = civil_from_days((t + self.std).div_euclid(86_400));
        let in_dst = |year: i64| -> bool {
            let base = days_from_civil(year, 1, 1) * 86_400;
            // Start is given in local standard time, end in local DST.
            let start = base + rule_day_seconds(self.start.0, year) + self.start.1 - self.std;
            let end = base + rule_day_seconds(self.end.0, year) + self.end.1 - self.dst;
            if start < end {
                t >= start && t < end
            } else {
                !(t >= end && t < start)
            }
        };
        if in_dst(year) { self.dst } else { self.std }
    }
}

impl Zone {
    /// Whether the zone has one offset, ever (`ZoneInfo._fixed_offset`).
    pub fn is_fixed(&self) -> bool {
        self.rule.is_none()
            && self.transitions.iter().all(|&(_, o)| o == self.before)
            && self.after == self.before
    }

    /// The UTC offset, in seconds, at the UTC instant `t`.
    pub fn offset_at(&self, t: i64) -> i64 {
        match self.transitions.partition_point(|&(at, _)| at <= t) {
            0 => self.before,
            n if n < self.transitions.len() => self.transitions[n - 1].1,
            _ => match &self.rule {
                Some(rule) => rule.offset_at(t),
                None => self.after,
            },
        }
    }

    /// The UTC instant of the wall time `local` (seconds), or `None` where
    /// it is ambiguous or does not exist: pandas' `tz_localize_to_utc` with
    /// `ambiguous='raise'` and `nonexistent='raise'`, which raise there.
    pub fn localize(&self, local: i64) -> Option<i64> {
        let mut offsets = vec![
            self.offset_at(local - 86_400),
            self.offset_at(local + 86_400),
            self.offset_at(local),
        ];
        offsets.sort_unstable();
        offsets.dedup();
        let found: Vec<i64> = offsets
            .into_iter()
            .map(|o| local - o)
            .filter(|&utc| utc + self.offset_at(utc) == local)
            .collect();
        match found.as_slice() {
            [one] => Some(*one),
            [a, b] if a == b => Some(*a),
            _ => None,
        }
    }
}
