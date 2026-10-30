// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-bound stable indirect ordering for inert object comparisons.
//! Natural runs, short minimum runs, gallop trimming and collapse order follow
//! the audited generic argsort policy; no native comparator is called here.
use std::cmp::Ordering;

type Run = (usize, usize);

pub(super) fn sort<E>(
    indices: &mut [usize],
    mut compare: impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<(), E> {
    let mut min_run = indices.len();
    let mut remainder = 0;
    while min_run > 16 {
        remainder |= min_run & 1;
        min_run >>= 1;
    }
    min_run += remainder;
    let mut stack: Vec<Run> = Vec::new();
    let mut start = 0;
    while start < indices.len() {
        let length = count_run(indices, start, min_run, &mut compare)?;
        stack.push((start, length));
        loop {
            let top = stack.len();
            if top < 2 {
                break;
            }
            let b = stack[top - 2].1;
            let c = stack[top - 1].1;
            let at = if (top > 2 && stack[top - 3].1 <= b + c)
                || (top > 3 && stack[top - 4].1 <= stack[top - 3].1 + b)
            {
                if stack[top - 3].1 <= c {
                    top - 3
                } else {
                    top - 2
                }
            } else if b <= c {
                top - 2
            } else {
                break;
            };
            merge(indices, &mut stack, at, &mut compare)?;
        }
        start += length;
    }
    while stack.len() > 1 {
        let top = stack.len();
        let at = if top > 2 && stack[top - 3].1 <= stack[top - 1].1 {
            top - 3
        } else {
            top - 2
        };
        merge(indices, &mut stack, at, &mut compare)?;
    }
    Ok(())
}

fn count_run<E>(
    a: &mut [usize],
    start: usize,
    min_run: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<usize, E> {
    if a.len() - start == 1 {
        return Ok(1);
    }
    let ascending = cmp(a[start], a[start + 1])? != Ordering::Greater;
    let mut end = start + 1;
    if ascending {
        while end < a.len() - 1 && cmp(a[end], a[end + 1])? != Ordering::Greater {
            end += 1;
        }
    } else {
        while end < a.len() - 1 && cmp(a[end + 1], a[end])? == Ordering::Less {
            end += 1;
        }
        a[start..=end].reverse();
    }
    end += 1;
    let size = (end - start).max(min_run.min(a.len() - start));
    for at in end..start + size {
        let value = a[at];
        let mut cursor = at;
        while cursor > start && cmp(value, a[cursor - 1])? == Ordering::Less {
            a[cursor] = a[cursor - 1];
            cursor -= 1;
        }
        a[cursor] = value;
    }
    Ok(size)
}

fn gallop_right<E>(
    a: &[usize],
    key: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<usize, E> {
    if cmp(key, a[0])? == Ordering::Less {
        return Ok(0);
    }
    let (mut last, mut offset) = (0, 1);
    while offset < a.len() {
        if cmp(key, a[offset])? == Ordering::Less {
            break;
        }
        last = offset;
        offset = offset.saturating_mul(2).saturating_add(1);
    }
    offset = offset.min(a.len());
    while last + 1 < offset {
        let middle = last + (offset - last) / 2;
        if cmp(key, a[middle])? == Ordering::Less {
            offset = middle;
        } else {
            last = middle;
        }
    }
    Ok(offset)
}

fn gallop_left<E>(
    a: &[usize],
    key: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<usize, E> {
    if cmp(a[a.len() - 1], key)? == Ordering::Less {
        return Ok(a.len());
    }
    let (mut last, mut offset) = (0, 1);
    while offset < a.len() {
        if cmp(a[a.len() - offset - 1], key)? == Ordering::Less {
            break;
        }
        last = offset;
        offset = offset.saturating_mul(2).saturating_add(1);
    }
    offset = offset.min(a.len());
    // Exclusive left bound represents the source's possible -1 sentinel.
    let mut left = a.len() - offset;
    let mut right = a.len() - last - 1;
    while left < right {
        let middle = left + (right - left - 1) / 2;
        if cmp(a[middle], key)? == Ordering::Less {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    Ok(right)
}

fn merge<E>(
    a: &mut [usize],
    stack: &mut Vec<Run>,
    at: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<(), E> {
    let (start, length) = stack[at];
    let (right, right_length) = stack[at + 1];
    let skip = gallop_right(&a[start..start + length], a[right], cmp)?;
    if skip < length {
        let left_start = start + skip;
        let left_length = length - skip;
        let take = gallop_left(&a[right..right + right_length], a[right - 1], cmp)?;
        if take < left_length {
            merge_right(a, left_start, left_length, right, take, cmp)?;
        } else {
            merge_left(a, left_start, left_length, right, take, cmp)?;
        }
    }
    stack[at].1 += right_length;
    stack.remove(at + 1);
    Ok(())
}

fn merge_left<E>(
    a: &mut [usize],
    start: usize,
    length: usize,
    right: usize,
    right_length: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<(), E> {
    let saved = a[start..start + length].to_vec();
    let end = right + right_length;
    let (mut out, mut source, mut held) = (start, right, 0);
    a[out] = a[source];
    out += 1;
    source += 1;
    while out < source && source < end {
        if cmp(a[source], saved[held])? == Ordering::Less {
            a[out] = a[source];
            source += 1;
        } else {
            a[out] = saved[held];
            held += 1;
        }
        out += 1;
    }
    if out != source {
        a[out..source].copy_from_slice(&saved[held..held + source - out]);
    }
    Ok(())
}

fn merge_right<E>(
    a: &mut [usize],
    start: usize,
    length: usize,
    right: usize,
    right_length: usize,
    cmp: &mut impl FnMut(usize, usize) -> Result<Ordering, E>,
) -> Result<(), E> {
    if right_length == 0 {
        return Ok(());
    }
    let saved = a[right..right + right_length].to_vec();
    let (mut source, mut out, mut held) = (start + length, right + right_length, right_length);
    out -= 1;
    source -= 1;
    a[out] = a[source];
    while source < out && start < source {
        out -= 1;
        if cmp(saved[held - 1], a[source - 1])? == Ordering::Less {
            source -= 1;
            a[out] = a[source];
        } else {
            held -= 1;
            a[out] = saved[held];
        }
    }
    if source != out {
        a[start..out].copy_from_slice(&saved[held - (out - start)..held]);
    }
    Ok(())
}
