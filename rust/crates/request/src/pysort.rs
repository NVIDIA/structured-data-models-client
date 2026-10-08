// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python's `list.sort(key=..., reverse=True)` over float keys, as CPython
//! 3.12 sorts (`Objects/listobject.c`): stable, so ties keep the order they
//! were given in, and with a NaN key, the order its timsort happens to leave,
//! since every comparison with a NaN is false.
//!
//! A list sorted with `reverse=True` is reversed, sorted ascending, and
//! reversed again. The sort is timsort as 3.12 writes it, step for step:
//! runs found by `count_run` and extended to `minrun` by binary insertion,
//! merged in the order powersort's `found_new_run` and
//! `merge_force_collapse` give, with `merge_lo` and `merge_hi` galloping as
//! `min_gallop` says. Keys that compare totally have one stable order, which
//! any stable sort gives; a NaN makes the comparisons inconsistent, and then
//! only the same comparisons in the same order give the same answer.
//!
//! CPython 3.13 rewrote `count_run`, so a NaN can leave another order there.

/// `MIN_GALLOP`.
const MIN_GALLOP: usize = 7;

/// A pending run: where it starts in the list, how long it is, and its
/// powersort power.
#[derive(Clone, Copy)]
struct Run {
    base: usize,
    len: usize,
    power: u32,
}

/// How a merge ended: with what is left of the run copied aside, or with
/// one element of it left to place after the other run.
enum End {
    Done,
    One,
}

/// `MergeState`, over the positions being sorted.
struct Sort<'a> {
    keys: &'a [f64],
    list: Vec<usize>,
    min_gallop: usize,
    pending: Vec<Run>,
}

impl Sort<'_> {
    /// `ISLT(x, y)` of two positions, by their keys: `float.__lt__`.
    fn lt(&self, x: usize, y: usize) -> bool {
        self.keys[x] < self.keys[y]
    }

    /// `count_run(lo, hi)`: the length of the run at `lo`, and whether it
    /// is strictly descending.
    fn count_run(&self, lo: usize, hi: usize) -> (usize, bool) {
        if lo + 1 == hi {
            return (1, false);
        }
        let mut n = 2;
        let mut at = lo + 1;
        if self.lt(self.list[at], self.list[at - 1]) {
            at += 1;
            while at < hi && self.lt(self.list[at], self.list[at - 1]) {
                at += 1;
                n += 1;
            }
            (n, true)
        } else {
            at += 1;
            while at < hi && !self.lt(self.list[at], self.list[at - 1]) {
                at += 1;
                n += 1;
            }
            (n, false)
        }
    }

    /// `binarysort(lo, hi, start)`: `lo..start` is sorted; insert the rest.
    fn binarysort(&mut self, lo: usize, hi: usize, start: usize) {
        let start = if start == lo { start + 1 } else { start };
        for at in start..hi {
            let pivot = self.list[at];
            let (mut l, mut r) = (lo, at);
            loop {
                let p = l + ((r - l) >> 1);
                if self.lt(pivot, self.list[p]) {
                    r = p;
                } else {
                    l = p + 1;
                }
                if l >= r {
                    break;
                }
            }
            self.list.copy_within(l..at, l + 1);
            self.list[l] = pivot;
        }
    }

    /// `gallop_left(key, a, n, hint)`: where `key` goes in sorted `a`,
    /// leftmost among equals.
    fn gallop_left(&self, key: usize, a: &[usize], hint: usize) -> usize {
        let n = a.len() as isize;
        let hint = hint as isize;
        let at = |i: isize| a[i as usize];
        let mut lastofs: isize = 0;
        let mut ofs: isize = 1;
        if self.lt(at(hint), key) {
            let maxofs = n - hint;
            while ofs < maxofs {
                if self.lt(at(hint + ofs), key) {
                    lastofs = ofs;
                    ofs = (ofs << 1) + 1;
                } else {
                    break;
                }
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            lastofs += hint;
            ofs += hint;
        } else {
            let maxofs = hint + 1;
            while ofs < maxofs {
                if self.lt(at(hint - ofs), key) {
                    break;
                }
                lastofs = ofs;
                ofs = (ofs << 1) + 1;
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            let k = lastofs;
            lastofs = hint - ofs;
            ofs = hint - k;
        }
        lastofs += 1;
        while lastofs < ofs {
            let m = lastofs + ((ofs - lastofs) >> 1);
            if self.lt(at(m), key) {
                lastofs = m + 1;
            } else {
                ofs = m;
            }
        }
        ofs as usize
    }

    /// `gallop_right(key, a, n, hint)`: where `key` goes in sorted `a`,
    /// rightmost among equals.
    fn gallop_right(&self, key: usize, a: &[usize], hint: usize) -> usize {
        let n = a.len() as isize;
        let hint = hint as isize;
        let at = |i: isize| a[i as usize];
        let mut lastofs: isize = 0;
        let mut ofs: isize = 1;
        if self.lt(key, at(hint)) {
            let maxofs = hint + 1;
            while ofs < maxofs {
                if self.lt(key, at(hint - ofs)) {
                    lastofs = ofs;
                    ofs = (ofs << 1) + 1;
                } else {
                    break;
                }
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            let k = lastofs;
            lastofs = hint - ofs;
            ofs = hint - k;
        } else {
            let maxofs = n - hint;
            while ofs < maxofs {
                if self.lt(key, at(hint + ofs)) {
                    break;
                }
                lastofs = ofs;
                ofs = (ofs << 1) + 1;
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            lastofs += hint;
            ofs += hint;
        }
        lastofs += 1;
        while lastofs < ofs {
            let m = lastofs + ((ofs - lastofs) >> 1);
            if self.lt(key, at(m)) {
                ofs = m;
            } else {
                lastofs = m + 1;
            }
        }
        ofs as usize
    }

    /// `merge_lo`: merge the run at `pa` with the one after it, `na <= nb`,
    /// with the first copied aside.
    fn merge_lo(&mut self, pa: usize, mut na: usize, pb: usize, mut nb: usize) {
        let tmp: Vec<usize> = self.list[pa..pa + na].to_vec();
        let (mut dest, mut ia, mut ib) = (pa, 0usize, pb);
        let end = 'merge: {
            self.list[dest] = self.list[ib];
            dest += 1;
            ib += 1;
            nb -= 1;
            if nb == 0 {
                break 'merge End::Done;
            }
            if na == 1 {
                break 'merge End::One;
            }
            let mut min_gallop = self.min_gallop;
            loop {
                let mut acount = 0usize;
                let mut bcount = 0usize;
                loop {
                    if self.lt(self.list[ib], tmp[ia]) {
                        self.list[dest] = self.list[ib];
                        dest += 1;
                        ib += 1;
                        bcount += 1;
                        acount = 0;
                        nb -= 1;
                        if nb == 0 {
                            break 'merge End::Done;
                        }
                        if bcount >= min_gallop {
                            break;
                        }
                    } else {
                        self.list[dest] = tmp[ia];
                        dest += 1;
                        ia += 1;
                        acount += 1;
                        bcount = 0;
                        na -= 1;
                        if na == 1 {
                            break 'merge End::One;
                        }
                        if acount >= min_gallop {
                            break;
                        }
                    }
                }
                min_gallop += 1;
                loop {
                    min_gallop -= usize::from(min_gallop > 1);
                    self.min_gallop = min_gallop;
                    let k = self.gallop_right(self.list[ib], &tmp[ia..ia + na], 0);
                    acount = k;
                    if k > 0 {
                        self.list[dest..dest + k].copy_from_slice(&tmp[ia..ia + k]);
                        dest += k;
                        ia += k;
                        na -= k;
                        if na == 1 {
                            break 'merge End::One;
                        }
                        if na == 0 {
                            break 'merge End::Done;
                        }
                    }
                    self.list[dest] = self.list[ib];
                    dest += 1;
                    ib += 1;
                    nb -= 1;
                    if nb == 0 {
                        break 'merge End::Done;
                    }
                    let k = self.gallop_left(tmp[ia], &self.list[ib..ib + nb], 0);
                    bcount = k;
                    if k > 0 {
                        self.list.copy_within(ib..ib + k, dest);
                        dest += k;
                        ib += k;
                        nb -= k;
                        if nb == 0 {
                            break 'merge End::Done;
                        }
                    }
                    self.list[dest] = tmp[ia];
                    dest += 1;
                    ia += 1;
                    na -= 1;
                    if na == 1 {
                        break 'merge End::One;
                    }
                    if acount < MIN_GALLOP && bcount < MIN_GALLOP {
                        break;
                    }
                }
                min_gallop += 1;
                self.min_gallop = min_gallop;
            }
        };
        match end {
            End::Done => {
                self.list[dest..dest + na].copy_from_slice(&tmp[ia..ia + na]);
            }
            End::One => {
                self.list.copy_within(ib..ib + nb, dest);
                self.list[dest + nb] = tmp[ia];
            }
        }
    }

    /// `merge_hi`: merge the run at `pa` with the one after it, `na > nb`,
    /// from the right, with the second copied aside.
    fn merge_hi(&mut self, pa: usize, mut na: usize, pb: usize, mut nb: usize) {
        let tmp: Vec<usize> = self.list[pb..pb + nb].to_vec();
        // Positions of the last element of each, which can pass below the
        // start of the list as the merge empties a run.
        let mut dest = (pb + nb - 1) as isize;
        let mut ia = (pa + na - 1) as isize;
        let mut ib = (nb - 1) as isize;
        let idx = |i: isize| i as usize;
        let end = 'merge: {
            self.list[idx(dest)] = self.list[idx(ia)];
            dest -= 1;
            ia -= 1;
            na -= 1;
            if na == 0 {
                break 'merge End::Done;
            }
            if nb == 1 {
                break 'merge End::One;
            }
            let mut min_gallop = self.min_gallop;
            loop {
                let mut acount = 0usize;
                let mut bcount = 0usize;
                loop {
                    if self.lt(tmp[idx(ib)], self.list[idx(ia)]) {
                        self.list[idx(dest)] = self.list[idx(ia)];
                        dest -= 1;
                        ia -= 1;
                        acount += 1;
                        bcount = 0;
                        na -= 1;
                        if na == 0 {
                            break 'merge End::Done;
                        }
                        if acount >= min_gallop {
                            break;
                        }
                    } else {
                        self.list[idx(dest)] = tmp[idx(ib)];
                        dest -= 1;
                        ib -= 1;
                        bcount += 1;
                        acount = 0;
                        nb -= 1;
                        if nb == 1 {
                            break 'merge End::One;
                        }
                        if bcount >= min_gallop {
                            break;
                        }
                    }
                }
                min_gallop += 1;
                loop {
                    min_gallop -= usize::from(min_gallop > 1);
                    self.min_gallop = min_gallop;
                    let k = self.gallop_right(tmp[idx(ib)], &self.list[pa..pa + na], na - 1);
                    let k = na - k;
                    acount = k;
                    if k > 0 {
                        dest -= k as isize;
                        ia -= k as isize;
                        self.list
                            .copy_within(idx(ia + 1)..idx(ia + 1) + k, idx(dest + 1));
                        na -= k;
                        if na == 0 {
                            break 'merge End::Done;
                        }
                    }
                    self.list[idx(dest)] = tmp[idx(ib)];
                    dest -= 1;
                    ib -= 1;
                    nb -= 1;
                    if nb == 1 {
                        break 'merge End::One;
                    }
                    let k = self.gallop_left(self.list[idx(ia)], &tmp[..nb], nb - 1);
                    let k = nb - k;
                    bcount = k;
                    if k > 0 {
                        dest -= k as isize;
                        ib -= k as isize;
                        let from = idx(ib + 1);
                        self.list[idx(dest + 1)..idx(dest + 1) + k]
                            .copy_from_slice(&tmp[from..from + k]);
                        nb -= k;
                        if nb == 1 {
                            break 'merge End::One;
                        }
                        if nb == 0 {
                            break 'merge End::Done;
                        }
                    }
                    self.list[idx(dest)] = self.list[idx(ia)];
                    dest -= 1;
                    ia -= 1;
                    na -= 1;
                    if na == 0 {
                        break 'merge End::Done;
                    }
                    if acount < MIN_GALLOP && bcount < MIN_GALLOP {
                        break;
                    }
                }
                min_gallop += 1;
                self.min_gallop = min_gallop;
            }
        };
        match end {
            End::Done => {
                if nb > 0 {
                    let at = idx(dest + 1 - nb as isize);
                    self.list[at..at + nb].copy_from_slice(&tmp[..nb]);
                }
            }
            End::One => {
                let at = idx(dest + 1 - na as isize);
                self.list
                    .copy_within(idx(ia + 1 - na as isize)..idx(ia + 1), at);
                dest -= na as isize;
                self.list[idx(dest)] = tmp[idx(ib)];
            }
        }
    }

    /// `merge_at(i)`: merge pending runs `i` and `i + 1`.
    fn merge_at(&mut self, i: usize) {
        let Run {
            base: mut pa,
            len: mut na,
            ..
        } = self.pending[i];
        let Run {
            base: pb, len: nb, ..
        } = self.pending[i + 1];
        self.pending[i].len = na + nb;
        if i + 3 == self.pending.len() {
            self.pending[i + 1] = self.pending[i + 2];
        }
        self.pending.pop();

        let k = self.gallop_right(self.list[pb], &self.list[pa..pa + na], 0);
        pa += k;
        na -= k;
        if na == 0 {
            return;
        }
        let nb = self.gallop_left(self.list[pa + na - 1], &self.list[pb..pb + nb], nb - 1);
        if nb == 0 {
            return;
        }
        if na <= nb {
            self.merge_lo(pa, na, pb, nb);
        } else {
            self.merge_hi(pa, na, pb, nb);
        }
    }

    /// `found_new_run(n2)`: merge what powersort says to, before a run of
    /// `n2` is pushed.
    fn found_new_run(&mut self, n2: usize) {
        let Some(last) = self.pending.last().copied() else {
            return;
        };
        let power = powerloop(last.base, last.len, n2, self.list.len());
        while self.pending.len() > 1 && self.pending[self.pending.len() - 2].power > power {
            self.merge_at(self.pending.len() - 2);
        }
        if let Some(last) = self.pending.last_mut() {
            last.power = power;
        }
    }

    /// `merge_force_collapse()`.
    fn merge_force_collapse(&mut self) {
        while self.pending.len() > 1 {
            let mut n = self.pending.len() - 2;
            if n > 0 && self.pending[n - 1].len < self.pending[n + 1].len {
                n -= 1;
            }
            self.merge_at(n);
        }
    }
}

/// `powerloop(s1, n1, n2, n)`: the powersort power of the boundary between
/// a run at `s1` of `n1` and the `n2` after it, in a list of `n`.
fn powerloop(s1: usize, n1: usize, n2: usize, n: usize) -> u32 {
    let mut result = 0;
    let mut a = 2 * s1 + n1;
    let mut b = a + n1 + n2;
    loop {
        result += 1;
        if a >= n {
            a -= n;
            b -= n;
        } else if b >= n {
            break;
        }
        a <<= 1;
        b <<= 1;
    }
    result
}

/// `merge_compute_minrun(n)`.
fn minrun(mut n: usize) -> usize {
    let mut r = 0;
    while n >= 64 {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

/// The positions of `keys` in the order `sorted(..., reverse=True)` of a
/// list holding them in this order leaves them in.
pub fn sort_desc(keys: &[f64]) -> Vec<usize> {
    let n = keys.len();
    // The list as it is sorted: reversed first.
    let mut sort = Sort {
        keys,
        list: (0..n).rev().collect(),
        min_gallop: MIN_GALLOP,
        pending: Vec::new(),
    };
    if n >= 2 {
        let minrun = minrun(n);
        let (mut lo, mut remaining) = (0, n);
        while remaining > 0 {
            let (mut run, descending) = sort.count_run(lo, lo + remaining);
            if descending {
                sort.list[lo..lo + run].reverse();
            }
            if run < minrun {
                let force = remaining.min(minrun);
                sort.binarysort(lo, lo + force, lo + run);
                run = force;
            }
            sort.found_new_run(run);
            sort.pending.push(Run {
                base: lo,
                len: run,
                power: 0,
            });
            lo += run;
            remaining -= run;
        }
        sort.merge_force_collapse();
    }
    let mut order = sort.list;
    order.reverse();
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_keep_their_order_and_nans_stay_where_python_leaves_them() {
        assert_eq!(sort_desc(&[0.5, 0.7, 0.5]), [1, 0, 2]);
        assert_eq!(sort_desc(&[f64::NAN, 0.7, 0.2]), [0, 1, 2]);
        assert_eq!(sort_desc(&[0.2, f64::NAN, 0.7]), [0, 1, 2]);
        assert_eq!(sort_desc(&[0.2, 0.9, 0.5, 0.9]), [1, 3, 2, 0]);
    }

    /// `n` keys of a 64-bit LCG: every `nan_every`th draw a NaN, the rest
    /// one of `m` tenths.
    fn generated(n: usize, m: u64, nan_every: u64, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = x >> 33;
                if r.is_multiple_of(nan_every) {
                    f64::NAN
                } else {
                    ((r / nan_every) % m) as f64 / 10.0
                }
            })
            .collect()
    }

    fn digest(order: &[usize]) -> u64 {
        order
            .iter()
            .enumerate()
            .map(|(i, &o)| (i as u64 + 1) * o as u64)
            .sum::<u64>()
            % 1_000_000_007
    }

    /// What `sorted(enumerate(keys), key=lambda p: p[1], reverse=True)`
    /// gives under CPython 3.12 for these keys, by the digest of the
    /// positions: from 64 keys on, where timsort merges runs, and with
    /// enough of them that `merge_lo`, `merge_hi` and galloping all run.
    #[test]
    fn a_nan_among_64_keys_or_more_is_left_where_python_3_12_leaves_it() {
        for (n, m, nan_every, seed, want) in [
            (64, 10, 7, 1, 83116),
            (65, 2, 3, 2, 91307),
            (100, 10, 20, 3, 259327),
            (130, 4, 2, 4, 730359),
            (257, 10, 9, 5, 5625694),
            (700, 6, 5, 6, 113766394),
            (1500, 10, 50, 7, 98947863),
            (3000, 2, 3, 8, 999807269),
            (5000, 10, 4, 9, 661682385),
        ] {
            let keys = generated(n, m, nan_every, seed);
            assert_eq!(digest(&sort_desc(&keys)), want, "{n} keys, seed {seed}");
        }
    }

    #[test]
    fn keys_that_compare_totally_sort_as_a_stable_sort_does() {
        let keys = generated(4000, 7, u64::MAX, 11);
        let mut stable: Vec<usize> = (0..keys.len()).collect();
        stable.sort_by(|&a, &b| keys[b].partial_cmp(&keys[a]).unwrap());
        assert_eq!(sort_desc(&keys), stable);
    }
}
