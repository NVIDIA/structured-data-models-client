// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guarded object-index quicksort with terminal comparison outcomes.

pub(super) fn argsort<E>(
    num: usize,
    mut less: impl FnMut(usize, usize) -> Result<bool, E>,
) -> Result<Vec<usize>, E> {
    const SMALL_QUICKSORT: usize = 16;
    let mut t: Vec<usize> = (0..num).collect();
    if num < 2 {
        return Ok(t);
    }
    let mut stack: Vec<(usize, usize, i64)> = Vec::new();
    let (mut pl, mut pr) = (0usize, num - 1);
    let mut cdepth = 2 * i64::from(usize::BITS - 1 - num.leading_zeros());
    loop {
        if cdepth < 0 {
            heap(&mut t[pl..=pr], &mut less)?;
        } else {
            while pr - pl > SMALL_QUICKSORT - 1 {
                let pm = pl + ((pr - pl) >> 1);
                if less(t[pm], t[pl])? {
                    t.swap(pm, pl);
                }
                if less(t[pr], t[pm])? {
                    t.swap(pr, pm);
                }
                if less(t[pm], t[pl])? {
                    t.swap(pm, pl);
                }
                let vp = t[pm];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    pi += 1;
                    while less(t[pi], vp)? && pi < pj {
                        pi += 1;
                    }
                    pj -= 1;
                    while less(vp, t[pj])? && pi < pj {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi, pj);
                }
                t.swap(pi, pr - 1);
                cdepth -= 1;
                // The larger part waits on the stack.
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            for pi in pl + 1..=pr {
                let vi = t[pi];
                let vp = vi;
                let mut pj = pi;
                while pj > pl && less(vp, t[pj - 1])? {
                    t[pj] = t[pj - 1];
                    pj -= 1;
                }
                t[pj] = vi;
            }
        }
        let Some((l, r, d)) = stack.pop() else {
            break;
        };
        pl = l;
        pr = r;
        cdepth = d;
    }
    Ok(t)
}

/// The generic object's depth fallback retains its native comparison order.
fn heap<E>(
    a: &mut [usize],
    less: &mut impl FnMut(usize, usize) -> Result<bool, E>,
) -> Result<(), E> {
    let n = a.len();
    if n < 2 {
        return Ok(());
    }
    // One-based, as the producer indexes its heap.
    let at = |a: &[usize], k: usize| a[k - 1];
    let mut l = n >> 1;
    while l > 0 {
        let tmp = at(a, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && less(at(a, j), at(a, j + 1))? {
                j += 1;
            }
            if less(tmp, at(a, j))? {
                a[i - 1] = at(a, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
        l -= 1;
    }
    let mut n = n;
    while n > 1 {
        let tmp = at(a, n);
        a[n - 1] = at(a, 1);
        n -= 1;
        let (mut i, mut j) = (1usize, 2usize);
        while j <= n {
            if j < n && less(at(a, j), at(a, j + 1))? {
                j += 1;
            }
            if less(tmp, at(a, j))? {
                a[i - 1] = at(a, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
    }
    Ok(())
}
