// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guarded object-index quicksort with terminal comparison outcomes.

/// Pure callers drive the same suspendable algorithm with immediately ready comparisons.
pub(super) fn argsort<E>(
    num: usize,
    mut less: impl FnMut(usize, usize) -> Result<bool, E>,
) -> Result<Vec<usize>, E> {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    let future = argsort_async(num, async |left, right| less(left, right));
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(result) => result,
        Poll::Pending => unreachable!("pure comparison cannot suspend"),
    }
}

pub(super) async fn argsort_async<E>(
    num: usize,
    mut less: impl AsyncFnMut(usize, usize) -> Result<bool, E>,
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
            heap(&mut t[pl..=pr], &mut less).await?;
        } else {
            while pr - pl > SMALL_QUICKSORT - 1 {
                let pm = pl + ((pr - pl) >> 1);
                if less(t[pm], t[pl]).await? {
                    t.swap(pm, pl);
                }
                if less(t[pr], t[pm]).await? {
                    t.swap(pr, pm);
                }
                if less(t[pm], t[pl]).await? {
                    t.swap(pm, pl);
                }
                let vp = t[pm];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    pi += 1;
                    while less(t[pi], vp).await? && pi < pj {
                        pi += 1;
                    }
                    pj -= 1;
                    while less(vp, t[pj]).await? && pi < pj {
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
                while pj > pl && less(vp, t[pj - 1]).await? {
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
async fn heap<E>(
    a: &mut [usize],
    less: &mut impl AsyncFnMut(usize, usize) -> Result<bool, E>,
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
            if j < n && less(at(a, j), at(a, j + 1)).await? {
                j += 1;
            }
            if less(tmp, at(a, j)).await? {
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
            if j < n && less(at(a, j), at(a, j + 1)).await? {
                j += 1;
            }
            if less(tmp, at(a, j)).await? {
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
