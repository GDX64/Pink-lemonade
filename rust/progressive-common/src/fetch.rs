//! Runtime-agnostic concurrent future join and gap-bridging byte-range coalescing.
//!
//! Avoids pulling in any async runtime or `futures` dependency so strategies built on
//! `progressive-common` compile cleanly on both native Tokio and `wasm32-unknown-unknown`.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Coalesce page or leaf indices into `(start, count)` runs, bridging small gaps of up to
/// `max_gap` unrequested items when doing so avoids an extra network RTT, while capping
/// each merged run at `max_run` items for parallelism and prompt cancellation.
pub fn coalesce_runs(mut indices: Vec<u64>, max_gap: u64, max_run: usize) -> Vec<(u64, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }
    let max_run = max_run.max(1);
    indices.sort_unstable();
    indices.dedup();

    let mut runs = Vec::new();
    let mut start = indices[0];
    let mut end = start + 1; // exclusive

    for &idx in &indices[1..] {
        let gap = idx - end;
        let merged_len = (idx - start + 1) as usize;
        if gap <= max_gap && merged_len <= max_run {
            end = idx + 1;
        } else {
            runs.push((start, (end - start) as usize));
            start = idx;
            end = idx + 1;
        }
    }
    runs.push((start, (end - start) as usize));
    runs
}

/// Coalesce disjoint variable-length `(start, len)` spans into `(start, end)` reads,
/// bridging gaps of up to `max_gap` units and capping merged reads at `max_run` units
/// (a single span longer than `max_run` is still read alone, in one piece).
pub fn coalesce_spans(mut spans: Vec<(u64, u64)>, max_gap: u64, max_run: u64) -> Vec<(u64, u64)> {
    spans.retain(|&(_, len)| len > 0);
    if spans.is_empty() {
        return Vec::new();
    }
    spans.sort_unstable();
    spans.dedup();
    let mut reads = Vec::new();
    let (mut start, mut end) = (spans[0].0, spans[0].0 + spans[0].1);
    for &(s, len) in &spans[1..] {
        let e = s + len;
        if s <= end.saturating_add(max_gap) && e.max(end) - start <= max_run {
            end = end.max(e);
        } else {
            reads.push((start, end));
            (start, end) = (s, e);
        }
    }
    reads.push((start, end));
    reads
}

/// Poll all `futures` concurrently on the caller's executor, returning all outputs in order
/// or the first error (dropping all remaining in-flight futures immediately).
pub fn try_join_all<F, T, E>(futures: Vec<F>) -> TryJoinAll<F, T>
where
    F: Future<Output = Result<T, E>>,
{
    let items = futures
        .into_iter()
        .map(|f| Slot::Pending(Box::pin(f)))
        .collect();
    TryJoinAll { items }
}

enum Slot<F, T> {
    Pending(Pin<Box<F>>),
    Done(T),
    Taken,
}

pub struct TryJoinAll<F, T> {
    items: Vec<Slot<F, T>>,
}

impl<F, T, E> Future for TryJoinAll<F, T>
where
    F: Future<Output = Result<T, E>>,
{
    type Output = Result<Vec<T>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let mut all_done = true;

        for slot in &mut this.items {
            if let Slot::Pending(fut) = slot {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(val)) => *slot = Slot::Done(val),
                    Poll::Ready(Err(err)) => {
                        this.items.clear();
                        return Poll::Ready(Err(err));
                    }
                    Poll::Pending => all_done = false,
                }
            }
        }

        if all_done {
            let mut out = Vec::with_capacity(this.items.len());
            for slot in &mut this.items {
                match std::mem::replace(slot, Slot::Taken) {
                    Slot::Done(val) => out.push(val),
                    _ => unreachable!(),
                }
            }
            Poll::Ready(Ok(out))
        } else {
            Poll::Pending
        }
    }
}

/// Poll two fallible futures concurrently, returning both outputs or the first error
/// (dropping the other future immediately).
pub fn join2<A, B, TA, TB, E>(a: A, b: B) -> Join2<A, B, TA, TB>
where
    A: Future<Output = Result<TA, E>>,
    B: Future<Output = Result<TB, E>>,
{
    Join2 {
        a: Slot::Pending(Box::pin(a)),
        b: Slot::Pending(Box::pin(b)),
    }
}

pub struct Join2<A, B, TA, TB> {
    a: Slot<A, TA>,
    b: Slot<B, TB>,
}

impl<A, B, TA, TB, E> Future for Join2<A, B, TA, TB>
where
    A: Future<Output = Result<TA, E>>,
    B: Future<Output = Result<TB, E>>,
{
    type Output = Result<(TA, TB), E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Both slots hold `Pin<Box<_>>` futures, so moving `Join2` never moves them.
        let this = unsafe { self.get_unchecked_mut() };
        if let Slot::Pending(fut) = &mut this.a {
            match fut.as_mut().poll(cx) {
                Poll::Ready(Ok(val)) => this.a = Slot::Done(val),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => {}
            }
        }
        if let Slot::Pending(fut) = &mut this.b {
            match fut.as_mut().poll(cx) {
                Poll::Ready(Ok(val)) => this.b = Slot::Done(val),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => {}
            }
        }
        if matches!(this.a, Slot::Done(_)) && matches!(this.b, Slot::Done(_)) {
            let (Slot::Done(a), Slot::Done(b)) = (
                std::mem::replace(&mut this.a, Slot::Taken),
                std::mem::replace(&mut this.b, Slot::Taken),
            ) else {
                unreachable!()
            };
            Poll::Ready(Ok((a, b)))
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_spans_bridges_gaps_and_caps_runs() {
        assert_eq!(coalesce_spans(vec![(10, 2), (0, 4), (5, 1)], 1, 100), vec![(0, 6), (10, 12)]);
        assert_eq!(coalesce_spans(vec![(0, 4), (5, 1)], 0, 100), vec![(0, 4), (5, 6)]);
        assert_eq!(coalesce_spans(vec![(0, 4), (4, 4), (8, 4)], 0, 8), vec![(0, 8), (8, 12)]);
        assert_eq!(coalesce_spans(vec![(0, 50), (50, 1)], 0, 8), vec![(0, 50), (50, 51)]);
        assert!(coalesce_spans(vec![(3, 0)], 0, 8).is_empty());
    }
}
