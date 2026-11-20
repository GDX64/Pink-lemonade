//! Runtime-agnostic concurrent future join and contiguous byte-range coalescing.
//!
//! Avoids pulling in any async runtime or `futures` dependency so `progressive-tree`
//! compiles cleanly on both native Tokio and `wasm32-unknown-unknown`.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Coalesce sorted/unsorted page or leaf indices into contiguous `(start, count)` runs,
/// each capped at `max_run` items so large spans still parallelize across concurrent streams.
pub fn coalesce_runs(mut indices: Vec<u64>, max_run: usize) -> Vec<(u64, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }
    let max_run = max_run.max(1);
    indices.sort_unstable();
    indices.dedup();

    let mut runs = Vec::new();
    let mut start = indices[0];
    let mut count = 1usize;

    for &idx in &indices[1..] {
        if idx == start + count as u64 && count < max_run {
            count += 1;
        } else {
            runs.push((start, count));
            start = idx;
            count = 1;
        }
    }
    runs.push((start, count));
    runs
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
