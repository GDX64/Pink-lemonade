//! Heap-tracking global allocator for standalone benchmark binaries.
//!
//! Install in a benchmark binary to verify that actual OS heap usage matches a client's
//! self-reported `ProgressiveClient::memory_bytes` and stays within budget:
//!
//! ```ignore
//! use pink_lemonade_progressive_harness::TrackingAllocator;
//!
//! #[global_allocator]
//! static ALLOC: TrackingAllocator = TrackingAllocator::new();
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Snapshot of allocator counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub total_allocated_bytes: usize,
    pub allocation_count: usize,
}

/// Thin [`GlobalAlloc`] wrapper around [`System`] that tracks live and peak heap bytes.
pub struct TrackingAllocator {
    current: AtomicUsize,
    peak: AtomicUsize,
    total_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            current_bytes: self.current.load(Ordering::Relaxed),
            peak_bytes: self.peak.load(Ordering::Relaxed),
            total_allocated_bytes: self.total_allocated.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }

    pub fn current_bytes(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Reset the peak watermark to the current live heap size, returning the previous peak.
    pub fn reset_peak(&self) -> usize {
        let now = self.current.load(Ordering::Relaxed);
        self.peak.swap(now, Ordering::Relaxed)
    }

    pub(crate) fn record_alloc(&self, bytes: usize) {
        let live = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(live, Ordering::Relaxed);
        self.total_allocated.fetch_add(bytes, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_dealloc(&self, bytes: usize) {
        self.current.fetch_sub(bytes, Ordering::Relaxed);
    }
}

impl Default for TrackingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            self.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.record_dealloc(layout.size());
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                self.record_alloc(new_size - old_size);
            } else {
                self.record_dealloc(old_size - new_size);
            }
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_live_peak_and_reset() {
        let alloc = TrackingAllocator::new();
        alloc.record_alloc(1024);
        alloc.record_alloc(2048);
        assert_eq!(alloc.current_bytes(), 3072);
        assert_eq!(alloc.peak_bytes(), 3072);

        alloc.record_dealloc(2048);
        assert_eq!(alloc.current_bytes(), 1024);
        assert_eq!(alloc.peak_bytes(), 3072);

        let old_peak = alloc.reset_peak();
        assert_eq!(old_peak, 3072);
        assert_eq!(alloc.peak_bytes(), 1024);
        assert_eq!(alloc.stats().allocation_count, 2);
    }
}
