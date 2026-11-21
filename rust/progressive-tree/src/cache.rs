//! Strictly bounded LRU cache for compact 64-byte tree node pages and raw leaf point blocks.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use pink_lemonade_progressive_core::Point;

use crate::node::PackedNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheKey {
    /// A page of up to `fanout` compact 64-byte nodes from `level_{level}.bin`.
    TreePage { level: u16, page: u64 },
    /// A leaf block of up to `leaf_points` raw points from `points.bin`.
    LeafPoints(u64),
}

#[derive(Clone)]
pub enum CacheValue {
    TreePage(Arc<[PackedNode]>),
    LeafPoints(Arc<[Point]>),
}

impl CacheValue {
    fn heap_bytes(&self) -> usize {
        match self {
            Self::TreePage(nodes) => nodes.len() * std::mem::size_of::<PackedNode>(),
            Self::LeafPoints(pts) => pts.len() * std::mem::size_of::<Point>(),
        }
    }
}

struct Entry {
    value: CacheValue,
    bytes: usize,
    tick: u64,
}

/// Byte-bounded LRU cache enforcing `used_bytes <= budget_bytes` at all times.
pub struct LruPageCache {
    budget_bytes: usize,
    used_bytes: usize,
    clock: u64,
    entries: HashMap<CacheKey, Entry>,
    order: BTreeSet<(u64, CacheKey)>,
}

impl LruPageCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            order: BTreeSet::new(),
        }
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn get_tree_page(&mut self, level: usize, page: u64) -> Option<Arc<[PackedNode]>> {
        let key = CacheKey::TreePage {
            level: level as u16,
            page,
        };
        match self.touch(&key)? {
            CacheValue::TreePage(nodes) => Some(nodes),
            CacheValue::LeafPoints(_) => None,
        }
    }

    pub fn get_leaf_points(&mut self, leaf: u64) -> Option<Arc<[Point]>> {
        let key = CacheKey::LeafPoints(leaf);
        match self.touch(&key)? {
            CacheValue::LeafPoints(pts) => Some(pts),
            CacheValue::TreePage(_) => None,
        }
    }

    pub fn insert_tree_page(&mut self, level: usize, page: u64, nodes: Arc<[PackedNode]>) {
        let key = CacheKey::TreePage {
            level: level as u16,
            page,
        };
        self.insert(key, CacheValue::TreePage(nodes));
    }

    pub fn insert_leaf_points(&mut self, leaf: u64, points: Arc<[Point]>) {
        let key = CacheKey::LeafPoints(leaf);
        self.insert(key, CacheValue::LeafPoints(points));
    }

    fn touch(&mut self, key: &CacheKey) -> Option<CacheValue> {
        let entry = self.entries.get_mut(key)?;
        self.order.remove(&(entry.tick, *key));
        self.clock += 1;
        entry.tick = self.clock;
        self.order.insert((entry.tick, *key));
        Some(entry.value.clone())
    }

    fn insert(&mut self, key: CacheKey, value: CacheValue) {
        let bytes = value.heap_bytes() + std::mem::size_of::<(CacheKey, Entry)>();
        if bytes > self.budget_bytes {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.order.remove(&(old.tick, key));
            self.used_bytes -= old.bytes;
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some(&(_, victim_key)) = self.order.first() else {
                break;
            };
            self.order.pop_first();
            if let Some(victim) = self.entries.remove(&victim_key) {
                self.used_bytes -= victim.bytes;
            }
        }
        if self.used_bytes + bytes <= self.budget_bytes {
            self.clock += 1;
            let tick = self.clock;
            self.entries.insert(
                key,
                Entry {
                    value,
                    bytes,
                    tick,
                },
            );
            self.order.insert((tick, key));
            self.used_bytes += bytes;
        }
    }
}
