//! Simplest possible concrete implementation of `pink-lemonade-progressive-core`.
//!
//! Stores all points in a single flat object (`points.bin`, 24 bytes per point).
//! On query, downloads the entire file (caching it only if it fits within
//! `ClientConfig::memory_budget_bytes`) and returns unmerged unit kernels for all
//! visible points.

use std::sync::Arc;

use pink_lemonade_progressive_core::{
    BuildError, ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, Frame, ObjectWriter,
    Point, ProgressiveClient, ProgressiveStrategy, StoreError, ViewQuery, async_trait,
};

const POINTS_KEY: &str = "points.bin";
const POINT_BYTES: usize = 24;

/// Baseline strategy: single raw file, full download, no downsampling.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveStrategy;

#[derive(Default)]
pub struct NaiveBuilder {
    points: Vec<Point>,
}

impl DatasetBuilder for NaiveBuilder {
    fn push_chunk(&mut self, points: &[Point]) -> Result<(), BuildError> {
        self.points.extend_from_slice(points);
        Ok(())
    }

    fn finish(self: Box<Self>, writer: &mut dyn ObjectWriter) -> Result<(), BuildError> {
        let mut sink = writer.create_object(POINTS_KEY)?;
        for p in &self.points {
            sink.write_all(&p.x.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.y.to_le_bytes()).map_err(StoreError::from)?;
            sink.write_all(&p.weight.to_le_bytes())
                .map_err(StoreError::from)?;
        }
        sink.flush().map_err(StoreError::from)?;
        Ok(())
    }
}

pub struct NaiveClient {
    store: Arc<dyn ByteRangeStore>,
    total_bytes: u64,
    memory_budget_bytes: usize,
    cached_points: Option<Vec<Point>>,
}

#[async_trait]
impl ProgressiveClient for NaiveClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        if self.total_bytes == 0 {
            return Ok(Frame::empty());
        }

        let points: Vec<Point> = if let Some(cached) = &self.cached_points {
            cached.clone()
        } else {
            let len = usize::try_from(self.total_bytes)
                .map_err(|_| ClientError::CorruptData("points.bin exceeds usize".to_owned()))?;
            let raw = self.store.read_range(POINTS_KEY, 0, len).await?;
            let decoded: Vec<Point> = raw
                .as_chunks::<POINT_BYTES>()
                .0
                .iter()
                .map(|c| {
                    Point::new(
                        f64::from_le_bytes(c[0..8].try_into().unwrap()),
                        f64::from_le_bytes(c[8..16].try_into().unwrap()),
                        f64::from_le_bytes(c[16..24].try_into().unwrap()),
                    )
                })
                .collect();

            if decoded.len() * std::mem::size_of::<Point>() <= self.memory_budget_bytes {
                self.cached_points = Some(decoded.clone());
            }
            decoded
        };

        let mut gpu_instances = Vec::new();
        let mut line_points = Vec::new();
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let inv_two_pi = 1.0 / (2.0 * std::f64::consts::PI);

        for p in &points {
            if p.x < query.min_x || p.x > query.max_x {
                continue;
            }
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            gpu_instances.extend_from_slice(&[
                p.x as f32,
                p.y as f32,
                (p.weight * inv_two_pi) as f32,
                1.0,
                0.0,
                0.0,
                1.0,
            ]);
            line_points.extend_from_slice(&[p.x as f32, p.y as f32]);
        }

        let y_range = if min_y.is_finite() {
            Some([min_y, max_y])
        } else {
            None
        };

        Ok(Frame {
            gpu_instances,
            line_points,
            y_range,
        })
    }

    fn memory_bytes(&self) -> usize {
        self.cached_points
            .as_ref()
            .map(|v| v.len() * std::mem::size_of::<Point>())
            .unwrap_or(0)
    }
}

#[async_trait]
impl ProgressiveStrategy for NaiveStrategy {
    fn name(&self) -> &str {
        "naive_full_download"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(NaiveBuilder::default())
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        let total_bytes = store.object_len(POINTS_KEY).await?;
        if !total_bytes.is_multiple_of(POINT_BYTES as u64) {
            return Err(ClientError::CorruptData(format!(
                "{POINTS_KEY} length {total_bytes} is not a multiple of {POINT_BYTES}"
            )));
        }
        Ok(Box::new(NaiveClient {
            store,
            total_bytes,
            memory_budget_bytes: config.memory_budget_bytes,
            cached_points: None,
        }))
    }
}
