//! Smoke tests exercising `pink-lemonade-progressive-core` and `pink-lemonade-progressive-harness`
//! using a tiny in-test stub strategy.

use std::sync::Arc;
use std::time::Duration;

use pink_lemonade_progressive_core::{
    BuildError, ByteRangeStore, ClientConfig, ClientError, DatasetBuilder, Frame, ObjectWriter,
    Point, ProgressiveClient, ProgressiveStrategy, StoreError, ViewQuery, async_trait,
};
use pink_lemonade_progressive_harness::{
    BenchmarkStore, ConformanceConfig, DatasetShape, DatasetStream, DiskStore, HarnessConfig,
    MemoryStore, NetworkProfile, SimulatedTransport, Workload, run_benchmark,
    run_conformance_suite,
};

const POINTS_KEY: &str = "points.bin";
const POINT_BYTES: usize = 24;

struct StubStrategy;

#[derive(Default)]
struct StubBuilder {
    points: Vec<Point>,
}

impl DatasetBuilder for StubBuilder {
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

struct StubClient {
    store: Arc<dyn ByteRangeStore>,
    total_bytes: u64,
    memory_budget_bytes: usize,
    cached_points: Option<Vec<Point>>,
}

#[async_trait]
impl ProgressiveClient for StubClient {
    async fn query(&mut self, query: ViewQuery) -> Result<Frame, ClientError> {
        if self.total_bytes == 0 {
            return Ok(Frame::empty());
        }

        let points: Vec<Point> = if let Some(cached) = &self.cached_points {
            cached.clone()
        } else {
            let raw = self
                .store
                .read_range(POINTS_KEY, 0, self.total_bytes as usize)
                .await?;
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
impl ProgressiveStrategy for StubStrategy {
    fn name(&self) -> &str {
        "in_test_stub"
    }

    fn create_builder(&self) -> Box<dyn DatasetBuilder> {
        Box::new(StubBuilder::default())
    }

    async fn open_client(
        &self,
        store: Arc<dyn ByteRangeStore>,
        config: ClientConfig,
    ) -> Result<Box<dyn ProgressiveClient>, ClientError> {
        let total_bytes = store.object_len(POINTS_KEY).await?;
        Ok(Box::new(StubClient {
            store,
            total_bytes,
            memory_budget_bytes: config.memory_budget_bytes,
            cached_points: None,
        }))
    }
}

#[tokio::test(start_paused = true)]
async fn stub_passes_full_conformance_suite() {
    let config = ConformanceConfig {
        point_count: 1_500,
        chunk_size: 256,
        require_line_points: true,
        ..ConformanceConfig::default()
    };
    run_conformance_suite(&StubStrategy, &config)
        .await
        .expect("conformance suite should pass for stub");
}

#[tokio::test(start_paused = true)]
async fn benchmark_harness_runs_all_workloads_in_virtual_time() {
    let shape = DatasetShape::SineWave;
    let point_count = 1_000;
    let ground_truth = DatasetStream::new(shape, point_count, 250).collect_all();
    let stream = DatasetStream::new(shape, point_count, 250);

    let config = HarnessConfig {
        network: NetworkProfile::broadband(),
        client: ClientConfig {
            memory_budget_bytes: 1024,
        },
        nmae_pixel_samples: 32,
        ..HarnessConfig::default()
    };
    let workloads = Workload::standard_suite(
        config.screen_w,
        config.screen_h,
        config.sigma_px,
        config.merge_threshold,
        0.5,
    );
    let mut store = MemoryStore::new();

    let report = run_benchmark(
        &StubStrategy,
        stream,
        &mut store,
        &workloads,
        &config,
        Some(&ground_truth),
    )
    .await
    .expect("benchmark run failed");

    assert_eq!(report.total_points, point_count);
    assert_eq!(report.object_count, 1);
    assert_eq!(report.workloads.len(), 6);

    // y_zoom: auto-y anchor, then relative box-zooms resolved into explicit, nested y-bounds.
    let y_zoom = &report.workloads[5];
    assert_eq!(y_zoom.name, "y_zoom");
    assert!(y_zoom.steps[0].view_y_bounds.is_none());
    let [lo2, hi2] = y_zoom.steps[1].view_y_bounds.expect("2x zoom has y bounds");
    let [lo8, hi8] = y_zoom.steps[2].view_y_bounds.expect("8x zoom has y bounds");
    assert!(lo2 < lo8 && hi8 < hi2, "8x band must nest inside 2x band");
    assert!(((hi2 - lo2) / (hi8 - lo8) - 4.0).abs() < 1e-9);
    assert!(y_zoom.steps.iter().all(|s| s.completed && s.nmae_percent.is_some()));

    let overview = &report.workloads[0];
    assert!(overview.steps[0].completed);
    assert!(overview.steps[0].latency.unwrap() > Duration::ZERO);
    assert!(overview.steps[0].nmae_percent.unwrap() < 0.05);

    let interrupt = &report.workloads[4];
    assert!(
        interrupt.total_transport.cancelled_requests >= 5,
        "expected cancelled requests in rapid_interrupt_pan, got {}",
        interrupt.total_transport.cancelled_requests
    );
    assert!(interrupt.steps.last().unwrap().completed);

    let formatted = report.to_string();
    assert!(formatted.contains("in_test_stub"));
    assert!(formatted.contains("rapid_interrupt_pan"));
}

#[tokio::test(start_paused = true)]
async fn simulated_transport_enforces_concurrency_and_tracks_cancellation_bytes() {
    let mut store = MemoryStore::new();
    store.write_object("blob.bin", &vec![7u8; 100_000]).unwrap();

    let profile = NetworkProfile {
        name: "test-slow",
        rtt: Duration::from_millis(10),
        bandwidth_bytes_per_sec: Some(100_000.0),
        max_concurrency: Some(2),
    };
    let transport = Arc::new(SimulatedTransport::new(store.reader(), profile));

    let read_fut = transport.read_range("blob.bin", 0, 100_000);
    tokio::select! {
        _ = read_fut => panic!("read should not finish before 510 ms"),
        _ = tokio::time::sleep(Duration::from_millis(510)) => {}
    }

    let m = transport.metrics();
    assert_eq!(m.completed_requests, 0);
    assert_eq!(m.cancelled_requests, 1);
    assert!(
        (45_000..=55_000).contains(&m.wasted_cancelled_bytes),
        "expected ~50 KB wasted bytes, got {}",
        m.wasted_cancelled_bytes
    );

    let mut handles = Vec::new();
    for _ in 0..4 {
        let t = Arc::clone(&transport);
        handles.push(tokio::spawn(async move {
            t.read_range("blob.bin", 0, 1_000).await.unwrap()
        }));
    }
    for h in handles {
        assert_eq!(h.await.unwrap().len(), 1_000);
    }

    let m = transport.metrics();
    assert_eq!(m.completed_requests, 4);
    assert_eq!(m.bytes_downloaded, 4_000);
    assert!(m.peak_in_flight <= 2);
}

#[tokio::test]
async fn disk_store_rejects_traversal_and_clamps_ranges() {
    let dir = std::env::temp_dir().join(format!(
        "pink_lemonade_disk_store_test_{}",
        std::process::id()
    ));
    let mut store = DiskStore::open(&dir).unwrap();
    store.write_object("nested/chunk.bin", b"0123456789").unwrap();

    assert_eq!(store.object_len("nested/chunk.bin").await.unwrap(), 10);
    assert_eq!(
        store.read_range("nested/chunk.bin", 6, 100).await.unwrap(),
        b"6789"
    );
    assert!(matches!(
        store.read_range("nested/chunk.bin", 11, 4).await,
        Err(StoreError::OutOfRange { .. })
    ));
    assert!(matches!(
        store.read_range("../escape.bin", 0, 4).await,
        Err(StoreError::InvalidKey(_))
    ));

    store.remove_dir_all().unwrap();
}
