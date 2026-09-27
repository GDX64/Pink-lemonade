use std::hint::black_box;
use std::time::{Duration, Instant};

pub struct Mybench {
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

pub struct Summary {
    pub mean: Duration,
    pub ci95_low: Duration,
    pub ci95_high: Duration,
    pub measured_runs: usize,
}

impl Mybench {
    pub fn run<T>(&self, mut operation: impl FnMut() -> T) -> Summary {
        assert!(self.measured_runs >= 2, "a confidence interval needs at least two runs");
        for _ in 0..self.warmup_runs {
            black_box(operation());
        }

        let mut seconds = Vec::with_capacity(self.measured_runs);
        for _ in 0..self.measured_runs {
            let start = Instant::now();
            let result = operation();
            let elapsed = start.elapsed();
            black_box(result);
            seconds.push(elapsed.as_secs_f64());
        }

        let n = seconds.len() as f64;
        let mean = seconds.iter().sum::<f64>() / n;
        let sample_variance = seconds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let margin = t_critical_95(self.measured_runs - 1) * (sample_variance / n).sqrt();
        Summary {
            mean: Duration::from_secs_f64(mean),
            ci95_low: Duration::from_secs_f64((mean - margin).max(0.0)),
            ci95_high: Duration::from_secs_f64(mean + margin),
            measured_runs: self.measured_runs,
        }
    }
}

// Two-sided Student t critical values for a 95% interval; the benchmark uses
// ten measured runs. Keep the supported sample counts explicit.
fn t_critical_95(degrees_of_freedom: usize) -> f64 {
    const VALUES: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    ];
    *VALUES.get(degrees_of_freedom - 1).expect("Mybench supports 2 to 31 measured runs")
}
