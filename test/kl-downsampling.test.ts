import { describe, expect, test } from "vitest";
import {
  KLDownsampler,
  createMixture,
  mergePair,
  runnallsCost,
  runnallsReduce,
  type Mixture,
} from "../src/examples/rasterizing/kl-downsampling";

type Comp = {
  x: number;
  y: number;
  w: number;
  p00: number;
  p01: number;
  p11: number;
};

function mixtureOf(comps: Comp[]): Mixture {
  const m = createMixture(comps.length);
  m.count = comps.length;
  comps.forEach((c, i) => {
    m.x[i] = c.x;
    m.y[i] = c.y;
    m.w[i] = c.w;
    m.p00[i] = c.p00;
    m.p01[i] = c.p01;
    m.p11[i] = c.p11;
  });
  return m;
}

function unit(x: number, y: number, w = 1): Comp {
  return { x, y, w, p00: 1, p01: 0, p11: 1 };
}

/** Moments of a whole mixture, computed directly (the invariant to preserve). */
function moments(m: Mixture) {
  let w = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < m.count; i++) {
    w += m.w[i]!;
    mx += m.w[i]! * m.x[i]!;
    my += m.w[i]! * m.y[i]!;
  }
  mx /= w;
  my /= w;

  let p00 = 0;
  let p01 = 0;
  let p11 = 0;
  for (let i = 0; i < m.count; i++) {
    const dx = m.x[i]! - mx;
    const dy = m.y[i]! - my;
    p00 += m.w[i]! * (m.p00[i]! + dx * dx);
    p01 += m.w[i]! * (m.p01[i]! + dx * dy);
    p11 += m.w[i]! * (m.p11[i]! + dy * dy);
  }
  return { w, mx, my, p00: p00 / w, p01: p01 / w, p11: p11 / w };
}

describe("Runnalls merging cost", () => {
  test("is zero for identical components", () => {
    const m = mixtureOf([unit(3, -2, 5), unit(3, -2, 7)]);
    expect(runnallsCost(m, 0, 1)).toBeCloseTo(0, 12);
  });

  test("is symmetric", () => {
    const m = mixtureOf([
      { x: 0, y: 0, w: 2, p00: 1, p01: 0.3, p11: 2 },
      { x: 1.5, y: -0.5, w: 5, p00: 3, p01: -0.4, p11: 1 },
    ]);
    expect(runnallsCost(m, 0, 1)).toBeCloseTo(runnallsCost(m, 1, 0), 12);
  });

  test("is non-negative and grows with separation", () => {
    let previous = -1;
    for (const d of [0, 0.5, 1, 2, 4, 8]) {
      const m = mixtureOf([unit(0, 0), unit(d, 0)]);
      const cost = runnallsCost(m, 0, 1);
      expect(cost).toBeGreaterThanOrEqual(0);
      expect(cost).toBeGreaterThan(previous);
      previous = cost;
    }
  });

  test("matches the closed form for two equal-weight unit kernels", () => {
    // wi = wj = 1, Pi = Pj = I, separation d along x:
    //   Pij = I + diag(d^2/4, 0), so B = 1/2 * 2 * log(1 + d^2/4).
    for (const d of [0.5, 1, 3]) {
      const m = mixtureOf([unit(0, 0), unit(d, 0)]);
      expect(runnallsCost(m, 0, 1)).toBeCloseTo(Math.log(1 + (d * d) / 4), 12);
    }
  });

  test("prefers equal covariances over equal means, unlike a Mahalanobis rule", () => {
    // Runnalls' point against Salmond's joining algorithm: d^2_ij ignores the
    // component covariances, so it would merge the co-located pair with wildly
    // different shapes. The KL bound charges for the shape change instead.
    const m = mixtureOf([
      { x: 0, y: 0, w: 1, p00: 1, p01: 0, p11: 1 },
      { x: 0, y: 0, w: 1, p00: 100, p01: 0, p11: 100 }, // same mean, far shape
      { x: 0.3, y: 0, w: 1, p00: 1, p01: 0, p11: 1 }, // near mean, same shape
    ]);
    expect(runnallsCost(m, 0, 2)).toBeLessThan(runnallsCost(m, 0, 1));
  });
});

describe("moment-preserving pair merge", () => {
  test("preserves mass, mean and covariance of the pair", () => {
    const m = mixtureOf([
      { x: -1, y: 2, w: 3, p00: 1.5, p01: 0.2, p11: 0.8 },
      { x: 4, y: -1, w: 1, p00: 2, p01: -0.5, p11: 3 },
    ]);
    const expected = moments(m);

    const out = createMixture(1);
    out.count = 1;
    mergePair(m, 0, 1, out, 0);

    expect(out.w[0]).toBeCloseTo(expected.w, 12);
    expect(out.x[0]).toBeCloseTo(expected.mx, 12);
    expect(out.y[0]).toBeCloseTo(expected.my, 12);
    expect(out.p00[0]).toBeCloseTo(expected.p00, 12);
    expect(out.p01[0]).toBeCloseTo(expected.p01, 12);
    expect(out.p11[0]).toBeCloseTo(expected.p11, 12);
  });
});

describe("runnallsReduce", () => {
  test("full reduction reproduces the moments of the original mixture", () => {
    const comps: Comp[] = [];
    for (let i = 0; i < 40; i++) {
      comps.push({
        x: Math.sin(i * 1.7) * 5,
        y: Math.cos(i * 0.9) * 3,
        w: 1 + (i % 7),
        p00: 1,
        p01: 0,
        p11: 1,
      });
    }
    const original = mixtureOf(comps);
    const expected = moments(original);

    const { mixture } = runnallsReduce(mixtureOf(comps), { targetCount: 1 });
    expect(mixture.count).toBe(1);
    expect(mixture.w[0]).toBeCloseTo(expected.w, 9);
    expect(mixture.x[0]).toBeCloseTo(expected.mx, 9);
    expect(mixture.y[0]).toBeCloseTo(expected.my, 9);
    expect(mixture.p00[0]).toBeCloseTo(expected.p00, 9);
    expect(mixture.p01[0]).toBeCloseTo(expected.p01, 9);
    expect(mixture.p11[0]).toBeCloseTo(expected.p11, 9);
  });

  test("conserves total weight at every budget", () => {
    const comps: Comp[] = [];
    for (let i = 0; i < 60; i++) comps.push(unit(i * 0.13, (i % 5) * 0.4, i + 1));
    const total = comps.reduce((s, c) => s + c.w, 0);

    for (const target of [1, 3, 10, 25, 60]) {
      const { mixture } = runnallsReduce(mixtureOf(comps), {
        targetCount: target,
      });
      expect(mixture.count).toBe(target);
      let sum = 0;
      for (let i = 0; i < mixture.count; i++) sum += mixture.w[i]!;
      expect(sum).toBeCloseTo(total, 9);
    }
  });

  test("merges the two closest components first", () => {
    // Three tight pairs, far apart. Reducing 6 -> 3 must recover the pairs.
    const comps = [
      unit(0, 0),
      unit(0.1, 0),
      unit(50, 0),
      unit(50.1, 0),
      unit(0, 80),
      unit(0.1, 80),
    ];
    const { mixture } = runnallsReduce(mixtureOf(comps), { targetCount: 3 });
    expect(mixture.count).toBe(3);
    const centers = Array.from({ length: 3 }, (_, i) => [
      mixture.x[i]!,
      mixture.y[i]!,
    ])
      .map(([x, y]) => `${x!.toFixed(2)},${y!.toFixed(2)}`)
      .sort();
    expect(centers).toEqual(["0.05,0.00", "0.05,80.00", "50.05,0.00"]);
  });

  test("is independent of input ordering", () => {
    const comps: Comp[] = [];
    for (let i = 0; i < 30; i++) {
      comps.push(unit(Math.sin(i * 2.3) * 4, Math.cos(i * 1.1) * 4, 1 + (i % 3)));
    }
    const shuffled = [...comps].reverse();

    const a = runnallsReduce(mixtureOf(comps), { targetCount: 5 }).mixture;
    const b = runnallsReduce(mixtureOf(shuffled), { targetCount: 5 }).mixture;

    const key = (m: Mixture) =>
      Array.from({ length: m.count }, (_, i) =>
        `${m.x[i]!.toFixed(6)}|${m.y[i]!.toFixed(6)}|${m.w[i]!.toFixed(6)}`,
      ).sort();

    expect(key(a)).toEqual(key(b));
  });

  test("stops on the KL threshold before the budget", () => {
    // Two well-separated singletons: any merge is expensive.
    const comps = [unit(0, 0), unit(1000, 0)];
    const { mixture, merges } = runnallsReduce(mixtureOf(comps), {
      targetCount: 1,
      klThreshold: 1e-6,
    });
    expect(merges).toBe(0);
    expect(mixture.count).toBe(2);
  });

  test("merge costs are monotonically non-decreasing", () => {
    // Runnalls / Salmond: the minimum merge cost rises as reduction proceeds.
    const comps: Comp[] = [];
    for (let i = 0; i < 50; i++) {
      comps.push(unit(Math.sin(i * 3.1) * 6, Math.cos(i * 2.7) * 6, 1 + (i % 4)));
    }
    const m = mixtureOf(comps);
    let previous = 0;
    for (let target = 49; target >= 1; target--) {
      const { lastCost } = runnallsReduce(m, { targetCount: target });
      expect(lastCost).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = lastCost;
    }
  });
});

describe("KLDownsampler", () => {
  function build(n: number) {
    const data = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      data[i * 3] = i / n;
      data[i * 3 + 1] = Math.sin(i * 0.05) * 0.4 + 0.5;
      data[i * 3 + 2] = 1 + (i % 5);
    }
    return data;
  }

  function configure(data: Float64Array, targetCount: number) {
    const d = new KLDownsampler();
    d.setViewMinX(0);
    d.setViewMaxX(1);
    d.setViewMinY(0);
    d.setViewMaxY(1);
    d.setScreenW(1920);
    d.setScreenH(1080);
    d.setSigmaSizePx(16);
    d.setTargetCount(targetCount);
    d.setDataF64(data);
    return d;
  }

  test("emits count * 7 floats and honours the budget", () => {
    const result = configure(build(300), 20).mergePoints();
    expect(result.count).toBe(20);
    expect(result.gpuInstances.length).toBe(20 * 7);
  });

  test("returns empty for an empty view range", () => {
    const d = configure(build(300), 20);
    d.setViewMinX(5);
    d.setViewMaxX(6);
    const result = d.mergePoints();
    expect(result.count).toBe(0);
    expect(result.gpuInstances.length).toBe(0);
  });

  test("emits symmetric positive-definite covariances", () => {
    const { gpuInstances, count } = configure(build(300), 15).mergePoints();
    for (let i = 0; i < count; i++) {
      const o = i * 7;
      const p00 = gpuInstances[o + 3]!;
      const p01 = gpuInstances[o + 4]!;
      const p10 = gpuInstances[o + 5]!;
      const p11 = gpuInstances[o + 6]!;
      expect(p01).toBe(p10);
      expect(p00).toBeGreaterThan(0);
      expect(p00 * p11 - p01 * p10).toBeGreaterThan(0);
      expect(gpuInstances[o + 2]!).toBeGreaterThan(0);
    }
  });

  test("centres stay inside the view in data space", () => {
    const { gpuInstances, count } = configure(build(300), 15).mergePoints();
    for (let i = 0; i < count; i++) {
      expect(gpuInstances[i * 7]!).toBeGreaterThanOrEqual(0);
      expect(gpuInstances[i * 7]!).toBeLessThanOrEqual(1);
    }
  });

  test("no merging leaves one kernel per visible sample", () => {
    const data = build(120);
    const d = configure(data, Number.POSITIVE_INFINITY);
    const { count, gpuInstances } = d.mergePoints();
    expect(count).toBe(120);
    // Unit covariance, amplitude w / 2pi, matching the single-point convention.
    expect(gpuInstances[3]!).toBeCloseTo(1, 5);
    expect(gpuInstances[4]!).toBeCloseTo(0, 5);
    expect(gpuInstances[2]!).toBeCloseTo(data[2]! / (2 * Math.PI), 5);
  });
});
