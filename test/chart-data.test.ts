import { describe, expect, test } from "vitest";
import { normalizeChartData } from "../src/chart/api";

describe("normalizeChartData", () => {
  test("columns: interleaves, defaults weight to 1, normalizes x", () => {
    const out = normalizeChartData({ x: [100, 150, 200], y: [1, 2, 3] });
    expect(out.n).toBe(3);
    expect(out.xMin).toBe(100);
    expect(out.xScale).toBe(100);
    expect(Array.from(out.dataF64)).toEqual([0, 1, 1, 0.5, 2, 1, 1, 3, 1]);
  });

  test("interleaved sorted input is normalized in place", () => {
    const points = new Float64Array([10, 5, 2, 20, 6, 3]);
    const out = normalizeChartData({ points });
    expect(out.dataF64).toBe(points);
    expect(Array.from(points)).toEqual([0, 5, 2, 1, 6, 3]);
  });

  test("unsorted input is sorted by x, keeping rows together", () => {
    const out = normalizeChartData({
      x: [30, 10, 20],
      y: [3, 1, 2],
      weight: [0.3, 0.1, 0.2],
    });
    expect(Array.from(out.dataF64)).toEqual([0, 1, 0.1, 0.5, 2, 0.2, 1, 3, 0.3]);
  });

  test("single point and empty input do not divide by zero", () => {
    expect(normalizeChartData({ x: [5], y: [1] }).dataF64[0]).toBe(0);
    expect(normalizeChartData({ points: new Float64Array(0) }).n).toBe(0);
  });

  test("rejects mismatched lengths", () => {
    expect(() => normalizeChartData({ x: [1, 2], y: [1] })).toThrow();
    expect(() =>
      normalizeChartData({ points: new Float64Array(4) }),
    ).toThrow();
  });
});
