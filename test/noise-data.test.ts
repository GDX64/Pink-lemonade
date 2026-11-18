import { describe, expect, test } from "vitest";
import { createNoiseData, createNoiseFloatData } from "../src/data/noise";

describe("packed noise data", () => {
  test.each([1, 64, 100])("matches the point generator for %i points", (n) => {
    const points = createNoiseData(n, 578211);
    const packed = createNoiseFloatData(n);
    expect(packed.n).toBe(n);
    for (let i = 0; i < n; i++) {
      expect(packed.dataF64[i * 3]).toBeCloseTo(n === 1 ? 0 : i / (n - 1), 12);
      expect(packed.dataF64[i * 3 + 1]).toBe(points[i]![1]);
      expect(packed.dataF64[i * 3 + 2]).toBe(points[i]![2] || 1);
    }
  });
});
