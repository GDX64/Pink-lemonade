import { describe, expect, it } from "vitest";
import {
  DEFAULT_PARAMS,
  buildRipmap,
  coalesceRuns,
  generateDataset,
  levelCounts,
  maxDepth,
  runQuery,
  TREE_PARAMS,
  type ViewQuery,
} from "../src/pages/ripmap/ripmap-model";

const view = (minX: number, maxX: number): ViewQuery => ({
  minX,
  maxX,
  yBounds: null,
  screenW: 900,
  screenH: 400,
  sigmaPx: 14,
  mergeThreshold: 1,
});

describe("ripmap explorer model", () => {
  it("mirrors the Rust layout helpers", () => {
    expect(levelCounts(10_000, 32, 4, 1024)).toEqual([313]);
    expect(levelCounts(300_000, 32, 4, 1024)).toEqual([9375, 2344, 586]);
    expect(maxDepth(32, 16)).toBe(1);
    expect(maxDepth(8192, 16)).toBe(9);
    expect(maxDepth(16, 16)).toBe(0);
    expect(coalesceRuns([5, 1, 2, 9, 30], 3, 100)).toEqual([[1, 9], [30, 1]]);
    expect(coalesceRuns([0, 1, 2, 3], 0, 2)).toEqual([[0, 2], [2, 2]]);
  });

  for (const shape of ["random_walk", "noisy_spikes"] as const) {
    it(`preserves exact mass and y-range on ${shape}`, () => {
      const data = generateDataset(shape, 120_000);
      const model = buildRipmap(data, DEFAULT_PARAMS);
      for (const [minX, maxX] of [[0, 1], [0.2137, 0.6011], [0.5, 0.52]] as const) {
        const r = runQuery(model, view(minX, maxX));
        let mass = 0;
        let lo = Infinity;
        let hi = -Infinity;
        for (let i = 0; i < data.xs.length; i++) {
          const x = data.xs[i]!;
          if (x < minX || x > maxX) continue;
          mass += data.ws[i]!;
          lo = Math.min(lo, data.ys[i]!);
          hi = Math.max(hi, data.ys[i]!);
        }
        let got = 0;
        for (const item of r.items) {
          if (item.kind === "plane") got += item.cells.reduce((s, c) => s + c.mass, 0);
          else if (item.kind === "resolved") got += item.node.mass;
          else {
            for (let i = item.node.start; i < item.node.end; i++) {
              const x = data.xs[i]!;
              if (x >= minX && x <= maxX) got += data.ws[i]!;
            }
          }
        }
        expect(got / mass).toBeCloseTo(1, 9);
        expect(r.yRange).toEqual([lo, hi]);
      }
    });
  }

  for (const strategy of ["ripmap", "tree"] as const) {
    it(`greedy kernel groups (${strategy}) preserve the visible mass exactly`, () => {
      const data = generateDataset("random_walk", 120_000);
      const model = buildRipmap(data, strategy === "tree" ? TREE_PARAMS : DEFAULT_PARAMS, strategy);
      for (const [minX, maxX] of [[0, 1], [0.3, 0.31]] as const) {
        const r = runQuery(model, view(minX, maxX));
        let mass = 0;
        for (let i = 0; i < data.xs.length; i++) {
          const x = data.xs[i]!;
          if (x >= minX && x <= maxX) mass += data.ws[i]!;
        }
        let got = r.groups.reduce((s, g) => s + g.node.mass, 0);
        for (const item of r.items) {
          if (item.kind === "plane") got += item.cells.reduce((s, c) => s + c.mass, 0);
        }
        expect(got / mass).toBeCloseTo(1, 9);
        const pieces = r.items.filter((i) => i.kind === "resolved").length;
        expect(r.groups.length).toBeLessThanOrEqual(pieces + r.groups.reduce((s, g) => s + g.rawPoints, 0));
      }
    });
  }

  it("x-tree mirrors PagedTreeStrategy layout and never uses planes", () => {
    const model = buildRipmap(generateDataset("noisy_spikes", 300_000), TREE_PARAMS, "tree");
    expect(model.levelCounts).toEqual([9375, 293, 10]);
    expect(model.objects.map((o) => o.key)).toEqual(["manifest.bin", "level_0.bin", "level_1.bin", "points.bin"]);
    const r = runQuery(model, view(0, 1));
    expect(r.items.some((i) => i.kind === "plane")).toBe(false);
    // White noise never fits as tall columns: the overview falls through to (nearly) every leaf.
    expect(r.wireBytes).toBeGreaterThan(model.rawBytes * 0.9);
  });

  it("serves the white-noise overview mostly from planes", () => {
    const model = buildRipmap(generateDataset("noisy_spikes", 300_000), DEFAULT_PARAMS);
    const r = runQuery(model, view(0, 1));
    expect(r.items.some((i) => i.kind === "plane")).toBe(true);
    expect(r.wireBytes).toBeLessThan(model.rawBytes / 10);
  });
});
