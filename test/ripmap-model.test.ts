import { describe, expect, it } from "vitest";
import {
  DEFAULT_PARAMS,
  buildRipmap,
  coalesceRuns,
  generateDataset,
  levelCounts,
  maxDepth,
  planeDepthsForLevel,
  runQuery,
  TREE_PARAMS,
  type ViewQuery,
} from "../src/pages/ripmap/ripmap-model";
import {
  QUAD_PARAMS,
  RECORD_BYTES,
  buildQuadtree,
  coalesceSpans,
  countInlineDepths,
  runQuadQuery,
} from "../src/pages/ripmap/quadtree-model";
import { GRID_PARAMS, buildGrid, indexAt, maxLevelFor, quantize, runGridQuery } from "../src/pages/ripmap/grid-model";

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
    expect(planeDepthsForLevel(0, 32, 16, 2)).toEqual([]);
    expect(planeDepthsForLevel(1, 128, 16, 2)).toEqual([1, 3, 5]);
    expect(planeDepthsForLevel(2, 512, 16, 2)).toEqual([1, 2, 3, 5, 7]);
    expect(planeDepthsForLevel(3, 2048, 16, 2)).toEqual([1, 2, 3, 4, 5, 7, 9]);
    expect(planeDepthsForLevel(4, 8192, 16, 2)).toEqual([1, 2, 3, 4, 5, 6, 7, 9, 11]);
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

  it("serves the white-noise overview mostly from planes in a 4-object PLPRIPM2 layout", () => {
    const model = buildRipmap(generateDataset("noisy_spikes", 300_000), DEFAULT_PARAMS);
    expect(model.objects.map((o) => o.key)).toEqual(["manifest.bin", "levels.bin", "planes.bin", "points.bin"]);
    const r = runQuery(model, view(0, 1));
    expect(r.items.some((i) => i.kind === "plane")).toBe(true);
    expect(r.wireBytes).toBeLessThan(model.rawBytes / 10);
  });
});

describe("quadtree explorer model", () => {
  it("mirrors inline_depths and coalesce_spans from Rust", () => {
    expect(countInlineDepths([1, 4, 16, 64, 256, 1024, 4096], 1024)).toBe(5);
    expect(countInlineDepths([1, 4, 16], 1024)).toBe(3);
    expect(countInlineDepths([1, 5000], 0)).toBe(1);
    expect(countInlineDepths([], 1024)).toBe(0);
    expect(coalesceSpans([[10, 2], [0, 4], [5, 1]], 1, 100)).toEqual([[0, 6], [10, 12]]);
    expect(coalesceSpans([[0, 4], [4, 4], [8, 4]], 0, 8)).toEqual([[0, 8], [8, 12]]);
    expect(coalesceSpans([[0, 50], [50, 1]], 0, 8)).toEqual([[0, 50], [50, 51]]);
  });

  for (const shape of ["random_walk", "noisy_spikes"] as const) {
    it(`partitions points and preserves exact mass and y-range on ${shape}`, () => {
      const data = generateDataset(shape, 60_000);
      const model = buildQuadtree(data, QUAD_PARAMS);
      // Every point in exactly one leaf, leaves x-sorted and within their quadrant.
      let covered = 0;
      for (const level of model.levels) {
        for (const node of level) {
          if (!node.isLeaf) continue;
          covered += node.end - node.start;
          expect(node.end - node.start).toBeLessThanOrEqual(QUAD_PARAMS.leafPoints);
          for (let i = node.start + 1; i < node.end; i++) expect(model.data.xs[i]!).toBeGreaterThanOrEqual(model.data.xs[i - 1]!);
          expect(node.minX).toBeGreaterThanOrEqual(node.rx0);
          expect(node.maxY).toBeLessThanOrEqual(node.ry1);
        }
      }
      expect(covered).toBe(data.xs.length);

      for (const [minX, maxX] of [[0, 1], [0.2137, 0.6011], [0.5, 0.52]] as const) {
        const r = runQuadQuery(model, view(minX, maxX));
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
        expect(r.groups.reduce((s, g) => s + g.node.mass, 0) / mass).toBeCloseTo(1, 9);
        expect(r.yRange).toEqual([lo, hi]);
        expect(r.items.some((i) => i.kind === "plane")).toBe(false);
      }
    });
  }

  it("stores every remote depth back to back in one nodes.bin", () => {
    const model = buildQuadtree(generateDataset("noisy_spikes", 300_000), QUAD_PARAMS);
    expect(model.objects.map((o) => o.key)).toEqual(["manifest.bin", "nodes.bin", "points.bin"]);
    let offset = 0;
    for (const [d, info] of model.depthInfo.entries()) {
      expect(info.inline).toBe(d < model.inlineDepths);
      if (info.inline) continue;
      expect(info.offset).toBe(offset);
      offset += info.count * RECORD_BYTES;
    }
    expect(model.objects[1]!.bytes).toBe(offset);
  });
});

describe("sparse grid explorer model", () => {
  it("mirrors max_level_for and nests levels exactly", () => {
    expect(maxLevelFor(1, 256)).toBe(0);
    expect(maxLevelFor(257, 256)).toBe(1);
    expect(maxLevelFor(10_000_000, 256)).toBe(16);
    for (const v of [0, 0.1, 0.333333, 0.5, 0.9999999, 1]) {
      const q = quantize(v, 0, 1);
      for (let level = 1; level <= 24; level++) expect(indexAt(q, level) >>> 1).toBe(indexAt(q, level - 1));
    }
  });

  it("stores every plane's cells column-major with consistent CSC pointers and exact mass", () => {
    const data = generateDataset("noisy_spikes", 100_000);
    const model = buildGrid(data, GRID_PARAMS);
    const total = data.ws.reduce((s, w) => s + w, 0);
    expect(model.planes.length).toBeGreaterThan(0);
    for (const p of model.planes) {
      expect(p.i - p.j).toBeGreaterThanOrEqual(GRID_PARAMS.minAspect);
      expect(p.cells.length * GRID_PARAMS.minPointsPerCell).toBeLessThanOrEqual(data.xs.length);
      expect(p.colStarts[p.colStarts.length - 1]).toBe(p.cells.length);
      for (let c = 0; c + 1 < p.colStarts.length; c++) {
        for (let k = p.colStarts[c]!; k < p.colStarts[c + 1]!; k++) expect(p.cells[k]!.col).toBe(c);
      }
      expect(p.cells.reduce((s, c) => s + c.mass, 0)).toBeCloseTo(total, 6);
    }
    expect(model.objects.map((o) => o.key)).toEqual(["manifest.bin", "columns.bin", "cells.bin", "points.bin"]);
  });

  it("answers views with exact mass and y-range in about two rounds", () => {
    for (const shape of ["noisy_spikes", "random_walk"] as const) {
      const data = generateDataset(shape, 100_000);
      const model = buildGrid(data, GRID_PARAMS);
      for (const [lo, hi] of [[0, 1], [0.4123, 0.4567], [0.31, 0.3105]] as const) {
        const r = runGridQuery(model, view(lo, hi));
        let mass = 0;
        let minY = Infinity;
        let maxY = -Infinity;
        for (let i = 0; i < data.xs.length; i++) {
          if (data.xs[i]! < lo || data.xs[i]! > hi) continue;
          mass += data.ws[i]!;
          minY = Math.min(minY, data.ys[i]!);
          maxY = Math.max(maxY, data.ys[i]!);
        }
        expect(r.yRange).toEqual([minY, maxY]);
        let got = r.groups.reduce((s, g) => s + g.node.mass, 0);
        for (const rd of r.chosen!.renders) if (rd.kind === "cells") got += rd.cells.reduce((s, c) => s + c.mass, 0);
        expect(got / mass).toBeCloseTo(1, 9);
        // open + records + cells/raw.
        expect(r.rounds).toBeLessThanOrEqual(3);
      }
    }
  });
});
