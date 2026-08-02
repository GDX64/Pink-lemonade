import { createNoiseFloatData } from "../chart/chart";
import { mergePoints } from "../examples/rasterizing/downsampling";
import { wasmKlMerge } from "../examples/rasterizing/kl-downsampling";
import { wasmSalmondMerge } from "../examples/rasterizing/salmond-downsampling";

/**
 * Largest dataset for which Runnalls' reduction is also evaluated. It is
 * O(N^2), so N=100,000 is out of reach (hours); above this the comparison
 * columns read "n/a".
 */
const KL_MAX_N = 1_000;

/**
 * Same for Salmond's clustering reduction, which reduces a whole pass at a time
 * instead of rescanning after every merge and so reaches 10,000 comfortably.
 */
const SALMOND_MAX_N = 10_000;

type GaussianComponent = {
  x: number;
  y: number;
  w: number;
  p00: number;
  p01: number;
  p10: number;
  p11: number;
};

const SAMPLE_COUNT = 10_000;
const PROGRESS_STEP = 250;
const MAX_SIGMA_DISTANCE = 5;
const MAX_SIGMA_DISTANCE_SQ = MAX_SIGMA_DISTANCE * MAX_SIGMA_DISTANCE;

type MonteCarloMetrics = {
  mse: number;
  nrmse: number;
  rme: number;
  acceptedSamples: number;
  mergedCount: number;
  /** Runnalls at the same kernel budget, or null when N exceeds KL_MAX_N. */
  kl: { mse: number; nrmse: number; rme: number } | null;
  /**
   * Salmond clustering, or null when N exceeds SALMOND_MAX_N. It combines
   * components in groups, so `count` can land below the merge's budget rather
   * than exactly on it --- the comparison is only fair read alongside it.
   */
  salmond: { mse: number; nrmse: number; rme: number; count: number } | null;
};

export type MseMethod = "merge" | "runnalls" | "salmond";

/**
 * One method's NRMSE at one N. Held as numbers rather than formatted strings so
 * a chart can fit a trend line; formatting is the renderer's job.
 */
export interface MseResult {
  method: MseMethod;
  n: number;
  kernels: number;
  nrmse: number;
  /** 95% relative margin of error on the MSE estimate, as a percentage. */
  rmePct: number;
  samples: number;
}

type MonteCarloEstimationEvent =
  | {
      kind: "scenario-start";
      n: number;
      viewMinX: number;
      viewMaxX: number;
      viewMinY: number;
      viewMaxY: number;
    }
  | {
      kind: "sample";
      n: number;
      x: number;
      y: number;
      accepted: boolean;
    }
  | {
      kind: "progress";
      n: number;
      acceptedSamples: number;
      targetSamples: number;
      progress: number;
    }
  | {
      kind: "result";
      n: number;
      metrics: MonteCarloMetrics;
    };

export async function runMsePage(
  onProgress?: (event: {
    scenarioIndex: number;
    totalScenarios: number;
    n: number;
    acceptedSamples: number;
    targetSamples: number;
    scenarioProgress: number;
    overallProgress: number;
  }) => void,
): Promise<MseResult[]> {
  const scenarios = [1_000, 10_000, 100_000] as const;
  const results: MseResult[] = [];
  const progressUi = createMseProgressUi();
  const klDownsampler = await wasmKlMerge();
  const salmondDownsampler = await wasmSalmondMerge();

  for (const [scenarioIndex, n] of scenarios.entries()) {
    let metrics: MonteCarloMetrics | null = null;
    for (const event of estimateMonteCarloMetrics(
      n,
      klDownsampler,
      salmondDownsampler,
    )) {
      if (event.kind === "scenario-start") {
        progressUi.resetPlot({
          n: event.n,
          viewMinX: event.viewMinX,
          viewMaxX: event.viewMaxX,
          viewMinY: event.viewMinY,
          viewMaxY: event.viewMaxY,
        });
        continue;
      }

      if (event.kind === "sample") {
        progressUi.plotPoint({
          x: event.x,
          y: event.y,
          accepted: event.accepted,
        });
        continue;
      }

      if (event.kind === "progress") {
        const scenarioProgress = event.progress;
        const overallProgress =
          (scenarioIndex + scenarioProgress) / scenarios.length;
        progressUi.update({
          n,
          acceptedSamples: event.acceptedSamples,
          targetSamples: event.targetSamples,
          scenarioProgress,
          overallProgress,
        });
        onProgress?.({
          scenarioIndex,
          totalScenarios: scenarios.length,
          n,
          acceptedSamples: event.acceptedSamples,
          targetSamples: event.targetSamples,
          scenarioProgress,
          overallProgress,
        });
        await waitForNextFrame();
        continue;
      }
      metrics = event.metrics;
    }

    if (!metrics) {
      throw new Error(`MSE estimation finished without result for N=${n}`);
    }

    results.push({
      method: "merge",
      n,
      kernels: metrics.mergedCount,
      nrmse: metrics.nrmse,
      rmePct: metrics.rme,
      samples: metrics.acceptedSamples,
    });
    if (metrics.kl) {
      results.push({
        method: "runnalls",
        n,
        // Runnalls is held to the merge's exact budget by construction.
        kernels: metrics.mergedCount,
        nrmse: metrics.kl.nrmse,
        rmePct: metrics.kl.rme,
        samples: metrics.acceptedSamples,
      });
    }
    if (metrics.salmond) {
      results.push({
        method: "salmond",
        n,
        kernels: metrics.salmond.count,
        nrmse: metrics.salmond.nrmse,
        rmePct: metrics.salmond.rme,
        samples: metrics.acceptedSamples,
      });
    }
  }

  progressUi.done();

  return results;
}

function* estimateMonteCarloMetrics(
  n: number,
  klDownsampler: Awaited<ReturnType<typeof wasmKlMerge>>,
  salmondDownsampler: Awaited<ReturnType<typeof wasmSalmondMerge>>,
): Generator<MonteCarloEstimationEvent, void, void> {
  const { dataF64, yMin, yMax } = createNoiseFloatData(n);

  const viewMinX = 0;
  const viewMaxX = 1;
  const viewMinY = yMin;
  const viewMaxY = yMax;
  const screenW = 1920;
  const screenH = 1080;
  const mergeThreshold = 1;
  const sigmaSizePx = 16;

  yield {
    kind: "scenario-start",
    n,
    viewMinX,
    viewMaxX,
    viewMinY,
    viewMaxY,
  };

  const merged = mergePoints({
    viewMinX,
    viewMaxX,
    viewMinY,
    viewMaxY,
    screenW,
    screenH,
    mergeThreshold,
    sigmaSizePx,
    dataF64,
  });

  const mergedComponents = decodeComponents(merged.gpuInstances);
  const originalComponents = decodeOriginalComponents(dataF64);

  // Runnalls at the same kernel budget: the quality ceiling for this many
  // kernels, against which the one-pass merge is judged.
  let klComponents: GaussianComponent[] | null = null;
  if (n <= KL_MAX_N) {
    klDownsampler.setViewMinX(viewMinX);
    klDownsampler.setViewMaxX(viewMaxX);
    klDownsampler.setViewMinY(viewMinY);
    klDownsampler.setViewMaxY(viewMaxY);
    klDownsampler.setScreenW(screenW);
    klDownsampler.setScreenH(screenH);
    klDownsampler.setSigmaSizePx(sigmaSizePx);
    klDownsampler.setTargetCount(merged.count);
    klDownsampler.setDataF64(dataF64);
    klComponents = decodeComponents(klDownsampler.mergePoints().gpuInstances);
  }

  // Salmond clustering at the same budget. It reaches a larger N than Runnalls
  // but can only guarantee *at most* the merge's kernel count, so the achieved
  // count is carried through to the table.
  let salmondComponents: GaussianComponent[] | null = null;
  let salmondCount = 0;
  if (n <= SALMOND_MAX_N) {
    salmondDownsampler.setViewMinX(viewMinX);
    salmondDownsampler.setViewMaxX(viewMaxX);
    salmondDownsampler.setViewMinY(viewMinY);
    salmondDownsampler.setViewMaxY(viewMaxY);
    salmondDownsampler.setScreenW(screenW);
    salmondDownsampler.setScreenH(screenH);
    salmondDownsampler.setSigmaSizePx(sigmaSizePx);
    salmondDownsampler.setTargetCount(merged.count);
    salmondDownsampler.setDataF64(dataF64);
    const salmondResult = salmondDownsampler.mergePoints();
    salmondCount = salmondResult.count;
    salmondComponents = decodeComponents(salmondResult.gpuInstances);
  }

  const toSX = buildToSX(viewMinX, viewMaxX, screenW, sigmaSizePx);
  const toSY = buildToSY(viewMinY, viewMaxY, screenH, sigmaSizePx);

  const rng = createRng(0xdecafbad ^ n);
  const minApproxKde = 0.01;
  let acceptedSamples = 0;
  let mse = 0;
  let sqErrorSum = 0;
  let sqErrorSumSq = 0;
  let minReference = Number.POSITIVE_INFINITY;
  let maxReference = Number.NEGATIVE_INFINITY;
  let klSqErrorSum = 0;
  let klSqErrorSumSq = 0;
  let salmondSqErrorSum = 0;
  let salmondSqErrorSumSq = 0;
  while (acceptedSamples < SAMPLE_COUNT) {
    const x = lerp(viewMinX, viewMaxX, rng());
    const y = lerp(viewMinY, viewMaxY, rng());
    const approximation = evaluateKdeAt(mergedComponents, x, y, toSX, toSY);
    const accepted = approximation >= minApproxKde;

    yield {
      kind: "sample",
      n,
      x,
      y,
      accepted,
    };

    if (!accepted) {
      continue;
    }

    const reference = evaluateKdeAt(originalComponents, x, y, toSX, toSY);
    const diff = approximation - reference;
    const sqError = diff * diff;
    mse += sqError;
    sqErrorSum += sqError;
    sqErrorSumSq += sqError * sqError;
    acceptedSamples++;
    if (reference < minReference) minReference = reference;
    if (reference > maxReference) maxReference = reference;

    // Scored on exactly the same accepted samples as the merge, so the two
    // NRMSE figures are directly comparable.
    if (klComponents) {
      const klDiff = evaluateKdeAt(klComponents, x, y, toSX, toSY) - reference;
      const klSqError = klDiff * klDiff;
      klSqErrorSum += klSqError;
      klSqErrorSumSq += klSqError * klSqError;
    }

    if (salmondComponents) {
      const salmondDiff =
        evaluateKdeAt(salmondComponents, x, y, toSX, toSY) - reference;
      const salmondSqError = salmondDiff * salmondDiff;
      salmondSqErrorSum += salmondSqError;
      salmondSqErrorSumSq += salmondSqError * salmondSqError;
    }

    if (
      acceptedSamples % PROGRESS_STEP === 0 ||
      acceptedSamples === SAMPLE_COUNT
    ) {
      yield {
        kind: "progress",
        n,
        acceptedSamples,
        targetSamples: SAMPLE_COUNT,
        progress: acceptedSamples / SAMPLE_COUNT,
      };
    }
  }
  mse /= Math.max(acceptedSamples, 1);

  const varianceSqError =
    acceptedSamples > 1
      ? (sqErrorSumSq - (sqErrorSum * sqErrorSum) / acceptedSamples) /
        (acceptedSamples - 1)
      : 0;
  const stdErrMse = Math.sqrt(
    Math.max(varianceSqError, 0) / Math.max(acceptedSamples, 1),
  );
  const margin95 = 1.96 * stdErrMse;
  const rme = mse > 0 ? (margin95 / mse) * 100 : 0;

  const referenceRange = maxReference - minReference;
  const nrmse = referenceRange > 0 ? Math.sqrt(mse) / referenceRange : 0;

  let kl: MonteCarloMetrics["kl"] = null;
  if (klComponents) {
    const klMse = klSqErrorSum / Math.max(acceptedSamples, 1);
    const klVariance =
      acceptedSamples > 1
        ? (klSqErrorSumSq - (klSqErrorSum * klSqErrorSum) / acceptedSamples) /
          (acceptedSamples - 1)
        : 0;
    const klStdErr = Math.sqrt(
      Math.max(klVariance, 0) / Math.max(acceptedSamples, 1),
    );
    kl = {
      mse: klMse,
      nrmse: referenceRange > 0 ? Math.sqrt(klMse) / referenceRange : 0,
      rme: klMse > 0 ? ((1.96 * klStdErr) / klMse) * 100 : 0,
    };
  }

  let salmond: MonteCarloMetrics["salmond"] = null;
  if (salmondComponents) {
    const salmondMse = salmondSqErrorSum / Math.max(acceptedSamples, 1);
    const salmondVariance =
      acceptedSamples > 1
        ? (salmondSqErrorSumSq -
            (salmondSqErrorSum * salmondSqErrorSum) / acceptedSamples) /
          (acceptedSamples - 1)
        : 0;
    const salmondStdErr = Math.sqrt(
      Math.max(salmondVariance, 0) / Math.max(acceptedSamples, 1),
    );
    salmond = {
      mse: salmondMse,
      nrmse: referenceRange > 0 ? Math.sqrt(salmondMse) / referenceRange : 0,
      rme:
        salmondMse > 0 ? ((1.96 * salmondStdErr) / salmondMse) * 100 : 0,
      count: salmondCount,
    };
  }

  yield {
    kind: "result",
    n,
    metrics: {
      mse,
      nrmse,
      rme,
      acceptedSamples,
      mergedCount: merged.count,
      kl,
      salmond,
    },
  };
}

function waitForNextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function createMseProgressUi() {
  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.inset = "0";
  host.style.display = "grid";
  host.style.placeItems = "center";
  host.style.background = "rgba(247, 251, 255, 0.88)";
  host.style.zIndex = "9999";

  const card = document.createElement("div");
  card.style.width = "min(840px, 96vw)";
  card.style.border = "1px solid #c8d9eb";
  card.style.borderRadius = "14px";
  card.style.background = "#ffffff";
  card.style.padding = "18px";
  card.style.boxShadow = "0 10px 30px rgba(10, 34, 65, 0.15)";

  const title = document.createElement("div");
  title.textContent = "Estimating Monte Carlo MSE...";
  title.style.fontWeight = "700";
  title.style.marginBottom = "10px";

  const detail = document.createElement("div");
  detail.style.fontSize = "14px";
  detail.style.color = "#2d4d6c";
  detail.style.marginBottom = "10px";
  detail.textContent = "Preparing...";

  const canvasWrap = document.createElement("div");
  canvasWrap.style.marginBottom = "10px";
  canvasWrap.style.display = "flex";
  canvasWrap.style.justifyContent = "center";

  const canvas = document.createElement("canvas");
  canvas.width = 800;
  canvas.height = 600;
  canvas.style.width = "800px";
  canvas.style.height = "600px";
  canvas.style.maxWidth = "100%";
  canvas.style.border = "1px solid #d0dce9";
  canvas.style.borderRadius = "8px";
  canvas.style.background = "#f8fbff";

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not create Canvas 2D context for MSE plot");
  }

  canvasWrap.appendChild(canvas);

  const barWrap = document.createElement("div");
  barWrap.style.height = "10px";
  barWrap.style.background = "#e7eef6";
  barWrap.style.borderRadius = "999px";
  barWrap.style.overflow = "hidden";

  const bar = document.createElement("div");
  bar.style.height = "100%";
  bar.style.width = "0%";
  bar.style.background = "linear-gradient(90deg, #6ca4d8, #2f6fa9)";
  bar.style.transition = "width 80ms linear";

  barWrap.appendChild(bar);
  card.appendChild(title);
  card.appendChild(detail);
  card.appendChild(canvasWrap);
  card.appendChild(barWrap);
  host.appendChild(card);
  document.body.appendChild(host);

  let plotViewMinX = 0;
  let plotViewMaxX = 1;
  let plotViewMinY = 0;
  let plotViewMaxY = 1;

  const drawBackground = () => {
    ctx.fillStyle = "#f8fbff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  drawBackground();

  return {
    resetPlot(input: {
      n: number;
      viewMinX: number;
      viewMaxX: number;
      viewMinY: number;
      viewMaxY: number;
    }) {
      plotViewMinX = input.viewMinX;
      plotViewMaxX = input.viewMaxX;
      plotViewMinY = input.viewMinY;
      plotViewMaxY = input.viewMaxY;
      drawBackground();
      detail.textContent = `Preparing N=${input.n.toLocaleString("en-US")}...`;
    },
    plotPoint(input: { x: number; y: number; accepted: boolean }) {
      const xRange = plotViewMaxX - plotViewMinX;
      const yRange = plotViewMaxY - plotViewMinY;
      if (xRange <= 0 || yRange <= 0) return;

      const nx = (input.x - plotViewMinX) / xRange;
      const ny = (input.y - plotViewMinY) / yRange;
      const px = Math.round(nx * (canvas.width - 1));
      const py = Math.round((1 - ny) * (canvas.height - 1));

      if (px < 0 || px >= canvas.width || py < 0 || py >= canvas.height) return;

      ctx.fillStyle = input.accepted ? "#1f6fb2" : "#c8d3df";
      ctx.fillRect(px, py, 2, 2);
    },
    update(input: {
      n: number;
      acceptedSamples: number;
      targetSamples: number;
      scenarioProgress: number;
      overallProgress: number;
    }) {
      const scenarioPct = (input.scenarioProgress * 100).toFixed(1);
      const overallPct = (input.overallProgress * 100).toFixed(1);
      detail.textContent = `N=${input.n.toLocaleString("en-US")} | samples ${input.acceptedSamples}/${input.targetSamples} | scenario ${scenarioPct}% | overall ${overallPct}%`;
      bar.style.width = `${overallPct}%`;
    },
    done() {
      host.remove();
    },
  };
}

function decodeOriginalComponents(dataF64: Float64Array): GaussianComponent[] {
  const components: GaussianComponent[] = [];
  for (let i = 0; i < dataF64.length; i += 3) {
    const x = dataF64[i]!;
    const y = dataF64[i + 1]!;
    const weight = dataF64[i + 2]!;
    components.push({
      x,
      y,
      w: weight / (2 * Math.PI),
      p00: 1,
      p01: 0,
      p10: 0,
      p11: 1,
    });
  }
  return components;
}

function decodeComponents(instances: Float32Array): GaussianComponent[] {
  const components: GaussianComponent[] = [];
  for (let i = 0; i < instances.length; i += 7) {
    components.push({
      x: instances[i]!,
      y: instances[i + 1]!,
      w: instances[i + 2]!,
      p00: instances[i + 3]!,
      p01: instances[i + 4]!,
      p10: instances[i + 5]!,
      p11: instances[i + 6]!,
    });
  }
  return components;
}

function buildToSX(
  viewMinX: number,
  viewMaxX: number,
  screenW: number,
  sigmaSizePx: number,
) {
  const xFactor = screenW / (viewMaxX - viewMinX) / sigmaSizePx;
  return (x: number) => (x - viewMinX) * xFactor;
}

function buildToSY(
  viewMinY: number,
  viewMaxY: number,
  screenH: number,
  sigmaSizePx: number,
) {
  const yFactor = screenH / (viewMaxY - viewMinY) / sigmaSizePx;
  return (y: number) => (y - viewMinY) * yFactor;
}

function evaluateKdeAt(
  components: GaussianComponent[],
  x: number,
  y: number,
  toSX: (x: number) => number,
  toSY: (y: number) => number,
): number {
  const sx = toSX(x);
  const sy = toSY(y);
  let sum = 0;

  for (const c of components) {
    const mux = toSX(c.x);
    const muy = toSY(c.y);
    const dx = sx - mux;
    const dy = sy - muy;

    const cartesianD2 = dx * dx + dy * dy;
    if (cartesianD2 > MAX_SIGMA_DISTANCE_SQ) continue;

    const det = c.p00 * c.p11 - c.p01 * c.p10;
    if (det <= 1e-12) continue;

    const inv00 = c.p11 / det;
    const inv01 = -c.p01 / det;
    const inv10 = -c.p10 / det;
    const inv11 = c.p00 / det;
    const d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

    sum += c.w * Math.exp(-0.5 * d2);
  }

  return sum;
}

function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
