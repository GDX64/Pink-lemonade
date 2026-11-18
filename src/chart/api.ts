import {
  GaussianChart,
  type DataRegion,
  type GaussianChartState,
  type LoadedData,
  type RefreshFlags,
} from "./gaussian-chart";

export type { DataRegion };

type NumericArray = ArrayLike<number>;

/**
 * Chart input, in the caller's own units (x is typically a timestamp in ms).
 *
 * - Columns: `{ x, y, weight? }`. Copied; `weight` defaults to 1.
 * - Interleaved: `{ points }` laid out as `[x, y, weight, x, y, weight, ...]`.
 *   Zero-copy when already sorted by x: the chart takes ownership of the
 *   array and normalizes x in place, so don't reuse it afterwards.
 *
 * Unsorted input is sorted by x (which costs a copy).
 */
export type ChartData =
  | { x: NumericArray; y: NumericArray; weight?: NumericArray }
  | { points: Float64Array };

export type ChartStyle = {
  /** Kernel standard deviation in CSS px. */
  sigma?: number;
  opacityCut?: number;
  /** Number of color bands; 0 = continuous. */
  quantSteps?: number;
  /** Low, mid and high palette colors (hex). */
  palette?: [string, string, string];
  /** Position of the mid color in [0, 1]. */
  paletteLevel?: number;
  heatmap?: { show?: boolean; mode?: "overlay" | "composited" };
  line?: { show?: boolean; width?: number; opacity?: number };
  scatter?: { show?: boolean };
  /** Debug overlays for the merged gaussians. */
  debug?: { quads?: boolean; sigmaCircle?: boolean };
};

export type ChartSettings = {
  style?: ChartStyle;
  axes?: { x?: { show?: boolean }; y?: { show?: boolean } };
  merge?: { thresholdSigmas?: number };
  pixelRatio?: number | "device";
};

export type CreateChartOptions = ChartSettings & {
  data: ChartData;
  /** Show the lil-gui debug panel. Defaults to false. */
  controls?: boolean;
};

export type ChartEvents = {
  viewchange: (view: DataRegion) => void;
  selectionchange: (selection: DataRegion | null) => void;
  frame: () => void;
};

export interface GaussianChartHandle {
  /** The element the chart renders into (fills the mount). */
  readonly element: HTMLElement;
  update(settings: ChartSettings): void;
  setData(data: ChartData): void;
  readonly view: {
    get(): DataRegion;
    set(range: { x: [number, number] }): void;
  };
  readonly selection: {
    get(): DataRegion | null;
    set(region: DataRegion | null): void;
  };
  on<E extends keyof ChartEvents>(event: E, cb: ChartEvents[E]): () => void;
  /** Integrate the rendered density over `region` (whole view if omitted). */
  integrate(
    region?: DataRegion,
  ): Promise<{ integral: number; actualSum: number; errorPercent: number } | null>;
  /** Render at the given size and return a PNG. */
  toImage(options: {
    width: number;
    height: number;
    pixelRatio?: number;
  }): Promise<Blob | null>;
  destroy(): void;
}

/**
 * Create a WebGPU gaussian-splat chart that fills `mount`.
 * `mount` must have a size (the chart follows it with a ResizeObserver).
 * Resolves once the chart is rendering.
 */
export async function createGaussianChart(
  mount: HTMLElement,
  options: CreateChartOptions,
): Promise<GaussianChartHandle> {
  const chart = new GaussianChart({
    data: normalizeChartData(options.data),
    controls: options.controls ?? false,
  });
  applySettings(chart.state, options);

  const element = chart.container;
  element.style.width = "100%";
  element.style.height = "100%";
  mount.appendChild(element);

  const listeners: { [E in keyof ChartEvents]: Set<ChartEvents[E]> } = {
    viewchange: new Set(),
    selectionchange: new Set(),
    frame: new Set(),
  };
  let selection: DataRegion | null = null;
  let lastView = "";

  chart.onSelectionChange = (region) => {
    selection = region;
    listeners.selectionchange.forEach((cb) => cb(region));
  };
  chart.onFrame = () => {
    listeners.frame.forEach((cb) => cb());
    if (listeners.viewchange.size === 0) return;
    const view = chart.getViewRange();
    const key = `${view.x}|${view.y}`;
    if (key === lastView) return;
    lastView = key;
    listeners.viewchange.forEach((cb) => cb(view));
  };

  try {
    await chart.start();
  } catch (err) {
    chart.destroy();
    element.remove();
    throw err;
  }
  void chart.setupRenderLoop();

  return {
    element,
    update(settings) {
      chart.refresh(applySettings(chart.state, settings));
    },
    setData(data) {
      chart.setData(normalizeChartData(data));
    },
    view: {
      get: () => chart.getViewRange(),
      set: (range) => chart.setViewRange(range),
    },
    selection: {
      get: () => selection,
      set: (region) => chart.setDataSelection(region),
    },
    on(event, cb) {
      const set = listeners[event] as Set<typeof cb>;
      set.add(cb);
      return () => set.delete(cb);
    },
    integrate: (region) => chart.integrateRegion(region),
    toImage: ({ width, height, pixelRatio }) =>
      chart.captureImage({
        width,
        height,
        pixelRatio: pixelRatio ?? window.devicePixelRatio,
      }),
    destroy() {
      chart.destroy();
      element.remove();
      Object.values(listeners).forEach((set) => set.clear());
    },
  };
}

/** Write settings into chart state; returns which side effects are needed. */
function applySettings(
  state: GaussianChartState,
  settings: ChartSettings,
): RefreshFlags {
  const flags: RefreshFlags = {};
  const { style, axes, merge, pixelRatio } = settings;

  if (style) {
    if (style.sigma !== undefined) {
      state.sigmaSize = style.sigma;
      flags.merge = true;
    }
    if (style.opacityCut !== undefined) {
      state.opacityCut = style.opacityCut;
      flags.palette = true;
    }
    if (style.quantSteps !== undefined) {
      state.quantSteps = style.quantSteps;
      flags.palette = true;
    }
    if (style.paletteLevel !== undefined) {
      state.paletteLevel = style.paletteLevel;
      flags.palette = true;
    }
    if (style.palette) {
      const [c0, c1, c2] = style.palette;
      state.paletteColors = { c0, c1, c2 };
      flags.palette = true;
    }
    if (style.heatmap?.show !== undefined) {
      state.showHeatmap = style.heatmap.show;
      flags.presentation = true;
    }
    if (style.heatmap?.mode !== undefined) {
      state.heatmapRenderMode = style.heatmap.mode;
      flags.presentation = true;
    }
    if (style.line?.show !== undefined) state.showLine = style.line.show;
    if (style.line?.width !== undefined) state.lineWidth = style.line.width;
    if (style.line?.opacity !== undefined) {
      state.lineOpacity = style.line.opacity;
    }
    if (style.scatter?.show !== undefined) {
      state.showScatter = style.scatter.show;
    }
    if (style.debug?.quads !== undefined) {
      state.showGaussianQuads = style.debug.quads;
    }
    if (style.debug?.sigmaCircle !== undefined) {
      state.showGaussian3SigmaCircle = style.debug.sigmaCircle;
    }
  }
  if (axes?.x?.show !== undefined) state.showTimescale = axes.x.show;
  if (axes?.y?.show !== undefined) {
    state.showYAxis = axes.y.show;
    flags.layout = true;
  }
  if (merge?.thresholdSigmas !== undefined) {
    state.mergeThresholdSigmas = merge.thresholdSigmas;
    flags.merge = true;
  }
  if (pixelRatio !== undefined) {
    state.pixelRatio =
      pixelRatio === "device" ? window.devicePixelRatio : pixelRatio;
    flags.layout = true;
  }
  return flags;
}

/**
 * Convert chart input into the internal layout: interleaved [x, y, w],
 * sorted by x, with x normalized to [0, 1] (keeps GPU math precise).
 */
export function normalizeChartData(data: ChartData): LoadedData {
  let points: Float64Array;
  if ("points" in data) {
    if (data.points.length % 3 !== 0) {
      throw new Error("points length must be a multiple of 3 (x, y, weight)");
    }
    points = data.points;
  } else {
    const n = data.x.length;
    if (data.y.length !== n || (data.weight && data.weight.length !== n)) {
      throw new Error("x, y and weight must have the same length");
    }
    points = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      points[i * 3] = data.x[i]!;
      points[i * 3 + 1] = data.y[i]!;
      points[i * 3 + 2] = data.weight ? data.weight[i]! : 1;
    }
  }

  const n = points.length / 3;
  if (!isSortedByX(points, n)) points = sortByX(points, n);

  const xMin = n > 0 ? points[0]! : 0;
  const xMax = n > 0 ? points[(n - 1) * 3]! : 0;
  const xScale = xMax - xMin || 1;
  for (let i = 0; i < n; i++) {
    points[i * 3] = (points[i * 3]! - xMin) / xScale;
  }
  return { n, dataF64: points, xMin, xScale };
}

function isSortedByX(points: Float64Array, n: number) {
  for (let i = 1; i < n; i++) {
    if (points[i * 3]! < points[(i - 1) * 3]!) return false;
  }
  return true;
}

function sortByX(points: Float64Array, n: number) {
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => points[a * 3]! - points[b * 3]!);
  const sorted = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    sorted.set(points.subarray(order[i]! * 3, order[i]! * 3 + 3), i * 3);
  }
  return sorted;
}
