import { type ClientCache, type DatasetShape, type ViewQuery, emptyCache, generateDataset } from "./ripmap-model";
import {
  CELLS_KEY,
  COLUMNS_KEY,
  GRID_CELL_BYTES,
  GRID_PARAMS,
  type GridCell,
  type GridModel,
  type GridParams,
  type GridPlane,
  type GridQueryResult,
  type GridRender,
  MAX_COL_WIDTH,
  PAGE_RECORDS,
  buildGrid,
  runGridQuery,
  slot,
  slotLo,
} from "./grid-model";
import {
  GUTTER,
  PAD_Y,
  type Palette,
  STYLE,
  colors,
  el,
  fmtBytes,
  fmtCount,
  fmtNum,
  lowerBound,
  makeSprite,
  setupCanvas,
} from "./ripmap-page";

/**
 * Interactive explainer for `rust/progressive-grid` (`sparse_grid`): build the sparse moment grid
 * pyramid in-browser, then pan/zoom and watch which stored plane the client picks, how the view
 * decomposes into inside columns and edge chains, and the two rounds of byte-range reads.
 */

const GRID_STYLE = `
.rm .cand { width:100%; border-collapse:collapse; font-size:12px; margin:6px 0; }
.rm .cand td, .rm .cand th { padding:3px 6px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }
.rm .cand th { color:var(--muted); font-weight:600; }
.rm .cand td:first-child, .rm .cand th:first-child { text-align:left; }
.rm .cand tr.win td { font-weight:700; color:var(--accent); }
`;

const ROW = 24;
const MATRIX_ROW = 22;

type Hover =
  | { kind: "render"; render: GridRender; cell: GridCell | null }
  | { kind: "plane"; i: number; j: number; plane: GridPlane | null }
  | null;

interface State {
  shape: DatasetShape;
  n: number;
  params: GridParams;
  sigmaPx: number;
  threshold: number;
  minX: number;
  maxX: number;
  yBounds: [number, number] | null;
  showRaw: boolean;
  showCells: boolean;
  showGroups: boolean;
  showKde: boolean;
  warmCache: boolean;
}

export async function runGridPage(args: { output: HTMLElement }) {
  const { output } = args;
  output.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = STYLE + GRID_STYLE;
  output.appendChild(style);

  const root = el("div", "rm");
  output.appendChild(root);
  root.innerHTML = `
    <h1>Sparse grid explorer</h1>
    <p class="lead">An in-browser model of <code>rust/progressive-grid</code> (<code>sparse_grid</code>). The data's bounding
      box is cut into power-of-two grids of 2<sup>i</sup> columns × 2<sup>j</sup> rows, stored for several cell
      <em>shapes</em> (d = i − j), keeping only non-empty cells. Every cell is direct-addressed, so a view is two rounds
      of reads: column records, then cells. Pan and zoom to see which grid the client picks, how the view splits into
      inside columns and edge chains, and every byte range it reads.</p>
    <div class="controls panel" data-ref="controls"></div>
    <div class="stats" data-ref="stats"></div>
    <div class="main">
      <div class="panel">
        <h2>Screen — what the client renders</h2>
        <div class="canvas-wrap"><canvas data-ref="screen"></canvas></div>
        <div class="legend">
          <span><i class="sw" style="background:var(--plane)"></i>stored cell (one kernel each)</span>
          <span><i class="sw" style="background:var(--resolved)"></i>whole column fits as one kernel</span>
          <span><i class="sw" style="background:var(--leaf)"></i>raw points (edge column, or no plane fits)</span>
          <span><i class="sw" style="background:var(--group)"></i>merged kernel (2+ pieces)</span>
          <span><i class="sw" style="background:var(--kde)"></i>rendered KDE preview</span>
        </div>
        <h2 style="margin-top:14px">Column decomposition — shared x-axis, one row per x-level</h2>
        <div class="canvas-wrap"><canvas data-ref="cols"></canvas></div>
        <div class="legend">
          <span><i class="sw" style="background:var(--plane)"></i>inside column, rendered from cells</span>
          <span><i class="sw" style="background:var(--resolved)"></i>inside column, one kernel</span>
          <span><i class="sw" style="background:var(--boundary)"></i>straddles the edge → split into its 2 children</span>
          <span><i class="sw" style="background:var(--leaf)"></i>raw</span>
          <span><i class="sw" style="background:var(--yrange)"></i>record page read (round 1)</span>
          <span><i class="sw" style="background:var(--expand)"></i>record page from cache</span>
        </div>
        <div class="hint">Wheel: zoom x · Shift+wheel: zoom y (switches to explicit <code>y_bounds</code>) · Drag: pan · Double-click: overview · Hover the screen, the rows or the plane map for details.</div>
      </div>
      <div class="panel inspector" data-ref="inspector"></div>
    </div>
    <div class="panel section">
      <h2>Stored planes — x-level i (columns) × y-level j (rows)</h2>
      <div class="canvas-wrap"><canvas data-ref="matrix"></canvas></div>
      <div class="legend">
        <span><i class="sw" style="background:var(--plane)"></i>stored plane (shade = non-empty cells, log scale)</span>
        <span><i class="sw" style="background:var(--yrange)"></i>candidate x-levels</span>
        <span><i class="sw" style="background:var(--accent)"></i>chosen plane</span>
        <span><i class="sw" style="border:2px dashed var(--accent)"></i>used by edge-chain columns</span>
        <span><i class="sw" style="border:2px solid var(--leaf)"></i>rows needed but plane not stored (→ raw)</span>
      </div>
    </div>
    <div class="two section">
      <div class="panel"><h2>Static objects &amp; byte ranges read</h2><div class="objects" data-ref="objects"></div></div>
      <div class="panel"><h2>Round trips</h2><div class="rounds" data-ref="rounds"></div></div>
    </div>
    <div class="section steps">${STEPS}</div>
  `;
  const ref = <T extends HTMLElement>(name: string) => root.querySelector(`[data-ref="${name}"]`) as T;
  const controls = ref<HTMLDivElement>("controls");
  const statsEl = ref<HTMLDivElement>("stats");
  const screen = ref<HTMLCanvasElement>("screen");
  const colsCanvas = ref<HTMLCanvasElement>("cols");
  const matrix = ref<HTMLCanvasElement>("matrix");
  const inspector = ref<HTMLDivElement>("inspector");
  const objectsEl = ref<HTMLDivElement>("objects");
  const roundsEl = ref<HTMLDivElement>("rounds");

  const state: State = {
    shape: "noisy_spikes",
    n: 300_000,
    params: { ...GRID_PARAMS },
    sigmaPx: 14,
    threshold: 1,
    minX: 0,
    maxX: 1,
    yBounds: null,
    showRaw: true,
    showCells: true,
    showGroups: true,
    showKde: false,
    warmCache: false,
  };

  const build = () => buildGrid(generateDataset(state.shape, state.n), state.params);
  let model: GridModel = build();
  let cache: ClientCache = emptyCache();
  let result: GridQueryResult | null = null;
  let hover: Hover = null;
  const sprite = makeSprite();

  // ---------------------------------------------------------------- controls
  const select = (label: string, options: Array<[string, string]>, value: string, onChange: (v: string) => void) => {
    const wrap = el("label", "ctl");
    wrap.append(label);
    const s = document.createElement("select");
    for (const [v, t] of options) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      s.appendChild(o);
    }
    s.value = value;
    s.onchange = () => onChange(s.value);
    wrap.appendChild(s);
    controls.appendChild(wrap);
  };
  const slider = (label: string, min: number, max: number, step: number, value: number, onInput: (v: number) => void) => {
    const wrap = el("label", "ctl");
    const head = el("span");
    const out = document.createElement("output");
    head.append(`${label} `, out);
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    out.textContent = String(value);
    input.oninput = () => {
      out.textContent = input.value;
      onInput(Number(input.value));
    };
    wrap.append(head, input);
    controls.appendChild(wrap);
  };
  const check = (label: string, value: boolean, onChange: (v: boolean) => void) => {
    const wrap = el("label", "chk");
    const c = document.createElement("input");
    c.type = "checkbox";
    c.checked = value;
    c.onchange = () => onChange(c.checked);
    wrap.append(c, label);
    controls.appendChild(wrap);
  };
  const button = (label: string, onClick: () => void) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = onClick;
    controls.appendChild(b);
  };
  const rebuild = () => {
    model = build();
    cache = emptyCache();
    update();
  };

  select("Dataset", [["random_walk", "random walk (weighted)"], ["noisy_spikes", "white noise + spikes"]], state.shape, (v) => {
    state.shape = v as DatasetShape;
    rebuild();
  });
  select("Points", [["10000", "10K"], ["100000", "100K"], ["300000", "300K"], ["1000000", "1M"]], String(state.n), (v) => {
    state.n = Number(v);
    rebuild();
  });
  select("Cell shapes (d = i − j)", [["-2,12,2", "−2…12 step 2 (default)"], ["0,0,1", "0 only (square, quadtree-like)"], ["-2,12,1", "−2…12 step 1"]], "-2,12,2", (v) => {
    const [lo, hi, step] = v.split(",").map(Number);
    state.params = { ...state.params, minAspect: lo!, maxAspect: hi!, aspectStep: step! };
    rebuild();
  });
  select("min_points_per_cell", [["4", "4"], ["16", "16 (default)"], ["64", "64"]], String(state.params.minPointsPerCell), (v) => {
    state.params = { ...state.params, minPointsPerCell: Number(v) };
    rebuild();
  });
  select("col_leaf", [["64", "64"], ["256", "256 (default)"], ["1024", "1024"]], String(state.params.colLeaf), (v) => {
    state.params = { ...state.params, colLeaf: Number(v) };
    rebuild();
  });
  slider("σ px", 4, 40, 1, state.sigmaPx, (v) => {
    state.sigmaPx = v;
    update();
  });
  slider("merge threshold", 0.5, 3, 0.1, state.threshold, (v) => {
    state.threshold = v;
    update();
  });
  check("raw data", state.showRaw, (v) => {
    state.showRaw = v;
    draw();
  });
  check("cells & columns", state.showCells, (v) => {
    state.showCells = v;
    draw();
  });
  check("merged kernels", state.showGroups, (v) => {
    state.showGroups = v;
    draw();
  });
  check("KDE preview", state.showKde, (v) => {
    state.showKde = v;
    draw();
  });
  check("keep cache warm", state.warmCache, (v) => {
    state.warmCache = v;
    cache = emptyCache();
    update();
  });
  button("Overview", () => setView(0, 1, null));
  button("Zoom 20×", () => setView(0.45, 0.5, null));
  button("Y zoom 4×", () => {
    const r = result?.yRange;
    if (!r) return;
    const mid = (r[0] + r[1]) / 2;
    const half = (r[1] - r[0]) / 8;
    state.yBounds = [mid - half, mid + half];
    update();
  });

  function setView(minX: number, maxX: number, yBounds: [number, number] | null) {
    state.minX = minX;
    state.maxX = maxX;
    state.yBounds = yBounds;
    update();
  }

  // ---------------------------------------------------------------- geometry
  const geom = { W: 0, H: 0, plotW: 10, plotH: 10 };
  const measure = () => {
    geom.W = screen.clientWidth;
    geom.H = Math.round(Math.min(Math.max(geom.W * 0.45, 280), 520));
    geom.plotW = Math.max(geom.W - GUTTER, 10);
    geom.plotH = Math.max(geom.H - 2 * PAD_Y, 10);
    screen.style.height = `${geom.H}px`;
  };
  const sx = (x: number) => GUTTER + ((x - state.minX) / (state.maxX - state.minX)) * geom.plotW;
  const invX = (px: number) => state.minX + ((px - GUTTER) / geom.plotW) * (state.maxX - state.minX);
  const yr = () => result?.effectiveY ?? [0, 1];
  const sy = (y: number) => {
    const [y0, y1] = yr();
    return PAD_Y + (1 - (y - y0) / Math.max(y1 - y0, 1e-12)) * geom.plotH;
  };
  const invY = (py: number) => {
    const [y0, y1] = yr();
    return y0 + (1 - (py - PAD_Y) / geom.plotH) * (y1 - y0);
  };
  /** Geometric x-extent of column `c` at `level`. */
  const colSpan = (level: number, c: number): [number, number] => {
    const b = model.bounds;
    const lo = slotLo(b.x0, b.w, c, level);
    return [lo, lo + slot(b.w, level)];
  };

  function update() {
    measure();
    const query: ViewQuery = {
      minX: state.minX,
      maxX: state.maxX,
      yBounds: state.yBounds,
      screenW: geom.plotW,
      screenH: geom.plotH,
      sigmaPx: state.sigmaPx,
      mergeThreshold: state.threshold,
    };
    if (!state.warmCache) cache = emptyCache();
    result = runGridQuery(model, query, cache);
    hover = null;
    renderStats();
    renderObjects();
    renderRounds();
    renderInspector();
    draw();
  }

  function draw() {
    if (!result) return;
    drawScreen(result);
    drawColumns(result);
    drawMatrix(result);
  }

  // ---------------------------------------------------------------- screen
  function drawScreen(r: GridQueryResult) {
    const { W, H } = geom;
    const ctx = setupCanvas(screen, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = c.muted;
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let t = 0; t <= 4; t++) {
      const py = PAD_Y + (t / 4) * geom.plotH;
      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(GUTTER, py);
      ctx.lineTo(W, py);
      ctx.stroke();
      ctx.fillText(fmtNum(invY(py)), GUTTER - 6, py);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, 0, W - GUTTER, H);
    ctx.clip();
    if (state.showKde) drawKde(ctx, r, c);
    if (state.showRaw) drawRaw(ctx, c, W, H);
    if (state.showCells) drawRenders(ctx, r, c);
    if (state.showGroups) drawGroups(ctx, r, c);
    if (hover?.kind === "render") {
      const col = hover.render.col;
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 2;
      const box = hover.cell ? [hover.cell.rx0, hover.cell.rx1, hover.cell.ry0, hover.cell.ry1] : [col.minX, col.maxX, col.minY, col.maxY];
      const x0 = sx(box[0]!);
      ctx.strokeRect(x0, sy(box[3]!), Math.max(sx(box[1]!) - x0, 2), Math.max(sy(box[2]!) - sy(box[3]!), 2));
      ctx.lineWidth = 1;
    }
    ctx.restore();

    const rad = state.sigmaPx * state.threshold;
    ctx.strokeStyle = c.accent;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(W - rad - 12, rad + 12, rad, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = c.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText("merge threshold (σ·t)", W - 2 * rad - 18, 12);
    if (state.yBounds) {
      ctx.textAlign = "left";
      ctx.fillText("explicit y_bounds", GUTTER + 6, 6);
    }
  }

  function drawRaw(ctx: CanvasRenderingContext2D, c: Palette, W: number, H: number) {
    const { xs, ys } = model.data;
    const lo = lowerBound(xs, state.minX);
    const hi = lowerBound(xs, state.maxX + 1e-15);
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.ceil(W * dpr);
    const ph = Math.ceil(H * dpr);
    const counts = new Uint16Array(pw * ph);
    let max = 1;
    for (let i = lo; i < hi; i++) {
      const px = Math.floor(sx(xs[i]!) * dpr);
      const py = Math.floor(sy(ys[i]!) * dpr);
      if (px < 0 || px >= pw || py < 0 || py >= ph) continue;
      const idx = py * pw + px;
      const v = Math.min(counts[idx]! + 1, 65535);
      counts[idx] = v;
      if (v > max) max = v;
    }
    const img = ctx.createImageData(pw, ph);
    const [r, g, b] = c.dataRgb;
    const logMax = Math.log1p(max);
    for (let i = 0; i < counts.length; i++) {
      const v = counts[i]!;
      if (v === 0) continue;
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 40 + Math.round((Math.log1p(v) / logMax) * 150);
    }
    const off = document.createElement("canvas");
    off.width = pw;
    off.height = ph;
    off.getContext("2d")!.putImageData(img, 0, 0);
    ctx.drawImage(off, 0, 0, W, H);
  }

  function drawRenders(ctx: CanvasRenderingContext2D, r: GridQueryResult, c: Palette) {
    if (!r.chosen) return;
    let maxMass = 0;
    for (const rd of r.chosen.renders) if (rd.kind === "cells") for (const cell of rd.cells) maxMass = Math.max(maxMass, cell.mass);
    for (const rd of r.chosen.renders) {
      const col = rd.col;
      const x0 = sx(col.minX);
      const x1 = Math.max(sx(col.maxX), x0 + 1);
      if (rd.kind === "cells") {
        ctx.fillStyle = c.plane;
        for (const cell of rd.cells) {
          ctx.globalAlpha = 0.12 + 0.6 * Math.sqrt(cell.mass / Math.max(maxMass, 1e-12));
          const cx0 = sx(cell.rx0);
          const top = sy(cell.ry1);
          ctx.fillRect(cx0, top, Math.max(sx(cell.rx1) - cx0, 1), Math.max(sy(cell.ry0) - top, 1));
        }
        const [g0, g1] = colSpan(col.level, col.index);
        if (sx(g1) - sx(g0) >= 4) {
          ctx.globalAlpha = 0.35;
          ctx.strokeStyle = c.plane;
          ctx.strokeRect(sx(g0) + 0.5, sy(col.maxY) + 0.5, sx(g1) - sx(g0), Math.max(sy(col.minY) - sy(col.maxY), 1));
        }
        ctx.globalAlpha = 1;
      } else if (rd.kind === "node") {
        ctx.strokeStyle = c.resolved;
        ctx.strokeRect(x0 + 0.5, sy(col.maxY) + 0.5, x1 - x0, Math.max(sy(col.minY) - sy(col.maxY), 1));
        ctx.fillStyle = c.resolved;
        ctx.beginPath();
        ctx.arc(sx(col.x), sy(col.y), 2, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = c.leaf;
        const { xs, ys } = model.data;
        for (let i = col.start; i < col.end; i++) {
          const x = xs[i]!;
          if (x < state.minX || x > state.maxX) continue;
          ctx.fillRect(sx(x) - 1.5, sy(ys[i]!) - 1.5, 3, 3);
        }
        ctx.strokeStyle = c.leaf;
        if (rd.edge) ctx.setLineDash([2, 2]);
        ctx.strokeRect(x0 + 0.5, sy(col.maxY) + 0.5, x1 - x0, Math.max(sy(col.minY) - sy(col.maxY), 1));
        ctx.setLineDash([]);
      }
    }
  }

  function drawGroups(ctx: CanvasRenderingContext2D, r: GridQueryResult, c: Palette) {
    ctx.strokeStyle = c.group;
    ctx.fillStyle = c.group;
    for (const g of r.groups) {
      if (g.parts < 2) continue;
      const n = g.node;
      const x0 = sx(n.minX);
      const x1 = Math.max(sx(n.maxX), x0 + 1);
      const y0 = sy(n.maxY);
      const y1 = Math.max(sy(n.minY), y0 + 1);
      ctx.globalAlpha = 0.85;
      ctx.strokeRect(x0 - 1.5, y0 - 1.5, x1 - x0 + 3, y1 - y0 + 3);
      ctx.beginPath();
      ctx.arc(sx(n.x), sy(n.y), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawKde(ctx: CanvasRenderingContext2D, r: GridQueryResult, c: Palette) {
    const kernels: Array<[number, number, number, number, number, number]> = [];
    const push = (mass: number, x: number, y: number, cxx: number, cxy: number, cyy: number) => {
      if (mass <= 0) return;
      const s2 = state.sigmaPx ** 2;
      kernels.push([mass, sx(x), sy(y), (1 + cxx * r.xFactor ** 2) * s2, -cxy * r.xFactor * r.yFactor * s2, (1 + cyy * r.yFactor ** 2) * s2]);
    };
    for (const rd of r.chosen?.renders ?? []) if (rd.kind === "cells") for (const cell of rd.cells) push(cell.mass, cell.x, cell.y, cell.cxx, cell.cxy, cell.cyy);
    for (const g of r.groups) push(g.node.mass, g.node.x, g.node.y, g.node.cxx, g.node.cxy, g.node.cyy);
    if (kernels.length === 0) return;
    const peaks = kernels.map(([m, , , a, b, d]) => m / Math.sqrt(Math.max(a * d - b * b, 1e-9)));
    const ref = [...peaks].sort((p, q) => p - q)[Math.floor(peaks.length * 0.98)] ?? 1;
    ctx.save();
    for (const [i, [, cx, cy, a, b, d]] of kernels.entries()) {
      const l11 = Math.sqrt(a);
      const l21 = b / l11;
      const l22 = Math.sqrt(Math.max(d - l21 * l21, 1e-9));
      ctx.globalAlpha = Math.min(1, (peaks[i]! / ref) * 0.35);
      ctx.setTransform(l11, l21, 0, l22, cx, cy);
      ctx.drawImage(sprite.get(c.kde), -3, -3, 6, 6);
    }
    ctx.restore();
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ---------------------------------------------------------------- column decomposition
  /** Rows drawn: from the chosen candidate level down to the deepest level any piece reached. */
  const rowLevels = (r: GridQueryResult) => {
    if (!r.chosen) return [] as number[];
    const top = r.chosen.level;
    const deepest = Math.max(top, ...r.chosen.pieces.map((p) => p.col.level), ...r.pages.map((p) => p.level));
    return Array.from({ length: deepest - top + 1 }, (_, k) => top + k);
  };

  function drawColumns(r: GridQueryResult) {
    const W = colsCanvas.clientWidth;
    const levels = rowLevels(r);
    const H = Math.max(levels.length, 1) * ROW + 8;
    const ctx = setupCanvas(colsCanvas, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    if (!r.chosen) return;
    const top = levels[0]!;
    const rowY = (level: number) => 4 + (level - top) * ROW;
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";

    // Straddling ancestors of every piece below the top level.
    const straddling = new Set<string>();
    for (const p of r.chosen.pieces) {
      let c2 = p.col.index;
      for (let level = p.col.level - 1; level >= top; level--) {
        c2 >>= 1;
        straddling.add(`${level}:${c2}`);
      }
    }
    const fill = (level: number, c0: number, color: string, alpha: number, inset = 3) => {
      const [g0, g1] = colSpan(level, c0);
      const x0 = Math.max(sx(g0), GUTTER);
      const x1 = Math.min(Math.max(sx(g1), x0 + 2), W);
      if (x1 < GUTTER) return;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.fillRect(x0, rowY(level) + inset, Math.max(x1 - x0 - 0.5, 1.5), ROW - 2 * inset - 4);
      ctx.globalAlpha = 1;
    };
    for (const level of levels) {
      const y = rowY(level);
      ctx.fillStyle = c.muted;
      ctx.textAlign = "left";
      const plane = r.chosen.renders.find((rd) => rd.kind === "cells" && rd.col.level === level);
      const tag = level === r.chosen.level ? " ◀" : "";
      ctx.fillText(`i=${level}${tag}`, 4, y + ROW / 2 - 4);
      if (plane && plane.kind === "cells") ctx.fillText(`cells j=${plane.plane.j}`, 4, y + ROW / 2 + 7);
      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(GUTTER, y + ROW - 2);
      ctx.lineTo(W, y + ROW - 2);
      ctx.stroke();
      // Record pages read for this level (thin bar along the row bottom).
      for (const pg of r.pages.filter((p) => p.level === level)) {
        const [a0] = colSpan(level, pg.page * PAGE_RECORDS);
        const [, a1] = colSpan(level, Math.min(pg.page * PAGE_RECORDS + PAGE_RECORDS, 2 ** level) - 1);
        const x0 = Math.max(sx(a0), GUTTER);
        const x1 = Math.min(Math.max(sx(a1), x0 + 2), W);
        if (x1 < GUTTER || x0 > W) continue;
        ctx.fillStyle = pg.cached ? c.expand : getComputedStyle(root).getPropertyValue("--yrange").trim();
        ctx.fillRect(x0, y + ROW - 6, x1 - x0, 3);
      }
    }
    for (const key of straddling) {
      const [level, c0] = key.split(":").map(Number);
      fill(level!, c0!, c.boundary, 0.55);
    }
    for (const rd of r.chosen.renders) {
      const color = rd.kind === "cells" ? c.plane : rd.kind === "node" ? c.resolved : c.leaf;
      fill(rd.col.level, rd.col.index, color, hover?.kind === "render" && hover.render === rd ? 1 : 0.75);
    }
  }

  // ---------------------------------------------------------------- plane matrix
  const matrixGeom = { cell: 20, rows: 1 };
  function drawMatrix(r: GridQueryResult) {
    const W = matrix.clientWidth;
    const cols = model.maxLevel + 1;
    const maxJ = Math.max(1, ...model.planes.map((p) => p.j), ...(r.candidates.map((c) => neededJ(r, c.level) ?? 0)));
    const rows = maxJ + 1;
    const cell = Math.max(8, Math.min(40, Math.floor((W - GUTTER) / Math.max(cols, 1))));
    matrixGeom.cell = cell;
    matrixGeom.rows = rows;
    const H = rows * MATRIX_ROW + 28;
    const ctx = setupCanvas(matrix, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    const cx = (i: number) => GUTTER + i * cell;
    const cy = (j: number) => 4 + j * MATRIX_ROW;
    const yrange = getComputedStyle(root).getPropertyValue("--yrange").trim();

    // Candidate columns band.
    for (const cand of r.candidates) {
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = yrange;
      ctx.fillRect(cx(cand.level), 0, cell, rows * MATRIX_ROW + 6);
      ctx.globalAlpha = 1;
    }
    const maxCells = Math.max(1, ...model.planes.map((p) => p.cells.length));
    for (const p of model.planes) {
      ctx.globalAlpha = 0.15 + 0.75 * (Math.log1p(p.cells.length) / Math.log1p(maxCells));
      ctx.fillStyle = c.plane;
      ctx.fillRect(cx(p.i) + 1, cy(p.j) + 1, cell - 2, MATRIX_ROW - 2);
    }
    ctx.globalAlpha = 1;
    // Planes used by edge-chain columns (dashed) and the chosen plane (solid).
    const used = new Set<GridPlane>();
    for (const rd of r.chosen?.renders ?? []) if (rd.kind === "cells") used.add(rd.plane);
    ctx.strokeStyle = c.accent;
    for (const p of used) {
      const isChosen = p.i === r.chosen?.level;
      ctx.lineWidth = isChosen ? 3 : 1.5;
      ctx.setLineDash(isChosen ? [] : [3, 2]);
      ctx.strokeRect(cx(p.i) + 1.5, cy(p.j) + 1.5, cell - 3, MATRIX_ROW - 3);
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    // Needed rows that are not stored → raw.
    for (const cand of r.candidates) {
      const j = neededJ(r, cand.level);
      if (j === null || cand.plane) continue;
      ctx.strokeStyle = c.leaf;
      ctx.strokeRect(cx(cand.level) + 2, cy(Math.min(j, rows - 1)) + 2, cell - 4, MATRIX_ROW - 4);
    }
    ctx.lineWidth = 1;
    if (hover?.kind === "plane") {
      ctx.strokeStyle = c.ink;
      ctx.strokeRect(cx(hover.i) + 0.5, cy(hover.j) + 0.5, cell - 1, MATRIX_ROW - 1);
    }
    // Axis labels.
    ctx.fillStyle = c.muted;
    ctx.textAlign = "right";
    for (let j = 0; j < rows; j++) if (rows <= 16 || j % 2 === 0) ctx.fillText(`j=${j}`, GUTTER - 6, cy(j) + MATRIX_ROW / 2);
    ctx.textAlign = "center";
    for (let i = 0; i < cols; i++) if (cell >= 18 || i % 2 === 0) ctx.fillText(`${i}`, cx(i) + cell / 2, rows * MATRIX_ROW + 16);
    ctx.textAlign = "left";
    ctx.fillText("i →", cx(cols) + 6, rows * MATRIX_ROW + 16);
  }

  /** Smallest y-level whose rows fit beside the candidate's columns (whether stored or not). */
  function neededJ(r: GridQueryResult, level: number): number | null {
    const w = slot(model.bounds.w, level) * r.xFactor;
    const t = state.threshold;
    if (w >= t) return null;
    const hMax = Math.sqrt(t * t - w * w);
    for (let j = 0; j <= 30; j++) if (slot(model.bounds.h, j) * r.yFactorEstimate < hMax) return j;
    return null;
  }

  // ---------------------------------------------------------------- panels
  function renderStats() {
    if (!result) return;
    const r = result;
    const { xs } = model.data;
    const visible = lowerBound(xs, state.maxX + 1e-15) - lowerBound(xs, state.minX);
    const rawWindow = visible * 24;
    const counts = { cells: 0, node: 0, raw: 0 };
    let cellCount = 0;
    let cellKernels = 0;
    for (const rd of r.chosen?.renders ?? []) {
      counts[rd.kind]++;
      if (rd.kind === "cells") {
        cellCount += rd.cells.length;
        cellKernels += rd.kernels;
      }
    }
    const chosen = r.chosen?.plane;
    const tiles: Array<[string, string, string]> = [
      ["Visible points", fmtCount(visible), `raw window ${fmtBytes(rawWindow)}`],
      ["Grid used", chosen ? `${2 ** chosen.i}×${2 ** chosen.j}` : r.chosen ? "raw" : "—", chosen ? `plane (i=${chosen.i}, j=${chosen.j}) · shape d=${chosen.i - chosen.j}` : "no plane fits at this level"],
      ["Kernels emitted", fmtCount(r.kernelCount), `${fmtCount(cellCount)} cells → ${fmtCount(cellKernels)} merged · ${counts.node} column kernels · ${counts.raw} raw runs`],
      ["Bytes on the wire", fmtBytes(r.wireBytes), rawWindow > 0 ? `${((r.wireBytes / rawWindow) * 100).toFixed(1)}% of raw window` : ""],
      ["Range requests", String(r.requests.length), state.warmCache ? `${r.cacheHits} cache hits` : "cold cache"],
      ["Round trips", String(r.rounds), "sequential RTTs"],
      ["Storage", fmtBytes(model.totalBytes), `${(model.totalBytes / Math.max(model.rawBytes, 1)).toFixed(2)}× raw · ${model.planes.length} planes · ${model.objects.length} objects`],
    ];
    statsEl.innerHTML = tiles.map(([l, v, s]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  }

  function renderObjects() {
    if (!result) return;
    const phaseColor: Record<string, string> = { open: "var(--open)", frontier: "var(--yrange)", "planes+leaves": "" };
    let html = `<div class="k" style="color:var(--muted)">object</div><div class="num">size</div><div style="color:var(--muted)">bytes read (each mark is one range request)</div><div class="num">read</div>`;
    for (const obj of model.objects) {
      const reqs = result.requests.filter((q) => q.key === obj.key);
      const read = reqs.reduce((s, q) => s + q.len, 0);
      const dim = reqs.length === 0 ? " dim" : "";
      const marks = reqs
        .map((q) => {
          const color = phaseColor[q.phase] || (obj.key === CELLS_KEY ? "var(--plane)" : "var(--leaf)");
          return `<i style="left:${(q.offset / obj.bytes) * 100}%;width:${(q.len / obj.bytes) * 100}%;background:${color}" title="${q.phase}: offset ${q.offset}, ${fmtBytes(q.len)}"></i>`;
        })
        .join("");
      html += `<div class="name${dim}">${obj.key}</div><div class="num${dim}">${fmtBytes(obj.bytes)}</div><div class="bar${dim}">${marks}</div><div class="num${dim}">${reqs.length ? `${fmtBytes(read)} · ${reqs.length}×` : "—"}</div>`;
    }
    objectsEl.innerHTML = html;
  }

  function renderRounds() {
    if (!result) return;
    const text: Record<string, string> = {
      open: "open: manifest (bbox, per-level record offsets, plane table)",
      frontier: "column records: visible columns at both candidate levels + edge chains",
      "planes+leaves": "cells of the chosen planes + raw edge points (concurrent)",
    };
    const rounds = new Map<number, GridQueryResult["requests"]>();
    for (const q of result.requests) rounds.set(q.round, [...(rounds.get(q.round) ?? []), q]);
    roundsEl.innerHTML =
      rounds.size === 0
        ? `<div>Everything served from cache — 0 round trips.</div>`
        : [...rounds.entries()]
            .sort((x, y) => x[0] - y[0])
            .map(([round, reqs]) => {
              const objs = new Map<string, number>();
              for (const q of reqs) objs.set(q.key, (objs.get(q.key) ?? 0) + 1);
              const list = [...objs.entries()].map(([k, n]) => `<code>${k}</code>${n > 1 ? `×${n}` : ""}`).join(", ");
              const bytes = reqs.reduce((s, q) => s + q.len, 0);
              return `<div><b>RTT ${round + 1}</b><span>${text[reqs[0]!.phase]} — ${list} <span style="color:var(--muted)">(${fmtBytes(bytes)})</span></span></div>`;
            })
            .join("");
  }

  function candidateTable(r: GridQueryResult) {
    const rows = r.candidates
      .map((cand) => {
        const counts = { cells: 0, node: 0, raw: 0 };
        for (const rd of cand.renders) counts[rd.kind]++;
        const j = cand.plane ? `${cand.plane.j}` : `${neededJ(r, cand.level) ?? "—"} ✗`;
        return `<tr class="${cand === r.chosen ? "win" : ""}"><td>i=${cand.level}</td><td>${cand.w.toFixed(2)}</td><td>${cand.hMax.toFixed(2)}</td><td>${j}</td><td>${counts.cells}/${counts.node}/${counts.raw}</td><td>${fmtBytes(cand.bytes)}</td></tr>`;
      })
      .join("");
    return `<table class="cand"><tr><th>level</th><th>w (σ)</th><th>h max</th><th>rows j</th><th>cells/kern/raw</th><th>bytes</th></tr>${rows}</table>`;
  }

  function renderInspector() {
    if (!result) return;
    const r = result;
    const b = model.bounds;
    if (!hover) {
      const colW = r.chosen ? slot(b.w, r.chosen.level) * r.xFactor : 0;
      inspector.innerHTML = `
        <h2>This view</h2>
        <div>Screen factors: x ${fmtNum(r.xFactor)} σ/unit, y ${fmtNum(r.yFactor)} σ/unit (estimate before round 2:
          ${fmtNum(r.yFactorEstimate)}, from the inside columns' exact y-range).</div>
        <div style="margin-top:6px">Coarsest level with columns ≤ ${MAX_COL_WIDTH}σ wide: <b>i=${r.base}</b>
          (${fmtCount(2 ** r.base)} columns). The client compares it with the next level and keeps the one that reads fewer bytes:</div>
        ${candidateTable(r)}
        <div class="k">For each inside column: one kernel if its whole box fits (w² + h² &lt; t²) or it is far off-screen;
          otherwise the coarsest stored row level j with h = H/2<sup>j</sup>·f<sub>y</sub> &lt; √(t² − w²); otherwise raw.
          ✗ = the rows needed are finer than any stored plane at that level.</div>
        ${r.chosen ? `<div style="margin-top:8px">Chosen: columns of <b>i=${r.chosen.level}</b> (${colW.toFixed(2)}σ wide on screen),
          ${r.chosen.pieces.filter((p) => p.kind === "inside").length} inside columns, ${r.chosen.pieces.filter((p) => p.kind === "edge").length} raw edge columns.</div>` : ""}
        <div class="k" style="margin-top:10px">Dataset: ${fmtCount(model.data.xs.length)} points · ${model.maxLevel + 1} x-levels
          (finest: ${fmtCount(2 ** Math.max(model.maxLevel, 0))} columns ≈ ${fmtCount(Math.round(model.data.xs.length / 2 ** Math.max(model.maxLevel, 0)))} pts)
          · bbox y ∈ [${fmtNum(b.y0)}, ${fmtNum(b.y0 + b.h)}]</div>
        <div class="k">Shapes stored: ${model.aspects
          .map((a) => `d=${a.d}: ${a.levels.length ? `${a.levels.length} planes (i ${a.levels[0]![0]}…${a.levels[a.levels.length - 1]![0]})` : "none"}`)
          .join(" · ")}</div>`;
      return;
    }
    if (hover.kind === "plane") {
      const { i, j, plane } = hover;
      const cols = 2 ** i;
      if (!plane) {
        inspector.innerHTML = `<h2>Plane (i=${i}, j=${j})</h2><div>Not stored: ${fmtCount(cols)} × ${fmtCount(2 ** j)} cells.</div>
          <div class="k">Either its shape d=${i - j} is not one of the stored aspects, or no column qualified (cells averaging
          ≥ ${model.params.minColumnPointsPerCell} points), so a view needing it reads raw points.</div>`;
        return;
      }
      const avg = model.data.xs.length / Math.max(plane.cells.length, 1);
      inspector.innerHTML = `<h2>Plane (i=${i}, j=${j}) · d=${i - j}</h2>
        <div>${fmtCount(cols)} columns × ${fmtCount(2 ** j)} rows = ${fmtCount(cols * 2 ** j)} slots, of which
          <b>${fmtCount(plane.cells.length)}</b> are non-empty (${((plane.cells.length / (cols * 2 ** j)) * 100).toFixed(2)}%), avg ${avg.toFixed(1)} points each.</div>
        <div class="k" style="margin-top:6px">Cell size: ${fmtNum(slot(b.w, i))} × ${fmtNum(slot(b.h, j))} data units.</div>
        <div class="k">Bytes: <code>cells.bin</code> [${fmtCount(plane.offset)}, +${fmtBytes(plane.cells.length * GRID_CELL_BYTES)}) — ${GRID_CELL_BYTES} B per cell
          (u16 row + 14-byte quantized moments), column-major. Columns that failed the gate have an empty run.</div>
        <div class="k">Its CSC pointers are the ${model.planesByLevel[i]!.indexOf(plane) + 1}ᵗʰ of ${model.planesByLevel[i]!.length} cell starts in every
          level-${i} record (${model.recordBytes[i]} B each in <code>columns.bin</code>).</div>`;
      return;
    }
    const { render, cell } = hover;
    const col = render.col;
    const rb = model.recordBytes[col.level]!;
    const recordAt = model.levelOffsets[col.level]! + col.index * rb;
    const [g0, g1] = colSpan(col.level, col.index);
    const head = `<h2>Column ${fmtCount(col.index)} of level i=${col.level}</h2>
      <div class="k">Grid x-extent [${fmtNum(g0)}, ${fmtNum(g1)}) · exact node x ∈ [${fmtNum(col.minX)}, ${fmtNum(col.maxX)}], y ∈ [${fmtNum(col.minY)}, ${fmtNum(col.maxY)}]</div>
      <div class="k">${fmtCount(col.end - col.start)} points = <code>points.bin</code> [${fmtCount(col.start)}, ${fmtCount(col.end)})</div>
      <div class="k">Record: <code>columns.bin</code> [${fmtCount(recordAt)}, +${rb} B), page ${Math.floor(col.index / PAGE_RECORDS)} of level ${col.level}</div>`;
    let body = "";
    if (render.kind === "cells") {
      const p = render.plane;
      const s = p.colStarts[col.index]!;
      body = `<div style="margin-top:8px"><span class="tag" style="background:var(--plane)">cells</span> plane (i=${p.i}, j=${p.j}): ${render.cells.length} non-empty cells
        out of ${fmtCount(2 ** p.j)} rows → <code>cells.bin</code> [${fmtCount(p.offset + s * GRID_CELL_BYTES)}, +${fmtBytes(render.cells.length * GRID_CELL_BYTES)}),
        merged into ${render.kernels} kernel${render.kernels === 1 ? "" : "s"}</div>`;
      if (cell) {
        body += `<div class="eq">row ${cell.row}: y ∈ [${fmtNum(cell.ry0)}, ${fmtNum(cell.ry1)})
${cell.count} points · mass ${fmtNum(cell.mass)}
mean (${fmtNum(cell.x)}, ${fmtNum(cell.y)})
on screen: ${(slot(b.w, p.i) * r.xFactor).toFixed(2)}σ × ${(slot(b.h, p.j) * r.yFactor).toFixed(2)}σ</div>`;
      }
    } else if (render.kind === "node") {
      body = `<div style="margin-top:8px"><span class="tag" style="background:var(--resolved)">one kernel</span> ${render.reason === "fits" ? "the column's whole box fits the merge threshold" : "far off-screen in y (explicit y_bounds)"} — no cells read.</div>`;
    } else {
      body = `<div style="margin-top:8px"><span class="tag" style="background:var(--leaf)">raw</span> ${render.edge ? "straddles the viewport edge at the bottom of its chain: points are read and clipped to the view" : "no stored plane at this level has rows fine enough for this column (or the column failed that plane's density gate): its points are read"}.</div>`;
    }
    inspector.innerHTML = head + body;
  }

  // ---------------------------------------------------------------- hit testing
  function hitScreen(px: number, py: number): Hover {
    if (!result?.chosen) return null;
    const x = invX(px);
    const y = invY(py);
    const tol = (2 / geom.plotW) * (state.maxX - state.minX);
    for (const rd of result.chosen.renders) {
      const [g0, g1] = rd.kind === "cells" ? colSpan(rd.col.level, rd.col.index) : [rd.col.minX - tol, rd.col.maxX + tol];
      if (x < g0 || x >= g1) continue;
      if (rd.kind === "cells") return { kind: "render", render: rd, cell: rd.cells.find((c) => y >= c.ry0 && y < c.ry1) ?? null };
      return { kind: "render", render: rd, cell: null };
    }
    return null;
  }

  function hitColumns(px: number, py: number): Hover {
    if (!result?.chosen) return null;
    const levels = rowLevels(result);
    const level = levels[0]! + Math.floor((py - 4) / ROW);
    const x = invX(px);
    const tol = (3 / geom.plotW) * (state.maxX - state.minX);
    const rd = result.chosen.renders.find((r) => {
      if (r.col.level !== level) return false;
      const [g0, g1] = colSpan(r.col.level, r.col.index);
      return x >= g0 - tol && x <= g1 + tol;
    });
    return rd ? { kind: "render", render: rd, cell: null } : null;
  }

  function hitMatrix(px: number, py: number): Hover {
    const i = Math.floor((px - GUTTER) / matrixGeom.cell);
    const j = Math.floor((py - 4) / MATRIX_ROW);
    if (i < 0 || i > model.maxLevel || j < 0 || j >= matrixGeom.rows) return null;
    return { kind: "plane", i, j, plane: model.planes.find((p) => p.i === i && p.j === j) ?? null };
  }

  const setHover = (h: Hover) => {
    const same =
      (h === null && hover === null) ||
      (h?.kind === "render" && hover?.kind === "render" && h.render === hover.render && h.cell === hover.cell) ||
      (h?.kind === "plane" && hover?.kind === "plane" && h.i === hover.i && h.j === hover.j);
    if (same) return;
    hover = h;
    renderInspector();
    draw();
  };

  let drag: { px: number; py: number; minX: number; maxX: number; y: [number, number] | null } | null = null;
  screen.addEventListener("pointerdown", (e) => {
    drag = { px: e.offsetX, py: e.offsetY, minX: state.minX, maxX: state.maxX, y: state.yBounds };
    screen.setPointerCapture(e.pointerId);
  });
  screen.addEventListener("pointermove", (e) => {
    if (drag) {
      const span = drag.maxX - drag.minX;
      const dx = ((e.offsetX - drag.px) / geom.plotW) * span;
      const minX = Math.min(Math.max(drag.minX - dx, -span * 0.25), 1 + span * 0.25 - span);
      state.minX = minX;
      state.maxX = minX + span;
      if (drag.y) {
        const ySpan = drag.y[1] - drag.y[0];
        const dy = ((e.offsetY - drag.py) / geom.plotH) * ySpan;
        state.yBounds = [drag.y[0] + dy, drag.y[1] + dy];
      }
      scheduleUpdate();
    } else setHover(hitScreen(e.offsetX, e.offsetY));
  });
  screen.addEventListener("pointerup", () => {
    drag = null;
  });
  screen.addEventListener("pointerleave", () => {
    if (!drag) setHover(null);
  });
  screen.addEventListener("dblclick", () => setView(0, 1, null));
  screen.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0015);
      if (e.shiftKey) {
        const [y0, y1] = state.yBounds ?? yr();
        const anchor = invY(e.offsetY);
        state.yBounds = [anchor - (anchor - y0) * factor, anchor + (y1 - anchor) * factor];
      } else {
        const anchor = invX(e.offsetX);
        const span = Math.min(Math.max((state.maxX - state.minX) * factor, 1e-6), 1.5);
        const t = (anchor - state.minX) / (state.maxX - state.minX);
        state.minX = anchor - t * span;
        state.maxX = state.minX + span;
      }
      scheduleUpdate();
    },
    { passive: false },
  );
  colsCanvas.addEventListener("pointermove", (e) => setHover(hitColumns(e.offsetX, e.offsetY)));
  colsCanvas.addEventListener("pointerleave", () => setHover(null));
  matrix.addEventListener("pointermove", (e) => setHover(hitMatrix(e.offsetX, e.offsetY)));
  matrix.addEventListener("pointerleave", () => setHover(null));

  let pending = false;
  function scheduleUpdate() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      update();
    });
  }
  new ResizeObserver(() => scheduleUpdate()).observe(screen.parentElement!);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => draw());
  update();
}

const STEPS = `
  <div class="panel"><h3>1 · one sparse grid per cell shape</h3>
    The bounding box is cut into 2<sup>i</sup> columns × 2<sup>j</sup> rows. Instead of square cells only, grids are kept for
    several shapes d = i − j (by default −2, 0, 2, …, 12: from rows 4× finer than columns to columns 4096× finer than rows).
    Only non-empty cells are stored, as shared 16-byte cells (u16 row + quantized moments). Each grid may hold at most N/16
    cells; that budget goes to the densest columns first (any column averaging ≥ 4 points per cell qualifies), so on evenly
    dense data whole grids are kept and on uneven data the dense columns still get fine rows.</div>
  <div class="panel"><h3>2 · column records point into every grid</h3>
    Points are stored in x order, so a column at any level is one run of <code>points.bin</code>. Each level has one record per
    column: the column's exact moment node, its first point, and its first cell in <em>every</em> grid of that level (CSC
    pointers). A plane's cells are column-major, so a run of neighbouring columns is one contiguous range of
    <code>cells.bin</code>.</div>
  <div class="panel"><h3>3 · query: two rounds, no tree walk</h3>
    The column width comes straight from the zoom: the client takes the coarsest level whose columns are ≤ 0.9σ wide and the
    next one. Round 1 reads their records, plus the few columns of finer levels that can contain the view's edges. Columns
    inside the view are summarized; a column straddling an edge splits into its children, down to raw points. The client then
    picks, per level, the coarsest stored rows that fit (w² + h² &lt; t²) and keeps the level that reads fewer bytes. Round 2
    reads those cells and the raw edge points together.</div>
  <div class="panel"><h3>4 · where it struggles</h3>
    Deep y-zooms on white noise still go raw: at those rows a column has fewer than 4 points per non-empty row, so none
    qualifies (Y zoom 4× after Zoom 20×). Try the "0 only" shape set to see the quadtree's problem come back
    when you zoom 20× on white noise.</div>`;

