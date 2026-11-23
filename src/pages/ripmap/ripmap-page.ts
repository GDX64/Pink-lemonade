import {
  type ClientCache,
  type DatasetShape,
  DEFAULT_PARAMS,
  type FrontierItem,
  MAX_PLANE_BLOCK_WIDTH,
  type PlaneCell,
  type QueryResult,
  type RipmapModel,
  type RipmapParams,
  type Strategy,
  TREE_PARAMS,
  type TraceEntry,
  type ViewQuery,
  blockPointsAt,
  buildRipmap,
  CELL_BYTES,
  emptyCache,
  generateDataset,
  recordBytesFor,
  runQuery,
} from "./ripmap-model";
import {
  QUAD_PARAMS,
  type QuadModel,
  type QuadNode,
  type QuadParams,
  RECORD_BYTES,
  buildQuadtree,
  runQuadQuery,
} from "./quadtree-model";

/** Explorer modes: the two x-hierarchies (ripmap, x-tree) and the 2D moment quadtree. */
export type PageStrategy = Strategy | "quadtree";
type AnyModel = RipmapModel | QuadModel;

/**
 * Interactive explainer for the hierarchy-based progressive strategies (`progressive-ripmap`,
 * the `progressive-tree` x-tree, and the `progressive-quadtree` moment quadtree): build a
 * dataset in-browser, then pan/zoom a viewport and
 * watch which hierarchy nodes resolve, which x-blocks are served through y-bin planes (ripmap
 * only), which leaves fall back to raw points, how the frame greedily merges everything into
 * kernels, and which byte ranges are fetched.
 */

export const STYLE = `
.rm { --bg:#f6f7fb; --panel:#ffffff; --ink:#1b2433; --muted:#5d6b80; --line:#dde3ec; --grid:#eef1f6;
  --plane:#7c4dff; --resolved:#0f9d8a; --leaf:#f07c1f; --expand:#b7c2d1; --boundary:#8a96a8;
  --open:#64748b; --yrange:#2f7de1; --data:#1b2433; --kde:#d6336c; --accent:#d6336c; --group:#c2255c;
  font-family: ui-sans-serif, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg);
  min-height:100vh; padding:20px 24px 48px; box-sizing:border-box; }
@media (prefers-color-scheme: dark) {
  .rm { --bg:#10141b; --panel:#171c25; --ink:#e6ebf2; --muted:#93a0b4; --line:#2a3240; --grid:#1f2631;
    --plane:#a58bff; --resolved:#2cc7b0; --leaf:#ff9b4a; --expand:#4a5566; --boundary:#7d8899;
    --open:#94a3b8; --yrange:#5ea2ff; --data:#e6ebf2; --kde:#ff6b9a; --accent:#ff6b9a; --group:#ff8fb3; }
}
.rm * { box-sizing:border-box; }
.rm h1 { font-size:22px; margin:0 0 4px; letter-spacing:-0.01em; }
.rm h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 10px; font-weight:650; }
.rm p.lead { margin:0 0 16px; color:var(--muted); max-width:900px; line-height:1.5; }
.rm code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size:.92em; }
.rm .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.rm .controls { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:flex-end; margin-bottom:14px; }
.rm .ctl { display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted); }
.rm .ctl select, .rm .ctl input[type=range] { font:inherit; color:var(--ink); }
.rm .ctl select { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 6px; font-size:13px; }
.rm .ctl output { color:var(--ink); font-variant-numeric:tabular-nums; }
.rm button { font:inherit; font-size:13px; border:1px solid var(--line); background:var(--panel); color:var(--ink);
  border-radius:6px; padding:5px 10px; cursor:pointer; }
.rm button:hover { border-color:var(--accent); }
.rm label.chk { display:flex; gap:6px; align-items:center; font-size:13px; color:var(--ink); }
.rm .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:14px; }
.rm .stat { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.rm .stat .v { font-size:20px; font-weight:650; font-variant-numeric:tabular-nums; }
.rm .stat .l { font-size:12px; color:var(--muted); }
.rm .stat .s { font-size:11px; color:var(--muted); margin-top:2px; }
.rm .main { display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:14px; align-items:start; }
@media (max-width: 1000px) { .rm .main { grid-template-columns:minmax(0,1fr); } }
.rm canvas { display:block; width:100%; }
.rm .canvas-wrap { position:relative; }
.rm .hint { font-size:12px; color:var(--muted); margin-top:6px; }
.rm .legend { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12px; margin:8px 0 2px; }
.rm .legend span { display:inline-flex; align-items:center; gap:6px; }
.rm .sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
.rm .inspector { font-size:13px; line-height:1.55; min-height:220px; }
.rm .inspector .k { color:var(--muted); }
.rm .inspector .eq { font-family: ui-monospace, Consolas, monospace; font-size:11px; overflow-x:auto; background:var(--grid);
  border-radius:6px; padding:6px 8px; margin:6px 0; white-space:pre; }
.rm .tag { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:650; color:#fff; }
.rm .objects { display:grid; grid-template-columns:190px 80px minmax(0,1fr) 90px; gap:3px 10px; align-items:center; font-size:12px; }
.rm .objects .name { font-family: ui-monospace, Consolas, monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rm .objects .num { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
.rm .objects .bar { position:relative; height:12px; background:var(--grid); border-radius:3px; overflow:hidden; }
.rm .objects .bar i { position:absolute; top:0; bottom:0; min-width:2px; }
.rm .objects .dim { opacity:.45; }
.rm .rounds { font-size:12px; display:flex; flex-direction:column; gap:4px; }
.rm .rounds div { display:flex; gap:8px; align-items:baseline; }
.rm .rounds b { min-width:54px; font-variant-numeric:tabular-nums; }
.rm .steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.rm .steps .panel { font-size:13px; line-height:1.55; }
.rm .steps h3 { margin:0 0 6px; font-size:14px; }
.rm .section { margin-top:14px; }
.rm .two { display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:14px; }
@media (max-width: 1000px) { .rm .two { grid-template-columns:minmax(0,1fr); } }
`;

export const GUTTER = 88; // left label gutter shared by the screen and the hierarchy canvas
export const PAD_Y = 12;
const LEVEL_ROW = 26;
const QUAD_ROW = 34;

interface State {
  shape: DatasetShape;
  n: number;
  params: RipmapParams;
  quad: QuadParams;
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

type Hover =
  | { kind: "item"; item: FrontierItem; cell: PlaneCell | null }
  | { kind: "trace"; entry: TraceEntry }
  | null;

export async function runRipmapPage(args: { output: HTMLElement; strategy?: PageStrategy }) {
  const { output } = args;
  const strategy: PageStrategy = args.strategy ?? "ripmap";
  const isTree = strategy === "tree";
  const isQuad = strategy === "quadtree";
  /** Strategies without moment planes (resolve-or-expand only). */
  const noPlanes = strategy !== "ripmap";
  const text = TEXT[strategy];
  output.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = STYLE;
  output.appendChild(style);

  const root = el("div", "rm");
  output.appendChild(root);
  root.innerHTML = `
    <h1>${text.title}</h1>
    <p class="lead">${text.lead}</p>
    <div class="controls panel" data-ref="controls"></div>
    <div class="stats" data-ref="stats"></div>
    <div class="main">
      <div class="panel">
        <h2>Screen — what the client renders</h2>
        <div class="canvas-wrap"><canvas data-ref="screen"></canvas></div>
        <div class="legend">${text.screenLegend}</div>
        <h2 style="margin-top:14px">${isQuad ? "Quadtree depths — shared x-axis, y compressed per row" : "Hierarchy traversal — shared x-axis"}</h2>
        <div class="canvas-wrap"><canvas data-ref="tree"></canvas></div>
        <div class="legend">${text.treeLegend}</div>
        <div class="hint">Wheel: zoom x · Shift+wheel: zoom y (switches to explicit <code>y_bounds</code>) · Drag: pan · Double-click: overview · Hover anything for the decision math.</div>
      </div>
      <div class="panel inspector" data-ref="inspector"></div>
    </div>
    <div class="two section">
      <div class="panel"><h2>Static objects &amp; byte ranges read</h2><div class="objects" data-ref="objects"></div></div>
      <div class="panel"><h2>Round trips</h2><div class="rounds" data-ref="rounds"></div></div>
    </div>
    <div class="section steps" data-ref="steps"></div>
  `;
  const ref = <T extends HTMLElement>(name: string) => root.querySelector(`[data-ref="${name}"]`) as T;
  const controls = ref<HTMLDivElement>("controls");
  const statsEl = ref<HTMLDivElement>("stats");
  const screen = ref<HTMLCanvasElement>("screen");
  const tree = ref<HTMLCanvasElement>("tree");
  const inspector = ref<HTMLDivElement>("inspector");
  const objectsEl = ref<HTMLDivElement>("objects");
  const roundsEl = ref<HTMLDivElement>("rounds");
  ref<HTMLDivElement>("steps").innerHTML = text.steps;

  const state: State = {
    shape: "random_walk",
    n: 300_000,
    // A small ripmap root cap makes the level-by-level descent visible; the Rust default is 1024.
    // The x-tree uses `PagedTreeStrategy::default()` (fanout 32, root cap = fanout).
    params: isTree ? { ...TREE_PARAMS } : { ...DEFAULT_PARAMS, rootCap: 16 },
    // A small root cap makes the depth-by-depth descent visible; the Rust default is 1024.
    quad: { ...QUAD_PARAMS, rootCap: 21 },
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

  const build = (): AnyModel => {
    const data = generateDataset(state.shape, state.n);
    return isQuad ? buildQuadtree(data, state.quad) : buildRipmap(data, state.params, strategy as Strategy);
  };
  let model: AnyModel = build();
  /** The x-hierarchy model (ripmap / x-tree modes only). */
  const rm = () => model as RipmapModel;
  const qm = () => model as QuadModel;
  /** x-sorted points (the quadtree stores its own copy in Z order). */
  const sortedData = () => (model.strategy === "quadtree" ? model.sorted : model.data);
  let cache: ClientCache = emptyCache();
  let result: QueryResult | null = null;
  let hover: Hover = null;
  const sprite = makeSprite();

  // ---------------------------------------------------------------- controls
  const select = (label: string, options: Array<[string, string]>, value: string, onChange: (v: string) => void) => {
    const wrap = el("label", "ctl");
    wrap.append(label);
    const s = document.createElement("select");
    for (const [v, text] of options) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = text;
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
  if (isQuad) {
    select("Leaf points", [["8", "8"], ["32", "32 (default)"], ["128", "128"]], String(state.quad.leafPoints), (v) => {
      state.quad.leafPoints = Number(v);
      rebuild();
    });
    select("Root cap (inlined nodes)", [["1", "1 (root only)"], ["21", "21"], ["1024", "1024 (Rust default)"]], String(state.quad.rootCap), (v) => {
      state.quad.rootCap = Number(v);
      rebuild();
    });
  } else if (isTree) {
    select("Fanout (= root cap)", [["4", "4"], ["8", "8"], ["16", "16"], ["32", "32 (default)"]], String(state.params.fanout), (v) => {
      state.params.fanout = Number(v);
      state.params.rootCap = Number(v);
      rebuild();
    });
  } else {
    select("Fanout", [["2", "2"], ["4", "4 (default)"], ["8", "8"]], String(state.params.fanout), (v) => {
      state.params.fanout = Number(v);
      rebuild();
    });
    select("Root cap", [["16", "16"], ["128", "128"], ["1024", "1024 (Rust default)"]], String(state.params.rootCap), (v) => {
      state.params.rootCap = Number(v);
      rebuild();
    });
    select("min_points_per_bin", [["4", "4"], ["16", "16 (default)"], ["64", "64"]], String(state.params.minPointsPerBin), (v) => {
      state.params.minPointsPerBin = Number(v);
      rebuild();
    });
  }
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
  check("frontier cells", state.showCells, (v) => {
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
  // Canvas size is measured once per update (`measure`), never inside per-point transforms:
  // reading `clientWidth` there forces a synchronous layout for every projected point.
  const geom = { W: 0, H: 0, plotW: 10, plotH: 10 };
  const measure = () => {
    geom.W = screen.clientWidth;
    geom.H = Math.round(Math.min(Math.max(geom.W * 0.45, 280), 520));
    geom.plotW = Math.max(geom.W - GUTTER, 10);
    geom.plotH = Math.max(geom.H - 2 * PAD_Y, 10);
    screen.style.height = `${geom.H}px`;
  };
  const plotW = () => geom.plotW;
  const plotH = () => geom.plotH;
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

  function update() {
    measure();
    const query: ViewQuery = {
      minX: state.minX,
      maxX: state.maxX,
      yBounds: state.yBounds,
      screenW: plotW(),
      screenH: plotH(),
      sigmaPx: state.sigmaPx,
      mergeThreshold: state.threshold,
    };
    if (!state.warmCache) cache = emptyCache();
    result = model.strategy === "quadtree" ? runQuadQuery(model, query, cache) : runQuery(model, query, cache);
    hover = null;
    renderStats();
    renderObjects();
    renderRounds();
    renderInspector();
    draw();
  }

  // ---------------------------------------------------------------- drawing
  function draw() {
    if (!result) return;
    drawScreen(result);
    drawTree(result);
  }

  function drawScreen(r: QueryResult) {
    const { W, H } = geom;
    const ctx = setupCanvas(screen, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);

    // y axis labels
    ctx.fillStyle = c.muted;
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let t = 0; t <= 4; t++) {
      const py = PAD_Y + (t / 4) * plotH();
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
    if (state.showCells) drawFrontier(ctx, r, c);
    if (state.showGroups) drawGroups(ctx, r, c);

    // hover highlight
    if (hover) {
      const node = hover.kind === "item" ? hover.item.node : hover.entry.node;
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 2;
      const x0 = sx(node.minX);
      const x1 = Math.max(sx(node.maxX), x0 + 2);
      if (hover.kind === "item" && hover.cell) {
        ctx.strokeRect(x0, sy(hover.cell.lo + hover.cell.h), x1 - x0, Math.max(sy(hover.cell.lo) - sy(hover.cell.lo + hover.cell.h), 2));
      } else {
        ctx.strokeRect(x0, sy(node.maxY), x1 - x0, Math.max(sy(node.minY) - sy(node.maxY), 2));
      }
      ctx.lineWidth = 1;
    }
    ctx.restore();

    // σ / threshold reference
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
    const { xs, ys } = sortedData();
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

  function drawFrontier(ctx: CanvasRenderingContext2D, r: QueryResult, c: Palette) {
    if (isQuad) {
      // The classic quadrant grid of every visited node (faint), under the tight boxes.
      ctx.strokeStyle = c.boundary;
      ctx.globalAlpha = 0.35;
      for (const entries of r.trace) {
        for (const e of entries) {
          const q = e.node as QuadNode;
          ctx.strokeRect(sx(q.rx0) + 0.5, sy(q.ry1) + 0.5, sx(q.rx1) - sx(q.rx0), sy(q.ry0) - sy(q.ry1));
        }
      }
      ctx.globalAlpha = 1;
    }
    let maxDensity = 0;
    for (const item of r.items) {
      if (item.kind !== "plane") continue;
      for (const cell of item.cells) maxDensity = Math.max(maxDensity, cell.mass);
    }
    for (const item of r.items) {
      const node = item.node;
      const x0 = sx(node.minX);
      const x1 = Math.max(sx(node.maxX), x0 + 1);
      if (item.kind === "plane") {
        for (const cell of item.cells) {
          const top = sy(cell.lo + cell.h);
          const bot = sy(cell.lo);
          if (cell.count > 0) {
            ctx.globalAlpha = 0.12 + 0.6 * Math.sqrt(cell.mass / Math.max(maxDensity, 1e-12));
            ctx.fillStyle = c.plane;
            ctx.fillRect(x0, top, x1 - x0, bot - top);
          }
        }
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = c.plane;
        ctx.strokeRect(x0 + 0.5, sy(node.maxY) + 0.5, x1 - x0 - 1, sy(node.minY) - sy(node.maxY));
        ctx.globalAlpha = 1;
      } else if (item.kind === "resolved") {
        ctx.strokeStyle = c.resolved;
        ctx.globalAlpha = 0.9;
        ctx.strokeRect(x0 + 0.5, sy(node.maxY) + 0.5, x1 - x0, Math.max(sy(node.minY) - sy(node.maxY), 1));
        ctx.fillStyle = c.resolved;
        ctx.beginPath();
        ctx.arc(sx(node.x), sy(node.y), 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = c.leaf;
        const { xs, ys } = model.data;
        for (let i = node.start; i < node.end; i++) {
          const x = xs[i]!;
          if (x < state.minX || x > state.maxX) continue;
          ctx.fillRect(sx(x) - 1.5, sy(ys[i]!) - 1.5, 3, 3);
        }
        ctx.strokeStyle = c.leaf;
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(x0 + 0.5, sy(node.maxY) + 0.5, x1 - x0, Math.max(sy(node.minY) - sy(node.maxY), 1));
        ctx.setLineDash([]);
      }
    }
  }

  function drawGroups(ctx: CanvasRenderingContext2D, r: QueryResult, c: Palette) {
    // Kernels built from more than one piece: outline the merged bounding box and mark the mean.
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

  function drawKde(ctx: CanvasRenderingContext2D, r: QueryResult, c: Palette) {
    // Every emitted kernel: covariance (σ units) = I + Σ·diag(fx, fy)², drawn as a Gaussian sprite.
    const kernels: Array<[number, number, number, number, number, number]> = [];
    const push = (mass: number, x: number, y: number, cxx: number, cxy: number, cyy: number) => {
      if (mass <= 0) return;
      const s2 = state.sigmaPx ** 2;
      const a = (1 + cxx * r.xFactor ** 2) * s2;
      const b = -cxy * r.xFactor * r.yFactor * s2; // screen y points down
      const d = (1 + cyy * r.yFactor ** 2) * s2;
      kernels.push([mass, sx(x), sy(y), a, b, d]);
    };
    // Exactly the frame's kernels: plane cells plus the greedily merged groups.
    for (const item of r.items) {
      if (item.kind !== "plane") continue;
      for (const cell of item.cells) push(cell.mass, cell.x, cell.y, cell.cxx, cell.cxy, cell.cyy);
    }
    for (const g of r.groups) {
      const n = g.node;
      push(n.mass, n.x, n.y, n.cxx, n.cxy, n.cyy);
    }
    if (kernels.length === 0) return;
    const peaks = kernels.map(([m, , , a, b, d]) => m / Math.sqrt(Math.max(a * d - b * b, 1e-9)));
    const sorted = [...peaks].sort((p, q) => p - q);
    const ref = sorted[Math.floor(sorted.length * 0.98)] ?? 1;
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

  function drawTree(r: QueryResult) {
    if (model.strategy === "quadtree") {
      drawQuadDepths(r, model);
      return;
    }
    const rows = model.levelCounts.length;
    const W = tree.clientWidth;
    const H = (rows + 1) * LEVEL_ROW + 8;
    const ctx = setupCanvas(tree, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";

    const rowY = (level: number) => 4 + (rows - 1 - level) * LEVEL_ROW;
    for (let level = rows - 1; level >= 0; level--) {
      const y = rowY(level);
      ctx.fillStyle = c.grid;
      ctx.fillRect(GUTTER, y + 3, W - GUTTER, LEVEL_ROW - 6);
      ctx.fillStyle = c.ink;
      ctx.textAlign = "left";
      ctx.fillText(`ℓ${level}${level === rows - 1 ? " root" : ""}`, 4, y + LEVEL_ROW / 2 - 6);
      ctx.fillStyle = c.muted;
      ctx.fillText(
        `${fmtCount(blockPointsAt(rm().params, level))}p${isTree ? "" : ` · k≤${rm().kmax[level]}`}`,
        4,
        y + LEVEL_ROW / 2 + 6,
      );
      ctx.save();
      ctx.beginPath();
      ctx.rect(GUTTER, y, W - GUTTER, LEVEL_ROW);
      ctx.clip();
      for (const entry of r.trace[level] ?? []) {
        const x0 = Math.max(sx(entry.node.minX), GUTTER - 2);
        const x1 = Math.min(Math.max(sx(entry.node.maxX), x0 + 1), W + 2);
        ctx.fillStyle = traceColor(entry, c);
        ctx.fillRect(x0, y + 3, Math.max(x1 - x0 - 0.5, 0.75), LEVEL_ROW - 6);
        if (entry.decision.kind === "plane" && x1 - x0 > 16) {
          ctx.fillStyle = "#fff";
          ctx.textAlign = "center";
          ctx.fillText(`k${entry.decision.k}`, (x0 + x1) / 2, y + LEVEL_ROW / 2);
        }
        if (hover?.kind === "trace" && hover.entry === entry) {
          ctx.strokeStyle = c.accent;
          ctx.lineWidth = 2;
          ctx.strokeRect(x0, y + 2, x1 - x0, LEVEL_ROW - 4);
          ctx.lineWidth = 1;
        }
      }
      ctx.restore();
    }
    // raw points row: leaves that fell back to points.bin
    const y = rowY(-1);
    ctx.fillStyle = c.ink;
    ctx.textAlign = "left";
    ctx.fillText("points.bin", 4, y + LEVEL_ROW / 2);
    ctx.fillStyle = c.grid;
    ctx.fillRect(GUTTER, y + 3, W - GUTTER, LEVEL_ROW - 6);
    ctx.fillStyle = c.leaf;
    for (const item of r.items) {
      if (item.kind !== "leaf") continue;
      const x0 = Math.max(sx(item.node.minX), GUTTER);
      const x1 = Math.min(Math.max(sx(item.node.maxX), x0 + 1), W);
      ctx.fillRect(x0, y + 3, x1 - x0, LEVEL_ROW - 6);
    }
  }

  /** One row per depth: each node drawn as its quadrant, y mapped into the row. */
  function drawQuadDepths(r: QueryResult, m: QuadModel) {
    const rows = m.levelCounts.length;
    const W = tree.clientWidth;
    const H = (rows + 1) * QUAD_ROW + 8;
    const ctx = setupCanvas(tree, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px ui-sans-serif, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    const rootNode = m.levels[0]?.[0];
    for (let depth = 0; depth < rows; depth++) {
      const y = 4 + depth * QUAD_ROW;
      const info = m.depthInfo[depth]!;
      ctx.fillStyle = c.grid;
      ctx.fillRect(GUTTER, y + 2, W - GUTTER, QUAD_ROW - 4);
      ctx.fillStyle = c.ink;
      ctx.textAlign = "left";
      ctx.fillText(`d${depth}${depth === 0 ? " root" : ""}`, 4, y + QUAD_ROW / 2 - 6);
      ctx.fillStyle = c.muted;
      ctx.fillText(
        `${fmtCount(info.count)} · ${info.inline ? "manifest" : "nodes.bin"}`,
        4,
        y + QUAD_ROW / 2 + 7,
      );
      if (!rootNode) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(GUTTER, y, W - GUTTER, QUAD_ROW);
      ctx.clip();
      for (const entry of r.trace[depth] ?? []) {
        const [x0, x1, y0, y1] = quadRowBox(entry.node as QuadNode, rootNode, y, W);
        ctx.fillStyle = traceColor(entry, c);
        ctx.fillRect(x0, y0, Math.max(x1 - x0 - 0.5, 0.75), Math.max(y1 - y0 - 0.5, 0.75));
        if (hover?.kind === "trace" && hover.entry === entry) {
          ctx.strokeStyle = c.accent;
          ctx.lineWidth = 2;
          ctx.strokeRect(x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2);
          ctx.lineWidth = 1;
        }
      }
      ctx.restore();
    }
    const y = 4 + rows * QUAD_ROW;
    ctx.fillStyle = c.ink;
    ctx.textAlign = "left";
    ctx.fillText("points.bin", 4, y + QUAD_ROW / 2);
    ctx.fillStyle = c.grid;
    ctx.fillRect(GUTTER, y + 2, W - GUTTER, QUAD_ROW - 4);
    ctx.fillStyle = c.leaf;
    for (const item of r.items) {
      if (item.kind !== "leaf" || !rootNode) continue;
      const [x0, x1, y0, y1] = quadRowBox(item.node as QuadNode, rootNode, y, W);
      ctx.fillRect(x0, y0, Math.max(x1 - x0, 1), Math.max(y1 - y0, 1));
    }
  }

  /** Screen box of quadrant `q` inside a depth row starting at `rowTop` (y spans the root region). */
  function quadRowBox(q: QuadNode, rootNode: QuadNode, rowTop: number, W: number): [number, number, number, number] {
    const x0 = Math.max(sx(q.rx0), GUTTER - 2);
    const x1 = Math.min(Math.max(sx(q.rx1), x0 + 1), W + 2);
    const span = Math.max(rootNode.ry1 - rootNode.ry0, 1e-12);
    const inner = QUAD_ROW - 6;
    const y0 = rowTop + 3 + (1 - (q.ry1 - rootNode.ry0) / span) * inner;
    const y1 = rowTop + 3 + (1 - (q.ry0 - rootNode.ry0) / span) * inner;
    return [x0, x1, y0, Math.max(y1, y0 + 1)];
  }

  /** An expanded node with nothing below it: its raw points come from `points.bin`. */
  function isRawLeaf(entry: TraceEntry) {
    if (entry.decision.kind !== "expand") return false;
    return isQuad ? (entry.node as QuadNode).isLeaf : entry.node.level === 0;
  }

  function traceColor(entry: TraceEntry, c: Palette) {
    const d = entry.decision;
    if (d.kind === "plane") return c.plane;
    if (d.kind === "resolve") return c.resolved;
    if (isRawLeaf(entry)) return c.leaf;
    return d.reason === "boundary" ? c.boundary : c.expand;
  }

  // ---------------------------------------------------------------- panels
  function renderStats() {
    if (!result) return;
    const r = result;
    const { xs } = sortedData();
    const visible = lowerBound(xs, state.maxX + 1e-15) - lowerBound(xs, state.minX);
    const counts = { resolved: 0, plane: 0, leaf: 0 };
    for (const item of r.items) counts[item.kind]++;
    const rawWindow = visible * 24;
    const tiles: Array<[string, string, string]> = [
      ["Visible points", fmtCount(visible), `raw window ${fmtBytes(rawWindow)}`],
      [
        "Kernels emitted",
        fmtCount(r.kernelCount),
        noPlanes
          ? `${counts.resolved} nodes + ${counts.leaf} leaves → ${fmtCount(r.groups.length)} after greedy merge`
          : `${counts.plane} plane blocks · ${counts.resolved} nodes · ${counts.leaf} leaves`,
      ],
      ["Bytes on the wire", fmtBytes(r.wireBytes), rawWindow > 0 ? `${((r.wireBytes / rawWindow) * 100).toFixed(1)}% of raw window` : ""],
      ["Range requests", String(r.requests.length), state.warmCache ? `${r.cacheHits} cache hits` : "cold cache"],
      ["Round trips", String(r.rounds), "sequential RTTs"],
      ["Storage", fmtBytes(model.totalBytes), `${(model.totalBytes / model.rawBytes).toFixed(2)}× raw · ${model.objects.length} objects`],
    ];
    statsEl.innerHTML = tiles
      .map(([l, v, s]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`)
      .join("");
  }

  function renderObjects() {
    if (!result) return;
    const byKey = new Map<string, typeof result.requests>();
    for (const req of result.requests) {
      const list = byKey.get(req.key) ?? [];
      list.push(req);
      byKey.set(req.key, list);
    }
    const phaseColor: Record<string, string> = {
      open: "var(--open)",
      "y-range": "var(--yrange)",
      frontier: "var(--boundary)",
      "planes+leaves": "",
    };
    let html = `<div class="k" style="color:var(--muted)">object</div><div class="num">size</div><div style="color:var(--muted)">bytes read (each mark is one range request)</div><div class="num">read</div>`;
    for (const obj of model.objects) {
      const reqs = byKey.get(obj.key) ?? [];
      const read = reqs.reduce((s, q) => s + q.len, 0);
      const dim = reqs.length === 0 ? " dim" : "";
      const marks = reqs
        .map((q) => {
          const color =
            phaseColor[q.phase] || (obj.kind === "plane" ? "var(--plane)" : "var(--leaf)");
          const left = (q.offset / obj.bytes) * 100;
          const width = (q.len / obj.bytes) * 100;
          return `<i style="left:${left}%;width:${width}%;background:${color}" title="${q.phase}: offset ${q.offset}, ${fmtBytes(q.len)}"></i>`;
        })
        .join("");
      const label = obj.kind === "plane" ? `${obj.key} <span style="color:var(--muted)">2^${obj.k} bins</span>` : obj.key;
      html += `<div class="name${dim}" title="${obj.key}">${label}</div><div class="num${dim}">${fmtBytes(obj.bytes)}</div><div class="bar${dim}">${marks}</div><div class="num${dim}">${reqs.length ? `${fmtBytes(read)} · ${reqs.length}×` : "—"}</div>`;
    }
    objectsEl.innerHTML = html;
  }

  function renderRounds() {
    if (!result) return;
    const byRound = new Map<number, typeof result.requests>();
    for (const req of result.requests) {
      const list = byRound.get(req.round) ?? [];
      list.push(req);
      byRound.set(req.round, list);
    }
    const phaseText: Record<string, string> = {
      open: isQuad
        ? `open: manifest (depth table + ${qm().inlineDepths} inlined depth${qm().inlineDepths === 1 ? "" : "s"})`
        : isTree
          ? "open: manifest (root level inlined)"
          : "open: manifest + ripmap header (join2)",
      "y-range": "exact visible y-range: descend straddling edge nodes",
      frontier: isQuad ? "frontier: child pages of expanded nodes (one depth per round trip)" : "frontier: fetch child pages of expanded nodes",
      "planes+leaves": isQuad
        ? "raw leaves (boundary + leaves still too big)"
        : isTree
          ? "raw leaves (boundary + still too big at ℓ0)"
          : "plane blocks + raw boundary leaves (join2)",
    };
    const rows = [...byRound.entries()].sort((a, b) => a[0] - b[0]);
    roundsEl.innerHTML =
      rows.length === 0
        ? `<div>Everything served from cache — 0 round trips.</div>`
        : rows
            .map(([round, reqs]) => {
              const objs = new Map<string, number>();
              for (const q of reqs) objs.set(q.key, (objs.get(q.key) ?? 0) + 1);
              const list = [...objs.entries()].map(([k, n]) => `<code>${k}</code>${n > 1 ? `×${n}` : ""}`).join(", ");
              const bytes = reqs.reduce((s, q) => s + q.len, 0);
              return `<div><b>RTT ${round + 1}</b><span>${phaseText[reqs[0]!.phase]} — ${list} <span style="color:var(--muted)">(${fmtBytes(bytes)})</span></span></div>`;
            })
            .join("");
  }

  function renderInspector() {
    if (!hover && model.strategy === "quadtree") {
      const m = model;
      const depthRows = m.depthInfo
        .map(
          (info, d) =>
            `<div>d${d}: ${fmtCount(info.count)} nodes → ${info.inline ? "manifest.bin" : `<code>nodes.bin</code> @ ${fmtBytes(info.offset)}`}</div>`,
        )
        .join("");
      inspector.innerHTML = `<h2>Inspector</h2>
        <div>Hover a node, a raw leaf, or a cell in the depth rows to see why the client chose it and which kernel it merged into.</div>
        <div class="eq">straddles viewport edge → expand
w² + span² &lt; t²        → resolve
otherwise              → expand
  (a leaf: raw points from points.bin)

w, span: the node's tight bbox on
screen — not its quadrant

then, in Z order (push_group):
  merge into previous kernel while
  dx² + dy² &lt; t²</div>
        <div class="k" style="margin-top:8px">Depths (${m.params.leafPoints}-point leaves · ${RECORD_BYTES}-byte records)</div>
        ${depthRows}`;
      return;
    }
    if (!hover && isTree) {
      const levelRows = model.levelCounts
        .map(
          (count, level) =>
            `<div>ℓ${level}: ${fmtCount(count)} nodes × ${fmtCount(blockPointsAt(rm().params, level))} pts${level === model.levelCounts.length - 1 ? " (in manifest)" : ` · pages of ${rm().params.fanout} × 64 B`}</div>`,
        )
        .reverse()
        .join("");
      inspector.innerHTML = `<h2>Inspector</h2>
        <div>Hover a node, a raw leaf, or a hierarchy cell to see why the client chose it and which kernel it merged into.</div>
        <div class="eq">straddles viewport edge → expand
w² + span² &lt; t²        → resolve
otherwise              → expand
  (at ℓ0: raw leaf from points.bin)

then, left to right (push_group):
  merge into previous kernel while
  dx² + dy² &lt; t²  (half-extents
  around the merged mean)
  raw leaf points: halve the run
  until it fits, then push_group</div>
        <div class="k" style="margin-top:8px">Levels</div>
        ${levelRows}`;
      return;
    }
    if (!hover) {
      const kmaxRows = model.levelCounts
        .map((count, level) => {
          const ks = rm().planeDepths[level]!;
          const desc = ks.length > 0 ? `k ∈ {${ks.join(", ")}}` : "no planes";
          return `<div>ℓ${level}: ${fmtCount(count)} blocks × ${fmtCount(blockPointsAt(rm().params, level))} pts → ${desc}</div>`;
        })
        .reverse()
        .join("");
      inspector.innerHTML = `<h2>Inspector</h2>
        <div>Hover a plane bin, a node, or a hierarchy cell to see why the client chose it.</div>
        <div class="eq">straddles viewport edge → expand
w² + span² &lt; t²        → resolve
w ≤ ${MAX_PLANE_BLOCK_WIDTH}·t and
  k = ⌈log₂(span/√(t²−w²))⌉
  stored sparse slot ≥ k → plane(ℓ, k)
otherwise              → expand
  (at ℓ0: raw leaf from points.bin)</div>
        <div class="k" style="margin-top:8px">Data-derived sparse plane depths (non-empty bins ≤ block_pts / ${rm().params.minPointsPerBin}, cap 14)</div>
        ${kmaxRows}`;
      return;
    }
    const node = hover.kind === "item" ? hover.item.node : hover.entry.node;
    const entry =
      hover.kind === "trace"
        ? hover.entry
        : result?.trace[node.level]?.find((e) => e.node === node) ?? null;
    let tag = "";
    let math = "";
    if (entry) {
      const { detail: d, decision } = entry;
      const t = state.threshold;
      const tagFor = (bg: string, text: string) => `<span class="tag" style="background:${bg}">${text}</span>`;
      tag =
        decision.kind === "plane"
          ? tagFor("var(--plane)", `plane(ℓ${node.level}, k=${decision.k})`)
          : decision.kind === "resolve"
            ? tagFor("var(--resolved)", decision.reason === "fits" ? "resolve: fits" : "resolve: off-screen in y")
            : isRawLeaf(entry)
              ? tagFor("var(--leaf)", "raw leaf")
              : tagFor(decision.reason === "boundary" ? "var(--boundary)" : "var(--expand)", `expand: ${decision.reason}`);
      const lines = [
        `w    = ${d.w.toFixed(3)}σ   (on-screen width)`,
        `span = ${d.span.toFixed(3)}σ   (on-screen height)`,
      ];
      if (!d.fullyInside) lines.push(`straddles the viewport edge → must descend`);
      else {
        lines.push(`w² + span² = ${(d.w * d.w + d.span * d.span).toFixed(2)} ${d.w * d.w + d.span * d.span < t * t ? "<" : "≥"} t² = ${(t * t).toFixed(2)}`);
        if (decision.kind !== "resolve" && !noPlanes) {
          lines.push(`w ${d.w <= MAX_PLANE_BLOCK_WIDTH * t ? "≤" : ">"} ${MAX_PLANE_BLOCK_WIDTH}·t = ${(MAX_PLANE_BLOCK_WIDTH * t).toFixed(2)}`);
        }
        if (d.kNeeded !== null) {
          const bins = Math.ceil(d.span / Math.max(d.hMax, 1e-12));
          lines.push(`h_max = √(t² − w²) = ${d.hMax.toFixed(3)}σ`);
          lines.push(`bins ≥ ⌈span / h_max⌉ = ${bins} → k = ${d.kNeeded}`);
          lines.push(`kmax(ℓ${node.level}) = ${d.kmax} → ${d.kNeeded <= d.kmax ? "use plane" : "too deep, expand"}`);
        }
      }
      math = `<div class="eq">${lines.join("\n")}</div>`;
    }
    let bytes = "";
    let cellsHtml = "";
    if (hover.kind === "item" && hover.item.kind === "plane") {
      const k = hover.item.k;
      const ks = rm().planeDepths[node.level]!;
      const slot = ks.indexOf(k);
      const dir = slot >= 0 ? rm().blockDirs[node.level]![node.index]![slot]! : null;
      const baseOff = slot >= 0 ? rm().planeByteOffsets[node.level]![slot]! : 0;
      const nonEmpty = dir ? dir.cellCount : hover.item.cells.filter((c) => c.count > 0).length;
      const blockBytes = nonEmpty * CELL_BYTES;
      const byteOff = dir ? baseOff + dir.cellOffset * CELL_BYTES : 0;
      bytes = `<div class="k">Bytes: <code>planes.bin</code> [${fmtCount(byteOff)}, +${fmtBytes(blockBytes)}) · ${nonEmpty}/${2 ** k} non-empty 16 B SparseCells</div>`;
      const cell = hover.cell;
      if (cell) {
        bytes += `<div>Bin ${cell.bin} / ${2 ** k}: y ∈ [${fmtNum(cell.lo)}, ${fmtNum(cell.lo + cell.h)}), ${fmtCount(cell.count)} pts, mass ${fmtNum(cell.mass)}</div>`;
      }
      cellsHtml = `<canvas data-ref="cells" style="height:${Math.min(2 ** k * 6 + 20, 260)}px;margin-top:6px"></canvas>`;
    } else if (hover.kind === "item" && hover.item.kind === "leaf") {
      bytes = `<div class="k">Bytes: <code>points.bin</code> [${fmtCount(node.start * 24)}, +${fmtBytes((node.end - node.start) * 24)})</div>`;
    } else if (model.strategy === "quadtree") {
      const info = model.depthInfo[node.level]!;
      bytes =
        info.inline
          ? `<div class="k">Record inlined in <code>manifest.bin</code> (depth ${node.level})</div>`
          : `<div class="k">Record: <code>nodes.bin</code> [${fmtCount(info.offset + node.index * RECORD_BYTES)}, +${RECORD_BYTES} B)</div>`;
    } else if (node.level < model.levelCounts.length - 1) {
      if (model.strategy === "ripmap") {
        const stride = recordBytesFor(rm().planeDepths[node.level]!.length);
        bytes = `<div class="k">Node bytes: <code>levels.bin</code> ℓ${node.level} page ${Math.floor(node.index / rm().params.fanout)} (${rm().params.fanout} × ${stride} B)</div>`;
      } else {
        bytes = `<div class="k">Node bytes: <code>level_${node.level}.bin</code> page ${Math.floor(node.index / rm().params.fanout)} (${rm().params.fanout} × 64 B)</div>`;
      }
    } else {
      bytes = `<div class="k">Root node — inlined in <code>manifest.bin</code></div>`;
    }
    const groupHtml = hover.kind === "item" && hover.item.kind !== "plane" ? describeGroup(hover.item) : "";
    inspector.innerHTML = `<h2>Inspector</h2>
      <div>${tag}</div>
      <div style="margin-top:6px">${isQuad ? `Depth ${node.level}, node ${fmtCount(node.index)} · ${fmtCount(node.end - node.start)} points (Z-order [${fmtCount(node.start)}, ${fmtCount(node.end)}))` : `Level ℓ${node.level}, block ${fmtCount(node.index)} · points [${fmtCount(node.start)}, ${fmtCount(node.end)})`}</div>
      <div class="k">${isQuad ? "tight bbox " : ""}x ∈ [${fmtNum(node.minX)}, ${fmtNum(node.maxX)}] · y ∈ [${fmtNum(node.minY)}, ${fmtNum(node.maxY)}]</div>${quadRegion(node)}
      ${math}${bytes}${groupHtml}${cellsHtml}`;
    const cellsCanvas = inspector.querySelector<HTMLCanvasElement>('[data-ref="cells"]');
    if (cellsCanvas && hover.kind === "item" && hover.item.kind === "plane") {
      drawCells(cellsCanvas, hover.item.cells, hover.cell);
    }
  }

  function quadRegion(node: TraceEntry["node"]) {
    if (!isQuad) return "";
    const q = node as QuadNode;
    const kind = q.isLeaf ? "leaf" : `${q.childCount} non-empty child quadrant${q.childCount === 1 ? "" : "s"}`;
    return `<div class="k">quadrant x ∈ [${fmtNum(q.rx0)}, ${fmtNum(q.rx1)}) · y ∈ [${fmtNum(q.ry0)}, ${fmtNum(q.ry1)}) · ${kind}</div>`;
  }

  function describeGroup(item: FrontierItem) {
    if (!result) return "";
    const idx = result.items.indexOf(item);
    const groups = result.groups.filter((g) => g.items.includes(idx));
    if (groups.length === 0) return "";
    const t = state.threshold;
    if (groups.length > 1) {
      const pts = groups.reduce((s, g) => s + g.rawPoints, 0);
      return `<div class="k" style="margin-top:8px">Greedy merge</div>
        <div>Raw points of this leaf were halved until each run fit, then merged into
        <b>${groups.length}</b> kernels (${fmtCount(pts)} points in total, shared with neighbours).</div>`;
    }
    const g = groups[0]!;
    const n = g.node;
    const dx = Math.max(n.x - n.minX, n.maxX - n.x) * result.xFactor;
    const dy = Math.max(n.y - n.minY, n.maxY - n.y) * result.yFactor;
    const parts = g.parts === 1 ? "on its own (nothing next to it fit)" : `with ${g.parts - 1} neighbouring piece${g.parts > 2 ? "s" : ""}`;
    return `<div class="k" style="margin-top:8px">Greedy merge (kernel #${result.groups.indexOf(g) + 1} of ${result.groups.length})</div>
      <div>Emitted ${parts}${g.rawPoints ? ` · ${fmtCount(g.rawPoints)} raw points` : ""}.</div>
      <div class="eq">dx = ${dx.toFixed(3)}σ, dy = ${dy.toFixed(3)}σ  (half-extents)
dx² + dy² = ${(dx * dx + dy * dy).toFixed(2)} &lt; t² = ${(t * t).toFixed(2)}</div>`;
  }

  function drawCells(canvas: HTMLCanvasElement, cells: PlaneCell[], active: PlaneCell | null) {
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const ctx = setupCanvas(canvas, W, H);
    const c = colors(root);
    ctx.fillStyle = c.panel;
    ctx.fillRect(0, 0, W, H);
    const maxMass = Math.max(...cells.map((x) => x.mass), 1e-12);
    const rowH = (H - 14) / cells.length;
    ctx.font = "10px ui-sans-serif, Segoe UI, sans-serif";
    ctx.fillStyle = c.muted;
    const nonEmpty = cells.filter((x) => x.count > 0).length;
    ctx.fillText(`${nonEmpty}/${cells.length} sparse 16 B cells (top = max_y), bar = mass`, 0, 10);
    for (const cell of cells) {
      const y = 14 + (cells.length - 1 - cell.bin) * rowH;
      ctx.fillStyle = c.grid;
      ctx.fillRect(0, y + 0.5, W, Math.max(rowH - 1, 0.5));
      ctx.fillStyle = cell === active ? c.accent : c.plane;
      ctx.fillRect(0, y + 0.5, (cell.mass / maxMass) * W, Math.max(rowH - 1, 0.5));
    }
  }

  // ---------------------------------------------------------------- interaction
  function hitItem(px: number, py: number): Hover {
    if (!result) return null;
    const x = invX(px);
    const y = invY(py);
    const tolX = (4 / plotW()) * (state.maxX - state.minX);
    for (const item of result.items) {
      const n = item.node;
      if (x < n.minX - tolX || x > n.maxX + tolX) continue;
      if (item.kind === "plane") {
        const cell = item.cells.find((c) => y >= c.lo && y <= c.lo + c.h) ?? null;
        if (cell || (y >= n.minY && y <= n.maxY)) return { kind: "item", item, cell };
      } else {
        const tolY = (4 / plotH()) * (yr()[1] - yr()[0]);
        if (y >= n.minY - tolY && y <= n.maxY + tolY) return { kind: "item", item, cell: null };
      }
    }
    return null;
  }

  function hitTrace(px: number, py: number): Hover {
    if (!result) return null;
    if (model.strategy === "quadtree") {
      const depth = Math.floor((py - 4) / QUAD_ROW);
      const rootNode = model.levels[0]?.[0];
      if (!rootNode || depth < 0 || depth >= model.levelCounts.length) return null;
      const rowTop = 4 + depth * QUAD_ROW;
      const W = tree.clientWidth;
      const entry = result.trace[depth]?.find((e) => {
        const [x0, x1, y0, y1] = quadRowBox(e.node as QuadNode, rootNode, rowTop, W);
        return px >= x0 - 2 && px <= x1 + 2 && py >= y0 - 2 && py <= y1 + 2;
      });
      return entry ? { kind: "trace", entry } : null;
    }
    const rows = model.levelCounts.length;
    const level = rows - 1 - Math.floor((py - 4) / LEVEL_ROW);
    if (level < 0 || level >= rows) return null;
    const x = invX(px);
    const tolX = (2 / plotW()) * (state.maxX - state.minX);
    const entry = result.trace[level]?.find((e) => x >= e.node.minX - tolX && x <= e.node.maxX + tolX);
    return entry ? { kind: "trace", entry } : null;
  }

  const setHover = (h: Hover) => {
    const same =
      (h === null && hover === null) ||
      (h?.kind === "item" && hover?.kind === "item" && h.item === hover.item && h.cell === hover.cell) ||
      (h?.kind === "trace" && hover?.kind === "trace" && h.entry === hover.entry);
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
      const dx = ((e.offsetX - drag.px) / plotW()) * span;
      let minX = drag.minX - dx;
      minX = Math.min(Math.max(minX, -span * 0.25), 1 + span * 0.25 - span);
      state.minX = minX;
      state.maxX = minX + span;
      if (drag.y) {
        const ySpan = drag.y[1] - drag.y[0];
        const dy = ((e.offsetY - drag.py) / plotH()) * ySpan;
        state.yBounds = [drag.y[0] + dy, drag.y[1] + dy];
      }
      scheduleUpdate();
    } else {
      setHover(hitItem(e.offsetX, e.offsetY));
    }
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
  tree.addEventListener("pointermove", (e) => setHover(hitTrace(e.offsetX, e.offsetY)));
  tree.addEventListener("pointerleave", () => setHover(null));

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

// ------------------------------------------------------------------------------------------------
// helpers

const RIPMAP_STEPS = `
  <div class="panel"><h3>1 · 4-object x-hierarchy with inline directories</h3>
    Points stay in x order (32-point leaves, <b>fanout 4</b>). Each node record packs a 64-byte moment node plus
    inline 6-byte <code>(cell_offset, cell_count)</code> slots for every stored plane at that level. Top levels live
    in <code>manifest.bin</code> (read in 1 RTT); deeper levels pack back-to-back in <code>levels.bin</code>,
    planes in <code>planes.bin</code>, raw points in <code>points.bin</code>.</div>
  <div class="panel"><h3>2 · sparse 16-byte y-bin planes</h3>
    For levels ℓ ≥ 1, data-derived y-depths store only <b>non-empty</b> bins as 16-byte quantized
    <code>SparseCell</code>s (Popoviciu-bounded variances). A block is stored at depth k only while its non-empty
    count stays ≤ <code>block_points / min_points_per_bin</code>, letting outlier-stretched blocks reach +2 extra
    depths without paying for empty zeros.</div>
  <div class="panel"><h3>3 · query: per-node level jump &amp; decoupled resolution</h3>
    Each root node jumps directly to the level where its own x-width satisfies w ≤ 0.9·t (so narrow burst blocks
    never get dragged down by a wide gap). Explicit <code>y_bounds</code> skips Phase 1. Fully-inside nodes resolve
    when w² + span² &lt; t², or pick the coarsest stored sparse plane slot ≥ k with non-empty cells.</div>
  <div class="panel"><h3>4 · fetch: coalesced spans &amp; greedy merging</h3>
    Inline directories give exact byte ranges in <code>planes.bin</code> with zero extra RTTs. Missing plane spans
    (reusing cached finer k when available) and boundary leaves fetch concurrently, then emitted plane bins are
    greedily merged (and off-screen y bins collapsed) into exact moment-preserving kernels.</div>`;

const TREE_STEPS = `
  <div class="panel"><h3>1 · x-ordered moment hierarchy</h3>
    The same paged hierarchy the ripmap uses, with a <b>fanout of 32</b>: 32-point leaves, then every 32 neighbouring
    nodes become one 64-byte parent, so one page of children is 2 KiB. Levels roll up until at most 32 root nodes
    remain, inlined in <code>manifest.bin</code>. A handful of static objects; storage ≈ 1.09× raw (level 0 adds 64 B per 768 B leaf).</div>
  <div class="panel"><h3>2 · query: descend until it fits</h3>
    After the exact visible y-range, the client walks the tree level by level. Nodes straddling the viewport edge
    always descend. A fully-inside node becomes one kernel only when its whole on-screen box fits,
    w² + h² &lt; t². Otherwise it expands, and at level 0 that means downloading its 32 raw points. Only x is ever
    split, so a tall node can only get thinner, never shorter.</div>
  <div class="panel"><h3>3 · assemble: greedy merging</h3>
    Left to right, each resolved node is merged into the previous kernel while the combined kernel's half-extents
    around its mean still fit (dx² + dy² &lt; t²). Raw leaf points are halved recursively until each run fits, then
    pushed through the same greedy merge. Moments stay exact; only the grouping changes.</div>
  <div class="panel"><h3>4 · fetch and where it struggles</h3>
    One round trip per level for 2 KiB pages (bridging gaps ≤ 16 KiB), then the raw leaves. Smooth signals resolve
    high up. Vertically incoherent data (white noise) never fits, because thinner x-slices are just as tall, so the
    overview descends to every leaf and downloads ~100% of <code>points.bin</code>. Compare with the ripmap page.</div>`;

const QUAD_STEPS = `
  <div class="panel"><h3>1 · split space, not x</h3>
    Starting from the data's bounding box, every node with more than 32 points splits its region at the midpoint into
    4 equal quadrants — squares in normalized data coordinates. Empty quadrants are dropped, so sparse areas cost
    nothing. Points are stored in quadtree (Z) order, which makes every subtree one contiguous run of
    <code>points.bin</code>; each leaf is sorted by x.</div>
  <div class="panel"><h3>2 · 72-byte records, packed objects</h3>
    Each node is the usual 64-byte moment node plus an 8-byte link: its children's position in the next depth, or its
    point range. Depths are arrays ordered by parent, so a node's ≤ 4 children are adjacent. The top depths are inlined
    in <code>manifest.bin</code>; every deeper depth follows the previous one in a single <code>nodes.bin</code>, found
    through a per-depth byte offset. Reads are byte ranges anyway, so the whole dataset is just three objects.</div>
  <div class="panel"><h3>3 · query: one depth per round trip</h3>
    Same rules as the x-tree, on 2D boxes: a node straddling the viewport edge descends; a fully-inside node whose tight
    bounding box fits (w² + h² &lt; t²) becomes one kernel; anything else expands, down to raw leaves. Cells are short as
    well as narrow, so white noise resolves without downloading every point — but every depth is a round trip.</div>
  <div class="panel"><h3>4 · where it struggles</h3>
    Cells are square, but time series are zoomed in x. Zoom 20× and the cells that are narrow enough on screen are also
    20× shorter than they need to be, so a dense band splits into far more cells than pixels — try white noise at
    Zoom 20×. Thin curves (random walks) also make the tree deep: more round trips than the x-tree.</div>`;

const TEXT: Record<PageStrategy, { title: string; lead: string; screenLegend: string; treeLegend: string; steps: string }> = {
  ripmap: {
    title: "Moment ripmap v2 explorer",
    lead: `An in-browser model of <code>rust/progressive-ripmap</code> (<code>PLPRIPM2</code>). The dataset is packed
      into 4 static objects: an x-ordered hierarchy with inline plane directories plus 16-byte sparse y-bin
      <em>planes</em>. Every pan or zoom re-runs the level-jumping query policy and shows which nodes resolve,
      which σ-wide x-blocks are answered from <code>planes.bin</code>, which boundary leaves fall back to raw
      points, and the byte ranges that would go over the wire.`,
    screenLegend: `
      <span><i class="sw" style="background:var(--plane)"></i>plane bin (one kernel per non-empty bin)</span>
      <span><i class="sw" style="background:var(--resolved)"></i>resolved node</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw boundary leaf points</span>
      <span><i class="sw" style="background:var(--group)"></i>merged kernel (2+ pieces)</span>
      <span><i class="sw" style="background:var(--kde)"></i>rendered KDE preview</span>`,
    treeLegend: `
      <span><i class="sw" style="background:var(--expand)"></i>expanded (too wide, or k &gt; kmax)</span>
      <span><i class="sw" style="background:var(--boundary)"></i>straddles viewport edge → must descend</span>
      <span><i class="sw" style="background:var(--resolved)"></i>resolved</span>
      <span><i class="sw" style="background:var(--plane)"></i>plane(k)</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw leaf</span>`,
    steps: RIPMAP_STEPS,
  },
  tree: {
    title: "Paged x-tree explorer",
    lead: `An in-browser model of <code>rust/progressive-tree</code> (<code>paged_moment_tree</code>). The dataset is
      built into an x-ordered, fanout-32 moment hierarchy; every pan or zoom re-runs the client's query and shows
      which nodes are small enough on screen to become one kernel, which leaves fall back to raw points, how the
      frame then greedily merges neighbours into the final kernels, and the byte ranges that would go over the wire.`,
    screenLegend: `
      <span><i class="sw" style="background:var(--resolved)"></i>resolved node (fits as one kernel)</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw leaf points</span>
      <span><i class="sw" style="background:var(--group)"></i>merged kernel (2+ pieces)</span>
      <span><i class="sw" style="background:var(--kde)"></i>rendered KDE preview</span>`,
    treeLegend: `
      <span><i class="sw" style="background:var(--expand)"></i>expanded (doesn't fit)</span>
      <span><i class="sw" style="background:var(--boundary)"></i>straddles viewport edge → must descend</span>
      <span><i class="sw" style="background:var(--resolved)"></i>resolved</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw leaf</span>`,
    steps: TREE_STEPS,
  },
  quadtree: {
    title: "Moment quadtree explorer",
    lead: `An in-browser model of <code>rust/progressive-quadtree</code> (<code>moment_quadtree</code>). The dataset is
      split into a classic point-region quadtree — midpoint quadrants, empty ones dropped — with exact moments per node.
      Every pan or zoom re-runs the client's query: which cells resolve to one kernel, which straddle the viewport edge or
      are still too big and descend, which leaves fall back to raw points, and the byte ranges read from the packed
      single <code>nodes.bin</code>.`,
    screenLegend: `
      <span><i class="sw" style="background:var(--boundary)"></i>quadrant grid of visited nodes</span>
      <span><i class="sw" style="background:var(--resolved)"></i>resolved node (tight bbox fits)</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw leaf points</span>
      <span><i class="sw" style="background:var(--group)"></i>merged kernel (2+ pieces)</span>
      <span><i class="sw" style="background:var(--kde)"></i>rendered KDE preview</span>`,
    treeLegend: `
      <span><i class="sw" style="background:var(--expand)"></i>expanded (doesn't fit)</span>
      <span><i class="sw" style="background:var(--boundary)"></i>straddles viewport edge → must descend</span>
      <span><i class="sw" style="background:var(--resolved)"></i>resolved</span>
      <span><i class="sw" style="background:var(--leaf)"></i>raw leaf</span>`,
    steps: QUAD_STEPS,
  },
};

export interface Palette {
  panel: string;
  ink: string;
  muted: string;
  grid: string;
  plane: string;
  resolved: string;
  leaf: string;
  expand: string;
  boundary: string;
  kde: string;
  accent: string;
  group: string;
  dataRgb: [number, number, number];
}

export function colors(root: HTMLElement): Palette {
  const s = getComputedStyle(root);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    panel: v("--panel"),
    ink: v("--ink"),
    muted: v("--muted"),
    grid: v("--grid"),
    plane: v("--plane"),
    resolved: v("--resolved"),
    leaf: v("--leaf"),
    expand: v("--expand"),
    boundary: v("--boundary"),
    kde: v("--kde"),
    accent: v("--accent"),
    group: v("--group"),
    dataRgb: hexToRgb(v("--data")),
  };
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return m ? [parseInt(m[1]!, 16), parseInt(m[2]!, 16), parseInt(m[3]!, 16)] : [30, 30, 30];
}

export function makeSprite() {
  const cache = new Map<string, HTMLCanvasElement>();
  return {
    get(color: string) {
      let c = cache.get(color);
      if (!c) {
        c = document.createElement("canvas");
        c.width = c.height = 64;
        const ctx = c.getContext("2d")!;
        const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        // Gaussian falloff over 3σ (radius 32px = 3σ).
        for (let i = 0; i <= 10; i++) {
          const r = i / 10;
          const a = Math.exp(-0.5 * (r * 3) ** 2);
          g.addColorStop(r, withAlpha(color, a));
        }
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, 64, 64);
        cache.set(color, c);
      }
      return c;
    },
  };
}

export function withAlpha(hex: string, a: number) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

export function setupCanvas(canvas: HTMLCanvasElement, w: number, h: number) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${h}px`;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function lowerBound(xs: Float64Array, value: number) {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid]! < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function el(tag: string, cls?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

export function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / 1024 / 1024).toFixed(2)} MiB`;
}

export function fmtCount(n: number) {
  return n.toLocaleString("en-US");
}

export function fmtNum(v: number) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 2 : 1);
}

