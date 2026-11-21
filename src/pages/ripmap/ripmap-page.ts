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
  type TraceEntry,
  type ViewQuery,
  blockPointsAt,
  buildRipmap,
  CELL_BYTES,
  emptyCache,
  generateDataset,
  runQuery,
} from "./ripmap-model";

/**
 * Interactive explainer for the `progressive-ripmap` strategy: build a dataset in-browser,
 * then pan/zoom a viewport and watch which hierarchy nodes resolve, which x-blocks are served
 * through y-bin planes, which leaves fall back to raw points, and which byte ranges are fetched.
 */

const STYLE = `
.rm { --bg:#f6f7fb; --panel:#ffffff; --ink:#1b2433; --muted:#5d6b80; --line:#dde3ec; --grid:#eef1f6;
  --plane:#7c4dff; --resolved:#0f9d8a; --leaf:#f07c1f; --expand:#b7c2d1; --boundary:#8a96a8;
  --open:#64748b; --yrange:#2f7de1; --data:#1b2433; --kde:#d6336c; --accent:#d6336c;
  font-family: ui-sans-serif, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg);
  min-height:100vh; padding:20px 24px 48px; box-sizing:border-box; }
@media (prefers-color-scheme: dark) {
  .rm { --bg:#10141b; --panel:#171c25; --ink:#e6ebf2; --muted:#93a0b4; --line:#2a3240; --grid:#1f2631;
    --plane:#a58bff; --resolved:#2cc7b0; --leaf:#ff9b4a; --expand:#4a5566; --boundary:#7d8899;
    --open:#94a3b8; --yrange:#5ea2ff; --data:#e6ebf2; --kde:#ff6b9a; --accent:#ff6b9a; }
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

const GUTTER = 88; // left label gutter shared by the screen and the hierarchy canvas
const PAD_Y = 12;
const LEVEL_ROW = 26;

interface State {
  shape: DatasetShape;
  n: number;
  params: RipmapParams;
  sigmaPx: number;
  threshold: number;
  minX: number;
  maxX: number;
  yBounds: [number, number] | null;
  showRaw: boolean;
  showCells: boolean;
  showKde: boolean;
  warmCache: boolean;
}

type Hover =
  | { kind: "item"; item: FrontierItem; cell: PlaneCell | null }
  | { kind: "trace"; entry: TraceEntry }
  | null;

export async function runRipmapPage(args: { output: HTMLElement }) {
  const { output } = args;
  output.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = STYLE;
  output.appendChild(style);

  const root = el("div", "rm");
  output.appendChild(root);
  root.innerHTML = `
    <h1>Moment ripmap explorer</h1>
    <p class="lead">An in-browser model of <code>rust/progressive-ripmap</code>. The dataset is built into an
      x-ordered moment hierarchy plus dense per-block y-bin <em>planes</em>; every pan or zoom re-runs the
      client's query policy and shows which nodes resolve to one kernel, which σ-wide x-blocks are answered by a
      plane, which boundary leaves fall back to raw points, and the byte ranges that would go over the wire.</p>
    <div class="controls panel" data-ref="controls"></div>
    <div class="stats" data-ref="stats"></div>
    <div class="main">
      <div class="panel">
        <h2>Screen — what the client renders</h2>
        <div class="canvas-wrap"><canvas data-ref="screen"></canvas></div>
        <div class="legend">
          <span><i class="sw" style="background:var(--plane)"></i>plane bin (one kernel per non-empty bin)</span>
          <span><i class="sw" style="background:var(--resolved)"></i>resolved node (one kernel)</span>
          <span><i class="sw" style="background:var(--leaf)"></i>raw boundary leaf points</span>
          <span><i class="sw" style="background:var(--kde)"></i>rendered KDE preview</span>
        </div>
        <h2 style="margin-top:14px">Hierarchy traversal — shared x-axis</h2>
        <div class="canvas-wrap"><canvas data-ref="tree"></canvas></div>
        <div class="legend">
          <span><i class="sw" style="background:var(--expand)"></i>expanded (too wide, or k &gt; kmax)</span>
          <span><i class="sw" style="background:var(--boundary)"></i>straddles viewport edge → must descend</span>
          <span><i class="sw" style="background:var(--resolved)"></i>resolved</span>
          <span><i class="sw" style="background:var(--plane)"></i>plane(k)</span>
          <span><i class="sw" style="background:var(--leaf)"></i>raw leaf</span>
        </div>
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
  ref<HTMLDivElement>("steps").innerHTML = STEPS_HTML;

  const state: State = {
    shape: "random_walk",
    n: 300_000,
    // A small root cap makes the level-by-level descent visible; the Rust default is 1024.
    params: { ...DEFAULT_PARAMS, rootCap: 16 },
    sigmaPx: 14,
    threshold: 1,
    minX: 0,
    maxX: 1,
    yBounds: null,
    showRaw: true,
    showCells: true,
    showKde: false,
    warmCache: false,
  };

  let model: RipmapModel = buildRipmap(generateDataset(state.shape, state.n), state.params);
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
    model = buildRipmap(generateDataset(state.shape, state.n), state.params);
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
  const plotW = () => Math.max(screen.clientWidth - GUTTER, 10);
  const plotH = () => Math.max(screen.clientHeight - 2 * PAD_Y, 10);
  const sx = (x: number) => GUTTER + ((x - state.minX) / (state.maxX - state.minX)) * plotW();
  const invX = (px: number) => state.minX + ((px - GUTTER) / plotW()) * (state.maxX - state.minX);
  const yr = () => result?.effectiveY ?? [0, 1];
  const sy = (y: number) => {
    const [y0, y1] = yr();
    return PAD_Y + (1 - (y - y0) / Math.max(y1 - y0, 1e-12)) * plotH();
  };
  const invY = (py: number) => {
    const [y0, y1] = yr();
    return y0 + (1 - (py - PAD_Y) / plotH()) * (y1 - y0);
  };

  const screenHeight = () => Math.round(Math.min(Math.max(screen.clientWidth * 0.45, 280), 520));

  function update() {
    screen.style.height = `${screenHeight()}px`;
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
    result = runQuery(model, query, cache);
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
    const W = screen.clientWidth;
    const H = screenHeight();
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

  function drawFrontier(ctx: CanvasRenderingContext2D, r: QueryResult, c: Palette) {
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
    for (const item of r.items) {
      if (item.kind === "plane") {
        for (const cell of item.cells) push(cell.mass, cell.x, cell.y, cell.cxx, cell.cxy, cell.cyy);
      } else if (item.kind === "resolved") {
        const n = item.node;
        push(n.mass, n.x, n.y, n.cxx, n.cxy, n.cyy);
      } else {
        const { xs, ys, ws } = model.data;
        for (let i = item.node.start; i < item.node.end; i++) {
          const x = xs[i]!;
          if (x >= state.minX && x <= state.maxX) push(ws[i]!, x, ys[i]!, 0, 0, 0);
        }
      }
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
      ctx.fillText(`${fmtCount(blockPointsAt(model.params, level))}p · k≤${model.kmax[level]}`, 4, y + LEVEL_ROW / 2 + 6);
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

  function traceColor(entry: TraceEntry, c: Palette) {
    const d = entry.decision;
    if (d.kind === "plane") return c.plane;
    if (d.kind === "resolve") return c.resolved;
    if (entry.node.level === 0) return c.leaf;
    return d.reason === "boundary" ? c.boundary : c.expand;
  }

  // ---------------------------------------------------------------- panels
  function renderStats() {
    if (!result) return;
    const r = result;
    const { xs } = model.data;
    const visible = lowerBound(xs, state.maxX + 1e-15) - lowerBound(xs, state.minX);
    const counts = { resolved: 0, plane: 0, leaf: 0 };
    for (const item of r.items) counts[item.kind]++;
    const rawWindow = visible * 24;
    const tiles: Array<[string, string, string]> = [
      ["Visible points", fmtCount(visible), `raw window ${fmtBytes(rawWindow)}`],
      ["Kernels emitted", `≤ ${fmtCount(r.kernelCount)}`, `${counts.plane} plane blocks · ${counts.resolved} nodes · ${counts.leaf} leaves`],
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
      open: "open: manifest + ripmap header (join2)",
      "y-range": "exact visible y-range: descend straddling edge nodes",
      frontier: "frontier: fetch child pages of expanded nodes",
      "planes+leaves": "plane blocks + raw boundary leaves (join2)",
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
    if (!hover) {
      const kmaxRows = model.levelCounts
        .map(
          (count, level) =>
            `<div>ℓ${level}: ${fmtCount(count)} blocks × ${fmtCount(blockPointsAt(model.params, level))} pts → kmax ${model.kmax[level]}</div>`,
        )
        .reverse()
        .join("");
      inspector.innerHTML = `<h2>Inspector</h2>
        <div>Hover a plane bin, a node, or a hierarchy cell to see why the client chose it.</div>
        <div class="eq">straddles viewport edge → expand
w² + span² &lt; t²        → resolve
w ≤ ${MAX_PLANE_BLOCK_WIDTH}·t and
  k = ⌈log₂(span/√(t²−w²))⌉
  k ≤ kmax(ℓ)          → plane(ℓ, k)
otherwise              → expand
  (at ℓ0: raw leaf from points.bin)</div>
        <div class="k" style="margin-top:8px">Stored plane depths (kmax = ⌊log₂(block_pts / ${model.params.minPointsPerBin})⌋, cap 14)</div>
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
            : node.level === 0
              ? tagFor("var(--leaf)", "raw leaf")
              : tagFor(decision.reason === "boundary" ? "var(--boundary)" : "var(--expand)", `expand: ${decision.reason}`);
      const lines = [
        `w    = ${d.w.toFixed(3)}σ   (on-screen width)`,
        `span = ${d.span.toFixed(3)}σ   (on-screen height)`,
      ];
      if (!d.fullyInside) lines.push(`straddles the viewport edge → must descend`);
      else {
        lines.push(`w² + span² = ${(d.w * d.w + d.span * d.span).toFixed(2)} ${d.w * d.w + d.span * d.span < t * t ? "<" : "≥"} t² = ${(t * t).toFixed(2)}`);
        if (decision.kind !== "resolve") {
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
      const blockBytes = 2 ** k * CELL_BYTES;
      bytes = `<div class="k">Bytes: <code>plane_${node.level}_${k}.bin</code> [${fmtCount(node.index * blockBytes)}, +${fmtBytes(blockBytes)})</div>`;
      const cell = hover.cell;
      if (cell) {
        bytes += `<div>Bin ${cell.bin} / ${2 ** k}: y ∈ [${fmtNum(cell.lo)}, ${fmtNum(cell.lo + cell.h)}), ${fmtCount(cell.count)} pts, mass ${fmtNum(cell.mass)}</div>`;
      }
      cellsHtml = `<canvas data-ref="cells" style="height:${Math.min(2 ** k * 6 + 20, 260)}px;margin-top:6px"></canvas>`;
    } else if (hover.kind === "item" && hover.item.kind === "leaf") {
      bytes = `<div class="k">Bytes: <code>points.bin</code> [${fmtCount(node.start * 24)}, +${fmtBytes((node.end - node.start) * 24)})</div>`;
    } else if (node.level < model.levelCounts.length - 1) {
      bytes = `<div class="k">Node bytes: <code>level_${node.level}.bin</code> page ${Math.floor(node.index / model.params.fanout)} (${model.params.fanout} × 64 B)</div>`;
    } else {
      bytes = `<div class="k">Root node — inlined in <code>manifest.bin</code></div>`;
    }
    inspector.innerHTML = `<h2>Inspector</h2>
      <div>${tag}</div>
      <div style="margin-top:6px">Level ℓ${node.level}, block ${fmtCount(node.index)} · points [${fmtCount(node.start)}, ${fmtCount(node.end)})</div>
      <div class="k">x ∈ [${fmtNum(node.minX)}, ${fmtNum(node.maxX)}] · y ∈ [${fmtNum(node.minY)}, ${fmtNum(node.maxY)}]</div>
      ${math}${bytes}${cellsHtml}`;
    const cellsCanvas = inspector.querySelector<HTMLCanvasElement>('[data-ref="cells"]');
    if (cellsCanvas && hover.kind === "item" && hover.item.kind === "plane") {
      drawCells(cellsCanvas, hover.item.cells, hover.cell);
    }
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
    ctx.fillText(`${cells.length} dense MomentCells (top = max_y), bar = mass`, 0, 10);
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

const STEPS_HTML = `
  <div class="panel"><h3>1 · x-ordered moment hierarchy</h3>
    Points are kept in x order and grouped into 32-point leaves, then summarised level by level with a small
    <b>fanout of 4</b> so x-block widths come in fine ×4 steps. Each 64-byte node stores mass, mean, covariance and its
    exact bounding box. Levels ≤ <code>root_cap</code> nodes are inlined in <code>manifest.bin</code>; the rest live
    in <code>level_ℓ.bin</code>, raw points in <code>points.bin</code>.</div>
  <div class="panel"><h3>2 · dense y-bin planes</h3>
    For every level ℓ and depth k = 1…kmax(ℓ), <code>plane_ℓ_k.bin</code> stores 2<sup>k</sup> 24-byte
    <code>MomentCell</code>s per x-block, splitting that block's own [min_y, max_y] into equal bins (empty bins are zero).
    Block j is the contiguous byte range [j·2<sup>k</sup>, (j+1)·2<sup>k</sup>)·24, so a run of neighbouring blocks is one
    range request. Planes stop once bins would average fewer than <code>min_points_per_bin</code> points.</div>
  <div class="panel"><h3>3 · query: decouple x and y resolution</h3>
    After the exact visible y-range, the client walks the hierarchy level by level. A node straddling the viewport edge
    always descends (exact moments come from x-containment). A fully-inside node that fits inside the merge threshold
    becomes one kernel. Otherwise, once it is at most 0.9σ wide, it is answered from plane (ℓ, k) with the smallest k
    whose bin height keeps w² + h² ≤ t² — instead of descending until every node is also less than σ tall.</div>
  <div class="panel"><h3>4 · fetch: one contiguous range per plane run</h3>
    Tree pages are fetched one level per round trip. Then the needed plane blocks (grouped per (ℓ, k), bridging gaps
    ≤ 16 KiB) and the raw boundary leaves (bridging ≤ 8 leaves) are fetched concurrently in a single final round.
    On vertically incoherent data such as white noise this replaces downloading nearly every raw point with a
    handful of plane ranges.</div>`;

interface Palette {
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
  dataRgb: [number, number, number];
}

function colors(root: HTMLElement): Palette {
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
    dataRgb: hexToRgb(v("--data")),
  };
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return m ? [parseInt(m[1]!, 16), parseInt(m[2]!, 16), parseInt(m[3]!, 16)] : [30, 30, 30];
}

function makeSprite() {
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

function withAlpha(hex: string, a: number) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function setupCanvas(canvas: HTMLCanvasElement, w: number, h: number) {
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

function lowerBound(xs: Float64Array, value: number) {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid]! < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function el(tag: string, cls?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / 1024 / 1024).toFixed(2)} MiB`;
}

function fmtCount(n: number) {
  return n.toLocaleString("en-US");
}

function fmtNum(v: number) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 2 : 1);
}

