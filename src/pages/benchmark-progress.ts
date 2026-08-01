/**
 * Blocking-work progress overlay for the benchmark page.
 *
 * The benchmark is a long series of synchronous wasm calls. Nothing here can
 * report itself: the caller has to hand back control to the browser often
 * enough for these elements to repaint, which is what `waitForRepaint` is for.
 *
 * Two bars rather than one. Phases differ in cost by four orders of magnitude
 * (1,000 merge iterations at ~25 us each, then 10 Runnalls iterations at ~500 ms
 * each), so a single iteration-counted bar would crawl and then leap. The
 * overall bar counts phases; the phase bar counts iterations within one.
 */

export interface BenchmarkProgressState {
  phaseLabel: string;
  phaseIndex: number;
  phaseCount: number;
  /** 0..1 within the current phase. */
  phaseProgress: number;
  /** Estimated milliseconds left in this phase, or null before the first run. */
  phaseEtaMs: number | null;
  elapsedMs: number;
}

function formatDuration(ms: number): string {
  if (ms < 1_000) return "<1 s";
  const totalSeconds = Math.round(ms / 1_000);
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes} m ${String(seconds).padStart(2, "0")} s`;
}

export function createBenchmarkProgressUi() {
  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.inset = "0";
  host.style.display = "grid";
  host.style.placeItems = "center";
  host.style.background = "rgba(247, 251, 255, 0.88)";
  host.style.zIndex = "9999";

  const card = document.createElement("div");
  card.style.width = "min(680px, 94vw)";
  card.style.border = "1px solid #c8d9eb";
  card.style.borderRadius = "14px";
  card.style.background = "#ffffff";
  card.style.padding = "20px";
  card.style.boxShadow = "0 10px 30px rgba(10, 34, 65, 0.15)";
  card.style.color = "#16324f";

  const title = document.createElement("div");
  title.textContent = "Running reduction benchmark...";
  title.style.fontWeight = "700";
  title.style.fontSize = "17px";
  title.style.marginBottom = "6px";

  const note = document.createElement("div");
  note.textContent =
    "Timing runs block the page between updates. This takes a while.";
  note.style.fontSize = "13px";
  note.style.color = "#5c708a";
  note.style.marginBottom = "16px";

  const makeBar = () => {
    const wrap = document.createElement("div");
    wrap.style.height = "10px";
    wrap.style.background = "#e7eef6";
    wrap.style.borderRadius = "999px";
    wrap.style.overflow = "hidden";
    const fill = document.createElement("div");
    fill.style.height = "100%";
    fill.style.width = "0%";
    fill.style.background = "linear-gradient(90deg, #6ca4d8, #2f6fa9)";
    fill.style.transition = "width 90ms linear";
    wrap.appendChild(fill);
    return { wrap, fill };
  };

  const makeLabel = () => {
    const el = document.createElement("div");
    el.style.display = "flex";
    el.style.justifyContent = "space-between";
    el.style.fontSize = "13px";
    el.style.color = "#2d4d6c";
    el.style.margin = "0 0 6px 0";
    const left = document.createElement("span");
    const right = document.createElement("span");
    right.style.fontVariantNumeric = "tabular-nums";
    el.appendChild(left);
    el.appendChild(right);
    return { el, left, right };
  };

  const overallLabel = makeLabel();
  const overallBar = makeBar();
  const phaseLabel = makeLabel();
  const phaseBar = makeBar();

  overallBar.wrap.style.marginBottom = "18px";

  card.appendChild(title);
  card.appendChild(note);
  card.appendChild(overallLabel.el);
  card.appendChild(overallBar.wrap);
  card.appendChild(phaseLabel.el);
  card.appendChild(phaseBar.wrap);
  host.appendChild(card);
  document.body.appendChild(host);

  return {
    /** Shown while datasets are generated, before any phase exists. */
    setPreparing(message: string) {
      overallLabel.left.textContent = message;
      overallLabel.right.textContent = "";
      phaseLabel.left.textContent = "";
      phaseLabel.right.textContent = "";
      overallBar.fill.style.width = "0%";
      phaseBar.fill.style.width = "0%";
    },

    update(state: BenchmarkProgressState) {
      const overall =
        (state.phaseIndex + state.phaseProgress) / Math.max(state.phaseCount, 1);

      overallLabel.left.textContent = `Phase ${state.phaseIndex + 1} of ${state.phaseCount}`;
      overallLabel.right.textContent = `${(overall * 100).toFixed(0)}% · ${formatDuration(state.elapsedMs)} elapsed`;
      overallBar.fill.style.width = `${(overall * 100).toFixed(1)}%`;

      phaseLabel.left.textContent = state.phaseLabel;
      phaseLabel.right.textContent =
        state.phaseEtaMs === null
          ? `${(state.phaseProgress * 100).toFixed(0)}%`
          : `${(state.phaseProgress * 100).toFixed(0)}% · ~${formatDuration(state.phaseEtaMs)} left`;
      phaseBar.fill.style.width = `${(state.phaseProgress * 100).toFixed(1)}%`;
    },

    done() {
      host.remove();
    },
  };
}

/**
 * Hands control back to the browser long enough for a repaint. A double
 * `requestAnimationFrame` is deliberate: the first callback fires *before* the
 * upcoming paint, so resolving on it would let the caller resume and re-block
 * without anything reaching the screen.
 */
export function waitForRepaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}
