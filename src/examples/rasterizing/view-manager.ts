export function lowerBound(pts: Float64Array, x: number): number {
  let lo = 0,
    hi = pts.length / 3;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pts[mid * 3]! < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function upperBound(pts: Float64Array, x: number): number {
  let lo = 0,
    hi = pts.length / 3;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (pts[mid * 3]! <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export class ViewManager {
  private readonly pts: Float64Array;
  private readonly dataMinY: number;
  private readonly dataMaxY: number;
  private readonly fullRangeX: number;
  private readonly minViewRangeX: number;
  private currentViewMinX: number;
  private currentViewMaxX: number;
  private targetViewMinX: number;
  private targetViewMaxX: number;
  private currentViewMinY: number;
  private currentViewMaxY: number;
  private isPanning = false;
  private lastPointerX = 0;
  private readonly interpolationRate = 12;

  constructor(pts: Float64Array) {
    this.pts = pts;
    const n = pts.length / 3;
    let minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const y = pts[i * 3 + 1]!;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    // x is normalized to [0, 1] and sorted
    this.dataMinY = minY;
    this.dataMaxY = maxY;
    this.fullRangeX = 1; // x span is always 1 after normalization
    this.minViewRangeX = Math.max(1 / 2 ** 16, 1e-9);
    this.currentViewMinX = 0;
    this.currentViewMaxX = 1;
    this.targetViewMinX = 0;
    this.targetViewMaxX = 1;
    this.currentViewMinY = minY;
    this.currentViewMaxY = maxY;
  }

  private computeVisibleYRange(): [number, number] {
    const { pts } = this;
    const startIdx = lowerBound(pts, this.currentViewMinX);
    const endIdx = upperBound(pts, this.currentViewMaxX);
    let minY = Infinity,
      maxY = -Infinity;
    for (let i = startIdx; i < endIdx; i++) {
      const y = pts[i * 3 + 1]!;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (!isFinite(minY)) return [this.dataMinY, this.dataMaxY];
    return [minY, maxY];
  }

  tick(dtSeconds: number): void {
    const alpha = 1 - Math.exp(-this.interpolationRate * dtSeconds);
    this.currentViewMinX +=
      (this.targetViewMinX - this.currentViewMinX) * alpha;
    this.currentViewMaxX +=
      (this.targetViewMaxX - this.currentViewMaxX) * alpha;
    if (Math.abs(this.targetViewMinX - this.currentViewMinX) < 1e-8)
      this.currentViewMinX = this.targetViewMinX;
    if (Math.abs(this.targetViewMaxX - this.currentViewMaxX) < 1e-8)
      this.currentViewMaxX = this.targetViewMaxX;

    const [targetMinY, targetMaxY] = this.computeVisibleYRange();
    this.currentViewMinY += (targetMinY - this.currentViewMinY) * alpha;
    this.currentViewMaxY += (targetMaxY - this.currentViewMaxY) * alpha;
  }

  getViewMinX(): number {
    return this.currentViewMinX;
  }
  getViewMaxX(): number {
    return this.currentViewMaxX;
  }
  getViewMinY(): number {
    return this.currentViewMinY;
  }
  getViewMaxY(): number {
    return this.currentViewMaxY;
  }

  setViewRangeX(minX: number, maxX: number): void {
    const clampedMin = Math.max(0, Math.min(1, minX));
    const clampedMax = Math.max(0, Math.min(1, maxX));
    let lo = Math.min(clampedMin, clampedMax);
    let hi = Math.max(clampedMin, clampedMax);
    const minSpan = this.minViewRangeX;
    if (hi - lo < minSpan) {
      hi = Math.min(1, lo + minSpan);
      lo = Math.max(0, hi - minSpan);
    }
    this.targetViewMinX = lo;
    this.targetViewMaxX = hi;
    this.currentViewMinX = lo;
    this.currentViewMaxX = hi;
    const [minY, maxY] = this.computeVisibleYRange();
    this.currentViewMinY = minY;
    this.currentViewMaxY = maxY;
  }

  bindCanvas(canvas: HTMLCanvasElement): void {
    canvas.style.touchAction = "none";

    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 1) return;
      e.preventDefault();
      this.isPanning = true;
      this.lastPointerX = e.clientX;
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!this.isPanning) return;
      if ((e.buttons & 4) === 0) {
        this.isPanning = false;
        canvas.releasePointerCapture(e.pointerId);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const deltaXRatio = (e.clientX - this.lastPointerX) / rect.width;
      this.lastPointerX = e.clientX;
      const span = this.targetViewMaxX - this.targetViewMinX;
      const deltaX = deltaXRatio * span;
      this.targetViewMinX -= deltaX;
      this.targetViewMaxX -= deltaX;
      this.clampTarget();
    });

    const stopPan = (e: PointerEvent) => {
      if (!this.isPanning) return;
      this.isPanning = false;
      canvas.releasePointerCapture(e.pointerId);
    };
    canvas.addEventListener("pointerup", stopPan);
    canvas.addEventListener("pointercancel", stopPan);

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const currentSpan = this.targetViewMaxX - this.targetViewMinX;

        const isLateral = Math.abs(e.deltaX) > Math.abs(e.deltaY);

        if (isLateral && e.deltaX !== 0) {
          const deltaX = (e.deltaX / rect.width) * currentSpan;
          this.targetViewMinX += deltaX;
          this.targetViewMaxX += deltaX;
          this.clampTarget();
        } else if (e.deltaY !== 0) {
          const anchorRatio = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / rect.width),
          );
          const zoomFactor = Math.exp(e.deltaY * 0.0015);
          const nextSpan = Math.max(
            this.minViewRangeX,
            Math.min(this.fullRangeX, currentSpan * zoomFactor),
          );
          if (!Number.isFinite(nextSpan) || nextSpan === currentSpan) return;
          const anchorX = this.targetViewMinX + anchorRatio * currentSpan;
          this.targetViewMinX = anchorX - anchorRatio * nextSpan;
          this.targetViewMaxX = this.targetViewMinX + nextSpan;
          this.clampTarget();
        }
      },
      { passive: false },
    );
  }

  private clampTarget(): void {
    const span = Math.max(
      this.targetViewMaxX - this.targetViewMinX,
      this.minViewRangeX,
    );
    this.targetViewMinX = Math.max(0, Math.min(this.targetViewMinX, 1 - span));
    this.targetViewMaxX = this.targetViewMinX + span;
  }
}
