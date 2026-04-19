# pink-lemonade

Canvas2D-style API rendered with WebGPU.

## Requirements

- Node.js 18+
- Chromium-compatible environment for browser-mode tests

## Setup

```bash
npm install
npx playwright install chromium
```

## Scripts

- `npm run typecheck`: run TypeScript checks
- `npm run build`: build ESM library output with Vite
- `npm run test`: run Vitest in browser mode (Chromium/Playwright)
- `npm run test:watch`: run Vitest in watch mode

## Quick start

```ts
import { createCanvas2DContext } from "pink-lemonade";

const canvas = document.querySelector("canvas");

if (canvas instanceof HTMLCanvasElement) {
  const ctx = await createCanvas2DContext(canvas);
  ctx.clear({ r: 1, g: 0.9, b: 0.6, a: 1 });
}
```

## Notes

WebGPU availability depends on browser/runtime support and system GPU capabilities. Tests intentionally account for unavailable or unusable WebGPU states.
