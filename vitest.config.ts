import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/browser/**/*.test.ts"],
    benchmark: {
      include: ["./test/**/*.bench.ts"],
      reporters: ["default"],
      outputFile: "./benchmark-results.json",
    },
    browser: {
      enabled: false,
      provider: playwright({
        launchOptions: {
          headless: false,
        },
      }),
      instances: [
        {
          browser: "chromium",
        },
      ],
    },
  },
});

