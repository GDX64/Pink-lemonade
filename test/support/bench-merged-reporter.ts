import { getTests } from "@vitest/runner/utils";
import type { Reporter, TestModule } from "vitest/node";

type BenchWithResult = {
  name: string;
  hz: number;
  mean: number;
  p99: number;
  rme: number;
  sampleCount: number;
  mergedCount: number | null;
};

function formatNumber(value: number): string {
  return value.toFixed(value < 100 ? 4 : 2);
}

function parseMergedCount(name: string): number | null {
  const match = /\bmerged\s*=\s*(\d+)\b/i.exec(name);
  return match ? Number.parseInt(match[1], 10) : null;
}

function stripMergedTag(name: string): string {
  return name.replace(/\s*\|\s*merged\s*=\s*\d+\s*$/i, "");
}

function pad(value: string, width: number, alignRight = false): string {
  return alignRight ? value.padStart(width, " ") : value.padEnd(width, " ");
}

export default class BenchMergedReporter implements Reporter {
  onTestRunEnd(testModules: ReadonlyArray<TestModule>): void {
    const rows: BenchWithResult[] = [];

    for (const testModule of testModules) {
      for (const test of getTests(testModule)) {
        const benchmark = test.meta.benchmark && test.result?.benchmark;
        if (!benchmark) {
          continue;
        }

        rows.push({
          name: stripMergedTag(test.name),
          hz: benchmark.hz ?? 0,
          mean: benchmark.mean ?? 0,
          p99: benchmark.p99 ?? 0,
          rme: benchmark.rme ?? 0,
          sampleCount: benchmark.sampleCount ?? 0,
          mergedCount: parseMergedCount(test.name),
        });
      }
    }

    if (!rows.length) {
      return;
    }

    const headers = [
      "name",
      "hz",
      "mean",
      "p99",
      "rme",
      "samples",
      "merged_count",
    ];
    const body = rows.map((row) => [
      row.name,
      formatNumber(row.hz),
      formatNumber(row.mean),
      formatNumber(row.p99),
      `±${row.rme.toFixed(2)}%`,
      String(row.sampleCount),
      row.mergedCount === null ? "n/a" : String(row.mergedCount),
    ]);

    const widths = headers.map((header, index) => {
      let width = header.length;
      for (const line of body) {
        width = Math.max(width, line[index].length);
      }
      return width;
    });

    const numericColumns = new Set([1, 2, 3, 4, 5, 6]);
    const formatRow = (line: string[]): string =>
      line
        .map((value, index) =>
          pad(value, widths[index], numericColumns.has(index)),
        )
        .join("  ");

    console.log("\nmerge benchmark report (extra merged_count column)");
    console.log(`   ${formatRow(headers)}`);
    for (const line of body) {
      console.log(`   ${formatRow(line)}`);
    }
  }
}
