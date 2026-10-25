import { lstatSync, mkdirSync, readlinkSync, rmSync, symlinkSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { defineConfig, devices } from "@playwright/test";
import { siteBase } from "./scripts/site-config.mjs";

const reportFile = process.env.PLAYWRIGHT_JSON_OUTPUT_NAME ?? ".playwright/report.json";
const port = Number(process.env.TRIALAGENTBENCH_TEST_PORT ?? "4174");
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error("Invalid product-test port");

// Serve the built dist itself at the base it was built for, as Pages does, from a root holding only that link.
function servedRoot(): string {
  if (siteBase === "/") return "dist";
  const root = resolve(".playwright/served");
  const link = join(root, siteBase.slice(1, -1));
  const dist = resolve("dist");
  if (!lstatSync(link, { throwIfNoEntry: false })?.isSymbolicLink() || readlinkSync(link) !== dist) {
    rmSync(link, { recursive: true, force: true });
    mkdirSync(dirname(link), { recursive: true });
    symlinkSync(dist, link, "dir");
  }
  return root;
}

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  outputDir: ".playwright/results",
  reporter: [["line"], ["json", { outputFile: reportFile }]],
  retries: 0,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}${siteBase}`,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `python3 -m http.server ${port} --directory ${JSON.stringify(servedRoot())}`,
    port,
    reuseExistingServer: false,
  },
});
