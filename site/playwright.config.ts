import { defineConfig, devices } from "@playwright/test";

const reportFile = process.env.PLAYWRIGHT_JSON_OUTPUT_NAME ?? ".playwright/report.json";
const port = Number(process.env.TRIALAGENTBENCH_TEST_PORT ?? "4174");
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error("Invalid product-test port");

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  outputDir: ".playwright/results",
  reporter: [["line"], ["json", { outputFile: reportFile }]],
  retries: 0,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `python3 -m http.server ${port} --directory dist`,
    port,
    reuseExistingServer: false,
  },
});
