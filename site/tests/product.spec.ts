import { readFileSync } from "node:fs";
import { join } from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { harnessRoot, siteBase } from "../scripts/site-config.mjs";
import { releaseEvidence } from "../src/lib/evidence";

const release = releaseEvidence();
// The README carries a model results section only once a campaign's results are published.
const hasResults = readFileSync(join(harnessRoot, "README.md"), "utf8").includes("\n## Model results\n");
const routes = ["/", ...(hasResults ? ["/results/"] : []), "/benchmark/", "/run/"];
// Routes are named from the site root; baseURL carries the base the site was built and is served under.
const fromBase = (route: string) => `.${route}`;
// Chromium can stall rasterizing the tall realism SVG in a nested scroll container.
// Capture its visible contents with static clipping; keep interactive scrolling for readers.
function screenshotOptions(page: Page) {
  // Beyond-viewport capture can collapse Chromium's layout width on this long mobile page.
  // Preserve the current viewport width during capture so text and figures do not reflow.
  return {
    scale: "css",
    style: `:root { min-width: ${page.viewportSize()!.width}px !important; } .figure-scroll { overflow: clip !important; }`,
  } as const;
}

async function figuresLoaded(page: Page): Promise<void> {
  for (const image of await page.locator("main img").all()) {
    await image.scrollIntoViewIfNeeded();
    await expect.poll(() => image.evaluate(element => element instanceof HTMLImageElement && element.complete && element.naturalWidth > 0)).toBe(true);
    // A figure is never drawn larger than its designed size, so its type stays at the exported size.
    const designed = await image.getAttribute("width");
    if (designed) expect((await image.boundingBox())!.width).toBeLessThanOrEqual(Number(designed) + 1);
  }
}

for (const route of routes) {
  test(`${route} is accessible, readable and within its viewport`, async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    expect((await page.goto(fromBase(route), { waitUntil: "networkidle" }))?.ok()).toBe(true);
    await expect(page.locator("main h1")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    if ((page.viewportSize()?.width ?? 0) <= 600) {
      const clipped = await page.locator("main p").evaluateAll(paragraphs => paragraphs.flatMap(paragraph => {
        const box = paragraph.getBoundingClientRect();
        if (box.width === 0 || box.height === 0 || (box.left >= 12 && box.right <= innerWidth - 12)) return [];
        return [{ text: paragraph.textContent?.slice(0, 80), left: box.left, right: box.right }];
      }));
      expect(clipped, "Compact prose must keep at least 12px on both sides").toEqual([]);
    }
    await figuresLoaded(page);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    expect(errors).toEqual([]);
    await expect(page.locator("main")).not.toContainText("undefined");
    for (const href of await page.getByRole("navigation", { name: "On this page" }).getByRole("link").evaluateAll(links => links.map(link => link.getAttribute("href")!))) {
      await expect(page.locator(`main [id="${href.slice(1)}"]`)).toHaveCount(1);
    }
    await page.screenshot({ path: info.outputPath("page.png"), fullPage: true, ...screenshotOptions(page) });
  });

  test(`${route} reads completely without JavaScript and in print`, async ({ browser, baseURL, viewport }, info) => {
    const context = await browser.newContext({ javaScriptEnabled: false, baseURL: baseURL!, viewport });
    const page = await context.newPage();
    await page.goto(fromBase(route), { waitUntil: "networkidle" });
    await figuresLoaded(page);
    const visible = await page.locator("main img:visible").count();
    await page.emulateMedia({ media: "print" });
    await expect(page.locator(".site-header")).toBeHidden();
    await expect(page.locator("main img:visible")).toHaveCount(visible);
    if (info.project.name === "desktop") await page.pdf({ path: info.outputPath("print.pdf"), format: "A4", printBackground: true });
    await context.close();
  });
}

test("the pages are linked and the dropped pages are gone", async ({ page, baseURL }) => {
  const url = (route: string) => new URL(fromBase(route), baseURL).href;
  await page.goto(fromBase("/"));
  const primary = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(primary.getByRole("link")).toHaveText([
    ...(hasResults ? ["Results"] : []),
    "Methods and evidence",
    "Run it",
    "Dataset",
    "Code",
  ]);
  await primary.getByRole("link", { name: "Methods and evidence" }).click();
  await expect(page).toHaveURL(url("/benchmark/"));
  await expect(page.locator("main h1")).toHaveText("Methods and evidence");
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Run it" }).click();
  await expect(page).toHaveURL(url("/run/"));
  await page.getByRole("link", { name: "TrialAgentBench", exact: true }).click();
  await expect(page).toHaveURL(url("/"));
  const dropped = ["/guide/", "/trials/", "/methods/", "/evidence/", "/trialeval/catalogue/1/"];
  for (const route of [...dropped, ...(hasResults ? [] : ["/results/"])]) {
    expect((await page.request.get(fromBase(route))).status(), route).toBe(404);
  }
});

test("the results page renders the README's model results tables", async ({ page }) => {
  test.skip(!hasResults, "no model results are published");
  await page.goto(fromBase("/"));
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Results" }).click();
  await expect(page).toHaveURL(/\/results\/$/);
  await expect(page.locator("main h1")).toHaveText("Model results");
  const tables = page.locator("main table");
  const trialeval = tables.filter({ has: page.getByRole("columnheader", { name: "Mean regret", exact: true }) });
  const trialdev = tables.filter({ has: page.getByRole("columnheader", { name: "Development value", exact: true }) });
  await expect(trialeval).toHaveCount(1);
  await expect(trialdev).toHaveCount(1);
  const record: {
    scorer: { version: string; sources_sha256: string };
    runs: object[];
    comparisons: { trialeval: object[]; trialdev: object[] }[];
  } = JSON.parse(readFileSync(join(harnessRoot, "docs/results.json"), "utf8"));
  await expect(trialeval.locator("tbody tr")).toHaveCount(record.runs.length);
  await expect(trialdev.locator("tbody tr")).toHaveCount(record.runs.length);
  await expect(page.locator("main")).toContainText(record.scorer.version);
  await expect(page.locator("main")).toContainText(record.scorer.sources_sha256);
  const paired = tables.filter({ has: page.getByRole("columnheader", { name: "Paired SE", exact: true }) });
  const pairedBenchmarks = ["trialeval", "trialdev"] as const;
  await expect(paired).toHaveCount(pairedBenchmarks.filter(benchmark => record.comparisons?.some(row => row[benchmark].length)).length);
  for (const table of await paired.all()) await expect(table.getByRole("columnheader", { name: "MDD", exact: true })).toBeVisible();
});

test("keyboard users can skip the navigation", async ({ page }) => {
  await page.goto(fromBase("/benchmark/"));
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("main")).toBeFocused();
});

test("the release evidence states the record's values and serves its files", async ({ page }, info) => {
  const te = release.trialeval;
  await page.goto(fromBase("/benchmark/#release-evidence-heading"));
  const section = page.locator("#release-evidence");
  await expect(section).toBeVisible();
  const publicRow = section.locator("tbody tr").first();
  await expect(publicRow).toContainText(`${te.public.regime_cells} (${te.public.a2_cells} A2, ${te.public.trap_cells} traps)`);
  await expect(publicRow).toContainText(`${te.public.independent_base_worlds} (${te.public.series} series × ${te.public.laws_per_series} laws)`);
  await expect(publicRow).toContainText(te.public.tasks.toLocaleString("en-US"));
  await expect(section).toContainText(`${te.consequence.admitted} of ${te.consequence.traps}`);
  await expect(section).toContainText(`${te.over_correction.consequential} of ${te.over_correction.twins}`);
  await expect(section).toContainText(`$${te.over_correction.mean_regret_musd[0].toFixed(1)}M`);
  if (te.policy_public.naive_cost) await expect(section).toContainText(`$${te.policy_public.naive_cost.regret_musd[1].toFixed(1)}M`);
  // Thresholds are the record's, not constants of the page.
  await expect(section).toContainText(`at least ${te.consequence.shift_threshold_sd.toFixed(1)} grading SD in every effect tertile`);
  if (te.over_correction.margin !== null) await expect(section).toContainText(`at least ${te.over_correction.margin} times the bar`);
  await expect(section).toContainText("withheld rule outcomes only");
  await expect(section).toContainText(te.policy_public.unpriced_worlds != null
    ? `${te.policy_public.unpriced_worlds.toLocaleString("en-US")} cell-worlds where no credited analysis fits add zero regret`
    : "The public panel's unpriced-world total was not recorded.");
  const bandRows = section.getByRole("region", { name: "Public cells failed and withheld rule outcomes across the economics band" }).locator("tbody tr");
  await expect(bandRows).toHaveCount(Object.keys(te.economics.band_failing_cells).length);
  // The withheld split is stated by rule outcome: whether the reference fails a withheld cell, never a count.
  for (const row of await bandRows.all()) await expect(row.locator("td").nth(3)).toHaveText(/^(yes|no)$/);
  // Trap and twin cells are counted and summarised, never named; neither is the withheld split's held-out trap.
  expect((await section.innerText()).match(/TE-S\d{2}-A[23]/g) ?? []).toEqual([]);
  await expect(section.locator('a[href$="/cells.md"]')).toHaveCount(0);
  const artifacts = await section.locator(`a[href^="${siteBase}artifacts/"]`).evaluateAll(links => links.map(link => link.getAttribute("href")!));
  expect(artifacts, "the section links the release evidence files").not.toEqual([]);
  for (const href of artifacts) {
    expect((await page.request.get(href)).ok(), href).toBe(true);
  }
  await figuresLoaded(page);
  await section.screenshot({ path: info.outputPath("release-evidence.png"), ...screenshotOptions(page) });
});
