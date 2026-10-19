import { readFileSync } from "node:fs";
import { join } from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { harnessRoot } from "../scripts/site-config.mjs";
import { releaseEvidence } from "../src/lib/evidence";

const release = releaseEvidence();
// The README carries a model results section only once a campaign's results are published.
const hasResults = readFileSync(join(harnessRoot, "README.md"), "utf8").includes("\n## Model results\n");
const routes = ["/", ...(hasResults ? ["/results/"] : []), "/benchmark/", "/run/"];

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
    expect((await page.goto(route, { waitUntil: "networkidle" }))?.ok()).toBe(true);
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
    await page.screenshot({ path: info.outputPath("page.png"), fullPage: true });
  });

  test(`${route} reads completely without JavaScript and in print`, async ({ browser, baseURL, viewport }, info) => {
    const context = await browser.newContext({ javaScriptEnabled: false, baseURL: baseURL!, viewport });
    const page = await context.newPage();
    await page.goto(route, { waitUntil: "networkidle" });
    await figuresLoaded(page);
    const visible = await page.locator("main img:visible").count();
    await page.emulateMedia({ media: "print" });
    await expect(page.locator(".site-header")).toBeHidden();
    await expect(page.locator("main img:visible")).toHaveCount(visible);
    if (info.project.name === "desktop") await page.pdf({ path: info.outputPath("print.pdf"), format: "A4", printBackground: true });
    await context.close();
  });
}

test("the pages are linked and the dropped pages are gone", async ({ page }) => {
  await page.goto("/");
  const primary = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(primary.getByRole("link")).toHaveText([
    ...(hasResults ? ["Results"] : []),
    "Methods and evidence",
    "Run it",
    "Dataset",
    "Code",
  ]);
  await primary.getByRole("link", { name: "Methods and evidence" }).click();
  await expect(page).toHaveURL(/\/benchmark\/$/);
  await expect(page.locator("main h1")).toHaveText("Methods and evidence");
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Run it" }).click();
  await expect(page).toHaveURL(/\/run\/$/);
  await page.getByRole("link", { name: "TrialAgentBench", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
  const dropped = ["/guide/", "/trials/", "/methods/", "/evidence/", "/trialeval/catalogue/1/"];
  for (const route of [...dropped, ...(hasResults ? [] : ["/results/"])]) {
    expect((await page.request.get(route)).status(), route).toBe(404);
  }
});

test("the results page renders the README's model results tables", async ({ page }) => {
  test.skip(!hasResults, "no model results are published");
  await page.goto("/");
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Results" }).click();
  await expect(page).toHaveURL(/\/results\/$/);
  await expect(page.locator("main h1")).toHaveText("Model results");
  const tables = page.locator("main table");
  await expect(tables).toHaveCount(2);
  await expect(tables.first().locator("thead")).toContainText("Mean regret");
  await expect(tables.last().locator("thead")).toContainText("Development value");
  expect(await tables.first().locator("tbody tr").count()).toBeGreaterThan(0);
});

test("keyboard users can skip the navigation", async ({ page }) => {
  await page.goto("/benchmark/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("main")).toBeFocused();
});

test("the release evidence states the record's values and serves its files", async ({ page }, info) => {
  const te = release.trialeval;
  await page.goto("/benchmark/#release-evidence-heading");
  const section = page.locator("#release-evidence");
  await expect(section).toBeVisible();
  const publicRow = section.locator("tbody tr").first();
  await expect(publicRow).toContainText(`${te.public.regime_cells} (${te.public.a2_cells} A2, ${te.public.trap_cells} traps)`);
  await expect(publicRow).toContainText(`${te.public.independent_base_worlds} (${te.public.series} series × ${te.public.laws_per_series} laws)`);
  await expect(publicRow).toContainText(te.public.tasks.toLocaleString("en-US"));
  for (const cost of te.policy_public.naive_costs) await expect(section).toContainText(`$${cost.regret_musd.toFixed(1)}M`);
  for (const row of te.over_corrections) await expect(section).toContainText(row.regime_cell_id);
  // The withheld split's held-out trap is counted but never named.
  const named = new Set((await section.innerText()).match(/TE-S\d{2}-A\d/g) ?? []);
  const publicCells = [...te.public.public_trap_cell_ids, ...te.public.public_a2_cell_ids];
  expect([...named].filter(cell => !cell.endsWith("-A1") && !publicCells.includes(cell))).toEqual([]);
  for (const href of await section.locator('a[href^="/artifacts/"]').evaluateAll(links => links.map(link => link.getAttribute("href")!))) {
    expect((await page.request.get(href)).ok(), href).toBe(true);
  }
  await section.screenshot({ path: info.outputPath("release-evidence.png") });
});
