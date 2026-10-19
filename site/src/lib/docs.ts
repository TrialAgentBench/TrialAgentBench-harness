/**
 * The harness repository's Markdown, rendered once by Astro and placed on the site without rewording.
 *
 * Pages select whole sections by their heading. Astro emits the figures the Markdown embeds. Links are re-pointed: the
 * methods and running guides to their site pages, and every other repository file to the same file on GitHub.
 */
import { parseHTML } from "linkedom";

import * as readmeSource from "@harness/README.md";
import * as methodsSource from "@harness/docs/METHODS.md";
import * as runningSource from "@harness/docs/RUNNING.md";
import * as examplesSource from "@harness/examples/README.md";
import { project } from "./project";
import { hasResults, RESULTS_HEADING } from "./results";
import { routeHref } from "./urls";

interface MarkdownModule { compiledContent(): string | Promise<string> }

export interface DocSection {
  id: string;
  title: string;
  /** The section's content without its heading. */
  html: string;
}

export interface Doc {
  /** Content before the first second-level heading, without the title. */
  preamble: string;
  sections: DocSection[];
}

function siteTarget(path: string, hash: string): string {
  if (path === "docs/METHODS.md") return `${routeHref("benchmark")}${hash}`;
  if (path === "docs/RUNNING.md") return `${routeHref("run")}${hash}`;
  if (path === "README.md") return `${routeHref("run")}${hash || "#quickstart"}`;
  return `${project.repositoryUrl}/${path.endsWith("/") ? "tree" : "blob"}/v${project.version}/${path}${hash}`;
}

function relink(document: Document, repositoryPath: string): void {
  for (const link of document.querySelectorAll("a[href]")) {
    const href = link.getAttribute("href") ?? "";
    if (href.startsWith("#") || /^[a-z]+:/i.test(href)) continue;
    const target = new URL(href, `https://repository.invalid/${repositoryPath}`);
    link.setAttribute("href", siteTarget(target.pathname.slice(1), target.hash));
  }
}

async function load(module: MarkdownModule, repositoryPath: string): Promise<Doc> {
  const { document } = parseHTML(`<!doctype html><html><body>${await module.compiledContent()}</body></html>`);
  relink(document, repositoryPath);
  // Wide tables and commands scroll inside keyboard-reachable regions; each table region is named by its heading.
  for (const command of document.querySelectorAll("pre")) command.setAttribute("tabindex", "0");
  const labels = new Map<string, number>();
  let heading = document.querySelector("h1")?.textContent ?? repositoryPath;
  for (const element of document.querySelectorAll("h2, h3, h4, table")) {
    if (element.tagName !== "TABLE") {
      heading = element.textContent ?? heading;
      continue;
    }
    const label = `${heading}: table of ${element.querySelector("th")?.textContent ?? "values"}`;
    const repeat = (labels.get(label) ?? 0) + 1;
    labels.set(label, repeat);
    const region = document.createElement("div");
    region.setAttribute("class", "table-scroll");
    region.setAttribute("tabindex", "0");
    region.setAttribute("role", "region");
    region.setAttribute("aria-label", repeat === 1 ? label : `${label} (${repeat})`);
    element.replaceWith(region);
    region.append(element);
  }
  const preamble: string[] = [];
  const sections: DocSection[] = [];
  for (const node of [...document.body.childNodes]) {
    const element = node as Element;
    if (element.tagName === "H1") continue;
    if (element.tagName === "H2") {
      sections.push({ id: element.getAttribute("id") ?? "", title: element.textContent ?? "", html: "" });
      continue;
    }
    const html = element.outerHTML ?? node.textContent ?? "";
    const current = sections.at(-1);
    if (current) current.html += html;
    else preamble.push(html);
  }
  return { preamble: preamble.join(""), sections };
}

/** Return the named section, failing the build if the source no longer has it. */
export function section(doc: Doc, title: string): DocSection {
  const found = doc.sections.find((candidate) => candidate.title === title);
  if (!found) throw new Error(`the harness documentation has no section "${title}"`);
  return found;
}

/** A demonstration's question, figure and reading, without the run command that sits between question and figure. */
export function demonstration(title: string): DocSection {
  const found = section(examples, title);
  const html = transform(found.html, (body) => {
    const blocks = [...body.children];
    const figure = blocks.findIndex((block) => block.querySelector("img"));
    if (figure < 1) throw new Error(`the demonstration "${title}" no longer opens with a question and then a figure`);
    for (const block of blocks.slice(1, figure)) block.remove();
  });
  return { ...found, html };
}

/** Apply a structural change to a fragment of rendered Markdown and return the new HTML. */
export function transform(html: string, change: (body: HTMLElement, document: Document) => void): string {
  const { document } = parseHTML(`<!doctype html><html><body>${html}</body></html>`);
  change(document.body, document);
  return document.body.innerHTML;
}

export const readme = await load(readmeSource, "README.md");
export const methods = await load(methodsSource, "docs/METHODS.md");
export const running = await load(runningSource, "docs/RUNNING.md");
const examples = await load(examplesSource, "examples/README.md");

/** The README's model results section, present only once a campaign's results are published. */
export const results = hasResults ? section(readme, RESULTS_HEADING) : undefined;
