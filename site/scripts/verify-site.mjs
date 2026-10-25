import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

import { siteBase } from "./site-config.mjs";

const outputRoot = resolve(process.argv[2] ?? "dist");
const manifestName = "site-build-manifest.json";

if (!existsSync(outputRoot) || !statSync(outputRoot).isDirectory()) {
  throw new Error(`site output directory does not exist: ${outputRoot}`);
}

function walk(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  });
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function verifyBuildManifest() {
  const manifestPath = join(outputRoot, manifestName);
  if (!existsSync(manifestPath)) throw new Error(`site lacks its complete build manifest`);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (
    manifest?.schema_id !== "trialagentbench.site_build/v1" ||
    !Array.isArray(manifest.artifacts) ||
    typeof manifest.checksum !== "string" ||
    !/^[0-9a-f]{64}$/.test(manifest.checksum) ||
    Object.keys(manifest).sort().join(",") !== "artifacts,checksum,schema_id"
  ) {
    throw new Error("site build manifest violates its public contract");
  }
  const expectedChecksum = createHash("sha256")
    .update(canonical({ schema_id: manifest.schema_id, artifacts: manifest.artifacts }))
    .digest("hex");
  if (manifest.checksum !== expectedChecksum) throw new Error("site build manifest checksum mismatch");
  const expected = walk(outputRoot)
    .filter((path) => relative(outputRoot, path).split("\\").join("/") !== manifestName)
    .map((path) => {
      const body = readFileSync(path);
      return {
        relative_path: relative(outputRoot, path).split("\\").join("/"),
        sha256: createHash("sha256").update(body).digest("hex"),
        byte_length: statSync(path).size,
      };
    })
    .sort((left, right) => left.relative_path.localeCompare(right.relative_path));
  if (canonical(manifest.artifacts) !== canonical(expected)) {
    throw new Error("site files differ from the checksum-bound build inventory");
  }
}

function pageUrl(page) {
  const outputPath = relative(outputRoot, page).split("\\").join("/");
  const route = outputPath === "index.html" ? "" : outputPath.replace(/index\.html$/, "");
  return new URL(`${siteBase}${route}`, "https://static.invalid");
}

function localTarget(href, page) {
  if (!href || /^(?:mailto:|data:|tel:)/.test(href)) return null;
  const url = new URL(href, pageUrl(page));
  if (url.origin !== "https://static.invalid") return null;
  if (!url.pathname.startsWith(siteBase)) {
    throw new Error(
      `${relative(outputRoot, page)} points outside configured base ${siteBase}: ${href}`,
    );
  }
  const webPath = decodeURIComponent(url.pathname.slice(siteBase.length));
  if (webPath === "") return join(outputRoot, "index.html");
  if (webPath.endsWith("/")) return join(outputRoot, webPath, "index.html");
  return join(outputRoot, webPath);
}

const htmlFiles = walk(outputRoot).filter((path) => path.endsWith(".html"));
if (htmlFiles.length === 0) throw new Error("site output contains no HTML pages");

const missing = [];
for (const page of htmlFiles) {
  const html = readFileSync(page, "utf8");
  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${relative(outputRoot, page)} contains duplicate element identifiers`);
  }
  for (const landmark of ["<header", "<nav", "<main", "<footer"]) {
    if (!html.includes(landmark)) {
      throw new Error(`${relative(outputRoot, page)} lacks ${landmark.slice(1)} landmark`);
    }
  }
  if ((html.match(/aria-current="page"/g) ?? []).length !== 1) {
    throw new Error(`${relative(outputRoot, page)} must identify exactly one current navigation item`);
  }
  const references = html.matchAll(/(?:href|src)="([^"]+)"/g);
  for (const match of references) {
    const href = match[1];
    if (/\.html(?:[?#]|$)/.test(href)) {
      throw new Error(`${relative(outputRoot, page)} exposes a file-format route: ${href}`);
    }
    const target = localTarget(href, page);
    if (target !== null && !existsSync(target)) {
      missing.push(`${relative(outputRoot, page)} -> ${href}`);
    } else if (target?.endsWith(".html") && href.includes("#")) {
      const fragment = decodeURIComponent(new URL(href, pageUrl(page)).hash.slice(1));
      const targetIds = [...readFileSync(target, "utf8").matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
      if (fragment && !targetIds.includes(fragment)) missing.push(`${relative(outputRoot, page)} -> ${href}`);
    }
  }
}

if (missing.length > 0) {
  throw new Error(`site contains missing local targets:\n${missing.join("\n")}`);
}

const expectedRoutes = ["benchmark", "run"];
for (const route of expectedRoutes) {
  const page = join(outputRoot, route, "index.html");
  if (!existsSync(page)) throw new Error(`site lacks directory route /${route}/`);
}
if (!existsSync(join(outputRoot, "artifacts", "SHA256SUMS"))) {
  throw new Error("site lacks the release evidence checksum list");
}
verifyBuildManifest();

process.stdout.write(
  `Verified ${htmlFiles.length} directory-format pages and local targets at base ${siteBase}\n`,
);
