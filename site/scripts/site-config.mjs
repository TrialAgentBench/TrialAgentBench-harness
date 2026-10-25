import { fileURLToPath } from "node:url";

const DEFAULT_SITE_BASE = "/";

/** The harness repository whose README, methods, running guide and demonstrations the site renders. */
export const harnessRoot = fileURLToPath(new URL("../../", import.meta.url));

export function normalizeSiteBase(value = process.env.TRIALAGENTBENCH_SITE_BASE) {
  const candidate = (value ?? DEFAULT_SITE_BASE).trim();
  if (candidate === "" || candidate === "/") return DEFAULT_SITE_BASE;
  if (candidate.includes("?") || candidate.includes("#") || candidate.includes("\\")) {
    throw new Error("TRIALAGENTBENCH_SITE_BASE must be a URL path without a query or fragment");
  }
  const segments = candidate.split("/").filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === "..")) {
    throw new Error("TRIALAGENTBENCH_SITE_BASE cannot traverse directories");
  }
  return `/${segments.join("/")}/`;
}

export const siteBase = normalizeSiteBase();
