import { defineConfig } from "astro/config";

import { harnessRoot, siteBase } from "./scripts/site-config.mjs";

export default defineConfig({
  base: siteBase,
  output: "static",
  // Commands render as plain text in the site palette.
  markdown: { syntaxHighlight: false },
  trailingSlash: "always",
  build: { format: "directory" },
  vite: { build: { cssMinify: "lightningcss" }, resolve: { alias: { "@harness": harnessRoot } } },
});
