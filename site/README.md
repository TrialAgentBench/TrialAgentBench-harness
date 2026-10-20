# TrialAgentBench site

Static pages built from the same sources as the public documentation (the harness repository,
`harnessRoot` in `scripts/site-config.mjs`):

- `/` renders the governing claim and "What it measures and why" from the harness `README.md`.
- `/results/` renders the README's "Model results" section. The section, and so the page and its
  navigation link, exist only once model results are published.
- `/benchmark/` renders `docs/METHODS.md` in full, with the release evidence after the
  construct-validity argument: the record's counts, and its costs as ranges across the trap and twin cells,
  which it never names (`src/components/ReleaseEvidence.astro`), the fixed-policy panel demonstration and the
  realism figures.
- `/run/` renders the README quickstart and `docs/RUNNING.md`.

`src/lib/docs.ts` renders the Markdown without rewording. Pages select whole sections by heading, and the
build fails if a selected heading disappears. Links to the methods and running guides become site links;
other repository links point to the file on GitHub. Astro emits the figures the Markdown embeds, so a
figure changes on the site only when its producer in `examples/` regenerates it.

## Evidence

The site reads the release evidence published with the dataset: its `SHA256SUMS` and its
`release-evidence/` directory (the public record, release facts, realism and demonstration figures).
Download them for the harness version:

```bash
uvx --from huggingface_hub hf download TrialAgentBench/TrialAgentBench --repo-type dataset \
  --revision v1.0.0 --include "SHA256SUMS" --include "release-evidence/*" --local-dir /path/to/site-evidence
```

The build refuses evidence whose `release-evidence/` files differ from the `SHA256SUMS` entries, byte for
byte, or whose record is not a public release evidence record.

## Build and check

Use Node.js 22.19 or later:

```bash
npm ci
export TRIALAGENTBENCH_EVIDENCE=/path/to/site-evidence
npm run check
npm run test:evidence
npm run build
npm run test:product
```

`build` verifies the release evidence against `SHA256SUMS` before rendering, copies it to `artifacts/`,
writes the site inventory and checks routes, landmarks and local links. Product checks run Chromium at desktop and compact (Pixel 7) sizes: accessibility (axe),
overflow, figures at no more than their designed size, no-JavaScript reading, print, and the release
evidence values against the record.

To serve below a URL subpath, rebuild with `TRIALAGENTBENCH_SITE_BASE=/chosen-path/ npm run build`. The
repository's Pages workflow (`.github/workflows/pages.yml`) builds the site this way, from the release
evidence of the harness version, and deploys it to GitHub Pages.
