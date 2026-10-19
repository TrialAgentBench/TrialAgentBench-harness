import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";

const outputRoot = resolve(process.argv[2] ?? "dist");
const manifestName = "site-build-manifest.json";

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

function walk(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  });
}

const artifacts = walk(outputRoot)
  .filter((path) => relative(outputRoot, path).split("\\").join("/") !== manifestName)
  .map((path) => {
    const payload = readFileSync(path);
    return {
      relative_path: relative(outputRoot, path).split("\\").join("/"),
      sha256: createHash("sha256").update(payload).digest("hex"),
      byte_length: statSync(path).size,
    };
  })
  .sort((left, right) => left.relative_path.localeCompare(right.relative_path));

const payload = { schema_id: "trialagentbench.site_build/v1", artifacts };
const checksum = createHash("sha256").update(canonical(payload)).digest("hex");
writeFileSync(join(outputRoot, manifestName), `${JSON.stringify({ ...payload, checksum }, null, 2)}\n`, "utf8");
process.stdout.write(`Bound ${artifacts.length} site files in ${manifestName}\n`);
