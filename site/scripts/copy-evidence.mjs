import { copyFileSync, cpSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

import { evidenceRoot, verifyEvidence } from "./evidence.mjs";

const source = evidenceRoot();
verifyEvidence(source);
const destination = resolve("dist", "artifacts");
if (existsSync(destination)) {
  throw new Error(`refusing to replace staged release evidence: ${destination}`);
}
mkdirSync(destination);
copyFileSync(resolve(source, "SHA256SUMS"), resolve(destination, "SHA256SUMS"));
cpSync(resolve(source, "release-evidence"), resolve(destination, "release-evidence"), {
  recursive: true,
  dereference: false,
  errorOnExist: true,
});
verifyEvidence(destination);
process.stdout.write(`Staged the release evidence at ${destination}\n`);
