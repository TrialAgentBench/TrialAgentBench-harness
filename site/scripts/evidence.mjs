// The site's evidence is the dataset's published release evidence: a download of the dataset's SHA256SUMS and its
// release-evidence/ directory, every file of which the checksum list binds.
import { createHash } from "node:crypto";
import { lstatSync, readFileSync, readdirSync } from "node:fs";
import { relative, resolve, sep } from "node:path";

export const RECORD_PATH = "release-evidence/release-evidence.json";
const RECORD_SCHEMA = "trialagentbench.public_release_evidence/v1";
let cachedVerification;

export function evidenceRoot() {
  const configured = process.env.TRIALAGENTBENCH_EVIDENCE;
  if (!configured) {
    throw new Error("TRIALAGENTBENCH_EVIDENCE must name a download of the dataset's release evidence");
  }
  return resolve(configured);
}

function filesBelow(root, directory = root) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    const stats = lstatSync(path);
    if (stats.isSymbolicLink()) throw new Error(`release evidence contains a symlink: ${path}`);
    if (stats.isDirectory()) files.push(...filesBelow(root, path));
    else if (stats.isFile()) files.push(relative(root, path).split(sep).join("/"));
  }
  return files.sort();
}

/** Verify that release-evidence/ holds exactly the files SHA256SUMS lists under it, byte for byte. */
export function verifyEvidence(root = evidenceRoot()) {
  if (cachedVerification?.root === root) return cachedVerification;
  if (lstatSync(root).isSymbolicLink()) throw new Error("release evidence root must not be a symlink");
  const listed = new Map();
  for (const line of readFileSync(resolve(root, "SHA256SUMS"), "utf8").split("\n")) {
    const match = /^([0-9a-f]{64}) {2}(release-evidence\/[^\n]+)$/.exec(line);
    if (match) listed.set(match[2], match[1]);
  }
  const observed = filesBelow(root, resolve(root, "release-evidence"));
  const expected = [...listed.keys()].sort();
  if (!listed.has(RECORD_PATH) || JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("release evidence files differ from the release-evidence/ entries of SHA256SUMS");
  }
  for (const [path, digest] of listed) {
    if (createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex") !== digest) {
      throw new Error(`release evidence file differs from SHA256SUMS: ${path}`);
    }
  }
  const record = JSON.parse(readFileSync(resolve(root, RECORD_PATH), "utf8"));
  if (record?.schema_id !== RECORD_SCHEMA) throw new Error(`the release evidence record is not ${RECORD_SCHEMA}`);
  for (const [status, result] of [["gradient_status", "evaluation"], ["withheld_gradient_status", "withheld_evaluation"]]) {
    const expectedStatus = record.trialdev[result] == null ? "pending" : "present";
    if (record.trialdev[status] !== expectedStatus) throw new Error(`TrialDev ${status} disagrees with its verdict`);
  }
  if ((record.trialdev.gradient == null) !== (record.trialdev.evaluation == null)) {
    throw new Error("TrialDev gradient and evaluation disagree");
  }
  for (const split of ["public", "withheld"]) {
    const td = record.trialdev[split];
    if (td.qualification_status === "pending") {
      if (split !== "public" || td.recovery_status !== "pending" || td.accepted !== null || td.failed !== null || td.census !== null) {
        throw new Error("Pending TrialDev evidence cannot claim a scientific pass");
      }
    } else if (td.qualification_status !== undefined && (td.qualification_status !== "present" || td.recovery_status !== "present" || typeof td.accepted !== "boolean" || typeof td.failed !== "number" || td.census == null)) {
      throw new Error("Present TrialDev evidence requires qualification results");
    }
  }
  cachedVerification = { root, files: expected, record };
  return cachedVerification;
}
