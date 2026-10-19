import assert from "node:assert/strict";
import { appendFileSync, cpSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import { RECORD_PATH, verifyEvidence } from "../scripts/evidence.mjs";

const fixture = process.env.TRIALAGENTBENCH_EVIDENCE ?? resolve("dist/artifacts");

function withCopy(change) {
  const root = mkdtempSync(join(tmpdir(), "trialagentbench-evidence-"));
  try {
    cpSync(fixture, root, { recursive: true });
    change(root);
  } finally {
    rmSync(root, { recursive: true });
  }
}

test("the evidence adapter accepts the published release evidence", () => {
  const verification = verifyEvidence(fixture);
  assert.ok(verification.files.includes(RECORD_PATH));
  assert.equal(verification.record.schema_id, "trialagentbench.release_evidence/v1");
});

test("the evidence adapter rejects a missing file", () => {
  withCopy((root) => {
    const file = verifyEvidence(fixture).files.find((path) => path !== RECORD_PATH);
    unlinkSync(join(root, file));
    assert.throws(() => verifyEvidence(root), /differ from the release-evidence\/ entries/);
  });
});

test("the evidence adapter rejects a file SHA256SUMS does not list", () => {
  withCopy((root) => {
    writeFileSync(join(root, "release-evidence", "notes.md"), "unlisted\n", "utf8");
    assert.throws(() => verifyEvidence(root), /differ from the release-evidence\/ entries/);
  });
});

test("the evidence adapter rejects a changed quantitative value", () => {
  withCopy((root) => {
    const path = join(root, RECORD_PATH);
    const record = JSON.parse(readFileSync(path, "utf8"));
    record.trialeval.public.tasks += 1;
    writeFileSync(path, `${JSON.stringify(record, null, 2)}\n`, "utf8");
    assert.throws(() => verifyEvidence(root), /differs from SHA256SUMS/);
  });
});

test("the evidence adapter ignores files outside release-evidence/", () => {
  withCopy((root) => {
    appendFileSync(join(root, "SHA256SUMS"), `${"0".repeat(64)}  full/SHA256SUMS\n`, "utf8");
    assert.equal(verifyEvidence(root).record.schema_id, "trialagentbench.release_evidence/v1");
  });
});
