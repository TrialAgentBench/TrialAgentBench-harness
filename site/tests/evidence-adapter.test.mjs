import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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

function rewriteRecord(root, change) {
  const path = join(root, RECORD_PATH);
  const record = JSON.parse(readFileSync(path, "utf8"));
  change(record);
  const payload = `${JSON.stringify(record, null, 2)}\n`;
  writeFileSync(path, payload, "utf8");
  const sums = join(root, "SHA256SUMS");
  const digest = createHash("sha256").update(payload).digest("hex");
  writeFileSync(
    sums,
    readFileSync(sums, "utf8").replace(new RegExp(`^[0-9a-f]{64}(  ${RECORD_PATH})$`, "m"), `${digest}$1`),
    "utf8",
  );
}

test("the evidence adapter accepts the published release evidence", () => {
  const verification = verifyEvidence(fixture);
  assert.ok(verification.files.includes(RECORD_PATH));
  assert.equal(verification.record.schema_id, "trialagentbench.public_release_evidence/v1");
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

test("the evidence adapter rejects the private record, which names trap and twin cells", () => {
  withCopy((root) => {
    rewriteRecord(root, (record) => { record.schema_id = "trialagentbench.release_evidence/v1"; });
    assert.throws(() => verifyEvidence(root), /is not trialagentbench\.public_release_evidence\/v1/);
  });
});

for (const [status, result] of [["gradient_status", "evaluation"], ["withheld_gradient_status", "withheld_evaluation"]]) {
  test(`the evidence adapter rejects ${status}=pending with a retained verdict`, () => {
    withCopy((root) => {
      rewriteRecord(root, (record) => {
        record.trialdev[status] = "pending";
        record.trialdev[result] = { stale: true };
      });
      assert.throws(() => verifyEvidence(root), /TrialDev .* disagrees with its verdict/);
    });
  });
  test(`the evidence adapter rejects ${status}=present without a verdict`, () => {
    withCopy((root) => {
      rewriteRecord(root, (record) => {
        record.trialdev[status] = "present";
        record.trialdev[result] = null;
      });
      assert.throws(() => verifyEvidence(root), /TrialDev .* disagrees with its verdict/);
    });
  });
}

test("the evidence adapter ignores files outside release-evidence/", () => {
  withCopy((root) => {
    appendFileSync(join(root, "SHA256SUMS"), `${"0".repeat(64)}  full/SHA256SUMS\n`, "utf8");
    assert.equal(verifyEvidence(root).record.schema_id, "trialagentbench.public_release_evidence/v1");
  });
});


function pendingTrialDev(record) {
  Object.assign(record.trialdev.public, {
    qualification_status: "pending", recovery_status: "pending", accepted: null, failed: null, census: null,
  });
  Object.assign(record.trialdev, {
    gradient_status: "pending", withheld_gradient_status: "pending", gradient: null, evaluation: null, withheld_evaluation: null,
  });
}

test("the evidence adapter accepts explicitly pending public scientific evidence", () => {
  withCopy((root) => {
    rewriteRecord(root, pendingTrialDev);
    assert.equal(verifyEvidence(root).record.trialdev.public.accepted, null);
  });
});

test("the evidence adapter rejects a pass claimed by pending public evidence", () => {
  withCopy((root) => {
    rewriteRecord(root, (record) => { pendingTrialDev(record); record.trialdev.public.accepted = true; });
    assert.throws(() => verifyEvidence(root), /cannot claim a scientific pass/);
  });
});

test("the evidence adapter rejects pending withheld qualification", () => {
  withCopy((root) => {
    rewriteRecord(root, (record) => { record.trialdev.withheld.qualification_status = "pending"; });
    assert.throws(() => verifyEvidence(root), /cannot claim a scientific pass/);
  });
});
