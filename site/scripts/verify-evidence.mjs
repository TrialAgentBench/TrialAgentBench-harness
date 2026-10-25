import { verifyEvidence } from "./evidence.mjs";

const verification = verifyEvidence();
process.stdout.write(`Verified ${verification.files.length} release evidence files against SHA256SUMS\n`);
