export interface VerifiedEvidence {
  readonly root: string;
  /** The release-evidence/ files, relative to the root. */
  readonly files: readonly string[];
  readonly record: unknown;
}

export const RECORD_PATH: string;
export function evidenceRoot(): string;
export function verifyEvidence(root?: string): VerifiedEvidence;
