import { verifyEvidence } from "../../scripts/evidence.mjs";

/** Least and greatest value across cells. */
type Span = [number, number];

interface ReleaseSplit {
  release_id: string;
  tasks: number;
  series: number;
  laws_per_series: number;
  independent_base_worlds: number;
  regime_cells: number;
  trap_cells: number;
  held_out_cells: number;
  a2_cells: number;
  tasks_by_context: Record<string, number>;
  accepted: boolean;
  failed_cells: number;
  pair_parity: { pairs: number; identical_pairs: number } | null;
}

interface ReleaseTrialDevSplit {
  qualification_status?: "pending" | "present";
  recovery_status?: "pending" | "present";
  programmes: number;
  tasks: number;
  accepted: boolean | null;
  failed: number | null;
}

interface PolicyPanel {
  worlds_per_cell: number;
  unpriced_worlds?: number | null;
  cells: number;
  failing_cells: Record<string, number>;
  naive_cost: {
    cells: number;
    traps: number;
    regret_musd: Span;
    consequential_share: Span;
    regret_at_truth_musd: Span;
  } | null;
}

/**
 * The withheld split's panel as rule outcomes only: by band point, then policy, whether the policy fails at least one
 * withheld cell. A withheld cost or count would differ from the public split's by the held-out cell alone.
 */
interface WithheldPolicyOutcomes {
  worlds_per_cell: number;
  cells: number;
  fails_a_cell: Record<string, Record<string, boolean>>;
}

/** Failing cells by band point, then by policy. */
type BandCounts = Record<string, Record<string, number>>;

/**
 * The fields of the public release evidence record (trialagentbench.public_release_evidence/v1) that the site
 * displays. The record states trap and twin cells by count and as ranges across cells, and the withheld split by
 * counts and rule outcomes; it names none of them.
 */
export interface ReleaseEvidence {
  trialeval: {
    public: ReleaseSplit;
    withheld: ReleaseSplit;
    base_worlds_shared_between_splits: number;
    recovery: { prior_drawn_worlds_per_cell: number; coverage_worlds_per_cell: number };
    consequence: {
      traps: number;
      admitted: number;
      shift_threshold_sd: number;
      shift: number;
      interval_only: number;
      smallest_tertile_shift_sd: number | null;
      width_ratio: Span;
      shortcut_coverage: Span;
      loss_ratio: Span;
    };
    over_correction: {
      twins: number;
      consequential: number;
      margin: number | null;
      band_min_mean_over_bar: number | null;
      mean_regret_musd: Span;
      consequential_share: Span;
      decision_flip_share: Span | null;
    };
    policy_public: PolicyPanel;
    policy_withheld: WithheldPolicyOutcomes;
    economics: {
      band_failing_cells: BandCounts;
      band_regret_bar_musd: Record<string, number>;
    };
    realism: {
      cells: number;
      status_counts: Record<string, number>;
      departures: number;
      trap_departures: number;
      effect_distribution: { worlds: number; share_beyond_196: number } | null;
    };
  };
  trialdev: {
    public: ReleaseTrialDevSplit;
    withheld: ReleaseTrialDevSplit;
    gradient_status: "pending" | "present";
    withheld_gradient_status: "pending" | "present";
  };
}

/** The verified public release evidence record; the build fails if the evidence directory lacks it. */
export function releaseEvidence(): ReleaseEvidence {
  return verifyEvidence().record as ReleaseEvidence;
}
