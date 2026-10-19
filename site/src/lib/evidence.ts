import { verifyEvidence } from "../../scripts/evidence.mjs";

type Interval = [number, number, number];

interface ReleaseSplit {
  release_id: string;
  tasks: number;
  series: number;
  laws_per_series: number;
  independent_base_worlds: number;
  regime_cells: number;
  trap_cells: number;
  held_out_cells: number;
  public_trap_cell_ids: string[];
  a2_cells: number;
  public_a2_cell_ids: string[];
  tasks_by_context: Record<string, number>;
  accepted: boolean;
  failed_cells: number;
  pair_parity: { pairs: number; identical_pairs: number };
}

interface ReleaseTrialDevSplit {
  programmes: number;
  tasks: number;
  accepted: boolean;
  failed: number;
}

interface NaiveCost {
  regime_cell_id: string;
  regret_musd: number;
  regret_se_musd: number;
  consequential_share: number;
  regret_at_truth_musd: number;
}

/** The fields of the release evidence record (trialagentbench.release_evidence/v1) that the site displays. */
export interface ReleaseEvidence {
  trialeval: {
    public: ReleaseSplit;
    withheld: ReleaseSplit;
    base_worlds_shared_between_splits: number;
    recovery: { prior_drawn_worlds_per_cell: number; coverage_worlds_per_cell: number };
    consequence: Array<{ regime_cell_id: string; width_ratio: number; loss_ratio: number; shortcut_coverage: number }>;
    over_corrections: Array<{ regime_cell_id: string; consequential_share: number; mean_regret_musd: number }>;
    policy_public: { worlds_per_cell: number; naive_costs: NaiveCost[] };
    realism: {
      cells: number;
      status_counts: Record<string, number>;
      departures: number;
      effect_distribution: { worlds: number; share_beyond_196: number } | null;
    };
  };
  trialdev: {
    public: ReleaseTrialDevSplit;
    withheld: ReleaseTrialDevSplit;
    panel: { public: { reference_normalised: Interval; reference_minus_stop_z: number } } | null;
  };
}

/** The verified release evidence record; the build fails if the evidence directory lacks it. */
export function releaseEvidence(): ReleaseEvidence {
  return verifyEvidence().record as ReleaseEvidence;
}
