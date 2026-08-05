"""Scenario bundle inspection helpers for benchmark users."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trialagentbench_harness.trialdev.share.io import read_json
from trialagentbench_harness.trialdev.share.models import PhaseModuleSpecV1, TrialDevelopmentEvalContractV1
from trialagentbench_harness.trialdev.share.safety_policy import serious_event_definitions_v1
from trialagentbench_harness.trialdev.share.validate import validate_public_scenario_bundle_v1

__all__ = ["inspect_scenario_bundle_v1"]


def _risk_at_horizon(*, time: pd.Series, event: pd.Series, horizon: float) -> float:
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(event, errors="coerce").to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("Scenario inspection cannot estimate risk from an empty evidence surface.")
    if not np.isfinite(e).all() or not np.isin(e, (0.0, 1.0)).all():
        raise ValueError("Scenario inspection requires complete binary 0/1 event indicators.")
    occurred = e.astype(bool) & np.isfinite(t) & (t <= float(horizon))
    return float(occurred.mean())


def inspect_scenario_bundle_v1(*, scenario_root: Path, include_hidden_diagnostics: bool = False) -> dict[str, object]:
    """
    Inspect a built scenario bundle for review and debugging.

    Parameters
    ----------
    scenario_root
        Scenario bundle root directory (contains `public/` and, for full releases,
        `hidden/`).

    Returns
    -------
    dict[str, object]
        JSON-serializable summary suitable for printing or writing to disk.
    """
    root = Path(scenario_root)
    public_dir = root / "public"
    if not public_dir.is_dir():
        raise FileNotFoundError("Scenario bundle missing public surface.")
    validate_public_scenario_bundle_v1(scenario_root=root)

    var_catalog = read_json(public_dir / "variable_catalog.json")
    variables = list(var_catalog.get("variables", []) or [])
    variable_ids = [str(v.get("variable_id", "")).strip() for v in variables if str(v.get("variable_id", "")).strip()]

    drug_catalog = read_json(public_dir / "candidate_drug_catalog.json")
    drugs = list(drug_catalog.get("candidate_drugs", []) or [])
    drug_ids = [
        str(d.get("candidate_drug_id", "")).strip() for d in drugs if str(d.get("candidate_drug_id", "")).strip()
    ]

    endpoint_catalog = read_json(public_dir / "endpoint_catalog.json")
    endpoints = list(endpoint_catalog.get("endpoints", []) or [])
    efficacy_endpoint_ids = [
        str(e.get("endpoint_id", "")).strip()
        for e in endpoints
        if str(e.get("endpoint_id", "")).strip() and str(e.get("kind", "")) == "efficacy"
    ]
    terminal_endpoint_ids = [
        str(endpoint.get("endpoint_id", "")).strip()
        for endpoint in endpoints
        if str(endpoint.get("kind", "")) == "efficacy" and str(endpoint.get("mechanism_role", "")) == "terminal"
    ]
    if len(terminal_endpoint_ids) != 1:
        raise ValueError("Endpoint catalog requires exactly one terminal efficacy endpoint.")

    serious_event_definitions = serious_event_definitions_v1(scenario_root=root)

    eval_contract = TrialDevelopmentEvalContractV1.model_validate(read_json(public_dir / "eval_contract.json"))
    modules: dict[str, PhaseModuleSpecV1] = {str(module.phase_id): module for module in eval_contract.phase_modules}
    phase_request_envelopes = {
        str(phase_id): {
            "allowed_follow_up_days": list(module.allowed_follow_up_days),
            "allowed_allocation_ratios": list(module.allowed_allocation_ratios),
            "max_sample_size": module.max_sample_size,
            "max_analysis_covariates": module.max_analysis_covariates,
            "max_subgroup_splits": module.max_subgroup_splits,
            "allowed_selection_objectives": list(module.allowed_selection_objectives),
        }
        for phase_id, module in sorted(modules.items())
    }
    phase2 = modules.get("phase2")
    phase3 = modules.get("phase3")
    if phase2 is None or not phase2.allowed_follow_up_days:
        raise ValueError("Public evaluation contract lacks a phase-2 follow-up horizon.")
    if phase3 is None or not phase3.allowed_follow_up_days:
        raise ValueError("Public evaluation contract lacks a phase-3 follow-up horizon.")
    phase2_horizon = float(phase2.allowed_follow_up_days[0])
    phase3_horizon = float(phase3.allowed_follow_up_days[0])

    obs_path = public_dir / "observational_extract.parquet"
    obs = pd.read_parquet(obs_path)

    if "TREATMENT" not in obs.columns:
        raise ValueError("Observational extract lacks required TREATMENT assignment.")
    treatment_shares = obs["TREATMENT"].astype("string").value_counts(normalize=True).to_dict()
    treatment_shares = {str(k): float(v) for k, v in sorted(treatment_shares.items())}

    missingness = obs.isna().mean().sort_values(ascending=False)
    top_missingness = {str(k): float(v) for k, v in missingness.head(12).to_dict().items()}

    endpoint_rates: dict[str, dict[str, float]] = {}
    for endpoint_id in efficacy_endpoint_ids:
        outcome_id = f"EFF_{endpoint_id}"
        t_col = f"{outcome_id}_T"
        e_col = f"{outcome_id}_E"
        if t_col not in obs.columns or e_col not in obs.columns:
            raise ValueError(f"Observational extract lacks efficacy endpoint columns for {endpoint_id!r}.")
        endpoint_rates[str(endpoint_id)] = {
            "risk_at_phase2_horizon": _risk_at_horizon(time=obs[t_col], event=obs[e_col], horizon=phase2_horizon),
            "risk_at_phase3_horizon": _risk_at_horizon(time=obs[t_col], event=obs[e_col], horizon=phase3_horizon),
        }

    terminal_endpoint_rates = {}
    terminal_t_col = f"EFF_{terminal_endpoint_ids[0]}_T"
    terminal_e_col = f"EFF_{terminal_endpoint_ids[0]}_E"
    if terminal_t_col in obs.columns and terminal_e_col in obs.columns:
        terminal_endpoint_rates = {
            "risk_at_phase2_horizon": _risk_at_horizon(
                time=obs[terminal_t_col], event=obs[terminal_e_col], horizon=phase2_horizon
            ),
            "risk_at_phase3_horizon": _risk_at_horizon(
                time=obs[terminal_t_col], event=obs[terminal_e_col], horizon=phase3_horizon
            ),
        }

    discontinuation = {}
    if "DISCONTINUATION_T" in obs.columns and "DISCONTINUATION_E" in obs.columns:
        discontinuation = {
            "risk_at_phase2_horizon": _risk_at_horizon(
                time=obs["DISCONTINUATION_T"],
                event=obs["DISCONTINUATION_E"],
                horizon=phase2_horizon,
            ),
            "risk_at_phase3_horizon": _risk_at_horizon(
                time=obs["DISCONTINUATION_T"],
                event=obs["DISCONTINUATION_E"],
                horizon=phase3_horizon,
            ),
        }

    ltfu = {}
    if "LTFU_T" in obs.columns and "LTFU_E" in obs.columns:
        ltfu = {
            "risk_at_phase2_horizon": _risk_at_horizon(
                time=obs["LTFU_T"], event=obs["LTFU_E"], horizon=phase2_horizon
            ),
            "risk_at_phase3_horizon": _risk_at_horizon(
                time=obs["LTFU_T"], event=obs["LTFU_E"], horizon=phase3_horizon
            ),
        }

    ae_event_rates: dict[str, float] = {}
    for definition in serious_event_definitions:
        t_col = definition.time_column
        e_col = definition.event_column
        if t_col not in obs.columns or e_col not in obs.columns:
            raise ValueError(f"Observational extract lacks serious-event columns for {definition.endpoint_id!r}.")
        ae_event_rates[definition.endpoint_id] = _risk_at_horizon(
            time=obs[t_col], event=obs[e_col], horizon=phase2_horizon
        )

    hidden_closure: dict[str, object] | None = None
    warnings: list[str] = []
    hidden_dir = root / "hidden"
    if bool(include_hidden_diagnostics) and hidden_dir.is_dir():
        qual_path = hidden_dir / "superpopulation_qualification_summary.json"
        if qual_path.is_file():
            qual = read_json(qual_path)
            realism = dict((qual.get("realism_summary", {}) or {}) if isinstance(qual, dict) else {})
            limiting = list(qual.get("limiting_factors", []) or []) if isinstance(qual, dict) else []
            hidden_closure = {
                "sufficient_for_release": (
                    bool(qual.get("sufficient_for_release")) if isinstance(qual, dict) else None
                ),
                "limiting_factors": [str(x) for x in limiting],
                "diagnostic_observations": [str(x) for x in list(qual.get("diagnostic_observations", []) or [])],
                "baseline_dimensionality": realism.get("baseline_dimensionality"),
                "candidate_drug_count": realism.get("candidate_drug_count"),
                "confounding_index": realism.get("confounding_index"),
                "severity_overlap_index": realism.get("severity_overlap_index"),
                "naive_phase2_rank_correlation": realism.get("naive_phase2_rank_correlation"),
                "principled_phase2_rank_correlation": realism.get("principled_phase2_rank_correlation"),
                "naive_best_drug_phase2": realism.get("naive_best_drug_phase2"),
                "standardized_best_drug_phase2": realism.get("standardized_best_drug_phase2"),
                "counterfactual_best_drug_phase2": realism.get("counterfactual_best_drug_phase2"),
            }
            if realism.get("naive_best_drug_phase2") != realism.get("counterfactual_best_drug_phase2"):
                warnings.append("Naive phase-2 endpoint ranking disagrees with the hidden counterfactual ranking.")

    summary: dict[str, object] = {
        "scenario_id": root.name.replace("scenario_", "", 1),
        "paths": {"scenario_root": str(root), "observational_extract": str(obs_path)},
        "dimensions": {
            "n_subjects": int(obs.shape[0]),
            "n_columns": int(obs.shape[1]),
            "n_baseline_variables": int(len(variable_ids)),
            "n_candidate_drugs": int(len(drug_ids)),
            "n_ae_families": int(len(serious_event_definitions)),
        },
        "treatment_shares": treatment_shares,
        "top_missingness": top_missingness,
        "endpoint_rates": endpoint_rates,
        "terminal_endpoint_rates": terminal_endpoint_rates,
        "discontinuation": discontinuation,
        "loss_to_followup": ltfu,
        "ae_event_rates_phase2_horizon": {k: float(v) for k, v in sorted(ae_event_rates.items())},
        "phase_request_envelopes": phase_request_envelopes,
        "warnings": warnings,
    }
    if hidden_closure is not None:
        summary["hidden_closure"] = hidden_closure
    return summary
