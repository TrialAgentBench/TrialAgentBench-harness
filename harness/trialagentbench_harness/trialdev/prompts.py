"""System prompt and per-phase context builders for the TrialDevBench agent loop.

The agent runs as a single conversation per program. It receives:

* a system prompt with the program brief, scenario surface, legal design
  levers, and the rules of the asset-development task; and
* per-phase context blocks that appear as user messages whenever the agent
  is asked to take the next action — fresh state, prior-phase summaries,
  asset lock, and materialization usage.

Prompts state the clinical-development question and available evidence without
prescribing a statistical method. The harness binds submitted science to the
current programme state after participant-schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevPublicObservationalMethodSpecV1,
)
from trialagentbench_harness.contracts.experiments import ProcedureAssistanceV1
from trialagentbench_harness.trialdev.schema import Program

SYSTEM_PROMPT_TEMPLATE = """\
You are the statistical lead for one clinical development programme. Use the
available observational and randomized evidence to advise the programme from
initial review through early safety, proof of concept, and confirmation.

PROGRAMME
* programme_id:   {program_id}
* objective:      {objective_id}

CLINICAL QUESTION
At each checkpoint, what does the available evidence show about efficacy,
safety, treatment retention, follow-up, and study feasibility, and which next
action does it support under the stated programme objective?

EVIDENCE AVAILABLE
The working directory contains the current programme files. Read files from the
root with relative paths such as `observational_extract.parquet` or
`eval_contract.json`. Available files:
{public_files}

Use execute_code (pandas as `pd`, numpy as `np` are pre-imported) or
inspect_parquet for quick parquet schema dumps.
A persistent writable `scratch/` workspace is available through dedicated
list/read/write tools for reusable scripts, intermediate files, and notes.
Workspace-tool paths omit the `scratch/` prefix; Python code uses relative paths
such as `Path("scratch/analysis.py")`, never an absolute path.
The isolated environment also includes PyArrow, SciPy, statsmodels, lifelines,
scikit-learn, and matplotlib.

After a study is completed, its analysis dataset lives at
`phase_<phase_id>/trial_output/` inside this same working directory.

WORKING APPROACH
{procedure_assistance}

OBSERVATIONAL ANALYSIS SELECTION
{observational_analysis_selection}

WORK REQUIRED
At each checkpoint, propose any required study design, analyse the available
participant-level data, report the estimate and its uncertainty, and decide
whether to continue or stop. End with the programme recommendation supported by
the complete evidence sequence.

For the **observational_review** phase, no randomized study has yet been
conducted. Submit one combined analysis-and-decision JSON through
``submit_obs_review_analysis_and_decision_file``.

ASSET CONTINUITY
After nomination, the candidate regimen is fixed for this programme. Subsequent phase
requests must use the same drug. A request for another drug is invalid.

OBJECTIVE POLICY
The programme objective is ``{objective_id}``. Its components, units,
exchange rates, and sensitivity values are declared in
``objective_charter.json`` and ``decision_charter.json``. Apply that
programme policy rather than an unstated clinical or financial convention.

A NOTE ON ENDPOINTS
The ``endpoint_id`` field on phase requests is the *primary efficacy
endpoint* (only meaningful from phase2 onward — phase1 is
dose-finding and safety, no efficacy endpoint). Safety is NOT entered as
an efficacy endpoint. Safety is monitored at every randomized phase and is
reported through the analysis submission contract.

OUTPUT CONTRACT
* Every submission must conform to its JSON schema. Validation errors
  are returned to you as tool replies; correct and resubmit.
* Observational-review and phase-analysis submissions have typed direct tools.
  Their schemas are also available in ``runtime_submission_contracts.json``.
  For a large payload, write the complete JSON object under ``scratch/``, review
  it, and call the corresponding ``submit_*_file`` tool with the prefix omitted.
  Direct and file transports validate the same analyst-owned fields. The
  harness attaches the current study and evidence identity.
* Stop when the evidence and programme policy support stopping. Do not advance
  an asset merely to reach a later checkpoint.

CONCLUSION
Recommend any action supported by the analysis. If the evidence supports more
than one action, state the uncertainty. If the requested comparison is not
identified under the stated assumptions and available data, explain the
limitation rather than reporting a causal ranking.
"""


_OUTPUT_CONTRACT_ONLY_ASSISTANCE = """\
Choose and execute a defensible analysis for the study design and available
data. State what was estimated, for whom, on which scale and time horizon, how
it was estimated, the assumptions needed for interpretation, and the associated
uncertainty."""

_TRIALDEV_ANALYSIS_COMPONENTS = (
    "inspect current state, prior evidence, and the prospective design contract",
    "submit a legal phase design and inspect the participant-level, endpoint, and safety records",
    "define the estimand, assess identification, design structure, data integrity, and relevant model assumptions, "
    "and execute the selected analysis",
    "submit effect and safety evidence with uncertainty and identify the records used",
    "choose an admissible action supported by that evidence and preserve the decision sequence",
)
_TRIALDEV_UNORDERED_COMPONENTS = "; ".join(_TRIALDEV_ANALYSIS_COMPONENTS)
_TRIALDEV_ORDERED_COMPONENTS = "\n".join(
    f"{index}. {component}." for index, component in enumerate(_TRIALDEV_ANALYSIS_COMPONENTS, start=1)
)
_TRIALDEV_ASSISTED_PREAMBLE = """\
The programme files define the scientific objective, admissible actions, design
bounds, available evidence, state transitions, and a neutral prospective
method catalogue."""
_TRIALDEV_ASSISTED_BOUNDARY = """\
Independently determine which analysis is defensible. A supplied method-route
identifier must agree with the estimator, estimand, scale, horizon, population,
and uncertainty actually executed. The catalogue and workflow do not prescribe
a conclusion or action."""
_UNORDERED_CHECKLIST_ASSISTANCE = f"""\
{_TRIALDEV_ASSISTED_PREAMBLE}
Complete the following operations in any order: {_TRIALDEV_UNORDERED_COMPONENTS}.
{_TRIALDEV_ASSISTED_BOUNDARY}"""

_ORDERED_SOP_ASSISTANCE = f"""\
{_TRIALDEV_ASSISTED_PREAMBLE}
Complete the same operations in this required order:
{_TRIALDEV_ORDERED_COMPONENTS}
{_TRIALDEV_ASSISTED_BOUNDARY}"""

_ASSISTANCE_TEXT: dict[ProcedureAssistanceV1, str] = {
    "output_contract_only": _OUTPUT_CONTRACT_ONLY_ASSISTANCE,
    "unordered_checklist": _UNORDERED_CHECKLIST_ASSISTANCE,
    "ordered_sop": _ORDERED_SOP_ASSISTANCE,
}


def procedure_assistance_prompt(procedure_assistance: ProcedureAssistanceV1) -> str:
    """Return the participant-facing workflow support for one condition."""

    return _ASSISTANCE_TEXT[procedure_assistance]


def build_system_prompt(
    program: Program,
    public_dir: Path,
    *,
    max_turns_per_step: int,
    procedure_assistance: ProcedureAssistanceV1,
    observational_analysis_specification: TrialDevPublicObservationalMethodSpecV1 | None = None,
) -> str:
    """Compose the system prompt for one program run."""
    if max_turns_per_step < 1:
        raise ValueError("max_turns_per_step must be at least 1.")
    public_files = sorted(p.name for p in public_dir.iterdir() if p.is_file())
    public_files_str = "\n".join(f"  - {name}" for name in public_files)
    if observational_analysis_specification is None:
        selection = (
            "Select and execute the observational method justified by the available evidence. "
            "No method has been selected for you."
        )
    else:
        selection = (
            "For observational_review, execute the complete prospective method in "
            "`observational_analysis_specification.json`. Do not substitute another method. "
            "The specification contains no fitted result, diagnostic conclusion, candidate ranking, "
            "or development action; derive all of those from the available evidence."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(
        program_id=program.program_id,
        objective_id=program.objective_id,
        public_files=public_files_str or "  (no files staged)",
        procedure_assistance=procedure_assistance_prompt(procedure_assistance),
        observational_analysis_selection=selection,
    )


# ---------------------------------------------------------------------------
# Per-phase context blocks (delivered as user messages between tool turns)
# ---------------------------------------------------------------------------


_OBS_REVIEW_BLOCK = """\
CHECKPOINT: Observational evidence review

You are at the programme-entry review. No randomized study has yet been
conducted. The available participant-level evidence is in
`observational_extract.parquet`.

Your task at this checkpoint:

  1. Inspect the observational extract.
  2. Decide whether to start a development programme. When candidate effects
     can be estimated under the stated assumptions, report each candidate's
     estimate and ranking even if none meets the entry criterion. Withhold a
     causal ranking only when the comparison is not identified or cannot be
     estimated defensibly, and cite the evidence supporting that limitation.
  3. Submit one evidence-linked observational analysis and decision using
     ``submit_obs_review_analysis_and_decision``. The runtime submission
     contract defines the required scientific content and legal actions. For a
     large payload, write it under ``scratch/`` and use
     ``submit_obs_review_analysis_and_decision_file`` instead.

Use the drug and variable names defined in the programme files. When the
evidence does not support a causal ranking, report that limitation and withhold
nomination rather than inventing a point comparison.

"""


_PHASE_REQUEST_BLOCK = """\
STUDY DESIGN: {phase_id}

It's time to design the {phase_id} trial.

Programme identity: {program_id}
Programme primary objective: {program_objective}
{objective_reminder}
Current programme state:
  - active_asset_id:          {active_asset}
  - completed_checkpoint_ids: {completed}
  - retired_asset_ids:        {retired}
{prior_phase_summaries}
Permitted design choices for this phase (from eval_contract.json):
{phase_levers}

Submit a phase request via submit_phase_request. The request records your
prospective design proposal. Once the design is accepted, the study records
available for analysis will be staged in the working directory.
"""


_PHASE_ANALYSIS_BLOCK = """\
STUDY DATA AVAILABLE — {phase_id}

The study is complete and its analysis records are available at:
  {trial_output_path}

Files (read via inspect_parquet / execute_code with paths relative to your CWD):
  - {output_relpath}/participants.parquet  ({n_participants} subjects)
  - {output_relpath}/endpoints.parquet     (with EVENT, TIME, FOLLOW_UP_DAYS columns)
  - {output_relpath}/safety.parquet        (AE_*, DISCONTINUATION_*, LTFU_* columns)
  - {output_relpath}/phase_summary_public.json (arm counts and event-rate summary)
  - {output_relpath}/arm_mapping.json      (control vs treatment arm IDs)

Analyse the study against its stated objective, estimand, and decision policy,
then call
``submit_phase_analysis``. For a large payload, write it as a complete JSON file
under ``scratch/`` and call ``submit_phase_analysis_file`` instead.
"""


_PHASE_DECISION_BLOCK = """\
PROGRAMME DECISION — {phase_id}

Your analysis was accepted. Now choose a decision_action.

Permitted decision_action values for this phase ({phase_id}):
{action_menu}
{policy_notes}
Set candidate_drug_id when the action advances the nominated asset or declares success
(this drug is asset-locked / nominated), and provide a clear decision_rationale
(max 4000 chars).
"""


def build_obs_review_block() -> str:
    """Render the fixed observational-review prompt."""

    return _OBS_REVIEW_BLOCK


def build_phase_request_block(
    *,
    phase_id: str,
    state_summary: dict[str, Any],
    phase_module: dict[str, Any],
    prior_phase_summaries: list[dict[str, Any]],
    program_id: str = "(unknown)",
    program_objective: str = "(unknown)",
) -> str:
    active_asset = str(state_summary.get("active_asset_id") or "(none)")
    completed = ", ".join(state_summary.get("completed_checkpoint_ids", [])) or "(none)"
    retired = ", ".join(state_summary.get("retired_asset_ids", [])) or "(none)"

    allowed_obj = list(phase_module.get("allowed_selection_objectives") or [])
    objective_reminder = (
        f"Programme objective: {program_objective}. Allowed selection objectives for this phase: {allowed_obj}."
    )

    levers_lines = []
    for key in (
        "allowed_selection_objectives",
        "allowed_endpoint_ids",
        "allowed_follow_up_days",
        "allowed_enrollment_window_days",
        "allowed_site_count_budgets",
        "allowed_allocation_ratios",
        "allowed_treatment_discontinuation_strategies",
        "allowed_interim_policies",
        "allowed_site_strategies",
        "max_sample_size",
        "max_analysis_covariates",
        "max_subgroup_splits",
        "allowed_variable_ids",
    ):
        value = phase_module.get(key)
        if value not in (None, [], (), {}):
            levers_lines.append(f"  - {key}: {value}")
    levers = "\n".join(levers_lines) or "  (none documented)"

    if prior_phase_summaries:
        prior_lines = ["", "Prior phase summaries (already completed):"]
        for summary in prior_phase_summaries:
            prior_lines.append(
                f"  - {summary.get('phase_id', '?')}: "
                f"effect={summary.get('primary_effect')}, "
                f"action={summary.get('decision_action')}, "
                f"chosen_drug={summary.get('candidate_drug_id')}"
            )
        prior_lines.append("")
        prior_block = "\n".join(prior_lines)
    else:
        prior_block = "\n"

    return _PHASE_REQUEST_BLOCK.format(
        phase_id=phase_id,
        program_id=program_id,
        program_objective=program_objective,
        objective_reminder=("\n" + objective_reminder + "\n") if objective_reminder else "",
        active_asset=active_asset,
        completed=completed,
        retired=retired,
        prior_phase_summaries=prior_block,
        phase_levers=levers,
    )


def build_phase_analysis_block(
    *,
    phase_id: str,
    trial_output_summary: dict[str, Any],
) -> str:
    relpath = trial_output_summary.get("trial_output_relpath")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("Phase-analysis prompt requires a participant-relative trial output path.")
    return _PHASE_ANALYSIS_BLOCK.format(
        phase_id=phase_id,
        trial_output_path=relpath,
        output_relpath=relpath,
        n_participants=trial_output_summary["n_participants"],
    )


def build_phase_decision_block(
    *,
    phase_id: str,
    action_spec: TrialDevelopmentPhaseActionSpecV1,
) -> str:
    allowed = list(action_spec.allowed_action_ids)
    terminal = set(action_spec.terminal_action_ids)
    lines = []
    for action in allowed:
        tag = " (terminal — ends programme)" if action in terminal else ""
        lines.append(f"  - {action}{tag}")
    action_menu = "\n".join(lines)
    notes = action_spec.notes
    policy_notes = f"\nPolicy note: {notes}\n" if notes else ""
    return _PHASE_DECISION_BLOCK.format(
        phase_id=phase_id,
        action_menu=action_menu,
        policy_notes=policy_notes,
    )


def get_phase_module(public_dir: Path, phase_id: str) -> dict[str, Any]:
    """Pull one phase's lever menu from the eval contract."""
    payload = json.loads((public_dir / "eval_contract.json").read_text(encoding="utf-8"))
    for module in payload.get("phase_modules", []) or []:
        if str(module.get("phase_id")) == str(phase_id):
            return dict(module)
    raise ValueError(f"Evaluation contract does not define phase module {phase_id!r}.")


__all__ = [
    "build_system_prompt",
    "build_obs_review_block",
    "build_phase_request_block",
    "build_phase_analysis_block",
    "build_phase_decision_block",
    "get_phase_module",
]
