"""Adapter boundary for TrialDev shared models and hashing utilities.

This module is the only place the harness should import `trialagentbench_harness.trialdev.share`
directly. Other modules should import the required symbols from here.
"""

from __future__ import annotations

from trialagentbench_harness.trialdev.share.hashing import sha256_file_hex
from trialagentbench_harness.trialdev.share.models import PhaseModuleSpecV1, TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPublicObservationalMethodCatalogV1,
    TrialDevPublicObservationalMethodSpecV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentAnalysisDiagnosticV1,
    TrialDevelopmentCandidateUtilityEstimateV1,
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
    TrialDevelopmentTrialOutputManifestV1,
    TrialDevProgrammeStateV1,
)
from trialagentbench_harness.trialdev.share.validate import candidate_ids_by_role_v1

__all__ = [
    "TrialDevelopmentAnalysisDiagnosticV1",
    "TrialDevelopmentCandidateUtilityEstimateV1",
    "TrialDevelopmentObservationalReviewSubmissionV1",
    "TrialDevelopmentPhaseAnalysisSubmissionV1",
    "TrialDevelopmentPhaseActionPolicyV1",
    "TrialDevelopmentPhaseActionSpecV1",
    "TrialDevelopmentPhaseDecisionSubmissionV1",
    "TrialDevProgrammeStateV1",
    "TrialDevPublicObservationalMethodCatalogV1",
    "TrialDevPublicObservationalMethodSpecV1",
    "PhaseModuleSpecV1",
    "TrialDevelopmentRequestV1",
    "TrialDevelopmentTrialOutputManifestV1",
    "candidate_ids_by_role_v1",
    "sha256_file_hex",
]
