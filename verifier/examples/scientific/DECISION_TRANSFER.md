# Do benchmark results transfer to real research decisions?

Proposed study, 11 October 2026. The study has not been registered or executed.

The public real-record experiment reconstructs treatment contrasts in 33 trials.
It tests analysis on real data. This study asks a different question: does the
better of two frozen TrialDev policies also make more acceptable research
recommendations on independent clinical programmes?

## Cases and comparisons

First, two practising trial analysts and the two policies attempt four disclosed
TrialDev programmes. Analysts report missing information, ambiguous questions and
disagreement with native scores. Repair material task defects before freezing the
external study. These pilot cases do not enter the external estimates.

Order the two policies using a prespecified synthetic development-value comparison,
with completion and resources reported. Fix a tie rule before scoring. External
case performance cannot select or tune the policies.

An independent custodian then assembles an eligibility frame of real development
programmes with authorized protocols, participant records, benefit and harm evidence,
and documented practical constraints. Exclude benchmark construction sources,
clinical calibration programmes, RCT_Bench records and pilot cases. Prefer dossiers
unavailable during model training; public historical cases require a separate
contamination assessment.

The working allocation is 12 programmes across three clinical areas, sampled
randomly from the frozen frame. Use one historical decision cutoff per programme.
Publish eligibility decisions, clinical coverage and the sampling seed. This is
a feasibility allocation, not a power calculation; finalize it after the pilot
and before exposing external cases.

## The task

Both policies and both analysts receive identical cutoff evidence, tools, local
reference materials and resource constraints. Each submits an executable benefit
and harm analysis with uncertainty, plus a recommendation to stop, advance, or
commission a specified next study. Several recommendations may be defensible.

Randomize case order and reset agent context between cases. Give analysts the same
onboarding and tutorial; analyst attempts use no LLM assistance. Subsequent outcomes,
historical decisions, reference judgments and internet access are withheld.

Two independent clinical-methodological adjudicators seal case-specific acceptable
alternatives and major-error criteria before seeing submissions. They then assess
anonymized submissions separately in random order. A third resolves disagreements.
Report initial agreement and all adjudicated changes. Adjudicators see neither
policy identity nor benchmark scores. Analyst attempts are comparators, not an
answer key.

A recommendation is acceptable when it is actionable and contains no major error
in the treatment question, benefit–harm interpretation, uncertainty or feasibility.
Each case rubric defines why an error is consequential. Retain component judgments.
The historical sponsor's choice and eventual treatment success cannot reveal what
an alternative unrun study would have achieved.

## Analysis

The primary estimand is the paired difference in adjudicated acceptability, higher-
ranked minus lower-ranked policy, averaged equally across programmes. Report both
marginal proportions and all four paired outcome counts. Construct conservative
95% bounds by subtracting opposite bounds of 97.5% Clopper–Pearson intervals for
the two discordant probabilities. The programme is the independent unit.

Secondary analyses compare each policy with the mean of the two analysts on each
case and describe major errors, completion, reproducibility and resource use.
Use paired programme-bootstrap intervals, conditional on these analysts, and retain
their individual results. No noninferiority margin is assumed. Two policies test
this transfer contrast; they cannot establish a general correlation between
benchmark scores and clinical ability.

## Execution and release

Allow one primary attempt per participant and case, with a two-hour limit including
replay. Freeze the same agent spending ceiling before launch. The working pilot
and external allocations imply 32 agent attempts and 64 analyst attempt-hours,
excluding onboarding and adjudication. Report tokens, compute, spending and time
separately.

Timeouts, invalid recommendations and unreproduced required analyses fail the
acceptability criterion. Permit one documented rerun for an external infrastructure
failure; retain the original. Unstarted funding-censored cases remain visible and
preclude a full-panel transfer claim.

Before pilot attempts, timestamp the participants, task selection and budgets.
Before external access, register the sampling frame, rubric, policy settings,
selection rule, analysis code, stopping and retry rules, and amendments. Holdout
dossiers and reference keys remain outside participant access. Changes after
exposure require new holdouts.

The design adapts matched resource comparisons and expert feasibility checks from
[RE-Bench](https://arxiv.org/html/2411.15114v2), and clinician-authored tasks with
curated references from [MedAgentBench](https://arxiv.org/html/2501.14654v2).
RE-Bench allowed human LLM use; this comparison explicitly measures unaided analysts.
