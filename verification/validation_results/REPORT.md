# Simulation validity

TrialAgentBench uses synthetic clinical-trial evidence to test whether an agent
changes its analysis when the design, assumptions, or available information
changes. That evaluation requires the data to preserve the trial structures
used by the analysis and to respond correctly when a statistical mechanism is
varied. Validation therefore combines external trial comparisons, controlled
mechanism changes, and independent statistical recovery.

## Evidence base

The evidence base combines the released TrialEval trials with public
participant-level randomized trials, detailed source-trial outcome studies,
recurrent-event studies, and cross-domain clinical datasets. These sources
play different roles: public trials anchor observable clinical structure;
controlled simulations test whether known changes are recovered; and
structure-breaking controls test whether an apparently plausible dataset fails
when patient, treatment, visit, or event-time relationships are disrupted.
Exact study and participant counts are reported with the corresponding
results.

## Validation questions

A synthetic dataset can match the distribution of every variable while losing
the relationships that make a clinical-trial analysis valid. Source-trial
comparisons therefore test observable outcomes and follow-up, while
participant-linkage controls test whether correlations and adjusted treatment
estimates survive generation. Controlled changes to trial design and
statistical assumptions then test whether independent analyses respond in the
expected direction. Independent grading reconstruction and downstream
development consequences are separate questions linked through the
[verification guide](../../docs/VERIFICATION_GUIDE.md).

## Analyses

| Chapter | Contents |
|---|---|
| [TrialEval release contents](reports/trialeval-release-contents.md) | Trial questions, accepted same-question analyses, a worked base trial, the base-trial census, and matched context reconstruction. |
| [Trial design and assumption response](reports/trial-design-and-assumption-response.md) | Randomization, pragmatic conduct, covariate structure, endpoint ascertainment, clustering, stepped-wedge rollout, sequential monitoring, and supported analyses under assumption failure. |
| [Source-trial anchoring](reports/source-trial-anchoring.md) | External trial distributions and source-sized PATENCY, HeadSOAR, and TERECO outcome comparisons. |
| [Participant-linkage preservation](reports/participant-linkage-preservation.md) | Whole-participant resampling against independent-column controls and the consequences for dependence and adjusted treatment analysis. |
| [Mechanism and effect recovery](reports/mechanism-and-effect-recovery.md) | Known-effect recovery, dropout, heterogeneity, competing risks, confounding, recurrent events, cross-domain linkage, structure-breaking controls, and independent estimation. |

The [methods](METHODS.md) define every comparison and uncertainty calculation.
[RESULTS.csv](RESULTS.csv) contains the complete numerical index, and
[SOURCES.md](SOURCES.md) identifies the external datasets and analysis sources.

## Results

| Evidence domain | Result | Interpretation |
|---|---|---|
| TrialEval release contents | 100 independent base trials, 500 matched context views, and 610,190 synthetic participants across seven design profiles | Context views change the analysis information without inflating the number of independent trials. |
| Source-trial anchoring | Three-year Kaplan-Meier error was 0.00134 survival-probability units; seven-level ordinal-category error was 0.00443 probability units; all 36 longitudinal arm-by-visit means and follow-up counts lay within repeated-trial 95% predictive intervals | Generated trials preserve the outcome, follow-up, and dependence structures used by the corresponding public-trial analyses. |
| Participant-linkage preservation | Whole-participant resampling reduced correlation error by 30% and adjusted-treatment bias by 58% relative to independently sampling columns | Keeping participant records intact preserves relationships required by adjusted analyses. |
| Trial-design and mechanism response | Cluster-aware and period-adjusted coverage were 0.957 at their declared reference settings; the binary treatment-effect recovery slope was 0.994 | Encoded designs and mechanisms produce their expected analytical consequences. |

Complete estimates, intervals, units, and denominators are indexed in
[RESULTS.csv](RESULTS.csv) and explained in the evidence chapters above.

## Interpretation

Across the evaluated trial families, the simulator reproduces outcomes on
their clinical scales, retains participant relationships that affect analysis,
responds smoothly to known changes in treatment and nuisance mechanisms, and
exposes the expected failure when those relationships are deliberately broken.

The results establish testable control over the statistical properties required
by survival, ordinal, longitudinal, competing-risk, confounding,
heterogeneous-effect, and recurrent-event analyses. Comparing the observed
process, varying a known mechanism, and recovering its consequence
independently provides the same test for additional trial families.

The [methods](METHODS.md) specify each analysis. Complete estimates, intervals,
denominators, and source identities are available in the
[numerical results](RESULTS.csv), [source register](SOURCES.md), and figure
data.
