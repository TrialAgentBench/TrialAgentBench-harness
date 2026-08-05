# How the benchmark data encode the statistical task

## TrialEval changes the evidence, not the question

TrialEval asks a model to perform a supported primary analysis of one synthetic
clinical trial. The construction deliberately injects known design and
assumption conditions so that the benchmark can distinguish methods that remain
supported from methods invalidated by that condition.

The hierarchy is:

| Level | Meaning | Release census |
| --- | --- | ---: |
| Design profile | allocation and study structure | 7 |
| Evaluation series | recurring scientific question and endpoint setting | 9 |
| Regime cell | series-specific assumption condition | 25 |
| Base trial | independently generated dataset | 100 |
| Context item | one evidence view of a base trial | 500 |

There are four independent base trials per regime cell and five contexts per
base trial.

### Design profiles

| ID | Design |
|---|---|
| TE-DP01 | Individually randomized parallel-group survival trial |
| TE-DP02 | Individually randomized treatment-use trial with switching, rescue, adherence, and discontinuation |
| TE-DP03 | Parallel-group survival trial with reference-population standardization |
| TE-DP04 | Parallel-group survival trial with a blinded endpoint-validation substudy |
| TE-DP05 | Parallel cluster-randomized survival trial |
| TE-DP06 | Cluster-randomized stepped-wedge survival trial |
| TE-DP07 | Parallel-group survival trial with comparative group-sequential monitoring |

The evaluation series cover time-to-event outcomes, intercurrent events,
covariate standardization, endpoint ascertainment, clustered allocation,
calendar-time adjustment, and sequential monitoring. Each item states its
scientific question and supplies the evidence needed to select a supported
analysis.

### D: Design

- **D1:** individually randomized trial.
- **D2:** pragmatic or policy-oriented randomized trial.
- **D3:** covariate-structure or endpoint-ascertainment design.
- **D4:** cluster, stepped-wedge, or group-sequential design requiring an
  explicit design-aware analysis.

The design label is descriptive; the more specific design subtype controls the
analysis adjustment.

### A: Assumption

- **A1, regular:** the conventional route's material assumptions hold.
- **A2, stressed:** a mild, diagnosable departure is present.
- **A3, consequential and recoverable:** the observed departure makes the default
  route materially misleading for the declared question, while at least one
  prespecified alternative remains supported.
- **A4, incompatible default:** the conventional route is not supported by the
  observed design, data support, or analysis timing. The appropriate conclusion
  may be an alternative estimate, an identified range, qualified
  non-estimability, or abstention.

These levels are series-specific. They describe the consequence of the
observed condition for the declared analysis; they are not claims about
universal method performance. A4 requires a supported non-default response for
the stated question.

### C: Context

- **C1:** analysis-ready data with the locked analysis specification.
- **C2:** analysis-ready data with protocol-level specification.
- **C3:** less processed data with the locked analysis specification.
- **C4:** less processed data with protocol-level specification.
- **C5:** C4 plus one declared, exactly repairable transport duplication. The
  public `validate_data_integrity` operation checks that the agent's repaired
  Parquet input removes one identical copy per duplicated compound key, fails
  on any ambiguous or inexact state, and returns the checksum-bound
  analysis-input record.

All five views retain the same question and scientific route universe. The
credit set exposed by the scoring key depends on the instructions supplied:
C1 and C3 prescribe exactly one accepted route through a locked analysis
specification; C2, C4, and C5 expose the complete accepted same-estimand route
set for protocol-guided selection. C5 has the same eligible route set as C4
after exact repair. The contexts test evidence use and data handling, not
different scientific targets.

### Intercurrent events and missing data

Every score-bearing estimand binds each relevant intercurrent event to an
explicit strategy. Treatment discontinuation, rescue therapy, nonadherence,
and switching enter that map only when they affect the stated treatment-effect
question. Endpoints, competing events, and safety outcomes retain their own
declared roles. Attendance, missing observations, censoring, and loss to
follow-up belong to the observation process and are handled separately. Death
can occupy different roles for different questions; it is not automatically an
intercurrent event. An empty intercurrent-event binding is valid when no event
changes the stated estimand.

## TrialDev carries the conclusion into later decisions

TrialDev asks a model to analyse the evidence available now and make the next
decision in a clinical development programme. It does not ask the model to
recover a hidden best trajectory. The harness executes a contract-valid legal
decision and the grader separately reports whether the analysis, uncertainty,
action, and study plan are supported by the available evidence and disclosed
programme policy. This permits the benchmark to measure both analytical
quality and the practical consequences of imperfect decisions.

The release contains two tasks that are reported separately:

| Stream | Decision problem | Released evaluation units |
|---|---|---:|
| Single-asset development | Decide whether to nominate one regimen, then design, analyse, continue, or stop an irreversible phase 1–3 chain | 50 trajectories evaluated under four objectives, yielding 200 matched programme views |
| Bounded portfolio reallocation | Select a lead and reserve from three regimens, retain or retire options, and make at most one reserve promotion under a fixed resource budget | 96 programme-policy views |

The 96 portfolio views are 12 independently generated evidence worlds crossed
with four objective policies and two resource budgets. Those eight views of one
world are matched policy questions, not eight independent datasets. The world
stores each asset–phase episode once; a reached action releases the corresponding
precommitted records and never generates a favourable outcome on demand.

### Common definitions

| Term | Operational meaning |
|---|---|
| Available evidence | Checksummed data and protocol records named by the current programme state; unreached and unselected evidence is unavailable. |
| Statistical conclusion | A declared estimate and interval for a comparison identified under the stated assumptions and public provenance, or a qualified conclusion that those conditions do not support a point comparison. |
| Feasible action | An action permitted by the current state, remaining resources, safety exclusions, and study-design menu. |
| Supported action | A feasible action compatible with the independently reconstructed statistical conclusion and the disclosed objective policy. |
| Programme state | The immutable evidence and decision history, current checkpoint, active and retired assets, switch use, and remaining resources. |
| Structural nonreach | A later checkpoint that correctly does not occur after a supported terminal action; it is not model failure. |

Observational evidence may support several lead–reserve orderings, one ordering,
no efficacy-qualified regimen, or no defensible causal ordering. In the last
case the correct task is to report the identification limitation and withhold
allocation; candidate causal effects must not be invented. Randomized phases
use the declared estimand, effect direction, interval, safety policy, and
prospective study design. Multiple actions receive equal primary credit whenever
all remain supported after uncertainty is propagated at an exploratory
checkpoint. Confirmation has one evidence-based conclusion: success when both
efficacy and safety clearly pass, failure when either clearly fails, and
inconclusive otherwise.

“Identified” does not mean that an observational assumption has been
empirically proved. It means that the point comparison follows under the
declared assumptions and the disclosed treatment-assignment provenance. Each
observational method also declares its deterministic bootstrap seed, random
number generator, standard-error convention, and interval construction so a
participant and an independent verifier can reproduce the uncertainty record.

The observational objective charter defines every utility term on a common
time horizon. Efficacy is the standardized control risk minus candidate risk;
serious adverse events, discontinuation, and loss to follow-up enter as the
standardized candidate risk minus control risk when included by the chosen
objective. These are comparative cumulative-incidence estimands, not absolute
full-record event counts. The charter supplies their weights and direction, so
the same participant-visible formula determines both the analysis and its
programme implication.

The participant receives only the current state and evidence. Earlier accepted
records are immutable, later data cannot repair an earlier score, and future or
counterfactual branches never enter the workspace. Programme trajectories,
rather than checkpoints or objective views, are the independent units for statistical
uncertainty.

## Why synthetic construction is used

Synthetic generation provides controlled conditions and repeatability. Each
released condition is evaluated for observable expression in participant data,
its stated statistical consequence, recovery by a supported method, and failure
of structure-breaking shortcuts. External trial data and methodological
literature anchor the evaluated distributions, designs, and analysis routes.

The separately distributed validation report tests source-scale outcome
replication, participant-level dependence, analysis consequences, effect
recovery, and structural controls across the benchmark's statistical
mechanisms. It binds its methods, numerical results, figures, and source
identities to the corresponding release.

## Why the archive roles are separated

The participant archive contains model-visible tasks, data dictionaries,
protocol or analysis-specification context, policies, schemas, and tool
instructions. The evaluator archive contains score-time route or action
records. The verification archive contains safe independent replay evidence.
Generating state, latent variables, construction-only diagnostics, and
answer-bearing evaluator records are forbidden from participant workspaces.
