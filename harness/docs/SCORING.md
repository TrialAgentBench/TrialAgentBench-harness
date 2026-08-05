# How analyses and decisions receive credit

## The question is fixed before the method

The benchmark scores the compatibility of an agent's declared analysis or
decision with a prespecified question, the evidence supplied, and the
assumptions that remain supportable. Simulation parameters are used to
construct and test the benchmark; they are not a universal answer key.

Each TrialEval scoring key contains one or more **credit-eligible routes**.
Routes attached to the same item must answer the same question: they share the
analysis population, treatment, comparator, endpoint, intercurrent-event
strategies, assessment horizon, and effect scale. They may differ in estimator,
uncertainty method, or result form when those alternatives were admitted
prospectively. For example, an A4 item may admit a prespecified identified set
and a coded conclusion that a unique point is not identified; both answer the
same fixed estimand.

An answer receives route credit only when one complete route matches. Combining
the estimand from one route, estimator from another, and result from a third is
not an additional route.

## TrialEval checks the complete analysis route

The grader evaluates seven components separately:

1. **Submission:** the scheduled item has one usable primary submission.
2. **Question:** population, contrast, endpoint, intercurrent-event strategies,
   horizon, and effect scale match the fixed question.
3. **Method:** `analysis_method_id` matches one complete eligible method record
   for that question.
4. **Evidence:** every diagnostic required by that route is linked to validated
   typed evidence.
5. **Integrity:** C5 contains the exact declared repair; other contexts have no
   repair requirement.
6. **Result structure:** the result kind, unit, interval, and vector components
   are compatible with the selected route.
7. **Route comparison:** the typed value or conclusion matches the route target
   under its prespecified comparison rule and numerical tolerance.

The primary grade applies these checks as an ordered, noncompensatory cascade
and reports the first causal failure. The same record also reports every
component that remained independently evaluable, so an early evidence failure
does not hide a later numerical discrepancy. Later component outcomes are
descriptive and cannot rescue the primary grade.

No response and an incomplete response are distinct typed failures. No
response is `missing_primary_submission`; a syntactically valid submission
that omits an item-required semantic deliverable is
`missing_required_deliverable`. Both remain failed observations in experiment
denominators. Unknown deliverables in an evaluator contract are invalid
benchmark state and fail loudly instead of becoming participant failures.

Required components are conjunctive. A numerically close estimate does not
compensate for analysing the wrong population or ignoring a broken assumption.
Likewise, a different eligible route is not penalized for differing from a
reference estimator.

### A4

A4 means that the conventional route is incompatible with the observed design,
data support, or analysis timing. It does not automatically mean “refuse.”
Full credit may require:

- a supported alternative estimator;
- a sensitivity analysis or identification bound;
- a qualified non-estimable conclusion; or
- abstention when the requested effect cannot be identified from the supplied
  evidence.

Unqualified use of the broken default is not credit eligible.

For the bounded-deviation routes in TE-S04 and TE-S06, the participant does
not choose a favourable sensitivity value. The task publishes the fixed grid
`delta = 0.05, 0.10, 0.20`. A conformant response declares that grid and
reports the lower and upper endpoint at every value as one six-component
sensitivity set. The grader checks the grid as part of the route identity and
then compares all six endpoints. A point inside a bound, one selected delta,
or a correct set attached to a different sensitivity model is not equivalent.
The worst-case bound and the typed non-identification response remain separate
eligible routes where the item admits them.

### Score-bearing and audit-only fields

The grader binds the declared population, treatment, comparator, endpoint,
intercurrent-event strategies, horizon, effect scale, and one exact
`analysis_method_id` to an eligible route. The method ID resolves to one
complete intrinsic record; participants cannot recombine its family,
uncertainty method, result kind, design modifiers, or sensitivity grid. The
grader then checks the typed result and the route-required diagnostic evidence.
For C5 it also checks the typed repair record.

Free-text estimator `implementation`, estimator `qualifications`, and top-level
`limitations` are retained for audit and interpretation but do not affect
method identity or score. Typed diagnostic evidence remains score-bearing when
the matched route requires it. Planning output is
reported as a separate prespecified consequence where eligible; it is not
folded into the primary route pass. No free-text similarity, keyword search,
or favourable search across unmatched routes is used.

For a qualified non-identification response, the participant declares the
corresponding method ID, result kind `limitation`, and one exact global code
supported by that method record. The current codes
distinguish censoring/support failure from endpoint-validation failure. The
grader first checks that this semantic route matches the item; possession of a
valid global code alone does not reveal or establish item eligibility.

### Numeric tolerances

Each numeric route declares its effect scale, reporting precision, independent
reconstruction tolerance, and comparison rule. Tolerances cover deterministic
floating-point and declared reporting-rounding differences; they are not an
empirical margin chosen after seeing model output. Verification reports both
absolute difference and difference-to-tolerance ratio.

## TrialDev checks the link from evidence to action

TrialDev retains seven shared score coordinates:

- asset nomination;
- phase design;
- phase analysis;
- decision action;
- route timing;
- final recommendation; and
- safety gate.

The bounded-portfolio stream adds two noncompensatory coordinates:

- portfolio allocation; and
- resource feasibility.

The single-asset and portfolio streams are reported separately. Their primary
results are not pooled into one TrialDev scalar.

Each lane is evaluated from evidence available at that checkpoint. Future
phase evidence cannot repair an earlier decision. The observational screen is
nonrandomized; each later randomized phase compares the nominated regimen with
control. Discontinuation is treated as tolerability or feasibility evidence
unless the public safety policy explicitly makes it a hard gate. Loss to
follow-up and censoring remain separate from discontinuation.

Where the public evidence supports several actions, the evaluation-target
register lists the credit-eligible set. The primary score is not proximity to a
single latent “best” action.

This set-valued rule applies where uncertainty leaves more than one defensible
development action, such as continuing or stopping after an exploratory study.
It does not turn an uncertain confirmatory result into optional success. At
confirmation, clear efficacy and safety support success, a clear failure in
either domain supports failure, and every remaining case supports an
inconclusive conclusion.

The checkpoint score is the minimum over the lanes required at that reached
checkpoint. The programme primary score is the minimum over observational
review, every reached randomized checkpoint, and the terminal decision. A
missing or invalid required checkpoint contributes zero. A checkpoint after a
supported stop is `structural_not_reached` and has no score; it is not converted
to model failure. A phase unavailable under the published programme design is
`not_scheduled`, while a phase blocked by invalid earlier work is
`not_reached_after_invalid`.

`chain_summary.json` retains all five checkpoint outcomes in order, with each
reached checkpoint's conditional and cumulative score. The phase grade reports
retain every lane record, and the trajectory grade retains stop/progress,
terminal recommendation, invalid-attempt, regret, and resource-consequence
records. These diagnostic coordinates do not compensate for an earlier
failure.

At observational review, a nomination is eligible when its upper efficacy
confidence bound reaches the declared minimum and its utility contrast from
the best such candidate is within the larger of the practical-equivalence
margin and the prespecified contrast confidence half-width. When the objective
is estimable but no candidate satisfies that rule, the submission must still
report complete candidate estimates and rankings before choosing not to
nominate. Qualified non-nomination is reserved for public evidence that the
requested comparison cannot be identified or estimated defensibly.

For bounded portfolios, one uncertainty estimate is reported for each unordered
candidate pair. The same half-width applies whichever member is considered for
the lead role; identifier order has no effect, and the signed utility difference
determines the direction.
The grader independently recomputes all candidate estimates, intervals, pairwise
contrast uncertainties, evidence checksums, randomized effects, safety results,
design feasibility, and resource feasibility. It then derives the complete
supported action set. Selecting any member receives the same primary action
credit. A point estimate outside its interval, an undeclared method, a stale
state, a missing evidence reference, an unaffordable design, a retired asset,
or a second switch fails the responsibility it violates. Contract-invalid or
non-executable records are returned for correction. A contract-valid legal
decision continues through the programme so that analytical quality and its
later practical consequences remain observable.

When the observational causal comparison is not identified, the accepted
record contains the public identification evidence but no candidate causal
estimates or ranking. This is a qualified scientific conclusion, not a proxy
for poor model performance. Conversely, broad uncertainty does not justify
withholding when the declared analysis remains identified: the supported set
may simply contain multiple actions.

Here, identified means point-identifiable under the declared assumptions and
the public treatment-assignment provenance. It does not imply empirical proof
of exchangeability. Numeric agreement uses the released method's complete
deterministic uncertainty contract, including bootstrap seed, random number
generator, standard-error degrees of freedom, and confidence-interval rule.

### Consequence and policy-value qualification

Primary TrialDev credit asks whether the submitted analysis is scientifically
defensible and whether the selected action belongs to the set supported by the
evidence and published policy. Exact reproduction is reported separately. The
grader does not choose an action using hidden simulation parameters.
Consequently, two equally credited actions can have different later simulated
outcomes when the available evidence cannot distinguish them. A legal but
unsupported action receives lower action credit and still exposes its resource
use and downstream outcome.

The separate policy-value study checks whether this uncertainty is controlled.
For qualification only, a mechanism oracle enumerates the finite legal
lead-reserve pairs using known generation parameters. Repeated worlds report
whether its action remains in the public supported set, the terminal-success
regret of the best and worst supported actions, simpler point-ranking
comparators, and expected resource use. Oracle-action coverage uses Wilson
intervals and regret uses world-bootstrap intervals. These diagnostics can
identify a weak decision rule or simulator, but they never alter a participant
grade or imply a universally optimal clinical policy.

## How failed and unreached tasks are represented

Every scheduled assignment remains in the denominator and has exactly one
completion state:

- `completed`;
- `failed`;
- `missing`; or
- `non_estimable`.

Non-completed states require a typed failure code. Examples include invalid
schema, resource limit, missing fixed evidence, unsupported analysis, and
non-finite result. Infrastructure failures are reported separately from model
failures but are never silently converted to successful or omitted rows.
For TrialDev, exhaustion of the correction budget after repeated invalid
phase requests is a model noncompletion distinct from turn-limit
exhaustion; both remain in the all-programme denominator with zero primary
credit.

## How task results are aggregated

The primary TrialEval experiment uses paired base-trial comparisons with
noncompletion counted as failure. TrialDev reports complete-programme,
cumulative-through-checkpoint, conditional-on-reaching-checkpoint, lane,
stop/progress, and consequence results, with uncertainty clustered on programme
trajectory. Model-panel summaries must:

- use the frozen assignment schedule;
- retain all scheduled rows;
- respect the declared independence and matching units;
- report component denominators and typed failures; and
- distinguish prespecified primary contrasts from descriptive subgroup
  summaries.

The response-interface contrast is defined prospectively as
`structured-minus-narrative`; reversing this sign changes the estimand.

Leaderboard-style scalar aggregation is optional presentation. The auditable
result is the component and lane record from which any aggregate is derived.
