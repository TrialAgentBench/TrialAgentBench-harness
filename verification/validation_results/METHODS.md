# Methods

Clinical-trial simulation requires separate evidence that outcomes are
replicated, participant relationships are preserved, and controlled mechanisms
remain recoverable. The [statistical report](REPORT.md) gives the corresponding
results; the methods below define each comparison and its uncertainty.

## Framework

The validation separates three questions that are often conflated.

### Outcome fidelity

Outcome replication compares observed and generated trials at the source
study's sample size. The comparisons use the clinical scale required by the
analysis:

- complete event-free curves, risk sets, and follow-up for survival outcomes;
- all ordered categories and cumulative probabilities for ordinal outcomes;
- arm-specific trajectories, attendance, and covariance for longitudinal
  outcomes; and
- complete cause-specific event processes for competing risks.

This step estimates predictive variation among source-fitted simulated trials.
It does not estimate sampling uncertainty in the observed source trial.

### Linkage preservation

Marginal distributions can remain correct after records have been attached to
the wrong participant, visit, treatment arm, or event time. The linkage
experiments therefore preserve the exact values in each affected column while
progressively permuting their participant keys.

Any resulting change in correlation or treatment analysis is attributable to
the lost linkage because the marginal values have not changed. Linked-subject
resampling is compared with independent-column sampling for the same reason:
both preserve the empirical values, while only linked-subject resampling
retains their observed joint structure. The comparison isolates the
measurement and analytical consequence of linkage; release trials are
evaluated separately against external trial distributions.

Marginal discrepancy is the Wasserstein distance divided by the source
standard deviation. Dependence discrepancy is the absolute difference between
source and simulated Spearman correlations for outcome, age, and body mass
index; medians are taken over variable pairs and simulation worlds within each
trial. Analysis discrepancy is the absolute bias in the adjusted treatment
coefficient divided by its source-trial standard error. Portfolio summaries
then take the median across 10 equally weighted trials.

The source-scale controls pair each simulated trial with one deterministic
permutation. In PATENCY, event indicators are permuted within randomized arm
while follow-up times are fixed; error is the mean absolute difference from
the fitted arm-specific survival curve over the declared time grid. In
HeadSOAR, treatment labels are permuted among complete high-dose outcomes
while arm sizes and category values are fixed; error is the mean absolute
difference from the fitted arm-specific category probabilities. Student *t*
intervals summarize the intact error, disrupted error, and within-trial
difference over 1,000 paired trials.

### Mechanism recovery

Mechanism experiments vary one generating parameter over a declared range that
includes a null or low setting. A separately fitted analysis estimates the
parameter in every simulated trial. The unit-scale recovery slope is:

\[
\text{unit-scale recovery slope}
=
\frac{\text{change in recovered estimate}}
     {\text{change in generating value}}.
\]

A slope near one means that the analysis recovers the configured change at the
correct scale. A slope near zero indicates that the mechanism is not reaching
the analysis. Settings with more information distinguish incorrect generation
from imprecision at the source trial's sample size.

Some source-specific experiments configure effects through a multiplier of the
source coefficient. Their **coefficient change per multiplier** retains the
coefficient's units and is compared with the source coefficient, rather than
with one. For example, a recovered change of -0.115 log hazard-ratio units per
multiplier is compared with the configured change of -0.114. The report names
these two summaries separately.

The PATENCY and HeadSOAR experiments use common random numbers across effect
settings within each simulated world. Allocation, baseline variation,
censoring or missingness, and latent outcome draws are therefore paired while
only the treatment-effect coefficient changes. Point recovery and interval
coverage are evaluated separately at each setting; the response slope and
within-world ordering use the paired contrast.

## Design

| Scientific question | Primary comparison | Unit used for uncertainty |
|---|---|---|
| Are clinical outcomes reproduced? | Observed source trial versus repeated source-fitted trials | Simulation trial for predictive variation |
| Does participant linkage affect dependence and analysis? | Linked-subject resampling versus independent columns | Source trial across the 10-trial portfolio |
| Does linkage affect analysis? | Intact records versus graded exact-marginal permutation | Source trial or study |
| Are known effects recoverable? | Separately fitted estimate versus generating value across several settings | Simulation trial within source; source trial across portfolios |
| Does added information improve precision? | Source-size versus fourfold-size trials generated from the same source | Paired source trial |
| Are reported results implementation-specific? | Statsmodels estimate versus an independently initialized SciPy calculation | Source trial |

## TrialEval release characterisation

The release catalogue contains five evidence contexts for each matched trial.
Only C1 is an independently generated trial; C2-C5 retain the same matched-set
identity and change the information supplied for analysis. The TrialEval
base-trial census therefore characterises each of the 100 C1 trials once and reports 500 context
views separately.

Participant records are linked one-to-one across the subject, primary
time-to-event, and operational-flag tables. Continuous variables are described
by the arithmetic mean and sample standard deviation with Student *t* and
chi-square intervals. Categorical variables use category proportions with
Wilson intervals. Age-BMI dependence is the complete-case Spearman correlation
with a 2,000-replicate participant bootstrap interval for general use and 200
replicates for the full TrialEval base-trial census. Event-free probability is estimated by
Kaplan-Meier at 25%, 50%, 75%, and 100% of each trial's declared follow-up
horizon with Greenwood log-log intervals.

The worked trial is the independent trial `TE-S03-A1_03`. Its
participant distributions use the synthetic participant as the independent
unit. The treated-minus-control 365-day event-risk difference is reproduced
from the public route reference using the declared Kaplan-Meier estimator and
delta-method Greenwood uncertainty. The base-trial figure then gives every
independent trial equal visual weight and marks the median within each design
profile; its ranges are a complete census, not sampling intervals.

## Design properties

Design properties are estimated once per independent C1 trial from the public
participant, endpoint, operational, and protocol records. For individually
randomized trials, the largest treated-control standardized difference across
age, body mass index, hypertension, diabetes, and chronic kidney disease is
compared with 499 arm-count-preserving random reassignments. These are the five
baseline variables used by the prespecified covariate-standardized analysis.
The observed value, randomization 95th percentile, and randomization percentile
are retained for each trial.

Pragmatic conduct is described by participant-level exposure received,
presence of any missed exposure, intercurrent-event occurrence, treatment
discontinuation, rescue therapy, treatment switching, and per-protocol
eligibility, both overall and by randomized arm. Covariate structure uses
participant-level Spearman correlations. A prespecified Cox model containing
age, body mass index, hypertension, diabetes, and chronic kidney disease is
standardized to the empirical population with baseline body mass index at
least 35 kg/m2 and compared with the unadjusted Kaplan-Meier risk difference
at the same horizon.

In endpoint-ascertainment trials, validation fraction, sensitivity, and
specificity are calculated against the adjudicated endpoint among participants
selected for validation. Observed and adjudicated event proportions are
reported separately. The corrected primary risk contrast is compared with a
Kaplan-Meier analysis of the routinely observed endpoint.

Cluster size is summarized across released site identifiers. Event
intraclass correlation uses a one-way random-effects variance decomposition
after subtracting the randomized-arm event mean. Unequal cluster sizes enter
through the effective cluster size

\[
\tilde m =
\frac{N-\sum_j m_j^2/N}{J-1}.
\]

The estimated event design effect is
\(1+(\tilde m-1)\widehat{\rho}\). The method-of-moments correlation remains
untruncated so the estimate is centred on zero under independence.
Stepped-wedge adoption days are recovered for each cluster and sequence.
Interim information fractions, stopping status, nominal alpha, and critical
values are read from the public monitoring plan.

Cluster response is evaluated in 1,000 matched parallel-trial worlds at five
cluster intensities spanning a 90th-to-10th percentile hazard ratio of 1.00
to 7.50. Each world contains 80 clusters of 85 participants, balanced
cluster-level treatment assignment, 280 days of follow-up, and a 3.5% event
risk in the absence of cluster heterogeneity. The empirical-reference log-hazard
standard deviation inverts the rare-event lognormal relation
\(\rho=p\{\exp(\sigma^2)-1\}/(1-p)\) at \(p=0.035\) and
\(\rho=0.011\), the adjusted median event ICC in the linked empirical study.
This gives a 90th-to-10th percentile cluster hazard ratio of 3.74. Treatment
multiplies the event hazard by 0.80. The same cluster normal variates,
randomization, and participant event quantiles are reused across intensity
levels within each world. Event intraclass correlation and design effect use
the same estimators as the TrialEval base-trial census. The known conditional risk
difference averages the treated and control counterfactual risks over all
generated clusters. Ordinary least squares estimates the risk difference.
Cluster-aware intervals use the small-sample-corrected cluster sandwich with
79 degrees of freedom; participant-independent intervals use the ordinary
model standard error. Coverage intervals are Wilson intervals and continuous
summaries use Student *t* intervals across trials. The complete world-level
results are retained.

Stepped-wedge response is evaluated in 1,000 matched worlds at five
period-4-to-period-1 secular hazard ratios: 1.00, 1.20, 1.65, 2.00, and 3.00.
Each world contains 80 clusters, four randomized adoption sequences, four
calendar periods, and 30 participants per cluster-period. Treatment has a
configured hazard ratio of \(\exp(-0.35)\), and the cluster 90th-to-10th
percentile hazard ratio is 3.74, matching the empirical cluster reference.
Cluster effects, adoption sequences, and participant event quantiles are
shared across trend levels within each world. Complementary-log-log binomial
models include or omit calendar period. Each model is standardized over the
four calendar periods to estimate the same participant-average
treated-minus-control event-risk difference. The world-specific reference
averages the corresponding counterfactual risks over all generated
cluster-periods. Both analyses use small-sample-corrected cluster-sandwich
uncertainty with 79 degrees of freedom and delta-method intervals for the risk
difference. Coverage intervals are Wilson intervals and continuous summaries
use Student *t* intervals across trials.

The finite-release design comparisons target the public treated-minus-control
event-risk difference in the same trial. Cluster-parallel analyses compare the
released cluster-aware interval with a participant-independent Kaplan-Meier
Greenwood interval. In stepped-wedge trials, calendar-period-adjusted and
period-omitting Poisson models quantify the finite-trial treatment contrast and
calendar pattern. No-trend A1 trials provide a specificity control; A2 trials
contain the secular-trend stress condition. Calendar-period baseline rates are
estimated from the person-period model after adjustment for current treatment
exposure. The repeated-trial experiment above then compares the two analysis
strategies on an exactly shared, standardized event-risk-difference estimand.

For group-sequential trials, Kaplan-Meier risk differences and standard errors
are recomputed from public endpoint rows. The absolute test statistic is
compared with the prespecified boundary at the actual analysis look, and the
resulting stop decision is compared with the released execution path.
Fixed-analysis intervals use the independently recomputed standard error and a
1.96 critical value. The randomized cluster is the independent unit for
cluster and stepped-wedge analyses; the trial at its released information
fraction is the unit for interim monitoring.

The monitoring operating-characteristic experiment uses only the public
information fractions and two-sided efficacy boundaries. For each of 500,000
independent standard Brownian paths, the test statistic at information
fraction \(t\) is \(W(t)/\sqrt{t}+\delta\sqrt{t}\). The standardized signal
sets the mean final statistic to 0, 0.5, 1, 1.5, or 2 times the final efficacy
boundary. The first absolute boundary crossing determines the stopping look.
At that look, the standardized treatment estimate is the observed statistic
divided by the square root of accrued information. Two intervals are evaluated:
the repeated confidence interval obtained by inverting the prespecified
look-specific boundary and the ordinary 1.96 interval that ignores repeated
testing. Coverage, rejection, and early-stopping probabilities use Wilson
intervals; mean bias uses a Student *t* interval. Expected information is the
mean realized information fraction. The same Brownian paths are reused across
signal levels, isolating the response to signal strength while retaining
500,000 independent trial paths at each level.

## Assumption response

The response experiment contains 32 independently generated trial replicates
for each of eight Assumption-axis series. Within a replicate, A1-A3 or A1-A2
tiers share the random-number stream, participant count, visit schedule,
source profile, endpoint, estimand, effect scale, and common analysis route.
The tier changes the specified mechanism. A public-data bridge records the
series, matched replicate, estimand, effect scale, observed mechanism, routine
analysis, prespecified same-estimand alternative where applicable, intervals,
diagnostic status, and numerical replay error for all 704 tier-specific
analyses from 256 independently generated matched trial replicates.

Mechanism intensity is measured on the scale supplied by each diagnostic:
log-hazard-ratio variation for proportional-hazards and censoring processes,
treated-arm recorded-dose nonadherence and endpoint disagreement as proportions, and
the treatment-adjusted time coefficient for secular trend. The analysis
response is a signed, prespecified same-estimand contrast in the direction of
the expected correction: direct minus Cox-projected risk for time-varying
effects; ordinary minus weighted restricted mean survival for prognostic
censoring; weighted minus ordinary risk for baseline-dependent censoring;
spline minus linear standardized risk for nonlinear prognosis; routine minus
validation-corrected risk for endpoint error; weighted minus unweighted risk
for clustered censoring; and period-omitting minus period-adjusted risk for
secular trend. This preserves the direction and native unit of each
consequence instead of converting sampling variation into a positive
absolute-distance floor. The nonadherence series keeps the randomized
treatment-policy analysis fixed and measures attenuation of the absolute risk
difference relative to its paired higher-adherence trial.

Tier means and 95% Student *t* intervals use the generated-trial replicate as
the independent unit. Adjacent-tier changes use paired Student *t* intervals
across the 32 shared random streams. The display retains mechanism intensity
and the analysis consequence in their native units. Comparisons within a
replicate are paired; uncertainty is therefore estimated across matched trial
replicates rather than by treating the two analyses as independent.

The TrialEval base-trial census is analyzed separately. It contains four independently
generated trials in each of 25 Assumption-axis cells and includes three A4
cells. A4 denotes a condition in which the prespecified conventional analysis
is not defensible; it is not an additional numeric dose. Identification is
assessed separately and determines the form of the replacement analysis. For
dependent censoring and incomplete endpoint-validation support, risk-difference
ranges are calculated at event-risk departures of 5, 10, and 20 percentage points,
followed by the unrestricted worst-case range. The range midpoint is used only
to position each line and is not interpreted as a point estimate. For
sequential monitoring, the replacement is the treatment-risk estimate and
repeated 95% confidence interval at the realized information fraction under
the declared group-sequential plan. An ordinary fixed-look interval is not a
valid replacement at a data-dependent analysis time.

The bounded-departure parameter is an absolute probability difference. For
dependent censoring, it limits how far the event probability by the fixed
horizon among participants censored before that horizon may differ from the
observed-status risk in the same randomized arm. For incomplete endpoint
validation, it limits how far the adjudicated event probability in the
unsupported prognostic stratum may differ from its routinely recorded event
probability. The 0.05, 0.10, and 0.20 values form a doubling sequence on the
0-1 risk scale; the unrestricted analysis allows any value from zero to one.
The full response curve is reported, so the conclusion is not determined by
one selected departure value.

## Context and standards

The context census groups the five public items that share one prospective
base-trial identity. Equality checks compare the generation seed and the full
estimand contract within each group. Analysis-ready C1-C2 tables are compared
directly. C3-C4 analysis tables are reconstructed from declared participant,
randomization, visit, exposure, disposition, endpoint, adjudication, and
baseline domains. Dates are converted to analysis time from each participant's
recorded origin; endpoint-window midpoints define event time; last contact and
the protocol horizon define censoring. Randomized-arm aliases come only from
the public protocol and task.

The matched-panel comparison requires one generation seed and one estimand.
C1 and C2 must have identical analysis-ready data; C3 and C4 must have
identical raw domains. C5 first identifies and removes the declared exact
duplicate, then applies the C4 reconstruction. Its repaired analysis content
is compared with the matched C4 content.

Every eligible public route is recomputed from its context-specific inputs.
C1 and C3 each prescribe one route per trial. C2, C4, and C5 expose the
complete supported route set for the same estimand. Numerical points and
identified intervals use their declared comparison rules; failed,
non-estimable, and successful routes retain separate dispositions.

The bounded standards analysis uses the official
`cdisc-org/sdtm-adam-pilot-project` at commit
`667511d4b183871d74392ba691c935c38d431d39`. XPT and Dataset-JSON values are
compared cell by cell for five SDTM and three ADaM datasets. The analysis also
checks declared keys, subject references, Define-XML dataset entries, and
equality of an adverse-event discontinuation risk difference derived
independently from SDTM and ADaM. Five in-memory mutations test duplicate
keys, orphan subjects, changed transport values, missing metadata, and changed
analysis derivation.

## External trial distributions

External and generated data are reduced to trial-level fingerprints before
comparison. The baseline frame contains 29 external trials and 4,776
participants. Ten trials and 1,044 participants have compatible outcomes for
the analysis-impact frame. The generated frame contains 75 trials and 442,524
participants.

The eight prespecified constructs are mean and standard deviation of age and
body mass index, age-BMI Spearman correlation, maximum scaled randomized-arm
imbalance, adjustment-induced estimate shift in unadjusted standard-error
units, and the adjusted-to-unadjusted standard-error ratio. Standardized
Wasserstein distance compares the external and generated trial-fingerprint
distributions.

The scale reference is generated solely from the external portfolio.
External trials are repeatedly divided into independent groups, and the 95th
percentile of their between-group distances is calculated for each construct.
The reported relative distance divides the generated-external distance by that
construct-specific reference. Confidence intervals use 2,000 trial-level
bootstrap replicates with seed 451014; participants do not enter the
cross-trial uncertainty calculation as independent observations.

## Outcomes

### Survival

Arm-specific Kaplan-Meier estimates are evaluated at prespecified time points
through the source trial's follow-up. The analysis reports:

- mean absolute curve error;
- numbers at risk;
- restricted mean survival time;
- event incidence and censoring;
- Cox log hazard-ratio recovery; and
- a control that breaks linkage between event occurrence and event time.

The displayed 95% envelope is the central range across independently simulated
trials. It is a predictive interval for a new simulated trial, not a confidence
interval for the source curve. The source-scale analysis uses 1,000 trials of
2,638 participants at each of four paired treatment-effect settings.

### Ordinal outcomes

Ordinal comparisons retain every category. Mean absolute errors are calculated
for category probabilities and cumulative probabilities by randomized arm.
The displayed intervals are the central 95% range of the source-size simulated
trials for each category and cumulative cutpoint. Treatment effects are
estimated with proportional-odds regression. The source-scale analysis uses
1,000 trials of 1,368 participants at each of four paired treatment-effect
settings. Cumulative predictive intervals are calculated within each simulated
trial before taking quantiles across trials.

The clustered analysis additionally preserves the number of observations per
participant and estimates within-participant ordinal dependence. Its treatment
analysis uses exchangeable generalized estimating equations with robust
standard errors. Permuting outcomes across participant identifiers tests
whether the dependence measure detects loss of clustering.

### Longitudinal outcomes

Longitudinal comparisons include arm-by-visit means, standard deviations,
attendance, and within-participant correlations. Treatment recovery estimates
the randomized arm-by-visit contrast with a saturated visit model and
participant-clustered uncertainty. The TERECO analysis uses 200 independently
simulated trials of 119 participants and jointly evaluates six outcomes, 36
arm-visit cells, and 135 within- and between-outcome correlations. Treatment
bias is divided by the pooled source standard deviation of each outcome;
simultaneous 95% intervals use the maximum standardized error across the six
outcomes.

The linkage experiment preserves 0%, 25%, 50%, 75%, or 100% of
within-participant associations while leaving each arm-visit marginal
distribution unchanged. Correlation mean absolute error is calculated against
the complete source correlation vector. A participant bootstrap of the source
trial supplies the sampling range expected when the observed trial is compared
with another sample from the same population.

## Mechanisms

### Dropout

Dropout probability depends on the previous observed outcome through a known
coefficient. The verifier estimates this coefficient directly from visit
records. It then compares:

- available-case analysis, which ignores the attendance mechanism;
- inverse-probability weighting using an estimated attendance model; and
- weighting using the known generating probabilities.

The last route checks implementation under a known mechanism; it is not
presented as an analysis available in an ordinary blinded trial. Results
include treatment-trajectory bias, coverage, weight concentration, effective
sample size, response slopes, and failed fits.

### Heterogeneity

Treatment effect varies with standardized age. Continuous outcomes use a
linear treatment-by-age interaction. Binary outcomes use the exact
finite-population risk-difference interaction induced by the logistic
generating model and the generated baseline covariates.

HC3 standard errors are used for the primary analysis. Classical least-squares
standard errors and the empirical standard deviation across trials provide
separate checks of the expected precision gain at fourfold information.

### Competing risks

Primary and competing events are generated in four discrete intervals. The
verifier reconstructs each risk set and fits both cause-specific coefficients
in one multinomial likelihood. It also calculates cumulative incidence for the
primary event and for either event.

The design varies the primary and competing coefficients separately. This
tests both coefficient recovery and the probability consequence of competition:
increasing competing-event risk should reduce the chance of observing the
primary event first while increasing the probability of any event.

### Confounding

Exposure is assigned through a clipped logistic function of generated age and
body mass index. Assignment strength varies in both directions. The endpoint
depends on exposure and the same baseline score.

The verifier compares the unadjusted exposure coefficient with a correctly
adjusted coefficient, then compares inverse-probability weighted risk
differences using the known and estimated assignment models with the exact
finite-population effect. Standardized
imbalance, extreme propensity mass, and effective sample size quantify the
progressive loss of overlap.

### Recurrent events

Participant event rates follow a Poisson-gamma model. A mean-one gamma
multiplier creates persistent heterogeneity between participants; its variance
is the configured parameter.

The verifier fits an NB2 mean model to participant event counts, with log
follow-up as an offset and treatment and source visit frequency as covariates.
It profiles the dispersion coefficient from the Poisson boundary through the
positive parameter space and obtains the 95% interval by likelihood-ratio
inversion. A second calculation applies the Poisson-gamma variance identity
directly to the released participant rates and follow-up. Agreement in the
direct calculation checks the generating law; agreement in the fitted
coefficient checks whether the mechanism is recoverable from finite trial data.

### Cross-domain linkage

Assessments, biosamples, adverse events, and interventions are progressively
permuted among participants while every domain retains its exact multiset of
values. The analysis measures the change in log-count correlations and in a
prespecified adverse-event-burden regression. Studies, rather than shuffled
datasets, are the units used for portfolio uncertainty.

### Response display

The mechanism-response display reports each analysis on its native scale.
Treatment-heterogeneity, competing-event, confounding, and cross-domain
intervals treat source trials or studies as independent units. Dropout
intervals use matched simulated trials within source. Recurrent-event points
compare the configured frailty variance with the direct Poisson-gamma
generating-law estimate. Dashed references denote zero response, exact
recovery, or identity as appropriate to the panel.

## Uncertainty

Monte Carlo intervals for a mean estimate use variation across the scheduled
simulation trials. Portfolio intervals give each source trial or study equal
weight.

Coverage is the proportion of model-based 95% intervals that contain the known
generating value. Wilson intervals describe uncertainty in coverage, rejection,
and other proportions. Bias, root mean squared error, empirical estimator
standard deviation, and mean reported standard error jointly characterize
point recovery and uncertainty calibration.

Every scheduled trial remains in the denominator for scheduled-world
summaries. Failed fits remain non-estimable. The report gives failure counts
and their concentration by information setting.

## Separate estimation

Validation estimates use Statsmodels, Lifelines, and SciPy without importing the
benchmark grader. Selected point estimates are repeated with a second library or
a separately initialized optimization:

- regression coefficients are compared between Statsmodels and SciPy;
- competing-risk likelihoods are optimized independently from a neutral
  starting value; and
- generating-law moments are calculated directly where a closed-form identity
  is available.

Numerical agreement across implementations reduces the chance that recovery
reflects shared code rather than the data-generating property under study.

## Data and code

The result set contains:

- [REPORT.md](REPORT.md), the evidence index and headline results;
- [reports/](reports/), the detailed statistical chapters;
- [RESULTS.csv](RESULTS.csv), the headline numerical summary;
- [data/operating_characteristics](data/operating_characteristics), complete
  dose, recovery, uncertainty, failure, and information-response tables;
- [SOURCES.md](SOURCES.md), dataset identities and method references;
- one SVG, one PDF, and the exact CSV data for each reported figure.

`validation_bundle.json` records the checksum and media type of each artifact.
