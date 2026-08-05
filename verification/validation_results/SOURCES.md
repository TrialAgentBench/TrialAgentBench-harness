# Sources

Participant-level clinical data configure source-specific simulations and
define external reference distributions. Participant records remain with their
source repositories; the result set contains aggregate statistics, figures,
and uncertainty intervals.

## Data

| Source | Version | Persistent identifier | Terms | Analysis contribution |
|---|---:|---|---|---|
| TrialEval public release | `trialagentbench-paired-release-003` | [TrialAgentBench dataset](https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/) | CC BY-NC 4.0 | Complete 100-trial characterisation, realised design properties, 500-context census, and exposed worked analysis |
| PATENCY | 9 | [Dryad 10.5061/dryad.dz08kps7j](https://doi.org/10.5061/dryad.dz08kps7j) | CC0 1.0 | Three-year survival, clustered graft outcomes, and structure-breaking controls |
| HeadSOAR | 4 | [Dryad 10.5061/dryad.jwstqjqr9](https://doi.org/10.5061/dryad.jwstqjqr9) | CC0 1.0 | Seven-level ordinal outcomes, safety risks, and treatment-outcome controls |
| TERECO | 2 | [Dryad 10.5061/dryad.59zw3r27n](https://doi.org/10.5061/dryad.59zw3r27n) | CC0 1.0 | Six longitudinal outcomes, repeated-measure dependence, attendance, and treatment-by-time effects |
| Surgical-nurse skin-barrier trial | v1 | [Zenodo 10.5281/zenodo.17562062](https://doi.org/10.5281/zenodo.17562062) | CC BY 4.0 | Four-occasion longitudinal reference and observation-process calibration |
| PENG-block concentration trial | v1 | [Zenodo 10.5281/zenodo.18849529](https://doi.org/10.5281/zenodo.18849529) | CC BY 4.0 | Seven-occasion longitudinal reference and observation-process calibration |
| RCTBench | `ef11941dc4749c135c71a1171c7ace70d7b7ca1c` | [Source repository](https://github.com/bingkaiwang/rct_bench-main) | Repository terms | Ten participant-level randomized trials for joint-distribution, effect-recovery, heterogeneity, competing-event, and confounding analyses |
| ImmPort shared studies | DR58 and DR60 packages | [ImmPort](https://www.immport.org/) | ImmPort data-use terms | Fifteen-study recurrent-event reference and eight-study cross-domain linkage analysis |

The RCTBench portfolio comprises trials 024, 044, 053, 055, 064, 075, 078,
082, 100, and 119. Together they contribute 10 independent trial units and
1,044 source-sized participants to each simulated portfolio.

The TrialEval characterisation uses the participant archive, public
simulation-properties catalogue, and verification archive from one paired
release. Their SHA-256 digests are recorded in
[release_characterisation.json](data/release_characterisation.json). The
characterisation contains synthetic participant rows and public analysis
records only.

The two Zenodo trials contribute distinct longitudinal settings: 35
participants measured at four occasions and 102 participants measured at
seven occasions. Their role is to test whether an observation mechanism
estimated in one clinical setting remains recoverable across different
measurement schedules and outcome scales.

The recurrent-event reference contains 2,628 participants and 15,818
adverse-event rows from 15 eligible ImmPort studies. The cross-domain analysis
uses eight studies with linked assessment, biosample, intervention, and
adverse-event records. ImmPort participant data are analysed under their
source terms.

The screened ImmPort source set comprises `SDY1`, `SDY91`, `SDY471`, `SDY473`,
`SDY545`, `SDY670`, `SDY689`, `SDY823`, `SDY824`, `SDY857`, `SDY1025`,
`SDY1028`, `SDY1039`, `SDY1437`, `SDY1515`, `SDY1644`, and `SDY1671` from
DR58, plus `SDY1520` from DR60. Each analysis retains the subset with the
participant linkage, follow-up, and outcome fields required for its estimand.
Source acquisition receipts record archive checksums; analysis records retain
the source identities included in each result.

## Study roles

Each dataset addresses the question supported by its design:

- PATENCY, HeadSOAR, and TERECO test whether complete clinical outcomes are
  reproduced at the original sample size and on their native scales.
- RCTBench tests whether marginal distributions, participant relationships,
  and analysis results remain coherent across a portfolio of independent
  trials.
- The Zenodo studies test longitudinal observation processes across different
  schedules and outcome types.
- ImmPort tests recurrent-event heterogeneity and participant linkage across
  independently collected clinical studies and data domains.

This division keeps each comparison tied to an observed data structure while
allowing the same generating mechanisms to be tested across heterogeneous
sources.

## Statistical methods

- Kaplan EL, Meier P. Nonparametric estimation from incomplete observations.
  *Journal of the American Statistical Association*. 1958.
  [doi:10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452).
- Cox DR. Regression models and life-tables. *Journal of the Royal Statistical
  Society: Series B*. 1972.
  [doi:10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x).
- McCullagh P. Regression models for ordinal data. *Journal of the Royal
  Statistical Society: Series B*. 1980.
  [doi:10.1111/j.2517-6161.1980.tb01109.x](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x).
- Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
  models. *Biometrika*. 1986.
  [doi:10.1093/biomet/73.1.13](https://doi.org/10.1093/biomet/73.1.13).
- Li F, Turner EL, Heagerty PJ, Murray DM, Vollmer WM, DeLong ER.
  An evaluation of constrained randomization for the design and analysis of
  group-randomized trials. *Statistics in Medicine*. 2017.
  [doi:10.1002/sim.7410](https://doi.org/10.1002/sim.7410).
- Martin J, Girling A, Nirantharakumar K, Ryan R, Marshall T, Hemming K.
  Intra-cluster and inter-period correlation coefficients for cross-sectional
  cluster randomised controlled trials for type-2 diabetes in UK primary care.
  *Trials*. 2016.
  [doi:10.1186/s13063-016-1532-9](https://doi.org/10.1186/s13063-016-1532-9).
- Adams G, Gulliford MC, Ukoumunne OC, Eldridge S, Chinn S, Campbell MJ.
  Patterns of intra-cluster correlation from primary care research to inform
  study design and analysis. *Journal of Clinical Epidemiology*. 2004.
  [doi:10.1016/j.jclinepi.2003.12.013](https://doi.org/10.1016/j.jclinepi.2003.12.013).
- Martin J, Girling A, Nirantharakumar K, Ryan R, Marshall T, Hemming K.
  Clustering of continuous and binary outcomes at the general practice level
  in individually randomised studies in primary care. *BMC Medical Research
  Methodology*. 2020.
  [doi:10.1186/s12874-020-00971-7](https://doi.org/10.1186/s12874-020-00971-7).
- Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster
  randomized trials. *Contemporary Clinical Trials*. 2007.
  [doi:10.1016/j.cct.2006.05.007](https://doi.org/10.1016/j.cct.2006.05.007).
- O'Brien PC, Fleming TR. A multiple testing procedure for clinical trials.
  *Biometrics*. 1979.
  [doi:10.2307/2530245](https://doi.org/10.2307/2530245).
- DeMets DL, Lan KKG. Interim analysis: the alpha spending function approach.
  *Statistics in Medicine*. 1994.
  [doi:10.1002/sim.4780131308](https://doi.org/10.1002/sim.4780131308).
- Wilson EB. Probable inference, the law of succession, and statistical
  inference. *Journal of the American Statistical Association*. 1927.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).

The exact estimands, controls, resampling units, and interval definitions are
specified in [METHODS.md](METHODS.md). Numerical results and denominators are
reported in [RESULTS.csv](RESULTS.csv).
