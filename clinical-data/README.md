# Clinical data tables

Shared deterministic SDTM/ADaM projections for Ariadne and TrialAgentBench.
These functions consume observed tables; they generate neither patients nor outcomes.
The public scientific review independently checks the exported values and linkage.

`clinical_timing` converts elapsed observations and optional participant reference
instants into calendar dates and study days. The exposure, findings and analysis
projectors accept the same row-aligned `reference_days` input for staggered entry.
An unknown calendar origin remains unknown; it is not replaced with an invented date.
