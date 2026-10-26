# Opt-in provider outage handling

This prospective condition must be explicitly recorded before the main runs.
Existing runs keep their original behavior. The default is
`retry.unavailability_policy = "charge_all_v1"`; its serialized configuration
and existing Dev checkpoint identity are unchanged.

With `"defer_unanswered_v1"`, the first qualifying transport failure is recorded
and propagated as `ProviderUnavailableError`. Qualifying failures are a
connection failure, HTTP 429/500/502/503/504, or a request timeout when the task
had enough remaining time for the configured full request timeout. No received
model response, reasoning payload, or earlier received response in the same
retry sequence may be reclassified as an unanswered outage. Authentication,
invalid-request and other permanent client errors do not qualify.

A timeout shortened by the task's remaining scientific allowance still ends
as ordinary task noncompletion. A usable response's waiting time, analysis
code execution and other normally charged work still consume the allowance.
Errors in model code, invalid scientific answers, refusal, malformed or empty
returned model outcomes, exhausted turns and genuine task timeouts are not
eligible for this outage deferral.

## Clock and evidence

The adapter measures the unanswered transport interval. The shared provider
boundary excludes exactly that interval from the active scientific clock and
writes it as `scientific_clock_excluded_seconds` on the failed request event,
with `unavailability_policy`, total observed `elapsed_seconds`, retry advice,
request identity and failure evidence. Successful provider preparation before
that interval remains charged. Impossible or inconsistent exclusion evidence
fails closed, preserves the request log and requires operator review.

This is not a fresh task allowance. A Dev checkpoint preserves all prior
successful turns, conversation, pending tool state and remaining charged time.
Recovery continues at the same next turn after downtime. A changed clock policy
is rejected on resume. Existing workspace recovery is reused; live interpreter
memory is not promised across process restarts.

Eval has no mid-task conversation checkpoint. An outage retains an
infrastructure-failed original primary and its evidence as `provider_unavailable`,
distinct from legacy `provider_stop` outcomes. The item carries a hash-bound
copy of the failed request event; both native grading and the independent
verifier require an unanswered typed outage, the same task/model and the frozen
`defer_unanswered_v1` policy. Missing or inconsistent evidence refuses recovery.
Its recovery uses a separately linked, once-only infrastructure replacement. The original
and replacement remain auditable; do not count both as independent primary
scientific observations. Retrying that entire Eval task is distinct from Dev
continuation and must be reported as such.

## Spending and bounded recovery

Clock exclusion does not make an API call free. All known charges and uncertain
spending reservations remain in the same campaign ledger. An unanswered
request may still have been billed; this change never releases its reservation
or bypasses a spending hold. Other available routes may continue only within
that same global cap.

The frozen dispatch configuration controls cooldown and bounded recovery;
see [STRATIFIED_DISPATCH.md](STRATIFIED_DISPATCH.md). A provider outage pauses its
route for the current command, retaining other models' completed work. A later
resume respects cooldowns and persistent deferral limits. Do not change model
identity, route, settings, spending limits or remove audit events to force work
through. Exhausted recovery limits require operator review.

## Offline validation

`tests/adapters/test_provider_unavailability.py` injects full and shortened
request timeouts, connection/HTTP errors, received and ambiguous payloads,
prior unusable responses, policy mismatches and failed spending reservations.
It checks complete event logs, exact clock exclusion, successful-time charges,
and retention of uncertain liability. Dev timing tests exercise native
checkpoint/resume across successful work, outages and long downtime without
resetting turns or previous time consumption. No real provider requests are
required by these tests.
