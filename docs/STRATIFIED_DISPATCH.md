# Balanced scheduling within one campaign

An optional `dispatch` configuration freezes an exact priority order for the
entire scientific panel. Prepare the full panel once. Execution slices resume
those same native invocations, use the same `requests.sqlite` spending ledger,
and do not create subcampaigns or extra primary attempts.

```json
"dispatch": {
  "task_order": [
    {"benchmark": "trialeval", "task_id": "released-eval-task-id"},
    {"benchmark": "trialdev", "task_id": "released-dev-task-id"}
  ],
  "tasks_per_slice": 1,
  "defer_seconds": 300,
  "maximum_provider_deferrals_per_task": 3,
  "maximum_consecutive_provider_deferrals": 3
}
```

The order must contain every explicitly selected task exactly once, using exact
task IDs. Choose the order from release metadata before observing outcomes.
`tasks_per_slice` cannot exceed `operations.task_workers`. A slice contains at
most that many pending tasks from one model/benchmark run. At most
`operations.model_workers` slices run simultaneously; a native run has one writer.
The model priority rotates at each task position. Work on available routes can
advance past unavailable routes; wave labels are not global barriers.

`campaign run` authenticates the original full invocation on every slice. Native
coverage retains the full denominator. Intermediate slices save terminal task
checkpoints and coverage but do not publish misleading complete-run summaries.
When every native task is terminal, normal native summary/index generation and
`campaign finish` operate on the full original selection. Resuming a completed
campaign does not repeat completed primaries.

For models explicitly configured with
`retry.unavailability_policy = "defer_unanswered_v1"`, confirmed unanswered
provider failures leave Dev work pending at its existing checkpoint. Eval has no
mid-task conversation checkpoint: its failed primary is retained, and recovery
uses the existing separately linked once-only infrastructure retry. No other
model's answer replaces the deferred task.

Each outage pauses its provider route for the remainder of that run command.
`dispatch-events.jsonl` records started/completed slices and provider deferrals,
including cooldown and safe error classification. A later `campaign run` may
resume after the recorded cooldown. Both the per-task and consecutive route deferral limits persist across
restarts; reaching either holds the route for operator review,
so a persistent outage is not retried on every remaining task. This log is an
operational audit, not a second result ledger. Do not delete or edit it to bypass
recovery limits. Confirmed failed transport time and uncertain spending remain
subject to the separately recorded provider clock/accounting policy.

A fatal engineering fault or campaign-wide commercial stop drains current work
and cancels further provider requests. Completed native checkpoints survive.
Route-local capacity stops leave the other routes free to proceed. A pending
panel cannot be passed off as complete by running `finish`.

## Source and scorer custody

Prepare only after the source, verifier/scorer, executor and scientific condition
are reviewed and pinned. Dispatch configuration is part of the campaign identity;
changing it requires a new prepared campaign. Native resume rejects source or
invocation drift. Do not refresh a source hash, executor attestation or scorer
identity inside a live campaign to force a resume after editing code. Keep prior
pilot outputs on their original source/scorer identities and label them separately.
