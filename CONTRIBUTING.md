# Contributing

Please open focused issues and pull requests that identify the affected
benchmark release, command, and evidence boundary.

For code changes:

1. add or update a deterministic test;
2. run `make check` and `make test`;
3. run `make build` and `make smoke` when packaging changes; and
4. avoid committing credentials, restricted data, evaluator archives, model
   traces, or local run outputs.

Scientific changes should state the treatment question, estimand, supported
analysis route, assumptions, uncertainty rule, and expected downstream action.
Changes to a released target or numerical result must update its report,
machine-readable table, figure source, and release manifest together.

Report security-sensitive problems through GitHub private vulnerability
reporting rather than a public issue.
