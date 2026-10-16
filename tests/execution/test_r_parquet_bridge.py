"""Real R ingestion preserves released scalar types, identities and missingness."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from trialagentbench_verifier.executor import DockerCodeSession

from trialagentbench.ports.code_execution import CodeExecutionLimitsV1


@pytest.mark.executor
def test_r_standard_models_fit_supported_design_controls(tmp_path: Path) -> None:
    """Exercise installed mixed, clustered, survival, count and nonlinear tools."""
    program = """
set.seed(6906)
n <- 300L
d <- data.frame(id=factor(seq_len(n)), site=factor(rep(seq_len(30L), each=10L)),
                arm=rep(0:1, n/2), x=rnorm(n))
u <- rnorm(30L, sd=0.4)
d$y <- 0.4*d$arm + 0.3*d$x + u[d$site] + rnorm(n)
d$count <- rnbinom(n, mu=exp(0.2*d$arm + 0.2*d$x), size=2)
d$time <- rexp(n, exp(-0.3*d$arm + 0.2*d$x))
d$event <- as.integer(d$time < 2)
d$time <- pmin(d$time, 2)
long <- d[rep(seq_len(n), each=3L), ]
long$visit <- rep(0:2, n)
long$y <- long$y + 0.2*long$visit + rnorm(n*3L, sd=0.4)
mixed <- nlme::lme(y ~ arm*visit + x, random=~1|id, data=long)
design <- survey::svydesign(ids=~site, weights=~1, data=d)
clustered <- survey::svyglm(y ~ arm + x, design=design)
survival_fit <- survival::coxph(survival::Surv(time,event) ~ arm + x, cluster=site, data=d)
count_fit <- MASS::glm.nb(count ~ arm + x, data=d)
smooth <- mgcv::gam(y ~ arm + s(x,k=5), data=d, method='REML')
spline <- lm(y ~ arm + splines::ns(x,df=3), data=d)
coefficients <- c(nlme::fixef(mixed), coef(clustered), coef(survival_fit),
                  coef(count_fit), coef(smooth), coef(spline))
standard_errors <- c(sqrt(diag(vcov(mixed))), sqrt(diag(vcov(clustered))),
                    sqrt(diag(vcov(survival_fit))), sqrt(diag(vcov(count_fit))))
stopifnot(all(is.finite(coefficients)), all(is.finite(standard_errors)), all(standard_errors > 0))
cat('mixed-clustered-survival-count-nonlinear-pass')
"""
    (tmp_path / "analysis.R").write_text(program, encoding="utf-8")
    session = DockerCodeSession(tmp_path, image="trialagentbench/executor:1.1.1", limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result("source('analysis.R')", language="r")
        assert result.status == "success", result.output
        assert "mixed-clustered-survival-count-nonlinear-pass" in result.output
    finally:
        session.close()


@pytest.mark.executor
def test_r_parquet_preserves_scalar_values_and_missingness(tmp_path: Path) -> None:
    """Typed transfer distinguishes literal NA, empty strings and actual nulls."""
    frame = pd.DataFrame(
        {
            "participant_id": ["001", "NA", "", "__TRIALAGENTBENCH_MISSING__", None, "é,\nquoted"],
            "event_observed": pd.array([True, False, None, False, True, None], dtype="boolean"),
            "count": pd.array([1, 0, None, 17, -2, 10], dtype="Int64"),
            "outcome": [0.123456789012345, 1.0, float("nan"), -0.25, 1e-12, 10.0],
        }
    )
    frame.to_parquet(tmp_path / "input.parquet", index=False)
    session = DockerCodeSession(tmp_path, image="trialagentbench/executor:1.1.1", limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); d <- read_trialagentbench_parquet('input.parquet'); "
            "stopifnot(identical(d$participant_id,c('001','NA','','__TRIALAGENTBENCH_MISSING__',NA_character_,'é,\\nquoted')), "
            "identical(d$event_observed,c(TRUE,FALSE,NA,FALSE,TRUE,NA)), "
            "identical(d$count,c(1,0,NA,17,-2,10)), "
            "is.na(d$outcome[3]),abs(d$outcome[1]-0.123456789012345)<1e-16, "
            "d$outcome[5]==1e-12); cat('typed-roundtrip-pass')",
            language="r",
        )
        assert result.status == "success", result.output
        assert "typed-roundtrip-pass" in result.output
    finally:
        session.close()


@pytest.mark.executor
def test_r_parquet_rejects_integer_precision_loss(tmp_path: Path) -> None:
    """A large identity must fail explicitly instead of silently rounding in R."""
    pq.write_table(pa.table({"identity": pa.array([2**53 + 1], type=pa.int64())}), tmp_path / "input.parquet")
    session = DockerCodeSession(tmp_path, image="trialagentbench/executor:1.1.1", limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); read_trialagentbench_parquet('input.parquet')", language="r"
        )
        assert result.status == "execution_error"
        assert "lose integer precision" in result.output
    finally:
        session.close()


@pytest.mark.executor
@pytest.mark.parametrize("rows", (0, 3))
def test_r_parquet_preserves_empty_and_null_columns(tmp_path: Path, rows: int) -> None:
    """No observations must not become a fabricated row or inferred text value."""
    pq.write_table(
        pa.table({"id": pa.array([None] * rows, type=pa.string()), "event": pa.array([None] * rows, type=pa.bool_())}),
        tmp_path / "input.parquet",
    )
    session = DockerCodeSession(tmp_path, image="trialagentbench/executor:1.1.1", limits=CodeExecutionLimitsV1())
    try:
        result = session.execute_result(
            "library(trialagentbenchR); d <- read_trialagentbench_parquet('input.parquet'); "
            f"stopifnot(nrow(d)=={rows},is.character(d$id),is.logical(d$event),all(is.na(d$id)),all(is.na(d$event)))",
            language="r",
        )
        assert result.status == "success", result.output
    finally:
        session.close()
