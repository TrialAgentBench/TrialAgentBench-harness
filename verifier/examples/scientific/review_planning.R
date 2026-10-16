# Recalculate the disclosed Ariadne planning examples with R's standard library.
# Usage: Rscript review_planning.R references/planning.csv planning-results.csv
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(2L, 3L)) stop("usage: review_planning.R input.csv output.csv [clinical-results-directory]")
plans <- read.csv(args[[1]], stringsAsFactors = FALSE)
required <- c("request_id", "design_type", "power", "two_sided_alpha",
              "treatment_allocation", "candidate_information", "relative_tolerance")
if (!all(required %in% names(plans)) || nrow(plans) == 0L) {
  stop("planning input requires complete design rows")
}
if (anyNA(plans[required]) || any(plans$treatment_allocation != 0.5)) {
  stop("these reference comparisons require equal allocation and complete inputs")
}
if (any(plans$power <= 0.5 | plans$power >= 1) ||
    any(plans$two_sided_alpha <= 0 | plans$two_sided_alpha >= 0.5) ||
    any(plans$candidate_information <= 0 | plans$relative_tolerance < 0)) {
  stop("invalid information, error rate or comparison tolerance")
}
results <- lapply(seq_len(nrow(plans)), function(i) {
  p <- plans[i, ]
  if (p$design_type == "continuous") {
    ref <- power.t.test(delta = abs(p$treatment_difference),
                        sd = p$outcome_standard_deviation,
                        sig.level = p$two_sided_alpha, power = p$power,
                        type = "two.sample", alternative = "two.sided")
    value <- ceiling(2 * ref$n)
    method <- "stats::power.t.test"
    unit <- "total participants"
  } else if (p$design_type == "binary") {
    ref <- power.prop.test(p1 = p$control_probability,
                           p2 = p$treatment_probability,
                           sig.level = p$two_sided_alpha, power = p$power,
                           alternative = "two.sided")
    value <- ceiling(2 * ref$n)
    method <- "stats::power.prop.test"
    unit <- "total participants"
  } else if (p$design_type == "survival") {
    value <- ceiling((qnorm(1 - p$two_sided_alpha / 2) + qnorm(p$power))^2 /
                     (0.25 * log(p$hazard_ratio)^2))
    method <- "Schoenfeld event identity"
    unit <- "primary events"
  } else {
    stop(paste("unsupported reference family:", p$design_type))
  }
  difference <- abs(p$candidate_information - value) / value
  data.frame(request_id = p$request_id, method = method, unit = unit,
             candidate_information = p$candidate_information,
             reference_information = value, relative_difference = difference,
             relative_tolerance = p$relative_tolerance,
             passed = difference <= p$relative_tolerance,
             R_version = as.character(getRversion()))
})
results <- do.call(rbind, results)
write.csv(results, args[[2]], row.names = FALSE)
print(results, row.names = FALSE)
if (!all(results$passed)) stop("planning reference comparison failed")

# If a worked-study directory is supplied, independently fit its observed data.
if (length(args) == 3L) {
  root <- args[[3]]
  records <- read.csv(file.path(root, "families", "asthma_standard_analysis.csv"))
  question <- read.csv(file.path(root, "families", "asthma_analysis_question.csv"))
  reported <- read.csv(file.path(root, "families", "asthma_results.csv"))
  if (nrow(question) != 1L || !setequal(unique(records$assigned_arm), c("active", "control"))) {
    stop("clinical reference requires one question and two declared arms")
  }
  flags <- tolower(as.character(records$event_observed))
  if (anyNA(flags) || !all(flags %in% c("true", "false"))) stop("invalid event status")
  records$event_observed <- flags == "true"
  arms <- lapply(c("active", "control"), function(arm) {
    data <- records[records$assigned_arm == arm, ]
    fit <- survival::survfit(survival::Surv(observed_time, event_observed) ~ 1, data = data)
    at_horizon <- summary(fit, times = question$horizon[[1]])
    if (length(at_horizon$surv) != 1L) stop("the declared horizon lacks observed support")
    c(risk = 1 - at_horizon$surv, variance = at_horizon$std.err^2)
  })
  estimate <- arms[[1]][["risk"]] - arms[[2]][["risk"]]
  margin <- qnorm(0.975) * sqrt(arms[[1]][["variance"]] + arms[[2]][["variance"]])
  python <- reported[grepl("^Kaplan", reported$analysis), ]
  if (nrow(python) != 1L) stop("one Python KM result is required")
  result <- data.frame(quantity = c("estimate", "lower", "upper"),
                       R_value = c(estimate, estimate - margin, estimate + margin),
                       Python_value = c(python$estimate, python$lower, python$upper))
  result$absolute_difference <- abs(result$R_value - result$Python_value)
  result$tolerance <- 1e-10
  result$passed <- result$absolute_difference <= result$tolerance
  result$input_view <- "current ADTTE export"
  result$R_version <- as.character(getRversion())
  result$survival_version <- as.character(packageVersion("survival"))
  write.csv(result, file.path(root, "clinical-reference-results.csv"), row.names = FALSE)
  print(result, row.names = FALSE)
  if (!all(result$passed)) stop("clinical reference comparison failed")
}
