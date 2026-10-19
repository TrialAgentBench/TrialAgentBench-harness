import readme from "@harness/README.md?raw";

/** The README carries a model results section only once a campaign's results are published. */
export const RESULTS_HEADING = "Model results";
export const hasResults = readme.includes(`\n## ${RESULTS_HEADING}\n`);
