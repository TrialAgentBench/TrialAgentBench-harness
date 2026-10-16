# TrialAgentBench analysis executor

The release image contains Python 3.12 and R 4.5 in one immutable Docker
environment. Python packages are hash-locked in `requirements.lock`. Direct R
packages are version-pinned in the Dockerfile, and every run records the image
ID plus the complete installed Debian, Python, and R package inventories.

## Obtain the executor

The qualified host route is Linux x86_64 with Docker. The image is separate from
the Python installation and benchmark data. Campaign preparation can build and
attest it from the authenticated release toolkit; this requires access to the
pinned base image and package sources on the first build. The base image digest,
dated Debian snapshots, Python hashes and R versions are recorded in the build
context.

**Identity contract.** The executor's release identity is `environment_sha256`:
the hash of the complete probed Debian, Python and R inventories, support files
and execution limits. The release manifest also records `context_sha256` (the
build context) and the release owner's observed `image_id`. A Docker image ID is
a local observation, not a release identity: rebuilding the same context yields
a new ID (build timestamps, install logs and bytecode differ), and Docker's
containerd and classic image stores report different IDs for the same image. A
campaign pins one local image ID so that every run, resume and replay uses the
same image, and preparation, every native run and every replay probe that image
and refuse it unless its environment hash equals the release's
`environment_sha256`. A rebuilt or reloaded image with the same environment hash
is therefore accepted; an image with any inventory difference is refused.

For an offline host, use the optional prebuilt executor archive supplied with the
release. Keep its trusted checksum file alongside it:

```sh
sha256sum -c trialagentbench-executor.tar.sha256
docker image load --input trialagentbench-executor.tar
```

Set `executor_image_id` in the campaign configuration to the ID Docker reports
for the loaded image (`docker image inspect --format '{{.Id}}' IMAGE`). This
selects the prebuilt image and avoids an automatic source build; do not use a
mutable tag as the pin. Loading an archive is not itself attestation: preparation
and independent execution still probe the loaded image and enforce the release's
environment hash. No registry publication is implied by a local image archive.

The source-build route remains available from the public GitHub checkout:

```sh
docker build --iidfile executor-image.id --tag trialagentbench/executor:local executor/
```

For campaigns using a local source build, let preparation perform the toolkit-bound attestation
and save `executor-attestation.json` beside the campaign configuration. Preserve
that record when resuming or returning results. Container images, Docker's build
cache and installed host dependencies are outside benchmark cache/scratch size
estimates and should be budgeted separately.

Python includes pandas, PyArrow, SciPy, statsmodels, lifelines, scikit-learn,
matplotlib, and jsonschema. R includes the recommended packages (including
MASS, Matrix, nlme, mgcv, and survival), jsonlite, survey, sandwich, and the small
`trialagentbenchR` package. R analyses read release Parquet files with:

```r
library(trialagentbenchR)
data <- read_trialagentbench_parquet("data/analysis.parquet")
```

The bridge uses the image's pinned PyArrow installation and a bounded temporary
CSV with explicit scalar types and a collision-free missing-value marker. It
preserves Boolean indicators, numeric values, string identifiers (including
leading zeros and literal `NA`), and nulls. Nested, date/time and binary columns
are outside this reader's released scalar contract and fail explicitly. Integers
outside R's exact numeric range also fail; identifiers in that range must be
released as strings. Neither language can install packages or access the network
while a task is running.
