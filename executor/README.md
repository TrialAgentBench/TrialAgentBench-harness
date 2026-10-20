# TrialAgentBench analysis executor

The release image contains Python 3.12 and R 4.5 in one immutable Docker
environment. Python packages are hash-locked in `requirements.lock`. Direct R
packages are version-pinned in the Dockerfile, and every run records the image
ID plus the complete installed Debian, Python, and R package inventories.

The lock installs binary wheels only. `wheels/` vendors autograd-gamma 0.5.0, a
lifelines dependency published on PyPI only as a source archive, as a wheel built
from that archive (kept beside it) so the lock can pin it by hash. It is MIT
licensed; the licence ships inside both files.

## Obtain the executor

The qualified host route is Linux x86_64 with Docker. The image is separate from
the Python installation and benchmark data. Campaign preparation can build and
attest it from the authenticated release toolkit; this requires access to the
pinned base image and package sources on the first build. The base image digest,
dated Debian snapshots, Python hashes and R versions are recorded in the build
context.

From the public repository root, build the image used by the offline clinical demo:

```sh
docker build --tag trialagentbench/executor:1.0.0 executor/
```

The demo needs this image even though its task data and scripted analyst are bundled. A campaign
instead authenticates the executor against its matching release manifest during preparation.
Each analysis container is limited to 2 CPUs, 4,096 MB and 300 seconds per execution; a concurrent
task reserves two container slots for the live session and independent replay. Size host and
Docker VM resources for that concurrency.

**Identity contract.** The executor's release identity is `environment_sha256`:
the hash of the complete probed Debian, Python and R inventories, support files
and execution limits. The release manifest also records `context_sha256` (the
build context) and the release owner's observed `image_id`. A Docker image ID is
a local observation, not a release identity: rebuilding the same context can yield
a different ID (build timestamps, install logs and bytecode differ), and Docker's
containerd and classic image stores report different IDs for the same image. A
campaign pins one local image ID so that every run, resume and replay uses the
same image, and preparation, every native run and every replay probe that image
and refuse it unless its environment hash equals the release's
`environment_sha256`. Preparation accepts a rebuilt or reloaded image with the same environment
hash; an existing campaign still requires its recorded local ID. An image with any inventory
difference is refused.

For an offline host, use a retained executor archive. Keep its independently trusted checksum
file alongside it:

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

To retain the image ID of a source build, run from the public GitHub repository root:

```sh
docker build --iidfile executor-image.id --tag trialagentbench/executor:1.0.0 executor/
```

For campaigns using a local source build, let preparation perform the toolkit-bound attestation
and save `executor-attestation.json` beside the campaign configuration. Preserve
that record when resuming or returning results. Rebuilt local image IDs are not globally
reproducible package identities.

## Retain a campaign image

Preparation tags the pinned image as
`trialagentbench/executor-campaign:<campaign-path-hash>-<plan-checksum-prefix>` and prints the exact
`docker image save` command for `CAMPAIGN/executor-image.tar`. The local tag protects against
dangling-image pruning; it does not protect against deleting tagged images. Preparation does not
write the archive or its digest. Run the printed command before starting or pruning images, then
record its SHA-256, for example from the campaign directory:

```sh
sha256sum executor-image.tar > executor-image.tar.sha256
```

Keep the archive, checksum, immutable local image ID in `configuration.json`, and
`executor-attestation.json` beside the input configuration. On recovery, verify the bytes before
loading:

```sh
sha256sum --check executor-image.tar.sha256
docker image load --input executor-image.tar
```

Resume still requires the campaign's pinned local image ID and a successful environment probe.
For preparation of a new attestation, `TRIALAGENTBENCH_EXECUTOR_IMAGE=LOCAL_TAG_OR_ID` selects an
equivalent local image: its complete probed environment hash must match the release. This override
is used when no valid attestation or explicit `executor_image_id` already selects the image.
It does not rewrite an existing campaign's pin.

If `executor-attestation.json` names an image that has disappeared, `campaign prepare` authenticates
the release toolkit, reuses a matching local image or builds from that toolkit, and re-attests. It
retains the old file as `executor-attestation.<digest-prefix>.json` before replacement. When the
toolkit is unavailable, including development inputs, the error identifies the stale file: restore
the saved image or prepare with the authenticated release toolkit. A differing environment hash
is refused.

Container images, Docker's build
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
