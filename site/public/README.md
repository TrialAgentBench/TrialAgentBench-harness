# TrialAgentBench site

Serve this directory with a local web server:

```bash
python3 -m http.server 8000 --directory .
```

Open `http://localhost:8000/`. The home page states what the benchmark measures, Methods and evidence
renders the methods with the release evidence, and Run it renders the running guide. `artifacts/` holds
the release evidence record, its release facts, cell cards and realism figures.

This build uses directory routes. To serve below a URL subpath, rebuild with
`TRIALAGENTBENCH_SITE_BASE=/chosen-path/` as described in the source README.
