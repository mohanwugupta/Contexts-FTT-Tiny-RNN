# Reproducibility

## Environment and data

The validated environment is Python 3.11 with the exact package versions in
`requirements.txt`. Create a fresh environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the raw human exports at the repository-relative paths declared in
`configs/human_final.json`:

```text
data/mixedBreak_clean.csv
data/threeContextsBreak_clean.csv
```

The SRN data are generated from the fixed curriculum. No external model weights
are downloaded.

## Tests

```bash
python -m pytest -q
```

The paper-contract tests freeze sequence/context definitions, A10/B10 exposure,
initialization scale, zero-update probes, A/B-only human profiles, transition
indexing, rank 2, final model ordering, headline schemas, and all baseline CSV
hashes.

## Canonical commands

From scratch:

```bash
python scripts/run_srn_final.py --config configs/srn_final.json
python scripts/run_human_final.py --config configs/human_final.json
python scripts/make_paper_figures.py
```

Equivalent one-command run:

```bash
python scripts/run_all_final.py
```

The deterministic v2.1 checkpoints can be reused for a faster frozen-analysis
replay:

```bash
python scripts/run_all_final.py --reuse-execution
```

For table-only verification, use `--no-plots` on the SRN/human runners or
`--no-analysis-plots` on the all-analysis runner. Paper figures are always
regenerated from CSVs by `scripts/make_paper_figures.py`.

## Outputs

Registered versioned output roots are declared in the two JSON configs. The
paper-facing layer is:

```text
outputs/final/
    figures/                 # Figures 1–5
    tables/                  # exact copies of every plotted source CSV
    paper_figure_sources.json
```

`paper_figure_sources.json` records the source path and SHA-256 for every copied
table. The figure script never fits a model.

## Pre-cleanup baseline and equivalence

`reproducibility/baseline_manifest.json` records:

- the pre-cleanup test result;
- exact final commands and configuration files;
- raw-input hashes;
- every frozen final CSV's schema and SHA-256;
- numerical snapshots underlying paper Figures 1–5;
- independent replay results for all feasible pipelines.

`reproducibility/baseline_checksums.sha256` contains 180 raw inputs, configs,
CSVs, reports, and analysis figures. The checksum-validated checkpoint manifest
separately registers all 240 checkpoint files. Before cleanup, all 180 baseline
artifacts validated and the v2.1, v2.2, v2.3, and v3 reruns matched every CSV
byte-for-byte. The full v3.1 pipeline was rerun immediately before the baseline
and selected rank 2.

After a rerun, use:

```bash
python reproducibility/verify_final_outputs.py
```

Exact hashes are required. The final pipelines have already reproduced
bit-for-bit on the validated environment, so no numerical-tolerance exception is
currently justified. Any future platform that cannot reproduce bitwise must
document the affected files and add a reviewed, tight comparison procedure; it
must not permit schema, row-order, categorical, rank, model-ordering, or
decision-code changes.

## Portability

Canonical configs and commands use only repository-relative paths. Historical
run audit JSONs may contain the absolute path of the machine on which they were
originally created; those strings are provenance fields, not runtime inputs.
They are intentionally not rewritten because doing so would invalidate the
frozen baseline.
