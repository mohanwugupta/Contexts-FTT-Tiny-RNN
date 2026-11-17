# Mechanism-first probe diagnostics

This analysis is read-only with respect to every human and SRN source. It does
not load a model checkpoint, construct an optimizer, train a network, rerun a
probe, or alter a canonical artifact.

## Run

From the repository root:

```bash
python scripts/run_probe_diagnostics.py \
  --config configs/mechanism_first_probe_diagnostics.json
```

To resume from a completed provenance stage, for example:

```bash
python scripts/run_probe_diagnostics.py \
  --config configs/mechanism_first_probe_diagnostics.json \
  --from-stage 3 --through-stage 5
```

Stage 0 must have run once before a later-stage restart. The runner rejects a
saved manifest created by another configuration, and Stage 5 rehashes every
input to enforce source immutability.

## Stages

1. Input manifest, schema, alias, transition, leakage, and canonicalization audit.
2. Exact `M = L + Z` profiles and unconstrained/constrained q plus additive delta,
   with 2,000 participant/seed bootstrap draws.
3. Raw-level versus centered-shape rescoring of unchanged held-out model rows;
   one additive delta is evaluated on nested held-out repetitions.
4. History support/confound/rank/VIF gates, followed by H0/Hdelta/Hq/Hboth only
   when the frozen gates pass.
5. Independent human no-context basis and Novel comparison; the corresponding
   SRN branch stops if no independent frozen RT source exists.
6. Cross-system decision table, source-immutability check, and Markdown report.

Outputs are in `outputs/mechanism_first_probe_diagnostics/`. Every CSV contains
an explicit unit and condition plus the configuration and input-manifest hashes.
Every generated figure has a `figure_N_source.csv` table. Figure 5 is omitted
when history is not identifiable, while its source/status placeholder remains.

The implementation decisions and run log are in
`reproducibility/MECHANISM_FIRST_PROBE_DIAGNOSTICS_SCRATCHPAD.md`.
