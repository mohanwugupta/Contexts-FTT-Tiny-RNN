# State-space bridge

## What is a “state” here?

A state is a compact, dynamically updated estimate of current task performance. At the
investigator’s direction, the canonical implementation is the no-fatigue family in
`state-space_SRN.Rmd`: Lump has one state; Cross and Split have context-gated A/B states. A
two-term OLS decoder maps the state to sequence RT for humans or normalized RT for SRNs.

The port preserves the R notebook’s indexing, initial states, bounds, decoder, and BIC parameter
counts. Both humans and SRNs call the same Python engine.

## Why is Mixed not part of the state-space learning model?

Lump/Cross/Split answer what was learned during pure A/B practice. Mixed, Novel, and No Context
are retrieval tests. They contribute zero rows to state dynamics, decoder fitting, BIC, and model
selection. Probe outcomes can therefore be permuted or removed without changing Stage 1.

## Why is low rank a vector extension?

The scalar model supplies one A and B value. The transition analysis replaces each scalar with a
seven-position vector and estimates a smaller basis spanning those profiles. The bridge reads the
rank chosen by held-out dimension selection and regression-checks the current rank-2 result; it
does not assign rank two inside the estimator.

## Corrected SRN gate

Final SRN inputs must contain High scale 2 + execution, Low scale + execution, and High scale 2
sequence-only. The old High scale .5 run is rejected. Before state-space fitting, the corrected
run must show CE down, confidence up, speed up, normalized RT down, and exact RT semantics within
float32 tolerance.

The faithful two-seed smoke run failed the speed/RT direction gate. In accordance with the PRD,
the full pipeline stops and writes `SCIENTIFIC_GATE_FAILURE.md`; it does not tune the old model
code to obtain a preferred result.

## Run order

```bash
python scripts/run_full_state_space_bridge.py --profile smoke --stage all --no-plots
python scripts/run_full_state_space_bridge.py --profile final --stage all
```

Use `--stage corrected-srn`, `downstream`, or `assemble` to resume an existing additive run. The
scripts never write into the frozen v2.1/v2.2/v2.3 output directories.

