# Mechanism-first probe diagnostics scratchpad

## Scope frozen before implementation

- No SRN training, checkpoint replay, probe rerun, architecture change, or new RT readout.
- Human primary scale is cleaned milliseconds; the full level/shape and q/delta analysis is repeated on log RT as robustness.
- Human cleaning is delegated to the existing `retrieval_v1` contract and `build_token_rows`; no new exclusion or outlier rule is added.
- The SRN source is the frozen 20-seed confirmatory `High_monotonic`, Stage 2 layout, zero-overlap probe output.
- Transition identity is positions 1--7, representing ordinal transitions 1→2 through 7→8 in both systems.
- Mixed is the internal alias for Combined/Both; Novel is the internal alias for Third. Original labels remain in row-level exports.

## Analysis decisions

1. A participant/seed condition profile is decomposed exactly as `M = L + Z`; q is fit only to centered `Z` and delta is computed afterward from levels.
2. Unconstrained q is primary for Stage 1 description. A clipped [0,1] value is a labeled sensitivity. Near-zero A/B centered contrasts retain a row but q is `NaN` and flagged; a neutral 0.5 anchor is used only to keep a finite, explicitly flagged delta diagnostic.
3. Human uncertainty resamples complete participants; SRN uncertainty resamples complete seeds. The configuration freezes 2,000 draws and seed 41027.
4. Whole-profile Fast/Slow canonicalization is independently re-derived from A/B means and required to reproduce the previously frozen A/B-only map. It never examines Mixed, Novel, or human similarity. As expected mathematically, swapping the endpoints can reorient q but cannot improve the fitted A/B line or its shape error.
5. Existing predictions are never refit. Raw and centered scores use identical original held-out rows. Rank-2 signed predictions are the previously reconstructed rows whose errors match the frozen v3.1 artifact.
6. The additive-cost augmentation uses one condition-wide intercept. It is estimated on already-held-out test repetitions 2/6 and evaluated on already-held-out repetitions 4/8. This is a nested repetition holdout and cannot alter centered predictions.
7. History candidates are audited before coefficients. The preregistered support, within-participant, epoch/order, rank, correlation, and VIF thresholds live in the JSON config. A failed gate is reported as non-identifiability, never as a psychological null.
8. The human Default basis comes only from disjoint participants in the canonical `Default`/no-context control condition in `threeContextsBreak_clean.csv`. Participant centering and the across-control mean are fixed without Novel rows. It is not `(A+B)/2`.
9. The frozen SRN sequence-only control has no execution RT. Therefore it is inventoried but rejected as an SRN default RT basis. No substitute network output is generated.
10. Human milliseconds and normalized SRN RT are not numerically compared. Cross-system adjudication compares qualitative global-cost signs and centered-profile reuse, with raw delta retained in system-specific units.

## Restart contract

Use `--from-stage N --through-stage M`. A restart after Stage 0 must match the saved configuration hash and manifest hash. Stage 5 rehashes every input and fails if any source changed.

## Implementation log

- Added `mechanism_probe_diagnostics/` with pure analysis, source/firewall, history, default-basis, plotting, and pipeline modules.
- Added `scripts/run_probe_diagnostics.py` and portable config `configs/mechanism_first_probe_diagnostics.json`.
- Added tests corresponding to PRD T1--T14. Final commands and observed branch decisions will be appended after verification.

## Verification and observed branch results

- Focused PRD suite: `14 passed`.
- Repository-wide regression suite: `174 passed in 29.16s`.
- A clean `--config configs/mechanism_first_probe_diagnostics.json` run completed
  Stages 0--5 and the final rehash reported every frozen source unchanged.
- History: stopped non-identifiable. No candidate simultaneously met repeated
  support, within-participant variation, order/epoch independence, rank, and
  collinearity gates. This is explicitly not a null history result.
- Human independent Default: accepted from 27 disjoint Novel-dataset no-context
  controls (5,103 transition rows; split-half profile correlation 0.7813).
- SRN independent Default: stopped. The 20-seed sequence-only control has no
  execution RT and no saved training-only invariant RT component exists.
- Existing held-out rank-2 structure retained centered-shape value (R² 0.3388
  for Mixed and 0.4026 for Novel on log RT in the final score table).
- Frozen High-scale SRN Mixed had positive global delta (mean 0.2726, bootstrap
  95% CI 0.2075--0.3473) but poor A/B centered-profile reuse (mean R² -0.0875).
  Whole-profile Fast/Slow canonicalization reoriented q only and left fit/error
  unchanged, as the algebra requires.
- The decision gate is Result B: the selected next target is an explicit
  compositional retrieval/gating mechanism, not another readout-only change.
