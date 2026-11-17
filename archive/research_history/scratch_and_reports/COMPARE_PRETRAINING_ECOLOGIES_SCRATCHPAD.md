# Compare Pretraining Ecologies — Implementation Scratchpad

## Status

- PRD translated into an isolated `compare_pretraining_ecologies` package.
- Existing frozen model, probe, and paper pipelines were not edited.
- Engineering smoke run completed against `/tmp`; it stopped at the internal
  pilot gate as designed and created no human-transfer files.

## Decisions

1. **Paired P1 controls.** P2--P5 do not have identical source distributions,
   so the implementation uses `P1_for_P2` ... `P1_for_P5`. Each pair shares the
   world seed, source cues, layouts, targets, and update count. Cue and layout
   assignments are independently deranged in P1. At human selection, the one
   P1 group is the matched control of the top-ranked structured ecology. This
   yields the requested P0 + P1 + top-two four-group design.
2. **Empirical layout manifold.** The four arrays in
   `human_svg_target_coordinates_v3.json` are centered and decomposed by SVD.
   World-specific directions lie in that empirical subspace; interpolation and
   jitter are clipped to the observed coordinate range. Registry names are not
   used.
3. **Smooth task law.** Local next-token uncertainty changes continuously with
   latent z at sparse positions. Targets remain ordinary sampled length-8 motor
   sequences and the loss remains the existing next-token plus normative
   execution objective. No transfer-profile or interpolation loss exists.
4. **P3 boundary law.** Within-regime task distributions change smoothly, while
   sparse response mappings change across a world-specific latent boundary.
5. **P4 composition.** Both factors are experienced jointly during pretraining;
   in V4, 11 is withheld within the new family and receives zero updates.
6. **P5 mixture.** A deterministic four-world block realizes the frozen
   50% smooth / 25% boundary / 25% compositional distribution exactly.
7. **Validation immutability.** V1--V4 use fresh descendants carrying only the
   pretrained weights/readout. New-family adaptation changes descendants; all
   probes are frozen. Parent model and optimizer hashes are asserted identical
   before and after the battery.
8. **Human firewall.** Human G2 files are not created until pilot admission,
   confirmatory gating, composite ranking, and top-two selection are complete.
   The primary first probe uses reset recurrent state and makes no update.
9. **Mechanism contingency.** Weight-subsystem ablations are implemented but
   run only when a selected structured condition both passes familiar gates and
   improves first-probe Mixed transfer relative to the selected P1 control.

## Verification log

- AST parse: all new package and runner files passed.
- Generator import/manifold construction: passed.
- Full engineering smoke (1 seed, 2 worlds, all P1--P5 pairs, V1--V4): passed.
- Isolated 20-sequence human branch plus all existing human analysis tables:
  passed (20 training rows, 70 frozen transition rows).
- Automated PRD contract tests: 7 passed, including the forced downstream
  selection/human integration path and all ten required figures.
- Full repository regression suite: 253 passed in 60.14 seconds.

## Commands

Engineering smoke:

```bash
python scripts/run_compare_pretraining_ecologies.py --smoke \
  --output /tmp/compare-pretraining-ecologies-smoke
```

Registered full pipeline (potentially long; automatic gates apply):

```bash
python scripts/run_compare_pretraining_ecologies.py \
  --config configs/compare_pretraining_ecologies.json \
  --output outputs/compare_pretraining_ecologies
```
