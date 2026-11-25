# Saved-table reproduction

From the repository root, in an environment with the existing numpy/pandas
dependencies:

```bash
PYTHONPATH=.:src python results/research_search_cloud/selected_analysis_verified_v3_1/verify_saved_tables.py --archive artifacts/selected-analysis-results.zip --out /tmp/selected-results-table-check
```

The destination must be new. The verifier checks the exact archive hash/CRC,
31 recovered outputs,10 native files, five pinned source hashes,8960 source
positions and score/probability consistency, six derived tables, all fixed-time
seed intervals and same-network classification on16 complete trajectories.
It reads ZIP contents directly and does not run any model, worker or optimizer.
The six derived CSVs and TABLE_VERIFICATION.json are written to the new directory.

For full139-original-input verification, first restore the original evidence
paths, then add `--source-root /PATH/TO/RESTORED/ROOT`. The public result archive
contains enough observations for selected table reproduction, but is not the
entire predecessor checkpoint/history archive. No original data is resampled or
retrained to supply missing evidence. VERIFICATION.json records the full local
139-input and original-position comparison already completed here.

Primary figures are byte-identical copies from the archive and can be viewed
directly under figures/. Source figure code is pinned by RUN_SPEC.json; table-only
rendering uses experiments.cloud_confidence_selected_v1.figures.render after
copying the archive's table products to a new directory. Cross-platform rendering
may differ in fonts/metadata and is not an original runtime-seal claim.

Keep RUN_SPEC.json, authorization.json and recovery receipts unchanged. The old
RUN_SPEC deadline is historical; the later authorization governs actual container
timing. Neither reproduction nor fixing output collection authorizes another
scientific execution, model call, follow-up or72-hour campaign.
