# Bounded saved-readout analysis

This separate namespace reads archived predictions only. It adds native/quarter-view and paired-initialization accounting at fixed saved updates, preserving all logits/probabilities and the frozen native evaluator. It never trains, loads checkpoints or forwards a model. The selected proposal and coordinator decision are supplied privately; no research result is embedded here.

The worker requires Linux/x86_64, Landlock ABI>=3 and a seccomp filter. It tests source-write denial, private read/link denial, denied network sockets and allowed output writes before reading observations. Unsupported/failed controls stop execution. This verifies only the worker's restrictions, not the parent Cloud agent's permissions or independent agent contexts. Sources: https://docs.kernel.org/userspace-api/landlock.html and https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html .

```bash
PYTHONPATH=.:src python -m pytest -q tests/cloud_confidence_selected_v1
PYTHONPATH=.:src python -m experiments.cloud_confidence_selected_v1.driver --selection PRIVATE_SELECTION.json --proposal PRIVATE_PROPOSAL.json --out NEW_PRIVATE_JOB_DIR --deadline-utc EXPLICIT_UTC_DEADLINE
```

Never launch an existing job again. The hard job caps are120CPU seconds and min(300wall seconds, time until the explicit deadline). A failed guard, missing file, changed hash, invalid pairing or failed score comparison remains failed/blocked; there is no unrestricted fallback. Native output flags and the row-level four-outcome criterion are unchanged. Source observations are hashed before/after; reports remain private. The original72-hour campaign is not touched.
