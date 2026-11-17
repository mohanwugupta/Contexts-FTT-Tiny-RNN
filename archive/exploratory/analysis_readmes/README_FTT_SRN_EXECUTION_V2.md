# FTT High-Scale SRN with execution control (PRD v2.1)

This is an isolated implementation of `PRD_v2_1_FTT_high_scale_SRN_execution_control.md`.
It does not read human-data files and does not modify or import the older PRD2/PRD3/GRU fitting
pipelines.

The experiment trains the exact fixed eight-unit tanh Elman SRN under three registered conditions:

- High-Scale (`SD=.5`) with sequence and execution objectives;
- Low-Scale (`SD=.02`) with the same objectives;
- High-Scale sequence-only objective control.

Training is 17 alternating blocks of ten A/B sequence repetitions. Mixed and Novel are frozen
counterfactuals only. Novel's input-to-hidden column starts at exact zero, receives zero gradient,
and is re-zeroed after every optimizer step.

## Commands

Run the quick end-to-end software check first:

```bash
python scripts/run_ftt_srn_execution_v2.py \
  --profile smoke \
  --output outputs/ftt_srn_execution_v2_smoke
```

Run the five-seed pilot:

```bash
python scripts/run_ftt_srn_execution_v2.py \
  --profile pilot \
  --output outputs/ftt_srn_execution_v2_pilot
```

Run the preregistered 20-seed analysis:

```bash
python scripts/run_ftt_srn_execution_v2.py \
  --profile final \
  --output outputs/ftt_srn_execution_v2_final
```

The CLI prints timestamped progress at every fit, training block, probe stage, output stage, and
figure stage. `--quiet` suppresses those messages. Output paths are written directly; the only
temporary location is the harmless Matplotlib font cache, not model results.

`--force-probe-interpretation` is restricted to smoke testing. It verifies downstream software when
a single seed cannot establish a cohort-level gate and must not be used for scientific inference.

## Interpretation and artifacts

The pipeline always writes the 18 required CSVs, `run_config_v2.json`, a Markdown decision report,
the checksum-protected checkpoint manifest used by PRD v2.2, and nine figures. Frozen probe values
remain available when a gate fails for auditability, but their
`interpretation_allowed` field is false.

The decision order is strict:

1. High versus Low representational splitting, mastery, switch cost, transfer, and context use;
2. confidence-grounded execution speed overall, within A/B, and controlling block/context;
3. within-network Mixed and Novel interpolation relative to sufficiently distinct A/B RT endpoints.

The grounding gate uses the final eight complete A/B blocks, after the network has acquired the
sequence. This tests the learned confidence–speed policy without conflating it with early adaptation
from random High-Scale execution-head weights.

`execution_control_report.md` explicitly separates representational interpolation from behavioral
interpolation. The primary Mixed input is norm matched; additive Mixed is only a sensitivity check.

## Tests

```bash
PYTHONPYCACHEPREFIX=/private/tmp/execv2-pycache \
MPLCONFIGDIR=/private/tmp/ftt-srn-v2-mpl \
XDG_CACHE_HOME=/private/tmp/ftt-srn-v2-xdg \
python -m pytest -q -p no:cacheprovider tests/test_execv2_*.py
```
