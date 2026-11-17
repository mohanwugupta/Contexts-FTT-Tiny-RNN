# Exploratory analyses

This directory preserves analyses that shaped the final project but are not
called by `scripts/run_all_final.py`.

The original task-SRN and probe-analysis chain documents the failure of a plain
task-trained network to provide the final human averaging account. The
environmental-pressure and partial-observability SRNs/GRUs tested alternative
mechanisms. PRDs and version-specific READMEs record the decisions made while
those hypotheses were being developed.

Historical scripts retain their original repository-relative imports. To rerun
one, temporarily place its archived package/script back at the recorded original
path or add the relevant archived package directory to `PYTHONPATH`; archived
output is already preserved under `archive/old_outputs/`. These workflows are
not supported as paper entry points and must not overwrite the frozen final
directories.

The root `pytest.ini` deliberately limits the paper regression to `tests/`.
Archived tests are preserved beside the historical code but are not collected
by the canonical test command.
