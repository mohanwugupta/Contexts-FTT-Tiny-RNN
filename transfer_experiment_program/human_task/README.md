# Local transfer task

Generate the study configurations first:

```bash
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e4
```

Then serve the repository and open the printed local URL:

```bash
python scripts/serve_transfer_human_task.py --port 8765
```

The shared engine supports E2 history/practice, E3 cue/layout-factorial, and E4 action-visibility configurations. A stored counterbalance code deterministically assigns a cell. Browser state is resumable in local storage; JSON and CSV event exports are available at completion or abort. “Simulated participant” is only an engineering mode and labels every exported event `synthetic_task_validation`.

The response keys are A/S/D/F. An error restarts the current sequence, matching the behavior recovered from the legacy exports. In `one_target`, only the current action label enters the DOM; future positions are generic unlabeled placeholders. Context cue and layout rendering are independent from action visibility.

Analyze a real export with:

```bash
python scripts/run_transfer_experiment_program.py \
  --config configs/transfer_experiment_program.yaml \
  --stage e4 --force --human-events path/to/events.csv
```

Human collection, consent, recruitment, and deployment are separate research operations and are not performed by this repository runner.
