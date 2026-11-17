"""Disposable synthetic validation of the prior during target-task learning."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from context_controlled_recurrent_architectures.analysis import decode_context_belief
from context_controlled_recurrent_architectures.models import build_model
from context_controlled_recurrent_architectures.synthetic import SyntheticRun, validate_runs


EVENTS = {0: "before_human_training", 20: "early_human_training", 40: "before_first_Mixed", 120: "before_final_probe_and_end"}


def prior_retention(config, upstream_config, runs, progress=None):
    rows = []; worlds = int(config.payload["retention_validation_worlds"])
    for run in runs:
        if run.architecture not in {"A", "B"}: continue
        for checkpoint, event in EVENTS.items():
            model = build_model(run.architecture, run.seed); model.load_state_dict(deepcopy(run.snapshots[checkpoint]["model_state_dict"]))
            synthetic = SyntheticRun(run.architecture, run.prior_condition, run.seed, "human_retention", model, pd.DataFrame(), pd.DataFrame(), {})
            validation = validate_runs(upstream_config, [synthetic], worlds)
            belief, _, _ = decode_context_belief(validation["belief_raw"], validation["continuum"])
            rows.append({"group": run.group, "architecture": run.architecture, "prior_condition": run.prior_condition, "seed": run.seed,
                         "checkpoint": checkpoint, "retention_event": event, "validation_worlds": worlds,
                         "synthetic_split_score": validation["split"].AB_hidden_distance.mean(),
                         "synthetic_ambiguity_NLL": validation["ambiguity"].ambiguous_query_NLL.mean(),
                         "context_belief_correlation": belief.posterior_correlation.mean(), "parameters_unchanged": validation["audit"].parameters_unchanged.all()})
            if progress: progress(f"retention {run.group} seed {run.seed} checkpoint {checkpoint}")
    return pd.DataFrame(rows)
