"""Read-only, post-gate human profiles used only for unfitted correspondence."""

from __future__ import annotations

import numpy as np

from human_configured_state.data.load_human import load_human_token_data


def _mean_profile(frame, mask):
    values = frame[mask].groupby("transition_position").log_rt.mean().reindex(range(1, 8))
    if values.isna().any(): return None
    return values.to_numpy(float)


def load_profiles(config):
    combined = load_human_token_data(config.combined_data, "Combined", target_probe="Both")
    novel = load_human_token_data(config.novel_data, "Novel", target_probe="Novel")
    mixed = _mean_profile(combined, combined.probe_epoch.eq("Mixed1") & combined.sequence_repetition.eq(1))
    novel_profile = _mean_profile(novel, novel.probe_epoch.eq("Novel1") & novel.sequence_repetition.eq(1))
    result = {"Mixed": mixed, "Novel": novel_profile}
    # Default is not tagged as a probe by the shared retrieval contract; recover
    # its first sequence from the earliest no-context block without changing rows.
    for path, dataset in ((config.combined_data, "Combined"), (config.novel_data, "Novel")):
        try:
            default = load_human_token_data(path, dataset, target_probe="Default")
            blocks = default[default.context.eq("Default")].block_order.dropna().sort_values().unique()
            if len(blocks):
                profile = _mean_profile(default, default.context.eq("Default") & default.block_order.eq(blocks[0]) & default.sequence_repetition.eq(1))
                if profile is not None: result["NoContext"] = profile; break
        except (ValueError, KeyError):
            continue
    return {key: value for key, value in result.items() if value is not None}
