"""Load and validate frozen v3 tables plus the unchanged token contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from human_configured_state.data import load_human_token_data


@dataclass
class V3Inputs:
    root: Path
    run_config: dict
    pure_profiles: pd.DataFrame
    mixed: pd.DataFrame
    novel: pd.DataFrame
    srn_mixed: pd.DataFrame
    srn_novel: pd.DataFrame
    srn_audit: pd.DataFrame
    tokens: pd.DataFrame
    combined_raw_path: Path
    novel_raw_path: Path


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Required frozen v3 input is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required frozen v3 input is empty: {path}")
    return frame


def _limit(frame: pd.DataFrame, selected: set[str]) -> pd.DataFrame:
    return frame[frame.participant_uid.astype(str).isin(selected)].copy()


def load_v3_inputs(
    v3_output_dir, combined_path, novel_path, *, participant_limit_per_dataset=None
) -> V3Inputs:
    root = Path(v3_output_dir).expanduser().resolve()
    config_path = root / "human_run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    record = json.loads(config_path.read_text(encoding="utf-8"))
    pure = _read(root, "pure_ab_token_profiles.csv")
    mixed = _read(root, "mixed_midpoint_deviation.csv")
    novel = _read(root, "novel_midpoint_deviation.csv")
    srn_mixed = _read(root, "srn_mixed_profile.csv")
    srn_novel = _read(root, "srn_novel_profile.csv")
    audit = _read(root, "srn_frozen_audit.csv")
    combined_tokens = load_human_token_data(combined_path, "Combined", target_probe="Both")
    novel_tokens = load_human_token_data(novel_path, "Novel", target_probe="Novel")
    tokens = pd.concat([combined_tokens, novel_tokens], ignore_index=True)

    if participant_limit_per_dataset is not None:
        selected = set()
        for dataset, group in pure.groupby("dataset", sort=False):
            selected.update(sorted(group.participant_uid.astype(str).unique())[
                :int(participant_limit_per_dataset)
            ])
        pure = _limit(pure, selected)
        mixed = _limit(mixed, selected)
        novel = _limit(novel, selected)
        tokens = _limit(tokens, selected)
    if set(mixed.context) != {"Both"} or set(novel.context) != {"Novel"}:
        raise ValueError("Frozen Mixed and Novel v3 outcomes must remain separate.")
    if not audit.weights_unchanged.astype(bool).all():
        raise ValueError("The source v3 SRN audit contains a changed checkpoint.")
    if int(audit.optimizer_steps.sum()) or int(audit.backward_calls.sum()):
        raise ValueError("The source v3 SRN audit is not frozen.")
    return V3Inputs(
        root=root,
        run_config=record,
        pure_profiles=pure,
        mixed=mixed,
        novel=novel,
        srn_mixed=srn_mixed,
        srn_novel=srn_novel,
        srn_audit=audit,
        tokens=tokens,
        combined_raw_path=Path(combined_path).expanduser().resolve(),
        novel_raw_path=Path(novel_path).expanduser().resolve(),
    )
