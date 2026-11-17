"""Compatibility-first human data adapter with a hard familiar-only firewall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from human_configured_state.data.load_human import RAW_COLUMNS, build_token_rows
from state_space_bridge.data import load_human_sequence_data, make_human_average_traces


@dataclass(frozen=True)
class ParticipantData:
    participant_id: str
    contexts: np.ndarray
    blocks: np.ndarray
    sequence_in_block: np.ndarray
    global_sequence: np.ndarray
    y: np.ndarray
    historical_y: np.ndarray

    @property
    def n_sequences(self) -> int:
        return int(self.y.shape[0])


def assert_familiar_only(frame: pd.DataFrame) -> None:
    contexts = set(frame["context"].dropna().astype(str))
    forbidden = contexts - {"A", "B"}
    if forbidden:
        raise ValueError(
            "Human state-space fitting accepts familiar A/B rows only; "
            f"found {sorted(forbidden)}. Mixed/Novel/NoContext are firewalled."
        )


def load_familiar_transition_data(path: str | Path, participant_limit: int | None = None) -> pd.DataFrame:
    """Reuse retrieval_v1 exclusions, then retain seven correct RT transitions.

    Cohort membership and the time of the first probe come from the validated
    sequence contract. Probe outcomes are never returned to the fitter.
    """
    source = Path(path)
    sequences = load_human_sequence_data(source, "Combined")
    accepted = sequences[
        sequences.context.isin(["A", "B"]) & sequences.before_first_probe
    ].copy()
    assert_familiar_only(accepted)
    if accepted.original_sequence_in_block.eq(1).any():
        raise AssertionError("The historical first sequence exclusion was not preserved.")
    if participant_limit is not None:
        keep = sorted(accepted.participant_id.unique())[: int(participant_limit)]
        accepted = accepted[accepted.participant_id.isin(keep)].copy()

    raw = pd.read_csv(source, usecols=list(RAW_COLUMNS))
    token = build_token_rows(raw, accepted, dataset="Combined")
    keys = accepted[[
        "participant_id", "block_id", "original_sequence_in_block", "observed",
        "global_sequence", "analysis_sequence_in_block", "before_first_probe",
    ]].drop_duplicates()
    token = token.merge(
        keys,
        on=["participant_id", "block_id", "original_sequence_in_block"],
        how="inner", validate="many_to_one", suffixes=("", "_historical"),
    )
    token = token[token.context.isin(["A", "B"])].copy()
    assert_familiar_only(token)
    token = token.rename(columns={"observed": "historical_sequence_rt"})
    token["analyzed_row_id"] = (
        token.participant_id.astype(str) + "|b" + token.block_id.astype(str)
        + "|s" + token.original_sequence_in_block.astype(str)
        + "|k" + token.transition_position.astype(str)
    )
    if token.analyzed_row_id.duplicated().any():
        raise AssertionError("Duplicate analyzed transition rows.")
    counts = token.groupby([
        "participant_id", "block_id", "original_sequence_in_block"
    ]).transition_position.nunique()
    if not counts.eq(7).all():
        raise AssertionError("Every analyzed sequence must have seven transitions.")
    token["rt_scale"] = "untransformed milliseconds"
    token["preprocessing_contract"] = "retrieval_v1 + historical pre-probe A/B scope"
    return token.sort_values([
        "participant_id", "block_order", "sequence_repetition", "transition_position"
    ], kind="stable").reset_index(drop=True)


def participant_data(frame: pd.DataFrame) -> list[ParticipantData]:
    assert_familiar_only(frame)
    result = []
    for participant, rows in frame.groupby("participant_id", sort=False):
        index = ["block_id", "original_sequence_in_block"]
        matrix = rows.pivot_table(index=index, columns="transition_position", values="mean_rt", aggfunc="first")
        meta = rows.drop_duplicates(index).set_index(index).loc[matrix.index]
        if list(matrix.columns.astype(int)) != list(range(1, 8)):
            raise AssertionError("Transition positions must be exactly 1..7.")
        result.append(ParticipantData(
            participant_id=str(participant),
            contexts=meta.context.to_numpy(dtype=str),
            blocks=matrix.index.get_level_values("block_id").to_numpy(),
            sequence_in_block=matrix.index.get_level_values("original_sequence_in_block").to_numpy(dtype=int),
            global_sequence=np.arange(1, len(meta) + 1),
            y=matrix.to_numpy(dtype=float),
            historical_y=meta.historical_sequence_rt.to_numpy(dtype=float),
        ))
    return result


def historical_average_traces(path: str | Path) -> pd.DataFrame:
    frame = load_human_sequence_data(path, "Combined")
    return make_human_average_traces(frame)
