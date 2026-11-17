#!/usr/bin/env python3
"""Run corrected scale-2 SRN, frozen downstream analyses, and final bridge."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ftt_srn_causal_representation.config import (  # noqa: E402
    final_config as causal_final, smoke_config as causal_smoke,
)
from ftt_srn_causal_representation.pipeline import run_causal_pipeline  # noqa: E402
from ftt_srn_context_sequence_interaction.config import (  # noqa: E402
    final_config as interaction_final, smoke_config as interaction_smoke,
)
from ftt_srn_context_sequence_interaction.pipeline import run_interaction_pipeline  # noqa: E402
from human_configured_state.config import final_config as human_final, smoke_config as human_smoke  # noqa: E402
from human_configured_state.pipeline import run_human_configured_state_pipeline  # noqa: E402
from human_low_rank_interpretation.config import (  # noqa: E402
    final_config as lowrank_final, smoke_config as lowrank_smoke,
)
from human_low_rank_interpretation.pipeline import run_low_rank_interpretation_pipeline  # noqa: E402
from state_space_bridge.config import final_config as bridge_final, smoke_config as bridge_smoke  # noqa: E402
from state_space_bridge.pipeline import run_state_space_bridge  # noqa: E402
from state_space_bridge.srn import ScientificGateError, run_corrected_srn  # noqa: E402


def _status(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] FULL-BRIDGE | {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete corrected state-space bridge.")
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument(
        "--stage", choices=("corrected-srn", "downstream", "assemble", "all"), default="all"
    )
    parser.add_argument("--combined", default="data/mixedBreak_clean.csv")
    parser.add_argument("--novel", default="data/threeContextsBreak_clean.csv")
    parser.add_argument("--output-prefix", default="outputs/state_space_bridge")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = args.profile
    prefix = args.output_prefix.rstrip("_")
    paths = {
        "srn": f"{prefix}_corrected_srn_{suffix}",
        "causal": f"{prefix}_causal_v22_{suffix}",
        "interaction": f"{prefix}_context_v23_{suffix}",
        "v3": f"{prefix}_human_v3_{suffix}",
        "v31": f"{prefix}_human_v31_{suffix}",
        "bridge": f"{prefix}_{suffix}",
    }
    progress = not args.quiet
    plots = not args.no_plots

    if args.stage in {"corrected-srn", "all"}:
        _status("Stage 1/6: corrected High-scale=2.0 SRN rerun and sanity gate")
        run_corrected_srn(paths["srn"], profile=args.profile, show_progress=progress)
    if args.stage in {"downstream", "all"}:
        _status("Stage 2/6: corrected frozen causal analysis")
        cfg = causal_smoke() if args.profile == "smoke" else causal_final()
        run_causal_pipeline(paths["srn"], paths["causal"], config=replace(
            cfg, make_plots=plots, show_progress=progress, profile=args.profile
        ))
        _status("Stage 3/6: corrected context-by-sequence analysis")
        cfg = interaction_smoke() if args.profile == "smoke" else interaction_final()
        run_interaction_pipeline(paths["srn"], paths["interaction"], config=replace(
            cfg, make_plots=plots, show_progress=progress, profile=args.profile
        ))
        _status("Stage 4/6: human retrieval ladder using the corrected frozen SRN")
        cfg = human_smoke() if args.profile == "smoke" else human_final()
        run_human_configured_state_pipeline(
            args.combined, args.novel, paths["srn"], paths["v3"],
            config=replace(cfg, make_plots=plots, show_progress=progress, profile=args.profile),
        )
        _status("Stage 5/6: data-driven human low-rank interpretation")
        cfg = lowrank_smoke() if args.profile == "smoke" else lowrank_final()
        run_low_rank_interpretation_pipeline(
            paths["v3"], args.combined, args.novel, paths["v31"],
            config=replace(cfg, make_plots=plots, show_progress=progress, profile=args.profile),
        )
    if args.stage in {"assemble", "all"}:
        _status("Stage 6/6: state-space fits, recovery, retrieval assembly, and bridge figures")
        cfg = bridge_smoke() if args.profile == "smoke" else bridge_final()
        run_state_space_bridge(
            args.combined, args.novel, paths["srn"], paths["v3"], paths["v31"],
            paths["bridge"], causal_dir=paths["causal"], interaction_dir=paths["interaction"],
            config=replace(cfg, make_plots=plots, show_progress=progress, profile=args.profile),
        )
    _status("Requested stages complete")


if __name__ == "__main__":
    try:
        main()
    except ScientificGateError as error:
        raise SystemExit(f"FULL-BRIDGE STOPPED: {error}") from None
