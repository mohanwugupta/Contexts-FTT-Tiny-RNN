#!/usr/bin/env python3
"""Run the gated Stage 0--5 structured-context ecology."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "structured-ecology-mpl")
)

from structured_context_ecology import (  # noqa: E402
    ecology_config, run_structured_context_ecology, smoke_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered structured-context stages in order. Later stages "
            "are automatically blocked when a prior gate fails."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "ecology"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ecology_config() if args.profile == "ecology" else smoke_config()
    config = replace(
        config, make_plots=not args.no_plots, show_progress=not args.quiet
    )
    output = args.output or ROOT / f"outputs/structured_context_ecology_{args.profile}"
    tables = run_structured_context_ecology(output, config=config)
    print("\nStructured-context stage summary:", flush=True)
    print(tables["structured_context_stage_summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
