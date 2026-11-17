#!/usr/bin/env python3
"""Run the familiar-only human state-space adjudication."""

from pathlib import Path
import argparse
import os
import sys

# Each process performs many small NumPy/SciPy operations. Let process-level
# parallelism own the cores instead of multiplying BLAS threads inside workers.
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable,"1")

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from human_state_space_adjudication import load_config,run_pipeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=ROOT/"data/mixedBreak_clean.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/human_state_space_adjudication")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/human_state_space_adjudication.json")
    parser.add_argument("--profile",choices=["smoke","final"],default="final")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--workers",type=int,default=None,help="Parallel worker processes; default auto (up to 6)")
    args=parser.parse_args()
    config=load_config(args.config,profile=args.profile)
    if args.no_plots or args.workers is not None:
        from dataclasses import replace
        config=replace(config,make_plots=False) if args.no_plots else config
        config=replace(config,workers=args.workers) if args.workers is not None else config
    run_pipeline(args.data,args.output,config=config)


if __name__=="__main__": main()
