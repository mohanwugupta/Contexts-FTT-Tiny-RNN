#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from output_relevant_compositionality import ExperimentConfig,run_pipeline


def main():
    parser=argparse.ArgumentParser(description="Two-phase output-relevant compositionality experiment")
    parser.add_argument("--output",type=Path,default=Path("outputs/output_relevant_compositionality"))
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(20)))
    parser.add_argument("--selection-seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--no-plots",action="store_true");args=parser.parse_args()
    run_pipeline(ExperimentConfig(output_dir=args.output,seeds=tuple(args.seeds),selection_seeds=tuple(args.selection_seeds),
                                  make_plots=not args.no_plots,profile="final" if len(args.seeds)>=10 else "custom"))


if __name__=="__main__":main()
