#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))

from srn_nonlinearity_compositionality import MechanismConfig,run_pipeline


def main():
    parser=argparse.ArgumentParser(description="Frozen SRN nonlinearity/output-compositionality analysis")
    parser.add_argument("--checkpoints",type=Path,default=Path("outputs/confirmatory_srn_final/confirmatory_checkpoints"))
    parser.add_argument("--reproduction",type=Path,default=Path("outputs/srn_mixed_novel_diagnostics"))
    parser.add_argument("--output",type=Path,default=Path("outputs/srn_nonlinearity_compositionality"))
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(20)))
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    run_pipeline(MechanismConfig(checkpoint_dir=args.checkpoints,reproduction_dir=args.reproduction,
                                 output_dir=args.output,seeds=tuple(args.seeds),make_plots=not args.no_plots,
                                 profile="final" if len(args.seeds)==20 else "custom"))


if __name__=="__main__": main()
