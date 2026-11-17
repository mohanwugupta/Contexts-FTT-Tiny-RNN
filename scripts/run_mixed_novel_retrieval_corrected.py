#!/usr/bin/env python3
from pathlib import Path
import argparse,os,sys

for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable,"1")
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from mixed_novel_retrieval_corrected import load_config,run_pipeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--combined",type=Path,default=ROOT/"data/mixedBreak_clean.csv")
    parser.add_argument("--novel",type=Path,default=ROOT/"data/threeContextsBreak_clean.csv")
    parser.add_argument("--familiar-dir",type=Path,default=ROOT/"outputs/human_state_space_adjudication")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/mixed_novel_retrieval_corrected")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/mixed_novel_retrieval_corrected.json")
    parser.add_argument("--profile",choices=["smoke","final"],default="final")
    parser.add_argument("--workers",type=int,default=None)
    parser.add_argument("--no-plots",action="store_true",
        help="Skip the older optional figures; mandatory absolute-fit figures A-J are always rendered.")
    args=parser.parse_args(); config=load_config(args.config,profile=args.profile)
    if args.workers is not None or args.no_plots:
        from dataclasses import replace
        if args.workers is not None: config=replace(config,workers=args.workers)
        if args.no_plots: config=replace(config,make_plots=False)
    run_pipeline(args.combined,args.novel,args.familiar_dir,args.output,config=config)


if __name__=="__main__": main()
