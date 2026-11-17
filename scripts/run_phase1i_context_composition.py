#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from phase1i_context_composition import run_pipeline
from phase1i_context_composition.config import load_config
def main():
    parser=argparse.ArgumentParser(description="Run Phase 1I compositional context-transformation audit")
    parser.add_argument("--config");parser.add_argument("--output");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();run_pipeline(load_config(args.config,repo_dir=ROOT,output=args.output,profile="smoke" if args.smoke else "confirmatory"))
if __name__=="__main__":main()
