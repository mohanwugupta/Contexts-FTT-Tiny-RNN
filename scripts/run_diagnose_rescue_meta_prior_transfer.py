#!/usr/bin/env python3
"""Run the hard-gated meta-prior diagnosis/rescue program."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from diagnose_rescue_meta_prior_transfer import load_config,run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/diagnose_rescue_meta_prior_transfer.json")
    parser.add_argument("--output",default=None)
    parser.add_argument("--smoke",action="store_true",help="Engineering smoke: one H8 replay seed, three native-test seeds, tiny conditional Stage C.")
    parser.add_argument("--force-stage-b-pass-for-integration-test",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(); config=load_config(args.config,smoke=args.smoke,output=args.output,force_stage_b_pass=args.force_stage_b_pass_for_integration_test); run_pipeline(config)


if __name__=="__main__": main()
