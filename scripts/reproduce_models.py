#!/usr/bin/env python3
"""Explicit optional historical training reproduction, never called by paper evaluation."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper_freeze_v2.json');p.add_argument('--mode',choices=['reproduce'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Training reproduction requires a fresh output directory')
    # Original objectives/cues remain unchanged; the new Novel registry is not imported.
    subprocess.run([sys.executable,str(ROOT/'scripts/run_confirmatory_srn.py'),'--profile','confirmatory','--output',str(out/'canonical')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/run_phase1l_contrast_bottleneck_trajectory_closure.py'),'--output',str(out/'constructive')],cwd=ROOT,check=True)
