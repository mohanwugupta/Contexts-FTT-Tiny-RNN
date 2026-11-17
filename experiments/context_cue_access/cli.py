import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/context-cue-access-mpl')
import argparse
import torch
from .config import DEFAULT
from .io import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['s0','s1','s2','s3','s4']);p.add_argument('--output',default=str(DEFAULT));p.add_argument('--eval-only',action='store_true');a=p.parse_args()
    if not a.eval_only:raise ValueError('--eval-only mandatory; cue inputs only are optimized')
    torch.set_num_threads(1);run(a.stage,a.output)
