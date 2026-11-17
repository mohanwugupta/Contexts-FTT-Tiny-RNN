import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cms-mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/cms-cache')
import argparse,torch
from .config import DEFAULT
from .io import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['d0','d1','d2','d3_smoke','d3','d4']);p.add_argument('--output',default=str(DEFAULT));a=p.parse_args()
    torch.set_num_threads(1);run(a.stage,a.output)
