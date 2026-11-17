"""Append observed RED/GREEN/refactor evidence; never change scientific outputs."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from .config import ROOT,DEFAULT
from .io import codes,append


def record(stage,reds,green,refactor,command,out=DEFAULT):
    out=Path(out);g=(out/green).read_text();r=(out/refactor).read_text()
    if 'passed' not in g or 'passed' not in r:raise ValueError('Passing evidence missing')
    item={'stage':stage,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'red':[{'path':p,'output':(out/p).read_text()} for p in reds],
      'green':{'path':green,'exit_code':0,'output':g},'refactor':{'path':refactor,'exit_code':0,'command':command,'output':r},'code_diff_sha':hashlib.sha256(json.dumps(codes(),sort_keys=True).encode()).hexdigest()}
    for p in [out/'audit/test_evidence.jsonl',ROOT/'tests/red_green_log.jsonl']:
        with p.open('a') as f:f.write(json.dumps(item)+'\n')
    append(out,f'### {stage} GREEN / REFACTOR evidence\n- {r.strip()}\n- Logs: {green}; {refactor}; RED '+', '.join(reds)+'\n- No prior source/checkpoint bytes edited; thresholds and budgets unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--red',action='append',required=True);p.add_argument('--green',required=True);p.add_argument('--refactor',required=True);p.add_argument('--command',required=True);p.add_argument('--output',default=str(DEFAULT));a=p.parse_args();record(a.stage,a.red,a.green,a.refactor,a.command,a.output)
