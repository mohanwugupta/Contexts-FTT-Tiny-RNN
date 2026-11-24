import argparse,json
from .probe import prepare,run_probe


def main():
    p=argparse.ArgumentParser(description='Versioned isolated backend repair: prepare or manually approve one fixed ping')
    p.add_argument('command',choices=['prepare','run']);p.add_argument('--approve')
    a=p.parse_args()
    try:r=prepare() if a.command=='prepare' else run_probe(a.approve)
    except (RuntimeError,PermissionError,ValueError) as ex:
        print(json.dumps(dict(status='BLOCKED',reason=str(ex),live_test_run=False),indent=2));raise SystemExit(2)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
