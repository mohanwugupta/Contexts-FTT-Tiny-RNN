import argparse,json
from .control import prepare,status,initialize


def main():
    p=argparse.ArgumentParser(description='Prepare/inspect controlled pilot v1. No agent or optimizer calls.')
    p.add_argument('command',choices=['prepare','status','init']);a=p.parse_args()
    try:r={'prepare':prepare,'status':status,'init':initialize}[a.command]()
    except RuntimeError as ex:
        print(json.dumps(dict(status='BLOCKED',reason=str(ex),agent_dispatched=False,campaign_created=False),indent=2));raise SystemExit(2)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
