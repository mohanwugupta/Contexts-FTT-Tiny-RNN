import argparse,json
from research_search.common import clean
from .controller import initialize,prepare_proposal,prepare_critic,call,compute,OUT
from .reporting import export


def main():
    p=argparse.ArgumentParser(description='Controlled pilot v2; each agent call needs individual approval')
    p.add_argument('command',choices=['init','prepare-proposal','proposal','compute','prepare-critic','critic','status','report','pause','stop']);p.add_argument('--approve')
    args=p.parse_args();s=initialize()
    try:
        if args.command=='init':r=s.state()
        elif args.command=='prepare-proposal':r=prepare_proposal(s)
        elif args.command=='prepare-critic':r=prepare_critic(s)
        elif args.command in ['proposal','critic']:r=call(s,'pilot2-'+args.command,args.approve)
        elif args.command=='compute':r=compute(s)
        elif args.command=='report':r=export(s,OUT)
        elif args.command in ['pause','stop']:s.control('paused' if args.command=='pause' else 'stopped');r=s.state()
        else:r=s.state()|dict(phase=s.get('phase'),jobs=[dict(x) for x in s.db.execute('SELECT id,status,error FROM jobs')])
        print(json.dumps(clean(r),indent=2))
    finally:s.close()

if __name__=='__main__':main()
