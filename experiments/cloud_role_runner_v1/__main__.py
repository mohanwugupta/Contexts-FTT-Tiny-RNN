import argparse,json
from pathlib import Path
from .runtime import prepare
from .contract import save
from .docker import launch,SOCKETS

def main():
 p=argparse.ArgumentParser(description='Linux role transport: no scientific worker or automatic role loop')
 p.add_argument('action',choices=['prepare','fixture','approval','call']);p.add_argument('--out',type=Path);p.add_argument('--runtime',type=Path);p.add_argument('--image');p.add_argument('--codex');p.add_argument('--ca');p.add_argument('--socket',choices=SOCKETS,default=SOCKETS[0]);p.add_argument('--synthetic',action='store_true');p.add_argument('--campaign',type=Path);p.add_argument('--ticket');p.add_argument('--approval');p.add_argument('--auth-file',type=Path)
 a=p.parse_args()
 if a.action=='prepare':
  if not a.out or not a.image:p.error('prepare requires --out and accepted --image')
  result=prepare(a.out,a.image,a.codex,a.ca,a.socket,a.synthetic)
 elif a.action=='fixture':
  if not a.runtime or not a.out:p.error('fixture requires --runtime and new --out')
  r=json.loads(a.runtime.read_text());result=launch(r,a.out,'fixture',{})
  result={k:v for k,v in result.items() if k not in ['stdout','stderr']};result.update(new_agent_calls=0,new_optimizer_steps=0,live_usage_verified=False)
 else:
  if not all([a.runtime,a.campaign,a.ticket]) or (a.action=='call' and not a.approval):p.error('existing runtime/campaign/ticket and exact approval SHA required for call')
  if not (a.campaign/'campaign.sqlite').is_file():p.error('Existing rehearsal ledger required; no new clock or budget is initialized')
  from .agent import invoke,prepare_approval,guard_campaign
  guard_campaign(a.campaign)
  from research_search.orchestration.store import Store
  s=Store(a.campaign,json.loads((a.campaign/'budgets.json').read_text()))
  try:result=prepare_approval(s,a.ticket,json.loads(a.runtime.read_text())) if a.action=='approval' else invoke(s,a.ticket,a.approval,json.loads(a.runtime.read_text()),a.auth_file)
  finally:s.close()
 print(json.dumps(result,indent=2));return 0 if result.get('passed',a.action in ['prepare','approval']) else 3
if __name__=='__main__':raise SystemExit(main())
