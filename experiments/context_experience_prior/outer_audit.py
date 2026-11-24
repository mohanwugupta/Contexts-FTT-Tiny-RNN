"""Read-only every-episode outer interpolation and initialization chain audit."""
from .core import *

def verify_outer(before,adapted,after,epsilon):
 assert before.keys()==adapted.keys()==after.keys()
 for n in before:assert torch.equal(after[n],before[n]+epsilon*(adapted[n]-before[n]))

def main():
 f=pd.read_csv(OUT/'tables/meta_episodes.csv');assert len(f)==19200 and f.status.eq('completed').all() and f.optimizer_empty_at_start.all() and f.policy_reset.all() and f.inner_steps_executed.eq(20).all();assert (f.Wh_norm_before>0).all();r=pd.read_csv(OUT/'tables/meta_registry.csv');assert len(r)==80;checked=0
 for row in r.itertuples():
  path=OUT/row.path;assert sha(path)==row.sha256;b=torch.load(path,weights_only=False);assert b['required_milestones']==[0,40,80,120,180,240] and len(b['records'])==241;last=b['records'][0]['meta']
  for i,st in enumerate(b['records'][1:],1):
   assert st['episode']==i and st['status']=='completed';eps=.1+(.01-.1)*(i-1)/239;assert st['epsilon']==eps and all(torch.equal(v,st['before'][n]) for n,v in last.items());verify_outer(st['before'],st['adapted'],st['meta'],eps);last=st['meta'];checked+=1
 assert checked==19200;dump(OUT/'audit/outer_verification.json',dict(passed=True,interpolations=checked,all80_episode_chains=True,required_milestones=True,reset_metadata=True,no_updates=True));append('## Outer audit GREEN\nAll19200before/adapted/afterReptilemoves and80meta-initializationchains matchregisterednativeformulaexactly;requiredmilestones/resetmetadata/384000innersteps verified. No training/evaluationinthischeck.')
if __name__=='__main__':main()
