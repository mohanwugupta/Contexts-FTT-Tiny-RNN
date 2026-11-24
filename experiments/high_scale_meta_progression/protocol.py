"""Fixed design and source-preserving state continuation contracts."""
from .common import *
from experiments.linear_scale_sweep.protocol import scaled

def config(path=None):
 c=json.loads(Path(path or ROOT/'configs/high_scale_meta_progression.yaml').read_text())
 assert c['scales']==[4,5,6] and c['seeds']==list(range(6)) and c['meta_episodes']==1920 and c['meta_doses']==[240,960,1920] and c['arms']==['T','B','O1','O2'] and c['target_final']==990 and c['target_intermediate']==330
 return c

def scaled_initial(seed,scale):
 if seed not in range(6) or scale not in [4,5,6]:raise ValueError('Unregistered initialization')
 p=HIGH/f'initial/s{seed:02d}_High.pt';d=copy.deepcopy(torch.load(p,weights_only=False)['model'])
 for k in ['recurrent.weight_ih_l0','sequence_head.weight']:d[k].mul_(scale/2)
 return d

def restore_target(s):
 m=source_model(s['model']);p=core.policy();p.load_state_dict(copy.deepcopy(s['policy']));o=core.optimizer(m,p);o.load_state_dict(copy.deepcopy(s['optimizer']))
 steps=[float(v['step']) for v in o.state.values()]
 if s['update']>0 and (len(steps)!=8 or any(v!=s['update'] for v in steps)):raise ValueError('Missing or inconsistent complete target Adam state')
 return m,p,o

def epsilon(k):
 if k not in range(1,1921):raise ValueError('Fixed Reptile budget')
 return .1+(.01-.1)*(k-1)/239 if k<=240 else .01

def validate_fork(actual,parent):
 if set(actual)!=set(parent) or not all(torch.equal(actual[k],v) for k,v in parent.items()):raise ValueError('Fork parent differs')

def validate_resume(cp,registry_sha,code):
 if cp['registry_sha']!=registry_sha or cp['code']!=code or cp['next_episode']!=cp['episode']+1:raise ValueError('Unverified resume identity')

def target_cue(update):
 if update not in range(1,991):raise ValueError('Outside registered990 target updates')
 return 'A' if ((update-1)//10)%2==0 else 'B'
