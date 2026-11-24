"""Read-only complete-grid, tensor-state, prediction and analysis verification."""
import argparse
from .core import *
from .analysis import validate_grid,block_table,switch_table,qualify

def hashes(root,m):
 for name,h in m.items():
  if sha(root/name)!=h:raise ValueError('Hash mismatch: '+name)

def verify(sealed=True):
 read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip')
 run=json.loads((OUT/'audit/training_complete.json').read_text());hashes(ROOT,run['inputs']);hashes(ROOT,run['code']);f=read('frozen_evaluations');validate_grid(f);t=read('training_trajectories');assert len(t)==13200 and not t.duplicated(['seed','arm','update']).any();assert t.status.eq('completed').all() and t.optimizer_step_executed.all();assert run['scientific_updates_executed']==13200
 assert all(r.training_cue==training_cue(r.update) for r in t.itertuples())
 lookup=f.set_index(['seed','arm','update','cue']).ce
 pre=np.array([lookup.loc[(r.seed,r.arm,r.update-1,r.training_cue)] for r in t.itertuples()]);np.testing.assert_allclose(t.ce_before,pre,atol=2e-5,rtol=1e-6)
 registry=read('checkpoint_registry');assert len(registry)==40;states=0
 for r in registry.itertuples():
  assert sha(OUT/r.path)==r.sha256;p=torch.load(OUT/r.path,map_location='cpu',weights_only=False);assert p['seed']==r.seed and p['arm']==r.arm and len(p['states'])==331;states+=331
  for k,s in enumerate(p['states']):
   assert s['update']==k and s['status']=='completed';model=s['model'];assert all(torch.isfinite(v).all() for v in model.values())
   if r.arm=='Centered':assert torch.count_nonzero(model['g'])==0
   paramcount=10 if r.arm=='Standard' else 9
   assert len(s['optimizer']['param_groups'][0]['params'])==paramcount
   if k:
    assert len(s['optimizer']['state'])==paramcount
    assert all(int(v['step'])==k for v in s['optimizer']['state'].values())
  # Initial tensors match source normalized draw, g=0; all other paired tensors tested in core suite.
  m=pair(r.seed)[r.arm]
  for n,v in m.state_dict().items():assert torch.equal(v,p['states'][0]['model'][n])
 pos=read('position_predictions');assert len(pos)==370720 and not pos.duplicated(['seed','arm','update','cue','position']).any();assert set(pos.position)==set(range(1,8));target=TARGET[pos.position.to_numpy()-1];assert np.array_equal(target,pos.target)
 z=pos[[f'logit_{j}' for j in range(4)]].to_numpy();v=z-z.max(1)[:,None];ce=np.log(np.exp(v).sum(1))-v[np.arange(len(z)),target];prob=np.exp(v);prob/=prob.sum(1)[:,None];np.testing.assert_allclose(ce,pos.ce,atol=1e-12);np.testing.assert_allclose(prob,pos[[f'probability_{j}' for j in range(4)]],atol=1e-12);assert np.array_equal(z.argmax(1)==target,pos.correct)
 agg=pos.groupby(['seed','arm','update','cue']).agg(ce=('ce','mean'),accuracy=('correct','mean'));primary=f.set_index(['seed','arm','update','cue']).sort_index();np.testing.assert_allclose(agg[['ce','accuracy']],primary[['ce','accuracy']],atol=1e-12)
 cm=pos[(pos.arm=='Centered')&pos.cue.isin(['C','Mixed'])].sort_values(['seed','update','position','cue']);fields=[f'{name}_{j}' for name,size in [('logit',4),('hidden',8),('probability',4)] for j in range(size)];a=cm[cm.cue=='C'][fields].to_numpy();b=cm[cm.cue=='Mixed'][fields].to_numpy();np.testing.assert_array_equal(a,b)
 checks=read('mathematical_checks');assert len(checks)==92680 and checks.passed.all();assert (checks.double_logit_midpoint_error<=checks.double_tolerance).all() and (checks.double_hidden_midpoint_error<=checks.double_tolerance).all();assert (checks.double_ce_bound_gap<=2*checks.double_tolerance).all();assert checks[checks.arm=='Centered'].C_Mixed_logit_error.eq(0).all()
 for name,computed in [('block_transfer',block_table(f)),('switch_costs',switch_table(f))]:pd.testing.assert_frame_equal(read(name),computed,check_exact=False,atol=1e-11,rtol=1e-9)
 q=qualify(f);end=read('generalization');qq=end[['seed','arm','update','strong_familiar']].drop_duplicates();pd.testing.assert_frame_equal(qq.sort_values(['seed','arm','update']).reset_index(drop=True),q[q['update'].isin(PROBES)][['seed','arm','update','strong_familiar']].sort_values(['seed','arm','update']).reset_index(drop=True))
 for r in read('paired_intervals').itertuples():
  x=read('paired_seed_effects');x=x[(x['update']==r.update)&(x.cue==r.cue)&(x.metric==r.metric)];np.testing.assert_allclose(interval(x.Centered_minus_Standard),[r.mean,r.ci_low,r.ci_high],atol=1e-12)
 if sealed:
  m=json.loads((OUT/'manifest.json').read_text());hashes(OUT,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,scientific_updates=len(t),states=states,evaluation_cells=len(f),positions=len(pos),mathematical_position_checks=len(checks),training_executed_by_verifier=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--unsealed',action='store_true');p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify(not a.unsealed)
 if a.regenerate:
  if a.regenerate.exists() and list(a.regenerate.iterdir()):raise ValueError('Regeneration destination must be empty')
  from .figures import generate
  names=generate(OUT,a.regenerate)
  for n in names:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_figure_files']=len(names)
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
