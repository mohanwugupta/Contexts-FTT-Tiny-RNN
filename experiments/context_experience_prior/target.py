"""Registered A10/B10 follow-up, exact P0 reuse, frozen six-cue audits."""
from .core import *
from .measures import measure_model
from .export_correction import training_record

def lift_P0_state(state,c):
 q=copy.deepcopy(state);q['model']['recurrent.weight_ih_l0'][:,6].copy_(c)
 for v in q['optimizer']['state'].values():
  if 'exp_avg' in v and tuple(v['exp_avg'].shape)==(8,7):
   assert torch.count_nonzero(v['exp_avg'][:,6])==torch.count_nonzero(v['exp_avg_sq'][:,6])==0
 return q

def archived_predictions():
 keep=[]
 for f in pd.read_csv(SOURCE/'part_b/tables/position_predictions.csv',chunksize=100000):
  f=f[f.arm.isin(['NI','NC'])&f.cue.ne('C')];keep.append(f)
 f=pd.concat(keep,ignore_index=True);return {key:g.sort_values('position') for key,g in f.groupby(['seed','arm','update','cue'])}

def saved_obs(f):
 return dict(logits=f[[f'logit_{i}' for i in range(4)]].to_numpy(),probabilities=f[[f'probability_{i}' for i in range(4)]].to_numpy(),hidden=f[[f'hidden_{i}' for i in range(8)]].to_numpy(),ce=f.ce.to_numpy(),accuracy=f.correct.to_numpy(bool),margin=f.margin.to_numpy())

def csv_gz(name,rows):
 p=OUT/'tables'/f'{name}.csv.gz';pd.DataFrame(rows).to_csv(p,index=False,compression='gzip',mode='a',header=not p.exists())

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());hashes(ROOT,seal['inputs']);hashes(OUT,seal['registry_hashes']);meta=json.loads((OUT/'audit/meta_complete.json').read_text());assert meta['inner_updates_executed']==384000 and meta['outer_updates_executed']==19200
 if (OUT/'audit/target_started.json').exists():raise RuntimeError('Target run already started')
 dump(OUT/'audit/target_started.json',dict(code=codes(),meta_complete_sha=sha(OUT/'audit/meta_complete.json'),selection='final240only',new_target_updates=26400,reused_target_updates=13200,diagnostic_updates=1440))
 archived=archived_predictions();original=pd.read_csv(SOURCE/'part_b/tables/training.csv');reg=[];actual=diagsteps=0
 for seed in range(20):
  bases=base_pair(seed)
  for org in ORG:
   for prior in PRIORS:
    condition=prior+'-'+org;arm='N'+org;m=copy.deepcopy(bases[org]);p=policy();o=optimizer(m,p);parent=None;prov='new registered final240 target'
    if prior=='P0':
     parent=torch.load(SOURCE/f'part_b/checkpoints/seed_{seed}_{arm}.pt',weights_only=False);prov='exact archived NI/NC lift; five cue branches reused; C evaluated'
    else:
     path=OUT/f'meta/seed_{seed}_{prior}_{org}.pt';b=torch.load(path,weights_only=False);assert len(b['records'])==241 and b['records'][-1]['status']=='completed';m.load_state_dict(b['records'][-1]['meta'])
    c=m.recurrent.weight_ih_l0[:,6].detach().clone();assert torch.count_nonzero(c)>0 and not o.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values())
    states=[];evals=[];positions=[];geom=[];controls=[];decomp=[];training=[];diagnostics=[];failed=None
    for k in range(331):
     if parent:
      st=lift_P0_state(parent['states'][k],c);m.load_state_dict(st['model']);p.load_state_dict(st['policy']);o.load_state_dict(st['optimizer']);assert st['status']=='completed'
     assert torch.equal(c,m.recurrent.weight_ih_l0[:,6]);ids=dict(seed=seed,organization=org,prior=prior,condition=condition,update=k)
     if failed:raise RuntimeError('Nonfinite target requires explicit unavailable-state recording: '+str(ids))
     states.append(dict(update=k,status='completed',model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),source=prov))
     obs={cue:(saved_obs(archived[seed,arm,k,cue]) if parent and cue!='C' else trace(m,cue)) for cue in CUES}
     for cue,d in obs.items():
      evals.append(dict(**ids,cue=cue,status='completed',provenance=prov if parent else 'new frozen natural evaluation',ce=float(d['ce'].mean()),accuracy=float(d['accuracy'].mean()),above_chance=bool(d['ce'].mean()>np.log(4))))
      for t in range(7):
       q=dict(**ids,cue=cue,position=t+1,token=TARGET_PATH[t],target=int(TARGET[t]),ce=d['ce'][t],correct=bool(d['accuracy'][t]),margin=d['margin'][t]);q.update({f'logit_{j}':d['logits'][t,j] for j in range(4)});q.update({f'probability_{j}':d['probabilities'][t,j] for j in range(4)});q.update({f'hidden_{j}':d['hidden'][t,j] for j in range(8)});positions.append(q)
     g,co,dc=measure_model(m,obs);geom.append(dict(**ids,**g));controls.extend(dict(**ids,**x) for x in co);decomp.extend(dict(**ids,**x) for x in dc)
     if k in DIAG:
      q=diagnostic(m,p,o);diagsteps+=2;diagnostics.extend(dict(**ids,**x) for x in q)
     if k<330:
      cue=training_cue(k+1)
      if parent:
       q=original[(original.seed==seed)&(original.arm==arm)&(original['update']==k+1)].iloc[0].to_dict();q={n:v for n,v in q.items() if n not in ['seed','arm','update','training_cue']};q['new_optimizer_step_executed']=False
      else:
       q=step(m,p,o,cue);actual+=int(q['optimizer_step_executed']);q['new_optimizer_step_executed']=q['optimizer_step_executed'];failed=None if q['status']=='completed' else q['status']
      assert abs(float(q['ce_before'])-obs[cue]['ce'].mean())<1e-4+1e-5*abs(q['ce_before']);training.append(training_record(dict(seed=seed,organization=org,prior=prior,condition=condition,update=k+1,training_cue=cue,archive_reused=bool(parent)),q))
    path=OUT/f'target/seed_{seed}_{prior}_{org}.pt';torch.save(dict(seed=seed,organization=org,prior=prior,condition=condition,states=states,C_constant=True,initial_C=c,target_policy_reset=True,target_optimizer_reset=True,source=prov),path);reg.append(dict(seed=seed,organization=org,prior=prior,condition=condition,path=str(path.relative_to(OUT)),sha256=sha(path),states=331,status='completed',C_constant=True))
    for name,rows in [('evaluations',evals),('geometry',geom),('training',training),('diagnostics',diagnostics)]:save_csv(name,rows)
    for name,rows in [('position_predictions',positions),('mathematical_checks',controls),('decomposition',decomp)]:csv_gz(name,rows)
    print(f'Target seed{seed:02d} {condition}:331 states, C constant',flush=True)
 save_csv('target_registry',reg);hashes(ROOT,seal['inputs']);dump(OUT/'audit/target_complete.json',dict(code=codes(),new_target_steps=actual,diagnostic_steps=diagsteps,reused_target_updates=13200,all120_runs_retained=True,states=39720));assert actual==26400 and diagsteps==1440;append(f'## Target phase complete\n{actual} newupdates,13200 exactP0 reusedupdates,{diagsteps} cloneddiagnosticupdates.120×331 states,six naturalcues,everyposition/logit retained. C unchangedallstates;parenthashes unchanged.')
if __name__=='__main__':main()
