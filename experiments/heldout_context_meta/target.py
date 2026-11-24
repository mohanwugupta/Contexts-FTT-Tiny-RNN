"""One fixed dose adaptation; main states untouched by restored-Adam probes."""
from .common import *
from experiments.context_experience_prior import core
from experiments.context_experience_prior.measures import measure_model
from experiments.meta_learning_scaling.target import validate_training_records
from .mechanisms import extra_measures

def validate_training_records(t,e):
 expected=np.where(((t['update'].to_numpy()-1)//10)%2==0,'A','B');assert np.array_equal(t.training_cue,expected)
 q=e[['update','cue','ce']].rename(columns={'cue':'training_cue'}).copy();q['update']+=1;z=t.merge(q,on=['update','training_cue'],validate='one_to_one');assert len(z)==len(t);assert (abs(z.ce_before-z.ce)<=1e-4+1e-5*abs(z.ce_before)).all();np.testing.assert_allclose(z.total_loss,z.ce_before+.2*z.execution_loss,atol=2e-6,rtol=1e-6)

def target_run(job):
 initialize_process();seed,prior,org,arm,dose=job;run=f's{seed:02d}_{prior}_{org}_{arm}_d{dose}';dest=OUT/'target'/run
 if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
 dest.mkdir();start=time.monotonic();dump(dest/'started.json',dict(seed=seed,prior=prior,organization=org,arm=arm,dose=dose,code=codes(),time=time.time()));initial_state,source=initial(seed,prior,org,arm,dose);m=source_model(initial_state) if initial_state is not None else None;p=core.policy();o=core.optimizer(m,p) if m is not None else None
 if m is not None:assert not o.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values());c=m.recurrent.weight_ih_l0[:,6].detach().clone();assert torch.count_nonzero(c)>0
 states=[];evals=[];positions=[];geom=[];checks=[];decomp=[];training=[];diagnostics=[];failed=None if initial_state is not None else 'blocked_nonfinite_meta';n=dn=0
 for k in range(331):
  if failed:
   states.append(dict(update=k,status=failed,model=None,policy=None,optimizer=None))
   for cue in CUES:
    evals.append(dict(update=k,cue=cue,status=failed,ce=np.nan,accuracy=np.nan,above_chance=None))
    for t in range(7):positions.append(dict(update=k,cue=cue,status=failed,position=t+1,token=core.TARGET_PATH[t],target=int(core.TARGET[t]),ce=np.nan,correct=None,margin=np.nan,**{f'logit_{j}':np.nan for j in range(4)},**{f'probability_{j}':np.nan for j in range(4)},**{f'hidden_{j}':np.nan for j in range(8)}))
   if k in DIAG:
    for a in ['A','B']:
     for b in ['A','B']:diagnostics.append(dict(update=k,train_context=a,evaluation_context=b,improvement=np.nan,loss_before=np.nan,loss_after=np.nan,diagnostic_status=failed,diagnostic_update_executed=False,main_unchanged=True,optimizer_restored=False,main_start_hash='',clone_start_hash=''))
   if k<330:training.append(dict(update=k+1,training_cue=core.training_cue(k+1),status=failed,optimizer_step_executed=False,ce_before=np.nan,accuracy_before=np.nan,execution_loss=np.nan,total_loss=np.nan,gradient_norm=np.nan,hidden_absmax=np.nan))
   continue
  assert torch.equal(c,m.recurrent.weight_ih_l0[:,6]);states.append(dict(update=k,status='completed',model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict())))
  obs={cue:core.trace(m,cue) for cue in CUES}
  for cue,x in obs.items():
   evals.append(dict(update=k,cue=cue,status='completed',ce=float(x['ce'].mean()),accuracy=float(x['accuracy'].mean()),above_chance=bool(x['ce'].mean()>np.log(4))))
   for t in range(7):positions.append(dict(update=k,cue=cue,status='completed',position=t+1,token=core.TARGET_PATH[t],target=int(core.TARGET[t]),ce=x['ce'][t],correct=bool(x['accuracy'][t]),margin=x['margin'][t],**{f'logit_{j}':x['logits'][t,j] for j in range(4)},**{f'probability_{j}':x['probabilities'][t,j] for j in range(4)},**{f'hidden_{j}':x['hidden'][t,j] for j in range(8)}))
  g,cc,dc=measure_model(m,obs);eg,ed=extra_measures(m,obs,cc);g.update(eg)
  for a,bb in zip(dc,ed):a.update(bb)
  geom.append(dict(update=k,**g));checks.extend(dict(update=k,status='completed',**v) for v in cc);decomp.extend(dict(update=k,**v) for v in dc)
  if k in DIAG:
   rr=core.diagnostic(m,p,o);dn+=sum(v['diagnostic_update_executed'] for v in rr)/2;diagnostics.extend(dict(update=k,**v) for v in rr)
  if k<330:
   q=core.step(m,p,o,core.training_cue(k+1));n+=int(q['optimizer_step_executed']);training.append(dict(update=k+1,training_cue=core.training_cue(k+1),**q));failed=None if q['status']=='completed' else q['status']
 t=pd.DataFrame(training);e=pd.DataFrame(evals);finite=t[t.status=='completed'];validate_training_records(finite,e)
 torch.save(dict(seed=seed,prior=prior,organization=org,arm=arm,dose=dose,states=states,source=str(source.relative_to(ROOT)),source_sha=sha(source),C_initial=c if initial_state is not None else None),dest/'states.pt')
 for name,rows in [('evaluations',evals),('geometry',geom),('training',training),('diagnostics',diagnostics)]:write_frame(dest/f'{name}.csv',rows)
 for name,rows in [('position_predictions',positions),('mathematical_checks',checks),('decomposition',decomp)]:write_frame(dest/f'{name}.csv.gz',rows)
 row=dict(run_id=run,seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,dose=dose,status=failed or 'completed',source=str(source.relative_to(ROOT)),source_sha=sha(source),path=str((dest/'states.pt').relative_to(ROOT)),sha256=sha(dest/'states.pt'),directory=str(dest.relative_to(OUT)),reused=False,states=331,target_updates=n,diagnostic_updates=int(dn),wall_seconds=time.monotonic()-start);dump(dest/'complete.json',row);return row
