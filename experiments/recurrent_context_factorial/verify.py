"""Read-only array/tensor/coverage/source verification. Never updates models."""
import argparse
from .core import *
from .analysis import assert_complete,block_table,switch_table,endpoint_table,factorial_contrasts
from experiments.centered_context_linear.core import interval

def verify_decomposition(z,rows):
 y=rows.target.to_numpy(int);r=np.arange(len(y));z0=z['C'];u=(z['A']+z['B'])/2-z0;v=(z['A']-z['B'])/2;za=z0+u;zb=z0+2*u;tol=1e-10+1e-12*max(abs(w).max() for w in [z0,z['A'],z['B'],za,zb])
 close=lambda a,b:np.testing.assert_allclose(a,b,atol=tol,rtol=1e-12)
 margin=lambda w:w[r,y]-np.max(np.where(np.arange(4)[None,:]==y[:,None],-np.inf,w),axis=1)
 ce=lambda w:np.logaddexp.reduce(w,axis=1)-w[r,y]
 for name,w in [('z0',z0),('u',u),('v',v)]:
  close(w,rows[[f'{name}_raw_{j}' for j in range(4)]]);wc=w-w.mean(1)[:,None];close(wc,rows[[f'{name}_centered_{j}' for j in range(4)]]);column={'z0':'baseline_norm','u':'u_norm','v':'v_norm'}[name];close(np.linalg.norm(wc,axis=1),rows[column])
 for prefix,w in [('baseline',z0),('avg',za),('both',zb)]:
  close(ce(w),rows[prefix+('_ce' if prefix=='baseline' else '_algebraic_ce')]);close(margin(w),rows[prefix+'_margin'])
 wrong=np.where(np.arange(4)[None,:]==y[:,None],-np.inf,z0).argmax(1);fixed=u[r,y]-u[r,wrong];penalty=np.max(np.where(np.arange(4)[None,:]==y[:,None],-np.inf,za),axis=1)-za[r,wrong];close(fixed,rows.u_fixed_competitor_margin);close(penalty,rows.margin_competitor_switch_penalty);close(margin(za)-margin(z0),rows.actual_avg_margin_change);close(fixed-penalty,rows.actual_avg_margin_change)


def verify_positions(part,evaluations,math):
 base=OUT/part/'tables';keys=['seed','arm','update','position'];n=0;aggregates=[];controls=math.set_index(keys).sort_index();decomp=pd.read_csv(base/'decomposition.csv',float_precision='round_trip').set_index(keys).sort_index();max_ce_error=0.
 for x in pd.read_csv(base/'position_predictions.csv',chunksize=42000,float_precision='round_trip'):
  n+=len(x);assert not x.duplicated(keys+['cue']).any();assert np.array_equal(x.target,TARGET[x.position.to_numpy()-1]);assert np.array_equal(x.token,SEQ[x.position.to_numpy()-1]);z=x[[f'logit_{j}' for j in range(4)]].to_numpy();mx=z.max(1);v=z-mx[:,None];ce=mx+np.log(np.exp(v).sum(1))-z[np.arange(len(z)),x.target.to_numpy()];probs=np.exp(v);probs/=probs.sum(1,keepdims=True);error=abs(ce-x.ce.to_numpy());max_ce_error=max(max_ce_error,float(error.max()));np.testing.assert_allclose(ce,x.ce,atol=1e-10,rtol=1e-12);np.testing.assert_allclose(probs,x[[f'probability_{j}' for j in range(4)]],atol=1e-12);assert np.array_equal(z.argmax(1)==x.target,x.correct)
  a=x.groupby(['seed','arm','update','cue']).agg(ce=('ce','mean'),accuracy=('correct','mean'),positions=('position','size'));assert a.positions.eq(7).all();aggregates.append(a)
  for family,size in [('hidden',8),('logit',4),('probability',4)]:
   pivots=[x.pivot(index=keys,columns='cue',values=f'{family}_{j}').sort_index() for j in range(size)]
   for p in pivots:np.testing.assert_array_equal(p.C,p.NoContext)
   if family=='probability':continue
   ci=controls.loc[pivots[0].index];field='logits' if family=='logit' else 'hidden'
   avg=np.stack([(p['Mixed-average']-(p.A+p.B)/2).to_numpy() for p in pivots],1);both=np.stack([(p['Mixed-both']-p.A-p.B+p.C).to_numpy() for p in pivots],1)
   assert (abs(avg).max(1)<=ci[f'avg_{field}_allowance'].to_numpy()).all();assert (abs(both).max(1)<=ci[f'both_{field}_allowance'].to_numpy()).all()
   if part=='part_a':
    mask=pivots[0].index.get_level_values('arm')=='Centered'
    for p in pivots:np.testing.assert_array_equal(p.loc[mask,'Mixed-average'],p.loc[mask,'Mixed-both']);np.testing.assert_array_equal(p.loc[mask,'Mixed-average'],p.loc[mask,'C'])
  zz={c:np.stack([x.pivot(index=keys,columns='cue',values=f'logit_{j}').sort_index()[c].to_numpy() for j in range(4)],1) for c in CUES};index=x.pivot(index=keys,columns='cue',values='logit_0').sort_index().index;verify_decomposition(zz,decomp.loc[index])
 agg=pd.concat(aggregates).sort_index();assert not agg.index.has_duplicates;expected=evaluations.set_index(['seed','arm','update','cue']).sort_index();np.testing.assert_allclose(agg[['ce','accuracy']],expected[['ce','accuracy']],atol=1e-10,rtol=1e-12)
 return n,max_ce_error

def verify(sealed=True):
 starts=json.loads((OUT/'audit/part_b_started.json').read_text());done=json.loads((OUT/'audit/part_b_complete.json').read_text());pa=json.loads((OUT/'audit/part_a_complete.json').read_text());check_hashes(ROOT,starts['inputs']);check_hashes(ROOT,pa['inputs']);assert guard()==starts['source_manifests']==pa['source_manifests'];assert done['new_optimizer_steps']==26300 and done['archived_updates_reused']==100
 # Original execution code and corrected continuation source both retained; core algorithm unchanged.
 correction=json.loads((OUT/'audit/validation_correction_resume.json').read_text());assert starts['code']['experiments/recurrent_context_factorial/core.py']==done['code']['experiments/recurrent_context_factorial/core.py']==sha(ROOT/'experiments/recurrent_context_factorial/core.py');assert starts['code']['experiments/recurrent_context_factorial/part_b.py']==sha(OUT/'audit/part_b_before_validation_correction.py');assert done['code']['experiments/recurrent_context_factorial/part_b.py']==sha(OUT/'audit/part_b_executed_continuation.py');assert sha(OUT/correction['retained_original_code'])==correction['original_code_sha256'];assert correction['changes_to_numerical_tolerances']==False and correction['scientific_optimizer_updates_repeated']==0
 for name,info in correction['existing_files'].items():
  p=OUT/name
  with p.open('rb') as f:assert hashlib.sha256(f.read(info['bytes'])).hexdigest()==info['sha256']
 read=lambda part,n:pd.read_csv(OUT/part/'tables'/f'{n}.csv',float_precision='round_trip');fa=read('part_a','evaluations');fb=read('part_b','evaluations');assert_complete(fa,['Standard','Centered']);assert_complete(fb);assert fa.status.eq('completed').all() and fb.status.eq('completed').all();poscounts={}
 for part,f,expected in [('part_a',fa,92680),('part_b',fb,185360)]:
  m=read(part,'mathematical_checks');assert len(m)==expected and m.passed.all();assert (m.double_avg_logits_error<=m.double_tolerance).all() and (m.double_both_logits_error<=m.double_tolerance).all();assert (m.double_avg_ce_gap<=2*m.double_tolerance).all();assert (m.float32_avg_ce_gap<=2*m.avg_logits_allowance).all();poscounts[part]=verify_positions(part,f,m)
  ep=read(part,'endpoints');expected_ep=endpoint_table(f[f['update'].isin(PROBES)]);pd.testing.assert_frame_equal(ep,expected_ep,check_exact=False,atol=1e-10,rtol=1e-12)
  summary=read(part,'endpoint_summary')
  for r in summary.itertuples():
   ff=ep[(ep.arm==r.arm)&(ep['update']==r.update)&(ep.cue==r.cue)];ff=ff if r.subgroup=='all' else ff[ff.strong_familiar];assert len(ff)==r.n;np.testing.assert_allclose(interval(ff[r.metric]),[r.mean,r.ci_low,r.ci_high],atol=1e-10,rtol=1e-12,equal_nan=True);np.testing.assert_allclose(ff[r.metric].median(),r.median,atol=1e-10,equal_nan=True)
 registry=read('part_b','checkpoint_registry');assert len(registry)==80 and not registry.duplicated(['seed','arm']).any();statecount=0
 for seed in range(20):
  initial=models(seed)
  for rr in registry[registry.seed==seed].itertuples():
   path=OUT/rr.path;assert sha(path)==rr.sha256;p=torch.load(path,map_location='cpu',weights_only=False);assert p['seed']==seed and p['arm']==rr.arm and len(p['states'])==331;statecount+=331
   for n,v in initial[rr.arm].state_dict().items():assert torch.equal(v,p['states'][0]['model'][n])
   for k,state in enumerate(p['states']):
    assert state['update']==k and state['status']=='completed';assert sum(v.numel() for v in state['model'].values())==172;assert torch.count_nonzero(state['model']['recurrent.weight_ih_l0'][:,6])==0;assert len(state['optimizer']['param_groups'][0]['params'])==8
    if k:assert len(state['optimizer']['state'])==8 and all(int(v['step'])==k for v in state['optimizer']['state'].values())
 t=read('part_b','training');assert len(t)==26400 and not t.duplicated(['seed','arm','update']).any();assert t.status.eq('completed').all();assert t.optimizer_step_executed.sum()==26300 and t.archived_update_reused.sum()==100;assert (t.optimizer_step_executed^t.archived_update_reused).all();assert all(r.training_cue==training_cue(r.update) for r in t.itertuples());lookup=fb.set_index(['seed','arm','update','cue']).ce;before=np.array([lookup.loc[(r.seed,r.arm,r.update-1,r.training_cue)] for r in t.itertuples()]);np.testing.assert_allclose(t.ce_before,before,atol=2e-5,rtol=1e-6)
 oldhist=pd.read_csv(HIGH/'training_measurements.csv',float_precision='round_trip');oldhist=oldhist[oldhist.architecture=='RNN-Linear'].set_index(['seed','update']);hi=t[t.arm=='HI'];np.testing.assert_array_equal(hi.ce_before,[oldhist.loc[(r.seed,r.update),'ce_before'] for r in hi.itertuples()]);assert read('part_b','historical_fidelity').matched.all()
 for r in read('part_b','source_reuse_registry').itertuples():assert sha(ROOT/r.path)==r.sha256
 for name,computed in [('block_transfer',block_table(fb)),('switch_costs',switch_table(fb))]:pd.testing.assert_frame_equal(read('part_b',name),computed,check_exact=False,atol=1e-9,rtol=1e-10)
 seed=read('part_b','seed_outcomes');contrasts=read('part_b','factorial_seed_contrasts')
 for metric in seed.columns[3:]:
  wide=seed.pivot(index=['seed','update'],columns='arm',values=metric).reset_index();z=factorial_contrasts(wide).sort_values(['seed','update']);saved=contrasts[contrasts.metric==metric].sort_values(['seed','update']);np.testing.assert_allclose(saved[['normalization','centering','interaction']],z[['normalization','centering','interaction']],atol=1e-10)
 for r in read('part_b','factorial_intervals').itertuples():
  x=contrasts[(contrasts['update']==r.update)&(contrasts.metric==r.metric)];assert len(x)==20;np.testing.assert_allclose(interval(x[r.effect]),[r.mean,r.ci_low,r.ci_high],atol=1e-10)
 if sealed:
  man=json.loads((OUT/'manifest.json').read_text());check_hashes(OUT,man['outputs']);check_hashes(ROOT,man['code'])
 return dict(passed=True,part_a_evaluations=len(fa),part_b_evaluations=len(fb),part_a_positions=poscounts['part_a'][0],part_b_positions=poscounts['part_b'][0],new_optimizer_steps=26300,archived_updates_reused=100,part_b_states=statecount,training_replayed=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--unsealed',action='store_true');p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify(not a.unsealed)
 if a.regenerate:
  if a.regenerate.exists() and list(a.regenerate.iterdir()):raise ValueError('Empty figure destination required')
  from .figures import figure_a,primary,remaining
  for part,generate in [('part_a',figure_a),('part_b',primary),('part_b',remaining)]:
   for name in generate(a.regenerate/part):assert sha(a.regenerate/part/name)==sha(OUT/part/'figures'/name)
  r['figure_files_byte_identical']=16
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
