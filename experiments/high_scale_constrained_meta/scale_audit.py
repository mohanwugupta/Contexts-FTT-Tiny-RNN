"""Validate and consolidate every nominal/actual radial projection and scale."""
from .common import *
from experiments.meta_objective_balance.statistics import summarize,time_summary
GROUPS=['token','context_A','context_B','context_C','output']

def group_arrays(x):
 w=x[:,:56].reshape(-1,8,7);return [w[:,:,:4].reshape(-1,32),w[:,:,4],w[:,:,5],w[:,:,6],x[:,136:168]]
def main():
 tall=[];scalars=[];effective=[];max_short=0.;nchanged=0;nominal=0
 for s in SEEDS:
  for arm in ['U','P']:
   for phase in ['Reptile','FOMAML']:
    stem=phase.lower();d=OUT/f'meta/s{s:02d}_{arm}';t=pd.read_csv(d/f'{stem}_episodes.csv',float_precision='round_trip');p=pd.read_csv(d/f'{stem}_projections.csv',float_precision='round_trip');initial=pd.read_csv(d/f'{stem}_initial_scale.csv',float_precision='round_trip').rename(columns={'rms':'after_rms'});initial['before_rms']=initial.after_rms;initial['episode']=0;initial['multiplier']=1.;initial['projected']=False;initial['parameter_change_norm']=0.;p=pd.concat([initial,p],ignore_index=True);p=p.assign(seed=s,arm=arm,phase=phase);p['cumulative_episode']=p.episode+(1920 if phase=='FOMAML' else 0);t=t.assign(seed=s,arm=arm,phase=phase);t['cumulative_episode']=t.episode+(1920 if phase=='FOMAML' else 0);changed=[]
    for path in sorted(d.glob(stem+'_chunk*.npz')):
     n=np.load(path);end=int(path.stem.split('chunk')[1]);start=end-60;pre=group_arrays(n['unprojected'][:,0]);post=group_arrays(n['after'][:,0]);adapt=group_arrays(n['adapted']);actual=np.zeros(60,int)
     for j,g in enumerate(GROUPS):
      a=pre[j].astype(float);b=post[j].astype(float);r0=np.sqrt(np.mean(a*a,axis=1));r1=np.sqrt(np.mean(b*b,axis=1));delta=np.linalg.norm(b-a,axis=1);q=p[p.group.eq(g)&p.episode.between(start+1,end)].sort_values('episode');np.testing.assert_allclose(q.before_rms,r0,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(q.after_rms,r1,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(q.parameter_change_norm,delta,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(n['scale_before_rms'][:,j],r0,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(n['scale_after_rms'][:,j],r1,rtol=1e-12,atol=1e-12);expected=np.maximum(1.,q.floor.to_numpy()/r0) if arm=='P' else np.ones(60);np.testing.assert_allclose(q.multiplier,expected,rtol=1e-12,atol=1e-12);actual+=delta>0
      if arm=='P':max_short=max(max_short,float(np.maximum(0,q.floor.to_numpy()-r1).max()));assert np.all(r1>=q.floor.to_numpy()-1e-7-2e-7*q.floor.to_numpy())
      # Adapted task weights are unconstrained by protocol; report them, never flag as a violation.
      idx=p.index[p.group.eq(g)&p.episode.between(start+1,end)];p.loc[idx,'adapted_rms']=np.sqrt(np.mean(adapt[j].astype(float)**2,axis=1));p.loc[idx,'actual_parameter_change']=delta>0
     changed.extend(actual.tolist());np.testing.assert_allclose(t.iloc[start:end].projection_change_norm,np.linalg.norm(n['after'][:,0].astype(float)-n['unprojected'][:,0].astype(float),axis=1),rtol=1e-12,atol=1e-12);np.testing.assert_allclose(t.iloc[start:end].outer_update_norm,np.linalg.norm(n['unprojected'][:,0].astype(float)-n['before'][:,0].astype(float),axis=1),rtol=1e-12,atol=1e-12)
     for i in range(60):
      for pos in range(7):effective.append(dict(seed=s,arm=arm,phase=phase,episode=start+i+1,cumulative_episode=start+i+1+(1920 if phase=='FOMAML' else 0),position=pos+1,raw_contrast=n['scale_after_effective_contrast'][i,pos],centered_contrast=n['scale_after_centered_contrast'][i,pos]))
    t['actually_changed_groups']=changed;tall.append(p);scalars.append(t);nchanged+=int(sum(changed));nominal+=int(t.projected_groups.sum())
 p=pd.concat(tall,ignore_index=True);t=pd.concat(scalars,ignore_index=True);e=pd.DataFrame(effective);write_frame(OUT/'tables/meta_group_scale_projection.csv.gz',p);write_frame(OUT/'tables/meta_training_histories.csv',t);write_frame(OUT/'tables/meta_effective_contrast_positions.csv.gz',e);windows=t.copy();windows['window_end']=((windows.episode-1)//60+1)*60;windows=windows.groupby(['seed','arm','phase','window_end']).mean(numeric_only=True).reset_index();write_frame(OUT/'tables/meta_training_windows.csv',windows);group_summary=p[p.episode.isin([0,240,960,1920])];write_frame(OUT/'tables/meta_scale_checkpoint_summary.csv',summarize(group_summary,['arm','phase','episode','group'],['after_rms','adapted_rms'],SEEDS));dump(OUT/'audit/scale_projection_verified.json',dict(passed=True,projection_group_records=230400,initial_group_records=120,all_recorded_RMS_multipliers_displacements_verified=True,nominal_group_projection_triggers=nominal,groups_with_nonzero_parameter_change=nchanged,projected_max_floor_shortfall=max_short,outside_registered_tolerance=0,inner_weights_not_constrained=True,optimizer_updates_in_audit=0));append('## Complete scale/projection record audit\nAll230400group records independently reconstructed from raw unprojected/projected tensors; RMS/multipliers/displacements match. Nominal trigger count and actual nonzero changes distinguished for float32 roundoff. NoPfloor violations outside the preregistered tolerance; no U floors enforced. Full7-position raw/centered effective contrasts retained. Inner-adapted RMS is reported separately, never treated as constrained or silently projected. No optimizer/model replay.')
 print('Scale/projection record audit PASSED',flush=True)
if __name__=='__main__':main()
