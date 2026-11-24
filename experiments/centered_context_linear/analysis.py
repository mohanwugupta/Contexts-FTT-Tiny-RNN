"""Seed-unit descriptive analyses, no model construction or update."""
from .core import *

def validate_grid(f,seeds=20,arms=ARMS,last=330,cues=list(CUES)):
 keys=['seed','arm','update','cue']
 expected=pd.MultiIndex.from_product([range(seeds),arms,range(last+1),cues],names=keys)
 actual=pd.MultiIndex.from_frame(f[keys])
 if actual.has_duplicates or len(actual)!=len(expected) or set(actual)!=set(expected):raise ValueError('Incomplete/duplicate frozen grid')

def block_table(f):
 rows=[]
 for (seed,arm),g in f.groupby(['seed','arm']):
  p=g.pivot(index='update',columns='cue',values='ce');last=int(p.index.max())
  for start in range(0,last-9,10):
   end=start+10;own='A' if start%20==0 else 'B';other='B' if own=='A' else 'A';gain=p.loc[start,own]-p.loc[end,own];cross=p.loc[start,other]-p.loc[end,other];floor=bool((p.loc[start,['A','B']]<=.05).all());tol=1e-10+1e-12*abs(p.loc[[start,end],['A','B']]).to_numpy().max()
   rows.append(dict(seed=seed,arm=arm,start=start,end=end,practiced=own,practiced_start_ce=p.loc[start,own],unpracticed_start_ce=p.loc[start,other],practiced_gain=gain,unpracticed_gain=cross,specificity=gain-cross,numerical_tolerance=tol,floor_at_start=floor,directional_sign_evidence=bool(not floor and gain-cross>tol),window='early' if end<=100 else 'late'))
 return pd.DataFrame(rows)

def switch_table(f):
 rows=[]
 for (seed,arm),g in f.groupby(['seed','arm']):
  p=g.pivot(index='update',columns='cue',values='ce');last=int(p.index.max())
  for boundary in range(10,last,10):
   outgoing='A' if boundary%20==10 else 'B';incoming='B' if outgoing=='A' else 'A';raw=p.loc[boundary,incoming]-p.loc[boundary,outgoing];baseline=p.loc[0,incoming]-p.loc[0,outgoing]
   rows.append(dict(seed=seed,arm=arm,boundary=boundary,direction=outgoing+'→'+incoming,incoming_ce=p.loc[boundary,incoming],outgoing_ce=p.loc[boundary,outgoing],switch_cost=raw,initial_cue_difference=baseline,baseline_adjusted_cost=raw-baseline,floor_at_boundary=bool((p.loc[boundary,['A','B']]<=.05).all())))
 return pd.DataFrame(rows)

def qualify(f):
 rows=[]
 for (seed,arm,k),g in f.groupby(['seed','arm','update']):
  d=g.set_index('cue');good=bool((d.loc[['A','B'],'ce']<=.5).all() and (d.loc[['A','B'],'accuracy']>=6/7).all())
  rows.append(dict(seed=seed,arm=arm,update=k,strong_familiar=good,familiar_ce=d.loc[['A','B'],'ce'].mean()))
 return pd.DataFrame(rows)

def time_intervals(frame,groups,time,metrics):
 rows=[];rng=np.random.default_rng(20260925);idx=rng.integers(20,size=(10000,20));weights=np.stack([(idx==i).sum(1)/20 for i in range(20)],1)
 grouper=groups[0] if len(groups)==1 else groups
 for key,g in frame.groupby(grouper):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:
   p=g.pivot(index='seed',columns=time,values=metric).reindex(range(20));x=p.to_numpy();valid=np.isfinite(x);den=weights@valid.astype(float);draw=np.divide(weights@np.nan_to_num(x),den,out=np.full_like(den,np.nan),where=den>0);ci=np.nanquantile(draw,[.025,.975],axis=0)
   for j,t in enumerate(p.columns):rows.append(dict(zip(groups,key),**{time:t,'metric':metric,'mean':np.nanmean(x[:,j]),'ci_low':ci[0,j],'ci_high':ci[1,j],'n_seeds':int(valid[:,j].sum())}))
 return pd.DataFrame(rows)

def position_geometry(pos):
 keys=['seed','arm','update','position'];a=pos[pos.cue=='A'].set_index(keys).sort_index();b=pos[pos.cue=='B'].set_index(keys).sort_index()
 if not a.index.equals(b.index) or a.index.has_duplicates:raise ValueError('Missing/duplicate paired position geometry')
 h=[f'hidden_{j}' for j in range(8)];z=[f'logit_{j}' for j in range(4)];dh=a[h].to_numpy()-b[h].to_numpy();za=a[z].to_numpy();zb=b[z].to_numpy();dz=(za-za.mean(1)[:,None])-(zb-zb.mean(1)[:,None]);r=a.reset_index()[keys].copy();r['hidden_separation']=np.linalg.norm(dh,axis=1);r['centered_logit_separation']=np.linalg.norm(dz,axis=1);return r


def main():
 read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip')
 f=read('frozen_evaluations');validate_grid(f);t=read('training_trajectories');g=read('geometry');b=block_table(f);s=switch_table(f);q=qualify(f);f=f.merge(q,on=['seed','arm','update'],validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['strong_performance']=(f.ce<=.5)&(f.accuracy>=6/7);f['above_chance']=f.ce>np.log(4)
 gap=f[f.cue.isin(['A','B'])].pivot(index=['seed','arm','update'],columns='cue',values='ce').reset_index();gap['signed_A_minus_B']=gap.A-gap.B;gap['absolute_gap']=abs(gap.signed_A_minus_B)
 counts=[];joint=[];endpoints=f[f['update'].isin(PROBES)].copy();summary=[]
 for (arm,k,cue),ff in endpoints.groupby(['arm','update','cue']):
  for subgroup in ['all','strong_familiar']:
   x=ff if subgroup=='all' else ff[ff.strong_familiar]
   for metric in ['ce','accuracy','gap']:
    mean,lo,hi=interval(x[metric]);summary.append(dict(arm=arm,update=k,cue=cue,subgroup=subgroup,metric=metric,n=len(x),mean=mean,ci_low=lo,ci_high=hi))
   counts.append(dict(arm=arm,update=k,cue=cue,subgroup=subgroup,n=len(x),strong_performance=int(x.strong_performance.sum()),above_chance=int(x.above_chance.sum())))
 for r in q[q['update'].isin(PROBES)].itertuples():
  geo=g[(g.seed==r.seed)&(g.arm==r.arm)&(g['update']==r.update)].iloc[0];c=f[(f.seed==r.seed)&(f.arm==r.arm)&(f['update']==r.update)&(f.cue=='C')].iloc[0];sep=geo.hidden_separation>geo.numerical_tolerance and geo.centered_logit_separation>geo.numerical_tolerance
  for window in ['early','late']:
   bb=b[(b.seed==r.seed)&(b.arm==r.arm)&(b.end<=r.update)&(b.window==window)];direction=bb.groupby('practiced').directional_sign_evidence.any();sign=bool(all(direction.get(cue,False) for cue in ['A','B']))
   joint.append(dict(seed=r.seed,arm=r.arm,update=r.update,window=window,strong_familiar=r.strong_familiar,C_success=c.strong_performance,nonzero_separation=sep,bidirectional_block_sign_evidence=sign,joint_descriptive=bool(r.strong_familiar and c.strong_performance and sep and sign),original_splitting_gate='unassessed'))
 paired=[]
 for (k,cue),ff in endpoints.groupby(['update','cue']):
  for metric in ['ce','accuracy','gap']:
   p=ff.pivot(index='seed',columns='arm',values=metric)
   for seed,row in p.iterrows():paired.append(dict(seed=seed,update=k,cue=cue,metric=metric,Centered_minus_Standard=row.Centered-row.Standard))
 paired=pd.DataFrame(paired);ps=[]
 for key,ff in paired.groupby(['update','cue','metric']):
  mean,lo,hi=interval(ff.Centered_minus_Standard);ps.append(dict(zip(['update','cue','metric'],key),mean=mean,ci_low=lo,ci_high=hi,n=20))
 frames={'block_transfer':b,'switch_costs':s,'performance_gaps':gap,'generalization':endpoints,'endpoint_summary':pd.DataFrame(summary),'endpoint_counts':pd.DataFrame(counts),'joint_descriptive':pd.DataFrame(joint),'paired_seed_effects':paired,'paired_intervals':pd.DataFrame(ps),'evaluation_intervals':time_intervals(f,['arm','cue'],'update',['ce','accuracy']),'training_intervals':time_intervals(t,['arm'],'update',['ce_before']),'geometry_intervals':time_intervals(g,['arm'],'update',['hidden_separation','centered_logit_separation','hidden_centroid_distance','d_norm','g_norm']),'block_intervals':time_intervals(b,['arm','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),'switch_intervals':time_intervals(s,['arm','direction'],'boundary',['switch_cost','baseline_adjusted_cost'])}
 for name,x in frames.items():x.to_csv(OUT/'tables'/f'{name}.csv',index=False)
 write_json(OUT/'audit/analysis_complete.json',dict(all_seeds=20,seed_unit=True,bootstrap_draws=10000,bootstrap_seed=20260925,model_selection='none; familiar-only reporting strata',original_splitting_gate='unassessed',floor=.05))
 append('## Analysis GREEN\nSigned block gains, raw and initial-bias-adjusted switches, all-state absolute gaps, familiar-only strata and paired differences saved. Bootstrap resamples whole seeds; no transition inference. Descriptive bidirectional signs are explicitly not the original splitting gate. Mathematical/nonzero tolerances and floor flag unchanged.')
if __name__=='__main__':main()
