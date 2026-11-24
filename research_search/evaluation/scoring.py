"""Fixed deterministic scorer. Candidate code never chooses metrics or thresholds."""
import numpy as np,pandas as pd
from experiments.context_experience_prior import core
from experiments.joint_phenotype_hypothesis_screen.analysis import seed_summary
from .metrics import first_blocks,learning_tables

PROBES=['A','B','C','Mixed-average','NoContext']

def observations(logits,updates,path,seed,candidate,allow_nonfinite=False):
    z=np.asarray(logits,float);updates=list(map(int,updates));path=list(map(int,path))
    if z.shape!=(len(updates),5,7,4) or len(path)!=8 or len(set(updates))!=len(updates) or not allow_nonfinite and not np.isfinite(z).all():raise ValueError('Invalid prediction grid; retain as failed outcome')
    rows=[];positions=[];targets=np.array(path[1:])
    for i,n in enumerate(updates):
        for j,c in enumerate(PROBES):
            valid_positions=np.isfinite(z[i,j]).all(axis=1);valid=bool(valid_positions.all())
            with np.errstate(invalid='ignore',over='ignore',under='ignore',divide='ignore'):s=core.score(z[i,j],targets)
            for field in ['ce','margin','probabilities']:s[field][~valid_positions]=np.nan
            ce=float(s['ce'].mean());count=int(s['accuracy'].sum()) if valid else np.nan;status='completed' if valid else 'nonfinite_probe'
            rows.append(dict(seed=seed,candidate=candidate,update=n,cue=c,ce=ce,correct_count=count,accuracy=count/7,margin=float(s['margin'].mean()),useful=bool(ce<=.5 and count>=6) if valid else None,strict=bool(ce<=.1 and count==7) if valid else None,status=status))
            for t in range(7):positions.append(dict(seed=seed,candidate=candidate,update=n,cue=c,position=t+1,token=path[t],target=path[t+1],predicted_class=int(z[i,j,t].argmax()) if valid_positions[t] else np.nan,correct=bool(s['accuracy'][t]) if valid_positions[t] else None,ce=float(s['ce'][t]),margin=float(s['margin'][t]),target_probability=float(s['probabilities'][t,targets[t]]),**{f'logit_{k}':float(z[i,j,t,k]) for k in range(4)},**{f'probability_{k}':float(s['probabilities'][t,k]) for k in range(4)},status='completed' if valid_positions[t] else 'nonfinite_probe'))
    f=pd.DataFrame(rows);p=pd.DataFrame(positions);familiar=f[f.cue.isin(['A','B'])].groupby('update').agg(familiar_ce=('ce',lambda x:float(x.mean()) if x.notna().all() else np.nan),strong_familiar=('useful',lambda x:bool(x.astype(bool).all()) if x.notna().all() else None))
    f=f.merge(familiar,on='update',validate='many_to_one');f['gap']=f.ce-f.familiar_ce
    return f,p

def score(logits,updates,path,seed,candidate,reference=None,sparse=False):
    f,p=observations(logits,updates,path,seed,candidate)
    if not sparse and sorted(set(updates))!=list(range(331)):raise ValueError('All331 frozen target states required')
    ref=reference or dict(resolved=False,Q_A=0,Q_B=0,E_normalized=0)
    summary=seed_summary(f,candidate,seed,'development',ref)
    hit=f.loc[f.strong_familiar,'update']
    summary['first_strong_familiar']=float(hit.min()) if len(hit) else None
    summary['original_late_functional']=None
    summary['development_feasible']=bool(summary['original_early_phenotype'] and summary['net_A']>0 and summary['net_B']>0 and summary['C_useful'] and summary['Mixed_useful'] and summary['A_to_B']>0 and summary['B_to_A']>0 and summary['sustained_cost']>=0)
    # This additional development routing rule is frozen before outcomes. It
    # does not replace any original measure or assert a late functional gate.
    if sparse:return dict(summary=summary,metrics=f,positions=p)
    blocks,switches,matched=learning_tables(f)
    return dict(summary=summary,metrics=f,positions=p,blocks=blocks,switch_costs=switches,matched_acquisition=matched)

def cohort(rows,seeds):
    """Average tasks within each trained seed before bootstrap; missing tasks persist."""
    from experiments.meta_objective_balance.statistics import interval
    f=pd.DataFrame(rows);metrics=['D_A','D_B','A_to_B','B_to_A','sustained_cost','C_ce','Mixed_ce','NoContext_ce']
    out={}
    expected=f.task_id.nunique()
    for field in metrics:
        by=f.groupby('seed')[field].agg(['mean','count']);x=by['mean'].where(by['count'].eq(expected)).reindex(seeds).to_numpy(float);out[field]=interval(x,list(seeds))
    return out
