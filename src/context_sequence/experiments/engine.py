"""Frozen experiments. All selection is structural; no outcome-dependent branches."""
import hashlib
import numpy as np
import pandas as pd
from context_sequence.evaluation.rollout import rollout,midpoint,centered,cue_projection
from context_sequence.metrics.geometry import geometry
from context_sequence.statistics.aggregation import novel_means
from context_sequence.task.novelty import VARIANTS
from context_sequence.task.sequence import SEQUENCE

METRICS=['familiar_mean_ce','raw_ce','raw_rescue','task_distance','task_residual','task_separation','hidden_distance','hidden_residual','hidden_separation','axis_coordinate','off_axis_distance','task_generation','cue_alpha','cue_discarded_norm','transfer_deficit']
IDENTITY=['cohort','model_family','training_objective','family','condition','seed','checkpoint_step','checkpoint_sha256']
CELL=IDENTITY+['probe','intervention','continuation','analysis_kind']


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def references(model,cues):
    # The familiar targets are constructed before any protected cue rollout.
    a=rollout(model,cues['A']);b=rollout(model,cues['B'])
    target=midpoint(a['hidden'],b['hidden'])
    natural={'A':a,'B':b}
    natural.update({p:rollout(model,c) for p,c in cues.items() if p not in natural})
    return a,b,target,natural


def measure(record,probe,k,h,z,a,b,native,*,intervention,continuation='D0',kind='trajectory',position=0,cue=None,prestate=None):
    z64=np.asarray(z,float);maximum=z64.max();ce=float(maximum+np.log(np.exp(z64-maximum).sum())-z64[SEQUENCE[k+1]])
    g=geometry(h,z,a,b,k);ng=geometry(native['hidden'][k],native['logits'][k],a,b,k)
    alpha,residual=cue_projection(cue)
    row={**{key:record[key] for key in IDENTITY},'probe':probe,'transition':k+1,'position':position,
         'lag':k-position+1 if position else -1,'intervention':intervention,'continuation':continuation,'analysis_kind':kind,
         'familiar_mean_ce':float((a['raw_ce'].mean()+b['raw_ce'].mean())/2),'raw_ce':ce,'correct_token_probability':float(np.exp(-ce)),'baseline_raw_ce':float(native['raw_ce'][k]),
         'raw_rescue':float(native['raw_ce'][k]-ce),'transfer_deficit':ce-float((a['raw_ce'].mean()+b['raw_ce'].mean())/2),
         **g,'task_generation':ng['task_distance']-g['task_distance'],'cue_alpha':alpha,'cue_discarded_norm':float(np.linalg.norm(residual)),
         'prestate_hash':digest(prestate) if prestate is not None else '',
         **{f'hidden_{i}':float(v) for i,v in enumerate(h)},**{f'logit_{i}':float(v) for i,v in enumerate(z)},
         **{f'centered_logit_{i}':float(v) for i,v in enumerate(centered(z))}}
    return row


def trajectory_rows(record,probe,run,a,b,native,intervention,position=0,continuation='D0',kind='trajectory'):
    return [measure(record,probe,k,run['hidden'][k],run['logits'][k],a,b,native,intervention=intervention,
                    continuation=continuation,kind=kind,position=position,cue=run['cues'][k],prestate=run['prestates'][k]) for k in range(7)]


def evaluate_constructive(model,cues,record):
    before=model.digest();a,b,target,natural=references(model,cues);rows=[]
    for probe,run in natural.items():rows.extend(trajectory_rows(record,probe,run,a,b,run,'natural'))
    if before!=model.digest():raise RuntimeError('Frozen model mutated')
    return pd.DataFrame(rows)


def evaluate_canonical(model,cues,record):
    before=model.digest();a,b,target,natural=references(model,cues);tables={k:[] for k in ['a1_canonical_phenotype_transition','a3_composite_sufficiency_transition','a4_maintenance_transition','a5_cue_generation_transition']}
    for probe,run in natural.items():tables['a1_canonical_phenotype_transition'].extend(trajectory_rows(record,probe,run,a,b,run,'natural'))
    probes=[p for p in cues if p not in ('A','B')]
    for probe in probes:
        native=natural[probe];cue=cues[probe]
        tables['a3_composite_sufficiency_transition'].extend(trajectory_rows(record,probe,native,a,b,native,'natural'))
        for label,patch,target_values in [('composite','correct',target),('wrong_position','wrong',target),('orthogonal','orthogonal',target),('A_endpoint','correct',a['hidden']),('B_endpoint','correct',b['hidden'])]:
            for position in [None,*range(7)]:
                run=rollout(model,cue,target=target_values,patch=patch,position=position,seed=record['seed'])
                name=('repeated_' if position is None else 'single_')+label
                tables['a3_composite_sufficiency_transition'].extend(trajectory_rows(record,probe,run,a,b,native,name,0 if position is None else position+1))
        for position in range(7):
            for clamp,continued in [('D0',None),('D1',cues['NoContext']),('D2',cues['Mixed'])]:
                run=rollout(model,cue,target=target,patch='correct',position=position,continuation=continued)
                tables['a4_maintenance_transition'].extend(trajectory_rows(record,probe,run,a,b,native,'single_composite',position+1,clamp))
    for probe,native in natural.items():
        for k in range(7):
            pre=native['prestates'][k]
            for branch,cue in [('native',cues[probe]),('Mixed',cues['Mixed']),('A',cues['A']),('B',cues['B']),('zero',cues['NoContext'])]:
                h,z=model.step(SEQUENCE[k],cue,pre)
                tables['a5_cue_generation_transition'].append(measure(record,probe,k,h,z,a,b,native,intervention=branch,kind='local',position=k+1,cue=cue,prestate=pre))
        for position in range(7):
            for label,cue,pulse in [('pulse_Mixed',cues['Mixed'],True),('persistent_Mixed',cues['Mixed'],False),('persistent_A',cues['A'],False),('persistent_B',cues['B'],False)]:
                run=rollout(model,cues[probe],takeover=cue,position=position,pulse=pulse)
                tables['a5_cue_generation_transition'].extend(trajectory_rows(record,probe,run,a,b,native,label,position+1,kind='takeover'))
    if before!=model.digest():raise RuntimeError('Frozen model mutated')
    return {k:pd.DataFrame(v) for k,v in tables.items()}


def seed_summary(frame):
    rows=[]
    # Equal position weights; within-position transition means, then position means.
    for key,g in frame.groupby(CELL,sort=False,dropna=False):
        base=dict(zip(CELL,key));position_means=g.groupby('position')[METRICS].mean()
        row={**base,**position_means.mean().to_dict(),'n_positions':len(position_means)}
        acute=g[g.position.gt(0)]
        if len(acute):
            row['immediate_rescue']=float(acute[acute.lag.eq(0)].groupby('position').raw_rescue.mean().mean())
            row['remaining_rescue']=float(acute[acute.lag.ge(0)].groupby('position').raw_rescue.mean().mean())
            row['post_task_distance']=float(acute[acute.lag.gt(0)].groupby('position').task_distance.mean().mean())
            row['post_task_residual']=float(acute[acute.lag.gt(0)].groupby('position').task_residual.mean().mean())
        rows.append(row)
    result=pd.DataFrame(rows)
    if result.probe.isin(VARIANTS).any():
        metrics=[c for c in result if c not in CELL and pd.api.types.is_numeric_dtype(result[c])]
        result=novel_means(result,[c for c in CELL if c!='probe'],metrics)
    return result


def lag_summary(frame):
    frame=frame[frame.lag.ge(0)]
    keys=CELL+['lag']
    result=frame.groupby(keys,as_index=False)[METRICS].mean()
    counts=frame.groupby(keys,as_index=False).position.nunique().rename(columns={'position':'n_positions'})
    result=result.merge(counts,on=keys,validate='one_to_one')
    if result.probe.isin(VARIANTS).any():result=novel_means(result,[c for c in keys if c!='probe'],METRICS+['n_positions'])
    return result
