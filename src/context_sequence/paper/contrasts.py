import numpy as np
import pandas as pd
from context_sequence.statistics.aggregation import paired_effect
from context_sequence.statistics.bootstrap import summarize

PROBES=['Mixed','NoContext','Novel30Mean','Novel60Mean','NovelMean']


def all_contrasts(tables,indices):
    rows=[]
    def add(analysis,label,left,right,metric,cohort,family,probe):
        effect=paired_effect(left.set_index('seed')[metric],right.set_index('seed')[metric])
        if list(effect.index)!=list(range(20)):raise ValueError('Primary paired contrast must contain registered seeds 0–19')
        for seed,value in effect.items():rows.append({'analysis':analysis,'cohort':cohort,'family':family,'probe':probe,'contrast':label,'metric':metric,'seed':seed,'effect':value,'orientation':'left_minus_right'})
    for name,df in tables.items():
        if name=='a1_canonical_phenotype_seed':
            for family,g in df.groupby('family'):
                for p in PROBES:
                    one=g[g.probe.eq(p)]
                    if len(one):
                        zero=one.copy();zero['transfer_deficit']=0
                        add('A1',p+' minus familiar',one,zero,'transfer_deficit','canonical',family,p)
                add('A1','Novel60Mean minus Novel30Mean',g[g.probe.eq('Novel60Mean')],g[g.probe.eq('Novel30Mean')],'raw_ce','canonical',family,'novelty_gradient')
            for p in ['A','B',*PROBES]:
                for metric in ['raw_ce','task_separation','hidden_separation','task_distance','task_residual']:
                    add('A1','High minus Low',df[df.family.eq('High_monotonic')&df.probe.eq(p)],df[df.family.eq('Low_monotonic')&df.probe.eq(p)],metric,'canonical','High_vs_Low',p)
        if name=='a3_composite_sufficiency_seed':
            for (family,p),g in df.groupby(['family','probe']):
                if p not in PROBES:continue
                for other in ['natural','repeated_wrong_position','repeated_orthogonal','repeated_A_endpoint','repeated_B_endpoint']:
                    add('A3','repeated_composite minus '+other,g[g.intervention.eq('repeated_composite')],g[g.intervention.eq(other)],'raw_ce','canonical',family,p)
                for other in ['single_wrong_position','single_orthogonal','single_A_endpoint','single_B_endpoint']:
                    add('A3','single_composite minus '+other,g[g.intervention.eq('single_composite')],g[g.intervention.eq(other)],'immediate_rescue','canonical',family,p)
                for iv,metric in [('repeated_composite','raw_rescue'),('single_composite','immediate_rescue'),('single_composite','remaining_rescue')]:
                    one=g[g.intervention.eq(iv)];zero=one.copy();zero[metric]=0
                    add('A3',iv+' rescue versus zero',one,zero,metric,'canonical',family,p)
            for family,g in df[df.intervention.eq('repeated_composite')].groupby('family'):
                add('A3','Novel60Mean minus Novel30Mean rescue',g[g.probe.eq('Novel60Mean')],g[g.probe.eq('Novel30Mean')],'raw_rescue','canonical',family,'novelty_gradient')
        if name=='a4_maintenance_seed':
            for (family,p),g in df.groupby(['family','probe']):
                if p not in PROBES:continue
                for d in ['D1','D2']:
                    for metric in ['remaining_rescue','post_task_distance','post_task_residual']:
                        add('A4',d+' minus D0',g[g.continuation.eq(d)],g[g.continuation.eq('D0')],metric,'canonical',family,p)
            for family,g in df[df.continuation.eq('D0')].groupby('family'):
                for metric in ['remaining_rescue','post_task_distance']:add('A4','Novel60Mean minus Novel30Mean',g[g.probe.eq('Novel60Mean')],g[g.probe.eq('Novel30Mean')],metric,'canonical',family,'novelty_gradient')
        if name=='a5_cue_generation_seed':
            for (family,p),g in df.groupby(['family','probe']):
                if p not in ['A','B',*PROBES]:continue
                local=g[g.analysis_kind.eq('local')];take=g[g.analysis_kind.eq('takeover')]
                for other in ['native','A','B','zero']:
                    for metric in ['raw_ce','task_distance']:
                        add('A5','local Mixed minus '+other,local[local.intervention.eq('Mixed')],local[local.intervention.eq(other)],metric,'canonical',family,p)
                for other in ['pulse_Mixed','persistent_A','persistent_B']:
                    add('A5','persistent_Mixed minus '+other,take[take.intervention.eq('persistent_Mixed')],take[take.intervention.eq(other)],'remaining_rescue','canonical',family,p)
        if name=='a6_constructive_seed':
            for p,g in df.groupby('probe'):
                if p not in ['A','B',*PROBES]:continue
                for left,right in [('C1_contrast_only','C0_canonical'),('C2_closure_only','C0_canonical'),('C3_full','C0_canonical'),('C3_full','C1_contrast_only'),('C3_full','C2_closure_only')]:
                    for metric in ['raw_ce','transfer_deficit','task_distance','task_residual','task_separation','hidden_separation']:
                        add('A6',left+' minus '+right,g[g.condition.eq(left)],g[g.condition.eq(right)],metric,'constructive','High_sequence_only',p)
    effects=pd.DataFrame(rows);summary=[]
    keys=['analysis','cohort','family','probe','contrast','metric','orientation']
    for key,g in effects.groupby(keys,sort=False):summary.append({**dict(zip(keys,key)),**summarize(g.sort_values('seed').effect.to_numpy(),indices)})
    return effects,pd.DataFrame(summary)


def geometry_behavior(summary):
    keys=['seed','probe'];a=summary[summary.condition.eq('C0_canonical')].set_index(keys);b=summary[summary.condition.eq('C3_full')].set_index(keys)
    metrics=['raw_ce','task_distance','task_residual','task_separation','hidden_separation']
    d=b[metrics]-a[metrics];d=d.rename(columns={m:'delta_'+m for m in metrics}).reset_index()
    d['cohort']='constructive';d['contrast']='C3_full minus C0_canonical'
    return d[d.probe.isin(PROBES)]
