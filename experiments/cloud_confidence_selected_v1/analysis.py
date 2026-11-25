"""Pure saved-observation joins/accounting, without model or optimizer access."""
import numpy as np
import pandas as pd
TIMES=[0,10,20,40,80,120,170,330]
CUES=['A','B','C','Mixed-average','NoContext']
KEYS=['condition','seed','task_id','update','cue','position','view']
CATEGORIES=['both_wrong','became_correct','became_wrong','both_correct']


def booleans(series):
    result=series.map({True:True,False:False,'True':True,'False':False})
    if result.isna().any():raise ValueError('Missing/invalid correctness flag')
    return result.astype(bool)


def validate(frame):
    expected=pd.MultiIndex.from_product([['candidate','control'],[21000,21001],['task00','task01','task02','task03'],TIMES,CUES,range(1,8),['native','quarter']],names=KEYS)
    if frame.duplicated(KEYS).any() or len(frame)!=len(expected):raise ValueError('Missing/duplicate saved observations')
    got=pd.MultiIndex.from_frame(frame[KEYS])
    if len(expected.difference(got)) or len(got.difference(expected)):raise ValueError('Wrong checkpoint/cue/task/seed/view grid')
    numeric=['ce','margin','target','token','predicted_class']+[f'logit_{i}' for i in range(4)]+[f'probability_{i}' for i in range(4)]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():raise ValueError('Nonfinite/missing source cell')
    for k in ['target','token','predicted_class']:
        if not frame[k].between(0,3).all() or not frame[k].mod(1).eq(0).all():raise ValueError('Invalid token/class indexing')
    z=frame[[f'logit_{i}' for i in range(4)]].to_numpy(float);s=z-z.max(axis=1,keepdims=True)
    lp=s-np.log(np.exp(s).sum(axis=1,keepdims=True));targets=frame.target.to_numpy(int);ids=np.arange(len(frame))
    ce=-lp[ids,targets];correct=z.argmax(axis=1)==targets
    other=z.copy();other[ids,targets]=-np.inf;margin=z[ids,targets]-other.max(axis=1)
    # Inherited double saved-logit score tolerances, not behavioral thresholds.
    for actual,wanted in [(frame.ce,ce),(frame.margin,margin),(frame[[f'probability_{i}' for i in range(4)]],np.exp(lp))]:
        if not np.allclose(actual,wanted,atol=2e-12,rtol=1e-12):raise ValueError('Saved scalar/probability/logit mismatch')
    if not np.array_equal(booleans(frame.correct),correct) or not np.array_equal(frame.predicted_class,z.argmax(axis=1)):raise ValueError('Saved decision/logit mismatch')


def classify(pairs):
    p=pairs.copy();a=booleans(p.correct_reference);b=booleans(p.correct_alternative)
    p['category']=np.select([~a&~b,~a&b,a&~b,a&b],CATEGORIES,default='invalid')
    p['CE_gain']=p.ce_reference-p.ce_alternative
    p['correct_count_change']=b.astype(int)-a.astype(int)
    if 'margin_reference' in p:p['margin_change']=p.margin_alternative-p.margin_reference
    if 'predicted_class_reference' in p:p['same_predicted_token']=p.predicted_class_reference.eq(p.predicted_class_alternative)
    return p


def paired(frame,axis,reference,alternative,keys):
    if frame.duplicated(keys+[axis]).any():raise ValueError('Ambiguous pair')
    a=frame[frame[axis].eq(reference)].drop(columns=[axis]);b=frame[frame[axis].eq(alternative)].drop(columns=[axis])
    p=a.merge(b,on=keys,how='outer',suffixes=('_reference','_alternative'),validate='one_to_one',indicator=True)
    if not p['_merge'].eq('both').all():raise ValueError('Missing matched reference/alternative')
    for k in ['target','token']:
        if not p[k+'_reference'].eq(p[k+'_alternative']).all():raise ValueError('Different matched tokens/targets')
    return classify(p.drop(columns=['_merge']))


def accounting(pairs,keys):
    counts=pairs.groupby(keys).position.agg(['size','nunique'])
    if not counts['size'].eq(7).all() or not counts['nunique'].eq(7).all():raise ValueError('Incomplete seven-position accounting')
    observed=pairs.groupby(keys+['category'],as_index=False).agg(positions=('position','size'),CE_gain_contribution=('CE_gain','sum'),correct_count_change=('correct_count_change','sum'))
    observed['CE_gain_contribution']/=7
    full=pairs[keys].drop_duplicates().merge(pd.DataFrame({'category':CATEGORIES}),how='cross')
    out=full.merge(observed,on=keys+['category'],how='left',validate='one_to_one')
    # All seven positions were validated above. An empty category contributes
    # exactly zero; no missing observation/failed feasibility is zero-imputed.
    for k in ['positions','CE_gain_contribution','correct_count_change']:out[k]=out[k].fillna(0)
    out['positions']=out.positions.astype(int);out['correct_count_change']=out.correct_count_change.astype(int)
    return out


def metric_table(pairs,keys):
    out=pairs.groupby(keys,as_index=False).agg(CE_gain=('CE_gain','mean'),correct_count_change=('correct_count_change','sum'),margin_change=('margin_change','mean'),reference_CE=('ce_reference','mean'),alternative_CE=('ce_alternative','mean'))
    baseline_keys=[k for k in keys if k!='update']
    base=out[out['update'].eq(0)][baseline_keys+['CE_gain','correct_count_change']].rename(columns={'CE_gain':'CE_gain_at0','correct_count_change':'correct_change_at0'})
    out=out.merge(base,on=baseline_keys,validate='many_to_one')
    out['CE_gain_change_since0']=out.CE_gain-out.CE_gain_at0
    out['correct_change_since0']=out.correct_count_change-out.correct_change_at0
    return out


def tables(frame):
    validate(frame)
    wk=['condition','seed','task_id','update','cue','position'];pk=['view','seed','task_id','update','cue','position']
    within=paired(frame,'view','native','quarter',wk);across=paired(frame,'condition','control','candidate',pk)
    return dict(within_readout_positions=within,within_readout_accounting=accounting(within,wk[:-1]),within_readout_metrics=metric_table(within,wk[:-1]),initialization_positions=across,initialization_accounting=accounting(across,pk[:-1]),initialization_metrics=metric_table(across,pk[:-1]))


def seed_intervals(metrics,dimensions):
    from experiments.meta_objective_balance.statistics import interval
    fields=['CE_gain','correct_count_change','margin_change','CE_gain_change_since0','correct_change_since0']
    by=metrics.groupby(dimensions+['seed'])[fields].mean().reset_index();rows=[]
    for label,part in by.groupby(dimensions,sort=True):
        if not isinstance(label,tuple):label=(label,)
        for field in fields:
            x=part.set_index('seed')[field].reindex([21000,21001]).to_numpy(float)
            rows.append(dict(zip(dimensions,label),metric=field,**interval(x,[21000,21001])))
    return by,pd.DataFrame(rows)
