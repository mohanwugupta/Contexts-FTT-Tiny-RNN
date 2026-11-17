"""D2 task-defined history signature and conservative familiar-only decoders."""
import numpy as np
import pandas as pd
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from .config import SEQUENCE,CONTRACT
from .source_adapter import records,load_original
from .replay import natural_trace
from .familiar_decoder import fit_familiar,decode
from .statistics import seed_bootstrap
from .io import save,dump,report,table


def history_mask(sequence=SEQUENCE):
    successors={}
    for token,target in zip(sequence[:-1],sequence[1:]):successors.setdefault(token,set()).add(target)
    return np.array([len(successors[token])>1 for token in sequence[:-1]],bool)


def excess_table(a,b,m):
    for f in [a,b,m]:
        if f.transition.tolist()!=list(range(1,8)) or f.token.tolist()!=list(SEQUENCE[:-1]) or f.target.tolist()!=list(SEQUENCE[1:]):raise ValueError('Token/target alignment differs from fixed grammar')
    f=m[['transition','token','target']].copy();f['history_dependent']=history_mask();f['A_ce']=a.raw_ce.to_numpy();f['B_ce']=b.raw_ce.to_numpy();f['mixed_ce']=m.raw_ce.to_numpy();f['familiar_ce']=(f.A_ce+f.B_ce)/2;f['excess_ce']=f.mixed_ce-f.familiar_ce
    return f


def seed_contrast(frame):
    rows=[]
    for seed,f in frame.groupby('seed'):
        if len(f)!=7 or sorted(f.transition.tolist())!=list(range(1,8)):raise ValueError('Exactly seven repeated observations required per seed')
        h=f[f.history_dependent].excess_ce.mean();d=f[~f.history_dependent].excess_ce.mean()
        rows.append({'seed':seed,'history_excess_ce':h,'diagnostic_excess_ce':d,'history_contrast':h-d,'overall_excess_ce':f.excess_ce.mean()})
    return pd.DataFrame(rows)


def signature(model):
    a=natural_trace(model,[1,0,0]);b=natural_trace(model,[0,1,0]);m=natural_trace(model,[.5,.5,0]);f=excess_table(a,b,m);f['seed']=0
    return f,seed_contrast(f).iloc[0].to_dict()


def run(root,out):
    raw=pd.read_csv(out/'tables/d0_native_replay.csv',float_precision='round_trip');long=[];decoded=[];fitrows=[];fits=[]
    for r in records():
        f=raw[raw.seed.eq(r['seed'])];a=f[f.cue.eq('A')].sort_values('transition');b=f[f.cue.eq('B')].sort_values('transition');m=f[f.cue.eq('Mixed')].sort_values('transition')
        x=excess_table(a,b,m);x['seed']=r['seed'];x['checkpoint_sha']=r['checkpoint_sha256'];long.append(x)
        cols=[f'hidden_{i}' for i in range(8)];fit=fit_familiar(a[cols].to_numpy(),b[cols].to_numpy());fitrows.append({'seed':r['seed'],'checkpoint_sha':r['checkpoint_sha256'],**{k:v for k,v in fit.items() if k not in ['ridge_pool','ridge_A','ridge_B','prototypes']}})
        fits.append({'seed':r['seed'],'familiar_only_fit':fit})
        # No probe state is passed to fitting/normalization. Mixed is scored only now.
        for cue,q in [('A',a),('B',b),('Mixed',m)]:
            d=decode(fit,q[cols].to_numpy());d['seed']=r['seed'];d['cue']=cue;d['fit_sha']=fit.get('fit_sha');decoded.append(d)
    x=pd.concat(long,ignore_index=True);s=seed_contrast(x);d=pd.concat(decoded,ignore_index=True);f=pd.DataFrame(fitrows)
    for name,value in [('d2_transition_history',x),('d2_history_seed',s),('d2_decoder_transition',d),('d2_decoder_sanity',f)]:save(out,name,value)
    dump(out/'audit/d2_familiar_only_coefficients.json',fits)
    for metric in ['prototype_correct','ridge_correct']:d[metric]=pd.to_numeric(d[metric])
    scores=d.groupby(['seed','cue','prototype_status','ridge_status'],as_index=False)[['prototype_correct','ridge_correct','true_position_distance']].mean();save(out,'d2_decoder_seed',scores)
    sboot=s.copy();sboot['analysis']='fixed_grammar_history_signature';boot=seed_bootstrap(sboot,['history_contrast','history_excess_ce','diagnostic_excess_ce','overall_excess_ce'],['analysis']);save(out,'d2_history_intervals',boot)
    report(out,'D2_SEQUENCE_INFORMATION.md','# D2 — sequence-history signature and readout limits\n\nThe four history-dependent positions are1,3,4,7; token-diagnostic positions2,5,6. We computed each seed’s Mixed−mean(A,B) CE within each mask before inference. Primary history-minus-diagnostic excess CE:\n\n'+table(boot[boot.metric.eq('history_contrast')][['mean','ci_low','ci_high','n_seeds']])+'\n\nThe masks are fixed by grammar, but also confound token, sequence position and difficulty; this is a motivated signature, not a unique measure of lost sequence progress.\n\nBoth decoders use familiar states only: Euclidean14-state prototypes, and seven-class ridge with fixed lambda1, A/B-only population SD floor1e-6 and an unpenalized intercept. A→B and B→A tests each require at least5/7 correct; pooled rank/variance/sample checks are frozen. Prototype sanity passed '+str(int(f.prototype_status.eq('shared_decoder_sanity_passed').sum()))+'/20 seeds; ridge sanity passed '+str(int(f.ridge_status.eq('shared_decoder_sanity_passed').sum()))+'/20. Failed/shared-decoder cases are explicitly nonidentifiable; their raw Mixed scores remain in tables, not interpreted as missing information. Fourteen states provide very limited support even when sanity passes. The decoders are not SRN weights and were never optimized on Mixed labels.\n')
    return {'models':20,'history_rows':len(x),'prototype_shared_sanity_pass':int(f.prototype_status.eq('shared_decoder_sanity_passed').sum()),'ridge_shared_sanity_pass':int(f.ridge_status.eq('shared_decoder_sanity_passed').sum()),'optimizer_steps':0,'ridge_lambda':1.}
