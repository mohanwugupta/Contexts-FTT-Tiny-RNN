"""Frozen source replay plus unchanged historical learning flags; no updates."""
import copy,numpy as np,pandas as pd,torch
from research_search.common import *
from experiments.a06_novel_cue_followup import protocol as source
from experiments.a06_novel_cue_followup.execution import original_cache
from experiments.joint_phenotype_hypothesis_screen.adapter import mathematics
from .scoring import score,PROBES

def run():
    accepted=PACKAGE/'audit/baseline_parity.json'
    if accepted.exists():
        r=read(accepted)
        for n,h in source.load_sources()['inputs'].items():assert sha(ROOT/n)==h,n
        assert r['passed'] and r['models']==12 and r['states']==96
        return r
    source.configure();before={n:sha(ROOT/n) for n in source.load_sources()['inputs']};checks=[];ref=read(source.SCREEN/'reference_thresholds.json');old=pd.read_csv(source.OUT/'learning_measures.csv')
    for seed in range(6):
        for condition in ['A00','A06']:
            if not (source.OUT/f'original_cache/{condition}_s{seed}.npz').is_file():raise FileNotFoundError('Restore exact archived cache; never recreate a sealed source file')
            arr=original_cache(seed,condition);states=source.load_states(seed,condition);index={int(k):i for i,k in enumerate(arr['updates'])};z=np.asarray(arr['logits']);r=score(z,arr['updates'],source.core.TARGET_PATH,seed,condition,ref,sparse=seed<2 and condition=='A06');row=old[old.seed.eq(seed)&old.condition.eq(condition)].iloc[0]
            for field in ['D_A','D_B','own_A','own_B','cross_A','cross_B','Q_A','Q_B','E','E_normalized','C_ce','NoContext_ce','Mixed_ce']:np.testing.assert_allclose(r['summary'][field],row[field],rtol=1e-12,atol=1e-12)
            for field in ['familiar','joint','original_early_phenotype']:assert r['summary'][field]==row[field]
            for n in source.NOVEL_TIMES:
                st=states[n];m=source.source_model(st['model']);before_model=source.digest(m.state_dict());rng=source.rng();obs={c:source.core.trace(m,c) for c in PROBES}
                for j,c in enumerate(PROBES):np.testing.assert_array_equal(obs[c]['logits'],arr['logits'][index[n],j])
                control=mathematics(m,obs,'A00');assert control['passed'];assert source.digest(m.state_dict())==before_model and source.digest(rng)==source.digest(source.rng())
                checks.append(dict(seed=seed,condition=condition,update=n,all5_native_logit_arrays_exact=True,midpoint_and_convexity=control,model_hash=source.digest(st['model'])))
    for n,h in before.items():assert sha(ROOT/n)==h,n
    result=dict(passed=True,models=12,states=96,native_sequence_forwards=480,double_control_sequence_forwards=480,total_sequence_forwards=960,source_function_double_and_native_controls='96 inherited mathematics calls, each includes5 double controls; no optimizer work',learning_measure_checks=180,original_flags_unchanged=True,source_files_unchanged=len(before),new_optimizer_steps=0,checks=checks)
    write(PACKAGE/'audit/baseline_parity.json',result,True);return result
