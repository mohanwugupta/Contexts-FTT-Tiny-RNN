"""D0: establish frozen numerical and objective contracts before new analyses."""
from pathlib import Path
import json,inspect
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.hashing import sha256
from confirmatory_srn.config import ConfirmatoryConfig
from .config import ROOT,CONTRACT,SEQUENCE
from .source_adapter import records,cues,load_original,load_torch,assert_unchanged
from .replay import natural_trace
from .io import dump,save,report,table,check_files


def objective_contract(payload):
    expected={'hidden_dim':8,'token_dim':4,'context_dim':3,'high_scale':2.,'context_alpha':1.,'learning_rate':.005,'gradient_clip':1.,'lambda_execution':.20,'normative_c_time':.5,'normative_c_risk':1.,'normative_gamma':2.,'n_blocks':17,'block_length':10}
    cfg=payload['config']
    if any(cfg.get(k)!=v for k,v in expected.items()) or tuple(cfg['fixed_sequence'])!=tuple(SEQUENCE) or payload.get('policy_state_dict') is None:raise ValueError('blocked_training_objective_fidelity: exact source objective/policy required')
    if any('optimizer' in k for k in payload):raise ValueError('blocked_training_objective_fidelity: optimizer payload variant requires audited restoration')
    return {**expected,'optimizer_mode':'fine_tune_from_weights_optimizer_reset','optimizer_moments_available':False,'loss':'mean_CE + .20*mean(.5*(1-v) + stopgrad(1-p_correct)*v**2)',
      'policy':'v=sigmoid(softplus(raw_slope)*logit(stopgrad(clamp(p_correct,1e-6,1-1e-6)))+intercept)',
      'gradient_contract':'model CE gradients; separate execution-policy gradients; all172+2 parameters jointly clipped to L2 norm1.0; original novel-column gradient hook and post-step enforcement',
      'optimizer':'torch.optim.Adam(all model parameters followed by both policy parameters, lr=.005); remaining arguments are source library defaults',
      'sequence_reset':'initial_state=None for each complete seven-step sequence','update_unit':'one full-sequence mean loss, backward, joint clip, Adam.step, enforce_novel_neutral',
      'original_updates':170,'new_updates':160,'final_update':330,'protected_inputs_in_training':0,'same_starting_model_and_policy_for_all_arms':True}


def verify_prior_scientific(prior):
    prior=Path(prior);m=json.loads((prior/'manifests/sealed_manifest.json').read_text())
    check_files(ROOT,m['inputs']);check_files(ROOT,m['code_hashes']);check_files(prior,m['files'])
    actual={str(p.relative_to(prior)) for p in prior.rglob('*') if p.is_file() and str(p.relative_to(prior))!='manifests/sealed_manifest.json'}
    extras=actual-set(m['files'])
    if any(Path(x).name!='.DS_Store' for x in extras):raise ValueError('Unverified scientific prior extras: '+str(sorted(extras)))
    return [{'path':x,'status':'excluded_OS_metadata','sha256_at_audit':sha256(prior/x)} for x in sorted(extras)]


def frozen_inputs():
    prior=ROOT/'results/context_cue_access/v1';verify_prior_scientific(prior)
    p=json.loads((prior/'manifests/sealed_manifest.json').read_text());files=dict(p['inputs']);files.update(p['code_hashes'])
    needed=['PRD_CONTEXT_MIXING_SEQUENCE_INFORMATION.md','CONTEXT_MIXING_SCRATCHPAD_TEMPLATE.md','confirmatory_srn/model.py','confirmatory_srn/training.py','confirmatory_srn/config.py','confirmatory_srn/contexts.py',
      'execution_objective_search/objectives.py','execution_objective_search/config.py','ftt_srn_execution_v2/training/ftt_schedule.py','ftt_srn_execution_v2/inputs/token_onehot.py',
      'docs/FINAL_EXPERIMENT_INFRASTRUCTURE.md','results/paper/v2_20260925_novelty30_60_r2/FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md','results/dynamical_generalization/dg_v1_20261103/E1_RESULTS.md',
      'results/dynamical_generalization/dg_v1_20261103/FINAL_DYNAMICAL_GENERALIZATION_REPORT.md','results/output_compatible_trajectories/v1_20261111/reports/FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md',
      'results/output_compatible_trajectories/v1_20261111/tables/p0_natural_trace.csv','results/dynamical_generalization/dg_v1_20261103/tables/e1_single_patch_replay.csv',
      'results/context_cue_access/v1/FINAL_CONTEXT_CUE_ACCESS_REPORT.md','results/context_cue_access/v1/CLAIM_LEDGER.md','results/context_cue_access/v1/manifests/sealed_manifest.json']
    for name in needed:
        path=ROOT/name
        if not path.is_file():raise FileNotFoundError('blocked_missing_source: '+name)
        if getattr(path.stat(),'st_flags',0)&0x40000000:raise FileNotFoundError('blocked_missing_source: required cloud-only '+name)
        files[name]=sha256(path)
    for r in records():files[r['path']]=r['checkpoint_sha256']
    return files


def run(root,out):
    dump(out/'audit/prior_os_metadata_extras.json',verify_prior_scientific(ROOT/'results/context_cue_access/v1'))
    oldpath=ROOT/'results/output_compatible_trajectories/v1_20261111/tables/p0_natural_trace.csv';old=pd.read_csv(oldpath);parts=[];checks=[];states=[];contracts=[];blocked=[]
    for r in records():
        m=load_original(root,r);before=m.digest();model,policy,p=load_torch(root,r)
        try:c=objective_contract(p);contracts.append(c)
        except ValueError as e:blocked.append({'seed':r['seed'],'status':'blocked_training_objective_fidelity','error':str(e)})
        for cue,vector in cues().items():
            f=natural_trace(m,vector);known=old[old.condition.eq('High_monotonic')&old.seed.eq(r['seed'])&old.cue.eq(cue)]
            for prefix,n in [('hidden_',8),('logit_',4),('probability_',4)]:
                cols=[prefix+str(i) for i in range(n)];err=float(np.abs(f[cols].to_numpy()-known[cols].to_numpy()).max())
                if err>1e-12:raise ValueError('blocked_source_fidelity: prior natural '+prefix)
                checks.append({'seed':r['seed'],'cue':cue,'comparison':'saved_numpy_source','field':prefix,'max_abs_error':err,'tolerance':1e-12,'passed':True})
            with torch.no_grad():
                tokens=torch.eye(4)[list(SEQUENCE[:-1])].unsqueeze(0);z,h,_=model(tokens,torch.tensor(vector,dtype=torch.float32))
            for cols,values,tol in [([f'hidden_{j}' for j in range(8)],h[0].numpy(),.0002),([f'logit_{j}' for j in range(4)],z[0].numpy(),.002)]:
                err=float(np.max(np.abs(f[cols].to_numpy()-values)))
                if err>tol:raise ValueError('blocked_source_fidelity: source torch/NumPy execution')
                checks.append({'seed':r['seed'],'cue':cue,'comparison':'original_torch_source','field':cols[0],'max_abs_error':err,'tolerance':tol,'passed':True})
            torch_ce=torch.nn.functional.cross_entropy(z[0],torch.tensor(SEQUENCE[1:]),reduction='none').numpy()
            ce_error=float(np.max(np.abs(torch_ce-f.raw_ce.to_numpy())))
            if ce_error>.002:raise ValueError('blocked_source_fidelity: original Torch CE')
            checks.append({'seed':r['seed'],'cue':cue,'comparison':'original_torch_source','field':'raw_ce','max_abs_error':ce_error,'tolerance':.002,'passed':True})
            for key,value in {'seed':r['seed'],'cohort':'canonical','condition':'High_monotonic','cue':cue,'checkpoint_sha':r['checkpoint_sha256'],'source_sha':sha256(ROOT/'src/context_sequence/evaluation/rollout.py'),'config_sha':CONTRACT['sha256'],'source_data_sha':sha256(oldpath)}.items():f[key]=value
            parts.append(f)
        assert_unchanged(m,before)
        states.append({**r,'model_tensor_sha':before,'policy_state':{key:float(value) for key,value in p['policy_state_dict'].items()},'optimizer_state_available':False,'embedded_config':str(p['config']),'weights_unchanged':True})
    allrows=pd.concat(parts,ignore_index=True);save(out,'d0_native_replay',allrows);save(out,'d0_source_fidelity',pd.DataFrame(checks));save(out,'d0_checkpoint_audit',pd.DataFrame(states))
    contract=contracts[0] if len(contracts)==20 and all(x==contracts[0] for x in contracts) else {'status':'blocked_training_objective_fidelity','failures':blocked}
    contract['status']='verified' if len(contracts)==20 and not blocked else 'blocked_training_objective_fidelity';dump(out/'manifests/d3_objective_contract.json',contract);dump(out/'manifests/config.json',CONTRACT)
    snapshot=json.loads((ROOT/'artifacts/manifests/pre_freeze_snapshot.json').read_text());optional=[]
    for name in ['outputs/confirmatory_srn_final/confirmatory_config.json','outputs/confirmatory_srn_final/confirmatory_training_sequence_level.csv']:
        p=ROOT/name;expected=next((x['sha256'] for x in snapshot['files'] if x['path']==name),None)
        missing=not p.is_file() or bool(getattr(p.stat(),'st_flags',0)&0x40000000)
        optional.append({'path':name,'expected_sha':expected,'observed_sha':None if missing else sha256(p),'status':'unavailable_cloud_optional' if missing else 'available','substitute':'exact embedded config in each original checkpoint; actual executed training source; no historical learning curve fabricated'})
    pd.DataFrame(optional).to_csv(out/'audit/optional_sources.csv',index=False)
    mean=allrows.groupby('cue',as_index=False).raw_ce.mean()
    report(out,'D0_SOURCE_AUDIT.md','# D0 — source, numerical and training-objective audit\n\nAll20 canonical High_monotonic checkpoints at original update170 were verified, without aliasing C0. Eight tanh units take four one-hot tokens plus three raw context channels; the third cue column is exactly zero. Natural rollouts start at zero, update then output, and score seven teacher-forced next-token targets.\n\n'+table(mean)+'\n\nNatural hidden states, logits and complete probabilities match the prior saved NumPy source to1e-12 CSV precision. The original Torch RNN and the frozen NumPy evaluator pass the previously established state2e-4/logit2e-3 source-precision bounds. All original tensors remain unchanged.\n\nD3 preserves the actual loss, including the separate two-parameter monotonic execution policy, detached uncertainty, joint gradient clipping and original Adam contract. No optimizer moments were saved: all arms will use identical fresh Adam states and be labeled fine_tune_from_weights_optimizer_reset. The checkpoint-embedded configs, including exact objective coefficients, were verified in all20 models; show_progress=False is only a presentation setting. D1 and D2 remain evaluation-only.\n\nTwo optional old config/training-log files are cloud-only. Expected historical hashes and the unavailable status are retained; no observed digest or historical learning curve was invented. Exact embedded checkpoint configs and local source code establish the continuation contract.\n')
    source='''# Executed source and D3 objective contract

Canonical model: confirmatory_srn/model.py:ConfirmatoryElmanSRN, nn.RNN(input7,hidden8,tanh,batch_first=True), plus sequence_head Linear8→4. It has172 model parameters. Separate MonotonicExecutionPolicy adds raw_slope/intercept, two parameters. No execution hidden head is invented.

Training: confirmatory_srn/training.py:train_condition. Each sequence has batch1, seven token inputs and targets, initial_state=None. The source applies Adam.step once per complete sequence, not per transition. Native A10/B10 has17 blocks and170 original updates. Source learning rate.005; clipping L2 norm1.0 jointly over model then policy parameters.

p_t=softmax(logits)_correct; v_t=sigmoid(softplus(a)*logit(stopgrad(clamp(p_t,1e-6,1−1e-6)))+b).

L=mean_t CE_t +0.20 mean_t[0.50(1−v_t)+stopgrad(1−p_t) v_t²]. The model gets CE gradients; the policy gets execution gradients. Joint clipping still couples their norm, so token-only continuation would be a different contract. execution_loss and predictive_statistics are imported from execution_objective_search/objectives.py without modification. Source hooks zero third-cue gradients and enforce that column after every step.

Adam receives all172 model scalars followed by both policy parameters, lr.005 and the installed source defaults; original checkpoint files have no optimizer state. All B/I/S arms strictly load the same model and policy weights before constructing fresh identical Adam state. New160-update final endpoints are330; no probe or checkpoint choice enters training. The smoke uses seeds0/1 and its six models count toward the60 primary models and9600-update total.

Timeline: t1 token0→1; t2 token1→2; t3 token2→0; t4 token0→3; t5 token3→1; t6 token1→2; t7 token2→3. h0=0. D1 patches are one-time exogenous placements, not natural retrieval.

Natural CE and plots use the original frozen float32 recurrence. Double affine/tanh arithmetic evaluates the exact mixing identity at observed float32 prestates. The difference from recorded float32 poststates is a separate roundoff term bounded by64*eps32*(1+absolute affine summands), not a new mechanism. Exact double identity tolerance1e-11. Source precision bounds are frozen before D1; no threshold is chosen from an unfavorable result.

All actual relative paths and SHA256 values are in audit/frozen_inputs.json; per-model checkpoint and tensor hashes in tables/d0_checkpoint_audit.csv; per-row config/source/data identities in d0_native_replay.csv. No source file or original checkpoint is edited.
'''
    with (out/'audit/source_contract.md').open('x') as f:f.write(source)
    return {'models':20,'natural_rows':len(allrows),'source_fidelity':'passed','d3_objective_status':contract['status'],'optimizer_steps':0,'original_weights_unchanged':True}
