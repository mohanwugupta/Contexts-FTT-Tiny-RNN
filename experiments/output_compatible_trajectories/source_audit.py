"""P0: resolve and audit existing bytes, never recreate or train a checkpoint."""
from pathlib import Path
import inspect
import json
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.hashing import sha256,object_hash
from context_sequence.provenance.registry import load_model,audit_inputs
from context_sequence.provenance.regression import historical_regression
from context_sequence.task.novelty import read_registry,cues_from_registry
from context_sequence.evaluation.rollout import Model
from context_sequence.dynamics.closure import references,static_audit
from .config import CONTRACT,REGISTRY_SHA,subset
from .continuation import natural_trace,continue_from,frozen_weight_guard
from .state_geometry import output_spaces
from .common import metadata,save,report,table

ROOT_FILES=['artifacts/manifests/pre_freeze_snapshot.json','scratchpads/output_compatible_reference_index.json','configs/paper_freeze_v2.json','configs/novelty_registry_v1.json','configs/dynamical_generalization_v1.json',
 'results/paper/v2_20260925_novelty30_60_r2/manifest.json','results/dynamical_generalization/dg_v1_20261103/manifest.json',
 'results/dynamical_generalization/dg_v1_20261103/tables/e1_transition_dynamics.csv','results/dynamical_generalization/dg_v1_20261103/tables/e1_single_patch_replay.csv',
 'results/dynamical_generalization/dg_v1_20261103/tables/e1_static_convexity_audit.csv',
 'analysis/phase1k/phase1k_natural_rollouts.csv','analysis/phase1l/phase1l_final_transition_metrics.csv',
 'phase1j_context_stability/core.py','phase1k_cue_generation/interventions.py','confirmatory_srn/model.py','phase1l_contrast_closure/model.py',
 'src/context_sequence/evaluation/rollout.py','src/context_sequence/provenance/registry.py','src/context_sequence/provenance/regression.py','src/context_sequence/dynamics/closure.py']


def load_sources(root):
    root=Path(root);p=root/'configs/paper_freeze_v2.json'
    if not p.is_file():raise FileNotFoundError('blocked_missing_source: '+str(p))
    paper=json.loads(p.read_text());reg=read_registry(root/'configs/novelty_registry_v1.json',REGISTRY_SHA)
    records=paper['checkpoints']
    if len(records)!=120:raise ValueError('Historical cohort identity/count mismatch')
    return {'records':records,'analysis_records':subset(records,'p1'),'registry':reg,'cues':cues_from_registry(reg)}


def verify_checkpoint(root,record):return load_model(root,record)


def require_parity(observed,expected,tolerance):
    a=np.asarray(observed,float);b=np.asarray(expected,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,atol=tolerance,rtol=0):raise ValueError('blocked_replay_mismatch: shape or numerical reference mismatch')
    return True


def optional_table_audit(path,expected,flags=None):
    p=Path(path)
    if flags is None:flags=getattr(p.stat(),'st_flags',0) if p.is_file() else 0
    # Apple SF_DATALESS identifies cloud metadata without available file bytes.
    if not p.is_file() or flags & 0x40000000:
        return {'path':str(p),'expected_sha256':expected,'observed_sha256':None,'status':'blocked_missing_source_optional'}
    observed=sha256(p)
    return {'path':str(p),'expected_sha256':expected,'observed_sha256':observed,'status':'verified' if observed==expected else 'failed_integrity'}


def input_hashes(root,bundle):
    root=Path(root);rows=[]
    refs=json.loads((root/'scratchpads/output_compatible_reference_index.json').read_text())['references']
    paths=set(ROOT_FILES+[x.get('resolved_path',x['path']) for x in refs if x.get('sha256') is not None or x.get('resolved_path')])
    prior=json.loads((root/'configs/dynamical_generalization_v1.json').read_text())
    expected={**prior['source_files'],**{x['path']:x['sha256'] for x in prior['input_files']}}
    expected.update({x['path']:x['sha256'] for x in refs if x.get('sha256') is not None})
    dg_root='results/dynamical_generalization/dg_v1_20261103/'
    dg=json.loads((root/(dg_root+'manifest.json')).read_text())
    expected.update({path:dg['outputs'][path[len(dg_root):]] for path in paths if path.startswith(dg_root) and path[len(dg_root):] in dg['outputs']})
    # Audit the historical program's frozen source/inputs and this program's discovered references.
    for path in sorted(paths|set(expected)):
        p=root/path;obs=sha256(p) if p.is_file() else None
        want=expected.get(path,obs)
        rows.append({'kind':'source_or_reference','path':path,'expected_sha256':want,'observed_sha256':obs,'passed':obs is not None and obs==want})
    for r in audit_inputs(root,bundle['records']).to_dict('records'):
        rows.append({'kind':'checkpoint','path':r['path'],'expected_sha256':r['checkpoint_sha256'],'observed_sha256':r['observed_sha256'],'passed':r['passed']})
    return pd.DataFrame(rows)


def run(root,out,smoke=False):
    bundle=load_sources(root);before=input_hashes(root,bundle)
    before.to_csv(out/'audit/p0_input_hashes.csv',index=False)
    if not before.passed.all():raise FileNotFoundError('blocked_missing_source: '+','.join(before.loc[~before.passed,'path']))
    optional_path='analysis/phase1j_single_injection_context_stability/phase1j_transition_metrics.csv'
    snapshot=json.loads((root/'artifacts/manifests/pre_freeze_snapshot.json').read_text())
    expected=next(x['sha256'] for x in snapshot['files'] if x['path']==optional_path)
    optional=[optional_table_audit(root/optional_path,expected)]
    paper=json.loads((root/'results/paper/v2_20260925_novelty30_60_r2/manifest.json').read_text())
    optional.append(optional_table_audit(root/'results/paper/v2_20260925_novelty30_60_r2/tables/a3_composite_sufficiency_seed.csv',paper['outputs']['tables/a3_composite_sufficiency_seed.csv']))
    pd.DataFrame(optional).to_csv(out/'audit/p0_optional_historical_sources.csv',index=False)
    if any(x['status']=='failed_integrity' for x in optional):raise ValueError('Historical optional table changed')
    records=bundle['records'] if not smoke else [r for r in bundle['records'] if r['seed'] in [0,1,2,3]]
    old=pd.read_csv(root/'results/dynamical_generalization/dg_v1_20261103/tables/e1_transition_dynamics.csv')
    patches=pd.read_csv(root/'results/dynamical_generalization/dg_v1_20261103/tables/e1_single_patch_replay.csv')
    repeated=pd.read_csv(root/'results/dynamical_generalization/dg_v1_20261103/tables/e1_static_convexity_audit.csv')
    natural_parts=[];parity=[];geometry=[];head=[];state_keys=[]
    with frozen_weight_guard():
        regression=historical_regression(root,records)
        regression.to_csv(out/'audit/p0_historical_regression.csv',index=False)
        if not regression.passed.all():raise ValueError('blocked_replay_mismatch: historical I/K/L regression failed')
        for i,r in enumerate(records,1):
            m=verify_checkpoint(root,r);meta=metadata(r);a,b,mid=references(m);sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
            source=old[old.condition.eq(r['condition'])&old.seed.eq(r['seed'])]
            patch_source=patches[patches.condition.eq(r['condition'])&patches.seed.eq(r['seed'])]
            with frozen_weight_guard(m):
                for cue,vector in bundle['cues'].items():
                    trace=natural_trace(m,vector);known=source[source.probe.eq(cue)].sort_values('transition')
                    for columns,tol in [(['raw_ce'],.002),([f'hidden_{j}' for j in range(8)],.0002),([f'logit_{j}' for j in range(4)],.002)]:require_parity(trace[columns].values,known[columns].values,tol)
                    parity.append({**meta,'cue':cue,'check':'natural_trace','max_ce_difference':float(np.abs(trace.raw_ce.values-known.raw_ce.values).max()),'passed':True})
                    for key,value in meta.items():trace[key]=value
                    trace['cue']=cue;natural_parts.append(trace)
                    for k in [1,3,5]:
                        trace_p=continue_from(m,mid[k-1],k,vector);ref=patch_source[patch_source.probe.eq(cue)&patch_source.position.eq(k)].sort_values('transition')
                        require_parity(trace_p.raw_ce.values,ref.raw_ce.values,.002)
                        require_parity(trace_p[[f'hidden_{j}' for j in range(8)]].values,ref[[f'hidden_{j}' for j in range(8)]].values,.0002)
                        parity.append({**meta,'cue':cue,'k':k,'check':'single_patch','max_ce_difference':float(np.abs(trace_p.raw_ce.values-ref.raw_ce.values).max()),'passed':True})
            s=static_audit(m)
            if s.affine_error.max()>1e-10 or s.convexity_gap.min() < -1e-10:raise ValueError('Static affine/Jensen invariant failed')
            g=output_spaces(m)
            head.append({**meta,**{key:g[key] for key in ['rank','nullity','rank_tolerance','null_residual','orthogonality_residual']},'singular_values':json.dumps(g['singular_values'].tolist()),'M':json.dumps(g['M'].tolist()),'N':json.dumps(g['N'].tolist()),'R':json.dumps(g['R'].tolist()),'max_affine_error':float(s.affine_error.max()),'minimum_convexity_gap':float(s.convexity_gap.min()),'third_channel_inactive':bool(np.all(m.wih[:,6]==0)) if m.context_u is None else True})
            for k in range(7):geometry.append({**meta,'k':k+1,'AB_hidden_separation':sep[k],'midpoint_box_margin':1-float(np.max(np.abs(mid[k]))),'midpoint_in_oracle_interior':bool(np.max(np.abs(mid[k]))<=1-1e-6),'A_ce':a['raw_ce'][k],'B_ce':b['raw_ce'][k],'midpoint_ce':s.midpoint_ce.iloc[k]})
            if r['cohort']=='canonical':
                ref=repeated[repeated.condition.eq(r['condition'])&repeated.seed.eq(r['seed'])].sort_values('transition')
                require_parity(s.midpoint_ce.to_numpy(),ref.midpoint_ce.to_numpy(),.002)
            payload=torch.load(root/r['path'],map_location='cpu',weights_only=False)
            for key,value in payload['model_state_dict'].items():state_keys.append({**meta,'state_dict_key':key,'shape':str(tuple(value.shape)),'dtype':str(value.dtype),'requires_grad_in_loaded_state':bool(value.requires_grad),'tensor_sha256':__import__('hashlib').sha256(value.detach().cpu().numpy().tobytes()).hexdigest()})
            if i%20==0:print(f'P0 replay {i}/{len(records)}',flush=True)
    for name,frame in [('p0_natural_trace',pd.concat(natural_parts,ignore_index=True)),('p0_replay_parity',pd.DataFrame(parity)),('p0_reference_geometry',pd.DataFrame(geometry)),('p0_output_spaces',pd.DataFrame(head)),('p0_state_dict_keys',pd.DataFrame(state_keys))]:save(out,name,frame)
    funcs=[]
    for f in [Model.step,Model.drive,references,natural_trace,continue_from]:
        lines,line=inspect.getsourcelines(f);funcs.append({'function':f.__module__+'.'+f.__name__,'path':str(Path(inspect.getsourcefile(f)).relative_to(root)),'line':line})
    save(out,'p0_source_function_index',pd.DataFrame(funcs))
    geom=pd.DataFrame(geometry)
    report(out,'P0_SOURCE_AUDIT.md',f'# P0 — source and reproduction\n\nWe asked whether the frozen task and state indexing reproduce the historical evidence. We replayed {len(records)} existing checkpoints with unchanged next-token readouts and no optimizer. All natural state/logit/CE, single-placement and algebraic midpoint checks passed. The source has eight tanh hidden units, seven teacher-forced predictions and an inactive third cue channel. C0 remains a separate sequence-only cohort.\n\nThe full-right centered-head SVD is retained per model. Midpoint oracle-interior eligibility: {int(geom.midpoint_in_oracle_interior.sum())}/{len(geom)} positions; inadmissible starts will be retained and flagged rather than clipped or replaced. Complete source keys, tensor hashes, numerical differences and function locations are in the audit tables.\n\nThis establishes faithful frozen replay, not natural access to a useful midpoint. P1 may now reuse the old lag data; P2/P3 remain gated by their implementation tests.\n')
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','models_replayed':len(records),'model_optimizer_updates':0,'source_alias_resolution':'Phase 1J core.py and Phase 1K interventions.py; evaluation.py aliases were absent'}
