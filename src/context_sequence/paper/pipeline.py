from contextlib import contextmanager
from unittest.mock import patch
import torch


@contextmanager
def optimizer_firewall():
    def denied(*args,**kwargs):raise RuntimeError('Optimizer/backward forbidden in evaluation_only mode')
    with patch.object(torch.optim.Optimizer,'__init__',denied),patch.object(torch.optim.Adam,'step',denied),patch.object(torch.optim.SGD,'step',denied),patch.object(torch.Tensor,'backward',denied):
        yield

from pathlib import Path
import json
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256,object_hash,source_hash
from context_sequence.provenance.registry import checkpoint_records,audit_inputs,load_model
from context_sequence.provenance.regression import historical_regression
from context_sequence.task.novelty import read_registry,cues_from_registry
from context_sequence.evaluation.rollout import cue_projection
from context_sequence.experiments.engine import evaluate_canonical,evaluate_constructive,seed_summary,lag_summary
from context_sequence.experiments.matching import match_acquisition
from context_sequence.statistics.bootstrap import bootstrap_indices
from .contrasts import all_contrasts,geometry_behavior


def load_contract(path):
    path=Path(path).resolve();config=json.loads(path.read_text());root=path.parent.parent
    content={k:v for k,v in config.items() if k!='contract_sha256'}
    if object_hash(content)!=config['contract_sha256']:raise ValueError('Paper contract hash mismatch')
    if not config['evaluation_only'] or config['seeds']!=list(range(20)):raise ValueError('Frozen evaluation contract changed')
    if source_hash(root)!=config['source_code_sha256']:raise ValueError('Implementation changed since prospective paper contract freeze')
    return root,config


def validate_keys(frame):
    keys=[k for k in ['cohort','condition','seed','probe','intervention','continuation','analysis_kind','position','transition'] if k in frame]
    if frame.duplicated(keys).any():raise ValueError('Nonunique paper table keys: '+str(keys))


def run(config_path,output=None):
    root,config=load_contract(config_path);registry=read_registry(root/config['novelty_registry'],config['novelty_registry_sha256'])
    cues=cues_from_registry(registry);records=config['checkpoints'];out=Path(output) if output else root/'results/paper'/config['freeze_id']
    if out.exists():raise FileExistsError('Freeze outputs are immutable; choose a fresh output directory')
    for sub in ['tables','figures','captions','audit']:(out/sub).mkdir(parents=True,exist_ok=True)
    common={'registry_sha256':registry['registry_sha256'],'source_code_sha256':config['source_code_sha256'],'contract_sha256':config['contract_sha256'],'evaluator':config['evaluator'],'numerical_dtype':config['numerical_dtype']}
    statuses=[];tables={}
    def save(name,frame):
        frame=frame.copy()
        for k,v in common.items():frame[k]=v
        frame.to_csv(out/'tables'/f'{name}.csv',index=False);tables[name]=frame
    audited=audit_inputs(root,records);audited.to_csv(out/'audit/checkpoint_hashes_before.csv',index=False)
    # Hash every declared supporting input before reading it.
    for entry in config['input_tables']:
        if sha256(root/entry['path'])!=entry['sha256']:raise ValueError('Frozen source table changed: '+entry['path'])
    # Selection is completed and saved before any new cue enters model evaluation.
    timeline=pd.read_csv(root/'analysis/phase1i_context_composition/phase1i_timeline_metrics.csv')
    candidates,matches=match_acquisition(timeline);save('a2_acquisition_match_candidates',candidates);save('a2_acquisition_matches',matches)
    primary=matches[matches.primary].copy()
    primary['evidence_status']='insufficient_common_support';primary['reason']='Intermediate matched model weights are not saved; no optimizer replay permitted.'
    save('a2_acquisition_robustness_seed',primary)
    statuses.append({'analysis':'A2','implementation_status':'validated','evidence_status':'insufficient_common_support','n_seeds':len(primary),'exclusions':f'{20-len(primary)} seeds lack primary familiar support; intermediate matched checkpoints unavailable',
                     'result':'Familiar-only matching completed; no new-Novel matched contrast is estimable from available frozen weights.'})
    (out/'audit/matching_frozen.json').write_text(json.dumps({'matching_rule':config['matching'],'matches_sha256':sha256(out/'tables/a2_acquisition_matches.csv'),'before_new_novel_evaluation':True},indent=2))
    audit_rows=[]
    for name,row in registry['cues'].items():audit_rows.append({'probe':name,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}})
    save('novelty_registry_audit',pd.DataFrame(audit_rows))
    index=bootstrap_indices();np.save(out/'audit/bootstrap_indices.npy',index)
    with optimizer_firewall():
        available=[r for r in records if bool(audited[audited.path.eq(r['path'])].passed.iloc[0])]
        # No technical regression failure may reach new-cue descendants.
        regression=historical_regression(root,available);regression.to_csv(out/'audit/evaluator_regression.csv',index=False)
        if not regression.passed.all():raise ValueError('Historical evaluator regression failed')
        for cohort,analyses in [('canonical',['A1','A3','A4','A5']),('constructive',['A6','A7','A8'])]:
            cohort_records=[r for r in records if r['cohort']==cohort]
            if any(not bool(audited[audited.path.eq(r['path'])].passed.iloc[0]) for r in cohort_records):
                for a in analyses:statuses.append({'analysis':a,'implementation_status':'validated','evidence_status':'blocked_missing_or_changed_artifact','n_seeds':0,'exclusions':'cohort incomplete','result':'Affected cohort evaluation not run'})
                continue
            collected={};cue_rows=[]
            for number,r in enumerate(cohort_records,1):
                model=load_model(root,r)
                print(f'{cohort} {number}/{len(cohort_records)}: {r["condition"]} seed {r["seed"]}',flush=True)
                batch=evaluate_canonical(model,cues,r) if cohort=='canonical' else {'a6_constructive_transition':evaluate_constructive(model,cues,r)}
                for name,f in batch.items():collected.setdefault(name,[]).append(f)
                if cohort=='constructive':
                    for name,cue in cues.items():
                        alpha,res=cue_projection(cue);drive=model.drive(cue)
                        cue_rows.append({**r,'probe':name,'raw_vector':json.dumps(cue.tolist()),**registry['cues'][name],
                                         'cue_alpha':alpha,'projection_magnitude':abs(alpha)*np.sqrt(.5),'discarded_component':json.dumps(res.tolist()),'discarded_norm':float(np.linalg.norm(res)),
                                         'effective_context_drive':json.dumps(drive.tolist()),'effective_drive_norm':float(np.linalg.norm(drive))})
            for name,parts in collected.items():
                frame=pd.concat(parts,ignore_index=True);validate_keys(frame);save(name,frame)
                summary=seed_summary(frame);save(name.replace('_transition','_seed'),summary)
                if name.startswith(('a3','a4','a5')):save(name.replace('_transition','_lag_seed'),lag_summary(frame[frame.analysis_kind.ne('local')]))
            if cohort=='constructive':
                save('a7_geometry_behavior',geometry_behavior(tables['a6_constructive_seed']));save('a8_effective_cue_audit',pd.DataFrame(cue_rows))
                # Exact behavioral aliases follow from identical effective coordinates.
                s=tables['a6_constructive_seed']
                for condition in ['C1_contrast_only','C3_full']:
                    g=s[s.condition.eq(condition)].set_index(['seed','probe'])
                    for left,right in [('Mixed','NoContext'),('Novel30_A','Novel60_A'),('Novel30_B','Novel60_B')]:
                        for metric in ['raw_ce','task_distance','task_residual']:
                            if not np.array_equal(g.xs(left,level='probe')[metric],g.xs(right,level='probe')[metric]):raise AssertionError('Effective cue alias invariant failed')
            for a in analyses:statuses.append({'analysis':a,'implementation_status':'validated','evidence_status':'completed','n_seeds':20,'exclusions':'none','result':'All declared frozen conditions evaluated; scientific sign does not gate completion.'})
    effects,bootstrap=all_contrasts(tables,index);save('paper_primary_contrasts',effects);save('paper_bootstrap_summary',bootstrap)
    after=audit_inputs(root,records);after.to_csv(out/'audit/checkpoint_hashes_after.csv',index=False)
    if not after.observed_sha256.equals(audited.observed_sha256):raise RuntimeError('Checkpoint bytes changed')
    # Immutable historical evidence copied into explicit supplementary tables.
    acq=pd.read_csv(root/'analysis/phase1i_context_composition/phase1i_true_acquisition.csv');save('s1_genuine_acquisition',acq)
    legacy=pd.read_csv(root/'analysis/phase1l/phase1l_novel_within_seed.csv');legacy['historical_probe']=legacy.probe;legacy['probe']=legacy.probe.str.replace('Novel','LegacyNovel',regex=False);save('s2_legacy_constructive',legacy)
    oldmatch=pd.read_csv(root/'outputs/transfer_experiment_program/e1_acquisition/strict_match_outcomes.csv');oldmatch['cohort']='historical_ordinal_layout';save('s3_historical_matching',oldmatch)
    neutral=pd.read_csv(root/'analysis/phase1l/phase1l_neutral_rollout_metrics.csv');neutral['cohort']='constructive';save('s4_historical_neutral_closure',neutral)
    status=pd.DataFrame(statuses).sort_values('analysis');status.to_csv(out/'audit/analysis_status.csv',index=False)
    from .figures import make_figures
    from .report import write_report
    claims=make_figures(out,tables,config);save('claim_source_map',claims)
    write_report(out,tables,status,config)
    output_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
    manifest={'freeze_id':config['freeze_id'],**common,'evaluation_only':True,'optimizer_updates':0,'registry_frozen_before_evaluation':True,
              'checkpoints':records,'input_tables':config['input_tables'],'bootstrap_indices_sha256':sha256(out/'audit/bootstrap_indices.npy'),
              'analysis_status':status.to_dict('records'),'outputs':output_hashes,'figure_panels':claims.to_dict('records')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
