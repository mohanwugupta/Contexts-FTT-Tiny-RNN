"""Frozen scientific tail; AST equivalence checked against selected v1."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    from .contract import sha as digest
    return digest(path)

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def compute(args, out, manifest):
    from experiments.research_search_cloud_v1.port import verify
    verify(ROOT,manifest)
    import jsonschema,pandas as pd,numpy as np,torch
    from unittest.mock import patch
    from research_search.common import read
    from experiments.a06_novel_cue_followup.verify import no_scientific_execution
    from experiments.research_search_integration_v1_1.protocol import schema,validate_proposal
    selection=read(args.selection);proposal=read(args.proposal)
    jsonschema.validate(selection,schema('coordinator_select'));validate_proposal(proposal,'initialization')
    if selection['selected']!=proposal['id'] or proposal['analysis']!='confidence_vs_correctness':raise ValueError('Wrong selected proposal')
    from experiments.research_search_integration_v1_1.cached import execute
    from experiments.cloud_confidence_selected_v1.analysis import TIMES,tables,seed_intervals
    request=out/'request';request.mkdir();save(request/'definition.json',dict(sources=manifest['native_consumed_inputs']))
    def forbidden(*a,**k):raise RuntimeError('No checkpoint/model loading authorized')
    with no_scientific_execution(),patch.object(torch,'load',forbidden):
        execute(request,'confidence_vs_correctness',out/'native')
        source=ROOT/'research_search/database/controlled_pilot_v2';frames=[]
        tasks={t['task_id']:t['path'] for t in read(source/'development.json')['tasks']}
        for condition,jobs in read(source/'paired_jobs.json').items():
            for job in jobs:
                p=source/'jobs'/job;meta=read(p/'evaluation.json')
                for view,name in [('native','positions.csv'),('quarter','static_readout_positions.csv')]:
                    f=pd.read_csv(p/name,float_precision='round_trip');f=f[f['update'].isin(TIMES)].copy()
                    path=tasks[meta['task_id']]
                    if not np.array_equal(f.target.to_numpy(),np.asarray(path)[f.position.to_numpy(int)]) or not np.array_equal(f.token.to_numpy(),np.asarray(path)[f.position.to_numpy(int)-1]):raise ValueError('Wrong frozen token/target indexing')
                    frames.append(f.assign(condition=condition,seed=meta['seed'],task_id=meta['task_id'],view=view))
        observations=pd.concat(frames,ignore_index=True);products=tables(observations)
        observations.to_csv(out/'fixed_time_positions.csv',index=False)
        for name,f in products.items():f.to_csv(out/(name+'.csv'),index=False)
        for name,dims in [('within_readout',['condition','update','cue']),('initialization',['view','update','cue'])]:
            means,intervals=seed_intervals(products[name+'_metrics'],dims)
            means.to_csv(out/(name+'_seed_means.csv'),index=False);intervals.to_csv(out/(name+'_seed_intervals.csv'),index=False)
    from experiments.cloud_confidence_selected_v1.figures import render
    render(out)
    verify(ROOT,manifest)
    save(out/'RESULT.json',dict(status='completed_cached_analysis',selected=selection['selected'],proposal_sha256=sha(args.proposal),selection_sha256=sha(args.selection),fixed_time_positions=len(observations),paired_position_rows_per_comparison=4480,independent_seeds=2,new_model_forwards=0,new_optimizer_steps=0,native_summary='native/evaluation_result.json',scientific_scope='Readout expression and saved developmental comparisons only; no natural retrieval or competent-prior mechanism established.',full_parent_agent_isolation_certified=False))

