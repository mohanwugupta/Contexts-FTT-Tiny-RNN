"""Verify/reproduce saved table arithmetic only. No worker/model/optimizer run."""
import argparse
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ARCHIVE_SHA256 = '26a1d5457b08c9e3633ca291142eafd70428df77f2131e8863833da57dbafb9a'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ROOT/'artifacts/selected-analysis-results.zip')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-root', type=Path, help='Optional restored original input root for all139 native-source hashes')
    args = p.parse_args()
    if args.out.exists():raise ValueError('Use a new output directory')
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Unexpected result archive')
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from experiments.cloud_confidence_selected_v1.analysis import validate, tables, seed_intervals
    from experiments.research_search_integration_v1_1.evaluation import four_outcome
    verified = {}
    with zipfile.ZipFile(args.archive) as z:
        names=z.namelist()
        if len(names)!=38 or len(set(names))!=len(names) or z.testzip() is not None:
            raise ValueError('Wrong file count, duplicate names or bad ZIP CRC')
        for entry in z.infolist():
            rel=PurePosixPath(entry.filename)
            if rel.is_absolute() or '..' in rel.parts or stat.S_ISLNK(entry.external_attr>>16):
                raise ValueError('Unsafe archive entry')
        read_json=lambda name:json.loads(z.read(name))
        read_csv=lambda name:pd.read_csv(BytesIO(z.read('worker-output/'+name)),float_precision='round_trip')
        recovered=read_json('RECOVERED_FILES.json')
        if set('worker-output/'+name for name in recovered)!=set(n for n in names if n.startswith('worker-output/')):
            raise ValueError('Recovered manifest file-set mismatch')
        for rel,expected in recovered.items():
            if hashlib.sha256(z.read('worker-output/'+rel)).hexdigest()!=expected:
                raise ValueError('Changed recovered output: '+rel)
        native=read_json('worker-output/native/complete.json')
        for rel,expected in native['files'].items():
            if hashlib.sha256(z.read('worker-output/native/'+rel)).hexdigest()!=expected:
                raise ValueError('Changed native output: '+rel)
        spec=read_json('RUN_SPEC.json')
        for rel,expected in spec['code'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
                raise ValueError('Pinned scientific code changed: '+rel)
        inputs=read_json('worker-output/request/definition.json')['sources']
        if args.source_root is not None:
            for rel,expected in inputs.items():
                if hashlib.sha256((args.source_root/rel).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Original frozen source changed: '+rel)
            verified['original_native_source_hashes']=len(inputs)
        else:verified['original_native_source_hashes']='not requested; restore original evidence and pass --source-root'
        fixed=read_csv('fixed_time_positions.csv');validate(fixed)
        products=tables(fixed)
        for name,computed in products.items():
            stored=read_csv(name+'.csv')
            sort=[c for c in ['condition','view','seed','task_id','update','cue','position','category'] if c in computed]
            assert_frame_equal(computed.sort_values(sort).reset_index(drop=True),stored.sort_values(sort).reset_index(drop=True),check_dtype=False,check_like=True,atol=2e-12,rtol=1e-12)
        for name,dims in [('within_readout',['condition','update','cue']),('initialization',['view','update','cue'])]:
            means,intervals=seed_intervals(products[name+'_metrics'],dims)
            for suffix,computed in [('seed_means',means),('seed_intervals',intervals)]:
                stored=read_csv(name+'_'+suffix+'.csv');sort=dims+(['seed'] if suffix=='seed_means' else ['metric'])
                assert_frame_equal(computed.sort_values(sort).reset_index(drop=True),stored.sort_values(sort).reset_index(drop=True),check_dtype=False,check_like=True,atol=2e-12,rtol=1e-12)
            if not intervals['n'].eq(2).all():raise ValueError('Wrong independent unit')
        metrics=read_csv('native/seed_task_metrics.csv');frozen=read_csv('native/frozen_metrics.csv')
        if len(metrics)!=16:raise ValueError('Wrong model/task grid')
        for _,row in metrics.iterrows():
            subset=frozen[frozen.condition.eq(row.condition)&frozen.job_id.eq(row.job_id)]
            if len(subset)!=331*5:raise ValueError('Missing full frozen trajectory')
            endpoint=subset[subset['update'].eq(330)].set_index('cue').to_dict('index')
            summary=row.to_dict();summary['original_early_phenotype']=bool(row.original_early_phenotype)
            result=four_outcome(summary,endpoint)
            for field in ['four_outcome_feasible_v1','endpoint_complete','A_useful_v1','B_useful_v1','C_useful_v1','Mixed_useful_v1','NoContext_useful_v1']:
                if result[field]!=row[field]:raise ValueError('Four-outcome classification changed')
        verified.update(passed=True,archive_entries=len(names),recovered_hashes=len(recovered),native_hashes=len(native['files']),selected_source_hashes=len(spec['code']),all_four_probability_score_positions=len(fixed),paired_position_rows_per_comparison=len(products['initialization_positions']),derived_tables=6,two_seed_intervals_recomputed=True,native_network_task_rows=len(metrics),new_model_forwards=0,new_optimizer_steps=0,new_agent_calls=0)
        args.out.mkdir(parents=True)
        for name,frame in products.items():frame.to_csv(args.out/(name+'.csv'),index=False)
        (args.out/'TABLE_VERIFICATION.json').write_text(json.dumps(verified,indent=2)+'\n')
        print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
