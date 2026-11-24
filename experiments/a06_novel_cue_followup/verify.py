"""Read-only independent arithmetic and provenance validation. Never rolls out models."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from .protocol import *

def score_arrays(z):
    z=np.asarray(z,dtype=np.float64)
    shifted=z-z.max(axis=-1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
    target=np.asarray(core.TARGET)
    correct=np.argmax(z,axis=-1)==target
    others=z.copy();np.put_along_axis(others,np.broadcast_to(target,z.shape[:-1])[...,None],-np.inf,axis=-1)
    chosen=np.take_along_axis(z,np.broadcast_to(target,z.shape[:-1])[...,None],axis=-1)[...,0]
    return dict(probabilities=np.exp(logp),ce=-np.take_along_axis(logp,np.broadcast_to(target,z.shape[:-1])[...,None],axis=-1)[...,0],accuracy=correct,margin=chosen-others.max(axis=-1))

def check_scores(z,stored):
    s=score_arrays(z)
    for field in s:np.testing.assert_allclose(s[field],stored[field],rtol=1e-12,atol=2e-12,err_msg=field)
    return s

def check_static(z,bias,view):
    np.testing.assert_array_equal(view,.25*(z-bias)+bias)

@contextmanager
def no_scientific_execution():
    def forbidden(*a,**k):raise RuntimeError('Read-only validation/reproduction prohibits model forwards, objectives and optimizer work')
    with ExitStack() as stack:
        for obj,name in [(core,'step'),(core,'trace'),(core,'objective'),(torch.Tensor,'backward'),(torch.optim.Adam,'step'),(torch.optim.SGD,'step'),(torch.nn.Module,'_call_impl')]:
            stack.enter_context(patch.object(obj,name,forbidden))
        yield

def supplement():
    """Additional metadata, leaving the pre-execution sealed 12-cue registry intact."""
    bank=json.loads((OUT/'registry/cue_vectors.json').read_text());rows=[]
    for name,vector,kind in [(c,core.CUES[c],'original') for c in PROBES]+[(r['cue'],r['vector'],r['kind']) for r in bank]:
        v=np.asarray(vector,float);norm=float(np.linalg.norm(v));rows.append(dict(cue=name,kind=kind,c1=v[0],c2=v[1],c3=v[2],norm=norm,**{f'distance_{c}':float(np.linalg.norm(v-e)) for c,e in zip(['A','B','C'],np.eye(3))},**{f'similarity_{c}':v[i]/norm if norm else np.nan for i,c in enumerate(['A','B','C'])},similarity_status='undefined_zero_vector' if norm==0 else 'defined'))
    csv(OUT/'all_input_cues.csv',rows)
    coverage=pd.read_csv(OUT/'registry/state_coverage.csv');hashes=[]
    for seed in SEEDS:
        initial=torch.load(ROOT/load_sources()['controls'][str(seed)]['initializer'],weights_only=False)['model']
        for condition in ['A00','A06']:
            states=load_states(seed,condition);g=coverage[coverage.seed.eq(seed)&coverage.condition.eq(condition)];path=g.iloc[0].path;filehash=sha(ROOT/path)
            assert digest(states[0]['model'])==digest(initial if condition=='A00' else a06_state(initial))
            for k,s in sorted(states.items()):
                assert torch.equal(s['model']['recurrent.weight_ih_l0'][:,6],states[0]['model']['recurrent.weight_ih_l0'][:,6])
                if condition=='A06' and seed>=2:
                    assert s['update']==s['schedule_cursor']==k and s['next_cue']==(cue(k+1) if k<330 else None)
                    assert set(s)>=set(['model','policy','optimizer','rng','settings'])
                hashes.append(dict(seed=seed,condition=condition,update=k,source=path,file_sha256=filehash,full_state_hash=digest(s),model_hash=digest(s['model']),policy_hash=digest(s['policy']),Adam_hash=digest(s['optimizer']),RNG_hash=digest(s['rng']) if 'rng' in s else '',RNG_status='archived' if 'rng' in s else 'not_archived_in_inherited_A00_state',C_column_unchanged=True))
    assert len(hashes)==3330
    csv(OUT/'registry/checkpoint_hashes.csv',hashes)

def numeric():
    paths=sorted((OUT/'original_cache').glob('*.npz'))+sorted((OUT/'novel_cache').glob('*/*.npz'))
    assert len(paths)==108
    for path in paths:
        a=dict(np.load(path));assert np.isfinite(a['logits']).all() and (a['status']=='completed').all()
        check_scores(a['logits'],a)
    p=pd.read_csv(OUT/'position_predictions.csv',float_precision='round_trip',low_memory=False)
    assert not p.duplicated(['seed','condition','update','cue','position']).any()
    np.testing.assert_array_equal(p.target,np.asarray(core.TARGET)[p.position.to_numpy(int)-1]);np.testing.assert_array_equal(p.token,np.asarray(core.TARGET_PATH[:-1])[p.position.to_numpy(int)-1])
    finite=p.status.eq('completed');valid=p[finite];z=valid[[f'logit_{j}' for j in range(4)]].to_numpy();shifted=z-z.max(axis=1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True));probs=np.exp(logp);targets=valid.target.to_numpy(int);ids=np.arange(len(valid));ce=-logp[ids,targets]
    np.testing.assert_allclose(valid[[f'probability_{j}' for j in range(4)]],probs,atol=2e-12,rtol=1e-12)
    np.testing.assert_allclose(valid.ce,ce,atol=2e-12,rtol=1e-12)
    np.testing.assert_array_equal(valid.predicted_class,z.argmax(1));np.testing.assert_array_equal(valid.correct.astype(bool),z.argmax(1)==targets)
    assert p[~finite][['ce','predicted_class']+[f'logit_{j}' for j in range(4)]+[f'probability_{j}' for j in range(4)]].isna().all().all()
    assert (~finite).sum()==3210*7
    indexed=p.set_index(['seed','condition','update','cue','position'])
    for path in paths:
        a=dict(np.load(path));is_original=path.parent.name=='original_cache';label=path.stem if is_original else path.parent.name;condition,seed_=label.split('_s');seed=int(seed_);times=a['updates'] if is_original else [int(a['update'])];labels=PROBES if is_original else list(a['cues']);raw=a['logits'] if is_original else a['logits'][None,...];bias=a['output_bias'][:,None,None,:] if is_original else a['output_bias'][None,None,None,:]
        key=pd.MultiIndex.from_product([[seed],[condition],times,labels,range(1,8)])
        np.testing.assert_array_equal(indexed.reindex(key)[[f'logit_{j}' for j in range(4)]].to_numpy().reshape(raw.shape),raw)
        if condition=='A00':
            key=pd.MultiIndex.from_product([[seed],['A00-static'],times,labels,range(1,8)])
            check_static(raw,bias,indexed.reindex(key)[[f'logit_{j}' for j in range(4)]].to_numpy().reshape(raw.shape))
    keys=['seed','condition','update','cue'];agg=valid.groupby(keys).agg(ce=('ce','mean'),accuracy=('correct','mean')).reset_index()
    for name in ['original_cue_results.csv','unfamiliar_cue_results.csv']:
        f=pd.read_csv(OUT/name,float_precision='round_trip',low_memory=False);matched=f[f.status.eq('completed')].merge(agg,on=keys,suffixes=('_csv','_positions'),validate='one_to_one')
        assert len(matched)==f.status.eq('completed').sum()
        for field in ['ce','accuracy']:np.testing.assert_allclose(matched[field+'_csv'].to_numpy(float),matched[field+'_positions'].to_numpy(float),atol=2e-12,rtol=1e-12)
    for seed in SEEDS:
        a=dict(np.load(OUT/f'original_cache/A00_s{seed}.npz'));view=.25*(a['logits']-a['output_bias'][:,None,None,:])+a['output_bias'][:,None,None,:];check_static(a['logits'],a['output_bias'][:,None,None,:],view)
        expected=score_arrays(view);f=pd.read_csv(OUT/'original_cue_results.csv',float_precision='round_trip',low_memory=False);g=f[f.seed.eq(seed)&f.condition.eq('A00-static')].set_index(['update','cue']).reindex(pd.MultiIndex.from_product([a['updates'],PROBES]))
        np.testing.assert_allclose(g.ce,expected['ce'].mean(-1).reshape(-1),rtol=1e-12,atol=2e-12)
    return dict(caches=len(paths),positions=len(p),missing_positions=int((~finite).sum()),all_four_probabilities=True,independent_stable_CE=True,static_checkpoint_bias_verified=True)

def main():
    configure()
    with no_scientific_execution():
        sources=load_sources()
        for name,h in sources['inputs'].items():assert sha(ROOT/name)==h,name
        supporting=OUT/'registry/supporting_import_hashes.json'
        if supporting.exists():
            for name,h in json.loads(supporting.read_text())['files'].items():assert sha(ROOT/name)==h,name
        assert identity()==json.loads((OUT/'registry/runtime_identity.json').read_text())
        seal=json.loads((OUT/'registry/cue_seal.json').read_text());assert sha(OUT/'cue_registry.csv')==seal['sha256'] and sha(OUT/'registry/cue_vectors.json')==seal['vectors_sha'];assert cue_bank()==json.loads((OUT/'registry/cue_vectors.json').read_text())
        ledger=Ledger();assert ledger.count==1320 and all(ledger.seed_count(s)==330 for s in [2,3,4,5])
        for s in [2,3,4,5]:
            r=json.loads((OUT/f'runs/A06_s{s}/complete.json').read_text());assert r['passed'] and r['physical_optimizer_calls']==330
            for name,key in [('states.pt','states_sha'),('original.npz','observations_sha')]:assert sha(OUT/f'runs/A06_s{s}/{name}')==r[key]
        for receipt in (OUT/'novel_cache').glob('*/*.json'):
            r=json.loads(receipt.read_text());assert r['passed'] and r['sequence_forward_evaluations']==12 and r['all_model_policy_Adam_RNG_states_unchanged'];assert sha(receipt.with_suffix('.npz'))==r['sha256']
        result=numeric()
    return dict(passed=True,**result,source_files_verified=len(sources['inputs']),optimizer_attempts=1320,new_optimizer_steps=0,model_forward_calls=0)

def seal():
    assert not (OUT/'manifest.json').exists(),'Already sealed'
    result=main();put(OUT/'audit/final_verification.json',result)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/a06_novel_cue_followup','tests/a06_novel_cue_followup'] for p in (ROOT/folder).glob('*.py')}
    for name in ['configs/a06_novel_cue_followup.yaml','docs/a06_novel_cue_followup/PRD.md']:code[name]=sha(ROOT/name)
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest.json'}
    inputs=load_sources()['inputs']|json.loads((OUT/'registry/supporting_import_hashes.json').read_text())['files']
    put(OUT/'manifest.json',dict(program='a06_novel_cue_followup',version=1,inputs=inputs,code=code,outputs=outputs,optimizer_steps=1320,status='complete; fixed experiment stopped'))
    return dict(passed=True,sealed_files=len(outputs),source_files=len(inputs),code_files=len(code))

def sealed():
    m=json.loads((OUT/'manifest.json').read_text())
    for kind,base in [('inputs',ROOT),('code',ROOT),('outputs',OUT)]:
        for name,h in m[kind].items():assert sha(base/name)==h,f'{kind}: {name}'
    actual={str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest.json'}
    assert actual==set(m['outputs']),'Changed sealed file set'
    return dict(passed=True,files_verified=sum(len(m[k]) for k in ['inputs','code','outputs']),optimizer_steps=0,model_forward_calls=0)
