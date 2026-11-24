"""Seal matched tasks and base tensors before any scientific update."""
import platform,sys
from .core import *
from .registry import make_registry,validate,transition_counts

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if (OUT/'audit/protocol_seal.json').exists():raise RuntimeError('Protocol already sealed')
 parent=json.loads((SOURCE/'manifest.json').read_text());hashes(SOURCE,parent['outputs']);hashes(ROOT,parent['code']);episodes=[];updates=[];sequences=[];base=[];initials={}
 for seed in range(20):
  r=make_registry(seed);validate(r);dump(OUT/'registry'/f'seed_{seed}.json',r)
  for ep,rows in zip(r['episodes'],r['updates']):
   meta={k:v for k,v in ep.items() if k!='paths'};episodes.append(dict(seed=seed,**meta,context_sequence_MI_P1=0.,context_sequence_MI_P2=np.log(2) if ep['kind']=='conflict' else 0.,shared_prefix=3 if ep['kind']=='conflict' else 0))
   for j,path in enumerate(ep['paths']):
    a=transition_counts(path);sequences.append(dict(seed=seed,episode=ep['episode'],sequence=j,full_path=''.join(map(str,path)),inputs=''.join(map(str,path[:-1])),targets=''.join(map(str,path[1:])),history_ambiguous_tokens=int(((a>0).sum(1)>1).sum()),all_four_symbols=True,target_dihedral_excluded=True,**{f'transition_{i}_{k}':int(a[i,k]) for i in range(4) for k in range(4)}))
   for prior in ['P1','P2']:
    for x in rows[prior]:updates.append(dict(seed=seed,episode=ep['episode'],prior=prior,**x))
  for org,m in base_pair(seed).items():
   key=f'{seed}_{org}';initials[key]=copy.deepcopy(m.state_dict());wi=m.recurrent.weight_ih_l0.detach();base.append(dict(seed=seed,organization=org,model_hash=hash_state(m),A_norm=float(wi[:,4].norm()),B_norm=float(wi[:,5].norm()),C_norm=float(wi[:,6].norm()),Wh_spectral_norm=float(torch.linalg.matrix_norm(m.recurrent.weight_hh_l0.detach().double(),2)),no_C_gradient_mask=not bool(m.recurrent.weight_ih_l0._backward_hooks)))
 torch.save(initials,OUT/'registry/base_initializations.pt')
 for name,rows in [('episodes',episodes),('updates',updates),('sequences',sequences),('base_initializations',base)]:pd.DataFrame(rows).to_csv(OUT/'registry'/f'{name}.csv',index=False)
 inputs={str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items() if n.startswith('part_b/') and (n.endswith('.pt') or n.endswith('.csv'))};inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json')
 for n in ['confirmatory_srn/model.py','confirmatory_srn/config.py','execution_objective_search/objectives.py','experiments/high_scale_initialization/core.py','experiments/recurrent_learning_transfer/measures.py','experiments/recurrent_learning_transfer/learning.py','docs/context_experience_prior/PRD.md']:inputs[n]=sha(ROOT/n)
 dump(OUT/'audit/environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),threads=1,deterministic_algorithms=True))
 registry_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*')};dump(OUT/'audit/protocol_seal.json',dict(inputs=inputs,registry_hashes=registry_hashes,code=codes(),source_parent_manifest=sha(SOURCE/'manifest.json'),meta_updates=384000,outer_updates=19200,new_target_updates=26400,P0_target_updates_reused=13200,diagnostic_updates=1440,total_new_gradient_updates=411840,episodes_per_seed=240,inner_updates=20,Reptile_reference='https://arxiv.org/abs/1803.02999',target_selection=False))
 append('## Registry/source protocol sealed\nAll20 registries:4800 episodes,7200 sequence slots,192000 matchedP1/P2 schedule rows. Leakage,historydependence,contextMI,marginal exposures andcounterbalancing pass. BaseI/C tensors saved withnonzero C/nohook. Sourceparent seal verified;scientifictraining not yetstarted. Fixed budgets,final240 selection andpaperreference recorded.')
if __name__=='__main__':main()
