"""Prospective matched family streams, sealed before screen/confirmation outcomes."""
from .common import *
from .generator import block_specs,family,episode,validate

def make_stream(seed,n,reserved):
 assert n in [480,960];families=[];arms={a:[] for a in 'ABC'};seen=set()
 for block in range(n//480):
  rng=np.random.default_rng(np.random.SeedSequence([20261121,928777,int(seed),block]));specs=block_specs(rng)
  for spec in specs:
   duplicate=0
   while True:
    f=family(rng,spec['roles'],spec['contrast'],spec['orientation'],spec['one_position'],reserved);signature=tuple(sorted(tuple(p) for p in f['base']+f['alternatives']))
    if signature not in seen:seen.add(signature);break
    duplicate+=1
   index=len(families)+1;f.update(family_id=f's{seed:02d}_e{index}',episode=index,seed=seed,block=block+1,k=spec['k'],profile=spec['profile'],duplicate_families_rejected=duplicate);families.append(f)
   for arm in 'ABC':
    e=episode(f,spec['k'][arm]);e.update(seed=seed,episode=index,arm=arm);validate(e,reserved);arms[arm].append(e)
 return families,arms

def summary(eps):
 return dict(episodes=len(eps),type_counts={k:sum(e['n_exceptions']==k for e in eps) for k in range(3)},exceptions=sum(e['n_exceptions'] for e in eps),eligible_cells=sum(e['eligible_count'] for e in eps),critical_cells=sum(e['critical_cells'] for e in eps),role_counts={r:{c:sum(e['roles'][r]==c for e in eps) for c in 'ABC'} for r in ['X','Y','Q']},outer_critical_mixture_mass=.5*.25*sum(e['n_exceptions']>0 for e in eps)/len(eps))

def main():
 gate=json.loads((OUT/'audit/capacity_decision.json').read_text());assert gate['passed'],'Mandatory pilot gate failed: no meta registry/training';assert json.loads((OUT/'audit/pilot_verified.json').read_text())['passed'];seal=json.loads((OUT/'audit/pilot_registry_seal.json').read_text());assert sha(ROOT/'experiments/structured_overlap_meta/generator.py')==seal['generator_sha'];reserved={tuple(p) for name in ['diagnostic_paths','pilot_paths'] for p in json.loads((OUT/f'registry/{name}.json').read_text())}
 for e in json.loads((OUT/'registry/benchmark_fixture_episodes.json').read_text()):reserved.update(tuple(p) for c in ['X','Y'] for p in e['paths'][c])
 rows=[];cells=[]
 for seed in range(20):
  n=480 if seed in SCREEN else 960;families,arms=make_stream(seed,n,reserved);dump(OUT/f'registry/families_seed_{seed}.json',families)
  for arm,eps in arms.items():
   dump(OUT/f'registry/episodes_seed_{seed}_{arm}.json',eps);s=summary(eps);rows.append(dict(seed=seed,arm=arm,episodes=n,zero=s['type_counts'][0],one=s['type_counts'][1],two=s['type_counts'][2],exceptions=s['exceptions'],eligible_cells=s['eligible_cells'],critical_cells=s['critical_cells'],outer_critical_mixture_mass=s['outer_critical_mixture_mass']))
   for e in eps:
    for role in ['X','Y','Q']:cells.append(dict(seed=seed,arm=arm,episode=e['episode'],kind='role',position=0,first=e['roles'][role],second=role,count=1))
    for j in e['active_subtrials']:
     p=e['registered_positions'][j];cells.append(dict(seed=seed,arm=arm,episode=e['episode'],kind='critical',position=p,first=e['paths']['X'][j][p],second=e['paths']['Y'][j][p],count=1))
    for j in range(2):
     for t,valid in enumerate(e['novel_mask'][j],1):
      if valid:cells.append(dict(seed=seed,arm=arm,episode=e['episode'],kind='eligible',position=t,first=j,second='',count=1))
  print(f'Sealed matched streams seed{seed}',flush=True)
 write_frame(OUT/'registry/realized_arm_summary.csv',rows);frame=pd.DataFrame(cells);write_frame(OUT/'registry/realized_position_role_target_counts.csv',frame.groupby(['seed','arm','kind','position','first','second'],dropna=False)['count'].sum().reset_index());files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*') if p.is_file()};dump(OUT/'audit/meta_registry_seal.json',dict(passed=True,files=files,generator_sha=seal['generator_sha'],families=16320,potential_arm_episode_records=48960,actual_meta_episodes_if_confirmation_runs=35520,unselected_candidate_records=13440,screen_seeds=SCREEN,confirmation_seeds=CONFIRM))
 append('## Prospective meta streams sealed\nAll20seedlibrariesfixedbeforeanyscreenoutcome. Samefamilies/roles/positions/alternativesacrossarms;onlyactiveexceptionsdiffer. A/Baggregatecritical/Qcountsandexpectedexceptionsmatch;Cdiffersasregistered. All480blocksroles/types/orderedtargetsbalanced,realizedpositioncross-tabsretained. ExistingP2alreadyhadsharedprefixes;neworganization/objectiveiscompoundcontrolledfork. Conditionalcriticaltermisactive50%ofA,75%ofB,50%ofCepisodes:outercriticalmixturemass .0625/.09375/.0625, despitefixedformula. Do NOT claim pure structural arrangement isolated from its conditional objective-mass consequence. No weights changed to equalize this; masks/normalization remain exactly prescribed.')
if __name__=='__main__':main()
