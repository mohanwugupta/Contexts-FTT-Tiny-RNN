"""Final source, paired-gap arithmetic, missingness and task-role checks."""
from .common import *
from .data import registry
from .protocol import validate,signature
from .analysis import summarize

def require_targets(f):
 expected={(s,c,a,d) for s in range(20) for c in CONDS for a,d in [('Historical',0),('Seen',240),('Transfer',240),('Seen',960),('Transfer',960)]};actual=list(f[['seed','condition','arm','dose']].itertuples(index=False,name=None));assert len(actual)==400 and set(actual)==expected

def main():
 reg=registry();require_targets(reg);f=read('evaluations');assert len(f)==400*331*6 and not f.duplicated(['seed','condition','arm','dose','update','cue']).any();assert f.groupby(['seed','condition','arm','dose','cue'])['update'].apply(lambda x: sorted(x)==list(range(331))).all();q=f[f.cue=='C'].pivot(index=['seed','condition','arm','dose','update'],columns='cue',values='gap');ep=read('endpoints');ef=read('paired_seed_effects')
 for c in CONDS:
  for dose in [240,960]:
   p=ep[(ep.condition==c)&(ep.cue=='C')&(ep['update']==330)&(ep.dose==dose)].pivot(index='seed',columns='arm',values='gap').reindex(range(20));v=ef[(ef.condition==c)&(ef.dose==dose)&(ef.cue=='C')&(ef['update']==330)&(ef.metric=='gap')&(ef.contrast=='Transfer-Seen')].sort_values('seed').effect;np.testing.assert_allclose(v,p.Transfer-p.Seen,atol=1e-12,rtol=1e-12,equal_nan=True)
 acq=read('acquisition');assert acq.loc[acq.status=='never_acquired','post_acquisition_gap'].isna().all();m=read('matched_acquisition');assert len(m)==400*4*3 and m.loc[~m.available,'specificity'].isna().all();assert ef.groupby(['condition','dose','contrast','update','cue','metric']).seed.apply(lambda x: sorted(x)==list(range(20))).all()
 # Retain allquery history summaries; average episodes withinseed, never increase network n.
 metareg=pd.read_csv(OUT/'registry/meta_runs.csv');queryseed=[]
 for r in metareg.itertuples():
  parts=[pd.read_csv(OUT/f'meta/s{r.seed:02d}_{r.prior}_{r.organization}_{r.arm}_b{b}_episodes.csv') for b in range(1,5)];df=pd.concat(parts,ignore_index=True);assert len(df)==960 and df.episode.tolist()==list(range(1,961));assert df.query_unchanged.all() and df.policy_reset.all() and df.inner_optimizer_empty.all()
  for dose in [240,960]:
   for window in ['cumulative','last240']:
    x=df[(df.episode<=dose)&(df.episode>(0 if window=='cumulative' else dose-240))]
    for kind,g in x.groupby('kind'):queryseed.append(dict(seed=r.seed,condition=r.prior+'-'+r.organization,arm=r.arm,dose=dose,window=window,kind=kind,query_ce=g.query_ce.mean(),query_gradient_norm=g.query_gradient_norm.mean(),n_episodes=len(g)))
 write_frame(OUT/'tables/meta_query_seed.csv',queryseed);write_frame(OUT/'tables/meta_query_summary.csv',summarize(pd.DataFrame(queryseed),['condition','arm','dose','window','kind'],['query_ce','query_gradient_norm']))
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());core.hashes(OUT,seal['registry'])
 for seed in range(20):
  for b in range(1,5):validate(json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text()))
 parent_verify();dump(OUT/'audit/final_tables_verified.json',dict(passed=True,physical_histories=400,frozen_cells=len(f),matched_cells=len(m),primary_pair_arithmetic=True,post_acquisition_missing_correct=True,all20_retained=True,source_and_registry_unchanged=True));append('## FinaltablearithmeticGREEN\n400plannedphysicaltargethistories/794400frozencells,all331updates/6cues,4800matchedcellswithmissingn,pairedprimaryGCeffectsandnever-acquiredmissingnessverified. Fixedqueryhistoriesaveragedwithinseedforkind/cumulative/last240diagnostics,notselection. Wholeparent/registryhashesunchanged.')
if __name__=='__main__':main()
