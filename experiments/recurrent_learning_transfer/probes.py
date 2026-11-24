"""Protected natural rollouts and familiar-only geometry after training is sealed."""
import json
import numpy as np
import pandas as pd
import torch
from .design import *
from .network import pair,trace,linear_control


def validate_seal(seal,r):
 expected={(s,a,k) for s in range(20) for a in ARCHS for k in DIAG}
 if not seal.get('all40_attempts_registered') or not seal.get('all320_checkpoint_slots_registered') or seal['scientific_updates_planned']!=13200 or seal['diagnostic_updates_planned']!=640:raise ValueError('All scientific attempts/diagnostics must be frozen')
 if r.duplicated(['seed','architecture','checkpoint']).any() or set(r[['seed','architecture','checkpoint']].itertuples(index=False,name=None))!=expected:raise ValueError('Incomplete/chosen checkpoint inventory')
 valid=r[r.status=='completed']
 if dict(zip(valid.checkpoint_path,valid.checkpoint_sha))!=seal['checkpoint_hashes']:raise ValueError('Checkpoint identities changed')
 return True

def pca_familiar(d):
 if set(d)!=set(['A','B']):raise ValueError('PCA fit must use familiar A/B states only')
 h=np.r_[d['A']['hidden'],d['B']['hidden']];center=h.mean(0);x=h-center;_,s,vt=np.linalg.svd(x,full_matrices=False);components=vt[:2].copy()
 for j in range(2):
  if components[j,np.argmax(abs(components[j]))]<0:components[j]*=-1
 return dict(coordinates=x@components.T,components=components,center=center,explained_variance_ratio=s[:2]**2/(np.sum(s*s) if np.sum(s*s)>0 else 1))

def geometry(d):
 if set(d)!=set(['A','B']):raise ValueError('Familiar geometry only')
 a,b=d['A'],d['B'];dh=a['hidden']-b['hidden'];dz=a['logits']-b['logits'];dz-=dz.mean(1,keepdims=True)
 return dict(hidden_centroid_distance=float(np.linalg.norm(a['hidden'].mean(0)-b['hidden'].mean(0))),position_matched_hidden_separation=float(np.linalg.norm(dh,axis=1).mean()),centered_logit_separation=float(np.linalg.norm(dz,axis=1).mean()),AB_probability_tv=float((.5*abs(a['probabilities']-b['probabilities']).sum(1)).mean()))

def run():
 seal=json.loads((OUT/'audit/training_seal.json').read_text());r=pd.read_csv(OUT/'tables/checkpoint_registry.csv');validate_seal(seal,r)
 for name,h in seal['code'].items():
  if hash_file(ROOT/name)!=h:raise ValueError('Training code changed before probes')
 for name,h in seal['checkpoint_hashes'].items():
  if hash_file(OUT/name)!=h:raise ValueError('Frozen checkpoint changed')
 if (OUT/'audit/probes_started.json').exists():raise FileExistsError('Protected evaluation already started')
 code=codes();dump(OUT/'audit/probes_started.json',dict(training_seal_sha=hash_file(OUT/'audit/training_seal.json'),code=code,analysis_order=ENDPOINTS));torch.set_num_threads(1)
 general=[];geos=[];pca=[];raw=[];residual=[];controls=[];d1=[];links=[]
 from context_sequence.evaluation.rollout import Model as FrozenModel
 from experiments.context_mixing_sequence_continuity.mixing_decomposition import decompose
 from experiments.context_mixing_sequence_continuity.replay import natural_trace
 for endpoint in ENDPOINTS:
  for record in r[r.checkpoint==endpoint].itertuples():
   meta=dict(seed=record.seed,architecture=record.architecture,endpoint=endpoint,status=record.status,checkpoint_sha=record.checkpoint_sha)
   if record.status!='completed':
    for cue in CUES:general.append(dict(**meta,cue=cue,ce=np.nan,accuracy=np.nan,gap=np.nan))
    geos.append(meta);continue
   payload=torch.load(OUT/record.checkpoint_path,map_location='cpu',weights_only=False)
   if (payload['seed'],payload['architecture'],payload['update'],payload['config'])!=(record.seed,record.architecture,endpoint,CONFIG):raise ValueError('Checkpoint metadata mismatch')
   m=pair(record.seed)[record.architecture];m.load_state_dict(payload['model_state'],strict=True);d={cue:trace(m,cue) for cue in CUES};fam={c:d[c] for c in ['A','B']};lf=float((d['A']['ce'].mean()+d['B']['ce'].mean())/2);fa=float((d['A']['accuracy'].mean()+d['B']['accuracy'].mean())/2)
   gg=geometry(fam);gg.update(meta);geo_pca=pca_familiar(fam);gg['pca_PC1_variance']=float(geo_pca['explained_variance_ratio'][0]);gg['pca_PC2_variance']=float(geo_pca['explained_variance_ratio'][1]);geos.append(gg)
   for j,coords in enumerate(geo_pca['coordinates']):pca.append(dict(**meta,cue='A' if j<7 else 'B',transition=j%7+1,PC1=coords[0],PC2=coords[1]))
   for cue,tr in d.items():
    row=dict(**meta,cue=cue,ce=float(tr['ce'].mean()),accuracy=float(tr['accuracy'].mean()),familiar_ce=lf,familiar_accuracy=fa,gap=float(tr['ce'].mean()-lf),minus_uniform=float(tr['ce'].mean()-np.log(4)))
    for t in range(7):
     row['ce_position_'+str(t+1)]=float(tr['ce'][t]);row['accuracy_position_'+str(t+1)]=int(tr['accuracy'][t]);z=tr['logits'][t];h=tr['hidden'][t];raw.append(dict(**meta,cue=cue,transition=t+1,token=int(SEQ[t]),target=int(TARGET[t]),ce=float(tr['ce'][t]),correct=bool(tr['accuracy'][t]),**{f'hidden_{j}':v for j,v in enumerate(h)},**{f'logit_{j}':v for j,v in enumerate(z)},**{f'probability_{j}':v for j,v in enumerate(tr['probabilities'][t])}))
    general.append(row)
   doubles={c:trace(m,c,torch.float64) for c in ['A','B','Mixed']}
   for t in range(7):
    h=d['Mixed']['hidden'][t]-(d['A']['hidden'][t]+d['B']['hidden'][t])/2;z=d['Mixed']['logits'][t]-(d['A']['logits'][t]+d['B']['logits'][t])/2;z-=z.mean();hd=doubles['Mixed']['hidden'][t]-(doubles['A']['hidden'][t]+doubles['B']['hidden'][t])/2;zd=doubles['Mixed']['logits'][t]-(doubles['A']['logits'][t]+doubles['B']['logits'][t])/2;zd-=zd.mean();residual.append(dict(**meta,transition=t+1,hidden_midpoint_residual=float(np.linalg.norm(h)),centered_logit_midpoint_residual=float(np.linalg.norm(z)),double_hidden_residual=float(np.linalg.norm(hd)),double_centered_logit_residual=float(np.linalg.norm(zd))))
   if not m.nonlinear:controls.append(dict(**meta,**linear_control(m)))
   else:
    arrays=lambda x:x.detach().cpu().numpy().copy()
    adapter=FrozenModel(arrays(m.wi),arrays(m.wh),arrays(m.bih)+arrays(m.bhh),arrays(m.wo),arrays(m.bo));linked=True
    for cue in ['A','B','Mixed']:
     old=natural_trace(adapter,CUES[cue]);dh=float(np.max(abs(old[[f'hidden_{j}' for j in range(8)]].to_numpy()-d[cue]['hidden'])));dz=float(np.max(abs(old[[f'logit_{j}' for j in range(4)]].to_numpy()-d[cue]['logits'])));ok=dh<=2e-4 and dz<=.002;linked=linked and ok;links.append(dict(**meta,cue=cue,hidden_max_error=dh,logit_max_error=dz,hidden_tolerance=2e-4,logit_tolerance=.002,linked=ok))
    dd=decompose(adapter)
    for key,val in meta.items():dd[key]=val
    dd['Torch_source_linked']=linked;d1.append(dd)
 pd.DataFrame(general).to_csv(OUT/'generalization.csv',index=False);pd.DataFrame(geos).to_csv(OUT/'representational_geometry.csv',index=False);pd.DataFrame(pca).to_csv(OUT/'tables/pca_trajectories.csv',index=False);pd.DataFrame(raw).to_csv(OUT/'tables/transition_predictions.csv',index=False);pd.DataFrame(residual).to_csv(OUT/'tables/midpoint_residuals.csv',index=False);pd.DataFrame(controls).to_csv(OUT/'tables/linear_controls.csv',index=False);pd.concat(d1,ignore_index=True).to_csv(OUT/'tables/tanh_D1_decomposition.csv',index=False);pd.DataFrame(links).to_csv(OUT/'tables/D1_source_linkage.csv',index=False)
 for name,h in seal['checkpoint_hashes'].items():
  if hash_file(OUT/name)!=h:raise ValueError('Protected evaluation changed weights')
 if codes()!=code:raise ValueError('Evaluation code changed during stage')
 dump(OUT/'audit/probes_complete.json',dict(code=code,analysis_order=ENDPOINTS,models=40,endpoints=80,probe_rows=640,transition_rows=4480,linear_controls=40,optimizer_steps=0,D1_source_linkage_passed=bool(pd.DataFrame(links).linked.all())));append('## Protected evaluation complete\nAll40 runs frozen before probes. Primary170 evaluated first,330 secondary. All80 model/endpoint cells,640 cue rows and4480 positions retained. PCA uses only A/B familiar states. Linear averaging controls and imported D1 decomposition audited without changing weights. No optimization/model selection.')
 print('Protected natural evaluations complete, primary170 first; no model updates',flush=True)

if __name__=='__main__':run()
