"""Post-seal frozen natural probes, familiar PCA and midpoint controls."""
import pandas as pd
from .core import *
from experiments.mixed_checkpoint_curves.evaluate import double_control,float_control
from experiments.recurrent_learning_transfer.probes import pca_familiar,geometry

def load_model(payload):
 # Construct shape-only source model, then replace every tensor; no optimizer/update.
 m=pair(payload['seed'])[payload['architecture']];m.load_state_dict(payload['model_state_dict'],strict=True);m.requires_grad_(False);m.eval();return m

def run():
 seal=json.loads((OUT/'audit/training_seal.json').read_text());assert seal['all40_attempts_retained'] and seal['all240_checkpoint_slots_retained'] and seal['protected_evaluations']==0
 if (OUT/'audit/probes_started.json').exists():raise FileExistsError('No replay of completed natural branches')
 for n,h in seal['checkpoint_hashes'].items():
  if sha(OUT/n)!=h:raise ValueError('Checkpoint changed')
 dump(OUT/'audit/probes_started.json',dict(training_seal_sha=sha(OUT/'audit/training_seal.json'),primary=170,secondary=330));torch.set_num_threads(1);registry=pd.read_csv(OUT/'tables/checkpoint_registry.csv');general=[];positions=[];geoms=[];pca=[];residuals=[];controls=[];bounds=[]
 # All cells fixed a priori; primary170 first, secondary330 next, then descriptive earlier states.
 for k in [170,330,80,100,120,0]:
  for r in registry[registry.checkpoint==k].itertuples():
   meta=dict(seed=r.seed,architecture=r.architecture,checkpoint=k,checkpoint_sha=r.checkpoint_sha,status=r.status)
   cues=['A','B'] if k==0 else ['A','B','Mixed']
   if r.status!='completed':
    for c in cues:general.append(dict(**meta,cue=c,ce=np.nan,accuracy=np.nan))
    geoms.append(dict(**meta));continue
   payload=torch.load(OUT/r.checkpoint_path,map_location='cpu',weights_only=False);m=load_model(payload);before=state_hash(m);d={c:trace(m,c) for c in cues}
   for c,t in d.items():
    general.append(dict(**meta,cue=c,ce=float(t['ce'].mean()) if t else np.nan,accuracy=float(t['accuracy'].mean()) if t else np.nan))
    if t:
     for j in range(7):positions.append(dict(**meta,cue=c,position=j+1,token=int(SEQ[j]),target=int(TARGET[j]),ce=float(t['ce'][j]),correct=bool(t['accuracy'][j]),**{f'hidden_{q}':v for q,v in enumerate(t['hidden'][j])},**{f'logit_{q}':v for q,v in enumerate(t['logits'][j])},**{f'probability_{q}':v for q,v in enumerate(t['probabilities'][j])}))
   if d['A'] and d['B']:
    gg=geometry({c:d[c] for c in ['A','B']});gg.update(meta);gg.update(**spectra(m),hidden_absmax=max(float(abs(d[c]['hidden']).max()) for c in ['A','B']),saturation_95=float(np.mean([np.mean(abs(d[c]['hidden'])>.95) for c in ['A','B']])),saturation_99=float(np.mean([np.mean(abs(d[c]['hidden'])>.99) for c in ['A','B']])));geoms.append(gg);pc=pca_familiar({c:d[c] for c in ['A','B']})
    for j,xy in enumerate(pc['coordinates']):pca.append(dict(**meta,cue='A' if j<7 else 'B',position=j%7+1,PC1=xy[0],PC2=xy[1],familiar_only=True))
   else:geoms.append(meta)
   if k and all(d[c] is not None for c in cues):
    for j in range(7):
     dh=d['Mixed']['hidden'][j]-(d['A']['hidden'][j]+d['B']['hidden'][j])/2;dz=d['Mixed']['logits'][j]-(d['A']['logits'][j]+d['B']['logits'][j])/2;dz-=dz.mean();residuals.append(dict(**meta,position=j+1,hidden_midpoint_residual=float(np.linalg.norm(dh)),centered_logit_midpoint_residual=float(np.linalg.norm(dz))))
    if r.architecture=='RNN-Linear':
     try:
      dd={c:trace(m,c,torch.float64) for c in cues}
      if any(t is None for t in dd.values()):raise ValueError('Double evaluation nonfinite')
      ctrl=double_control(dd);br=float_control(mathematical_adapter(m),d);controls.append(dict(**meta,**ctrl,control_status='passed'))
      for x in br:bounds.append(dict(**meta,**x))
     except ValueError as e:controls.append(dict(**meta,passed=False,control_status='failed_or_blocked',reason=str(e)))
   if state_hash(m)!=before or sha(OUT/r.checkpoint_path)!=r.checkpoint_sha:raise ValueError('Frozen probe altered weights')
 for f,n in [(pd.DataFrame(general),'generalization.csv'),(pd.DataFrame(positions),'tables/transition_predictions.csv'),(pd.DataFrame(geoms),'tables/geometry.csv'),(pd.DataFrame(pca),'tables/pca_trajectories.csv'),(pd.DataFrame(residuals),'tables/midpoint_residuals.csv'),(pd.DataFrame(controls),'tables/linear_controls.csv'),(pd.DataFrame(bounds),'tables/linear_position_bounds.csv')]:f.to_csv(OUT/n,index=False)
 for n,h in seal['checkpoint_hashes'].items():
  if sha(OUT/n)!=h:raise ValueError('Frozen source changed')
 dump(OUT/'audit/probes_complete.json',dict(training_seal_sha=sha(OUT/'audit/training_seal.json'),generalization_rows=len(general),position_rows=len(positions),linear_controls=len(controls),all_linear_controls_passed=all(x['passed'] for x in controls),protected_selection=False,optimizer_steps=0));append('## Fixed protected probes completed\nAllrequestedcells retained. Primary170 evaluated first,330 secondary, earlierstates descriptive. NaturalA/B/Mixed, familiarPCA,hidden/outputresiduals andLinear double/float32 controls use immutable source checkpoints. No probe selection or extraoptimizer updates.');print('Natural probes complete; allLinearcontrols:',all(x['passed'] for x in controls))

if __name__=='__main__':run()
