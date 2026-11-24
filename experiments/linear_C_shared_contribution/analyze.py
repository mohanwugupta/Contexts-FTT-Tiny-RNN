"""Saved-logit decomposition and input-operator audit; zero network forward calls."""
from .core import *

def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Completed diagnostic cannot be replaced')
 sources=guard();torch.set_num_threads(1);raw=pd.read_csv(SOURCE/'tables/position_predictions.csv',float_precision='round_trip');raw=raw[(raw.architecture=='RNN-Linear')&raw.cue.isin(['A','B','Mixed','NoContext'])];old=pd.read_csv(SOURCE/'tables/seed_values.csv',float_precision='round_trip');cp=pd.read_csv(SOURCE/'tables/checkpoint_hashes.csv');cp=cp[cp.architecture=='RNN-Linear'];inherited_bounds=pd.read_csv(SOURCE/'tables/linear_affine_float32.csv',float_precision='round_trip');inherited_bounds=inherited_bounds[inherited_bounds.cue=='Mixed'];position=[];long=[];neutral=[];identities=[];hashes=[]
 config=dict(checkpoints=STEPS,seeds=list(range(20)),architecture='RNN-Linear',A=[1,0,0],B=[0,1,0],Mixed=[.5,.5,0],C=[0,0,1],NoContext=[0,0,0],strong_ce=.5,strong_accuracy=6/7,bootstrap_draws=10000,bootstrap_seed=20260925,loss_partition='CE=softplus(-strongest_wrong_margin)+other_alternatives',network_forward_calls=0,optimizer_constructions=0,training_updates=0);dump(OUT/'audit/config.json',config)
 for k in STEPS:
  for seed in range(20):
   record=cp[(cp.seed==seed)&(cp.checkpoint==k)]
   if len(record)!=1:raise ValueError('Missing original checkpoint slot')
   rec=record.iloc[0];path=ROOT/rec.path
   if sha(path)!=rec.sha256:raise ValueError('Changed checkpoint')
   payload=torch.load(path,map_location='cpu',weights_only=False);st=payload['model_state_dict'];wi=st['recurrent.weight_ih_l0'].numpy();wh=st['recurrent.weight_hh_l0'].numpy();wo=st['sequence_head.weight'].numpy();bi=st['recurrent.bias_ih_l0'].numpy();hashes.append(dict(seed=seed,checkpoint=k,path=rec.path,sha256=rec.sha256))
   # C-NoContext input/state/output differences: prove numerically without replaying either branch.
   for dtype,label in [(np.float32,'float32'),(np.float64,'float64')]:
    x0=np.c_[np.eye(4)[SEQ[:-1]],np.zeros((7,3))].astype(dtype);xc=x0.copy();xc[:,6]=1;proj0=x0@wi.astype(dtype).T+bi.astype(dtype);projC=xc@wi.astype(dtype).T+bi.astype(dtype);equal=bool(np.array_equal(projC,proj0))
    if not equal:raise ValueError('C input projection differs from NoContext')
    for r in neutral_proof(wi,wh,wo,dtype):neutral.append(dict(seed=seed,checkpoint=k,dtype=label,input_projection_bitwise_equal=equal,third_channel_nonzero_count=int(np.count_nonzero(wi[:,6])),**r))
   native={};double={}
   for c in ['A','B','Mixed','NoContext']:
    f=raw[(raw.seed==seed)&(raw.checkpoint==k)&(raw.cue==c)].sort_values('position')
    if len(f)!=7:raise ValueError('Missing archived position')
    native[c]=f[[f'logit_{j}' for j in range(4)]].to_numpy()
    with np.load(SOURCE/'cache'/f'{seed}_RNN-Linear_{k}_{c}_float64.npz',allow_pickle=False) as z:
     if str(z['checkpoint_sha'])!=rec.sha256:raise ValueError('Wrong double cache source')
     double[c]=z['logits'].copy()
   familiar=old[(old.seed==seed)&(old.architecture=='RNN-Linear')&(old.checkpoint==k)].set_index('cue');good=bool(familiar.loc['A','strong_familiar']);ops=operator_components(wi,wh,wo)
   for t,target in enumerate(TARGET):
    d=decompose(native['NoContext'][t],native['A'][t],native['B'][t],int(target));dd=decompose(double['NoContext'][t],double['A'][t],double['B'][t],int(target));j0=d['baseline_competitor'];scores={};parts={};vectors={'C':d['z0'],'C_plus_u':d['restored'],'AB_logit_mean':d['retrieval'],'Mixed':native['Mixed'][t],'A':native['A'][t],'B':native['B'][t]}
    for condition,z in vectors.items():
     s=score(z,int(target));scores[condition]=s;parts[condition]=loss_partition(z,int(target));long.append(dict(seed=seed,checkpoint=k,position=t+1,token=int(SEQ[t]),target=int(target),condition=condition,strong_familiar=good,ce=s['ce'],accuracy=s['accuracy'],predicted=s['predicted'],strongest_incorrect=s['competitor'],correct_class_margin=s['margin'],**parts[condition],**{f'logit_{j}':float(z[j]) for j in range(4)},**{f'centered_logit_{j}':float(center(z)[j]) for j in range(4)},**{f'probability_{j}':float(s['probabilities'][j]) for j in range(4)}))
    tol=dd['tolerance'];source_bound=inherited_bounds[(inherited_bounds.seed==seed)&(inherited_bounds.checkpoint==k)&(inherited_bounds.position==t+1)].iloc[0];mixed_error=float(abs(native['Mixed'][t]-d['restored']).max());double_error=float(abs(double['Mixed'][t]-dd['restored']).max());retrieval_error=float(abs(d['restored']-d['retrieval']).max());uerror=float(abs(dd['u']-ops[t][0]).max());verror=float(abs(dd['v']-ops[t][1]).max())
    if double_error>tol or max(uerror,verror)>tol or mixed_error>source_bound.logit_bound_max+2*d['tolerance'] or abs(scores['C_plus_u']['ce']-scores['AB_logit_mean']['ce'])>2*d['tolerance']:raise ValueError('Inherited precision/operator/restoration identity failed')
    identities.append(dict(seed=seed,checkpoint=k,position=t+1,double_tolerance=tol,double_Mixed_error=double_error,double_operator_u_error=uerror,double_operator_v_error=verror,native_Mixed_error=mixed_error,native_Mixed_logit_bound=float(source_bound.logit_bound_max),restored_retrieval_error=retrieval_error,restored_retrieval_ce_difference=scores['C_plus_u']['ce']-scores['AB_logit_mean']['ce'],restored_native_Mixed_ce_difference=scores['C_plus_u']['ce']-scores['Mixed']['ce'],restored_native_Mixed_prediction_agree=scores['C_plus_u']['predicted']==scores['Mixed']['predicted'],passed=True))
    ce_gain=scores['C']['ce']-scores['C_plus_u']['ce'];margin_gain=parts['C']['binary_loss']-parts['C_plus_u']['binary_loss'];other_gain=parts['C']['other_loss']-parts['C_plus_u']['other_loss']
    if abs(ce_gain-margin_gain-other_gain)>2*d['tolerance']:raise ValueError('Loss partition not exact')
    row=dict(seed=seed,checkpoint=k,position=t+1,token=int(SEQ[t]),target=int(target),strong_familiar=good,C_competitor=j0,restored_competitor=d['restored_competitor'],ce_C=scores['C']['ce'],ce_restored=scores['C_plus_u']['ce'],ce_native_Mixed=scores['Mixed']['ce'],ce_gain=ce_gain,margin_loss_gain=margin_gain,other_alternatives_loss_gain=other_gain,C_accuracy=scores['C']['accuracy'],restored_accuracy=scores['C_plus_u']['accuracy'],wrong_to_correct=int(not scores['C']['accuracy'] and scores['C_plus_u']['accuracy']),correct_to_wrong=int(scores['C']['accuracy'] and not scores['C_plus_u']['accuracy']),C_margin=scores['C']['margin'],restored_margin=scores['C_plus_u']['margin'],actual_margin_change=d['actual_margin_change'],fixed_C_competitor_u_margin=d['fixed_margin_u'],competitor_switch_penalty=d['competitor_switch_penalty'],centered_target_u=float(d['center_u'][target]),centered_wrong_suppression_u=float(-d['center_u'][j0]),u_raises_centered_target=int(d['center_u'][target]>0),u_suppresses_C_competitor=int(d['center_u'][j0]<0))
    for name in ['z0','u','v']:
     vec=d[name];cv=d['center_'+name];row[name+'_centered_norm']=float(np.linalg.norm(cv));row[name+'_fixed_C_competitor_margin']=float(vec[target]-vec[j0])
     for j in range(4):row[name+f'_raw_{j}']=float(vec[j]);row[name+f'_centered_{j}']=float(cv[j]);row[name+f'_target_minus_class_{j}']=float(vec[target]-vec[j])
    position.append(row)
   if sha(path)!=rec.sha256:raise ValueError('Checkpoint changed during arithmetic')
 position=pd.DataFrame(position);long=pd.DataFrame(long);seed_rows=[]
 for (seed,k),f in position.groupby(['seed','checkpoint']):
  record=dict(seed=seed,checkpoint=k,strong_familiar=bool(f.strong_familiar.iloc[0]),checkpoint_sha=cp[(cp.seed==seed)&(cp.checkpoint==k)].iloc[0].sha256)
  for name in ['ce_C','ce_restored','ce_native_Mixed','ce_gain','margin_loss_gain','other_alternatives_loss_gain','C_accuracy','restored_accuracy','C_margin','restored_margin','actual_margin_change']:record[name]=float(f[name].mean())
  record.update(wrong_to_correct_positions=int(f.wrong_to_correct.sum()),correct_to_wrong_positions=int(f.correct_to_wrong.sum()));seed_rows.append(record)
 for f,n in [(position,'position_decomposition.csv'),(long,'condition_predictions.csv'),(pd.DataFrame(seed_rows),'seed_values.csv'),(pd.DataFrame(neutral),'C_NoContext_identity.csv'),(pd.DataFrame(identities),'decomposition_checks.csv'),(pd.DataFrame(hashes),'checkpoint_hashes.csv')]:f.to_csv(OUT/'tables'/n,index=False)
 intervals=[];metrics=['ce_C','ce_restored','ce_native_Mixed','ce_gain','margin_loss_gain','other_alternatives_loss_gain','C_accuracy','restored_accuracy','C_margin','restored_margin','actual_margin_change'];metrics_pos=metrics+['wrong_to_correct','correct_to_wrong','z0_centered_norm','u_centered_norm','v_centered_norm','z0_fixed_C_competitor_margin','u_fixed_C_competitor_margin','v_fixed_C_competitor_margin','centered_target_u','centered_wrong_suppression_u','fixed_C_competitor_u_margin','competitor_switch_penalty','u_raises_centered_target','u_suppresses_C_competitor']
 for level,frame,keys,mets in [('seed',pd.DataFrame(seed_rows),['checkpoint'],metrics),('position',position,['checkpoint','position'],metrics_pos)]:
  for key,g in frame.groupby(keys):
   key=key if isinstance(key,tuple) else (key,)
   for qual in ['all','strong_familiar']:
    for metric in mets:
     f=g[['seed',metric]].copy()
     if qual=='strong_familiar':f[metric]=f[metric].where(g.strong_familiar.to_numpy(),np.nan)
     intervals.append(dict(level=level,**dict(zip(keys,key)),qualification=qual,metric=metric,**interval(f,metric)))
 pd.DataFrame(intervals).to_csv(OUT/'tables/intervals.csv',index=False)
 shares=[]
 for k,g in pd.DataFrame(seed_rows).groupby('checkpoint'):
  for qual in ['all','strong_familiar']:
   f=g if qual=='all' else g[g.strong_familiar];den=float(f.ce_gain.sum());num=float(f.margin_loss_gain.sum());shares.append(dict(checkpoint=k,qualification=qual,n_valid=len(f),total_ce_gain=den,total_binary_margin_loss_gain=num,total_other_alternatives_gain=float(f.other_alternatives_loss_gain.sum()),margin_share_percent=100*num/den if den>0 else np.nan,ratio_of_sums_not_mean_ratios=True))
 pd.DataFrame(shares).to_csv(OUT/'tables/cohort_loss_accounting.csv',index=False)
 if guard()!=sources:raise ValueError('Source seal changed')
 dump(OUT/'audit/analysis_complete.json',dict(passed=True,source_manifests=sources,inputs={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'tables/position_predictions.csv',SOURCE/'tables/seed_values.csv',SOURCE/'tables/checkpoint_hashes.csv',SOURCE/'tables/linear_affine_float32.csv',ROOT/'experiments/high_scale_initialization/core.py']},code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('core.py')]},model_checkpoint_slots=40,position_rows=len(position),prediction_rows=len(long),network_forward_calls=0,training_updates=0,optimizer_constructions=0,all_native_Mixed_predictions_match=all(r['restored_native_Mixed_prediction_agree'] for r in identities)))
 append('## Archived decomposition completed\nAll40 source model/states,280 positions,1680 conditionprediction rows and560 zerochannelproof rows retained. Usedarchivednative/f64trajectories only,noforwardoroptimizer. Operatoru/v andCidentity verified; nativeMixedroundoff separate fromexactretrieval. Seed/positionCIs andsignedmargin-lossaccounting saved; all20/qualified separate.');print('Archived diagnostic complete; nativeMixedpredictionsallmatch:',all(r['restored_native_Mixed_prediction_agree'] for r in identities))

if __name__=='__main__':run()
