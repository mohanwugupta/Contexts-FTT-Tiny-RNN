"""Independent archived-array verification; no recurrent model is instantiated."""
import argparse
from .core import *

def check_accounting(r,tol=1e-9):
 assert abs(r['ce_gain']-r['margin_loss_gain']-r['other_alternatives_loss_gain'])<=tol
 assert abs(r.get('restored_retrieval_ce_difference',0))<=tol
 return True

def hashes(root,d):
 for n,h in d.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Changed file: '+n)

def verify(out=OUT):
 out=Path(out);read=lambda n:pd.read_csv(out/'tables'/n,float_precision='round_trip');audit=json.loads((out/'audit/analysis_complete.json').read_text());assert guard()==audit['source_manifests'];hashes(ROOT,audit['inputs']);hashes(ROOT,audit['code']);d=read('position_decomposition.csv');s=read('seed_values.csv');long=read('condition_predictions.csv');ident=read('decomposition_checks.csv');neutral=read('C_NoContext_identity.csv');cp=read('checkpoint_hashes.csv')
 from .figures import validate
 validate(d);assert len(s)==40 and len(long)==1680 and len(ident)==280 and len(neutral)==560 and len(cp)==40
 assert neutral.input_projection_bitwise_equal.all() and neutral.third_channel_nonzero_count.eq(0).all() and neutral.delta_hidden_max.eq(0).all() and neutral.delta_logit_max.eq(0).all()
 archived=pd.read_csv(SOURCE/'tables/position_predictions.csv',float_precision='round_trip')
 for r in cp.itertuples():
  assert sha(ROOT/r.path)==r.sha256;payload=torch.load(ROOT/r.path,map_location='cpu',weights_only=False);st=payload['model_state_dict'];wi=st['recurrent.weight_ih_l0'].numpy();wh=st['recurrent.weight_hh_l0'].numpy();wo=st['sequence_head.weight'].numpy();assert all(x['delta_logit_max']==0 for x in neutral_proof(wi,wh,wo));f=long[(long.seed==r.seed)&(long.checkpoint==r.checkpoint)]
  for condition,cue in [('C','NoContext'),('A','A'),('B','B'),('Mixed','Mixed')]:
   a=archived[(archived.seed==r.seed)&(archived.architecture=='RNN-Linear')&(archived.checkpoint==r.checkpoint)&(archived.cue==cue)].sort_values('position');b=f[f.condition==condition].sort_values('position');np.testing.assert_array_equal(a[[f'logit_{j}' for j in range(4)]],b[[f'logit_{j}' for j in range(4)]])
  for t in range(1,8):
   ff=f[f.position==t].set_index('condition');get=lambda c:ff.loc[c,[f'logit_{j}' for j in range(4)]].to_numpy(float);target=int(ff.loc['C','target']);x=decompose(get('C'),get('A'),get('B'),target);np.testing.assert_allclose(x['restored'],get('C_plus_u'),atol=1e-10);np.testing.assert_array_equal(x['retrieval'],get('AB_logit_mean'));row=d[(d.seed==r.seed)&(d.checkpoint==r.checkpoint)&(d.position==t)].iloc[0]
   for name in ['z0','u','v']:
    np.testing.assert_allclose(row[[name+f'_raw_{j}' for j in range(4)]].to_numpy(float),x[name],atol=1e-10);np.testing.assert_allclose(row[[name+f'_centered_{j}' for j in range(4)]].to_numpy(float),center(x[name]),atol=1e-10)
   check_accounting(row.to_dict(),tol=2*x['tolerance'])
 for r in long.itertuples():
  z=np.array([getattr(r,f'logit_{j}') for j in range(4)]);x=score(z,r.target);parts=loss_partition(z,r.target);np.testing.assert_allclose([r.ce,r.correct_class_margin,r.binary_loss,r.other_loss],[x['ce'],x['margin'],parts['binary_loss'],parts['other_loss']],atol=1e-10);np.testing.assert_allclose([getattr(r,f'probability_{j}') for j in range(4)],x['probabilities'],atol=1e-12);assert r.predicted==x['predicted'] and r.accuracy==x['accuracy']
 for r in s.itertuples():
  f=d[(d.seed==r.seed)&(d.checkpoint==r.checkpoint)]
  for metric in ['ce_C','ce_restored','ce_native_Mixed','ce_gain','margin_loss_gain','other_alternatives_loss_gain','C_accuracy','restored_accuracy']:np.testing.assert_allclose(getattr(r,metric),f[metric].mean(),atol=1e-10)
  before=long[(long.seed==r.seed)&(long.checkpoint==r.checkpoint)&long.condition.isin(['A','B'])].groupby('condition')[['ce','accuracy']].mean();good=bool(before.loc['A','ce']<=.5 and before.loc['B','ce']<=.5 and before.loc['A','accuracy']>=6/7 and before.loc['B','accuracy']>=6/7);assert r.strong_familiar==good
 for r in read('intervals.csv').itertuples():
  f=s[s.checkpoint==r.checkpoint] if r.level=='seed' else d[(d.checkpoint==r.checkpoint)&(d.position==r.position)];values=f[['seed',r.metric]].copy()
  if r.qualification=='strong_familiar':values[r.metric]=values[r.metric].where(f.strong_familiar.to_numpy(),np.nan)
  x=interval(values,r.metric);np.testing.assert_allclose([r.mean,r.ci_low,r.ci_high],[x[k] for k in ['mean','ci_low','ci_high']],rtol=1e-10,atol=1e-9,equal_nan=True);assert r.n_valid==x['n_valid']
 assert ident.passed.all() and ident.restored_native_Mixed_prediction_agree.all();assert (ident.double_Mixed_error<=ident.double_tolerance).all() and (ident.double_operator_u_error<=ident.double_tolerance).all() and (ident.double_operator_v_error<=ident.double_tolerance).all()
 if (out/'manifest.json').exists():
  m=json.loads((out/'manifest.json').read_text());hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,seed_checkpoint_cells=40,positions=280,condition_predictions=1680,C_NoContext_proof_rows=560,all20_seeds_retained=True,checkpoints_unchanged=True,network_forward_calls=0,training_updates=0,optimizer_constructions=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify()
 if a.regenerate:
  from .figures import generate
  files=generate(OUT,a.regenerate)
  for n in files:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_regenerated_figure_files']=len(files)
 print(json.dumps(r,indent=2))

if __name__=='__main__':main()
