import numpy as np,pandas as pd,pytest
from experiments.ab_update_transfer_analysis import verify

def test_budget_accounting_includes_uncertain_attempts_and_reuse():
 r=dict(new_diagnostic_control_steps=9502,new_reconstruction_steps=0,new_fixture_steps=0,new_meta_steps=0,archived_branch_optimizer_steps_reused=8606,logical_completed_branch_steps=18108,failed_or_uncertain_attempts=0)
 verify.check_exposure(r)
 with pytest.raises(AssertionError):verify.check_exposure(dict(r,new_diagnostic_control_steps=20001))
 with pytest.raises(AssertionError):verify.check_exposure(dict(r,new_meta_steps=1))
 with pytest.raises(AssertionError):verify.check_exposure(dict(r,new_diagnostic_control_steps=1))

def test_scalar_gains_cannot_hide_negative_signs():
 f=pd.DataFrame([dict(pre_ce=1.,post_ce=2.,gain=-1.,own_gain=.5,cross_gain=-.5,specificity=1.)]);verify.check_effects(f)
 with pytest.raises(AssertionError):verify.check_effects(f.assign(gain=1.))
 with pytest.raises(AssertionError):verify.check_effects(f.assign(specificity=.5))

def test_all_four_probabilities_and_correct_positions_are_verified():
 from experiments.context_experience_prior.core import TARGET
 z=np.zeros((1,5,7,4));p=np.full_like(z,.25);a=dict(logits=z,probabilities=p,ce=np.full((1,5,7),np.log(4)),accuracy=np.broadcast_to(TARGET==0,(1,5,7)).copy(),margin=np.zeros((1,5,7)))
 verify.check_raw(a)
 bad=dict(a,probabilities=p.copy());bad['probabilities'][0,0,0,3]+=.01
 with pytest.raises(AssertionError):verify.check_raw(bad)
 bad=dict(a,accuracy=np.zeros((1,5,7),bool))
 with pytest.raises(AssertionError):verify.check_raw(bad)

def test_visual_receipt_requires_complete_files_and_full_resolution_families(tmp_path):
 from experiments.ab_update_transfer_analysis.common import sha
 files={}
 for i in range(1,7):
  for suffix in ['png','svg']:
   p=tmp_path/f'figure_{i}_test.{suffix}';p.write_bytes(f'image{i}{suffix}'.encode());files[p.name]=sha(p)
 png=[n for n in files if n.endswith('.png')];r=dict(passed=True,hashes=files,reviewed_pngs=png,full_resolution_pngs=png)
 verify.visual(tmp_path,r)
 with pytest.raises(AssertionError):verify.visual(tmp_path,dict(r,reviewed_pngs=png[:-1]))
 with pytest.raises(AssertionError):verify.visual(tmp_path,dict(r,full_resolution_pngs=png[:-1]))
 (tmp_path/png[0]).write_bytes(b'changed')
 with pytest.raises(AssertionError):verify.visual(tmp_path,r)

def test_sealed_output_set_rejects_unregistered_additions(tmp_path):
 (tmp_path/'a.csv').write_text('gain\n-1\n');verify.check_output_set(tmp_path,{'a.csv':'hash'})
 (tmp_path/'extra.csv').write_text('unregistered')
 with pytest.raises(AssertionError):verify.check_output_set(tmp_path,{'a.csv':'hash'})

def test_public_cli_blocks_scientific_stages_after_seal():
 from experiments.ab_update_transfer_analysis.__main__ import after_seal
 for stage in ['inspect','preflight','run','summarize']:
  with pytest.raises(RuntimeError):after_seal(stage,None,True)
 after_seal('verify',None,True);after_seal('figures','/tmp/external',True)
 with pytest.raises(RuntimeError):after_seal('figures',None,True)
