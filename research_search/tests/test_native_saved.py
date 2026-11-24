import copy
import numpy as np,torch
from research_search.common import PACKAGE,read
from research_search.runners.candidate import DEFAULT,build_model,export_predictions
from research_search.evaluation.linear import controls

def test_saved_prior_target_transformation_and_affine_controls():
    torch.set_num_threads(1)
    base=PACKAGE/'database/native_prior_fixture_v2';cp=torch.load(base/'target/progress.pt',weights_only=False);prior=torch.load(base/'prior/pretrained.pt',weights_only=False);zero=cp['extra']['states'][0]['model']
    for name,value in prior['model'].items():torch.testing.assert_close(zero[name],value*.25 if name=='sequence_head.weight' else value,atol=0,rtol=0)
    config=copy.deepcopy(DEFAULT);config['prior'].update(episodes=1,cue_rotation=True);config['target']['readout_initial_multiplier']=.25
    for n,s in cp['extra']['states'].items():
        m=build_model(config,170004).double();m.load_state_dict(s['model']);d=export_predictions(m,cp['extra']['task']);r=controls(s['model'],cp['extra']['observations'][n],cp['extra']['hidden_observations'][n],d,cp['extra']['task']);assert r['passed']

def test_zero_gate_forward_matches_native_additive_without_training():
    from experiments.context_experience_prior import core
    c=copy.deepcopy(DEFAULT);m=build_model(c,777);c['model']['gate']=True;g=build_model(c,777);x,_=core.tensors([0,0,1,0,2,3,1,3])
    with torch.no_grad():
        z=m(x,torch.tensor([1.,0.,0.]))[0];zz=g(x,torch.tensor([1.,0.,0.]))[0]
    torch.testing.assert_close(z,zz,rtol=0,atol=0)
