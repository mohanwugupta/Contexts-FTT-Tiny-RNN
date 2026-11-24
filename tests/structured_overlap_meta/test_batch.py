import copy
import numpy as np
import torch
import pytest
from experiments.structured_overlap_meta.common import core,source,source_model,initialize_process
from experiments.structured_overlap_meta.generator import family,episode
from experiments.structured_overlap_meta.batch import forward_pair,step_batch

def data():return episode(family(np.random.default_rng(7791),dict(X='A',Y='B',Q='C'),(0,1),0,5,set()),2)

def test_independent_zero_hidden_subtrials_and_no_extra_cue():
 initialize_process();state,_,_=source(0);m=source_model(state);e=data();z,h,y=forward_pair(m,e['paths']['X'],'A');separate=[]
 for path in e['paths']['X']:
  x,yy=core.tensors(path);separate.append(m(x,torch.tensor(core.CUES['A']))[0])
 torch.testing.assert_close(z,torch.cat(separate),rtol=2e-6,atol=2e-6);assert z.shape==(2,7,4) and h.shape==(2,7,8) and y.shape==(2,7)
 p=core.policy();o=core.optimizer(m,p)
 with pytest.raises(ValueError):step_batch(m,p,o,'Mixed-both',e['paths']['X'])

def test_optimization_batched_average_against_independent_native_subtrial_losses():
 initialize_process();state,_,_=source(0);a=source_model(state);b=source_model(state);ap=core.policy();bp=core.policy();ao=core.optimizer(a,ap);bo=core.optimizer(b,bp);paths=data()['paths']['X'];r=step_batch(a,ap,ao,'A',paths)
 bo.zero_grad();losses=[]
 for path in paths:
  x,y=core.tensors(path);z,_,_=b(x,torch.tensor(core.CUES['A']));losses.append(core.objective(z,bp,y)[0])
 total=sum(losses)/2;total.backward();norm=torch.nn.utils.clip_grad_norm_(list(b.parameters())+list(bp.parameters()),1.);bo.step()
 assert abs(r['total_loss']-float(total.detach()))<2e-6 and abs(r['gradient_norm']-float(norm))<2e-5 and r['optimizer_step_executed'] and r['examples']==2
 for x,y in zip(list(a.parameters())+list(ap.parameters()),list(b.parameters())+list(bp.parameters())):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
 assert all(float(s['step'])==1 for s in ao.state.values())
