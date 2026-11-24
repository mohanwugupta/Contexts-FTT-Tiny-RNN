import numpy as np
import torch
from experiments.meta_objective_balance.common import source,source_model,core,initialize_process
from experiments.meta_objective_balance.protocol import pair_block,old,critical_mask
from experiments.meta_objective_balance.verify import analytic
from experiments.heldout_context_meta.fomaml import query_loss

def test_saved_bptt_all_and_only_critical_transitions_without_optimizer():
 initialize_process();state,_,_=source(0);m=source_model(state);ep=pair_block(old.make_block(0,5,set()))[0]['conflict'];loss,z,h,qs=query_loss(m,ep,'P2');y=torch.tensor([ep['paths'][q['sequence']][1:] for q in qs]);mask=torch.tensor(critical_mask(ep,qs));critical=torch.nn.functional.cross_entropy(z[mask],y[mask]);params=list(m.parameters())
 ga=torch.autograd.grad(loss,params,retain_graph=True);gc=torch.autograd.grad(critical,params)
 for crit,gs in [(False,ga),(True,gc)]:np.testing.assert_allclose(analytic(state,ep,z.detach().numpy(),h.detach().numpy(),crit),np.concatenate([g.numpy().ravel() for g in gs]),rtol=2e-4,atol=1e-4)
 assert not np.allclose(np.concatenate([g.numpy().ravel() for g in ga]),np.concatenate([g.numpy().ravel() for g in gc]))
