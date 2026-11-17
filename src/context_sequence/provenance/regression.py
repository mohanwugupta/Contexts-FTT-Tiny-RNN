"""Pre-Novel regression: familiar and legacy cues only."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from context_sequence.evaluation.rollout import rollout
from context_sequence.task.sequence import SEQUENCE
from .registry import load_model


def historical_regression(root,records):
    root=Path(root);torch.set_num_threads(1);rows=[]
    kc=pd.read_csv(root/'analysis/phase1k/phase1k_natural_rollouts.csv')
    lc=pd.read_csv(root/'analysis/phase1l/phase1l_final_transition_metrics.csv')
    old={'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0],'NoContext':[0,0,0]}
    for i,v in enumerate([(1,1,1),(1,-1,1),(-1,1,1),(-1,-1,1)],1):old[f'Novel{i}']=np.array(v,dtype=np.float32)/np.sqrt(3.)
    for r in records:
        model=load_model(root,r);saved=kc if r['cohort']=='canonical' else lc
        group=saved[saved.seed.eq(r['seed'])]
        group=group[group.family.eq(r['family'])] if r['cohort']=='canonical' else group[group.condition.eq(r['condition'])]
        state=torch.load(root/r['path'],map_location='cpu',weights_only=False)['model_state_dict']
        max_ce=max_h=max_z=max_torch_ce=0.
        for cue,value in old.items():
            run=rollout(model,value);reference=group[group.probe.eq(cue)].sort_values('transition')
            max_ce=max(max_ce,float(np.max(np.abs(run['raw_ce']-reference.raw_ce.to_numpy()))))
            # Independent historical Torch ordering for natural forward regression.
            h=torch.zeros(8);hs=[];zs=[]
            for token in SEQUENCE[:-1]:
                x=torch.eye(4)[token];c=torch.tensor(value,dtype=torch.float32)
                if 'context_u' in state:
                    d=state['contrast_direction'];alpha=((c-state['cbar'])*d).sum()/(d.square().sum()+1e-12)
                    h=torch.tanh(x@state['weight_seq'].T+state['context_u']*alpha+h@state['weight_hh'].T+state['bias_ih']+state['bias_hh'])
                else:
                    # nn.RNN CPU one-step expression with independent bias additions.
                    h=torch.tanh(torch.nn.functional.linear(torch.cat([x,c]),state['recurrent.weight_ih_l0'],state['recurrent.bias_ih_l0'])+torch.nn.functional.linear(h,state['recurrent.weight_hh_l0'],state['recurrent.bias_hh_l0']))
                hs.append(h.numpy());zs.append((h@state['sequence_head.weight'].T+state['sequence_head.bias']).numpy())
            z=np.stack(zs);ce=torch.nn.functional.cross_entropy(torch.tensor(z),torch.tensor(SEQUENCE[1:]),reduction='none').numpy()
            max_h=max(max_h,float(np.abs(run['hidden']-np.stack(hs)).max()));max_z=max(max_z,float(np.abs(run['logits']-z).max()));max_torch_ce=max(max_torch_ce,float(np.abs(run['raw_ce']-ce).max()))
        tolerance=1e-10 if r['cohort']=='canonical' else 2e-3
        rows.append({**r,'check':'historical_natural','max_saved_ce_difference':max_ce,'max_numpy_torch_hidden_difference':max_h,
                     'max_numpy_torch_logit_difference':max_z,'max_numpy_torch_ce_difference':max_torch_ce,
                     'saved_ce_tolerance':tolerance,'passed':max_ce<=tolerance and max_h<=2e-4 and max_z<=2e-3 and max_torch_ce<=2e-3})
        if r['condition']=='C0_canonical':
            other=torch.load(root/f'outputs/confirmatory_srn_final/confirmatory_checkpoints/High_sequence_only_seed_{r["seed"]}.pt',map_location='cpu',weights_only=False)['model_state_dict']
            same=set(state)==set(other) and all(torch.equal(state[k],other[k]) for k in state)
            rows.append({**r,'check':'C0_identity','passed':same})
    return pd.DataFrame(rows)
