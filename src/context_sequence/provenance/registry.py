"""Load only declared immutable historical artifacts; never synthesize replacements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from context_sequence.evaluation.rollout import Model
from .hashing import sha256


def checkpoint_records(root):
    root=Path(root);records=[]
    specs=[('canonical','analysis/phase1i_context_composition/phase1i_checkpoint_provenance.csv'),
           ('constructive','analysis/phase1l/phase1l_provenance.csv')]
    for cohort,path in specs:
        for r in pd.read_csv(root/path).to_dict('records'):
            p=Path(r['checkpoint_path'])
            if p.is_absolute():
                # Portability without accepting arbitrary alternate checkpoint files.
                marker='analysis/phase1l/checkpoints/' if cohort=='constructive' else 'outputs/confirmatory_srn_final/confirmatory_checkpoints/'
                p=Path(marker)/p.name
            family=r['family'];condition=r.get('condition',family)
            records.append({'cohort':cohort,'model_family':'contrast_srn' if condition in ('C1_contrast_only','C3_full') else 'canonical_srn',
                'training_objective':'sequence_CE_plus_closure' if condition in ('C2_closure_only','C3_full') else ('sequence_CE' if cohort=='constructive' else 'sequence_plus_monotonic_execution'),
                'family':family,'condition':condition,'seed':int(r['seed']),'checkpoint_step':170,'path':str(p),
                'checkpoint_sha256':r['checkpoint_hash'],'historical_label':condition})
    return records


def load_model(root,record):
    p=Path(root)/record['path']
    if not p.is_file():raise FileNotFoundError(f'Missing frozen checkpoint: {p}; replacement training forbidden')
    if sha256(p)!=record['checkpoint_sha256']:raise ValueError(f'Checkpoint hash mismatch: {p}')
    state=torch.load(p,map_location='cpu',weights_only=False)['model_state_dict']
    def a(key):return state[key].detach().cpu().numpy().copy()
    if 'context_u' in state:
        if not np.array_equal(a('cbar'),[.5,.5,0]) or not np.array_equal(a('contrast_direction'),[.5,-.5,0]):raise ValueError('Unexpected frozen contrast projection')
        m=Model(a('weight_seq'),a('weight_hh'),a('bias_ih')+a('bias_hh'),a('sequence_head.weight'),a('sequence_head.bias'),a('context_u'))
    else:m=Model(a('recurrent.weight_ih_l0'),a('recurrent.weight_hh_l0'),a('recurrent.bias_ih_l0')+a('recurrent.bias_hh_l0'),a('sequence_head.weight'),a('sequence_head.bias'))
    for v in vars(m).values():
        if isinstance(v,np.ndarray):v.setflags(write=False)
    return m


def audit_inputs(root,records):
    rows=[]
    for r in records:
        p=Path(root)/r['path'];observed=sha256(p) if p.is_file() else None
        rows.append({**r,'observed_sha256':observed,'passed':observed==r['checkpoint_sha256']})
    return pd.DataFrame(rows)
