import pandas as pd
import pytest
from experiments.nonzero_context_generalization.verify import verify

def test_complete_frozen_evidence_no_training():
 r=verify(sealed=False)
 assert r['primary_probe_cells']==529600 and r['all_scale_probe_cells']==2648000
 assert r['trained_seeds']==20 and r['vectors_per_seed']==20 and r['parameter_mutations']==0
 assert r['training_updates']==0 and r['optimizer_constructions']==0
