import torch
from experiments.joint_phenotype_hypothesis_screen.audit_outputs import gate_values
def test_integer_registered_vectors_use_native_float_dtype():
 v=torch.zeros(8,3);g=gate_values(v,[1,0,0]);assert g.dtype==torch.float32 and torch.equal(g,torch.ones(8))
 v.fill_(100.);assert (gate_values(v,[1,0,0])<=1.5).all() and torch.equal(gate_values(v,[0,0,0]),torch.ones(8))
