import copy
import numpy as np,pytest,torch
from experiments.high_scale_constrained_scaling_10k.verify import verify_projection
from experiments.high_scale_constrained_scaling_10k.common import SOURCE,source_model
from experiments.high_scale_constrained_scaling_10k.protocol import floors,projection
from experiments.high_scale_constrained_meta.runtime import pack

def test_projection_reconstruction_rejects_recurrence_and_floor_changes():
 d=torch.load(SOURCE/'initial/s00_scale4.pt',weights_only=False)['model'];m=source_model(d);f=floors(d)
 with torch.no_grad():m.recurrent.weight_ih_l0.mul_(.8);m.sequence_head.weight.mul_(.8)
 raw=pack(m,None);rows=projection(m,f,None,'P');p=pack(m,None)[0];arr=np.array([[r[v] for v in ['before_rms','before_rms','after_rms','floor','multiplier','parameter_change_norm','projected','enabled']] for r in rows]);verify_projection(raw,p,arr,f,'P');bad=p.copy();bad[56]+=.01
 with pytest.raises(AssertionError):verify_projection(raw,bad,arr,f,'P')
 bad=arr.copy();bad[0,3]=2
 with pytest.raises(AssertionError):verify_projection(raw,p,bad,f,'P')

def test_before_outer_RMS_cannot_be_replaced_by_raw_RMS():
 from experiments.high_scale_constrained_scaling_10k.verify import GROUPS,INDEX
 d=torch.load(SOURCE/'initial/s00_scale4.pt',weights_only=False)['model'];m=source_model(d);f=floors(d);raw=pack(m,None);after=raw[0].copy();rows=[]
 for g in GROUPS:
  r=float(torch.sqrt(torch.mean(torch.from_numpy(raw[0,INDEX[g]]).double()**2)));rows.append([r,r,r,f[g],1.,0.,0.,0.])
 arr=np.array(rows);verify_projection(raw,after,arr,f,'U',previous=raw[0])
 bad=arr.copy();bad[0,0]+=.1
 with pytest.raises(AssertionError):verify_projection(raw,after,bad,f,'U',previous=raw[0])
