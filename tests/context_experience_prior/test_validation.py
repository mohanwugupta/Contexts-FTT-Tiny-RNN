import numpy as np
import pandas as pd
import pytest
from experiments.context_experience_prior.verify import verify_position_rows,verify_component_rows

def test_all_four_probabilities_not_only_winner_and_loss():
 z=np.array([1.,2.,3.,4.]);p=np.exp(z-z.max());p/=p.sum();f=pd.DataFrame([dict(target=1,ce=-np.log(p[1]),correct=False,margin=-2.,**{f'logit_{j}':z[j] for j in range(4)},**{f'probability_{j}':p[j] for j in range(4)})]);verify_position_rows(f)
 bad=f.copy();bad.loc[0,'probability_0']+=.01;bad.loc[0,'probability_2']-=.01
 with pytest.raises(AssertionError):verify_position_rows(bad)
 bad=f.copy();bad.loc[0,'target']=2
 with pytest.raises(AssertionError):verify_position_rows(bad)

def test_centering_and_operator_decomposition_poison():
 d={}
 for name,v in [('z0',[1,2,3,4]),('u',[4,3,2,1]),('v',[-1,1,-1,1]),('C_response',[2,3,4,5])]:
  for j,x in enumerate(v):d[f'{name}_raw_{j}']=x;d[f'{name}_centered_{j}']=x-np.mean(v)
 f=pd.DataFrame([d]);verify_component_rows(f)
 f.loc[0,'u_centered_1']+=1
 with pytest.raises(AssertionError):verify_component_rows(f)
