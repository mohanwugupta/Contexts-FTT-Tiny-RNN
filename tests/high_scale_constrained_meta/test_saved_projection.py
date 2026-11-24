import numpy as np,pytest
from experiments.high_scale_constrained_meta.meta_verify import project_array
from experiments.high_scale_constrained_meta.initialization import initialize
from experiments.high_scale_constrained_meta.runtime import flat

def test_independent_projection_rejects_wrong_floor_and_protects_recurrence():
 m,_,floors,_=initialize(1);template=m.state_dict();x=flat(m.parameters());y=x.copy();y[:56]*=.5;y[136:168]*=.5;z=project_array(y,template,floors,True);assert np.array_equal(z[56:136],y[56:136]) and np.array_equal(z[168:],y[168:]);assert np.array_equal(project_array(y,template,floors,False),y)
 y[:56]=0
 with pytest.raises(ValueError):project_array(y,template,floors,True)
