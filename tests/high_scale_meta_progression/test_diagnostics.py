import pytest
from experiments.high_scale_meta_progression.diagnostics import jobs,validate_reserved_tasks
from experiments.high_scale_meta_progression.common import OUT
import json

def test_parent_alias_not_rerun_and_original_critical_labels():
 j=jobs();assert len(j)==234 and len(set(j))==234
 assert sum(x[2]=='R' for x in j)==18 and all(d==1920 for s,k,a,d in j if a=='R')
 for seed in range(6):
  eps=json.loads((OUT/f'registry/heldout_v2_seed_{seed}.json').read_text())['episodes'];validate_reserved_tasks(eps)
  bad=json.loads(json.dumps(eps));bad[0]['paths'][1][3]=bad[0]['paths'][0][3]
  with pytest.raises(ValueError):validate_reserved_tasks(bad)
