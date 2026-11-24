import json
import pytest
from experiments.high_scale_constrained_scaling_10k.registries import build,validate_bank
from experiments.high_scale_constrained_scaling_10k.common import SOURCE

def test_exact_prefix_native_extensions_partial_blocks_and_exclusions():
 reserved={tuple(p) for p in json.loads((SOURCE/'registry/diagnostic_paths.json').read_text())};banks,unused=build(0,reserved)
 for arm in ['R','T','B','O1','O2']:
  old=json.loads((SOURCE/f'registry/{arm}_s00.json').read_text())['records'];assert banks[arm][:len(old)]==old;validate_bank(banks[arm],arm,reserved)
 assert [len(banks[a]) for a in ['R','T','B','O1','O2']]==[1920,10000,5000,10000,10000]
 assert len(unused['T'])==80 and len(unused['B'])==40 and len(unused['O1'])==len(unused['O2'])==80
 bad=json.loads(json.dumps(banks['T']));bad[1920]['paths'][0]=list(next(iter(reserved)))
 with pytest.raises(ValueError):validate_bank(bad,'T',reserved)
 bad=json.loads(json.dumps(banks['B']));bad[960]['pair']=960
 with pytest.raises(ValueError):validate_bank(bad,'B',reserved)
