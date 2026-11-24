import copy
import pytest
from experiments.high_scale_meta_progression.registries import structured_stream,paired_stream,validate_exclusions
from experiments.high_scale_constrained_meta.registries import generate

def test_streams_matching_counts_exclusions_and_cadence():
 reserved={(0,1,2,0,3,1,2,3)};f,arms=structured_stream(0,480,reserved)
 assert len(f)==480
 for arm,counts in [('O1',[120,240,120]),('O2',[240,192,48])]:
  assert [sum(e['n_exceptions']==k for e in arms[arm]) for k in range(3)]==counts
  validate_exclusions(arms[arm],reserved,structured=True)
 for a,b in zip(arms['O1'],arms['O2']):assert a['family_id']==b['family_id'] and a['roles']==b['roles'] and a['paths']['X']==b['paths']['X']
 eps=generate(0,'FOMAML',1,reserved)['episodes'];pairs=paired_stream(eps);assert len(pairs)==120
 assert sorted(e['episode'] for p in pairs for e in [p['shared'],p['conflict']])==list(range(1,241))
 bad=copy.deepcopy(eps);reserved.add(tuple(bad[0]['paths'][0]))
 with pytest.raises(ValueError):validate_exclusions(bad,reserved)
