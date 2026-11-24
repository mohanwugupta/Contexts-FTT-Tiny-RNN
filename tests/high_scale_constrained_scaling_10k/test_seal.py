import pytest
from experiments.high_scale_constrained_scaling_10k import seal
from experiments.high_scale_constrained_scaling_10k.common import sha

def test_seal_rejects_tamper_missing_escape(tmp_path):
 p=tmp_path/'a';p.write_text('saved');h=sha(p);seal.check_hashes(tmp_path,{'a':h})
 p.write_text('changed')
 with pytest.raises(ValueError):seal.check_hashes(tmp_path,{'a':h})
 with pytest.raises(ValueError):seal.check_hashes(tmp_path,{'missing':h})
 with pytest.raises(ValueError):seal.check_hashes(tmp_path,{'../outside':h})
 assert 'final_visual_review' in seal.REQUIRED
 assert 'figure_regeneration' in seal.REQUIRED

def test_visual_review_requires_all_companions_and_seven_families(tmp_path):
 files={}
 for i in range(46):
  family=i%7+1
  for suffix in ['png','svg']:
   p=tmp_path/f'figure_{family}_fixture_{i}.{suffix}';p.write_text(f'fixture{i}');files[p.name]=sha(p)
 pngs=sorted(n for n in files if n.endswith('.png'))
 full=[next(n for n in pngs if n.startswith(f'figure_{k}_')) for k in range(1,8)]
 review=dict(passed=True,hashes=files,reviewed_pngs=pngs,full_resolution_pngs=full)
 seal.verify_visual(tmp_path,review)
 with pytest.raises(AssertionError):seal.verify_visual(tmp_path,dict(review,hashes={n:h for n,h in files.items() if n!=pngs[0]}))
 with pytest.raises(AssertionError):seal.verify_visual(tmp_path,dict(review,reviewed_pngs=pngs[:-1]))
 with pytest.raises(AssertionError):seal.verify_visual(tmp_path,dict(review,full_resolution_pngs=full[:-1]))
