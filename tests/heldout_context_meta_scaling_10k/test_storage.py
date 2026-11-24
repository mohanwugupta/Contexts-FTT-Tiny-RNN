import pytest
from experiments.heldout_context_meta_scaling_10k.storage import Journal

def test_atomic_restart_sequential_integrity_and_corruption(tmp_path):
 p=tmp_path/'run.sqlite';j=Journal(p);j.initialize({'episode':960,'x':1});assert j.latest()['episode']==960
 j.commit({'episode':961,'x':2},{'episode':961,'status':'completed'},registered=True)
 j.close();j=Journal(p);assert j.latest()=={'episode':961,'x':2};assert j.checkpoint(961)['x']==2
 with pytest.raises(ValueError):j.commit({'episode':963},{'episode':963},False)
 with pytest.raises(ValueError):j.commit({'episode':961},{'episode':961},False)
 j.db.execute("UPDATE latest SET hash='bad'");j.db.commit()
 with pytest.raises(ValueError):j.latest()
 j.close()
