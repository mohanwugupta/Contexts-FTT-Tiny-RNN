import pytest
from experiments.high_scale_constrained_scaling_10k.storage import ChunkJournal

def test_atomic_complete_chunks_are_restartable_and_reject_skips(tmp_path):
 j=ChunkJournal(tmp_path/'x.sqlite');j.initialize(dict(episode=1920,model='source'));j.commit_chunk(dict(episode=1922,model='next'),[dict(episode=1921,x=1),dict(episode=1922,x=2)],registered=True);assert j.latest()['episode']==1922 and j.checkpoint(1922)['model']=='next' and len(j.summaries())==2
 with pytest.raises(ValueError):j.commit_chunk(dict(episode=1924),[dict(episode=1924)])
 assert j.latest()['episode']==1922 and len(j.summaries())==2
 j.close();k=ChunkJournal(tmp_path/'x.sqlite');assert k.latest()['model']=='next';k.close()
