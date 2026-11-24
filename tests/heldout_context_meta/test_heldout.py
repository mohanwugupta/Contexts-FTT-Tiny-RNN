from experiments.heldout_context_meta.heldout import conflict_cells
from experiments.heldout_context_meta.protocol import make_block

def test_conflict_evaluation_is_identifiable_and_same_prefix():
 ep=next(e for e in make_block(0,1,set())['episodes'] if e['kind']=='conflict');cells=conflict_cells(ep)
 assert len(cells)==2 and cells[0][0]!=cells[1][0]
 paths=[ep['paths'][j] for c,j in cells];assert paths[0][:3]==paths[1][:3] and paths[0][3]!=paths[1][3]
