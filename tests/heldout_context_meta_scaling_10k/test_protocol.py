import copy
import pytest
from experiments.heldout_context_meta_scaling_10k.protocol import DOSES, NEW_DOSES, execution_prefix, reserve_block
from experiments.heldout_context_meta import protocol as source

def test_original_distribution_and_reserved_paths():
 original=source.make_block(0,5,set());reserved={tuple(original['episodes'][0]['paths'][0])}
 block=reserve_block(0,5,set(),reserved)
 assert source.validate(block)
 assert not reserved.intersection(tuple(p) for e in block['episodes'] for p in e['paths'])
 assert block['reserved_rejections']
 assert reserve_block(0,5,set(),reserved)==block
 assert [(e['kind'],e.get('query_context'),e.get('pair'),e['order'],e['mapping'],e['pattern']) for e in block['episodes']]==[(e['kind'],e.get('query_context'),e.get('pair'),e['order'],e['mapping'],e['pattern']) for e in original['episodes']]
 assert source.draw is not None

def test_prespecified_doses_and_final_prefix():
 assert DOSES==[0,240,960,1920,3840,5760,7680,10000]
 assert NEW_DOSES==[1920,3840,5760,7680,10000]
 b=source.make_block(1,42,set());x=execution_prefix(b)
 assert len(x)==160 and x[-1]['episode']==10000
 assert len(execution_prefix(source.make_block(1,41,set())))==240
 assert sum(len(execution_prefix(source.make_block(0,k,set()))) for k in range(5,43))==9040
