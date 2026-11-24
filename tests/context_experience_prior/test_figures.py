import inspect
from experiments.context_experience_prior.figures import draw_all,OUT_NAMES

def test_five_primary_and_table_only_graphics():
 assert all(f'figure_{i}' in OUT_NAMES for i in range(1,6))
 src=inspect.getsource(__import__('experiments.context_experience_prior.figures',fromlist=['x']))
 assert 'torch.load' not in src and '.backward(' not in src and 'optimizer(' not in src
 assert callable(draw_all)
