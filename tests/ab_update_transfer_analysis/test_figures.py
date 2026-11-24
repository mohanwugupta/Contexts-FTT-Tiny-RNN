import ast,inspect
from experiments.ab_update_transfer_analysis import figures

def test_table_only_figures_and_required_families():
 tree=ast.parse(inspect.getsource(figures));names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
 assert {'actual_curves','gain_matrices','joint','starting_error','parameter_effects','adam_history'}.issubset(names)
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   text=ast.unparse(n);assert not any(x in text for x in ['adapter','engine','torch','core'])
  if isinstance(n,ast.Call):assert not (isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward'])
