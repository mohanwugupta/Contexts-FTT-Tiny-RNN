import ast
from pathlib import Path
from experiments.high_scale_constrained_scaling_10k import figures as f

def test_table_only_all_seven_families_and_no_scientific_import():
 assert f.FAMILIES==list(range(1,8))
 tree=ast.parse(Path(f.__file__).read_text())
 assert not any(isinstance(x,ast.ImportFrom) and (x.level or 'torch' in (x.module or '') or 'experiments' in (x.module or '')) for x in ast.walk(tree))
 assert not any(isinstance(x,ast.Import) and any('torch' in n.name for n in x.names) for x in ast.walk(tree))
 assert f.VIEWS==[330,100,20,990]
 assert f.METRIC_FAMILIES[3]==['D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost']
 assert f.METRIC_FAMILIES[4]==['C_ce','gap','Mixed_average_ce','Mixed_both_ce','NoContext_ce','strict_C','Mixed_both_success']

def test_reptile_retention_is_separate_from_FOMAML_zero_alias():
 assert f.REPTILE_RETENTION_DOSES==[0,240,960,1920]
 assert f.EXPECTED_PAIRS==46

def test_joint_acquisition_marker_does_not_select_or_relabel_missing():
 import numpy as np,pandas as pd
 g=pd.DataFrame(dict(seed=[0,1,2],strong_familiar=[1.,0.,np.nan],gap=[-2.,-1.,0.]))
 assert f.familiar_failures(g).seed.tolist()==[1]
 assert len(g)==3
