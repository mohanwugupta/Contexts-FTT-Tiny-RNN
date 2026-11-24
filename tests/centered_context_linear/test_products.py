import numpy as np
import pandas as pd
import pytest
from experiments.centered_context_linear.verify import verify
from experiments.centered_context_linear.figures import validate_inputs
from experiments.centered_context_linear.core import OUT

def test_full_registered_outputs():
 r=verify(sealed=False)
 assert r['scientific_updates']==13200 and r['states']==13240 and r['evaluation_cells']==52960 and r['positions']==370720

def test_figure_inputs_reject_lost_seed():
 f=pd.read_csv(OUT/'tables/frozen_evaluations.csv');validate_inputs(f)
 with pytest.raises(ValueError):validate_inputs(f[f.seed!=19])

def test_report_from_saved_tables():
 from experiments.centered_context_linear.report import main
 main()
 text=(OUT/'REPORT.md').read_text()
 assert '| Standard | 330 |' in text and '| Centered | 330 |' in text
 assert 'Persistent separate learning is not established' in text
 assert len(pd.read_csv(OUT/'tables/claim_ledger.csv'))==10
