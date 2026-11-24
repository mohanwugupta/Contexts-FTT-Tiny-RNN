from experiments.context_experience_prior.report import render_report,claim_ledger

def test_report_has_fixed_question_and_limits():
 import inspect
 s=inspect.getsource(render_report)
 for text in ['Reptile','NoContext','C','functional','floor','P2','paired','temporal','natural','20']:
  assert text in s
 assert callable(claim_ledger)
