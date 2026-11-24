import copy
import pytest
from experiments.research_search_integration_v1_1.canonical import canonical_hash, normalize_config, prior_identity
from experiments.research_search_integration_v1_1.evaluation import four_outcome
from experiments.research_search_integration_v1_1.protocol import LIMITS, RESERVATIONS, validate_proposal
from research_search.runners.candidate import DEFAULT


def test_numeric_semantics_and_provenance():
    assert canonical_hash({'x':5,'y':[-0.0,1e-3]})==canonical_hash({'x':5.0,'y':[0,.001]})
    assert canonical_hash({'x':True})!=canonical_hash({'x':1})
    assert canonical_hash({'x':5.01})!=canonical_hash({'x':5})
    assert canonical_hash({'x':'5'})!=canonical_hash({'x':5})
    for x in [float('nan'),float('inf')]:
        with pytest.raises(ValueError):canonical_hash({'x':x})
    a=copy.deepcopy(DEFAULT);b=copy.deepcopy(a);b['model']['nominal_scale']=5
    assert normalize_config(a)==normalize_config(b)
    assert type(normalize_config(b)['model']['nominal_scale']) is float
    p=dict(source_sha='a',code_sha='b',task_exclusion_sha='c',rng_contract='native',adam_contract='native')
    assert prior_identity(a,21000,p)==prior_identity(b,21000,p)
    b['target']['readout_initial_multiplier']=.5
    assert prior_identity(a,21000,p)==prior_identity(b,21000,p)
    b['prior']['episodes']=240
    assert prior_identity(a,21000,p)!=prior_identity(b,21000,p)
    assert prior_identity(a,21000,p)!=prior_identity(a,21001,p)
    assert prior_identity(a,21000,p)!=prior_identity(a,21000,p|{'task_exclusion_sha':'different'})
    for group,key,value in [('model','hidden_dim',8.0),('model','hidden_dim',True),('target','steps',330.0),('model','nominal_scale',True)]:
        b=copy.deepcopy(a);b[group][key]=value
        with pytest.raises(ValueError):normalize_config(b)


def row():
    return dict(familiar=True,original_early_phenotype=True,joint=False,early=False,reference_status='reference_unresolved',net_A=-1.,net_B=2.,A_to_B=-.4,B_to_A=.3,sustained_cost=-.2)


def endpoint():
    return {c:dict(ce=.5,correct_count=6) for c in ['A','B','C','Mixed-average','NoContext']}


def test_four_outcome_conjunction_and_legacy_flags():
    r=row();e=endpoint();v=four_outcome(r,e)
    assert v['four_outcome_feasible_v1'] is True
    assert v['original_joint'] is False and v['screen_preservation_valid'] is None
    assert r['joint'] is False and r['net_A']==-1
    e['NoContext']['correct_count']=5
    assert four_outcome(r,e)['four_outcome_feasible_v1'] is False
    e=endpoint();e['C']['ce']=.500001
    assert four_outcome(r,e)['four_outcome_feasible_v1'] is False
    e=endpoint();del e['NoContext']
    assert four_outcome(r,e)['four_outcome_feasible_v1'] is None
    r=row();r['original_early_phenotype']=False
    assert four_outcome(r,endpoint())['four_outcome_feasible_v1'] is False
    r=row();r['reference_status']='resolved';r['early']=True
    assert four_outcome(r,endpoint())['screen_preservation_valid'] is True
    e=endpoint();e['A']['correct_count']=5
    assert four_outcome(row(),e)['four_outcome_feasible_v1'] is False
    e=endpoint();e['C']['correct_count']=6.5
    with pytest.raises(ValueError):four_outcome(row(),e)


def test_protocol_budgets_no_automatic_expansion():
    assert LIMITS['tokens']==100000 and LIMITS['wall_seconds']==21600
    assert LIMITS['optimizer_steps']==0 and LIMITS['cpu_seconds']==900
    assert LIMITS['active_branches']==1 and LIMITS['proposals']==3
    assert sum(RESERVATIONS.values())==100000
    p=dict(id='initialization-proposal',research_area='initialization',analysis='confidence_vs_correctness',hypothesis='less confidence on errors',prediction='CE without corrections',alternatives=['learned trajectory','confidence'],novelty='cached control',cost_reason='zero steps',stop_conditions=['missing cache'])
    validate_proposal(p,'initialization')
    with pytest.raises(ValueError):validate_proposal(p|{'analysis':'new_training'},'initialization')
    with pytest.raises(ValueError):validate_proposal(p,'learning')


def test_zero_update_ceiling_rejects_any_training(tmp_path):
    from research_search.orchestration.store import Store
    from research_search.orchestration.budget import BudgetError
    from research_search.orchestration.queue import enqueue
    s=Store(tmp_path,LIMITS)
    with pytest.raises(BudgetError):s.charge_step('forbidden',1)
    with pytest.raises(BudgetError):enqueue(s,'training','candidate',dict(kind='target'))
    assert s.totals()['optimizer_steps']==0
    s.close()
