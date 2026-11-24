import pytest
from research_search.orchestration.agent_backend import parse_events,subscription_env,validate_proposal
from research_search.orchestration.experiment_tree import frontier

def test_reported_usage_not_cached_or_reasoning_double_counted():
    r=parse_events([{'type':'turn.completed','usage':{'input_tokens':100,'cached_input_tokens':40,'output_tokens':60}},{'type':'item.completed','item':{'type':'agent_message','text':'{}'}}])
    assert r['usage']==dict(input_tokens=100,cached_input_tokens=40,output_tokens=60)
    assert parse_events([{'type':'turn.failed','error':{'message':'quota'}}])['usage'] is None
    with pytest.raises(ValueError):parse_events([{'type':'turn.completed','usage':{}},{'type':'turn.completed','usage':{}}])

def test_no_implicit_paid_credentials_or_nested_agents():
    e=subscription_env({'PATH':'x','OPENAI_API_KEY':'secret','CODEX_API_KEY':'secret','ANTHROPIC_API_KEY':'secret','CODEX_HOME':'x'})
    assert not any(k.endswith('API_KEY') for k in e) and e['PATH']=='x'

def test_pareto_retains_tradeoffs_and_missing_not_zero():
    rows=[dict(id='a',x=2,y=1),dict(id='b',x=1,y=2),dict(id='c',x=0,y=0),dict(id='missing',x=None,y=100)]
    assert frontier(rows,{'x':'max','y':'max'})==['a','b']

def test_proposals_require_falsifiable_controls_and_finite_budget():
    with pytest.raises(ValueError):validate_proposal({'id':'x','hypothesis':'works'})
