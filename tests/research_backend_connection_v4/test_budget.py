import json
from pathlib import Path
import pytest
from research_search.common import ROOT,PACKAGE,read
from experiments.research_backend_connection_v4.probe import TOKEN_RESERVATION,STATE,assess_reply


def events(total):
    return [dict(type='item.completed',item=dict(type='agent_message',text='{"fixture":true}')),dict(type='turn.completed',usage=dict(input_tokens=total-15,output_tokens=15,cached_input_tokens=0))]


def test_new_allowance_separate_from_exhausted_v3():
    assert TOKEN_RESERVATION==50000
    assert STATE==PACKAGE/'database/backend_validation_v4'
    old=read(PACKAGE/'database/backend_validation_v3/receipt.json')
    assert old['ledger']['status']=='budget_exhausted' and old['reported_tokens']==14191


@pytest.mark.parametrize('total,passes',[(14191,True),(50000,True),(50001,False)])
def test_exact_token_ceiling(total,passes):
    r=assess_reply(dict(returncode=0,timed_out=False,cancelled=False),events(total))
    assert r['budget_passed'] is passes and r['passed'] is passes


def test_cached_diagnostic_check_is_not_changed_by_cap():
    prior=read(PACKAGE/'database/backend_validation_v3/receipt.json')
    raw=Path(prior['raw_event_directory'])/'events.jsonl'
    r=assess_reply(dict(returncode=0,timed_out=False,cancelled=False),[json.loads(x) for x in raw.read_text().splitlines()])
    assert r['budget_passed'] is True and r['reply_exact'] is True
    assert r['unexpected_tool_item'] is True and r['passed'] is False


def test_missing_usage_remains_unknown():
    r=assess_reply(dict(returncode=0),[])
    assert not r['passed'] and r['reported_tokens'] is None and r['budget_passed'] is None
