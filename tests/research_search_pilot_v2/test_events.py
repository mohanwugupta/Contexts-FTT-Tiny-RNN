import json
from pathlib import Path
import pytest
from research_search.common import PACKAGE,read
from experiments.research_search_pilot_v2.events import assess,NOTICE


def fixture(message=NOTICE,late=False):
    notice=dict(type='item.completed',item=dict(type='error',message=message))
    start=dict(type='turn.started')
    rest=[dict(type='item.completed',item=dict(type='agent_message',text='{"ok":true}')),dict(type='turn.completed',usage=dict(input_tokens=14000,output_tokens=15,cached_input_tokens=0))]
    return ([start,notice] if late else [notice,start])+rest


def test_exact_early_notice_retained_not_misclassified_as_tool():
    r=assess(dict(returncode=0),fixture(),25000)
    assert r['passed'] and r['notices']==[NOTICE] and r['reported_tokens']==14015


@pytest.mark.parametrize('message,late',[(NOTICE+' extra',False),('real auth failure',False),(NOTICE,True)])
def test_other_errors_are_not_whitelisted(message,late):
    assert not assess(dict(returncode=0),fixture(message,late),25000)['passed']


def test_duplicate_tools_missing_usage_and_budget_still_fail():
    base=fixture()
    for bad in [[base[0]]+base,base+[dict(type='item.completed',item=dict(type='command_execution'))],base[:-1]]:
        assert not assess(dict(returncode=0),bad,25000)['passed']
    assert not assess(dict(returncode=0),base,14000)['passed']
    assert not assess(dict(returncode=1),base,25000)['passed']


def test_cached_v4_replays_without_rewriting_original_failure():
    old=read(PACKAGE/'database/backend_validation_v4/receipt.json')
    events=[json.loads(x) for x in (Path(old['raw_event_directory'])/'events.jsonl').read_text().splitlines()]
    r=assess(dict(returncode=0),events,50000)
    assert r['passed'] and old['passed'] is False and old['unexpected_tool_item'] is True
