"""Narrow pre-turn diagnostic qualification; all other failures stay visible."""
import json
from research_search.orchestration.agent_backend import parse_events
from research_search.orchestration.budget import usage

NOTICE='Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.'


def assess(receipt,events,allowance):
    seen_turn=False;notices=[];rejected=[]
    for event in events:
        if event.get('type')=='turn.started':seen_turn=True
        if event.get('type') in ['item.started','item.completed']:
            item=event.get('item',{});kind=item.get('type')
            if kind=='error' and event['type']=='item.completed' and item.get('message')==NOTICE and not seen_turn and not notices:
                notices.append(NOTICE)
            elif kind not in ['agent_message','reasoning']:
                rejected.append(dict(event=event.get('type'),item_type=kind,message=item.get('message')))
    try:
        parsed=parse_events(events);total,cached=usage(parsed['usage'])
    except (ValueError,KeyError,TypeError):
        return dict(passed=False,status='missing_or_invalid_usage',reported_usage=None,reported_tokens=None,notices=notices,rejected_items=rejected,reply=None)
    try:reply=json.loads(parsed['text']);valid=isinstance(reply,dict)
    except (ValueError,TypeError):reply=None;valid=False
    passed=bool(valid and not parsed['failed'] and not parsed['nested_agent'] and not rejected and receipt['returncode']==0 and not receipt.get('timed_out') and not receipt.get('cancelled') and total<=allowance)
    return dict(passed=passed,status='accepted' if passed else 'execution_or_reply_or_usage_failure',reported_usage=parsed['usage'],reported_tokens=total,cached_input_tokens=cached,notices=notices,rejected_items=rejected,reply=reply)
