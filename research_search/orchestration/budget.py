"""Ceilings are checked inside the durable store transaction, never in agent prompts."""
class BudgetError(RuntimeError):pass

ALLOCATIONS={'setup':100000,'hypothesis':250000,'criticism':250000,'implementation':250000,'reporting':150000}
CALL_LIMITS={'proposal':10000,'coding':20000,'review':8000,'critic':10000,'coordinator':8000}

def usage(value):
    if not isinstance(value,dict):raise ValueError('Provider usage unavailable')
    for key in ['input_tokens','output_tokens','cached_input_tokens']:
        if type(value.get(key)) is not int or value[key]<0:raise ValueError('Missing/invalid provider token field '+key)
    if value['cached_input_tokens']>value['input_tokens']:raise ValueError('Cached input is a subset of input')
    # Codex turn.completed is already aggregate usage. Do not add cached or
    # nested reasoning breakdowns to input+output again.
    total=value['input_tokens']+value['output_tokens']
    if 'total_tokens' in value:
        if type(value['total_tokens']) is not int or value['total_tokens']<total:raise ValueError('Inconsistent provider aggregate')
        total=value['total_tokens']
    return total,value['cached_input_tokens']
