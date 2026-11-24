"""Fixed pilot scope and native grammar; no post-outcome choice of thresholds."""
import copy
from research_search.runners.candidate import DEFAULT,validate
from research_search.orchestration.coordinator import leaf_changes

LIMITS=dict(tokens=50000,cpu_seconds=3600,optimizer_steps=6000,wall_seconds=7200,proposals=2,active_branches=1)
BASELINE=copy.deepcopy(DEFAULT);BASELINE['prior']['episodes']=0
SEEDS=[21000,21001]
CALL_RESERVATION=25000
ALLOWED={'model.centered':[True], 'model.cue_timing':['onset'],
         'target.readout_initial_multiplier':[.25,.5,2.],
         'target.update_rule':['slow_recurrence','reset_first_moment','freeze_context','readout_only']}


def validate_candidate(config):
    validate(config);diff=leaf_changes(BASELINE,config)
    if len(diff)!=1 or diff[0] not in ALLOWED:raise ValueError('Pilot requires exactly one registered affordable change')
    group,key=diff[0].split('.')
    if config[group][key] not in ALLOWED[diff[0]]:raise ValueError('Unregistered pilot value')
    return diff


def _object(fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
def _string():return dict(type='string')


def proposal_schema():
    groups={}
    for group,fields in BASELINE.items():
        properties={}
        for key,value in fields.items():
            name=group+'.'+key;values=[value]+ALLOWED.get(name,[])
            kind='boolean' if type(value) is bool else 'integer' if type(value) is int else 'number' if isinstance(value,float) else 'string'
            properties[key]=dict(type=kind,enum=values)
        groups[group]=_object(properties)
    fields={k:_string() for k in ['hypothesis','proposed_change','predicted_behavior','evaluation_plan','novelty_justification']}
    fields|=dict(id=dict(type='string',enum=['pilot2-candidate-01']),parent_candidate=dict(type='null'),
        research_area=dict(type='string',enum=['architecture','learning','initialization']),
        mechanism=_object(dict(field=_string(),explanation=_string())),
        alternative_predictions=dict(type='array',items=_string()),
        matched_control=dict(type='string',enum=['pilot-baseline-zero-prior']),
        required_code_changes=dict(type='array',items=_string()),
        estimated_training_steps=dict(type='integer'),estimated_runtime=dict(type='integer'),estimated_agent_tokens=dict(type='integer'),
        stop_conditions=dict(type='array',items=_string()),
        ablation_primary=_object(dict(metric=dict(type='string',enum=['C_ce','Mixed_ce','D_A','D_B','A_to_B','B_to_A','sustained_cost']),direction=dict(type='string',enum=['min','max']))),
        config=_object(groups))
    return _object(fields)


def critic_schema():
    return _object(dict(failed_prediction=_string(),explanations=dict(type='array',items=_string()),
        technical_assessment=_string(),scientific_usefulness=_string(),limitations=dict(type='array',items=_string()),
        cheapest_discriminating_test=_string(),recommendation=dict(type='string',enum=['retain','revise','abandon','inconclusive'])))
