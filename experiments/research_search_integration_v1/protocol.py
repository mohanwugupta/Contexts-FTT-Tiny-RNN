"""Frozen scope, schemas and finite cached analysis grammar."""
from pathlib import Path
from research_search.common import ROOT,PACKAGE
LIMITS=dict(tokens=100000,cpu_seconds=900,optimizer_steps=0,wall_seconds=21600,proposals=3,active_branches=1)
RESERVATIONS=dict(specialist_initialization=19000,specialist_learning=19000,coordinator_select=21000,critic=21000,coordinator_review=20000)
PHASES=list(RESERVATIONS)
ANALYSES=dict(initialization='confidence_vs_correctness',learning='acquisition_vs_interference')
DIRECTORY=PACKAGE/'database/integration_v1'
OUT=ROOT/'results/research_search_integration/v1'
SOURCE=PACKAGE/'database/controlled_pilot_v2'
PUBLIC=ROOT/'results/research_search_pilot/v2'


def obj(fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
def string():return dict(type='string')
def strings():return dict(type='array',items=string(),minItems=1,maxItems=4)


def proposal_schema(area):
    return obj(dict(id=dict(type='string',enum=[area+'-proposal']),research_area=dict(type='string',enum=[area]),analysis=dict(type='string',enum=[ANALYSES[area]]),hypothesis=string(),prediction=string(),alternatives=strings(),novelty=string(),cost_reason=string(),stop_conditions=strings()))


def validate_proposal(p,area):
    expected=set(proposal_schema(area)['required'])
    if set(p)!=expected or p['research_area']!=area or p['analysis']!=ANALYSES[area] or p['id']!=area+'-proposal':raise ValueError('Unregistered specialist proposal')
    if len(p['alternatives'])<2 or not all(isinstance(p[k],str) and p[k].strip() for k in ['hypothesis','prediction','novelty','cost_reason']):raise ValueError('Incomplete falsifiable proposal')
    if len(str(p).split())>700:raise ValueError('Proposal too long')
    return p


def schema(role):
    if role.startswith('specialist_'):return proposal_schema(role.split('_',1)[1])
    if role=='coordinator_select':return obj(dict(selected=dict(type='string',enum=[a+'-proposal' for a in ANALYSES]),comparison=string(),duplication_check=string(),scientific_value=string(),cost_assessment=string(),selection_reason=string(),rejected_hypothesis_reason=string(),scientific_scope=dict(type='string',enum=['cached_descriptive_inconclusive_for_pretrained_mechanism'])))
    if role=='critic':return obj(dict(failed_outcomes=strings(),competing_explanations=strings(),scientific_usefulness=string(),mechanistic_limits=string(),cheapest_followup=string(),recommendation=dict(type='string',enum=['reject','revise','retain_for_separate_test','inconclusive'])))
    if role=='coordinator_review':return obj(dict(decision=dict(type='string',enum=['reject','revise_design_only','investigate_later','inconclusive']),evidence=string(),critic_response=string(),next_step_design_only=string(),readiness_limits=strings(),stop_now=dict(type='boolean',enum=[True])))
    raise ValueError('Unknown role')
