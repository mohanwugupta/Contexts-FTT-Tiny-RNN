"""Additive four-outcome criterion; original scorer/flags remain untouched."""
import math
CUES=['A','B','C','Mixed-average','NoContext']


def four_outcome(summary,endpoint):
    useful={}
    for cue in CUES:
        cell=endpoint.get(cue)
        if cell is None or cell.get('ce') is None or cell.get('correct_count') is None:
            useful[cue]=None;continue
        ce=float(cell['ce']);count=float(cell['correct_count'])
        if not math.isfinite(ce) or not math.isfinite(count):useful[cue]=None;continue
        if count!=int(count) or not 0<=count<=7 or ce<0:raise ValueError('Invalid scalar cell')
        useful[cue]=bool(ce<=.5 and count>=6)
    complete=all(useful[c] is not None for c in CUES) and type(summary.get('original_early_phenotype')) is bool
    feasible=bool(all(useful.values()) and summary['original_early_phenotype']) if complete else None
    return dict(four_outcome_feasible_v1=feasible,endpoint_complete=complete,
        **{c.replace('-average','')+'_useful_v1':useful[c] for c in CUES},
        original_joint=summary.get('joint'),original_early_phenotype=summary.get('original_early_phenotype'),
        original_screen_flag=summary.get('early'),
        screen_preservation_valid=summary.get('early') if summary.get('reference_status')=='resolved' else None,
        screen_reference_status=summary.get('reference_status','missing'),
        original_late_functional=summary.get('original_late_functional'))
