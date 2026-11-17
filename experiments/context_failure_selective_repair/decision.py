"""R4 is a decision memo only. This module has no model/optimizer execution."""
import json
from .core import dump,report,read,table


def validate_claim(claim):
 unsupported={'natural retrieval','lost sequence information','global unreachability','human mechanism','attractor'}
 if unsupported.intersection(claim.get('asserts',[])):raise ValueError('Unsupported mechanistic label')
 return True

def recommendation(evidence):
 if not evidence['selective_mechanism_gate']:
  return dict(recommendation='no_new_training',scientific_updates=0,design_only=True,diagnostic='First audit the fixed intervention rays against the tanh-interior tangent cone, from saved states/bases only, to distinguish absence of an admissible matched intervention from absence of a useful computation. Any new intervention geometry needs a separate fixed protocol; do not relax margin or retune U here.')
 return dict(recommendation='separate_fixed_protocol_required',scientific_updates=0,design_only=True,diagnostic='Derive a familiar-only U-preservation constraint and its W matched objective before any learning; protected cues excluded from fitting/selection.')

def run(out):
 e=json.loads((out/'audit/r3_decision.json').read_text());r=recommendation(e);dump(out/'audit/r4_decision.json',r)
 report(out,'R4_LEARNING_FOLLOWUP_DECISION.md','# R4 — learning follow-up decision, design only\n\nRecommendation: **'+r['recommendation']+'**. Scientific weights updated:0. The prespecified selective mechanism gate failed. We therefore propose no new training objective or curriculum. D3 already worsened familiar competence without clear Mixed benefit, and the present data do not justify another training experiment.\n\n'+r['diagnostic']+'\n\nThe smallest theory-revising diagnostic is geometric, table-only: identify coordinates that forbid each fixed U/W/random ray at natural Mixed poststates, and compare their required lower/upper ray intervals. It asks whether this protocol can implement its matched intervention at all. It cannot establish that no feasible beneficial state exists. If a subsequent study changes the admissible intervention family, freeze it before outcomes and retain the current failed/infeasible record. No silent margin reduction, clipping, sign search or optimization is authorized.\n\nAny later mechanism-specific learning study requires its own protocol, familiar-only training and selection, preservation of the original full loss/policy contract, competence and raw/output-space separation checks, and protected Mixed/NoContext/Novel evaluation after fitting. No such study is executed or approved by this memo.\n')
 return r
