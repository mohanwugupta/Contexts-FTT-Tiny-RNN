"""Read-only full-trace coverage adapter; frozen training math is unchanged."""
from .common import *
from . import train
from experiments.heldout_context_meta_scaling_10k.adapters import bind
ORIGINAL_RUN=train.run_meta
ORIGINAL_TRACE=train.trace_dose

def trace_dose(arm,dose,initial_dose):
 return ORIGINAL_TRACE(arm,dose,initial_dose) or dose==(1922 if arm=='B' else 1921)

def worker(job):
 with bind(train,trace_dose=trace_dose):return ORIGINAL_RUN(job)

def main():
 receipt=OUT/'audit/continuation_trace_policy.json'
 if not receipt.exists():dump(receipt,dict(adapter_sha=sha(Path(__file__)),extra_full_trace_doses={'B':1922,'T':1921,'O1':1921,'O2':1921},additional_optimizer_steps=0,critical_code=critical_codes()))
 else:assert json.loads(receipt.read_text())['adapter_sha']==sha(Path(__file__))
 with bind(train,run_meta=worker):train.main('D')
