"""Execute the originally sealed gate; stop or authorize the fixed meta protocol."""
from .common import *
from .gate import decide

def main():
 assert json.loads((OUT/'audit/preflight_verified.json').read_text())['passed'] and json.loads((OUT/'audit/preflight_analysis_complete.json').read_text())['passed'];seal=json.loads((OUT/'audit/initial_registry_seal.json').read_text());registered=seal['code']['experiments/high_scale_constrained_meta/gate.py'];assert sha(ROOT/'experiments/high_scale_constrained_meta/gate.py')==registered,'Gate source changed since initial registry seal';f=read('preflight_seed_outcomes');r=decide(f);dump(OUT/'audit/preflight_decision.json',dict(**r,gate_source_sha=registered,preregistration_sha=sha(OUT/'registry/PREREGISTRATION.md'),seed_table_sha=sha(OUT/'tables/preflight_seed_outcomes.csv'),time=time.time()));write_frame(OUT/'tables/preflight_gate_checks.csv',[dict(check=k,passed=v) for k,v in r['checks'].items()]);append('## Mandatory preflight decision\n'+json.dumps(r,sort_keys=True)+'\n'+('Fixed Reptile/FOMAML arms authorized, unchanged source algorithms only.' if r['passed'] else 'STOP: no Reptile or FOMAML scientific training, no held-out diagnostic clones of pretrained models, no rescue/retuning. Preserve all preflight rows and issue stopped-stage reports.'))
 if not r['passed']:
  dump(OUT/'audit/meta_stage_status.json',dict(status='not_run_preflight_gate_failed',reptile_episodes=0,fomaml_episodes=0,meta_inner_updates=0,meta_outer_updates=0,projected_U_P_comparison='not_established',reason=r['checks'],no_rescue=True));print('Preflight FAILED: meta-training stopped by fixed protocol',flush=True)
 else:print('Preflight PASSED: fixed meta protocol authorized',flush=True)
if __name__=='__main__':main()
