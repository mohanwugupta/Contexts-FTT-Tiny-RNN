"""One cached-analysis entry point; deterministic external regeneration."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import analysis as a

ARTIFACTS=['comparison.csv','positions.csv','paired.csv','position_differences.csv','early.csv','practice.csv.gz','endpoint.csv','seed0_C_positions.csv','seed0_NoContext_positions.csv','seed1_C_positions.csv','seed1_NoContext_positions.csv','REPORT.md','POSITION_TABLES.md','decisions.csv','REPRODUCIBILITY.md','source_audit.json','missing_inputs.json','validation.json','figure_1_readout_comparison.png','figure_1_readout_comparison.svg','figure_2_split_and_position_errors.png','figure_2_split_and_position_errors.svg']
def reproduce_doc(out):
 (out/'REPRODUCIBILITY.md').write_text('''# Cached reproduction

Run from repository root in the existing numerical environment:

```sh
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.a06_cached_readout_comparison --output-dir /tmp/a06_readout_new --compare-with results/a06_cached_readout_comparison/v1
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.a06_cached_readout_comparison --verify results/a06_cached_readout_comparison/v1
```

Use a new empty external output directory. The canonical package's manifest pins code/PRD, every output and all read inputs. source_audit.json gives exact paths/hashes and static-view lineage; missing_inputs.json records missing paths/keys and unmatched checkpoints. The source is results/joint_phenotype_hypothesis_screen/v1; its sealed selection, thresholds and files are unchanged. No checkpoint was needed to recover bias in this completed run.

Input mapping: recorded per-run metrics labels and resolved_config.target.probe_updates match the pinned producer engine.TIMES/PROBES order, never inferred from array indices. All10 recorded updates0/10/20/30/40/50/60/100/170/330 are mutually available. Native core.TARGET=[1,2,0,3,1,2,3], token order[0,1,2,0,3,1,2], all seven positions weighted1/7; all four classes retained. Physical Mixed-average logits are scored directly, never averaged losses. Source native float32 logits are promoted stored values; source core.score reuses float64 stable softmax/CE/margin scoring.

For A00-static use the same checkpoint's bias:0.25*(z_A00-b_A00)+b_A00. Cached original A06_static files are found through registry/derived_controls.csv; both views match exactly. A06 saved logits are not rescaled. Native source.from_logits and core.score are reused. Checkpoint loading solely for missing bias is supported; no model construction/forward is permitted. Missing essential data never authorize a rollout or new training.

comparison.csv contains all300 seed/checkpoint/cue/arm rows and the three requested paired CE/count differences. Negative CE delta favors the first arm; positive correct-count delta favors it. positions.csv holds all2100 position rows with all four logits/probabilities, target/predicted class, target probability, margin and traceable source. paired.csv and position_differences.csv expose every pair and signed contribution ΔpositionCE/7; each seven-position contribution sum recovers aggregate ΔCE. seed1_C_positions.csv and seed1_NoContext_positions.csv are seven-row wide comparisons. Seed0 has the same exports and a compact companion in POSITION_TABLES.md.

early.csv copies original metrics/eligibility/floors/classifications; original source functions cross-check signed own/cross/D/Q and raw/normalized/directional/sustained switching. Original joint flags and selection remain frozen. Static has no independent practice history. practice.csv.gz contains only the four actual A00/A06 histories,1320 updates. Figure2 shows early1–60 and full330 practice. No positions/checkpoints are new network replications; no new threshold for similar gains is applied.

Only the PRD acceptance checks are run: archived CE/count agreement under the inherited1e−12 scoring contract; cached static bias formula under the existing double geometry tolerance; all positions/aggregate arithmetic; recorded labels/coverage; copied early classifications. Source array matches are exact here. Guards prohibit model construction, forward/trace/objective, backward/autograd, Adam/SGD.step and native optimizer use. Zero new training, optimizer steps, fitted parameters or confirmation seeds. No dependency or training-infrastructure change.
''')
def run(out):
 out=Path(out)
 if (out/'manifest.json').exists():return verify(out)
 torch.set_num_threads(1)
 with a.cached_only() as calls:
  result=a.build(out)
  from .reporting import render as report
  from .figures import render as figures
  report(out,*result);figures(out,result[0],result[1],result[5]);reproduce_doc(out)
  a.put(out/'validation.json',result[-1]|dict(prohibited_calls=len(calls)))
 audit=json.loads((out/'source_audit.json').read_text())
 for name,h in audit['inputs'].items():assert a.sha(a.ROOT/name)==h,'Changed source '+name
 print(json.dumps(result[-1]|dict(prohibited_calls=len(calls))),flush=True)
 return result[-1]
def verify(out):
 out=Path(out);m=json.loads((out/'manifest.json').read_text());actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'};assert actual==set(m['outputs'])
 for name,h in m['outputs'].items():assert a.sha(out/name)==h,name
 for name,h in (m['inputs']|m['code']).items():assert a.sha(a.ROOT/name)==h,name
 print(json.dumps(dict(passed=True,outputs=len(m['outputs']),inputs=len(m['inputs']),optimizer_steps=0)),flush=True);return m
def seal(out=a.OUT):
 out=Path(out);assert json.loads((out/'audit/visual_review.json').read_text())['passed'];assert json.loads((out/'audit/reproduction.json').read_text())['passed'];v=json.loads((out/'validation.json').read_text());assert v['passed'] and v['prohibited_calls']==0
 visual=json.loads((out/'audit/visual_review.json').read_text());assert visual['hashes']=={n:a.sha(out/n) for n in ARTIFACTS if n.endswith(('.png','.svg'))}
 code={str(p.relative_to(a.ROOT)):a.sha(p) for folder in ['experiments/a06_cached_readout_comparison','tests/a06_cached_readout_comparison'] for p in (a.ROOT/folder).glob('*.py')};code['docs/a06_cached_readout_comparison/PRD.md']=a.sha(a.ROOT/'docs/a06_cached_readout_comparison/PRD.md')
 code['experiments/a06_cached_readout_comparison/README.md']=a.sha(a.ROOT/'experiments/a06_cached_readout_comparison/README.md')
 inputs=json.loads((out/'source_audit.json').read_text())['inputs'];outputs={str(p.relative_to(out)):a.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'};a.put(out/'manifest.json',dict(program='a06_cached_readout_comparison/v1',inputs=inputs,code=code,outputs=outputs,scientific_artifacts=ARTIFACTS,validation=v));verify(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(a.OUT));p.add_argument('--verify');p.add_argument('--compare-with');args=p.parse_args()
 if args.verify:verify(args.verify);return
 out=Path(args.output_dir).resolve();assert out!=a.SOURCE.resolve() and not out.is_relative_to(a.SOURCE.resolve());run(out)
 if args.compare_with:
  base=Path(args.compare_with);hashes={n:a.sha(out/n) for n in ARTIFACTS};assert hashes=={n:a.sha(base/n) for n in ARTIFACTS};a.put(out/'reproduction.json',dict(passed=True,scientific_artifacts=len(hashes),hashes=hashes,optimizer_steps=0));print('All22 scientific artifacts reproduce byte-identically',flush=True)
if __name__=='__main__':main()
