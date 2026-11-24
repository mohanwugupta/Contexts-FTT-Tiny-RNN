"""Read-only source/init/reuse audit and pretraining gate."""
import numpy as np
import pandas as pd
import torch
from .core import *

def run():
 if (OUT/'audit/pretraining_gate.json').exists():raise FileExistsError('Audit already complete')
 from experiments.recurrent_learning_transfer.design import OUT as NORMAL
 previous=json.loads((NORMAL/'manifest.json').read_text())
 for root,items in [(NORMAL,previous['outputs']),(ROOT,previous['code'])]:
  for n,h in items.items():
   if sha(root/n)!=h:raise ValueError('Sealed normalized input changed')
 inherited=json.loads((NORMAL/'audit/pretraining_gate.json').read_text())['inputs']
 for n,h in inherited.items():
  if sha(ROOT/n)!=h:raise ValueError('Original source/checkpoint changed')
 files=['confirmatory_srn/model.py','confirmatory_srn/training.py','confirmatory_srn/config.py','confirmatory_srn/contexts.py','execution_objective_search/objectives.py','outputs/confirmatory_srn_final/confirmatory_training_audit.csv','outputs/confirmatory_srn_final/confirmatory_training_sequence_level.csv','results/recurrent_learning_transfer/v1/tables/historical_original_trace.csv','results/recurrent_learning_transfer/v1/tables/endpoint_summary.csv','results/mixed_checkpoint_curves/v1/manifest.json','docs/high_scale_initialization/PRD.md']
 inputs={**inherited,**{n:sha(ROOT/n) for n in files}};old=pd.read_csv(ROOT/files[5]);old=old[old.condition=='High_monotonic'];rows=[]
 for seed in range(20):
  arms=pair(seed);p=policy();ref=old[old.seed==seed].iloc[0];t=arms['RNN-Tanh'];initial=state_hash(t,p)
  if initial!=ref.initial_state_sha256 or state_hash(t)!=ref.initial_model_sha256:raise ValueError('Canonical initial draw not reproduced')
  if state_hash(arms['RNN-Linear'],p)!=initial:raise ValueError('New arms not exactly paired')
  for a,m in arms.items():
   if torch.count_nonzero(m.recurrent.weight_ih_l0[:,6]) or any(torch.count_nonzero(v) for n,v in m.named_parameters() if 'bias' in n):raise ValueError('Incorrect canonical neutral channel or biases')
   row=dict(seed=seed,architecture=a,initial_model_hash=state_hash(m),initial_model_policy_hash=state_hash(m,p),canonical_initial_hash_matched=True,model_parameters=sum(v.numel() for v in m.parameters()),policy_parameters=sum(v.numel() for v in p.parameters()),**spectra(m))
   for n,v in m.named_parameters():row[n+'_std']=float(v.detach().std())
   rows.append(row)
 pd.DataFrame(rows).to_csv(OUT/'tables/initialization_audit.csv',index=False)
 code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','tests'] for p in (ROOT/folder/'high_scale_initialization').glob('*.py')}
 dump(OUT/'audit/config.json',CONFIG);dump(OUT/'audit/pretraining_gate.json',dict(passed=True,inputs=inputs,normalized_manifest_sha=sha(NORMAL/'manifest.json'),code=code,canonical_initial_model_policy_matches=20,paired_new_arms=20,recurrent_normalization=False))
 with (OUT/'IMPLEMENTATION_AUDIT.md').open('x') as f:f.write('''# Implementation and source fidelity

All20 canonical initial model and execution-policy hashes reproduce the archived High_monotonic audit. Source module construction consumes default RNN/head initialization before Normal(0,2) draws; this differs from the previous normalized program's direct local-generator draws. New Linear/Tanh seeds are exactly paired; normalized/new integer labels do not establish paired initialization.

Tanh uses the original ConfirmatoryElmanSRN native nn.RNN forward. Linear preserves its six named parameter tensors, biases and neutral third cue channel but replaces the tanh recurrence with explicit affine recurrence. Kernel differences between native Tanh and explicit Linear are documented. No recurrent normalization, rescaling, projection, hidden clipping, extra history/position inputs or altered objective. The source training objective, batched shapes, detached uncertainty, policy, Adam.005 and global jointclip1 are preserved; the330-step curriculum extends the original first170 steps. Source smoke parity verifies every20 pre-update losses and final model/policy against train_condition. Unit smoke updates are isolated fixtures, not scientific trajectories.

Alloriginal checkpoint/source and normalized sealed bytes verified. Exactly40 new trajectories planned,13200 updates maximum; all40 seeds/arms and240 requested checkpoint slots must be sealed before protected probes. Missing/failed cells stay explicit. A/B frozen evaluations and pre-update training loss are distinct. No restoredAdam learning-transfer clones added, so the earlier joint splitting gate is not newly assessed here. Block preference, geometry and post-floor competence are reported separately.

Primary170 before secondary330. Canonical170 new-Tanh tensor/loss parity audit is mandatory before interpretation; even exact reproduction would not make historical copies independent evidence. Normalized comparison is descriptive and unpaired, with initialization-draw and Tanh-kernel differences disclosed. No protected-result tuning or follow-up permitted.
''')
 append('## Pretraining audit passed\nAll20 sourcecanonical initial model/policy hashes and20 new-arm pairs exact. Full source/predecessor hashes verified; zeroWh rescaling. Exactsource20-update smoke parity tested separately,40fixtureupdates. Fixed configuration saved. Proceed once to registered40 scientific trajectories; noMixed probes untilsealed.')
 print('Pretraining gate passed:20 canonical initial hashes,20 paired tensors, no normalization')

if __name__=='__main__':run()
