#!/usr/bin/env python3
"""Freeze geometric/training design without evaluating a model or protected cue."""
from pathlib import Path
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from context_sequence.provenance.hashing import object_hash,sha256
from context_sequence.task.novelty import read_registry
from context_sequence.evaluation.cue_grid import grid_registry
from context_sequence.task.replication import sequence_registry,validate_sequences
from context_sequence.dynamics.registry import dump_exclusive,environment


def freeze():
    novel=read_registry(ROOT/'configs/novelty_registry_v1.json','2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727')
    grid=grid_registry(novel);sequences=sequence_registry();assert validate_sequences(sequences).valid.all()
    decision=json.loads((ROOT/'scratchpads/decisions.jsonl').read_text().splitlines()[0])
    factorial={'version':'factorial_v1','base_architecture':'source_matched_8_unit_tanh_unrestricted','scale':2.,'learning_rate':.005,'gradient_clip':1.,'steps':170,'pilot_seeds':list(range(1000,1005)),'confirmatory_seeds':list(range(20)),'replication_seeds':list(range(20)),
        'arms':['F00','F01','F10','F11'],'candidates':decision['pilot_candidates'],'pilot_rules':decision['pilot_rules'],'training_ce':'current A/B context next-token CE only','closure_normalization':'fixed T times H; detached familiar midpoint; recursively inherited neutral state','physical_neutral_cue':[.5,.5,0.],'physical_neutral_equals_Mixed':True,'protected_behavioral_targets_used':False}
    factorial['config_sha256']=object_hash(factorial)
    for name,value in [('context_grid_v1.json',grid),('sequence_registry_v1.json',sequences),('factorial_v1.json',factorial)]:dump_exclusive(ROOT/'configs'/name,value)
    paper=json.loads((ROOT/'configs/paper_freeze_v2.json').read_text())
    oldfiles=subprocess.check_output(['git','ls-tree','-r','--name-only','1240883','src/context_sequence'],cwd=ROOT,text=True).splitlines()
    oldscope={p:sha256(ROOT/p) for p in oldfiles if p.endswith('.py')}
    assert object_hash(oldscope)==paper['source_code_sha256'],'Historical paper source code changed'
    inputs={x['path']:x['sha256'] for x in paper['input_tables']}
    for r in json.loads((ROOT/'scratchpads/dynamical_generalization_reference_index.json').read_text())['references']:inputs[r['path']]=r['sha256']
    extras=['configs/context_grid_v1.json','configs/sequence_registry_v1.json','configs/factorial_v1.json','scratchpads/dynamical_generalization_reference_index.json','results/paper/v2_20260925_novelty30_60_r2/manifest.json','analysis/phase1i_context_composition/phase1i_checkpoint_provenance.csv','analysis/phase1l/phase1l_provenance.csv']
    for seed in range(20):extras.append(f'outputs/confirmatory_srn_final/confirmatory_checkpoints/High_sequence_only_seed_{seed}.pt')
    for p in extras:inputs[p]=sha256(ROOT/p)
    sources=list((ROOT/'src/context_sequence').rglob('*.py'))
    sources += [ROOT/p for p in ['confirmatory_srn/model.py','confirmatory_srn/config.py','confirmatory_srn/training.py','phase1l_contrast_closure/model.py','phase1l_contrast_closure/training.py','ftt_srn_execution_v2/training/ftt_schedule.py']]
    sources += list((ROOT/'scripts').glob('*dynamical_generalization*.py'))
    sources += [ROOT/'tests/unit'/p for p in ['test_e1_dynamics.py','test_e2_grid.py','test_e3_factorial.py','test_e4_sequences.py']]
    sources += [ROOT/'tests/regression/test_historical_parity.py',ROOT/'tests/integration/test_dynamical_program.py']
    config={'version':'dynamical_generalization_v1','freeze_id':'dg_v1_20261103','identity':'new prospective extension; not part of original confirmatory paper freeze','novelty_registry':'configs/novelty_registry_v1.json','novelty_registry_sha256':novel['registry_sha256'],'cue_grid':'configs/context_grid_v1.json','sequence_registry':'configs/sequence_registry_v1.json','factorial_config':'configs/factorial_v1.json','checkpoints':paper['checkpoints'],
        'input_files':[{'path':p,'sha256':v} for p,v in sorted(inputs.items())],'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(sources))},'historical_paper_source_scope':oldscope,'historical_paper_source_sha256':object_hash(oldscope),'tolerances':decision['tolerances'],'cv_protocol':decision['cv_protocol'],'evaluator':'numpy_explicit_recurrence_v1; float64 diagnostics and smooth-real Jacobians','bootstrap_seed':20260925,'bootstrap_draws':10000,'seeds':list(range(20)),**environment()}
    config['config_sha256']=object_hash(config);dump_exclusive(ROOT/'configs/dynamical_generalization_v1.json',config)
    print(config['config_sha256']);print(validate_sequences(sequences)[['sequence_id','tokens','replaced_equivalent_candidate']].to_string(index=False))
if __name__=='__main__':freeze()
