"""Decision-gated prospective program. Evaluation never falls back to training."""
from pathlib import Path
import json
import itertools
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.hashing import sha256,object_hash
from context_sequence.provenance.registry import load_model,audit_inputs
from context_sequence.provenance.regression import historical_regression
from context_sequence.task.novelty import read_registry,cues_from_registry
from context_sequence.evaluation.cue_grid import cue_geometry
from context_sequence.analysis.aggregation import seed_summary,paired_contrasts,bootstrap_effects
from context_sequence.analysis.contrasts import grid_controls,projection_invariants,familiar_feasibility
from context_sequence.analysis import figures,reporting
from context_sequence.task.replication import validate_sequences
from .closure import decompose,static_audit,single_patch_replay
from .propagation import grouped_predictions
from .registry import load_config,read_hashed,audit,dump_exclusive,finalize,verify_stage,environment


def check_mode(stage,mode):
    allowed={'e1':{'eval_only'},'e2':{'eval_only'},'e3':{'ab_pilot','confirmatory'},'e4':{'confirmatory'},'report':{'eval_only'}}
    if stage not in allowed or mode not in allowed[stage]:raise ValueError(f'Forbidden stage/mode combination: {stage}/{mode}; no implicit training fallback')


@contextmanager
def evaluation_firewall():
    def denied(*a,**k):raise RuntimeError('Optimizer/backward forbidden during frozen evaluation')
    with patch.object(torch.optim.Optimizer,'__init__',denied),patch.object(torch.optim.Adam,'step',denied),patch.object(torch.optim.SGD,'step',denied),patch.object(torch.Tensor,'backward',denied):yield


def save(out,name,frame):
    p=out/'tables'/f'{name}.csv'
    if p.exists():raise FileExistsError('Immutable output exists: '+str(p))
    frame.to_csv(p,index=False)


def metadata(record):
    return {k:record[k] for k in ['cohort','condition','family','seed','model_family','training_objective','checkpoint_sha256']}


def attach(frame,record,probe,vector):
    f=frame.copy()
    for k,v in metadata(record).items():f[k]=v
    f['probe']=probe;f['sequence_id']='original'
    for k,v in cue_geometry(vector).items():f[k]=v
    return f


def _integrity(rows):
    if rows.identity_error.max()>2e-6:raise AssertionError('Exact nonlinear recurrence identity failed')
    if not np.isfinite(rows.select_dtypes('number').to_numpy()).all():
        # Angle at zero is deliberately undefined, and is not an arithmetic failure.
        numeric=rows.select_dtypes('number').drop(columns=['nearest_endpoint_angle_degrees'],errors='ignore')
        if not np.isfinite(numeric.to_numpy()).all():raise AssertionError('Nonfinite dynamical measurement')


def run_e1(root,out,config):
    cues=cues_from_registry(read_registry(root/config['novelty_registry'],config['novelty_registry_sha256']))
    natural_parts=[];static_parts=[];patch_parts=[]
    with evaluation_firewall():
        regression=historical_regression(root,config['checkpoints'])
        regression.to_csv(out/'audit/e1_historical_regression.csv',index=False)
        if not regression.passed.all():raise AssertionError('Historical evaluator regression failed; descendants blocked')
        for i,r in enumerate(config['checkpoints'],1):
            model=load_model(root,r);before=model.digest()
            static=static_audit(model)
            if static.affine_error.max()>1e-10 or static.convexity_gap.min() < -1e-10:raise AssertionError('Static midpoint affine/convexity invariant failed')
            for k,v in metadata(r).items():static[k]=v
            static_parts.append(static)
            for name,cue in cues.items():
                f=attach(decompose(model,cue),r,name,cue);_integrity(f);natural_parts.append(f)
                p=attach(single_patch_replay(model,cue),r,name,cue);_integrity(p);patch_parts.append(p)
            if model.digest()!=before:raise AssertionError('Frozen in-memory parameters mutated')
            if i%10==0:print(f'E1 frozen checkpoints {i}/{len(config["checkpoints"])}',flush=True)
    transitions=pd.concat(natural_parts,ignore_index=True);static=pd.concat(static_parts,ignore_index=True);patched=pd.concat(patch_parts,ignore_index=True)
    summary=seed_summary(transitions);pred=grouped_predictions(summary[summary.probe.isin(list(cues))])
    changes=paired_contrasts(summary[summary.cohort.eq('constructive')],{'C2_minus_C0':{'C2_closure_only':1,'C0_canonical':-1},'C3_minus_C1':{'C3_full':1,'C1_contrast_only':-1}},metrics=('raw_ce','closure_task_norm','error_task_norm'))
    cv=pred.groupby(['cohort','seed','model']).squared_error.mean().unstack();cvrows=[]
    for (cohort,seed),r in cv.iterrows():cvrows.append({'sequence_id':'original','cohort':cohort,'seed':seed,'probe':'all_registered_raw_cues','contrast':'dynamics_minus_static','metric':'held_out_MSE','effect':r.dynamics_augmented-r['static']})
    effects=pd.concat([changes,pd.DataFrame(cvrows)],ignore_index=True);boot=bootstrap_effects(effects)
    for name,f in [('e1_transition_dynamics',transitions),('e1_seed_summary',summary),('e1_static_convexity_audit',static),('e1_single_patch_replay',patched),('e1_grouped_cv_predictions',pred),('e1_seed_effects',effects),('e1_effects_bootstrap',boot)]:save(out,name,f)
    # Preserve genuine pre-update acquisition data, without fitting any RT readout.
    for label,path in [('canonical','analysis/phase1i_context_composition/phase1i_true_acquisition.csv'),('constructive','analysis/phase1l/phase1l_training_sequence_metrics.csv')]:
        f=pd.read_csv(root/path);cols=[c for c in f if c in ['condition','family','seed','training_sequence_index','training_context','context','raw_sequence_loss_pre_update','raw_sequence_ce_pre_update','raw_sequence_loss','parameter_hash','parameter_hash_pre','parameters_changed']]
        save(out,'e1_'+label+'_genuine_acquisition',f[cols])
    figures.e1_figures(out,summary,patched,changes);reporting.e1_report(out,summary,static,pred,boot)
    return {'implementation_status':'validated','scientific_status':'completed_mixed','evidence_status':'completed','integrity_passed':True,'optimizer_updates':0,'n_checkpoint_models':len(config['checkpoints']),'n_transition_rows':len(transitions),'max_exact_identity_error':float(transitions.identity_error.max())}


def run_e2(root,out,config):
    prerequisite=verify_stage(out,'e1',config)
    if not prerequisite.get('integrity_passed'):raise ValueError('E2 blocked by E1 integrity')
    registry=json.loads((root/config['cue_grid']).read_text());parts=[]
    with evaluation_firewall():
        for i,r in enumerate(config['checkpoints'],1):
            model=load_model(root,r);before=model.digest()
            for c in registry['cues']:
                f=attach(decompose(model,c['vector']),r,c['name'],c['vector']);_integrity(f)
                f['cue_kind']=c['kind'];f['cue_sha256']=c['cue_sha256']
                f['cue_alpha']=c['cue_alpha'];f['cue_beta']=c['cue_beta']
                drive=model.drive(c['vector'])
                for j,value in enumerate(drive):f[f'context_drive_{j}']=float(value)
                parts.append(f)
            if model.digest()!=before:raise AssertionError('Frozen in-memory parameters mutated')
            if i%10==0:print(f'E2 grid checkpoints {i}/{len(config["checkpoints"])}',flush=True)
    transition=pd.concat(parts,ignore_index=True);summary=seed_summary(transition)
    kinds={x['name']:x['kind'] for x in registry['cues']};summary['cue_kind']=summary.probe.map(kinds)
    controls=grid_controls(summary,registry);invariants=projection_invariants(transition,registry)
    if not invariants.passed.all():raise AssertionError('C1/C3 same-alpha projection invariant failed')
    for name,f in [('e2_grid_transition',transition),('e2_grid_seed',summary),('e2_matched_geometry_controls',controls),('e2_projection_invariants',invariants),('e2_controls_bootstrap',bootstrap_effects(controls))]:save(out,name,f)
    figures.e2_figures(out,summary);reporting.e2_report(out,controls,invariants)
    return {'implementation_status':'validated','scientific_status':'completed_mixed','evidence_status':'completed','integrity_passed':True,'optimizer_updates':0,'n_grid_points_per_model':35,'n_anchors_per_model':8,'n_rays_per_model':6,'n_transition_rows':len(transition)}


def _save_checkpoint(out,prefix,sequence_id,seed,condition,settings,model,states,config):
    path=out/'checkpoints'/f'{prefix}_{sequence_id}_{condition}_seed_{seed}.pt'
    if path.exists():raise FileExistsError(path)
    history={k:torch.stack([s[k] for s in states]) for k in states[0]}
    payload={'model_state_dict':model.state_dict(),'parameter_history':history,'initial_state_sha256':object_hash({k:sha256_array(s) for k,s in states[0].items()}),'sequence_id':sequence_id,'seed':seed,'condition':condition,'settings':settings,'program_config_sha256':config['config_sha256'],'optimizer_updates':170,'training_contexts':['A','B'],'closure_cue_derived_from_A_B':[.5,.5,0.],'third_channel_zero':True}
    torch.save(payload,path)
    return {'sequence_id':sequence_id,'seed':int(seed),'condition':condition,'path':str(path.relative_to(out)),'checkpoint_sha256':sha256(path),'initial_state_sha256':payload['initial_state_sha256'],'optimizer_updates':170}


def sha256_array(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _prerequisites(out,config):
    for stage in ['e1','e2']:
        m=verify_stage(out,stage,config)
        if not m.get('integrity_passed'):raise ValueError('Training blocked by integrity gate: '+stage)


def run_pilot(root,out,config):
    _prerequisites(out,config)
    from context_sequence.training.factorial import candidate_settings,candidate_id,train_run,familiar_metrics,pilot_choice
    fc=read_hashed(root/config['factorial_config']);cache={};attempts=[];logs=[];checkpoint_rows=[]
    for setting in candidate_settings():
        cid=candidate_id(setting)
        for seed,condition in itertools.product(range(1000,1005),['F00','F01','F10','F11']):
            # Weak arms do not depend on hinge margin; closure-off arms do not depend on lambda_close.
            key=(seed,condition,setting['lambda_sep'],setting['lambda_close'] if condition[2]=='1' else 0.,setting['margin'] if condition[1]=='1' else 0.)
            cacheid=object_hash(list(key))[:20]
            if key not in cache:
                try:
                    model,log,states=train_run(seed,condition,setting)
                    metric=familiar_metrics(model,margin=setting['margin']);metric['improved']=metric['familiar_mean_ce']<log.initial_familiar_mean_ce.iloc[0]
                    metric['initial_familiar_mean_ce']=log.initial_familiar_mean_ce.iloc[0];metric['error']=''
                    log['pilot_cache_id']=cacheid;logs.append(log)
                    rec=_save_checkpoint(out,'e3_pilot',cacheid,seed,condition,setting,model,states,config);checkpoint_rows.append(rec)
                    # Reused weak-arm margin attainment is recomputed from its retained A/B trajectory if needed.
                    ha=model.trajectory((0,1,2,0,3,1,2,3),[1,0,0])[1].detach();hb=model.trajectory((0,1,2,0,3,1,2,3),[0,1,0])[1].detach()
                    cache[key]=(metric,torch.linalg.vector_norm(ha-hb,dim=-1).numpy())
                except FloatingPointError as e:
                    metric={k:np.nan for k in ['A_ce','B_ce','familiar_mean_ce','hidden_separation','task_separation','closure_fixed_H','weak_gradient_norm','strong_gradient_norm','closure_gradient_norm','initial_familiar_mean_ce']}
                    metric.update(improved=False,accuracy=0.,nonconstant_prediction=False,error=str(e));cache[key]=(metric,np.zeros(7))
                if len(cache)%10==0:print(f'E3 familiar-only distinct objectives {len(cache)}/400',flush=True)
            metric,distances=cache[key]
            attempts.append({'candidate_id':cid,'condition':condition,'seed':seed,**setting,**metric,'margin_attainment':float(np.mean(distances>=setting['margin'])),'pilot_cache_id':cacheid,'protected_probes_evaluated':False})
    attempts=pd.DataFrame(attempts);chosen,candidates=pilot_choice(attempts,fc['pilot_rules'])
    save(out,'e3_pilot_ab_only',attempts);save(out,'e3_pilot_candidate_audit',candidates);save(out,'e3_pilot_training',pd.concat(logs,ignore_index=True));save(out,'e3_pilot_checkpoint_registry',pd.DataFrame(checkpoint_rows))
    selected={'program_config_sha256':config['config_sha256'],'factorial_config_sha256':fc['config_sha256'],'pilot_seeds':list(range(1000,1005)),'confirmatory_seeds':list(range(20)),'settings':chosen,'selection_uses_familiar_only':True,'protected_probe_evaluations':0,'pilot_table_sha256':sha256(out/'tables/e3_pilot_ab_only.csv'),'pilot_candidate_audit_sha256':sha256(out/'tables/e3_pilot_candidate_audit.csv'),'selection_rule':fc['pilot_rules'],'scientific_status':'insufficient_familiar_support' if chosen is None else 'not_run',**environment()}
    selected['selection_sha256']=object_hash(selected);dump_exclusive(out/'manifests/e3_selected_config_manifest.json',selected)
    figures.pilot_figure(out,candidates)
    status={'implementation_status':'validated','scientific_status':'insufficient_familiar_support' if chosen is None else 'not_run','evidence_status':'insufficient_familiar_support' if chosen is None else 'familiar_configuration_selected','integrity_passed':True,'protected_probe_evaluations':0,'candidate_settings':48,'attempted_candidate_arm_seed_rows':len(attempts),'distinct_trained_objectives':len(cache),'selected_settings':chosen}
    reporting.write(out,'E3_PILOT_RESULTS.md','# E3 familiar-only pilot\n\n'+reporting.table(candidates)+'\n\nAll five reserved seeds and all four arms were assessed for every candidate. Algebraically identical weak/margin and closure-off/lambda-close runs were cached; no protected behavior entered the pilot.\n\nSelected settings: '+json.dumps(chosen))
    return status


def _selected(out,config):
    verify_stage(out,'e3_pilot',config)
    p=out/'manifests/e3_selected_config_manifest.json';d=json.loads(p.read_text())
    if object_hash({k:v for k,v in d.items() if k!='selection_sha256'})!=d['selection_sha256'] or d['program_config_sha256']!=config['config_sha256']:raise ValueError('Frozen familiar-only selection manifest changed')
    return d


def _factorial_evaluate(root,out,config,prefix,sequences,settings):
    from context_sequence.training.factorial import train_run,familiar_metrics,initialize,numpy_model
    records=[];logs=[];familiar=[]
    for seq in sequences:
        sid=seq['sequence_id'];tokens=tuple(seq['tokens'])
        for seed,condition in itertools.product(range(20),['F00','F01','F10','F11']):
            model,log,states=train_run(seed,condition,settings,tokens)
            metric=familiar_metrics(model,tokens,settings['margin']);metric.update(sequence_id=sid,seed=seed,condition=condition,initial_familiar_mean_ce=float(log.initial_familiar_mean_ce.iloc[0]),stable=True)
            familiar.append(metric);log['sequence_id']=sid;logs.append(log)
            record=_save_checkpoint(out,prefix,sid,seed,condition,settings,model,states,config);record['tokens']=list(tokens);records.append(record)
            if len(records)%10==0:print(f'{prefix.upper()} frozen new training runs {len(records)}/{len(sequences)*80}',flush=True)
    familiar=pd.DataFrame(familiar);feasible=familiar_feasibility(familiar,settings)
    save(out,prefix+'_training',pd.concat(logs,ignore_index=True));save(out,prefix+'_familiar_seed',familiar);save(out,prefix+'_familiar_feasibility',feasible);save(out,prefix+'_checkpoint_registry',pd.DataFrame(records))
    gate={'familiar_seed_sha256':sha256(out/'tables'/f'{prefix}_familiar_seed.csv'),'feasibility_sha256':sha256(out/'tables'/f'{prefix}_familiar_feasibility.csv'),'checkpoint_hashes':{r['path']:r['checkpoint_sha256'] for r in records},'before_protected_probe_evaluation':True,'no_seed_exclusions':True}
    dump_exclusive(out/'audit'/f'{prefix}_familiar_gate_frozen.json',gate)
    # All requested runs and the familiar gate are frozen before loading/evaluating protected cues.
    cues=cues_from_registry(read_registry(root/config['novelty_registry'],config['novelty_registry_sha256']));parts=[]
    with evaluation_firewall():
        for r in records:
            path=out/r['path']
            if sha256(path)!=r['checkpoint_sha256']:raise ValueError('New checkpoint hash changed before evaluation: '+str(path))
            state=torch.load(path,map_location='cpu',weights_only=False);model=initialize(r['seed']);model.load_state_dict(state['model_state_dict']);npmodel=numpy_model(model);before=npmodel.digest()
            for probe,cue in cues.items():
                f=decompose(npmodel,cue,tuple(r['tokens']));_integrity(f)
                for k,v in {'sequence_id':r['sequence_id'],'cohort':'factorial','condition':r['condition'],'seed':r['seed'],'probe':probe,'checkpoint_sha256':r['checkpoint_sha256'],**cue_geometry(cue)}.items():f[k]=v
                parts.append(f)
            if before!=npmodel.digest() or sha256(path)!=r['checkpoint_sha256']:raise AssertionError('New frozen checkpoint mutated')
    transition=pd.concat(parts,ignore_index=True);summary=seed_summary(transition)
    effects=paired_contrasts(summary,metrics=('raw_ce','hidden_separation','task_separation','closure_task_norm','error_task_norm','task_residual'));boot=bootstrap_effects(effects)
    save(out,prefix+'_probe_transition',transition);save(out,prefix+'_seed_summary',summary);save(out,prefix+'_seed_contrasts',effects);save(out,prefix+'_interaction_bootstrap',boot)
    return familiar,feasible,summary,boot,records


def run_e3(root,out,config):
    _prerequisites(out,config);selection=_selected(out,config)
    if selection['settings'] is None:
        status={'implementation_status':'validated','scientific_status':'insufficient_familiar_support','evidence_status':'insufficient_familiar_support','integrity_passed':True,'optimizer_updates':0,'confirmatory_runs':0,'protected_probe_evaluations':0}
        reporting.factorial_report(out,'e3',status);return status,[]
    registry=json.loads((root/config['sequence_registry']).read_text());settings=selection['settings']
    familiar,feasible,summary,boot,runs=_factorial_evaluate(root,out,config,'e3',registry['sequences'][:1],settings)
    closure_ok=all(familiar[familiar.condition.eq(f'F{i}1')].closure_fixed_H.mean() < familiar[familiar.condition.eq(f'F{i}0')].closure_fixed_H.mean() for i in [0,1])
    engaged=bool(feasible.separation_target_engaged.all() and closure_ok)
    science='failed_target_engagement' if not engaged else ('competence_confound' if not feasible.competence_comparable.all() else 'completed_mixed')
    status={'implementation_status':'validated','scientific_status':science,'evidence_status':'completed','integrity_passed':True,'confirmatory_runs':80,'optimizer_updates':13600,'protected_probe_training_updates':0,'target_engagement_passed':engaged,'competence_comparable':bool(feasible.competence_comparable.all())}
    figures.factorial_figure(out,'e3',familiar,summary);reporting.factorial_report(out,'e3',status,familiar,feasible,boot)
    return status,runs


def run_e4(root,out,config):
    e3=verify_stage(out,'e3',config);selection=_selected(out,config);registry=json.loads((root/config['sequence_registry']).read_text());validation=validate_sequences(registry);save(out,'e4_sequence_validation',validation)
    if selection['settings'] is None or not e3.get('integrity_passed'):
        status={'implementation_status':'validated','scientific_status':'inconclusive','evidence_status':'blocked_by_e3','integrity_passed':True,'optimizer_updates':0,'n_sequences_run':0,'reason':'No feasible frozen E3 familiar-only configuration; no replacement model or sequence-dependent calibration permitted.'}
        for name,cols in [('e4_sequence_seed_outcomes',['sequence_id','condition','seed','probe','raw_ce']),('e4_sequence_effects',['sequence_id','seed','probe','contrast','metric','effect']),('e4_heterogeneity',['probe','contrast','metric','sequence_count','sign_consistency'])]:save(out,name,pd.DataFrame(columns=cols))
        figures.replication_figure(out,pd.DataFrame(),registry,blocked=True)
        reporting.write(out,'E4_RESULTS.md','# E4 — Fixed sequence replication\n\nDesign-only; evidence_status=blocked_by_e3. '+status['reason']+'\n\n'+reporting.table(validation[['sequence_id','tokens','valid','replaced_equivalent_candidate']])+'\n\nThe registered design retains 20 paired seeds for each sequence and all four factorial arms, with no per-sequence coefficient search. No replication model was trained; empty outcome tables contain no imputed measurements.\n')
        return status,[]
    fam,feas,summary,boot,runs=_factorial_evaluate(root,out,config,'e4',registry['sequences'][1:],selection['settings'])
    effects=pd.read_csv(out/'tables/e4_seed_contrasts.csv');save(out,'e4_sequence_seed_outcomes',summary);save(out,'e4_sequence_effects',effects)
    rows=[]
    for key,g in boot.groupby(['probe','contrast','metric']):
        rows.append({'probe':key[0],'contrast':key[1],'metric':key[2],'sequence_count':g.sequence_id.nunique(),'sequence_mean_effect':g['mean'].mean(),'between_sequence_sd':g['mean'].std(),'minimum_sequence_effect':g['mean'].min(),'maximum_sequence_effect':g['mean'].max(),'fraction_sequence_means_positive':float((g['mean']>0).mean()),'universal_population_inference':False})
    heterogeneity=pd.DataFrame(rows);save(out,'e4_heterogeneity',heterogeneity);figures.replication_figure(out,boot,registry)
    status={'implementation_status':'validated','scientific_status':'completed_mixed','evidence_status':'completed','integrity_passed':True,'optimizer_updates':40800,'n_sequences_run':3,'n_seeds_per_sequence':20,'n_training_runs':240,'protected_probe_training_updates':0}
    reporting.write(out,'E4_RESULTS.md','# E4 — Sequence replication\n\n'+reporting.table(fam.groupby(['sequence_id','condition'])[['A_ce','B_ce','hidden_separation','closure_fixed_H']].mean().reset_index())+'\n\n'+reporting.table(feas)+'\n\nPer-sequence CE interactions:\n\n'+reporting.table(boot[boot.contrast.eq('interaction')&boot.metric.eq('raw_ce')])+'\n\nEvery sequence uses the unchanged frozen E3 objective and all 20 paired seeds. Effects and intervals are estimated separately per fixed sequence. Heterogeneity summarizes sequence means descriptively; seed × sequence rows are not independent evidence about a universal sequence population. This is replication within one eight-unit tanh architecture.\n')
    return status,runs


def claim_map(out):
    rows=[('E1','Static midpoint compatibility','e1_static_convexity_audit.csv','E1_T01_T02','e1_accessibility'),('E1','Recurrent closure and exact nonlinear propagation','e1_transition_dynamics.csv','E1_T03_T06_T07','e1_accessibility'),('E1','Incremental held-out-seed prediction','e1_grouped_cv_predictions.csv','E1_T10','e1_accessibility'),('E2','Same-alpha information and norm controls','e2_matched_geometry_controls.csv','E2_T01_T08','e2_context_grid'),('E2','Contrast pathway structural identities','e2_projection_invariants.csv','E2_T05','e2_context_grid'),('E3','Familiar-only selection and failures','e3_pilot_candidate_audit.csv','E3_T01_T13','e3_pilot_feasibility'),('E3','Factorial effects subject to familiar feasibility','e3_seed_contrasts.csv','E3_T01_T13','e3_factorial'),('E4','Sequence-specific replication or explicit blocked design','e4_sequence_effects.csv','E4_T01_T08','e4_replication_grid')]
    return pd.DataFrame([{'stage':s,'claim':c,'source_table':'tables/'+t,'source_available':(out/'tables'/t).is_file(),
        'source_sha256':sha256(out/'tables'/t) if (out/'tables'/t).is_file() else '',
        'evidence_status':'available' if (out/'tables'/t).is_file() and len(pd.read_csv(out/'tables'/t,nrows=1)) else 'unestimated_or_blocked',
        'filter_and_estimator':'Fixed panel filters in analysis/figures.py; directions/positions within seed before cohort means; separate sequence effects',
        'test_group':test,'figure_prefix':'figures/'+fig} for s,c,t,test,fig in rows])


def run(config_path,stage,mode,output=None):
    check_mode(stage,mode);root,config=load_config(config_path);torch.set_num_threads(1)
    out=Path(output).resolve() if output else root/'results/dynamical_generalization'/config['freeze_id']
    for folder in ['manifests','tables','figures','audit','checkpoints']:(out/folder).mkdir(parents=True,exist_ok=True)
    sid='e3_pilot' if stage=='e3' and mode=='ab_pilot' else stage
    if (out/'manifests'/f'{sid}.started.json').exists():raise FileExistsError('Stage already started; immutable results require a fresh output root: '+sid)
    before=audit(root,config);before.to_csv(out/'audit'/f'{sid}_input_hashes_before.csv',index=False)
    dump_exclusive(out/'manifests'/f'{sid}.started.json',{'stage':stage,'mode':mode,'program_config_sha256':config['config_sha256'],**environment()})
    np.save(out/'audit'/f'{sid}_bootstrap_indices.npy',np.random.default_rng(20260925).integers(0,20,(10000,20)))
    if not before.passed.all():
        bad=before[~before.passed];status={'implementation_status':'blocked','scientific_status':'not_run','evidence_status':'blocked_missing_source' if bad.status.eq('blocked_missing_source').any() else 'failed_integrity','integrity_passed':False,'missing_or_changed':bad[['path','expected_sha256','observed_sha256']].to_dict('records')}
        reporting.write(out,sid.upper()+'_RESULTS.md','# Blocked input audit\n\n'+reporting.table(bad));finalize(out,sid,config,status);raise FileNotFoundError('blocked_missing_source/failed_integrity: '+','.join(bad.path))
    runs=[]
    if stage=='e1':status=run_e1(root,out,config)
    elif stage=='e2':status=run_e2(root,out,config)
    elif mode=='ab_pilot':status=run_pilot(root,out,config)
    elif stage=='e3':status,runs=run_e3(root,out,config)
    elif stage=='e4':status,runs=run_e4(root,out,config)
    else:return build_final(out,config)
    after=audit(root,config);after.to_csv(out/'audit'/f'{sid}_input_hashes_after.csv',index=False)
    if not after.passed.all() or not after.observed_sha256.equals(before.observed_sha256):raise AssertionError('Immutable historical source/checkpoint hashes changed')
    m=finalize(out,sid,config,status,runs=runs)
    scratch=root/'scratchpads/dynamical_generalization_program.md'
    with scratch.open('a') as f:f.write(f'\n### Run {sid}\n- Config {config["config_sha256"]}; seeds registered in stage manifest; checkpoint/sequence identities retained; status {status}; output `{out}`. Source and checkpoint hashes identical before/after.\n')
    if stage=='e4':build_final(out,config)
    print(json.dumps(status,sort_keys=True),flush=True)
    return out


def build_final(out,config):
    stages=[]
    for stage in ['e1','e2','e3','e4']:
        m=verify_stage(out,stage,config);stages.append({k:m[k] for k in ['stage','implementation_status','scientific_status','evidence_status']})
    save(out,'claim_source_map',claim_map(out));reporting.final_report(out,stages)
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    dump_exclusive(out/'manifest.json',{'freeze_id':config['freeze_id'],'program_config_sha256':config['config_sha256'],'source_files':config['source_files'],'stages':stages,'outputs':files,**environment()})
    return out
