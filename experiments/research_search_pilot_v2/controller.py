"""Single-iteration pilot using the unchanged native scorer and worker numerics."""
import copy,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from research_search.common import ROOT,PACKAGE,read,write,sha,fingerprint,clean
from research_search.preflight import verify_artifacts
from research_search.orchestration.store import Store
from research_search.orchestration.queue import enqueue
from research_search.orchestration.budget import BudgetError
from research_search.orchestration.cpu import controller_budget
from research_search.orchestration.coordinator import admit,collect,manifest_snapshot,control_config,result_view
from research_search.evaluation.scoring import cohort
from experiments.meta_objective_balance.statistics import interval
from .protocol import BASELINE,LIMITS,SEEDS,CALL_RESERVATION,ALLOWED,validate_candidate,proposal_schema,critic_schema
from .events import assess
from .agent import invoke
from .worker import once

DIRECTORY=PACKAGE/'database/controlled_pilot_v2'
OUT=ROOT/'results/research_search_pilot/v2'


def code_files():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'experiments/research_backend_repair_v1/backend.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def initialize():
    verify_artifacts()
    connection=read(PACKAGE/'database/backend_validation_v4/receipt.json');events=[json.loads(x) for x in (Path(connection['raw_event_directory'])/'events.jsonl').read_text().splitlines()]
    accepted=assess(dict(returncode=0),events,50000)
    if not accepted['passed'] or not connection['budget_passed'] or not connection['original_auth_unchanged']:raise RuntimeError('Versioned backend replay validation failed')
    DIRECTORY.mkdir(parents=True,exist_ok=True);write(DIRECTORY/'budgets.json',LIMITS,True)
    bank=read(PACKAGE/'database/benchmark_banks/development.json');write(DIRECTORY/'development.json',bank,True)
    definition=dict(version='controlled-pilot-v2',code=code_files(),native_manifest_sha=sha(PACKAGE/'IMPLEMENTATION_MANIFEST.json'),native_runner=read(PACKAGE/'audit/evaluator_seal.json')['runner_hash'],development_sha=sha(DIRECTORY/'development.json'),limits=LIMITS,seeds=SEEDS,target_updates=330,prior_episodes=0,planned_target_histories=16,planned_optimizer_updates=5280,confirmation_authorized=False,tier_expansion=False,two_calls=['proposal','result_critic'],call_reservation=CALL_RESERVATION,manual_approval_each_call=True,hard_token_ceiling_guaranteed=False,old_connection_usage_kept_separate=True)
    write(DIRECTORY/'definition.json',definition,True)
    s=Store(DIRECTORY,LIMITS)
    if not s.get('phase'):
        s.set('phase','proposal_pending');s.set('manual_checkpoints',True);s.set('proposal_attempts',0)
        s.event('registered_pilot',definition)
    return s


def integrity(store):
    verify_artifacts();d=read(store.directory/'definition.json')
    if d['code']!=code_files():raise RuntimeError('Pilot wrapper code changed; versioned revalidation required')
    if sha(store.directory/'development.json')!=d['development_sha']:raise RuntimeError('Task bank changed')
    if store.limits!=LIMITS:raise RuntimeError('Limits changed')


def request(store,role,prompt,schema):
    integrity(store)
    if len(prompt)>16000:raise ValueError('Compact prompt length exceeded')
    ticket='pilot2-'+role;path=store.directory/'requests_agent'/f'{ticket}.json'
    value=dict(ticket=ticket,phase=store.get('phase'),role=role,prompt=prompt,schema=schema,reservation=CALL_RESERVATION,codex_sha256=sha(Path(shutil.which('codex')).resolve()),pilot_definition_sha=sha(store.directory/'definition.json'))
    write(path,value,True)
    return dict(ticket=ticket,approval_sha=sha(path),role=role,reservation=CALL_RESERVATION,phase=store.get('phase'),prompt_characters=len(prompt))


def prepare_proposal(store):
    if store.get('phase')!='proposal_pending':raise RuntimeError('Proposal phase is not pending')
    history=(PACKAGE/'history/KNOWN_FAILURES.md').read_text()[:2500]
    context=dict(baseline=BASELINE,allowed_one_leaf_changes=ALLOWED,planned_updates=5280,seeds=SEEDS,development_tasks=4,token_budget=50000,call_reservation=25000,scientific_limitations='Two-seed zero-prior technical pilot. No background competence claim, no canonical target/confirmation, no reward or objective changes. Historical mechanisms may be replicated across new tasks; this is not a claim of novel discovery.',historical_qualifications=history)
    prompt='You are the hypothesis specialist for one controlled ordinary-learning research pilot. Return only the supplied JSON schema. Use no tools/files/shell/apps/subagents. Propose one falsifiable mechanism and matched-control prediction using exactly one allowed leaf change from the complete baseline. Keep all other config fields unchanged. matched_control must be pilot-baseline-zero-prior and required_code_changes empty. Give at least two competing predictions, honest runtime/token estimates including about14k default input tokens, the full paired5280 target updates, and explicit stopping rules. No model selection or tuning on transfer probes, no reward for splitting, no new scale sweep. Background dose is zero in both arms; restrict any human/causal claims. Existing native code executes everything; do not author Python. Context:\n'+json.dumps(context,separators=(',',':'))
    return request(store,'proposal',prompt,proposal_schema())


def call(store,ticket,approval):
    integrity(store)
    role=read(store.directory/'requests_agent'/f'{ticket}.json')['role']
    expected='proposal_pending' if role=='proposal' else 'critic_pending'
    if store.get('phase')!=expected:raise RuntimeError('This role is not awaiting a call; no replay')
    if role=='proposal':
        n=store.get('proposal_attempts') or 0
        if n>=2:raise BudgetError('Two proposal attempts exhausted')
        # Count attempts immediately before the explicitly approved call.
        path=store.directory/'requests_agent'/f'{ticket}.json'
        if approval!=sha(path):raise PermissionError('Individual call approval missing')
        store.set('proposal_attempts',n+1)
    r=invoke(store,ticket,approval)
    if not r['passed']:
        store.set('phase','agent_failed_or_unknown');return r
    if role=='proposal':
        raw=r['reply'];write(store.directory/'proposal_agent.json',raw,True)
        try:
            diff=validate_candidate(raw['config'])
            if raw['matched_control']!='pilot-baseline-zero-prior' or raw['required_code_changes'] or raw['mechanism']['field']!=diff[0]:raise ValueError('Control/code/mechanism does not match one-leaf proposal')
            if raw['estimated_training_steps']>6000 or len(raw['alternative_predictions'])<2:raise ValueError('Bad resource estimate/competing predictions')
            resolved=copy.deepcopy(raw);resolved['matched_control']=dict(config=copy.deepcopy(BASELINE))
            store.control('ready');store.set('pending',dict(call_id=ticket));admit(store,resolved);store.set('active',resolved);store.set('phase','compute_ready');store.control('manual_checkpoint')
            write(store.directory/'proposal_resolution.json',dict(passed=True,changed_fields=diff,matched_control_resolution='symbolic name to predeclared baseline only; original JSON retained'),True)
        except (ValueError,KeyError,TypeError) as ex:
            store.set('phase','proposal_invalid');write(store.directory/'proposal_resolution.json',dict(passed=False,error=str(ex),no_automatic_revision=True),True)
    elif role=='critic':
        write(store.directory/'critic_feedback.json',r['reply'],True);store.set('phase','complete');store.control('stopped')
    return r|dict(pilot_phase=store.get('phase'))


def _queue(store,id,candidate,payload):
    row=store.db.execute('SELECT status FROM jobs WHERE id=?',(id,)).fetchone()
    if row:
        if row['status'] in ['failed','uncertain']:raise RuntimeError('Failed/uncertain branch cannot be rerun')
        return row['status']
    enqueue(store,id,candidate,payload);return 'queued'


def ensure_jobs(store,candidate,config):
    identity=fingerprint(read(store.directory/'definition.json'));ids=[]
    for seed in SEEDS:
        key=fingerprint(dict(model=config['model'],prior=config['prior'],seed=seed,identity=identity))[:24];init='p2-init-'+key
        status=_queue(store,init,candidate,dict(kind='prior',config=config,seed=seed,identity=identity,tasks=[]))
        if status!='completed':continue
        prior=store.directory/'jobs'/init/'pretrained.pt'
        for task in read(store.directory/'development.json')['tasks']:
            id='p2-target-'+fingerprint(dict(config=config,seed=seed,task=task))[:24]
            ids.append(id);_queue(store,id,candidate,dict(kind='target',config=config,seed=seed,identity=identity,task=task,phase='development',prior_file=str(prior),prior_sha=sha(prior)))
    return ids


def assess_rows(p,a,b,steps):
    frames=pd.DataFrame(a);feasible=all(r['development_feasible'] for r in a)
    means={k:float(frames[k].mean()) for k in ['D_A','D_B','A_to_B','B_to_A','sustained_cost','C_ce','C_accuracy','Mixed_ce','Mixed_accuracy']};means['familiar_fraction']=float(frames.familiar.mean())
    paired=[]
    for metric in ['D_A','D_B','A_to_B','B_to_A','sustained_cost','C_ce','Mixed_ce','NoContext_ce']:
        aa=pd.DataFrame(a).groupby('seed')[metric].mean().reindex(SEEDS);bb=pd.DataFrame(b).groupby('seed')[metric].mean().reindex(SEEDS)
        paired.append(dict(metric=metric,**interval((aa-bb).to_numpy(float),SEEDS)))
    for cue in ['C','NoContext','Mixed-average']:
        for metric in ['ce','accuracy']:
            key='static_view_'+cue+'_'+metric;native=cue.replace('-average','')+'_'+metric
            aa=pd.DataFrame(a).groupby('seed')[native].mean().reindex(SEEDS);bb=pd.DataFrame(b).groupby('seed')[key].mean().reindex(SEEDS)
            paired.append(dict(metric=native,contrast='candidate-minus-control-checkpoint-static-quarter-view',**interval((aa-bb).to_numpy(float),SEEDS)))
    return clean(dict(id=p['id'],phase='development',seeds=SEEDS,tasks=len(a)//len(SEEDS),n_network_seeds=2,background_adequate=True,background_competence='not_applicable_zero_prior',development_feasible=bool(feasible),classification='joint_development_feasible' if feasible else 'partial_or_negative',ablation_supported=False,ablation_primary=p['ablation_primary'],summary=means,paired_seed_intervals=paired,cohort=cohort(a,SEEDS),individual_results=a,matched_control_results=b,optimizer_calls=steps,claim='technical two-seed development pilot only; late functional gate unassessed; no human mechanism inference'))


def assessment(store,p):
    ids=ensure_jobs(store,p['id'],p['config']);control=control_config(store,p)
    bid=ensure_jobs(store,'control-'+fingerprint(control)[:16],control)
    if len(ids)!=8 or len(bid)!=8:return None
    a,b=collect(store,ids),collect(store,bid)
    if a is None or b is None:return None
    result=assess_rows(p,a,b,store.totals()['optimizer_steps'])
    manifest_snapshot(store,p,ids,[],'pilot_development2')
    write(store.directory/'paired_jobs.json',dict(candidate=ids,control=bid),True)
    return result


def compute(store):
    integrity(store)
    if store.get('phase') not in ['compute_ready','computing']:raise RuntimeError('No valid candidate awaiting compute')
    store.control('running');store.set('phase','computing');p=store.get('active')
    try:
        while store.get('status')=='running':
            with controller_budget(store):
                integrity(store);result=assessment(store,p)
                if result is not None:
                    write(store.directory/'development_result.json',result,True);store.result(p['id']+'-pilot',result)
                    store.set('phase','critic_pending');store.control('manual_checkpoint');return result
                if not store.db.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone():raise RuntimeError('Incomplete grid without queued work')
                receipt=once(store)
                if not receipt.get('passed'):
                    store.event('pilot_worker_failure',receipt);store.set('phase','compute_incomplete');store.control('paused');return dict(passed=False,partial=True,receipt=receipt)
                store.event('pilot_job_complete',dict(job=receipt['job'],kind=receipt.get('kind'),steps=receipt.get('attempts')))
    except BaseException as ex:
        store.event('pilot_compute_stopped',dict(error=repr(ex)))
        store.set('phase','compute_incomplete')
        if store.get('status') not in ['stopped','budget_exhausted']:store.control('paused')
        raise


def prepare_critic(store):
    if store.get('phase')!='critic_pending':raise RuntimeError('Complete evaluation required before critic approval')
    result=read(store.directory/'development_result.json')
    context=dict(proposal=store.get('active'),result=result_view(result),instruction='Two seeds/four repeated tasks are not8 independent networks. Distinguish technical completion, exploratory usefulness, nulls/tradeoffs, floor limitations and unsupported causal/human claims. Static quarter-readout is expression control, not separate learning. Do not change scientific criteria, optimize a new state or launch follow-up.')
    # Full exact outcomes stay local; compact whole-seed summaries are supplied.
    context['result'].pop('diagnostic_example',None)
    prompt='You are the scientific result critic. Use no tools or external files. Return strict JSON. Identify exact failed/surprising prediction, at least two competing explanations, scientific and technical usefulness, limitations and the smallest supported follow-up. Recommend retain/revise/abandon/inconclusive. This pilot stops after your feedback; no new experiment is launched. Context:\n'+json.dumps(clean(context),separators=(',',':'))
    return request(store,'critic',prompt,critic_schema())
