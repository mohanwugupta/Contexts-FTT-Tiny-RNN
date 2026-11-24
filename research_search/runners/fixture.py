"""Bounded non-scientific end-to-end exercise, never an exploratory candidate."""
import copy,time
from pathlib import Path
import numpy as np,torch
from research_search.common import *
from research_search.orchestration.store import Store
from research_search.orchestration.queue import enqueue,claim,finish,worker_lock
from research_search.orchestration.recovery import resume_job
from research_search.orchestration.budget import BudgetError
from research_search.runners.candidate import DEFAULT,build_model,run_target_learning
from research_search.runners.train import NativeRuntime
from research_search.runners.benchmark import FIXTURE_PATH
from research_search.evaluation.scoring import observations
from experiments.joint_phenotype_hypothesis_screen.common import digest

def receipt():return read(PACKAGE/'audit/fixture.json')

def run():
    if (PACKAGE/'audit/fixture.json').exists():return receipt()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    directory=PACKAGE/'database/fixture';limits=dict(tokens=50,cpu_seconds=60,optimizer_steps=5,wall_seconds=300,proposals=2,active_branches=2);s=Store(directory,limits)
    if s.totals()['optimizer_steps']:raise RuntimeError('Fixture has previous operations; explicit audited recovery required')
    flags={};c=copy.deepcopy(DEFAULT);c['prior']['episodes']=0;identity=fingerprint(c)
    proposal=dict(id='synthetic',parent_candidate=None,mechanism=dict(fixture=True),config=c,hypothesis='Non-scientific infrastructure exercise')
    s.register(proposal,identity)
    try:s.register(dict(proposal,id='duplicate'),identity)
    except ValueError:flags['duplicate_rejected']=True
    s.reserve_call('synthetic-usage','fixture',20);s.reconcile_call('synthetic-usage',dict(input_tokens=8,output_tokens=2,cached_input_tokens=3))
    try:s.reserve_call('over-budget','fixture',100)
    except BudgetError:flags['token_limit_enforced']=True
    enqueue(s,'fixture-target','synthetic',dict(fixture=True));job=claim(s);model=build_model(c,170002);runtime=NativeRuntime(s,'fixture-target',directory/'run',c,170002,identity,fixture_steps=4);runtime.attach(model)
    original=s.complete_step
    def pause_after_two(job,k,h):
        original(job,k,h)
        if k==2:s.control('paused')
    s.complete_step=pause_after_two;start=time.monotonic();cpu=time.process_time();task=dict(task_id='non-scientific-fixture',path=FIXTURE_PATH,context_assignment=['A','B'])
    with worker_lock(PACKAGE/'database'):
        try:run_target_learning(model,c,task)
        except BudgetError:pass
    assert runtime.k==2;finish(s,'fixture-target','paused');saved=torch.load(directory/'run/progress.pt',weights_only=False);s.close();s=Store(directory,limits);s.control('ready');resume_job(s,'fixture-target',2,sha(directory/'run/progress.pt'));job=claim(s);flags['queue_restarted']=True
    runtime=NativeRuntime(s,'fixture-target',directory/'run',c,170002,identity,fixture_steps=4);m,p,o=runtime.restore(saved,True);flags['state_restored']=digest((m.state_dict(),p.state_dict(),o.state_dict()))==digest((saved['model'],saved['policy'],saved['optimizer']))
    with worker_lock(PACKAGE/'database'):run_target_learning(m,c,task)
    finish(s,'fixture-target','completed');arr=np.load(directory/'run/predictions.npz');f,positions=observations(arr['logits'],arr['updates'],FIXTURE_PATH,170002,'synthetic');f.to_csv(directory/'fixture_metrics.csv',index=False);positions.to_csv(directory/'fixture_positions.csv',index=False);flags['fixed_evaluator_used']=len(positions)==5*5*7
    enqueue(s,'simulated-failure','synthetic',dict(fixture=True));claim(s);s.charge_step('simulated-failure',1);finish(s,'simulated-failure','failed','Simulated crash after attempted operation; no actual optimizer call')
    flags['simulated_failure_retained']=s.db.execute("SELECT status FROM jobs WHERE id='simulated-failure'").fetchone()[0]=='failed'
    try:s.recover('simulated-failure',0)
    except BudgetError:flags['uncertain_step_not_replayed']=True
    try:s.charge_step('another',1)
    except BudgetError:flags['optimizer_limit_enforced']=True
    report=directory/'FINAL_RESEARCH_REPORT.md';report.write_text('# Non-scientific fixture report\n\n## Central question\nValidate infrastructure, not a cognitive hypothesis.\n\n## Experiment-search history\nOne synthetic candidate, intentional duplicate, restart and failed run.\n\n## Strongest models\nNone; fixtures are not research evidence.\n\n## Splitting and transfer\nUnassessed because only4 updates were authorized. Raw fixed-scorer predictions are retained.\n\n## Negative findings\nSimulated failure and uncertain operation retained.\n\n## Mechanistic ablations\nNot performed.\n\n## Reproducibility and costs\nFour native optimizer calls; one additional simulated uncertain reservation;10 synthetic usage tokens (not provider usage). Full states restored after update2.\n\n## Remaining uncertainty\nThis fixture does not validate a discovery or a live agent token cap.\n\n## Smallest confirmatory experiment\nNone authorized by a fixture.\n');flags['final_report_written']=report.exists();s.control('stopped')
    result=dict(passed=all(flags.values()),**flags,actual_optimizer_steps=4,attempted_step_reservations=5,simulated_uncertain_no_optimizer=1,synthetic_tokens=10,real_agent_calls=0,active_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,report=str(report.relative_to(ROOT)),no_scientific_campaign=True)
    write(PACKAGE/'audit/fixture.json',result,True);s.close();return result
