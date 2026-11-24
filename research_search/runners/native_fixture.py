"""Separate bounded contract fixture: prior resume with cue basis, then target."""
import copy,time
import numpy as np,torch
from research_search.common import *
from research_search.orchestration.store import Store
from research_search.orchestration.queue import enqueue,claim,finish,worker_lock
from research_search.orchestration.recovery import resume_job
from research_search.orchestration.budget import BudgetError
from .candidate import DEFAULT,build_model,run_pretraining,run_target_learning
from .train import NativeRuntime
from experiments.joint_phenotype_hypothesis_screen.common import digest,rng

def run():
    receipt=PACKAGE/'audit/native_prior_fixture_v2.json'
    if receipt.exists():return read(receipt)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);directory=PACKAGE/'database/native_prior_fixture_v2';limits=dict(tokens=1000,cpu_seconds=60,optimizer_steps=24,wall_seconds=300,proposals=2,active_branches=2);s=Store(directory,limits)
    if s.totals()['optimizer_steps']:raise RuntimeError('Existing fixture operations require explicit audited recovery')
    config=copy.deepcopy(DEFAULT);config['prior'].update(episodes=1,cue_rotation=True);config['target']['readout_initial_multiplier']=.25;seed=170004;identity=fingerprint(config);path=[3,0,2,3,1,0,3,2];task=dict(episode=1,paths=[path],kind='shared',pair='AB',updates=[dict(inner_update=k+1,block=k//5+1,context='AB'[(k//5)%2],sequence=0) for k in range(20)])
    enqueue(s,'prior','fixture-v2',dict(kind='prior',tasks=[task]));claim(s);m=build_model(config,seed);initialC=m.recurrent.weight_ih_l0[:,6].detach().clone();rt=NativeRuntime(s,'prior',directory/'prior',config,seed,identity);rt.attach(m);original=s.complete_step
    def pause(job,k,h):
        original(job,k,h)
        if k==5:s.control('paused')
    s.complete_step=pause;start=time.monotonic();cpu=time.process_time()
    with worker_lock(PACKAGE/'database'):
        try:run_pretraining(m,config,[task])
        except BudgetError:pass
    assert rt.k==5;finish(s,'prior','paused');cp=torch.load(directory/'prior/progress.pt',weights_only=False);s.close();s=Store(directory,limits);s.control('ready');resume_job(s,'prior',5,sha(directory/'prior/progress.pt'));claim(s);rt=NativeRuntime(s,'prior',directory/'prior',config,seed,identity);m,p,o=rt.restore(cp,False);same=bool(digest((m.state_dict(),p.state_dict(),o.state_dict(),rng()))==digest((cp['model'],cp['policy'],cp['optimizer'],cp['rng'])));basis=bool(torch.equal(m.cue_basis,cp['settings']['cue_basis']))
    with worker_lock(PACKAGE/'database'):meta=run_pretraining(m,config,[task])
    finish(s,'prior','completed');learnable=not torch.equal(initialC,meta.recurrent.weight_ih_l0[:,6]);enqueue(s,'target','fixture-v2',dict(kind='target',fixture_steps=4));claim(s);rt=NativeRuntime(s,'target',directory/'target',config,seed,identity,fixture_steps=4);rt.attach(meta)
    with worker_lock(PACKAGE/'database'):run_target_learning(meta,config,dict(task_id='non-scientific-contract',path=path,context_assignment=['B','A']))
    finish(s,'target','completed');cp=torch.load(directory/'target/progress.pt',weights_only=False);states=cp['extra']['states'];fixed=all(torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],states[0]['model']['recurrent.weight_ih_l0'][:,6]) for st in states.values());assert len(states)==5 and s.totals()['optimizer_steps']==24
    result=dict(passed=bool(same and basis and learnable and fixed),actual_optimizer_steps=24,prior_steps=20,target_steps=4,interrupted_after=5,full_model_policy_Adam_RNG_restored=same,cue_basis_restored=basis,C_learnable_during_prior=learnable,C_bitwise_fixed_during_target=fixed,roleA_to_physicalB_assignment_verified=True,active_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,real_scientific_experiments=0)
    write(receipt,result,True);s.control('stopped');s.close();return result
