"""Native ordinary learning with before-attempt charging and full-state commits."""
import copy,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.joint_phenotype_hypothesis_screen import adapter
from experiments.joint_phenotype_hypothesis_screen.banks import haar
from experiments.high_scale_meta_progression.protocol import epsilon
from experiments.joint_phenotype_hypothesis_screen.common import digest,rng,set_rng,atomic
from experiments.joint_phenotype_hypothesis_screen.common import cue
from research_search.common import write,read,sha,fingerprint,clean
from research_search.evaluation.validity import validate_exposure
from .candidate import build_model,variant,target_optimizer,build_training_schedule,export_predictions

PROBES=['A','B','C','Mixed-average','NoContext']

class NativeRuntime:
    def __init__(self,store,job,directory,config,seed,identity,deadline=None,fixture_steps=None):
        self.store=store;self.job=job;self.folder=Path(directory);self.folder.mkdir(parents=True,exist_ok=True);self.config=config;self.seed=seed;self.identity=identity;self.deadline=deadline;self.fixture_steps=fixture_steps;self.path=self.folder/'progress.pt';self.k=0;self.saved=None
        cap=float(os.environ.get('RESEARCH_CPU_SECONDS','inf'));self.cpu_deadline=time.process_time()+max(0,cap-5)
        if self.path.exists():
            cp=torch.load(self.path,weights_only=False)
            if cp['identity']!=identity or cp['config_hash']!=fingerprint(config):raise ValueError('Changed code/config cannot resume')
            store.recover(job,cp['attempts']);last=store.db.execute('SELECT state_hash FROM steps WHERE job=? AND k=?',(job,cp['attempts'])).fetchone()
            if last and last[0]!=sha(self.path):raise ValueError('Complete state changed since commit')
            self.k=cp['attempts'];self.saved=cp
    def check(self):
        self.store.check()
        if time.process_time()>=self.cpu_deadline:raise BudgetError('Clean CPU stop before next update; complete state retained')
        if self.deadline and time.monotonic()>=self.deadline:raise TimeoutError('Clean wall-time stop before next update')
    def attach(self,model):model._research_runtime=self;return model
    def state(self,m,p,o,extra):
        return dict(identity=self.identity,config_hash=fingerprint(self.config),attempts=self.k,model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()),rng=rng(),settings=dict(onset=m.onset,cue_basis=copy.deepcopy(m.cue_basis),mask_cue=m.mask_cue,modes={n:v.training for n,v in m.named_modules()},policy_training=p.training),extra=copy.deepcopy(extra))
    def restore(self,cp,target):
        m=build_model(self.config,self.seed);m.load_state_dict(cp['model']);m.onset=cp['settings']['onset'];m.cue_basis=cp['settings']['cue_basis'];m.mask_cue=cp['settings']['mask_cue']
        if m.cue_basis is not None:
            import types
            m.forward=types.MethodType(adapter.flexible_forward,m)
        for n,v in m.named_modules():v.training=cp['settings'].get('modes',{}).get(n,True)
        p=core.policy();p.load_state_dict(cp['policy']);p.training=cp['settings'].get('policy_training',True);o=target_optimizer(m,p,self.config) if target else core.optimizer(m,p);o.load_state_dict(cp['optimizer']);set_rng(cp['rng']);return self.attach(m),p,o
    def step(self,m,p,o,c,path,extra,after=None,target=False):
        if target:
            n=extra['next_update'];assignment=extra['task'].get('context_assignment',['A','B']);expected=assignment[0 if cue(n)=='A' else 1]
            if c not in ['A','B'] or c!=expected or list(path)!=extra['task']['path']:raise ValueError('Protected target, cue or indexing mismatch')
        self.check();self.k+=1;self.store.charge_step(self.job,self.k)
        old=m.recurrent.weight_ih_l0[:,6].detach().clone();vg=m.V[:,2].detach().clone() if hasattr(m,'V') else None
        try:
            if target:
                # Bypass only the core's one-canonical-path assertion; the
                # wrapper above owns the immutable bank path and A/B schedule.
                if variant(self.config)=='A05':adapter.reset_first_moments(m,o,n)
                r=adapter.update(m,p,o,c,variant(self.config),n,lambda calls:None,path=path,meta=True)
            else:r=core.step(m,p,o,c,path=path,meta=True)
            if target:
                assert old.contiguous().numpy().tobytes()==m.recurrent.weight_ih_l0[:,6].detach().contiguous().numpy().tobytes()
                if vg is not None:assert vg.contiguous().numpy().tobytes()==m.V[:,2].detach().contiguous().numpy().tobytes()
            if r['status']!='completed':raise RuntimeError('Native nonfinite/failed update; attempted operation remains uncertain')
        except BaseException as ex:
            failed=self.state(m,p,o,extra);failed['gradients']={n:None if v.grad is None else v.grad.detach().clone() for n,v in list(m.named_parameters())+[('policy.'+n,v) for n,v in p.named_parameters()]};atomic(self.folder/f'failed_state_{self.k}.pt',failed);write(self.folder/f'failed_state_{self.k}.json',dict(attempt=self.k,error=repr(ex),status='uncertain_or_nonfinite',native_record=locals().get('r'),no_automatic_replay=True),True);raise
        if after:after(r)
        cp=self.state(m,p,o,extra);atomic(self.path,cp);self.store.complete_step(self.job,self.k,sha(self.path));self.saved=cp;return r
    def pretrain(self,model,config,tasks):
        if self.saved:
            if self.saved['extra']['phase']!='prior':raise ValueError('Wrong run phase')
            m,p,o=self.restore(self.saved,False);extra=self.saved['extra'];meta=build_model(config,self.seed);meta.load_state_dict(extra['meta']);start=extra['episode'];inner=extra['inner']
        else:meta=model;m=p=o=None;start=0;inner=0;extra=dict(phase='prior',episode=0,inner=0,meta=copy.deepcopy(meta.state_dict()),tasks_hash=fingerprint(tasks),records=[])
        if extra['tasks_hash']!=fingerprint(tasks):raise ValueError('Background tasks changed')
        for e in range(start,len(tasks)):
            task=tasks[e]
            if e!=start or inner==0:
                m=build_model(config,self.seed);m.load_state_dict(meta.state_dict());p=core.policy();o=core.optimizer(m,p)
                if config['prior']['cue_rotation']:
                    q=haar(np.random.default_rng([202611230803,self.seed,e]));m.cue_basis=torch.tensor(q,dtype=torch.float32)
                    if config['model']['activation']!='linear':raise ValueError('Cue rotation requires registered linear adapter')
                    import types
                    m.forward=types.MethodType(adapter.flexible_forward,m)
            for j in range(inner if e==start else 0,20):
                row=task['updates'][j];path=task['paths'][row['sequence']]
                def after(r,e=e,j=j):
                    record=dict(episode=e+1,inner=j+1,context=row['context'],path=path,**r);extra['records']=(extra['records']+[record])[-20:];extra['record_count']=extra.get('record_count',0)+1;extra['inner']=j+1
                    import json
                    with (self.folder/'prior_practice.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                    if j==19:
                        core.reptile(meta,m,epsilon(e+1));extra.update(episode=e+1,inner=0,meta=copy.deepcopy(meta.state_dict()))
                self.step(m,p,o,row['context'],path,extra,after,target=False)
            inner=0
        result=self.folder/'pretrained.pt';atomic(result,dict(model=meta.state_dict(),rng=rng(),config_hash=fingerprint(config),tasks_hash=fingerprint(tasks),attempts=self.k))
        write(self.folder/'prior_complete.json',dict(passed=True,attempts=self.k,sha256=sha(result),tasks_hash=fingerprint(tasks)),True);return self.attach(meta)
    def target(self,model,config,task):
        path=list(task['path']);assignment=task.get('context_assignment',['A','B']);steps=self.fixture_steps or 330
        if self.saved:
            if self.saved['extra']['phase']!='target' or self.saved['extra']['task']!=task:raise ValueError('Task changed on resume')
            m,p,o=self.restore(self.saved,True);extra=self.saved['extra']
        else:
            m=model;m.cue_basis=None;p=core.policy();o=target_optimizer(m,p,config)
            with torch.no_grad():m.sequence_head.weight.mul_(config['target']['readout_initial_multiplier'])
            extra=dict(phase='target',task=task,next_update=1,observations={},hidden_observations={},practice=[],exposure=[],states={})
            obs=export_predictions(m,dict(path=path,context_assignment=assignment,policy=p,optimizer=o));extra['observations'][0]=np.stack([obs[c]['logits'] for c in PROBES]);extra['hidden_observations'][0]=np.stack([obs[c]['hidden'] for c in PROBES]);extra['states'][0]=adapter.snapshot(m,p,o,0,'registered-native')
        for n in range(extra['next_update'],steps+1):
            role=cue(n);physical=assignment[0 if role=='A' else 1]
            def after(r,n=n):
                extra['practice'].append(dict(update=n,training_cue=role,physical_context=physical,**r));extra['exposure'].append(dict(update=n,context=role,physical_context=physical,path=path));extra['next_update']=n+1;obs=export_predictions(m,dict(path=path,context_assignment=assignment,policy=p,optimizer=o));extra['observations'][n]=np.stack([obs[c]['logits'] for c in PROBES]);extra['hidden_observations'][n]=np.stack([obs[c]['hidden'] for c in PROBES]);extra['states'][n]=adapter.snapshot(m,p,o,n,'registered-native')
            self.step(m,p,o,physical,path,extra,after,target=True)
        validate_exposure(extra['exposure'],path,assignment,expected=steps)
        updates=sorted(extra['observations']);np.savez_compressed(self.folder/'predictions.npz',updates=np.array(updates),logits=np.stack([extra['observations'][n] for n in updates]),hidden=np.stack([extra['hidden_observations'][n] for n in updates]),path=np.array(path),probes=np.array(PROBES));pd.DataFrame(extra['practice']).to_csv(self.folder/'practice.csv',index=False);write(self.folder/'exposure.json',extra['exposure'],True)
        atomic(self.folder/'states.pt',dict(states=extra['states'],task=task,config_hash=fingerprint(config),identity=self.identity))
        write(self.folder/'complete.json',dict(passed=True,fixture=bool(self.fixture_steps),attempts=self.k,states=len(updates),prediction_sha=sha(self.folder/'predictions.npz'),checkpoint_sha=sha(self.path),C_target_column_fixed=True,identity=self.identity),True)
        return dict(model=m,predictions=self.folder/'predictions.npz',attempts=self.k)
