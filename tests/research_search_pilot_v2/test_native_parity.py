import ast,copy,inspect,json
import pandas as pd
from research_search.common import read,write
from research_search.orchestration.store import Store
from experiments.research_search_pilot_v2.protocol import BASELINE,LIMITS,SEEDS
from experiments.research_search_pilot_v2.controller import assess_rows,ensure_jobs


def rows(effect):
    out=[]
    for i,seed in enumerate(SEEDS):
        for task in range(4):
            r=dict(seed=seed,task_id=f't{task}',D_A=(-.2 if i else .1)+effect,D_B=.2+effect,A_to_B=.3+effect,B_to_A=-.1+effect,sustained_cost=.01+effect,C_ce=2+i+effect,Mixed_ce=.6+i+effect,NoContext_ce=1+i+effect,C_accuracy=3/7,Mixed_accuracy=5/7,NoContext_accuracy=4/7,familiar=True,development_feasible=False)
            for cue in ['C','NoContext','Mixed-average']:
                for metric in ['ce','accuracy']:r['static_view_'+cue+'_'+metric]=1.0 if metric=='ce' else .5
            out.append(r)
    return out


def test_aggregation_matches_native_assessment(tmp_path,monkeypatch):
    from research_search.orchestration import coordinator as n
    a,b=rows(.05),rows(0);c=copy.deepcopy(BASELINE);c['target']['readout_initial_multiplier']=.25
    p=dict(id='candidate',config=c,matched_control=dict(config=BASELINE),ablation_primary=dict(metric='C_ce',direction='min'))
    write(tmp_path/'development.json',dict(tasks=[dict(task_id=f't{x}') for x in range(4)]))
    class S:
        directory=tmp_path
        def totals(self):return dict(optimizer_steps=5280)
    monkeypatch.setattr(n,'ensure_jobs',lambda store,name,*args,**kw:([name]*8,[]))
    monkeypatch.setattr(n,'collect',lambda store,ids:a if ids[0]=='candidate' else b)
    monkeypatch.setattr(n,'manifest_snapshot',lambda *args:None)
    native=n.assessment(S(),p,SEEDS);pilot=assess_rows(p,a,b,5280)
    for field in ['summary','paired_seed_intervals','cohort','development_feasible','ablation_supported','classification','n_network_seeds','tasks']:
        assert pilot[field]==native[field],field
    assert any(r['D_A']<0 for r in pilot['individual_results'])


def test_worker_orchestration_diff_is_only_entrypoint():
    from research_search.orchestration.worker import once as native
    from experiments.research_search_pilot_v2.worker import once as pilot
    a,b=ast.parse(inspect.getsource(native)),ast.parse(inspect.getsource(pilot))
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='experiments.research_search_pilot_v2.worker_cli':node.value='research_search.runners.evaluate'
            return node
    assert ast.dump(a,include_attributes=False)==ast.dump(Normalize().visit(b),include_attributes=False)


def test_zero_prior_grid_reserves_only_5280_target_updates(tmp_path):
    s=Store(tmp_path,LIMITS)
    try:
        write(tmp_path/'definition.json',dict(fixture=True))
        write(tmp_path/'development.json',dict(tasks=[dict(task_id=f't{i}',path=[0,1,2,0,3,1,2,3],context_assignment=['A','B']) for i in range(4)]))
        c=copy.deepcopy(BASELINE);c['target']['readout_initial_multiplier']=.25
        assert ensure_jobs(s,'candidate',c)==[]
        for row in s.db.execute('SELECT id FROM jobs').fetchall():
            p=tmp_path/'jobs'/row['id']/'pretrained.pt';p.parent.mkdir(parents=True);p.write_bytes(b'fixture-only-not-a-model')
            s.db.execute("UPDATE jobs SET status='completed' WHERE id=?",(row['id'],))
        assert len(ensure_jobs(s,'candidate',c))==8
        assert len(ensure_jobs(s,'control',BASELINE))==8
        payloads=[json.loads(r[0]) for r in s.db.execute('SELECT payload FROM jobs')]
        assert sum(p['kind']=='target' for p in payloads)==16
        assert all(p['kind']!='competence' and (p['kind']!='prior' or p['tasks']==[]) for p in payloads)
        assert s.totals()['reserved_optimizer_steps']==5280 and s.totals()['optimizer_steps']==0
    finally:s.close()
