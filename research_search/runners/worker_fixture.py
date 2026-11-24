"""Real isolated queue/CLI/evaluator integration with accepted arrays only."""
import copy,shutil
from research_search.common import *
from research_search.orchestration.store import Store
from research_search.orchestration.queue import enqueue
from research_search.orchestration.worker import once
from .candidate import DEFAULT

def run():
    receipt=PACKAGE/'audit/worker_readonly_fixture_v2.json'
    if receipt.exists():return read(receipt)
    p=PACKAGE/'database/worker_readonly_fixture_v2';limits=dict(tokens=1000,cpu_seconds=60,optimizer_steps=0,wall_seconds=120,proposals=1,active_branches=1);s=Store(p,limits);write(p/'budgets.json',limits,True)
    if s.db.execute('SELECT 1 FROM jobs').fetchone():raise RuntimeError('Existing worker fixture requires adjudication; no silent rerun')
    source=PACKAGE/'database/native_prior_fixture_v2/target/predictions.npz';target=p/'accepted_predictions.npz';shutil.copyfile(source,target);c=copy.deepcopy(DEFAULT);c['prior']['episodes']=0
    enqueue(s,'cached-only','non-scientific-worker',dict(kind='cached_fixture',config=c,seed=170004,identity=fingerprint(c),prediction_file=str(target),prediction_sha=sha(target)))
    r=once(s);result=dict(passed=bool(r.get('passed') and s.totals()['optimizer_steps']==0 and r['position_rows']==175),actual_optimizer_steps=0,model_forwards=0,position_rows=r.get('position_rows'),isolated_worker=True,forwards_backward_Adam_disabled=True,accepted_prediction_sha=sha(source),worker_receipt=r,cpu_seconds=s.totals()['cpu_seconds'])
    write(receipt,result,True);s.control('stopped');s.close();return result
