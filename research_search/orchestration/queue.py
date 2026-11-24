import contextlib,fcntl,json,os
from pathlib import Path
from research_search.common import identifier,encoded

@contextlib.contextmanager
def worker_lock(directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    with (p/'worker.lock').open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as ex:raise RuntimeError('A compute/agent worker already holds the global slot') from ex
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def enqueue(store,id,candidate,payload):
    with store.transaction():
        store.check()
        kind=payload.get('kind');steps=len(payload.get('tasks',[]))*20 if kind in ['prior','competence'] else payload.get('fixture_steps') or 330 if kind=='target' else 0
        if kind:
            t=store.totals()
            if t['optimizer_steps']+t['reserved_optimizer_steps']+steps>store.limits['optimizer_steps']:
                from .budget import BudgetError
                raise BudgetError('Queue refuses work beyond remaining optimizer reserves')
            store.db.execute('INSERT INTO update_reservations VALUES(?,?)',(id,steps))
        store.db.execute("INSERT INTO jobs VALUES(?,?,?,'queued',NULL,NULL)",(identifier(id),candidate,encoded(payload).decode()));store.event('queued',dict(id=id,reserved_updates=steps))

def claim(store):
    with store.transaction():
        store.check()
        if store.db.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():raise RuntimeError('Unrecovered running job; never silently rerun')
        row=store.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
        if not row:return None
        store.db.execute("UPDATE jobs SET status='running',pid=? WHERE id=?",(os.getpid(),row['id']));return dict(row)

def finish(store,id,status,error=None):
    if status not in ['completed','failed','paused','uncertain']:raise ValueError(status)
    with store.transaction():store.db.execute('UPDATE jobs SET status=?,error=?,pid=NULL WHERE id=?',(status,error,id));store.event('job_'+status,id)
