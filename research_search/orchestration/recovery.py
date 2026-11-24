"""Recovery never assumes that an interrupted update did not happen."""
from .budget import BudgetError
def inspect(store):
    return dict(jobs=[dict(r) for r in store.db.execute("SELECT * FROM jobs WHERE status IN ('running','uncertain','paused')")],steps=[dict(r) for r in store.db.execute("SELECT * FROM steps WHERE status!='completed'")],calls=[dict(r) for r in store.db.execute("SELECT id,activity,reserved,status FROM calls WHERE status!='reconciled'")],cpu=[dict(r) for r in store.db.execute("SELECT * FROM cpu WHERE status!='reconciled'")])

def resume_job(store,id,saved,checkpoint_hash):
    store.recover(id,saved)
    if saved:
        r=store.db.execute('SELECT state_hash FROM steps WHERE job=? AND k=?',(id,saved)).fetchone()
        if r[0]!=checkpoint_hash:raise BudgetError('Saved complete-state hash does not match last committed step')
    with store.transaction():
        row=store.db.execute('SELECT status FROM jobs WHERE id=?',(id,)).fetchone()
        if row is None or row[0] not in ['paused','running']:raise BudgetError('Failed/uncertain jobs require explicit adjudication, not automatic rerun')
        store.db.execute("UPDATE jobs SET status='queued',pid=NULL WHERE id=?",(id,));store.event('verified_resume',dict(id=id,saved=saved))
