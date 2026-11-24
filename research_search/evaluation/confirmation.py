"""Once-only family registration. No confirmation outcomes enter agent contexts."""
from research_search.common import encoded,fingerprint,PACKAGE
import sqlite3

def commit(store,family,code_hash,plan):
    if not plan.get('frozen') or not plan.get('development_qualified') or not plan.get('ablation_supported'):raise ValueError('Freeze code, replicated development and matched ablation first')
    with store.transaction():
        store.check()
        if store.db.execute('SELECT 1 FROM confirmations WHERE family=?',(family,)).fetchone():raise ValueError('Confirmation family already committed; never resample or retry scientific failure')
        path=(PACKAGE/'database/global_confirmation.sqlite') if store.directory.resolve().is_relative_to(PACKAGE/'database') else store.directory/'test_global_confirmation.sqlite'
        g=sqlite3.connect(path,isolation_level=None);g.execute('PRAGMA synchronous=FULL');g.execute('CREATE TABLE IF NOT EXISTS families(identity TEXT PRIMARY KEY,code TEXT,plan TEXT,campaign TEXT)')
        definition=store.get('definition') or {};key=fingerprint(dict(code_hash=code_hash,config_hash=plan.get('config_hash',family),bank=definition.get('confirmation_hash','fixture-only')))
        try:
            g.execute('BEGIN IMMEDIATE');g.execute('INSERT INTO families VALUES(?,?,?,?)',(key,code_hash,encoded(plan).decode(),str(store.directory)));g.execute('COMMIT')
        except sqlite3.IntegrityError as ex:g.execute('ROLLBACK');raise ValueError('Frozen configuration already committed globally; an alias/new campaign cannot resample confirmation') from ex
        finally:g.close()
        store.db.execute("INSERT INTO confirmations VALUES(?,?,?,'committed')",(family,code_hash,encoded(plan).decode()));store.set('confirmation_started',True);store.event('confirmation_committed',dict(family=family,code_hash=code_hash,plan_hash=fingerprint(plan)))

def agent_view(results):
    return [r for r in results if r.get('phase')=='development']
