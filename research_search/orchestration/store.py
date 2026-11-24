"""Single durable authority for candidates, reservations, queue and attempted work."""
import sqlite3,time,json,contextlib
from pathlib import Path
from research_search.common import encoded,fingerprint,identifier
from .budget import BudgetError,usage,ALLOCATIONS

class Store:
    def __init__(self,directory,limits):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.directory/'campaign.sqlite',timeout=30,isolation_level=None)
        self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,activity TEXT,reserved INTEGER,status TEXT,total INTEGER,cached INTEGER,raw TEXT);
        CREATE TABLE IF NOT EXISTS call_links(call TEXT PRIMARY KEY,candidate TEXT);
        CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY,parent TEXT,signature TEXT UNIQUE,proposal TEXT,code_hash TEXT,status TEXT,repairs INTEGER DEFAULT 0,revisions INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS events(n INTEGER PRIMARY KEY AUTOINCREMENT,at REAL,kind TEXT,data TEXT);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,candidate TEXT,payload TEXT,status TEXT,pid INTEGER,error TEXT);
        CREATE TABLE IF NOT EXISTS steps(job TEXT,k INTEGER,status TEXT,state_hash TEXT,PRIMARY KEY(job,k));
        CREATE TABLE IF NOT EXISTS update_reservations(job TEXT PRIMARY KEY,ceiling INTEGER);
        CREATE TABLE IF NOT EXISTS cpu(id TEXT PRIMARY KEY,reserved REAL,actual REAL,status TEXT);
        CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS confirmations(family TEXT PRIMARY KEY,code_hash TEXT,plan TEXT,status TEXT);''')
        with self.transaction():
            if not self.get('limits'):
                self.set('limits',limits);self.set('status','ready');self.set('started',None)
            elif self.get('limits')!=limits:raise ValueError('Campaign ceilings are immutable')
        self.limits=limits
    def close(self):self.db.close()
    @contextlib.contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield;self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
    def get(self,key):
        r=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(r[0]) if r else None
    def set(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,encoded(value).decode()))
    def event(self,kind,data):self.db.execute('INSERT INTO events(at,kind,data) VALUES(?,?,?)',(time.time(),kind,encoded(data).decode()))
    def state(self):return dict(status=self.get('status'),started=self.get('started'),totals=self.totals())
    def totals(self):
        q=self.db.execute("SELECT coalesce(sum(total),0),coalesce(sum(cached),0),coalesce(sum(CASE WHEN status!='reconciled' THEN reserved ELSE 0 END),0) FROM calls").fetchone()
        c=self.db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN status!='reconciled' THEN reserved ELSE 0 END),0) FROM cpu").fetchone()
        setup=self.get('unmetered_setup_reservation') or 0
        reserved=self.db.execute('SELECT coalesce(sum(max(0,ceiling-(SELECT count(*) FROM steps s WHERE s.job=r.job))),0) FROM update_reservations r').fetchone()[0]
        return dict(tokens=q[0],cached_input_tokens=q[1],reserved_tokens=q[2]+setup,unmetered_setup_reserved=setup,setup_actual_tokens=self.get('setup_actual_tokens'),optimizer_steps=self.db.execute('SELECT count(*) FROM steps').fetchone()[0],reserved_optimizer_steps=reserved,cpu_seconds=c[0],reserved_cpu_seconds=c[1],proposals=self.db.execute('SELECT count(*) FROM nodes').fetchone()[0])
    def check(self):
        status=self.get('status')
        if status in ['stopped','paused','manual_checkpoint','budget_exhausted']:raise BudgetError('Campaign '+status)
        start=self.get('started')
        if start is not None and time.time()-start>=self.limits['wall_seconds']:raise BudgetError('Wall-clock ceiling reached')
        t=self.totals()
        if t['cpu_seconds']>=self.limits['cpu_seconds']:raise BudgetError('CPU ceiling reached')
    def control(self,status):
        if status not in ['ready','running','paused','stopped','manual_checkpoint','budget_exhausted']:raise ValueError(status)
        with self.transaction():
            if self.get('status') in ['stopped','budget_exhausted'] and status not in [self.get('status'),'stopped']:raise BudgetError('Stop or exhausted budget is terminal')
            if status in ['ready','running'] and (self.db.execute("SELECT 1 FROM calls WHERE status!='reconciled'").fetchone() or self.db.execute("SELECT 1 FROM steps WHERE status!='completed'").fetchone()):raise BudgetError('Unreconciled usage or uncertain optimizer work requires explicit recovery')
            if status=='running' and self.get('started') is None:self.set('started',time.time())
            self.set('status',status);self.event('control',status)
    def reserve_call(self,id,activity,maximum,speculative=True):
        if type(maximum) is not int or maximum<=0:raise ValueError('Positive reservation required')
        with self.transaction():
            self.check();t=self.totals()
            if t['tokens']+t['reserved_tokens']+maximum>self.limits['tokens']:raise BudgetError('Global token reserve exceeded')
            if speculative and t['tokens']+t['reserved_tokens']+maximum>.8*self.limits['tokens']:raise BudgetError('80% speculative stop')
            if activity in ALLOCATIONS:
                used=self.db.execute("SELECT coalesce(sum(CASE WHEN status='reconciled' THEN total ELSE reserved END),0) FROM calls WHERE activity=?",(activity,)).fetchone()[0]
                cap=ALLOCATIONS[activity]*self.limits['tokens']/1000000
                if used+maximum>cap:raise BudgetError('Activity allocation exceeded')
            self.db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?)',(identifier(id),activity,maximum,'reserved',None,None,None));self.event('agent_reserved',dict(id=id,maximum=maximum))
    def reconcile_call(self,id,reported):
        with self.transaction():
            row=self.db.execute('SELECT * FROM calls WHERE id=?',(id,)).fetchone()
            if row is None or row['status']=='reconciled':raise ValueError('Absent/already reconciled call')
            try:total,cached=usage(reported)
            except ValueError:
                self.db.execute("UPDATE calls SET status='unknown',raw=? WHERE id=?",(json.dumps(reported),id))
                if self.get('status') not in ['stopped','budget_exhausted']:self.set('status','manual_checkpoint')
                self.event('usage_unknown',id);return
            self.db.execute("UPDATE calls SET status='reconciled',total=?,cached=?,raw=? WHERE id=?",(total,cached,json.dumps(reported),id))
            if (total>row['reserved'] or self.totals()['tokens']>self.limits['tokens']) and self.get('status') not in ['stopped','budget_exhausted']:self.set('status','budget_exhausted')
            self.event('usage_reported',dict(id=id,total=total,cached=cached,reservation_overrun=total>row['reserved']))
    def reserve_cpu(self,id,maximum):
        with self.transaction():
            self.check();t=self.totals()
            if maximum<=0 or t['cpu_seconds']+t['reserved_cpu_seconds']+maximum>self.limits['cpu_seconds']:raise BudgetError('CPU reserve exceeded')
            self.db.execute('INSERT INTO cpu VALUES(?,?,?,?)',(id,maximum,None,'reserved'))
    def reconcile_cpu(self,id,actual):
        with self.transaction():
            r=self.db.execute('SELECT * FROM cpu WHERE id=?',(id,)).fetchone()
            if r is None or r['status']=='reconciled' or actual<0:raise ValueError('CPU receipt invalid')
            self.db.execute("UPDATE cpu SET actual=?,status='reconciled' WHERE id=?",(actual,id))
            if actual>r['reserved'] and self.get('status') not in ['stopped','budget_exhausted']:self.set('status','budget_exhausted')
    def charge_step(self,job,k):
        with self.transaction():
            self.check()
            if self.totals()['optimizer_steps']>=self.limits['optimizer_steps']:raise BudgetError('Optimizer ceiling reached')
            last=self.db.execute('SELECT coalesce(max(k),0) FROM steps WHERE job=?',(job,)).fetchone()[0]
            if type(k) is not int or k!=last+1:raise BudgetError('Update cursor must be consecutive; never skip or replay')
            reserve=self.db.execute('SELECT ceiling FROM update_reservations WHERE job=?',(job,)).fetchone()
            if reserve and k>reserve['ceiling']:raise BudgetError('Per-job optimizer ceiling reached')
            if reserve is None and self.totals()['optimizer_steps']+self.totals()['reserved_optimizer_steps']>=self.limits['optimizer_steps']:raise BudgetError('Optimizer capacity already reserved')
            if self.db.execute('SELECT 1 FROM steps WHERE job=? AND k=?',(job,k)).fetchone():raise BudgetError('Never replay an attempted step')
            self.db.execute("INSERT INTO steps VALUES(?,?,'uncertain',NULL)",(job,k))
    def complete_step(self,job,k,state_hash):
        with self.transaction():
            r=self.db.execute('SELECT status FROM steps WHERE job=? AND k=?',(job,k)).fetchone()
            if r is None or r[0]!='uncertain':raise ValueError('No unique attempted step')
            self.db.execute("UPDATE steps SET status='completed',state_hash=? WHERE job=? AND k=?",(state_hash,job,k))
    def recover(self,job,saved):
        rows=self.db.execute('SELECT k,status FROM steps WHERE job=? ORDER BY k',(job,)).fetchall()
        if any(r['status']!='completed' for r in rows) or (max([r['k'] for r in rows],default=0)!=saved):raise BudgetError('Uncertain step or cursor mismatch; no replay')
        return saved
    def register(self,p,code_hash):
        identifier(p['id']);signature=fingerprint(dict(config=p['config'],parent=p.get('parent_candidate')))
        with self.transaction():
            self.check()
            if p.get('parent_candidate') and not self.db.execute('SELECT 1 FROM nodes WHERE id=?',(p['parent_candidate'],)).fetchone():raise ValueError('Unknown parent')
            if self.totals()['proposals']>=self.limits['proposals']:raise BudgetError('Proposal ceiling')
            if self.db.execute("SELECT count(*) FROM nodes WHERE status='active'").fetchone()[0]>=self.limits['active_branches']:raise BudgetError('Active branch ceiling')
            try:self.db.execute("INSERT INTO nodes(id,parent,signature,proposal,code_hash,status) VALUES(?,?,?,?,?,'active')",(p['id'],p.get('parent_candidate'),signature,encoded(p).decode(),code_hash))
            except sqlite3.IntegrityError as ex:raise ValueError('Duplicate/edited candidate or equivalent experiment') from ex
            self.event('registered',p['id'])
    def nodes(self):return [dict(r) for r in self.db.execute('SELECT * FROM nodes ORDER BY id')]
    def result(self,id,value):
        with self.transaction():self.db.execute('INSERT INTO results VALUES(?,?)',(id,encoded(value).decode()));self.event('result',id)
    def repair(self,id,scientific=False):
        field,cap=('revisions',1) if scientific else ('repairs',2)
        with self.transaction():
            r=self.db.execute('SELECT * FROM nodes WHERE id=?',(id,)).fetchone()
            if not r or r[field]>=cap:raise BudgetError('Branch repair/revision cap reached')
            self.db.execute(f'UPDATE nodes SET {field}={field}+1 WHERE id=?',(id,));self.event('scientific_revision_requires_new_id' if scientific else 'implementation_repair',id)
