"""Finalize SQLite storage for archival sealing, without changing scientific records."""
import sqlite3
from .common import OUT,dump,append

def main():
 rows=[]
 queries=['SELECT episode,hash FROM latest','SELECT episode,hash FROM episodes ORDER BY episode','SELECT episode,hash FROM checkpoints ORDER BY episode']
 for path in sorted((OUT/'meta').glob('*/journal.sqlite')):
  c=sqlite3.connect(path,timeout=5)
  before=[c.execute(q).fetchall() for q in queries]
  assert c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]==0
  assert c.execute('PRAGMA quick_check').fetchone()==('ok',)
  # DELETE requires exclusive access; a live reader/writer must fail this step.
  assert c.execute('PRAGMA journal_mode=DELETE').fetchone()==('delete',)
  assert before==[c.execute(q).fetchall() for q in queries]
  assert len(before[1])==9040 and before[0][0][0]==10000 and len(before[2])==5
  c.close()
  # The synchronized filesystem can retain empty WAL / derived SHM artifacts.
  # DELETE mode has committed all pages and obtained exclusive access above.
  wal=path.with_name(path.name+'-wal');shm=path.with_name(path.name+'-shm')
  if wal.exists():
   assert wal.stat().st_size==0,'Refuse to remove nonempty WAL'
   wal.unlink()
  if shm.exists():shm.unlink()
  rows.append(dict(run=path.parent.name,episodes=9040,latest=10000,checkpoints=5,all_record_hashes_unchanged=True,quick_check=True))
 assert len(rows)==160
 assert not list(OUT.rglob('*.sqlite-wal')) and not list(OUT.rglob('*.sqlite-shm'))
 dump(OUT/'audit/sqlite_closed.json',dict(passed=True,runs=rows,journal_mode='DELETE',scientific_records_unchanged=True,model_updates=0))
 append('## SQLite archival finalization\nA first checkpoint/close-only pass preserved all logical record hashes but failed the no-sidecar assertion: zero-byte WAL and derived SHM files persisted on this synchronized filesystem. The failed check is retained. Finalization obtained exclusive access via SQLite journal_mode=DELETE, quick-checked each database, verified all episode/restart/checkpoint record hashes unchanged, closed connections, and removed only empty WAL/derived SHM artifacts. All160×9040 episode records,160 restart states and800 registered blobs remain unchanged. No model update or scientific evaluation occurred. A future authorized continuation reopens WAL through the existing Journal constructor; sealed reproduction uses hashes and tables only.')
if __name__=='__main__':main()
