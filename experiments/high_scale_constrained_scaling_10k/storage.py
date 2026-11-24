"""Native Journal encoding/schema with atomic complete numerical chunks."""
import hashlib
from experiments.heldout_context_meta_scaling_10k.storage import Journal,encode,decode,atomic_save

class ChunkJournal(Journal):
 def commit_chunk(self,value,summaries,registered=False):
  previous=self.db.execute('SELECT episode FROM latest').fetchone()[0]
  if not summaries or [s['episode'] for s in summaries]!=list(range(previous+1,value['episode']+1)):raise ValueError('Would repeat/skip an episode within chunk')
  blob=encode(value);digest=hashlib.sha256(blob).hexdigest();encoded=[(s['episode'],encode(s)) for s in summaries]
  with self.db:
   self.db.executemany('INSERT INTO episodes VALUES(?,?,?)',[(e,b,hashlib.sha256(b).hexdigest()) for e,b in encoded]);self.db.execute('UPDATE latest SET episode=?,data=?,hash=? WHERE id=1',(value['episode'],blob,digest))
   if registered:self.db.execute('INSERT INTO checkpoints VALUES(?,?,?)',(value['episode'],blob,digest))
