"""Count every attempted optimizer call before execution; immutable step caches."""
from .common import *
class Ledger:
 def __init__(self,path,limits=None):
  self.path=Path(path);self.limits=limits or {'diagnostic':20000,'reconstruction':21000,'fixture':64};self._events=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else [];self._attempts={r['attempt'] for r in self._events};self._counts={k:sum(r['category']==k for r in self._events) for k in self.limits}
 def events(self):return list(self._events)
 def counts(self):
  return dict(self._counts)
 def reserve(self,category,attempt):
  if self.counts()[category]>=self.limits[category]:raise RuntimeError('Budget exhausted: '+category)
  if attempt in self._attempts:raise RuntimeError('Uncertain/previous optimizer attempt; no automatic retry: '+attempt)
  self.path.parent.mkdir(parents=True,exist_ok=True)
  r=dict(category=category,attempt=attempt,time=time.time())
  with self.path.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  self._events.append(r);self._attempts.add(attempt);self._counts[category]+=1
def branch_key(state,identity,source,horizon,kind):return digest(dict(state=state,identity=identity,source=source,horizon=horizon,kind=kind))
