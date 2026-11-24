"""Durable attempted-operation accounting; uncertain work is never repeated."""
from .common import *
class Exposure:
 def __init__(self,folder=OUT/'registry',caps=CAPS):
  self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.path=self.folder/'optimizer_events.jsonl';self.caps=caps;self.events=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else [];self.keys={(r['run'],r['step']) for r in self.events};self.total={k:sum(r['kind']==k for r in self.events) for k in caps}
 def counts(self):return dict(self.total)
 def physical_calls(self):return sum(r['physical_optimizer_calls'] for r in self.events)
 def charge(self,kind,run,step,physical_calls=1):
  if (run,step) in self.keys:raise ValueError('Attempt would repeat accepted/uncertain update')
  if kind not in self.caps or self.total[kind]>=self.caps[kind]:raise ValueError('Hard compute ceiling '+kind)
  row=dict(kind=kind,run=run,step=step,physical_optimizer_calls=physical_calls);payload=json.dumps(row)+'\n'
  with self.path.open('a') as f:f.write(payload);f.flush();os.fsync(f.fileno())
  self.events.append(row);self.keys.add((run,step));self.total[kind]+=1
 def run_count(self,run):return sum(r['run']==run for r in self.events)
def validate_resume(cp,identity,completed):
 if cp['identity']!=identity or cp['cursor']!=completed:raise ValueError('Changed identity or uncertain operations beyond saved cursor')
 return True
