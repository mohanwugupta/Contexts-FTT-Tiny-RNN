"""Recreate exact numeric tables and figures externally; no forward or update."""
import argparse
from unittest.mock import patch
from .common import *
from . import engine,analysis

def main(destination):
 dest=Path(destination)
 if dest.exists() and any(dest.iterdir()):raise ValueError('Use an empty external destination')
 if dest.resolve().is_relative_to(OUT.resolve()):raise ValueError('Never regenerate into sealed evidence')
 dest.mkdir(parents=True,exist_ok=True);before=sha(OUT/'registry/optimizer_events.jsonl');metric=[];fields=['probabilities','ce','accuracy','margin'];max_error={k:0. for k in fields};positions=0
 def forbidden(*args,**kwargs):raise AssertionError('Table-only reproduction attempted model evaluation or optimizer operation')
 with patch.object(core,'step',forbidden),patch.object(core,'trace',forbidden),patch.object(core,'objective',forbidden),patch.object(torch.optim.Adam,'step',forbidden),patch.object(torch.optim.SGD,'step',forbidden):
  for folder in sorted((OUT/'runs').iterdir()):
   if not (folder/'complete.json').exists():continue
   c=json.loads((folder/'complete.json').read_text());a=dict(np.load(folder/'observations.npz'));z=a['logits'];maximum=z.max(-1,keepdims=True);exponential=np.exp(z-maximum);probs=exponential/exponential.sum(-1,keepdims=True);label=core.TARGET;idx=np.arange(7);loss=maximum[...,0]+np.log(exponential.sum(-1))-z[:,:,idx,label];wrong=z.copy();wrong[:,:,idx,label]=-np.inf;rebuilt=dict(logits=z,output_bias=a['output_bias'],probabilities=probs,ce=loss,accuracy=z.argmax(-1)==label,margin=z[:,:,idx,label]-wrong.max(-1))
   for k in fields:
    error=float(np.max(abs(rebuilt[k].astype(float)-a[k].astype(float))));max_error[k]=max(max_error[k],error);np.testing.assert_array_equal(rebuilt[k],a[k])
   positions+=10*5*7;metric.extend(engine.metric_rows(rebuilt,c['candidate'],int(c['seed']),c['role']))
  csv(dest/'metrics.csv',metric);assert sha(dest/'metrics.csv')==sha(OUT/'metrics.csv')
  f=analysis.collect()
  for name,key in [('summary','summary'),('candidate_registry','registry'),('background_summary','background')]:csv(dest/f'{name}.csv',f[key]);assert sha(dest/f'{name}.csv')==sha(OUT/f'{name}.csv')
  csv(dest/'practice.csv.gz',f['practice']);assert sha(dest/'practice.csv.gz')==sha(OUT/'practice.csv.gz')
  from .figures import main as figures
  figures(dest/'figures');old=json.loads((OUT/'figures/FIGURE_INDEX.json').read_text());new=json.loads((dest/'figures/FIGURE_INDEX.json').read_text());assert old==new
 assert sha(OUT/'registry/optimizer_events.jsonl')==before
 result=dict(passed=True,position_predictions=positions,scalar_rows=len(metric),max_errors=max_error,numerical_tables_identical=True,all_six_figure_hashes_identical=True,model_forward_calls=0,optimizer_operations=0,ledger_sha=before);put(dest/'regeneration.json',result);print(json.dumps(result),flush=True);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);main(p.parse_args().output_dir)
