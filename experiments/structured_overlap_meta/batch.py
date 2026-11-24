"""One native Adam update on two independent reset sequences, averaged14 cells."""
from .common import core,torch,np

def forward_pair(m,paths,cue):
 if len(paths)!=2 or cue not in 'ABC' or any(len(p)!=8 for p in paths):raise ValueError('Two7-position examples and one registered physical cue required')
 xy=[core.tensors(path) for path in paths];x=torch.stack([v[0] for v in xy]);y=torch.stack([v[1] for v in xy]);z,h,_=m(x,torch.tensor(core.CUES[cue]));return z,h,y

def batch_objective(z,p,y):
 ce=torch.nn.functional.cross_entropy(z.reshape(-1,4),y.reshape(-1));ex=torch.stack([core.objective(z[j:j+1],p,y[j])[2] for j in range(2)]).mean();return ce+.2*ex,ce,ex

def step_batch(m,p,o,cue,paths):
 if cue not in ['A','B','C']:raise ValueError('Registered meta/pilot physical cues only')
 params=list(m.parameters())+list(p.parameters());assert [id(v) for g in o.param_groups for v in g['params']]==[id(v) for v in params];o.zero_grad();z,h,y=forward_pair(m,paths,cue);loss,ce,ex=batch_objective(z,p,y);r=dict(ce_before=float(ce.detach()),accuracy_before=float(z.argmax(-1).eq(y).float().mean()),execution_loss=float(ex.detach()),total_loss=float(loss.detach()),gradient_norm=np.nan,hidden_absmax=float(h.detach().abs().max()),optimizer_step_executed=False,status='nonfinite_loss',examples=2,positions=14)
 if not torch.isfinite(loss) or not torch.isfinite(z).all() or not torch.isfinite(h).all():return r
 loss.backward()
 if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params):r['status']='nonfinite_gradient';return r
 norm=torch.nn.utils.clip_grad_norm_(params,1.);r['gradient_norm']=float(norm)
 if not torch.isfinite(norm):r['status']='nonfinite_norm';return r
 o.step();r['optimizer_step_executed']=True;r['status']='completed' if all(torch.isfinite(v).all() for v in params) else 'nonfinite_postupdate';return r
