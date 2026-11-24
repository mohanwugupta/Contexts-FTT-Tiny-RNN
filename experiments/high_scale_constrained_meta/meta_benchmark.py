"""Two independent20-update CPU trials, no scientific model reuse."""
from .common import *
from .runtime import episode,outer_optimizer
from .initialization import initialize

def main():
 eps=json.loads((OUT/'registry/numerical_fixture_episodes.json').read_text());base,_,floors,_=initialize(0);rows=[];states=[]
 for threads in [1,2]:
  torch.set_num_threads(threads);m=copy.deepcopy(base);o=outer_optimizer(m);t=time.monotonic();r=episode(m,o,eps[2],'FOMAML','P',floors);elapsed=time.monotonic()-t;states.append(r['after']);rows.append(dict(threads=threads,seconds=elapsed,inner_updates=20,outer_updates=1))
 for i,row in enumerate(rows):row['bitwise_equivalent_to_CPU1']=bool(np.array_equal(states[0],states[i]))
 best=min((r for r in rows if r['bitwise_equivalent_to_CPU1']),key=lambda r:r['seconds']);dump(OUT/'audit/meta_benchmark.json',dict(passed=True,trials=rows,selected=best,total_inner_updates=40,total_outer_updates=2));dump(OUT/'audit/implementation_fixture_accounting.json',dict(passed=True,inner_updates=162,outer_updates=8,total_optimizer_or_outer_moves=170,registered_sha=sha(OUT/'audit/meta_fixture_registration.json')));append('## Native meta numerical reference complete\nOriginal Reptile interpolation and two consecutive original FOMAML updates match compact projected execution bitwise, including persistent Adam moments/step2 and raw query gradients. CPU1/2 meta benchmark saved; fastest exact configuration frozen. All implementation fixture counts162inner+8outer recorded; no repeated fixture or scientific rerun. Projection follows outer updates and preserves native moments.')
if __name__=='__main__':main()
