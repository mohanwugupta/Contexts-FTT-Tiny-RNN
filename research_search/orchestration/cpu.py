"""CPU accounting for the trusted controller as well as isolated workers."""
import contextlib,math,signal,time,uuid
from .budget import BudgetError

@contextlib.contextmanager
def controller_budget(store,maximum=120):
    totals=store.totals();remaining=store.limits['cpu_seconds']-totals['cpu_seconds']-totals['reserved_cpu_seconds'];cap=min(maximum,math.floor(remaining))
    if cap<1:raise BudgetError('No controller CPU reserve remains')
    id='controller-'+uuid.uuid4().hex;store.reserve_cpu(id,cap);start=time.process_time();previous=signal.getsignal(signal.SIGPROF);timer=signal.getitimer(signal.ITIMER_PROF)
    def stop(*_):raise BudgetError('Controller CPU slice reached; no unbounded task generation')
    signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,cap)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_PROF,*timer);signal.signal(signal.SIGPROF,previous);store.reconcile_cpu(id,time.process_time()-start)
